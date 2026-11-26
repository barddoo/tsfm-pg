#!/usr/bin/env bash
# One entry point for the local dev loop.
#
#   scripts/dev.sh pginit     initialise a local cluster (first-time setup)
#   scripts/dev.sh build      build + install into $PG_HOME
#   scripts/dev.sh start      start the local cluster if it is not running
#   scripts/dev.sh stop       stop it
#   scripts/dev.sh regress    build, start, run pg_regress
#   scripts/dev.sh smoke      load a real model and run one forecast (needs TSFM_TEST_MODEL)
#   scripts/dev.sh test       zig engine tests (no Postgres)
#   scripts/dev.sh all        test + regress + smoke
#
# Environment:
#   PG_HOME      Postgres install prefix with bin/pg_config
#                (default: brew postgresql@18 if available, else <repo>/out/18)
#   PGDATA_DIR   cluster data directory  (default: <repo>/out/data)
#   PGPORT       port                    (default: 5432)
#   PGUSER       role                    (default: postgres)
#   TSFM_CPU     value for -Dcpu         (default: native)
#   TSFM_TEST_MODEL  model dir for `smoke`; skipped when unset
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

_brew_pg() { brew --prefix postgresql@18 2>/dev/null || true; }
PG_HOME="${PG_HOME:-$(_brew_pg)}"
PG_HOME="${PG_HOME:-$root/out/18}"
PGDATA_DIR="${PGDATA_DIR:-$root/out/data}"
PGPORT="${PGPORT:-5432}"
PGUSER="${PGUSER:-postgres}"
cpu="${TSFM_CPU:-native}"
logfile="$root/out/pg.log"

[[ -x "$PG_HOME/bin/pg_config" ]] || {
    echo "error: $PG_HOME/bin/pg_config not found; set PG_HOME to your Postgres prefix" >&2
    exit 1
}
export PG_CONFIG="$PG_HOME/bin/pg_config" # must be absolute, see README
psql_cmd=("$PG_HOME/bin/psql" -X -q -h /tmp -p "$PGPORT" -U "$PGUSER" -v ON_ERROR_STOP=1)

running() { "$PG_HOME/bin/pg_isready" -q -h /tmp -p "$PGPORT"; }

zig_build() { (cd "$root" && zig build "$@" -Dcpu="$cpu" -p "$PG_HOME" -Dpg-port="$PGPORT" -Dpg-user="$PGUSER"); }

cmd_pginit() {
    [[ -d "$PGDATA_DIR" ]] && { echo "cluster already exists at $PGDATA_DIR"; return; }
    mkdir -p "$PGDATA_DIR" "$(dirname "$logfile")"
    "$PG_HOME/bin/initdb" -D "$PGDATA_DIR" -U "$PGUSER" --no-instructions >/dev/null
    echo "initialised cluster at $PGDATA_DIR"
}

cmd_build() { (cd "$root" && zig build -Dcpu="$cpu" -p "$PG_HOME"); echo "installed into $PG_HOME"; }

cmd_start() {
    [[ -d "$PGDATA_DIR" ]] || { echo "error: no cluster at $PGDATA_DIR; run: scripts/dev.sh pginit" >&2; exit 1; }
    if running; then echo "already running on :$PGPORT"; return; fi
    mkdir -p "$(dirname "$logfile")"
    "$PG_HOME/bin/pg_ctl" -D "$PGDATA_DIR" -l "$logfile" -o "-k /tmp -p $PGPORT" -w start >/dev/null
    echo "started on :$PGPORT (log: $logfile)"
}

cmd_stop() {
    running || { echo "not running"; return; }
    "$PG_HOME/bin/pg_ctl" -D "$PGDATA_DIR" -m fast -w stop >/dev/null
    echo "stopped"
}

cmd_regress() {
    cmd_build
    cmd_start
    zig_build pg_regress
    rm -rf "$root"/results "$root"/regression.out "$root"/regression.diffs
}

cmd_test() { (cd "$root" && zig build test -Dcpu="$cpu" --summary all); }

cmd_smoke() {
    if [[ -z "${TSFM_TEST_MODEL:-}" ]]; then
        echo "skip: set TSFM_TEST_MODEL to a model directory"
        return
    fi
    cmd_build
    cmd_start
    local db="tsfm_smoke_$$"
    "${psql_cmd[@]}" -d postgres -c "CREATE DATABASE $db"
    trap '"${psql_cmd[@]}" -d postgres -c "DROP DATABASE IF EXISTS '"$db"'" >/dev/null' EXIT
    "${psql_cmd[@]}" -d "$db" -v "model_dir=$TSFM_TEST_MODEL" <<'SQL'
CREATE EXTENSION tsfm;
SET tsfm.model_dir = :'model_dir';
SELECT tsfm_load() AS loaded;
SELECT tsfm_model_info()->>'model' AS model,
       tsfm_model_info()->>'simd_f32_lanes' AS simd_lanes,
       tsfm_model_info()->>'loaded' AS loaded;

DO $$
DECLARE
    h integer := 8;
    n integer;
    bad integer;
    med float8;
BEGIN
    SELECT count(*) INTO n FROM tsfm_forecast(
        ARRAY(SELECT i::float8 FROM generate_series(1, 64) i), h);
    IF n <> h * array_length(tsfm_quantiles(), 1) THEN
        RAISE EXCEPTION 'smoke: expected % rows, got %', h * array_length(tsfm_quantiles(), 1), n;
    END IF;

    SELECT count(*) INTO bad FROM tsfm_forecast(
        ARRAY(SELECT i::float8 FROM generate_series(1, 64) i), h)
    WHERE value IS NULL OR value = 'NaN' OR value = 'Infinity' OR value = '-Infinity';
    IF bad > 0 THEN
        RAISE EXCEPTION 'smoke: % non-finite forecast values', bad;
    END IF;

    SELECT value INTO med FROM tsfm_forecast(
        ARRAY(SELECT i::float8 FROM generate_series(1, 64) i), h)
    WHERE step = 1 AND quantile = 0.5;
    IF med < 40 OR med > 100 THEN
        RAISE EXCEPTION 'smoke: step-1 median % is implausible for 1..64', med;
    END IF;
    RAISE NOTICE 'smoke ok: step-1 median = %', med;
END $$;
SQL
}

case "${1:-}" in
    pginit)  cmd_pginit ;;
    build)   cmd_build ;;
    start)   cmd_start ;;
    stop)    cmd_stop ;;
    regress) cmd_regress ;;
    test)    cmd_test ;;
    smoke)   cmd_smoke ;;
    all)     cmd_test; cmd_regress; cmd_smoke ;;
    *) sed -n '2,21p' "${BASH_SOURCE[0]}"; exit 2 ;;
esac
