tsfm-pg
=======

`tsfm-pg` runs the [tsfm-zig](https://github.com/barddoo/tsfm-zig) forecaster
inside Postgres. It supports **Chronos-2** and **Timestral** checkpoints: the
model and the weights sit on the database host, and a query never ships the
series to a cloud API.

```sql
CREATE EXTENSION tsfm;

SELECT step, quantile, value
FROM tsfm_forecast(
       (SELECT array_agg(value ORDER BY ts) FROM metrics WHERE name = 'sku-123'),
       90)
WHERE quantile IN (0.1, 0.5, 0.9)
ORDER BY step, quantile;
```

Chronos-2 handles one univariate series per call. Timestral adds multiple target
variates plus past-only, known-future and categorical covariates. The engine is
linked as a Zig module, so there is no C ABI, no `dlopen`, and no second
artifact: `tsfm.so` is the whole extension.

## What it provides

| function | returns |
|---|---|
| `tsfm_version()` | extension and pinned engine version |
| `tsfm_model_info()` | `jsonb` metadata (model, dir, quantiles, lengths) |
| `tsfm_quantiles()` | the model's `double precision[]` quantile levels |
| `tsfm_load()` | `boolean`; explicit warmup |
| `tsfm_forecast(history, horizon)` | `TABLE(step, quantile, value)` |
| `tsfm_forecast_ex(history, horizon, n_variates, past, known, cat_past, cat_known)` | `TABLE(variate, step, quantile, value)` |
| `tsfm_forecast_batch(history, horizon, n_series, n_variates, past, known, cat_past, cat_known)` | `TABLE(series, variate, step, quantile, value)` |
| `tsfm_forecast_matrix(history, horizon)` | `double precision[]`, `[step][quantile]` |
| `tsfm_forecast_matrix_ex(history, horizon, n_variates, past, known, cat_past, cat_known)` | `double precision[]`, `[variate][step][quantile]` |
| `tsfm_forecast_ts(timestamps, history, horizon, step_size)` | `TABLE(ts, step, quantile, value)` |
| `tsfm_model_status()` | `jsonb`: is a usable checkpoint in `tsfm.model_dir`, and if not, what to do |

`history` is ordered oldest → newest. `NULL` and `'NaN'::float8` elements are
missing values. Rows are emitted step-major, with `step` starting at 1.
Values are converted to float32 for the engine: infinities and values beyond the
float32 range become missing, and integers above 2^24 (16,777,216) lose
precision, so scale or offset very large series such as epoch seconds first.
`tsfm_forecast_ex` also numbers variates from 1.

The long form is usually what you want:

```sql
-- every series in a table, one forecast each
SELECT m.name, f.step, f.value
FROM (SELECT name, array_agg(value ORDER BY ts) AS h
      FROM metrics GROUP BY name) m,
     LATERAL tsfm_forecast(m.h, 90) f
WHERE f.quantile = 0.5;
```

The outer executor calls the function once per group, so `statement_timeout`,
`LIMIT` and cancellation work between series. A single call is not
interruptible; keep horizons short.

For multiple target variates and covariates, use `tsfm_forecast_ex`. Arrays are
flattened row-major: `history` is `V × T`, `past` is `C × T`, and `known` is
`C × (T + horizon)`.

```sql
-- two variates, one past-only and one known-future covariate, 2 steps
SELECT variate, step, quantile, value
FROM tsfm_forecast_ex(
       ARRAY[10, 11, 12, 20, 21, 22]::float8[], -- 2 variates × 3 steps
       2,
       2,
       ARRAY[0, 1, 0]::float8[],                -- 1 past covariate × 3
       ARRAY[0, 1, 0, 1, 0]::float8[],          -- 1 known × (3 + 2)
       NULL,
       NULL)
WHERE quantile IN (0.1, 0.5, 0.9)
ORDER BY variate, step, quantile;
```

Timestral also accepts categorical arrays through `cat_past` and `cat_known`.
Chronos-2 rejects anything beyond `n_variates = 1` with no covariates.

For many independent series, `tsfm_forecast_batch` passes `B` series in one
engine call, which reuses each weight read across the batch:

```sql
-- 100 series, 365 values each, one value per series per step
SELECT series, step, value
FROM tsfm_forecast_batch(ARRAY[/* B × 365 values */], 90, 100, 1)
WHERE quantile = 0.5
ORDER BY series, step;
```

The batch checks for query cancellation between chunks of 64 series. A single
forward pass is still not interruptible.

`tsfm_forecast_ts` attaches timestamps using the last timestamp and a step:

```sql
SELECT ts, step, quantile, value
FROM tsfm_forecast_ts(
       ARRAY['2024-01-01','2024-01-02','2024-01-03']::timestamptz[],
       ARRAY[10, 11, 12]::float8[],
       7,
       interval '1 day')
WHERE quantile = 0.5;
```

`CREATE EXTENSION` is superuser-only and the extension revokes `EXECUTE` from
`PUBLIC`; grant it explicitly:

```sql
GRANT EXECUTE ON FUNCTION tsfm_forecast(double precision[], integer) TO analyst;
GRANT EXECUTE ON FUNCTION tsfm_forecast_ex(double precision[], integer, integer, double precision[], double precision[], double precision[], double precision[]) TO analyst;
GRANT EXECUTE ON FUNCTION tsfm_forecast_batch(double precision[], integer, integer, integer, double precision[], double precision[], double precision[], double precision[]) TO analyst;
GRANT EXECUTE ON FUNCTION tsfm_forecast_matrix(double precision[], integer) TO analyst;
GRANT EXECUTE ON FUNCTION tsfm_forecast_matrix_ex(double precision[], integer, integer, double precision[], double precision[], double precision[], double precision[]) TO analyst;
GRANT EXECUTE ON FUNCTION tsfm_forecast_ts(timestamptz[], double precision[], integer, interval) TO analyst;
```

`tsfm_forecast_ts` calls `tsfm_forecast`, so a role needs EXECUTE on both.

## Settings

Settings that decide what runs or what is read from disk are **superuser-only**
(`SUSET`): `tsfm.model_dir`, `tsfm.model`, `tsfm.threads`, `tsfm.enabled` and
`tsfm.stub`. Set them in `postgresql.conf`, with `ALTER SYSTEM`, or per
role/database with `ALTER ROLE ... SET`. That makes `tsfm.enabled = off` a real
kill switch (a role cannot turn it back on) and stops ordinary roles from
pointing the server at arbitrary files. The remaining knobs are `USERSET`, so
`SET LOCAL` works for them in a transaction.

Note that `symmetric` is an SQL keyword: write `SET tsfm."symmetric" = off`.

| GUC | default | notes |
|---|---|---|
| `tsfm.model_dir` | `<sharedir>/tsfm/models` | `config.json` + `model.safetensors` |
| `tsfm.model` | `auto` | `auto`, `chronos-2`, `timestral`; `auto` reads `config.json` |
| `tsfm.threads` | `1` | 0..64 (larger values are clamped); `0` = auto (up to 4 workers based on CPU count); ignored after the model has loaded |
| `tsfm.max_context` | `0` | keep the most recent N values; `0` = model default |
| `tsfm.sort_quantiles` | `on` | per-step monotonic quantiles; turn off for raw parity |
| `tsfm.enabled` | `on` | kill switch; forecasts error when off |
| `tsfm.stub` | `off` | naive last-value engine, for tests without weights |
| `tsfm.block_patches` | `32` | Timestral-only: patches decoded per block |
| `tsfm.symmetric` | `on` | Timestral-only: average with the mirrored forecast |
| `tsfm.clip_nonnegative` | `on` | Timestral-only: clamp non-negative series at 0 |
| `tsfm.quantile_real_cap_k` | `1e4` | Timestral-only: clamp to observed range + k×scale; 0 disables |
| `tsfm.extreme_value_threshold` | `1e30` | Timestral-only: fed-back medians at/above this become 0 |

The engine is process-global per backend and loaded lazily on the first call.
`tsfm.model_dir` and `tsfm.model` are fixed once it has loaded; changing them
later raises an error with a hint to use a new session. `tsfm.threads` sizes the
worker pool once, so a later change has no effect; `tsfm_model_info()` reports
the values actually in use. `tsfm.enabled = off` also blocks `tsfm_load()` and
`tsfm_quantiles()`, so a disabled session never loads the model.
`tsfm_load()` is an explicit warmup for connection pools.

`tsfm.model = 'auto'` selects Chronos-2 when `config.json` declares
`Chronos2Model`; otherwise it tries Timestral. Set the family explicitly if a
checkpoint is unusual.

The 480 MB checkpoint is a read-only mmap. Every backend shares the file-backed
pages through the page cache, so the private cost per connection is the model
structs plus forward-pass scratch (a few MB), not the weights.

Engine allocations use libc (`std.heap.c_allocator`), not `palloc`, so they can
outlive transactions and be used by worker threads. Allocation failures come
back as Zig errors, all `defer` cleanup runs, and the SQL layer maps them to a
normal Postgres `out of memory` error. Model-load allocations sit in an arena:
a failed or partial checkpoint load is freed in one step. One forecast result
is capped at 16 million f32 values (64 MiB) to turn an accidental huge
`n_variates` into an error instead of a backend-sized allocation. SQL output
arrays and JSON use the Postgres context allocator and are freed by Postgres.

`tsfm.threads > 1` is opt-in. Workers touch only engine memory: they never call
`ereport`, never touch `palloc`, and never check PostgreSQL interrupts. `0`
means auto: up to four workers based on the logical CPU count. The default
stays `1` because many backends each starting a pool can oversubscribe the
host.

## Requirements and build

- Zig 0.17 (same compiler as tsfm-zig; the module is compiled in, not loaded)
- PostgreSQL 16 or 17 with server headers
- A Chronos-2 checkpoint (Apache-2.0), for example `amazon/chronos-2`
- Optionally a Timestral checkpoint (Modified MIT, user-supplied)

The dev shell in `flake.nix` provides Postgres 16 and the pgzx scripts:

```sh
nix develop
pglocal && pginit
pgstart

# Nix's pg_config prints a relative --bindir when invoked through PATH, which
# makes pgzx install tsfm.so outside $libdir. Point at it by absolute path:
export PG_CONFIG="$PG_HOME/bin/pg_config"
zig build -Dcpu=native -p "$PG_HOME"

psql -U postgres -c 'CREATE EXTENSION tsfm'
psql -U postgres -c 'SELECT tsfm_version()'
```

### Dev loop script

`scripts/dev.sh` wraps the steps above (absolute `PG_CONFIG`, `-p`, `-Dcpu`,
cluster start/stop) and works inside or outside `nix develop`:

```sh
scripts/dev.sh test      # engine tests, no Postgres
scripts/dev.sh regress   # build + start cluster + pg_regress
scripts/dev.sh smoke     # real model, one forecast; needs TSFM_TEST_MODEL, else skips
scripts/dev.sh all
```

It honours `PG_HOME` (default `out/16`), `PGPORT`, `PGUSER` and `TSFM_CPU`
(default `native`). `smoke` runs in a throwaway database and checks the row
count, finiteness and a plausible median.

`zig build` defaults to `ReleaseSafe`: a bounds or overflow bug in the model
must become a Postgres error, not a segmentation fault that takes the backend
down. Use `-Doptimize=ReleaseFast` only for benchmarks.

`zig build` fetches `tsfm-zig` from the commit pinned in `build.zig.zon`. `pgzx`
is a path dependency (`../pgzx`) because its published package omits
`build.zig.zon` from `.paths`; clone it next to this repository. Both commits
are recorded in `build.zig.zon` / `Dockerfile`.

### GPU

The engine's matrix kernels can run on a GPU. This is opt-in, because it keeps a
second copy of the model's GEMM weights in device memory:

```sh
zig build -Dgpu=metal          # macOS
zig build -Dgpu=cuda           # Linux with an NVIDIA driver
```

The default is `-Dgpu=none`. The extension is still one `tsfm.so`: the CUDA
backend loads `libcudart`/`libcublas` with `dlopen` at model-load time, so the
build needs no CUDA toolkit or link-time libraries. At runtime, a normal CUDA
installation is found automatically; set `TSFM_CUDA_LIB_DIR` to use a private
copy. The variable must be in the **Postgres server process's** environment
(systemd `Environment=`, the compose service, or the shell that runs `pg_ctl`),
not just the `psql` session. If the device or libraries are missing, the engine
logs one warning and uses the CPU kernels instead.

The backend is selected once, before the model is loaded. `tsfm.threads` still
controls the CPU worker pool for elementwise work; cuBLAS/MPS parallelize the
large matrix products internally. See `tsfm-zig/docs/plans/gpu-cuda.md` and
`gpu-metal.md` for thresholds and measured results.

### SIMD

The model's matrix kernels (in `tsfm-zig`, built with `-Dblas=builtin`) and the
loops in this extension use Zig `@Vector` code whose width comes from the
**target CPU**, so the build's CPU decides how wide it runs. `tsfm_model_info()`
reports the result as `cpu_arch` and `simd_f32_lanes` (f32 lanes per vector
register; `1` means scalar).

- **Apple Silicon / aarch64:** NEON is part of the baseline, so the default
  build already gets 4 lanes.
- **x86-64:** without `-Dcpu` the target is baseline SSE2 (4 lanes). Build with
  `-Dcpu=x86_64_v3` for AVX2/FMA (8 lanes) or `-Dcpu=native` for the build
  machine. The resulting `tsfm.so` only runs on CPUs with those features (an
  older CPU fails with an illegal-instruction crash).
- **Docker:** the `Dockerfile` builds with `-Dcpu=${TSFM_CPU}`, default
  `x86_64_v3`. Use `--build-arg TSFM_CPU=baseline` for a portable image.

Check a deployment with `SELECT tsfm_model_info()->>'simd_f32_lanes'`.

### Model files

Downloads are install-time, never query-time. The extension never contacts the
network from SQL.

```sh
# Chronos-2 (Apache-2.0); can be baked into a public image
scripts/fetch-model.sh /var/lib/postgresql/tsfm/chronos-2

# Timestral (Modified MIT with a revenue cap); download it yourself
HF_TOKEN=... scripts/fetch-model.sh /var/lib/postgresql/tsfm/timestral-28m \
  mistralai/Timestral-28M
```

Then point the extension at the directory. These settings are superuser-only, so
use `postgresql.conf`, `ALTER SYSTEM` or `ALTER ROLE ... SET` (a plain
`SET` works in a superuser session for testing):

```sql
ALTER SYSTEM SET tsfm.model = 'timestral';
ALTER SYSTEM SET tsfm.model_dir = '/var/lib/postgresql/tsfm/timestral-28m';
SELECT pg_reload_conf();
```

Timestral weights are **not** redistributed or baked into the published Docker
image. Users are responsible for accepting the model terms and downloading the
weights. The default directory is `<pg_config --sharedir>/tsfm/models`.
Restrict `EXECUTE` to the roles that should be able to run forecasts; the
extension revokes it from `PUBLIC`.

## Tests

```sh
zig build test          # engine integration, no Postgres (stub + optional model)
zig build pg_regress    # SQL surface, no weights (tsfm.stub = on)
zig build parity        # numerical parity; needs TSFM_TEST_MODEL and/or TSFM_TEST_TIMESTRAL
```

`zig build parity` runs `tests/parity.sql` (Chronos-2) and
`tests/timestral_parity.sql` (Timestral, 2 variates + covariates) twice, with
`tsfm.threads=1` and `tsfm.threads=8`. It skips cleanly when the corresponding
environment variable is unset.

```sh
TSFM_TEST_MODEL=~/.cache/huggingface/hub/models--amazon--chronos-2/snapshots/<rev> \
TSFM_TEST_TIMESTRAL=~/.cache/huggingface/hub/models--mistralai--Timestral-28M/snapshots/<rev> \
  zig build parity
```

## Performance

Measured on an Apple M4, ReleaseSafe, warm engine (model load excluded):

| case | `tsfm.threads=1` | `tsfm.threads=8` |
|---|---|---|
| one series, 365 → 90 | 155 ms | 38 ms |
| batch of 32, 365 → 90 | 4.4 s | 1.0 s |

The forward pass streams the weight matrix once per pass and is
memory-bandwidth bound. `tsfm_forecast_batch` reuses each weight read across up
to 64 series per chunk; `LATERAL` remains useful when series need different
horizons or settings.

## Docker

```sh
# extension only; mount or fetch a checkpoint at run time
docker build -t tsfm-pg:slim .

# weights baked in at build time (~500 MB)
docker build --target chronos2 -t tsfm-pg:chronos2 .

docker run --rm -e POSTGRES_PASSWORD=pg tsfm-pg:chronos2
```

The weights layer uses `curl` against `huggingface.co` during `docker build`,
not at query time. Only Chronos-2 can be baked into a public image; for
Timestral, use the slim image and mount a user-supplied checkpoint directory at
`/usr/share/postgresql/17/tsfm/models` (or set `tsfm.model_dir`).

### Checking the install: `tsfm_model_status()`

A read-only diagnostic. It stats the files in `tsfm.model_dir` and reads a small
`config.json`; it loads no weights, writes nothing and never touches the
network, so it is safe to run anywhere and even with `tsfm.enabled = off`.

```sql
SELECT jsonb_pretty(tsfm_model_status());
-- or just the verdict:
SELECT s->>'ready', s->>'problem', s->>'fix' FROM (SELECT tsfm_model_status() s) x;
```

Fields: `model_dir`, `config_json` and `weights` (`present`, `missing` or
`unreadable`), `weights_bytes`, `configured_model` (the setting) and
`detected_model` (what `config.json` describes), `loaded`, `ready`, and the
human-readable `problem`, `fix` and `note`. It flags a missing or unreadable
file, an empty or truncated `model.safetensors`, an invalid `config.json`, and a
`tsfm.model` that disagrees with the checkpoint. The "model file not found"
error points here. Like the other functions, `EXECUTE` is revoked from `PUBLIC`
because it reveals server paths.

### Fetch on first start

The image's entrypoint can download the checkpoint once, before Postgres
starts, when `TSFM_MODEL_REPO` is set and the models directory is empty:

```sh
docker run -d -e POSTGRES_PASSWORD=pg \
  -e TSFM_MODEL_REPO=amazon/chronos-2 \
  -e TSFM_MODEL_REVISION=main \
  -e HF_TOKEN=... \
  -v tsfm-models:/usr/share/postgresql/17/tsfm/models \
  tsfm-pg:slim
```

Set `HF_TOKEN` only for gated repos. Use a volume so the download happens once. The download is atomic (a temporary
name, then rename), so a failed start never leaves a truncated checkpoint, and
the container exits if the download fails. Without `TSFM_MODEL_REPO` the
entrypoint does nothing. This happens at container start, never from SQL.

Examples: [`deploy/compose.yaml`](deploy/compose.yaml) (Docker Compose with
named volumes) and [`deploy/k8s/postgres-tsfm.yaml`](deploy/k8s/postgres-tsfm.yaml)
(an init container fetches the model into a shared volume, so the database
container needs no outbound network and mounts it read-only). Treat both as
starting points: replace the passwords and pin `TSFM_MODEL_REVISION` to a
commit.

## Deploying: who can install what

The extension is a native shared library plus model files, so installing it
needs access to the database host's filesystem. SQL roles cannot do this, by
design: model settings are superuser-only and nothing downloads from SQL.

| Environment | Works? | How |
|---|---|---|
| Self-hosted Postgres / VM | yes | An operator runs `scripts/fetch-model.sh` as the `postgres` OS user, installs the library, and sets `tsfm.model_dir` |
| Docker / Compose | yes | `tsfm-pg` image; bake Chronos-2 (`--target chronos2`) or fetch on first start |
| Kubernetes | yes | StatefulSet with an init container, see `deploy/k8s/` |
| Managed Postgres (RDS, Cloud SQL, Azure, Supabase, Neon, ...) | **no** | These services only allow a fixed list of extensions and do not accept custom shared libraries or filesystem access. Run your own Postgres (VM, container, Kubernetes) instead |

Application roles only need `GRANT EXECUTE` on the forecast functions they use.

## Limitations

- Timestral weights are Modified MIT with a revenue cap and are not
  redistributed; the user downloads them.
- `LATERAL` still works for many series; `tsfm_forecast_batch` batches them in
  one engine call, but there is no cross-series attention.
- Horizons above 1024 are not supported upstream; long-horizon unrolling is
  not ported.
- `tsfm_forecast_ts` assumes a regular timestamp step; it does not inspect
  gaps or calendars.
- The extension and tsfm-zig share a Zig version, so upgrades are coordinated
  across both repositories.

## Layout

```
build.zig                 extension + engine dependency, default ReleaseSafe
extension/tsfm.control    CREATE EXTENSION metadata
extension/tsfm--0.1.sql   SQL surface and REVOKEs
src/main.zig              PG_MODULE_MAGIC, function registration, _PG_init/_PG_fini
src/guc.zig               the settings table above
src/engine.zig            model lifetime, Chronos-2 + Timestral load/forecast, stub
src/forecast.zig          array marshalling, SRF iterators, error translation
src/cursor.zig            flat-result cursor for the SRF iterators
src/functions.zig         SQL-visible scalar functions and model info
src/engine_test.zig       engine-only tests
tests/parity.sql          Chronos-2 numerical parity fixtures
tests/timestral_parity.sql Timestral variate + covariate parity fixtures
docs/plans/pg-extension.md design notes
```

## Licensing

This extension is Apache-2.0, like tsfm-zig. Keep the `NOTICE` chain and the
pgzx attribution. Chronos-2 code and weights are Apache-2.0. Do not redistribute
Timestral weights without checking their Modified MIT terms.
