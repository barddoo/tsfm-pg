#!/usr/bin/env bash
# Wraps the postgres image's entrypoint: when TSFM_MODEL_REPO is set and the
# model directory has no checkpoint yet, download it once before Postgres
# starts. The download happens at container start, never from SQL.
#
#   docker run -e TSFM_MODEL_REPO=amazon/chronos-2 -v tsfm-models:/usr/share/postgresql/17/tsfm/models ...
#
# Env: TSFM_MODEL_REPO (Hugging Face repo; unset = do nothing), TSFM_MODEL_REVISION,
#      HF_TOKEN (gated repos), TSFM_MODEL_DIR (default below; must match tsfm.model_dir).
set -euo pipefail

dir="${TSFM_MODEL_DIR:-/usr/share/postgresql/${PG_MAJOR:-17}/tsfm/models}"

if [[ "${1:-}" == postgres && -n "${TSFM_MODEL_REPO:-}" ]]; then
    if [[ -s "$dir/model.safetensors" && -s "$dir/config.json" ]]; then
        echo "tsfm: model already present in $dir"
    else
        echo "tsfm: fetching ${TSFM_MODEL_REPO} into $dir"
        mkdir -p "$dir"
        if [[ "$(id -u)" == 0 ]]; then
            chown postgres:postgres "$dir"
            gosu postgres tsfm-fetch-model "$dir" "$TSFM_MODEL_REPO"
        else
            tsfm-fetch-model "$dir" "$TSFM_MODEL_REPO"
        fi
    fi
fi

exec docker-entrypoint.sh "$@"
