#!/usr/bin/env bash
# Fetch a checkpoint at install time. SQL statements must never make outbound
# requests; run this from a package step, an init container, or a Docker layer
# instead.
#
#   scripts/fetch-model.sh /var/lib/postgresql/tsfm/chronos-2
#   scripts/fetch-model.sh /var/lib/postgresql/tsfm/timestral-28m mistralai/Timestral-28M
#
# Timestral weights are Modified MIT with a revenue cap, and may be gated on
# Hugging Face. Accept the terms on the model page and export HF_TOKEN before
# downloading. This script does not redistribute the weights.
set -euo pipefail

dest="${1:?usage: fetch-model.sh <directory> [repo]}"
repo="${2:-${TSFM_MODEL_REPO:-amazon/chronos-2}}"
revision="${TSFM_MODEL_REVISION:-main}"
base="https://huggingface.co/${repo}/resolve/${revision}"

case "$repo" in
    mistralai/Timestral-*)
        echo "note: ${repo} weights are Modified MIT with a revenue cap." >&2
        echo "      Accept the model terms and set HF_TOKEN if the repo is gated." >&2
        ;;
esac

auth=()
if [[ -n "${HF_TOKEN:-}" ]]; then
    auth=(-H "Authorization: Bearer ${HF_TOKEN}")
fi

mkdir -p "$dest"
for file in config.json model.safetensors; do
    echo "fetching ${repo}/${file} -> ${dest}/${file}"
    # Download to a temp name so an interrupted fetch never leaves a truncated
    # model.safetensors that later looks like a valid install.
    curl "${auth[@]}" -fL --retry 3 --retry-delay 2 "${base}/${file}" -o "${dest}/${file}.part"
    mv "${dest}/${file}.part" "${dest}/${file}"
done

echo "done. SET tsfm.model_dir = '${dest}';"
