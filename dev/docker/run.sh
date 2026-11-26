#!/usr/bin/env bash
set -e

IMAGE_NAME=tsfm-pg-dev
IMAGE_TAG=pg17

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PRJ_ROOT=${PRJ_ROOT:-$(realpath "$SCRIPT_DIR/../..")}

docker run -i -t --rm \
	-v "$PRJ_ROOT:/workspace" \
	-w /workspace \
	"$IMAGE_NAME:$IMAGE_TAG" \
	"$@"
