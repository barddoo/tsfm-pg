# syntax=docker/dockerfile:1.7
#
# Two runtime images:
#   docker build --target chronos2 -t tsfm-pg:chronos2 .   # weights baked in (~500 MB)
#   docker build                    -t tsfm-pg:slim .       # mount a model directory
#
# The extension itself is the same in both. The default target is `final`,
# which is the slim image; select `chronos2` explicitly for the baked weights.

FROM postgres:17-bookworm AS build
ARG ZIG_VERSION=0.17.0
ARG PGZX_COMMIT=4d99bbc56be9a00aac3e27c6e733840245e54e89
# CPU target for the SIMD kernels. x86_64_v3 = AVX2/FMA (Haswell, 2013, and
# newer); the image will not start on older CPUs. Build with
# `--build-arg TSFM_CPU=baseline` for a portable SSE2 image.
ARG TSFM_CPU=x86_64_v3
# Engine GPU backend: none or cuda. A cuda build compiles without the toolkit
# (the engine dlopens libcudart/libcublas), but the runtime image must provide
# those libraries and the NVIDIA container runtime.
ARG TSFM_GPU=none

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates curl git xz-utils build-essential pkg-config \
        libpq-dev postgresql-server-dev-17 \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL "https://ziglang.org/download/${ZIG_VERSION}/zig-x86_64-linux-${ZIG_VERSION}.tar.xz" \
        | tar -xJ -C /opt \
    && ln -s "/opt/zig-x86_64-linux-${ZIG_VERSION}/zig" /usr/local/bin/zig

# pgzx is a path dependency in the source tree; its published package omits
# build.zig.zon from .paths, so clone the pinned commit next to tsfm-pg.
RUN git clone https://github.com/barddoo/pgzx /src/pgzx \
    && git -C /src/pgzx checkout "${PGZX_COMMIT}"

WORKDIR /src/tsfm-pg
COPY . .

ENV PG_CONFIG=/usr/lib/postgresql/17/bin/pg_config
RUN zig build -Doptimize=ReleaseSafe -Dcpu="${TSFM_CPU}" -Dgpu="${TSFM_GPU}" -p /usr/lib/postgresql/17

FROM postgres:17-bookworm AS runtime
ARG PG_MAJOR=17

COPY --from=build /usr/lib/postgresql/${PG_MAJOR}/lib/tsfm.so /usr/lib/postgresql/${PG_MAJOR}/lib/
# pg_config --sharedir is /usr/share/postgresql/<major> on Debian, not
# /usr/lib/postgresql/<major>/share/postgresql, which is where the build's
# `zig build -p /usr/lib/postgresql/<major>` installs them.
COPY --from=build /usr/share/postgresql/${PG_MAJOR}/extension/tsfm.control \
                  /usr/share/postgresql/${PG_MAJOR}/extension/
COPY --from=build /usr/share/postgresql/${PG_MAJOR}/extension/tsfm--0.1.sql \
                  /usr/share/postgresql/${PG_MAJOR}/extension/

# Default tsfm.model_dir is $sharedir/tsfm/models. Mount a checkpoint here for
# the slim image, or use the chronos2 target to bake Chronos-2 in. Timestral
# weights are Modified MIT with a revenue cap and are deliberately not baked;
# mount a user-supplied Timestral directory instead.
RUN mkdir -p /usr/share/postgresql/${PG_MAJOR}/tsfm/models

# Optional first-start download: set TSFM_MODEL_REPO (see the entrypoint) and
# mount a volume at the models directory so the checkpoint is fetched once.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*
COPY scripts/fetch-model.sh /usr/local/bin/tsfm-fetch-model
COPY scripts/docker-entrypoint-tsfm.sh /usr/local/bin/tsfm-entrypoint
# Setting ENTRYPOINT clears the base image's CMD, so restate it.
ENTRYPOINT ["tsfm-entrypoint"]
CMD ["postgres"]

FROM runtime AS slim

# Install-time download, never query-time: a SQL statement must not make
# outbound requests. The weights layer is cached independently of the build.
FROM postgres:17-bookworm AS weights
ARG CHRONOS2_REPO=amazon/chronos-2
ARG CHRONOS2_REVISION=main
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /weights \
    && curl -fsSL --retry 3 \
        "https://huggingface.co/${CHRONOS2_REPO}/resolve/${CHRONOS2_REVISION}/config.json" \
        -o /weights/config.json \
    && curl -fsSL --retry 3 \
        "https://huggingface.co/${CHRONOS2_REPO}/resolve/${CHRONOS2_REVISION}/model.safetensors" \
        -o /weights/model.safetensors

FROM runtime AS chronos2
ARG PG_MAJOR=17
COPY --from=weights /weights /usr/share/postgresql/${PG_MAJOR}/tsfm/models

# The default target is the slim image. Use `--target chronos2` for weights.
FROM slim AS final
