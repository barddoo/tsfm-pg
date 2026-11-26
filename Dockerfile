# syntax=docker/dockerfile:1.7
#
# Two runtime images:
#   docker build --target chronos2 -t tsfm-pg:chronos2 .   # weights baked in (~500 MB)
#   docker build                    -t tsfm-pg:slim .       # mount a model directory
#
# The extension itself is the same in both. The default target is `final`,
# which is the slim image; select `chronos2` explicitly for the baked weights.
#
# Builds for the platform Docker selects (amd64 or arm64). On amd64 the SIMD
# kernels default to x86_64_v3 (AVX2/FMA, Haswell 2013 and newer); pass
# --build-arg TSFM_CPU=baseline for an image that also runs on older CPUs.

ARG PG_VERSION=17

FROM postgres:${PG_VERSION}-bookworm AS build
ARG PG_VERSION
ARG TARGETARCH
ARG ZIG_VERSION=0.17.1
# Empty = x86_64_v3 on amd64, target baseline elsewhere. "native" tunes to the
# build host: fast, but do not reuse that image on a different CPU.
ARG TSFM_CPU=
# Engine GPU backend: none or cuda. A cuda build compiles without the toolkit
# (the engine dlopens libcudart/libcublas), but the runtime image must provide
# those libraries and the NVIDIA container runtime.
ARG TSFM_GPU=none

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates curl git xz-utils build-essential pkg-config \
        libpq-dev postgresql-server-dev-${PG_VERSION} \
    && rm -rf /var/lib/apt/lists/*

# Match the Zig tarball to the image architecture, so `docker build` also works
# on arm64 hosts (and with buildx --platform).
RUN case "${TARGETARCH}" in \
        amd64) zig_arch=x86_64 ;; \
        arm64) zig_arch=aarch64 ;; \
        *) echo "unsupported TARGETARCH: ${TARGETARCH}" >&2; exit 1 ;; \
    esac \
    && curl -fsSL --proto '=https' --tlsv1.2 --retry 3 \
        "https://ziglang.org/download/${ZIG_VERSION}/zig-${zig_arch}-linux-${ZIG_VERSION}.tar.xz" \
        | tar -xJ -C /opt \
    && ln -s "/opt/zig-${zig_arch}-linux-${ZIG_VERSION}/zig" /usr/local/bin/zig

WORKDIR /src/tsfm-pg
COPY . .

ENV PG_CONFIG=/usr/lib/postgresql/${PG_VERSION}/bin/pg_config
RUN cpu_arg=""; \
    if [ -z "${TSFM_CPU}" ] && [ "${TARGETARCH}" = "amd64" ]; then \
        cpu_arg="-Dcpu=x86_64_v3"; \
    elif [ -n "${TSFM_CPU}" ]; then \
        cpu_arg="-Dcpu=${TSFM_CPU}"; \
    fi; \
    zig build -Doptimize=ReleaseSafe ${cpu_arg} -Dgpu="${TSFM_GPU}" -p "/usr/lib/postgresql/${PG_VERSION}"

FROM postgres:${PG_VERSION}-bookworm AS runtime
ARG PG_VERSION

COPY --from=build /usr/lib/postgresql/${PG_VERSION}/lib/tsfm.so /usr/lib/postgresql/${PG_VERSION}/lib/
# pg_config --sharedir is /usr/share/postgresql/<major> on Debian, not
# /usr/lib/postgresql/<major>/share/postgresql, which is where the build's
# `zig build -p /usr/lib/postgresql/<major>` installs them.
COPY --from=build /usr/share/postgresql/${PG_VERSION}/extension/tsfm.control \
                  /usr/share/postgresql/${PG_VERSION}/extension/
COPY --from=build /usr/share/postgresql/${PG_VERSION}/extension/tsfm--0.1.sql \
                  /usr/share/postgresql/${PG_VERSION}/extension/

# Default tsfm.model_dir is $sharedir/tsfm/models. Mount a checkpoint here for
# the slim image, or use the chronos2 target to bake Chronos-2 in. Timestral
# weights are Modified MIT with a revenue cap and are deliberately not baked;
# mount a user-supplied Timestral directory instead.
RUN mkdir -p /usr/share/postgresql/${PG_VERSION}/tsfm/models

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
FROM postgres:${PG_VERSION}-bookworm AS weights
ARG CHRONOS2_REPO=amazon/chronos-2
ARG CHRONOS2_REVISION=main
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /weights \
    && curl -fsSL --proto '=https' --tlsv1.2 --retry 3 \
        "https://huggingface.co/${CHRONOS2_REPO}/resolve/${CHRONOS2_REVISION}/config.json" \
        -o /weights/config.json \
    && curl -fsSL --proto '=https' --tlsv1.2 --retry 3 \
        "https://huggingface.co/${CHRONOS2_REPO}/resolve/${CHRONOS2_REVISION}/model.safetensors" \
        -o /weights/model.safetensors

FROM runtime AS chronos2
ARG PG_VERSION
COPY --from=weights /weights /usr/share/postgresql/${PG_VERSION}/tsfm/models

# The default target is the slim image. Use `--target chronos2` for weights.
FROM slim AS final
