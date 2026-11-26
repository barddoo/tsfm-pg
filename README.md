# tsfm-pg

`tsfm-pg` runs the [tsfm-zig](https://github.com/barddoo/tsfm-zig) forecaster
inside Postgres. It supports **Chronos-2** and **Timestral** checkpoints. The
model and weights sit on the database host, and a query never ships the series to
a cloud API.

The engine is linked as a Zig module: no C ABI, no `dlopen`, no second artifact.
`tsfm.so` is the whole extension.

## Quickstart

Requires Zig 0.17.1 and PostgreSQL 16 or 17 with server headers. `PG_HOME` is a
Postgres install prefix:

```sh
export PG_HOME=/path/to/postgres
export PG_CONFIG="$PG_HOME/bin/pg_config"

zig build -Dcpu=native -p "$PG_HOME"                 # CPU-only; see GPU below
"$PG_HOME/bin/psql" -U postgres -c 'CREATE EXTENSION tsfm'

# install-time download, never from SQL; defaults to amazon/chronos-2
scripts/fetch-model.sh "$("$PG_HOME/bin/pg_config" --sharedir)/tsfm/models"
```

The default `tsfm.model_dir` is `<sharedir>/tsfm/models`, so the fetched
checkpoint is found without further settings. Run a forecast:

```sql
SELECT step, quantile, value
FROM tsfm_forecast(
       (SELECT array_agg(value ORDER BY ts) FROM metrics WHERE name = 'sku-123'),
       90)
WHERE quantile IN (0.1, 0.5, 0.9)
ORDER BY step, quantile;
```

The engine loads lazily on first use. `tsfm_load()` warms it explicitly (useful
for connection pools). A new session is needed to change `tsfm.model_dir` or
`tsfm.model` after the engine has loaded.

## SQL API

| function | returns |
|---|---|
| `tsfm_version()` | `text`: extension and pinned engine version |
| `tsfm_model_info()` | `jsonb`: model, dir, quantiles, context/prediction lengths, threads, SIMD |
| `tsfm_model_status()` | `jsonb`: is the configured directory usable, and if not, what to do |
| `tsfm_quantiles()` | `double precision[]`: the model's quantile levels |
| `tsfm_load()` | `boolean`: explicit warmup |
| `tsfm_forecast(history, horizon)` | `TABLE(step, quantile, value)` |
| `tsfm_forecast_ex(history, horizon, n_variates = 1, past = NULL, known = NULL, cat_past = NULL, cat_known = NULL)` | `TABLE(variate, step, quantile, value)` |
| `tsfm_forecast_batch(history, horizon, n_series, n_variates = 1, past = NULL, known = NULL, cat_past = NULL, cat_known = NULL)` | `TABLE(series, variate, step, quantile, value)` |
| `tsfm_forecast_matrix(history, horizon)` | `double precision[]`, `[step][quantile]` |
| `tsfm_forecast_matrix_ex(history, horizon, n_variates = 1, past = NULL, known = NULL, cat_past = NULL, cat_known = NULL)` | `double precision[]`, `[variate][step][quantile]` |
| `tsfm_forecast_ts(timestamps, history, horizon, step_size)` | `TABLE(ts, step, quantile, value)` |

All forecast functions are `STABLE PARALLEL RESTRICTED`; `tsfm_load()` is
`VOLATILE`. The long form arrives step-major, with `step` starting at 1.

## Data conventions

- `history` is ordered oldest → newest.
- `NULL` and `'NaN'::float8` elements are missing values.
- Values are converted to float32 for the engine. Infinities and values beyond
  the float32 range become missing; integers above 2^24 lose precision, so
  scale or offset large series such as epoch seconds first.
- Arrays are flattened row-major:
  - `history`: `V × T` (variates × steps)
  - `past` / `cat_past`: `C × T`
  - `known` / `cat_known`: `C × (T + horizon)`
  - batch `history`: `B × V × T`; batch covariates: `B × C × T` or
    `B × C × (T + horizon)`
- `variate` and `series` are numbered from 1.
- Chronos-2 accepts only `n_variates = 1` with no covariates. Timestral accepts
  multiple variates and numeric/categorical covariates.

## Functions by example

Every series in a table, one forecast each. The outer executor calls the
function once per group, so `statement_timeout`, `LIMIT` and cancellation work
between series:

```sql
SELECT m.name, f.step, f.value
FROM (SELECT name, array_agg(value ORDER BY ts) AS h
      FROM metrics GROUP BY name) m,
     LATERAL tsfm_forecast(m.h, 90) f
WHERE f.quantile = 0.5;
```

Two variates with one past-only and one known-future covariate:

```sql
SELECT variate, step, quantile, value
FROM tsfm_forecast_ex(
       ARRAY[10, 11, 12, 20, 21, 22]::float8[], -- 2 variates × 3 steps
       2,
       2,
       ARRAY[0, 1, 0]::float8[],                -- 1 past covariate × 3
       ARRAY[0, 1, 0, 1, 0]::float8[],          -- 1 known × (3 + 2)
       NULL,
       NULL)
WHERE quantile IN (0.1, 0.5, 0.9);
```

`tsfm_forecast_batch` passes `B` independent series in one engine call, reusing
each weight read across the batch. It checks for query cancellation between
chunks of 64 series; one forward pass is still not interruptible:

```sql
SELECT series, step, value
FROM tsfm_forecast_batch(ARRAY[/* B × 365 values */], 90, 100, 1)
WHERE quantile = 0.5;
```

The matrix forms return one array instead of rows:

```sql
SELECT tsfm_quantiles();
SELECT tsfm_forecast_matrix(ARRAY[1, 2, 3, 4, 5]::float8[], 4);       -- [step][quantile]
SELECT tsfm_forecast_matrix_ex(ARRAY[1, 2, 3, 10, 20, 30]::float8[], 2, 2);
```

Timestamps are attached from the last value and a step:

```sql
SELECT ts, step, quantile, value
FROM tsfm_forecast_ts(
       ARRAY['2024-01-01','2024-01-02','2024-01-03']::timestamptz[],
       ARRAY[10, 11, 12]::float8[],
       7,
       interval '1 day')
WHERE quantile = 0.5;
```

`tsfm_forecast_ts` assumes a regular step; it does not inspect gaps or
calendars.

## Privileges

`CREATE EXTENSION` is superuser-only, and the extension revokes `EXECUTE` from
`PUBLIC`. Grant it explicitly to application roles:

```sql
GRANT EXECUTE ON FUNCTION tsfm_version() TO analyst;
GRANT EXECUTE ON FUNCTION tsfm_model_info() TO analyst;
GRANT EXECUTE ON FUNCTION tsfm_model_status() TO analyst;
GRANT EXECUTE ON FUNCTION tsfm_quantiles() TO analyst;
GRANT EXECUTE ON FUNCTION tsfm_load() TO analyst;
GRANT EXECUTE ON FUNCTION tsfm_forecast(double precision[], integer) TO analyst;
GRANT EXECUTE ON FUNCTION tsfm_forecast_ex(double precision[], integer, integer, double precision[], double precision[], double precision[], double precision[]) TO analyst;
GRANT EXECUTE ON FUNCTION tsfm_forecast_batch(double precision[], integer, integer, integer, double precision[], double precision[], double precision[], double precision[]) TO analyst;
GRANT EXECUTE ON FUNCTION tsfm_forecast_matrix(double precision[], integer) TO analyst;
GRANT EXECUTE ON FUNCTION tsfm_forecast_matrix_ex(double precision[], integer, integer, double precision[], double precision[], double precision[], double precision[]) TO analyst;
GRANT EXECUTE ON FUNCTION tsfm_forecast_ts(timestamptz[], double precision[], integer, interval) TO analyst;
```

`tsfm_forecast_ts` calls `tsfm_forecast`, so a role needs `EXECUTE` on both.

## Settings

Settings that decide what runs or what is read from disk are **superuser-only**
(`SUSET`): `tsfm.model_dir`, `tsfm.model`, `tsfm.threads`, `tsfm.enabled` and
`tsfm.stub`. Set them in `postgresql.conf`, with `ALTER SYSTEM`, or per
role/database with `ALTER ROLE ... SET`. That makes `tsfm.enabled = off` a real
kill switch and stops ordinary roles from pointing the server at arbitrary
files. The decoding knobs are `USERSET`, so `SET LOCAL` works for them.

`tsfm.symmetric` is an SQL keyword: write `SET tsfm."symmetric" = off`.

| GUC | default | notes |
|---|---|---|
| `tsfm.model_dir` | `<sharedir>/tsfm/models` | directory with `config.json` + `model.safetensors` |
| `tsfm.model` | `auto` | `auto`, `chronos-2`, `timestral`; `auto` reads `config.json` |
| `tsfm.threads` | `1` | 0..64, larger values clamped; `0` = auto (up to 4 workers by CPU count); fixed after load |
| `tsfm.max_context` | `0` | keep the most recent N values; `0` = model default |
| `tsfm.sort_quantiles` | `on` | per-step monotonic quantiles; off for raw parity |
| `tsfm.enabled` | `on` | kill switch; forecasts, `tsfm_load()` and `tsfm_quantiles()` error when off |
| `tsfm.stub` | `off` | naive last-value engine, for tests without weights |
| `tsfm.block_patches` | `32` | Timestral only: patches decoded per block (1..1024) |
| `tsfm.symmetric` | `on` | Timestral only: average with the mirrored forecast |
| `tsfm.clip_nonnegative` | `on` | Timestral only: clamp non-negative series at 0 |
| `tsfm.quantile_real_cap_k` | `1e4` | Timestral only: clamp to observed range + k×scale; 0 disables |
| `tsfm.extreme_value_threshold` | `1e30` | Timestral only: fed-back medians at/above this become 0 |

The engine is process-global per backend and loaded lazily. `tsfm.model_dir`
and `tsfm.model` are fixed once loaded; changing them later raises an error with
a hint to reconnect. `tsfm.threads` sizes the worker pool once.
`tsfm_model_info()` reports the values actually in use.

`tsfm.threads > 1` is opt-in. Workers touch only engine memory: they never call
`ereport`, never touch `palloc`, and never check PostgreSQL interrupts. The
default stays `1` because many backends each starting a pool can oversubscribe
the host.

The checkpoint is a read-only mmap. Every backend shares its file-backed pages
through the page cache, so the per-connection private cost is the model structs
plus forward-pass scratch (a few MB), not the weights. Engine allocations use
libc, not `palloc`, so they can outlive transactions and be used by workers. A
failed or partial checkpoint load is freed in one arena step. One forecast
result is capped at 16 million f32 values (64 MiB).

## Build and install

```sh
export PG_CONFIG="$PG_HOME/bin/pg_config"
zig build -Dcpu=native -p "$PG_HOME"
```

| option | meaning |
|---|---|
| `-Dcpu=` | target CPU (e.g. `native`, `baseline`, `x86_64_v3`) |
| `-Dtarget=` | target triple for cross-compilation |
| `-Doptimize=` / `--release=` | defaults to `ReleaseSafe` |
| `-Dgpu=` | `none` (default), `metal`, `cuda` |
| `-Dpg-port=` / `-Dpg-user=` | defaults for `pg_regress` |

`zig build` defaults to `ReleaseSafe`: a bounds or overflow bug in the model must
become a Postgres error, not a segmentation fault that takes the backend down.
Use `ReleaseFast` only for benchmarks.

Both `tsfm` and `pgzx` are fetched from the commits pinned in `build.zig.zon`;
no manual clone required.

### Dev loop script

```sh
scripts/dev.sh build      # build + install into $PG_HOME
scripts/dev.sh start      # start the local cluster if needed
scripts/dev.sh stop
scripts/dev.sh test       # engine tests, no Postgres
scripts/dev.sh regress    # build + start + pg_regress
scripts/dev.sh smoke      # one real forecast; needs TSFM_TEST_MODEL
scripts/dev.sh all
```

Env: `PG_HOME` (default `out/18`), `PGPORT` (5432), `PGUSER` (postgres),
`TSFM_CPU` (native), `TSFM_TEST_MODEL`.

## CPU and SIMD

Model GEMMs come from `tsfm-zig` built with `-Dblas=builtin`; the extension's
hot loops use Zig `@Vector` too. Vector width comes from the build's CPU, so
`-Dcpu=native` gives the best result on the build machine. `tsfm_model_info()`
reports `cpu_arch` and `simd_f32_lanes` (1 means scalar). Check a deployment
with:

```sql
SELECT tsfm_model_info()->>'simd_f32_lanes';
```

## GPU

The default build is **CPU-only**: `-Dcpu=native` only widens SIMD, it does not
enable a GPU. GPU offload is opt-in because the backend keeps a second copy of
the model's GEMM weights in device memory:

```sh
zig build -Dgpu=metal -Dcpu=native -p "$PG_HOME"   # macOS (MPS)
zig build -Dgpu=cuda  -Dcpu=native -p "$PG_HOME"   # Linux + NVIDIA driver
```

Only large matrix products are offloaded; elementwise work and small products
stay on the CPU. A product goes to the device when `m * n * k` reaches the
backend's per-call threshold: `1 << 28` on Metal for both linear and activation
GEMMs, and `1 << 20` (linear) / `1 << 24` (activation) on CUDA.

The extension is still one `tsfm.so`: the CUDA backend loads
`libcudart`/`libcublas` with `dlopen` at model-load time, so the build needs no
CUDA toolkit or link-time libraries. At runtime, a normal CUDA installation is
found automatically; set `TSFM_CUDA_LIB_DIR` to a private copy. The variable
must be in the **Postgres server process's** environment (systemd
`Environment=`, the compose service, or the shell that runs `pg_ctl`), not just
the `psql` session.

The backend is selected once, before the model is loaded. If the device or
libraries are missing, the engine uses the CPU kernels instead: CUDA logs one
warning and disables itself; a Metal operation that cannot run (no device)
returns to the CPU path for that call. `tsfm.threads` still controls the CPU
worker pool; cuBLAS/MPS parallelize the large products internally.

## Model files

Downloads are install-time, never query-time; the extension never contacts the
network from SQL.

```sh
# Chronos-2 (Apache-2.0), the default repo
scripts/fetch-model.sh /var/lib/postgresql/tsfm/chronos-2

# Timestral (Modified MIT with a revenue cap); accept the terms first
HF_TOKEN=... scripts/fetch-model.sh /var/lib/postgresql/tsfm/timestral-28m \
  mistralai/Timestral-28M
```

Then point the extension at the directory. These settings are superuser-only;
use `postgresql.conf`, `ALTER SYSTEM` or `ALTER ROLE ... SET`:

```sql
ALTER SYSTEM SET tsfm.model = 'timestral';
ALTER SYSTEM SET tsfm.model_dir = '/var/lib/postgresql/tsfm/timestral-28m';
SELECT pg_reload_conf();
```

Timestral weights are **not** redistributed or baked into the published Docker
image.

## Checking the install (`tsfm_model_status()`)

Read-only diagnostic: it stats the files in `tsfm.model_dir` and reads a small
`config.json`. It loads no weights, writes nothing and never touches the
network, so it is safe anywhere, even with `tsfm.enabled = off`.

```sql
SELECT jsonb_pretty(tsfm_model_status());
SELECT s->>'ready', s->>'problem', s->>'fix'
FROM (SELECT tsfm_model_status() s) x;
```

Fields: `model_dir`, `config_json` and `weights` (`present`, `missing`,
`unreadable`), `weights_bytes`, `configured_model`, `detected_model`, `loaded`,
`ready`, and the human-readable `problem`, `fix`, `note`. It flags missing or
unreadable files, empty/truncated weights, invalid `config.json`, and a
`tsfm.model` that disagrees with the checkpoint. The "model file not found" load
error points here.

## Tests

```sh
zig build test          # engine integration, no Postgres (stub + optional model)
zig build pg_regress    # SQL surface, no weights (tsfm.stub = on)
zig build parity        # numerical parity; needs TSFM_TEST_MODEL / TSFM_TEST_TIMESTRAL
```

`parity` runs `tests/parity.sql` and `tests/timestral_parity.sql` twice, with
`tsfm.threads=1` and `tsfm.threads=8`, and skips when the corresponding env var
is unset:

```sh
TSFM_TEST_MODEL=~/.cache/huggingface/hub/models--amazon--chronos-2/snapshots/<rev> \
TSFM_TEST_TIMESTRAL=~/.cache/huggingface/hub/models--mistralai--Timestral-28M/snapshots/<rev> \
  zig build parity
```

## Performance

Apple M4, `ReleaseSafe`, warm engine (load excluded):

| case | `tsfm.threads=1` | `tsfm.threads=8` |
|---|---|---|
| one series, 365 → 90 | 155 ms | 38 ms |
| batch of 32, 365 → 90 | 4.4 s | 1.0 s |

The forward pass is memory-bandwidth bound. `tsfm_forecast_batch` reuses each
weight read across up to 64 series per chunk; `LATERAL` remains the right choice
when series need different horizons or settings.

## Docker

```sh
docker build -t tsfm-pg:slim .                              # extension only
docker build --target chronos2 -t tsfm-pg:chronos2 .        # weights baked in (~500 MB)
docker run --rm -e POSTGRES_PASSWORD=pg tsfm-pg:chronos2
```

The weights layer uses `curl` against `huggingface.co` during `docker build`,
not at query time. Only Chronos-2 can be baked into a public image; for
Timestral, use the slim image and mount a user-supplied checkpoint at
`/usr/share/postgresql/17/tsfm/models` (or set `tsfm.model_dir`).

The entrypoint can download the checkpoint once, before Postgres starts, when
`TSFM_MODEL_REPO` is set and the models directory is empty:

```sh
docker run -d -e POSTGRES_PASSWORD=pg \
  -e TSFM_MODEL_REPO=amazon/chronos-2 \
  -e TSFM_MODEL_REVISION=main \
  -v tsfm-models:/usr/share/postgresql/17/tsfm/models \
  tsfm-pg:slim
```

`HF_TOKEN` is only needed for gated repos. The download is atomic (temporary
name, then rename), so a failed start never leaves a truncated checkpoint; the
container exits if the fetch fails. Without `TSFM_MODEL_REPO` the entrypoint
does nothing.

Examples: [`deploy/compose.yaml`](deploy/compose.yaml) and
[`deploy/k8s/postgres-tsfm.yaml`](deploy/k8s/postgres-tsfm.yaml) (init container
fetches into a shared, read-only volume). Replace the passwords and pin
`TSFM_MODEL_REVISION` to a commit.

## Deployment

Installing the extension needs access to the database host's filesystem. SQL
roles cannot do this by design: model settings are superuser-only and nothing
downloads from SQL.

| environment | works? | how |
|---|---|---|
| Self-hosted Postgres / VM | yes | operator runs `scripts/fetch-model.sh`, installs the library, sets `tsfm.model_dir` |
| Docker / Compose | yes | `tsfm-pg` image; bake Chronos-2 or fetch on first start |
| Kubernetes | yes | StatefulSet with an init container, see `deploy/k8s/` |
| Managed Postgres (RDS, Cloud SQL, Azure, Supabase, Neon, ...) | **no** | custom shared libraries and filesystem access are not available |

## Limitations

- Timestral weights are Modified MIT with a revenue cap and are not
  redistributed; the user downloads them.
- `tsfm_forecast_batch` batches independent series in one call, but there is no
  cross-series attention.
- Horizons above 1024 are not supported upstream; long-horizon unrolling is not
  ported.
- `tsfm_forecast_ts` assumes a regular timestamp step.
- This extension and `tsfm-zig` share a Zig version, so upgrades are coordinated
  across both repositories.

## Layout

```
build.zig                 extension + engine dependency, default ReleaseSafe
extension/tsfm.control    CREATE EXTENSION metadata
extension/tsfm--0.1.sql   SQL surface and REVOKEs
src/main.zig              PG_MODULE_MAGIC, function registration, _PG_init/_PG_fini
src/guc.zig               settings table
src/engine.zig            model lifetime, Chronos-2 + Timestral load/forecast, stub
src/forecast.zig          array marshalling, SRF iterators, error translation
src/cursor.zig            flat-result cursor for the SRF iterators
src/functions.zig         SQL-visible scalar functions and JSON info/status
src/engine_test.zig       engine-only tests
tests/parity.sql          Chronos-2 numerical parity fixtures
tests/timestral_parity.sql Timestral variate + covariate parity fixtures
```

## License

Apache-2.0, like `tsfm-zig`. Keep the `NOTICE` chain and the pgzx attribution.
Chronos-2 code and weights are Apache-2.0. Do not redistribute Timestral
weights without checking their Modified MIT terms.
