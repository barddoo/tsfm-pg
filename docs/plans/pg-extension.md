# A Postgres extension for tsfm

**Status:** Implemented (Chronos-2 + Timestral, batch, timestamps, Docker, parity fixtures)
**Date:** 2026-11-03
**Scope:** Ship `tsfm-zig` as a Postgres extension: forecasting as a SQL function, weights on the
database host, no data leaving the cluster. Chronos-2 (Apache-2.0) and Timestral are supported;
Timestral weights are user-supplied because of their Modified MIT terms.

## 1. Goal

The tsfm engine is a CPU-only, no-Python, single-artifact forecaster. That is interesting on its own,
but the buyer is the one who cannot ship their series to a cloud API: banks, healthcare, government,
utilities, and any company whose data-residency rules make "POST this table to a vendor" a
non-starter. Postgres is where that data already lives.

The product shape is therefore narrow and honest: **`SELECT ... FROM tsfm_forecast(...)` runs the
model next to the data.** The extension is a thin SQL adapter over the engine; the model, the
weights and the kernels stay in `tsfm-zig` and are not reimplemented here.

Non-goals for this repo: training, model serving over HTTP, a forecasting UI, model selection across
families (that router belongs to the caller), and anything that requires copying the series out of
Postgres.

## 2. What already exists

`tsfm-pg` is the pgzx starter template (`xataio/pgzx`), not yet a git repository:

- `build.zig` / `build.zig.zon` declaring `my_extension` and the `pgzx` dependency (hash still unset)
- `extension/my_extension.control`, `extension/my_extension--0.1.sql`
- `sql/my_extension_test.sql`, `expected/my_extension_test.out`
- `src/main.zig` with a `hello()` function and pgzx test registration
- a Nix devshell (`devshell.nix`, `flake.nix`) providing Postgres 16 and the pgzx scripts
  (`pglocal`, `pginit`, `pgstart`, `pgstop`)

Everything below assumes this scaffold is renamed `tsfm` and kept.

`tsfm-zig` provides `chronos2.config.parse`, `chronos2.model.Model.load`,
`chronos2.forecaster.forecast` (`b × t` in, `b × horizon × q` out, NaN = missing),
`core.safetensors` (F32 tensors used in place from a read-only mmap), `core.hub` (Hub download, not
used at query time) and `core.par` (`std.Io.Threaded` fork-join). The CLI is
`zig-out/bin/timestral`.

Both projects build with **Zig 0.17**, so there is no process or ABI boundary between them. That is
the main architectural fact below.

## 3. Architecture

### 3.1 One toolchain, one shared object

`pgzx` and `tsfm-zig` are on the same Zig version, so `tsfm-pg` takes `.tsfm` as a build dependency,
imports its modules, and produces a single `tsfm.so` that Postgres loads:

```
tsfm.so (Zig 0.17)
├── pgzx              SQL types, GUCs, SRF, error mapping
└── tsfm (src/lib.zig)
    ├── core          safetensors mmap, nn kernels, par, io
    └── chronos2      config, model, forecaster
        └── model.safetensors (mmap, read-only)
```

What this buys compared with a separate engine library:

- `chronos2.forecaster.forecast` is called directly. No `float` marshalling, no `tsfm_*` ABI to
  version, no second artifact to install, no `dlopen` and no `LD_LIBRARY_PATH`.
- One `zig build`, one toolchain, one file to ship.
- The allocator and `std.Io` are created in the extension, next to the PG code that uses them, not
  in a foreign runtime with its own panic handler.

Build wiring has one rough edge: pgzx's `Project` helper installs its extension lib without exposing
a hook for extra imports (`configureLib` adds only `pgzx` and `build_options`). In `build.zig`:

```zig
const tsfm_dep = b.dependency("tsfm", .{ .target = target, .optimize = optimize, .blas = .builtin });
const proj = PGBuild.Project.init(b, .{ .name = "tsfm", .version = .{ .major = 0, .minor = 1 },
                                         .root_dir = "src/", .root_source_file = "src/main.zig" });

const lib = proj.extensionLib();                       // build the lib once
lib.root_module.addImport("tsfm", tsfm_dep.module("timestral"));
proj.pgbuild.addInstallExtensionLibArtifact(lib, proj.config.name);
```

`src/lib.zig` already re-exports `core` and `chronos2`, so one import is enough. The unit-test lib
(`createLib("{name}_unit", ...)`) needs the same import; the cleanest fix is a five-line
`Project.addImport` upstream in pgzx, otherwise duplicate the two lines above.

Keep **`-Dblas=builtin` as the default for the extension**: one self-contained `.so` with no
OpenBLAS in the Postgres install. `accelerate` (macOS dev) and `openblas` (benchmarks) stay
available. If OpenBLAS is ever worth it for throughput, it becomes an explicit build variant, not
the default.

A C ABI in `tsfm-zig` is still a reasonable later addition for non-Zig consumers, but the extension
does not need it and v1 should not carry the extra surface.

### 3.2 Model lifetime: lazy load, mmap, never in the postmaster

Postgres forks a backend per connection, so "load once per process" means once per backend, and the
naive reading is 480 MB × `max_connections`.

That is not what happens, because `core.safetensors` uses F32 tensors in place from a read-only
`mmap`: the 480 MB is clean, file-backed and shared by every backend through the page cache. Each
backend's private cost is the model structs plus the forward-pass scratch (a few MB), not the
weights. Two consequences:

- **Chronos-2 is the right default**: its published checkpoint is F32, so nothing is copied to the
  heap. The published Timestral checkpoints are F32 as well, so they are also mmap-shared; the
  reason Timestral is not the default is licensing (Modified MIT with a revenue cap), not memory.
  F16/BF16 checkpoints would be converted per backend, which is another reason to keep F32 as the
  supported shape.
- **Do not preload in the postmaster.** A crash while loading would take down the cluster, and the
  postmaster must stay single-threaded. Load on the first `tsfm_forecast` (or an explicit
  `tsfm_load()`), after the fork.

`shared_preload_libraries` is not required. `_PG_init` only registers GUCs.

### 3.3 Threads

Postgres backends are single-threaded and not thread-safe; `std.Io.Threaded` is only safe if the
workers touch nothing but engine memory.

Default `tsfm.threads = 1`; `0` means auto (up to four workers based on the logical CPU count). The
default stays conservative because many backends each starting a pool can oversubscribe the host.
**Construct `std.Io.Threaded` anyway**: it is the `Io` implementation
behind file reads and `mmap`, not just the worker pool (`main.zig` does the same with
`async_limit = .limited(threads - 1)`, i.e. zero workers at `threads = 1`). Only the `Io` passed to
`forecast` switches — `threaded.io()` when `threads > 1`, `std.Io.Threaded.global_single_threaded.io()`
otherwise. So the single-threaded path has no pool and no signals but still has a normal `Io`.

`tsfm.threads > 1` is opt-in and documented with the constraints: workers never call `ereport`,
never touch `palloc`, never check interrupts, and the engine installs no signal handlers.

### 3.4 Errors: unwind before `ereport`

A Zig `defer` is not run when Postgres `longjmp`s out of `ereport`. Raising a Postgres error from
inside the engine would leak every allocation and lock held in the skipped frames.

Rule: engine-facing functions in `src/engine.zig` return Zig errors and never call `ereport`. The
SQL-facing layer translates an error to `ereport(ERROR, ...)` only after every engine frame has
returned. Attach `detail`/`hint` where they help (`model file not found: ...`, `horizon must be
1..1024`), and map `error.OutOfMemory` separately: it usually means the model was not loaded.

## 4. Engine integration

`src/engine.zig` owns the same sequence the CLI's `run()` does, minus argument parsing and the Hub
client. One process-global engine, lazily initialized under a mutex, freed in `_PG_fini`:

```zig
const std = @import("std");
const tsfm = @import("tsfm");                 // src/lib.zig, re-exports core + chronos2

const Engine = struct {
    allocator: std.mem.Allocator,             // std.heap.c_allocator: outlives transactions
    threaded: std.Io.Threaded,                // always constructed, also when threads == 1
    io: std.Io,                               // threaded.io(): config, safetensors, mmap
    par_io: std.Io,                           // threaded.io() when threads > 1, else single-threaded
    weights: tsfm.safetensors.Weights,
    model: tsfm.chronos2.model.Model,
    quantiles: []const f64,
    family: enum { chronos2, timestral },
};

pub fn load(dir: []const u8, threads: usize, max_context: ?usize) !void {
    // Mirror the CLI: the pool exists regardless of thread count, because it is also the `Io`
    // behind file access and mmap. Only the `Io` handed to `forecast` switches.
    g.threaded = .init(g.allocator, .{ .async_limit = .limited(threads - 1) });
    g.io = g.threaded.io();
    g.par_io = if (threads > 1) g.io else std.Io.Threaded.global_single_threaded.io();

    const cfg_text = try readFile(g.io, dir ++ "/config.json");
    const is_c2 = try tsfm.chronos2.config.isChronos2(g.allocator, cfg_text);
    const parsed = try tsfm.chronos2.config.parse(g.allocator, cfg_text);   // Chronos-2 path
    // `Weights` must outlive the model and keep a stable address: both are Engine fields,
    // never locals that `Model` would point into. Deinit in reverse order.
    g.weights = try tsfm.safetensors.Weights.open(g.io, g.allocator, dir ++ "/model.safetensors");
    g.model = try tsfm.chronos2.model.Model.load(g.io, g.allocator, parsed.value, &g.weights);
}
```

Two details borrowed from `main.zig` that are easy to get wrong:

- `Engine` is a global, so `g.threaded` has a stable address; `Threaded.io()` may capture a pointer
  to it, so never copy the `Threaded` value after `init`.
- `forecast` takes `par_io`, not `io` (line 355 of `main.zig`); the `io` from `threaded.io()` is for
  files, mmap and diagnostics.

Forecasting converts the SQL array to `f32` (the engine's dtype), calls
`forecast(allocator, par_io, &model, inputs, b, t, horizon)`, and copies `[series][step][quantile]` into
the SRF's output. The result slice is freed before returning; nothing per-call survives the
transaction. The batch entry point passes `b > 1`, which Chronos-2 already supports through its
independent-group path.

Because the extension links the engine directly, there is no ABI to freeze, but there is a **version
coupling**: `tsfm-pg` and `tsfm-zig` must move together. Pin the dependency by hash in
`build.zig.zon` and have `tsfm_version()` report both the extension version and the pinned engine
commit.

## 5. SQL surface (v1)

### Functions

```sql
-- engine
tsfm_version()                                   RETURNS text
tsfm_model_info()                                RETURNS jsonb
tsfm_quantiles()                                 RETURNS double precision[]
tsfm_load()                                      RETURNS boolean   -- explicit warmup

-- forecasting
tsfm_forecast(history double precision[], horizon integer)
  RETURNS TABLE (step integer, quantile double precision, value double precision)

tsfm_forecast_matrix(history double precision[], horizon integer)
  RETURNS double precision[]        -- [step][quantile], step-major; one row to transfer

-- extended (Timestral): multiple variates and covariates
tsfm_forecast_ex(history double precision[], horizon integer,
                 n_variates integer DEFAULT 1,
                 past double precision[] DEFAULT NULL,
                 known double precision[] DEFAULT NULL,
                 cat_past double precision[] DEFAULT NULL,
                 cat_known double precision[] DEFAULT NULL)
  RETURNS TABLE (variate integer, step integer, quantile double precision, value double precision)

-- batch of independent series
tsfm_forecast_batch(history double precision[], horizon integer,
                    n_series integer, n_variates integer DEFAULT 1,
                    past double precision[] DEFAULT NULL, known double precision[] DEFAULT NULL,
                    cat_past double precision[] DEFAULT NULL, cat_known double precision[] DEFAULT NULL)
  RETURNS TABLE (series integer, variate integer, step integer, quantile double precision, value double precision)

tsfm_forecast_matrix_ex(history double precision[], horizon integer, n_variates integer DEFAULT 1, ...)
  RETURNS double precision[]        -- [variate][step][quantile]

tsfm_forecast_ts(timestamps timestamptz[], values double precision[],
                 horizon integer, step_size interval)
  RETURNS TABLE (ts timestamptz, step integer, quantile double precision, value double precision)
```

- `history` is ordered oldest → newest. `'NaN'::float8` or `NULL` elements are missing values, which
  Chronos-2 handles natively (masks). The adapter maps `NULL` → `NaN` before the call.
- Quantiles are the model's configured levels (21 for Chronos-2). The long form is ordered `step`,
  then `quantile`, matching the engine's layout, so no sort is needed.
- Quantiles can cross. Default `tsfm.sort_quantiles = on` sorts per step (monotonic), because that
  is what every consumer assumes; turn it off for bit-parity with the Python pipeline.
- Errors: empty/`NULL` history, `horizon < 1`, `horizon > 1024`, model not installed, engine
  disabled → a normal Postgres error with a hint.

### GUCs (`tsfm.*`, registered in `_PG_init`)

| GUC | type | default | notes |
|---|---|---|---|
| `tsfm.model_dir` | string | `$sharedir/tsfm/models` | directory containing `config.json` + `model.safetensors` |
| `tsfm.model` | enum | `auto` | `auto` (read `architectures`), `chronos-2`, `timestral` |
| `tsfm.threads` | int | `1` | clamp `[1, 64]`; `>1` has the caveats in §3.3 |
| `tsfm.max_context` | int | model default | truncate from the left, like `--max-context` |
| `tsfm.sort_quantiles` | bool | `on` | per-step monotonic quantiles |
| `tsfm.enabled` | bool | `on` | kill switch; forecast errors when off |
| `tsfm.stub` | bool | `off` | naive last-value engine, for tests without weights |
| `tsfm.block_patches` | int | `32` | Timestral only; patches decoded per block |
| `tsfm.symmetric` | bool | `on` | Timestral only; mirrored averaging |
| `tsfm.clip_nonnegative` | bool | `on` | Timestral only; clamp non-negative series |
| `tsfm.quantile_real_cap_k` | real | `1e4` | Timestral only; observed-range clamp, 0 disables |
| `tsfm.extreme_value_threshold` | real | `1e30` | Timestral only; fed-back medians at/above become 0 |

`SET LOCAL` works, so a transaction can pick a different model directory, but the engine is global:
changing `model_dir` after load errors with `hint = "call tsfm_load() in a new session"`. Prefer
making the path a per-call argument later if that proves annoying.

### Usage

```sql
CREATE EXTENSION tsfm;
SET tsfm.model_dir = '/var/lib/postgresql/tsfm/chronos-2';

-- one series, three quantiles
SELECT step, quantile, value
FROM tsfm_forecast(
       (SELECT array_agg(value ORDER BY ts) FROM metrics WHERE name = 'sku-123'),
       90)
WHERE quantile IN (0.1, 0.5, 0.9)
ORDER BY step, quantile;

-- every series in a table, one forecast each
SELECT m.name, f.step, f.value
FROM (SELECT name, array_agg(value ORDER BY ts) AS h FROM metrics GROUP BY name) m,
     LATERAL tsfm_forecast(m.h, 90) f
WHERE f.quantile = 0.5;
```

The outer executor calls the function once per group, so `statement_timeout`, `Cancel` and
`LIMIT` all work between series. A single call is uninterruptible; keep calls short (see §6).

## 6. Performance

Per call the engine streams the weight matrix once per forward pass; with ~30 tokens (365-step
context, 90-step horizon) that is ~480 MB of reads for Chronos-2, i.e. memory-bandwidth bound, not
FLOP bound. Three consequences:

- Single series, single thread: expect **tens to low hundreds of milliseconds**. Fill in the table
  on an M4 and on the Linux image in M1 — do not guess.
- `tsfm_forecast_batch` with 16–64 rows reuses each weight read across rows and is the only way to
  forecast thousands of series in a query. The engine already supports `b > 1`.
- Keep the first call's load time (config parse + `mmap` + struct build) out of the benchmark; report
  warm numbers, like the CLI's `--time`.

| case | M4, `tsfm.threads=1` | M4, `threads=8` | Linux/builtin |
|---|---|---|---|
| one series (365 → 90) | 155 ms | 38 ms | |
| batch of 32 (365 → 90) | 4.4 s | 1.0 s | |
| 1k series, batch 32 | not measured | not measured | |

Measured with the ReleaseSafe extension on an Apple M4, warm engine, model load excluded. Use
`LATERAL` over groups until `tsfm_forecast_batch` lands.

## 7. Repository layout

```
tsfm-pg/
  build.zig                 # project renamed to `tsfm`; adds the `.tsfm` dependency + import
  build.zig.zon             # .tsfm pinned by url + hash
  Dockerfile                # PG 17 + extension (+ optional weights layer)
  extension/tsfm.control
  extension/tsfm--0.1.sql
  sql/tsfm_test.sql
  expected/tsfm_test.out
  src/
    main.zig                # PG_MODULE_MAGIC, PG_FUNCTION_V1, _PG_init / _PG_fini
    guc.zig                 # the table in §5
    engine.zig              # load, forecast, batch, error mapping, version
    forecast.zig            # SRF iterators, array marshalling, quantile sorting
  tests/
    fixtures/               # series + expected quantiles from the Python reference
    reference/              # copy of tsfm-zig's tests/reference_chronos2.py
  docs/plans/pg-extension.md
```

`build.zig.zon` gains:

```zig
.tsfm = .{
    .url = "git+https://github.com/<you>/tsfm-zig#<commit>",
    .hash = "tsfm-0.1.0-...",
},
```

and the scaffold's `pgzx` hash still needs filling in (the `zig build` error message prints it, as
the template README describes).

## 8. Postgres integration details

- **Arrays**: `float8[]` → `deconstruct_array` → `float[]` (one conversion, engine is f32);
  output via the SRF iterator. `NULL` array element → `NaN`.
- **SRF**: `SETOF`/`TABLE` per the pgzx `srf` example. Allocate the result in the multi-call memory
  context; never hold the engine pointer in it.
- **Function marks**: `STABLE`, not `IMMUTABLE` (results depend on the GUCs and the model file);
  `PARALLEL RESTRICTED` (process-global engine and optional threads); not `STRICT` (a `NULL` history
  gets a clear error, not `NULL`).
- **Privileges**: `CREATE EXTENSION` is superuser-only; the extension SQL does
  `REVOKE ALL ON FUNCTION ... FROM PUBLIC` and the admin grants `EXECUTE`. A 480 MB model and a CPU
  core per call should not be reachable by every role by default.
- **Interrupts**: between rows only (see §5). Optional later: pass a `volatile sig_atomic_t *cancel`
  into the engine and check it per layer.
- **Memory**: the engine allocates with `malloc`/`mmap` (`std.heap.c_allocator`), not `palloc`, so it
  survives `MemoryContextReset`; this is intentional (the model outlives the transaction) and must
  be freed in `_PG_fini`: `model.deinit`, `weights.close(io)`, then `threaded.deinit()`, in that
  order. Everything per-call is `palloc` in the multi-call context.
- **Safety**: build the extension `ReleaseSafe`, not `ReleaseFast`. A bounds/overflow bug in the
  model must become a Postgres error, not a segmentation fault that takes the backend (and, with
  `restart_after_crash`, the cluster) down. Budget for fuzzing `history`/`horizon` before release;
  this is the main product risk.
- **Signals**: the engine installs none. Zig's panic handler aborts the backend; treat any abort as a
  release blocker.
- **Fork safety**: never call `engine.load` from `_PG_init` if `_PG_init` runs before the fork (it does
  not for a normal extension load, but would for a preload); lazy load on first use is the invariant.

## 9. Testing

Three layers, cheapest first:

1. **Engine integration, no Postgres** (`zig build test` in `tsfm-pg`, or a small test step): load the
   model through `engine.zig`, forecast a fixture, compare with the CLI/Python output to 1e-6.
   This proves the dependency wiring before any PG type is involved.
2. **SQL surface, no weights** (`tsfm.stub = on`, naive last-value): `pg_regress` covers array
   marshalling, `NULL`/`NaN`, quantile ordering, error messages, GUC clamping, `REVOKE`.
3. **Numerical parity, with weights** (`TSFM_TEST_MODEL=1`): fixtures generated by
   `tsfm-zig/tests/reference_chronos2.py` and compared with `abs(a-b) <= 1e-3` in plpgsql (pg_regress
   text comparison cannot express a tolerance). Cases from the Chronos-2 plan: 365→90, length not a
   multiple of 16, NaNs at start/middle, constant series, all-zero series, context > 8192,
   horizon 1024, plus `tsfm.threads=1` vs `8` equal.

Also: a soak test (10k calls in one session, RSS flat), a 50-connection test (RSS ≈ scratch only,
proving the mmap sharing), and a `statement_timeout` test over a table of series.

CI: Linux x86_64, PG 16/17/18, `-Dblas=builtin` and `openblas`; macOS dev only. The weights job is
cached separately and allowed to be slow.

## 10. Packaging

- **Docker**: `postgres:17` + `tsfm.so` + PGXS-installed SQL. Two tags: `-slim` (no weights, mount or
  `tsfm-fetch`) and `-chronos2` (weights baked, ~500 MB).
- **Nix**: the devshell already exists; add a `packages.tsfm` derivation and a
  `postgresql_17.withPackages` overlay, which is the path for `nix run`.
- **PGXN/deb/rpm**: after M5, not before. The extension is one `.so` and two SQL files; there is no
  reason to block on a package format.
- **Downloads are install-time, never query-time**: a SQL statement must not make outbound requests.
  `tsfm-fetch chronos-2` uses `tsfm-zig`'s Hub client, or the Docker layer, or a mounted directory.

## 11. Milestones

Each gate is demonstrable; do not start the next one before it passes.

| | what | gate |
|---|---|---|
| M0 | rename scaffold to `tsfm`, add `.tsfm` dependency, import `timestral`, forecast a CSV from a test step with no Postgres | output matches the CLI to 1e-6; `zig build test` green |
| M1 | link into the extension, GUCs, `tsfm_version()`, `tsfm_model_info()`, `tsfm_load()` | `CREATE EXTENSION tsfm; SELECT tsfm_version();` on PG 17 in Docker; `pglocal`/`pginit` flow works |
| M2 | `tsfm_forecast` SRF, single series, Chronos-2, `NaN`, quantile reorder/sort | manual forecast of a real series in `psql`; errors are readable |
| M3 | parity fixtures, stub mode, `REVOKE`, memory soak, 50-connection RSS | `zig build pg_regress` green; RSS per backend < 40 MB after warmup |
| M4 | `tsfm_forecast_batch`, thread caveats, Timestral, interrupt check, benchmarks | throughput table filled; `threads=1` vs `8` equal |
| M5 | Docker/Nix packaging, docs, versioned release | `docker run ... -c "SELECT ..."` on a clean machine |

M0 is the risky one (dependency wiring, pgzx's `addImport` gap). Timebox it to two days; if the
module import fights the pgzx build helper, the fallback is a tiny local patch to
`Project.configureLib` rather than an architectural change.

## 12. Risks and open questions

1. **Version coupling** — `tsfm-pg` and `tsfm-zig` now share a compiler version, so a Zig upgrade is
   a coordinated change across both repos. Pin `.tsfm` by commit, and make `tsfm_version()` report
   it. This is cheaper than the two-toolchain design, but it is a real constraint to write down.
2. **A backend abort takes the cluster down.** Budget time for input fuzzing, `ReleaseSafe`, and,
   if feasible, a panic hook that logs the model/horizon before aborting. This is the main
   "product" risk, more than speed.
3. **Which model first?** Chronos-2 is Apache-2.0, F32 (no per-backend copies) and univariate — the
   clean default. Timestral brings covariates and multivariate targets, which are genuinely useful
   in SQL, but its weights are Modified MIT with a revenue cap, so a commercial deployment needs a
   licensing conversation. Chronos-2 remains the default; Timestral is available behind
   `tsfm.model = 'timestral'`, and its weights are user-supplied rather than redistributed.
4. **Long horizon**: > 1024 is unsupported by the engine. Document it; do not promise rolling
   forecasts until the autoregressive unroll lands upstream.
5. **Timestamps**: implemented as `tsfm_forecast_ts(timestamps, values, horizon, step_size)`, which
   attaches a regular step to the last timestamp. Gaps, calendars and irregular timestamps are not
   interpreted.
6. **Batching in SQL**: `tsfm_forecast_batch` takes flattened `B x V x T` arrays and returns a long
   table. If that proves awkward for very wide batches, the next step is an SRF over `record`/JSON
   rather than widening the matrix types.
7. **Quantile crossing**: sorting hides the model's raw output. Keep the GUC so the parity tests can
   turn it off, and document which shape the product promises.

## 13. Out of scope

- Cross-series/group attention; Timestral known-future covariates are supported when the checkpoint's
  block schedule allows them
- Long-horizon unrolling, fine-tuning, embedding extraction
- Logical replication of models, `pg_dump` of weights (they are a directory, not a table)
- Windows builds (Postgres extensions are built per-platform; start with Linux and macOS)
- A C ABI for non-Zig consumers (possible later, not needed for the extension)
- A hosted/managed offering — this design exists for the opposite case

## 14. Licensing

- This extension: Apache-2.0, like `tsfm-zig`. Keep the `NOTICE` chain and add the pgzx attribution.
- Chronos-2 weights and code: Apache-2.0. Safe to redistribute in the `-chronos2` image.
- Timestral weights: Modified MIT with a revenue cap. Do **not** bake them into a public image
  without checking the terms; document that the user fetches them.

## 15. Commercial note

The extension itself should be open source — it is the distribution mechanism, not the moat, and a
closed Postgres extension is a support burden with no upside. What can be sold around it:

- **Support and packaging** for on-prem deployments (the buyer is a bank/gov/health org that needs a
  vendor to call; annual contract, not per-seat).
- **A verified, reproducible image** with pinned weights, checksums and a benchmark report.
- **Integration work**: schema conventions, `LATERAL` query patterns, quantile-to-scenario
  conversion, cron/batch orchestration inside the database host.

Price as an annual license or support retainer, not per query. The differentiator is "the data never
leaves your Postgres," which is a compliance sale, not a modeling sale.
