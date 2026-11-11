//! Process-global inference engine.
//!
//! One model per backend, loaded lazily on first use. The weights are a
//! read-only mmap, so the checkpoint is shared through the page cache; each
//! backend only pays for the model structs and the forward-pass scratch.
//!
//! Chronos-2 and Timestral are both supported. `tsfm.model = 'auto'` reads the
//! checkpoint's `architectures`; a config that is not Chronos-2 is parsed as
//! Timestral. Timestral additionally accepts past/known/categorical covariates
//! and multiple target variates.
//!
//! This file deliberately does not import pgzx. Engine-facing functions return
//! Zig errors and never call `ereport`, so every `defer` has run before the SQL
//! layer turns an error into a Postgres error (a `longjmp` skips defers).

const std = @import("std");
const tsfm = @import("tsfm");
const simd = @import("simd.zig");

const Allocator = std.mem.Allocator;

/// Quantile levels used by the stub engine. They match both published
/// checkpoints, so SQL tests exercise the same result shape without weights.
pub const stub_quantiles = [_]f64{
    0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
    0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.99,
};

pub const Family = enum { chronos2, timestral };
/// The `tsfm.model` setting. Backed by a C int because the GUC layer stores it.
pub const ModelChoice = enum(c_int) { auto = 0, chronos2 = 1, timestral = 2 };

/// Errors raised by this file. The SQL layer (`forecast.mapError`) also matches
/// a few names that come from the tsfm-zig module (CovariatesUnsupported,
/// KnownCovariatesUnsupported, AsymmetricQuantiles, DegenerateInput,
/// InvalidConfig, ...). Those are matched by name, so an upstream rename
/// degrades to a generic internal error that carries the name as its detail.
pub const Error = error{
    ModelNotLoaded,
    ModelDirChanged,
    ModelFamilyChanged,
    ModelFileNotFound,
    /// The engine is already executing (re-entrant call).
    Busy,
    InvalidThreads,
    EmptyHistory,
    /// Horizon outside 1..1024.
    HorizonTooLong,
    /// Horizon is within 1..1024 but beyond what the loaded model predicts.
    HorizonExceedsModel,
    MultivariateUnsupported,
    InvalidCovariateShape,
    CovariatesUnsupported,
    KnownCovariatesUnsupported,
    AsymmetricQuantiles,
    DegenerateInput,
    ResultTooLarge,
};

/// Timestral decoding options. Chronos-2 ignores them.
pub const TimestralOptions = struct {
    /// Most recent context steps to keep; 0 = the config's context length.
    max_context: usize = 0,
    block_patches: usize = 32,
    symmetric: bool = true,
    clip_nonnegative: bool = true,
    /// Clamp to `[min - k * s, max + k * s]` of the observed context; 0 disables.
    quantile_real_cap_k: f32 = 1e4,
    /// Fed-back medians at or above this magnitude become 0.
    extreme_value_threshold: f32 = 1e30,
};

/// Optional cooperative cancellation, checked between batch chunks. The
/// callback runs on the calling (Postgres) thread, not on engine workers.
pub const CancelFn = *const fn (ctx: ?*anyopaque) anyerror!void;

/// Covariates for one series (b = 1). Numeric arrays are `(C, T)` float32 with
/// NaN for missing; categorical arrays are `(C, T)` float64 codes (NaN =
/// missing). Past arrays have the context length; known arrays have
/// `T + horizon`.
pub const Covariates = struct {
    past: ?[]const f32 = null,
    n_past: usize = 0,
    known: ?[]const f32 = null,
    n_known: usize = 0,
    cat_past: ?[]const f64 = null,
    n_cat_past: usize = 0,
    cat_known: ?[]const f64 = null,
    n_cat_known: usize = 0,

    pub fn any(self: Covariates) bool {
        return self.n_past > 0 or self.n_known > 0 or self.n_cat_past > 0 or self.n_cat_known > 0;
    }
};

const Parsed = union(Family) {
    chronos2: tsfm.chronos2.config.Parsed,
    timestral: tsfm.cfg.Parsed,
};

const LoadedModel = union(Family) {
    chronos2: tsfm.chronos2.model.Model,
    timestral: tsfm.model.Model,
};

const Engine = struct {
    allocator: Allocator = std.heap.c_allocator,
    threaded: std.Io.Threaded = undefined,
    // True only when `threaded` was initialized (threads > 1). The default
    // single-threaded configuration borrows the std-global `Io` instead, so it
    // never installs a SIGIO handler or allocates executor state.
    threaded_live: bool = false,
    io: std.Io = undefined,
    par_io: std.Io = undefined,
    weights: tsfm.safetensors.Weights = undefined,
    model: LoadedModel = undefined,
    // Allocates the model structs (layers, qkv permutations, inv_freq). An
    // arena makes a failed load leak-free even if the model loader errors
    // halfway through: deinit frees every partial allocation at once. The
    // default arena has no buffers; `ensure` replaces it with the engine
    // allocator when a model is actually loaded.
    model_arena: std.heap.ArenaAllocator = std.heap.ArenaAllocator.init(std.heap.c_allocator),
    parsed: Parsed = undefined,
    family: Family = .chronos2,
    quantiles: []const f64 = &.{},
    model_dir: []u8 = &.{},
    // JSON string slices in the parsed config point into this buffer, so it
    // must outlive the model (Timestral uses block_schedule during inference).
    config_text: []u8 = &.{},
    threads: usize = 0,
    context_length: usize = 0,
    prediction_length: usize = 0,
    loaded: bool = false,
};

var g: Engine = .{};
// A backend is single-threaded, so this is only a guard against re-entrancy
// (for example a cancel callback that calls back into the engine). Contention
// is reported as `Error.Busy`: spinning would hang the backend without ever
// checking for interrupts.
var mutex: std.atomic.Mutex = .unlocked;

fn lock() !void {
    if (!mutex.tryLock()) return Error.Busy;
}

fn unlock() void {
    mutex.unlock();
}

/// Load the checkpoint in `dir` if no engine is loaded yet. Repeating the same
/// call is a no-op; changing the directory or model family after load is an
/// error (the model is process-global, so a new session is the only clean way
/// to switch). A different `threads` after load is ignored: the worker pool is
/// sized once and `loadedThreads()` reports what is in effect.
pub fn ensure(dir: []const u8, threads: usize, choice: ModelChoice) !void {
    try lock();
    defer unlock();

    if (g.loaded) {
        if (!std.mem.eql(u8, g.model_dir, dir)) return Error.ModelDirChanged;
        if (choice != .auto) {
            const chosen: Family = switch (choice) {
                .chronos2 => .chronos2,
                .timestral => .timestral,
                .auto => unreachable,
            };
            if (chosen != g.family) return Error.ModelFamilyChanged;
        }
        return;
    }

    if (threads < 1 or threads > 64) return Error.InvalidThreads;

    g.allocator = std.heap.c_allocator;
    const model_dir = try g.allocator.dupe(u8, dir);
    var dir_live = true;
    errdefer if (dir_live) g.allocator.free(model_dir);

    // Default to the process-wide single-threaded `Io`: synchronous file I/O,
    // no worker pool, no signal handlers. Only a multi-threaded configuration
    // needs a `Threaded` executor. It is created here, in the backend, after
    // fork -- never in `_PG_init` under shared_preload_libraries.
    g.io = std.Io.Threaded.global_single_threaded.io();
    g.par_io = g.io;
    if (threads > 1) {
        // The `Threaded` value must never move after `init`: `io()` may capture
        // a pointer to it, so it is initialized in place, in the global engine.
        // The pool is sized once and reused by every query in this backend.
        g.threaded = .init(g.allocator, .{ .async_limit = .limited(threads - 1) });
        g.threaded_live = true;
        g.io = g.threaded.io();
        g.par_io = g.io;
    }
    errdefer if (g.threaded_live) {
        g.threaded.deinit();
        g.threaded_live = false;
    };
    const prev_par_threads = tsfm.par.threads;
    errdefer tsfm.par.threads = prev_par_threads;
    tsfm.par.threads = threads;

    const cfg_path = try std.fmt.allocPrint(g.allocator, "{s}/config.json", .{dir});
    defer g.allocator.free(cfg_path);
    const cfg_text = tsfm.sys.readFile(g.io, g.allocator, cfg_path) catch |e| switch (e) {
        error.FileNotFound => return Error.ModelFileNotFound,
        else => return e,
    };
    var cfg_live = true;
    errdefer if (cfg_live) g.allocator.free(cfg_text);

    const family: Family = switch (choice) {
        .chronos2 => .chronos2,
        .timestral => .timestral,
        .auto => detectFamily(g.allocator, cfg_text) orelse return error.InvalidConfig,
    };

    var parsed: Parsed = undefined;
    var parsed_live = false;
    switch (family) {
        .chronos2 => {
            parsed = .{ .chronos2 = try tsfm.chronos2.config.parse(g.allocator, cfg_text) };
            parsed_live = true;
        },
        .timestral => {
            parsed = .{ .timestral = try tsfm.cfg.parse(g.allocator, cfg_text) };
            parsed_live = true;
        },
    }
    errdefer if (parsed_live) deinitParsed(family, &parsed);

    const w_path = try std.fmt.allocPrint(g.allocator, "{s}/model.safetensors", .{dir});
    defer g.allocator.free(w_path);
    g.weights = tsfm.safetensors.Weights.open(g.io, g.allocator, w_path) catch |e| switch (e) {
        error.FileNotFound => return Error.ModelFileNotFound,
        else => return e,
    };
    var weights_live = true;
    errdefer if (weights_live) g.weights.close(g.io);

    var model: LoadedModel = undefined;
    var model_live = false;
    g.model_arena = std.heap.ArenaAllocator.init(g.allocator);
    var arena_live = true;
    errdefer if (arena_live) g.model_arena.deinit();
    switch (family) {
        .chronos2 => {
            model = .{ .chronos2 = try tsfm.chronos2.model.Model.load(g.io, g.model_arena.allocator(), tsfm.core.nn.defaultBackend(), parsed.chronos2.value, &g.weights) };
            model_live = true;
        },
        .timestral => {
            model = .{ .timestral = try tsfm.model.Model.load(g.io, g.model_arena.allocator(), tsfm.core.nn.defaultBackend(), parsed.timestral.value, &g.weights) };
            model_live = true;
        },
    }
    errdefer if (model_live) deinitModel(family, &model, g.model_arena.allocator());

    g.parsed = parsed;
    g.model = model;
    g.family = family;
    g.model_dir = model_dir;
    g.config_text = cfg_text;
    switch (family) {
        .chronos2 => {
            const c = parsed.chronos2.value;
            g.quantiles = c.chronos_config.quantiles;
            g.context_length = c.chronos_config.context_length;
            g.prediction_length = c.predictionLength();
        },
        .timestral => {
            const c = parsed.timestral.value;
            g.quantiles = c.quantile_knots;
            g.context_length = c.context_length;
            // Timestral decodes block by block; there is no fixed prediction
            // length. The SQL layer caps horizons at 1024.
            g.prediction_length = 0;
        },
    }
    g.threads = threads;
    g.loaded = true;
    dir_live = false;
    parsed_live = false;
    weights_live = false;
    model_live = false;
    arena_live = false;
    cfg_live = false;
}

/// Which family a `config.json` describes, or null when it is not a JSON
/// object. Mirrors `auto`: anything that is not Chronos-2 is tried as Timestral.
/// (tsfm-zig's own `isChronos2` indexes the root as an object without checking,
/// so a config such as `[]` would trap; this checks first.)
pub fn detectFamily(allocator: Allocator, text: []const u8) ?Family {
    const parsed = std.json.parseFromSlice(std.json.Value, allocator, text, .{}) catch return null;
    defer parsed.deinit();
    if (parsed.value != .object) return null;
    const arch = parsed.value.object.get("architectures") orelse return .timestral;
    if (arch != .array) return .timestral;
    for (arch.array.items) |item| {
        if (item == .string and std.mem.eql(u8, item.string, "Chronos2Model")) return .chronos2;
    }
    return .timestral;
}

pub const FileState = enum { present, missing, unreadable };

/// What is on disk in a model directory. Read-only: it never loads weights.
pub const Status = struct {
    config: FileState = .missing,
    weights: FileState = .missing,
    weights_bytes: u64 = 0,
    /// Null when config.json is absent, unreadable, or not a JSON object.
    family: ?Family = null,
};

/// Inspect `dir` for `config.json` and `model.safetensors`. Touches only the
/// file system (stat, and a bounded read of config.json), so it is safe to call
/// before, after, or without loading the engine.
pub fn inspect(dir: []const u8) !Status {
    const allocator = std.heap.c_allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    var st: Status = .{};

    const w_path = try std.fmt.allocPrint(allocator, "{s}/model.safetensors", .{dir});
    defer allocator.free(w_path);
    if (std.Io.Dir.cwd().statFile(io, w_path, .{})) |s| {
        st.weights = .present;
        st.weights_bytes = s.size;
    } else |e| {
        st.weights = if (e == error.FileNotFound or e == error.NotDir) .missing else .unreadable;
    }

    const cfg_path = try std.fmt.allocPrint(allocator, "{s}/config.json", .{dir});
    defer allocator.free(cfg_path);
    // A model config is a few KiB; refuse to slurp anything absurd.
    if (std.Io.Dir.cwd().readFileAlloc(io, cfg_path, allocator, .limited(1 << 20))) |text| {
        defer allocator.free(text);
        st.config = .present;
        st.family = detectFamily(allocator, text);
    } else |e| {
        st.config = if (e == error.FileNotFound or e == error.NotDir) .missing else .unreadable;
    }
    return st;
}

fn deinitParsed(family: Family, parsed: *Parsed) void {
    switch (family) {
        .chronos2 => parsed.chronos2.deinit(),
        .timestral => parsed.timestral.deinit(),
    }
}

fn deinitModel(family: Family, model: *LoadedModel, allocator: Allocator) void {
    switch (family) {
        .chronos2 => model.chronos2.deinit(allocator),
        .timestral => model.timestral.deinit(allocator),
    }
}

/// Free the checkpoint. Postgres 15+ never calls `_PG_fini`, so in a backend
/// process exit reclaims everything; this exists for tests and older servers.
/// The order matters because the model and quantile slices point into the
/// parsed config and the mmap.
pub fn deinit() void {
    // Not `lock()`: this runs at teardown and must not fail.

    if (g.loaded) {
        deinitModel(g.family, &g.model, g.model_arena.allocator());
        g.model_arena.deinit();
        g.weights.close(g.io);
        deinitParsed(g.family, &g.parsed);
        g.allocator.free(g.config_text);
        if (g.threaded_live) g.threaded.deinit();
        g.allocator.free(g.model_dir);
    }
    g = .{};
}

/// Forecast one univariate series. `history` is oldest -> newest with NaN
/// marking missing values. Returns `horizon * quantiles().len` values in the
/// engine layout: step-major, then the model's own quantile order.
pub fn forecast(history: []const f32, horizon: usize, sort: bool, opts: TimestralOptions) ![]f32 {
    try lock();
    defer unlock();

    if (!g.loaded) return Error.ModelNotLoaded;
    if (history.len == 0) return Error.EmptyHistory;
    try checkHorizon(horizon);

    switch (g.family) {
        .chronos2 => return forecastChronosLocked(history, horizon, sort, opts.max_context),
        .timestral => {
            const result = try timestralBatchChunk(history, 0, 1, 1, history.len, horizon, .{}, opts);
            if (sort) sortRows(result, horizon, g.quantiles.len);
            return result;
        },
    }
}

/// 1..1024 for every family, and additionally within the model's own
/// prediction length for Chronos-2. Callers hold the lock and have a loaded
/// model.
fn checkHorizon(horizon: usize) !void {
    if (horizon == 0 or horizon > 1024) return Error.HorizonTooLong;
    if (g.family == .chronos2 and horizon > g.prediction_length) return Error.HorizonExceedsModel;
}

/// Bound one result at 16 Mi f32 values (64 MiB): a huge variate or series
/// count must be a clear error, not a backend-sized allocation.
fn checkResultSize(rows: usize, horizon: usize) !void {
    const q = g.quantiles.len;
    const max_elems: usize = 16 * 1024 * 1024;
    if (q > 0 and rows > max_elems / horizon / q) return Error.ResultTooLarge;
}

fn validateCovariates(cov: Covariates, n_series: usize, t: usize, horizon: usize) !void {
    if ((cov.n_past > 0) != (cov.past != null)) return Error.InvalidCovariateShape;
    if ((cov.n_known > 0) != (cov.known != null)) return Error.InvalidCovariateShape;
    if ((cov.n_cat_past > 0) != (cov.cat_past != null)) return Error.InvalidCovariateShape;
    if ((cov.n_cat_known > 0) != (cov.cat_known != null)) return Error.InvalidCovariateShape;
    if (cov.past) |p| if (p.len != n_series * cov.n_past * t) return Error.InvalidCovariateShape;
    if (cov.known) |k| if (k.len != n_series * cov.n_known * (t + horizon)) return Error.InvalidCovariateShape;
    if (cov.cat_past) |p| if (p.len != n_series * cov.n_cat_past * t) return Error.InvalidCovariateShape;
    if (cov.cat_known) |k| if (k.len != n_series * cov.n_cat_known * (t + horizon)) return Error.InvalidCovariateShape;
}

/// Batch independent series in one call. `history` is `B x V x T` flattened,
/// result is `(B, V, horizon, quantiles)` flattened. Timestral supports
/// covariates; Chronos-2 requires `V = 1` and no covariates. The optional
/// cancel callback is checked between chunks of `min(B, 64)` series, so a
/// long batch remains cancellable even though one forward pass is not.
pub fn forecastBatch(
    history: []const f32,
    n_series: usize,
    n_variates: usize,
    horizon: usize,
    cov: Covariates,
    opts: TimestralOptions,
    sort: bool,
    cancel: ?CancelFn,
    cancel_ctx: ?*anyopaque,
) ![]f32 {
    try lock();
    defer unlock();

    if (!g.loaded) return Error.ModelNotLoaded;
    if (n_series == 0 or n_variates == 0) return Error.EmptyHistory;
    const rows = n_series * n_variates;
    if (history.len == 0 or history.len % rows != 0) return Error.InvalidCovariateShape;
    const t = history.len / rows;
    if (t == 0) return Error.EmptyHistory;
    try checkHorizon(horizon);
    try checkResultSize(rows, horizon);
    try validateCovariates(cov, n_series, t, horizon);

    const q = g.quantiles.len;
    if (g.family == .chronos2 and (n_variates != 1 or cov.any()))
        return Error.MultivariateUnsupported;

    const total = rows * horizon * q;

    // Common case: everything fits in one chunk. Return the engine's buffer
    // directly instead of allocating a second result and copying into it.
    if (n_series <= 64) {
        if (cancel) |f| try f(cancel_ctx);
        const part = switch (g.family) {
            .chronos2 => try chronosBatchChunk(history, 0, n_series, t, horizon, opts.max_context),
            .timestral => try timestralBatchChunk(history, 0, n_series, n_variates, t, horizon, cov, opts),
        };
        if (sort) sortRows(part, rows * horizon, q);
        return part;
    }

    const full = try g.allocator.alloc(f32, total);
    errdefer g.allocator.free(full);

    var s0: usize = 0;
    while (s0 < n_series) {
        if (cancel) |f| try f(cancel_ctx);
        const s1 = @min(n_series, s0 + 64);
        const part = switch (g.family) {
            .chronos2 => try chronosBatchChunk(history, s0, s1, t, horizon, opts.max_context),
            .timestral => try timestralBatchChunk(history, s0, s1, n_variates, t, horizon, cov, opts),
        };
        defer g.allocator.free(part);

        const offset = s0 * n_variates * horizon * q;
        @memcpy(full[offset..][0..part.len], part);
        s0 = s1;
    }

    if (sort) sortRows(full, rows * horizon, q);
    return full;
}

fn forecastChronosLocked(history: []const f32, horizon: usize, sort: bool, max_context: usize) ![]f32 {
    const ctx = if (max_context > 0 and history.len > max_context)
        history[history.len - max_context ..]
    else
        history;
    const result = try tsfm.chronos2.forecaster.forecast(
        g.allocator,
        g.par_io,
        &g.model.chronos2,
        ctx,
        1,
        ctx.len,
        horizon,
    );
    if (sort) sortRows(result, horizon, g.quantiles.len);
    return result;
}

fn chronosBatchChunk(
    history: []const f32,
    s0: usize,
    s1: usize,
    t: usize,
    horizon: usize,
    max_context: usize,
) ![]f32 {
    const nb = s1 - s0;
    if (max_context > 0 and t > max_context) {
        const keep = max_context;
        const ctx = try g.allocator.alloc(f32, nb * keep);
        defer g.allocator.free(ctx);
        for (0..nb) |i| {
            const src = history[(s0 + i) * t + (t - keep) ..][0..keep];
            @memcpy(ctx[i * keep ..][0..keep], src);
        }
        return tsfm.chronos2.forecaster.forecast(
            g.allocator,
            g.par_io,
            &g.model.chronos2,
            ctx,
            nb,
            keep,
            horizon,
        );
    }
    return tsfm.chronos2.forecaster.forecast(
        g.allocator,
        g.par_io,
        &g.model.chronos2,
        history[s0 * t .. s1 * t],
        nb,
        t,
        horizon,
    );
}

fn timestralBatchChunk(
    history: []const f32,
    s0: usize,
    s1: usize,
    n_variates: usize,
    t: usize,
    horizon: usize,
    cov: Covariates,
    opts: TimestralOptions,
) ![]f32 {
    const nb = s1 - s0;
    const n_elems = nb * n_variates * t;

    const inputs = try g.allocator.dupe(f32, history[s0 * n_variates * t ..][0..n_elems]);
    defer g.allocator.free(inputs);
    const observed = try g.allocator.alloc(bool, n_elems);
    defer g.allocator.free(observed);
    simd.maskMissing(inputs, observed);

    const sub: Covariates = .{
        .past = if (cov.past) |p| p[s0 * cov.n_past * t ..][0 .. nb * cov.n_past * t] else null,
        .n_past = cov.n_past,
        .known = if (cov.known) |k| k[s0 * cov.n_known * (t + horizon) ..][0 .. nb * cov.n_known * (t + horizon)] else null,
        .n_known = cov.n_known,
        .cat_past = if (cov.cat_past) |p| p[s0 * cov.n_cat_past * t ..][0 .. nb * cov.n_cat_past * t] else null,
        .n_cat_past = cov.n_cat_past,
        .cat_known = if (cov.cat_known) |k| k[s0 * cov.n_cat_known * (t + horizon) ..][0 .. nb * cov.n_cat_known * (t + horizon)] else null,
        .n_cat_known = cov.n_cat_known,
    };
    const tcov: tsfm.forecaster.Covariates = .{
        .past = sub.past,
        .n_past = sub.n_past,
        .known = sub.known,
        .n_known = sub.n_known,
        .cat_past = sub.cat_past,
        .n_cat_past = sub.n_cat_past,
        .cat_known = sub.cat_known,
        .n_cat_known = sub.n_cat_known,
    };

    return tsfm.forecaster.forecast(
        g.allocator,
        g.par_io,
        &g.model.timestral,
        inputs,
        observed,
        nb,
        n_variates,
        t,
        horizon,
        tcov,
        .{
            .max_context = opts.max_context,
            .per_shot_horizon_patches = opts.block_patches,
            .symmetric_averaging = opts.symmetric,
            .clip_nonnegative = opts.clip_nonnegative,
            .quantile_real_cap_k = opts.quantile_real_cap_k,
            .extreme_value_threshold = opts.extreme_value_threshold,
        },
    );
}

/// The stub engine: every quantile is the most recent finite value. It exists
/// so the SQL surface can be tested without a checkpoint.
pub fn stubForecast(history: []const f32, horizon: usize, sort: bool) ![]f32 {
    return stubForecastBatch(history, 1, 1, horizon, sort);
}

pub fn stubForecastBatch(
    history: []const f32,
    n_series: usize,
    n_variates: usize,
    horizon: usize,
    sort: bool,
) ![]f32 {
    if (n_series == 0 or n_variates == 0) return Error.EmptyHistory;
    const rows = n_series * n_variates;
    if (history.len == 0) return Error.EmptyHistory;
    if (history.len % rows != 0) return Error.InvalidCovariateShape;
    const t = history.len / rows;
    if (horizon == 0 or horizon > 1024) return Error.HorizonTooLong;

    const q = stub_quantiles.len;
    const result = try std.heap.c_allocator.alloc(f32, rows * horizon * q);
    errdefer std.heap.c_allocator.free(result);

    for (0..rows) |row| {
        var last: f32 = std.math.nan(f32);
        var i = row * t + t;
        while (i > row * t) {
            i -= 1;
            if (std.math.isFinite(history[i])) {
                last = history[i];
                break;
            }
        }
        @memset(result[row * horizon * q ..][0 .. horizon * q], last);
    }
    if (sort) sortRows(result, rows * horizon, q);
    return result;
}

/// Free a slice returned by `forecast`/`forecastExtended`/`stubForecast`.
pub fn freeResult(result: []f32) void {
    g.allocator.free(result);
}

pub fn isLoaded() bool {
    return g.loaded;
}

pub fn quantiles() []const f64 {
    return if (g.loaded) g.quantiles else &stub_quantiles;
}

pub fn contextLength() usize {
    return g.context_length;
}

pub fn predictionLength() usize {
    return g.prediction_length;
}

pub fn loadedModelDir() []const u8 {
    return g.model_dir;
}

pub fn loadedThreads() usize {
    return g.threads;
}

pub fn familyName() []const u8 {
    return if (!g.loaded) "none" else switch (g.family) {
        .chronos2 => "chronos-2",
        .timestral => "timestral",
    };
}

pub fn sortRows(values: []f32, rows: usize, q: usize) void {
    if (q < 2) return;
    var row_index: usize = 0;
    while (row_index < rows) : (row_index += 1) {
        const row = values[row_index * q ..][0..q];
        // Insertion sort: q is 21 for both checkpoints, and this keeps NaN
        // handling deterministic without a comparator with a total order.
        var i: usize = 1;
        while (i < q) : (i += 1) {
            const x = row[i];
            var j = i;
            while (j > 0 and row[j - 1] > x) : (j -= 1) row[j] = row[j - 1];
            row[j] = x;
        }
    }
}
