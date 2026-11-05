//! SQL-facing forecasting: argument marshalling, the SRF iterators, and the
//! translation from Zig errors to Postgres errors.
//!
//! Everything here runs on the caller's thread. The engine may use worker
//! threads (tsfm.threads > 1), but only for its own memory; no worker calls
//! into Postgres.
//!
//! Every forecast flavour (single series, extended, batch) is one `Run`
//! produced by `runBatch`; the single-series and extended entry points are
//! the `n_series = 1` special cases. Argument order matches the SQL signatures:
//! `(history, horizon, [n_series], [n_variates], past, known, cat_past, cat_known)`.

const std = @import("std");
const pgzx = @import("pgzx");
const pg = pgzx.c;
const srf = pgzx.srf;

const engine = @import("engine.zig");
const guc = @import("guc.zig");
const simd = @import("simd.zig");

/// One forecast result: `(n_series, n_variates, horizon, quantiles)` flattened.
/// Owned by the caller and freed through `engine`.
pub const Run = struct {
    result: []f32,
    quantiles: []const f64,
    n_series: usize,
    n_variates: usize,
    horizon: usize,

    pub fn deinit(self: *Run) void {
        engine.freeResult(self.result);
    }
};

const max_horizon: i64 = 1024;

fn checkEnabled() !void {
    if (!guc.enabled.value) return error.ForecastingDisabled;
}

/// Validate the horizon argument shared by every entry point.
fn horizonArg(horizon: ?i32) !usize {
    const value = horizon orelse return error.HorizonNull;
    if (value < 1 or value > max_horizon) return error.HorizonTooLong;
    return @intCast(value);
}

/// Validate a dimension argument (`n_series`, `n_variates`).
fn dimArg(value: ?i32) !usize {
    const v = value orelse return error.NullArgument;
    if (v < 1) return error.InvalidCovariateShape;
    return @intCast(v);
}

fn timestralOptions() engine.TimestralOptions {
    return .{
        .max_context = guc.maxContext(),
        .block_patches = @intCast(guc.block_patches.value),
        .symmetric = guc.symmetric.value,
        .clip_nonnegative = guc.clip_nonnegative.value,
        .quantile_real_cap_k = @floatCast(guc.quantile_real_cap_k.value),
        .extreme_value_threshold = @floatCast(guc.extreme_value_threshold.value),
    };
}

fn ensureEngine() !void {
    if (guc.stub.value) return;
    try engine.ensure(guc.model_dir.value(), guc.threadCount(), guc.modelKind());
}

/// SQL float8 -> engine float32. NULL, NaN, infinities and values that overflow
/// float32 all become NaN ("missing"). Values keep only float32 precision, so
/// integers above 2^24 are rounded.
fn toF32Value(v: ?f64) f32 {
    const x = v orelse return std.math.nan(f32);
    if (!std.math.isFinite(x)) return std.math.nan(f32);
    const f: f32 = @floatCast(x);
    return if (std.math.isFinite(f)) f else std.math.nan(f32);
}

fn toCodeValue(v: ?f64) f64 {
    const x = v orelse return std.math.nan(f64);
    return if (std.math.isFinite(x)) x else std.math.nan(f64);
}

fn convertHistory(allocator: std.mem.Allocator, values: []const ?f64) ![]f32 {
    const out = try allocator.alloc(f32, values.len);
    for (values, out) |v, *dst| dst.* = toF32Value(v);
    return out;
}

/// Convert a flattened `C x row_len` float8 array to float32 with NaN for
/// missing/non-finite values. A NULL or empty array means "no covariates".
fn toF32(allocator: std.mem.Allocator, values: ?[]const ?f64, row_len: usize) !?[]const f32 {
    const vals = values orelse return null;
    if (vals.len == 0) return null;
    if (row_len == 0 or vals.len % row_len != 0) return error.InvalidCovariateShape;
    return try convertHistory(allocator, vals);
}

/// Convert a flattened `C x row_len` double array of category codes. Missing
/// and non-finite codes become NaN.
fn toCodes(allocator: std.mem.Allocator, values: ?[]const ?f64, row_len: usize) !?[]const f64 {
    const vals = values orelse return null;
    if (vals.len == 0) return null;
    if (row_len == 0 or vals.len % row_len != 0) return error.InvalidCovariateShape;

    const out = try allocator.alloc(f64, vals.len);
    for (vals, out) |v, *dst| dst.* = toCodeValue(v);
    return out;
}

fn hasCovariates(a: ?[]const ?f64, b: ?[]const ?f64, c: ?[]const ?f64, d: ?[]const ?f64) bool {
    inline for (.{ a, b, c, d }) |arr| {
        if (arr) |x| if (x.len > 0) return true;
    }
    return false;
}

/// Run a forecast for `n_series` independent series of `n_variates` target
/// rows each. `history` is `B x V x T` flattened; covariates are
/// `B x C x T` / `B x C x (T + horizon)`. The caller owns the result.
pub fn runBatch(
    history: ?[]const ?f64,
    horizon: ?i32,
    n_series: ?i32,
    n_variates: ?i32,
    past: ?[]const ?f64,
    known: ?[]const ?f64,
    cat_past: ?[]const ?f64,
    cat_known: ?[]const ?f64,
) !Run {
    try checkEnabled();
    const ns = try dimArg(n_series);
    const nv = try dimArg(n_variates);
    const h = try horizonArg(horizon);

    const values = history orelse return error.EmptyHistory;
    if (values.len == 0) return error.EmptyHistory;
    const rows = ns * nv;
    if (values.len % rows != 0) return error.InvalidCovariateShape;
    const t = values.len / rows;

    const allocator = std.heap.c_allocator;
    const inputs = try convertHistory(allocator, values);
    defer allocator.free(inputs);

    const sort = guc.sort_quantiles.value;

    if (guc.stub.value) {
        if (hasCovariates(past, known, cat_past, cat_known)) return error.StubUnsupported;

        // The stub does not know about tsfm.max_context, so apply it here.
        // Keep the last `limit` steps of every row (layout is B x V x T).
        const limit = guc.maxContext();
        var trimmed: ?[]f32 = null;
        defer if (trimmed) |buf| allocator.free(buf);
        var ctx: []const f32 = inputs;
        if (limit > 0 and t > limit) {
            const buf = try allocator.alloc(f32, rows * limit);
            trimmed = buf;
            for (0..rows) |row| {
                @memcpy(buf[row * limit ..][0..limit], inputs[row * t + (t - limit) ..][0..limit]);
            }
            ctx = buf;
        }
        return .{
            .result = try engine.stubForecastBatch(ctx, ns, nv, h, sort),
            .quantiles = &engine.stub_quantiles,
            .n_series = ns,
            .n_variates = nv,
            .horizon = h,
        };
    }

    const past32 = try toF32(allocator, past, ns * t);
    defer if (past32) |p| allocator.free(p);
    const known32 = try toF32(allocator, known, ns * (t + h));
    defer if (known32) |k| allocator.free(k);
    const cat_past64 = try toCodes(allocator, cat_past, ns * t);
    defer if (cat_past64) |p| allocator.free(p);
    const cat_known64 = try toCodes(allocator, cat_known, ns * (t + h));
    defer if (cat_known64) |k| allocator.free(k);

    const cov: engine.Covariates = .{
        .past = past32,
        .n_past = if (past32) |p| p.len / (ns * t) else 0,
        .known = known32,
        .n_known = if (known32) |k| k.len / (ns * (t + h)) else 0,
        .cat_past = cat_past64,
        .n_cat_past = if (cat_past64) |p| p.len / (ns * t) else 0,
        .cat_known = cat_known64,
        .n_cat_known = if (cat_known64) |k| k.len / (ns * (t + h)) else 0,
    };

    try ensureEngine();
    return .{
        .result = try engine.forecastBatch(inputs, ns, nv, h, cov, timestralOptions(), sort, checkCancel, null),
        .quantiles = engine.quantiles(),
        .n_series = ns,
        .n_variates = nv,
        .horizon = h,
    };
}

/// One univariate series.
pub fn run(history: ?[]const ?f64, horizon: ?i32) !Run {
    return runBatch(history, horizon, 1, 1, null, null, null, null);
}

/// One series with `n_variates` target rows and optional covariates.
pub fn runExtended(
    history: ?[]const ?f64,
    horizon: ?i32,
    n_variates: ?i32,
    past: ?[]const ?f64,
    known: ?[]const ?f64,
    cat_past: ?[]const ?f64,
    cat_known: ?[]const ?f64,
) !Run {
    const nv = n_variates orelse return error.NullArgument;
    return runBatch(history, horizon, 1, nv, past, known, cat_past, cat_known);
}

fn checkCancel(_: ?*anyopaque) anyerror!void {
    try pgzx.intr.CheckForInterrupts();
}

/// Explicit warmup (`tsfm_load()`). In stub mode there is nothing to load.
pub fn warmup() !bool {
    try checkEnabled();
    if (guc.stub.value) return true;
    try ensureEngine();
    return true;
}

/// `tsfm_quantiles()`.
pub fn quantiles() ![]const f64 {
    try checkEnabled();
    if (guc.stub.value) return &engine.stub_quantiles;
    try ensureEngine();
    return engine.quantiles();
}

pub const Row = struct {
    step: i32,
    quantile: f64,
    value: f64,
};

pub const ExtendedRow = struct {
    variate: i32,
    step: i32,
    quantile: f64,
    value: f64,
};

pub const BatchRow = struct {
    series: i32,
    variate: i32,
    step: i32,
    quantile: f64,
    value: f64,
};

const Cursor = @import("cursor.zig").Cursor;

fn cursorFrom(r: Run) Cursor {
    return .init(r.result, r.quantiles, r.n_variates, r.horizon);
}

/// Iterator behind `tsfm_forecast(...)`: one row per step and quantile.
pub const ForecastSet = struct {
    cursor: Cursor,

    pub fn next(self: *ForecastSet) ?Row {
        const c = &self.cursor;
        if (c.index >= c.result.len) return null;
        defer c.advance();
        return .{
            .step = @intCast(c.step + 1),
            .quantile = c.quantiles[c.quantile],
            .value = @floatCast(c.result[c.index]),
        };
    }

    pub fn deinit(self: *ForecastSet) void {
        engine.freeResult(self.cursor.result);
    }
};

/// Iterator behind `tsfm_forecast_ex(...)`: variate, step, quantile, value.
pub const ExtendedSet = struct {
    cursor: Cursor,

    pub fn next(self: *ExtendedSet) ?ExtendedRow {
        const c = &self.cursor;
        if (c.index >= c.result.len) return null;
        defer c.advance();
        return .{
            .variate = @intCast(c.variate + 1),
            .step = @intCast(c.step + 1),
            .quantile = c.quantiles[c.quantile],
            .value = @floatCast(c.result[c.index]),
        };
    }

    pub fn deinit(self: *ExtendedSet) void {
        engine.freeResult(self.cursor.result);
    }
};

/// Iterator behind `tsfm_forecast_batch(...)`:
/// series, variate, step, quantile, value.
pub const BatchSet = struct {
    cursor: Cursor,

    pub fn next(self: *BatchSet) ?BatchRow {
        const c = &self.cursor;
        if (c.index >= c.result.len) return null;
        defer c.advance();
        return .{
            .series = @intCast(c.series + 1),
            .variate = @intCast(c.variate + 1),
            .step = @intCast(c.step + 1),
            .quantile = c.quantiles[c.quantile],
            .value = @floatCast(c.result[c.index]),
        };
    }

    pub fn deinit(self: *BatchSet) void {
        engine.freeResult(self.cursor.result);
    }
};

pub fn forecastSet(history: ?[]const ?f64, horizon: ?i32) !srf.Table(ForecastSet) {
    return .{ .iter = .{ .cursor = cursorFrom(try run(history, horizon)) } };
}

pub fn forecastExtendedSet(
    history: ?[]const ?f64,
    horizon: ?i32,
    n_variates: ?i32,
    past: ?[]const ?f64,
    known: ?[]const ?f64,
    cat_past: ?[]const ?f64,
    cat_known: ?[]const ?f64,
) !srf.Table(ExtendedSet) {
    return .{ .iter = .{ .cursor = cursorFrom(try runExtended(history, horizon, n_variates, past, known, cat_past, cat_known)) } };
}

pub fn forecastBatchSet(
    history: ?[]const ?f64,
    horizon: ?i32,
    n_series: ?i32,
    n_variates: ?i32,
    past: ?[]const ?f64,
    known: ?[]const ?f64,
    cat_past: ?[]const ?f64,
    cat_known: ?[]const ?f64,
) !srf.Table(BatchSet) {
    return .{ .iter = .{ .cursor = cursorFrom(try runBatch(history, horizon, n_series, n_variates, past, known, cat_past, cat_known)) } };
}

/// Copy a run into a Postgres-owned array. `out_len` must have been derived
/// from the same quantile count the run uses.
fn copyOut(r: Run, out: []f64) !void {
    if (r.result.len != out.len) return error.InvalidInput;
    simd.widen(out, r.result);
}

/// `tsfm_forecast_matrix(...)`: [step][quantile] as one step-major array.
pub fn forecastMatrix(history: ?[]const ?f64, horizon: ?i32) ![]const f64 {
    return forecastMatrixEx(history, horizon, 1, null, null, null, null);
}

/// `tsfm_forecast_matrix_ex(...)`: [variate][step][quantile] as one array.
pub fn forecastMatrixEx(
    history: ?[]const ?f64,
    horizon: ?i32,
    n_variates: ?i32,
    past: ?[]const ?f64,
    known: ?[]const ?f64,
    cat_past: ?[]const ?f64,
    cat_known: ?[]const ?f64,
) ![]const f64 {
    // Validate everything that does not need the engine first, so a disabled
    // kill switch or a bad argument never loads the model.
    try checkEnabled();
    const v = try dimArg(n_variates);
    const h = try horizonArg(horizon);
    const values = history orelse return error.EmptyHistory;
    if (values.len == 0) return error.EmptyHistory;

    // Allocate the Postgres-owned output before any per-call engine result, so
    // a palloc OOM cannot strand a c_allocator buffer. `quantiles` may load the
    // model, but that memory is persistent engine state, not per-call.
    const q = (try quantiles()).len;
    const max_elems: usize = 16 * 1024 * 1024;
    if (q > 0 and v > max_elems / h / q) return error.ResultTooLarge;

    const allocator = pgzx.mem.PGCurrentContextAllocator;
    const out = try allocator.alloc(f64, v * h * q);
    errdefer allocator.free(out);

    var r = try runExtended(history, horizon, n_variates, past, known, cat_past, cat_known);
    defer r.deinit();
    try copyOut(r, out);
    return out;
}

/// Translate an engine error into a Postgres error and return it as a Zig
/// error. `ereportNoJump` captures the Postgres error stack without
/// longjmping, so callers' defers still run; fmgr rethrows it on the way out.
pub fn mapError(src: std.builtin.SourceLocation, e: anyerror) error{PGErrorStack} {
    // Preserve a Postgres error that was already captured (for example a
    // query cancellation raised by CheckForInterrupts). fmgr rethrows it.
    if (e == error.PGErrorStack) return error.PGErrorStack;

    return switch (e) {
        error.ForecastingDisabled => raise(src, pg.ERRCODE_OBJECT_NOT_IN_PREREQUISITE_STATE, .{
            .msg = "tsfm is disabled",
            .detail = "tsfm.enabled is off for this session.",
            .hint = "SET tsfm.enabled = on to allow forecasts.",
        }),
        error.HorizonTooLong => raise(src, pg.ERRCODE_INVALID_PARAMETER_VALUE, .{
            .msg = "horizon must be between 1 and 1024",
            .detail = "tsfm caps horizons at 1024 steps.",
            .hint = "Use a horizon of 1..1024.",
        }),
        error.HorizonExceedsModel => blk: {
            var buf: [96]u8 = undefined;
            const msg = std.fmt.bufPrint(&buf, "horizon exceeds the model's prediction length of {d} steps", .{engine.predictionLength()}) catch "horizon exceeds the model's prediction length";
            break :blk raise(src, pg.ERRCODE_INVALID_PARAMETER_VALUE, .{
                .msg = msg,
                .detail = "The loaded checkpoint cannot predict that many steps in one call.",
                .hint = "Use a smaller horizon, or a checkpoint with a longer prediction length.",
            });
        },
        error.HorizonNull => raise(src, pg.ERRCODE_INVALID_PARAMETER_VALUE, .{
            .msg = "horizon must not be NULL",
            .detail = "A forecast needs an explicit number of steps.",
            .hint = "Use a horizon of 1..1024.",
        }),
        error.NullArgument => raise(src, pg.ERRCODE_INVALID_PARAMETER_VALUE, .{
            .msg = "shape arguments must not be NULL",
            .detail = "n_series and n_variates describe the flattened history array and must be given explicitly.",
            .hint = "Pass positive integers; n_variates defaults to 1.",
        }),
        error.EmptyHistory => raise(src, pg.ERRCODE_INVALID_PARAMETER_VALUE, .{
            .msg = "history must contain at least one value",
            .detail = "The history array is NULL or empty.",
            .hint = "Pass an ordered (oldest -> newest) double precision[].",
        }),
        error.ModelDirChanged => raise(src, pg.ERRCODE_OBJECT_NOT_IN_PREREQUISITE_STATE, .{
            .msg = "tsfm.model_dir changed after the engine was loaded",
            .detail = "The model is process-global, so a new directory needs a new backend.",
            .hint = "Call tsfm_load() in a new session, or reset tsfm.model_dir.",
        }),
        error.ModelFamilyChanged => raise(src, pg.ERRCODE_OBJECT_NOT_IN_PREREQUISITE_STATE, .{
            .msg = "tsfm.model changed after the engine was loaded",
            .detail = "The model family is process-global.",
            .hint = "Call tsfm_load() in a new session, or reset tsfm.model.",
        }),
        error.Busy => raise(src, pg.ERRCODE_OBJECT_NOT_IN_PREREQUISITE_STATE, .{
            .msg = "the tsfm engine is already running a forecast",
            .detail = "A forecast was started from inside another forecast in the same session.",
            .hint = "Do not call tsfm functions from within a tsfm call.",
        }),
        error.InvalidThreads => raise(src, pg.ERRCODE_INVALID_PARAMETER_VALUE, .{
            .msg = "tsfm.threads must resolve to 1..64",
            .detail = "The effective worker count is outside the supported range.",
            .hint = "SET tsfm.threads to a value between 0 and 64.",
        }),
        error.ModelFileNotFound => raise(src, pg.ERRCODE_UNDEFINED_FILE, .{
            .msg = "tsfm model file not found",
            .detail = guc.model_dir.value(),
            .hint = "Run SELECT tsfm_model_status() for what is missing and how to fix it.",
        }),
        error.InvalidConfig,
        error.UnsupportedConfig,
        error.UnsupportedActivation,
        error.UnsupportedBias,
        error.UnknownLayerKind,
        error.MissingTensor,
        error.ShapeMismatch,
        error.InvalidSafetensors,
        error.UnsupportedDtype,
        error.MisalignedTensor,
        => raise(src, pg.ERRCODE_DATA_EXCEPTION, .{
            .msg = "invalid tsfm checkpoint",
            .detail = @errorName(e),
            .hint = "Check that config.json and model.safetensors belong together and are not truncated, and that tsfm.model matches the checkpoint.",
        }),
        error.MultivariateUnsupported => raise(src, pg.ERRCODE_FEATURE_NOT_SUPPORTED, .{
            .msg = "Chronos-2 forecasts one univariate series per call",
            .detail = "This function received multiple variates or covariates.",
            .hint = "Use a Timestral checkpoint with tsfm.model = 'timestral', or call tsfm_forecast() per series.",
        }),
        error.InvalidCovariateShape => raise(src, pg.ERRCODE_INVALID_PARAMETER_VALUE, .{
            .msg = "invalid covariate array shape",
            .detail = "history must be V x T; past is C x T; known is C x (T + horizon).",
            .hint = "Pass flattened row-major arrays and make the history length divisible by n_variates.",
        }),
        error.CovariatesUnsupported => raise(src, pg.ERRCODE_FEATURE_NOT_SUPPORTED, .{
            .msg = "this checkpoint does not support covariates",
            .detail = "The model's covariate_enabled setting is off.",
            .hint = "Use a Timestral checkpoint trained with covariates.",
        }),
        error.KnownCovariatesUnsupported => raise(src, pg.ERRCODE_FEATURE_NOT_SUPPORTED, .{
            .msg = "this checkpoint cannot route known-future covariates",
            .detail = "The block schedule has no time -> variate path after a time op.",
            .hint = "Use past-only covariates, or a checkpoint whose config supports known future values.",
        }),
        error.AsymmetricQuantiles => raise(src, pg.ERRCODE_INVALID_PARAMETER_VALUE, .{
            .msg = "Timestral quantile knots are not symmetric",
            .detail = "Mirrored averaging requires q_i + q_{n-1-i} = 1.",
            .hint = "SET tsfm.symmetric = off.",
        }),
        error.DegenerateInput => raise(src, pg.ERRCODE_DATA_EXCEPTION, .{
            .msg = "the forecast produced a non-finite value",
            .detail = "The context is too short, constant, or contains no observations.",
            .hint = "Check the history and try a longer or non-degenerate series.",
        }),
        error.ResultTooLarge => raise(src, pg.ERRCODE_PROGRAM_LIMIT_EXCEEDED, .{
            .msg = "forecast result would be too large",
            .detail = "One call is limited to 16 million f32 values (64 MiB).",
            .hint = "Reduce n_variates or the horizon, or forecast in smaller batches.",
        }),
        error.StubUnsupported => raise(src, pg.ERRCODE_FEATURE_NOT_SUPPORTED, .{
            .msg = "tsfm.stub does not support covariates or multivariate forecasts",
            .detail = "The naive last-value engine only handles one univariate series.",
            .hint = "SET tsfm.stub = off and point tsfm.model_dir at a Timestral checkpoint.",
        }),
        error.OutOfMemory => raise(src, pg.ERRCODE_OUT_OF_MEMORY, .{
            .msg = "tsfm ran out of memory",
            .detail = "The engine could not allocate its model or forecast scratch space.",
            .hint = "Check that the model is installed and that the backend has enough memory.",
        }),
        else => raise(src, pg.ERRCODE_INTERNAL_ERROR, .{
            .msg = "tsfm forecast failed",
            .detail = @errorName(e),
            .hint = "Check the Postgres log and tsfm_model_info().",
        }),
    };
}

const Raise = struct {
    msg: []const u8,
    detail: []const u8,
    hint: []const u8,
};

fn raise(src: std.builtin.SourceLocation, comptime code: c_int, r: Raise) error{PGErrorStack} {
    pgzx.elog.ereportNoJump(src, .Error, .{
        pgzx.elog.errcode(code),
        pgzx.elog.errmsg("{s}", .{r.msg}),
        pgzx.elog.errdetail("{s}", .{r.detail}),
        pgzx.elog.errhint("{s}", .{r.hint}),
    }) catch {};
    return error.PGErrorStack;
}
