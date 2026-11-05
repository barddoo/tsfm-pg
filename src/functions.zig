//! SQL-visible functions. Everything that can fail is wrapped in
//! `forecast.mapError` *after* the engine call has returned, so no Zig `defer`
//! is skipped by a Postgres longjmp.

const std = @import("std");
const pgzx = @import("pgzx");
const srf = pgzx.srf;
const build_options = @import("build_options");

const engine = @import("engine.zig");
const forecast = @import("forecast.zig");
const guc = @import("guc.zig");
const simd = @import("simd.zig");

const version_text = std.fmt.comptimePrint(
    "tsfm-pg {s} (engine tsfm-zig @ {s})",
    .{ build_options.version, build_options.engine_commit },
);

pub fn version() []const u8 {
    return version_text;
}

/// `tsfm_load()`: warm the engine and return true.
pub fn load() !bool {
    return forecast.warmup() catch |e| return forecast.mapError(@src(), e);
}

/// `tsfm_quantiles()`: the model's configured quantile levels.
pub fn quantiles() ![]const f64 {
    return forecast.quantiles() catch |e| return forecast.mapError(@src(), e);
}

/// `tsfm_model_info_text()`: JSON metadata; the SQL function `tsfm_model_info`
/// casts the text to `jsonb` so the extension does not need a jsonb converter.
pub fn modelInfoText() ![]const u8 {
    return modelInfoTextImpl() catch |e| return forecast.mapError(@src(), e);
}

/// `tsfm_model_status_text()`: JSON describing whether the configured model
/// directory is usable and, if not, what to do about it. Read-only: no weights
/// are loaded, nothing is written, and the network is never touched.
pub fn modelStatusText() ![]const u8 {
    return modelStatusTextImpl() catch |e| return forecast.mapError(@src(), e);
}

const fetch_hint = "scripts/fetch-model.sh {s} amazon/chronos-2 (run as the postgres OS user on the database host), " ++
    "or mount a checkpoint with config.json and model.safetensors there; " ++
    "alternatively point tsfm.model_dir at a directory that has them (superuser setting)";

fn modelStatusTextImpl() ![]const u8 {
    const allocator = pgzx.mem.PGCurrentContextAllocator;
    const dir = guc.model_dir.value();
    const stub = guc.stub.value;
    const st = try engine.inspect(dir);

    // `problem` is why forecasts would fail; `note` is context that is not a
    // failure. Both are plain text for humans.
    var problem: ?[]const u8 = null;
    var fix: ?[]const u8 = null;
    var note: ?[]const u8 = null;

    const configured = guc.modelKind();
    const want: ?engine.Family = switch (configured) {
        .auto => null,
        .chronos2 => .chronos2,
        .timestral => .timestral,
    };

    if (stub) {
        note = "tsfm.stub is on: forecasts use the naive last-value engine and need no model files";
    } else if (st.config == .missing and st.weights == .missing) {
        problem = try std.fmt.allocPrint(allocator, "no model found in {s}", .{dir});
        fix = try std.fmt.allocPrint(allocator, fetch_hint, .{dir});
    } else if (st.config == .unreadable or st.weights == .unreadable) {
        problem = try std.fmt.allocPrint(allocator, "the postgres OS user cannot read the model files in {s}", .{dir});
        fix = "check ownership and permissions of the directory and files (the postgres user needs read access)";
    } else if (st.config == .missing) {
        problem = try std.fmt.allocPrint(allocator, "config.json is missing in {s}", .{dir});
        fix = try std.fmt.allocPrint(allocator, fetch_hint, .{dir});
    } else if (st.weights == .missing) {
        problem = try std.fmt.allocPrint(allocator, "model.safetensors is missing in {s}", .{dir});
        fix = try std.fmt.allocPrint(allocator, fetch_hint, .{dir});
    } else if (st.weights_bytes < 1024) {
        problem = try std.fmt.allocPrint(allocator, "model.safetensors in {s} is only {d} bytes: empty or truncated", .{ dir, st.weights_bytes });
        fix = "download it again; an interrupted download leaves a partial file";
    } else if (st.family == null) {
        problem = try std.fmt.allocPrint(allocator, "config.json in {s} is not a valid JSON model config", .{dir});
        fix = "check that config.json belongs to a Chronos-2 or Timestral checkpoint and is not truncated";
    } else if (want != null and want.? != st.family.?) {
        problem = try std.fmt.allocPrint(allocator, "tsfm.model is '{s}' but config.json describes a {s} checkpoint", .{
            if (want.? == .chronos2) "chronos-2" else "timestral",
            if (st.family.? == .chronos2) "chronos-2" else "timestral",
        });
        fix = "SET tsfm.model = 'auto' (or the matching family); it is a superuser setting";
    }

    if (problem == null and !stub and engine.isLoaded() and !std.mem.eql(u8, engine.loadedModelDir(), dir)) {
        note = try std.fmt.allocPrint(allocator, "this session already loaded {s}; reconnect to use {s}", .{ engine.loadedModelDir(), dir });
    }

    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(allocator);

    try out.appendSlice(allocator, "{\"model_dir\":");
    try appendJsonString(&out, allocator, dir);
    try out.appendSlice(allocator, ",\"stub\":");
    try out.appendSlice(allocator, if (stub) "true" else "false");
    try out.appendSlice(allocator, ",\"config_json\":");
    try appendJsonString(&out, allocator, @tagName(st.config));
    try out.appendSlice(allocator, ",\"weights\":");
    try appendJsonString(&out, allocator, @tagName(st.weights));
    if (st.weights == .present) {
        try out.print(allocator, ",\"weights_bytes\":{d}", .{st.weights_bytes});
    } else {
        try out.appendSlice(allocator, ",\"weights_bytes\":null");
    }
    try out.appendSlice(allocator, ",\"configured_model\":");
    try appendJsonString(&out, allocator, switch (configured) {
        .auto => "auto",
        .chronos2 => "chronos-2",
        .timestral => "timestral",
    });
    try out.appendSlice(allocator, ",\"detected_model\":");
    if (st.family) |f| {
        try appendJsonString(&out, allocator, if (f == .chronos2) "chronos-2" else "timestral");
    } else {
        try out.appendSlice(allocator, "null");
    }
    try out.appendSlice(allocator, ",\"loaded\":");
    try out.appendSlice(allocator, if (!stub and engine.isLoaded()) "true" else "false");
    try out.appendSlice(allocator, ",\"ready\":");
    try out.appendSlice(allocator, if (problem == null) "true" else "false");
    inline for (.{ .{ "problem", problem }, .{ "fix", fix }, .{ "note", note } }) |entry| {
        try out.appendSlice(allocator, ",\"" ++ entry[0] ++ "\":");
        if (entry[1]) |text| try appendJsonString(&out, allocator, text) else try out.appendSlice(allocator, "null");
    }
    try out.appendSlice(allocator, "}");
    return out.toOwnedSlice(allocator);
}

fn modelInfoTextImpl() ![]const u8 {
    const allocator = pgzx.mem.PGCurrentContextAllocator;
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(allocator);

    try out.appendSlice(allocator, "{");
    try out.appendSlice(allocator, "\"extension_version\":");
    try appendJsonString(&out, allocator, build_options.version);
    try out.appendSlice(allocator, ",\"engine\":\"tsfm-zig\"");
    try out.appendSlice(allocator, ",\"engine_commit\":");
    try appendJsonString(&out, allocator, build_options.engine_commit);

    // In stub mode the engine is bypassed, so report it as not loaded even if a
    // real model was loaded earlier in this session. Values that only exist for
    // a loaded model are JSON null otherwise; model_dir and threads then show
    // the settings the next load would use.
    const stub = guc.stub.value;
    const loaded = !stub and engine.isLoaded();
    try out.appendSlice(allocator, ",\"loaded\":");
    try out.appendSlice(allocator, if (loaded) "true" else "false");

    const model_name: []const u8 = if (stub) "stub" else engine.familyName();
    try out.appendSlice(allocator, ",\"model\":");
    try appendJsonString(&out, allocator, model_name);

    try out.appendSlice(allocator, ",\"stub\":");
    try out.appendSlice(allocator, if (stub) "true" else "false");

    try out.appendSlice(allocator, ",\"model_dir\":");
    try appendJsonString(&out, allocator, if (loaded) engine.loadedModelDir() else guc.model_dir.value());

    try out.appendSlice(allocator, ",\"cpu_arch\":");
    try appendJsonString(&out, allocator, simd.arch());
    try out.print(allocator, ",\"simd_f32_lanes\":{d}", .{simd.f32_lanes});
    try out.print(allocator, ",\"threads\":{d}", .{if (loaded) engine.loadedThreads() else guc.threadCount()});
    try appendOptionalCount(&out, allocator, ",\"context_length\":", if (loaded) engine.contextLength() else 0);
    // Timestral decodes block by block and has no fixed prediction length.
    try appendOptionalCount(&out, allocator, ",\"prediction_length\":", if (loaded) engine.predictionLength() else 0);

    try out.appendSlice(allocator, ",\"quantiles\":[");
    const levels: []const f64 = if (stub)
        &engine.stub_quantiles
    else if (loaded)
        engine.quantiles()
    else
        &.{};
    for (levels, 0..) |level, i| {
        if (i > 0) try out.appendSlice(allocator, ",");
        try out.print(allocator, "{d}", .{level});
    }
    try out.appendSlice(allocator, "]}");

    return out.toOwnedSlice(allocator);
}

/// `tsfm_forecast(...)`: one row per (step, quantile).
pub fn forecastTable(history: ?[]const ?f64, horizon: ?i32) !srf.Table(forecast.ForecastSet) {
    return forecast.forecastSet(history, horizon) catch |e| return forecast.mapError(@src(), e);
}

/// `tsfm_forecast_ex(...)`: variates and optional covariates.
pub fn forecastEx(
    history: ?[]const ?f64,
    horizon: ?i32,
    n_variates: ?i32,
    past: ?[]const ?f64,
    known: ?[]const ?f64,
    cat_past: ?[]const ?f64,
    cat_known: ?[]const ?f64,
) !srf.Table(forecast.ExtendedSet) {
    return forecast.forecastExtendedSet(history, horizon, n_variates, past, known, cat_past, cat_known) catch |e|
        return forecast.mapError(@src(), e);
}

/// `tsfm_forecast_batch(...)`: independent series in one engine call.
pub fn forecastBatch(
    history: ?[]const ?f64,
    horizon: ?i32,
    n_series: ?i32,
    n_variates: ?i32,
    past: ?[]const ?f64,
    known: ?[]const ?f64,
    cat_past: ?[]const ?f64,
    cat_known: ?[]const ?f64,
) !srf.Table(forecast.BatchSet) {
    return forecast.forecastBatchSet(history, horizon, n_series, n_variates, past, known, cat_past, cat_known) catch |e|
        return forecast.mapError(@src(), e);
}

/// `tsfm_forecast_matrix(...)`: [step][quantile] as one array.
pub fn forecastMatrix(history: ?[]const ?f64, horizon: ?i32) ![]const f64 {
    return forecast.forecastMatrix(history, horizon) catch |e| return forecast.mapError(@src(), e);
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
    return forecast.forecastMatrixEx(history, horizon, n_variates, past, known, cat_past, cat_known) catch |e|
        return forecast.mapError(@src(), e);
}

/// Append `key` followed by a JSON number, or `null` when the value is 0 (unknown).
fn appendOptionalCount(out: *std.ArrayList(u8), allocator: std.mem.Allocator, key: []const u8, value: usize) !void {
    try out.appendSlice(allocator, key);
    if (value == 0) {
        try out.appendSlice(allocator, "null");
    } else {
        try out.print(allocator, "{d}", .{value});
    }
}

fn appendJsonString(out: *std.ArrayList(u8), allocator: std.mem.Allocator, s: []const u8) !void {
    try out.append(allocator, '"');
    for (s) |c| {
        switch (c) {
            '"' => try out.appendSlice(allocator, "\\\""),
            '\\' => try out.appendSlice(allocator, "\\\\"),
            '\n' => try out.appendSlice(allocator, "\\n"),
            '\r' => try out.appendSlice(allocator, "\\r"),
            '\t' => try out.appendSlice(allocator, "\\t"),
            else => {
                if (c < 0x20) {
                    try out.print(allocator, "\\u{x:0>4}", .{c});
                } else {
                    try out.append(allocator, c);
                }
            },
        }
    }
    try out.append(allocator, '"');
}
