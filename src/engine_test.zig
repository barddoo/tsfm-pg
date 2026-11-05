const std = @import("std");
const engine = @import("engine.zig");
const simd = @import("simd.zig");

fn env(name: []const u8) ?[]const u8 {
    for (std.mem.span(std.c.environ)) |maybe| {
        const entry = maybe orelse continue;
        const s = std.mem.span(entry);
        const eq = std.mem.indexOfScalar(u8, s, '=') orelse continue;
        if (std.mem.eql(u8, s[0..eq], name)) return s[eq + 1 ..];
    }
    return null;
}

test "stub forecast uses the last finite value" {
    const history = [_]f32{ 1, 2, std.math.nan(f32), 4 };
    const result = try engine.stubForecast(&history, 3, true);
    defer engine.freeResult(result);

    try std.testing.expectEqual(@as(usize, 3 * engine.stub_quantiles.len), result.len);
    for (result) |v| try std.testing.expectApproxEqAbs(@as(f32, 4), v, 1e-6);
}

test "stub forecast skips trailing missing values" {
    const history = [_]f32{ 7, std.math.nan(f32) };
    const result = try engine.stubForecast(&history, 1, true);
    defer engine.freeResult(result);
    for (result) |v| try std.testing.expectApproxEqAbs(@as(f32, 7), v, 1e-6);
}

test "stub forecast rejects empty history and bad horizons" {
    try std.testing.expectError(engine.Error.EmptyHistory, engine.stubForecast(&.{}, 1, true));
    try std.testing.expectError(engine.Error.HorizonTooLong, engine.stubForecast(&.{1}, 0, true));
    try std.testing.expectError(engine.Error.HorizonTooLong, engine.stubForecast(&.{1}, 1025, true));
}

test "stub forecast sorts rows when asked and handles batches" {
    const history = [_]f32{ 1, 2, 3, 10, 20, 30 };
    const result = try engine.stubForecastBatch(&history, 2, 1, 2, true);
    defer engine.freeResult(result);
    const q = engine.stub_quantiles.len;
    try std.testing.expectEqual(@as(usize, 2 * 2 * q), result.len);
    for (result[0 .. 2 * q]) |v| try std.testing.expectEqual(@as(f32, 3), v);
    for (result[2 * q ..]) |v| try std.testing.expectEqual(@as(f32, 30), v);

    try std.testing.expectError(engine.Error.InvalidCovariateShape, engine.stubForecastBatch(&history, 4, 1, 2, true));
}

test "sortRows makes every row monotonic" {
    var values = [_]f32{ 3, 1, 2, 9, 7, 8 };
    engine.sortRows(&values, 2, 3);
    try std.testing.expectEqualSlices(f32, &.{ 1, 2, 3, 7, 8, 9 }, &values);
}

test "forecasting without a loaded model is an error, not a crash" {
    try std.testing.expectError(engine.Error.ModelNotLoaded, engine.forecast(&.{1}, 1, true, .{}));
    try std.testing.expectError(
        engine.Error.ModelNotLoaded,
        engine.forecastBatch(&.{1}, 1, 1, 1, .{}, .{}, true, null, null),
    );
}

test "simd maskMissing matches the scalar definition for every tail length" {
    const nan = std.math.nan(f32);
    const inf = std.math.inf(f32);
    var n: usize = 0;
    while (n <= 3 * simd.f32_lanes + 1) : (n += 1) {
        var x: [64]f32 = undefined;
        var observed: [64]bool = undefined;
        var want: [64]f32 = undefined;
        for (0..n) |i| {
            x[i] = switch (i % 5) {
                0 => nan,
                1 => inf,
                2 => -inf,
                else => @as(f32, @floatFromInt(i)) - 3.5,
            };
            want[i] = if (std.math.isFinite(x[i])) x[i] else 0;
        }
        simd.maskMissing(x[0..n], observed[0..n]);
        for (0..n) |i| {
            try std.testing.expectEqual(want[i], x[i]);
            try std.testing.expectEqual(i % 5 > 2, observed[i]);
        }
    }
}

test "simd widen is exact for every tail length" {
    var src: [40]f32 = undefined;
    for (&src, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i)) * 0.1 - 1.7;
    var n: usize = 0;
    while (n <= src.len) : (n += 1) {
        var dst: [40]f64 = undefined;
        simd.widen(dst[0..n], src[0..n]);
        for (0..n) |i| try std.testing.expectEqual(@as(f64, src[i]), dst[i]);
    }
}

test "detectFamily reads architectures and rejects non-objects" {
    const a = std.testing.allocator;
    try std.testing.expectEqual(@as(?engine.Family, .chronos2), engine.detectFamily(a, "{\"architectures\":[\"Chronos2Model\"]}"));
    try std.testing.expectEqual(@as(?engine.Family, .timestral), engine.detectFamily(a, "{\"architectures\":[\"Other\"]}"));
    try std.testing.expectEqual(@as(?engine.Family, .timestral), engine.detectFamily(a, "{}"));
    try std.testing.expectEqual(@as(?engine.Family, .timestral), engine.detectFamily(a, "{\"architectures\":\"Chronos2Model\"}"));
    // None of these may trap; they are simply not model configs.
    try std.testing.expectEqual(@as(?engine.Family, null), engine.detectFamily(a, "[]"));
    try std.testing.expectEqual(@as(?engine.Family, null), engine.detectFamily(a, "42"));
    try std.testing.expectEqual(@as(?engine.Family, null), engine.detectFamily(a, "not json"));
    try std.testing.expectEqual(@as(?engine.Family, null), engine.detectFamily(a, ""));
}

test "inspect reports missing files without failing" {
    const st = try engine.inspect("/nonexistent/tsfm-models");
    try std.testing.expectEqual(engine.FileState.missing, st.config);
    try std.testing.expectEqual(engine.FileState.missing, st.weights);
    try std.testing.expectEqual(@as(u64, 0), st.weights_bytes);
    try std.testing.expectEqual(@as(?engine.Family, null), st.family);

    // An existing directory with no checkpoint is the same situation.
    const root = try engine.inspect("/");
    try std.testing.expectEqual(engine.FileState.missing, root.config);
    try std.testing.expectEqual(engine.FileState.missing, root.weights);
}

test "optional real-model smoke test" {
    const dir = env("TSFM_TEST_MODEL") orelse return error.SkipZigTest;
    try engine.ensure(dir, 1, .auto);
    defer engine.deinit();

    const history = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8 };
    const result = try engine.forecast(&history, 4, true, .{});
    defer engine.freeResult(result);

    const q = engine.quantiles().len;
    try std.testing.expectEqual(@as(usize, 4 * q), result.len);
    // Quantiles are monotonic per step after sorting.
    var step: usize = 0;
    while (step < 4) : (step += 1) {
        const row = result[step * q ..][0..q];
        var i: usize = 1;
        while (i < q) : (i += 1) {
            try std.testing.expect(row[i - 1] <= row[i]);
        }
    }
}

test "optional real-model forecast with a worker pool" {
    const dir = env("TSFM_TEST_MODEL") orelse return error.SkipZigTest;
    // threads > 1 is the only path that owns a `Threaded` executor; re-ensuring
    // after `deinit` also covers its creation and teardown across loads.
    try engine.ensure(dir, 4, .auto);
    defer engine.deinit();

    const history = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8 };
    const result = try engine.forecast(&history, 4, true, .{});
    defer engine.freeResult(result);
    try std.testing.expectEqual(@as(usize, 4 * engine.quantiles().len), result.len);
}
