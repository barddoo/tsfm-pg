//! Small explicit-SIMD helpers for the loops this extension owns. The heavy
//! kernels (matmul, attention, softmax) live in tsfm-zig and size themselves
//! from the same target CPU; see "SIMD" in the README for the `-Dcpu` option.

const std = @import("std");
const builtin = @import("builtin");

/// f32 lanes per vector register for the build target (1 = no SIMD).
pub const f32_lanes: usize = std.simd.suggestVectorLength(f32) orelse 1;

const L = f32_lanes;

/// Replace non-finite values (NaN, +/-inf) with 0 and record which entries were
/// observed. Timestral takes zeros plus an explicit mask for missing values.
pub fn maskMissing(x: []f32, observed: []bool) void {
    std.debug.assert(x.len == observed.len);
    const V = @Vector(L, f32);
    const zero: V = @splat(0);
    const max: V = @splat(std.math.floatMax(f32));

    var i: usize = 0;
    while (i + L <= x.len) : (i += L) {
        const v: V = x[i..][0..L].*;
        // `<=` is false for NaN, and |inf| > floatMax, so one compare covers both.
        const ok = @abs(v) <= max;
        x[i..][0..L].* = @select(f32, ok, v, zero);
        const flags: [L]bool = ok;
        observed[i..][0..L].* = flags;
    }
    while (i < x.len) : (i += 1) {
        const ok = std.math.isFinite(x[i]);
        observed[i] = ok;
        if (!ok) x[i] = 0;
    }
}

/// `dst[i] = @as(f64, src[i])`.
pub fn widen(dst: []f64, src: []const f32) void {
    std.debug.assert(dst.len == src.len);
    const VF = @Vector(L, f32);
    const VD = @Vector(L, f64);

    var i: usize = 0;
    while (i + L <= src.len) : (i += L) {
        const v: VF = src[i..][0..L].*;
        const w: VD = @floatCast(v);
        dst[i..][0..L].* = w;
    }
    while (i < src.len) : (i += 1) dst[i] = src[i];
}

/// Target description for `tsfm_model_info()`.
pub fn arch() []const u8 {
    return @tagName(builtin.cpu.arch);
}
