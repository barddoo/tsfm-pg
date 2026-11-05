//! Flat result cursor shared by the SRF iterators.
//!
//! A run is one flat array in `(series, variate, step, quantile)` order. The
//! cursor walks it sequentially so every row touches the next `f32` in memory,
//! and it tracks its position incrementally so the row loop has no integer
//! divide or modulo.

const std = @import("std");

/// Layout is deliberate: two slices, the flat index, and six `u32` counters fit
/// exactly in one 64-byte cache line. The `next()` loops read this state for
/// every emitted row; growing it past one line costs an extra line per row.
pub const Cursor = struct {
    result: []f32,
    quantiles: []const f64,
    index: usize = 0,
    horizon: u32,
    n_variates: u32,
    series: u32 = 0,
    variate: u32 = 0,
    step: u32 = 0,
    quantile: u32 = 0,

    comptime {
        std.debug.assert(@sizeOf(Cursor) <= 64);
    }

    pub fn init(result: []f32, quantiles: []const f64, n_variates: usize, horizon: usize) Cursor {
        return .{
            .result = result,
            .quantiles = quantiles,
            .horizon = @intCast(horizon),
            .n_variates = @intCast(n_variates),
        };
    }

    pub fn done(self: *const Cursor) bool {
        return self.index >= self.result.len;
    }

    pub fn value(self: *const Cursor) f32 {
        return self.result[self.index];
    }

    pub fn quantileLevel(self: *const Cursor) f64 {
        return self.quantiles[self.quantile];
    }

    /// Move to the next flat position. Callers read the position/value first.
    pub fn advance(self: *Cursor) void {
        self.index += 1;
        self.quantile += 1;
        if (self.quantile == self.quantiles.len) {
            self.quantile = 0;
            self.step += 1;
            if (self.step == self.horizon) {
                self.step = 0;
                self.variate += 1;
                if (self.variate == self.n_variates) {
                    self.variate = 0;
                    self.series += 1;
                }
            }
        }
    }
};
