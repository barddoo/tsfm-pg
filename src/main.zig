//! tsfm: local time-series forecasting inside Postgres.
//!
//! The model, weights and kernels live in tsfm-zig and are linked directly as a
//! Zig module; this file only registers the SQL functions, the GUCs and the
//! per-backend engine lifetime.

const std = @import("std");
const pgzx = @import("pgzx");

const engine = @import("engine.zig");
const functions = @import("functions.zig");
const guc = @import("guc.zig");

comptime {
    pgzx.PG_MODULE_MAGIC();

    pgzx.PG_FUNCTION_V1("tsfm_version", functions.version);
    pgzx.PG_FUNCTION_V1("tsfm_model_info_text", functions.modelInfoText);
    pgzx.PG_FUNCTION_V1("tsfm_model_status_text", functions.modelStatusText);
    pgzx.PG_FUNCTION_V1("tsfm_quantiles", functions.quantiles);
    pgzx.PG_FUNCTION_V1("tsfm_load", functions.load);
    pgzx.PG_FUNCTION_V1("tsfm_forecast", functions.forecastTable);
    pgzx.PG_FUNCTION_V1("tsfm_forecast_ex", functions.forecastEx);
    pgzx.PG_FUNCTION_V1("tsfm_forecast_batch", functions.forecastBatch);
    pgzx.PG_FUNCTION_V1("tsfm_forecast_matrix", functions.forecastMatrix);
    pgzx.PG_FUNCTION_V1("tsfm_forecast_matrix_ex", functions.forecastMatrixEx);
}

/// Register GUCs. Do *not* load the model here: `_PG_init` must stay cheap and
/// must never run before the backend fork (it would for a preloaded library).
pub export fn _PG_init() void {
    guc.init();
}

/// Only called by servers older than Postgres 15; newer ones never unload a
/// library, and process exit reclaims the engine's libc allocations and the
/// weight mmap. Kept so an older server frees the checkpoint explicitly.
pub export fn _PG_fini() void {
    engine.deinit();
}

const Tests = struct {
    pub fn testVersion() !void {
        try std.testing.expect(std.mem.startsWith(u8, functions.version(), "tsfm-pg "));
    }
};

comptime {
    pgzx.testing.registerTests(@import("build_options").testfn, .{Tests});
}
