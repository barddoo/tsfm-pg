//! Custom GUCs for the extension. `_PG_init` in main.zig calls `init`.
//!
//! Settings that decide *what runs and what is read from disk* are `SUSET`:
//! `tsfm.model_dir` (an ordinary role must not point the server at arbitrary
//! files), `tsfm.model`, `tsfm.threads`, `tsfm.enabled` (a kill switch a
//! session can undo is no kill switch) and `tsfm.stub` (silently swaps real
//! forecasts for a constant). Set them in postgresql.conf, with `ALTER SYSTEM`,
//! or per role/database with `ALTER ROLE ... SET`. The decoding knobs stay
//! `USERSET`, so `SET LOCAL` works for them.
//!
//! The engine itself is process-global per backend (see engine.zig); changing
//! `tsfm.model_dir` or `tsfm.model` after the model is loaded is rejected by
//! `engine.ensure` with a hint to reconnect, and `tsfm.threads` is ignored
//! after load.

const std = @import("std");
const pgzx = @import("pgzx");
const pg = pgzx.c;

const engine = @import("engine.zig");

pub const ModelKind = engine.ModelChoice;

pub var model_dir: pgzx.guc.CustomStringVariable = undefined;
pub var model: pgzx.guc.CustomEnumVariable = undefined;
pub var threads: pgzx.guc.CustomIntVariable = undefined;
pub var max_context: pgzx.guc.CustomIntVariable = undefined;
pub var sort_quantiles: pgzx.guc.CustomBoolVariable = undefined;
pub var enabled: pgzx.guc.CustomBoolVariable = undefined;
pub var stub: pgzx.guc.CustomBoolVariable = undefined;
// Timestral-only decoding options. They are ignored by Chronos-2.
pub var block_patches: pgzx.guc.CustomIntVariable = undefined;
pub var symmetric: pgzx.guc.CustomBoolVariable = undefined;
pub var clip_nonnegative: pgzx.guc.CustomBoolVariable = undefined;
pub var quantile_real_cap_k: pgzx.guc.CustomRealVariable = undefined;
pub var extreme_value_threshold: pgzx.guc.CustomRealVariable = undefined;

/// Default `tsfm.model_dir`: `<pg sharedir>/tsfm/models`, baked in at build
/// time from the `pg_config` the extension was compiled against.
var default_model_dir: [:0]const u8 = "";

/// Clamp `tsfm.threads` to [0, 64] instead of making `SET` fail (0 = auto). The
/// GUC's declared max admits the values the hook is meant to clamp.
fn clampThreads(newval: *c_int, extra: *?*anyopaque, source: pg.GucSource) bool {
    _ = extra;
    _ = source;
    if (newval.* < 0) newval.* = 0;
    if (newval.* > 64) newval.* = 64;
    return true;
}

pub fn init() void {
    // GUC initial values must outlive the setting. TopMemoryContext is the
    // right lifetime for the concatenated default path. Resolve $sharedir at
    // runtime from my_exec_path so the default is correct even when the
    // extension was built with a relative pg_config path.
    var top = pgzx.mem.MemoryContextAllocator.init(pg.TopMemoryContext, .{});
    var share_buf: [pg.MAXPGPATH]u8 = undefined;
    pg.get_share_path(pg.my_exec_path, &share_buf[0]);
    const sharedir = std.mem.sliceTo(share_buf[0..], 0);
    default_model_dir = std.fmt.allocPrintSentinel(
        top.allocator(),
        "{s}/tsfm/models",
        .{sharedir},
        0,
    ) catch registerOom();

    model_dir.register(.{
        .name = "tsfm.model_dir",
        .short_desc = "Directory containing config.json and model.safetensors",
        .initial_value = default_model_dir,
        .context = pg.PGC_SUSET,
    });

    // Register the enum by hand so the option table is allocated with an
    // explicit OOM path. pgzx's helper uses `catch unreachable` internally.
    registerModelEnum(top.allocator());

    threads.register(.{
        .name = "tsfm.threads",
        .short_desc = "Worker threads used inside one forecast call; 0 = auto (up to 4)",
        .initial_value = 1,
        .min_value = 0,
        .max_value = std.math.maxInt(c_int),
        .context = pg.PGC_SUSET,
        .check_hook = pgzx.guc.checkIntHook(clampThreads),
    });

    max_context.register(.{
        .name = "tsfm.max_context",
        .short_desc = "Keep only the most recent N history values (0 = model default)",
        .initial_value = 0,
        .min_value = 0,
        .max_value = 1 << 20,
    });

    sort_quantiles.register(.{
        .name = "tsfm.sort_quantiles",
        .short_desc = "Sort quantiles per step so they are monotonic",
        .initial_value = true,
    });

    enabled.register(.{
        .name = "tsfm.enabled",
        .short_desc = "Allow forecasting; turn off as a kill switch",
        .initial_value = true,
        .context = pg.PGC_SUSET,
    });

    stub.register(.{
        .name = "tsfm.stub",
        .short_desc = "Use a naive last-value engine instead of loading weights",
        .initial_value = false,
        .context = pg.PGC_SUSET,
    });

    block_patches.register(.{
        .name = "tsfm.block_patches",
        .short_desc = "Timestral: patches decoded per block (default 32)",
        .initial_value = 32,
        .min_value = 1,
        .max_value = 1024,
    });

    symmetric.register(.{
        .name = "tsfm.symmetric",
        .short_desc = "Timestral: average with the mirrored forecast",
        .initial_value = true,
    });

    clip_nonnegative.register(.{
        .name = "tsfm.clip_nonnegative",
        .short_desc = "Timestral: clamp non-negative series at 0",
        .initial_value = true,
    });

    quantile_real_cap_k.register(.{
        .name = "tsfm.quantile_real_cap_k",
        .short_desc = "Timestral: clamp to observed range + k*scale (0 disables)",
        .initial_value = 1e4,
        .min_value = 0,
        .max_value = 1e12,
    });

    extreme_value_threshold.register(.{
        .name = "tsfm.extreme_value_threshold",
        .short_desc = "Timestral: fed-back medians at or above this magnitude become 0",
        .initial_value = 1e30,
        .min_value = 0,
        .max_value = 1e38,
    });

    // Reject `tsfm.<typo>` rather than silently creating a placeholder.
    pgzx.guc.markPrefixReserved("tsfm");
}

/// `_PG_init` cannot return an error. Report a real Postgres OOM (which aborts
/// the library load cleanly) instead of aborting the whole backend.
fn registerOom() noreturn {
    pgzx.elog.ereport(@src(), .Error, .{
        pgzx.elog.errcode(pg.ERRCODE_OUT_OF_MEMORY),
        pgzx.elog.errmsg("tsfm: out of memory registering GUCs", .{}),
    });
    unreachable;
}

fn registerModelEnum(allocator: std.mem.Allocator) void {
    model.value = @backingInt(ModelKind.auto);
    const entries = allocator.alloc(pg.struct_config_enum_entry, 4) catch registerOom();
    entries[0] = .{ .name = "auto", .val = @backingInt(ModelKind.auto), .hidden = false };
    entries[1] = .{ .name = "chronos-2", .val = @backingInt(ModelKind.chronos2), .hidden = false };
    entries[2] = .{ .name = "timestral", .val = @backingInt(ModelKind.timestral), .hidden = false };
    entries[3] = .{ .name = null, .val = 0, .hidden = false };

    pg.DefineCustomEnumVariable(
        "tsfm.model",
        "Model family: auto, chronos-2 or timestral",
        null,
        &model.value,
        model.value,
        entries.ptr,
        pg.PGC_SUSET,
        0,
        null,
        null,
        null,
    );
}

/// Current model family setting.
pub fn modelKind() ModelKind {
    return @fromBackingInt(@intCast(model.value));
}

/// `tsfm.threads`, always within the supported range. 0 means auto: up to four
/// workers, capped by the logical CPU count. The default stays 1 because many
/// backends each starting a pool can oversubscribe the host.
pub fn threadCount() usize {
    const v = threads.value;
    if (v == 0) {
        const cpus = std.Thread.getCpuCount() catch 1;
        return @min(std.math.clamp(cpus, 1, 64), 4);
    }
    return std.math.clamp(@as(usize, @intCast(v)), 1, 64);
}

/// `tsfm.max_context`; 0 means "no explicit limit".
pub fn maxContext() usize {
    return @intCast(max_context.value);
}
