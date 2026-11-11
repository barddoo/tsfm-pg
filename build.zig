const std = @import("std");

// Load pgzx build support. The build utilities use pg_config to find all dependencies
// and provide functions to create and test extensions.
const PGBuild = @import("pgzx").Build;

const Gpu = enum { none, metal, cuda };

// Single source of truth for the version and the engine commit: build.zig.zon.
const zon = @import("build.zig.zon");
const engine_commit = blk: {
    const url = zon.dependencies.tsfm.url;
    const hash = std.mem.indexOfScalar(u8, url, '#') orelse @compileError("tsfm dependency URL must be pinned with #<commit>");
    break :blk url[hash + 1 ..];
};

pub fn build(b: *std.Build) void {
    // GPU offload is opt-in: `metal` on macOS, `cuda` on Linux/NVIDIA. The
    // engine falls back to CPU kernels when no device is available.
    const gpu = b.option(Gpu, "gpu", "Engine GPU backend: none, metal (macOS), cuda (Linux/NVIDIA)") orelse .none;

    const proj = PGBuild.Project.init(b, .{
        .name = "tsfm",
        .version = .{ .major = 0, .minor = 1 },
        .root_dir = "src/",
        .root_source_file = "src/main.zig",
    });

    // A bounds/overflow bug in the model must become a Postgres error, not a
    // segmentation fault. Default to ReleaseSafe unless the caller explicitly
    // picked an optimization mode (pgzx's Project.init defaults to Debug).
    if (b.user_input_options.get("optimize") == null and b.graph.release_mode == .off) {
        proj.pgbuild.options.optimize = .safe;
    }

    // The engine is a normal Zig module from tsfm-zig. There is no C ABI and no
    // second shared object: the extension imports the module directly.
    //
    // `-Dblas=builtin` keeps the resulting tsfm.so self-contained: no OpenBLAS
    // in the Postgres installation. Build a separate variant if throughput on
    // x86 ever justifies one.
    const tsfm_dep = b.dependency("tsfm", .{
        .target = proj.pgbuild.options.target,
        .optimize = proj.pgbuild.options.optimize,
        .blas = .builtin,
        .gpu = gpu,
    });
    const tsfm_mod = tsfm_dep.module("timestral");

    // Values baked into the extension at build time.
    proj.build_options.addOption([]const u8, "version", zon.version);
    proj.build_options.addOption([]const u8, "engine_commit", engine_commit);

    // Build the extension library once and attach the engine import. pgzx's
    // `Project.extensionLib` has no hook for extra imports, so the install
    // artifact uses this same instance rather than calling it a second time.
    const lib = proj.extensionLib();
    lib.root_module.addImport("tsfm", tsfm_mod);

    const install = b.getInstallStep();
    install.dependOn(&proj.pgbuild.addInstallExtensionLibArtifact(lib, proj.config.name).step);
    install.dependOn(&proj.installExtensionDir().step);

    const check = b.step("check", "Check that the extension compiles");
    const check_lib = proj.extensionLib();
    check_lib.root_module.addImport("tsfm", tsfm_mod);
    check_lib.linkage = null;
    check.dependOn(&check_lib.step);

    // Engine-only tests, without Postgres: load the module, run a forecast and
    // compare fixtures. These prove the dependency wiring before any SQL type
    // is involved.
    const engine_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/engine_test.zig"),
            .target = proj.pgbuild.options.target,
            .optimize = proj.pgbuild.options.optimize,
            .link_libc = true,
            .imports = &.{.{ .name = "tsfm", .module = tsfm_mod }},
        }),
    });
    const run_engine_tests = b.addRunArtifact(engine_tests);
    const test_step = b.step("test", "Run engine integration tests (no Postgres)");
    test_step.dependOn(&run_engine_tests.step);

    // pg_regress tests. They install the extension into the local cluster
    // first, so `zig build pg_regress` is self-contained after `pgstart`.
    const pg_user = b.option([]const u8, "pg-user", "Role for pg_regress (default: postgres)") orelse "postgres";
    const pg_port = b.option(u16, "pg-port", "Port for pg_regress (default: 5432)") orelse 5432;
    const regress = proj.addRegressTests(.{
        .db_user = pg_user,
        .db_port = pg_port,
        .scripts = &[_][]const u8{"tsfm_test"},
    });
    regress.step.dependOn(install);
    const regress_step = b.step("pg_regress", "Run SQL regression tests");
    regress_step.dependOn(&regress.step);

    // Numerical parity against fixtures produced by the tsfm-zig CLI. The
    // script skips cleanly when TSFM_TEST_MODEL is unset, so CI without the
    // 480 MB checkpoint can still call this step.
    const parity = b.addSystemCommand(&.{ "bash", "tests/run_parity.sh" });
    parity.step.dependOn(install);
    const parity_step = b.step("parity", "Run numerical parity tests (needs TSFM_TEST_MODEL)");
    parity_step.dependOn(&parity.step);
}
