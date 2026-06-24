"""Generate golden binary test fixtures for A4.

Produces deterministic .bin files for:
- single-region: 1 static region with 2 tasks, 1 dep
- multi-region: 2 static regions (2 tasks each), no cross-region deps
- empty-deps: 1 static region with 3 tasks, no deps
- string-table: 1 region, tasks with named args → string table populated

Run: PYTHONPATH=src python tests/sonata/fixtures/binary/gen_golden.py
"""

from pathlib import Path

from sonata.schedule import (
    ArgBinding,
    ArgDirection,
    ScheduleDep,
    ScheduledRegion,
    ScheduledTask,
    SonataScheduleContract,
)

FIXTURE_DIR = Path(__file__).parent


def gen_all() -> dict[str, SonataScheduleContract]:
    """Generate all golden fixtures. Returns {name: contract}."""
    fixtures: dict[str, SonataScheduleContract] = {}

    # 1. Single region, 2 tasks, 1 dep
    t1 = ScheduledTask(task_id=0, kernel_identity="add", func_id=1, core_type="aic",
        args=(ArgBinding(arg_identity="x", direction=ArgDirection.INPUT),
              ArgBinding(arg_identity="y", direction=ArgDirection.OUTPUT)))
    t2 = ScheduledTask(task_id=1, kernel_identity="mul", func_id=2, core_type="aic",
        args=(ArgBinding(arg_identity="z", direction=ArgDirection.INPUT),
              ArgBinding(arg_identity="w", direction=ArgDirection.OUTPUT)))
    r0 = ScheduledRegion(region_id="r0", kind="static",
        tasks=(t1, t2), deps=(ScheduleDep(producer=0, consumer=1),))
    fixtures["single-region"] = SonataScheduleContract(
        fingerprint="golden_single", regions=(r0,))

    # 2. Multi-region: 2 static regions, 2 tasks each, deps within each
    t3 = ScheduledTask(task_id=2, kernel_identity="sub", func_id=3, core_type="aic",
        args=(ArgBinding(arg_identity="a", direction=ArgDirection.INPUT),
              ArgBinding(arg_identity="b", direction=ArgDirection.OUTPUT)))
    t4 = ScheduledTask(task_id=3, kernel_identity="div", func_id=4, core_type="aic",
        args=(ArgBinding(arg_identity="c", direction=ArgDirection.INPUT),))
    r1 = ScheduledRegion(region_id="r1", kind="static",
        tasks=(t3, t4), deps=(ScheduleDep(producer=0, consumer=1),))
    fixtures["multi-region"] = SonataScheduleContract(
        fingerprint="golden_multi", regions=(r0, r1))

    # 3. Empty deps: 1 static region, 3 tasks, no deps
    tasks_no_deps = tuple(
        ScheduledTask(task_id=i, kernel_identity=f"k{i}", func_id=i, core_type="aic",
            args=(ArgBinding(arg_identity=f"arg{i}"),))
        for i in range(3)
    )
    r2 = ScheduledRegion(region_id="r2", kind="static",
        tasks=tasks_no_deps, deps=())
    fixtures["empty-deps"] = SonataScheduleContract(
        fingerprint="golden_nodeps", regions=(r2,))

    # 4. String table: named args, multiple tasks with distinct identities
    t5 = ScheduledTask(task_id=5, kernel_identity="tile_relu", func_id=10, core_type="aic",
        args=(ArgBinding(arg_identity="input_tensor", direction=ArgDirection.INPUT),
              ArgBinding(arg_identity="output_tensor", direction=ArgDirection.OUTPUT)))
    t6 = ScheduledTask(task_id=6, kernel_identity="tile_abs", func_id=11, core_type="aic",
        args=(ArgBinding(arg_identity="scratch_buf", direction=ArgDirection.INOUT),))
    r3 = ScheduledRegion(region_id="r3", kind="static",
        tasks=(t5, t6), deps=(ScheduleDep(producer=0, consumer=1),))
    fixtures["string-table"] = SonataScheduleContract(
        fingerprint="golden_str", regions=(r3,))

    return fixtures


def write_binaries(fixtures: dict[str, SonataScheduleContract]) -> None:
    """Write .bin files."""
    for name, contract in fixtures.items():
        bin_path = FIXTURE_DIR / f"{name}.bin"
        bin_path.write_bytes(contract.to_binary())
        print(f"  wrote {bin_path.name} ({len(bin_path.read_bytes())} bytes)")


if __name__ == "__main__":
    print("Generating golden binary fixtures...")
    fixtures = gen_all()
    write_binaries(fixtures)
    print("Done.")
