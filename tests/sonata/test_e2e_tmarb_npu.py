#!/usr/bin/env python
"""
test_e2e_sonata_tmarb_npu.py — Minimal end-to-end proof that sonata_tmarb runs on NPU.

Design
======

Five stages, each independently verifiable:

  0. ENVIRONMENT CHECK — ensure sonata_tmarb runtime binaries exist and the
     scheduler / kernel runtime is compiled in.
  1. COMPILE — compile a trivial abs kernel through the standard pypto pipeline.
  2. ANALYSE — run Sonata analysis on the certified IR; build a FlatSchedule
     binary from the resulting schedule.
  3. REASSEMBLE — from the standard ChipCallable produced by step 1, extract
     the child kernel binaries; build a NEW ChipCallable whose ``binary`` is
     the FlatSchedule blob instead of the orchestration ELF.
  4. EXECUTE — run the new ChipCallable under Worker(runtime="sonata_tmarb"),
     bypassing RuntimeBuilder's discovery (which does not know sonata_tmarb)
     by constructing RuntimeBinaries manually.
  5. VALIDATE — compare the output tensor against torch.abs().

The known gaps are documented inline with [GAP-*] markers.

Prerequisites
=============

  * pypto and simpler must be installed in the venv.
  * sonata_tmarb must be compiled (``pip install --no-build-isolation .`` from
    ``upstream/pypto/runtime/`` compiles it alongside tensormap_and_ringbuffer).
  * For NPU: a2a3 NPU hardware, ``task-submit`` available, or equivalent
    device access.
  * torch (for golden comparison).

What makes this hard
====================

[HARD-1] RuntimeBuilder does not discover sonata_tmarb
  The simpler RuntimeBuilder looks for runtimes in
  ``<upstream>/runtime/src/a2a3/runtime/<name>/build_config.py``.
  sonata_tmarb lives in ``<sonata>/runtime/sonata_tmarb/`` and is therefore
  invisible.  We work around this by constructing RuntimeBinaries manually.

[HARD-2] ChipCallable flow replaces orch SO with FlatSchedule binary
  The standard ``compile_and_assemble()`` builds a ChipCallable whose
  ``binary`` field is the orchestration ELF (``callable->binary_data()``
  on the C++ side).  We must rebuild the ChipCallable with the FlatSchedule
  binary in its place.  ChipCallable.build() accepts arbitrary bytes, but
  this path has never been exercised in a real run.

[HARD-3] Runtime binary paths must be resolved
  We need absolute paths to ``libhost_runtime.so``, ``libaicpu_kernel.so``,
  ``libaicore_kernel.so``, ``libsimpler_log.so``, and (on sim)
  ``libcpu_sim_context.so``.  These live under
  ``<upstream>/runtime/build/lib/``.

[HARD-4] Interpreter symbol must be reachable
  On sim, aicpu_execute is found via ``dlsym(RTLD_DEFAULT, "aicpu_execute")``
  because aicpu_kernel.so is loaded by the process.  On real NPU the process
  boundary is different; the env var ``SONATA_AICPU_PATH`` is required.
  We set it unconditionally for robustness.

[HARD-5] Schedule func_ids must match compiled kernel func_ids
  The FlatSchedule's task entries reference kernels by ``func_id``.  The
  Sonata analysis pipeline assigns func_ids from the compiled program's
  ``kernel_config.py``.  If the schedule carries a func_id that does not
  match any compiled child kernel, the interpreter will fail to dispatch.

What we already have working
============================

[EASY-1] Sonata analysis pipeline
  ``sonata_analyze()`` runs successfully on certified IR (tested by
  ``--with-sonata`` integration tests).  ``SonataScheduleContract.to_binary()``
  produces the correct binary format (tested by test_binary_serialization.py).

[EASY-2] Kernel compilation
  ``pypto.ir.compile()`` followed by ``compile_and_assemble()`` produces
  working ChipCallable objects (tested by thousands of ST tests).

[EASY-3] ChipCallable.child() API
  ``ChipCallable.child(i)`` returns ``PyCoreCallable`` objects, and
  ``ChipCallable.build()`` accepts ``[(func_id, CoreCallable), ...]`` as the
  ``children`` argument.  The extraction-and-rebuild cycle is mechanically
  straightforward.

[EASY-4] Binary schedule format
  ``SonataScheduleContract.to_binary()`` and ``from_binary()`` are round-trip
  tested.  The format matches ``flat_schedule.h`` struct layout.

[EASY-5] Golden comparison
  For a simple abs kernel, ``torch.abs(input)`` is the unambiguous reference.
"""

from __future__ import annotations

import ctypes
import os
import struct
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Imports — fail early with clear messages
# ---------------------------------------------------------------------------

try:
    import pypto.language as pl
    import pypto.ir as pir
    from pypto.runtime.device_runner import (
        compile_and_assemble,
        build_orch_args_from_inputs,
    )
    from pypto.runtime.task_interface import ChipCallable, ChipStorageTaskArgs, CallConfig
except ImportError as e:
    sys.exit(f"pypto not importable: {e}. Activate the project venv first.")

try:
    from simpler.task_interface import ChipWorker
    from simpler_setup.runtime_builder import RuntimeBinaries
    from simpler_setup.environment import PROJECT_ROOT
except ImportError as e:
    sys.exit(f"simpler not importable: {e}. Activate the project venv first.")

try:
    from sonata.pipeline import sonata_analyze, _extract_certified_ir, _write_bound_schedule
    from sonata.schedule import SonataScheduleContract
except ImportError as e:
    sys.exit(f"sonata not importable: {e}. Install the sonata package first.")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

M, N = 16, 16
PLATFORM = os.environ.get("SONATA_TEST_PLATFORM", "a2a3sim")
DEVICE_ID = int(os.environ.get("SONATA_TEST_DEVICE", "0"))
RTOL = float(os.environ.get("SONATA_TEST_RTOL", "1e-5"))
ATOL = float(os.environ.get("SONATA_TEST_ATOL", "1e-5"))

print(f"[config] platform={PLATFORM}, device={DEVICE_ID}")


# ===========================================================================
# Step 1: Program definition
# ===========================================================================

@pl.program
class AbsProgram:
    """Tile-level absolute value — the simplest possible kernel."""

    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        a: pl.Tensor[[M, N], pl.FP32],
        out: pl.Out[pl.Tensor[[M, N], pl.FP32]],
    ) -> pl.Tensor[[M, N], pl.FP32]:
        tile_a: pl.Tile[[M, N], pl.FP32] = pl.load(a, [0, 0], [M, N])
        tile_c: pl.Tile[[M, N], pl.FP32] = pl.tile.abs(tile_a)
        out = pl.store(tile_c, [0, 0], out)
        return out

    @pl.function(type=pl.FunctionType.Orchestration)
    def orchestrator(
        self,
        a: pl.Tensor[[M, N], pl.FP32],
        out: pl.Out[pl.Tensor[[M, N], pl.FP32]],
    ) -> pl.Tensor[[M, N], pl.FP32]:
        out = self.kernel(a, out)
        return out


# ===========================================================================
# Step 2: Compile program (standard pypto pipeline)
# ===========================================================================

print("\n=== Step 1: Compile program ===")

with tempfile.TemporaryDirectory(suffix="_sonata_tmarb_e2e") as tmpdir:
    work_dir = Path(tmpdir)
    compiled = pir.compile(AbsProgram, output_dir=str(work_dir))
    print(f"  compiled output: {work_dir}")

    # -----------------------------------------------------------------------
    # Step 3: Assemble standard ChipCallable (needed for kernel binaries)
    # -----------------------------------------------------------------------
    print("\n=== Step 2: Assemble standard ChipCallable ===")
    std_chip, runtime_name, runtime_config = compile_and_assemble(work_dir, platform=PLATFORM)
    print(f"  runtime_name={runtime_name}")
    print(f"  child_count={std_chip.child_count}")
    print(f"  binary_size={std_chip.binary_size} (standard orch SO)")

    # Extract kernel children for later reassembly
    children_list = [
        (std_chip.child_func_id(i), std_chip.child(i))
        for i in range(std_chip.child_count)
    ]
    sig = [std_chip.sig(i) for i in range(std_chip.sig_count)]
    func_name = std_chip.func_name
    config_name = std_chip.config_name
    print(f"  extracted {len(children_list)} children, sig_count={len(sig)}")

    # -----------------------------------------------------------------------
    # Step 4: Sonata analysis → FlatSchedule binary
    # -----------------------------------------------------------------------
    print("\n=== Step 3: Sonata analysis → FlatSchedule binary ===")
    certified_ir = _extract_certified_ir(AbsProgram)
    if certified_ir is None:
        raise RuntimeError("certified IR extraction returned None")

    result = sonata_analyze(certified_ir)
    if not result or not result.eligible:
        print(f"  SONATA NOT ELIGIBLE — falling back to standard TMARB path")
        print(f"  reasons: {result.fallback_reasons if result else 'N/A'}")
        # [GAP-6] If analysis says not eligible, we can't generate a schedule.
        # This is a real risk: the simple abs program may fail eligibility if
        # the analysis pipeline has requirements the program doesn't meet.
        # Fallback: manually construct a minimal schedule for the single-task case.
        sys.exit(1)

    print(f"  eligible=True, tasks={result.task_count}")
    print(f"  regions={result.region_statuses}")

    # Build schedule contract → binary schedule
    from sonata.schedule import build_schedule
    schedule = build_schedule(result.score, result)
    schedule_binary = schedule.to_binary()
    schedule_path = work_dir / "sonata_schedule.bin"
    schedule_path.write_bytes(schedule_binary)
    print(f"  schedule binary: {schedule_path} ({len(schedule_binary)} bytes)")

    # -----------------------------------------------------------------------
    # Step 5: Reassemble ChipCallable with schedule binary
    # -----------------------------------------------------------------------
    print("\n=== Step 4: Reassemble ChipCallable with schedule binary ===")
    sonata_chip = ChipCallable.build(
        signature=sig,
        func_name=func_name,
        binary=schedule_binary,     # ← FlatSchedule, NOT orch SO!
        children=children_list,
        config_name=config_name,
    )
    print(f"  sonata_chip.binary_size={sonata_chip.binary_size} (schedule binary)")

    # -----------------------------------------------------------------------
    # Step 6: Construct RuntimeBinaries manually
    # -----------------------------------------------------------------------
    # [HARD-1] RuntimeBuilder doesn't discover sonata_tmarb. We construct
    # RuntimeBinaries directly using known build output paths.
    print("\n=== Step 5: Construct sonata_tmarb RuntimeBinaries ===")

    UPSTREAM_RUNTIME = Path(__file__).resolve().parents[
        # Navigate to upstream/pypto/runtime
    ]  # [GAP-1] Path resolution — we need the correct relative path

    # Actually, use PROJECT_ROOT which is the pypto upstream root.
    arch, variant = _parse_platform(PLATFORM)
    lib_dir = PROJECT_ROOT / "build" / "lib" / arch / variant / "sonata_tmarb"
    sonata_bins = RuntimeBinaries(
        host_path=lib_dir / "libhost_runtime.so",
        aicpu_path=lib_dir / "libaicpu_kernel.so",
        aicore_path=lib_dir / "libaicore_kernel.so",
        simpler_log_path=PROJECT_ROOT / "build" / "lib" / "libsimpler_log.so",
        sim_context_path=(
            PROJECT_ROOT / "build" / "lib" / "libcpu_sim_context.so"
            if variant == "sim" else None
        ),
        dispatcher_path=None,  # sim has no dispatcher
    )

    # Validate all paths exist
    for field_name in ("host_path", "aicpu_path", "aicore_path", "simpler_log_path"):
        p = getattr(sonata_bins, field_name)
        if p and not p.exists():
            raise FileNotFoundError(
                f"sonata_tmarb binary {field_name}={p} not found. "
                f"Run 'pip install --no-build-isolation .' from upstream/pypto/runtime/."
            )
    if sonata_bins.sim_context_path and not sonata_bins.sim_context_path.exists():
        raise FileNotFoundError(
            f"sim_context={sonata_bins.sim_context_path} not found."
        )
    print(f"  host:      {sonata_bins.host_path}")
    print(f"  aicpu:     {sonata_bins.aicpu_path}")
    print(f"  aicore:    {sonata_bins.aicore_path}")
    print(f"  simpler_log: {sonata_bins.simpler_log_path}")

    # -----------------------------------------------------------------------
    # Step 7: Set env vars for sonata_tmarb
    # -----------------------------------------------------------------------

    # [HARD-4] Set SONATA_AICPU_PATH so runtime_maker.cpp can dlopen
    # the aicpu_kernel.so and find aicpu_execute / aicpu_entry.
    # [GAP-2] On real NPU, the interpreter may need different env vars.
    # [GAP-3] On sim, dlsym(RTLD_DEFAULT) should find aicpu_execute, but
    # we set the var anyway for consistency.
    env = os.environ.copy()
    env["SONATA_AICPU_PATH"] = str(sonata_bins.aicpu_path)
    # Path B: env var fallback (redundant if we embedded in ChipCallable, but
    # useful for debugging).
    env["SONATA_SCHEDULE_PATH"] = str(schedule_path)

    # -----------------------------------------------------------------------
    # Step 8: Build input tensor and orch_args
    # -----------------------------------------------------------------------
    print("\n=== Step 6: Prepare input data ===")

    # [GAP-4] build_orch_args_from_inputs requires the work_dir kernel_config.py
    # which has the parameter ordering. We use it, but the exact tensor names
    # must match kernel_config.py's PARAMS/MUTABLE_PARAMS.
    input_tensor = torch.randn(M, N, dtype=torch.float32)
    output_tensor = torch.zeros(M, N, dtype=torch.float32)
    # The tensor names must match what kernel_config.py lists as PARAMS.
    # By convention: first param is "a", output param is "out".
    inputs_result = [("a", input_tensor), ("out", output_tensor)]
    output_names = {"out"}

    orch_args, all_tensors, input_dict, output_dict = build_orch_args_from_inputs(
        inputs_result, output_names
    )
    print(f"  input shape: {input_tensor.shape}")
    print(f"  output shape: {output_tensor.shape}")

    # -----------------------------------------------------------------------
    # Step 9: Execute via ChipWorker (bypass Worker abstraction)
    # -----------------------------------------------------------------------
    print("\n=== Step 7: Execute on device ===")

    # [GAP-5] We use ChipWorker directly instead of the simpler Worker because
    # Worker._init_level2 calls RuntimeBuilder.get_binaries() which validates
    # against discovered runtimes.  ChipWorker.init() takes RuntimeBinaries
    # directly and does not validate the runtime name.

    cw = ChipWorker()
    cw.init(DEVICE_ID, sonata_bins)

    # Register the Sonata ChipCallable
    sonata_handle = cw.register(sonata_chip)

    cfg = CallConfig()
    # Use defaults from kernel_config if available
    bd = runtime_config.get("block_dim")
    if bd is not None:
        cfg.block_dim = bd
    atn = runtime_config.get("aicpu_thread_num")
    if atn is not None:
        cfg.aicpu_thread_num = atn

    print(f"  running on device {DEVICE_ID}...")
    timing = cw.run(sonata_handle, orch_args, cfg)
    print(f"  host wall:    {timing.host_wall_us} us")
    print(f"  device wall:  {timing.device_wall_us} us")

    # -----------------------------------------------------------------------
    # Step 10: Validate output
    # -----------------------------------------------------------------------
    print("\n=== Step 8: Validate output ===")

    # The output tensor was passed by reference; the runtime wrote into it.
    actual = output_dict["out"]
    expected = torch.abs(input_tensor)

    if torch.allclose(actual, expected, rtol=RTOL, atol=ATOL):
        print(f"  PASSED — output matches torch.abs within rtol={RTOL}, atol={ATOL}")
        print(f"  max diff: {(actual - expected).abs().max().item()}")
    else:
        diff = (actual - expected).abs()
        print(f"  FAILED — output does not match torch.abs")
        print(f"  max diff: {diff.max().item()}")
        print(f"  mean diff: {diff.mean().item()}")
        n_show = min(5, diff.numel())
        flat_diff = diff.flatten()
        flat_actual = actual.flatten()
        flat_expected = expected.flatten()
        for i in range(n_show):
            idx = flat_diff.argmax().item()
            print(f"    worst: [{idx}] actual={flat_actual[idx].item()}, "
                  f"expected={flat_expected[idx].item()}, diff={flat_diff[idx].item()}")
            flat_diff[idx] = 0  # suppress for next iteration
        sys.exit(1)

    # -----------------------------------------------------------------------
    # Cleanup
    # -----------------------------------------------------------------------
    cw.finalize()

print("\n=== ALL PASSED ===")


# ===========================================================================
# Helper
# ===========================================================================

def _parse_platform(platform: str) -> tuple[str, str]:
    """Return (arch, variant) e.g. ('a2a3', 'sim') or ('a2a3', 'onboard')."""
    if platform.endswith("sim"):
        return platform[:-3], "sim"
    return platform, "onboard"
