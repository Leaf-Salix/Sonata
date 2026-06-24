"""Tests for binary flat schedule serialization (from/to sonata_tmarb flat format)."""

import ctypes
import struct
from pathlib import Path

import pytest

from sonata.schedule import (
    ArgBinding,
    ArgDirection,
    BINARY_FORMAT_VERSION,
    ScheduleDep,
    ScheduleDecodeError,
    ScheduledRegion,
    ScheduledTask,
    ScopeMode,
    SonataScheduleContract,
)


# ── ctypes struct definitions matching flat_schedule.h (packed) ──

class _CFlatSchedule(ctypes.Structure):
    """Matches FlatSchedule in flat_schedule.h with #pragma pack(push, 1)."""
    _pack_ = 1
    _fields_ = [
        ("magic", ctypes.c_int32),
        ("version", ctypes.c_int32),
        ("num_regions", ctypes.c_int32),
        ("total_tasks", ctypes.c_int32),
        ("total_args", ctypes.c_int32),
        ("total_deps", ctypes.c_int32),
        ("fingerprint", ctypes.c_char * 64),
    ]


class _CFlatRegion(ctypes.Structure):
    """Matches FlatRegion in flat_schedule.h."""
    _pack_ = 1
    _fields_ = [
        ("kind", ctypes.c_int32),
        ("scope_mode", ctypes.c_int32),
        ("task_start", ctypes.c_int32),
        ("num_tasks", ctypes.c_int32),
        ("dep_start", ctypes.c_int32),
        ("num_deps", ctypes.c_int32),
    ]


class _CFlatTask(ctypes.Structure):
    """Matches FlatTask in flat_schedule.h."""
    _pack_ = 1
    _fields_ = [
        ("task_id", ctypes.c_int32),
        ("func_id", ctypes.c_int32),
        ("core_type", ctypes.c_int16),
        ("num_args", ctypes.c_int16),
        ("arg_base", ctypes.c_int32),
    ]


class _CFlatArg(ctypes.Structure):
    """Matches FlatArg in flat_schedule.h."""
    _pack_ = 1
    _fields_ = [
        ("runtime_slot", ctypes.c_int32),
        ("direction", ctypes.c_int16),
    ]


class _CFlatDep(ctypes.Structure):
    """Matches FlatDep in flat_schedule.h."""
    _pack_ = 1
    _fields_ = [
        ("producer", ctypes.c_int32),
        ("consumer", ctypes.c_int32),
    ]


# ── Struct size and packing assertions (Phase A1) ──

def _sum_field_sizes(struct_cls):
    """Sum of ctypes field sizes. If == sizeof, then pack(1) is enforced."""
    return sum(
        ctypes.sizeof(field_type)
        for field_name, field_type in struct_cls._fields_
    )


def _field_offsets(struct_cls):
    """Compute field offsets for a pack(1) ctypes struct by iterating fields.

    With pack(1) and no padding, offsets are the cumulative sum of prior
    field sizes.  This is verified by the *_size and *_padding tests.
    """
    offsets = {}
    off = 0
    for name, ftype in struct_cls._fields_:
        offsets[name] = off
        off += ctypes.sizeof(ftype)
    return offsets


class TestBinaryStructLayout:
    """A1: Validate binary format against flat_schedule.h struct alignment."""

    def test_flat_schedule_size(self):
        """FlatSchedule must be exactly 88 bytes: 6×int32 + 64-char fingerprint."""
        assert ctypes.sizeof(_CFlatSchedule) == 88

    def test_flat_region_size(self):
        """FlatRegion must be exactly 24 bytes: 6×int32."""
        assert ctypes.sizeof(_CFlatRegion) == 24

    def test_flat_task_size(self):
        """FlatTask must be exactly 16 bytes: 2×int32 + 2×int16 + int32."""
        assert ctypes.sizeof(_CFlatTask) == 16

    def test_flat_arg_size(self):
        """FlatArg must be exactly 6 bytes: int32 + int16."""
        assert ctypes.sizeof(_CFlatArg) == 6

    def test_flat_dep_size(self):
        """FlatDep must be exactly 8 bytes: 2×int32."""
        assert ctypes.sizeof(_CFlatDep) == 8

    def test_pack_no_padding_flat_schedule(self):
        """Assert #pragma pack(push, 1) — no padding in FlatSchedule."""
        assert ctypes.sizeof(_CFlatSchedule) == _sum_field_sizes(_CFlatSchedule)

    def test_pack_no_padding_flat_region(self):
        """Assert no padding in FlatRegion."""
        assert ctypes.sizeof(_CFlatRegion) == _sum_field_sizes(_CFlatRegion)

    def test_pack_no_padding_flat_task(self):
        """Assert no padding in FlatTask."""
        assert ctypes.sizeof(_CFlatTask) == _sum_field_sizes(_CFlatTask)

    def test_pack_no_padding_flat_arg(self):
        """Assert no padding in FlatArg."""
        assert ctypes.sizeof(_CFlatArg) == _sum_field_sizes(_CFlatArg)

    def test_pack_no_padding_flat_dep(self):
        """Assert no padding in FlatDep."""
        assert ctypes.sizeof(_CFlatDep) == _sum_field_sizes(_CFlatDep)

    def test_field_offsets_flat_schedule(self):
        """Verify field offsets in FlatSchedule match C struct layout."""
        assert _field_offsets(_CFlatSchedule) == {
            "magic": 0,
            "version": 4,
            "num_regions": 8,
            "total_tasks": 12,
            "total_args": 16,
            "total_deps": 20,
            "fingerprint": 24,
        }

    def test_field_offsets_flat_region(self):
        """Verify field offsets in FlatRegion match C struct layout."""
        assert _field_offsets(_CFlatRegion) == {
            "kind": 0,
            "scope_mode": 4,
            "task_start": 8,
            "num_tasks": 12,
            "dep_start": 16,
            "num_deps": 20,
        }

    def test_field_offsets_flat_task(self):
        """Verify field offsets in FlatTask match C struct layout."""
        assert _field_offsets(_CFlatTask) == {
            "task_id": 0,
            "func_id": 4,
            "core_type": 8,
            "num_args": 10,
            "arg_base": 12,
        }

    def test_field_offsets_flat_arg(self):
        """Verify field offsets in FlatArg match C struct layout."""
        assert _field_offsets(_CFlatArg) == {
            "runtime_slot": 0,
            "direction": 4,
        }

    def test_field_offsets_flat_dep(self):
        """Verify field offsets in FlatDep match C struct layout."""
        assert _field_offsets(_CFlatDep) == {
            "producer": 0,
            "consumer": 4,
        }

    def test_header_bytes_written_by_python_match_flat_schedule_size(self):
        """Assert to_binary() header byte count equals sizeof(FlatSchedule).

        The Python serializer writes 24 bytes of int fields + 64 bytes of
        fingerprint = 88 bytes total for the header, matching the C struct.
        """
        c = SonataScheduleContract(fingerprint="fp_a1")
        data = c.to_binary()
        # Header is at [0:88]; verify via ctypes int fields + raw fp at expected offset
        s = _CFlatSchedule.from_buffer_copy(data[:88])
        assert s.magic == 0x534F4E41
        assert s.version == BINARY_FORMAT_VERSION
        assert s.num_regions == 0
        assert s.num_regions == len(c.regions)
        # Verify fingerprint bytes directly at offset 24 (int fields = 6×4 = 24)
        fp_raw = data[24:88]
        assert fp_raw.startswith(b"fp_a1")
        assert fp_raw.count(b"\x00") >= 59  # null-padded (fp_a1 = 5 chars)

    def test_ctypes_overlay_header_values(self):
        """Overlay ctypes FlatSchedule on binary data and validate field values."""
        c = SonataScheduleContract(fingerprint="overlay_test", regions=(
            ScheduledRegion(region_id="r0", kind="static", tasks=(
                ScheduledTask(task_id=0, kernel_identity="k", func_id=42, core_type="aic",
                    args=(ArgBinding(arg_identity="x", direction=ArgDirection.INPUT),)),
            )),
        ))
        data = c.to_binary()
        # Overlay header
        s = _CFlatSchedule.from_buffer_copy(data[:88])
        assert s.magic == 0x534F4E41
        assert s.version == BINARY_FORMAT_VERSION
        assert s.num_regions == 1
        assert s.total_tasks == 1
        assert s.total_args == 1
        assert s.total_deps == 0
        assert s.fingerprint == b"overlay_test"  # c_char*64 truncates at null

    def test_ctypes_overlay_region(self):
        """Overlay ctypes FlatRegion and verify field values match contract."""
        r0 = ScheduledRegion(region_id="r0", kind="static", scope_mode=ScopeMode.MANUAL)
        c = SonataScheduleContract(fingerprint="fp_r", regions=(r0,))
        data = c.to_binary()
        r_off, t_off, a_off, d_off = _compute_offsets(data)
        region_bytes = data[r_off:r_off + 24]  # sizeof = 24
        r = _CFlatRegion.from_buffer_copy(region_bytes)
        assert r.kind == 0        # static
        assert r.scope_mode == 1  # manual
        assert r.task_start == 0
        assert r.num_tasks == 0
        assert r.dep_start == 0
        assert r.num_deps == 0

    def test_ctypes_overlay_task(self):
        """Overlay ctypes FlatTask and verify field values."""
        t1 = ScheduledTask(task_id=0, kernel_identity="add", func_id=7, core_type="aiv",
            args=(ArgBinding(arg_identity="x"), ArgBinding(arg_identity="y")))
        r0 = ScheduledRegion(region_id="r0", kind="static", tasks=(t1,))
        c = SonataScheduleContract(fingerprint="fp_t", regions=(r0,))
        data = c.to_binary()
        r_off, t_off, a_off, d_off = _compute_offsets(data)
        task_bytes = data[t_off:t_off + 16]  # sizeof = 16
        t = _CFlatTask.from_buffer_copy(task_bytes)
        assert t.task_id == 0
        assert t.func_id == 7
        assert t.core_type == 1    # aiv
        assert t.num_args == 2
        # arg_base depends on contract state — just verify it's >= 0
        assert t.arg_base >= 0

    def test_ctypes_overlay_arg(self):
        """Overlay ctypes FlatArg and verify direction and runtime_slot."""
        args = (
            ArgBinding(arg_identity="in", runtime_slot=0, direction=ArgDirection.INPUT),
            ArgBinding(arg_identity="out", runtime_slot=1, direction=ArgDirection.OUTPUT),
            ArgBinding(arg_identity="scalar", runtime_slot=None, direction=ArgDirection.SCALAR),
            ArgBinding(arg_identity="nodep", runtime_slot=2, direction=ArgDirection.NO_DEP),
        )
        t1 = ScheduledTask(task_id=0, kernel_identity="k", func_id=1, core_type="aic", args=args)
        r0 = ScheduledRegion(region_id="r0", kind="static", tasks=(t1,))
        c = SonataScheduleContract(fingerprint="fp_a", regions=(r0,))
        data = c.to_binary()
        r_off, t_off, a_off, d_off = _compute_offsets(data)
        arg_bytes = data[a_off:a_off + 4 * 6]  # 4 args × 6 bytes = 24
        arg0 = _CFlatArg.from_buffer_copy(arg_bytes[0:6])
        assert arg0.runtime_slot == 0
        assert arg0.direction == 0  # INPUT
        arg1 = _CFlatArg.from_buffer_copy(arg_bytes[6:12])
        assert arg1.runtime_slot == 1
        assert arg1.direction == 1  # OUTPUT
        arg2 = _CFlatArg.from_buffer_copy(arg_bytes[12:18])
        assert arg2.runtime_slot == -1  # SCALAR → None encoded as -1
        assert arg2.direction == 3  # SCALAR
        arg3 = _CFlatArg.from_buffer_copy(arg_bytes[18:24])
        assert arg3.direction == 4  # NO_DEP

    def test_ctypes_overlay_dep(self):
        """Overlay ctypes FlatDep and verify producer/consumer values."""
        t1 = ScheduledTask(task_id=0, kernel_identity="a", func_id=1, core_type="aic",
            args=(ArgBinding(arg_identity="x"),))
        t2 = ScheduledTask(task_id=1, kernel_identity="b", func_id=2, core_type="aic",
            args=(ArgBinding(arg_identity="y"),))
        deps = (ScheduleDep(producer=0, consumer=1), ScheduleDep(producer=0, consumer=2))
        r0 = ScheduledRegion(region_id="r0", kind="static", tasks=(t1, t2), deps=deps)
        c = SonataScheduleContract(fingerprint="fp_d", regions=(r0,))
        data = c.to_binary()
        r_off, t_off, a_off, d_off = _compute_offsets(data)
        d0 = _CFlatDep.from_buffer_copy(data[d_off:d_off + 8])
        assert d0.producer == 0
        assert d0.consumer == 1
        d1 = _CFlatDep.from_buffer_copy(data[d_off + 8:d_off + 16])
        assert d1.producer == 0
        assert d1.consumer == 2

    def test_python_serialized_size_matches_struct_formula(self):
        """Assert total binary size matches what from_binary expects.

        total_bytes = 88 + nr*24 + total_tasks*16 + total_args*6 + total_deps*8
        """
        tasks = tuple(
            ScheduledTask(task_id=i, kernel_identity=f"k{i}", func_id=i, core_type="aic",
                args=(ArgBinding(arg_identity=f"x{i}"),))
            for i in range(3)
        )
        deps = tuple(ScheduleDep(producer=i, consumer=i + 1) for i in range(2))
        r0 = ScheduledRegion(region_id="r0", kind="static", tasks=tasks, deps=deps)
        c = SonataScheduleContract(fingerprint="fp_sz", regions=(r0,))
        data = c.to_binary()
        expected = 88 + 1*24 + 3*16 + 3*6 + 2*8
        assert len(data) >= expected, (
            f"Binary too small: {len(data)} < {expected}"
        )
        # String table appended after deps — account for it
        str_table_overhead = len(data) - expected
        assert str_table_overhead > 0  # kernel + arg identities present


def _compute_offsets(data: bytes):
    """Compute struct array offsets from binary header (helper for tests).

    v1: header (88 bytes), no CRC → payload at offset 88.
    v2: header (88 bytes) + CRC-32 (4 bytes) → payload at offset 92.
    """
    s = _CFlatSchedule.from_buffer_copy(data[:88])
    payload_off = 92 if s.version >= 2 else 88
    r_off = payload_off
    t_off = r_off + s.num_regions * ctypes.sizeof(_CFlatRegion)
    a_off = t_off + s.total_tasks * ctypes.sizeof(_CFlatTask)
    d_off = a_off + s.total_args * ctypes.sizeof(_CFlatArg)
    return r_off, t_off, a_off, d_off


def _make_static_region(tasks=(), deps=()):
    return ScheduledRegion(region_id="r0", kind="static", tasks=tuple(tasks), deps=tuple(deps))


class TestBinaryCrc:
    """A2: CRC-32 checksum validation in v2 binary format."""

    def test_crc_present_in_v2(self):
        """v2 binary has 4-byte CRC after the 88-byte header."""
        c = SonataScheduleContract(fingerprint="fp_crc")
        data = c.to_binary()
        # Header = 88 bytes, CRC = 4 bytes at 88..91
        assert len(data) >= 92, f"v2 data too short: {len(data)}"
        crc_bytes = data[88:92]
        import struct, zlib
        crc_value = struct.unpack_from("<I", crc_bytes, 0)[0]
        # CRC covers only struct arrays (deterministic from header fields),
        # not the optional string table.
        s = _CFlatSchedule.from_buffer_copy(data[:88])
        arrays_size = (s.num_regions * ctypes.sizeof(_CFlatRegion)
                       + s.total_tasks * ctypes.sizeof(_CFlatTask)
                       + s.total_args * ctypes.sizeof(_CFlatArg)
                       + s.total_deps * ctypes.sizeof(_CFlatDep))
        expected = zlib.crc32(data[92:92 + arrays_size])
        assert crc_value == expected, (
            f"CRC mismatch: stored={crc_value:#010x}, computed={expected:#010x}"
        )

    def test_crc_corrupted_raises_error(self):
        """Corrupted CRC in v2 blob raises ScheduleDecodeError."""
        c = SonataScheduleContract(fingerprint="fp_crc_bad")
        data = c.to_binary()
        # Corrupt the CRC byte
        bad_data = bytearray(data)
        bad_data[88] ^= 0xFF  # flip all bits in first CRC byte
        with pytest.raises(ScheduleDecodeError, match="CRC mismatch"):
            SonataScheduleContract.from_binary(bytes(bad_data))

    def test_crc_round_trip_validates(self):
        """A valid v2 blob passes CRC check in from_binary."""
        t1 = ScheduledTask(task_id=0, kernel_identity="k", func_id=1, core_type="aic",
            args=(ArgBinding(arg_identity="x"),))
        r0 = ScheduledRegion(region_id="r0", kind="static", tasks=(t1,))
        c = SonataScheduleContract(fingerprint="fp_crc_rt", regions=(r0,))
        data = c.to_binary()
        c2 = SonataScheduleContract.from_binary(data)
        assert c2.fingerprint == "fp_crc_rt"
        assert len(c2.regions) == 1

    def test_v1_blob_accepted_without_crc(self):
        """v1 blob (no CRC) is still accepted for backward compatibility."""
        # Manually construct a v1 binary blob
        import struct, zlib
        magic = 0x534F4E41
        version = 1
        nr = 0
        total_tasks = 0
        total_args = 0
        total_deps = 0
        header = struct.pack("<iiiiii", magic, version, nr, total_tasks, total_args, total_deps)
        fp = b"v1_compat" + b"\x00" * 55  # pad to 64 bytes
        v1_data = header + fp  # exactly 88 bytes, no CRC
        c = SonataScheduleContract.from_binary(v1_data)
        assert c.fingerprint == "v1_compat"
        assert len(c.regions) == 0

    def test_unsupported_version_rejected(self):
        """Unknown version raises ScheduleDecodeError."""
        import struct
        magic = 0x534F4E41
        version = 99  # unknown
        nr = 0
        header = struct.pack("<iiiiii", magic, version, nr, 0, 0, 0)
        fp = b"x" * 64
        bad_data = header + fp
        with pytest.raises(ScheduleDecodeError, match="unsupported version"):
            SonataScheduleContract.from_binary(bad_data)

    def test_payload_size_includes_crc(self):
        """Total v2 binary = header (88) + CRC (4) + payload."""
        c = SonataScheduleContract(fingerprint="fp_crc_sz")
        data = c.to_binary()
        # Header: 88, CRC: 4, rest is payload
        assert len(data) >= 92
        # Round trip still works
        c2 = SonataScheduleContract.from_binary(data)
        assert c2.fingerprint == "fp_crc_sz"


class TestBinarySerialization:
    def test_empty_schedule_binary(self):
        c = SonataScheduleContract(fingerprint="fp")
        data = c.to_binary()
        c2 = SonataScheduleContract.from_binary(data)
        assert c2.fingerprint == "fp"
        assert len(c2.regions) == 0

    def test_static_only_round_trip(self):
        t1 = ScheduledTask(task_id=0, kernel_identity="add", func_id=3, core_type="aic",
            args=(ArgBinding(arg_identity="x", direction=ArgDirection.INPUT),))
        r0 = ScheduledRegion(region_id="r0", kind="static", tasks=(t1,),
            scope_mode=ScopeMode.AUTO)
        c = SonataScheduleContract(fingerprint="fp_001", regions=(r0,))
        data = c.to_binary()
        c2 = SonataScheduleContract.from_binary(data)
        assert c2.fingerprint == "fp_001"
        assert len(c2.regions) == 1
        assert c2.regions[0].kind == "static"
        assert len(c2.regions[0].tasks) == 1

    def test_static_with_deps_round_trip(self):
        t1 = ScheduledTask(task_id=0, kernel_identity="add", func_id=3, core_type="aic",
            args=(ArgBinding(arg_identity="x"),))
        t2 = ScheduledTask(task_id=1, kernel_identity="mul", func_id=5, core_type="aic",
            args=(ArgBinding(arg_identity="z"),))
        r0 = ScheduledRegion(region_id="r0", kind="static", tasks=(t1, t2),
            deps=(ScheduleDep(producer=0, consumer=1),))
        c = SonataScheduleContract(fingerprint="fp_deps", regions=(r0,))
        data = c.to_binary()
        c2 = SonataScheduleContract.from_binary(data)
        assert len(c2.regions) == 1
        assert len(c2.regions[0].tasks) == 2
        assert len(c2.regions[0].deps) == 1

    def test_static_dynamic_mixed(self):
        t1 = ScheduledTask(task_id=0, kernel_identity="k", func_id=1, core_type="aic",
            args=(ArgBinding(arg_identity="x"),))
        r0 = ScheduledRegion(region_id="r0", kind="static", tasks=(t1,))
        r1 = ScheduledRegion(region_id="r1", kind="dynamic", dynamic_mode="backend_dynamic")
        c = SonataScheduleContract(fingerprint="fp_mix", regions=(r0, r1))
        data = c.to_binary()
        c2 = SonataScheduleContract.from_binary(data)
        assert len(c2.regions) == 2
        assert c2.regions[0].kind == "static"
        assert c2.regions[1].kind == "dynamic"

    def test_all_directions_in_args(self):
        args = [
            ArgBinding(arg_identity="a", direction=ArgDirection.INPUT),
            ArgBinding(arg_identity="b", direction=ArgDirection.OUTPUT),
            ArgBinding(arg_identity="c", direction=ArgDirection.INOUT),
            ArgBinding(arg_identity="d", direction=ArgDirection.OUTPUT_EXISTING),
            ArgBinding(arg_identity="e", direction=ArgDirection.NO_DEP),
            ArgBinding(arg_identity="n", direction=ArgDirection.SCALAR),
        ]
        t1 = ScheduledTask(task_id=0, kernel_identity="k", func_id=1, core_type="aic",
            args=tuple(args))
        r0 = ScheduledRegion(region_id="r0", kind="static", tasks=(t1,))
        c = SonataScheduleContract(fingerprint="fp_dir", regions=(r0,))
        data = c.to_binary()
        c2 = SonataScheduleContract.from_binary(data)
        assert len(c2.regions[0].tasks[0].args) == 6

    def test_magic_number(self):
        c = SonataScheduleContract(fingerprint="fp")
        data = c.to_binary()
        import struct
        magic = struct.unpack_from("<i", data, 0)[0]
        assert magic == 0x534F4E41  # "SONA"

    def test_deterministic(self):
        t1 = ScheduledTask(task_id=0, kernel_identity="k", func_id=1, core_type="aic",
            args=(ArgBinding(arg_identity="x"),))
        r0 = ScheduledRegion(region_id="r0", kind="static", tasks=(t1,))
        c = SonataScheduleContract(fingerprint="fp_det", regions=(r0,))
        d1 = c.to_binary()
        d2 = c.to_binary()
        assert d1 == d2

    def test_all_directions_round_trip_values(self):
        """Verify each direction round-trips to its correct value, not just count."""
        args = [
            ArgBinding(arg_identity="a", direction=ArgDirection.INPUT),
            ArgBinding(arg_identity="b", direction=ArgDirection.OUTPUT),
            ArgBinding(arg_identity="c", direction=ArgDirection.INOUT),
            ArgBinding(arg_identity="d", direction=ArgDirection.OUTPUT_EXISTING),
            ArgBinding(arg_identity="e", direction=ArgDirection.NO_DEP),
            ArgBinding(arg_identity="n", direction=ArgDirection.SCALAR),
        ]
        t1 = ScheduledTask(task_id=0, kernel_identity="k", func_id=1, core_type="aic", args=tuple(args))
        r0 = ScheduledRegion(region_id="r0", kind="static", tasks=(t1,))
        c = SonataScheduleContract(fingerprint="fp_dir", regions=(r0,))
        d = c.to_binary()
        c2 = SonataScheduleContract.from_binary(d)
        rtasks = c2.regions[0].tasks[0].args
        dirs = [a.direction for a in rtasks]
        assert dirs[0] == ArgDirection.INPUT
        assert dirs[1] == ArgDirection.OUTPUT
        assert dirs[2] == ArgDirection.INOUT
        assert dirs[3] == ArgDirection.OUTPUT_EXISTING
        assert dirs[4] == ArgDirection.NO_DEP
        assert dirs[5] == ArgDirection.SCALAR

    def test_scope_mode_round_trip(self):
        t1 = ScheduledTask(task_id=0, kernel_identity="k", func_id=1, core_type="aic",
            args=(ArgBinding(arg_identity="x"),))
        r0 = ScheduledRegion(region_id="r0", kind="static", tasks=(t1,), scope_mode=ScopeMode.MANUAL)
        c = SonataScheduleContract(fingerprint="fp_scope", regions=(r0,))
        d = c.to_binary()
        c2 = SonataScheduleContract.from_binary(d)
        assert c2.regions[0].scope_mode == ScopeMode.MANUAL

    def test_aiv_core_type(self):
        t1 = ScheduledTask(task_id=0, kernel_identity="k", func_id=1, core_type="aiv",
            args=(ArgBinding(arg_identity="x"),))
        r0 = _make_static_region(tasks=(t1,))
        c = SonataScheduleContract(fingerprint="fp_aiv", regions=(r0,))
        d = c.to_binary()
        c2 = SonataScheduleContract.from_binary(d)
        assert c2.regions[0].tasks[0].core_type == "aiv"

    def test_bad_magic_rejected(self):
        # Header is 88 bytes; provide enough for "too short" not to trigger first
        data = b"\x00" * 88
        with pytest.raises(ValueError, match="bad magic"):
            SonataScheduleContract.from_binary(data)

    def test_truncated_data_rejected(self):
        with pytest.raises(ScheduleDecodeError, match="too short"):
            SonataScheduleContract.from_binary(b"\x00" * 10)

    def test_large_schedule(self):
        tasks = tuple(
            ScheduledTask(task_id=i, kernel_identity=f"k{i}", func_id=i, core_type="aic",
                args=(ArgBinding(arg_identity=f"x{i}"),))
            for i in range(100)
        )
        deps = tuple(
            ScheduleDep(producer=i, consumer=i + 1)
            for i in range(99)
        )
        r0 = ScheduledRegion(region_id="r0", kind="static", tasks=tasks, deps=deps)
        c = SonataScheduleContract(fingerprint="fp_large", regions=(r0,))
        data = c.to_binary()
        assert len(data) > 0
        c2 = SonataScheduleContract.from_binary(data)
        assert len(c2.regions[0].tasks) == 100
        assert len(c2.regions[0].deps) == 99

    def test_multi_region_deps_round_trip(self):
        """S4: Deps in each region round-trip independently."""
        t0 = ScheduledTask(task_id=0, kernel_identity="a", func_id=1, core_type="aic",
            args=(ArgBinding(arg_identity="x"),))
        t1 = ScheduledTask(task_id=1, kernel_identity="b", func_id=2, core_type="aic",
            args=(ArgBinding(arg_identity="y"),))
        t2 = ScheduledTask(task_id=2, kernel_identity="c", func_id=3, core_type="aic",
            args=(ArgBinding(arg_identity="z"),))
        t3 = ScheduledTask(task_id=3, kernel_identity="d", func_id=4, core_type="aic",
            args=(ArgBinding(arg_identity="w"),))
        r0 = ScheduledRegion(region_id="r0", kind="static",
            tasks=(t0, t1), deps=(ScheduleDep(producer=0, consumer=1),))
        r1 = ScheduledRegion(region_id="r1", kind="static",
            tasks=(t2, t3), deps=(ScheduleDep(producer=0, consumer=1),))
        c = SonataScheduleContract(fingerprint="fp_mr", regions=(r0, r1))
        data = c.to_binary()
        c2 = SonataScheduleContract.from_binary(data)
        assert len(c2.regions) == 2
        assert len(c2.regions[0].deps) == 1
        assert len(c2.regions[1].deps) == 1
        assert c2.regions[0].deps[0] == ScheduleDep(producer=0, consumer=1)
        assert c2.regions[1].deps[0] == ScheduleDep(producer=0, consumer=1)
        assert c2.regions[0].tasks[0].func_id == 1
        assert c2.regions[1].tasks[0].func_id == 3

    def test_runtime_slot_round_trip(self):
        """S5: runtime_slot=0 survives round-trip (not conflated with None)."""
        a_slot0 = ArgBinding(arg_identity="a", runtime_slot=0, direction=ArgDirection.INPUT)
        a_none = ArgBinding(arg_identity="b", runtime_slot=None, direction=ArgDirection.OUTPUT)
        a_slot5 = ArgBinding(arg_identity="c", runtime_slot=5, direction=ArgDirection.INOUT)
        t1 = ScheduledTask(task_id=0, kernel_identity="k", func_id=1, core_type="aic",
            args=(a_slot0, a_none, a_slot5))
        r0 = ScheduledRegion(region_id="r0", kind="static", tasks=(t1,))
        c = SonataScheduleContract(fingerprint="fp_slot", regions=(r0,))
        d = c.to_binary()
        c2 = SonataScheduleContract.from_binary(d)
        args = c2.regions[0].tasks[0].args
        assert args[0].runtime_slot == 0
        assert args[1].runtime_slot is None
        assert args[2].runtime_slot == 5

    def test_string_table_kernel_identity_round_trip(self):
        """W4: kernel_identity and arg_identity survive binary round-trip via string table."""
        a1 = ArgBinding(arg_identity="input_tensor_x", runtime_slot=0, direction=ArgDirection.INPUT)
        a2 = ArgBinding(arg_identity="output_tensor_z", runtime_slot=1, direction=ArgDirection.OUTPUT)
        t1 = ScheduledTask(task_id=0, kernel_identity="tile_abs", func_id=3, core_type="aic",
            args=(a1, a2))
        t2 = ScheduledTask(task_id=1, kernel_identity="tile_add", func_id=5, core_type="aic",
            args=(ArgBinding(arg_identity="scratch_buf"),))
        r0 = ScheduledRegion(region_id="r0", kind="static", tasks=(t1, t2),
            deps=(ScheduleDep(producer=0, consumer=1),))
        c = SonataScheduleContract(fingerprint="fp_str", regions=(r0,))
        d = c.to_binary()
        c2 = SonataScheduleContract.from_binary(d)
        assert c2.regions[0].tasks[0].kernel_identity == "tile_abs"
        assert c2.regions[0].tasks[1].kernel_identity == "tile_add"
        assert c2.regions[0].tasks[0].args[0].arg_identity == "input_tensor_x"
        assert c2.regions[0].tasks[0].args[1].arg_identity == "output_tensor_z"
        assert c2.regions[0].tasks[1].args[0].arg_identity == "scratch_buf"


class TestGoldenFixtures:
    """A4: Golden binary regression tests."""

    FIXTURE_DIR = Path(__file__).parent / "fixtures" / "binary"

    @pytest.mark.parametrize("name,exp_fp,exp_regions", [
        ("single-region", "golden_single", 1),
        ("multi-region", "golden_multi", 2),
        ("empty-deps", "golden_nodeps", 1),
        ("string-table", "golden_str", 1),
    ])
    def test_golden_binary_loads(self, name, exp_fp, exp_regions):
        """Golden .bin files load correctly via from_binary."""
        bin_path = self.FIXTURE_DIR / f"{name}.bin"
        assert bin_path.exists(), f"Missing golden fixture: {bin_path}"
        data = bin_path.read_bytes()
        c = SonataScheduleContract.from_binary(data)
        assert c.fingerprint == exp_fp, f"{name}: fingerprint mismatch"
        assert len(c.regions) == exp_regions, f"{name}: region count mismatch"

    @pytest.mark.parametrize("name", [
        "single-region", "multi-region", "empty-deps", "string-table",
    ])
    def test_golden_deterministic(self, name):
        """Re-serializing a golden fixture produces identical bytes."""
        bin_path = self.FIXTURE_DIR / f"{name}.bin"
        data = bin_path.read_bytes()
        c = SonataScheduleContract.from_binary(data)
        re_encoded = c.to_binary()
        assert data == re_encoded, (
            f"{name}: re-serialized bytes differ from golden (determinism failure)"
        )
