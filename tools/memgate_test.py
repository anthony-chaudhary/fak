#!/usr/bin/env python3
"""memgate_test.py — unit tests for the memory-pressure gate (pure parsing logic)."""
import importlib.util
import sys
import os

spec = importlib.util.spec_from_file_location(
    "memgate", os.path.join(os.path.dirname(__file__), "memgate.py")
)
memgate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(memgate)


VM_STAT_SAMPLE = """Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free:                          1971366.
Pages active:                         286640.
Pages inactive:                       176020.
Pages speculative:                      1200.
Pages wired down:                     880000.
Pages purgeable:                        4047.
Pages occupied by compressor:          12345.
"""


def test_darwin_parsing(monkeypatch_cap=False):
    def fake_run(cmd):
        if cmd == ["vm_stat"]:
            return VM_STAT_SAMPLE
        if cmd == ["sysctl", "-n", "hw.pagesize"]:
            return "16384\n"
        if cmd == ["sysctl", "-n", "hw.memsize"]:
            return "38654705664\n"
        return ""

    orig = memgate._run
    memgate._run = fake_run
    try:
        d = memgate.read_darwin()
    finally:
        memgate._run = orig

    assert d["total_bytes"] == 38654705664, d
    # 1971366 free pages * 16384 = 32,314,533,888 bytes ~ 32.3GB
    assert d["free_bytes"] == 1971366 * 16384, d
    # 880000 wired * 16384 = 14,417,920,000 ~ 14.4GB (a resident 27B GPU model)
    assert d["wired_bytes"] == 880000 * 16384, d
    # available folds purgeable and keeps the 2GB safety margin
    expected_avail = d["free_bytes"] + 4047 * 16384 - int(memgate.SAFETY_MARGIN_GB * 1e9)
    assert d["available_bytes"] == max(expected_avail, 0), d
    print("test_darwin_parsing OK")


def test_snapshot_high_wired_flag():
    """When wired > 40% of RAM the snapshot flags a likely GPUresident."""
    orig_read = memgate.read_memory
    memgate.read_memory = lambda: {
        "total_bytes": 38654705664,
        "free_bytes": int(32e9),
        "available_bytes": int(30e9),
        "purgeable_bytes": int(1e9),
        "wired_bytes": int(16e9),  # 16GB / 38.6GB ~ 41% -> HIGH
        "compressed_bytes": 0,
    }
    orig_holders = memgate.big_holders
    memgate.big_holders = lambda: []
    try:
        snap = memgate.snapshot()
    finally:
        memgate.read_memory = orig_read
        memgate.big_holders = orig_holders
    assert snap["high_wired"] is True, snap
    assert "GPU" in snap["note"].upper() or "METAL" in snap["note"].upper(), snap["note"]
    print("test_snapshot_high_wired_flag OK")


def test_admit_decision_threshold():
    """admit is True iff available >= require AND wired is not high."""
    orig_read = memgate.read_memory
    orig_holders = memgate.big_holders

    def mem_with(free_gb, wired_gb):
        return {
            "total_bytes": 38654705664,
            "free_bytes": int(free_gb * 1e9),
            "available_bytes": int(free_gb * 1e9),
            "purgeable_bytes": 0,
            "wired_bytes": int(wired_gb * 1e9),
            "compressed_bytes": 0,
        }

    memgate.big_holders = lambda: []
    try:
        # Plenty free, low wired -> admit
        memgate.read_memory = lambda: mem_with(30, 5)
        snap = memgate.snapshot()
        assert snap["available_gb"] >= 20 and not snap["high_wired"]
        # Tight free -> would refuse a 20GB require
        memgate.read_memory = lambda: mem_with(10, 5)
        snap = memgate.snapshot()
        assert snap["available_gb"] < 20
        # High wired (GPUresident) -> flag even if 'free' looks ok
        memgate.read_memory = lambda: mem_with(25, 20)
        snap = memgate.snapshot()
        assert snap["high_wired"] is True
    finally:
        memgate.read_memory = orig_read
        memgate.big_holders = orig_holders
    print("test_admit_decision_threshold OK")


if __name__ == "__main__":
    failures = 0
    for t in [test_darwin_parsing, test_snapshot_high_wired_flag, test_admit_decision_threshold]:
        try:
            t()
        except AssertionError as e:
            print(f"FAIL {t.__name__}: {e}", file=sys.stderr)
            failures += 1
    print(f"\n{'OK' if not failures else str(failures)+' FAILED'} — memgate_test")
    sys.exit(1 if failures else 0)
