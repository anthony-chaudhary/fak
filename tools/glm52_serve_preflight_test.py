#!/usr/bin/env python3
"""Tests for glm52_serve_preflight.py — the GLM-5.2 SGLang/vLLM go/no-go gate.

Run:  python -m pytest tools/glm52_serve_preflight_test.py
  or:  python tools/glm52_serve_preflight_test.py   (standalone)

Deterministic and offline: nvidia-smi and the engine-import probes are injected
through a fake runner, so the verdict logic is exercised against the real node
shapes (A100 DGX = sm_80, H200/B200 = Hopper/Blackwell, RTX 4090 = Ada) without
any GPU or installed engine present.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import glm52_serve_preflight as pf  # noqa: E402


# --------------------------------------------------------------------------- #
# Fake runner
# --------------------------------------------------------------------------- #

def make_runner(smi_rows: list[tuple[str, str, int, str]] | None, installed: dict[str, str] | None):
    """smi_rows: (index, name, mem_mib, compute_cap). installed: engine -> version."""
    installed = installed or {}

    def runner(cmd):
        argv = list(cmd)
        joined = " ".join(argv)
        if argv and argv[0].endswith("nvidia-smi") or "nvidia-smi" in (argv[0] if argv else ""):
            if smi_rows is None:
                return pf.CmdResult(9, "", "NVIDIA-SMI has failed")
            want_cc = "compute_cap" in joined
            lines = []
            for idx, name, mem, cc in smi_rows:
                if want_cc:
                    lines.append(f"{idx}, {name}, {mem}, {cc}")
                else:
                    lines.append(f"{idx}, {name}, {mem}")
            return pf.CmdResult(0, "\n".join(lines) + "\n", "")
        # engine import probe: python -c "import X,sys; ..."
        if len(argv) >= 3 and argv[1] == "-c":
            code = argv[2]
            for engine, version in installed.items():
                if f"import {engine}," in code:
                    return pf.CmdResult(0, version + "\n", "")
            return pf.CmdResult(1, "", "ModuleNotFoundError")
        return pf.CmdResult(127, "", "unknown command")

    return runner


def report_for(smi_rows, installed, **kw):
    runner = make_runner(smi_rows, installed)
    return pf.build_report(runner=runner, smi_path="/usr/bin/nvidia-smi", python_exe="python", **kw)


# --------------------------------------------------------------------------- #
# capability mapping
# --------------------------------------------------------------------------- #

def test_capability_from_name_known_gpus():
    assert pf.capability_from_name("NVIDIA A100-SXM4-40GB") == 8.0
    assert pf.capability_from_name("NVIDIA A800-SXM4-80GB") == 8.0
    assert pf.capability_from_name("NVIDIA H100 80GB HBM3") == 9.0
    assert pf.capability_from_name("NVIDIA H200") == 9.0
    assert pf.capability_from_name("NVIDIA B200") == 10.0
    assert pf.capability_from_name("NVIDIA GB300") == 10.0
    assert pf.capability_from_name("NVIDIA GeForce RTX 4090") == 8.9
    assert pf.capability_from_name("NVIDIA L40S") == 8.9
    assert pf.capability_from_name("Tesla V100-SXM2-32GB") == 7.0


def test_capability_a100_not_shadowed_by_a10():
    # "a100" must resolve to 8.0 even though "a10" is a substring of "a100".
    assert pf.capability_from_name("NVIDIA A100") == 8.0


def test_capability_unknown_returns_none():
    assert pf.capability_from_name("Some Future GPU XYZ") is None
    assert pf.capability_from_name("") is None


def test_arch_labels():
    assert "Blackwell" in pf.arch_label(10.0)
    assert "Hopper" in pf.arch_label(9.0)
    assert "Ada" in pf.arch_label(8.9)
    assert "Ampere" in pf.arch_label(8.0)
    assert pf.arch_label(None) == "unknown"


# --------------------------------------------------------------------------- #
# memory math
# --------------------------------------------------------------------------- #

def test_required_vram_includes_overhead():
    assert pf.required_vram_gb("fp8", 0.0) == 753.0
    assert pf.required_vram_gb("fp8", 0.15) == round(753.0 * 1.15, 1)
    assert pf.required_vram_gb("int4", 0.0) == 376.0


def test_recommended_quant_picks_highest_fidelity_that_fits():
    # 8x H200 (1128 GB, Hopper): fp8 (~866 GB with 15% overhead) fits -> fp8.
    assert pf.recommended_quant(1128.0, 0.15, 9.0) == "fp8"
    # 320 GB (8x A100-40): even int4 (~432 GB) does not fit.
    assert pf.recommended_quant(320.0, 0.15, 9.0) is None
    # ~500 GB on Hopper: fp8 no, nvfp4 Blackwell-only (skip), w4afp8 (~423) fits.
    assert pf.recommended_quant(500.0, 0.15, 9.0) == "w4afp8"
    # 8x H100-80 (640 GB, Hopper): fp8 no, nvfp4 skip, w4afp8 (~423) -> w4afp8.
    assert pf.recommended_quant(640.0, 0.15, 9.0) == "w4afp8"


def test_recommended_quant_arch_gates_nvfp4_to_blackwell():
    # 744 GB on Hopper: nvfp4 skipped; w4afp8 (~423) outranks int4 -> w4afp8.
    assert pf.recommended_quant(744.0, 0.15, 9.0) == "w4afp8"
    # Same VRAM on Blackwell: nvfp4 (~528) fits and outranks int4 -> nvfp4.
    assert pf.recommended_quant(744.0, 0.15, 10.0) == "nvfp4"
    # Unknown arch (cc None): no gate, size-only -> nvfp4 before int4.
    assert pf.recommended_quant(744.0, 0.15, None) == "nvfp4"


def test_quant_arch_ok():
    assert pf.quant_arch_ok("fp8", 9.0) is True
    assert pf.quant_arch_ok("int4", 9.0) is True
    assert pf.quant_arch_ok("nvfp4", 9.0) is False   # Hopper: no FP4 tensor cores
    assert pf.quant_arch_ok("nvfp4", 10.0) is True    # Blackwell
    assert pf.quant_arch_ok("fp8", None) is False


# --------------------------------------------------------------------------- #
# the A100 DGX — the headline case: BLOCKED on arch (and memory)
# --------------------------------------------------------------------------- #

def a100_dgx_rows(cc: str = "8.0"):
    return [(str(i), "NVIDIA A100-SXM4-40GB", 40960, cc) for i in range(8)]


def test_a100_dgx_blocked_arch_for_both_engines():
    rep = report_for(a100_dgx_rows(), installed={})
    assert rep["node"]["compute_cap"] == 8.0
    assert rep["node"]["gpu_count"] == 8
    assert rep["summary"]["total_vram_gb"] == 320.0
    assert rep["summary"]["node_verdict"] == "BLOCKED_ARCH"
    assert rep["summary"]["any_engine_ready"] is False
    for e in rep["engines"]:
        assert e["verdict"] == "BLOCKED_ARCH"
        assert e["arch_ok"] is False
        assert any("llama.cpp" in n or "H200" in n for n in e["notes"])


def test_a100_dgx_falls_back_to_name_map_when_smi_lacks_compute_cap():
    # Older driver: compute_cap column missing -> name map must recover sm_80.
    rep = report_for(a100_dgx_rows(cc="N/A"), installed={})
    # smi returns N/A; runner drops cc; resolve_node should use name map.
    assert rep["node"]["compute_cap"] == 8.0
    assert rep["node"]["compute_cap_source"] == "name-map"
    assert rep["summary"]["node_verdict"] == "BLOCKED_ARCH"


# --------------------------------------------------------------------------- #
# H200 / B200 — the READY targets
# --------------------------------------------------------------------------- #

def test_h200_x8_ready_pending_install_when_engine_absent():
    rows = [(str(i), "NVIDIA H200", 144384, "9.0") for i in range(8)]  # ~141 GB each
    rep = report_for(rows, installed={})
    assert rep["summary"]["node_verdict"] == "READY_PENDING_INSTALL"
    assert rep["summary"]["any_engine_ready"] is True
    assert rep["summary"]["recommended_quant"] == "fp8"
    for e in rep["engines"]:
        assert e["verdict"] == "READY_PENDING_INSTALL"
        assert e["arch_ok"] is True and e["memory_ok"] is True


def test_h200_x8_ready_when_sglang_installed():
    rows = [(str(i), "NVIDIA H200", 144384, "9.0") for i in range(8)]
    rep = report_for(rows, installed={"sglang": "0.5.0"})
    sg = next(e for e in rep["engines"] if e["engine"] == "sglang")
    vl = next(e for e in rep["engines"] if e["engine"] == "vllm")
    assert sg["verdict"] == "READY"
    assert sg["engine_version"] == "0.5.0"
    assert vl["verdict"] == "READY_PENDING_INSTALL"
    assert rep["summary"]["node_verdict"] == "READY"
    assert "sglang" in rep["summary"]["ready_engines"]


def test_b200_x8_ready_pending_install():
    rows = [(str(i), "NVIDIA B200", 196608, "10.0") for i in range(8)]  # 192 GB each
    rep = report_for(rows, installed={})
    assert rep["summary"]["any_engine_ready"] is True
    assert rep["node"]["arch"].startswith("Blackwell")


# --------------------------------------------------------------------------- #
# RTX 4090 — Ada: blocked on stock, community-port note surfaced
# --------------------------------------------------------------------------- #

def test_rtx4090_blocked_arch_with_ada_port_note():
    rows = [("0", "NVIDIA GeForce RTX 4090", 24564, "8.9")]
    rep = report_for(rows, installed={"vllm": "0.11.0"})
    assert rep["summary"]["node_verdict"] == "BLOCKED_ARCH"
    for e in rep["engines"]:
        assert e["verdict"] == "BLOCKED_ARCH"
        assert any("ada_dsa" in n or "renning22" in n for n in e["notes"])


# --------------------------------------------------------------------------- #
# memory-blocked case: right arch, not enough VRAM
# --------------------------------------------------------------------------- #

def test_blocked_memory_when_arch_ok_but_vram_short():
    # 2x H200 = ~282 GB: arch OK (sm_90) but no quant fits -> BLOCKED_MEMORY.
    rows = [(str(i), "NVIDIA H200", 144384, "9.0") for i in range(2)]
    rep = report_for(rows, installed={"sglang": "0.5.0"})
    assert rep["summary"]["node_verdict"] == "BLOCKED_MEMORY"
    sg = next(e for e in rep["engines"] if e["engine"] == "sglang")
    assert sg["verdict"] == "BLOCKED_MEMORY"
    assert sg["arch_ok"] is True and sg["memory_ok"] is False


def test_blocked_memory_suggests_smaller_quant():
    # 4x H200 = ~564 GB (Hopper): fp8 (~866 GB) does not fit; nvfp4 is Blackwell-
    # only (skipped); w4afp8 (~423 GB) fits and outranks int4 -> suggest w4afp8.
    rep = report_for(
        [(str(i), "NVIDIA H200", 144384, "9.0") for i in range(4)],  # ~564 GB
        installed={"sglang": "0.5.0"},
        quant="fp8",
    )
    sg = next(e for e in rep["engines"] if e["engine"] == "sglang")
    assert sg["verdict"] == "BLOCKED_MEMORY"
    assert sg["recommended_quant"] == "w4afp8"
    assert any("w4afp8" in n for n in sg["notes"])


# --------------------------------------------------------------------------- #
# NVFP4 is Blackwell-only: requesting it on Hopper is BLOCKED_QUANT_ARCH
# --------------------------------------------------------------------------- #

def test_nvfp4_on_hopper_h100_is_blocked_quant_arch():
    # 8x H100-80 (640 GB): nvfp4 fits by size (~528 GB) but needs Blackwell.
    rows = [(str(i), "NVIDIA H100 80GB HBM3", 81920, "9.0") for i in range(8)]
    rep = report_for(rows, installed={"sglang": "0.5.0"}, quant="nvfp4")
    sg = next(e for e in rep["engines"] if e["engine"] == "sglang")
    assert sg["verdict"] == "BLOCKED_QUANT_ARCH"
    assert sg["recommended_quant"] == "w4afp8"   # Hopper 4-bit path
    assert any("INT4" in n or "W4AFP8" in n for n in sg["notes"])
    assert rep["summary"]["node_verdict"] == "BLOCKED_QUANT_ARCH"


def test_int4_on_hopper_h100_is_ready():
    # The chosen GCP tier: 8x H100-80, INT4 -> READY (engine installed).
    rows = [(str(i), "NVIDIA H100 80GB HBM3", 81920, "9.0") for i in range(8)]
    rep = report_for(rows, installed={"sglang": "0.5.0"}, quant="int4")
    sg = next(e for e in rep["engines"] if e["engine"] == "sglang")
    assert sg["verdict"] == "READY"
    assert sg["arch_ok"] is True and sg["memory_ok"] is True


def test_w4afp8_on_hopper_h100_is_ready():
    # The actual chosen path: 8x H100-80, PhalaCloud/GLM-5.2-W4AFP8 -> READY.
    rows = [(str(i), "NVIDIA H100 80GB HBM3", 81920, "9.0") for i in range(8)]
    rep = report_for(rows, installed={"sglang": "0.5.13.post1"}, quant="w4afp8")
    sg = next(e for e in rep["engines"] if e["engine"] == "sglang")
    assert sg["verdict"] == "READY"
    assert sg["arch_ok"] is True and sg["memory_ok"] is True
    assert pf.quant_arch_ok("w4afp8", 9.0) is True


def test_nvfp4_on_blackwell_b200_is_ready():
    rows = [(str(i), "NVIDIA B200", 196608, "10.0") for i in range(8)]
    rep = report_for(rows, installed={"vllm": "0.11.0"}, quant="nvfp4")
    vl = next(e for e in rep["engines"] if e["engine"] == "vllm")
    assert vl["verdict"] == "READY"


# --------------------------------------------------------------------------- #
# planner mode (no GPU on this box) + no nvidia-smi
# --------------------------------------------------------------------------- #

def test_planner_mode_overrides_without_gpu():
    runner = make_runner(smi_rows=None, installed={})  # nvidia-smi fails
    rep = pf.build_report(
        runner=runner,
        smi_path=None,           # simulate "nvidia-smi not found"
        smi_autodetect=False,    # ...and do not fall back to the host's real one
        python_exe="python",
        override_name="NVIDIA H200",
        override_count=8,
        override_total_gb=1128.0,
        probe_engines=False,
    )
    assert rep["node"]["compute_cap"] == 9.0
    assert rep["node"]["compute_cap_source"] == "name-map"
    assert rep["summary"]["node_verdict"] == "READY_PENDING_INSTALL"
    assert rep["nvidia_smi"]["status"] == "UNAVAILABLE"


def test_no_nvidia_smi_no_override_is_blocked_arch():
    runner = make_runner(smi_rows=None, installed={})
    rep = pf.build_report(runner=runner, smi_path=None, smi_autodetect=False,
                          python_exe="python", probe_engines=False)
    assert rep["node"]["compute_cap"] is None
    assert rep["summary"]["node_verdict"] == "BLOCKED_ARCH"
    assert rep["summary"]["any_engine_ready"] is False


# --------------------------------------------------------------------------- #
# report shape + renderers + CLI gate
# --------------------------------------------------------------------------- #

def test_report_schema_and_markdown():
    rep = report_for(a100_dgx_rows(), installed={})
    assert rep["schema"] == pf.SCHEMA
    assert rep["generated_at"].endswith("Z")
    md = pf.render_markdown(rep)
    assert "GLM-5.2 SGLang/vLLM serving-readiness preflight" in md
    assert "BLOCKED_ARCH" in md
    # JSON round-trips.
    json.loads(json.dumps(rep))


def test_require_ready_exit_code():
    # --require-ready gates the exit code. Force nvidia-smi "absent" so the
    # planner overrides drive the verdict deterministically on any host.
    import glm52_serve_preflight as mod
    orig = mod.real_runner
    mod.real_runner = make_runner(smi_rows=None, installed={})
    try:
        rc = mod.main(["--gpu-name", "NVIDIA A100-SXM4-40GB", "--gpu-count", "8",
                       "--gpu-memory-total-gb", "320", "--no-probe-engines", "--require-ready"])
        assert rc == 1
        rc2 = mod.main(["--gpu-name", "NVIDIA H200", "--gpu-count", "8",
                        "--gpu-memory-total-gb", "1128", "--no-probe-engines", "--require-ready"])
        assert rc2 == 0
    finally:
        mod.real_runner = orig


def test_single_engine_selection():
    rows = [(str(i), "NVIDIA H200", 144384, "9.0") for i in range(8)]
    rep = report_for(rows, installed={}, engines=("vllm",))
    assert [e["engine"] for e in rep["engines"]] == ["vllm"]


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-q"]))
