#!/usr/bin/env python3
"""Contract test for the claude-glm-gcp preset: GLM-5.2 on the GCP kernel setup, used
from here through the kernel via one preset command.

No GCP, no GPU, and no live GLM endpoint needed — this pins the WIRING that makes the
preset resolve to fak's openai backend at the GLM /v1 with the served id glm-5.2, across
the bash launcher, the PowerShell twin, and the GCP bring-up script. The live model turn
is hardware-gated (stand the node up with scripts/gcp-glm-serve.sh); the preset, the wire,
and the bring-up plan are proven here. See docs/fak/claude-glm-gcp.md.
"""

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SH = (ROOT / "scripts" / "dogfood-claude.sh").read_text(encoding="utf-8")
PS1 = (ROOT / "scripts" / "dogfood-claude.ps1").read_text(encoding="utf-8")
GCP = (ROOT / "scripts" / "gcp-glm-serve.sh").read_text(encoding="utf-8")


def test_bash_launcher_has_glm_gcp_preset():
    # The preset: openai backend, served id glm-5.2, the FAK_GLM_GCP_BASE_URL knob.
    assert "glm-gcp)" in SH
    assert 'DEFAULT_BACKEND="openai"' in SH
    assert 'DEFAULT_OPENAI_BASE_URL="${FAK_GLM_GCP_BASE_URL:-http://127.0.0.1:8200/v1}"' in SH
    assert 'DEFAULT_MODEL="${FAK_GLM_GCP_MODEL:-glm-5.2}"' in SH
    # The invoked-name -> preset mapping and the install symlink.
    assert "claude-glm-gcp)" in SH
    assert 'PRESET="glm-gcp"' in SH
    assert 'glm_name="claude-glm-gcp"' in SH


def test_ps1_launcher_has_glm_gcp_preset_and_openai_backend():
    # The .ps1 had no preset/openai backend before; assert both landed.
    assert "'glm-gcp'" in PS1
    assert "FAK_GLM_GCP_BASE_URL" in PS1
    assert "$OpenaiBackend = ($Backend -eq 'openai')" in PS1
    assert "Resolve-OpenAiBaseUrl" in PS1  # the reachability gate (no dead upstream)
    assert "Get-FirstOpenAiModel" in PS1
    # The install shim that pins the preset for the child only.
    assert "claude-glm-gcp.cmd" in PS1
    assert "FAK_DOGFOOD_PRESET=glm-gcp" in PS1


def test_bringup_runs_preflight_gated_serve_and_hands_off_to_the_preset():
    assert "glm52_sglang_vllm_serve.sh" in GCP  # the preflight-gated on-node serve
    assert "gcp_accel.py" in GCP and "--emit-shell" in GCP  # machine strings from the single registry
    assert "claude-glm-gcp" in GCP  # the hand-off to the preset
    assert "FAK_GLM_GCP_BASE_URL" in GCP
    assert 'MODE="plan"' in GCP  # plan-by-default (reviewable without creds)


def test_bringup_plan_renders_without_creds():
    bash = shutil.which("bash")
    if not bash:
        import pytest

        pytest.skip("bash not on PATH")
    out = subprocess.run(
        [bash, str(ROOT / "scripts" / "gcp-glm-serve.sh")],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )
    assert out.returncode == 0, out.stderr
    text = out.stdout + out.stderr
    assert "gcloud compute instances create" in text
    assert "glm52_sglang_vllm_serve.sh" in text
    assert "claude-glm-gcp" in text
    assert "a3-ultragpu-8g" in text  # the default sm_90+ tier resolved from the registry


def test_default_bringup_tier_clears_the_dsa_floor():
    sys.path.insert(0, str(ROOT / "tools"))
    import gcp_accel

    t = gcp_accel.by_slug("a3-ultra-h200")
    assert t is not None
    # sm_90 is the stock-SGLang/vLLM DSA floor the on-node preflight enforces.
    assert int(t.compute_capability) >= 90


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
