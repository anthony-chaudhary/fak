---
title: "Issue #9827 — AMD RX 7600 WSL ROCm admission witness for vLLM/SGLang"
description: "Coordinator-backed witness separating successful user-scoped ROCm setup from ROCm GPU admission failure and unattempted engine throughput."
---

# Issue #9827 — AMD RX 7600 WSL ROCm admission witness for vLLM/SGLang

**Overall outcome:** blocked at GPU runtime admission.

- **Environment setup:** partial success.
  - User-scoped ROCm Core SDK Python install succeeded in a venv.
  - Official WSL `librocdxg` package also installed.
- **GPU admission:** failed.
  - ROCm in WSL did **not** enumerate a supported GPU.
  - This is the gating failure.
- **Engine throughput:** not attempted.
  - vLLM and SGLang server runs are deferred until `rocminfo` admits `gfx1102`.

## Captured environment

- Captured by coordinator: **2026-08-28 UTC**.
- Windows host:
  - Microsoft Windows 11 Pro 10.0.26200 build 26200 x64.
  - Discrete GPU: AMD Radeon RX 7600, PCI `DEV_7480`, driver `32.0.31041.1004`.
  - Integrated GPU also present: AMD Radeon Graphics.
- WSL guest:
  - Ubuntu 24.04.4 LTS.
  - Kernel `6.18.33.2-microsoft-standard-WSL2`.
  - Device nodes before/after setup state relevant to admission:
    - `/dev/dxg=true`
    - `/dev/kfd=false`
    - `/dev/dri=false`

## Setup success: user-scoped ROCm SDK

A user-scoped ROCm Core SDK install succeeded **without host driver changes**.

- Python venv:
  - `~/.cache/fak/issue-9827/rocm-10-gfx1102/venv`
- Python package index:
  - `https://stable.repo.amd.com/rocm/whl-next/`
- Installed package request:
  - ``rocm[device-gfx1102,libraries]==10.0.0``
- Installed components witnessed:
  - `rocm-sdk-core 10.0.0`
  - `libraries 10.0.0`
  - `device-gfx1102 10.0.0`
- Validation witnesses:
  - `rocm-sdk targets` includes `gfx1102`.
  - `hipconfig` reports:
    - `HIP version: 7.15.26333`
    - `AMD clang version: 23.0.0git`

This proves packaging/userland installation worked. It does **not** prove the WSL runtime can admit the GPU.

## GPU admission failure: ROCm cannot see the GPU in this WSL envelope

The official ROCm WSL bridge package was installed:

- Package: `librocdxg` v`1.2.2`
- Asset source: GitHub release asset
- SHA256:
  - `28ded1254811192ebace1f76c0227580184af7b27ab2475fb9728295a702d541`

After installation, runtime probes still failed:

- `rocminfo`:
  - exit code: `1`
  - witnessed messages include:
    - `No WDDM adapters found`
    - `WSL environment detected`
    - `hsa_init Failed, possibly no supported GPU devices`
- `amd-smi`:
  - exit code: `255`
  - witnessed message: drivers not loaded

**Meaning:** GPU compute is not visible to ROCm in the current WSL + Windows driver envelope.

## Support boundary from cited upstream docs

### AMD WSL compatibility matrix

Source:
- `https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/compatibility/compatibilityrad/wsl/wsl_compatibility.html`
- Accessed: `2026-08-28`

Witnessed support statement relevant here:
- ROCm `7.2.1` / Adrenalin `26.1.1` WSL2 compatibility lists RX `9070`/`9060`/`7900`/`7800`/`7700` families.
- **RX 7600 is not listed.**

### vLLM AMD GPU support docs

Source:
- `https://docs.vllm.ai/en/stable/getting_started/installation/gpu/`
- Accessed: `2026-08-28`

Witnessed support statement relevant here:
- AMD requires Linux and ROCm `>= 6.3`.
- Listed GPU families include:
  - MI200 / MI300 / MI350
  - RX 7900 `gfx1100/1101`
  - RX 9000 `gfx1200/1201`
  - Ryzen AI MAX / AI 300
- **RX 7600 (`gfx1102`) is not listed.**

Pinned engine candidate for deferred rerun:
- PyPI latest probe: `0.28.0`
- Upstream tag: `v0.11.0`
- Commit: `b8b302cde434df8c9289a2b465406b47ebab1c2d`

### SGLang AMD docs

Source:
- `https://docs.sglang.ai/platforms/amd_gpu.html`
- Accessed: `2026-08-28`

Witnessed support statement relevant here:
- AMD guidance recommends Docker and images/builds based on `rocm.Dockerfile`.
- Docs discuss AMD generally, but do **not** establish RX 7600 WSL support.

Pinned engine candidate for deferred rerun:
- Latest/pinned probe: `0.5.18`
- Upstream tag: `v0.5.18`
- Commit: `ff4c6e641d9f9bb174d34ff651c01c114aea8e40`

## Engine outcome separation

### vLLM
- Typed outcome: `non_comparable_runtime_gpu_not_admitted`
- Reason:
  - Runtime prerequisite failed before engine startup.
  - No server process was started.
  - No throughput/latency/result row exists.

### SGLang
- Typed outcome: `non_comparable_runtime_gpu_not_admitted`
- Reason:
  - Runtime prerequisite failed before engine startup.
  - No server process was started.
  - No throughput/latency/result row exists.

## No substitution / no fallback

- No CPU fallback was used.
- No fak-native substitute was used.
- This is a **hardware/runtime admission** non-comparability, not an engine performance comparison.

## Next gate

Proceed to engine execution **only if** `rocminfo` successfully enumerates `gfx1102` in WSL.

Likely operator gate before any rerun:
- Installing the **WSL-specific Windows AMD driver** may require an operator-approved driver change and reboot.
- That step was **not performed** in this capture.

Important caveat even if admission later succeeds:
- RX 7600 remains outside the cited AMD official WSL matrix and outside vLLM's listed AMD GPU families.
- Any future successful run would still be **unsupported/experimental** unless upstream support statements change.

## Rerun commands

These commands are the honest rerun path. Engine commands remain deferred until runtime admission succeeds.

### 1) Environment probes

```bash
uname -a
lsb_release -a
python3 --version
test -e /dev/dxg && echo /dev/dxg=true || echo /dev/dxg=false
test -e /dev/kfd && echo /dev/kfd=true || echo /dev/kfd=false
test -e /dev/dri && echo /dev/dri=true || echo /dev/dri=false
which rocminfo || true
which rocm-smi || true
which hipcc || true
dpkg -l | grep -E 'rocm-core|rocm-hip-runtime|hip-runtime-amd' || true
```

### 2) User-scoped ROCm pip install

```bash
python3 -m venv ~/.cache/fak/issue-9827/rocm-10-gfx1102/venv
source ~/.cache/fak/issue-9827/rocm-10-gfx1102/venv/bin/activate
python -m pip install --upgrade pip
python -m pip install --extra-index-url https://stable.repo.amd.com/rocm/whl-next/ 'rocm[device-gfx1102,libraries]==10.0.0'
rocm-sdk targets
hipconfig
```

### 3) WSL ROCm bridge package install

**This mutates the system and requires root.**

Asset to use:
- `librocdxg` v`1.2.2`
- SHA256 `28ded1254811192ebace1f76c0227580184af7b27ab2475fb9728295a702d541`

```bash
sudo dpkg -i ./librocdxg_1.2.2_amd64.deb
sha256sum ./librocdxg_1.2.2_amd64.deb
```

### 4) Admission probes

```bash
rocminfo
echo $?
amd-smi
echo $?
```

Expected current failing witness from this capture:
- `rocminfo`: exit `1`, includes `No WDDM adapters found`, `WSL environment detected`, `hsa_init Failed, possibly no supported GPU devices`
- `amd-smi`: exit `255`, drivers not loaded

### 5) Deferred engine pins and intended launch/loadgen placeholders

Only run these **after** `rocminfo` enumerates `gfx1102`.

vLLM pin:
- tag `v0.11.0`
- commit `b8b302cde434df8c9289a2b465406b47ebab1c2d`

SGLang pin:
- tag `v0.5.18`
- commit `ff4c6e641d9f9bb174d34ff651c01c114aea8e40`

OpenAI-compatible loadgen geometry placeholder:

```bash
# placeholder only; do not claim executed until runtime admission succeeds
# server: host/port TBD
# model: TBD
# concurrency: <N>
# prompt_tokens: <P>
# output_tokens: <O>
# duration_or_requests: <D>
```
