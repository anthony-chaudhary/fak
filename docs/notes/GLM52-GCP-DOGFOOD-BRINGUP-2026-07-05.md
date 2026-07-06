# GLM-5.2 GCP dogfood bring-up, 2026-07-05

This note records the worked path for standing up GLM-5.2 on GCP and driving it
from this local machine through `claude-glm-gcp` / `fak serve`. It is a scrubbed
operator trail: no secrets, no tokens, and no ephemeral VM IPs.

## Goal

Run a real GLM-5.2 endpoint on a GCP GPU VM, front it locally with fak's Claude
Code preset, and prove a headless dogfood turn from this workstation:

```powershell
claude-glm-gcp --probe "say pong"
```

## Live state during proof

- Instance: `fak-glm-dogfood`
- Zone: `us-central1-c`
- Shape: `a3-megagpu-8g`
- GPU: 8x H100 80GB (`nvidia-h100-mega-80gb`)
- Serve path under test: SGLang W4AFP8, fronted locally by `claude-glm-gcp`
  through the same OpenAI-compatible `/v1` tunnel.
- Cost guards: idle delete timer enabled, budget delete timer set for 240 minutes.
- Native fak/llama.cpp phase did become reachable, but was not fast enough for
  a full Claude Code dogfood turn. It remains useful as a smoke path.
- Service phase observed at 2026-07-05 19:01 UTC:

```text
glm52sglang.service active; W4AFP8 weights loaded; /v1/models healthy;
max_model_len=65536
```

## What was tried

1. Confirmed local GCP auth and project wiring:

```powershell
gcloud config list --format=json
gcloud auth list --format=json
```

2. Checked live resources and quota:

```powershell
gcloud compute instances list --format="table(name,zone,status,machineType.basename())"
python tools\gcp_gpu_probe.py --all-tiers
```

The probe showed the standard 8x H100 tier was quota-provisionable, but live
capacity was still unknown until create time.

3. Confirmed the GLM-5.2 GGUF source is public and not locally blocked by a
missing Hugging Face token:

```powershell
Invoke-WebRequest https://huggingface.co/api/models/unsloth/GLM-5.2-GGUF
```

Observed metadata: public, not gated.

4. Tried the canonical pure-fak-kernel H100 create:

```bash
GCP_TIER=a3-high-h100 \
VM_NAME=fak-glm-dogfood \
SERVE=fak \
MAX_RUNTIME_MINUTES=240 \
MAX_RUNTIME_ACTION=delete \
ON_IDLE=delete \
IDLE_MINUTES=45 \
GRACE_MINUTES=90 \
bash scripts/gcp-glm-serve.sh --apply
```

Results:

- `us-central1-a`: `ZONE_RESOURCE_POOL_EXHAUSTED_WITH_DETAILS`
- `us-central1-b`: `ZONE_RESOURCE_POOL_EXHAUSTED_WITH_DETAILS`
- `us-central1-c`: `ZONE_RESOURCE_POOL_EXHAUSTED_WITH_DETAILS`

5. Added first-class Flex-start knobs to `scripts/gcp-glm-serve.sh` and verified
the plan/test:

```powershell
wsl.exe bash -lc 'cd /mnt/c/work/fak && go test ./cmd/fak -run TestClaudeGLMGCPBringupPlanWiring'
```

6. Tried Flex-start for `a3-high-h100`:

```bash
PROVISIONING_MODEL=FLEX_START \
REQUEST_VALID_FOR_DURATION=2h \
GCP_TIER=a3-high-h100 \
VM_NAME=fak-glm-dogfood \
SERVE=fak \
bash scripts/gcp-glm-serve.sh --apply
```

Result: refused by quota, because Flex-start consumes preemptible quota and the
project only had 3 preemptible H100 GPUs in `us-central1`; GLM-5.2 needs 8.

7. Found and wired the H100 Mega fallback:

```powershell
gcloud compute machine-types list --filter="name~'a3-mega|a3-highgpu-8g'"
gcloud compute accelerator-types list --filter="name='nvidia-h100-mega-80gb'"
```

Added:

- `tools/gcp_accel.py`: `a3-mega-h100`
- `tools/gcp_gpu_probe.py`: `nvidia-h100-mega-80gb` -> `NVIDIA_H100_MEGA`

Verified:

```powershell
python tools\gcp_accel.py --emit-shell a3-mega-h100
python tools\gcp_gpu_probe.py --all-tiers --zone us-central1-a
python -m pytest tools\gcp_accel_test.py
```

8. Tried H100 Mega:

- `us-central1-a`: stocked out, but GCP returned `zonesAvailable: us-central1-c`
- `us-central1-c`: create succeeded

Apply command:

```bash
GCP_TIER=a3-mega-h100 \
GCP_ZONE=us-central1-c \
VM_NAME=fak-glm-dogfood \
SERVE=fak \
MAX_RUNTIME_MINUTES=240 \
MAX_RUNTIME_ACTION=delete \
ON_IDLE=delete \
IDLE_MINUTES=45 \
GRACE_MINUTES=90 \
bash scripts/gcp-glm-serve.sh --apply
```

9. Fixed the native fak service lifecycle after observing a false-ready state.
`tools/glm52_fak_native_serve.sh` originally started `fak serve` in the
background, ran a smoke request, wrote READY, and then exited. Under
`systemd-run --collect`, that exit tore down the background server. The script
now writes READY and then waits on the child server process:

```text
GLM52_FAK_NATIVE_SERVE_READY port=8000 model=glm-5.2
```

After the fix, `glm52serve` stayed active and kept `fak serve` in the service
cgroup.

10. Ran the local `claude-glm-gcp` path against the native fak endpoint through
the IAP tunnel. The endpoint answered minimal OpenAI-compatible smoke requests,
but full Claude Code prompts repeatedly hit 300-second cancellation windows:

```text
gateway: upstream model error mid-stream (messages): context canceled
```

The VM-side logs showed the corresponding OpenAI chat completion calls canceled
after roughly 299 seconds. Native decode was about 0.3 tok/s with
`--cpu-offload-experts`, which is not acceptable for the dogfood proof.

11. Stopped the native service to free the GPUs and started installing SGLang on
the same H100 Mega node. The local preflight classified the node as
`READY_PENDING_INSTALL` for `sglang` with W4AFP8:

```text
arch=Hopper (sm_90)
total_vram_gb=637.2
required_vram_gb=423.2
recommended_quant=w4afp8
```

12. Raised stale inherited timeout floors in the `claude-glm-gcp` launchers.
`FAK_PLANNER_TIMEOUT_S`, `FAK_HTTP_WRITE_TIMEOUT_S`, and `API_TIMEOUT_MS` now
get lifted to the backend floor when unset, malformed, or set too low. This did
not make the native backend viable, but it removes a misleading local timeout
failure mode before the SGLang retry.

13. Installed SGLang on the node and fixed a serve-wrapper portability bug:
`tools/glm52_sglang_vllm_serve.sh` used a bare `python` preflight call, but the
GCP image provides `python3`. The wrapper now uses
`PYTHON_BIN="${PYTHON_BIN:-python3}"`.

14. Retried SGLang W4AFP8. Preflight passed as `READY` with SGLang `0.5.14`,
Torch `2.11.0+cu130`, and CUDA available. Startup reached distributed
initialization and W4AFP8 weight loading:

```text
launching sglang for PhalaCloud/GLM-5.2-W4AFP8 (tp=8 ctx=32768 quant=w4afp8)
Load weight begin. avail mem=77.33 GB
```

15. SGLang completed both the main GLM-5.2 W4AFP8 shard load and the EAGLE /
NextN draft load, allocated fp8 KV cache, initialized FlashInfer allreduce, and
entered CUDA graph capture. The server did not bind port 8000 during this phase;
local tunnel probes reset until startup completes. The live logs showed repeated
DeepGEMM JIT warmup entries such as:

```text
Capture target verify CUDA graph begin...
Entering DeepGEMM JIT Pre-Compile session
Try DeepGEMM JIT Compiling for <GEMM_NT_F8F8BF16>
```

16. The first SGLang run reached `/v1/models`, but chat completion failed before
the dogfood probe because the image had `jinja2` 3.0.3 and SGLang's Hugging Face
chat template path requires `jinja2>=3.1.0`. The same run exposed the SGLang
wrapper lifecycle bug: after logging READY it exited, and `systemd-run --collect`
cleaned up the server process. The wrapper now fails closed on old `jinja2`,
requires the smoke completion to contain `GLM52_OK`, and waits on the server
process after READY.

17. The next `claude-glm-gcp` probe reached fak and SGLang, but SGLang rejected
the request because the inherited 32,768-token context was too small for Claude
Code's Anthropic turn:

```text
Requested token count exceeds the model's maximum context length of 32768 tokens.
You requested a total of 64657 tokens: 32657 tokens from the input messages and
32000 tokens for the completion.
```

Restarting the service with `CTX=65536` made `/v1/models` report:

```json
{"id":"glm-5.2","max_model_len":65536}
```

18. The GLM chat template can place useful answer text in reasoning-only fields
unless thinking is disabled. The successful dogfood run used:

```powershell
$env:FAK_PROVIDER_EXTRA_BODY_JSON='{"chat_template_kwargs":{"enable_thinking":false}}'
```

The `claude-glm-gcp` preset now defaults that extra body so future probes do not
need the manual env var.

19. Proved the full local path through Claude Code, fak, the IAP tunnel, and the
GCP SGLang service:

```powershell
$env:FAK_GLM_GCP_BASE_URL='http://127.0.0.1:8200/v1'
$env:FAK_DOGFOOD_TIMEOUT_S='900'
claude-glm-gcp --probe 'say pong'
```

Result:

```text
[dogfood] kernel healthy: {"engine":"inkernel","model":"glm-5.2","ok":true,"planner":"proxy"}
[dogfood] result: pong
[dogfood] subtype=success  is_error=False  turns=1  model=glm-5.2
[dogfood] live turn ok - Claude Code completed a turn against the local kernel-fronted model
```

Witness copied to:

```text
experiments/agent-live/dogfood-claude-glm52-gcp-20260705.json
```

The witness records `duration_ms=12619`, `duration_api_ms=11797`,
`ttft_ms=11816`, `input_tokens=32641`, `output_tokens=2`, and result `pong`.
The local gateway log recorded `POST /v1/messages` returning HTTP 200 in
11.8 seconds for `claude-cli/2.1.200`.

20. Deleted the expensive proof VM after capturing the witness:

```powershell
gcloud compute instances delete fak-glm-dogfood `
  --project <redacted-project> `
  --zone us-central1-c `
  --quiet
```

Follow-up `gcloud compute instances describe` returned `not found`, confirming
the spend source was removed. The cleanup confirmation is on issue #2955:
`https://github.com/anthony-chaudhary/fak/issues/2955#issuecomment-4887262984`.

## Automation gap

Filed as GitHub issue #2955:
`ops(gcp): automate GLM-5.2 capacity fallback and dogfood proof`.

The issue was updated with both the Jinja2/lifecycle trap observed mid-bring-up
and the final 65k context requirement plus dated live-proof witness:
`https://github.com/anthony-chaudhary/fak/issues/2955#issuecomment-4887253599`.

The operator should not have had to manually translate:

- `ZONE_RESOURCE_POOL_EXHAUSTED_WITH_DETAILS`
- `zonesAvailable: us-central1-c`
- Flex-start's separate preemptible quota failure
- the separate `NVIDIA_H100_MEGA` family

into a working VM. The provisioner should fold those into an automatic capacity
ladder and then continue through tunnel, health polling, and `claude-glm-gcp`
proof capture.
