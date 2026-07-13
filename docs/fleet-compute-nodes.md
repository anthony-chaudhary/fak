# The sanctioned compute nodes — the local machine is the control point, not the compute boundary

This is the in-repo SSOT for **where hardware-gated work runs**. The workstation an agent
runs on (a Windows laptop, in the reference fleet) is the **control point**: it plans,
dispatches, and ledgers. It is **not** the compute boundary. When a task needs something the
local box lacks — a CUDA/GPU device, a specific accelerator arch, DC-network reach, or heavy
CPU — the sanctioned move is to **dispatch the work to a fleet node that has it**, then report
the real result.

**The anti-pattern this page kills:** ending a turn with *"not yet — no GPU on this host"* (or
any *"blocked for lack of local hardware"*) as a terminal answer. That treats the laptop as the
boundary. It never is. There is always a route: a fleet node, or — if the credential/bridge
session for that node is missing — a ready-to-run command sequence handed to the operator. Both
count as "used the lab." Only `no local hardware` does not.

> The complement, so the two never blur: the **kernel's own** 60-second proof needs no GPU, no
> key, no model (see [`AGENTS.md`](../AGENTS.md#the-60-second-proof-no-key-no-model-no-gpu--verified)).
> This page is about the *other* case — a task that **genuinely** needs a device (a live CUDA
> serve, a device-GEMM witness). That task is dispatched, never abandoned.

## The nodes (route "no local X" here)

| Need | Node | How to reach it |
| --- | --- | --- |
| Live CUDA serve / GPU burst | **GCP** `fak-realmodel` (g2-standard-8, **NVIDIA L4 sm_89**); spot A100/H100 for bigger runs | `python tools/gcp_gpu_probe.py --all-tiers` (read-only quota probe) · `python tools/gcp_bench.py --dry-run` (prints every gcloud command offline) · `GCP_PROJECT=<id> ./scripts/gcp-qwen-serve.sh --apply` (stand up an in-kernel CUDA serve) |
| Device-GEMM / kernel witness (**8-GPU datacenter server sm_80**) | the on-prem **GPU server** — no inbound SSH; reached only via the private Slack control bridge (its hostname lives in local fleet memory, not the repo) | `go build -o dgxbridge ./cmd/dgxbridge && ./dgxbridge doctor` → `run` / `bg` / `wait` / `pull`; needs a live operator Slack session |
| DC network + tokens (Confluence, CPU bench) | **CPU server** (AVX2 EPYC-7742, CPU-only) | operator `dgxbridge` session; the laptop can't reach the DC, `da33` can |
| Nightly device-witness ledger | **nightrun** pipeline | dispatch device-witness tasks → `docs/nightrun/collected.jsonl` (produced on GPU hardware, not the laptop) |

**Capacity reality (as of the 2026-07 lab-resource sweep):** the reference project's GPU quota
is **H100-in-us-central1 only** (A100/L4/T4 report NO_QUOTA); on-demand `a3-highgpu-1g` can
STOCKOUT across all zones (surfaced as a bare "Internal error"), while `--spot` draws from a
separate, larger, ~3× cheaper pool and has succeeded where on-demand failed. So for a GPU
witness: try `--spot` early, and note that failed `create`s don't bill. `gcp_bench` pins
`--max-run-duration=7200s ...action=DELETE` so a box can't run away.

**Credential-missing is still not a wall.** If the creds/bridge session for the right node are
absent (no GCP creds on the control box by default; `dgxbridge` needs a live operator Slack
session), produce the **exact ready-to-run command sequence** and hand it to the operator. That
is "used the lab," never "blocked by local hardware."

**A device witness runs ON the GPU node.** `FAK_CUDA_ARCH=sm_89 bash tools/cuda_acceptance.sh`
(or `make cuda-accept`) honestly SKIPs (exit 3) on a non-GPU host — an honest skip, never a
false green. The green comes from the node that can actually witness it.

> **"Node" = fleet compute node, not Node.js.** There is no Node.js project in-repo (no
> `package.json`); the non-Go runtime is Python (`tools/`), frozen by `internal/pythongate`.

## How this is enforced

The doctrine is not honor-system — a guard hook catches the regression and redirects to this page.

- **`fak hwgate-lint`** — scans an agent's final output for the "no local GPU/hardware" stop
  pattern and types each hit to a class (`NO_LOCAL_GPU`, `NO_LOCAL_RUNTIME`, `LOCAL_BOUNDARY`),
  each carrying the fixed redirect to a sanctioned node. A turn that already names a sanctioned
  route is suppressed (the fleet was used). Run it directly, or drive the fixtures with
  `fak hwgate-lint --selfcheck`. It is the hardware-gate dual of `fak headless-lint`.
- **The Stop-hook gate.** When `fak guard` installs the Claude Code Stop hook, a headless child
  whose final turn declares a local-hardware blocker as terminal is caught by the
  `--hardware-gate` rung. On `enforce` the stop is **blocked** (exit 2) and the sanctioned-node
  redirect is fed back so the agent dispatches instead of stopping; `warn` (the shipped default)
  and `shadow` observe only, so the pattern can soak before promotion. The gate runs **before**
  the operator-directed gate: a "no GPU here — want me to try elsewhere?" turn is the more
  specific misroute and takes the fixed hardware remedy, not the generic "act on your own
  question" nudge.
- **The tally.** `fak guard-stops` rolls up how many stops ended by declaring a local-hardware
  blocker as terminal (vs redirected), so the soak → promote decision reads straight off the
  ledger.

## Related

- [`cmd/dgxbridge`](../cmd/dgxbridge) — the Slack control bridge to the on-prem GPU server.
- [`docs/nightrun/`](nightrun/) — the nightly device-witness ledger the GPU nodes feed.
- [`AGENTS.md`](../AGENTS.md) — the agent doctrine that points here.
