# SWE-bench Verified resolve compare — fak-gateway vs raw-SGLang (GPU server, 2026-06-22)

Artifacts for [`docs/benchmarks/SWEBENCH-VERIFIED-DGX-RESOLVE-COMPARE.md`](../../../docs/benchmarks/SWEBENCH-VERIFIED-DGX-RESOLVE-COMPARE.md).
Instance: `astropy__astropy-12907` (SWE-bench Verified). Model: `Qwen/Qwen3.6-27B`,
SGLang TP=8 bf16 (`--tool-call-parser qwen3_coder`), on the lab GPU server (8-GPU datacenter server).
Agent: mini-swe-agent 2.2.8. Grader: `swebench.harness.run_evaluation` 4.1.0 (Docker).

Denominator note: the harness `report.json` `total_instances` is the whole 500-set,
so the first run's `compare.json` shows `total:500` — the honest denominator is the
**1 instance submitted** (`instances:1`, `resolved_ids` are the truth). The fixed
driver records `submitted` + `total:1` + `dataset_total:500` (see the fak-allow file).

| arm | gateway policy | turns | wall (agent) | patch | completed | **resolved (of 1)** | gateway verdicts |
|---|---|---:|---:|---:|---:|:---:|---|
| raw-sglang | — (unguarded) | 72 | 223 s | 504 B | 1/1 | **1/1 ✓** | n/a |
| fak-gateway | DefaultPolicy | 251 (loop) | 410 s | 0 B | 0/1 | **0/1 ✗** | 224× `DEFAULT_DENY` |
| fak-gateway | `allow:[bash]` + `sources:{bash:trusted_local}` | 251 (loop) | 342 s | 0 B | 0/1 | **0/1 ✗** | 246× `TRUST_VIOLATION` (ESCALATE) |

**Finding.** Same model, opposite completion — decided by fak's capability/trust
floor, not the model. Raw SGLang lets the agent run free and it lands the *correct*
fix (`raw-sglang.preds.json`: `cright[...] = right` in `astropy/modeling/separable.py`,
the canonical astropy-12907 solution → harness ✓). Through fak the agent never lands
a patch: DefaultPolicy default-denies every `bash` call; even after allow-listing
`bash` and marking it `trusted_local`, fak's trust floor escalates each call to
`TRUST_VIOLATION` and denies it (the same floor that, adversarially, blocks
injection-driven exfiltration). Completing a coding task *through* fak requires
deliberately provisioning trust (an escalation approver / authorized data-flow), not
just allow-listing the tool.

## Files
- `raw-vs-fakdefault.compare.json` / `.COMPARE.md` — run #1 (raw-sglang + fak DefaultPolicy).
- `fak-allow-trustedlocal.compare.json` — run #2 (fak gateway, allow `bash` + trusted_local).
- `raw-sglang.preds.json` — the 504-byte patch that resolved the instance (correct fix).

## Reproduce
`tools/dgx_swebench_compare.py` (driver), `tools/dgx_swebench_launch.sh` (detached
launcher), `tools/dgx_fak_gateway_repolicy.sh` (re-policy the gateway),
`examples/swebench-coding-agent-policy.json` (allow-bash policy). Drive via
`cmd/dgxbridge`. See the doc for the exact command sequence.
