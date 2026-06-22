---
title: "SWE-bench Verified resolve: fak-gateway vs raw-SGLang on the DGX"
description: "A real coding-agent SWE-bench Verified run on the lab A100 DGX, driving Qwen3.6-27B through both the fak adjudication gateway and raw SGLang, graded with the official harness. Headline: the same model resolves the instance through raw SGLang but is blocked by fak's capability/trust floor — overall completion is decided by the floor, not the model."
---

# SWE-bench Verified — fak-gateway vs raw-SGLang, overall completion (DGX)

> **What this is.** The **resolve-rate / overall-completion** arm that
> [`SWEBENCH-RESULTS.md`](SWEBENCH-RESULTS.md) marks *comparable but DGX-gated*.
> A real `mini-swe-agent` coding agent solves the SWE-bench Verified instance
> `astropy__astropy-12907` on the lab A100 DGX, driving **`Qwen/Qwen3.6-27B`**
> (SGLang TP=8, bf16) through several request paths, graded with the **official**
> `swebench.harness.run_evaluation`. Every arm hits the *same* model weights; the
> only variable is what sits in the request path.
>
> **Date:** 2026-06-22 · **Hardware:** lab A100 DGX (8× A100-40GB) ·
> **Serving:** SGLang 0.5.10.post1 (TP=8, `--tool-call-parser qwen3_coder`) ·
> **Agent:** mini-swe-agent 2.2.8 · **Harness:** swebench 4.1.0 (Docker on the DGX).

## Headline

| arm | request path | gateway policy | agent turns | patch | **resolved** |
|---|---|---|---:|---:|:---:|
| **raw-sglang** | agent → SGLang `:30000` | — (unguarded) | 72 | 504 B | **✓ 1/1** |
| **fak-gateway** | agent → `fak serve :8080` → SGLang | DefaultPolicy | 251 (looped) | 0 B | **✗ 0/1** |
| **fak-gateway** | agent → `fak serve :8080` → SGLang | allow `bash` + `trusted_local` | 251 (looped) | 0 B | **✗ 0/1** |

**Overall completion is decided by fak's capability/trust floor, not by the model.**
The identical Qwen3.6-27B that **resolves** `astropy__astropy-12907` through raw
SGLang (a correct 504-byte patch, ✓ in the official harness) **never lands a patch**
through fak — because fak adjudicates every `bash` tool call and refuses it:

- **DefaultPolicy → `DEFAULT_DENY`.** No tool is permitted by default; the gateway
  denied all 224 `bash` calls. The agent sees "no tool calls", loops to its step
  limit (`LimitsExceeded`), submits an empty patch.
- **`allow: ["bash"]` + `sources: {"bash": "trusted_local"}` → `TRUST_VIOLATION`.**
  Allowing the tool lets the first 1–2 calls through (`verdict: ALLOW`), then fak's
  **trust floor escalates** every subsequent call (`disposition: ESCALATE`,
  `reason: TRUST_VIOLATION`) and denies it. Marking `bash` a trusted-local source
  does **not** disable this — the floor still requires escalation/authorization that
  an unattended benchmark cannot provide. Empty patch again.

This is fak working **exactly as designed** — *"the kernel that doesn't believe the
agents."* It is a capability + information-flow + residency floor, **not** a
transparent OpenAI passthrough. Raw SGLang executes whatever the model emits; fak
will not let an unattended agent freely run `bash` against a model it treats as a
remote trust boundary.

## So what is the honest answer to "does fak preserve completion vs SGLang"?

**No — not transparently, and that is the point.** The comparison isolates exactly
what fak adds and costs:

- **Raw SGLang**: maximal completion, zero guardrail. The agent's every command runs
  verbatim; an injected/poisoned tool result would too.
- **fak gateway**: the agent's actions are adjudicated against a policy and an
  information-flow/trust floor. Under the default and the simple allow policies that
  floor **stops** an unattended coding agent — which is the same mechanism that, in
  the adversarial direction, stops prompt-injection-driven exfiltration (see the
  gateway exfil-floor tests). Completing a coding task *through* fak therefore
  requires deliberately provisioning trust (an escalation approver / authorized
  data-flow), not just allow-listing the tool.

The resolve number is thus a **policy outcome**, not a model score: same model,
opposite completion, decided entirely by the floor.

## Method (exactly reproducible)

One self-contained driver, [`tools/dgx_swebench_compare.py`](../../tools/dgx_swebench_compare.py),
launched detached on the DGX via the Slack control bridge
([`cmd/dgxbridge`](../../cmd/dgxbridge)) and polled from a host-shared `/tmp` log:

1. **Serve** `Qwen/Qwen3.6-27B`, SGLang TP=8, `--mem-fraction-static 0.75`,
   **`--tool-call-parser qwen3_coder`**. Qwen3.6's chat template emits XML
   `<function=…><parameter=…>` tool calls; `qwen25`/Hermes mismatches and collapses
   generation to a single token (empty patches) — see [[qwen36-sglang-tool-call-parser]].
2. **Front** with `fak serve --provider openai --base-url …:30000/v1` on `:8080`,
   `--policy <manifest>` for the allow arms
   ([`tools/dgx_fak_gateway_repolicy.sh`](../../tools/dgx_fak_gateway_repolicy.sh)).
3. **Tool-call self-test** both endpoints before the agent runs (proves SGLang
   returns OpenAI `tool_calls`, and shows the gateway's per-call verdict).
4. Per arm: `mini-extra swebench --subset verified --split test
   --filter astropy__astropy-12907` against that arm's `model.model_kwargs.api_base`
   → `preds.json`.
5. **Grade** each with `python -m swebench.harness.run_evaluation
   --dataset_name princeton-nlp/SWE-bench_Verified` (resolve denominator = the 1
   instance submitted, *not* the report's 500-set `total_instances`).

Allow-arm policy: [`examples/swebench-coding-agent-policy.json`](../../examples/swebench-coding-agent-policy.json)
— `allow: ["bash"]`, `arg_rules` blocking `rm -rf`/`sudo`/`curl|sh`/`git push`,
`sources: {"bash": "trusted_local"}`.

## Honest fences

- This is **SGLang-serves + fak-fronts**, *not* fak's native CUDA engine (which
  cannot serve a multi-GPU 27B). Same fence as the throughput doc.
- A 1-instance selection is a **single sample** — the claim is the *mechanism*
  (the floor decides completion), not a 500-set score.
- The raw-SGLang resolve (1/1) shows the model + harness + `qwen3_coder` serving
  path are genuinely capable; the fak 0/1 is a policy/trust-floor outcome, not a
  model or plumbing failure.
- Earlier failed configs are themselves findings: with an unparseable tool format
  (`qwen25`), **raw SGLang fails *silent*** (empty response) while the **fak gateway
  fails *loud*** (`502 "upstream tool-call format not recognized; refusing to skip
  adjudication"`).
- All numbers trace to the run's committed artifacts (`compare.json`, `COMPARE.md`,
  per-arm `preds.json`, agent + gateway logs).
