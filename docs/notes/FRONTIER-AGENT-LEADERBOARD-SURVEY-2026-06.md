# Frontier Agent Leaderboard Survey — enter/skip map (#1068)

**Generated:** 2026-06-27
**Issue:** [#1068](https://github.com/anthony-chaudhary/fak/issues/1068) (parent epic [#1063](https://github.com/anthony-chaudhary/fak/issues/1063))
**Status:** COMPLETED — **pick: BFCL V4** (cost/latency cost-delta lane)
**Sourcing date:** all leaderboard facts retrieved 2026-06 (per #1068's Entry-mechanics survey); URLs are clickable and dated below.

## What this is (and is not)

A **research / decision** deliverable: a ranked enter/skip table over the five frontier
*non-coding* agent leaderboards plus HAL as a cost-axis anchor, naming the single board to
pursue. **No leaderboard submission happens here**; executing the pick's run-contract is a
separate follow-on ticket.

The hard constraint that shapes every cell — fak has **no native frontier model and no
resolve-% of its own** on any of these boards. Every headline rank is the *fronted model's*
capability. fak rides *inside* an agent via its OpenAI-compatible / LiteLLM front
(`OPENAI_BASE_URL=http://localhost:8080/v1`, the [#873](https://github.com/anthony-chaudhary/fak/issues/873)
Packet E seam) or it co-defines a sub-metric. So each number below is labelled by **who
controls it**:

- **WITNESSED** — fak-authored, fak controls: the gateway-attributable *delta on the same
  fronted run* (the ~3 % gateway tax at saturation; the cross-session / prefix-KV-reuse cost
  saving as a **marginal-over-TUNED** figure; the model-free **injection / refusal floor**).
- **OBSERVED** — the fronted model's: any accuracy / resolve-% / Pass^k / rank / cost-column
  number. Appearing in a board's accuracy column publishes the *model's* number, not fak's.

## fak-attributable numbers used below (all WITNESSED, sourced in-repo)

| Number | Value | Witness artifact / row |
|---|---|---|
| Gateway tax at saturation | **~3 %** (peak C=64 0.75×; C=128 0.97×) | `experiments/qwen36/dgx-r4-20260622/compare.json` ([#921](https://github.com/anthony-chaudhary/fak/issues/921) Rung-4, commit `a2559041`); `BENCHMARK-AUTHORITY.md` row "Qwen3.6-27B 8-GPU SGLang serving"; `docs/benchmarks/VLLM-HEADTOHEAD-RESULTS.md` §4 |
| KV-reuse cost saving (marginal-over-TUNED) | **~1.0–1.31× single-session; 4.1× on a 50×5 fleet** (`net_value_add_vs_tuned=4.125`) | `experiments/session/headline-qwen-50x5.json` (commit `2bbda6f`); `BENCHMARK-AUTHORITY.md` rows L35/L367/L376 |
| Injection / refusal floor (model-free) | **full-stack ASR 0.000 vs detection-only 0.763**, 38-case corpus (29/38 reach context detection-only, 0/38 full-stack) | `experiments/agent-live/agentdojo-fak-fullstack-20260625.json` (`asr_fullstack=0`, `asr_detection=0.763…`) |
| Drop-in decide cost floor | **~0.55 ns/op · 0 allocs** registry read; 362 ns full allow | `internal/abi/registry_scaling_test.go`; `BENCHMARK-AUTHORITY.md` row "Adjudication overhead… [#451]" |
| Bit-exact KV / DeletionCertificate | **max\|Δ\|=0, L3 all-miss read-back** | [#56](https://github.com/anthony-chaudhary/fak/issues/56) — an *infrastructure property*, **no benchmark home in this family** (report WITNESSED, never as a rank) |

The **forbidden** number on every cost row: the **vs-naive** reuse multiple (the 17.9–23.4×
class, or the 60.3× headline). It is *not* the cost delta — see Honesty fence §1.

---

## Ranked enter/skip table

Columns per #1068 AC-2: **(1)** does it score a non-accuracy axis fak can witness
(cost / latency / safe-completion / policy)? · **(2)** submit mechanism + dated source URL ·
**(3)** cost to run · **(4)** the exact fak-attributable number that would ride it · **(5)**
that number's `WITNESSED`/`OBSERVED` label · verdict.

| Rank | Board | (1) Witnessable non-accuracy axis? | (2) Submit mechanism + source (retrieved 2026-06) | (3) Cost to run | (4) fak-attributable number that rides it | (5) Label | Verdict |
|---|---|---|---|---|---|---|---|
| **1** | **BFCL V4** (Berkeley Gorilla) | **Y** — has **both a USD-cost and a latency-seconds column** (strongest cost-axis fit in the family) | Contact the Gorilla team via **Discord** (no automated form); self-run harness available (**EvalScope**). [gorilla.cs.berkeley.edu/leaderboard.html](https://gorilla.cs.berkeley.edu/leaderboard.html) (last updated 2026-04-12); snapshot [llm-stats.com/benchmarks/bfcl-v4](https://llm-stats.com/benchmarks/bfcl-v4) | Paid key / GPU for the fronted model | ~3 % gateway tax at saturation on the *same fronted run*, accuracy held constant; KV-reuse marginal-over-TUNED (1.0–1.31× single / 4.1× 50×5) | **WITNESSED** for the delta; the board's USD/latency column itself = **OBSERVED** (model's) | **ENTER (the pick)** |
| 2 | **AA Agentic Index** (Artificial Analysis) | **Y** — per-task input/reasoning/answer **cost breakdown** | AA runs models itself in its open-source **Stirrup** harness (E2B sandbox + 6 tools); self-reproduce = Stirrup + E2B + a fronted model. [artificialanalysis.ai/models/capabilities/agentic](https://artificialanalysis.ai/models/capabilities/agentic); τ³-Banking [artificialanalysis.ai/evaluations/tau3-banking](https://artificialanalysis.ai/evaluations/tau3-banking) | E2B sandbox + model key | Same ~3 % tax / KV-reuse marginal-over-TUNED on a self-reproduced Stirrup run, accuracy held constant | **WITNESSED** for the delta; index rank = **OBSERVED** | SKIP-for-now (self-reproduce only; AA controls the listed set — viable **secondary**) |
| 3 | **Galileo Agent Leaderboard v2** | **Y** — reports **cost-per-session** | Dataset + harness public ([`galileo-ai/agent-leaderboard-v2`](https://huggingface.co/spaces/galileo-ai/agent-leaderboard), GitHub `rungalileo/agent-leaderboard`), reproducible locally; Galileo curates the listed model set. [huggingface.co/spaces/galileo-ai/agent-leaderboard](https://huggingface.co/spaces/galileo-ai/agent-leaderboard) | API key for the model | cost-per-session delta on a locally-reproduced run, accuracy held constant | **WITNESSED** for the delta; listed rank = **OBSERVED** (Galileo-curated) | SKIP-for-now (local reproduce only, not a controlled board slot) |
| 4 | **GAIA2** (Meta ARE) | **N** for the ranked score — has informational **budget-scaling curves** but **no cost/latency in the ranked score** | Self-run `gaia2-run` on the validation set, `--hf_upload` traces, submit the form; model-agnostic via LiteLLM (fak's front fits). [huggingface.co/spaces/meta-agents-research-environments/leaderboard](https://huggingface.co/spaces/meta-agents-research-environments/leaderboard); [eval docs](https://facebookresearch.github.io/meta-agents-research-environments/user_guide/gaia2_evaluation.html) | Fronted frontier key or GPU | **None rideable in the ranked score** — no cost/latency axis; only validation-set accuracy exists | accuracy = **OBSERVED** (model's) | **SKIP** (no scored non-accuracy axis) |
| 5 | **Vals Index** | Y (cost-per-test + latency) **but no self-submit** | Vals runs **all evals in-house**; enterprise engagement to be listed. [vals.ai/home](https://www.vals.ai/home) | Enterprise engagement | **None postable** — no self-run path | any rank = **OBSERVED** | **SKIP** (no self-submit) |
| — | **HAL** (Princeton — cost-axis *anchor*, not a submit target) | Y — built around **accuracy-vs-dollars** (closest match to fak's cost-per-task framing) | **Not a self-submit form** — a research harness + released logs. [hal.cs.princeton.edu](https://hal.cs.princeton.edu); arXiv 2510.11977 | Harness + fronted model | Cite as the **cost-axis reference**; the Berkeley RDI / HAL 2026 "8 major agent benchmarks gameable to near-perfect *without solving tasks*" finding **motivates** fak's model-free floor (ASR 0.000) — it does **not** hand fak a rank | floor = **WITNESSED**; framing = reference | **ANCHOR** (reference only) |

---

## The pick: BFCL V4 — one-paragraph justification

**BFCL V4 is the only board in this family that carries *both* a USD-cost and a
latency-seconds column.** That is what makes a *held-accuracy-constant, fak-in-vs-fak-out*
delta directly postable against an **existing public column** — the ~3 % gateway tax at
saturation (WITNESSED, `dgx-r4-20260622/compare.json`) and the KV-reuse marginal-over-TUNED
saving (1.0–1.31× single / 4.1× on a 50×5 fleet, WITNESSED, `headline-qwen-50x5.json`) — with
**no invented metric**. The alternatives each fail the cost-axis test in a specific way: AA
and Galileo expose a cost column but only via *self-reproduce* (no controlled board slot);
GAIA2 has *no* cost/latency in its ranked score; Vals has the column but *no self-submit*. So
BFCL is the one board where fak can attach a cost-axis number to a real public column — and
even there, **fak rides the cost column, it never tops the accuracy rank** (that rank is the
fronted model's, OBSERVED). The entry mechanic (Discord contact; EvalScope self-run) keeps the
first move a *self-run* with both arms before any listing request.

**Next step (named, not executed):** run-contract stub
[`experiments/agent-live/bfcl-v4-raw-fak-contract-20260627.json`](../../experiments/agent-live/bfcl-v4-raw-fak-contract-20260627.json)
— `result_claim_allowed=false`, raw + fak arms, shared model / task-ids / budget. Executing it
is a separate follow-on ticket.

---

## Negative result — no first-class adjudication-gateway axis exists

The structural finding #1068 asked to verify, stated explicitly and recorded as a negative
result (per-board evidence that policy/safety is folded into the model's pass-rate, **not**
scored standalone):

| Board | What it scores | Is safe-completion / policy-compliance / provable-deletion a *first-class* axis? |
|---|---|---|
| BFCL V4 | function-call accuracy + USD cost + latency | **No** — irrelevance/error detection is part of *accuracy*, not a standalone policy axis |
| AA Agentic Index | composite (solve + cost); τ³-Banking included | **No** — policy adherence folded into the model's pass-rate |
| Galileo v2 | tool-selection quality + cost-per-session | **No** — adherence folded into the model's score |
| GAIA2 | 0/1 task accuracy + budget-scaling curves | **No** — no safe-completion axis |
| Vals Index | cost/utility tradeoffs | **No** — no standalone policy-compliance / provable-deletion axis |
| HAL (anchor) | accuracy-vs-dollars | **No** — and it explicitly demonstrates benches are *gameable to near-perfect without solving tasks* |

**Conclusion:** *no public board in this family scores a tool-call adjudication gateway's
safe-completion / policy-compliance / provable-deletion as a first-class axis.* Where policy
adherence exists (τ³-Bench, Galileo) it is folded into the model's pass-rate. Therefore fak's
structural floor (full-stack **ASR 0.000** over its **own fixed 38-case corpus**,
`agentdojo-fak-fullstack-20260625.json`, model-free, **WITNESSED**) has **no public-leaderboard
home** and must be reported as *fak's own floor with the corpus boundary stated* — never as a
leaderboard placement. That model-free boundary is also its **measurement-validity** edge:
the ASR is assigned by a deterministic adjudication witness (`succeeded := injectionReachedContext
&& sinkExecuted`, **no judge model**), not by a safety classifier or an LLM-as-judge, so it
cannot be moved by the calibration drift or the benign-framing / white-box-GCG attacks that
make most *judge-mediated* reported ASR unreliable — see the metric-side idea-scout triage
[#1097](RESEARCH-jailbreak-judge-asr-reliability-triage-2026-06-27.md). ([#56](https://github.com/anthony-chaudhary/fak/issues/56) is the
DeletionCertificate *mechanism*, not a deletion *benchmark*.)

---

## Honesty fence (the three lines not to cross)

1. **Cost-column conflation.** BFCL / AA / Galileo / Vals cost columns measure the *model's*
   token cost. fak's ~3 % tax and KV-reuse saving are a **separate** delta — never blended
   into the board's number, and never implying fak lowered the model's bill by the vs-naive
   multiple. Quote the **marginal-over-TUNED** figure (~1.0–1.31× single; 4.1× on a 50×5
   fleet); **never** the 17.9–23.4× (or 60.3×) vs-naive number. The delta is valid **only with
   resolve-% held constant** (same model, same harness, fak-in vs fak-out) — a resolve-% move
   in either direction invalidates the comparison.
2. **No throughput-loss-as-win.** fak loses raw single-stream and aggregate throughput by
   design (fronting raw SGLang on 8×A100 = 0.60–0.97×). None of that is dressed as a win on a
   cost row.
3. **No fabricated safety rank.** There is **no** public board that scores safe-completion /
   policy-compliance / provable-deletion as a first-class axis, so any "fak tops a safety
   leaderboard" claim would be fabricated. The ASR-0.000 floor is reportable **only** as fak's
   own floor over its own corpus, with the boundary stated.

Every row above states the fronted-model dependency, labels resolve-%/rank **OBSERVED**, and
reserves **WITNESSED** for the gateway-attributable delta on the *same* run. No row claims a
fak rank or implies "fak scored X on board Y."

## Related

[#1063](https://github.com/anthony-chaudhary/fak/issues/1063) (benchmark-entry portfolio epic) ·
[#1010](https://github.com/anthony-chaudhary/fak/issues/1010) (GLM-5.2 cache-value, the
no-paid-key cost/cache lane this complements) ·
[#873](https://github.com/anthony-chaudhary/fak/issues/873) Packet E (the OpenAI-compatible
external-run seam reused here) ·
[#56](https://github.com/anthony-chaudhary/fak/issues/56) (per-span DeletionCertificate) ·
[#968](https://github.com/anthony-chaudhary/fak/issues/968) (A100 weights-access block for any
open-weight front) · governance
[#416](https://github.com/anthony-chaudhary/fak/issues/416) /
[#9](https://github.com/anthony-chaudhary/fak/issues/9) /
[#72](https://github.com/anthony-chaudhary/fak/issues/72) (lineage + real-measurement
discipline applied above). Coding/tool-state agent benches already contracted:
[#871](https://github.com/anthony-chaudhary/fak/issues/871) /
[#872](https://github.com/anthony-chaudhary/fak/issues/872) /
[#873](https://github.com/anthony-chaudhary/fak/issues/873) /
[#874](https://github.com/anthony-chaudhary/fak/issues/874) /
[#875](https://github.com/anthony-chaudhary/fak/issues/875).
