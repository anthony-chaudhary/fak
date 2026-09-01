# Benchmark-Portfolio Execution Roadmap & Ground-Truth Status — 2026-07-10

> **Class:** execution/status note (dated). Not an authority row — this note claims **no new
> benchmark number**. Witnessed numbers live in [`BENCHMARK-AUTHORITY.md`](../../BENCHMARK-AUTHORITY.md);
> the ranked lane board + SKIP list live in epic **#1063**. This note ties tonight's
> ground-truth checks to the epic tree and names the **operator decision points** that gate
> every remaining public number.

## TL;DR — what changed tonight

1. **The GPU lab server is NOT down.** It is reachable (via `private bridge`, ~50% flaky readback) and is
   an **multi-GPU Ampere server with its accelerators idle**. The earlier
   `private bridge doctor -probe → live_session=false` was readback flakiness, not an unavailable box.
2. **The model-free structural floors are green on this tree**, re-witnessed tonight:
   AgentDojo full-stack **ASR 0/38 (gate PASS)**, causal-invalidation `max|Δ|=0`, provable-deletion
   certificate mint+verify+tamper-reject. These are the numbers fak **owns**.
3. **The real gate on a first public capability number is not hardware availability — it is a
   model/serving choice and two operator approvals.** GLM-5.2 is a poor fit for sm_80 A100 (below);
   a *servable coder model on the idle A100s* is the shortest honest path to a first graded pass@1.
4. **No public number was produced or committed tonight** — that is deliberate. Every capability
   lane holds `result_claim_allowed=false` until an official grader + raw-vs-fak compare artifact
   are checked in; those runs are outward-facing and operator-gated. This note de-risks them.

---

## 1. The vision (restated, so the roadmap is grounded in it)

**fak is an agent _kernel_, not a token engine.** It sits in front of the model as an admission /
adjudication / cache-coherence layer. On raw throughput it *structurally loses* to a bare serving
stack — the committed gateway-tax rows show **0.75× peak → ~3% at saturation** vs raw SGLang. So the
portfolio must **never point at a throughput leaderboard as a fak win.**

fak's honest axes are the ones it owns end-to-end, where the whole number is fak's:

- **Structural safety** — prompt-injection / tool-abuse defense (AgentDojo, ToolSandbox). Model-free.
- **Provable isolation & deletion** — KV/cache containment, causal invalidation-on-external-write.
- **Cross-agent reuse / cache-value** — RadixAttention, session value-add, served-inline dedup.
- **The raw-vs-fak _differential_** on capability boards — *same model in both arms*, measuring what
  fak adds/costs (safety, cache-value, tokens, TTS), **never** the fronted model's resolve-%.

The honesty fence that makes the portfolio credible: **a fronted model's SWE-bench / LiveCodeBench /
GAIA resolve-% is that model's number, never fak's.** fak's claim is always the delta between the
raw arm and the fak arm.

---

## 2. Ground truth verified tonight (the corrections)

| Claim under test | Prior belief | Verified tonight | Evidence |
|---|---|---|---|
| GPU lab availability | "down / no live session" | **UP, 8× A100-40GB, all idle** | live `private bridge run nvidia-smi` → multiple idle Ampere accelerators |
| AgentDojo full-stack floor | 0/38 PASS (committed 2026-06-25) | **Still 0/38, gate PASS** on this tree | `go run ./cmd/agentdojoredteam -json` (this box) |
| Causal invalidation witness | green | **green, `max\|Δ\|=0`** | `go run ./cmd/causalbench -selfcheck` → exit 0 |
| Provable-deletion witness | green | **green (mint+verify+tamper-reject)** | `go run ./cmd/deletioncert -selfcheck` → exit 0 |
| Served-inline dedup (#1350) | "zero baseline / unmeasured" | **already witnessed & committed** | authority row: 35.7% read-only-shaped names, 0% Claude-native |

> **Working-tree caveat (governance).** The tree carries a large in-progress M5 diff. The AgentDojo
> re-run reflects the *dirty tree* (corpus hash `9c67ecb6…` vs committed `ddc5b9ae…`; detection arm
> 30/38 vs committed 29/38). The load-bearing gate (full-stack 0/38 PASS) is unchanged, so this is a
> **regression-confirmation, not a new authority row.** No number was committed from a dirty tree.

---

## 3. Portfolio status — witnessed vs blocked (ranked)

### Tier 0 — WITNESSED & owned by fak (keep green; these are the release story)

| Lane | Number | Provenance |
|---|---|---|
| AgentDojo structural safety floor | full-stack **ASR 0/38 (0.000)** vs detection-only 0.763; benign 2/2; PASS | model-free, fak-authored 38-case corpus (not public 629) |
| AgentDojo `fak_gateway` non-model defense entry (#1064) | local intercept **ASR 0/7**; module 26-check test PASS | utility arms `NEEDS_KEY`; public run operator-gated |
| Causal invalidation-on-external-write | PASS, `max\|Δ\|=0` (targeted evict, re-admit refused) | CPU-only structural witness |
| Provable deletion certificate | mint+verify+tamper-reject PASS | CPU-only structural witness |
| RadixAttention live ladder | **4.58× → 6.95×** live / 7.50× token / 86.7% hit | committed JSON, model-independent determinism |
| Gateway tax (Qwen3.6-27B 8-GPU SGLang) | **0.75× peak → ~3% at saturation** | real 8-GPU compare artifact |
| Pure-kernel decide latency | **362 ns** allow · 0.55 ns/op read floor | committed bench artifacts |
| Served-inline proxy-dedup (#1350) | 35.7% read-only-shaped / **0% Claude-native** | committed report + sheet |

### Tier 1 — BLOCKED on a run, no fak code residual (shortest paths to a public number)

| Lane | Epic / issue | Code state | What's missing |
|---|---|---|---|
| **LiveCodeBench raw arm** | epic **#2085 CLOSED (29/29)**, exec **#3060** | harness + official `lcb_runner` interop shipped | **serving standup only** (#3059) + weights (#968). "Needs only serving + an `lcb_runner` model registration pointing at the gateway." |
| Terminal-Bench 2.1 | epic **#897 OPEN**, children **5/5 closed** | contract/adapter/taxonomy/packet shipped | `OPENAI_API_KEY` + live Docker/Harbor + Codex `/v1/responses` wire (#925) + ≥86% go/no-go bar (model pin gpt-5.6) |
| AgentDojo public 629+97 | #1064 | module + local witness shipped | **~US$39 paid fronted-model run** + upstream PR into `ethz-spylab/agentdojo` fork |

### Tier 2 — BLOCKED on GPU generation (harder; GLM-5.2-specific)

| Lane | Epic / issue | Residual |
|---|---|---|
| GLM-5.2 fak-kernel cache-value | **#1010 OPEN (3/4)**, open child **#1012** | GPU-server replay capturing `kv_prefix.reused_tokens` on a solved ticket. Observation seam shipped (`52dfea0d`). |
| Memory benchmark matrix | **#2236 / #2244 OPEN** | every fak cell unfilled; matrix not run (rung R1, mechanism-in-code) |
| FrontierSWE TTS | #1721 | GATED — no number until official grader + score-parity |

---

## 4. Shortest honest path to a FIRST public capability number

**LiveCodeBench raw arm (#3060, Milestone 1).** The whole 29-child harness (#2085) is shipped and
closed; #3060 states the raw arm has **no fak code residual** — it "needs only serving + an
`lcb_runner` model registration pointing at the gateway."

**Key roadmap insight — decouple the raw arm from GLM-5.2/sm_80.** #3059 targets *GLM-5.2 on the GPU server*,
but GLM-5.2 cannot be fast-served on sm_80 A100 (§5). The raw arm does **not require GLM-5.2** — it
requires *a servable model behind the gateway*, and the box is **already provisioned to serve one**:
per the standing lab inventory (verified 2026-07), **vLLM 0.19.1** is installed
(`/root/mandar/aiwasp_venv`, V1-only; `vllm serve <model>` positional) and resident weights include
**Qwen2.5-0.5B / 1.5B-Instruct** (harness smoke), **gpt-oss-120b** (a ~120B MoE that fits across
8× 40 GB = 320 GB — the resident *credible-pass@1* candidate), and DeepSeek-R1/V3 (large). So the raw
arm needs **no model download and no vLLM install** — `vllm serve <resident>` → fak gateway front →
`lcb_runner` registration → run. This yields a **real graded pass@1 tonight-class run**, and:

- proves the #2085 harness end-to-end on real hardware (it has never produced a number);
- establishes the **raw baseline** that the fak arm (M2) then compares against — and fak's actual
  axis is the *raw-vs-fak differential* (safety/cache/cost), which is **model-agnostic**. The absolute
  pass@1 is the model's (per the fence); the differential is fak's.

So the fast path to a first public number is **a servable coder model on the idle A100s**, not
GLM-5.2. This is tracked as a new recommendation below (§6). GLM-5.2 stays on the GCP H100 fast lane
for the cache-value lane (#1010/#1012), which is a *different* claim.

---

## 5. Performance gaps → RSI goals

| Gap | Issue | Why it blocks a number | RSI goal |
|---|---|---|---|
| **Host expert-GEMM wall** — GLM-5.2 under `--cpu-offload-experts` decodes **0.03–0.17 tok/s** on A100 | **#996 / #971** | too slow to *generate* a benchmark pass → forces #1010 to pivot to cache-value-on-solved-ticket | close the sm_80 expert-GEMM path (or land a W4A8/quant MoE forward that fits 40 GB/GPU) so fak's own kernel generates at usable tok/s |
| **Served-inline name gate** serves 0% on Claude-native `Read`/`Grep`/`Glob` | **#1350** | the dominant real-agent read path gets no dedup | teach the read-only name gate to recognize native read tools (or scope default-on to MCP/snake_case) → unlocks the dominant-path value |
| **Codex full-turn probe window** — the ~20 KB Codex system+developer prompt doesn't decode inside the readiness probe window on the current bridge/model latency | **#2974** (siblings #2952/#2953) | blocks the unshaped Codex coding-turn witness through `--remote-serve @lab` | faster serving lane / lower-latency tunnel / prompt compaction |
| **Public-suite conversion** — every floor is witnessed on a fak-authored corpus, not the public suite | #1064 (AgentDojo), #3060 (LCB), #897 (TB) | internal witness ≠ public leaderboard credibility | convert each internal floor to a public-suite run behind the same honesty fence |

---

## 6. Operator decision points (what needs a human — nothing to "restart")

The box is up; there is nothing to restart. The remaining gates are **choices and approvals**, not
infrastructure:

1. **[Tier 1, cheapest]** Pick the servable model for the **LiveCodeBench raw arm** on the idle
   8× A100-40GB. Box is already provisioned (vLLM 0.19.1 resident; no install/download): use
   **gpt-oss-120b** (resident, fits 320 GB) for a credible pass@1, or a resident Qwen2.5-1.5B for an
   immediate harness smoke, or download a 32B-class coder if a coder-specific number is wanted. This
   produces the first real graded pass@1 and the raw baseline for the fak differential.
   *No GLM-5.2, no GCP, no new spend.*
2. **[Tier 1]** Approve the **AgentDojo public 629+97 run** (~US$39 paid fronted model) + the upstream
   PR into the `ethz-spylab/agentdojo` fork. This is the single highest-credibility *external* proof
   of the safety floor.
3. **[Tier 2]** For the **GLM-5.2 cache-value lane (#1012)** decide the fast serving lane:
   GCP H100 (`scripts/gcp-glm-serve.sh --apply`, needs `GCP_PROJECT` + `HF_TOKEN`, 8-GPU shape
   > $500/day burst) — GLM-5.2 fast-serve is already witnessed once there
   (`experiments/agent-live/dogfood-claude-glm52-gcp-20260705.json`). A100 cannot fast-serve GLM-5.2.
4. **[Tier 1, Terminal-Bench]** Provision `OPENAI_API_KEY` + Docker/Harbor + the Codex `/v1/responses`
   wire (#925) only if TB rank-1 is a priority; it carries more residual than LCB.

---

## 7. What I did NOT do, and why (honesty ledger)

- **Did not stand up serving or produce a public benchmark number.** Those runs are outward-facing and
  operator-gated (`result_claim_allowed=false` until official grader + compare artifact). Autonomously
  emitting a "public number" would violate the repo's own governance.
- **Did not commit a new authority row.** The tree is dirty (M5 in progress); the only new data I have
  (AgentDojo re-run) is a dirty-tree regression-confirmation, not a committable number.
- **Did not create duplicate tickets.** The gaps in §5 are already tracked (#996/#971, #1350, #2974);
  the one genuinely-new recommendation (servable-coder LCB raw arm on A100) is §4/§6.1 and should be
  filed as a scoped issue if the operator agrees with the decoupling.

---

### Reproduce the tonight checks

```bash
go run ./cmd/agentdojoredteam -json          # full-stack ASR 0/38, gate PASS
go run ./cmd/causalbench -selfcheck          # causal invalidation, exit 0
go run ./cmd/deletioncert -selfcheck         # provable deletion, exit 0
# GPU-server reachability (flaky readback — retry):
#   private bridge -timeout 120s run 'nvidia-smi -L'   → 8× A100-SXM4-40GB
```
