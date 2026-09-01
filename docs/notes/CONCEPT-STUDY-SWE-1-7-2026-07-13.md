# Study-repo: SWE-1.7 (Cognition) → fak (2026-07-13)

A `study-repo` pass over **Cognition's SWE-1.7 blog**. SWE-1.7 is a **proprietary, RL-trained coding model** (Kimi K2.7 base, served on Cerebras) — **there is no acquirable repo**, so every borrow is **`inspire`-only**: the source anchor is a blog section plus any named open reference impl (NVIDIA Dynamo, DeepSeek-R1, Kimi K2). No bytes vendored; the license gate is N/A (nothing to copy). The primary blog page is hard-gated behind this environment's fetch policy, so the technique map was assembled from search summaries and cross-checked against the named open references — **re-witness the source text before acting**.

Method: five technique-axes, one parallel witness agent each (grounded at real fak `path:line`), then an **adversarial verify + dedup** pass per candidate (refute the gap; search existing issues/notes; correct overstatements). Filed under **epic #4597** with the `swe17-inspired` label.

## Decisive finding: the marquee technique is a *deliberate divergence*, not a gap

SWE-1.7's headline **self-compaction** — the model writes its own lossy summary of working state and resumes from that generated text — is **precisely the hazard fak's core context invariant exists to keep off the load-bearing path**. fak's compression is **extractive by construction** (`buildSummary` returns a verbatim source prefix, `internal/contextq/contextq.go:645-705`) with a load-bearing **`FaithfulnessProbe=1.0`** ("no model in the loop and thus no hallucination surface", `internal/contextq/viewindex.go:23-29`), and fak chose bounded-context **relays over compaction** (perpetual-sessions epic #1860). The #554 relinking triage (`docs/notes/RESEARCH-relinking-compression-boundary-triage-2026-06-23.md`) records the named invariant: *any model-authored abstractive summarizer on the agentic path must be re-screened at the result-admit boundary, never trusted as a faithful view.* Adopting self-compaction verbatim would regress that. **SWE-1.7 corroborates fak's invariant rather than exposing a gap** — a second external instance (after the #554 triage) of the exact failure mode fak designed around.

The one residual hunted — *does fak content-witness the **harness's own** generative auto-compaction summary (Claude Code) at the result-admit boundary?* — **dissolved under refutation**: (1) it is a **deliberately-fenced honest limit** ("fak controls the wire bytes, not the harness's transcript or summarizer; visibility ends at the wire" — `internal/compactcohere/doc.go`, `INDEX.md:364`, `HARNESS-CACHE-COHERENCE-AUDIT-2026-06-28.md` §2.4/§5, shipped rung-D detector #1134); and (2) a content-address/shingle fingerprint of a sealed span **cannot** match an *abstractive* summary whose content was never present whole in the source — the wrong tool for the real hazard. No leaf.

## Per-axis disposition

| Axis (SWE-1.7 technique) | Verdict | fak seam | Outcome |
|---|---|---|---|
| **Self-compaction** (model self-summarizes + resumes) | DIVERGENT | `contextq.go:645`, `viewindex.go:23`, `compactcohere/doc.go`, `INDEX.md:364` | No leaf — see decisive finding. |
| **Numerical drift** train↔inference (entropy preservation; DeepSeek-R1) | PRESENT | `internal/compute/fp8*`, quantization-specific quality budgets (#4540) | No leaf — fak already bounds quantized-forward error vs a reference with a budget. |
| **Fault-tolerant in-flight reroute** (Dynamo reroutes trajectories onto a warm peer replica) | PARTIAL — real, **duplicate** | `internal/agent/retry.go:95`, `cmd/fak/guard_child.go:1177`, `guard_rotation.go:110-128` | No new leaf — already tracked by **#3514**; cross-linked with the Dynamo framing. fak already has request-level retry + warm rotate-to-peer-seat reroute; the residual is a bounded warm-relaunch arm for a *transient-wire* child crash. |
| **Pre-dispatch task-hardening vs reward-hacking** (harden the grader before rollout; DeepSeek-R1 verifiable reward) | PARTIAL — real, **novel** | `internal/issuecontract/contract.go`, `scale.go` | **Filed #4598** — a pre-dispatch **forgeability grade** on a ticket's done-condition (advisory-first `StrictWitness`), the pre-flight complement to fak's post-hoc reward-hack guards (`witness.go` notests, `testintegrity.go` `TEST_CANNOT_FAIL`). |
| **Efficient reasoning / scope discipline** (reward discourages editing outside the blast radius) | PARTIAL — real, **novel** | `internal/dispatchtick/witness.go:24-47`, `cmd/fak/dispatch_tick_witness.go:167`, `internal/workerworktree/workerworktree.go:406` | **Filed #4599** — witness a worker's **committed footprint against its declared lane** (`CLAIM_OUT_OF_LANE`, advisory); the structural (non-forgeable-witness) restatement of a trained scope reward. |

## What the two filed leaves have in common

Both restate an SWE-1.7 **trained-reward** discipline as a **fak structural witness**: fak cannot train Claude, so instead of shaping a reward it (a) grades the *contract* before dispatch so a gameable "done" fails at triage (#4598), and (b) grades the *committed artifact* after so scope-creep is surfaced as a non-forgeable claim (#4599). Both are **advisory-first**, mirroring `StrictScale`/`CLAIM_TEST_*` fail-open discipline exactly. Adversarial verify sustained both, corrected two overstatements in #4598 (the worker prompt *does* inject generic hardening; `internal/assumecheck` already ships a forgeability-ordered `WitnessKind` to reuse) and one design error in #4599 (a resolving-SHA footprint check is *vacuous* on the worktree path — grade the honest per-worker source instead).

## Related passes

Same genre as the [colibri](CONCEPT-STUDY-COLIBRI-2026-07-11.md), [Dynamo](CONCEPT-STUDY-DYNAMO-2026-07-08.md), and [deepagents](CONCEPT-STUDY-DEEPAGENTS-2026-07-10.md) study-repo passes. Companion skills: `.claude/skills/field-borrow`, `.claude/skills/sota-check`.

_Witness is lexical + a 2026-07-13 snapshot; re-witness before acting._
