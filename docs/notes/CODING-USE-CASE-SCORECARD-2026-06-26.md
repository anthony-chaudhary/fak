---
title: "Coding as a primary use case — scorecard + next-steps spine (2026-06-26)"
description: "Steps through every example under examples/ from a coding-agent lens, grades fak's support for the coding-agent use case across five dimensions (adopt / floor / cost / witness / in-kernel), and maps the ordered next-steps spine onto the live epics (#897, #745, #610, the examples cluster)."
---

# Coding as a primary use case — where fak stands, and what to do next

A readout for one question: *if a coding agent (Claude Code, OpenAI Codex, Cursor,
a SWE-bench harness) is the primary thing in front of fak, how well is it served
today, and what is the ordered next work?* Two passes — step through the examples,
then grade the use case — feed one spine of next steps mapped to live epics.

Honest framing up front: **adopt and floor are fak's strongest axes; proving the
value (a measured coding-agent number, the cost lever on real traffic) is the
frontier.** Composite grade ≈ **B− / C+**. Nothing below promotes an unwitnessed
claim to shipped.

---

## Part 1 — Examples status (coding lens)

29 example entries under `examples/`. Grouped by relevance to a code-writing agent.

### Coding-core, runnable (offline + deterministic unless noted)

| Example | Demonstrates | Runs? |
|---|---|---|
| `agentdojo-redteam/` | Adaptive injection battery, ASR detection-only vs full-stack IFC | runs (Go-only) |
| `wire-quarantine-demo/` | Result-side quarantine of untrusted tool output before it reaches context | runs |
| `wire-proof/` | Over-the-wire floor: every call passes the policy floor before execution | runs (offline mock planner) |
| `mcp/` | MCP stdio adjudication (deny `git_push`, allow `git_status`), zero deps | runs (seconds, deterministic) |
| `openai-agents-guardrail/` | Maps fak verdicts → OpenAI Agents SDK behaviors, no SDK/key needed | runs (self-contained) |
| `adjudication-demo/` | Live kernel verdict-before-decode against a real local model | runs — **hard-requires ollama**, no hostless fallback (**#632**) |

### Coding-core, static (policy floors — coherent, comprehensive, no runner)

| Policy | What a coding agent gets |
|---|---|
| `presets/coding-agent-safe.json` | The hardened, round-trip-gated floor: gitgate denials (force-push, amend, `commit --all`, `add -A`, `tag -f/-d`, `rebase -i`, `--no-verify/--no-gpg-sign`), destructive shell, out-of-tree writes, `lint_writes` on malformed Go/JSON, `SECRET_EXFIL`. Load-bearing for Claude-Code-type agents. |
| `swebench-coding-agent-policy.json` | SWE-bench floor: blocks `git push`, `rm -rf`, `sudo`, device writes, pipe-to-shell; allows read/write/bash. |
| `protected-push-floor-policy.json` | Argument-scoped `git_push` (#449): allows feature branches, denies protected refs + force by **arg value** — finer than a bash regex. |
| `dogfood-claude-policy.json` · `repo-guard-policy.json` · `dev-agent-policy.json` | The internal-dev floors (fail-closed posture, self-modify-glob protection of the kernel/policy spine). |

### Adjacent (generic, applies to a coding deployment) — all runnable & offline

`auth-hardening/` (bearer/api-key gate), `escalation-demo/` (deny → safe-sink + redact),
`mcp-client/` (reference six-tool MCP client, both transports), `observability/`
(`/metrics` + `/debug/vars`), `policy-hot-reload/` (in-place floor swap),
`trace-reset/` (per-trace IFC reset), `extdriver/` (out-of-tree adjudicator module),
`mem0-openmemory-policy.json`, `devops-dryrun-policy.json`.

### Routing (config-only, no runner) — relevant to model routing (#595)

`routing-presets/` (×5: cost-saver, best-of-quality, guard-writes, scout-then-route,
gardening), `routing-bench/` (×3 corpus + configs), `model-routing.example.json`,
`model-accounts.example.json`. These are static JSON; there is **no runnable routing
demo** that executes a per-aspect decision.

### Other-domain (not coding) — static policy floors / fixtures

`customer-support-readonly`, `research-agent`, `flight-booking`, `healthcare-phi`,
`sql-analyst`, `policy.example`, `shared-task-record*`, `trajectory`.

### The example-level finding

The coding **floors are solid and coherent**; the coding **demos are mostly
generic-capability proofs**. What is missing is a re-runnable example that wraps a
*coding agent behind `fak guard`* and shows a dangerous call denied in a live
session — today that behavior is proven by unit tests + a recorded live pilot
(under #747), not by an example a skeptic can run. That gap is exactly the open
examples cluster: **#318** (dogfood-claude adoption wrapper), **#344** (dev-agent
ship-gate demo), **#321** (live A/B walkthrough), **#632** (ollama-free
adjudication-demo).

---

## Part 2 — Use-case scorecard (five dimensions)

Grades are "how well does fak serve a **coding agent** on this axis **today**,"
not how good the underlying mechanism is in the abstract.

| Dimension | Grade | One-line |
|---|---|---|
| **Adopt** — harness integration in one repoint | **A−** | `fak guard -- claude` (one command) or a base-URL repoint fronts 38/44 surveyed harnesses; agent/model/prompts unchanged. Gaps: OpenAI/Codex wire lacks a recorded live-transit proof; no prebuilt binary (needs Go 1.26+ or curl script). |
| **Floor** — security for code-writing agents | **B+** | Deny-by-structure before the model sees the call: gitgate (push/commit/add/tag/rebase variants), self-modify globs on the kernel spine, destructive shell (`rm -rf`/`sudo`/`dd`/`curl\|sh`/fork-bomb), inline-eval routing, synth-tool ledger (#543), secret redaction. Proven in `adjudicator_test.go` / `devagent_test.go`. Gaps: no **re-runnable** coding-agent-blocked demo; arg predicates are regex-only (relative escapes caught, absolute paths need the DOS `repo_guard.py` backstop). |
| **Cost** — lever for long coding sessions | **C+** | Cache-preserving compaction sheds 92–95% of old turns while keeping the cache prefix **sha256-identical** (memcpy splice, fail-safe identity). Dogfooded end-to-end at 142k/236k inbound. **But:** the dogfood is a **mock upstream** — the cascade hypothesis (that Anthropic's cache actually reuses the byte-stable prefix after a middle drop) is unverified on real traffic (**#745**), and the lever is Anthropic-wire only. The realized cost win for one real coding session is **not yet measured**. |
| **Witness** — a measured coding-agent result | **D+** | **Zero** raw-vs-fak coding-agent solve number exists. SWE-bench Verified shows fak-native work-elimination (16–23× A/C) but `pass_rate_pct = 0` (resolve-rate is GPU-gated; ≈0 with the local 135M model). Terminal-Bench 2.1 is all contract-phase (**#897**); every artifact carries `result_claim_allowed=false`. The older #868–875 agentic-benchmark family is **closed/superseded**. |
| **In-kernel** — run the coding loop on fak's own forward | **B−** | Pure-Go forward is argmax-exact on CUDA / Vulkan / Metal; tokenizer is oracle-parity; `fak serve --gguf --engine inkernel` ships. **But:** no real model has solved a coding instance through this path, and a multi-turn Claude-Code loop on fak's own forward is unwitnessed (**#610**, #609, #463). |

**Reading the grades:** the coding adopter's value chain is *adopt → floor → cost →
witness*. fak is strong at the front of that chain (A−/B+) and weakest at the end
(D+). The whole point of the spine below is to walk the grade rightward.

---

## Part 3 — The spine of next steps

Ordered by leverage — what most moves a real coding-agent adopter from "interesting"
to "proven." Each rung names the live epic that owns it and the grade it lifts.

1. **Witness a coding-agent number — `#897` Terminal-Bench 2.1 rank-1 (P0).**
   The single highest-leverage item: the only path to a measured raw-vs-fak coding
   result. Children: **#898** (pin target + contract + gates), **#899**
   (Harbor/Codex CLI adapter), **#900** (smoke + full rehearsal with compare
   artifacts), **#901** (failure taxonomy / retry / command-state recovery),
   **#902** (submission packet + BENCHMARK-AUTHORITY row). *Lifts Witness D+ → a real grade.*

2. **Settle the cost lever on real traffic — `#745` (with `#747`, `#746`).**
   The compaction lever is one credentialed Anthropic session from settlement.
   Open DoD items: a `[SHIPPED]` tag in `CLAIMS.md`, a `BENCHMARK-AUTHORITY.md`
   row, and one **combined** long-session guard dogfood capturing a denied call +
   intact audit chain + body shed in a single artifact. *Lifts Cost C+ → B.*

3. **Make the floor visible — the examples cluster (`#318`, `#344`, `#321`, `#632`).**
   Turn the test-proven floor into a re-runnable coding-agent demo: a `fak guard --`
   wrapper that denies a dangerous call live (#318), the ship-gate demo (#344), a
   point-at-your-own-model A/B (#321), and an ollama-free `adjudication-demo` (#632).
   Cheapest credibility-per-token. *Lifts Floor B+ → A−.*

4. **Finish the adopt secondary paths.** A recorded OpenAI/Codex-wire
   gateway-transit proof (folds into #747/#746) and prebuilt binaries (a release /
   persona item). *Closes the Adopt A− gaps.*

5. **In-kernel coding loop — `#610` (with #609, #463).** Witness a ≥2-tool-call
   Claude-Code loop on fak's own forward; GPU-gated. *Lifts In-kernel B− → B.*

Supporting / longer-horizon, already tracked: **#595** (per-aspect + ensemble model
routing — route a tool call vs a reasoning step), **#748** (agent-OS process model
for long coding loops), **#805 / #809** (intent conduit + speculative-loop perf).

### The one gap with no owner

No single epic frames *"coding agent behind fak"* as the primary use case and ties
the four pillars (adopt · floor · cost · witness) into one adoption-quality track —
the threads live in #897, #745, #610, and the examples cluster independently. A
candidate umbrella: `epic(coding): make "coding agent behind fak" the proven primary
use case — adopt · floor · cost · witness`, owning the cross-cutting scorecard and
the examples-visibility gap, referencing the four existing epics as its pillars.
**Not filed** pending operator go-ahead (the backlog is dispatcher-managed at
`AT_CAP`); this note is its durable form until then.

---

*Method: an Explore-agent fan-out read all 29 examples + five use-case dimensions;
issue states verified against the live backlog on 2026-06-26 (the #868–875 family is
closed). Grades are the assessment's, not a benchmark.*
