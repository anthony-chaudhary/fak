# Concept study: Ponytail — portable behavioral instructions and their witnesses (2026-08-13)

**Observed at:** 2026-08-13 (America/Los_Angeles)  
**Source:** <https://github.com/DietrichGebert/ponytail>  
**Pinned revision:** [`2ed6c52c9d7e5e56942508591085fd45dea277d3`](https://github.com/DietrichGebert/ponytail/tree/2ed6c52c9d7e5e56942508591085fd45dea277d3) (commit event 2026-08-07T21:44:01Z)  
**Source state:** `main`, after release `v4.9.0`; 159 tracked files; open upstream work was treated as direction, not shipped proof.  
**License at revision:** MIT, root `LICENSE`; no NOTICE, vendored tree, generated-code provenance marker, or submodule was found. This study uses **INSPIRE-ONLY** routes and copies no expressive source.  
**Refresh trigger:** new Ponytail release; material change to hooks, rule-copy checker, benchmark harness, or subagent scoping; or implementation of #6671–#6673.

## Verdict

Ponytail is not an agent kernel and should not be imported wholesale. It is a compact behavioral instruction product optimized for users who want the same “avoid shortcuts, then review” discipline in many agent hosts with nearly no setup. The useful borrows are therefore not its prose rules themselves. They are three narrower verification mechanisms around that product: scorer controls that prove a behavioral oracle before API spend, canonical payload parity coupled to host-envelope conformance, and role-selective child instruction injection. Those gaps are independently shippable as #6671, #6672, and #6673.

## Worldview reconstructed

Ponytail targets an individual developer who changes agents frequently but wants one lightweight coding discipline to follow them. The evidence is its broad host packaging (`.claude-plugin`, `.codex-plugin`, `.cursor`, `.opencode`, `.qoder`, `.grok-plugin`, `pi-extension`, `ponytail-mcp`), its default activation and persisted mode in `hooks/ponytail-activate.js` and `hooks/ponytail-mode-tracker.js`, and its benchmark focus on ordinary coding tasks and classic lazy edge-case failures. It optimizes **behavioral consistency and low operator touch across heterogeneous hosts**, accepting duplicated packaging and host adapters to reach users where they already work.

fak instead optimizes a resident mediation boundary: reuse, routing, context residency, policy, and witnessed operations. That makes Ponytail's host portability a useful source of adapter tests, but its global prose discipline is not a replacement for fak's structural gate. The comparison is complementary, not “one is better.”

## Evidence coverage

The read was deep and subsystem-oriented rather than a README skim:

- **Map and product contract:** root README, package/plugin manifests, command and skill entry points, `docs/agent-portability.md`, and `docs/platform-native.md`.
- **Runtime:** `hooks/ponytail-runtime.js`, activation, config, instruction filtering, mode tracking, statusline scripts, and SubagentStart injection.
- **Host adapters:** Claude/Codex, Copilot, Qoder, Grok, OpenCode, Pi, MCP, native rule copies, and their manifests.
- **Tests:** hook behavior and Windows paths, command/package/uninstall checks, adapter tests, Pi and MCP tests, and correctness/behavior fixtures.
- **Evaluation:** behavior, correctness, robustness-audit, promptfoo configurations, agentic tasks/judge, result ledgers, and comparison arms.
- **History and rationale:** releases/tags/changelog-equivalent history; open and closed issues; open, closed, and merged PRs plus selected discussion on portability, scoping, mode parsing, and benchmark failures. Recent commits were read for Codex envelope placement, PowerShell compatibility, Grok adapter reuse, and agent-type scoping.
- **Provenance:** exact-revision root license, repository tree, submodule state, and license/provenance filename scan.

### Completeness critic

Nothing material remained unopened at the subsystem level. Spanish/Korean README translations, static assets, example solutions, and every mechanically repeated platform copy were not read line-by-line after their canonical source, checker, manifests, and representative tests established their role. They cannot change the three borrow decisions. There are no source directories hidden behind the absent `src/` tree; the load-bearing implementation is in hooks, skills, adapters, scripts, tests, and benchmarks.

## Candidate ledger

| Borrow | Source anchor | Axis | Their-worldview reason | fak witness on-axis | Route | Filed |
|---|---|---|---|---|---|---|
| Paired known-good and lazy-wrong controls that self-test each behavioral scorer before API spend | `benchmarks/robustness-audit.js:2-5,35-177@2ed6c52c9d7e5e56942508591085fd45dea277d3` | Scorer discriminative validity before non-deterministic execution | A tiny instruction product must prove failures are real behavior changes, not a permissive checker | **ABSENT:** `internal/benchcatalog/catalog.go:73-91` records runnable metadata/baselines, but queries and raw searches found no required paired oracle controls | INSPIRE-ONLY | [#6671](https://github.com/anthony-chaudhary/fak/issues/6671) |
| Derive/check canonical instructions while separately validating each host's output envelope | `scripts/check-rule-copies.js:15-43`; `tests/hooks.test.js:390-520@2ed6c52c9d7e5e56942508591085fd45dea277d3` | Semantic payload parity plus platform-schema conformance | Their value disappears if one of many supported hosts receives stale text or valid text in the wrong field | **PARTIAL:** `internal/toolplugin` validates descriptors and `internal/hooks` tests fak protocols, but no canonical fixture projects through a host adapter matrix and round-trips semantic parity | INSPIRE-ONLY | [#6672](https://github.com/anthony-chaudhary/fak/issues/6672) |
| Inject one canonical instruction fragment only into configured child-agent roles | `hooks/ponytail-subagent.js:30-87`; `hooks/claude-codex-hooks.json:24-32@2ed6c52c9d7e5e56942508591085fd45dea277d3` | Role-selective instruction residency | Specialist children should receive the discipline only where it is relevant, limiting context and behavioral distortion | **PARTIAL:** `internal/syspromptmmu` profiles by tool shape, but searches found no child-role/audience allowlist for one instruction fragment | INSPIRE-ONLY | [#6673](https://github.com/anthony-chaudhary/fak/issues/6673) |
| Persist an operator-selected mode and expose it in a statusline | `hooks/ponytail-mode-tracker.js:1-103`; `hooks/ponytail-activate.js:1-117@2ed6c52c9d7e5e56942508591085fd45dea277d3` | Low-touch visibility of active behavioral posture | A user should know which discipline is active without re-explaining it every turn | **PRESENT-on-axis:** fak has durable session controls and operator readback surfaces; duplicating a prose-mode tracker would add a second authority rather than improve this axis | — | — |
| Normalize short aliases and reject an unsafe default mode | `hooks/ponytail-runtime.js:48-97`; `tests/behavior.test.js:150-227@2ed6c52c9d7e5e56942508591085fd45dea277d3` | Deterministic configuration normalization | Near-zero setup requires forgiving input without silently selecting a weak posture | **PRESENT-on-axis:** fak's typed config/CLI parsers validate enums and defaults at the structural boundary; a Ponytail-specific mode vocabulary has no independent value | — | — |
| Regex-filter a full skill body into concise/default/audit variants | `hooks/ponytail-instructions.js:1-124`; `tests/behavior.test.js:229-337@2ed6c52c9d7e5e56942508591085fd45dea277d3` | Prompt footprint versus retained rule meaning | The same discipline must fit hosts and tasks with different context tolerance | **DIVERGENT:** Ponytail accepts text parsing because one Markdown skill is its authority; fak's managed-context contract favors typed segments and stable semantic identity, so regex-derived prompt meaning would weaken cache identity and auditability | — | — |
| Package one behavior layer across many native agent ecosystems | host manifests plus `docs/agent-portability.md:1-176@2ed6c52c9d7e5e56942508591085fd45dea277d3` | Distribution reach with minimal operator setup | Users choose their agent host first and expect the discipline to follow | **WORLDVIEW-FINDING:** relevant to adoption, but not one checkable kernel leaf. #6672 captures the first concrete interoperability mechanism; broader packaging remains note-only until a named host and outcome clear the ship-alone test | — | — |

Searches used varied phrases through `fak capabilities` and raw repository scans because capability indexing is lexical. GitHub issue searches then checked title/body variants for scorer controls, canonical adapter projection, hook-schema matrices, and role-scoped child injection. No open duplicate of the three filed leaves was found.

## Negative knowledge and direction

- A correct semantic payload can still fail when a host expects another envelope. Ponytail's Codex `additionalContext` placement fix and PowerShell hook fix make this a concrete adapter-contract lesson, not theoretical portability work.
- Parsing prose into modes is fragile: upstream fixes narrowed accidental `ponytail:` markers and stopped rule bullets beginning with mode words from being swallowed. fak should keep instruction selection typed rather than inherit that parser.
- Open/merged adapter work shows ecosystems converge enough to reuse canonical hooks but diverge enough to require host-specific conformance witnesses. Open work is directional evidence only.

## Problem and value frame

- **Centrality:** Enabling. These borrows improve the witnesses and delivery surfaces around fak's core performance/security checkpoint rather than introducing a new core.
- **P1 managed context:** #6673 scopes resident instructions; #6672 avoids divergent duplicate context.
- **P2 net-true efficiency:** #6671 prevents paid runs with broken scorers; #6673 requests a token delta but forbids a gain claim without a workload witness.
- **P3 bounded adaptation:** canonical inputs, typed scopes, negative controls, and schema validation bound all three mechanisms.
- **P4 integrated operations:** each leaf attaches to an existing Go seam and an offline witness, not a new standalone script.
- **For:** fak maintainers and operators evolving prompts, benchmarks, and host integrations.
- **Problem:** semantic behavior can drift before provider calls, across host envelopes, or across child roles without a local discriminating witness.
- **Today:** neighboring infrastructure exists, but these three axes are partial or absent.
- **Better because:** failures become deterministic and local before they consume model spend or reach an operator.
- **Witness:** the acceptance tests in #6671, #6672, and #6673.

## Companions

- Skill discipline: `study-repo`
- Per-capability witness discipline: `field-borrow`
- Parent context: #6552, #6019, #6042
- Filed leaves: #6671, #6672, #6673
