---
title: "deepagents borrow study: 1 witnessed PARTIAL filed under #4200 from a scout pass over langchain-ai/deepagents"
description: >
  A /study-repo scout-loop pass over langchain-ai/deepagents @ c87e8fe (LangChain's
  deep-agent harness — middleware-composed agent loop, pluggable sandbox-provider
  registry merged by precedence, HITL approval-mode tool gating, and a data-driven
  dependency preflight that emits actionable install hints before spawning a backend
  subprocess). Candidates dogfooded against fak's self-index (fak_feature_query /
  fak_index_* + raw grep of the fak seam). Decisive signal: fak is MATURE on this axis
  — HITL tool gating, config-declared provider registries, and preflight-with-install-
  hint are all PRESENT; class_path runtime load and managed binary auto-download are
  off-axis for a single static Go binary. One witnessed PARTIAL filed: fak hand-rolls
  the host-readiness preflight per benchmark instead of declaring the install hint as
  data behind a shared probe. All borrows inspire (MIT source, Python->Go clean-room);
  no bytes vendored.
metadata:
  type: project
---

# deepagents borrow study (2026-07-10)

## What was studied

- **Repo:** [langchain-ai/deepagents](https://github.com/langchain-ai/deepagents)
- **Pinned SHA:** `c87e8fe558f0b56dba3ce92a994cc6aac0c93ccf` (`c87e8fe`, HEAD dated 2026-07-10 — a fresh-lane lead). All `path:line@c87e8fe` citations below resolve against this pin.
- **What it is:** LangChain's deep-agent harness. Subsystems mined: the sandbox trio `libs/code/deepagents_code/integrations/` — `sandbox_provider.py` (the `SandboxProvider` ABC + `SandboxProviderMetadata` + the `SandboxInstallHint` dataclass), `sandbox_registry.py` (built-in / entry-point / config providers merged by precedence), `sandbox_factory.py` (`create_sandbox`, the `verify_sandbox_deps` preflight, ensure-snapshot-by-name idempotency, cleanup-only-if-created contextmanager) — plus `managed_tools.py` (managed ripgrep binary), and the middleware-composed agent loop `agent.py` (HITL `interrupt_on` / approval modes).

## Method

Read the load-bearing modules + their unit tests at the pinned SHA, extracted candidate borrows each grounded at a real `path:line@c87e8fe`, then dogfooded fak's own self-index (`fak_feature_query`, `fak_index_verbs|docs|leaves`) plus a raw `Grep` of the concrete fak seam before grading PRESENT / PARTIAL / ABSENT. Note: `fak_feature_query` blobs overflowed the tool budget (300–830 KB); fell back to `limit=`-capped queries + the lighter `fak_index_*` verbs + raw grep, same as the sibling claude-code pass.

## License gate

`langchain-ai/deepagents` ships an OSI **MIT** `LICENSE` (Copyright LangChain, Inc.; both `libs/code` and `libs/deepagents` declare `license = MIT`). The one filed borrow is a small idiom **re-implemented in Go**, not a byte copy ⇒ **`inspire`**, no bytes vendored. MIT would permit either.

## Decisive finding

fak is **mature on deepagents' governance axis** — most candidates witnessed PRESENT and were correctly dropped. The one survivor is a *consolidation* gap (fak has the capability but hand-rolls it), not a missing capability.

## Witness table

| # | Candidate (deepagents seam) | fak witness | Verdict | Disposition |
|---|---|---|---|---|
| C1 | Data-driven dependency preflight: install hint carried as data (`SandboxInstallHint`, `sandbox_provider.py:14`) + one generic probe (`verify_sandbox_deps`, `sandbox_factory.py:1077`) that reads module+hint from metadata and errors *before* spawning the subprocess | fak has host-readiness preflight with install hints **twice** — `internal/terminalbench/preflight.go:233` + `internal/livecodebench/preflight.go:138` — but each hand-rolls a per-reason switch + per-tool `*Detail` strings; no shared "dep + install hint" data structure | **PARTIAL** | **Filed #4203** |
| C2 | HITL tool-call gating (`agent.py` `interrupt_on` / approval modes) | fak's `preflight` adjudicates one tool call against a policy (ALLOW/DENY by structure, no model in the loop) — tool-call gating is fak's core value prop | PRESENT | Dropped |
| C3 | Config-declared provider registry merged by precedence (config > entry-point > built-in) (`sandbox_registry.py:38-79`) | fak already resolves models / MCP servers / DOS drivers by name from config | PRESENT-adjacent | Dropped |
| C4 | `class_path` config escape hatch: declare a provider class via `module:Class` with an explicit "executes arbitrary Python, same trust model" warning (`sandbox_config.py`) | Runtime class loading does not map to a single static Go binary; fak's registries are code-registered | off-axis | Inspire-only, not filed |
| C5 | Managed binary auto-download (ripgrep) (`managed_tools.py`) | fak deliberately relies on the system `rg` via the Grep tool; bundling a second binary is anti-philosophy (one static binary) | off-axis | Dropped |
| C6 | Capability-flag-gated option validation: reject `snapshot_name`/`sandbox_id` for a provider whose metadata does not advertise it, *before* any work (`sandbox_factory.py:113-126`) | `internal/modelreg` grep: no `supports`/`capability`/`advertise` — the flag-gated early-reject pattern is absent — **but** no crisp fak seam where an unsupported option is silently accepted and fails late was located this pass | ABSENT (no seam) | **Not filed** — candidate for a future pass |
| C7 | Best-effort cleanup that never masks the original error; cleanup-only-if-created (`sandbox_factory.py:132-172`) | Robustness idiom; no mis-behaving fak lifecycle seam located | — | Not filed |

## Filed

- **Epic #4200** — `epic(deepagents-study): mine langchain-ai/deepagents ... — witnessed borrows` (mirrors the sibling study-epic convention: #4040 claudecode-study, #3983 kvcache-factory-study, #3366 lmcache-study).
- **Leaf #4203** — `feat(preflight): consolidate per-bench host-readiness preflight onto an install-hint-as-data helper (borrow deepagents SandboxInstallHint)`. PARTIAL. First checkable step: extract `internal/hostdep` (`Dep{Name, Probe, Install}` + `Classify`), refactor the two existing preflights onto it, `go test ./internal/terminalbench ./internal/livecodebench` stays green. `inspire`.

## Not filed (recorded above)

- **C6** capability-flag early-reject — genuinely ABSENT in `modelreg`, but per the field-borrow "no file:line seam, no issue" rule it is held for a future pass once a late-failure seam is located.
- **C4 / C5 / C7** — off-axis for a single static Go binary or robustness idioms without a mis-behaving fak seam.

## Companions / cross-links

Same genre as the sibling foreign-repo study passes: [claude-code](CLAUDECODE-STUDY-2026-07-10.md) (#4040), [KVCache-Factory](https://github.com/anthony-chaudhary/fak/issues/3983), [lmcache](CONCEPT-STUDY-LMCACHE-2026-07-08.md) (#3366). A crawl is not a borrow; a study is not a ship — this note + #4200/#4203 grow backlog; ancestry (`Fixes #4203` on the trunk) resolves it later, by a different worker.
