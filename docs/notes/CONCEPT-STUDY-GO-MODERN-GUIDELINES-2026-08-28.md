# JetBrains go-modern-guidelines: version-aware guidance is the borrow; the current corpus is WATCH

> Studied 2026-08-28. Source: [JetBrains/go-modern-guidelines](https://github.com/JetBrains/go-modern-guidelines), pinned at [`40781f167719913666fe2a7dc1c77ea6f256df0a`](https://github.com/JetBrains/go-modern-guidelines/commit/40781f167719913666fe2a7dc1c77ea6f256df0a). License: Apache-2.0. Tracker: [#9848](https://github.com/anthony-chaudhary/fak/issues/9848). Durable receipt: `study_b146a02126f7253aa7e746be05d330b20cf6e5e8acd58cfaf7882442ba1d6778`.

## Verdict

The strongest transferable mechanism is not a static Go style prompt. It is a small deterministic CLI that resolves the target module's Go version, returns a complete compact list of compatible rule IDs, and expands rationale/examples only for selected IDs. That design directly addresses model training lag and frequency bias without placing every rule in every prompt.

FAK already has the stronger generic substrate: skill name/description cards stay resident, whole bodies are content-addressed pages, and `fak skill` can query, inspect residency, and hot-swap versions. What FAK lacks on-axis is the Go-specific optional module: target-version resolution, a modern-Go rule catalog, nested rule-level `list`/`explain`, and a safety gate that prevents unverified before/after examples from reaching agents.

The pinned upstream corpus is **not safe to vendor as authoritative**. It passes its package tests, but post-pin [issue #14](https://github.com/JetBrains/go-modern-guidelines/issues/14) and PRs [#13](https://github.com/JetBrains/go-modern-guidelines/pull/13), [#20](https://github.com/JetBrains/go-modern-guidelines/pull/20), [#22](https://github.com/JetBrains/go-modern-guidelines/pull/22), [#23](https://github.com/JetBrains/go-modern-guidelines/pull/23), and [#24](https://github.com/JetBrains/go-modern-guidelines/pull/24) document behavior-changing examples, wrong API-version metadata, fallback/version-detection errors, and permissive malformed-version parsing. The architecture is ADAPT; the current content snapshot is WATCH.

## Value frame

- **For:** agents and maintainers editing Go across FAK and FAK-created harnesses.
- **Problem:** models emit older idioms, while timeless advice can recommend APIs a target module cannot compile or can erase behavior-changing caveats.
- **Today:** FAK exposes generic skill paging and Go build/test tooling but no version-filtered modern-Go capability.
- **Better because:** a bounded optional module can resolve applicability, page detail selectively, and fail closed on unwitnessed transformations.
- **Witness:** epic [#9854](https://github.com/anthony-chaudhary/fak/issues/9854) requires a captured edit session that resolves `go.mod`, selects compatible rules, pages one explanation, rejects a deliberately non-equivalent rule, and remains build/test green.

Centrality: **Enabling**. P1 improves bounded task context; P2 requires net tool-call/retry/validation accounting; P3 keeps rules versioned and optional; P4 makes resolution, validation, disclosure, and receipts queryable.

## What shipped at the pin

Merged [PR #11](https://github.com/JetBrains/go-modern-guidelines/pull/11) replaced an approximately 291-line static skill with a typed Go CLI, embedded structured data, cross-platform wrappers, and thin Junie/Claude/Codex/Cursor manifests. The load-bearing paths are:

- applicability precedence and module/workspace resolution: [`internal/goversion/goversion.go:18-100`](https://github.com/JetBrains/go-modern-guidelines/blob/40781f167719913666fe2a7dc1c77ea6f256df0a/internal/goversion/goversion.go#L18-L100);
- embedded catalog and version filtering: [`internal/guidelines/guidelines.go:25-185`](https://github.com/JetBrains/go-modern-guidelines/blob/40781f167719913666fe2a7dc1c77ea6f256df0a/internal/guidelines/guidelines.go#L25-L185);
- list-first, selected-explain discipline: [`plugin/skills/use-modern-go/SKILL.md:24-66`](https://github.com/JetBrains/go-modern-guidelines/blob/40781f167719913666fe2a7dc1c77ea6f256df0a/plugin/skills/use-modern-go/SKILL.md#L24-L66);
- exact-tag lazy install, environment isolation, version read-back, and staged promotion: [`run-tool.sh:29-65`](https://github.com/JetBrains/go-modern-guidelines/blob/40781f167719913666fe2a7dc1c77ea6f256df0a/plugin/skills/use-modern-go/scripts/run-tool.sh#L29-L65);
- table integrity and output-shape tests: [`guidelines_test.go:53-168`](https://github.com/JetBrains/go-modern-guidelines/blob/40781f167719913666fe2a7dc1c77ea6f256df0a/internal/guidelines/guidelines_test.go#L53-L168).

The pinned tree has 29 tracked files, 54 rules, and 63 before/after examples. `go test ./...` passed for `internal/cli`, `internal/goversion`, and `internal/guidelines` on Go 1.26.6/Windows amd64.

## Worldview and tradeoff

The repository is for code-generating agents working inside real Go modules, not a human style-guide reader. Its design assumes the model's training cutoff and frequency bias are persistent, so a deterministic tool should supply the current delta. Its distribution boundary is multi-harness: one semantic skill, thin host adapters, equivalent POSIX and PowerShell wrappers.

The central tradeoff is **compactness versus safety visibility**. The short list saves context, but the pinned skill calls `explain` mainly before skipping a rule. That allows an agent to apply a terse recommendation without seeing its caveats. Post-pin PR #13/#20 correctly shifts behavior-changing caveats into the summary boundary. Any FAK adaptation must keep a caveat in the smallest page that can cause action.

## Source ledger

Observed at `2026-08-28T21:09:39Z`.

| Source | Event/state | Immutable anchor | What it changed | Refresh trigger |
|---|---|---|---|---|
| `main` | 2026-08-19, shipped/current | [`40781f1`](https://github.com/JetBrains/go-modern-guidelines/commit/40781f167719913666fe2a7dc1c77ea6f256df0a) | Pins the studied plugin 1.1.1 / CLI v0.1.1 state. | `main` moves |
| CLI redesign | merged 2026-08-11 | [PR #11](https://github.com/JetBrains/go-modern-guidelines/pull/11) | Proves static-prompt -> deterministic versioned disclosure was intentional. | architecture changes |
| tags | v0.1.0-dev.1, v0.1.0, v0.1.1; no GitHub Releases | `v0.1.1 -> 937827b86eeff98f45150ce00236bc34a30ed20f` | Shows module tags and plugin versions are separate release sequences. | new/moved tag or Release |
| semantic correction | merged 2026-02-25 | [PR #2](https://github.com/JetBrains/go-modern-guidelines/pull/2) | Shows external language facts change and need source-pinned review. | rule change/revert |
| deterministic modernization | open/proposed | [issue #9](https://github.com/JetBrains/go-modern-guidelines/issues/9) | Supplies a recipe candidate: `go fix -diff` before residual model work. | issue/implementation changes |
| correctness cluster | open, created 2026-08-26..27 | [#14](https://github.com/JetBrains/go-modern-guidelines/issues/14), [#13](https://github.com/JetBrains/go-modern-guidelines/pull/13), [#20](https://github.com/JetBrains/go-modern-guidelines/pull/20), [#22](https://github.com/JetBrains/go-modern-guidelines/pull/22), [#24](https://github.com/JetBrains/go-modern-guidelines/pull/24) | Demotes the corpus to WATCH and makes semantic conformance a prerequisite. | any merge/close/review/force-push |
| license/provenance | unchanged since first commit | [`LICENSE@40781f1`](https://github.com/JetBrains/go-modern-guidelines/blob/40781f167719913666fe2a7dc1c77ea6f256df0a/LICENSE) | Apache-2.0 permits adaptation with license/notice/modification obligations; no repository NOTICE exists. | license/provenance changes |

## FAK on-axis witness

Three self-query forms and raw search were used for each surviving axis.

- `fak capabilities "target project language version filtered coding guidance"` returned no on-point result.
- `fak capabilities "progressive disclosure list ids then explain selected guidance"` returned no match.
- `fak dev index docs|leaves|verbs|claims` found generic Go/build/skill surfaces but no modern-Go catalog or workflow.
- `fak skill query "modern Go version-aware code guidance" --budget 5 --json` faulted unrelated skills and reported 69,528 body bytes; no modern-Go skill exists.
- Raw search found many local `go.mod` walks but no `go fix`, modernize integration, rule catalog, or rule-level equivalence fixtures.

The ablated result is not “FAK lacks progressive disclosure.” Generic paging is PRESENT in `internal/policy/skill.go` and `cmd/fak/skill.go`; exact skill version/digest/install read-back is also PRESENT. The gap is the optional Go-specific data/tool path and nested rule-level disclosure. Existing [#1103](https://github.com/anthony-chaudhary/fak/issues/1103) and [#7110](https://github.com/anthony-chaudhary/fak/issues/7110) remain the generic substrate.

## Candidate matrix

| Borrow/axis | FAK verdict | Portfolio/license | Decision and disconfirming witness |
|---|---|---|---|
| Target-file Go-version resolution + compatible terse rule list | **ABSENT** | **OPTIONAL-MODULE · ADAPT** | File [#9855](https://github.com/anthony-chaudhary/fak/issues/9855). Exclude if a two-version fixture cannot outperform a static repo-only instruction without unsafe fallback. |
| Selected ID detail paging with summary-visible caveats | **PARTIAL** | **OPTIONAL-MODULE · ADAPT** | Generic paging exists; [#9858](https://github.com/anthony-chaudhary/fak/issues/9858) dogfoods a smaller content-backed page. Reject if tool-call + prompt + retry cost is not net lower. |
| Syntax/type/version/observable-behavior witness per rule | **ABSENT** | **OPTIONAL-MODULE · ADAPT** | File [#9857](https://github.com/anthony-chaudhary/fak/issues/9857). A rule without its declared witnesses stays out of the agent-visible list. |
| Deterministic `go fix -diff` -> apply -> rerun -> build/test before model residuals | **ABSENT** | **RECIPE · INSPIRE-ONLY** | File [#9856](https://github.com/anthony-chaudhary/fak/issues/9856). Keep preview/read-only default and run away from the live peer-dirty tree. |
| Whole-skill metadata/body paging, versioning, audit, hot-swap | **PRESENT** | **DEFAULT** | Keep #1103/#7110; no duplicate issue. |
| Exact plugin artifact identity/read-back | **PRESENT** | **DEFAULT** | Keep FAK's digest/version/install surfaces; upstream's wrapper adds no on-axis gap. |
| Current 54-rule corpus | **PARTIAL** | **WATCH** | Do not vendor until the open correctness cluster resolves and an independent FAK conformance run passes. |
| Self-installing wrapper that can fetch during skill invocation | **DIVERGENT** | **EXCLUDE as FAK default** | FAK keeps extension acquisition explicit and kernel-governed; lazy network installation may remain an external plugin behavior, not a widened default authority. |

Default frontier: keep FAK's kernel-owned skill paging/version/digest model. Coverage frontier: add the optional modern-Go resolver/catalog, rule-detail paging, conformance gate, and `go fix` recipe without widening the default binary or tool authority.

## Negative knowledge

- “Modern” is not synonymous with behavior-preserving: nil versus empty values, NaN comparison, callback lifetime, aliasing, serialization, and absent-separator branches all break naive rewrite equivalence.
- Table-shape and golden-output tests do not typecheck examples or prove semantics.
- A directive-less module must not inherit the host's newest Go version and receive uncompilable advice.
- Grouped feature documentation can conflate distinct API introduction versions; each rule needs its own source-of-truth version witness.
- Inline shell snippets are a portability and agent-permission hazard; issue #7 was fixed by typed cross-platform tooling.
- Even a repository devoted to knowledge-cutoff repair received a stale “generic methods are invalid” PR after Go 1.27 shipped them. Human contributions need the same version witness as model output.

## Completeness and limits

The deep fan-out opened every one of the 29 tracked files and all 54 rules/63 examples; deepened and read all 25 commits and all refs/tags; exhausted all 12 issues and all 12 PRs including comments, reviews, inline threads, and timelines; checked releases, discussions, wiki, roadmap/RFC/ADR/unfinished-work-marker, license, NOTICE, contributions, submodules, vendoring/generated provenance, and the FAK seams. No load-bearing source/content/validation subsystem remains unopened.

Discussions are disabled. No releases, milestones, wiki content, RFC/ADR/roadmap/unfinished-work-marker files, submodules, vendor tree, generated markers, or NOTICE exist. GitHub Projects is enabled, but ProjectsV2 could not be enumerated because the token lacks `read:project`; that is an unavailable surface, not proof that no board exists.

Refresh on a new upstream commit/tag/release, any correctness-cluster disposition, a new semantic-conformance path, or a changed FAK skill-runtime/modern-Go frontier. This note does not claim the four FAK issues are implemented.

## Filed trail

- [#9848](https://github.com/anthony-chaudhary/fak/issues/9848): study tracker.
- [#9854](https://github.com/anthony-chaudhary/fak/issues/9854): optional modern-Go epic.
- [#9855](https://github.com/anthony-chaudhary/fak/issues/9855): version resolver + compact list spine.
- [#9858](https://github.com/anthony-chaudhary/fak/issues/9858): selected detail + caveat boundary.
- [#9857](https://github.com/anthony-chaudhary/fak/issues/9857): rule conformance witnesses.
- [#9856](https://github.com/anthony-chaudhary/fak/issues/9856): deterministic `go fix`-first recipe.

Companions: [`field-borrow`](../../.claude/skills/field-borrow/SKILL.md) · [`study-repo`](../../.claude/skills/study-repo/SKILL.md) · [queried skill loader #1103](https://github.com/anthony-chaudhary/fak/issues/1103) · [dynamic skill runtime #7110](https://github.com/anthony-chaudhary/fak/issues/7110).
