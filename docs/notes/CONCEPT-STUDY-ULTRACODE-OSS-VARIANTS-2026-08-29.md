# Ultracode OSS variants: mechanisms are real; comparative effectiveness is not yet proven

> Audited 2026-08-29 against FAK `62a8402072ee24bd72de77b7390fda31dea329e3`. Durable receipt: `study_e9b23af8a4f1607b7df8192abc25b87c429b798d293832476c77703881b9a870`. This is a bounded current-tree inventory, not an exhaustive history or forge-conversation audit.

## Verdict

FAK has a real, fail-closed Ultracode planning and evaluation surface, but broad effectiveness is **not proven**. The offline self-check proves a four-role DAG, leases, budgets, independent readback, and reconciliation without launching workers. The current live single-versus-fleet replay returns `ABSTAIN`: both arms accepted 3/3 effects, but the fleet took 34.576 seconds versus 16.277 seconds for the single arm (`0.47076x` concurrency speedup, `-0.529242` accepted-outcome-per-wall gain). Activation, independent witness digests, billed tokens, spend, and a current budget receipt are incomplete.

The positive small-model evidence is narrower. The qwen2.5:0.5b/Ollama campaign shows scoped-context and prefix-access avoidance at widths 1, 2, 4, and 8, including 117,410 avoided accesses attributed 62.7 percent to FAK role scoping. It does not prove coding throughput, broad fleet speed, billed savings, or provider-independent value.

The public variants ship useful orchestration mechanisms, but none of the inspected evidence is commensurate with FAK's accepted-outcome denominator. Transcript token summaries, static price multiplication, unit tests, cassette replay, and unmatched task studies establish implementation shape—not comparative productivity. The next honest step is a dry capability/acceptance matrix followed by a matched live comparison whose missing evidence becomes `ABSTAIN`, not a ranking.

## Value frame

- **For:** operators deciding whether and how to use FAK Ultracode, provider-native collaboration, or a public workflow implementation.
- **Problem:** the name covers unrelated product shapes, while evidence ranges from prompt definitions to executable runtimes and incomparable token anecdotes.
- **Today:** FAK's live value pair abstains; public projects demonstrate mechanisms but lack a common accepted-outcome denominator.
- **Better because:** a pinned inventory separates shipped mechanics, license boundaries, and evidence strength before any matched comparison is attempted.
- **Witness:** one contract-valid benchmark leaf requires identical task/start revision, an accepted-artifact oracle, effective activation, terminal-effect readback, retries, repeated trials, and authority-labelled wall/token/cache/billed/spend fields.

Centrality: **Core** for the comparison leaf and **Stewardship** for this inventory. P1 advances context attribution; P2 keeps net value coupled to accepted outcomes; P3 bounds every arm and abstains on incomparable capability; P4 preserves pins, authorities, receipts, and terminal evidence.

## Current FAK evidence

| Surface | Status | What the witness actually proves |
|---|---|---|
| `fak ultracode --selfcheck --json` | **PRESENT** | Offline plan shape: four roles, a dependency DAG, 65,536-token declared budget, leases, independent readback, and reconciliation. It launches zero workers and proves no live effectiveness. |
| `fak ultracode bench --pair docs/_witnesses/issue-8168-ultracode-live/pair.json --json` | **ABSTAIN** | Equal 3/3 accepted effects; fleet critical path 34.576s versus single 16.277s; activation, witness, billed/spend, and budget-receipt gaps prevent a gain claim. |
| `docs/_witnesses/issue-8624-ultracode-smallmodel/` | **GAIN in one access envelope** | Scoped-context and prefix reuse avoid context accesses on qwen2.5:0.5b/Ollama. Width 1 also gains; this is not a fleet-speed result. |
| `internal/ultracodebench/ultracodebench.go:130-215` | **PRESENT** | Identity/budget equality, activation, passing equal outcomes, retry exclusion, independent witnesses, and requested cost authority are required before `GAIN`; gaps become `ABSTAIN`. |
| Live operational tickets | **CONTRADICTED** | #8801, #8855, #10019, #10163, and #10164 record cumulative or duplicate usage, stale task identity, post-hoc 426x budget failure, descendant-accounting ambiguity, and missing structured wait state. |
| Launcher default | **REGRESSED** | #5016 required `auto`; `cmd/fak/accounts.go:173` now defaults to `on` even though `accounts_launch.go:75-112` calls grind work pure overhead and maps auto grind/unknown to off. |

## Pinned public inventory

| Source | Shape and strongest shipped mechanism | Effectiveness evidence and boundary | License disposition |
|---|---|---|---|
| [`diepquynh/ultracode@63f0d6e`](https://github.com/diepquynh/ultracode/tree/63f0d6e4dbf62258190e938ee4ebf9b93278f119) | Prompt/hook pipeline with per-repository ordered phases, cross-repository ready-node parallelism, approval/fact-check gates, review loops, and a repeated-build circuit breaker. | Its benchmark counts transcript tokens, reads, searches, errors, and failure streaks, not accepted outcomes. Tests are optional at the closing gate, weaker than FAK's proof-by-default contract. | MIT; **INSPIRE/ADAPT**, no code copied. |
| [`hesreallyhim/ultracode-workflows@9b5404d`](https://github.com/hesreallyhim/ultracode-workflows/tree/9b5404d11b885b28380d3eb17471ef7b17601b5e) | Deterministic workflow scripts: multi-angle review, file/line dedupe, independent adversarial verification, bounded loop-until-dry, resume state, static parse lint, and cassette replay. | Good shape and replay evidence; the contract checker is AST-only, resume trusts supplied state, and no paired accepted-task study is committed. | MIT; **INSPIRE/ADAPT**, no code copied. |
| [`Suraj1235/open-dynamic-workflows@972bb98`](https://github.com/Suraj1235/open-dynamic-workflows/tree/972bb98494ea23f907df88850024bd7022b099d4) | Executable QuickJS-WASM workflow runtime with memory/interrupt limits, bounded queues/retries, side-effect cache refusal, critic quorum plus green-test gate, checkpoints, and orphan recovery. | Integration tests prove crash/resume and fail-closed dual gating. Cost is static token-price multiplication; runtime checkpoints are not independent effect proof, and its own evidence note leaves real-provider workflow evidence incomplete. | MIT; **INSPIRE/ADAPT**, no code copied. |
| [`Caleb0796/codex-ultracode-kit@8176113`](https://github.com/Caleb0796/codex-ultracode-kit/tree/8176113ed18b0df1ef1a2717e2eb556aaf132018) | Real `codex exec` harness with a global semaphore, capped fanout, strict result schema, adversarial ledger, and HEAD-bound prefix resume for eligible read-only work. | The test plan records a roughly 30K baseline and 273K two-agent input with 76 percent cached, but most behavioral rows are unrun. Budgets are soft, writer/worktree agents do not replay, and its branch/worktree/`git add -A` path conflicts with FAK's shared-trunk rules. | MIT; **CONCEPT-ONLY**, no code copied. |
| [`KangarooKi/Ultracode@4a7d1b6`](https://github.com/KangarooKi/Ultracode/tree/4a7d1b6eb10e4e020b628b7fbaa21c1837b31edd) | Adjacent Python coding CLI with a synchronous depth-capped child, JSON task graph, hooks, permissions, MCP, memory, and compaction. | Unit/mock evidence only; task tests cover graph CRUD rather than concurrent graph execution. This is a homonym/product-surface comparator, not a demonstrated fleet runtime. | No root license found; **DO-NOT-COPY / INSPIRE-ONLY**. |
| [`toel1234/oh-my-opencode@18d7643`](https://github.com/toel1234/oh-my-opencode/tree/18d7643bc1eb4f3e99e0e80e7b8cca9fe199ff1a) | Historical Ultrawork lineage with orchestrator, specialists, background delegation, and persistent completion framing. | Adjacent prompt/runtime lineage rather than a matched executable benchmark. Current lineage has since moved and split into componentized OpenAgent work. | Sustainable Use License; **INSPIRE-ONLY / DO-NOT-PORT**. |
| [`QuintinShaw/pi-dynamic-workflows@2f85c9e`](https://github.com/QuintinShaw/pi-dynamic-workflows/tree/2f85c9eb350319e14a2ff16b98986a30b3c8bb4d) | Batch-scoped cancellation, nested caps, ordered state deltas, prefix resume, and draining of unawaited children before terminal completion. | Strong directional study: 74/81 versus 73/81 tasks and median loaded context 2,564 versus 8,096, but prompts/fixtures differ and the raw provider artifact is uncommitted. | MIT; **WATCH / INSPIRE**, no code copied. |
| [`code-yeongyu/oh-my-openagent@b4e0094`](https://github.com/code-yeongyu/oh-my-openagent/tree/b4e0094c04df49360271fd9e449681d0c386eb35) | Current Ultrawork component: observable stop goal, atomic todos, cleanup, real-surface QA scenarios, append-only notes, and capped high-reasoning review. | Component/evaluation plumbing, not a general coding-productivity result. Repository license metadata is `NOASSERTION`; component-level license must be rechecked before reuse. | **INSPIRE-ONLY pending provenance**. |
| [`Yeachan-Heo/oh-my-claudecode@adf4bf3`](https://github.com/Yeachan-Heo/oh-my-claudecode/tree/adf4bf3280c8a8d7b932d5c11aef84ba22d6a11d) | Historical Ultrawork persistence mode and stop-hook loop. | Ultrawork was retired; issues documented a loop that kept echoing initial intent after work completed. This is negative evidence for live-goal-derived reinforcement and immediate cancellation. | MIT; **NEGATIVE-KNOWLEDGE / INSPIRE**, no code copied. |

## Candidate dispositions

| Candidate | FAK status | Disposition |
|---|---|---|
| Matched dry capability/acceptance matrix, then live accepted-outcome variants | **PARTIAL** | **FILED #10176** under the Ultracode runtime program; consume #8559 and #5971 rather than duplicating receipt/runtime work. |
| Effective activation, effect readback, equal identity/budgets/outcomes, retry exclusion, and authority-labelled costs | **PRESENT/PARTIAL operationally** | **KEEP** the fail-closed evaluator; resolve #8559/#5971/#10019/#10163/#10164 before a gain claim. |
| Task-adaptive default posture | **REGRESSED** | **REOPEN #5016**; do not file a duplicate. |
| Longest-prefix resume, ordered replay, effect-cache refusal, graph state, and adapters | **PRESENT or already owned** | **NO NEW ISSUE**; route to #2444/#8490/#5970/#5971/#5973 as applicable. |
| Batch cancellation/drain, static source contracts, repeated-build breaker | **UNCONFIRMED gap** | **WATCH**. Current bounded self-query does not justify a new implementation ticket; full history/forge study is still missing. |
| Unlicensed or non-permissive source code | **INTENTIONALLY ABSENT** | **EXCLUDE** direct reuse. |

## Backlog reconciliation

- Reopen #5964 as the durable runtime/effectiveness parent: its broad definition of done remains unchecked, while the closing comment cited only an accounting commit.
- Reopen #5016 because a later commit deliberately changed its required `auto` default back to `on`.
- Keep #8168 closed but explicitly superseded by #8559 for current receipt recapture; do not claim its `ABSTAIN` artifact as a completed effectiveness proof.
- Scope #8801 to direct-worker session baseline/delta accounting; scope #10163 to descendant-inclusive Codex root-goal authority; #10019 consumes both for live admission/enforcement.
- Keep #8855 as task/transcript/effect identity, and #10164 as structured Codex wait lifecycle. These are prerequisites, not effectiveness results.
- Add the active leaves as real subissues of #5964 so issue prose is no longer the only hierarchy.

## Completeness and limits

The coordinator independently re-read the pinned heads, root license presence, load-bearing implementation paths, current FAK evaluator, live artifacts, and related issue bodies. The nine-source inventory covers current-tree mechanisms and explicitly separates a historical lineage and a homonym.

It does not cover every commit, issue, pull request, discussion, release, external package artifact, provider trace, or raw benchmark record for every source. The shallow clones therefore remain `candidate` in the monitored-repository registry. Those omissions block broad historical, licensing, and productivity claims; they do not change the observed absence of a matched accepted-outcome denominator.
