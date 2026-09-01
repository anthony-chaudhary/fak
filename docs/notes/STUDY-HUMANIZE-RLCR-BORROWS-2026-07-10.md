---
title: "Study: PolyArch/humanize (RLCR) — borrow candidates for fak (2026-07-10)"
description: "Deep-read of the PolyArch/humanize Claude-Code plugin (RLCR: Ralph-Loop with Codex Review) @ 0ec921a36b4365df503511c5567bbd3e02db0df5 — 10-agent parallel study (8 subsystem readers + completeness critic + citation-checked synthesis, ~1.05M tokens). humanize is a bash+CC-config harness that turns one Claude build agent into a self-terminating loop whose correctness is graded by an INDEPENDENT second model (Codex), pinned to a fixed base SHA, emitting closed-vocabulary severity/verdict tokens. This note is the borrow backlog: 14 ranked candidates (each grounded at a real path:line under the pinned SHA and ablated to one axis), the TOP-3, the anti-patterns fak should NOT copy, and a dedup list of what fak already has. Every PRESENT/PARTIAL/ABSENT tag is a HYPOTHESIS to be witnessed against fak's own code before any issue is filed."
---

# Study: PolyArch/humanize (RLCR) — borrow candidates for fak

> Date: 2026-07-10. Source: `github.com/PolyArch/humanize` @ `0ec921a36b4365df503511c5567bbd3e02db0df5`
> (read-only clone in scratch, never in-tree). Method: 10-agent workflow — 8 parallel
> subsystem readers → completeness critic (read every file class the readers skipped) →
> synthesis with 4 citations spot-checked and confirmed. ~1.05M subagent tokens.
>
> **Status: backlog candidates, NOT yet witnessed against fak.** Every PRESENT/PARTIAL/ABSENT
> below is the *readers'* hypothesis from reading humanize; each must be witnessed against
> fak's actual code (at the axis it targets, not the capability name) before a gh issue is
> filed. Filing is a separate, explicitly-authorized step.

## 1. What humanize is

The RLCR ("Ralph-Loop with Codex Review") Claude-Code plugin: a bash + CC-config harness that
turns a single Claude "build" agent into a self-terminating iterate-until-done loop. The whole
control structure is one **Stop hook** (`hooks/hooks.json:63-73`) that intercepts every
turn-end and returns `{"decision":"block","reason":…}` to force another round while feeding the
next round's prompt — so refusal and dispatch are the same primitive. Correctness is
adjudicated not by the builder but by an **independent second model** (Codex) pinned to a fixed
base SHA (`hooks/loop-codex-stop-hook.sh:1214-1221,1266`), whose findings must arrive in
machine-parseable closed vocabularies (`[P0-9]` severity, `Mainline Progress Verdict:
ADVANCED/STALLED/REGRESSED`). Around that spine: fail-closed PreToolUse validators, an
evidence-bound "BitLesson" project-memory subsystem, a plan-comprehension gate, and a post-run
methodology self-audit — every phase encoded as an unforgeable on-disk `*-state.md` filename
rather than a mutable field.

## 2. Borrow-candidate table (ranked; ABSENT/high first)

Legend — (a) mechanism @ source, (b) fak landing, (c) hypothesized state vs fak, (d) value.

**#1 — Independent CROSS-MODEL reviewer as a gating loop phase.** ABSENT · HIGH.
(a) A *different* model runs `codex review --base <captured-sha>` with its own hooks disabled
(`loop-codex-stop-hook.sh:1266`, recursion-guard `:1169-1177`), pinned to a fixed SHA so it
sees the whole session delta not the last commit (`:1214-1221`), forced to higher effort than
the builder (`:139`); the loop refuses to advance while any `[P0-9]` marker appears in the last
50 lines (`lib/loop-common.sh:759-766`); review failure/empty = hard block, never skip
(`:1295-1298`). (b) A gating phase in super-loop / dos-dispatch-loop; map `[P0]..[P9]` into the
closed refusal vocabulary (`dos_refuse_reasons`/`dos_check_reason`), e.g. `INDEPENDENT_REVIEW_P0`.
(c) fak's `dos_review` explicitly carries "ZERO new trust over dos_commit_audit" — it
re-projects the *author's own* commit evidence, not an independent grader. (d) Load-bearing
"no blind spots" property, squarely on fak's "kernel doesn't believe the agents" philosophy;
a different model family is structurally harder to collude with a same-family author.

**#2 — Required progress-verdict token + graduated stall→replan→stop breaker.** PARTIAL · HIGH.
(a) Reviewer MUST emit `Mainline Progress Verdict: ADVANCED/STALLED/REGRESSED`; absence blocks
the round (`loop-codex-stop-hook.sh:1826-1829`). STALLED/REGRESSED increments a persisted stall
count; ≥2 → `drift_status=replan_required` (soft, swaps in a replan prompt); ≥3 →
`MAINLINE_DRIFT_STOP` hard-ends the loop (`:1837-1848`; COMPLETE resets `:1855-1861`). Verdict
parser rejects ambiguous multi-keyword lines as `unknown` (`loop-common.sh:653-659`). (b)
`PROGRESS_VERDICT_MISSING` + `MAINLINE_DRIFT` tokens (TRUE_DRAIN category); a persisted
`mainline_stall_count`/`drift_status` in the run ledger; soft `dos_replan` at N, hard break at
N+1. (c) `dos_status` already computes a git-*witnessed* STALLED rung with no `claimed` field —
but syntactic (commits-since-start_sha); missing the independent-model semantic verdict-or-block
gate and the persisted double-threshold ladder. (d) Most-cited borrow across every reader.

**#3 — Write-time, evidence-bound memory-delta gate (BitLesson delta).** ABSENT · HIGH.
(a) Every non-finalize round summary must carry `## BitLesson Delta` with `Action:
none|add|update` + concrete `BL-YYYYMMDD-*` IDs + non-placeholder Notes; the harness refuses
exit unless each cited ID *already exists* in the KB, verified by awk lookup
(`scripts/bitlesson-validate-delta.sh:322,328-340,369-384`); empty-KB ratchet forbids
`Action: none` once the KB has real lessons (`:254-267`); extraction is fence/HTML-comment-aware
so the agent can't spoof the parser (`:92-179`). (b) A Stop/commit-time gate on
super-loop / run-it-all-night / dos-dispatch-loop; reasons `MEMORY_DELTA_MISSING` /
`MEMORY_DELTA_UNWITNESSED` / `MEMORY_NONE_ON_EMPTY_KB`; cross-check cited slugs via `dos_recall`.
(c) fak verifies memory on *read* (`dos_recall`) but has no *write-time* gate proving a round
banked a witnessed, reusable takeaway. (d) The write-time complement to `dos_recall` — same
claim-vs-artifact discipline as `dos_commit_audit`, applied to memory capture at loop-exit.

**#4 — Cheap-model per-task memory selector (retrieval-over-accumulation).** ABSENT · HIGH.
(a) `bitlesson-select.sh` runs a haiku-tier router per sub-task returning exactly two lines
`LESSON_IDS: <ids|NONE>` / `RATIONALE: <one sentence>` (`:173-180`), precision-over-recall,
read-only / no-tools / no-repo-inspection (`:166-171`), 120s timeout, deterministic KB-empty
short-circuit (`:105-109`), output re-enforced to the 2-line contract via sed (`:242-267`),
provider fallback so the loop never stalls (`:126-131`). (b) A pre-dispatch step in the dispatch
loop / `dos-plan-price`, and/or the guard SessionStart injection — select only task-relevant
note IDs instead of injecting the whole index. (c) No per-task memory retrieval routing
evident; MEMORY.md is injected essentially wholesale (the oversized index in this very session
is the evidence). (d) Directly attacks fak's demonstrated memory-index bloat.

**#5 — Corroborating-witness file so a self-reported state flag can't be forged.** ABSENT · HIGH.
(a) `review_started=true` in the agent-adjacent state.md is believed only if a separate
`.review-phase-started` marker exists; a flag set without the marker is refused as a "manual
toggle attack" (`loop-codex-stop-hook.sh:1902-1938`); missing schema fields terminate the loop
rather than defaulting (`:186-206`). (b) Guard layer / dos state validation — a loop/lease
state field the agent could edit must be backed by an independent artifact before the kernel
acts on it. (c) `dos_verify`/`dos_commit_audit` distrust *commit* claims; the analog for the
kernel's *own* control state is not evident. (d) Cheap, high-value guard against loop-state
tampering — pure fak "don't believe the agent" turned inward.

**#6 — Bash write-footprint extractor (turn "unknown blast radius" into a path set).** PARTIAL/ABSENT · HIGH.
(a) `command_modifies_file()` matches `>`,`>>`,`tee`,`sed -i`,`awk -i inplace`,`perl -i`,
`mv`/`cp`,`rm`,`dd of=`,`truncate`,`exec N>` (`lib/loop-common.sh:1485-1510`); the Bash
validator splits on `; | & && || |&`, re-checks each segment for `mv/cp <governed>` as source,
strips leading redirections, unwraps `sh -c`/`bash -c` payloads
(`loop-bash-validator.sh:321-488`). (b) PreToolUse guard / `dos_arbitrate` admission — extract a
Bash command's write footprint so arbitration can prove (non-)collision. (c) fak's own DOS
PRE-admission advisory fires on *every* Bash call ("lane 'Bash' has an EMPTY tree (unknown
blast radius)") — it currently cannot resolve a Bash command's write set. (d) Converts an
unresolvable Bash lane into a concrete tree tree-disjointness can adjudicate. **Borrow the
footprint extraction; NOT the shell-form denylist treadmill (see anti-pattern C).**

**#7 — Post-run methodology self-audit → sanitized cross-project lesson (RSI on the harness).** PARTIAL/ABSENT · MED-HIGH.
(a) On terminal exit an independent Opus agent grades all round/review records on 7 loop-health
axes incl. stagnation and plan-to-execution drift (`methodology-analysis-prompt.md:33-39`), then
a hard sanitization allow-list strips paths/symbols/hashes/domain terms/code (`:22-30`) to yield
a portable lesson; fail-closed on non-empty report + valid exit-reason enum
(`lib/methodology-analysis.sh:114-174`). (b) A terminal methodology-analysis phase in
super-loop/RSI feeding guard-journal + refusal history to a cross-model auditor, filing a
domain-free lesson into docs/notes. (c) fak has guard-accuracy RSI; the *sanitized,
project-agnostic methodology lesson* (RSI on the orchestration process, not guard accuracy) is
the delta; the allow-list is directly reusable as a pre-notes filter.

**#8 — Plan-comprehension discrimination quiz gating the OPERATOR.** ABSENT · MED-HIGH.
(a) An opus quiz emits 2 four-option MCQs testing HOW the plan is implemented (not WHAT), with
distractors from this plan's specifics so a title-skimmer picks wrong
(`agents/plan-understanding-quiz.md:40-42`), randomized answer position (`:81`); **advisory not
a gate** (`commands/start-rlcr-loop.md:86,100-105`); skip when the plan came from a converged
planner (`gen-plan.md:608`). (b) A pre-dispatch advisory reason `PLAN_UNCOMPREHENDED`; a skill
beside `dos-plan-price`; skip when the plan was produced by fak's own converged planner. (c) fak
prices/audits the plan artifact but nothing verifies the dispatching operator *understands* it
before a loop amplifies it. (d) On-target given fak's memory of "dispatch #N already resolved /
misroute" failures. **Keep advisory (anti-pattern D).**

**#9 — Corrective, injection-safe templated refusal payloads.** PARTIAL · MED.
(a) ~40 externalized block templates, each with an inline hardcoded fallback so a missing
template still denies (`lib/template-loader.sh:188-211`), rendered by a single-pass awk
`{{VAR}}` substituter that deliberately does NOT re-scan substituted values (`:56-136`, esp.
`:116-117`) — an attacker-controlled path can't expand into another placeholder; each block
names the one legal next move. (b) Give each fak reason a fallback-guarded, parameterized,
redirect-to-the-one-legal-move template. (c) fak reasons carry `summary`/`fix`/`see_also`; the
delta is the per-instance rendered redirect + the non-re-scanning single-pass renderer.

**#10 — Typed CMT:/ENDCMT comment→classify→disposition ledger for plan refinement.** ABSENT · MED.
(a) A stateful POSIX-awk scanner tracks fence/HTML-comment/CMT/heading state so markers in
code are ignored, fail-closed on malformed annotation with line+column+heading+excerpt
(`scripts/validate-refine-plan-io.sh:16-315`); each block → {question, change_request,
research_request} → {answered, applied, researched, deferred, resolved}, emitting a comment-free
plan + an auditable QA ledger; transactional temp-file-per-output
(`commands/refine-plan.md:508-565`). (b) A `dos-replan`-adjacent skill: operator annotates a fak
plan with `CMT:` blocks; the skill yields a refined plan + a ledger binding every comment to a
closed disposition.

**#11 — Plan-level branch-switch pre-check (shift OFF_TRUNK left).** PARTIAL · MED.
(a) `plan-compliance-checker` greps the plan prose for `git checkout -b`/`git switch`/worktree
creation and fails `FAIL_BRANCH_SWITCH` before the loop runs, disambiguating safe lookalikes
(`git checkout -- <file>`, negations, `--base-branch`) to hold FPs down
(`agents/plan-compliance-checker.md:32-46`). (b) A plan-admission linter emitting
`PLAN_MANDATES_OFF_TRUNK`; the disambiguation table is what guard-RSI would tune for FP rate.
(c) fak enforces OFF_TRUNK only at commit time; humanize catches the intent in the plan text.

**#12 — Adversarial / property test corpus for the guard+path+concurrency surface.** PARTIAL/ABSENT · MED.
(a) `tests/robustness/*`: a shell-metacharacter injection matrix + a symlink battery incl.
chains A→B→C (`test-path-validation-robustness.sh:199-332,367-377`); atomic-write-via-`mv`
gives 20/20 consistent reads under a concurrent writer + zombie-loop protection (newest-dir-only)
(`test-concurrent-state-robustness.sh:279-322,186-201`); nproc-throttled parallel harness with
mock-codex injection (`run-all-tests.sh:133-141,202-218`). (b) An adversarial test corpus for
fak's guard/path/arbitration surface. (c) Best guess: fak manage tests skew happy-path.

**#13 — Parent-only canonicalization as the path-authorization primitive.** PARTIAL · MED.
(a) `canonicalize_path_prefix` resolves ONLY the parent and reattaches the basename verbatim
(`hooks/lib/project-root.sh:72-96`), with an explicit note that the leaf-dereferencing
`canonicalize_path` must NOT authorize a user-supplied path (`:106-109`) — because `mv`/write
acts on the link path itself, a leaf symlink alias must not canonicalize away. (b) fak path-guard
sites comparing a write target against an expected location. (c) The load-bearing helper behind
"canonicalize parent, compare basename verbatim."

**#14 — Lockstep version-bump CI gate.** PARTIAL · MED-LOW.
(a) Exactly-+1 semver enforced in lockstep across `plugin.json`, `marketplace.json`, `README.md`
with a distinct error per violation class (`.github/workflows/version-bump-check.yml:40-120`);
`pr-target-check.yml:19-28` lets only `dev` target `main`. (b) fak's `release` skill / CI.
(c) Skill discipline vs a mechanized CI reject.

**Folded / lower-value** (cite for completeness; fold into adjacent tickets): cost-ordered gates
(free checks before Codex tokens, `loop-codex-stop-hook.sh:402-782`); cross-vendor
archived-consult-as-artifact (`ask-gemini.sh` forced web-research + cite-sources + full
{input,output,metadata} archival, `:236-241,294-380`) → a `fak research` verb `dos_verify`/
`dos_citation_resolve` can witness; `gen-idea` directed-orthogonal-diversity swarm with an
anti-fabrication sentinel `exploratory, no concrete precedent` (`commands/gen-idea.md:108-142`);
finalize/simplify gated terminal phase constrained to this run's diff with a tests-pass witness
(`finalize-phase-prompt.md:9-24,42-52`); PostToolUse leader-only session-binding hook
(`loop-post-bash-hook.sh:20-21,85-101`); task-list-completeness gate, lane-typed, DEFAULT-to-blocking
(`check-todos-from-transcript.py:24-35,110-112`); model-router capability clamp
(`scripts/lib/model-router.sh:62-91`); filesystem-as-bus observability (terminal reason = which
`*-state.md` exists, `monitor-common.sh:154-192`); portable timeout wrapper
(`scripts/portable-timeout.sh`).

## 3. Top 3 (highest value, fak most plausibly lacks)

- **TOP-1 — Independent cross-model reviewer as a gating phase, wired into the closed refusal
  vocabulary (#1).** A ralph/dispatch-loop phase that, before a round advances, runs a *different*
  model's review pinned to the run's captured start SHA, that reviewer's own hooks disabled to
  stop Stop-recursion, refusing to advance while any `[P0-9]` marker appears — severities mapped
  into `dos_refuse_reasons` (e.g. `INDEPENDENT_REVIEW_P0`). fak's whole trust model is "the kernel
  doesn't believe the agents," yet `dos_review` only re-projects the author's own commit-audit
  rung; the missing piece (different model, fixed base SHA, self-hooks off, severity token in the
  first 10 chars) is cheap and drops straight into the existing closed vocabulary.
- **TOP-2 — Memory pair: write-time evidence-bound memory-delta gate + cheap-model per-task
  selector (#3 + #4).** A Stop/commit-time gate refusing to close a round unless it emits a
  structured memory delta whose cited IDs resolve in fak's notes store (reuse `dos_recall` as the
  cross-check; STALE_CLAIM-family refusal otherwise; empty-KB ratchet), paired with a haiku-tier
  selector returning only task-relevant note IDs pre-dispatch. fak proves memory on *read* but has
  no *write-time* proof a round banked a durable lesson, and injects MEMORY.md wholesale (the
  oversized index here is the evidence). Together: an enforced, retrieved KB instead of an
  accumulating dump.
- **TOP-3 — Required progress-verdict token driving a graduated stall→replan→stop breaker, with
  missing-verdict-is-a-block (#2), backed by the corroborating-witness discipline (#5).** Demand a
  closed-vocab `ADVANCED/STALLED/REGRESSED` verdict each round; absence = `PROGRESS_VERDICT_MISSING`;
  persist a stall counter; soft `dos_replan` at ≥2, hard break at ≥3
  (`loop-codex-stop-hook.sh:1837-1848`); reinforce with the "any agent-editable loop-state field
  must be backed by an independent artifact" rule (`:1902-1938`). `dos_status` detects STALLED
  syntactically but lacks the independent-model semantic gate and the double-threshold ladder.

## 4. Anti-patterns — what fak should NOT copy

- **A. Don't restructure fak's loop around "the Stop-hook IS the loop," nor adopt the 7200s Stop
  timeout (`hooks/hooks.json:63-73`).** That is forced by the CC-plugin substrate — humanize has
  no driver process, so the hook must both prevent exit and carry the next prompt. fak already has
  real dispatch/ralph/superloop drivers. Borrow the *semantics* (block-as-continuation: the
  refusal payload IS the next-step prompt) into fak's refusal emitter, not the *mechanism*.
- **B. Don't downgrade completion verification to text-marker scraping.** humanize's "done" oracle
  is grep-for-`COMPLETE`-on-its-own-line + `[P0-9]` in the last 50 lines
  (`loop-common.sh:759-766`) — *weaker* than fak's git-evidence `dos_verify` ("source=none can't
  masquerade as strong"). Keep `dos_verify` as the completion witness; use the reviewer's tokens
  as an *additional semantic axis*, never a replacement.
- **C. Don't import humanize's whole apparatus for defending an agent-writable state file.** Its
  control state lives in an agent-adjacent `state.md`, so it must spend enormous effort defending
  it (`loop-bash-validator.sh:321-488`, marker corroboration, manual-toggle detection). fak's
  stronger move: keep loop/lease/journal state in a store the agent has *no tool path to* — then
  most of that defense is unnecessary. Borrow the corroborating-witness *idea* (TOP-3) and the
  write-footprint extractor (#6); do NOT inherit the shell-form denylist treadmill — resolve the
  footprint into `dos_arbitrate` instead of growing a blocklist.
- **D. Don't make the plan-comprehension quiz a hard gate.** humanize keeps it advisory and
  degrades to "quiz unavailable, continuing" (`start-rlcr-loop.md:86`); a hard comprehension gate
  on a dispatcher is operator-hostile and itself a livelock source. If fak adopts #8, keep
  `PLAN_UNCOMPREHENDED` warn-class.
- **E. Don't vendor `statusline.sh`** — explicitly AS-IS / reference-only (`:8-14`) and its
  zombie-loop resolver (`:67-155`) duplicates `find_active_loop` elsewhere. Borrow the *concept*
  (session-scoped verdict field; terminal-reason-as-filename for dumb-reader observability).
- **F. Minor:** humanize keeps its BitLesson KB in-tree under `.humanize/` and must then exclude
  it from the git-clean gate (`loop-codex-stop-hook.sh:692-698`). fak's out-of-tree
  logvault/scratchpad discipline is already cleaner — keep memory/artifacts out of the worktree.

## 5. Dedup — likely PRESENT in fak (do NOT file as new)

Closed-vocabulary structured refusals (`dos_refuse_reasons`/`dos_check_reason`);
PreToolUse/PostToolUse guards (mirror `hooks.json:14-51`); git-evidence completion (`dos_verify`,
stronger than grep-COMPLETE); a guard/decision journal (analog of the per-round
`.humanize/rlcr/<ts>/` trail); lane-lease tree-disjointness (`dos_arbitrate`) — the
machine-checked *superior* of humanize's prompt-only "assign strict file ownership boundaries"
plea (`agent-teams-core.md:18`). The genuinely novel borrows are **#1–#6 and #8**; everything
below #9 refines something fak already has.

## 6. Suggested filing shape (when authorized)

Per the study-repo discipline, witness each candidate against fak at its *axis* first, then file
survivors as small independent tickets — never one "adopt humanize" monolith. Strongest
standalone tickets: **#1, #3+#4 (memory pair), #2+#5 (progress/corroboration), #6**; then #7, #8,
#12 as a test-discipline ticket. #9–#14 fold into adjacent guard/release work. Anti-patterns
A–F belong in each ticket's "explicitly out of scope" fence so a future implementer doesn't
re-derive humanize's substrate-forced choices.
