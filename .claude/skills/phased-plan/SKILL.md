---
name: phased-plan
description: Ceremony rules for shipping a phase of a phased plan — when to release, when to emit a handoff prompt, how far to go on type-strengthening, when to fold or split phases, and the hero-exit rule that prevents plans from becoming open-ended. Load when shipping a phase of a docs/*-plan.md (or equivalent) plan.
disable-model-invocation: true
user-invocable: false
allowed-tools: Read, Edit, Bash
---

# Phased-Plan Ceremony

Load this skill when you finish (or are about to finish) a phase of a phased plan. It governs four things: handoff prompts, releases, plan-shape rules (hero-exit, phase-split, hygiene-queue), type-strengthening. "The project's release skill" referenced below is this repo's own `/release`.

**Git authorization.** Loading this skill is the user's explicit authorization to run `git add` and `git commit` for the phase-ship commit and for any Phase-0 baseline snapshot the ceremony spells out. The "never commit unless asked" default does NOT apply here — committing the phase artifact IS the skill's job. The skill does NOT authorize `git push` or `git tag` directly — those happen via the project's release skill (which has its own authorization), or require explicit user confirmation if invoked outside it. Force-push, history rewrites, branch deletion, and `git reset --hard` always require explicit user confirmation. Use targeted pathspecs only — never `git add -A`/`-u`/`.`.

## Handoff prompts — session-boundary only

Emit a copy-paste handoff prompt **only at a session boundary** — when the user signals they are wrapping up, clearing context, or otherwise ending the session ("wrapping up", "/clear next", "done for the day", compaction imminent). Do NOT emit one after every phase completion within a live session; the plan doc + issue tracker already carry continuation context for in-session work. Format when you do emit:

```
--- NEXT-STEP PROMPT (copy into new session) ---
Continue <plan-id> (<doc path>). Just shipped: <one-line summary of completed phase>.
Next phase: <next phase id + title>. Goal: <1-sentence goal>. Key files/refs: <paths>.
Start by reading <doc path> and the listed files, then propose a short implementation plan before editing.
--- END ---
```

≤6 lines, self-contained, includes plan doc path + next phase id.

## Releases — only on shippable code change

Run the project's release skill **only when the phase produced a user-visible behavior change or modifies shipped code paths**. The commit still lands either way — the release ceremony (version bump, changelog entry, tag, push, release artifact) is reserved for code that will be visible to the operator running the tool.

Do NOT release for:

- Docs-only edits to in-flight plans (plan flesh-out, phase re-sequencing, prose tightening)
- Proposal / baseline-snapshot / measurement-checkpoint phases that add no runtime code
- Observation-only / shadow-soak / instrumentation-only phases
- Single-checkbox issue ticks with no code diff
- Memory-file or roadmap-index updates

Do release for: new/changed CLI flag or subcommand, new/changed handler or API endpoint, behavior change in core pipelines, schema changes, bug fixes that affect runtime, new skill or meaningful skill-logic change.

When in doubt, skip the release and batch with the next shippable phase. Version numbers should carry signal — raise the bar, not the cadence.

### Pre-release 3-question gate

Before invoking the release skill, answer:

1. Does the release window contain at least one runtime code-path edit?
2. Could an operator trigger something from CLI/UI and see different output?
3. Is the headline a behavior name (not "plan-doc refresh", "phase X captured", "ceremony", "rollup")?

Three "no"s → docs-only commit on the working branch with no version bump, no tag, no archive, no release entry.

## Hero-exit rule — close the plan when the hero metric is met

Every phased plan has a single 1-hop hero metric in its header. When that metric reaches its target, the **next phase queued is the close-out** — not another sub-phase, not a refactor of an adjacent surface, not "while we're in here."

Before queuing the next phase of a plan whose hero metric you just moved, re-read the metric:

1. **Hero met?** Next phase = plan close-out. In the **same close-out commit**: tombstone the plan-doc, close the umbrella issue, move pending sub-phases to PARK with a "remaining tail" label.
2. **Hero not met?** Continue per the plan's sequencing.

Plans without hero-exit become open-ended consolidation projects that consume slot budget without delivering closure. If work past the hero looks legitimately load-bearing, that means the plan named the wrong hero — fix the metric first (one PR, header only), then continue. Don't ship past a stated hero without auditing it.

## TOMB-at-impl-plus-monitor — soak gates are a lane, not a phase

A plan is tombstone-eligible the moment **implementation ships and a monitor is attached**. "Monitor" = either a registered soak watch or a CI invariant that fires on regression. The 100%-phases criterion conflates engineering work with passive evidence-gathering and produces the "85-95% stuck for weeks" portfolio pattern.

**Rules:**

1. **Last phase is never "soak gate + tomb."** Those are separate kinds of work. Final engineering phase ships code; soak goes to the soak registry; tomb is administrative.
2. **TOMB-with-soak-attached is a recognized terminal state.** Plan exits the active queue, soak continues, memory anchor preserves the decision trail.
3. **A failing soak keeps the plan active.** If a soak's current metric is below its pass-criterion, the plan is *not* tomb-eligible — the risk is real, not theoretical.
4. **Soak pass-criterion shape:** prefer event-count gates over calendar gates when the system runs hot (e.g. `≥5 events` beats `7d` when traffic is bursty); use composite gates (`metric ≥ X for N consecutive periods`) for noise-prone metrics.

## Phase-split test — fold phases that don't produce intermediate evidence

Before splitting work into two phases, ask: **"does phase N+1 need evidence that phase N produces?"**

- **No** (N+1 is just code chained off N's API surface) → fold them. One phase, one ship-stamp.
- **Yes** (N+1 reads telemetry, soak result, or measurement that only exists after N ships) → split is justified.

Splitting on operator-fatigue lines is a major producer of fragmentation. The historical default ("each phase fits one agent session") is wrong when phase N and N+1 are reversible code chains the same agent could ship together.

**Anti-pattern:** observability-before-guard splits ("observe" then "act"). When the change is low-risk, bake telemetry into the implementation phase and let the soak prove it.

## Hygiene-umbrella plans are queues, not roadmaps

Hygiene-umbrella plans (cleanup, deprecation, tech-debt collections) accumulate open-ended cleanup. Convention:

- One substrate phase at the top (baseline) — fine.
- Everything after is a **queue**, not pre-numbered phases. Track `inflight items` + `closed in last 7d` count.
- Drop `% complete` from the plan-doc header (it forces inventing new phase numbers when sibling cleanups surface mid-session).
- Sibling items batch into one dispatch when they chain on the same substrate.

Distinguishing test: if you can answer "does the new sub-task need evidence the prior sub-task produces?" with "no, it's just more cleanup of the same shape" → it's a queue item, not a phase.

## One investigation is one plan — do not split it into a series

A single investigation with several suspected causes is **one plan with phases** (Phase 0 baseline → Phase 1..N each a measured fix), not N sibling plans. Sibling plans are justified only when the causes are already proven independent and each has its own hero metric.

Rule: **no new plan doc until a Phase-0 baseline names a metric that no existing plan can host.** Before opening any new plan, check whether an existing active or hygiene plan can host the work. Prefer a phase/queue-item on an existing plan; open a new doc only when that genuinely fails.

## Phase 0 baselines — register-or-delete invariant

**Optional for low-blast-radius edits.** A phase may skip the Phase 0 baseline freeze when **all three** hold: change is ≤50 LOC, rollback is a single revert, and failure mode is observable post-hoc from existing telemetry. The plan still names the metric and expected direction; it just doesn't pre-freeze a numeric. The measure-then-change axiom is load-bearing for risky changes — wiring substrate, changing dispatch behaviour, touching a hot path — and remains mandatory for those.

When a phase DOES produce a measure-then-change baseline, it **must** land in git the same way every time: same commit, registry row (in whatever baseline-tracking file the project uses), no exceptions. Skipping the registry row is the producer for orphan baselines that survive plan-aborts and pile up.

```bash
# Atomic — one commit, both assets. Never just the dir, never just the row.
git add <baseline-dir>/ <baseline-registry-file>
git diff --cached --name-only      # AUDIT — both paths present
git commit -m "<plan>: <phase> — baseline snapshot (measure-then-change)" -- \
        <baseline-dir>/ <baseline-registry-file>
```

The registry row encodes "this baseline exists, here's what it measures, here's its rebaseline cadence." Without it, future tools can't audit the on-disk dir against ground truth — only the operator's memory ties them.

**If the plan aborts:** keep the baseline dir + registry row in git as historical evidence. Annotate the registry row's notes with the abort reason. Deletion is a separate operator decision after the plan has been tombstoned.

No release ceremony, no version bump — Phase 0 is observation-only by definition (per "Releases — only on shippable code change" above).

## Issue-tracker conventions

If the project uses GitHub/Linear/etc. for umbrella issues:

- Each active phased plan has one umbrella issue. Tombstoned/parked plans do not.
- Pending phases are rendered as `- [ ]` task-list checkboxes inside the issue body so the issue card shows X-of-N progress automatically.
- When you ship a phase: tick its checkbox.
- When a plan completes: close the umbrella issue. If the plan tombstones, link the tombstone in the closing comment.
- When proposing a new plan: create a new umbrella issue, add a link in the plan-doc header.

The project's CLAUDE.md / AGENTS.md should name the issue-tracker URL and label conventions.

## Type-strengthening — payload-only scope

Type the **new payload/struct/enum/config-key the phase itself adds**. Do not scope-creep into surrounding untyped code — that's a separate cleanup, not part of the phase.

- Python: Pydantic model (or dataclass) for any new structured payload crossing a module boundary; `Literal`/`Enum` for closed string sets the phase introduces; return-type annotations on new functions.
- Go: named struct for any new payload crossing a package boundary; typed enum (`type Foo string` + consts) for any new closed string set; concrete error type when callers of new code will branch on it.
- Contract lockstep: if the phase adds a new config key or event-payload field, add/update the matching struct in the same commit.

Skip silently when the new thing is genuinely dynamic (third-party shape) or typing would force a breaking change outside the phase's scope.
