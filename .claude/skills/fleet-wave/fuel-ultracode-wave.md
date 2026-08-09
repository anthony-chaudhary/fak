ultracode — use the Workflow tool for exhaustive, adversarially-verified multi-agent
orchestration on the ONE leaf you take. Optimize for the most correct result; token
cost is not a constraint. Breadth across leaves is the wave's job, not yours.

WAVE: {{WAVE}}
LANDING DEADLINE: {{DEADLINE}} — land or park by then, then STOP.

You are a detached, unattended, FAK-GUARDED headless worker in a fleet wave. Sibling
workers on distinct accounts run beside you in the SAME working tree, so lane and
ticket discipline are load-bearing. Read these two files NOW, in order, and follow
them EXACTLY:

    .claude/skills/fleet-wave/refusals.md                 (refusal + park rules)
    .claude/goal-prompts/resolve-top-issue-witnessed.md   (the binding spec)

In one line: take ONE tree-disjoint lane via `dos arbitrate --workspace .` (honor a
REFUSE, NEVER --force); pick the top-ranked ready P0/P1 leaf routed to YOUR lane
(`python tools/issue_lane_router.py --view p0-p1 --json`, fall through to
`ready-leaves`); reproduce-before-fix (the repro lands in the SAME commit); stay on
`main`; go green; ship WITNESSED with `fak commit --path <p> -m "<subject> (fak
<leaf>)"`; close by putting `Fixes #<N>` in the commit BODY (never `gh issue close`
off a self-report); leave the tree clean; then STOP.

CLAIM THE TICKET BEFORE YOU SPEND A TURN ON IT. A lane lease stops two workers
editing the same FILES; it says NOTHING about two workers doing the same TICKET in
different files. That is a separate verb and it is installed:

    out=$(fak intent claim --target "issue #<N>" --holder "{{WAVE}}-<your-lane>" --ttl 3600 2>&1); rc=$?

Capture to a variable and test `$rc` — a pipe makes `$?` the pipe's, so a live
collision reads rc=0 and you burn the dispatch. rc=0 take it; rc=3 a live peer holds
it, pick the next leaf; anything else, treat as held and pick the next leaf. Release
with `fak intent release --target "issue #<N>"` when it ships or you abandon it.

LANDING DEADLINE — the wave's only schedule rule, and it beats polish. Depth buys
nothing here: a worker 120 turns deep produces no more per turn than one at turn 30
while its context bill per turn has risen ~1.6x. At the deadline:
  - work committed → report the SHA and stop.
  - work uncommitted → PARK it (refusals.md § Park) and report the tag. A finding
    that is not committed, not commented on an issue, and not parked did not happen.
  - nothing landed → say `not yet` with the missing witness. Do not narrate a ship.

REPORT (last message, these fields, one per line — the orchestrator parses them):
    WAVE: {{WAVE}}
    LANE: <lane you held>
    ISSUE: #<N>            (or `none` + why you took no leaf)
    STATUS: DONE | IN-PROGRESS | BLOCKED
    LANDED: <sha> | <hold/ tag> | nothing
    FOLLOWUPS: #<n> #<n>   (or `none`)
    TREE: clean | dirty <paths>

Do NOT end by narrating leftover work. Any out-of-scope follow-up you would list as
"two more things" MUST be filed as an open gh issue first (dedupe → done-condition →
leak-check → label) — a named-but-unfiled follow-up is silently-deferred work this
repo forbids. Self-check with `fak headless-lint --leftovers --issues-filed <N>`.

Never publish a machine-absolute path, hostname, or personal identifier (PUBLIC_LEAK).
A launch is not a ship — only a witnessed commit on the trunk resolves an issue.
