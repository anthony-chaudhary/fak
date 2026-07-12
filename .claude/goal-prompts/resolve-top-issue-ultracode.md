ultracode — use the Workflow tool for exhaustive, adversarially-verified multi-agent
orchestration on the single leaf you take. You are running Opus 4.8. Optimize for the
most correct, thorough result; token cost is not a constraint.

You are a detached, unattended, FAK-GUARDED headless worker in a bulk fan-out. Other
workers (distinct accounts) run beside you in the SAME working tree, so lane discipline
is load-bearing. FIRST read the full witnessed operating spec and follow it EXACTLY:

    .claude/goal-prompts/resolve-top-issue-witnessed.md   (read this file now)

That spec is binding. In one line: take ONE tree-disjoint lane via `dos arbitrate
--workspace .` (honor a REFUSE, NEVER --force); pick the top-ranked ready P0/P1 leaf
routed to YOUR lane (`python tools/issue_lane_router.py --view p0-p1 --json`, fall
through to `ready-leaves`) that no sibling is already on; reproduce-before-fix (the repro
lands in the SAME commit); stay on `main`; go green (`make ci`); ship WITNESSED with
`fak commit --path <p> -m "<subject> (fak <leaf>)"` (preview first); close by putting
`Fixes #<N>` in the commit BODY (never `gh issue close` off a self-report); leave the
tree clean; then STOP.

ULTRACODE overlay for your one leaf:
- Decompose the leaf; run parallel finder + verifier subagents via the Workflow tool.
- Adversarially verify the fix with independent skeptics BEFORE you commit.
- A launch is not a ship — only a witnessed commit on the trunk resolves the issue.
- Do NOT widen your lane's diff to absorb out-of-scope findings; file them as issues.
- Never publish a machine-absolute path, hostname, or personal identifier (PUBLIC_LEAK).
- If a guard refuses you (OFF_TRUNK / COLLISION_RISK / STALE_BASE_DELETION /
  MERGE_IN_PROGRESS): reconcile in place or STOP. Do not route around it.

Do NOT end by narrating leftover work. Any remaining or out-of-scope follow-up you
would otherwise list as "two more things" at the end MUST be filed as an open gh
issue first (dedupe → done-condition → leak-check → label) — a named-but-unfiled
follow-up is silently-deferred work this repo forbids. Self-check before you stop
with `fak headless-lint --leftovers --issues-filed <N>`: the fold (#3670) refuses
a summary that narrates leftovers while zero issues were filed, and passes once
each is filed (or with `--override` for genuinely nothing left).

Report faithfully: the issue number, the witnessing commit SHA (or `not yet` + the
missing witness), the issue numbers of any follow-ups you filed, and whether the
tree was left clean.
