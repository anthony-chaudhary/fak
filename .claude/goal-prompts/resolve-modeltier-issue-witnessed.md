You are a detached headless worker in a bulk super-loop fan-out for the model-tier
batch (modeltier C1-C10, issues #3038-#3047). Claim ONE READY menu item, resolve
it, ship it WITNESSED, then stop. Siblings (distinct accounts) run beside you in
the SAME tree — lane discipline is load-bearing.

## The menu (claim the FIRST item that is READY and whose lane lease is free)

READY = every prereq issue is CLOSED
(`gh issue view <P> --repo anthony-chaudhary/fak --json state -q .state`==CLOSED).
If your first pick is blocked or its lane is leased, try the next; if NONE is
ready, stop cleanly — that is a correct, honest run, do not force one.

- C1 #3038 — capability-score registry. Lane `modelroute`. NEW pkg
  internal/modelscore/** (don't touch the dirty route.go/fleetaccounts). Prereq: none.
- C2 #3039 — bench ingestors (Terminal-Bench/SWE/FrontierSWE), Go-only. Lane
  `benchcatalog`. internal/benchcatalog/**, cmd/fak/frontierswe.go; writes rows into
  internal/modelscore. Prereq: #3038.
- C3 #3040 — score-vector -> T0/T1/T2 policy. Lane `modelroute`. NEW file in
  internal/modelroute/** (no god-file). Prereq: #3038.
- C4 #3041 — issue-contract tier tags. Lane `issuecontract`.
  internal/issuecontract/**, cmd/fak/issue_contract.go (own hunks; priority/P1 !=
  tier/T1). Prereq: none.
- C5 #3042 — dispatch cheapest tier + over-tier waste. Lane `dispatchtick`.
  internal/dispatchtick/**, cmd/fak/dispatch_*.go (tier hunks only). Prereq: #3041.
- C6 #3043 — benchmark cheap models for super-loop meta work. Lane `superloop`.
  internal/superloop/** (read-only eval). Prereq: none.
- C7 #3044 — risk limits for cheap-model meta orchestration. Lane `policy`. Floor
  tests in internal/policy/** via exported API (internal/adjudicator is core-locked,
  refuses new *_test.go). Prereq: #3040.
- C8 #3045 — observe tier decisions. Lane `dispatchtick`. Extend
  internal/dispatchtick + tools/dispatch_status.py + internal/gateway/metrics.go
  (own hunks; route.go/metrics.go dirty). Prereq: #3040, #3041, #3042.
- C9 #3046 — calibrate tier policy from outcomes. Lane `issuecost`.
  internal/issuecost/**, internal/sessionaudit/** (calibration hunks). Prereq:
  #3040, #3045.
- C10 #3047 — shadow/canary rollout. Lane `dispatchtick`. internal/dispatchtick +
  internal/superloop (canary stays read-only T2, no default-on routing). Prereq:
  #3041, #3042, #3045.

## The one loop

1. Take YOUR lane: `dos arbitrate --workspace . --lane <lane>`. A REFUSE = a sibling
   holds it; claim the next ready item or stop. NEVER --force.
2. Read before writing: `gh issue view <N>` + the named tree. New tooling is Go
   (pure logic in internal/<leaf>, thin shell in cmd/fak).
3. Reproduce first: a test that fails before, passes after, in the SAME commit. If
   you cannot capture it, report `not yet` — do not claim a fix.
4. Ship on the trunk by explicit path. Green first: make ci (Windows: ./test.ps1
   under WSL). Then `fak commit --preview` then `fak commit --path ... -m "<subject>
   (fak <leaf>)"`. Never git add -A. Leaf = the file-tree package, NOT the routed lane.
5. Close by ancestry: `Fixes #<N>` in the commit BODY. Do NOT gh issue close.
   Verify: `dos commit-audit --json`.
6. Leave the tree clean, release the lease, stop. One witnessed leaf = a complete
   run — do not spin.

## Hard boundaries (enforced below you)

- A launch is not a ship. Only a witnessed commit on the trunk resolves an issue.
- Do not widen your diff into a sibling's tree; commit only your item's hunks.
- Out-of-scope findings: file an issue, do not absorb them.
- Never publish a machine-absolute path, hostname, or personal id (PUBLIC_LEAK /
  FILE_ADMISSION refuse it).
- On a guard refusal (OFF_TRUNK, COLLISION_RISK, CORE_SELF_MODIFY, MERGE_IN_PROGRESS):
  recover per the AGENTS.md table or STOP; do not route around it.

Do NOT end by narrating leftover work: any remaining or out-of-scope follow-up
you'd otherwise list as "two more things" MUST be filed as an open gh issue first
(dedupe → done-condition → leak-check → label) — a named-but-unfiled follow-up is
silently-deferred work this repo forbids.

Report faithfully: the C-item + issue number (plus any follow-up issue numbers),
the witnessing commit SHA (or `not yet` + missing witness), and whether the tree
was left clean.
