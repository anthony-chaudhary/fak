---
title: "fak loop-tool-map — the right verb is one ask away at every loop stage"
description: "The loop-stage to tool map (#1153): at each stage of the agentic-coding loop (orient -> plan -> act -> verify -> ship -> learn) it names the concrete fak/dos verb to reach for right now, so the kernel's levers get used by default instead of by luck. Queryable from the loop via `fak loop-map`."
---

# loop-tool-map — what tool do I reach for *right now*?

<!-- loop-tool-map - process: this doc is the human view of `fak loop-map`; the data lives in internal/loopmap -->

The agentic-coding loop is **orient -> plan -> act -> verify -> ship -> learn**
(the six stages owned by the [loop-index scorecard](loop-index-scorecard.md)). At each
stage there is a *right* verb to reach for, but that knowledge is tribal: it is scattered
across `AGENTS.md`, agent memory, and the kernel's tool surface, so a mid-tier agent
guesses and reaches for the wrong tool, or none. This map makes the right verb **one ask
away** — query it from inside the loop with `fak loop-map`.

> This is lever **#1153** of the 10x dev-experience epic ([#1148](../../)). Where
> `fak loop-index-scorecard` *measures* whether the loop is getting faster, this makes the
> right verb cheap to *find* so every downstream lever gets used **by default**.

## Query it from the loop

```bash
fak loop-map                      # the whole map, grouped by loop stage
fak loop-map --stage verify       # just the rows for one stage
fak loop-map --ask "about to claim done"   # match a free-text "what now?" to the verb
fak loop-map --nudge              # the "did you verify?" verify/ship-boundary reminder
fak loop-map --json               # the machine-readable map (for tooling)
```

## The map

| Stage | When you are... | Reach for | Why |
|---|---|---|---|
| **orient** | about to trust a recalled memory | `fak recall` | re-verify recalled memory against the tree at read time before you trust it; the kernel-side equivalent is the `dos_recall` MCP tool. |
| **plan** | about to fan out N agents / parallel work | `dos arbitrate` | prove the lanes are disjoint **before** launching, so workers do not collide or needlessly serialize — honor a REFUSE. |
| **act** | about to run a tool call | `fak guard` | the kernel adjudicates each call, repairs a malformed one in place, and serves an identical repeat locally instead of spending a turn. |
| **verify** | about to claim the work is done | `dos verify` | confirm the claim landed from git evidence, not a self-report; pair with `fak commit --preview` to pre-check lane + ship-stamp. |
| **ship** | the tree is green, about to commit | `fak commit` | commit-by-path under the green gate (refuses OFF_TRUNK / PATHSPEC_RACE); then `dos commit-audit <sha>` grades it diff-witnessed. |
| **learn** | the session is over | `fak sessions` | ingest + score this host's transcripts so the session -> outcome loop learns from the run instead of forgetting it. |

The three anchors the issue calls out by name: *claim-done -> verify*, *fan-out ->
arbitrate*, *trust-a-memory -> recall*.

## No hand-maintained drift

The map is **data** (`internal/loopmap`), not prose. Every `fak <verb>` it names is bound
to fak's real verb registry: `TestNoFakVerbDrift` re-parses `cmd/fak/main.go` and reds the
gate if the map points at a verb the binary does not have. The scripted-scenario witness
(`TestScriptedScenario`) proves a cold agent, given only the map, reaches for the correct
verb at each stage. So this doc cannot rot away from the tool surface it describes.

> A note on verb honesty: the issue text mentions `dos answer` and `dos recall`, but the
> live `dos` CLI has neither (it has `answer-shape`, not `answer`; `review`/`resume`, not
> `recall`). The recall anchor therefore points at the real, drift-checked `fak recall`
> verb and names the `dos_recall` MCP tool as the kernel-side equivalent, rather than
> sending an agent at a verb that does not exist.

## Follow-on

This lands the fak-side affordance (the queryable map + the `fak loop-map` verb + the
verify nudge). Registering these rows into the `dos` answer corpus so the same map is one
ask away from the `dos` side as well is the labeled next step — it depends on the external
`dos` tool's corpus, not this repo.
