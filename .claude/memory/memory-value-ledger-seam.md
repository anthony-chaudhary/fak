---
name: memory-value-ledger-seam
description: "Where the memory-value P&L lives and how recall events reach it — the seam every loop turn feeds"
metadata:
  type: project
---

`fak memory recall` is the loop-turn orientation verb AND the memory-value
event feed: each default-store recall that witnesses events appends one
fak-memory-value-ledger/1 row to docs/nightrun/memory-value.jsonl (fresh =
claim-verified renders, withheld_stale = stale claims refused before
injection). The seam lives in cmd/fak/memory_recall.go; the unbounded
frontier/pressure/debt fold that consumes the ledger lives in
internal/memvaluescore/score.go. An explicit --store never feeds this repo's
ledger unless --ledger names a path, so fixtures and ad-hoc stores cannot
inflate the P&L. Run recall at turn start (intent = the turn's task) — that is
what moves the frontier; store size never does.
