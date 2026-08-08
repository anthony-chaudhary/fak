---
date: 2026-08-08
issue: 5592
headline: "Daily lock-aware Git hygiene is ready for the next fak release."
---

# Daily Lock-aware Git Hygiene

The next release includes `fak git daily`, a deterministic maintenance pass for
shared checkouts. It inspects repository health, reports stale Git lock files,
and deliberately leaves every lock in place so an operator can distinguish a
dead writer from a live peer before cleanup. The same command is available to
the registered daily cron path.

Reproduce the shipped self-check without changing repository state:

```powershell
fak git daily --selfcheck
```

The command returns a structured verdict and exits non-zero when maintenance
needs operator action. The implementation and behavior witnesses shipped in
`internal/gitdaily/` and `cmd/fak/git_daily_test.go` with the working spine for
issue #5577.

Scope: this note drafts the release entry; it does not cut or tag a release.
