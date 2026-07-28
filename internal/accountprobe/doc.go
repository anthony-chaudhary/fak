// Package accountprobe reads the active account-probe ledger (probe_ledger.jsonl)
// that tools/account_probe.py writes — one JSON line per probe, append-ordered, with a
// closed status vocabulary (OK / AUTH / ACCESS / CREDIT / LIMIT / APIERR / TRANSPORT).
//
// It is the Go port of account_probe's ledger-READ surface (probe_ledger_path,
// last_probe_by_account, recent_probe_age_min): the piece the fleet roster needs to
// fold a fresh active probe back over a stale carried block. The probe WRITER (the
// subprocess that launches `claude -p "say pong"` and classifies the result) stays in
// Python; this package only reads what it recorded, so a fresh OK probe can override a
// carried limit and a fresh LIMIT/AUTH can set one, with a freshness gate so a stale
// OK cannot mask a real current limit.
//
// It also owns the question of WHICH registry dir a host means (regdir.go). One host can end
// up running two: the per-user Fleet registry the prober writes its ledger under, and a
// cwd-relative tools/_registry a fak process started from the clone root maintains beside
// it. Only the first can derive a block, because only it has a ledger — so resolving to the
// second silently reports a fleet with zero blocked seats (#5390). ResolveRegDir picks by
// AUTHORITY (a declared dir, then a ledger-bearing dir, then any registry state) and never
// by modification time, grades the chosen dir's block-derivability so "cannot tell" stays
// distinguishable from "nothing blocked", and reports a fork rather than letting a second
// registry exist silently.
package accountprobe
