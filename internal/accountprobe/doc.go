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
package accountprobe
