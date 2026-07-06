package knownbad

import "testing"

// TestWithRevokeStampsTerminalRow is the core arm of the W8 (#2720) unwitnessed release
// valve: WithRevoke flips an open row to the terminal "revoked" status, stamps the operator,
// instant, and prose reason, and — crucially — makes the row NOT Live so it retracts the
// signature through the SAME supersede fold a resolve uses. Unlike WithResolve it carries no
// witness (a revoke is a judgement, not a proof); it preserves the signature/tree/reason and
// the claim bookkeeping so the operator card can still name the fixer.
func TestWithRevokeStampsTerminalRow(t *testing.T) {
	const now = int64(1_700_000_000)
	open := NewRecord("build", []string{"internal/foo/**"}, "n", "finder", "", now, 0)
	if !open.Live(now) {
		t.Fatalf("precondition: a fresh open row must be live")
	}

	rev := open.WithRevoke("operator", now+5, "  flaky, not a shared bug  ")
	if rev.Status != StatusRevoked {
		t.Errorf("WithRevoke status = %q, want %q", rev.Status, StatusRevoked)
	}
	if !rev.Revoked() {
		t.Errorf("Revoked() = false for a revoked row: %+v", rev)
	}
	if rev.RevokedBy != "operator" || rev.RevokedAtUnix != now+5 {
		t.Errorf("revoke stamp wrong: by=%q at=%d", rev.RevokedBy, rev.RevokedAtUnix)
	}
	if rev.RevokeReason != "flaky, not a shared bug" {
		t.Errorf("RevokeReason not trimmed/stamped: %q", rev.RevokeReason)
	}
	if rev.Witness != "" {
		t.Errorf("a revoke must carry NO witness (it is a judgement, not a proof): %q", rev.Witness)
	}
	// The retraction is the point: a revoked row is never live, so it stops matching.
	if rev.Live(now + 5) {
		t.Errorf("a revoked row must NOT be live — it retracts the signature")
	}
	// Identity preserved so the supersede fold keeps grouping it with the open row.
	if rev.Signature != open.Signature || rev.ReasonClass != open.ReasonClass {
		t.Errorf("WithRevoke changed the signature identity: %+v vs %+v", rev, open)
	}
}

// TestLatestStateClassifiesRetraction pins the shell's expired/revoked/resolved-vs-never-seen
// distinction (the KNOWN_BAD_EXPIRED_OR_REVOKED gate): LatestState folds to the LATEST row and
// reports the terse state a claim/resolve/revoke uses to choose between a structured refuse
// (seen but retracted) and a plain usage error (never recorded).
func TestLatestStateClassifiesRetraction(t *testing.T) {
	const now = int64(1_700_000_000)
	sig := Signature("build", []string{"internal/foo/**"}, "")

	// Never recorded -> not seen, empty state.
	if rec, seen, state := LatestState(nil, sig, now); seen || state != "" || rec.Signature != "" {
		t.Errorf("empty ledger: got seen=%v state=%q, want unseen", seen, state)
	}

	open := NewRecord("build", []string{"internal/foo/**"}, "", "finder", "", now, 100)

	// Live (open, unexpired).
	if _, seen, state := LatestState([]Record{open}, sig, now+10); !seen || state != "live" {
		t.Errorf("open+unexpired: got seen=%v state=%q, want live", seen, state)
	}
	// Expired: same open row, clock past its TTL.
	if _, seen, state := LatestState([]Record{open}, sig, now+1000); !seen || state != "expired" {
		t.Errorf("open+past-ttl: got seen=%v state=%q, want expired", seen, state)
	}
	// Revoked: a superseding revoked row is the latest.
	revoked := open.WithRevoke("op", now+5, "wrong tree")
	if _, seen, state := LatestState([]Record{open, revoked}, sig, now+10); !seen || state != "revoked" {
		t.Errorf("revoked latest: got seen=%v state=%q, want revoked", seen, state)
	}
	// Resolved: a superseding resolved row is the latest.
	resolved := open.WithResolve("fixer", now+5, "tests")
	if _, seen, state := LatestState([]Record{open, resolved}, sig, now+10); !seen || state != "resolved" {
		t.Errorf("resolved latest: got seen=%v state=%q, want resolved", seen, state)
	}
	// Latest-wins: an open row appended AFTER a revoke re-opens the signature (the failure
	// re-fired and re-recorded). LatestState reflects only the LATEST row.
	reopened := NewRecord("build", []string{"internal/foo/**"}, "", "finder", "", now+20, 100)
	if _, seen, state := LatestState([]Record{open, revoked, reopened}, sig, now+25); !seen || state != "live" {
		t.Errorf("revoke then re-record: got seen=%v state=%q, want live", seen, state)
	}
}

// TestDefaultRecordTTLIsBounded guards the W8 self-healing invariant at the core: the default
// record TTL is a positive, bounded window (not the old 0 = forever). A signature stamped with
// the default expires; the guard is that the constant stays a real fuse, so a forgotten
// signature cannot park the fleet indefinitely.
func TestDefaultRecordTTLIsBounded(t *testing.T) {
	if DefaultRecordTTLSeconds <= 0 {
		t.Fatalf("DefaultRecordTTLSeconds = %d, want a positive bounded fuse", DefaultRecordTTLSeconds)
	}
	const now = int64(1_700_000_000)
	rec := NewRecord("build", []string{"internal/foo/**"}, "", "finder", "", now, DefaultRecordTTLSeconds)
	if !rec.Live(now) {
		t.Errorf("a just-recorded default-ttl row must be live")
	}
	if rec.Live(now + DefaultRecordTTLSeconds + 1) {
		t.Errorf("a default-ttl row must expire after its window (%d s)", DefaultRecordTTLSeconds)
	}
}
