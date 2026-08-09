package main

// leaseref_audit_age_test.go witnesses the audit's AGE rung for TTL-LESS leases — the
// half of `age_threshold_seconds` the row had always advertised and never applied.
//
// WHAT WOULD BREAK WITHOUT IT. A lease taken with `fak leaseref acquire` carries ttl 0
// ("no expiry") by default. leaseref.Record.Expired short-circuits false at ttl<=0, so
// such a record never enters Live's expired partition, Reap (which deletes only what Live
// called expired) can never collect it, and the audit reported it `stale=false,
// TTL_LIVE` no matter how old it got. The lane it names is then refused for the life of
// the repository with nothing anywhere reporting it.
//
// THE TRAP THESE TESTS ALSO PIN. The tempting staleness test — "is the holder's process
// gone?" — is wrong here: the acquiring process is a per-invocation CLI child that dies
// almost immediately, so a dead pid is true of a perfectly HEALTHY lease and a pid probe
// would call every live lane in the fleet abandoned.
// TestLeaserefAuditRowIgnoresHolderPidForStaleness is the control that fixes the rule to
// age: a young lease whose holder string names a long-dead pid stays not-stale.
//
// Every assertion runs against literal records at a FIXED now, or against a t.TempDir()
// git repo — never the live refs/fak/locks namespace, and nothing here deletes a lease.

import (
	"bytes"
	"context"
	"encoding/json"
	"os/exec"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/leaseref"
)

// auditNow is the pinned read instant every row assertion below is taken at. A literal
// clock is what keeps the age comparison from being a time-of-day flake.
var auditNow = time.Unix(1_800_000_000, 0)

// auditRow renders one record at auditNow through the production row builder, applying
// the same TTL verdict runLeaserefAudit would (Record.Expired) so the test cannot drift
// from the caller's partition.
func auditRow(r leaseref.Record) map[string]any {
	return leaserefAuditLeaseRow(r, auditNow, r.Expired(auditNow))
}

// noTTLLease is a lease in exactly the shape the wedge arrives in: no TTL at all, so it
// can never expire, acquired agoS seconds before the pinned now.
func noTTLLease(id string, agoS int64) leaseref.Record {
	return leaseref.Record{
		ID:         id,
		TreeGlobs:  []string{"internal/" + id + "/**"},
		Holder:     "nodeA/sess-" + id,
		AcquiredAt: auditNow.Unix() - agoS,
		TTLSeconds: 0,
	}
}

// TestLeaserefAuditRowNoTTLLeasePastFloorIsStaleByAge is the headline witness: a lease
// that cannot expire, held far past the age floor, is reported stale with the age reason,
// and the threshold it was judged against is the AGE floor rather than the 0 TTL that
// made the old comparison unfireable.
func TestLeaserefAuditRowNoTTLLeasePastFloorIsStaleByAge(t *testing.T) {
	row := auditRow(noTTLLease("ghost", 9*24*60*60)) // nine days, the observed wedge shape

	if row["ttl_seconds"] != int64(0) {
		t.Fatalf("ttl_seconds = %v, want 0 — the witness must exercise the no-expiry record", row["ttl_seconds"])
	}
	if row["stale"] != true {
		t.Fatalf("stale = %v, want true: a TTL-less lease nine days old is abandoned", row["stale"])
	}
	if row["reason"] != leaserefReasonNoTTLStale {
		t.Fatalf("reason = %v, want %q", row["reason"], leaserefReasonNoTTLStale)
	}
	if row["age_threshold_seconds"] != leaserefNoTTLStaleAgeS {
		t.Fatalf("age_threshold_seconds = %v, want the age floor %d — echoing the 0 TTL is the defect",
			row["age_threshold_seconds"], leaserefNoTTLStaleAgeS)
	}
	if got, want := row["age_seconds"], int64(9*24*60*60); got != want {
		t.Fatalf("age_seconds = %v, want %d", got, want)
	}
	ev, _ := row["evidence"].(string)
	if !strings.Contains(ev, "age_seconds=777600") || !strings.Contains(ev, "age_threshold_seconds=86400") {
		t.Fatalf("evidence = %q, want the age comparison spelled out", ev)
	}
	if strings.Contains(strings.ToLower(ev), "pid") && !strings.Contains(ev, "never by the holder's pid") {
		t.Fatalf("evidence = %q, must not rest a staleness verdict on a pid", ev)
	}
}

// TestLeaserefAuditRowYoungNoTTLLeaseIsNotStale is the false-reclaim guard, and it is the
// test that stops the rung from degenerating into "every TTL-less lease is a ghost". A
// lease taken a minute ago has no TTL either, and it is very much alive.
func TestLeaserefAuditRowYoungNoTTLLeaseIsNotStale(t *testing.T) {
	row := auditRow(noTTLLease("fresh", 60))

	if row["stale"] != false {
		t.Fatalf("stale = %v, want false: a lease acquired 60s ago is live", row["stale"])
	}
	if row["reason"] != leaserefReasonNoTTLYoung {
		t.Fatalf("reason = %v, want %q", row["reason"], leaserefReasonNoTTLYoung)
	}
	if row["age_threshold_seconds"] != leaserefNoTTLStaleAgeS {
		t.Fatalf("age_threshold_seconds = %v, want the age floor %d even when not stale — the reader must see what the age was compared to",
			row["age_threshold_seconds"], leaserefNoTTLStaleAgeS)
	}

	// The boundary itself: one second short of the floor is still live; the floor is
	// reached exactly at the floor. An off-by-one here is a lane stolen from a live peer.
	if r := auditRow(noTTLLease("edge-under", leaserefNoTTLStaleAgeS-1)); r["stale"] != false {
		t.Fatalf("a lease one second under the floor reported stale=%v, want false", r["stale"])
	}
	if r := auditRow(noTTLLease("edge-at", leaserefNoTTLStaleAgeS)); r["stale"] != true {
		t.Fatalf("a lease exactly at the floor reported stale=%v, want true", r["stale"])
	}
}

// TestLeaserefAuditRowIgnoresHolderPidForStaleness pins the rule the whole design turns
// on. The holder string names a process that is long gone — the ordinary state of a
// healthy lease, because the acquirer is a per-invocation CLI child. Judged by that pid
// this lease is abandoned; judged by age it is thirty seconds old and must be untouchable.
func TestLeaserefAuditRowIgnoresHolderPidForStaleness(t *testing.T) {
	rec := noTTLLease("pid-trap", 30)
	rec.Holder = "nodeA:pid=999999" // a pid that has certainly exited

	row := auditRow(rec)
	if row["stale"] != false || row["reason"] != leaserefReasonNoTTLYoung {
		t.Fatalf("stale=%v reason=%v, want a fresh lease to survive a dead holder pid — staleness must never consult a pid",
			row["stale"], row["reason"])
	}
	// And the converse half of the same rule: a lease is stale on AGE alone, with no pid
	// evidence available at all. leaseref.Record carries no pid field, and that is the
	// structural reason the trap cannot be fallen into here.
	old := noTTLLease("pid-trap-old", 30*24*60*60)
	old.Holder = "nodeA:pid=999999"
	if r := auditRow(old); r["stale"] != true || r["reason"] != leaserefReasonNoTTLStale {
		t.Fatalf("stale=%v reason=%v, want a month-old TTL-less lease stale by age", r["stale"], r["reason"])
	}
}

// TestLeaserefAuditRowUndatableNoTTLLeaseFailsClosed: a record with neither an acquired
// nor a renewed stamp has no age to compare, so it must NOT be called abandoned on the
// strength of an absence — the same fail-closed posture the session classifier takes
// toward a missing descriptor.
func TestLeaserefAuditRowUndatableNoTTLLeaseFailsClosed(t *testing.T) {
	row := auditRow(leaseref.Record{ID: "unstamped", Holder: "nodeA/sess-x"})

	if row["stale"] != false || row["reason"] != leaserefReasonNoTTLYoung {
		t.Fatalf("stale=%v reason=%v, want not-stale: an unknown age is not evidence of death",
			row["stale"], row["reason"])
	}
	if ev, _ := row["evidence"].(string); !strings.Contains(ev, "fails closed") {
		t.Fatalf("evidence = %q, want it to name the fail-closed rule", ev)
	}
}

// TestLeaserefAuditRowTTLLeasesAreUnchanged is the no-regression control: every lease
// that DOES carry a TTL keeps the pre-existing verdict, reason and threshold. The age
// rung must be unreachable for them — otherwise a short-TTL lease held legitimately for
// more than a day would flip class.
func TestLeaserefAuditRowTTLLeasesAreUnchanged(t *testing.T) {
	live := auditRow(leaseref.Record{ID: "ttl-live", Holder: "A", AcquiredAt: auditNow.Unix() - 10, TTLSeconds: 3600})
	if live["stale"] != false || live["reason"] != leaserefReasonTTLLive || live["age_threshold_seconds"] != int64(3600) {
		t.Fatalf("live TTL row = %+v, want stale=false TTL_LIVE threshold=3600", live)
	}

	// Older than the AGE floor, but its own TTL has not lapsed: the TTL rule owns it and
	// the age rung must not fire.
	longHold := auditRow(leaseref.Record{ID: "ttl-long", Holder: "A", AcquiredAt: auditNow.Unix() - 2*leaserefNoTTLStaleAgeS, TTLSeconds: 30 * 24 * 60 * 60})
	if longHold["stale"] != false || longHold["reason"] != leaserefReasonTTLLive {
		t.Fatalf("long-TTL row = %+v, want stale=false TTL_LIVE — a lease with a TTL is judged by that TTL alone", longHold)
	}

	expired := auditRow(leaseref.Record{ID: "ttl-dead", Holder: "B", AcquiredAt: auditNow.Unix() - 100, TTLSeconds: 10})
	if expired["stale"] != true || expired["reason"] != leaserefReasonTTLExpired || expired["age_threshold_seconds"] != int64(10) {
		t.Fatalf("expired TTL row = %+v, want stale=true TTL_EXPIRED threshold=10", expired)
	}
}

// TestLeaserefAuditReportsAgeStaleGhostsEndToEnd drives the real verb over a t.TempDir()
// git repo and asserts the envelope an operator (and `fak garden`) actually reads: the
// ghost is counted on its own keys, kept OUT of would_reap because the reaper provably
// cannot collect it, the verdict trips, and the reason names the remedy that works.
func TestLeaserefAuditReportsAgeStaleGhostsEndToEnd(t *testing.T) {
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git not on PATH")
	}
	dir := t.TempDir()
	for _, args := range [][]string{
		{"init", "-q"},
		{"config", "user.email", "t@example.test"},
		{"config", "user.name", "t"},
	} {
		c := exec.Command("git", args...)
		c.Dir = dir
		if out, err := c.CombinedOutput(); err != nil {
			t.Fatalf("git %v: %v\n%s", args, err, out)
		}
	}

	store := leaseref.NewInDir(dir)
	ctx := context.Background()
	now := time.Now().Unix()
	// The ghost: no TTL, acquired nine days ago — unreapable and, before this rung,
	// unreported. And a live neighbour with the same absent TTL, taken a minute ago.
	for _, rec := range []leaseref.Record{
		{ID: "ghost-lane", TreeGlobs: []string{"internal/ghost/**"}, Holder: "nodeA/sess-1", AcquiredAt: now - 9*24*60*60},
		{ID: "fresh-lane", TreeGlobs: []string{"internal/fresh/**"}, Holder: "nodeB/sess-2", AcquiredAt: now - 60},
	} {
		if _, err := store.Acquire(ctx, rec); err != nil {
			t.Fatalf("Acquire(%s): %v", rec.ID, err)
		}
	}

	var out, errb bytes.Buffer
	if code := runLeaseref(&out, &errb, []string{"audit", "--dir", dir}); code != 0 {
		t.Fatalf("leaseref audit exit=%d stderr=%q", code, errb.String())
	}
	var audit struct {
		Verdict      string           `json:"verdict"`
		Reason       string           `json:"reason"`
		LiveCount    int              `json:"live_count"`
		ExpiredCount int              `json:"expired_count"`
		WouldReap    []map[string]any `json:"would_reap"`
		AgeStaleIDs  []string         `json:"age_stale_ids"`
		AgeStale     []map[string]any `json:"age_stale"`
		AgeStaleN    int              `json:"age_stale_count"`
		FloorS       int64            `json:"no_ttl_stale_age_seconds"`
	}
	if err := json.Unmarshal(out.Bytes(), &audit); err != nil {
		t.Fatalf("audit JSON unmarshal: %v\nout=%s", err, out.String())
	}

	// Neither lease can expire, so the TTL partition sees two live and nothing reapable —
	// the load-bearing detail: without the age rung this envelope is entirely green.
	if audit.LiveCount != 2 || audit.ExpiredCount != 0 || len(audit.WouldReap) != 0 {
		t.Fatalf("TTL partition = %d live / %d expired / %d would_reap, want 2/0/0",
			audit.LiveCount, audit.ExpiredCount, len(audit.WouldReap))
	}
	if audit.AgeStaleN != 1 || len(audit.AgeStaleIDs) != 1 || audit.AgeStaleIDs[0] != "ghost-lane" {
		t.Fatalf("age_stale = %d %v, want exactly [ghost-lane] — the fresh lease must not be swept up",
			audit.AgeStaleN, audit.AgeStaleIDs)
	}
	if len(audit.AgeStale) != 1 || audit.AgeStale[0]["reason"] != leaserefReasonNoTTLStale {
		t.Fatalf("age_stale rows = %+v, want one %s row", audit.AgeStale, leaserefReasonNoTTLStale)
	}
	if audit.FloorS != leaserefNoTTLStaleAgeS {
		t.Fatalf("no_ttl_stale_age_seconds = %d, want %d", audit.FloorS, leaserefNoTTLStaleAgeS)
	}
	if audit.Verdict != "ACTION" {
		t.Fatalf("verdict = %q, want ACTION: a permanently-unreapable lane is not an OK state", audit.Verdict)
	}
	if !strings.Contains(audit.Reason, "ghost-lane") || !strings.Contains(audit.Reason, "fak leaseref release") {
		t.Fatalf("reason = %q, want it to name the ghost and the remedy that works", audit.Reason)
	}
	if !strings.Contains(audit.Reason, "`reap` cannot collect these") {
		t.Fatalf("reason = %q, must not imply the reaper will collect a TTL-less lease", audit.Reason)
	}
}
