package waiting

import (
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/loopmgr"
)

// ev is a minimal loop-event builder for fixture rows. Fold reads only the
// fields set here (it does not validate the hash chain — that is loopmgr.Load's
// job), mirroring how operatortouches/loopscore build their fold fixtures.
func ev(kind loopmgr.EventKind, loop, run, reason string, ts time.Time, seq uint64) loopmgr.Event {
	return loopmgr.Event{
		Schema:     loopmgr.SchemaEvent,
		Seq:        seq,
		TSUnixNano: ts.UnixNano(),
		LoopID:     loop,
		RunID:      run,
		Kind:       kind,
		Reason:     reason,
	}
}

func TestFoldBlockedNotifyInFlightBecomesRowWithAgeHeldSeatAndDeadline(t *testing.T) {
	asOf := time.Unix(0, 0).UTC().Add(30 * time.Minute) // 30 min after the notify
	events := []loopmgr.Event{
		ev(loopmgr.EventStart, "dispatch/iss-1", "run-1", "", time.Unix(0, 0), 1),
		ev(loopmgr.EventNotify, "dispatch/iss-1", "run-1", "ESCALATE_APPROVAL", time.Unix(0, 0), 2),
	}

	q := Fold(events, Params{AsOf: asOf, Deadline: 2 * time.Hour})

	if len(q.Items) != 1 {
		t.Fatalf("Items = %d rows, want 1 (notify in -> row)", len(q.Items))
	}
	it := q.Items[0]
	if it.Status != StatusWaiting {
		t.Fatalf("status = %q, want waiting (under the 2h deadline)", it.Status)
	}
	if it.Held.WorkerSeat != true {
		t.Fatalf("Held.WorkerSeat = false, want true (run started, no end -> it holds a seat)")
	}
	if it.Held.LoopID != "dispatch/iss-1" {
		t.Fatalf("Held.LoopID = %q, want dispatch/iss-1", it.Held.LoopID)
	}
	// age ~30 min; deadline = filed + 2h.
	const wantAge = 30 * 60
	if !approx(it.AgeSeconds, wantAge, 1) {
		t.Fatalf("AgeSeconds = %g, want ~%g", it.AgeSeconds, float64(wantAge))
	}
	wantDeadline := time.Unix(0, 0).Add(2 * time.Hour).UnixNano()
	if it.DeadlineUnixNano != wantDeadline {
		t.Fatalf("DeadlineUnixNano = %d, want %d (filed + 2h)", it.DeadlineUnixNano, wantDeadline)
	}
	if it.PastDeadline {
		t.Fatal("PastDeadline = true, want false at 30 min under a 2h deadline")
	}
	if it.SafeDefault != "" {
		t.Fatalf("SafeDefault = %q, want empty while still waiting (prescribed only on expiry)", it.SafeDefault)
	}
	if q.ByStatus[string(StatusWaiting)] != 1 {
		t.Fatalf("by_status[waiting] = %d, want 1", q.ByStatus[string(StatusWaiting)])
	}
}

func TestFoldRunEndClosesTheRowNotAnOperatorAck(t *testing.T) {
	asOf := time.Unix(0, 0).UTC().Add(30 * time.Minute)
	events := []loopmgr.Event{
		ev(loopmgr.EventStart, "dispatch/iss-2", "run-2", "", time.Unix(0, 0), 1),
		ev(loopmgr.EventNotify, "dispatch/iss-2", "run-2", "NEEDS_HUMAN_REVIEW", time.Unix(0, 0), 2),
		ev(loopmgr.EventEnd, "dispatch/iss-2", "run-2", "", time.Unix(0, 0).Add(5*time.Minute), 3),
	}

	q := Fold(events, Params{AsOf: asOf, Deadline: 2 * time.Hour})

	if len(q.Items) != 0 {
		t.Fatalf("Items = %d rows, want 0 (the run ended -> no longer waiting)", len(q.Items))
	}
	if len(q.Resolved) != 1 {
		t.Fatalf("Resolved = %d rows, want 1 (closed this window)", len(q.Resolved))
	}
	if q.Resolved[0].Status != StatusClosedRunEnded {
		t.Fatalf("resolved status = %q, want closed_run_ended", q.Resolved[0].Status)
	}
	// The honesty fence: closure today is the run end, NOT a proven operator ack.
	if !strings.Contains(q.AckClosureNotYet, "#2271") {
		t.Fatalf("AckClosureNotYet = %q, must name the missing R2 ack row (#2271)", q.AckClosureNotYet)
	}
}

func TestFoldPastDeadlineExpiresAndPrescribesTheSafeDefault(t *testing.T) {
	// Filed 3 hours ago; deadline 2 hours -> the row is 1h past deadline.
	filed := time.Unix(0, 0).UTC()
	asOf := filed.Add(3 * time.Hour)
	events := []loopmgr.Event{
		ev(loopmgr.EventStart, "dispatch/iss-3", "run-3", "", filed, 1),
		ev(loopmgr.EventNotify, "dispatch/iss-3", "run-3", "ESCALATE_APPROVAL", filed, 2),
	}

	q := Fold(events, Params{AsOf: asOf, Deadline: 2 * time.Hour})

	if len(q.Items) != 1 {
		t.Fatalf("Items = %d rows, want 1", len(q.Items))
	}
	it := q.Items[0]
	if it.Status != StatusExpiredDefault {
		t.Fatalf("status = %q, want expired_default (past the 2h deadline)", it.Status)
	}
	if !it.PastDeadline {
		t.Fatal("PastDeadline = false, want true")
	}
	// APPROV -> release_held_resources_and_requeue (SafeDefaultFor).
	if it.SafeDefault != "release_held_resources_and_requeue" {
		t.Fatalf("SafeDefault = %q, want release_held_resources_and_requeue", it.SafeDefault)
	}
	if q.PastDeadline != 1 {
		t.Fatalf("queue PastDeadline = %d, want 1", q.PastDeadline)
	}
	// Expired rows rank first (cost-of-delay).
	if q.Items[0].Status != StatusExpiredDefault {
		t.Fatal("expired row must rank ahead of waiting rows")
	}
}

func TestFoldNonBlockedNotifyIsNotAQueueRow(t *testing.T) {
	// A witnessed-done completion notify carries no human decision — it must NOT
	// become a waiting row (the conservative BlockedReasonHints exclude it).
	asOf := time.Unix(0, 0).UTC().Add(10 * time.Minute)
	events := []loopmgr.Event{
		ev(loopmgr.EventStart, "dispatch/iss-4", "run-4", "", time.Unix(0, 0), 1),
		ev(loopmgr.EventNotify, "dispatch/iss-4", "run-4", "DONE_WITNESSED", time.Unix(0, 0), 2),
	}

	q := Fold(events, Params{AsOf: asOf, Deadline: 2 * time.Hour})

	if len(q.Items) != 0 || len(q.Resolved) != 0 {
		t.Fatalf("Items=%d Resolved=%d, want 0/0 (DONE_WITNESSED is informational, not blocked-on-operator)",
			len(q.Items), len(q.Resolved))
	}
}

func TestFoldNoRowWithoutABackingBlockedNotify(t *testing.T) {
	// The non-vacuous witness: an empty / information-only ledger yields an empty
	// queue — the fold never invents a row without a blocked notify behind it.
	asOf := time.Unix(0, 0).UTC().Add(1 * time.Hour)
	q := Fold(nil, Params{AsOf: asOf, Deadline: 2 * time.Hour})

	if len(q.Items) != 0 || len(q.Resolved) != 0 {
		t.Fatalf("empty ledger -> Items=%d Resolved=%d, want 0/0", len(q.Items), len(q.Resolved))
	}
	if q.Schema != Schema {
		t.Fatalf("Schema = %q, want %q", q.Schema, Schema)
	}
}

func TestFoldExpiredRanksBeforeWaitingAtEqualAge(t *testing.T) {
	filed := time.Unix(0, 0).UTC()
	asOf := filed.Add(3 * time.Hour) // both past the 2h deadline -> both expired
	events := []loopmgr.Event{
		ev(loopmgr.EventStart, "dispatch/iss-a", "run-a", "", filed, 1),
		ev(loopmgr.EventNotify, "dispatch/iss-a", "run-a", "ESCALATE", filed, 2),
		ev(loopmgr.EventStart, "dispatch/iss-b", "run-b", "", filed, 3),
		ev(loopmgr.EventNotify, "dispatch/iss-b", "run-b", "NEEDS_HUMAN", filed.Add(1*time.Second), 4),
	}
	q := Fold(events, Params{AsOf: asOf, Deadline: 2 * time.Hour})
	if len(q.Items) != 2 {
		t.Fatalf("Items = %d, want 2", len(q.Items))
	}
	if q.Items[0].FiledAtUnixNano > q.Items[1].FiledAtUnixNano {
		t.Fatal("rows must be oldest-first within the same status band")
	}
	if q.OldestAgeSeconds <= 0 {
		t.Fatalf("OldestAgeSeconds = %g, want > 0", q.OldestAgeSeconds)
	}
}

// approx compares two second values within eps (test clock jitter tolerance).
func approx(got, want float64, eps float64) bool {
	d := got - want
	if d < 0 {
		d = -d
	}
	return d <= eps
}
