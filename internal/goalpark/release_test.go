package goalpark

import (
	"errors"
	"strings"
	"testing"
	"time"
)

// release_test.go — the witness for the park's FOURTH clearing path (#5953).
//
// The three internal paths (timer, probe slot, MaxWait clamp) share one blind spot:
// none of them can hear that the wall's CAUSE was removed. An operator who enrolls a
// seat has fixed the problem, and the parks on the box keep holding anyway. Release
// is what listens; these arms pin that it retires a park exactly once, only with an
// attributable reason, and never twice.

func parkedRecord(goal, account string, parkedAt, until time.Time) Record {
	return Record{
		Schema:      Schema,
		Goal:        goal,
		Account:     account,
		Reason:      "LONG_RETRY_AFTER",
		ParkedAt:    parkedAt.Unix(),
		ParkedUntil: until.Unix(),
		Command:     []string{"fak", "guard", "--", "claude"},
	}
}

// TestReleaseRetiresAParkThatIsNotYetDue is the whole point: ClaimDue refuses a park
// before parked_until (ErrNotDue), so nothing that existed before this could clear a
// wall early. Release can, and the record must come back UNBLOCKING for the account it
// was holding.
func TestReleaseRetiresAParkThatIsNotYetDue(t *testing.T) {
	s := Store{Dir: t.TempDir()}
	now := time.Date(2026, 8, 8, 12, 0, 0, 0, time.UTC)
	rec := parkedRecord("lane-alpha", "acct-a", now, now.Add(20*time.Hour))
	if err := s.Park(rec); err != nil {
		t.Fatalf("Park: %v", err)
	}
	loaded, err := s.Load("lane-alpha")
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if !loaded.Blocks("acct-a", now) {
		t.Fatal("precondition: a fresh 20h park must block its account")
	}
	if _, err := s.ClaimDue("lane-alpha", "supervisor", now); !errors.Is(err, ErrNotDue) {
		t.Fatalf("ClaimDue before parked_until = %v, want ErrNotDue (this is the gap Release fills)", err)
	}

	released, err := s.Release("lane-alpha", "guard-box-a-77", "seat enrolled", now)
	if err != nil {
		t.Fatalf("Release: %v", err)
	}
	if released.ClaimedAt != now.Unix() || released.ClaimedBy != "guard-box-a-77" {
		t.Fatalf("released record = claimed_at %d by %q, want %d by the releaser",
			released.ClaimedAt, released.ClaimedBy, now.Unix())
	}
	if !strings.Contains(released.NextAction, "seat enrolled") {
		t.Fatalf("next_legal_action = %q, want it to carry the operator's reason verbatim", released.NextAction)
	}

	// The release must be VISIBLE to the one predicate every reader consults, from disk.
	reloaded, err := s.Load("lane-alpha")
	if err != nil {
		t.Fatalf("Load after release: %v", err)
	}
	if reloaded.Blocks("acct-a", now) {
		t.Fatal("a released park still blocks its account — the release never reached the reader")
	}
	if _, blocked := s.Resolve("lane-alpha", "acct-a", "guard", now); blocked {
		t.Fatal("Resolve still says blocked after a release")
	}
}

// TestReleaseIsExactlyOnceAcrossRacers is the property that lets seat-refresh be
// broadcast. Every guard on the box drains the same directive and races for the same
// records; exactly one may count the release as its own affected work, or the fold at
// the control point double-counts one park as N.
func TestReleaseIsExactlyOnceAcrossRacers(t *testing.T) {
	s := Store{Dir: t.TempDir()}
	now := time.Date(2026, 8, 8, 12, 0, 0, 0, time.UTC)
	if err := s.Park(parkedRecord("lane-beta", "acct-b", now, now.Add(20*time.Hour))); err != nil {
		t.Fatalf("Park: %v", err)
	}
	var won, lost int
	for _, racer := range []string{"guard-1", "guard-2", "guard-3"} {
		switch _, err := s.Release("lane-beta", racer, "seat enrolled", now); {
		case err == nil:
			won++
		case errors.Is(err, ErrClaimed):
			lost++
		default:
			t.Fatalf("%s: unexpected error %v", racer, err)
		}
	}
	if won != 1 || lost != 2 {
		t.Fatalf("won=%d lost=%d, want exactly one winner and two ErrClaimed", won, lost)
	}
	// A park the TIMER already retired is not re-releasable either — the two paths
	// share one claim, so they cannot both report retiring the same record.
	if _, err := s.Release("lane-beta", "guard-4", "seat enrolled", now.Add(time.Hour)); !errors.Is(err, ErrClaimed) {
		t.Fatalf("second-wave release = %v, want ErrClaimed", err)
	}
}

// TestReleaseRefusesAnUnattributedReason: on disk, a park retired with no reason is
// indistinguishable from one the timer retired, and an operator reading the record
// later cannot tell whether a wall lifted or somebody cut it short.
func TestReleaseRefusesAnUnattributedReason(t *testing.T) {
	s := Store{Dir: t.TempDir()}
	now := time.Date(2026, 8, 8, 12, 0, 0, 0, time.UTC)
	if err := s.Park(parkedRecord("lane-gamma", "acct-c", now, now.Add(20*time.Hour))); err != nil {
		t.Fatalf("Park: %v", err)
	}
	if _, err := s.Release("lane-gamma", "guard-1", "   ", now); err == nil {
		t.Fatal("Release accepted a blank reason")
	}
	loaded, err := s.Load("lane-gamma")
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if loaded.ClaimedAt != 0 {
		t.Fatal("a refused release still touched the record")
	}
	if _, err := s.Release("missing-goal", "guard-1", "seat enrolled", now); !errors.Is(err, ErrNotFound) {
		t.Fatalf("Release of an absent goal = %v, want ErrNotFound", err)
	}
}

// TestClaimDueStillHonoursItsOwnGate — Release shares takeClaim with ClaimDue, so this
// pins that sharing the write did not relax the timer path's precondition.
func TestClaimDueStillHonoursItsOwnGate(t *testing.T) {
	s := Store{Dir: t.TempDir()}
	now := time.Date(2026, 8, 8, 12, 0, 0, 0, time.UTC)
	if err := s.Park(parkedRecord("lane-delta", "acct-d", now, now.Add(2*time.Hour))); err != nil {
		t.Fatalf("Park: %v", err)
	}
	if _, err := s.ClaimDue("lane-delta", "sup", now.Add(time.Hour)); !errors.Is(err, ErrNotDue) {
		t.Fatalf("ClaimDue before the wall lifts = %v, want ErrNotDue", err)
	}
	claimed, err := s.ClaimDue("lane-delta", "sup", now.Add(3*time.Hour))
	if err != nil {
		t.Fatalf("ClaimDue after the wall lifts: %v", err)
	}
	if !strings.Contains(claimed.NextAction, "resume claimed") {
		t.Fatalf("timer-path next_legal_action = %q, want the resume wording, not the release wording", claimed.NextAction)
	}
	if strings.Contains(claimed.NextAction, "operator directive") {
		t.Fatalf("timer-path next_legal_action = %q, want it distinguishable from a release", claimed.NextAction)
	}
}
