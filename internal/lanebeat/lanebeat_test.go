package lanebeat

import (
	"testing"
	"time"
)

var (
	t0    = time.Date(2026, 8, 7, 12, 0, 0, 0, time.UTC)
	spawn = t0.Add(-20 * time.Minute)
)

// working is a holder the supervisor can see running and producing output: the
// only shape that may be attested.
func working() Holder {
	return Holder{
		Lane:         "gateway",
		HostID:       "DESKTOP-BB3FMHP",
		PID:          4242,
		Alive:        true,
		StartedAt:    spawn,
		LastOutputAt: t0.Add(-30 * time.Second),
		MaxHold:      40 * time.Minute,
	}
}

// heldByWorker is the lease that worker took after it started.
func heldByWorker() Lease {
	return Lease{
		Lane:       "gateway",
		Holder:     "claude-2404",
		HostID:     "DESKTOP-BB3FMHP",
		LoopTS:     "2026-08-07T11:41:00Z",
		AcquiredAt: spawn.Add(1 * time.Minute),
	}
}

func TestDecideBeatsALiveProgressingHolder(t *testing.T) {
	got := Decide(working(), []Lease{heldByWorker()}, t0)
	if !got.Beat || got.Reason != ReasonBeat {
		t.Fatalf("a live, progressing, in-budget holder must be beaten: %+v", got)
	}
	// The identity must be COPIED off the matched record. The kernel credits a
	// beat by (loop_ts, lane) and authenticates it against the recorded holder;
	// re-deriving any of the three mints a different identity that folds as a
	// no-op against the real lease.
	if got.Lane != "gateway" || got.Owner != "claude-2404" || got.LoopTS != "2026-08-07T11:41:00Z" {
		t.Fatalf("beat identity must come from the matched lease record, got %+v", got)
	}
}

// THE defect this package exists to prevent: a beat that keeps beating after the
// holder is gone would make a dead lease look permanently alive — the gap's own
// shape with the sign flipped, and strictly worse than the silent status quo
// because the TTL backstop that currently does all the work would stop firing.
func TestDecideNeverBeatsADeadHolder(t *testing.T) {
	h := working()
	h.Alive = false
	// Everything else is maximally favourable: fresh output, in budget, a lease
	// that is unambiguously this worker's. None of it may rescue a dead holder.
	got := Decide(h, []Lease{heldByWorker()}, t0)
	if got.Beat {
		t.Fatalf("a dead holder must never be beaten, got %+v", got)
	}
	if got.Reason != ReasonHolderDead {
		t.Fatalf("reason = %q, want %q", got.Reason, ReasonHolderDead)
	}
}

// Alive is not progressing. A resident-but-silent worker stops being attested
// and falls back to exactly today's TTL-only lifetime — this rung can only
// WITHHOLD an extension, never shorten a lease.
func TestDecideRefusesAQuietHolder(t *testing.T) {
	h := working()
	h.LastOutputAt = t0.Add(-(DefaultQuietAfter + time.Minute))
	got := Decide(h, []Lease{heldByWorker()}, t0)
	if got.Beat || got.Reason != ReasonHolderQuiet {
		t.Fatalf("a holder quiet past the window must not be beaten: %+v", got)
	}

	// Just inside the window still counts as progressing.
	h.LastOutputAt = t0.Add(-(DefaultQuietAfter - time.Minute))
	if got := Decide(h, []Lease{heldByWorker()}, t0); !got.Beat {
		t.Fatalf("a holder inside the quiet window must still be beaten: %+v", got)
	}

	// A worker that has produced nothing YET is inside its OPENING window rather
	// than instantly quiet: the zero LastOutputAt falls back to StartedAt, so the
	// window is measured from the spawn.
	fresh := working()
	fresh.StartedAt = t0.Add(-2 * time.Minute)
	fresh.LastOutputAt = time.Time{}
	lease := heldByWorker()
	lease.AcquiredAt = fresh.StartedAt.Add(10 * time.Second)
	if got := Decide(fresh, []Lease{lease}, t0); !got.Beat {
		t.Fatalf("a just-spawned holder with no output yet must still be beaten: %+v", got)
	}

	// ...and that fallback is a window, not a licence: the same silent worker
	// twenty minutes in has still proven nothing, so it stops being attested.
	fresh.StartedAt = t0.Add(-(DefaultQuietAfter + time.Minute))
	lease.AcquiredAt = fresh.StartedAt.Add(10 * time.Second)
	if got := Decide(fresh, []Lease{lease}, t0); got.Beat || got.Reason != ReasonHolderQuiet {
		t.Fatalf("a holder that never produced output must age out of its opening window: %+v", got)
	}
}

// A process still running past the budget it was spawned under is a hang the
// supervisor is about to reap. Attesting it is how ONE wedged worker pins a lane
// forever, which is the unbounded-lifetime failure this fix must not introduce.
func TestDecideStopsAttestingPastTheHoldersOwnDeadline(t *testing.T) {
	h := working()
	h.StartedAt = t0.Add(-41 * time.Minute) // MaxHold is 40m
	lease := heldByWorker()
	lease.AcquiredAt = h.StartedAt.Add(time.Minute)
	got := Decide(h, []Lease{lease}, t0)
	if got.Beat || got.Reason != ReasonHolderPastDeadline {
		t.Fatalf("an over-budget holder must not be attested: %+v", got)
	}

	// MaxHold == 0 means the caller has no budget to enforce; the rung is off and
	// the remaining evidence decides.
	h.MaxHold = 0
	if got := Decide(h, []Lease{lease}, t0); !got.Beat {
		t.Fatalf("MaxHold=0 must disable the deadline rung, got %+v", got)
	}
}

// The false-revival guard from the SUPERVISOR side. Without it, any live process
// on a lane licenses a beat on whatever orphan happens to sit there.
func TestDecideRefusesALeaseThatPredatesTheHolder(t *testing.T) {
	lease := heldByWorker()
	lease.Holder = "some-crashed-peer"
	lease.AcquiredAt = spawn.Add(-3 * time.Hour) // taken long before this worker existed
	got := Decide(working(), []Lease{lease}, t0)
	if got.Beat || got.Reason != ReasonLeasePredatesHolder {
		t.Fatalf("a stranger's older orphan must not be revived: %+v", got)
	}
}

// An absent/unparseable acquire stamp is UNPROVABLE, not old: it cannot satisfy
// the binding, so it refuses rather than defaulting into a beat.
func TestDecideRefusesAnUnstampedLease(t *testing.T) {
	lease := heldByWorker()
	lease.AcquiredAt = time.Time{}
	got := Decide(working(), []Lease{lease}, t0)
	if got.Beat || got.Reason != ReasonLeasePredatesHolder {
		t.Fatalf("an unstamped lease is unprovable, must not be beaten: %+v", got)
	}
}

func TestDecideRefusesForeignHostAndUnattributableHolder(t *testing.T) {
	foreign := heldByWorker()
	foreign.HostID = "SOME-OTHER-BOX"
	if got := Decide(working(), []Lease{foreign}, t0); got.Beat || got.Reason != ReasonForeignHost {
		t.Fatalf("this host's process table is not evidence about another box: %+v", got)
	}

	// The kernel itself refuses to beat a holder-less lease (no owner can
	// authenticate against an absent holder). Mirror it rather than discovering
	// it as a non-zero exit.
	anon := heldByWorker()
	anon.Holder = "  "
	if got := Decide(working(), []Lease{anon}, t0); got.Beat || got.Reason != ReasonUnattributableHolder {
		t.Fatalf("an unattributable lease must not be beaten: %+v", got)
	}
}

// A beat must never CREATE a lease.
func TestDecideRefusesWhenNothingIsHeldOnTheLane(t *testing.T) {
	other := heldByWorker()
	other.Lane = "docs"
	if got := Decide(working(), []Lease{other}, t0); got.Beat || got.Reason != ReasonNoLeaseOnLane {
		t.Fatalf("no live lease on the lane means nothing to beat: %+v", got)
	}
	if got := Decide(working(), nil, t0); got.Beat || got.Reason != ReasonNoLeaseOnLane {
		t.Fatalf("empty live set must refuse: %+v", got)
	}
}

func TestDecideRefusesAnUnboundHolder(t *testing.T) {
	h := working()
	h.Lane = "   "
	if got := Decide(h, []Lease{heldByWorker()}, t0); got.Beat || got.Reason != ReasonNoLane {
		t.Fatalf("a holder bound to no lane has nothing to refresh: %+v", got)
	}
}

// Same-lane siblings: the newest record wins, matching the kernel's own
// last-wins rule when loop_ts is omitted, so a beat cannot land on the wrong one.
func TestDecidePicksTheNewestLeaseOnTheLane(t *testing.T) {
	older := heldByWorker()
	older.Holder = "older-sibling"
	older.LoopTS = "2026-08-07T11:41:00Z"
	older.AcquiredAt = spawn.Add(1 * time.Minute)

	newer := heldByWorker()
	newer.Holder = "newer-sibling"
	newer.LoopTS = "2026-08-07T11:50:00Z"
	newer.AcquiredAt = spawn.Add(10 * time.Minute)

	// Order of the input must not matter.
	for _, live := range [][]Lease{{older, newer}, {newer, older}} {
		got := Decide(working(), live, t0)
		if !got.Beat || got.Owner != "newer-sibling" || got.LoopTS != "2026-08-07T11:50:00Z" {
			t.Fatalf("newest lease on the lane must win, got %+v", got)
		}
	}
}

// Every refusal path must be silent about identity: a Decision that does not
// admit must never carry an Owner/LoopTS a careless caller could still pass to
// the writer.
func TestRefusalsCarryNoBeatIdentity(t *testing.T) {
	dead := working()
	dead.Alive = false
	quiet := working()
	quiet.LastOutputAt = t0.Add(-2 * time.Hour)
	stale := working()
	stale.StartedAt = t0.Add(-10 * time.Hour)

	orphan := heldByWorker()
	orphan.AcquiredAt = spawn.Add(-time.Hour)

	cases := []struct {
		name string
		h    Holder
		live []Lease
	}{
		{"dead", dead, []Lease{heldByWorker()}},
		{"quiet", quiet, []Lease{heldByWorker()}},
		{"past-deadline", stale, []Lease{heldByWorker()}},
		{"no-lease", working(), nil},
		{"predates", working(), []Lease{orphan}},
	}
	for _, tc := range cases {
		got := Decide(tc.h, tc.live, t0)
		if got.Beat {
			t.Fatalf("%s: expected refusal, got %+v", tc.name, got)
		}
		if got.Owner != "" || got.LoopTS != "" {
			t.Fatalf("%s: a refusal must carry no beat identity, got %+v", tc.name, got)
		}
		if got.Reason == "" || got.Reason == ReasonBeat {
			t.Fatalf("%s: a refusal must name itself, got %q", tc.name, got.Reason)
		}
	}
}
