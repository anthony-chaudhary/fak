package main

import (
	"errors"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/loopfleet"
	"github.com/anthony-chaudhary/fak/internal/loopmgr"
	"github.com/anthony-chaudhary/fak/internal/relay"
	"github.com/anthony-chaudhary/fak/internal/superloop"
)

// writeDispatchLedger lays down a hermetic dispatch tick ledger under root. The rows
// are the real `fleet-issue-resolve-progress/1` shape, and they deliberately carry
// upbeat SELF-REPORTED counters (ok/resolved_toward_target) that the witness must
// ignore: only witnessed_numbers (the emitted refs) is read from here.
func writeDispatchLedger(t *testing.T, root, rows string) {
	t.Helper()
	dir := filepath.Join(root, ".dispatch-runs")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "progress.jsonl"), []byte(rows), 0o644); err != nil {
		t.Fatal(err)
	}
}

// TestMemberFollowonOrphanedEndToEnd drives the whole #4957 shell seam over the REAL
// production emission binding (the dispatch loop's own tick ledger) and a hermetic
// live-issue resolver, covering every branch the issue's "Done when" names:
// orphaned-and-ranked-as-debt, advanced/closed-reads-clean, unreadable-fails-closed,
// and the gate-off default that leaves the axis unread rather than clean.
func TestMemberFollowonOrphanedEndToEnd(t *testing.T) {
	root := t.TempDir()
	// The tail row is the one the witness joins; the earlier row must be ignored.
	writeDispatchLedger(t, root, `{"schema":"fleet-issue-resolve-progress/1","ok":true,"witnessed_numbers":[111]}
{"schema":"fleet-issue-resolve-progress/1","ok":true,"resolved_toward_target":176,"witnessed_numbers":[4591,4835]}
`)

	c := &superloopCollector{root: root}
	m := superloop.Member{Kind: superloop.KindLoop, Ref: "dispatch"}
	// The dispatch loop declares an HOURLY cadence — the window floor must widen it,
	// or every emitted issue untouched for an hour would be slandered orphaned.
	live := loopfleet.LoopHealth{Kind: "dispatch", State: loopmgr.HealthLive, CadenceSeconds: 3600}

	old := followonIssueState
	t.Cleanup(func() { followonIssueState = old })

	// Gate OFF (the default): the axis is UNREAD — no verdict, no reason, and the
	// live resolver is never called. Unread is not clean; it is weighed by nothing.
	calls := 0
	followonIssueState = func(int) (string, time.Time, error) {
		calls++
		return "OPEN", time.Now(), nil
	}
	if got, reason := c.memberFollowon(m, live); got != "" || reason != "" {
		t.Fatalf("gate off: memberFollowon = (%q, %q), want the unread axis (\"\", \"\")", got, reason)
	}
	if calls != 0 {
		t.Fatalf("gate off made %d live issue read(s); the default walk must stay offline", calls)
	}

	t.Setenv(followonWitnessEnv, "1")

	// ORPHANED: both emitted issues are OPEN and untouched well past the window. The
	// ledger row above claims ok/resolved — the verdict must come from live state.
	stale := time.Now().Add(-72 * time.Hour)
	followonIssueState = func(int) (string, time.Time, error) { return "OPEN", stale, nil }
	got, reason := c.memberFollowon(m, live)
	if got != superloop.FollowonOrphaned {
		t.Fatalf("open+idle emissions: follow-on = %q, want %q", got, superloop.FollowonOrphaned)
	}
	if reason != relay.ReasonOrphanedFollowon {
		t.Fatalf("orphaned reason = %q, want the closed token %q", reason, relay.ReasonOrphanedFollowon)
	}
	if debt := loopDebt(live, "", got); debt != 1 {
		t.Fatalf("loopDebt(live, -, orphaned) = %d, want 1 (the #4957 follow-on term; without it an orphaned loop reads clean)", debt)
	}
	// Distinct from SPINNING (#4956): a loop can be orphaned while its own progress
	// axis is untouched, and the two terms compound rather than collapse.
	if debt := loopDebt(live, superloop.ProgressSpinning, got); debt != 2 {
		t.Fatalf("loopDebt(live, spinning, orphaned) = %d, want 2 (progress term + follow-on term)", debt)
	}

	// ADVANCING: an emission touched inside the window reads clean, and so does a
	// CLOSED one regardless of when it was last touched.
	followonIssueState = func(int) (string, time.Time, error) { return "OPEN", time.Now().Add(-time.Hour), nil }
	if got, reason = c.memberFollowon(m, live); got != superloop.FollowonAdvancing || reason != "" {
		t.Fatalf("recently-advanced emissions: (%q, %q), want (%q, \"\")", got, reason, superloop.FollowonAdvancing)
	}
	if debt := loopDebt(live, "", got); debt != 0 {
		t.Fatalf("loopDebt(live, -, advancing) = %d, want 0", debt)
	}
	followonIssueState = func(int) (string, time.Time, error) { return "CLOSED", stale, nil }
	if got, _ = c.memberFollowon(m, live); got != superloop.FollowonAdvancing {
		t.Fatalf("closed emission: follow-on = %q, want %q (a closed follow-on is carried, not orphaned)", got, superloop.FollowonAdvancing)
	}

	// UNREADABLE: a `gh` outage must NOT be read as an orphan. Fail closed to
	// unknown — surfaced, never fabricated into debt.
	followonIssueState = func(int) (string, time.Time, error) {
		return "", time.Time{}, errors.New("gh: could not resolve host")
	}
	if got, reason = c.memberFollowon(m, live); got != superloop.FollowonUnknown {
		t.Fatalf("unreadable emission: follow-on = %q, want %q (an orphan is never fabricated from an absence)", got, superloop.FollowonUnknown)
	}
	if reason != "" {
		t.Fatalf("unknown carried reason %q, want \"\" — only an orphan binds the closed token", reason)
	}
	if debt := loopDebt(live, "", got); debt != 0 {
		t.Fatalf("loopDebt(live, -, unknown) = %d, want 0 (unknown is surfaced, never debt)", debt)
	}
}

// TestFollowonEmissionRefsBinding pins the production emission binding: the dispatch
// loop's LATEST durable tick row supplies the emitted refs in relay.ArtifactIssue
// pointer form, and every missing/unparsable/foreign surface fails closed to nil (the
// axis unread) rather than inventing an emission.
func TestFollowonEmissionRefsBinding(t *testing.T) {
	root := t.TempDir()

	// No ledger on this host: nothing emitted here, so nothing is owed.
	if refs := followonEmissionRefs(root, "dispatch"); refs != nil {
		t.Fatalf("absent ledger: refs = %v, want nil (fail closed)", refs)
	}

	writeDispatchLedger(t, root, `{"schema":"fleet-issue-resolve-progress/1","witnessed_numbers":[1]}
not json at all
{"schema":"fleet-issue-resolve-progress/1","witnessed_numbers":[4591,0,4835]}

`)
	got := followonEmissionRefs(root, "dispatch")
	want := []string{"#4591", "#4835"}
	if len(got) != len(want) {
		t.Fatalf("refs = %v, want %v (latest tick only, non-positive numbers dropped)", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("refs = %v, want %v", got, want)
		}
	}

	// A loop with no emission record of its own is not joined at all.
	if refs := followonEmissionRefs(root, "dojo"); refs != nil {
		t.Fatalf("unbound loop kind: refs = %v, want nil", refs)
	}

	// A tail row of a foreign schema is not mistaken for a dispatch tick.
	writeDispatchLedger(t, root, `{"schema":"some-other-ledger/1","witnessed_numbers":[9999]}
`)
	if refs := followonEmissionRefs(root, "dispatch"); refs != nil {
		t.Fatalf("foreign schema: refs = %v, want nil", refs)
	}

	// A tick that emitted nothing owes nothing — an empty ref set, not an orphan.
	writeDispatchLedger(t, root, `{"schema":"fleet-issue-resolve-progress/1","witnessed_numbers":[]}
`)
	if refs := followonEmissionRefs(root, "dispatch"); len(refs) != 0 {
		t.Fatalf("empty tick: refs = %v, want none", refs)
	}
}

// TestFollowonWindowFloor pins the fail-closed window: a fast-ticking loop's cadence
// is widened to the day floor so the witness cannot fabricate orphans from a window
// narrower than a work day, while a slower declared cadence is honored as declared.
func TestFollowonWindowFloor(t *testing.T) {
	if got := followonWindow(3600); got != defaultFollowonCadence {
		t.Fatalf("followonWindow(hourly) = %s, want the %s floor", got, defaultFollowonCadence)
	}
	if got := followonWindow(0); got != defaultFollowonCadence {
		t.Fatalf("followonWindow(undeclared) = %s, want the %s floor", got, defaultFollowonCadence)
	}
	if got := followonWindow(7 * 24 * 3600); got != 7*24*time.Hour {
		t.Fatalf("followonWindow(weekly) = %s, want 168h0m0s (a wider declared cadence is honored)", got)
	}
}

// TestResolveFollowonEmissionUnparsableRef pins the ref-parse edge: a ref this reader
// cannot join stays UNRESOLVED, which ClassifyFollowon folds to unknown — never an
// orphan invented from a ref shape the join did not understand.
func TestResolveFollowonEmissionUnparsableRef(t *testing.T) {
	old := followonIssueState
	t.Cleanup(func() { followonIssueState = old })
	followonIssueState = func(int) (string, time.Time, error) {
		t.Fatal("unparsable ref must not reach the live resolver")
		return "", time.Time{}, nil
	}
	for _, ref := range []string{"", "#", "ledger:abc", "#-3"} {
		em := resolveFollowonEmission(ref, defaultFollowonCadence, time.Now())
		if em.Resolved {
			t.Fatalf("resolveFollowonEmission(%q) resolved; want unresolved (fail closed)", ref)
		}
	}
	verdict, reason := superloop.ClassifyFollowon(
		superloop.Member{Kind: superloop.KindLoop, Ref: "dispatch"},
		superloop.FollowonRead{Emissions: []superloop.FollowonEmission{{Ref: "ledger:abc"}}})
	if verdict != superloop.FollowonUnknown || reason != "" {
		t.Fatalf("unparsable emission folded to (%q, %q), want (%q, \"\")", verdict, reason, superloop.FollowonUnknown)
	}
}
