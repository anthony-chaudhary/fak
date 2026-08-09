package main

// Witnesses for #5571: the loop-drive tick's ambient lease-ref sync report is
// SURFACED, and surfaced on an EDGE. #5564 gave the ambient sync a budget and a
// give-up path; #5569 made a breaker skip report itself as DEGRADED with a
// stable token, explicitly so a skipped sync is never mistaken for a clean one.
// Both were produced correctly and then discarded by every call site in
// loop_drive_region.go, so a node whose lease-ref transport was wedged ran with
// no peer visibility and printed nothing at all.
//
// These tests pin BOTH halves of the cure, because either one alone is a
// regression the other would hide: the loud half (a degradation reaches the
// operator's stream, naming its reason and its cause) and the QUIET half (a
// standing degradation does not re-print on every crossing, and a healthy tick
// says nothing). A fix that printed on every crossing would trade an invisible
// signal for an unreadable one on exactly the box the signal exists for.

import (
	"bytes"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/leaseref"
	"github.com/anthony-chaudhary/fak/internal/loopdrive"
)

// leaseRefSyncNoticeDegraded is a real degraded report — folded through the same
// loopdrive.ReportLeaseRefSync the ambient boundary uses, not a hand-built
// struct — carrying a distinctive transport failure the announcement must quote.
func leaseRefSyncNoticeDegraded() loopdrive.LeaseRefSyncReport {
	step := loopdrive.LeaseRefSyncPlan(loopdrive.LeaseRefSyncPlanInput{Remote: "origin"})[0]
	return loopdrive.ReportLeaseRefSync([]loopdrive.LeaseRefSyncAttempt{
		{Step: step, Err: "fixture-transport-wedged"},
	})
}

// stubAmbientLeaseRefSync points the ambient sync seam at a report the test
// controls, so no git transport and no remote is involved at all. The returned
// setter swaps which report every later crossing yields.
func stubAmbientLeaseRefSync(t *testing.T, initial loopdrive.LeaseRefSyncReport) (set func(loopdrive.LeaseRefSyncReport), crossings *int) {
	t.Helper()
	current := initial
	count := 0
	orig := ambientLeaseRefSync
	ambientLeaseRefSync = func(loopdrive.LeaseRefSyncSurface, *leaseref.Store, string, bool) loopdrive.LeaseRefSyncReport {
		count++
		return current
	}
	t.Cleanup(func() { ambientLeaseRefSync = orig })
	return func(r loopdrive.LeaseRefSyncReport) { current = r }, &count
}

// TestLoopDriveRegionAnnouncesLeaseRefSyncDegradationOnTheEdge is the #5571
// witness. A wedged lease-ref transport must reach the operator's stream ONCE
// per transition — not once per crossing (ensure crosses the boundary twice on a
// fresh acquire), not once per turn, and not never.
func TestLoopDriveRegionAnnouncesLeaseRefSyncDegradationOnTheEdge(t *testing.T) {
	initRegionTestRepo(t)
	set, crossings := stubAmbientLeaseRefSync(t, leaseRefSyncNoticeDegraded())

	var out bytes.Buffer
	spec := loopdrive.Spec{Loop: "region-loop", Lane: "gateway"}
	hold := newLoopDriveRegionHold(loopDriveOptions{}, spec).announceOn(&out)
	if hold == nil {
		t.Fatal("a lane-declaring spec must produce a region hold")
	}
	defer hold.release()

	degraded := func() int { return strings.Count(out.String(), "lease-ref sync degraded") }
	recovered := func() int { return strings.Count(out.String(), "lease-ref sync recovered") }

	// 1. The first turn: a fresh acquire crosses the sync boundary TWICE
	// (fetch-before-decide, push-after-write) and both crossings are degraded.
	// The operator must be told — exactly once.
	if refuse, err := hold.ensure(time.Now()); err != nil || refuse != nil {
		t.Fatalf("fresh acquire must hold: refuse=%+v err=%v", refuse, err)
	}
	if *crossings < 2 {
		t.Fatalf("ensure crossed the sync boundary %d time(s), want at least 2 — the test is not exercising the multi-crossing turn", *crossings)
	}
	if got := degraded(); got != 1 {
		t.Fatalf("degradation announcements after one turn = %d, want exactly 1 (a discarded report prints 0; a per-crossing print puts %d on the terminal every turn):\n%s",
			got, *crossings, out.String())
	}
	line := strings.TrimSpace(out.String())
	if !strings.Contains(line, loopdrive.ReasonLeaseRefSyncTransport) {
		t.Errorf("announcement does not carry the closed reason token %q, so an operator cannot grep it: %q", loopdrive.ReasonLeaseRefSyncTransport, line)
	}
	if !strings.Contains(line, "fixture-transport-wedged") {
		t.Errorf("announcement drops the transport failure that caused it — a content-free ping: %q", line)
	}
	if !strings.Contains(line, "nonfatal") {
		t.Errorf("announcement does not say it is nonfatal; an operator would read a fail-open advisory as a stop: %q", line)
	}
	if !strings.Contains(line, hold.id) {
		t.Errorf("announcement does not name the region lease it is about: %q", line)
	}

	// 2. THE QUIET HALF. The transport stays wedged for the next turns. The edge
	// already fired, so nothing more is printed — this is the assertion a
	// print-every-crossing "fix" fails.
	now := time.Now()
	for turn := 2; turn <= 4; turn++ {
		now = now.Add(time.Minute)
		if refuse, err := hold.ensure(now); err != nil || refuse != nil {
			t.Fatalf("turn %d renew must hold: refuse=%+v err=%v", turn, refuse, err)
		}
		if got := degraded(); got != 1 {
			t.Fatalf("turn %d: degradation announcements = %d, want still 1 — a standing degradation re-printed every tick spams the operator off the terminal on exactly the box this signal is for:\n%s",
				turn, got, out.String())
		}
	}

	// 3. The transport recovers. Silence here would be indistinguishable from
	// "still wedged", so the recovery edge prints exactly one line.
	set(loopdrive.ReportLeaseRefSync([]loopdrive.LeaseRefSyncAttempt{
		{Step: loopdrive.LeaseRefSyncPlan(loopdrive.LeaseRefSyncPlanInput{Remote: "origin"})[0]},
	}))
	now = now.Add(time.Minute)
	if refuse, err := hold.ensure(now); err != nil || refuse != nil {
		t.Fatalf("recovery turn must hold: refuse=%+v err=%v", refuse, err)
	}
	if got := recovered(); got != 1 {
		t.Fatalf("recovery announcements = %d, want exactly 1:\n%s", got, out.String())
	}

	// 4. ...and a healthy boundary then stays quiet, on both counters.
	now = now.Add(time.Minute)
	if refuse, err := hold.ensure(now); err != nil || refuse != nil {
		t.Fatalf("healthy turn must hold: refuse=%+v err=%v", refuse, err)
	}
	if got, want := recovered(), 1; got != want {
		t.Fatalf("recovery announcements = %d, want %d — a clean boundary must not announce itself every turn:\n%s", got, want, out.String())
	}

	// 5. It goes wedged AGAIN. The prior was reset by the recovery, so the new
	// edge fires: the discipline is edge-triggered, not print-once-per-process.
	set(leaseRefSyncNoticeDegraded())
	now = now.Add(time.Minute)
	if refuse, err := hold.ensure(now); err != nil || refuse != nil {
		t.Fatalf("re-degraded turn must hold: refuse=%+v err=%v", refuse, err)
	}
	if got := degraded(); got != 2 {
		t.Fatalf("degradation announcements after a second wedge = %d, want 2 — a later isolation must not be swallowed by the first one having been reported:\n%s",
			got, out.String())
	}
}

// TestLoopDriveRegionStaysSilentWhenLeaseRefSyncIsClean pins the floor: a
// healthy drive's stderr must be byte-for-byte empty on this axis. Without it,
// "announce the degradation" could be satisfied by announcing everything.
func TestLoopDriveRegionStaysSilentWhenLeaseRefSyncIsClean(t *testing.T) {
	initRegionTestRepo(t)
	stubAmbientLeaseRefSync(t, loopdrive.ReportLeaseRefSync([]loopdrive.LeaseRefSyncAttempt{
		{Step: loopdrive.LeaseRefSyncPlan(loopdrive.LeaseRefSyncPlanInput{Remote: "origin"})[0]},
	}))

	var out bytes.Buffer
	hold := newLoopDriveRegionHold(loopDriveOptions{}, loopdrive.Spec{Loop: "region-loop", Lane: "gateway"}).announceOn(&out)
	defer hold.release()
	now := time.Now()
	for turn := 1; turn <= 3; turn++ {
		if refuse, err := hold.ensure(now); err != nil || refuse != nil {
			t.Fatalf("turn %d must hold: refuse=%+v err=%v", turn, refuse, err)
		}
		now = now.Add(time.Minute)
	}
	if out.Len() != 0 {
		t.Fatalf("a clean lease-ref sync boundary wrote to the operator stream:\n%s", out.String())
	}
}

// TestLoopDriveRegionAnnouncesTheBreakerSkipAndToleratesNoStream closes the
// #5569 half of the ticket: an OPEN convergence breaker skips the plan entirely,
// and that skip is the report shape most at risk of reading as "fine". The
// announcement must carry the breaker's stable token, so an operator can tell a
// skipped crossing from a crossing that really ran and failed.
//
// It also pins the superloop posture: superloopDriveRegionAdmit builds a hold
// with no operator stream and returns a structured verdict instead, so a hold
// whose stderr was never attached must announce nothing and must not panic.
func TestLoopDriveRegionAnnouncesTheBreakerSkipAndToleratesNoStream(t *testing.T) {
	initRegionTestRepo(t)
	steps := loopdrive.LeaseRefSyncPlan(loopdrive.LeaseRefSyncPlanInput{Remote: "origin", LeaseRefsWritten: true})
	skipped := ambientLeaseRefSyncSkippedReport(steps, ambientLeaseRefSyncBreakerSkips-1)
	if skipped.Outcome != loopdrive.LeaseRefSyncDegraded {
		t.Fatalf("fixture precondition: a breaker skip must fold to degraded, got %q", skipped.Outcome)
	}
	stubAmbientLeaseRefSync(t, skipped)

	var out bytes.Buffer
	hold := newLoopDriveRegionHold(loopDriveOptions{}, loopdrive.Spec{Loop: "region-loop", Lane: "gateway"}).announceOn(&out)
	defer hold.release()
	if refuse, err := hold.ensure(time.Now()); err != nil || refuse != nil {
		t.Fatalf("fresh acquire must hold: refuse=%+v err=%v", refuse, err)
	}
	if !strings.Contains(out.String(), ambientLeaseRefSyncBreakerOpen) {
		t.Fatalf("a skipped sync is announced without the breaker token %q, so it reads like an ordinary transport failure:\n%s",
			ambientLeaseRefSyncBreakerOpen, out.String())
	}
	if got := strings.Count(out.String(), "lease-ref sync degraded"); got != 1 {
		t.Fatalf("breaker-skip announcements = %d, want exactly 1:\n%s", got, out.String())
	}

	// The superloop surface: constructed with no stream at all.
	silent := newLoopDriveRegionHold(loopDriveOptions{Lane: "docs"}, loopdrive.Spec{Loop: "superloop-fixture"})
	defer silent.release()
	if silent.stderr != nil {
		t.Fatal("a hold built without announceOn must carry no operator stream")
	}
	if refuse, err := silent.ensure(time.Now()); err != nil || refuse != nil {
		t.Fatalf("streamless hold must still acquire: refuse=%+v err=%v", refuse, err)
	}
	if !silent.leaseRefSyncDegraded {
		t.Fatal("a streamless hold must still track the degradation edge, so attaching a stream later cannot replay a stale one")
	}
}
