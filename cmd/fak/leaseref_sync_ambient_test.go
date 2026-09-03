package main

// Witnesses for the #2302 ambient leaseref-sync wiring (leaseref_sync_ambient.go):
// the boundary runs the convergence plan, stays NONFATAL when origin is
// unreachable (the fail-open posture the dispatch tick already uses), honors the
// FAK_LEASEREF_SYNC disable and FAK_LEASEREF_REMOTE override, and the loop-drive
// region hold actually drives it at the before-decide / after-write boundaries.

import (
	"bytes"
	"context"
	"fmt"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/leaseref"
	"github.com/anthony-chaudhary/fak/internal/loopdrive"
)

func TestAmbientLeaseRefSyncDisabledByEnv(t *testing.T) {
	t.Setenv("FAK_LEASEREF_SYNC", "off")
	store := leaseref.NewInDir("")
	report := ambientLeaseRefSyncImpl(loopdrive.LeaseRefSyncSurfaceLoopDriveTick, store, "origin", true)
	if report.Outcome != loopdrive.LeaseRefSyncOK {
		t.Fatalf("disabled sync outcome = %q, want ok", report.Outcome)
	}
	if report.Fatal {
		t.Fatalf("disabled sync must never be fatal: %+v", report)
	}
	if len(report.Failures) != 0 {
		t.Fatalf("disabled sync ran transport anyway: %+v", report.Failures)
	}
}

func TestAmbientLeaseRefSyncRemoteOverride(t *testing.T) {
	t.Setenv("FAK_LEASEREF_REMOTE", "upstream")
	if got := ambientLeaseRefSyncRemote(); got != "upstream" {
		t.Fatalf("remote = %q, want upstream", got)
	}
}

func TestAmbientLeaseRefSyncDefaultsOrigin(t *testing.T) {
	t.Setenv("FAK_LEASEREF_REMOTE", "")
	if got := ambientLeaseRefSyncRemote(); got != "origin" {
		t.Fatalf("default remote = %q, want origin", got)
	}
}

// TestAmbientLeaseRefSyncDegradesNonFatalWithoutOrigin: in a clone with no
// configured remote (single-box dev, CI) the convergence transport fails, but
// the boundary folds it to a NONFATAL degraded report — the caller must keep
// running on local lease evidence, never honest-stop on a dead link.
func TestAmbientLeaseRefSyncDegradesNonFatalWithoutOrigin(t *testing.T) {
	ambientLeaseRefSyncBreakerReset()
	t.Cleanup(ambientLeaseRefSyncBreakerReset)
	dir := initRegionTestRepo(t)
	store := leaseref.NewInDir(dir)

	report := ambientLeaseRefSyncImpl(loopdrive.LeaseRefSyncSurfaceLoopDriveTick, store, "origin", true)
	if report.Outcome != loopdrive.LeaseRefSyncDegraded {
		t.Fatalf("outcome = %q, want degraded (no origin configured)", report.Outcome)
	}
	if report.Fatal {
		t.Fatalf("transport failure must be surfaced, not fatal: %+v", report)
	}
	if report.Reason != loopdrive.ReasonLeaseRefSyncTransport {
		t.Fatalf("reason = %q, want %q", report.Reason, loopdrive.ReasonLeaseRefSyncTransport)
	}
	if len(report.Failures) == 0 {
		t.Fatalf("expected surfaced failures, got none: %+v", report)
	}

	// written=false runs only the before-decide fetch step: one failure, still
	// nonfatal — admission proceeds on the local lease set.
	report = ambientLeaseRefSyncImpl(loopdrive.LeaseRefSyncSurfaceLoopDriveTick, store, "origin", false)
	if report.Outcome != loopdrive.LeaseRefSyncDegraded || report.Fatal {
		t.Fatalf("fetch-only degrade = %+v, want nonfatal degraded", report)
	}
}

// TestAmbientLeaseRefSyncGivesUpOnAStalledGitRunner is the #5564 witness: with a
// git that never returns — an auth-stalled or black-holed remote, simulated
// deterministically through internal/leaseref's injected-Runner seam rather than
// against a real network — the ambient boundary must GIVE UP and report, not
// block the tick that called it.
//
// It fails in two distinguishable ways on a regression, both of them loud. If
// the call site goes back to an unexpirable context the runner's <-ctx.Done()
// never fires and the select below fails the test in bounded time with the
// diagnosis, instead of wedging the package suite until the go-test timeout —
// which is exactly how this bug spent its career being misread as a flake. If
// the deadline survives but moves to a per-step budget, the second transport
// call arrives with a LIVE context and that assertion names it.
func TestAmbientLeaseRefSyncGivesUpOnAStalledGitRunner(t *testing.T) {
	restore := ambientLeaseRefSyncBudget
	ambientLeaseRefSyncBudget = 200 * time.Millisecond
	t.Cleanup(func() { ambientLeaseRefSyncBudget = restore })
	// The breaker (#5569) is process-global: start from a closed one so this
	// witness really crosses the transport, and leave it closed for the next test.
	ambientLeaseRefSyncBreakerReset()
	t.Cleanup(ambientLeaseRefSyncBreakerReset)

	type transportCall struct {
		verb           string
		hasDeadline    bool
		expiredOnEntry bool
	}
	var calls []transportCall

	// The stall. Only the two transport verbs hang; the local preamble git calls
	// answer normally, so what is being witnessed is a wedged NETWORK git and not
	// a store that cannot read its own ref database.
	stalled := func(ctx context.Context, _ string, args ...string) (string, int, error) {
		verb := ""
		if len(args) != 0 {
			verb = args[0]
		}
		switch verb {
		case "push", "fetch":
			_, hasDeadline := ctx.Deadline()
			calls = append(calls, transportCall{verb: verb, hasDeadline: hasDeadline, expiredOnEntry: ctx.Err() != nil})
			<-ctx.Done() // returns ONLY when the context does — never, if it cannot expire
			return "", -1, ctx.Err()
		case "for-each-ref":
			// A non-empty local namespace, so the push direction is really attempted
			// rather than short-circuited as PushSkippedEmpty (leaseref/sync.go).
			return "refs/fak/locks/loop-region-loop\n", 0, nil
		default:
			return "", 0, nil
		}
	}
	store := leaseref.NewWithRunner(stalled, t.TempDir())

	// written=true plans BOTH steps (fetch-before-decide, push-after-write), which
	// is what makes the one-budget-per-call property observable.
	done := make(chan loopdrive.LeaseRefSyncReport, 1)
	go func() {
		done <- ambientLeaseRefSyncImpl(loopdrive.LeaseRefSyncSurfaceLoopDriveTick, store, "origin", true)
	}()
	var report loopdrive.LeaseRefSyncReport
	select {
	case report = <-done:
	case <-time.After(30 * ambientLeaseRefSyncBudget):
		t.Fatal("ambient lease-ref sync never returned against a stalled git: the sync context cannot expire, so a slow remote hangs the tick (#5564)")
	}

	if report.Outcome != loopdrive.LeaseRefSyncDegraded {
		t.Fatalf("outcome = %q, want degraded — a give-up must be SURFACED, not swallowed: %+v", report.Outcome, report)
	}
	if report.Fatal {
		t.Fatalf("a spent convergence budget must stay advisory, not fatal: %+v", report)
	}
	if report.Reason != loopdrive.ReasonLeaseRefSyncTransport {
		t.Fatalf("reason = %q, want %q", report.Reason, loopdrive.ReasonLeaseRefSyncTransport)
	}
	if len(report.Failures) != 2 {
		t.Fatalf("failures = %d, want both planned steps reported: %+v", len(report.Failures), report.Failures)
	}
	for _, f := range report.Failures {
		if !strings.Contains(f, "budget") {
			t.Errorf("failure %q does not name the deadline as the cause — the ledger reader would blame the remote", f)
		}
	}

	if len(calls) != 2 {
		t.Fatalf("transport calls = %+v, want one fetch then one push", calls)
	}
	if calls[0].verb != "fetch" || calls[1].verb != "push" {
		t.Fatalf("transport order = %+v, want fetch-before-decide then push-after-write", calls)
	}
	if !calls[0].hasDeadline {
		t.Error("the first transport call carried NO deadline: the ambient sync is back on an unexpirable context (#5564)")
	}
	if calls[0].expiredOnEntry {
		t.Error("the first transport call started already expired: the budget is not being spent on real work")
	}
	if !calls[1].expiredOnEntry {
		t.Error("the second transport call got a fresh live context: the budget is per-STEP, so the tick's exposure still multiplies by the plan's length")
	}
}

// TestAmbientLeaseRefSyncBreakerSkipsAfterASpentBudget is the #5569 witness: a
// bounded cost that is re-paid on EVERY crossing is still a node re-learning,
// at full price and forever, an answer it already has. After one crossing spends
// the whole budget the boundary must skip the plan for N crossings, let exactly
// ONE probe through, and resume the moment the transport answers again.
//
// Every assertion here is timing-free: it counts the transport verbs the
// injected runner was actually asked to run, so "skipped" is witnessed as NO
// GIT AT ALL rather than as a stopwatch reading. It runs the production N (the
// skips cost no wall-clock) and shrinks only the budget, so the two crossings
// that really stall cost 200ms each instead of 10s. NO REMOTE IS CONTACTED.
func TestAmbientLeaseRefSyncBreakerSkipsAfterASpentBudget(t *testing.T) {
	restore := ambientLeaseRefSyncBudget
	ambientLeaseRefSyncBudget = 200 * time.Millisecond
	t.Cleanup(func() { ambientLeaseRefSyncBudget = restore })
	ambientLeaseRefSyncBreakerReset()
	t.Cleanup(ambientLeaseRefSyncBreakerReset)

	transport := 0  // git verbs that actually reached the wire
	stalled := true // flipped when the node's credentials are "fixed"
	run := func(ctx context.Context, _ string, args ...string) (string, int, error) {
		verb := ""
		if len(args) != 0 {
			verb = args[0]
		}
		switch verb {
		case "push", "fetch":
			transport++
			if stalled {
				<-ctx.Done() // the wedged credential helper: returns only with the context
				return "", -1, ctx.Err()
			}
			return "", 0, nil
		case "for-each-ref":
			// A non-empty local namespace, so the push direction is really attempted
			// rather than short-circuited as PushSkippedEmpty (leaseref/sync.go).
			return "refs/fak/locks/loop-region-loop\n", 0, nil
		default:
			return "", 0, nil
		}
	}
	store := leaseref.NewWithRunner(run, t.TempDir())

	// written=true plans BOTH steps, so one full crossing is two transport verbs.
	cross := func() loopdrive.LeaseRefSyncReport {
		return ambientLeaseRefSyncImpl(loopdrive.LeaseRefSyncSurfaceLoopDriveTick, store, "origin", true)
	}
	nonfatal := func(when string, report loopdrive.LeaseRefSyncReport) {
		t.Helper()
		if report.Fatal {
			t.Fatalf("%s: report is FATAL — a breaker that can fail a tick is worse than the cost it saves: %+v", when, report)
		}
	}
	wantTransport := func(when string, want int) {
		t.Helper()
		if transport != want {
			t.Fatalf("%s: %d transport verbs run, want %d", when, transport, want)
		}
	}

	// 1. The first crossing pays the budget in full — #5564's behaviour, unchanged.
	report := cross()
	nonfatal("first crossing", report)
	wantTransport("first crossing", 2)
	if report.Outcome != loopdrive.LeaseRefSyncDegraded {
		t.Fatalf("first crossing outcome = %q, want degraded: %+v", report.Outcome, report)
	}

	// 2. The next N crossings are SKIPPED: no git runs at all, and the skip is
	// SURFACED — degraded, with every planned step naming the breaker — so a node
	// cannot go quietly isolated from the fleet while its report reads clean.
	for i := 1; i <= ambientLeaseRefSyncBreakerSkips; i++ {
		when := fmt.Sprintf("skip %d/%d", i, ambientLeaseRefSyncBreakerSkips)
		report = cross()
		nonfatal(when, report)
		if transport != 2 {
			t.Fatalf("%s: transport verbs rose to %d — the breaker never opened, so a wedged node still re-buys the answer every crossing (#5569)", when, transport)
		}
		if report.Outcome != loopdrive.LeaseRefSyncDegraded {
			t.Fatalf("%s: outcome = %q, want degraded — a skipped sync must never read as 'synced fine': %+v", when, report.Outcome, report)
		}
		if len(report.Failures) != 2 {
			t.Fatalf("%s: failures = %d, want both planned steps reported as skipped: %+v", when, len(report.Failures), report.Failures)
		}
		for _, f := range report.Failures {
			if !strings.Contains(f, ambientLeaseRefSyncBreakerOpen) {
				t.Errorf("%s: failure %q does not name the breaker, so a reader cannot tell a skip from a crossing that really failed", when, f)
			}
		}
		if !strings.Contains(report.Summary, ambientLeaseRefSyncBreakerOpen) {
			t.Errorf("%s: summary %q does not name the breaker", when, report.Summary)
		}
	}

	// 3. Crossing N+1 is the PROBE: exactly one is let through. The transport is
	// still wedged, so it spends the budget again and re-arms the breaker.
	report = cross()
	nonfatal("probe while still wedged", report)
	wantTransport("probe while still wedged", 4)
	if report.Outcome != loopdrive.LeaseRefSyncDegraded {
		t.Fatalf("probe outcome = %q, want degraded: %+v", report.Outcome, report)
	}

	// 4. The credentials are fixed. The breaker is armed, so its remaining skips
	// are still spent — the declared, bounded cost of remembering.
	stalled = false
	for i := 1; i <= ambientLeaseRefSyncBreakerSkips; i++ {
		report = cross()
		nonfatal("skip after recovery", report)
		wantTransport("skip after recovery", 4)
	}

	// 5. The next probe finds a healthy transport, syncs for real, and CLOSES.
	report = cross()
	nonfatal("recovery probe", report)
	wantTransport("recovery probe", 6)
	if report.Outcome != loopdrive.LeaseRefSyncOK {
		t.Fatalf("recovery probe outcome = %q, want ok — a recovered transport must sync: %+v", report.Outcome, report)
	}

	// 6. ...and the crossing AFTER it syncs immediately: a successful probe closes
	// the breaker outright, so recovery costs one crossing, not another N.
	report = cross()
	nonfatal("after recovery", report)
	if transport != 8 {
		t.Fatalf("the crossing after a successful probe ran %d transport verbs, want 8: the breaker stayed armed on a verdict the transport already disproved", transport)
	}
	if report.Outcome != loopdrive.LeaseRefSyncOK {
		t.Fatalf("post-recovery outcome = %q, want ok: %+v", report.Outcome, report)
	}
}

// TestAmbientLeaseRefSyncBreakerIgnoresAnOrdinaryTransportFailure pins the OTHER
// half of the trip rule (#5569): only a SPENT BUDGET arms the breaker. An
// ordinary transport failure — no remote configured, an unknown remote, a
// rejected update — already degrades instantly and cheaply, so there is nothing
// to amortize, and a breaker there would only delay the recovery of a clone that
// just gained its remote. Every crossing below must still reach the wire.
func TestAmbientLeaseRefSyncBreakerIgnoresAnOrdinaryTransportFailure(t *testing.T) {
	ambientLeaseRefSyncBreakerReset()
	t.Cleanup(ambientLeaseRefSyncBreakerReset)

	transport := 0
	// A fast non-zero exit: git's shape for "no such remote", with no stall and no
	// deadline involved, so the budget is never spent.
	broken := func(_ context.Context, _ string, args ...string) (string, int, error) {
		verb := ""
		if len(args) != 0 {
			verb = args[0]
		}
		switch verb {
		case "push", "fetch":
			transport++
			return "", 128, nil
		case "for-each-ref":
			return "refs/fak/locks/loop-region-loop\n", 0, nil
		default:
			return "", 0, nil
		}
	}
	store := leaseref.NewWithRunner(broken, t.TempDir())

	for i := 1; i <= ambientLeaseRefSyncBreakerSkips+2; i++ {
		report := ambientLeaseRefSyncImpl(loopdrive.LeaseRefSyncSurfaceLoopDriveTick, store, "origin", true)
		if report.Fatal {
			t.Fatalf("crossing %d: transport failure must stay advisory: %+v", i, report)
		}
		if report.Outcome != loopdrive.LeaseRefSyncDegraded {
			t.Fatalf("crossing %d: outcome = %q, want degraded: %+v", i, report.Outcome, report)
		}
		if transport != 2*i {
			t.Fatalf("crossing %d ran %d transport verbs in total, want %d: a cheap transport failure tripped the breaker, so a clone that just gained its remote waits %d crossings to find out (#5569)",
				i, transport, 2*i, ambientLeaseRefSyncBreakerSkips)
		}
		for _, f := range report.Failures {
			if strings.Contains(f, ambientLeaseRefSyncBreakerOpen) {
				t.Fatalf("crossing %d reported a breaker skip %q for a failure the budget was never spent on", i, f)
			}
			if strings.Contains(f, "budget") {
				t.Fatalf("crossing %d blamed the budget for a plain non-zero git exit: %q", i, f)
			}
		}
	}
}

// TestLoopDriveEnsureRunsAmbientSyncAtBoundaries: a fresh hold's ensure() must
// drive the ambient sync BOTH before the decide read (fetch-only, written=false)
// AND after the fenced acquire succeeds (push, written=true), at the loop-drive
// tick surface — the actual #2302 wiring, witnessed through the test seam.
func TestLoopDriveEnsureRunsAmbientSyncAtBoundaries(t *testing.T) {
	initRegionTestRepo(t)

	type rec struct {
		surface loopdrive.LeaseRefSyncSurface
		remote  string
		written bool
	}
	var got []rec
	orig := ambientLeaseRefSync
	ambientLeaseRefSync = func(surface loopdrive.LeaseRefSyncSurface, store *leaseref.Store, remote string, written bool) loopdrive.LeaseRefSyncReport {
		got = append(got, rec{surface, remote, written})
		return loopdrive.LeaseRefSyncReport{Outcome: loopdrive.LeaseRefSyncOK}
	}
	t.Cleanup(func() { ambientLeaseRefSync = orig })

	spec := loopdrive.Spec{Loop: "region-loop", Lane: "gateway"}
	hold := newLoopDriveRegionHold(loopDriveOptions{}, spec)
	defer hold.release()
	if refuse, err := hold.ensure(time.Now()); err != nil || refuse != nil {
		t.Fatalf("fresh acquire must hold: refuse=%+v err=%v", refuse, err)
	}
	if !hold.held {
		t.Fatal("hold must be held after a clean acquire")
	}

	var sawFetchBeforeDecide, sawPushAfterWrite bool
	for _, c := range got {
		if c.surface != loopdrive.LeaseRefSyncSurfaceLoopDriveTick {
			t.Fatalf("sync surface = %q, want loop_drive_tick", c.surface)
		}
		if c.remote != "origin" {
			t.Fatalf("remote = %q, want origin", c.remote)
		}
		if !c.written {
			sawFetchBeforeDecide = true
		} else {
			sawPushAfterWrite = true
		}
	}
	if !sawFetchBeforeDecide {
		t.Fatalf("ensure never ran the before-decide fetch sync: %+v", got)
	}
	if !sawPushAfterWrite {
		t.Fatalf("ensure never ran the after-write push sync: %+v", got)
	}

	// A renew (second ensure on the held lease) must also publish the write.
	got = nil
	if refuse, err := hold.ensure(time.Now().Add(time.Minute)); err != nil || refuse != nil {
		t.Fatalf("renew must hold: refuse=%+v err=%v", refuse, err)
	}
	sawPushAfterWrite = false
	for _, c := range got {
		if c.written {
			sawPushAfterWrite = true
		}
	}
	if !sawPushAfterWrite {
		t.Fatalf("renew never ran the after-write push sync: %+v", got)
	}
}

// TestIntentClaimAndReleaseAmbientSync verifies that fak intent claim and release
// invoke the ambient sync boundary (fetch before decide, push after write) and
// preserve nonfatal degradation when the remote transport is degraded (#10849).
func TestIntentClaimAndReleaseAmbientSync(t *testing.T) {
	dir := initRegionTestRepo(t)

	type rec struct {
		surface loopdrive.LeaseRefSyncSurface
		written bool
	}
	var got []rec
	orig := ambientLeaseRefSync
	ambientLeaseRefSync = func(surface loopdrive.LeaseRefSyncSurface, store *leaseref.Store, remote string, written bool) loopdrive.LeaseRefSyncReport {
		got = append(got, rec{surface, written})
		return loopdrive.LeaseRefSyncReport{Outcome: loopdrive.LeaseRefSyncDegraded, Reason: loopdrive.ReasonLeaseRefSyncTransport, Fatal: false}
	}
	t.Cleanup(func() { ambientLeaseRefSync = orig })

	var stdout, stderr bytes.Buffer
	// 1. fak intent claim: should run fetch before decide and push after write.
	code := runIntent(&stdout, &stderr, []string{"claim", "--dir", dir, "--target", "issue #10849", "--holder", "worker-1", "--ttl", "300"})
	if code != 0 {
		t.Fatalf("intent claim failed with code %d: %s", code, stderr.String())
	}
	if len(got) != 2 {
		t.Fatalf("expected 2 sync calls for claim (fetch then push), got %d: %+v", len(got), got)
	}
	if got[0].surface != loopdrive.LeaseRefSyncSurfaceIntentClaim || got[0].written {
		t.Errorf("claim first sync = %+v, want surface intent_claim and written=false", got[0])
	}
	if got[1].surface != loopdrive.LeaseRefSyncSurfaceIntentClaim || !got[1].written {
		t.Errorf("claim second sync = %+v, want surface intent_claim and written=true", got[1])
	}

	// 2. Conflicting claim: should run fetch before decide, but NOT push since claim is refused.
	got = nil
	stdout.Reset()
	stderr.Reset()
	code = runIntent(&stdout, &stderr, []string{"claim", "--dir", dir, "--target", "issue #10849", "--holder", "worker-2", "--ttl", "300"})
	if code != 3 { // INTENT_COLLISION
		t.Fatalf("conflicting claim code = %d, want 3", code)
	}
	if len(got) != 1 || got[0].written {
		t.Fatalf("conflicting claim should only run fetch before decide: %+v", got)
	}

	// 3. fak intent release: should run fetch before decide and push after write.
	got = nil
	stdout.Reset()
	stderr.Reset()
	code = runIntent(&stdout, &stderr, []string{"release", "--dir", dir, "--target", "issue #10849"})
	if code != 0 {
		t.Fatalf("intent release failed with code %d: %s", code, stderr.String())
	}
	if len(got) != 2 {
		t.Fatalf("expected 2 sync calls for release (fetch then push), got %d: %+v", len(got), got)
	}
	if got[0].surface != loopdrive.LeaseRefSyncSurfaceIntentRelease || got[0].written {
		t.Errorf("release first sync = %+v, want surface intent_release and written=false", got[0])
	}
	if got[1].surface != loopdrive.LeaseRefSyncSurfaceIntentRelease || !got[1].written {
		t.Errorf("release second sync = %+v, want surface intent_release and written=true", got[1])
	}
}

// TestLeaserefAcquireReleaseRenewAmbientSync verifies that fak leaseref acquire, renew,
// and release invoke the ambient sync boundary (fetch before decide, push after write)
// and preserve nonfatal degradation when the remote transport is degraded (#10849).
func TestLeaserefAcquireReleaseRenewAmbientSync(t *testing.T) {
	dir := initRegionTestRepo(t)

	type rec struct {
		surface loopdrive.LeaseRefSyncSurface
		written bool
	}
	var got []rec
	orig := ambientLeaseRefSync
	ambientLeaseRefSync = func(surface loopdrive.LeaseRefSyncSurface, store *leaseref.Store, remote string, written bool) loopdrive.LeaseRefSyncReport {
		got = append(got, rec{surface, written})
		return loopdrive.LeaseRefSyncReport{Outcome: loopdrive.LeaseRefSyncDegraded, Reason: loopdrive.ReasonLeaseRefSyncTransport, Fatal: false}
	}
	t.Cleanup(func() { ambientLeaseRefSync = orig })

	var stdout, stderr bytes.Buffer
	// 1. fak leaseref acquire
	code := runLeaseref(&stdout, &stderr, []string{"acquire", "--dir", dir, "--id", "lease-10849", "--holder", "worker-1", "--ttl", "300"})
	if code != 0 {
		t.Fatalf("leaseref acquire failed with code %d: %s", code, stderr.String())
	}
	if len(got) != 2 {
		t.Fatalf("expected 2 sync calls for acquire (fetch then push), got %d: %+v", len(got), got)
	}
	if got[0].surface != loopdrive.LeaseRefSyncSurfaceLeaserefAcquire || got[0].written {
		t.Errorf("acquire first sync = %+v, want surface leaseref_acquire and written=false", got[0])
	}
	if got[1].surface != loopdrive.LeaseRefSyncSurfaceLeaserefAcquire || !got[1].written {
		t.Errorf("acquire second sync = %+v, want surface leaseref_acquire and written=true", got[1])
	}

	// 2. fak leaseref renew
	got = nil
	stdout.Reset()
	stderr.Reset()
	code = runLeaseref(&stdout, &stderr, []string{"renew", "--dir", dir, "--id", "lease-10849", "--holder", "worker-1", "--ttl", "600"})
	if code != 0 {
		t.Fatalf("leaseref renew failed with code %d: %s", code, stderr.String())
	}
	if len(got) != 2 {
		t.Fatalf("expected 2 sync calls for renew (fetch then push), got %d: %+v", len(got), got)
	}
	if got[0].surface != loopdrive.LeaseRefSyncSurfaceLeaserefRenew || got[0].written {
		t.Errorf("renew first sync = %+v, want surface leaseref_renew and written=false", got[0])
	}
	if got[1].surface != loopdrive.LeaseRefSyncSurfaceLeaserefRenew || !got[1].written {
		t.Errorf("renew second sync = %+v, want surface leaseref_renew and written=true", got[1])
	}

	// 3. fak leaseref release (fenced)
	got = nil
	stdout.Reset()
	stderr.Reset()
	code = runLeaseref(&stdout, &stderr, []string{"release", "--dir", dir, "--id", "lease-10849", "--holder", "worker-1"})
	if code != 0 {
		t.Fatalf("leaseref release failed with code %d: %s", code, stderr.String())
	}
	if len(got) != 2 {
		t.Fatalf("expected 2 sync calls for release (fetch then push), got %d: %+v", len(got), got)
	}
	if got[0].surface != loopdrive.LeaseRefSyncSurfaceLeaserefRelease || got[0].written {
		t.Errorf("release first sync = %+v, want surface leaseref_release and written=false", got[0])
	}
	if got[1].surface != loopdrive.LeaseRefSyncSurfaceLeaserefRelease || !got[1].written {
		t.Errorf("release second sync = %+v, want surface leaseref_release and written=true", got[1])
	}
}

// TestDispatchLaneLeaseAmbientSync verifies that acquireDispatchLaneLease invokes
// the ambient sync boundary with LeaseRefSyncSurfaceDispatchPreflight and preserves
// nonfatal degradation when the remote transport fails (#10849).
func TestDispatchLaneLeaseAmbientSync(t *testing.T) {
	dir := initRegionTestRepo(t)

	type rec struct {
		surface loopdrive.LeaseRefSyncSurface
		written bool
	}
	var got []rec
	orig := ambientLeaseRefSync
	ambientLeaseRefSync = func(surface loopdrive.LeaseRefSyncSurface, store *leaseref.Store, remote string, written bool) loopdrive.LeaseRefSyncReport {
		got = append(got, rec{surface, written})
		return loopdrive.LeaseRefSyncReport{Outcome: loopdrive.LeaseRefSyncDegraded, Reason: loopdrive.ReasonLeaseRefSyncTransport, Fatal: false}
	}
	t.Cleanup(func() { ambientLeaseRefSync = orig })

	// 1. Clean acquire
	res := acquireDispatchLaneLease(dir, "lane-10849", "gateway", []string{"internal/gateway/**"}, 600, "")
	if acquired, _ := res["acquired"].(bool); !acquired {
		t.Fatalf("dispatch lane lease acquire should succeed despite degraded sync: %+v", res)
	}
	if len(got) != 2 {
		t.Fatalf("expected 2 sync calls for dispatch preflight acquire (fetch then push), got %d: %+v", len(got), got)
	}
	if got[0].surface != loopdrive.LeaseRefSyncSurfaceDispatchPreflight || got[0].written {
		t.Errorf("dispatch preflight first sync = %+v, want surface dispatch_preflight and written=false", got[0])
	}
	if got[1].surface != loopdrive.LeaseRefSyncSurfaceDispatchPreflight || !got[1].written {
		t.Errorf("dispatch preflight second sync = %+v, want surface dispatch_preflight and written=true", got[1])
	}

	// 2. Conflicting acquire
	got = nil
	res2 := acquireDispatchLaneLease(dir, "lane-10849-peer", "gateway", []string{"internal/gateway/**"}, 600, "")
	if refused, _ := res2["refused"].(bool); !refused {
		t.Fatalf("conflicting dispatch lane lease should refuse: %+v", res2)
	}
	if len(got) != 1 || got[0].written {
		t.Fatalf("refused acquire should only run fetch before decide: %+v", got)
	}
}
