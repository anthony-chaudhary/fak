package main

// leaseref_sync_ambient.go is the #2302 wiring: make the `fak leaseref sync`
// convergence AMBIENT at the loop-drive tick boundary, so cross-node lease
// visibility no longer depends on an operator remembering a refspec. The
// boundary MODEL lives in internal/loopdrive (LeaseRefSyncPlanForSurface —
// fetch-only before the decide read, push-only after the tick wrote a lease
// ref, both advisory); this file is the shell side that runs that plan against
// a real leaseref.Store.
//
// The dispatch tick's lane-lease acquire (acquireDispatchLaneLease) is the
// other #2302 surface; it shares this helper by passing the
// LeaseRefSyncSurfaceDispatchPreflight surface. It is wired in its own commit
// — the boundary is identical to the loop-drive one wired here.
//
// Sync is NONFATAL by design (internal/loopdrive.ReportLeaseRefSync): a node
// that cannot reach origin — single-box dev, air-gapped CI, a clone with no
// remote configured — continues on local lease evidence, the same fail-open
// posture as the dispatch tick's existing acquire. Transport failures fold
// into a LeaseRefSyncReport a caller may surface but never act on as a verdict.
//
// Nonfatal was only HALF true until #5564, because "the node cannot reach
// origin" and "the node cannot FINISH reaching origin" are different failures
// and only the first one used to end. This file is where the second one is
// given an end: it is the single funnel every ambient sync goes through, so the
// convergence budget below is the one place that decides how much wall-clock a
// best-effort sync may take from a tick (see ambientLeaseRefSyncBudget).
//
// #5569 adds the other half of that end: MEMORY. A bound that is re-paid every
// crossing is still a node re-learning, at full price and forever, an answer it
// already has. The breaker below (ambientLeaseRefSyncBreakerSkips) skips the
// plan for N crossings after one of them spent the whole budget, then lets
// exactly one probe through. It changes only how OFTEN this file asks; it never
// changes what an answer means, so the nonfatal posture above is untouched — a
// skipped crossing is reported as a nonfatal degradation, exactly like the
// failure it stands in for.

import (
	"context"
	"fmt"
	"os"
	"strings"
	"sync"
	"time"

	"github.com/anthony-chaudhary/fak/internal/leaseref"
	"github.com/anthony-chaudhary/fak/internal/loopdrive"
)

// ambientLeaseRefSync is the test seam: tests swap it to a recorder to witness
// that a tick ran the sync plan at the right boundary without driving real git.
// The default runs the real plan against the store.
var ambientLeaseRefSync = ambientLeaseRefSyncImpl

// ambientLeaseRefSyncRemote resolves the remote sync converges with. Defaults
// to "origin" (the protocol in docs/notes/MULTINODE-DEV-GROUNDWORK-2026-07-02.md);
// FAK_LEASEREF_REMOTE overrides it for setups whose canonical remote is named
// otherwise.
func ambientLeaseRefSyncRemote() string {
	if v := strings.TrimSpace(os.Getenv("FAK_LEASEREF_REMOTE")); v != "" {
		return v
	}
	return "origin"
}

// ambientLeaseRefSyncEnabled reports whether ambient sync is on. It is ON by
// default — #2302 makes convergence ambient, not operator muscle memory;
// FAK_LEASEREF_SYNC=off (also 0/false/no/disable) disables it for air-gapped
// runs or tests that want zero git transport.
func ambientLeaseRefSyncEnabled() bool {
	raw, ok := os.LookupEnv("FAK_LEASEREF_SYNC")
	if !ok {
		return true
	}
	switch strings.ToLower(strings.TrimSpace(raw)) {
	case "", "1", "on", "true", "yes", "enable", "enabled":
		return true
	}
	return false
}

// ambientLeaseRefSyncBudget bounds ONE ambient convergence call — the WHOLE
// plan, not one step (#5564). Before it existed this file handed store.Sync a
// context.Background(); Sync shells out to a real `git push` / `git fetch`, and
// internal/leaseref carries no deadline of its own, so an auth-stalled or
// black-holed remote pinned the loop-drive tick inside exec.Cmd.Run with
// nothing anywhere below it able to end the wait.
//
// WHY THE WHOLE PLAN, NOT ONE STEP PER BUDGET. LeaseRefSyncPlanForSurface
// returns up to two steps (fetch-before-decide, push-after-write) and
// loopDriveRegionHold.ensure runs this boundary twice a turn, so a per-step
// bound is not a bound on the tick: it multiplies by the plan's length, and it
// would grow silently the day a surface gains a third step. One deadline shared
// by every step of one call means the boundary costs at most this much whatever
// the plan's shape. A step that inherits an already-spent context fails at once
// and is reported — the honest answer after the convergence budget is gone.
//
// WHY THIS NUMBER. The repo already bounds this exact operation once: a single
// store.Sync push of refs/fak/locks/* to origin is bounded by leasePublishTimeout
// (30s, cmd/fak/leasewrite_endpoint.go), whose own doc records why it may be
// that wide — the publish runs on its own goroutine and holds no caller waiting.
// This call does hold a caller waiting; the tick blocks on it before it may
// decide and again after it writes. So the ambient budget is deliberately
// tighter than its asynchronous sibling's, while staying several times over the
// couple of seconds a healthy push of a handful of tiny blob refs costs. It is
// a CONVERGENCE bound, not a correctness one: the surface it guards is
// explicitly nonfatal, so spending the budget costs cross-node visibility until
// the next tick and never a wrong verdict — which is what makes erring on the
// short side the safe direction here.
//
// A var rather than a const ONLY so a test can shrink it and witness the
// give-up path without sleeping. Nothing in production assigns it.
var ambientLeaseRefSyncBudget = 10 * time.Second

// ambientLeaseRefSyncBreakerSkips is N: how many ambient convergence CROSSINGS
// are skipped after one crossing spent the whole budget, before exactly one
// probe crossing is let through again (#5569). The budget bounded a wedged
// transport's cost per crossing; it did not make the node REMEMBER, so a box
// with a stalled credential helper paid that bound on every crossing forever,
// re-learning an answer it already had.
//
// WHY A COUNT OF CROSSINGS, NOT A DURATION. This file has no scheduler and no
// clock of its own — the only event it observes is the boundary being crossed,
// which is exactly the thing whose cost is being amortized. A wall-clock
// cooldown would additionally have to guess a tick's period, which varies with
// the agent turn behind it (see loopDriveRegionTTLS's comment: a turn can be
// long). Note the unit honestly: loopDriveRegionHold.ensure crosses this
// boundary up to twice a turn (fetch-before-decide, push-after-write, and a
// renew-only turn crosses once), so N crossings is between N/2 and N turns.
//
// WHY THIS NUMBER. ambientLeaseRefSyncBudget's own doc above already supplies
// both terms: the budget is 10s, and it is that wide because it is "several
// times over the couple of seconds a healthy push of a handful of tiny blob
// refs costs". A breaker that admits one crossing in every N+1 spends at most
// one budget per N+1 crossings, so N=5 puts a WEDGED node at 10s/6 ≈ 1.7s of
// tick time per crossing — at or under what a HEALTHY crossing costs. That is
// the whole target: once the node has learned the remote is not answering, an
// unreachable remote must not cost the tick more than a reachable one.
//
// WHY FIXED, NOT BACKING OFF. internal/bgloop doubles a failing loop's retry
// schedule up to backoffMax because there the retry is cheap and the WAIT is
// the cost. Here it is the other way round: skipping is free and the retry is
// the whole 10s. Once the amortized cost is already at the healthy floor,
// doubling buys nothing measurable, while it costs without bound the one thing
// this surface exists to provide — cross-node lease visibility — on a node
// whose credentials get fixed forty minutes in. A fixed N bounds BOTH the
// re-learning cost and the recovery latency; bgloop's own "a clean tick resets
// it" is the half of that precedent kept here (see the record path below).
//
// A const, unlike the budget above: a skipped crossing costs no wall-clock, so
// a test can exercise the real N by crossing N times instead of shrinking it.
const ambientLeaseRefSyncBreakerSkips = 5

// ambientLeaseRefSyncBreakerOpen is the stable token an open breaker puts in the
// report, so a skip is never mistaken for a clean sync by a reader (or a test)
// matching on text. It is deliberately distinct from the budget-spent wording
// ambientLeaseRefSyncStepErr uses for a crossing that really ran.
const ambientLeaseRefSyncBreakerOpen = "lease-ref sync skipped: convergence breaker open"

// ambientLeaseRefSyncBreaker is the breaker's IN-PROCESS state: how many more
// ambient crossings to skip, and the remote that verdict is about.
//
// IN-PROCESS ON PURPOSE, NOT PERSISTED. What is remembered is one process's
// own measurement of its own transport, and the thing being protected is a
// long-running loop's tick — the only caller that crosses this boundary often
// enough to care. Persisting it would let a fresh `fak` invocation inherit a
// PEER's verdict about the network and skip its first sync on hearsay, which is
// a far bigger claim than the one measurement supports. The cost of that choice
// is explicit: a fleet of short-lived one-shot invocations gets no protection
// at all, each paying one budget and exiting. That is the correct trade — such
// a process pays the bound once, which #5564 already made survivable, whereas a
// loop pays it every turn, which is the defect this closes.
//
// Keyed by remote rather than by surface: the fact learned is "this node's
// transport to THIS remote is not answering", which is a node fact, not a
// per-surface one, and every ambient surface funnels through this one file. A
// call naming a different remote resets the breaker — nothing has been measured
// about that transport. One field rather than a map because the ambient remote
// is process-wide (ambientLeaseRefSyncRemote reads one env var), so at most one
// is live at a time.
var ambientLeaseRefSyncBreaker struct {
	mu     sync.Mutex
	remote string
	skips  int
}

// ambientLeaseRefSyncBreakerTake consumes one skip if the breaker is open for
// remote, reporting how many remain after this one. A false means the caller
// must run the plan for real — either the breaker is closed, or this is the one
// probe crossing that re-measures the transport.
func ambientLeaseRefSyncBreakerTake(remote string) (skip bool, remaining int) {
	ambientLeaseRefSyncBreaker.mu.Lock()
	defer ambientLeaseRefSyncBreaker.mu.Unlock()
	if ambientLeaseRefSyncBreaker.remote != remote {
		ambientLeaseRefSyncBreaker.remote = remote
		ambientLeaseRefSyncBreaker.skips = 0
	}
	if ambientLeaseRefSyncBreaker.skips <= 0 {
		return false, 0
	}
	ambientLeaseRefSyncBreaker.skips--
	return true, ambientLeaseRefSyncBreaker.skips
}

// ambientLeaseRefSyncBreakerRecord folds the outcome of a crossing that really
// ran. ONLY a spent budget arms the breaker.
//
// WHY NOT ANY TRANSPORT FAILURE. The two are different failures with opposite
// economics. A spent budget means the remote is answering slowly or not at all,
// and re-asking costs the tick the full bound every time — that is the thing
// worth remembering. An ordinary transport failure (no remote configured, an
// unknown remote, a rejected update) already degrades instantly and cheaply:
// it costs a fast non-zero git exit, so a breaker would save nothing and would
// instead delay the recovery of a clone that just gained its remote. That the
// two are distinguishable at all is #5564's doing — ambientLeaseRefSyncStepErr
// exists precisely because git cannot name a deadline this file set — so the
// signal a breaker needs is already on the wire.
//
// ANY OTHER OUTCOME CLOSES IT, including an ordinary failure on a probe
// crossing: bgloop's "a clean tick resets it", applied to the only evidence
// this file has.
func ambientLeaseRefSyncBreakerRecord(remote string, budgetSpent bool) {
	ambientLeaseRefSyncBreaker.mu.Lock()
	defer ambientLeaseRefSyncBreaker.mu.Unlock()
	ambientLeaseRefSyncBreaker.remote = remote
	if budgetSpent {
		ambientLeaseRefSyncBreaker.skips = ambientLeaseRefSyncBreakerSkips
		return
	}
	ambientLeaseRefSyncBreaker.skips = 0
}

// ambientLeaseRefSyncBreakerReset closes the breaker and forgets the remote it
// measured. Process-global state is shared by every test in this package, so a
// test that trips the breaker must clear it rather than leave the next test
// silently skipping the transport it means to exercise.
func ambientLeaseRefSyncBreakerReset() {
	ambientLeaseRefSyncBreaker.mu.Lock()
	defer ambientLeaseRefSyncBreaker.mu.Unlock()
	ambientLeaseRefSyncBreaker.remote = ""
	ambientLeaseRefSyncBreaker.skips = 0
}

// ambientLeaseRefSyncSkippedReport renders an open breaker as a report in the
// SAME shape a crossing that really failed produces, by folding one skipped
// attempt per planned step through loopdrive.ReportLeaseRefSync.
//
// THE SKIP IS VISIBLE, which is the point: a silently skipped sync is how a node
// ends up quietly isolated from the fleet for an hour. The fold's Outcome is
// therefore DEGRADED, never OK — a reader that only checks the outcome still
// sees that this boundary did not converge — and each planned step names the
// breaker and how many crossings remain before the next probe, so the skip is
// also distinguishable from a crossing that really ran and failed. The fold also
// supplies Fatal=false, which is what keeps a breaker from ever becoming a
// verdict the tick acts on.
func ambientLeaseRefSyncSkippedReport(steps []loopdrive.LeaseRefSyncStep, remaining int) loopdrive.LeaseRefSyncReport {
	attempts := make([]loopdrive.LeaseRefSyncAttempt, 0, len(steps))
	for _, step := range steps {
		attempts = append(attempts, loopdrive.LeaseRefSyncAttempt{
			Step: step,
			Err: fmt.Sprintf("%s (budget %s already spent on %s; %d more crossing(s) skipped before the next probe)",
				ambientLeaseRefSyncBreakerOpen, ambientLeaseRefSyncBudget, step.Remote, remaining),
		})
	}
	report := loopdrive.ReportLeaseRefSync(attempts)
	if len(attempts) != 0 {
		report.Summary = ambientLeaseRefSyncBreakerOpen + "; continuing with local lease evidence"
	}
	return report
}

// ambientLeaseRefSyncImpl runs the surface's sync plan against store and folds
// the per-step transport results into one advisory report. It never turns a
// transport failure into a fatal verdict — the caller proceeds on local lease
// evidence regardless.
func ambientLeaseRefSyncImpl(surface loopdrive.LeaseRefSyncSurface, store *leaseref.Store, remote string, written bool) loopdrive.LeaseRefSyncReport {
	if !ambientLeaseRefSyncEnabled() {
		return loopdrive.LeaseRefSyncReport{Outcome: loopdrive.LeaseRefSyncOK, Summary: "lease-ref sync disabled (FAK_LEASEREF_SYNC=off)"}
	}
	if strings.TrimSpace(remote) == "" {
		remote = ambientLeaseRefSyncRemote()
	}
	steps := loopdrive.LeaseRefSyncPlanForSurface(surface, loopdrive.LeaseRefSyncPlanInput{Remote: remote, LeaseRefsWritten: written})
	// #5569: a node that already spent a whole budget on this remote has LEARNED
	// that the transport is not answering. Skip the plan rather than re-buy that
	// answer at full price every crossing; the skip is reported, not swallowed.
	if skip, remaining := ambientLeaseRefSyncBreakerTake(remote); skip {
		return ambientLeaseRefSyncSkippedReport(steps, remaining)
	}
	// ONE deadline for the whole plan, so the tick's exposure is this budget and
	// not this budget times however many steps the surface plans.
	ctx, cancel := context.WithTimeout(context.Background(), ambientLeaseRefSyncBudget)
	defer cancel()
	budgetSpent := false
	attempts := make([]loopdrive.LeaseRefSyncAttempt, 0, len(steps))
	for _, step := range steps {
		push, fetch, err := loopdrive.LeaseRefSyncDirections(step)
		if err != nil {
			attempts = append(attempts, loopdrive.LeaseRefSyncAttempt{Step: step, Err: err.Error()})
			continue
		}
		if _, err := store.Sync(ctx, step.Remote, push, fetch); err != nil {
			// The same condition ambientLeaseRefSyncStepErr attributes the failure
			// on: a failed step under an expired budget is the ONE signal that arms
			// the breaker (see ambientLeaseRefSyncBreakerRecord).
			if ctx.Err() != nil {
				budgetSpent = true
			}
			attempts = append(attempts, loopdrive.LeaseRefSyncAttempt{Step: step, Err: ambientLeaseRefSyncStepErr(ctx, err)})
			continue
		}
		attempts = append(attempts, loopdrive.LeaseRefSyncAttempt{Step: step})
	}
	ambientLeaseRefSyncBreakerRecord(remote, budgetSpent)
	return loopdrive.ReportLeaseRefSync(attempts)
}

// ambientLeaseRefSyncStepErr names the one failure the raw leaseref error cannot
// name for itself: a spent convergence budget. leaseref reports transport
// outcomes in git's own terms, and git does not know it was killed by a
// deadline — a cancelled push surfaces as a plain non-zero exit, and a step that
// starts after the budget is already gone surfaces as "git not executable". Both
// would blame the remote for a deadline THIS file set, so the ledger reader
// would chase a network that was fine. Every other error crosses unchanged; the
// underlying text is kept either way, since it still says which direction and
// which remote failed.
func ambientLeaseRefSyncStepErr(ctx context.Context, err error) string {
	if ctx.Err() != nil {
		return fmt.Sprintf("lease-ref sync budget %s spent: %v", ambientLeaseRefSyncBudget, err)
	}
	return err.Error()
}

// syncLoopDriveTickLeaseRefs runs the loop-drive-tick sync boundary against the
// hold's store. written is false before the decide read (fetch-only: converge
// peer lease refs so admission sees them) and true after the hold acquired or
// renewed its lease ref (push-only: publish the local write).
func syncLoopDriveTickLeaseRefs(store *leaseref.Store, written bool) loopdrive.LeaseRefSyncReport {
	return ambientLeaseRefSync(loopdrive.LeaseRefSyncSurfaceLoopDriveTick, store, ambientLeaseRefSyncRemote(), written)
}
