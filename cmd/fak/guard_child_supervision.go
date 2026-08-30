package main

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"runtime"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/cachevalueledger"
	"github.com/anthony-chaudhary/fak/internal/gateway"
	"github.com/anthony-chaudhary/fak/internal/goalpark"
	"github.com/anthony-chaudhary/fak/internal/harnessres"
	"github.com/anthony-chaudhary/fak/internal/journal"
	"github.com/anthony-chaudhary/fak/internal/procguard"
	"github.com/anthony-chaudhary/fak/internal/resume"
	"github.com/anthony-chaudhary/fak/internal/session"
	"github.com/anthony-chaudhary/fak/internal/toolprocgate"
	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

func guardRecordWireRetry(j *journal.Journal, stderr io.Writer, agentName, traceID string, runErr error, state *os.ProcessState, started time.Time, retries int) {
	appendGuardChildExitWitness(j, agentName, traceID, runErr, state, started)
	guardEmitRestartHop(j, stderr, agentName, traceID, guardWireRetryHop(traceID, agentName, retries))
	time.Sleep(guardCrashRestartDelay(retries))
}

func guardRecordCrashRestart(j *journal.Journal, stderr io.Writer, agentName, traceID string, runErr error, state *os.ProcessState, started time.Time, retries int) {
	appendGuardChildExitWitness(j, agentName, traceID, runErr, state, started)
	guardEmitRestartHop(j, stderr, agentName, traceID, guardCrashRestartHop(traceID, agentName, retries))
}

func guardAdoptRecoveredCommand(command *[]string, next []string, ok bool) bool {
	if ok {
		*command = next
	}
	return ok
}

func guardRecoverCapCrash(command *[]string, runErr error, agentName string, childStarted time.Time, quiet bool, maxWait time.Duration, stderr io.Writer) bool {
	next, recovered := guardMaybeRecoverCapCrash(runErr, *command, agentName, childStarted, quiet, maxWait, nil, nil, stderr)
	return guardAdoptRecoveredCommand(command, next, recovered)
}

// guardGoalParked answers "is THIS account still walled off this goal?" — never
// the account-blind "is this lane parked?" it used to answer. Every branch below
// consults it BEFORE rotation.rotateAfterExit, so a positive verdict
// short-circuits account rotation entirely; when the verdict was lane-scoped
// that meant one account's 1h Retry-After stopped every account on the lane for
// as long as the park lasted, while the dispatcher kept dispatching into it and
// each child was killed mid-tool_use with no report and no commit. The scoping
// (and the retire-when-due claim) lives in goalpark.Store.Resolve so the guard
// cannot re-acquire an ad-hoc, account-blind condition here; a blank or foreign
// account falls through to rotation, which is the whole point.
func guardGoalParked() (goalpark.Record, bool) {
	goal, account := parkGoalIdentity()
	if goal == "" {
		return goalpark.Record{}, false
	}
	// Name this guard in the park's exactly-once claim ledger, so `fak goal-park
	// status` shows WHICH supervisor resumed a due park rather than an anonymous
	// claim.
	supervisor := "fak-guard"
	if account != "" {
		supervisor += "/" + account
	}
	return goalParkStore().Resolve(goal, account, supervisor, time.Now())
}

// guardParkProbeStatus renders WHY a park declined this run rather than only
// that it did. A park that reports nothing but "parked until T" is unfalsifiable
// from the log: an operator reading a torn-down worker cannot tell whether the
// wall is still being tested or has sealed shut. The probe ledger is the whole
// anti-self-seal contract (goalpark.AdmitProbe), so it belongs on the same line
// as the teardown it explains.
func guardParkProbeStatus(rec goalpark.Record, now time.Time) string {
	next := "budget spent"
	if rec.Probes < goalpark.ProbeBudget {
		since := rec.LastProbeAt
		if since == 0 {
			since = rec.ParkedAt
		}
		next = time.Until(time.Unix(since, 0).Add(rec.ProbeInterval())).Round(time.Second).String()
		if !now.Before(time.Unix(since, 0).Add(rec.ProbeInterval())) {
			next = "open"
		}
	}
	return fmt.Sprintf("probes=%d/%d next_probe=%s", rec.Probes, goalpark.ProbeBudget, next)
}

func runGuardChildAndReport(command []string, injected [][2]string, pinUpstream bool, credPath string, rotation *guardRotationRuntime, spawnMeta guardChildSpawnMetadata, codexSessionStatePath string, wireErrors *guardWireErrorGauge, srv *gateway.Server, cancel context.CancelFunc, serveErr <-chan error, quiet bool, auditJournal *journal.Journal, auditSeq0 uint64, guardTraceID, agentName, provider string, dojoMode bool, sampler *harnessres.Sampler, dumpStartupOnLaunchFail bool, startupProgress *guardStartupProgress) {
	// The startup renderer created the card and queued its control replies. Bind its
	// periodic status fold to the live gateway before the child starts; finalizeOutcome
	// below stops the updater and replaces the root with the terminal state.
	guardSessionCardHandle.startUpdater(srv)
	spawnBroker := toolprocgate.NewSpawnBroker()
	// #4686 in-place crash restart: a generic harness crash (OOM/SIGNAL/NONZERO_EXIT) matches none of
	// the narrow recovery seams above, so without this it would tear the guard master down. Bounded by
	// the bounded crashLimit (explicit 0 = off) so a systematic crash is surfaced, not masked.
	crashRestarts := 0
	crashLimit := guardCrashRestartLimit()
	var crashProgressHead string
	crashNoProgress := 0
	crashNoProgressLimit := guardCrashNoProgressLimit(crashLimit)
	resourceRetries := newGuardResourceRetryState()
	wireRetries := 0
	wireLimit := guardWireRetryLimit()
	for {
		startupProgress.Phase("broker/preparing child")
		_, child, err := launchGuardChildWithBroker(command, injected, pinUpstream, spawnMeta, spawnBroker, rotation.launcher())
		if err != nil {
			startupProgress.Abort()
			guardDumpStartupReportOnLaunchFail(os.Stderr, srv, dumpStartupOnLaunchFail)
			finishGuardChildAndReport(err, nil, srv, cancel, serveErr, quiet, auditJournal, auditSeq0, guardTraceID, agentName, provider, dojoMode, sampler)
			return
		}
		childStderr := guardCaptureChildStderr(child, agentName)
		childStdout := guardCaptureChildStdout(child, command, agentName)
		maybeStartGuardChildHarnessTerminalRestorePulseForPlan(spawnMeta.LaunchPlan)
		childStarted := time.Now()
		srv.BeginChildStartup(childStarted)
		rotationEvidenceBefore := srv.RotationEvidenceSnapshot()
		startupProgress.Phase("OS process start")
		resourcePolicy := guardResourcePolicyConfigured()
		job, startErr := windowgate.StartManagedAgentInNewJob(child, windowgate.ManagedJobConfig{MemoryLimitBytes: resourcePolicy.MaxTreeBytes})
		if startErr != nil {
			startupProgress.Abort()
			terminalGuardChild(child, startErr, "launch_failed")
			finishGuardChildAndReport(startErr, nil, srv, cancel, serveErr, quiet, auditJournal, auditSeq0, guardTraceID, agentName, provider, dojoMode, sampler)
			return
		}
		startupProgress.Phase("child registration")
		if err := startBoundGuardRegistration(child); err != nil {
			startupProgress.Abort()
			_ = child.Process.Kill()
			_, _ = child.Process.Wait()
			_ = job.Close()
			finishGuardChildAndReport(err, child.ProcessState, srv, cancel, serveErr, quiet, auditJournal, auditSeq0, guardTraceID, agentName, provider, dojoMode, sampler)
			return
		}
		startupProgress.Started()
		lifecycle := startCrashJournalPulse(guardTraceID, child.Process.Pid)
		wait := make(chan error, 1)
		go func() { wait <- child.Wait() }()
		var runErr error
		resourceStop := make(chan struct{})
		resourcePolicy.Stop = resourceStop
		resourceEvents := startGuardChildResourceMonitor(child.Process.Pid, guardTraceID, agentName, resourcePolicy)
		select {
		case event := <-resourceEvents:
			markGuardChildTerminalIntent(child, "resource_limit")
			_ = job.Close()
			runErr = stopGuardChild(child, wait, 0)
			lifecycle.finish(false)
			terminalGuardChild(child, runErr, "resource_limit")
			receiptErr := guardWriteResourceReceipt(event, guardTraceID, agentName, child.Process.Pid)
			fmt.Fprintf(os.Stderr, "fak guard: reaped child resource runaway: %s\n", event.Reason)
			resourceErr := fmt.Errorf("child resource limit: %s", event.Reason)
			appendGuardChildExitWitnessWithReason(auditJournal, agentName, guardTraceID, resourceErr, child.ProcessState, childStarted, event.Resource.Reason)
			if receiptErr != nil {
				resourceErr = fmt.Errorf("child resource receipt failed after containment: %w", receiptErr)
				fmt.Fprintf(os.Stderr, "fak guard: %v\n", resourceErr)
				finishGuardChildAndReport(resourceErr, child.ProcessState, srv, cancel, serveErr, quiet, auditJournal, auditSeq0, guardTraceID, agentName, provider, dojoMode, sampler)
				return
			}
			verdict := resourceRetries.decide(event, agentName, sessionStartSHA())
			if verdict.Action == guardResourceRetryRelaunch {
				nextCommand, reattachErr := guardResourceReattachCommand(command, agentName, codexSessionStatePath, guardTraceID)
				if reattachErr != nil {
					guardRecordResourceReattachUnavailable(auditJournal, agentName, guardTraceID)
					fmt.Fprintln(os.Stderr, guardResourceReattachUnavailableStatus(agentName, guardTraceID, reattachErr))
					resourceErr = fmt.Errorf("%s: %w", guardResourceReattachUnavailable, resourceErr)
					finishGuardChildAndReport(resourceErr, child.ProcessState, srv, cancel, serveErr, quiet, auditJournal, auditSeq0, guardTraceID, agentName, provider, dojoMode, sampler)
					return
				}
				guardReportResourceRestart(os.Stderr, agentName, verdict, nextCommand)
				time.Sleep(verdict.Delay)
				guardRecordResourceRestart(auditJournal, os.Stderr, agentName, guardTraceID, verdict.Attempt)
				command = nextCommand
				continue
			}
			if verdict.Action == guardResourceRetryExhausted {
				guardRecordResourceRestartGiveUp(auditJournal, agentName, guardTraceID)
				fmt.Fprintln(os.Stderr, guardResourceRestartGiveUpStatus(verdict, guardTraceID))
				resourceErr = fmt.Errorf("%s: %w", guardResourceRestartExhaustedReason, resourceErr)
			} else if verdict.Cause == guardResourceRestartCauseNoReattach {
				guardRecordResourceReattachUnavailable(auditJournal, agentName, guardTraceID)
				fmt.Fprintln(os.Stderr, guardResourceReattachUnavailableStatus(agentName, guardTraceID, nil))
				resourceErr = fmt.Errorf("%s: %w", guardResourceReattachUnavailable, resourceErr)
			}
			finishGuardChildAndReport(resourceErr, child.ProcessState, srv, cancel, serveErr, quiet, auditJournal, auditSeq0, guardTraceID, agentName, provider, dojoMode, sampler)
			return
		case runErr = <-wait:
			close(resourceStop)
		}
		lifecycle.finish(runErr == nil)
		_ = job.Close()
		terminalGuardChild(child, runErr, "")
		if rec, parked := guardGoalParked(); parked {
			fmt.Fprintf(os.Stderr, "fak guard: goal parked outside active context budget until %d; reason=%s; %s; next=%s\n", rec.ParkedUntil, rec.Reason, guardParkProbeStatus(rec, time.Now()), rec.NextAction)
			// #5862: a bare `break` here left the loop with NO teardown at all — no
			// witness row, no journal flush/Close, no refusal carry-forward sidecar, and
			// no gateway cancel — so a parked session's journal stayed zero-byte and its
			// cause was unrecoverable. Tear down exactly like the supervised loop's parked
			// branches: witness first (finishGuardChildAndReport closes the journal), then
			// report. runErr is deliberately dropped for the same reason the supervised
			// sibling drops it — a park is a scheduled resume, not a session failure, so
			// the process keeps the exit-0 semantics the `break` already had.
			appendGuardChildExitWitness(auditJournal, agentName, guardTraceID, nil, child.ProcessState, childStarted)
			finishGuardChildAndReport(nil, child.ProcessState, srv, cancel, serveErr, quiet, auditJournal, auditSeq0, guardTraceID, agentName, provider, dojoMode, sampler)
			return
		}
		if guardRefuseCodexCLIUsage(runErr, child.ProcessState, agentName, guardTraceID, childStderr.String(), childStarted, auditJournal, os.Stderr) ||
			guardRefuseCodexInvalidJSON(runErr, child.ProcessState, agentName, guardTraceID, childStdout.String(), childStarted, auditJournal, os.Stderr) {
			finishGuardChildAndReport(runErr, child.ProcessState, srv, cancel, serveErr, quiet, auditJournal, auditSeq0, guardTraceID, agentName, provider, dojoMode, sampler)
			return
		}
		if next, ok := guardMaybeRecoverAuthCrash(runErr, command, credPath, agentName, quiet, os.Stderr); ok {
			command = next
			continue
		}
		if next, ok := guardMaybeRetryTransientWireCrash(runErr, child.ProcessState, command, agentName, wireErrors.Consume(time.Now()), wireRetries, wireLimit, true, nil); ok {
			wireRetries++
			guardRecordWireRetry(auditJournal, os.Stderr, agentName, guardTraceID, runErr, child.ProcessState, childStarted, wireRetries)
			command = next
			continue
		}
		if nextCommand, nextInjected, ok := rotation.rotateAfterExit(runErr, guardRotationEvidenceSince(rotationEvidenceBefore, srv.RotationEvidenceSnapshot()), command, injected, auditJournal, guardTraceID, os.Stderr); ok {
			command, injected = nextCommand, nextInjected
			continue
		}
		if guardRecoverCapCrash(&command, runErr, agentName, childStarted, quiet, 0, os.Stderr) {
			continue
		}
		if class, code, ok := guardMaybeRestartOnCrash(runErr, child.ProcessState, crashRestarts, crashLimit); ok {
			guardMaybeLaunchCrashRSI(os.Stderr, guardTraceID, agentName, class, code, crashRestarts)
			crashRestarts++
			var reap bool
			crashProgressHead, crashNoProgress, reap = guardCrashNoProgressStep(crashProgressHead, sessionStartSHA(), crashNoProgress, crashNoProgressLimit)
			if reap {
				guardRecordCrashRestartGiveUp(auditJournal, agentName, guardTraceID)
				fmt.Fprintln(os.Stderr, guardCrashRestartGiveUpStatus(crashNoProgressLimit, guardTraceID))
				finishGuardChildAndReport(runErr, child.ProcessState, srv, cancel, serveErr, quiet, auditJournal, auditSeq0, guardTraceID, agentName, provider, dojoMode, sampler)
				return
			}
			guardReportCrashRestart(os.Stderr, agentName, class, code, crashRestarts, crashLimit, command)
			time.Sleep(guardCrashRestartDelay(crashRestarts))
			guardRecordCrashRestart(auditJournal, os.Stderr, agentName, guardTraceID, runErr, child.ProcessState, childStarted, crashRestarts)
			command = guardRestartRelaunchCommand(command, agentName)
			continue
		}
		if guardChildIsLaunchFailure(runErr) {
			guardDumpStartupReportOnLaunchFail(os.Stderr, srv, dumpStartupOnLaunchFail)
		}
		appendGuardChildExitWitness(auditJournal, agentName, guardTraceID, runErr, child.ProcessState, childStarted)
		finishGuardChildAndReport(runErr, child.ProcessState, srv, cancel, serveErr, quiet, auditJournal, auditSeq0, guardTraceID, agentName, provider, dojoMode, sampler)
		return
	}
}

// guardTimeBudgetTickInterval is the coarse cadence at which the supervision loop
// polls the session wall-clock envelope. 15s is fine-grained enough that a
// --max-duration overshoot is bounded to one tick, and coarse enough to add no
// measurable overhead to a guarded run.
const guardTimeBudgetTickInterval = 15 * time.Second

// guardTimeBudgetExhausted is the production caller of the session wall-clock gate
// that issue #2229 found missing: the supervision ticker asks it, once per tick,
// whether traceID's --max-duration envelope has elapsed as of now. It returns
// (true, TIME_BUDGET_EXHAUSTED) only on a genuine wall-clock exhaustion — an
// unbounded (--max-duration 0), still-within-budget, paused, or terminal session
// reports (false, "") and is left untouched, so an unconfigured envelope never
// stops a run. The transition to Draining/Stopped and the final elapsed-time fold
// are done by DecideTimeBudget itself (see internal/session/timebudget.go); this
// wrapper only classifies its verdict for the loop, and is the seam #2229's
// enforcement test drives.
func guardTimeBudgetExhausted(sessions *session.Table, traceID string, now time.Time) (bool, string) {
	if sessions == nil || strings.TrimSpace(traceID) == "" {
		return false, ""
	}
	v := sessions.DecideTimeBudget(traceID, now)
	if v.Stop && v.Reason == session.ReasonTimeBudgetExhausted {
		return true, v.Reason
	}
	return false, ""
}

func runGuardChildSupervisedAndReport(command []string, injected [][2]string, pinUpstream bool, credPath string, rotation *guardRotationRuntime, spawnMeta guardChildSpawnMetadata, codexSessionStatePath string, restarter *guardBudgetRestarter, wireErrors *guardWireErrorGauge, srv *gateway.Server, cancel context.CancelFunc, serveErr <-chan error, quiet bool, auditJournal *journal.Journal, auditSeq0 uint64, guardTraceID, agentName, provider string, dojoMode bool, sampler *harnessres.Sampler, dumpStartupOnLaunchFail bool, startupProgress *guardStartupProgress) {
	// Same live card as the unsupervised path; child restarts stay one session and one
	// Slack thread, so the updater spans the whole supervision loop and finalizes once.
	guardSessionCardHandle.startUpdater(srv)
	spawnBroker := toolprocgate.NewSpawnBroker()
	var extraEnv [][2]string
	restarts := 0
	// #4609 progress-aware early reap: track HEAD across budget restarts so a worker that keeps
	// restarting WITHOUT landing a commit is reaped early, while a committing worker keeps its
	// full runway. progressHead is the HEAD SHA at the last restart that made progress (seeded at
	// the loop's start SHA); "" when git offers no signal (e.g. a generic `fak guard` off a repo),
	// which disables the reap — as does noProgressLimit == 0.
	noProgressRestarts := 0
	noProgressLimit := guardNoProgressRestartLimit()
	equivalentRestarts := guardEquivalentRestarts{}
	var progressHead string
	// #4686 in-place crash restart: a generic harness crash (OOM/SIGNAL/NONZERO_EXIT) matches none of
	// the narrow recovery seams above, so without this it would tear the guard master down. Bounded by
	// the bounded crashLimit (explicit 0 = off) so a systematic crash is surfaced, not masked.
	crashRestarts := 0
	crashLimit := guardCrashRestartLimit()
	var crashProgressHead string
	crashNoProgress := 0
	crashNoProgressLimit := guardCrashNoProgressLimit(crashLimit)
	resourceRetries := newGuardResourceRetryState()
	wireRetries := 0
	wireLimit := guardWireRetryLimit()
	// Wall-clock enforcement (#2229): poll the session time budget on a coarse ticker so a
	// --max-duration envelope is actually ENFORCED here, not merely armed/persisted/displayed.
	// guardTimeBudgetExhausted is a no-op for an unbounded/paused/still-fine session, so a run
	// without a configured envelope is untouched; on a TIME_BUDGET_EXHAUSTED verdict the child
	// is stopped through the existing stopGuardChild path and the session is reported.
	budgetTicker := time.NewTicker(guardTimeBudgetTickInterval)
	defer budgetTicker.Stop()
	stopLoginHijackWatch := guardStartLoginHijackWatch(credPath, os.Stderr)
	defer stopLoginHijackWatch()
	relaunchFiles, err := captureGuardRelaunchFiles(command)
	if err != nil {
		startupProgress.Abort()
		finishGuardChildAndReport(err, nil, srv, cancel, serveErr, quiet, auditJournal, auditSeq0, guardTraceID, agentName, provider, dojoMode, sampler)
		return
	}
	for {
		if err := relaunchFiles.ensure(); err != nil {
			startupProgress.Abort()
			finishGuardChildAndReport(err, nil, srv, cancel, serveErr, quiet, auditJournal, auditSeq0, guardTraceID, agentName, provider, dojoMode, sampler)
			return
		}
		startupProgress.Phase("broker/preparing child")
		_, child, err := launchGuardChildWithBroker(command, injected, pinUpstream, spawnMeta, spawnBroker, nil, extraEnv...)
		wait := make(chan error, 1)
		if err != nil {
			startupProgress.Abort()
			guardDumpStartupReportOnLaunchFail(os.Stderr, srv, dumpStartupOnLaunchFail)
			finishGuardChildAndReport(err, nil, srv, cancel, serveErr, quiet, auditJournal, auditSeq0, guardTraceID, agentName, provider, dojoMode, sampler)
			return
		}
		childStderr := guardCaptureChildStderr(child, agentName)
		childStdout := guardCaptureChildStdout(child, command, agentName)
		maybeStartGuardChildHarnessTerminalRestorePulseForPlan(spawnMeta.LaunchPlan)
		childStarted := time.Now()
		srv.BeginChildStartup(childStarted)
		rotationEvidenceBefore := srv.RotationEvidenceSnapshot()
		startupProgress.Phase("OS process start")
		resourcePolicy := guardResourcePolicyConfigured()
		job, err := windowgate.StartManagedAgentInNewJob(child, windowgate.ManagedJobConfig{MemoryLimitBytes: resourcePolicy.MaxTreeBytes})
		if err != nil {
			startupProgress.Abort()
			// Start/containment failing IS a launch failure: either the child never ran, or
			// StartInNewJob reaped it because the teardown invariant could not be armed.
			guardDumpStartupReportOnLaunchFail(os.Stderr, srv, dumpStartupOnLaunchFail)
			finishGuardChildAndReport(err, child.ProcessState, srv, cancel, serveErr, quiet, auditJournal, auditSeq0, guardTraceID, agentName, provider, dojoMode, sampler)
			return
		}
		startupProgress.Phase("child registration")
		if err := startBoundGuardRegistration(child); err != nil {
			startupProgress.Abort()
			_ = child.Process.Kill()
			_, _ = child.Process.Wait()
			_ = job.Close()
			finishGuardChildAndReport(err, child.ProcessState, srv, cancel, serveErr, quiet, auditJournal, auditSeq0, guardTraceID, agentName, provider, dojoMode, sampler)
			return
		}
		startupProgress.Started()
		lifecycle := startCrashJournalPulse(guardTraceID, child.Process.Pid)
		go func() {
			runErr := child.Wait()
			lifecycle.finish(runErr == nil)
			terminalGuardChild(child, runErr, "")
			_ = job.Close()
			wait <- runErr
		}()
		resourceStop := make(chan struct{})
		resourcePolicy.Stop = resourceStop
		resourceEvents := startGuardChildResourceMonitor(child.Process.Pid, guardTraceID, agentName, resourcePolicy)
		event := waitGuardChild(wait, restarter.events, budgetTicker.C, func(now time.Time) (bool, string) {
			return guardTimeBudgetExhausted(serveSessions, guardTraceID, now)
		}, resourceEvents)
		close(resourceStop)
		switch event.Kind {
		case guardChildResourceLimit:
			markGuardChildTerminalIntent(child, "resource_limit")
			_ = job.Close()
			_ = stopGuardChild(child, wait, 0)
			receiptErr := guardWriteResourceReceipt(event, guardTraceID, agentName, child.Process.Pid)
			fmt.Fprintf(os.Stderr, "fak guard: reaped child resource runaway: %s\n", event.Reason)
			resourceErr := fmt.Errorf("child resource limit: %s", event.Reason)
			appendGuardChildExitWitnessWithReason(auditJournal, agentName, guardTraceID, resourceErr, child.ProcessState, childStarted, event.Resource.Reason)
			if receiptErr != nil {
				resourceErr = fmt.Errorf("child resource receipt failed after containment: %w", receiptErr)
				if restarter.stderr != nil {
					fmt.Fprintf(restarter.stderr, "fak guard: %v\n", resourceErr)
				}
				finishGuardChildAndReport(resourceErr, child.ProcessState, srv, cancel, serveErr, quiet, auditJournal, auditSeq0, guardTraceID, agentName, provider, dojoMode, sampler)
				return
			}
			verdict := resourceRetries.decide(event, agentName, sessionStartSHA())
			if verdict.Action == guardResourceRetryRelaunch {
				nextCommand, reattachErr := guardResourceReattachCommand(command, agentName, codexSessionStatePath, guardTraceID)
				if reattachErr != nil {
					guardRecordResourceReattachUnavailable(auditJournal, agentName, guardTraceID)
					if restarter.stderr != nil {
						fmt.Fprintln(restarter.stderr, guardResourceReattachUnavailableStatus(agentName, guardTraceID, reattachErr))
					}
					resourceErr = fmt.Errorf("%s: %w", guardResourceReattachUnavailable, resourceErr)
					finishGuardChildAndReport(resourceErr, child.ProcessState, srv, cancel, serveErr, quiet, auditJournal, auditSeq0, guardTraceID, agentName, provider, dojoMode, sampler)
					return
				}
				guardReportResourceRestart(restarter.stderr, agentName, verdict, nextCommand)
				time.Sleep(verdict.Delay)
				guardRecordResourceRestart(auditJournal, restarter.stderr, agentName, guardTraceID, verdict.Attempt)
				command = nextCommand
				continue
			}
			if verdict.Action == guardResourceRetryExhausted {
				guardRecordResourceRestartGiveUp(auditJournal, agentName, guardTraceID)
				if restarter.stderr != nil {
					fmt.Fprintln(restarter.stderr, guardResourceRestartGiveUpStatus(verdict, guardTraceID))
				}
				resourceErr = fmt.Errorf("%s: %w", guardResourceRestartExhaustedReason, resourceErr)
			} else if verdict.Cause == guardResourceRestartCauseNoReattach {
				guardRecordResourceReattachUnavailable(auditJournal, agentName, guardTraceID)
				if restarter.stderr != nil {
					fmt.Fprintln(restarter.stderr, guardResourceReattachUnavailableStatus(agentName, guardTraceID, nil))
				}
				resourceErr = fmt.Errorf("%s: %w", guardResourceReattachUnavailable, resourceErr)
			}
			finishGuardChildAndReport(resourceErr, child.ProcessState, srv, cancel, serveErr, quiet, auditJournal, auditSeq0, guardTraceID, agentName, provider, dojoMode, sampler)
			return
		case guardChildCompleted:
			runErr := event.RunErr
			if rec, parked := guardGoalParked(); parked {
				fmt.Fprintf(os.Stderr, "fak guard: goal parked outside active context budget until %d; reason=%s; %s; next=%s\n", rec.ParkedUntil, rec.Reason, guardParkProbeStatus(rec, time.Now()), rec.NextAction)
				// #5862: this is the branch the FLEET takes. Dispatch always passes
				// --max-duration (1740s), and maxDurationLimit > 0 routes every dispatched
				// worker into this supervised loop (guard.go), so a turn-0 provider 429 that
				// parks the goal tears down HERE. It used to skip the witness while its
				// guardChildRestart sibling below wrote one — and that asymmetry alone is why
				// a parked session's journal was zero-byte even though its .refusals.json
				// sidecar (written inside finishGuardChildAndReport) was not. The child has
				// already exited on this branch, so there is nothing to stop: append the
				// witness before the report, which closes the journal.
				appendGuardChildExitWitness(auditJournal, agentName, guardTraceID, nil, child.ProcessState, childStarted)
				finishGuardChildAndReport(nil, child.ProcessState, srv, cancel, serveErr, quiet, auditJournal, auditSeq0, guardTraceID, agentName, provider, dojoMode, sampler)
				return
			}
			if guardRefuseCodexCLIUsage(runErr, child.ProcessState, agentName, guardTraceID, childStderr.String(), childStarted, auditJournal, restarter.stderr) ||
				guardRefuseCodexInvalidJSON(runErr, child.ProcessState, agentName, guardTraceID, childStdout.String(), childStarted, auditJournal, restarter.stderr) {
				finishGuardChildAndReport(runErr, child.ProcessState, srv, cancel, serveErr, quiet, auditJournal, auditSeq0, guardTraceID, agentName, provider, dojoMode, sampler)
				return
			}
			if next, ok := guardMaybeRecoverAuthCrash(runErr, command, credPath, agentName, quiet, os.Stderr); ok {
				command = next
				continue
			}
			if next, ok := guardMaybeRetryTransientWireCrash(runErr, child.ProcessState, command, agentName, wireErrors.Consume(time.Now()), wireRetries, wireLimit, true, nil); ok {
				wireRetries++
				guardRecordWireRetry(auditJournal, restarter.stderr, agentName, guardTraceID, runErr, child.ProcessState, childStarted, wireRetries)
				command = next
				continue
			}
			if nextCommand, nextInjected, ok := rotation.rotateAfterExit(runErr, guardRotationEvidenceSince(rotationEvidenceBefore, srv.RotationEvidenceSnapshot()), command, injected, auditJournal, guardTraceID, os.Stderr); ok {
				command, injected = nextCommand, nextInjected
				continue
			}
			if guardRecoverCapCrash(&command, runErr, agentName, childStarted, quiet, func() time.Duration {
				v := serveSessions.QueryTimeBudget(guardTraceID, time.Now())
				if !v.Bounded {
					return 0
				}
				if v.Exceeded || v.Remaining <= 0 {
					return -1
				}
				return v.Remaining
			}(), os.Stderr) {
				continue
			}
			if class, code, ok := guardMaybeRestartOnCrash(runErr, child.ProcessState, crashRestarts, crashLimit); ok {
				guardMaybeLaunchCrashRSI(restarter.stderr, guardTraceID, agentName, class, code, crashRestarts)
				crashRestarts++
				var reap bool
				crashProgressHead, crashNoProgress, reap = guardCrashNoProgressStep(crashProgressHead, sessionStartSHA(), crashNoProgress, crashNoProgressLimit)
				if reap {
					guardRecordCrashRestartGiveUp(auditJournal, agentName, guardTraceID)
					fmt.Fprintln(restarter.stderr, guardCrashRestartGiveUpStatus(crashNoProgressLimit, guardTraceID))
					break
				}
				guardReportCrashRestart(restarter.stderr, agentName, class, code, crashRestarts, crashLimit, command)
				time.Sleep(guardCrashRestartDelay(crashRestarts))
				guardRecordCrashRestart(auditJournal, restarter.stderr, agentName, guardTraceID, runErr, child.ProcessState, childStarted, crashRestarts)
				command = guardRestartRelaunchCommand(command, agentName)
				continue
			}
			finishGuardChildAndReport(runErr, child.ProcessState, srv, cancel, serveErr, quiet, auditJournal, auditSeq0, guardTraceID, agentName, provider, dojoMode, sampler)
			return
		case guardChildRestart:
			if rec, parked := guardGoalParked(); parked {
				if !quiet {
					fmt.Fprintf(os.Stderr, "fak guard: context budget signal ignored as terminal; goal parked until %d reason=%s %s\n", rec.ParkedUntil, rec.Reason, guardParkProbeStatus(rec, time.Now()))
				}
				markGuardChildTerminalIntent(child, "cancelled")
				stopGuardChild(child, wait, 2*time.Second)
				appendGuardChildExitWitness(auditJournal, agentName, guardTraceID, nil, child.ProcessState, childStarted)
				finishGuardChildAndReport(nil, child.ProcessState, srv, cancel, serveErr, quiet, auditJournal, auditSeq0, guardTraceID, agentName, provider, dojoMode, sampler)
				return
			}
			ev := event.Restart
			equivalentRestarts = equivalentRestarts.step(ev.Reason)
			if equivalentRestarts.count >= guardEquivalentRestartLimit {
				if restarter.stderr != nil {
					fmt.Fprintln(restarter.stderr, guardEquivalentRestartStatus(equivalentRestarts, ev))
				}
				runErr := <-wait
				finishGuardChildAndReport(runErr, child.ProcessState, srv, cancel, serveErr, quiet, auditJournal, auditSeq0, guardTraceID, agentName, provider, dojoMode, sampler)
				return
			}
			// #4609: advance or reset the no-progress counter by whether HEAD moved since the
			// last progress checkpoint. A restart that landed a commit (HEAD advanced) resets both
			// the counter and the checkpoint — a committing worker earns back its full runway; one
			// that did not increments it. Skipped entirely when the reap is disabled or git offers
			// no HEAD signal. This is a strictly EARLIER, progress-aware trip that sits on top of
			// the raw --restart-limit backstop below (16, pinned by TestClaudeGuardRestartLimit),
			// never a replacement for it.
			if noProgressLimit > 0 && progressHead != "" {
				progressHead, noProgressRestarts = guardNoProgressStep(progressHead, sessionStartSHA(), noProgressRestarts)
			}
			if noProgressLimit > 0 && progressHead != "" && noProgressRestarts >= noProgressLimit {
				if restarter.stderr != nil {
					fmt.Fprintln(restarter.stderr, guardNoProgressReapStatus(noProgressLimit, ev))
				}
				runErr := <-wait
				finishGuardChildAndReport(runErr, child.ProcessState, srv, cancel, serveErr, quiet, auditJournal, auditSeq0, guardTraceID, agentName, provider, dojoMode, sampler)
				return
			}
			if restarter.limit > 0 && restarts >= restarter.limit {
				if restarter.stderr != nil {
					fmt.Fprintln(restarter.stderr, guardRestartLimitStatus(restarter.limit, ev))
				}
				runErr := <-wait
				finishGuardChildAndReport(runErr, child.ProcessState, srv, cancel, serveErr, quiet, auditJournal, auditSeq0, guardTraceID, agentName, provider, dojoMode, sampler)
				return
			}
			restarts++
			// Decide the relaunch command AND the handback mode it represents, so the
			// hop recorded below matches the command actually launched.
			handback := ""
			if restarter.seedHandback {
				// #3056 + authoritative-seed default (--restart-seed-handback, now ON by default):
				// inject the bounded carryover seed as the recognized child's initial prompt
				// (--append-system-prompt) AND strip any --continue, so the relaunch boots FRESH on the
				// distilled seed instead of reattaching the exhausted transcript. Reattaching would
				// re-inflate the very context window that just overflowed, so the child re-exhausts and
				// restarts again (the restart loop this fix breaks); booting on the seed shrinks the
				// window. A no-op for an unrecognized agent or an empty seed, which falls through to the
				// #3055 default below.
				if next, hb, injected := guardSeedPromptRelaunchCommand(command, agentName, ev.SeedText, restarter.stderr); injected {
					command, handback = guardStripContinueFlag(next, agentName), hb
				}
			}
			if handback == "" {
				// #3055 fallback: no usable seed (empty seed, or an unrecognized agent whose prompt
				// syntax fak will not guess), or --restart-seed-handback=false. Reattach the existing
				// transcript on relaunch. The FAK_RESET_* env vars set below are advisory only (Claude
				// Code reads none of them), so continuity comes from the wrapped agent's own resume flag
				// — a recognized child resumes the captured conversation instead of booting cold and
				// losing the task. Idempotent across repeated restarts; an unrecognized agent is
				// relaunched unchanged (handback derives to continue/ORPHANED in the hop below). This
				// path RE-INFLATES the exhausted window (no context shrink) — the hop one-liner marks it
				// shrink=no so an operator can spot the restart that did not reduce the window.
				command = guardRestartRelaunchCommand(command, agentName)
			}
			// #3057: ONE correlated record per restart — a RESTART_HOP row in the
			// guard audit journal plus a single stderr one-liner carrying the same
			// fields (from/to trace, seed size + file, handback mode, child session,
			// continuity status) — replacing the two uncorrelated lines that used to
			// be the only evidence a hidden relaunch ever happened. Queryable later
			// via `fak guard restart-audit` and `fak session status <id>`.
			guardEmitRestartHop(auditJournal, restarter.stderr, agentName, guardTraceID,
				guardRestartHopFromEventHandback(ev, restarts, agentName, handback))
			srv.SetDefaultTraceID(ev.ToTraceID)
			extraEnv = guardRestartEnv(ev)
			// Let the triggering response finish flushing to the wrapped client before
			// stopping the process that initiated it.
			time.Sleep(750 * time.Millisecond)
			markGuardChildTerminalIntent(child, "restart")
			stopGuardChild(child, wait, 2*time.Second)
		case guardChildTimeBudget:
			// The wall-clock envelope elapsed: stop the wrapped agent and report, rather
			// than let it keep burning tokens past its --max-duration (the #2229 gap).
			if !quiet {
				fmt.Fprintf(os.Stderr, "fak guard: %s — wall-clock --max-duration envelope elapsed for %s; stopping the wrapped agent\n", event.Reason, guardTraceID)
			}
			markGuardChildTerminalIntent(child, "time_budget")
			stopGuardChild(child, wait, 2*time.Second)
			appendGuardChildExitWitness(auditJournal, agentName, guardTraceID, nil, child.ProcessState, childStarted)
			finishGuardChildAndReport(nil, child.ProcessState, srv, cancel, serveErr, quiet, auditJournal, auditSeq0, guardTraceID, agentName, provider, dojoMode, sampler)
			return
		}
	}
}

// guardChildTreeKill is the destructive escalation used when a wrapped child does not exit
// within the graceful-interrupt window. It defaults to procguard.KillPID — a process-TREE
// kill (native job termination / taskkill /T on Windows, process-group/descendant SIGKILL on
// POSIX) — so the child's own descendants (the node runtime, MCP/tool subprocesses a `claude`
// spawns) are reaped too. A bare child.Process.Kill() only reaps the immediate PID and leaves
// that subtree orphaned; across a budget restart those orphans accumulate under one guard
// parent (#2989: three live claude children from two restarts, poisoning dispatch preflight
// with unattributed_live). Injectable for tests. Mirrors fleetKillPID in fleet.go.
var guardChildTreeKill = procguard.KillPID

// stopGuardChild stops the wrapped child on restart/timeout. It first asks politely
// (os.Interrupt) and waits out the grace window; if the child is still alive it escalates to a
// TREE kill of the child's PID so no descendant subtree survives the transition. Returning
// only after <-wait guarantees the previous child is fully reaped before the caller relaunches,
// so a budget restart is single-child (#2989).
func stopGuardChild(child *exec.Cmd, wait <-chan error, grace time.Duration) error {
	if child == nil || child.Process == nil {
		return nil
	}
	_ = child.Process.Signal(os.Interrupt)
	select {
	case err := <-wait:
		return err
	case <-time.After(grace):
	}
	// Tree-kill: reap the child AND its descendants, not just the immediate PID.
	if ok, detail := guardChildTreeKill(child.Process.Pid); !ok {
		return fmt.Errorf("tree kill pid %d failed: %s", child.Process.Pid, detail)
	}
	select {
	case err := <-wait:
		return err
	case <-time.After(5 * time.Second):
		return fmt.Errorf("tree kill pid %d did not join within 5s", child.Process.Pid)
	}
}

// formatGuardSessionResumeCommand returns the exact, copy/paste-ready command for
// reopening the provider session that owns the guard's current trace. SessionStart
// is the authority: missing, mismatched, unsupported, or malformed identity stays
// silent rather than printing a command that could open the wrong conversation.
func formatGuardSessionResumeCommand(agentName, traceID string) string {
	provider := guardAgentBaseName(agentName)
	if provider != "claude" && provider != "codex" {
		return ""
	}

	match := resume.ResolveIdentity(resume.LoadIdentityRows(resolveSweepRegDir("")), traceID)
	if !match.OK || match.Direction != "trace->uuid" {
		return ""
	}
	rowProvider := strings.ToLower(strings.TrimSpace(match.Row.Provider))
	if rowProvider != provider || !validGuardProviderSessionID(match.Paired) {
		return ""
	}

	var argv []string
	if provider == "claude" {
		argv = []string{"fak", "guard", "--", "claude", "--resume", match.Paired}
	} else {
		argv = []string{"fak", "guard", "--", "codex", "resume", match.Paired}
	}
	return fmt.Sprintf("\nfak guard: resume this session with:\n  %s\n", shellJoin(argv))
}

func validGuardProviderSessionID(value string) bool {
	value = strings.TrimSpace(value)
	if len(value) != 36 {
		return false
	}
	for i, r := range value {
		if i == 8 || i == 13 || i == 18 || i == 23 {
			if r != '-' {
				return false
			}
			continue
		}
		if !((r >= '0' && r <= '9') || (r >= 'a' && r <= 'f') || (r >= 'A' && r <= 'F')) {
			return false
		}
	}
	return true
}

// formatGuardResumeGuidance is the concise, actionable note printed when the wrapped agent
// exits abnormally (a non-zero code — a crash, an OOM, or a terminal upstream error). The
// guard process holds no agent conversation state itself — the wrapped tool owns that — so
// "resume" means re-running the same `fak guard -- <command>` with the agent's own
// resume/continue flag. The last line encodes a hard-won recovery: a guarded resume that
// dies IMMEDIATELY with "upstream model error" (a malformed-request rejection that can
// follow a mid-conversation quarantine) usually clears if that one resume is retried WITHOUT
// fak guard, then re-wrapped. Returned as a string (not printed) so it is unit-testable.
func formatGuardResumeGuidance(agentName string, code int) string {
	return formatGuardResumeGuidanceWithRefusals(agentName, code, nil)
}

func formatGuardResumeGuidanceWithRefusals(agentName string, code int, refusals []guardRefusalCarry) string {
	guardActivity := ""
	if len(refusals) > 0 {
		var b strings.Builder
		b.WriteString("  guard activity: the last guarded run hit floor refusal(s); treat the resume as recovery/debugging, not a blind retry.\n")
		for _, item := range refusals {
			if strings.TrimSpace(item.Reason) == "" {
				continue
			}
			fmt.Fprintf(&b, "    - %s x%d", item.Reason, item.Count)
			if fix := strings.TrimSpace(item.Fix); fix != "" {
				fmt.Fprintf(&b, " — fix: %s", fix)
			}
			b.WriteByte('\n')
		}
		b.WriteString("  guard resume: keep fak guard wrapped after clearing the refusal; do not retry the same refused call unchanged.\n")
		guardActivity = b.String()
	}
	return fmt.Sprintf(
		"\nfak guard: %s exited abnormally (code %d).\n"+
			"  resume: re-run the same `fak guard -- %s …` — launch the agent with its own resume/continue flag (e.g. `claude --continue`) so it picks the conversation back up.\n"+
			"%s"+
			"  this session's decision journal is replayable with `fak audit verify` (path in the audit summary above).\n"+
			"  if a guarded resume dies IMMEDIATELY with \"upstream model error\", retry that one resume WITHOUT fak guard to recover, then re-wrap.\n",
		agentName, code, agentName, guardActivity)
}

// formatVCacheSnapshotPointer is the exit pointer that closes the loop between the LIVE
// guard cache summary and the OFFLINE `fak vcache` family. After a session persists its
// OBSERVED provider-cache window (vcachesnapshot.Write), this line tells the operator the
// window is on disk AND names the one command that re-derives THIS session's REALIZED cache
// multiplier from it: `fak vcache score` reads the well-known snapshot path with no flags
// (it folds the same turns through the same vcacheobserve engine the summary line priced),
// and `fak vcache observe` renders the per-sub-concept panels. Without this, the snapshot is
// written silently and the related vcache tools look like they only have a synthetic forecast
// to chew on — the operator never learns the real session data is right there to replay.
//
// Empty when no turns were recorded (a run that never saw a cached turn writes an empty
// snapshot, which the score correctly treats as "no observed window" and falls open to the
// forecast), so a no-cache session stays quiet rather than printing a vacuous 0-turn pointer.
func formatVCacheSnapshotPointer(turns int, path string) string {
	if turns <= 0 {
		return ""
	}
	var b strings.Builder
	b.WriteString(guardSection("cache window"))
	b.WriteString(guardRow("recorded", fmt.Sprintf("%d turn(s)", turns)))
	b.WriteString(guardRow("to", path))
	b.WriteString(guardRow("replay realized multiplier", "fak vcache score"))
	b.WriteString(guardNote("no flags reads this snapshot; `fak vcache observe` renders the per-sub-concept panels"))
	return b.String()
}

// guardSummaryResetPrefix is the terminal escape the exit summary emits before its first line
// so it never inherits a dangling SGR style or a hidden cursor the wrapped agent's torn-down
// alt-screen left. "\x1b[0m" resets all SGR attributes (color, bold, reverse), "\x1b[?25h"
// re-shows the cursor a TUI may have hidden. It is emitted ONLY to a real terminal (isTTY): a
// summary piped to a file or a `-p` JSON capture must stay byte-clean, so a non-TTY sink gets
// the empty string. Pure (string in, string out) so the TTY-gated behavior is unit-tested.
func guardSummaryResetPrefix(isTTY bool) string {
	if !isTTY {
		return ""
	}
	return "\x1b[0m\x1b[?25h"
}

// guardClassifyChildCrash maps a completed child's exit into the closed crash
// vocabulary for a CHILD_CRASH journal row. It is pure and portable (no Unix-only
// syscall.WaitStatus — it reads only the exit code and the ProcessState's portable
// String(), so the SAME classification runs on the Windows dev host and the Unix
// fleet). isCrash is false for a clean exit (nil runErr, or a zero exit code) and
// for a run failure that never produced a child (a spawn error is reported by the
// caller's existing path, not journaled as a crash). When isCrash is true, class
// is one of the journal.Crash* constants and exitCode is the child's code (-1 when
// the platform reports a signaled kill).
//
//   - A signaled kill that looks like an OOM (exit 137 = 128+SIGKILL, or a state
//     string naming "killed") -> OOM: a resource exhaustion the loop tracks apart
//     from a logic fault.
//   - Any other signaled kill (ExitCode -1, or a state string naming "signal") ->
//     SIGNAL_CRASH: SIGSEGV/SIGABRT/an external kill.
//   - A plain non-zero exit (a Go panic the runtime turned into exit 2, an
//     unrecovered error) -> NONZERO_EXIT.
func guardClassifyChildCrash(runErr error, childState *os.ProcessState) (class string, exitCode int, isCrash bool) {
	ee, isExit := runErr.(*exec.ExitError)
	if !isExit {
		return "", 0, false // nil, or a spawn failure the caller reports directly
	}
	code := ee.ExitCode()
	if code == 0 {
		return "", 0, false // an ExitError that somehow carries a clean code is not a crash
	}
	stateStr := ""
	if childState != nil {
		stateStr = strings.ToLower(childState.String())
	}
	windowsForcedTermination := runtime.GOOS == "windows" && uint32(code) == 0xffffffff
	if windowsForcedTermination {
		code = -1
	}
	signaled := code < 0 || strings.Contains(stateStr, "signal")
	oom := code == 137 || strings.Contains(stateStr, "killed")
	switch {
	case oom:
		return journal.CrashOOM, code, true
	case signaled:
		return journal.CrashSignal, code, true
	default:
		return journal.CrashNonzeroExit, code, true
	}
}

func appendGuardChildExitWitness(j *journal.Journal, agentName, traceID string, runErr error, state *os.ProcessState, started time.Time) journal.Row {
	return appendGuardChildExitWitnessWithReason(j, agentName, traceID, runErr, state, started, "")
}

// appendGuardChildExitWitnessWithReason lets a supervisor-owned termination
// carry its stable typed cause directly. Resource-monitor errors are synthetic
// supervisor errors rather than *exec.ExitError values, so routing them through
// the ordinary process-exit classifier would otherwise produce a blank reason.
func appendGuardChildExitWitnessWithReason(j *journal.Journal, agentName, traceID string, runErr error, state *os.ProcessState, started time.Time, reasonClass string) journal.Row {
	if j == nil {
		return journal.Row{}
	}
	class, exitCode := journal.CrashCleanExit, 0
	if reasonClass = strings.TrimSpace(reasonClass); reasonClass != "" {
		class = reasonClass
		if state != nil {
			exitCode = state.ExitCode()
		}
	} else if runErr != nil {
		class, exitCode, _ = guardClassifyChildCrash(runErr, state)
	}
	lastHook := ""
	for _, row := range j.Recent(64) {
		if row.Seq == 0 || row.Kind == "CHILD_CRASH" || row.Kind == "CHILD_EXIT" {
			continue
		}
		lastHook = row.Kind
		if row.Reason != "" {
			lastHook += ":" + row.Reason
		}
		break
	}
	wall := time.Duration(0)
	if !started.IsZero() {
		wall = time.Since(started)
	}
	return j.AppendChildExit(agentName, traceID, class, exitCode, wall, lastHook)
}

func finishGuardChildAndReport(runErr error, childState *os.ProcessState, srv *gateway.Server, cancel context.CancelFunc, serveErr <-chan error, quiet bool, auditJournal *journal.Journal, auditSeq0 uint64, guardTraceID, agentName, provider string, dojoMode bool, sampler *harnessres.Sampler) {
	var currentRefusals []guardRefusalCarry

	// Tear the gateway down and report what the kernel decided this session.
	if sampler != nil {
		sampler.FoldChildExit(childState)
	}
	cancel()
	serr := <-serveErr
	// The session's wall-clock window, when the resource sampler tracked one — the
	// honest scope for the hook-latency exit line below (0 = no anchor, fold all-time).
	var sessionWindow time.Duration
	if sampler != nil {
		snap := sampler.Stop()
		snap.GuardStops = foldGuardStopCounts()
		sessionWindow = snap.Elapsed
		appendHarnessResources("guard", provider, agentName, snap)
		if !quiet {
			fmt.Fprintln(os.Stderr, "fak guard: "+snap.Report())
		}
	}
	// Pin THIS session's compaction health durably BEFORE the banner and OUTSIDE the !quiet
	// gate (#3152). The counters live only on the gateway we just tore down, so without this
	// row a finished session leaves no checkable answer to "did compaction fire for THIS
	// session?" — and a headless `--quiet` worker, which prints no banner at all, is exactly
	// the session an auditor comes back to. Best-effort: an unwritable ledger returns "" and
	// the banner simply omits the block, never failing the exit.
	compactionWitness := recordGuardCompactionWitness(guardCompactionWitnessLedger(), guardTraceID, srv.AdjudicationSummary(), time.Now())
	if !quiet {
		// The wrapped agent (Claude Code) paints a full-screen alternate-screen TUI over this
		// same terminal and, on a crash or an abnormal exit, can tear it down mid-escape-sequence
		// — leaving a dangling SGR color/style or a hidden cursor. The exit summary then renders
		// mis-colored or invisible. Emit a soft reset (SGR reset + show-cursor) onto a clean
		// baseline FIRST, but only when stderr is a real terminal: piping the summary to a file or
		// a JSON capture must stay byte-clean, so a non-TTY stderr gets no escape bytes.
		summaryTTY := guardFdIsTerminal(int(os.Stderr.Fd()))
		fmt.Fprint(os.Stderr, guardSummaryResetPrefix(summaryTTY))
		fmt.Fprintln(os.Stderr)
		sum := srv.AdjudicationSummary()
		kc := srv.KernelCounters()
		// Each formatter returns byte-clean text (no escape bytes — pure and unit-tested);
		// color is layered on HERE, gated on a real terminal, so the section rules read as
		// headings and the demoted notes recede while the value rows stand out, yet a piped
		// or JSON-captured summary stays exactly the plain text the tests assert.
		emit := func(text string) { fmt.Fprint(os.Stderr, guardColorizeSummary(text, summaryTTY)) }
		emit(formatAuditSummary(sum, kc))
		emit(formatAmplification(kc, sum))
		emit(formatTurnsTimeSaved(kc, sum))
		emit(formatJournalSummary(auditJournal, auditSeq0))
		// The guard-hook wall-clock tax (#1993): what the pre+post adjudication hooks
		// cost per tool call this session. Best-effort — a hook-less session prints
		// nothing rather than a vacuous zero row.
		emit(guardHookLatencySummaryLine(sessionWindow, time.Now()))
		// The tool process table (#2445): any event-stream monitor that went silent
		// past its cadence this session folded to a killed TOOL_HEARTBEAT_STALLED —
		// surface that count here (silence is not success). Best-effort, empty when
		// nothing is outstanding.
		emit(guardToolprocSummary(time.Now()))
		// The injected-directive negframe signal (#3568): which arm of the #3546 steerability
		// A/B this session ran, plus the post-reframe residual negatives and the
		// fail-safe-to-verbatim fallbacks. Best-effort — silent when nothing was injected.
		emit(guardNegframeSummary())
		emit(guardTrajectoryWarningLine())
		// The context-health verdict (#3099): fold this session's LIVE trajectory
		// corpus through the #3098 HEALTHY/STALL/DRIFT/DETOUR_OVERRUN scorer + the
		// #3096 shed-span use-after-free detector onto the same guard status channel,
		// so a repeat-failure loop or a shed-then-reference is visible here, not only
		// in a post-hoc `fak traj score`. Empty (silent) unless trajectory recording
		// is on, exactly like the sibling lines stay quiet when their signal is absent.
		emit(guardContextHealthLine())
		// The durable compaction witness (#3152), rendered from the row pinned above —
		// not from the live counters — so the operator reads the same bytes a later
		// `fak guard compaction-witness` audit will read and the banner can never
		// disagree with the witness of record. Empty (silent) when there was no session
		// id or the ledger round trip failed — exactly the cases where there is no
		// durable row to point an auditor at, so printing a block would overclaim.
		emit(compactionWitness)
		// The amendment posture (#5184): who could have moved which policy surface
		// this session. Read from the compiled-in PolicyKnobRegistry, never from
		// session state, so it is a property of the BINARY the operator is running
		// rather than a self-report -- and the load-bearing row is that the
		// agent-writable frontier is empty. Always printed: unlike the best-effort
		// lines above it has no absent case to stay quiet about.
		emit(formatAmendmentPosture())
	}
	// Append cache-value observation to ledger (epic #1072, issue #1075) AND surface it.
	// Persist both tracks, then — for a non-quiet (interactive) session — print the
	// dollar-aware evidence summary so the operator SEES the savings this session earned
	// instead of it landing silently in the ledger. The summary formatter had been built
	// and tested but wired to nothing (detection-without-enforcement); this is the felt
	// "usage savings moment" for a `fak c` / `fak guard` session. A headless --quiet
	// worker still persists silently, exactly as before.
	cvReport := buildCacheValuePersistenceReport(srv, "guard", agentName, provider, cachevalueledger.DefaultLedgerRel, time.Now())
	if !quiet {
		fmt.Fprint(os.Stderr, guardColorizeSummary(formatCacheValuePersistenceSummary("fak guard", cvReport), guardFdIsTerminal(int(os.Stderr.Fd()))))
	}
	// Append the gateway-usage exit row (#1610), same writer as the serve exits, so a
	// guard session's full served-turn counter family — compaction fired/bailed/shed
	// among them — survives the process instead of dying with the console summary.
	// Until now only `fak serve` exits reached this ledger, so the per-session WHY
	// behind a zero fak-slice (burst_unprofitable vs anchor-starved vs under_budget)
	// was unrecoverable after exit on the flagship guard path (epic #1601 gap).
	// The guard path is the one that CAN name its session: one guard process wraps one
	// agent session, so guardTraceID is a true per-session join key (unless it is the
	// shared non-durable sentinel, which gatewayUsageSessionID drops). Stamping it is what
	// lets `fak fleet metrics` drill a fleet roll-up down to this named session instead of
	// stopping at a process-level total.
	persistGatewayUsageObservation(srv, "guard", agentName, sessionWindow, gatewayUsageSessionID(guardTraceID))
	if dojoMode {
		_ = persistLiveDojoEpisode("guard", srv)
	}
	// Persist this session's OBSERVED provider-cache window so a later `fak vcache score`
	// (a separate process) reports the REALIZED multiplier from real traffic instead of the
	// synthetic-Zipf forecast (#1090). Best-effort: a write failure never fails the session,
	// and an empty window leaves the snapshot empty so the score falls open to the forecast.
	// On a clean write, point the operator at the offline vcache tools that now hold this
	// session's data — otherwise the snapshot is silent and the related vcache items look
	// like they only have a synthetic forecast to score.
	if turns, _ := srv.VCacheTurnsSnapshot(); len(turns) > 0 {
		if snapPath, ok, err := writeConfiguredVCacheSnapshot(turns); err == nil && ok && !quiet {
			fmt.Fprint(os.Stderr, formatVCacheSnapshotPointer(len(turns), snapPath))
		}
	}
	// Flush + fsync the durable trail before exit so a row returned to the agent is
	// never lost to a buffered write (Close is safe on a nil/in-memory journal).
	if auditJournal != nil {
		var err error
		currentRefusals, err = guardWriteRefusalCarryForwardAndReturn(auditJournal, auditSeq0, guardTraceID, guardFindReasonRoot())
		if err != nil && !quiet {
			fmt.Fprintf(os.Stderr, "fak guard: refusal carry-forward unavailable: %v\n", err)
		}
		_ = auditJournal.Close()
	}
	// Terminal edit for the live session card (#2259): classify the exit exactly as the
	// surfacing logic below does, then stop the updater and write the final outcome line. The
	// card handle is nil unless this session queued a Slack root, so finalizeOutcome is a no-op
	// otherwise; it flushes synchronously because an os.Exit follows immediately below.
	guardExitCode := 0
	if runErr != nil {
		if ee, isExit := runErr.(*exec.ExitError); isExit {
			guardExitCode = ee.ExitCode()
		} else {
			guardExitCode = 1
		}
	} else if serr != nil && !errors.Is(serr, http.ErrServerClosed) && !errors.Is(serr, context.Canceled) {
		guardExitCode = 1
	}
	guardSessionCardHandle.finalizeOutcome(guardExitCode, srv.AdjudicationSummary())
	if !quiet {
		renderGuardPromotionOffers(os.Stderr)
	}
	resumeCommand := formatGuardSessionResumeCommand(agentName, srv.DefaultTraceID())
	emitResumeCommand := func() {
		if !quiet {
			fmt.Fprint(os.Stderr, resumeCommand)
		}
	}
	recordGuardUsage(guardExitCode)
	// The session is over: drop this session's allow-overlay scope (#5180) so a
	// `fak guard allow --session <tool>` widening does not silently become the next
	// launch's floor. HERE, and not as a `defer` in cmdGuard, because every branch below
	// ends in os.Exit — which runs no defers — so a deferred drop would fire only on a
	// clean exit and leak on exactly the crash paths. Repo/user/env layers are untouched.
	dropGuardAllowSessionScopeAtSessionEnd(quiet, os.Stderr)
	// Faithfully surface the child's exit code first (so `fak guard -- claude -p …`
	// scripts see what the agent returned).
	if runErr != nil {
		if ee, isExit := runErr.(*exec.ExitError); isExit {
			// An abnormal (non-zero) exit is a crash / OOM / terminal upstream error — print
			// the actionable resume note so the operator isn't left with a bare exit code.
			// Suppressed under --quiet (scripted `-p` runs) and skipped on a clean 0 exit.
			if code := ee.ExitCode(); code != 0 && !quiet {
				fmt.Fprint(os.Stderr, formatGuardResumeGuidanceWithRefusals(agentName, code, currentRefusals))
			}
			emitResumeCommand()
			os.Exit(ee.ExitCode())
		}
		fmt.Fprintf(os.Stderr, "fak guard: could not run %q: %v\n", agentName, runErr)
		emitResumeCommand()
		os.Exit(1)
	}
	// The child succeeded — but if the gateway itself failed mid-session (Serve returned
	// something other than a clean shutdown), the adjudication boundary was down for part
	// of the run, so do not report a silent success. A clean teardown returns nil.
	if serr != nil && !errors.Is(serr, http.ErrServerClosed) && !errors.Is(serr, context.Canceled) {
		fmt.Fprintf(os.Stderr, "fak guard: gateway error during the session: %v\n", serr)
		emitResumeCommand()
		os.Exit(1)
	}
	emitResumeCommand()
}
