package main

// resume_watchdog_cli.go â€” `fak resume watchdog`, ONE TICK of the cross-account resume
// layer for ALL autonomous Claude sessions on this host. Go port of
// tools/fleet_resume_watchdog.py / .ps1 (the executing half; every decision it enforces
// is an already-audited leaf):
//
//	fak resume watchdog                 # dry-run: log what it WOULD resume
//	fak resume watchdog --live          # actually resume (outcome-gated, capped, paced)
//	FAK_LIVE=1 fak resume watchdog      # same, for cron/scheduled ticks
//	fak resume watchdog --plan p.json   # targeted run: consume EXACTLY this plan file (#2367)
//
// Each tick:
//  1. EXTRACT-IN-ADVANCE â€” refresh the on-disk session registry + AUTO_RESUME plan via
//     tools/fleet_sessions.py (the one remaining Python child; a missing interpreter
//     degrades to the existing plan file with a note, never a crash).
//  2. Gate each planned session through the audited decisions: the self-resume guard,
//     the worker-account policy, the outcome-aware once-gate (resume.RetryGate â€” a
//     resume that died recoverably stays eligible up to the attempt cap; a clean finish
//     or an auth wall burns it), and on a live tick the host-wide per-source admission
//     (resume.AdmitSource â€” the 529 burst-wall ceiling `fak resume admit` exposes).
//  3. Re-home the transcript when the plan says so (rehome.RehomeTranscript), spawn
//     `claude --resume` under the owning account's CLAUDE_CONFIG_DIR (hidden window on
//     Windows), record the launch in the durable ledger BEFORE anything else, and pace
//     the next spawn so a burst does not self-congest.
//  4. Alert (notifications.log + macOS toast when available) on accounts that need a
//     human re-login â€” once per account blocker.
//
// Safety rails (faithful to the .ps1/.py): DRY-RUN by default; per-tick launch cap;
// launch spacing; ledger-first recording so a crash cannot double-launch in one tick.
// Slack posting is NOT ported yet (the Python's --slack seam) â€” follow-on, see the
// tools/slack_post parity note in the goal issue.

import (
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/anthony-chaudhary/fak/internal/knownbad"
	"github.com/anthony-chaudhary/fak/internal/procguard"
	"github.com/anthony-chaudhary/fak/internal/resume"
	"github.com/anthony-chaudhary/fak/internal/resume/rehome"
	"github.com/anthony-chaudhary/fak/internal/resumebackoff"
	"github.com/anthony-chaudhary/fak/internal/resumemetrics"
	"github.com/anthony-chaudhary/fak/internal/sessionctl"
	"github.com/anthony-chaudhary/fak/internal/sessiondiag"
	"github.com/anthony-chaudhary/fak/internal/sessionsignals"
)

// resumeWatchdogPrompt is the standing re-entry instruction a resumed session receives.
const resumeWatchdogPrompt = "Resume where you left off; re-establish any /goal or /loop and continue toward it."

// The plan row type is the leaf's resume.WatchdogPlanRow (json-tagged with the
// resume_plan.json key names), so the shell and the pure guard chain read one shape.

// runResumeWatchdog executes one watchdog tick. Exit codes: 0 ok (including a clean
// dry-run), 1 runtime error, 2 usage error.
func runResumeWatchdog(stdout, stderr io.Writer, argv []string) int {
	return runResumeWatchdogTick(stdout, stderr, argv)
}

func runResumeWatchdogTick(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("resume watchdog", flag.ContinueOnError)
	fs.SetOutput(stderr)
	live := fs.Bool("live", rwEnvBool("FAK_LIVE"), "actually resume (default: dry-run; env FAK_LIVE=1)")
	windowH := fs.Float64("window-h", rwEnvFloat("FAK_WINDOW_H", 6), "registry window in hours (env FAK_WINDOW_H)")
	maxPerTick := fs.Int("max-per-tick", rwEnvInt("FAK_MAX_PER_TICK", 4), "max resumes launched per tick (env FAK_MAX_PER_TICK)")
	maxAttempts := fs.Int("max-attempts", rwEnvInt("FAK_MAX_ATTEMPTS", resume.DefaultMaxResumeAttempts), "give-up cap on automatic resumes of one session (env FAK_MAX_ATTEMPTS)")
	spacingSec := fs.Float64("spacing-sec", rwEnvFloat("FAK_LAUNCH_SPACING_SEC", 8), "seconds between spawns in one tick, so a burst does not trip the per-source 529 wall (0 = all at once; env FAK_LAUNCH_SPACING_SEC)")
	probeMode := fs.String("probe", envOrDefault("FAK_PROBE", "auto"), "registry refresh probe mode: auto|blocked|stale|all|none (auto = blocked on --live, none on dry-run; env FAK_PROBE)")
	probeMinIntervalMin := fs.Int("probe-min-interval-min", rwEnvInt("FAK_PROBE_MIN_INTERVAL_MIN", 20), "min minutes between active probes of one account (env FAK_PROBE_MIN_INTERVAL_MIN)")
	regDirFlag := fs.String("reg-dir", "", "registry dir holding resume_plan.json / resume_ledger.jsonl / sessions.json (default: $FLEET_REG_DIR, else host Fleet registry when present, else <repo>/tools/_registry)")
	logDirFlag := fs.String("log-dir", "", "watchdog log dir (default: $FAK_WATCHDOG_LOG_DIR, else <repo>/tools/_watchdog)")
	noRefresh := fs.Bool("no-refresh", false, "skip the fleet_sessions.py registry refresh and act on the existing plan file (offline/test)")
	planFlag := fs.String("plan", "", "explicit plan file for a TARGETED run: consume exactly this file, skip the registry refresh, and fail closed if it is missing/unreadable/empty â€” the shared resume_plan.json can be regenerated by a concurrent fleet refresh between an operator's write and this tick's read (#2367)")
	statusOnly := fs.Bool("status", false, "print the read-only drain status from resume_plan.json + resume_ledger.jsonl, then exit")
	asJSON := fs.Bool("json", false, "with --status, emit the machine-readable drain report")
	silentHours := fs.Float64("silent-hours", rwEnvFloat("FAK_RESUME_SILENT_HOURS", 2), "with --status, mark red when any unrecovered queued row is silent this many hours (env FAK_RESUME_SILENT_HOURS)")
	unprovenMinutes := fs.Float64("unproven-minutes", rwEnvFloat("FAK_RESUME_UNPROVEN_MINUTES", 10), "with --status, mark red when a launched resume has no progress witness for this many minutes (env FAK_RESUME_UNPROVEN_MINUTES; 0 disables)")
	monotonicTicks := fs.Int("monotonic-ticks", rwEnvInt("FAK_RESUME_MONOTONIC_TICKS", 3), "with --status, mark red when AUTO_RESUME depth grows for this many consecutive ticks (env FAK_RESUME_MONOTONIC_TICKS)")
	launchStaleMinutes := fs.Float64("launch-stale-minutes", rwEnvFloat("FAK_RESUME_LAUNCH_STALE_MIN", 30), "with --status, mark red when the plan carries queued sessions but the durable ledger has had no launch for this many minutes â€” the dead-but-silent auto-resume the status view used to mask off fabricated timestamps (#3460; env FAK_RESUME_LAUNCH_STALE_MIN; 0 disables)")
	backlogThreshold := fs.Int("backlog-threshold", rwEnvInt("FAK_RESUME_BACKLOG_THRESHOLD", 20), "page when AUTO_RESUME depth stays above this for --backlog-ticks consecutive ticks AND 0 accounts are throttled â€” BOTTLENECK-MAP Â§7's \"the cap is the real limiter\" decision as a standing gate (#3582; env FAK_RESUME_BACKLOG_THRESHOLD; 0 disables)")
	backlogTicks := fs.Int("backlog-ticks", rwEnvInt("FAK_RESUME_BACKLOG_TICKS", 3), "consecutive ticks the AUTO_RESUME backlog must stay above --backlog-threshold before paging (#3582; env FAK_RESUME_BACKLOG_TICKS; 0 disables)")
	if !parseFlags(fs, argv) {
		return 2
	}

	regDir := resolveSweepRegDir(*regDirFlag)
	logDir := resolveWatchdogLogDir(*logDirFlag)
	_ = os.MkdirAll(logDir, 0o755)
	note := func(format string, a ...any) { rwNote(logDir, stdout, fmt.Sprintf(format, a...)) }

	// Host-level tick lock (#3110): serialize overlapping ticks so a slow tick and a
	// concurrent cron/--live/manual tick sharing this regDir cannot both read
	// independent stale process-table snapshots and admit the SAME still-starting
	// session (briefly running two `claude --resume` on one transcript). A --status
	// read is side-effect-free and launches nothing, so it never contends for the lock.
	if !*statusOnly {
		release, acquired, lockErr := resume.TryTickLock(regDir)
		if lockErr != nil {
			// Fail open: a broken lock must never strand the whole watchdog.
			note("  tick-lock: %v â€” proceeding unlocked (fail open)", lockErr)
		} else if !acquired {
			note("another resume-watchdog tick holds the lock; skipping this tick")
			return 0
		} else {
			defer release()
		}
	}

	home, _ := os.UserHomeDir()
	claudeExe := rwClaudeExe(home)
	selfSID := strings.TrimSpace(os.Getenv("CLAUDE_CODE_SESSION_ID"))

	// 1. Refresh registry + plan (extract in advance). On a live tick the refresh also
	// actively re-probes blocked accounts so a silently-recovered one re-enters the pool
	// instead of riding a stale carried verdict; dry-run stays side-effect-free. A
	// targeted --plan run never refreshes: its whole point is consuming an operator's
	// immutable file, not whatever a concurrent fleet refresh just regenerated.
	targetedPlan := strings.TrimSpace(*planFlag)
	mode := resume.ResolveWatchdogProbeMode(*probeMode, *live)
	if !*noRefresh && !*statusOnly && targetedPlan == "" {
		if msg := rwRefreshRegistry(regDir, claudeExe, *windowH, mode, *probeMinIntervalMin); msg != "" {
			note("  refresh: %s", msg)
		}
	}

	planPath := filepath.Join(regDir, "resume_plan.json")
	if targetedPlan != "" {
		planPath = targetedPlan
	}
	plan := rwLoadPlan(planPath)
	if targetedPlan == "" {
		var candidateReport sessiondiag.WatchdogCandidateReport
		plan, candidateReport = rwMergeSessiondiagCandidates(plan, *windowH)
		if !*statusOnly && (len(candidateReport.Candidates) > 0 || len(candidateReport.Exclusions) > 0) {
			note("  candidates: included=%d excluded=%d source=sessiondiag", len(candidateReport.Candidates), len(candidateReport.Exclusions))
		}
	}
	ledgerPath := filepath.Join(regDir, "resume_ledger.jsonl")
	backoffHistory := rwBackoffHistory(ledgerPath)
	for _, completed := range rwLoadCodexCompletions(plan) {
		r := completed.Result
		rwAppendLedger(ledgerPath, map[string]any{"ts": rwNowISO(), "session": completed.Session, "harness": "codex", "phase": "terminal", "outcome": r.Outcome, "useful_work": r.UsefulWork, "task_completed": r.TaskCompleted, "process_exit": r.ProcessExit, "forced_reclaim": r.ForcedReclaim, "duration_ms": r.DurationMS})
	}
	rwBoundWatchdogArtifacts(logDir, ledgerPath, time.Now())
	if targetedPlan != "" {
		// Fail closed (#2367): a targeted run must consume the exact plan the operator
		// selected â€” silently acting on nothing (or on the shared file a concurrent
		// refresh owns) is the race this flag exists to close.
		if len(plan) == 0 {
			note("PLAN RACE GUARD: targeted plan %s is missing, unreadable, or empty â€” refusing to act (#2367)", targetedPlan)
			return 1
		}
		note("  plan: targeted file %s (%d row(s)); shared resume_plan.json ignored", targetedPlan, len(plan))
	}
	tickMode := "DRY-RUN"
	if *live {
		tickMode = "LIVE"
	}

	statusLedgerPath := rwWatchdogStatusLedger(regDir)
	statusEvents := rwLoadWatchdogStatusEvents(ledgerPath)
	statusEvents = append(statusEvents, rwLoadWatchdogStatusEvents(statusLedgerPath)...)
	throttledAccounts, throttledKnown := rwThrottledAccounts(regDir)
	statusTh := watchdogStatusThresholds{
		silentHours: *silentHours, unprovenMinutes: *unprovenMinutes,
		monotonicTicks: *monotonicTicks, launchStaleMinutes: *launchStaleMinutes,
		backlogThreshold: *backlogThreshold, backlogTicks: *backlogTicks,
		throttledAccounts: throttledAccounts, throttledKnown: throttledKnown,
	}
	if *statusOnly {
		// A --status read stays side-effect-free: it REPORTS the page (rep.Page) but never
		// notifies or touches the dedup store. Only a real tick pages, so a monitoring loop
		// polling --status cannot inflate the occurrence count.
		return reportResumeWatchdogStatus(stdout, stderr, tickMode, plan, statusEvents, statusTh, *asJSON)
	}
	note("TICK %s plan=%d window=%gh cap=%d", tickMode, len(plan), *windowH, *maxPerTick)
	rwRecordWatchdogStatusTick(statusLedgerPath, tickMode, plan, statusEvents)
	// Post-reset backlog SLO gate (#3582). Folded from the SAME plan + ledger this tick just
	// recorded, so the page reflects this tick's depth. statusEvents predates this tick's
	// depth row on purpose â€” the fold appends the live plan depth as the current sample.
	rwEmitBacklogPage(regDir, logDir, resume.FoldWatchdogStatus(resume.WatchdogStatusInput{
		Mode:                   tickMode,
		NowUnix:                time.Now().Unix(),
		BacklogThreshold:       *backlogThreshold,
		BacklogTicks:           *backlogTicks,
		ThrottledAccounts:      throttledAccounts,
		ThrottledAccountsKnown: throttledKnown,
		Plan:                   plan,
		Events:                 statusEvents,
	}).Page, note)

	// Defense-in-depth: the accounts policy still offers as workers. fleet_sessions.py
	// already excludes non-workers when it writes the plan, but a stale plan file could
	// predate the policy â€” re-check here too. An empty roster disables the check (fail
	// open, matching the Python's tolerance of a broken accounts read).
	driveCarry := rwLoadDriveCarry(regDir)
	guards := resume.WatchdogGuards{
		SelfSID:        selfSID,
		WorkerAccounts: rwWorkerAccounts(home),
		MaxAttempts:    *maxAttempts,
		// Liveness gate (#3459): never fire a second `claude --resume` onto a session a live
		// driver is already advancing. The plan (fleet_sessions.py) walks the on-disk
		// transcripts and can classify a stale/older copy as STOPPED_APIERR while a newer copy
		// under another account dir is alive; this consults the same audited process census
		// `fak resume admit` counts with, so a live session is skipped regardless of the plan's
		// disposition. Fail-open: an unreadable process table yields an empty set (inert).
		LiveSIDs: liveResumeSIDs(),
		// Honor an operator's durable pause/drain/stop of a specific session (`fak resume
		// hold`). Keyed by the Claude transcript UUID the plan row carries â€” the one key the
		// watchdog and the operator surface share (the descriptor registry is guardTraceID-
		// keyed, a disjoint space). Fail-open: an absent/unreadable store leaves the guard inert.
		DriveStates: rwLoadDriveStates(regDir),
	}

	history := rwLoadHistory(ledgerPath)
	// Re-witness every launched-but-unproven session, including sessions no longer
	// present in resume_plan.json. A productive child may remove its own eligibility
	// before the next tick; the durable launch ledger remains the authoritative set
	// that still needs a transcript witness.
	rwRewitnessDroppedSessions(home, statusLedgerPath, tickMode, plan, history, statusEvents)

	fakExe, posture := rwChildArgs(note)

	launched := 0
	// One tick per sweep: a live-but-idle watchdog (all SKIPs, no launches) is now
	// distinguishable from a dead one â€” zero ticks means the sweep never ran. This is the
	// authoritative in-process count the ledger-derived reconstruction used to lose (#3803).
	resumemetrics.Tick()
	progressRecorded := map[string]bool{}
	procScan := &rwProcScan{}
	traceByUUID := rwLoadIdentity(regDir)
	for _, p := range plan {
		if launched >= *maxPerTick {
			note("  per-tick cap reached (%d)", *maxPerTick)
			break
		}
		sid8 := shortID(p.Session)
		if rwPlanAuthBlocked(p.Disp) {
			reason := fmt.Sprintf("plan disposition %s requires auth/login; automatic resume cannot fix it", strings.TrimSpace(p.Disp))
			note("  SKIP %s â€” %s", sid8, reason)
			if *live {
				rwAppendLedger(ledgerPath, map[string]any{
					"ts": rwNowISO(), "session": p.Session, "account": p.Account,
					"resume_account": p.ResumeAccount,
					"phase":          "settled", "action": "consolidate-auth-plan-row",
					"outcome": "unrecoverable", "cause": p.Disp, "reason": reason,
				})
			}
			continue
		}
		// Outcome-aware once-gate input: how did the LAST attempt actually end, per the
		// transcript's own terminal turn (ground truth, never the launcher's ledger row)?
		hist := history[p.Session]
		if handled, decision := rwApplyTrajectoryWatchdog(p, hist, procScan, traceByUUID, ledgerPath, *live); handled {
			note("  %s %s - %s", decision.Action, sid8, decision.Reason)
			continue
		}
		progress := rwReadResumeProgress(home, p.Session, hist)
		outcome := progress.Outcome
		// A child may die before writing a terminal transcript event. Its stderr is
		// still durable; read the newest capture back and classify the otherwise
		// UNKNOWN attempt so deterministic 400/auth deaths do not burn all retries.
		if len(hist) > 0 && progress.NewTurns == 0 {
			if childErr, ok := rwNewestResumeChildError(logDir, p.Session); ok {
				cause := resume.ClassifyAttemptError(childErr)
				if cause != resume.AttemptErrorUnknown {
					rwRecordAttemptCause(ledgerPath, tickMode, p.Session, cause)
					if cause.Unrecoverable() {
						note("SKIP %s: prior child failed deterministically (%s)", shortID(p.Session), cause)
						continue
					}
				}
			}
		}
		if len(hist) > 0 {
			rwRecordResumeProgress(statusLedgerPath, tickMode, p.Session, progress, hist, statusEvents, progressRecorded)
		}
		// Stale-latch check (#2368): a took/unknown outcome burns the once-gate, but the
		// latch lies once the session dies AGAIN â€” the sweep re-plans it every tick and
		// the gate skips it every tick, forever. With proof of a new death (no live
		// process holds the session id, transcript silent past the dead floor) the
		// outcome is revived to recoverable and the same attempt cap takes over.
		if resume.CountAttempts(hist) > 0 &&
			(outcome == resume.OutcomeProgressed || outcome == resume.OutcomeUnknown) {
			ev := rwReDeathEvidence(home, p.Session, procScan, time.Now(),
				resume.LastLaunchUnix(hist), progress.NewTurns > 0)
			if revived, released := resume.ReviveOutcome(outcome, ev); released {
				outcome = revived
				graceAge, basis := ev.TranscriptIdleSeconds, "post-progress transcript"
				if !ev.PostLaunchProgress {
					graceAge, basis = ev.LaunchAgeSeconds, "launch without a post-launch turn"
				}
				note("  REVIVE %s â€” no live process holds it and the %s has aged %s: stale once-latch released, session re-eligible (#2368/#8722)",
					sid8, basis, humanIdle(graceAge))
			}
		}
		effective := guards
		if rwHarness(p) == "codex" {
			// Codex exec sessions are not Anthropic seats and have no CLAUDE_CONFIG_DIR.
			// Their explicit targeted-plan coordinates are the admission contract.
			effective.WorkerAccounts = nil
		}
		d := resume.DecideWatchdogRow(p, effective, hist, outcome)
		nextMove := sessionctl.Move{
			Kind: sessionctl.MoveHalt, Render: sessionctl.RenderStop, Session: sessionctl.SessionAutonomous,
			Gate: string(d.Action), Source: "resume-watchdog", Reason: d.Reason,
		}
		if d.Action == resume.WatchdogLaunch {
			nextMove.Kind, nextMove.Render = sessionctl.MoveContinue, sessionctl.RenderSystemDirective
			// Identify the resume by session id, not p.ResumeTarget(): the resume target is
			// the account config dir (e.g. .../.claude-secret), and this move is witnessed
			// into the decision ledger â€” a raw config-dir path there leaks the account's
			// secret dir. The actual resume still reads ResumeTarget() directly below.
			nextMove.Payload = p.Session
		}
		result := sessionctl.ApplyResult{Applied: d.Action == resume.WatchdogLaunch}
		if d.Action != resume.WatchdogLaunch {
			result.Refusal = d.Reason
		}
		if nextRecord, err := sessionctl.WitnessMove(nextMove, result); err != nil {
			note("  NEXT-WITNESS %s skipped (fail-open): %v", sid8, err)
		} else {
			rwAppendLedger(ledgerPath, map[string]any{"ts": rwNowISO(), "session": p.Session, "phase": "decision", "next": nextRecord})
		}
		// Count the per-session verdict the moment it is decided â€” launch, skip_self,
		// skip_blocked, skip_operator_hold, â€¦ â€” so /debug/vars carries the live verdict mix
		// independent of any ledger write that may or may not land (#3803).
		resumemetrics.RecordAction(string(d.Action))
		if d.Action != resume.WatchdogLaunch {
			note("  SKIP %s â€” %s", sid8, d.Reason)
			continue
		}
		signature := rwResumeStormSignature(p)
		backoff := resumebackoff.Decide(resumebackoff.Input{
			Session:         p.Session,
			Signature:       signature,
			Now:             time.Now().UTC(),
			History:         backoffHistory,
			CrashLoopBudget: rwCrashLoopBudget(),
		})
		if !backoff.Eligible {
			phase := "deferred"
			if backoff.Quarantined || backoff.Reason == resumebackoff.ReasonCrashLoopQuarantined {
				phase = "quarantined"
			}
			note("  SKIP %s — %s repeat=%d next=%s", sid8, backoff.Reason, backoff.Repeat, backoff.NextEligible.Format(time.RFC3339))
			rwAppendLedger(ledgerPath, map[string]any{"ts": rwNowISO(), "session": p.Session, "signature": signature, "phase": phase, "reason": backoff.Reason, "repeat": backoff.Repeat, "next_eligible": backoff.NextEligible})
			continue
		}
		acct := rwAccountTag(p.Account)
		resumeCfg := p.ResumeTarget()
		grant := launchSpawnBroker(rwResumeBrokerAttempt(fakExe, claudeExe, p, resumeCfg, posture, driveCarry[p.Session]))
		if !*live {
			if !grant.Allow {
				note("  WOULD DENY %s acct=%s proj=%s â€” spawn broker: %s agent_run=%s policy_digest=%s",
					sid8, acct, p.Project, grant.Reason, grant.Metadata.AgentRunID, grant.Metadata.PolicyDigest)
				continue
			}
			note("  WOULD RESUME %s acct=%s proj=%s agent_run=%s policy_digest=%s",
				sid8, acct, p.Project, grant.Metadata.AgentRunID, grant.Metadata.PolicyDigest)
			continue
		}

		// Host-wide per-source admission (#1341/#1344): may the BOX take one more live
		// resume across ALL accounts right now? A DEFER is recorded with a non-launch
		// phase so it never counts as launch pressure, and the session stays eligible
		// next tick. Fails open â€” a broken gate must never strand the whole watchdog.
		if admit, reason := rwSourceAdmit(ledgerPath, regDir, time.Now()); !admit {
			note("  DEFER %s acct=%s â€” per-source gate: %s", sid8, acct, reason)
			rwAppendLedger(ledgerPath, map[string]any{
				"ts": rwNowISO(), "session": p.Session, "account": p.Account,
				"resume_account": p.ResumeAccount,
				"phase":          "deferred", "cause": "source_concurrency_gate", "reason": reason,
			})
			continue
		}

		if !grant.Allow {
			note("  DENY %s acct=%s â€” spawn broker: %s agent_run=%s policy_digest=%s",
				sid8, acct, grant.Reason, grant.Metadata.AgentRunID, grant.Metadata.PolicyDigest)
			rwAppendLedger(ledgerPath, map[string]any{
				"ts": rwNowISO(), "session": p.Session, "account": p.Account,
				"resume_account": p.ResumeAccount,
				"phase":          "broker_denied", "cause": p.Disp, "reason": grant.Reason,
				"agent_run_id": grant.Metadata.AgentRunID, "policy_digest": grant.Metadata.PolicyDigest,
			})
			continue
		}

		if p.Rehomed {
			if !rehome.RehomeTranscript(p.RehomeSource(), resumeCfg, p.Project, p.Session, nil) {
				note("  SKIP %s â€” re-home source transcript missing", sid8)
				continue
			}
			note("  RE-HOME %s %s -> %s (transcript copied; resuming on healthy account)",
				sid8, p.Account, p.ResumeAccount)
		}

		pid, err := rwSpawnResumeLaunch(claudeExe, p, resumeCfg, logDir, grant)
		if err != nil {
			note("  FAIL %s â€” spawn: %v", sid8, err)
			continue
		}
		// Record the launch BEFORE anything else â€” a crash cannot double-launch in this
		// tick. The gate keys on OUTCOME, not mere presence: phase="launched" marks an
		// attempt whose result is unknown until the next tick reads the transcript.
		attempt := d.Attempt
		anchor := rwResumeAnchor(p.Session)
		row := map[string]any{
			"ts": rwNowISO(), "session": p.Session, "account": p.Account,
			"resume_account": p.ResumeAccount, "rehomed": p.Rehomed,
			"project": p.Project, "pid": pid, "cause": p.Disp,
			"phase": "launched", "attempt": attempt, "signature": signature,
			"resume_anchor": anchor, "harness": rwHarness(p),
		}
		if rwHarness(p) == "codex" {
			row["result_file"] = p.ResultFile
		}
		rwAppendLedger(ledgerPath, row)
		// #4139/#4216: record the OS-relaunch reset transaction â€” the transcript-UUID-keyed
		// analogue of session.ResetTransaction (#1582) â€” into its own durable, append-only store
		// next to the launched ledger row, so a hidden relaunch is no longer invisible to the
		// reset-transaction audit chain. The pure constructor (#4139) carries no clock and leaves
		// TS ""; the shell stamps the write time here, mirroring the drivestate store's discipline.
		if rwHarness(p) != "codex" {
			relaunchReset := resume.NewRelaunchResetRow(p, attempt)
			relaunchReset.TS = rwNowISO()
			rwAppendLedger(rwRelaunchResetLedger(regDir), relaunchReset)
		}
		// #4140: record the cache route this relaunch carried (RelaunchCacheAffinityEnv on the
		// child env, derived from the transcript UUID) into its own durable, append-only store,
		// so an operator can audit which warm route a relaunch used and the fold
		// (rwLoadRelaunchAffinity) gives launch plumbing a last-row-wins lookup. Same TS
		// discipline as the reset row: the pure constructor carries no clock, the shell stamps.
		if rwHarness(p) != "codex" {
			relaunchAffinity := resume.NewRelaunchAffinityRow(p.Session)
			relaunchAffinity.TS = rwNowISO()
			rwAppendLedger(rwRelaunchAffinityLedger(regDir), relaunchAffinity)
		}
		// A3 (#4114): refresh the durable uuid<->trace identity row for the just-resumed UUID
		// so its newest row names the account the resume re-homed onto. The resumed child can
		// never self-record this â€” WatchdogChildEnv strips CLAUDE_CODE_SESSION_ID from its env
		// (the mass-crash fix), so the guard-SessionStart producer is blind on the resume path;
		// the watchdog, which still holds the UUID as p.Session, records it here. Best-effort
		// and inherently live-only â€” this block runs only past the live spawn above, mirroring
		// the ledger-append gating.
		if rwHarness(p) != "codex" {
			rwRefreshResumeIdentity(regDir, p)
		}
		history[p.Session] = append(hist, resume.Attempt{UnixSeconds: time.Now().Unix(), Phase: "launched"})
		launched++
		note("  RESUMED %s acct=%s pid=%d (attempt %d/%d; re-eligible only if it fails recoverably)",
			sid8, acct, pid, attempt, *maxAttempts)
		rwToast(logDir, "Resumed dead session", fmt.Sprintf("%s  (%s / %s)", sid8, acct, p.Project), "info")
		if *spacingSec > 0 && launched < *maxPerTick {
			// Pace the next spawn so a burst does not slam the shared rate budget and trip
			// the transient 529 that strands a whole batch.
			time.Sleep(time.Duration(*spacingSec * float64(time.Second)))
		}
	}

	// 2. Alert on true login-blocked accounts â€” once per account blocker.
	rwAuthAlerts(regDir, logDir, note)

	note("  done: launched=%d sessions_in_ledger=%d", launched, len(history))
	return 0
}

// The pre-gate screens (self-resume guard, worker-account policy) and the probe-mode
// resolution live in the pure leaf: resume.DecideWatchdogRow / ResolveWatchdogProbeMode.

// resolveWatchdogLogDir resolves the watchdog log dir from the --log-dir flag, then
// $FAK_WATCHDOG_LOG_DIR, then <repo>/tools/_watchdog.
func rwChildArgs(note func(string, ...any)) (string, []string) {
	fakExe, err := os.Executable()
	if err != nil || strings.TrimSpace(fakExe) == "" {
		fakExe = "fak"
	}
	posture, postureErr := fleetGuardCachePostureArgs()
	if postureErr != nil {
		note("  WARN managed-cache: %v — ignoring; resuming passive", postureErr)
		posture = nil
	}
	if len(posture) > 0 && strings.TrimSpace(fakExe) == "" {
		note("  WARN managed-cache posture configured but `fak` is unavailable — resuming children directly (passive, no posture banner)")
		posture = nil
	} else if len(posture) > 0 {
		note("  managed-cache posture -> fronting resumed children with `fak guard %s --`", strings.Join(posture, " "))
	}
	return fakExe, posture
}

func resolveWatchdogLogDir(logDirFlag string) string {
	logDir := logDirFlag
	if logDir == "" {
		if v := strings.TrimSpace(os.Getenv("FAK_WATCHDOG_LOG_DIR")); v != "" {
			logDir = v
		} else {
			cwd, _ := os.Getwd()
			logDir = filepath.Join(findRepoRoot(cwd), "tools", "_watchdog")
		}
	}
	return logDir
}

// watchdogStatusThresholds groups the --status red-flag thresholds so the status reporter
// does not take a wide positional parameter list.
type watchdogStatusThresholds struct {
	silentHours        float64
	unprovenMinutes    float64
	monotonicTicks     int
	launchStaleMinutes float64
	// The post-reset backlog SLO gate (#3582) and the roster fact it needs. throttledKnown
	// false (unreadable roster) keeps the gate silent â€” see rwThrottledAccounts.
	backlogThreshold  int
	backlogTicks      int
	throttledAccounts int
	throttledKnown    bool
}

// reportResumeWatchdogStatus renders the read-only drain status (--status): fold the plan +
// status events into a verdict, print it (JSON with --json, else the table), and return the
// exit code (3 when the drain is red, else 0; a JSON encode failure returns its own code).
func reportResumeWatchdogStatus(stdout, stderr io.Writer, tickMode string, plan []resume.WatchdogPlanRow, statusEvents []resume.WatchdogStatusEvent, th watchdogStatusThresholds, asJSON bool) int {
	rep := resume.FoldWatchdogStatus(resume.WatchdogStatusInput{
		Mode:               tickMode,
		NowUnix:            time.Now().Unix(),
		SilentSeconds:      int64(th.silentHours * 3600),
		UnprovenSeconds:    int64(th.unprovenMinutes * 60),
		LaunchStaleSeconds: int64(th.launchStaleMinutes * 60),
		MonotonicTicks:     th.monotonicTicks,

		BacklogThreshold:       th.backlogThreshold,
		BacklogTicks:           th.backlogTicks,
		ThrottledAccounts:      th.throttledAccounts,
		ThrottledAccountsKnown: th.throttledKnown,

		Plan:   plan,
		Events: statusEvents,
	})
	if asJSON {
		code := encodeJSONOrFail(stdout, stderr, rep, "fak resume watchdog --status")
		if code != 0 {
			return code
		}
	} else {
		renderResumeWatchdogStatus(stdout, rep)
	}
	if rep.Verdict == resume.WatchdogDrainRed {
		return 3
	}
	return 0
}

func rwRewitnessDroppedSessions(home, statusPath, mode string, plan []resume.WatchdogPlanRow, history map[string][]resume.Attempt, events []resume.WatchdogStatusEvent) {
	planned := map[string]bool{}
	for _, row := range plan {
		planned[row.Session] = true
	}
	proven := map[string]bool{}
	for _, event := range events {
		if strings.EqualFold(event.Phase, "progress") || event.NewTurns > 0 || event.ProgressWitnessUnix > 0 {
			proven[event.Session] = true
		}
	}
	for sid, attempts := range history {
		if sid == "" || planned[sid] || proven[sid] || resume.CountAttempts(attempts) == 0 {
			continue
		}
		progress := rwReadResumeProgress(home, sid, attempts)
		if progress.NewTurns > 0 && progress.Outcome == resume.OutcomeProgressed {
			rwAppendLedger(statusPath, map[string]any{
				"ts": rwNowISO(), "mode": mode, "session": sid, "phase": "progress",
				"new_turns": progress.NewTurns, "progress_model": progress.ProgressModel,
				"progress_witnessed_at": progress.ProgressUnix, "rewitnessed_after_plan_drop": true,
			})
		}
	}
}

type rwResumeProgress struct {
	Outcome      resume.Outcome
	NewTurns     int
	ProgressUnix int64
	// ProgressModel is the model that served the recovery turns â€” the witness that a resume
	// took AND on which model, so the durable ledger can later prove not just "it recovered"
	// but "it recovered on Opus 4.8" (or flag that it drifted onto something else).
	ProgressModel string
}

func rwNewestResumeChildError(logDir, sid string) (string, bool) {
	pattern := filepath.Join(logDir, fmt.Sprintf("resume-%s-*.log.err", shortID(sid)))
	paths, _ := filepath.Glob(pattern)
	if len(paths) == 0 {
		return "", false
	}
	sort.Slice(paths, func(i, j int) bool {
		ai, _ := os.Stat(paths[i])
		aj, _ := os.Stat(paths[j])
		return ai != nil && aj != nil && ai.ModTime().After(aj.ModTime())
	})
	raw, err := os.ReadFile(paths[0])
	if err != nil || len(bytes.TrimSpace(raw)) == 0 {
		return "", false
	}
	if len(raw) > 64<<10 {
		raw = raw[len(raw)-(64<<10):]
	}
	return string(raw), true
}

func rwRecordAttemptCause(path, mode, sid string, cause resume.AttemptErrorClass) {
	rwAppendLedger(path, map[string]any{
		"ts": rwNowISO(), "mode": mode, "session": sid, "phase": "attempt_failed", "reason": cause,
	})
}

// rwReadResumeProgress reads the newest transcript copy for a session and returns the
// transcript-grounded facts that prove whether a prior resume took: the terminal outcome
// and the first real model turn after the last launch. No transcript means unknown, which
// preserves the conservative burn-once gate but emits no recovery witness.
func rwReadResumeProgress(home, sid string, hist []resume.Attempt) rwResumeProgress {
	out := rwResumeProgress{Outcome: resume.OutcomeUnknown}
	path := rwNewestTranscript(home, sid)
	if path == "" {
		return out
	}
	f, err := os.Open(path)
	if err != nil {
		return out
	}
	tr := scanTranscriptForStatus(f)
	f.Close()
	out.Outcome = resume.ClassifyOutcome(rwTerminalSignal(tr.terminalText))
	lastLaunch := resume.LastLaunchUnix(hist)
	for _, ts := range tr.turnTimes {
		if ts > lastLaunch {
			out.NewTurns++
			if out.ProgressUnix == 0 {
				out.ProgressUnix = ts
			}
		}
	}
	out.ProgressModel, _ = resume.ResumeModels(tr.turnTimes, tr.turnModels, lastLaunch)
	return out
}

func rwRecordResumeProgress(ledgerPath, mode, sid string, progress rwResumeProgress, hist []resume.Attempt, events []resume.WatchdogStatusEvent, recorded map[string]bool) {
	lastLaunch := resume.LastLaunchUnix(hist)
	if sid == "" || lastLaunch <= 0 || progress.NewTurns <= 0 || progress.ProgressUnix <= lastLaunch {
		return
	}
	if recorded[sid] || rwHasProgressWitness(events, sid, lastLaunch) {
		return
	}
	row := map[string]any{
		"ts":                      time.Unix(progress.ProgressUnix, 0).UTC().Format("2006-01-02T15:04:05Z"),
		"session":                 sid,
		"phase":                   "progress",
		"mode":                    mode,
		"new_turns":               progress.NewTurns,
		"progress_witnessed_at":   time.Unix(progress.ProgressUnix, 0).UTC().Format("2006-01-02T15:04:05Z"),
		"progress_witness_source": "transcript_real_turn_after_resume",
	}
	if progress.ProgressModel != "" {
		row["progress_model"] = progress.ProgressModel
	}
	rwAppendLedger(ledgerPath, row)
	recorded[sid] = true
	// Live twin of the ledger's "progress" row: a resume proven to have produced a real
	// post-launch turn, counted at the moment it is witnessed (#3803).
	resumemetrics.ProgressWitnessed()
}

// rwSoftWatchdog is the tick-spanning soft-watchdog episode tracker (#5287). It is
// process-wide so a session that stays wedged across many ticks is dumped ONCE per
// stall episode rather than once per tick, and re-arms the moment its curve moves
// again or it dies into the hard revive path.
var rwSoftWatchdog = resume.NewSoftWatchdog(0)

func rwApplyTrajectoryWatchdog(p resume.WatchdogPlanRow, hist []resume.Attempt, scan *rwProcScan, traceByUUID map[string]string, ledgerPath string, live bool) (bool, resume.TrajectoryWatchdogDecision) {
	anchor := rwResumeAnchor(p.Session)
	if !anchor.Present || anchor.Curve == nil {
		return false, resume.TrajectoryWatchdogDecision{}
	}
	cmdline, alive, known := scan.sessionProcess(p.Session)
	if !known {
		return false, resume.TrajectoryWatchdogDecision{}
	}
	nudged := false
	for _, a := range hist {
		if strings.EqualFold(strings.TrimSpace(a.Action), "trajectory_nudge") {
			nudged = true
		}
	}
	// #5287 soft watchdog (sglang's soft/hard split, clean-room): BEFORE the
	// intervention core decides anything, capture the alive-but-stalled session's
	// diagnostic state into the SAME durable session record the decision itself
	// lands in, so the wedge stays debuggable after the nudge/revive has moved it.
	// Strictly observe-only: the decision below is byte-identical with or without
	// this block, the capture is gated behind the soft grace timeout, and a dead
	// session is left entirely to the hard path.
	if dump, captured := rwSoftWatchdog.Observe(resume.SoftObservationFromCurve(p.Session, alive, anchor.Curve, cmdline, time.Now().UTC())); captured {
		row := resume.NewSoftDumpRow(dump, traceByUUID[p.Session])
		row.TS = rwNowISO()
		rwAppendLedger(ledgerPath, row)
	}
	decision := resume.DecideTrajectoryWatchdog(resume.TrajectoryWatchdogInput{Alive: alive, Signal: anchor.Curve.Signal, NudgeAttempted: nudged})
	if decision.Action == resume.TrajectoryReviveAnchor {
		return false, decision
	}
	if decision.Action == resume.TrajectoryNudge && live {
		trace := traceByUUID[p.Session]
		ref := sessionctl.EnqueueRedirect(trace, sessionctl.Redirect{ObjectiveID: anchor.ObjectiveID, Goal: anchor.Objective, Witness: anchor.Curve.Detail})
		row := map[string]any{"ts": rwNowISO(), "session": p.Session, "phase": "trajectory_decision", "action": "trajectory_nudge", "decision": decision, "resume_anchor": anchor, "trace": trace}
		if ref != nil {
			row["refusal"] = ref.Error()
		}
		rwAppendLedger(ledgerPath, row)
	}
	return true, decision
}

func rwHasProgressWitness(events []resume.WatchdogStatusEvent, sid string, afterUnix int64) bool {
	for _, e := range events {
		if e.Session != sid {
			continue
		}
		at := rwFirstNonZeroInt64(e.ProgressWitnessUnix, e.UnixSeconds)
		if at <= afterUnix {
			continue
		}
		if e.NewTurns > 0 || e.ProgressWitnessUnix > 0 || e.LedgerProgress || strings.TrimSpace(e.CommitSHA) != "" ||
			strings.EqualFold(strings.TrimSpace(e.Phase), "progress") {
			return true
		}
	}
	return false
}

func rwFirstNonZeroInt64(vals ...int64) int64 {
	for _, v := range vals {
		if v > 0 {
			return v
		}
	}
	return 0
}

// rwCollectProcCmdlines returns every live process's command line (via the same
// collector `fak ps` uses) â€” one snapshot per tick, taken lazily. Injectable seam for
// tests. ok=false means the table could not be read, so liveness is unknown.
var rwCollectProcCmdlines = func() ([]string, bool) {
	procs, errStr := procguard.CollectRelations()
	if errStr != "" {
		return nil, false
	}
	out := make([]string, 0, len(procs))
	for _, p := range procs {
		if p.Cmdline != "" {
			out = append(out, p.Cmdline)
		}
	}
	return out, true
}

// rwProcScan lazily snapshots the process table once per tick â€” only a row that would
// otherwise be skipped on the stale took-latch pays for the scan.
type rwProcScan struct {
	once     sync.Once
	cmdlines []string
	ok       bool
}

// sessionLive reports whether any live process's command line names sid (the
// `claude --resume <sid>` child), and whether the scan itself succeeded.
func (s *rwProcScan) sessionLive(sid string) (live, ok bool) {
	_, live, ok = s.sessionProcess(sid)
	return live, ok
}

// sessionProcess is sessionLive plus the matched command line â€” the only worker-side
// state the tick's existing snapshot can show, and what the soft watchdog's diagnostic
// dump records as the stalled session's pending action (#5287). It costs nothing extra:
// the same lazily-taken scan answers both.
func (s *rwProcScan) sessionProcess(sid string) (cmdline string, live, ok bool) {
	s.once.Do(func() { s.cmdlines, s.ok = rwCollectProcCmdlines() })
	if !s.ok || sid == "" {
		return "", false, s.ok
	}
	for _, c := range s.cmdlines {
		if strings.Contains(c, sid) {
			return c, true, true
		}
	}
	return "", false, true
}

// rwReDeathEvidence gathers the stale-latch facts (#2368/#8722) for one planned
// session: process liveness from the tick's table snapshot, transcript idleness from
// the newest copy's mtime, and the launch age used only when no post-launch model turn
// exists. Missing/unreadable timestamps stay -1 and never prove a retry.
func rwReDeathEvidence(home, sid string, scan *rwProcScan, now time.Time, lastLaunchUnix int64, postLaunchProgress bool) resume.ReDeathEvidence {
	ev := resume.ReDeathEvidence{
		TranscriptIdleSeconds: -1,
		LaunchAgeSeconds:      -1,
		PostLaunchProgress:    postLaunchProgress,
	}
	ev.ProcessLive, ev.ProcessScanOK = scan.sessionLive(sid)
	if lastLaunchUnix > 0 {
		ev.LaunchAgeSeconds = int64(now.Sub(time.Unix(lastLaunchUnix, 0)).Seconds())
	}
	if path := rwNewestTranscript(home, sid); path != "" {
		if fi, err := os.Stat(path); err == nil {
			ev.TranscriptIdleSeconds = int64(now.Sub(fi.ModTime()).Seconds())
		}
	}
	return ev
}

func rwResumeStormSignature(p resume.WatchdogPlanRow) string {
	// Disposition and CWD describe a recovery cohort, not one crashing workload.
	// Include the transcript UUID so repeats of the same logical session retain a
	// stable backoff key without parking unrelated sessions sharing a repository.
	return knownbad.Signature(firstString(p.Disp, "resume_crash"), []string{p.CWD},
		"resume-session:"+strings.TrimSpace(p.Session))
}

// rwTerminalSignal classifies a transcript's terminal-turn text into the closed
// TerminalSignal facts resume.ClassifyOutcome folds. One deliberate widening over the
// Python watchdog's ad-hoc "overloaded/529" check: the transient family is
// sessionsignals.IsAPIError â€” the same taxonomy every other tool in this family reads,
// so the watchdog can never disagree with the sweep about what is transient.
func rwTerminalSignal(text string) resume.TerminalSignal {
	if strings.TrimSpace(text) == "" {
		return resume.TerminalSignal{}
	}
	return resume.TerminalSignal{
		Found:             true,
		AuthWall:          sessionsignals.IsAuthError(text) || sessionsignals.NeedsLoginPrompt(text),
		LimitWall:         sessionsignals.IsLimitError(text),
		TransientAPIError: sessionsignals.IsAPIError(text),
	}
}

// rwNewestTranscript is the most-recently-modified copy of a session's transcript across
// ALL account dirs (a re-home writes a fresh copy under the target account).
func rwNewestTranscript(home, sid string) string {
	matches, _ := filepath.Glob(filepath.Join(home, ".claude*", "projects", "*", sid+".jsonl"))
	best, bestMod := "", time.Time{}
	for _, m := range matches {
		if fi, err := os.Stat(m); err == nil && !fi.IsDir() && fi.ModTime().After(bestMod) {
			best, bestMod = m, fi.ModTime()
		}
	}
	return best
}

// rwLoadPlan reads the AUTO_RESUME plan fleet_sessions.py writes. Missing/malformed
// yields an empty plan (the tick logs plan=0 and does nothing), never a crash.
func rwLoadPlan(path string) []resume.WatchdogPlanRow {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil
	}
	raw = bytes.TrimPrefix(raw, []byte{0xEF, 0xBB, 0xBF})
	var doc struct {
		Plan []resume.WatchdogPlanRow `json:"plan"`
	}
	if json.Unmarshal(raw, &doc) != nil {
		return nil
	}
	return doc.Plan
}

func rwPlanAuthBlocked(disp string) bool {
	return strings.Contains(strings.ToUpper(strings.TrimSpace(disp)), "AUTH")
}

// rwLoadHistory groups the durable resume ledger per session as typed Attempts, so the
// gate reasons about the OUTCOME and attempt count of prior resumes, not their existence.
func rwLoadHistory(path string) map[string][]resume.Attempt {
	out := map[string][]resume.Attempt{}
	raw, err := os.ReadFile(path)
	if err != nil {
		return out
	}
	for _, line := range strings.Split(string(raw), "\n") {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		var rec struct {
			TS             string `json:"ts"`
			Session        string `json:"session"`
			Phase          string `json:"phase"`
			Action         string `json:"action"`
			ManualOverride bool   `json:"manual_override"`
		}
		if json.Unmarshal([]byte(line), &rec) != nil || rec.Session == "" {
			continue
		}
		var unix int64
		if t, err := time.Parse(time.RFC3339, rec.TS); err == nil {
			unix = t.Unix()
		}
		out[rec.Session] = append(out[rec.Session], resume.Attempt{
			UnixSeconds: unix, Phase: rec.Phase, Action: rec.Action, ManualOverride: rec.ManualOverride,
		})
	}
	return out
}

// rwLoadWatchdogStatusEvents reads the same durable ledger as typed drain-steward facts.
// It accepts forward-extended rows and ignores malformed lines, so a status readout never
// depends on a perfect ledger.
func rwLoadWatchdogStatusEvents(path string) []resume.WatchdogStatusEvent {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil
	}
	var out []resume.WatchdogStatusEvent
	for _, line := range strings.Split(string(raw), "\n") {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		var rec struct {
			TS                  string `json:"ts"`
			Session             string `json:"session"`
			Phase               string `json:"phase"`
			Action              string `json:"action"`
			ManualOverride      bool   `json:"manual_override"`
			Mode                string `json:"mode"`
			AutoResumeDepth     int    `json:"auto_resume_depth"`
			NewTurns            int    `json:"new_turns"`
			CommitSHA           string `json:"commit_sha"`
			Commit              string `json:"commit"`
			LedgerProgress      bool   `json:"ledger_progress"`
			DetectedAt          string `json:"detected_at"`
			ResumedAt           string `json:"resumed_at"`
			ProgressWitnessedAt string `json:"progress_witnessed_at"`
		}
		if json.Unmarshal([]byte(line), &rec) != nil {
			continue
		}
		if rec.Phase == "" && rec.Session == "" {
			continue
		}
		commit := strings.TrimSpace(rec.CommitSHA)
		if commit == "" {
			commit = strings.TrimSpace(rec.Commit)
		}
		phase := rec.Phase
		if rec.ManualOverride || strings.HasPrefix(strings.ToLower(strings.TrimSpace(rec.Action)), "consolidate") {
			phase = "settled"
		}
		out = append(out, resume.WatchdogStatusEvent{
			UnixSeconds:         parseTranscriptUnix(rec.TS),
			Session:             rec.Session,
			Phase:               phase,
			Mode:                rec.Mode,
			AutoResumeDepth:     rec.AutoResumeDepth,
			NewTurns:            rec.NewTurns,
			CommitSHA:           commit,
			LedgerProgress:      rec.LedgerProgress,
			DetectedUnix:        parseTranscriptUnix(rec.DetectedAt),
			ResumedUnix:         parseTranscriptUnix(rec.ResumedAt),
			ProgressWitnessUnix: parseTranscriptUnix(rec.ProgressWitnessedAt),
		})
	}
	return out
}

// rwRecordWatchdogStatusTick leaves the durable breadcrumbs --status needs later:
// a depth sample every tick, and one first-seen queue row per planned session.
func rwRecordWatchdogStatusTick(ledgerPath, mode string, plan []resume.WatchdogPlanRow, existing []resume.WatchdogStatusEvent) {
	ts := rwNowISO()
	rwAppendLedger(ledgerPath, map[string]any{
		"ts":                ts,
		"phase":             "status",
		"mode":              mode,
		"auto_resume_depth": len(plan),
	})
	seenQueued := map[string]bool{}
	for _, e := range existing {
		switch strings.ToLower(strings.TrimSpace(e.Phase)) {
		case "queued", "detected", "auto_resume":
			if e.Session != "" {
				seenQueued[e.Session] = true
			}
		}
	}
	for _, p := range plan {
		if p.Session == "" || seenQueued[p.Session] {
			continue
		}
		rwAppendLedger(ledgerPath, map[string]any{
			"ts":             ts,
			"session":        p.Session,
			"account":        p.Account,
			"resume_account": p.ResumeAccount,
			"project":        p.Project,
			"phase":          "queued",
			"mode":           mode,
			"cause":          p.Disp,
		})
		seenQueued[p.Session] = true
	}
}
