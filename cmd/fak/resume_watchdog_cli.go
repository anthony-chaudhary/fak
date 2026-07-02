package main

// resume_watchdog_cli.go — `fak resume watchdog`, ONE TICK of the cross-account resume
// layer for ALL autonomous Claude sessions on this host. Go port of
// tools/fleet_resume_watchdog.py / .ps1 (the executing half; every decision it enforces
// is an already-audited leaf):
//
//	fak resume watchdog                 # dry-run: log what it WOULD resume
//	fak resume watchdog --live          # actually resume (outcome-gated, capped, paced)
//	FAK_LIVE=1 fak resume watchdog      # same, for cron/scheduled ticks
//
// Each tick:
//  1. EXTRACT-IN-ADVANCE — refresh the on-disk session registry + AUTO_RESUME plan via
//     tools/fleet_sessions.py (the one remaining Python child; a missing interpreter
//     degrades to the existing plan file with a note, never a crash).
//  2. Gate each planned session through the audited decisions: the self-resume guard,
//     the worker-account policy, the outcome-aware once-gate (resume.RetryGate — a
//     resume that died recoverably stays eligible up to the attempt cap; a clean finish
//     or an auth wall burns it), and on a live tick the host-wide per-source admission
//     (resume.AdmitSource — the 529 burst-wall ceiling `fak resume admit` exposes).
//  3. Re-home the transcript when the plan says so (rehome.RehomeTranscript), spawn
//     `claude --resume` under the owning account's CLAUDE_CONFIG_DIR (hidden window on
//     Windows), record the launch in the durable ledger BEFORE anything else, and pace
//     the next spawn so a burst does not self-congest.
//  4. Alert (notifications.log + macOS toast when available) on accounts that need a
//     human re-login — once per account blocker.
//
// Safety rails (faithful to the .ps1/.py): DRY-RUN by default; per-tick launch cap;
// launch spacing; ledger-first recording so a crash cannot double-launch in one tick.
// Slack posting is NOT ported yet (the Python's --slack seam) — follow-on, see the
// tools/slack_post parity note in the goal issue.

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/fleetaccounts"
	"github.com/anthony-chaudhary/fak/internal/resume"
	"github.com/anthony-chaudhary/fak/internal/resume/rehome"
	"github.com/anthony-chaudhary/fak/internal/sessionsignals"
	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

// resumeWatchdogPrompt is the standing re-entry instruction a resumed session receives.
const resumeWatchdogPrompt = "Resume where you left off; re-establish any /goal or /loop and continue toward it."

// The plan row type is the leaf's resume.WatchdogPlanRow (json-tagged with the
// resume_plan.json key names), so the shell and the pure guard chain read one shape.

// runResumeWatchdog executes one watchdog tick. Exit codes: 0 ok (including a clean
// dry-run), 1 runtime error, 2 usage error.
func runResumeWatchdog(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("resume watchdog", flag.ContinueOnError)
	fs.SetOutput(stderr)
	live := fs.Bool("live", rwEnvBool("FAK_LIVE"), "actually resume (default: dry-run; env FAK_LIVE=1)")
	windowH := fs.Float64("window-h", rwEnvFloat("FAK_WINDOW_H", 6), "registry window in hours (env FAK_WINDOW_H)")
	maxPerTick := fs.Int("max-per-tick", rwEnvInt("FAK_MAX_PER_TICK", 4), "max resumes launched per tick (env FAK_MAX_PER_TICK)")
	maxAttempts := fs.Int("max-attempts", rwEnvInt("FAK_MAX_ATTEMPTS", resume.DefaultMaxResumeAttempts), "give-up cap on automatic resumes of one session (env FAK_MAX_ATTEMPTS)")
	spacingSec := fs.Float64("spacing-sec", rwEnvFloat("FAK_LAUNCH_SPACING_SEC", 8), "seconds between spawns in one tick, so a burst does not trip the per-source 529 wall (0 = all at once; env FAK_LAUNCH_SPACING_SEC)")
	probeMode := fs.String("probe", rwEnvStr("FAK_PROBE", "auto"), "registry refresh probe mode: auto|blocked|stale|all|none (auto = blocked on --live, none on dry-run; env FAK_PROBE)")
	probeMinIntervalMin := fs.Int("probe-min-interval-min", rwEnvInt("FAK_PROBE_MIN_INTERVAL_MIN", 20), "min minutes between active probes of one account (env FAK_PROBE_MIN_INTERVAL_MIN)")
	regDirFlag := fs.String("reg-dir", "", "registry dir holding resume_plan.json / resume_ledger.jsonl / sessions.json (default: $FLEET_REG_DIR, else <repo>/tools/_registry)")
	logDirFlag := fs.String("log-dir", "", "watchdog log dir (default: $FAK_WATCHDOG_LOG_DIR, else <repo>/tools/_watchdog)")
	noRefresh := fs.Bool("no-refresh", false, "skip the fleet_sessions.py registry refresh and act on the existing plan file (offline/test)")
	statusOnly := fs.Bool("status", false, "print the read-only drain status from resume_plan.json + resume_ledger.jsonl, then exit")
	asJSON := fs.Bool("json", false, "with --status, emit the machine-readable drain report")
	silentHours := fs.Float64("silent-hours", rwEnvFloat("FAK_RESUME_SILENT_HOURS", 2), "with --status, mark red when any unrecovered queued row is silent this many hours (env FAK_RESUME_SILENT_HOURS)")
	monotonicTicks := fs.Int("monotonic-ticks", rwEnvInt("FAK_RESUME_MONOTONIC_TICKS", 3), "with --status, mark red when AUTO_RESUME depth grows for this many consecutive ticks (env FAK_RESUME_MONOTONIC_TICKS)")
	if err := fs.Parse(argv); err != nil {
		return 2
	}

	regDir := resolveSweepRegDir(*regDirFlag)
	logDir := *logDirFlag
	if logDir == "" {
		if v := strings.TrimSpace(os.Getenv("FAK_WATCHDOG_LOG_DIR")); v != "" {
			logDir = v
		} else {
			cwd, _ := os.Getwd()
			logDir = filepath.Join(findRepoRoot(cwd), "tools", "_watchdog")
		}
	}
	_ = os.MkdirAll(logDir, 0o755)
	note := func(format string, a ...any) { rwNote(logDir, stdout, fmt.Sprintf(format, a...)) }

	home, _ := os.UserHomeDir()
	claudeExe := rwClaudeExe(home)
	selfSID := strings.TrimSpace(os.Getenv("CLAUDE_CODE_SESSION_ID"))

	// 1. Refresh registry + plan (extract in advance). On a live tick the refresh also
	// actively re-probes blocked accounts so a silently-recovered one re-enters the pool
	// instead of riding a stale carried verdict; dry-run stays side-effect-free.
	mode := resume.ResolveWatchdogProbeMode(*probeMode, *live)
	if !*noRefresh && !*statusOnly {
		if msg := rwRefreshRegistry(regDir, claudeExe, *windowH, mode, *probeMinIntervalMin); msg != "" {
			note("  refresh: %s", msg)
		}
	}

	plan := rwLoadPlan(filepath.Join(regDir, "resume_plan.json"))
	tickMode := "DRY-RUN"
	if *live {
		tickMode = "LIVE"
	}
	ledgerPath := filepath.Join(regDir, "resume_ledger.jsonl")
	statusLedgerPath := rwWatchdogStatusLedger(regDir)
	statusEvents := rwLoadWatchdogStatusEvents(ledgerPath)
	statusEvents = append(statusEvents, rwLoadWatchdogStatusEvents(statusLedgerPath)...)
	if *statusOnly {
		rep := resume.FoldWatchdogStatus(resume.WatchdogStatusInput{
			Mode:           tickMode,
			NowUnix:        time.Now().Unix(),
			SilentSeconds:  int64(*silentHours * 3600),
			MonotonicTicks: *monotonicTicks,
			Plan:           plan,
			Events:         statusEvents,
		})
		if *asJSON {
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
	note("TICK %s plan=%d window=%gh cap=%d", tickMode, len(plan), *windowH, *maxPerTick)
	rwRecordWatchdogStatusTick(statusLedgerPath, tickMode, plan, statusEvents)

	// Defense-in-depth: the accounts policy still offers as workers. fleet_sessions.py
	// already excludes non-workers when it writes the plan, but a stale plan file could
	// predate the policy — re-check here too. An empty roster disables the check (fail
	// open, matching the Python's tolerance of a broken accounts read).
	guards := resume.WatchdogGuards{
		SelfSID:        selfSID,
		WorkerAccounts: rwWorkerAccounts(home),
		MaxAttempts:    *maxAttempts,
	}

	history := rwLoadHistory(ledgerPath)

	launched := 0
	for _, p := range plan {
		if launched >= *maxPerTick {
			note("  per-tick cap reached (%d)", *maxPerTick)
			break
		}
		sid8 := shortID(p.Session)
		// Outcome-aware once-gate input: how did the LAST attempt actually end, per the
		// transcript's own terminal turn (ground truth, never the launcher's ledger row)?
		hist := history[p.Session]
		outcome := resume.OutcomeUnknown
		if len(hist) > 0 {
			outcome = resume.ClassifyOutcome(rwTerminalSignal(rwTerminalText(rwNewestTranscript(home, p.Session))))
		}
		d := resume.DecideWatchdogRow(p, guards, hist, outcome)
		if d.Action != resume.WatchdogLaunch {
			note("  SKIP %s — %s", sid8, d.Reason)
			continue
		}
		acct := rwAccountTag(p.Account)
		resumeCfg := p.ResumeTarget()
		grant := launchSpawnBroker(rwResumeBrokerAttempt(claudeExe, p, resumeCfg))
		if !*live {
			if !grant.Allow {
				note("  WOULD DENY %s acct=%s proj=%s — spawn broker: %s agent_run=%s policy_digest=%s",
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
		// next tick. Fails open — a broken gate must never strand the whole watchdog.
		if admit, reason := rwSourceAdmit(ledgerPath, time.Now()); !admit {
			note("  DEFER %s acct=%s — per-source gate: %s", sid8, acct, reason)
			rwAppendLedger(ledgerPath, map[string]any{
				"ts": rwNowISO(), "session": p.Session, "account": p.Account,
				"resume_account": p.ResumeAccount,
				"phase":          "deferred", "cause": "source_concurrency_gate", "reason": reason,
			})
			continue
		}

		if !grant.Allow {
			note("  DENY %s acct=%s — spawn broker: %s agent_run=%s policy_digest=%s",
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
				note("  SKIP %s — re-home source transcript missing", sid8)
				continue
			}
			note("  RE-HOME %s %s -> %s (transcript copied; resuming on healthy account)",
				sid8, p.Account, p.ResumeAccount)
		}

		pid, err := rwSpawnResumeLaunch(claudeExe, p, resumeCfg, logDir, grant)
		if err != nil {
			note("  FAIL %s — spawn: %v", sid8, err)
			continue
		}
		// Record the launch BEFORE anything else — a crash cannot double-launch in this
		// tick. The gate keys on OUTCOME, not mere presence: phase="launched" marks an
		// attempt whose result is unknown until the next tick reads the transcript.
		attempt := d.Attempt
		row := map[string]any{
			"ts": rwNowISO(), "session": p.Session, "account": p.Account,
			"resume_account": p.ResumeAccount, "rehomed": p.Rehomed,
			"project": p.Project, "pid": pid, "cause": p.Disp,
			"phase": "launched", "attempt": attempt,
		}
		rwAppendLedger(ledgerPath, row)
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

	// 2. Alert on true login-blocked accounts — once per account blocker.
	rwAuthAlerts(regDir, logDir, note)

	note("  done: launched=%d sessions_in_ledger=%d", launched, len(history))
	return 0
}

// The pre-gate screens (self-resume guard, worker-account policy) and the probe-mode
// resolution live in the pure leaf: resume.DecideWatchdogRow / ResolveWatchdogProbeMode.

// rwTerminalSignal classifies a transcript's terminal-turn text into the closed
// TerminalSignal facts resume.ClassifyOutcome folds. One deliberate widening over the
// Python watchdog's ad-hoc "overloaded/529" check: the transient family is
// sessionsignals.IsAPIError — the same taxonomy every other tool in this family reads,
// so the watchdog can never disagree with the sweep about what is transient.
func rwTerminalSignal(text string) resume.TerminalSignal {
	if strings.TrimSpace(text) == "" {
		return resume.TerminalSignal{}
	}
	return resume.TerminalSignal{
		Found:             true,
		AuthWall:          sessionsignals.IsAuthError(text) || sessionsignals.NeedsLoginPrompt(text),
		LimitWall:         sessionsignals.LimitReset(text) != "",
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

// rwTerminalText is the text of the transcript's TERMINAL user/assistant record — the
// last real turn, ignoring trailing control/metadata records. Classification must read
// only this: a banner five turns back that a later clean turn superseded is NOT the
// session's current outcome.
func rwTerminalText(path string) string {
	if path == "" {
		return ""
	}
	copyRecs := loadSweepCopy(path).Records
	for i := len(copyRecs) - 1; i >= 0; i-- {
		if r := copyRecs[i]; r.Role == "user" || r.Role == "assistant" {
			return r.Text
		}
	}
	return ""
}

// rwLoadPlan reads the AUTO_RESUME plan fleet_sessions.py writes. Missing/malformed
// yields an empty plan (the tick logs plan=0 and does nothing), never a crash.
func rwLoadPlan(path string) []resume.WatchdogPlanRow {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil
	}
	var doc struct {
		Plan []resume.WatchdogPlanRow `json:"plan"`
	}
	if json.Unmarshal(raw, &doc) != nil {
		return nil
	}
	return doc.Plan
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
		out = append(out, resume.WatchdogStatusEvent{
			UnixSeconds:         parseTranscriptUnix(rec.TS),
			Session:             rec.Session,
			Phase:               rec.Phase,
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

func rwWatchdogStatusLedger(regDir string) string {
	return filepath.Join(regDir, "resume_watchdog_status.jsonl")
}

func renderResumeWatchdogStatus(w io.Writer, rep resume.WatchdogDrainStatus) {
	fmt.Fprintf(w, "resume watchdog status — %s mode=%s auto_resume_depth=%d silent_max=%s\n",
		strings.ToUpper(string(rep.Verdict)), rep.Mode, rep.AutoResumeDepth, humanIdle(rep.SilentSeconds))
	for _, r := range rep.Reasons {
		fmt.Fprintf(w, "  red: %s\n", r)
	}
	if len(rep.MTTRSessions) == 0 {
		fmt.Fprintln(w, "  no AUTO_RESUME rows or watchdog ledger sessions found")
		return
	}
	fmt.Fprintln(w)
	fmt.Fprintf(w, "%-10s %-19s %-8s %12s %12s %12s %10s  %s\n",
		"session", "status", "mode", "detected", "resumed", "progress", "silent", "evidence")
	for _, row := range rep.MTTRSessions {
		fmt.Fprintf(w, "%-10s %-19s %-8s %12s %12s %12s %10s  %s\n",
			shortID(row.Session), row.Status, row.Mode,
			watchdogUnix(row.DetectedAt), watchdogUnix(row.ResumedAt), watchdogUnix(row.ProgressWitnessedAt),
			humanIdle(row.SilentSeconds), row.Evidence)
	}
	fmt.Fprintln(w, "\n  recovered requires progress evidence after a launch; a launched ledger row alone stays launched_unproven.")
}

func watchdogUnix(unix int64) string {
	if unix <= 0 {
		return "-"
	}
	return time.Unix(unix, 0).UTC().Format("01-02 15:04")
}

// rwAppendLedger appends one JSONL row to the durable ledger. Best-effort: a failed
// append is reported by the caller's next read, never a tick crash.
func rwAppendLedger(path string, row any) {
	f, err := os.OpenFile(path, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644)
	if err != nil {
		return
	}
	defer f.Close()
	if b, err := json.Marshal(row); err == nil {
		_, _ = f.Write(append(b, '\n'))
	}
}

// rwSourceAdmit asks the per-source admission decision (the same fold `fak resume admit`
// exposes) whether this box may take one more live resume across ALL accounts. Fails
// OPEN on a policy-load error: a broken gate must never strand the whole watchdog — the
// per-tick cap and spacing remain as the fallback bound.
func rwSourceAdmit(ledgerPath string, now time.Time) (bool, string) {
	policies, err := resume.LoadSourcePolicy(defaultResumeSourcePolicy())
	if err != nil {
		return true, "gate-error: " + err.Error()
	}
	policy := policies.Default
	// A fresh/permissive policy file inherits the same CLI-default ceilings
	// `fak resume admit` applies, so the two entrances enforce one bound.
	if policy.MaxLiveResumes == 0 {
		policy.MaxLiveResumes = 4
	}
	if policy.MaxLaunchesPerWindow == 0 {
		policy.MaxLaunchesPerWindow = 10
	}
	if policy.WindowSeconds == 0 {
		policy.WindowSeconds = 300
	}
	if policy.MinLaunchSpacingSeconds == 0 {
		policy.MinLaunchSpacingSeconds = 8
	}
	d := resume.AdmitSource(foldSourceSnapshot(ledgerPath, now), policy, now)
	if d.Admit {
		return true, "admitted"
	}
	return false, d.Reason + ": " + d.Summary
}

// rwRefreshRegistry runs the fleet_sessions.py registry refresh (the one remaining
// Python child of this tick — porting fleet_sessions.py is the follow-on). FLEET_REG_DIR
// is pinned so the child writes where this tick reads; FLEET_CLAUDE_EXE is passed when it
// resolves so any probe spends its tiny turn on the same binary the resume uses. Returns
// a note string when the refresh was skipped or failed ("" on success).
func rwRefreshRegistry(regDir, claudeExe string, windowH float64, probeMode string, probeMinIntervalMin int) string {
	py := rwPythonExe()
	if py == "" {
		return "skipped (no python interpreter on PATH; acting on the existing plan file)"
	}
	cwd, _ := os.Getwd()
	script := filepath.Join(findRepoRoot(cwd), "tools", "fleet_sessions.py")
	if _, err := os.Stat(script); err != nil {
		return "skipped (tools/fleet_sessions.py not found; acting on the existing plan file)"
	}
	args := []string{script, "registry", "--window", strconv.FormatFloat(windowH, 'f', -1, 64)}
	if probeMode != "none" && probeMode != "" {
		args = append(args, "--probe", probeMode, "--min-interval-min", strconv.Itoa(probeMinIntervalMin))
	}
	cmd := exec.Command(py, args...)
	cmd.Env = append(os.Environ(), "FLEET_REG_DIR="+regDir)
	if claudeExe != "" {
		if _, err := os.Stat(claudeExe); err == nil {
			cmd.Env = append(cmd.Env, "FLEET_CLAUDE_EXE="+claudeExe)
		}
	}
	windowgate.ConfigureBackgroundCommand(cmd)
	if err := cmd.Run(); err != nil {
		return "refresh child failed (" + err.Error() + "); acting on the existing plan file"
	}
	return ""
}

// rwSpawnResume launches the detached `claude --resume` under the target account's
// CLAUDE_CONFIG_DIR, stdout/stderr teed to per-session files, with a hidden console on
// Windows (the CREATE_NO_WINDOW discipline every fak background spawn takes).
var rwSpawnResumeLaunch = rwSpawnResume

func rwResumeBrokerAttempt(claudeExe string, p resume.WatchdogPlanRow, resumeCfg string) launchBrokerAttempt {
	return newLaunchBrokerAttempt("resume_watchdog", "claude", rwResumeArgv(claudeExe, p.Session),
		envMap(resume.WatchdogChildEnv(os.Environ(), resumeCfg)), rwResumeCWD(p))
}

func rwResumeArgv(claudeExe, session string) []string {
	return []string{claudeExe, "--resume", session, "-p", resumeWatchdogPrompt, "--dangerously-skip-permissions"}
}

func rwResumeCWD(p resume.WatchdogPlanRow) string {
	if p.CWD != "" && rwIsDir(p.CWD) {
		return p.CWD
	}
	cwd, _ := os.Getwd()
	return findRepoRoot(cwd)
}

func rwSpawnResume(claudeExe string, p resume.WatchdogPlanRow, resumeCfg, logDir string, grant launchBrokerGrant) (int, error) {
	if claudeExe == "" {
		return 0, fmt.Errorf("no claude binary (set FLEET_CLAUDE_EXE)")
	}
	wd := firstString(grant.CWD, rwResumeCWD(p))
	outPath := filepath.Join(logDir, fmt.Sprintf("resume-%s-%d.log", shortID(p.Session), time.Now().Unix()))
	stdout, err := os.OpenFile(outPath, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644)
	if err != nil {
		return 0, err
	}
	stderr, err := os.OpenFile(outPath+".err", os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644)
	if err != nil {
		stdout.Close()
		return 0, err
	}
	defer stdout.Close()
	defer stderr.Close()

	argv := rwResumeArgv(claudeExe, p.Session)
	if len(grant.Argv) > 0 {
		argv = grant.Argv
	}
	cmd := exec.Command(argv[0], argv[1:]...)
	cmd.Dir = wd
	// The child env drops the parent's guard-gateway/model-API wiring and harness
	// identity, and pins CLAUDE_CONFIG_DIR to the target seat (resume.WatchdogChildEnv —
	// the 2026-07-01 whole-wave-crash fix: a resumed child inheriting a guarded parent's
	// ANTHROPIC_BASE_URL routes through the parent's loopback proxy and dies with it).
	cmd.Env = resume.WatchdogChildEnv(os.Environ(), resumeCfg)
	if len(grant.Env) > 0 {
		cmd.Env = envSliceFromMap(grant.Env)
	}
	cmd.Stdout = stdout
	cmd.Stderr = stderr
	windowgate.ConfigureBackgroundCommand(cmd)
	if err := cmd.Start(); err != nil {
		return 0, err
	}
	pid := cmd.Process.Pid
	// Detach: the tick never waits on the resumed session.
	_ = cmd.Process.Release()
	return pid, nil
}

// rwWorkerAccounts is the set of account dir-basenames policy still offers as workers.
// Empty on any discovery problem (which disables the defense-in-depth check, fail-open).
func rwWorkerAccounts(home string) map[string]bool {
	cwd, _ := os.Getwd()
	paths := fleetaccounts.ResolvePaths(filepath.Join(findRepoRoot(cwd), "tools"))
	if home == "" {
		home = paths.Home
	}
	pol := fleetaccounts.LoadPolicy(paths)
	out := map[string]bool{}
	for _, a := range fleetaccounts.Discover(home, paths.ConfigHome, pol) {
		if a.Kind == fleetaccounts.KindWorker {
			out[a.Account] = true
		}
	}
	return out
}

// rwAuthAlerts surfaces accounts whose sessions are stopped behind a human-fixable auth
// wall — once per (account, reason) blocker, tracked in _notified.json.
func rwAuthAlerts(regDir, logDir string, note func(string, ...any)) {
	raw, err := os.ReadFile(filepath.Join(regDir, "sessions.json"))
	if err != nil {
		return
	}
	var reg struct {
		Accounts []struct {
			Account             string `json:"account"`
			Tag                 string `json:"tag"`
			Blocked             bool   `json:"blocked"`
			Throttled           bool   `json:"throttled"`
			BlockKind           string `json:"block_kind"`
			BlockReason         string `json:"block_reason"`
			AuthBlockedSessions int    `json:"auth_blocked_sessions"`
		} `json:"accounts"`
	}
	if json.Unmarshal(raw, &reg) != nil {
		return
	}
	notifiedPath := filepath.Join(regDir, "_notified.json")
	notified := map[string]bool{}
	if b, err := os.ReadFile(notifiedPath); err == nil {
		_ = json.Unmarshal(b, &notified)
	}
	changed := false
	for _, a := range reg.Accounts {
		if !a.Blocked || a.Throttled || a.BlockKind != "auth" {
			continue
		}
		key := fmt.Sprintf("auth-account:%s:%s", a.Account, a.BlockReason)
		if notified[key] {
			continue
		}
		acct := a.Tag
		if acct == "" {
			acct = rwAccountTag(a.Account)
		}
		reason := a.BlockReason
		if reason == "" {
			reason = "auth/login required"
		}
		sessions := ""
		if a.AuthBlockedSessions > 0 {
			sessions = fmt.Sprintf(" / %d stopped session(s)", a.AuthBlockedSessions)
		}
		rwToast(logDir, "Account needs re-login", fmt.Sprintf("%s : %s%s", acct, reason, sessions), "warn")
		note("  ALERT auth-blocked acct=%s reason=%s (notified)", acct, reason)
		notified[key] = true
		changed = true
	}
	if changed {
		if b, err := json.Marshal(notified); err == nil {
			_ = os.WriteFile(notifiedPath, b, 0o644)
		}
	}
}

// rwNote appends one timestamped line to resume_watchdog.log and echoes it to the tick's
// stdout — the watchdog's single narration seam.
func rwNote(logDir string, w io.Writer, msg string) {
	line := rwNowISO() + "  " + msg
	if f, err := os.OpenFile(filepath.Join(logDir, "resume_watchdog.log"),
		os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644); err == nil {
		_, _ = f.WriteString(line + "\n")
		f.Close()
	}
	fmt.Fprintln(w, line)
}

// rwToast records an operator notification durably (notifications.log) and raises the
// macOS Notification Center toast when osascript exists. Best-effort everywhere: a
// notification failure must never kill a tick.
func rwToast(logDir, title, message, level string) {
	if f, err := os.OpenFile(filepath.Join(logDir, "notifications.log"),
		os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644); err == nil {
		fmt.Fprintf(f, "%s  [%s] %s — %s\n", rwNowISO(), level, title, message)
		f.Close()
	}
	if runtime.GOOS == "darwin" {
		if osa, err := exec.LookPath("osascript"); err == nil {
			script := fmt.Sprintf("display notification %q with title %q", message, title)
			cmd := exec.Command(osa, "-e", script)
			windowgate.ConfigureBackgroundCommand(cmd)
			_ = cmd.Run()
		}
	}
}

// rwClaudeExe resolves the claude binary from the fleet-wide convention: FLEET_CLAUDE_EXE,
// the FAK_CLAUDE_EXE back-compat fallback, PATH, then the conventional install path.
func rwClaudeExe(home string) string {
	if v := strings.TrimSpace(os.Getenv("FLEET_CLAUDE_EXE")); v != "" {
		return v
	}
	if v := strings.TrimSpace(os.Getenv("FAK_CLAUDE_EXE")); v != "" {
		return v
	}
	for _, name := range []string{"claude", "claude.exe"} {
		if p, err := exec.LookPath(name); err == nil {
			return p
		}
	}
	return filepath.Join(home, ".local", "bin", "claude")
}

// rwPythonExe resolves a Python interpreter for the registry-refresh child, or "".
func rwPythonExe() string {
	for _, name := range []string{"python", "python3", "py"} {
		if p, err := exec.LookPath(name); err == nil {
			return p
		}
	}
	return ""
}

// rwAccountTag is the human account tag of a config-dir basename (".claude-gem7" → "gem7").
func rwAccountTag(account string) string {
	tag := strings.TrimPrefix(strings.TrimPrefix(account, ".claude-"), ".claude")
	if tag == "" {
		return "default"
	}
	return tag
}

func rwIsDir(p string) bool {
	fi, err := os.Stat(p)
	return err == nil && fi.IsDir()
}

func rwNowISO() string { return time.Now().UTC().Format("2006-01-02T15:04:05Z") }

func rwEnvBool(name string) bool {
	switch strings.ToLower(strings.TrimSpace(os.Getenv(name))) {
	case "1", "true", "yes", "on":
		return true
	}
	return false
}

func rwEnvStr(name, def string) string {
	if v := strings.TrimSpace(os.Getenv(name)); v != "" {
		return v
	}
	return def
}

func rwEnvFloat(name string, def float64) float64 {
	if v := strings.TrimSpace(os.Getenv(name)); v != "" {
		if f, err := strconv.ParseFloat(v, 64); err == nil {
			return f
		}
	}
	return def
}

func rwEnvInt(name string, def int) int {
	if v := strings.TrimSpace(os.Getenv(name)); v != "" {
		if n, err := strconv.Atoi(v); err == nil {
			return n
		}
	}
	return def
}
