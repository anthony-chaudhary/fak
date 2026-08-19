package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/codexresume"
	"github.com/anthony-chaudhary/fak/internal/fleetaccounts"
	"github.com/anthony-chaudhary/fak/internal/jsonlledger"
	"github.com/anthony-chaudhary/fak/internal/resume"
	"github.com/anthony-chaudhary/fak/internal/resumeactuator"
	"github.com/anthony-chaudhary/fak/internal/trajctl"
	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

type rwCodexCompletion struct {
	Session string
	Result  codexresume.Result
}

func rwLoadCodexCompletions(plan []resume.WatchdogPlanRow) []rwCodexCompletion {
	var out []rwCodexCompletion
	for _, p := range plan {
		if rwHarness(p) != "codex" || strings.TrimSpace(p.ResultFile) == "" {
			continue
		}
		raw, err := os.ReadFile(p.ResultFile)
		if err != nil {
			continue
		}
		var result codexresume.Result
		if json.Unmarshal(raw, &result) != nil {
			continue
		}
		out = append(out, rwCodexCompletion{Session: p.Session, Result: result})
	}
	return out
}

func rwWatchdogStatusLedger(regDir string) string {
	return filepath.Join(regDir, "resume_watchdog_status.jsonl")
}

func rwEnvInt64(name string, fallback int64) int64 {
	if v, err := strconv.ParseInt(strings.TrimSpace(os.Getenv(name)), 10, 64); err == nil && v >= 0 {
		return v
	}
	return fallback
}

func rwRotateFile(path string, maxBytes int64) {
	if maxBytes <= 0 {
		return
	}
	if st, err := os.Stat(path); err == nil && st.Size() >= maxBytes {
		_ = os.Remove(path + ".1")
		_ = os.Rename(path, path+".1")
	}
}

func rwPruneResumeLogs(logDir string, retainDays float64, now time.Time) int {
	if retainDays < 0 {
		return 0
	}
	cutoff := now.Add(-time.Duration(retainDays * float64(24*time.Hour)))
	entries, _ := os.ReadDir(logDir)
	pruned := 0
	for _, entry := range entries {
		if entry.IsDir() || !strings.HasPrefix(entry.Name(), "resume-") || (!strings.HasSuffix(entry.Name(), ".log") && !strings.HasSuffix(entry.Name(), ".log.err")) {
			continue
		}
		if info, err := entry.Info(); err == nil && info.ModTime().Before(cutoff) {
			if os.Remove(filepath.Join(logDir, entry.Name())) == nil {
				pruned++
			}
		}
	}
	return pruned
}

func rwBoundWatchdogArtifacts(logDir, ledgerPath string, now time.Time) {
	maxBytes := rwEnvInt64("FAK_WATCHDOG_LOG_MAX_BYTES", 5*1024*1024)
	rwRotateFile(filepath.Join(logDir, "resume_watchdog.log"), maxBytes)
	rwRotateFile(filepath.Join(logDir, "notifications.log"), maxBytes)
	_ = rwPruneResumeLogs(logDir, rwEnvFloat("FAK_RESUME_LOG_RETAIN_DAYS", 14), now)
	compactBytes := rwEnvInt64("FAK_RESUME_LEDGER_COMPACT_BYTES", 512*1024)
	_ = rwCompactResumeLedger(ledgerPath, rwEnvFloat("FAK_RESUME_LEDGER_RETAIN_DAYS", 30), compactBytes, now)
	rwRotateFile(ledgerPath, compactBytes)
}

func rwCompactResumeLedger(path string, retainDays float64, compactBytes int64, now time.Time) int {
	if retainDays <= 0 || compactBytes <= 0 {
		return 0
	}
	if st, err := os.Stat(path); err != nil || st.Size() < compactBytes {
		return 0
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return 0
	}
	cutoff := now.Add(-time.Duration(retainDays * float64(24*time.Hour)))
	kept := make([][]byte, 0)
	dropped := 0
	for _, line := range bytes.Split(data, []byte{'\n'}) {
		if len(bytes.TrimSpace(line)) == 0 {
			continue
		}
		var row map[string]any
		if json.Unmarshal(line, &row) != nil {
			kept = append(kept, line)
			continue
		}
		action, _ := row["action"].(string)
		manual, _ := row["manual_override"].(bool)
		if strings.HasPrefix(action, "consolidate") || manual {
			kept = append(kept, line)
			continue
		}
		tsText, _ := row["ts"].(string)
		ts, err := time.Parse("2006-01-02T15:04:05Z", tsText)
		if err != nil || !ts.Before(cutoff) {
			kept = append(kept, line)
		} else {
			dropped++
		}
	}
	if dropped == 0 {
		return 0
	}
	out := bytes.Join(kept, []byte{'\n'})
	out = append(out, '\n')
	tmp := path + ".compact.tmp"
	if os.WriteFile(tmp, out, 0o644) != nil {
		return 0
	}
	if os.Rename(tmp, path) != nil {
		_ = os.Remove(tmp)
		return 0
	}
	return dropped
}

func renderResumeWatchdogStatus(w io.Writer, rep resume.WatchdogDrainStatus) {
	fmt.Fprintf(w, "resume watchdog status — %s mode=%s auto_resume_depth=%d silent_max=%s unproven_max=%s\n",
		strings.ToUpper(string(rep.Verdict)), rep.Mode, rep.AutoResumeDepth,
		humanIdle(rep.SilentSeconds), humanIdle(rep.UnprovenSeconds))
	for _, r := range rep.Reasons {
		fmt.Fprintf(w, "  red: %s\n", r)
	}
	if rep.Page != nil {
		// The dedup key an operator can grep for in _paged.json / the filed issue (#3582).
		fmt.Fprintf(w, "  page: %s (signature %s)\n", rep.Page.Reason, rep.Page.Signature)
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
func rwSourceAdmit(ledgerPath, regDir string, now time.Time) (bool, string) {
	policies, err := resume.LoadSourcePolicy(defaultResumeSourcePolicy())
	if err != nil {
		return true, "gate-error: " + err.Error()
	}
	policy := policies.Default
	// Resolve the host-wide live-resume ceiling from the highest-precedence rail
	// present — FAK_RESUME_MAX_LIVE env/param → explicit policy value → healthy-seat
	// derivation → the static default 4 — instead of the old hard-coded `= 4`. A fresh
	// host with more healthy seats now scales past 4 rather than turning eligible
	// sessions away (#5093); the same resolver + provenance backs `fak resume admit`.
	envVal, envPresent := resumeMaxLiveEnv()
	resolved := resume.ResolveMaxLiveResumes(resume.MaxLiveResumesInput{
		EnvPresent:  envPresent,
		EnvValue:    envVal,
		ConfigValue: policy.MaxLiveResumes,
		Seats:       rwLoadHeadroomSeats(regDir),
		Floor:       resume.DefaultMaxLiveResumes,
		Ceiling:     resumeLiveDeriveCeiling,
		SeatCap:     resumeLiveDerivePerSeat,
	})
	policy.MaxLiveResumes = resolved.Value
	// The other three axes keep their historical CLI-default fallbacks so the two
	// entrances (`fak resume admit` and this tick) still enforce one bound.
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
		return true, fmt.Sprintf("admitted (live ceiling %d via %s)", resolved.Value, resolved.Source)
	}
	return false, d.Reason + ": " + d.Summary
}

// resumeLiveDeriveCeiling / resumeLiveDerivePerSeat bound the healthy-seat
// derivation of the host-wide live ceiling: at most one live `claude --resume` per
// healthy seat, capped at 16 so a large pool cannot re-open the per-source 529 burst
// wall the ceiling exists to hold. The static floor stays resume.DefaultMaxLiveResumes.
const (
	resumeLiveDeriveCeiling = 16
	resumeLiveDerivePerSeat = 1
)

// resumeMaxLiveEnv reads the FAK_RESUME_MAX_LIVE override. present is false when the
// var is unset, blank, or unparseable — an unparseable value fails safe to the next
// rail (policy/derived/default) rather than silently disabling the ceiling (value 0).
func resumeMaxLiveEnv() (value int, present bool) {
	raw, ok := os.LookupEnv("FAK_RESUME_MAX_LIVE")
	if !ok {
		return 0, false
	}
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return 0, false
	}
	n, err := strconv.Atoi(raw)
	if err != nil {
		return 0, false
	}
	return n, true
}

// rwLoadHeadroomSeats reads the fleet sessions.json seat census (accounts[]) for the
// live-ceiling derivation, the same registry `fak resume cap` folds for the per-tick
// cap. A missing/unreadable/blank registry yields nil (no derivation) rather than an
// error, so the resolver falls through to env/policy/default.
func rwLoadHeadroomSeats(regDir string) []resume.HeadroomSeat {
	if regDir == "" {
		return nil
	}
	b, err := os.ReadFile(filepath.Join(regDir, "sessions.json"))
	if err != nil {
		return nil
	}
	var reg struct {
		Accounts []resume.HeadroomSeat `json:"accounts"`
	}
	if json.Unmarshal(b, &reg) != nil {
		return nil
	}
	return reg.Accounts
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

func rwHarness(p resume.WatchdogPlanRow) string {
	switch strings.ToLower(strings.TrimSpace(p.Harness)) {
	case "codex", "codex_exec":
		return "codex"
	default:
		return "claude"
	}
}

func rwActuatorRequest(p resume.WatchdogPlanRow, claudeExe string) resumeactuator.Request {
	prompt := resumeWatchdogPrompt
	if block := rwResumeAnchor(p.Session).Prompt(); block != "" {
		prompt = block + "\n\n" + prompt
	}
	return resumeactuator.Request{
		Harness: p.Harness, Session: p.Session, Rollout: p.Rollout,
		GoalFile: p.GoalFile, ResultFile: p.ResultFile, CWD: rwResumeCWD(p), Prompt: prompt, ClaudeExe: claudeExe, CodexExe: rwCodexExe(),
	}
}

func rwManagedResumeArgv(fakExe, claudeExe string, p resume.WatchdogPlanRow, postureArgs []string, carry ...resume.DriveCarryRow) ([]string, error) {
	var budget []string
	if len(carry) > 0 {
		if spec := rwDriveCarryEnvelope(carry[0]); spec != "" {
			budget = []string{"--budget-envelope", spec}
		}
	}
	return rwActuatorRequest(p, claudeExe).ManagedArgv(fakExe, postureArgs, budget)
}

func rwCodexResumeArgv(fakExe string, p resume.WatchdogPlanRow) []string {
	argv, _ := rwActuatorRequest(p, "").ContinuationArgv(fakExe)
	return argv
}

func validateCodexResumeCoordinates(p resume.WatchdogPlanRow) error {
	_, err := rwActuatorRequest(p, "").ContinuationArgv("fak")
	return err
}

func rwCodexExe() string {
	if exe := strings.TrimSpace(os.Getenv("FLEET_CODEX_EXE")); exe != "" {
		return exe
	}
	return "codex"
}

func rwResumeBrokerAttempt(fakExe, claudeExe string, p resume.WatchdogPlanRow, resumeCfg string, postureArgs []string, carry ...resume.DriveCarryRow) launchBrokerAttempt {
	argv, err := rwManagedResumeArgv(fakExe, claudeExe, p, postureArgs, carry...)
	if err != nil {
		argv = nil
	}
	return newLaunchBrokerAttempt("resume_watchdog", rwHarness(p), argv,
		rwChildEnvironment(p, resumeCfg), rwResumeCWD(p))
}

func rwChildEnvironment(p resume.WatchdogPlanRow, resumeCfg string) map[string]string {
	if rwHarness(p) == resumeactuator.HarnessCodex {
		return envMap(resume.CodexWatchdogChildEnv(os.Environ()))
	}
	return rwResumeChildEnv(p.Session, resumeCfg)
}

func rwResumeChildEnv(session, resumeCfg string) map[string]string {
	env := envMap(resume.WatchdogChildEnv(os.Environ(), resumeCfg))
	if key := resume.RelaunchCacheAffinityKey(session); key != "" {
		env[resume.RelaunchCacheAffinityEnv] = key
	}
	return env
}

var rwResumeAnchor = func(session string) resume.ResumeAnchor {
	path := filepath.Join(repoRoot(), trajctl.DefaultLedgerRel)
	return resume.BuildResumeAnchor(session, trajctl.Fold(trajctl.ReadLedgerFile(path)))
}

// rwResumeArgv retains the legacy Claude helper shape for compatibility callers.
// The watchdog broker uses rwManagedResumeArgv, which always fronts launches with fak m.
func rwResumeArgv(fakExe, claudeExe, session string, postureArgs []string, carry ...resume.DriveCarryRow) []string {
	prompt := resumeWatchdogPrompt
	anchor := rwResumeAnchor(session)
	if block := anchor.Prompt(); block != "" {
		prompt = block + "\n\n" + prompt
	}
	child := []string{claudeExe, "--resume", session, "-p", prompt, "--dangerously-skip-permissions"}
	if len(postureArgs) > 0 && strings.TrimSpace(fakExe) != "" {
		front := make([]string, 0, len(postureArgs)+len(child)+3)
		front = append(front, fakExe, "guard")
		front = append(front, postureArgs...)
		if len(carry) > 0 {
			if spec := rwDriveCarryEnvelope(carry[0]); spec != "" {
				front = append(front, "--budget-envelope", spec)
			}
		}
		front = append(front, "--")
		return append(front, child...)
	}
	return child
}

func rwResumeCWD(p resume.WatchdogPlanRow) string {
	if p.CWD != "" && rwIsDir(p.CWD) {
		return p.CWD
	}
	cwd, _ := os.Getwd()
	return findRepoRoot(cwd)
}

func rwSpawnResume(claudeExe string, p resume.WatchdogPlanRow, resumeCfg, logDir string, grant launchBrokerGrant) (int, error) {
	harness, err := rwActuatorRequest(p, claudeExe).HarnessName()
	if err != nil {
		return 0, err
	}
	if harness == resumeactuator.HarnessCodex {
		if err := validateCodexResumeCoordinates(p); err != nil {
			return 0, err
		}
	} else if claudeExe == "" {
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

	// grant.Argv is authoritative and carries any `fak guard <posture> --` front the broker
	// attempt built (rwResumeBrokerAttempt); this bare form is only a defensive fallback for an
	// (unreachable on the live path) empty grant, so it deliberately stays posture-free.
	argv := rwResumeArgv("", claudeExe, p.Session, nil)
	if rwHarness(p) == "codex" {
		argv = rwCodexResumeArgv("fak", p)
	}
	if len(grant.Argv) > 0 {
		argv = grant.Argv
	}
	argv, promptStdin, _ := guardPromptStdinTransport(argv)
	cmd := exec.Command(argv[0], argv[1:]...)
	if promptStdin != "" {
		cmd.Stdin = strings.NewReader(promptStdin)
	}
	cmd.Dir = wd
	// The child env drops the parent's guard-gateway/model-API wiring and harness
	// identity, and pins CLAUDE_CONFIG_DIR to the target seat (resume.WatchdogChildEnv —
	// the 2026-07-01 whole-wave-crash fix: a resumed child inheriting a guarded parent's
	// ANTHROPIC_BASE_URL routes through the parent's loopback proxy and dies with it).
	// Same base + cache-affinity carry as the broker attempt (#4140), so even the
	// defensive empty-grant fallback keeps the transcript's warm route.
	cmd.Env = envSliceFromMap(rwResumeChildEnv(p.Session, resumeCfg))
	if rwHarness(p) == "codex" {
		cmd.Env = resume.CodexWatchdogChildEnv(os.Environ())
	}
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

// rwDriveStateLedger is the durable, UUID-keyed operator drive-state store the watchdog reads
// to honor an operator's pause/drain/stop of a specific session. It lives in the SAME regDir
// the tick already resolves, so `fak resume hold` / `fak resume release` (the writer) and this
// reader always agree on the file. It is deliberately NOT the descriptor registry
// (session-registry.json): that store is keyed by guardTraceID, a disjoint keyspace from the
// Claude transcript UUID the plan row carries, AND it is TTL-GC'd — a Stopped descriptor would
// evaporate 30 min after the last stamp, so a durable Stop veto must live in an un-swept file.
func rwDriveCarryLedger(regDir string) string {
	return filepath.Join(regDir, "resume_drivecarry.jsonl")
}

func rwLoadDriveCarry(regDir string) map[string]resume.DriveCarryRow {
	raw, err := os.ReadFile(rwDriveCarryLedger(regDir))
	if err != nil {
		return nil
	}
	rows := jsonlledger.Parse[resume.DriveCarryRow](string(raw), nil)
	if len(rows) == 0 {
		return nil
	}
	return resume.FoldDriveCarryRows(rows)
}

func rwDriveCarryEnvelope(c resume.DriveCarryRow) string {
	var fields []string
	axis := func(name string, n int64) {
		if n > 0 {
			fields = append(fields, name+"="+strconv.FormatInt(n, 10))
		} else if n < 0 {
			fields = append(fields, name+"=unbounded")
		}
	}
	axis("turns", c.TurnsLeft)
	axis("tokens", c.TokensLeft)
	axis("context", c.ContextTokensLeft)
	if c.TimeLeftNanos > 0 {
		fields = append(fields, "wall="+time.Duration(c.TimeLeftNanos).String())
	} else if c.TimeLeftNanos < 0 {
		fields = append(fields, "wall=unbounded")
	}
	if c.SpendMicroCentsLeft > 0 {
		fields = append(fields, "spend="+strconv.FormatFloat(float64(c.SpendMicroCentsLeft)/100000000, 'f', 8, 64))
	} else if c.SpendMicroCentsLeft < 0 {
		fields = append(fields, "spend=unbounded")
	}
	if c.PaceMaxTokensPerTurn > 0 {
		fields = append(fields, "max-tokens="+strconv.Itoa(c.PaceMaxTokensPerTurn))
	}
	if c.PaceMinTurnGapMs > 0 {
		fields = append(fields, "gap="+(time.Duration(c.PaceMinTurnGapMs)*time.Millisecond).String())
	}
	return strings.Join(fields, ",")
}

func rwDriveStateLedger(regDir string) string {
	return filepath.Join(regDir, "resume_drivestate.jsonl")
}

// rwLoadDriveStates reads the append-only drive-state store and folds it — through the pure
// leaf resume.FoldDriveStates — into the one current operator drive-state per session id (the
// Claude transcript UUID the plan row carries). The fold is last-row-wins per session with one
// exception: a Stop is STICKY (terminal), so a later paused/running row can never revive a
// session the operator deliberately stopped, while a `running` release DOES lift a prior
// paused/draining hold. There is no TTL — a hold survives the descriptor registry's 30-minute
// GC. A missing / unreadable / empty file yields a nil map, which leaves the operator-hold
// guard INERT (fail-open, matching rwWorkerAccounts) — an absent store never strands a
// legitimately-crashed session.
func rwLoadDriveStates(regDir string) map[string]resume.WatchdogDriveState {
	raw, err := os.ReadFile(rwDriveStateLedger(regDir))
	if err != nil {
		return nil
	}
	states := resume.FoldDriveStates(jsonlledger.Parse[resume.DriveStateRow](string(raw), nil))
	if len(states) == 0 {
		return nil
	}
	return states
}

// rwRelaunchResetLedger is the durable, transcript-UUID-keyed store of OS-relaunch reset rows
// (#4139/#4216) — the resume-keyspace analogue of session's ResetTransactionLog (#1582). It
// lives in the SAME regDir the tick resolves, next to resume_drivestate.jsonl, so the launch-site
// writer (rwSpawnResumeLaunch's "launched" block) and this reader always agree on the file. It is
// append-only and folded last-write-per-session, exactly like the drivestate store it mirrors.
func rwRelaunchResetLedger(regDir string) string {
	return filepath.Join(regDir, "resume_relaunch_reset.jsonl")
}

// rwLoadRelaunchResets reads the append-only relaunch-reset store and folds it — through the pure
// leaf resume.FoldRelaunchResets — into the latest reset per session id (the Claude transcript
// UUID the plan row carries). A missing / unreadable / empty store yields a nil map (fail-open),
// mirroring rwLoadDriveStates: an absent store never strands a legitimately-crashed session.
func rwLoadRelaunchResets(regDir string) map[string]resume.RelaunchResetRow {
	raw, err := os.ReadFile(rwRelaunchResetLedger(regDir))
	if err != nil {
		return nil
	}
	resets := resume.FoldRelaunchResets(jsonlledger.Parse[resume.RelaunchResetRow](string(raw), nil))
	if len(resets) == 0 {
		return nil
	}
	return resets
}

// rwRelaunchAffinityLedger is the durable, transcript-UUID-keyed store of relaunch
// cache-affinity rows (#4140) — the cache-route sibling of the relaunch-reset store
// above. It lives in the SAME regDir the tick resolves, so the launch-site writer (the
// "launched" block in runResumeWatchdog) and this reader always agree on the file. It is
// append-only and folded last-write-per-session, exactly like the stores it mirrors.
func rwRelaunchAffinityLedger(regDir string) string {
	return filepath.Join(regDir, "resume_relaunch_affinity.jsonl")
}

// rwLoadRelaunchAffinity reads the append-only relaunch cache-affinity store and folds
// it — through the pure leaf resume.FoldRelaunchAffinity — into the latest cache route
// per session id (the Claude transcript UUID the plan row carries). A missing /
// unreadable / empty store yields a nil map (fail-open), mirroring rwLoadRelaunchResets:
// affinity is a warmth bias, and its absence must never strand a resume.
func rwLoadRelaunchAffinity(regDir string) map[string]string {
	raw, err := os.ReadFile(rwRelaunchAffinityLedger(regDir))
	if err != nil {
		return nil
	}
	routes := resume.FoldRelaunchAffinity(jsonlledger.Parse[resume.RelaunchAffinityRow](string(raw), nil))
	if len(routes) == 0 {
		return nil
	}
	return routes
}

// rwLoadIdentity reads the append-only identity store and folds it — through the pure leaf
// resume.FoldIdentity — into the traceByUUID direction the watchdog needs: a plan row is keyed
// by the transcript UUID, and this join is the only bridge from that UUID to the gateway TRACE
// the rich drive State (internal/session/table.go) is keyed on. Mirrors rwLoadDriveStates: a
// missing / unreadable / empty store yields a nil map, leaving any UUID->trace join INERT
// (fail-open) rather than stranding a legitimately-crashed session. The reverse direction is
// dropped here — the watchdog only ever holds a UUID and wants its trace.
func rwLoadIdentity(regDir string) map[string]string {
	traceByUUID, _ := resume.LoadIdentity(regDir)
	if len(traceByUUID) == 0 {
		return nil
	}
	return traceByUUID
}

// rwRefreshResumeIdentity best-effort refreshes the durable uuid<->trace identity row
// (resume_identity.jsonl, the A1 store) for a session the watchdog just resumed, so the
// join's newest row for that UUID names the account the resume re-homed onto. It is the
// resume-path twin of the guard-SessionStart producer: a watchdog-resumed child can NEVER
// self-record the join, because WatchdogChildEnv strips CLAUDE_CODE_SESSION_ID from its env
// (the mass-crash fix, WatchdogChildEnvDrop), so the UUID the SessionStart hook keys on is
// blank in the child. The watchdog holds that UUID as p.Session, so it records it here.
//
// The trace endpoint is carried forward from the existing store (via rwLoadIdentity, the A4
// forward reader): a re-home changes only the ACCOUNT, not the uuid<->trace pairing, and both
// FoldIdentity and ResolveIdentity skip a row missing either endpoint — so re-stamping the
// known trace keeps the refreshed row a full join whose account is now authoritative. A UUID
// with no prior join still records its account (a half row the fold ignores, but the audit
// read and any later full row still see it). Fail-open like every other watchdog write: a
// blank UUID or an append error is a silent no-op, never a strand.
func rwRefreshResumeIdentity(regDir string, p resume.WatchdogPlanRow) {
	uuid := strings.TrimSpace(p.Session)
	if uuid == "" {
		return
	}
	account := strings.TrimSpace(p.ResumeAccount)
	if account == "" {
		account = strings.TrimSpace(p.Account)
	}
	_ = appendJSONL(resume.IdentityLedgerPath(regDir), resume.IdentityRow{
		TS:      rwNowISO(),
		UUID:    uuid,
		Trace:   rwLoadIdentity(regDir)[uuid], // carry the known trace; "" = a half row the fold skips
		Account: account,
		Via:     "resume-watchdog",
	})
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
