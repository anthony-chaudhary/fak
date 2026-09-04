package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"text/tabwriter"
	"time"

	"github.com/anthony-chaudhary/fak/internal/accounts"
	"github.com/anthony-chaudhary/fak/internal/guardsessions"
	"github.com/anthony-chaudhary/fak/internal/hostresurrect"
	"github.com/anthony-chaudhary/fak/internal/resume"
	resumesweep "github.com/anthony-chaudhary/fak/internal/resume/sweep"
	"github.com/anthony-chaudhary/fak/internal/sessiondiag"
	"github.com/anthony-chaudhary/fak/internal/sessionjournal"
	"github.com/anthony-chaudhary/fak/internal/sessionrecovery"
	"github.com/anthony-chaudhary/fak/internal/sessionregistry"
)

type threadFlags map[string]bool

func (v threadFlags) String() string { return "" }
func (v threadFlags) Set(s string) error {
	s = strings.TrimSpace(s)
	if s == "" {
		return fmt.Errorf("empty thread")
	}
	v[s] = true
	return nil
}

var recoveryInventorySources = sessiondiag.ReadCodexInventorySources
var recoveryInventoryRegistryPath = sessionregistry.DefaultPath

var recoveryInventory = collectRecoveryInventory
var recoveryProviderCommand = exec.Command

func collectRecoveryInventory(since time.Duration) (sessionrecovery.InventoryReport, error) {
	now := recoveryNow()
	input, err := recoveryInventorySources("", since, now)
	if err != nil {
		return sessionrecovery.InventoryReport{}, fmt.Errorf("sessiondiag inventory: %w", err)
	}
	rows, registryErr := (sessionregistry.Store{Path: recoveryInventoryRegistryPath()}).ReadAll()
	if registryErr == nil {
		input.Registrations = rows
	} else if !errors.Is(registryErr, os.ErrNotExist) {
		input.SourceErrors = append(input.SourceErrors, sessiondiag.SourceError{Source: "child_registrations", Code: "READ_FAILED"})
	}
	input.Window = since
	input.StaleAfter = 10 * time.Minute
	raw, err := json.Marshal(sessiondiag.ReconcileInventory(input, now))
	if err != nil {
		return sessionrecovery.InventoryReport{}, fmt.Errorf("encode sessiondiag inventory: %w", err)
	}
	var report sessionrecovery.InventoryReport
	if err := json.Unmarshal(raw, &report); err != nil {
		return report, fmt.Errorf("decode sessiondiag inventory: %w", err)
	}
	report = mergeClaudeRecoveryInventory(report, resolveFleetUserHome("", ""), since, now, liveResumeSIDs())
	regDir := resolveSweepRegDir("")
	cohort, cohortErr := hostresurrect.LoadCohort(hostresurrect.CohortPath(regDir))
	if cohortErr != nil {
		return report, fmt.Errorf("host-resurrection cohort: %w", cohortErr)
	}
	guardRows := guardsessions.Load(regDir)
	guards := make([]sessionrecovery.HostEvidenceRow, 0, len(guardRows))
	for _, row := range guardRows {
		guards = append(guards, sessionrecovery.HostEvidenceRow{
			Handle: row.Handle, TraceID: row.TraceID, ResumeHandle: row.ResumeHandle,
			PID: row.PID, StartedAt: row.StartedAt, CWD: row.CWD, Command: append([]string(nil), row.Command...),
		})
	}
	members := make([]sessionrecovery.HostCohortEntry, 0, len(cohort.Sessions))
	for _, row := range cohort.Sessions {
		members = append(members, sessionrecovery.HostCohortEntry{Handle: row.Handle, PID: row.PID, StartedAt: row.StartedAt})
	}
	_, uuidByTrace := resume.LoadIdentity(regDir)
	report = sessionrecovery.MergeCodexHostCohort(report, guards, members, uuidByTrace)
	return report, nil
}

func mergeClaudeRecoveryInventory(report sessionrecovery.InventoryReport, home string, since time.Duration, now time.Time, live map[string]bool) sessionrecovery.InventoryReport {
	if strings.TrimSpace(home) == "" {
		return report
	}
	paths, _ := filepath.Glob(filepath.Join(home, ".claude*", "projects", "*", "*.jsonl"))
	bySID := map[string][]string{}
	for _, path := range paths {
		sid := strings.TrimSuffix(filepath.Base(path), ".jsonl")
		if len(sid) == 36 {
			bySID[sid] = append(bySID[sid], path)
		}
	}
	cutoff := now.Add(-since)
	candidates := sweepCwdCandidates(home)
	fallback, _ := os.Getwd()
	for sid, copiesOnDisk := range bySID {
		newest := time.Time{}
		copies := make([]resumesweep.Copy, 0, len(copiesOnDisk))
		for _, path := range copiesOnDisk {
			if info, err := os.Stat(path); err == nil && info.ModTime().After(newest) {
				newest = info.ModTime()
			}
			copies = append(copies, loadSweepCopy(path))
		}
		if newest.Before(cutoff) {
			continue
		}
		classified := resumesweep.Classify(sid, copies, live, now)
		if classified.Bucket == resumesweep.BucketOther {
			continue
		}
		cwd := resumesweep.CwdForSlug(classified.Project, candidates, fallback)
		cursor, cursorAt := resumesweep.LastAssistantCursor(copies)
		session := sessionrecovery.Session{
			Thread:   &sessionrecovery.Thread{ID: sid, Source: "claude_transcript", CWD: cwd},
			Provider: sessionrecovery.ProviderClaude, Category: sessionrecovery.CategorySubstantive,
			Action: sessionrecovery.ActionRecover, Reason: strings.ToLower(classified.Bucket),
			Bucket: classified.Bucket, Cursor: cursor, CursorAt: cursorAt,
		}
		if cursor != "" || cursorAt != "" {
			session.LatestTurn = &sessionrecovery.Turn{ID: cursor, Status: "inProgress", StartedAt: cursorAt}
		}
		switch {
		case resumesweep.IsSemanticProbe(copies):
			session.Category = sessionrecovery.CategoryProbe
			session.Action = sessionrecovery.ActionExcludeProbe
			session.Reason = "semantic_probe:" + strings.ToLower(resumesweep.FirstUserPrompt(copies))
		case classified.Bucket == resumesweep.BucketLive:
			session.Category = sessionrecovery.CategoryLive
			session.Action = sessionrecovery.ActionLeaveLive
			session.Reason = "live_claude_driver"
		case classified.Bucket == resumesweep.BucketAuth:
			session.Category = sessionrecovery.CategoryIdentityBlocked
			session.Action = sessionrecovery.ActionLoginRequired
			session.Reason = "claude_login_required"
		case classified.Bucket == resumesweep.BucketLimitResetFuture:
			session.Action = sessionrecovery.ActionWaitReset
			session.Reason = "claude_reset_not_elapsed"
		}
		report.Sessions = append(report.Sessions, session)
	}
	return report
}

var recoveryJournalCrashes = func(path string, now time.Time) ([]sessionjournal.Classified, error) {
	boot, _ := sessionjournal.BootTime(now)
	sessions := sessionjournal.FoldEvents(sessionjournal.LoadFile(path))
	return sessionjournal.Classify(sessions, sessionjournal.ClassifyConfig{Now: now, BootTime: boot}), nil
}
var recoveryLaunch sessionrecovery.Launcher = sessionrecovery.VisibleLauncher{}
var recoveryNow = time.Now
var recoverySleep = time.Sleep
var recoveryClaudeAuthProbe = func(ctx context.Context) (bool, error) {
	if err := accounts.DefaultRefreshSpawn(ctx, guardClaudeConfigDir()); err != nil {
		return false, err
	}
	return true, nil
}

const recoveryClaudeAuthProbeTimeout = 30 * time.Second

func reconcileHistoricalClaudeAuth(report *sessionrecovery.InventoryReport) {
	var authRows []int
	for i := range report.Sessions {
		row := &report.Sessions[i]
		if row.Thread == nil ||
			row.Thread.Source != "claude_transcript" ||
			row.Provider != sessionrecovery.ProviderClaude ||
			row.Category != sessionrecovery.CategoryIdentityBlocked ||
			row.Action != sessionrecovery.ActionLoginRequired ||
			row.Bucket != resumesweep.BucketAuth {
			continue
		}
		authRows = append(authRows, i)
	}
	if len(authRows) == 0 {
		return
	}

	// Historical transcript state is not current account state. Spend one bounded
	// Claude-owned inference probe for the whole cohort, then admit every historical
	// AUTH row only on a positive live result. Failure and unknown remain blocked.
	ctx, cancel := context.WithTimeout(context.Background(), recoveryClaudeAuthProbeTimeout)
	defer cancel()
	authenticated, err := recoveryClaudeAuthProbe(ctx)
	if err != nil || !authenticated {
		return
	}
	for _, i := range authRows {
		report.Sessions[i].Category = sessionrecovery.CategorySubstantive
		report.Sessions[i].Action = sessionrecovery.ActionRecover
		report.Sessions[i].Reason = "claude_current_auth_live"
	}
}

func runSessionRecover(stdout, stderr io.Writer, args []string) int {
	for _, arg := range args {
		if arg == "--provider-launch" || strings.HasPrefix(arg, "--provider-launch=") {
			return runSessionRecoverProviderLaunch(stdout, stderr, args)
		}
	}
	fs := flag.NewFlagSet("session recover", flag.ContinueOnError)
	fs.SetOutput(stderr)
	since := fs.Duration("since", 24*time.Hour, "candidate evidence window")
	limit := fs.Int("limit", 0, "maximum launches per wave (0 = all actionable candidates)")
	liveMode := fs.Bool("live", false, "write receipts and launch one visible wrapper window per selected session")
	apply := fs.Bool("apply", false, "deprecated alias for --live")
	all := fs.Bool("all", false, "with --live, confirm every selected candidate instead of one explicit --thread")
	cwd := fs.String("cwd", "", "explicit override for cwd_unknown candidates")
	prompt := fs.String("prompt", "", "optional resume prompt staged in a private file outside native launch argv")
	journal := fs.Bool("journal", true, "include crash candidates from the session journal")
	journalPath := fs.String("journal-path", "", "session journal path (default machine journal)")
	settle := fs.Duration("settle", 0, "optional initial delay before verification")
	verifyTimeout := fs.Duration("verify-timeout", 30*time.Second, "bounded verification deadline")
	pollInterval := fs.Duration("poll-interval", 500*time.Millisecond, "interval between verification observations")
	jsonOutput := fs.Bool("json", false, "emit the versioned cohort summary as JSON")
	receipts := fs.String("receipts", "", "receipt directory")
	interactive := fs.Bool("interactive", false, "preserve native Codex TUI on relaunch (uses codex resume instead of codex exec resume)")
	threads := threadFlags{}
	fs.Var(threads, "thread", "thread ID to recover (repeatable)")
	if err := fs.Parse(args); err != nil {
		return 2
	}
	limitExplicit := false
	fs.Visit(func(f *flag.Flag) {
		if f.Name == "limit" {
			limitExplicit = true
		}
	})
	if fs.NArg() != 0 || *since <= 0 || *limit < 0 || *settle < 0 || *verifyTimeout < 0 || *pollInterval <= 0 {
		fmt.Fprintln(stderr, "usage: fak session recover [--thread ID] [--since 24h] [--limit N] [--cwd DIR] [--prompt TEXT] [--live (--thread ID | --all)] [--verify-timeout 30s] [--poll-interval 500ms]")
		return 2
	}
	doLive := *liveMode || *apply
	if doLive && len(threads) == 0 && !*all {
		fmt.Fprintln(stderr, "fak session recover: --live requires an explicit --thread ID or --all after reviewing the preview")
		return 2
	}
	if *all && len(threads) > 0 {
		fmt.Fprintln(stderr, "fak session recover: --all cannot be combined with --thread")
		return 2
	}
	if *receipts == "" {
		base, err := os.UserConfigDir()
		if err != nil {
			fmt.Fprintln(stderr, "fak session recover: config directory unavailable")
			return 2
		}
		*receipts = filepath.Join(base, "fak", "session-recovery")
	}
	regDir := resolveSweepRegDir("")
	if _, invalid, err := resume.LoadIdentityRowsStrict(regDir); err != nil || invalid > 0 {
		path := resume.IdentityLedgerPath(regDir)
		if err != nil {
			fmt.Fprintf(stderr, "fak session recover: identity authority unreadable: %s: %v\n", path, err)
		} else {
			fmt.Fprintf(stderr, "fak session recover: identity authority malformed: %s: %d invalid non-empty row(s)\n", path, invalid)
		}
		return 2
	}
	selectionLimit := *limit
	if !limitExplicit {
		selectionLimit = 0
	}
	before, err := recoveryInventory(*since)
	if err != nil {
		fmt.Fprintln(stderr, "fak session recover:", err)
		return 2
	}
	reconcileHistoricalClaudeAuth(&before)
	managerPath, exeErr := os.Executable()
	if exeErr != nil {
		fmt.Fprintln(stderr, "fak session recover: resolve current executable:", exeErr)
		return 2
	}
	requests := sessionrecovery.Select(before, sessionrecovery.Options{ManagerBin: managerPath, Threads: threads, Limit: selectionLimit, CWDOverride: *cwd, Prompt: *prompt, ReceiptDir: *receipts, Interactive: *interactive})
	if *journal {
		classified, journalErr := recoveryJournalCrashes(*journalPath, recoveryNow())
		if journalErr != nil {
			fmt.Fprintln(stderr, "fak session recover: session journal:", journalErr)
			return 2
		}
		requests = sessionrecovery.MergeJournalCrashes(requests, classified, sessionrecovery.Options{ManagerBin: managerPath, Threads: threads, Limit: selectionLimit, CWDOverride: *cwd, Prompt: *prompt, ReceiptDir: *receipts, Interactive: *interactive})
	}
	mode := "preview"
	if doLive {
		mode = "live"
	}
	startedAt := recoveryNow()
	summary := sessionrecovery.NewSummary(mode, before, requests, startedAt)
	summary.WitnessPath = sessionrecovery.SummaryPath(*receipts, startedAt)
	if err := persistRecoverySummary(&summary); err != nil {
		fmt.Fprintln(stderr, "fak session recover: write initial run witness:", err)
		return 1
	}
	if doLive {
		pending := make(map[int]time.Time)
		requestByResult := make(map[int]int)
		for i := range requests {
			if requests[i].Status != "candidate" {
				continue
			}
			resultIndex := recoveryResultIndex(summary.Results, requests[i].ThreadID)
			if resultIndex < 0 {
				continue
			}
			wrote, err := sessionrecovery.WriteReceipt(requests[i], recoveryNow())
			if err != nil {
				requests[i].Status = "receipt_failed"
				requests[i].Reason = err.Error()
				if !persistRecoveryResult(stderr, &summary, resultIndex, sessionRecoveryResult(requests[i])) {
					return 1
				}
				continue
			}
			if !wrote {
				requests[i].Status = "already_receipted"
				if !persistRecoveryResult(stderr, &summary, resultIndex, sessionRecoveryResult(requests[i])) {
					return 1
				}
				continue
			}
			if err := sessionrecovery.StagePrompt(requests[i]); err != nil {
				requests[i].Status = "prompt_failed"
				requests[i].Reason = err.Error()
				_ = sessionrecovery.FinalizeReceipt(requests[i], requests[i].Status, requests[i].Reason, recoveryNow())
				if !persistRecoveryResult(stderr, &summary, resultIndex, sessionRecoveryResult(requests[i])) {
					return 1
				}
				continue
			}
			intent := sessionRecoveryResult(requests[i])
			intent.Status = "launch_intent"
			intent.Reason = "receipt persisted before visible launch"
			if !persistRecoveryResult(stderr, &summary, resultIndex, intent) {
				return 1
			}
			launchedAt := recoveryNow()
			handle, launchErr := recoveryLaunch.Launch(requests[i])
			if launchErr != nil {
				requests[i].Status = "launch_failed"
				requests[i].Reason = launchErr.Error()
				_ = sessionrecovery.FinalizeReceipt(requests[i], requests[i].Status, requests[i].Reason, recoveryNow())
				if !persistRecoveryResult(stderr, &summary, resultIndex, sessionRecoveryResult(requests[i])) {
					return 1
				}
				continue
			}
			requests[i].Status = "launched_unproven"
			result := sessionRecoveryResult(requests[i])
			result.LaunchIdentity = handle.Identity()
			result.LaunchedAt = launchedAt.UTC().Format(time.RFC3339Nano)
			result.BaselineCursor = summary.Results[resultIndex].BaselineCursor
			result.BaselineAt = summary.Results[resultIndex].BaselineAt
			summary.Results[resultIndex] = result
			pending[resultIndex] = launchedAt.Add(*verifyTimeout)
			requestByResult[resultIndex] = i
			if err := persistRecoverySummary(&summary); err != nil {
				fmt.Fprintln(stderr, "fak session recover: update run witness:", err)
				return 1
			}
		}
		if *settle > 0 && len(pending) > 0 {
			recoverySleep(*settle)
		}
		for len(pending) > 0 {
			after, inventoryErr := recoveryInventory(*since)
			observedAt := recoveryNow()
			for resultIndex, deadline := range pending {
				result := summary.Results[resultIndex]
				if inventoryErr != nil {
					result.Status = "verification_failed"
					result.Reason = inventoryErr.Error()
					result.Remediation = sessionrecovery.Remediation(result)
				} else {
					result = sessionrecovery.Observe(before, after, result)
				}
				requestIndex := requestByResult[resultIndex]
				requests[requestIndex].Status, requests[requestIndex].Reason = result.Status, result.Reason
				deadlineReached := !observedAt.Before(deadline) || *verifyTimeout == 0
				if (sessionrecovery.TerminalStatus(result.Status) && result.Status != "verification_failed") || deadlineReached {
					if finalizeErr := sessionrecovery.FinalizeReceipt(requests[requestIndex], result.Status, result.Reason, recoveryNow()); finalizeErr != nil {
						result.Status = "receipt_failed"
						result.Reason = finalizeErr.Error()
						result.Remediation = sessionrecovery.Remediation(result)
					}
					delete(pending, resultIndex)
				}
				summary.Results[resultIndex] = result
				if err := persistRecoverySummary(&summary); err != nil {
					fmt.Fprintln(stderr, "fak session recover: update run witness:", err)
					return 1
				}
			}
			if len(pending) > 0 {
				recoverySleep(*pollInterval)
			}
		}
	}
	summary.FinishedAt = recoveryNow().UTC().Format(time.RFC3339Nano)
	summary.Recount()
	if err := sessionrecovery.WriteSummary(summary.WitnessPath, summary); err != nil {
		fmt.Fprintln(stderr, "fak session recover: write run witness:", err)
		return 1
	}
	if *jsonOutput {
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		if err := enc.Encode(summary); err != nil {
			return 1
		}
	} else {
		renderRecoverySummary(stdout, summary)
	}
	if summary.Counts.Failed > 0 || summary.Counts.LaunchedUnproven > 0 {
		return 1
	}
	return 0
}

func runSessionRecoverProviderLaunch(stdout, stderr io.Writer, args []string) int {
	fs := flag.NewFlagSet("session recover provider launch", flag.ContinueOnError)
	fs.SetOutput(stderr)
	provider := fs.String("provider-launch", "", "provider to resume")
	thread := fs.String("thread", "", "provider thread/session id")
	cwd := fs.String("cwd", "", "provider working directory")
	promptFile := fs.String("prompt-file", "", "private continuation prompt file")
	codexBin := fs.String("codex", "codex", "Codex executable")
	interactive := fs.Bool("interactive", false, "preserve native Codex TUI on relaunch")
	if err := fs.Parse(args); err != nil {
		return 2
	}
	if fs.NArg() != 0 || strings.TrimSpace(*thread) == "" || strings.TrimSpace(*cwd) == "" || strings.TrimSpace(*promptFile) == "" {
		fmt.Fprintln(stderr, "usage: fak session recover --provider-launch claude|codex --thread ID --cwd DIR --prompt-file FILE [--codex BIN] [--interactive]")
		return 2
	}
	prompt, err := os.ReadFile(*promptFile)
	if err != nil {
		fmt.Fprintln(stderr, "fak session recover provider launch: read prompt file:", err)
		return 1
	}
	managerBin, err := os.Executable()
	if err != nil {
		fmt.Fprintln(stderr, "fak session recover provider launch: resolve guard executable")
		return 1
	}
	argv := []string{"guard", "--"}
	switch *provider {
	case sessionrecovery.ProviderClaude:
		// Claude print mode consumes the continuation prompt from stdin when no
		// positional prompt is present. Keep both control and provider argv
		// independent of prompt bytes; fak guard preserves stdin at the boundary.
		argv = append(argv, "claude", "--print", "--resume", *thread)
	case sessionrecovery.ProviderCodex:
		if *interactive {
			argv = append(argv, *codexBin, "resume", *thread)
		} else {
			argv = append(argv, *codexBin, "exec", "--cd", *cwd, "resume", *thread, "-")
		}
	default:
		fmt.Fprintf(stderr, "fak session recover provider launch: unsupported provider %q\n", *provider)
		return 2
	}
	cmd := recoveryProviderCommand(managerBin, argv...)
	cmd.Dir = *cwd
	cmd.Stdin = strings.NewReader(string(prompt))
	cmd.Stdout = stdout
	cmd.Stderr = stderr
	if err := cmd.Run(); err != nil {
		// Do not include argv or prompt-bearing command diagnostics. The staged
		// file remains available for an explicit retry after provider failure.
		fmt.Fprintln(stderr, "fak session recover provider launch: guarded provider failed")
		return 1
	}
	if err := os.Remove(*promptFile); err != nil && !errors.Is(err, os.ErrNotExist) {
		fmt.Fprintln(stderr, "fak session recover provider launch: remove staged prompt:", err)
		return 1
	}
	return 0
}

func recoveryResultIndex(results []sessionrecovery.Result, threadID string) int {
	for i := range results {
		if results[i].ThreadID == threadID {
			return i
		}
	}
	return -1
}

func persistRecoverySummary(summary *sessionrecovery.Summary) error {
	summary.FinishedAt = recoveryNow().UTC().Format(time.RFC3339Nano)
	summary.Recount()
	return sessionrecovery.WriteSummary(summary.WitnessPath, *summary)
}

func persistRecoveryResult(stderr io.Writer, summary *sessionrecovery.Summary, index int, result sessionrecovery.Result) bool {
	summary.Results[index] = result
	if err := persistRecoverySummary(summary); err != nil {
		fmt.Fprintln(stderr, "fak session recover: update run witness:", err)
		return false
	}
	return true
}

func sessionRecoveryResult(req sessionrecovery.Request) sessionrecovery.Result {
	result := sessionrecovery.Result{ThreadID: req.ThreadID, CWD: req.CWD, Source: req.Source, Provider: req.Provider, Category: req.Category, Action: req.Action, Status: req.Status, Reason: req.Reason, ReceiptPath: req.ReceiptPath, Argv: append([]string(nil), req.Argv...), HostHandles: append([]string(nil), req.HostHandles...), IdentityProvenance: req.IdentityProvenance}
	if req.Status == "candidate" {
		result.SelectionReason = "crashed session has an in-progress turn and no live process tree"
	}
	result.Remediation = sessionrecovery.Remediation(result)
	return result
}

func renderRecoverySummary(w io.Writer, summary sessionrecovery.Summary) {
	fmt.Fprintf(w, "SESSION RECOVERY %s\n", strings.ToUpper(summary.Mode))
	fmt.Fprintf(w, "witness=%s\n", summary.WitnessPath)
	fmt.Fprintf(w, "discovered=%d selected=%d actionable=%d omitted=%d launched=%d active=%d productive=%d completed=%d failed=%d stalled=%d unproven=%d exact_cardinality=%d\n", summary.Counts.Discovered, summary.Counts.Selected, summary.Counts.Actionable, summary.Counts.OmittedByLimit, summary.Counts.Launched, summary.Counts.Active, summary.Counts.Productive, summary.Counts.Completed, summary.Counts.Failed, summary.Counts.Stalled, summary.Counts.LaunchedUnproven, summary.Counts.ExactCardinality)
	if summary.RecoverAllCommand != "" {
		fmt.Fprintf(w, "recover-all: %s\n", summary.RecoverAllCommand)
	}
	if len(summary.Results) == 0 {
		fmt.Fprintln(w, "No crashed sessions need recovery.")
		return
	}
	tw := tabwriter.NewWriter(w, 0, 4, 2, ' ', 0)
	fmt.Fprintln(tw, "THREAD\tPROVIDER\tCATEGORY\tSTATUS\tTREES\tCWD")
	for _, result := range summary.Results {
		fmt.Fprintf(tw, "%s\t%s\t%s\t%s\t%d\t%s\n", result.ThreadID, result.Provider, result.Category, result.Status, result.GuardedProcessTrees, result.CWD)
		if result.Reason != "" {
			fmt.Fprintf(tw, "\twhy: %s\t\t\n", result.Reason)
		}
		if result.Remediation != "" {
			fmt.Fprintf(tw, "\tnext: %s\t\t\n", result.Remediation)
		}
	}
	_ = tw.Flush()
}
