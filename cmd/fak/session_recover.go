package main

import (
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"text/tabwriter"
	"time"

	"github.com/anthony-chaudhary/fak/internal/devcmd"
	"github.com/anthony-chaudhary/fak/internal/sessionjournal"
	"github.com/anthony-chaudhary/fak/internal/sessionrecovery"
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

var recoverySessionDiag = func(stdout, stderr io.Writer, args []string) int {
	return devcmd.RunSessionDiag(stdout, stderr, args, nil)
}

var recoveryInventory = func(since time.Duration) (sessionrecovery.InventoryReport, error) {
	var out, er bytes.Buffer
	args := []string{"--inventory", "--json", "--since", since.String()}
	if code := recoverySessionDiag(&out, &er, args); code != 0 {
		reason := strings.TrimSpace(er.String())
		if reason == "" {
			reason = fmt.Sprintf("exit %d", code)
		}
		return sessionrecovery.InventoryReport{}, fmt.Errorf("sessiondiag inventory: %s", reason)
	}
	var report sessionrecovery.InventoryReport
	if err := json.Unmarshal(out.Bytes(), &report); err != nil {
		return report, fmt.Errorf("decode sessiondiag inventory: %w", err)
	}
	return report, nil
}
var recoveryJournalCrashes = func(path string, now time.Time) ([]sessionjournal.Classified, error) {
	boot, _ := sessionjournal.BootTime(now)
	sessions := sessionjournal.FoldEvents(sessionjournal.LoadFile(path))
	return sessionjournal.Classify(sessions, sessionjournal.ClassifyConfig{Now: now, BootTime: boot}), nil
}
var recoveryLaunch sessionrecovery.Launcher = sessionrecovery.VisibleLauncher{}
var recoveryNow = time.Now
var recoverySleep = time.Sleep

func cmdSessionRecover(args []string) { os.Exit(runSessionRecover(os.Stdout, os.Stderr, args)) }

func runSessionRecover(stdout, stderr io.Writer, args []string) int {
	fs := flag.NewFlagSet("session recover", flag.ContinueOnError)
	fs.SetOutput(stderr)
	since := fs.Duration("since", 24*time.Hour, "candidate evidence window")
	limit := fs.Int("limit", 1, "maximum launches per wave")
	apply := fs.Bool("apply", false, "write receipts and launch visible resumes")
	all := fs.Bool("all", false, "with --apply, confirm every selected candidate instead of one explicit --thread")
	cwd := fs.String("cwd", "", "explicit override for cwd_unknown candidates")
	prompt := fs.String("prompt", "", "optional resume prompt passed as one exact argv element")
	journal := fs.Bool("journal", true, "include crash candidates from the session journal")
	journalPath := fs.String("journal-path", "", "session journal path (default machine journal)")
	settle := fs.Duration("settle", 0, "optional initial delay before verification")
	verifyTimeout := fs.Duration("verify-timeout", 30*time.Second, "bounded verification deadline")
	pollInterval := fs.Duration("poll-interval", 500*time.Millisecond, "interval between verification observations")
	jsonOutput := fs.Bool("json", false, "emit the versioned cohort summary as JSON")
	receipts := fs.String("receipts", "", "receipt directory")
	threads := threadFlags{}
	fs.Var(threads, "thread", "thread ID to recover (repeatable)")
	if err := fs.Parse(args); err != nil {
		return 2
	}
	if fs.NArg() != 0 || *since <= 0 || *limit <= 0 || *settle < 0 || *verifyTimeout < 0 || *pollInterval <= 0 {
		fmt.Fprintln(stderr, "usage: fak session recover [--thread ID] [--since 24h] [--limit 1] [--cwd DIR] [--prompt TEXT] [--apply (--thread ID | --all)] [--verify-timeout 30s] [--poll-interval 500ms]")
		return 2
	}
	if *apply && len(threads) == 0 && !*all {
		fmt.Fprintln(stderr, "fak session recover: --apply requires an explicit --thread ID or --all after reviewing the preview")
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
	before, err := recoveryInventory(*since)
	if err != nil {
		fmt.Fprintln(stderr, "fak session recover:", err)
		return 2
	}
	managerPath, exeErr := os.Executable()
	if exeErr != nil {
		fmt.Fprintln(stderr, "fak session recover: resolve current executable:", exeErr)
		return 2
	}
	requests := sessionrecovery.Select(before, sessionrecovery.Options{ManagerBin: managerPath, Threads: threads, Limit: *limit, CWDOverride: *cwd, Prompt: *prompt, ReceiptDir: *receipts})
	if *journal {
		classified, journalErr := recoveryJournalCrashes(*journalPath, recoveryNow())
		if journalErr != nil {
			fmt.Fprintln(stderr, "fak session recover: session journal:", journalErr)
			return 2
		}
		requests = sessionrecovery.MergeJournalCrashes(requests, classified, sessionrecovery.Options{ManagerBin: managerPath, Threads: threads, Limit: *limit, CWDOverride: *cwd, Prompt: *prompt, ReceiptDir: *receipts})
	}
	mode := "preview"
	startedAt := recoveryNow()
	summary := sessionrecovery.NewSummary(mode, before, requests, startedAt)
	if *apply {
		summary.Mode = "apply"
		results := make([]sessionrecovery.Result, 0, len(requests))
		for i := range requests {
			if requests[i].Status != "candidate" {
				results = append(results, sessionRecoveryResult(requests[i]))
				continue
			}
			wrote, err := sessionrecovery.WriteReceipt(requests[i], recoveryNow())
			if err != nil {
				requests[i].Status = "receipt_failed"
				requests[i].Reason = err.Error()
				results = append(results, sessionRecoveryResult(requests[i]))
				continue
			}
			if !wrote {
				requests[i].Status = "already_receipted"
				results = append(results, sessionRecoveryResult(requests[i]))
				continue
			}
			if err := recoveryLaunch.Launch(requests[i]); err != nil {
				requests[i].Status = "launch_failed"
				requests[i].Reason = err.Error()
				_ = sessionrecovery.FinalizeReceipt(requests[i], requests[i].Status, requests[i].Reason, recoveryNow())
				results = append(results, sessionRecoveryResult(requests[i]))
				continue
			}
			requests[i].Status = "launched_unproven"
			result := sessionRecoveryResult(requests[i])
			deadline := recoveryNow().Add(*verifyTimeout)
			if *settle > 0 {
				recoverySleep(*settle)
			}
			for {
				after, inventoryErr := recoveryInventory(*since)
				if inventoryErr != nil {
					result.Status = "verification_failed"
					result.Reason = inventoryErr.Error()
					result.Remediation = sessionrecovery.Remediation(result)
				} else {
					result = sessionrecovery.Observe(before, after, result)
				}
				deadlineReached := !recoveryNow().Before(deadline) || *verifyTimeout == 0
				if sessionrecovery.TerminalStatus(result.Status) && result.Status != "verification_failed" {
					break
				}
				if deadlineReached {
					// An exact guarded Codex tree proves the interactive resume is alive.
					// A later fresh turn upgrades active to productive; it is not required
					// merely to keep a usable, waiting TUI from being reported failed.
					break
				}
				recoverySleep(*pollInterval)
			}
			requests[i].Status, requests[i].Reason = result.Status, result.Reason
			if finalizeErr := sessionrecovery.FinalizeReceipt(requests[i], requests[i].Status, requests[i].Reason, recoveryNow()); finalizeErr != nil {
				result.Status = "receipt_failed"
				result.Reason = finalizeErr.Error()
				result.Remediation = sessionrecovery.Remediation(result)
				requests[i].Status, requests[i].Reason = result.Status, result.Reason
			}
			results = append(results, result)
		}
		summary.Results = results
	}
	summary.FinishedAt = recoveryNow().UTC().Format(time.RFC3339Nano)
	summary.Recount()
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

func sessionRecoveryResult(req sessionrecovery.Request) sessionrecovery.Result {
	result := sessionrecovery.Result{ThreadID: req.ThreadID, CWD: req.CWD, Source: req.Source, Status: req.Status, Reason: req.Reason, ReceiptPath: req.ReceiptPath, Argv: append([]string(nil), req.Argv...)}
	if req.Status == "candidate" {
		result.SelectionReason = "crashed session has an in-progress turn and no live process tree"
	}
	result.Remediation = sessionrecovery.Remediation(result)
	return result
}

func renderRecoverySummary(w io.Writer, summary sessionrecovery.Summary) {
	fmt.Fprintf(w, "SESSION RECOVERY %s\n", strings.ToUpper(summary.Mode))
	fmt.Fprintf(w, "discovered=%d selected=%d launched=%d active=%d productive=%d completed=%d failed=%d unproven=%d exact_cardinality=%d\n", summary.Counts.Discovered, summary.Counts.Selected, summary.Counts.Launched, summary.Counts.Active, summary.Counts.Productive, summary.Counts.Completed, summary.Counts.Failed, summary.Counts.LaunchedUnproven, summary.Counts.ExactCardinality)
	if len(summary.Results) == 0 {
		fmt.Fprintln(w, "No crashed sessions need recovery.")
		return
	}
	tw := tabwriter.NewWriter(w, 0, 4, 2, ' ', 0)
	fmt.Fprintln(tw, "THREAD\tSTATUS\tTREES\tCWD")
	for _, result := range summary.Results {
		fmt.Fprintf(tw, "%s\t%s\t%d\t%s\n", result.ThreadID, result.Status, result.GuardedProcessTrees, result.CWD)
		if result.Reason != "" {
			fmt.Fprintf(tw, "\twhy: %s\t\t\n", result.Reason)
		}
		if result.Remediation != "" {
			fmt.Fprintf(tw, "\tnext: %s\t\t\n", result.Remediation)
		}
	}
	_ = tw.Flush()
}
