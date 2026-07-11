// dispatchaudit.go — `fak dispatch audit`, the value-side complement of
// `fak dispatch status` (which reports backend HEALTH). audit folds the
// .dispatch-runs/ worker logs (+ .backend sidecars + progress ledger) into a
// per-worker outcome classification and a per-backend wasted-spawn / wasted-
// wall-clock rollup — the actual-value number the status verb lacks.
//
//	# read-only rollup
//	fak dispatch audit
//	# machine-readable findings (each with a stable fingerprint)
//	fak dispatch audit --json
//	# detect -> fingerprint -> dedup -> file: opens a gh issue for NEW fingerprints only
//	fak dispatch audit --file-issues
//	# opt-in: append a STARTED/NEVER_STARTED heartbeat event per worker to the loop ledger
//	fak dispatch audit --heartbeat
//
// The classification is PURE (internal/dispatchaudit.Fold); this file is the I/O
// shell — it reads the dir, renders the table, and (only with --file-issues)
// writes the dedup markers and shells out to `gh issue create`.
package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os/exec"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/dispatchaudit"
)

// runDispatchAudit is the testable core of `fak dispatch audit`.
func runDispatchAudit(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("dispatch audit", flag.ContinueOnError)
	fs.SetOutput(stderr)
	runsDir := fs.String("runs-dir", dispatchProgressRunsDir, "directory of dispatch worker logs")
	asJSON := fs.Bool("json", false, "emit the raw Report JSON instead of the human table")
	fileIssues := fs.Bool("file-issues", false, "file a gh issue for each NEW finding fingerprint (default: read-only)")
	maxIssues := fs.Int("max-issues", 0, "with --file-issues: hard cap on issues filed per run, worst-first (0 = no cap). The anti-storm bound for the scheduled feeder's first run over a large backlog")
	stormErrs := fs.Int("storm-errors", 0, "RETRY_STORM threshold: min provider-error lines (0 = default)")
	stormMins := fs.Float64("storm-mins", 0, "RETRY_STORM threshold: min wall-clock minutes (0 = default)")
	hookMin := fs.Int("hook-min", 0, "hook-failure-storm signature floor: min `hook: … Failed` lines per session (0 = default 3)")
	heartbeat := fs.Bool("heartbeat", false, "append a STARTED/NEVER_STARTED heartbeat event per worker to the loop ledger (default: off — a routine read-only audit never writes)")
	ledger := fs.String("ledger", "", "loop JSONL ledger path for --heartbeat (default: the loop ledger)")
	if !parseFlags(fs, argv) {
		return 2
	}

	workers, err := dispatchaudit.ScanDir(*runsDir)
	if err != nil {
		fmt.Fprintf(stderr, "fak dispatch audit: scan %s: %v\n", *runsDir, err)
		return 1
	}

	th := dispatchaudit.DefaultThresholds()
	if *stormErrs > 0 {
		th.StormMinErrors = *stormErrs
	}
	if *stormMins > 0 {
		th.StormMinMins = *stormMins
	}
	rep := dispatchaudit.Fold(workers, th)

	// Log-signature failure classifiers (issue #3337): scan the raw log text for
	// panic/traceback, hook-failure storms, OFF_TRUNK refusal storms, auth walls,
	// and banner-only no-ops, then merge them into the fileable findings so
	// --file-issues covers BOTH waste and log-signature failures. Best-effort: a
	// scan error is reported, never fatal to the waste rollup above.
	sigTh := dispatchaudit.DefaultSignatureThresholds()
	if *hookMin > 0 {
		sigTh.HookMin = *hookMin
	}
	if sigs, sigErr := dispatchaudit.ScanDirSignatures(*runsDir, sigTh); sigErr != nil {
		fmt.Fprintf(stderr, "fak dispatch audit: signature scan %s: %v\n", *runsDir, sigErr)
	} else {
		for _, sf := range sigs {
			rep.Findings = append(rep.Findings, sf.AsFinding())
		}
	}

	if *heartbeat {
		ledgerPath := firstNonEmpty(*ledger, defaultLoopLedger())
		n, err := appendWorkerHeartbeats(ledgerPath, rep.Classifications)
		if err != nil {
			fmt.Fprintf(stderr, "fak dispatch audit: heartbeat: %v\n", err)
			return 1
		}
		if !*asJSON {
			fmt.Fprintf(stdout, "heartbeat: recorded %d event(s) to %s\n\n", n, ledgerPath)
		}
	}

	if *asJSON {
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		if err := enc.Encode(rep); err != nil {
			fmt.Fprintf(stderr, "fak dispatch audit: encode json: %v\n", err)
			return 1
		}
		return 0
	}

	renderAuditTable(stdout, rep)

	if *fileIssues {
		return fileAuditFindings(stdout, stderr, *runsDir, rep, *maxIssues)
	}
	return 0
}

// renderAuditTable prints the per-backend rollup and the deduped findings.
func renderAuditTable(w io.Writer, rep dispatchaudit.Report) {
	fmt.Fprintf(w, "dispatch audit — %d workers, %d distinct findings\n\n", len(rep.Classifications), len(rep.Findings))
	fmt.Fprintf(w, "%-10s %7s %7s %7s %7s %7s %7s %7s %7s %12s\n",
		"backend", "workers", "shipped", "running", "wasted", "walled", "storm", "no-op", "errored", "wasted-min")
	for _, r := range rep.Rollups {
		fmt.Fprintf(w, "%-10s %7d %7d %7d %7d %7d %7d %7d %7d %12.1f\n",
			r.Backend, r.Workers, r.Shipped, r.Running, r.WastedSpawns, r.QuotaWalled, r.RetryStorms, r.NoOps, r.Errored, r.WastedMinutes)
	}
	if len(rep.Findings) > 0 {
		fmt.Fprintf(w, "\nfindings (fingerprint  outcome):\n")
		for _, f := range rep.Findings {
			fmt.Fprintf(w, "  %s  %s\n      %s\n", f.Fingerprint, f.Title, f.Detail)
		}
	}
}

// fileAuditFindings is the detect->fingerprint->dedup->file half: it files a gh
// issue for each finding whose fingerprint is neither already-marked nor an
// existing open-issue title, then writes the marker so it is never re-filed.
// maxIssues (>0) caps the run worst-first — the anti-storm bound the scheduled
// feeder passes on its first pass over a large historical backlog; the withheld
// remainder is reported, never silently dropped.
func fileAuditFindings(stdout, stderr io.Writer, runsDir string, rep dispatchaudit.Report, maxIssues int) int {
	filed := map[string]bool{}
	for _, f := range rep.Findings {
		if dispatchaudit.AlreadyFiled(runsDir, f.Fingerprint) {
			filed[f.Fingerprint] = true
		}
	}
	openTitles := openIssueTitles(stderr)

	fresh, withheld := dispatchaudit.SelectFindingsToFile(rep.Findings, filed, openTitles, maxIssues)
	if len(fresh) == 0 {
		fmt.Fprintln(stdout, "\nfile-issues: nothing new to file (all findings deduped).")
		return 0
	}
	if withheld > 0 {
		fmt.Fprintf(stdout, "\nfile-issues: filing %d worst-first; %d more withheld by --max-issues=%d (re-runs next cadence).\n", len(fresh), withheld, maxIssues)
	}

	rc := 0
	for _, f := range fresh {
		kind := string(f.Outcome)
		kindLabel := "outcome"
		if f.SignatureClass != "" {
			kind = string(f.SignatureClass)
			kindLabel = "signature"
		}
		body := fmt.Sprintf("Auto-filed by `fak dispatch audit`.\n\n- dispatchability: `triage_only`\n- %s: `%s`\n- backend: `%s`\n- code-site: `%s`\n- fingerprint: `%s`\n- first log: `%s`\n\n%s",
			kindLabel, kind, f.Backend, f.CodeSite, f.Fingerprint, f.Log, f.Detail)
		args := []string{"issue", "create",
			"--title", f.Title,
			"--body", body}
		for _, label := range dispatchAuditIssueLabels() {
			args = append(args, "--label", label)
		}
		ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
		cmd := exec.CommandContext(ctx, "gh", args...)
		configureDispatchHelperCommand(cmd)
		cmd.WaitDelay = 10 * time.Second
		out, err := cmd.CombinedOutput()
		cancel()
		if err != nil {
			fmt.Fprintf(stderr, "file-issues: gh issue create for %s failed: %v\n%s\n", f.Fingerprint, err, out)
			rc = 1
			continue
		}
		fmt.Fprintf(stdout, "filed %s: %s", f.Fingerprint, strings.TrimSpace(string(out)))
		fmt.Fprintln(stdout)
		if err := dispatchaudit.MarkFiled(runsDir, f.Fingerprint); err != nil {
			fmt.Fprintf(stderr, "file-issues: mark %s: %v\n", f.Fingerprint, err)
		}
	}
	return rc
}

func dispatchAuditIssueLabels() []string {
	return []string{"dispatch", "observability", "needs-triage", "triage-only"}
}

// openIssueTitles scans `gh issue list` for open titles so the dedup can avoid a
// duplicate even when no marker exists yet (e.g. a peer filed it). Best-effort —
// a gh failure yields an empty set, never a hard error.
func openIssueTitles(stderr io.Writer) map[string]bool {
	out := map[string]bool{}
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, "gh", "issue", "list", "--state", "open", "--limit", "400", "--json", "title")
	configureDispatchHelperCommand(cmd)
	cmd.WaitDelay = 10 * time.Second
	b, err := cmd.Output()
	if err != nil {
		fmt.Fprintf(stderr, "file-issues: gh issue list (dedup scan) failed; relying on markers only: %v\n", err)
		return out
	}
	var rows []struct {
		Title string `json:"title"`
	}
	if err := json.Unmarshal(b, &rows); err != nil {
		return out
	}
	for _, r := range rows {
		out[r.Title] = true
	}
	return out
}
