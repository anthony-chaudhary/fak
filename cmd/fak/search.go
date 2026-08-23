package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/fleetsearch"
	"github.com/anthony-chaudhary/fak/internal/sessionjournal"
	"github.com/anthony-chaudhary/fak/internal/sessionregistry"
)

func cmdSearch(args []string) { os.Exit(runSearch(os.Stdout, os.Stderr, args)) }

func runSearch(stdout, stderr io.Writer, args []string) int {
	fs := flag.NewFlagSet("search", flag.ContinueOnError)
	fs.SetOutput(stderr)
	lifecyclePath := fs.String("lifecycle", sessionjournal.DefaultPath(), "session lifecycle JSONL store")
	registrationPath := fs.String("registrations", sessionregistry.DefaultPath(), "child-registration JSONL store")
	toolProcessPath := fs.String("tool-processes", filepath.Join(".fak", "toolproc", "journal.jsonl"), "tool-process JSONL store")
	skipLifecycle := fs.Bool("skip-lifecycle", false, "skip lifecycle store and report partial coverage")
	skipRegistration := fs.Bool("skip-registrations", false, "skip child-registration store and report partial coverage")
	skipToolProcess := fs.Bool("skip-tool-processes", false, "skip tool-process store and report partial coverage")
	limit := fs.Int("limit", fleetsearch.DefaultLimit, "maximum returned sessions (1..100; limit:N in the query overrides it)")
	staleAfter := fs.Duration("stale-after", 15*time.Minute, "open-session heartbeat age classified as STALE (0 disables)")
	nowUnix := fs.Int64("now", 0, "observation Unix timestamp (tests/replay)")
	bootUnix := fs.Int64("boot-time", 0, "known machine boot Unix timestamp (0 leaves reboot liveness unknown)")
	asJSON := fs.Bool("json", false, "emit the stable machine-readable report")
	fs.Usage = func() { searchUsage(stdout, fs) }
	if rc, ok := parseFlagsOrHelp(fs, args); !ok {
		return rc
	}
	if fs.NArg() != 1 || strings.TrimSpace(fs.Arg(0)) == "" {
		fmt.Fprintln(stderr, "fak search: exactly one QUERY is required")
		fmt.Fprintln(stderr, "try: fak search --json \"confluence is:active\"")
		return 2
	}
	if *limit < 1 || *limit > fleetsearch.MaxLimit {
		fmt.Fprintf(stderr, "fak search: --limit must be between 1 and %d\n", fleetsearch.MaxLimit)
		return 2
	}
	if *staleAfter < 0 {
		fmt.Fprintln(stderr, "fak search: --stale-after cannot be negative")
		return 2
	}
	now := time.Now().UTC()
	if *nowUnix != 0 {
		now = time.Unix(*nowUnix, 0).UTC()
	}
	var boot time.Time
	if *bootUnix != 0 {
		boot = time.Unix(*bootUnix, 0).UTC()
	}
	report, err := fleetsearch.Run(fs.Arg(0), fleetsearch.Config{
		LifecyclePath: *lifecyclePath, RegistrationPath: *registrationPath, ToolProcessPath: *toolProcessPath,
		SkipLifecycle: *skipLifecycle, SkipRegistration: *skipRegistration, SkipToolProcess: *skipToolProcess,
		Now: now, BootTime: boot, StaleAfter: *staleAfter, Limit: *limit,
	})
	if err != nil {
		fmt.Fprintf(stderr, "fak search: %v\n", err)
		return 2
	}
	if *asJSON {
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		if err := enc.Encode(report); err != nil {
			fmt.Fprintf(stderr, "fak search: encode report: %v\n", err)
			return 1
		}
		return 0
	}
	writeSearchHuman(stdout, report)
	return 0
}

func searchUsage(w io.Writer, fs *flag.FlagSet) {
	fmt.Fprintln(w, "usage: fak search [flags] QUERY")
	fmt.Fprintln(w)
	fmt.Fprintln(w, "Search lifecycle, child-registration, and tool-process stores by logical session ID.")
	fmt.Fprintln(w, "Plain terms rank safe durable evidence; facets include is:active|stale|crashed|completed,")
	fmt.Fprintln(w, "store:lifecycle|registration|tool-process, and limit:N. A missing or skipped store")
	fmt.Fprintln(w, "returns INCOMPLETE_EVIDENCE instead of claiming sole-match or no-match certainty.")
	fmt.Fprintf(w, "JSON schema: %s\n\n", fleetsearch.Schema)
	prior := fs.Output()
	fs.SetOutput(w)
	fs.PrintDefaults()
	fs.SetOutput(prior)
}

func writeSearchHuman(w io.Writer, report fleetsearch.Report) {
	fmt.Fprintf(w, "VERDICT %s\n", report.Verdict)
	fmt.Fprintf(w, "QUERY %q terms=%d matches=%d returned=%d\n", report.Query.Raw, len(report.Query.Terms), report.TotalMatches, report.Returned)
	fmt.Fprint(w, "COVERAGE")
	for _, coverage := range report.Coverage {
		fmt.Fprintf(w, " %s=%s", coverage.Store, coverage.Status)
	}
	fmt.Fprintln(w)
	for i, hit := range report.Hits {
		fmt.Fprintf(w, "%d. %s %s score=%d\n", i+1, hit.SessionID, hit.Liveness, hit.Score)
		if len(hit.Objectives) > 0 {
			fmt.Fprintf(w, "   objective: %s\n", strings.Join(hit.Objectives, "; "))
		}
		if len(hit.Scope) > 0 {
			fmt.Fprintf(w, "   scope: %s\n", strings.Join(hit.Scope, "; "))
		}
		if len(hit.Tools) > 0 {
			fmt.Fprintf(w, "   tools: %s\n", strings.Join(hit.Tools, ", "))
		}
		fmt.Fprintf(w, "   evidence: %s\n", strings.Join(searchEvidenceStores(hit.Evidence), ", "))
	}
}

func searchEvidenceStores(evidence []fleetsearch.Evidence) []string {
	seen := map[fleetsearch.Store]bool{}
	var stores []string
	for _, row := range evidence {
		if !seen[row.Store] {
			stores = append(stores, string(row.Store))
			seen[row.Store] = true
		}
	}
	return stores
}
