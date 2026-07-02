package main

import (
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"time"

	"github.com/anthony-chaudhary/fak/internal/sessionaudit"
)

func cmdSessionAudit(argv []string) { os.Exit(runSessionAudit(os.Stdout, os.Stderr, argv)) }

// runSessionAudit is the testable shell for `fak session-audit`.
func runSessionAudit(stdout, stderr io.Writer, argv []string) int {
	if len(argv) == 0 {
		sessionAuditUsage(stderr)
		return 2
	}
	switch argv[0] {
	case "discover":
		return runSessionAuditDiscover(stdout, stderr, argv[1:])
	case "audit":
		return runSessionAuditAudit(stdout, stderr, argv[1:])
	case "deep":
		return runSessionAuditDeep(stdout, stderr, argv[1:])
	case "-h", "--help", "help":
		sessionAuditUsage(stdout)
		return 0
	default:
		fmt.Fprintf(stderr, "fak session-audit: unknown subcommand %q\n", argv[0])
		sessionAuditUsage(stderr)
		return 2
	}
}

func sessionAuditUsage(w io.Writer) {
	fmt.Fprintln(w, "usage: fak session-audit discover [--since-days N] [--root DIR ...] [--ns-prefix PREFIX|--here] [--all] [--include-subagents] [--max N]")
	fmt.Fprintln(w, "       fak session-audit audit    [--since-days N] [--root DIR ...] [--ns-prefix PREFIX|--here] [--all] [--include-subagents] [--max N] [--json OUT] [--md OUT]")
	fmt.Fprintln(w, "       fak session-audit deep <session.jsonl>")
}

type rootFlags []string

func (r *rootFlags) String() string { return fmt.Sprint([]string(*r)) }
func (r *rootFlags) Set(v string) error {
	*r = append(*r, v)
	return nil
}

func sessionAuditCommonFlags(name string, stderr io.Writer) (*flag.FlagSet, *rootFlags, *float64, *string, *bool, *bool, *bool, *int) {
	fs := flag.NewFlagSet(name, flag.ContinueOnError)
	fs.SetOutput(stderr)
	roots := rootFlags{}
	sinceDays := fs.Float64("since-days", -1, "only include transcripts modified within N days")
	nsPrefix := fs.String("ns-prefix", sessionaudit.NamespaceIncludePrefix, "namespace prefix filter")
	here := fs.Bool("here", false, "use the current working directory's Claude projects namespace as --ns-prefix")
	allNS := fs.Bool("all", false, "include all non-excluded namespaces")
	includeSubagents := fs.Bool("include-subagents", false, "include subagent/workflow transcripts")
	max := fs.Int("max", 0, "maximum transcripts to read or render")
	fs.Var(&roots, "root", "transcript projects root (repeatable)")
	return fs, &roots, sinceDays, nsPrefix, here, allNS, includeSubagents, max
}

func discoverOptions(roots rootFlags, sinceDays float64, nsPrefix string, here bool, allNS bool, includeSubagents bool) sessionaudit.DiscoverOptions {
	var since *float64
	if sinceDays >= 0 {
		v := sinceDays
		since = &v
	}
	if allNS {
		nsPrefix = ""
	} else if here && nsPrefix == "" {
		if cwd, err := os.Getwd(); err == nil {
			nsPrefix = sessionaudit.ProjectNamespace(cwd)
		}
	}
	return sessionaudit.DiscoverOptions{
		Roots:            []string(roots),
		SinceDays:        since,
		NamespacePrefix:  nsPrefix,
		IncludeSubagents: includeSubagents,
	}
}

func runSessionAuditDiscover(stdout, stderr io.Writer, argv []string) int {
	fs, roots, sinceDays, nsPrefix, here, allNS, includeSubagents, max := sessionAuditCommonFlags("session-audit discover", stderr)
	asJSON := fs.Bool("json", false, "emit discovered transcript records as JSON")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	if fs.NArg() != 0 {
		fs.Usage()
		return 2
	}
	recs, err := sessionaudit.Discover(discoverOptions(*roots, *sinceDays, *nsPrefix, *here, *allNS, *includeSubagents))
	if err != nil {
		fmt.Fprintf(stderr, "fak session-audit discover: %v\n", err)
		return 1
	}
	total := len(recs)
	if *max > 0 && len(recs) > *max {
		recs = recs[:*max]
	}
	if *asJSON {
		if err := sessionaudit.WriteJSON(stdout, recs); err != nil {
			fmt.Fprintf(stderr, "fak session-audit discover: encode json: %v\n", err)
			return 1
		}
		return 0
	}
	if *max > 0 && total > *max {
		fmt.Fprintf(stdout, "%d sessions (showing first %d of %d; use --ns-prefix or --here to target a namespace before --max)\n", len(recs), len(recs), total)
	} else {
		fmt.Fprintf(stdout, "%d sessions\n", len(recs))
	}
	for _, r := range recs {
		mt := time.Unix(0, int64(r.MTime*1e9)).Format("2006-01-02T15:04:05")
		fmt.Fprintf(stdout, "  %s  %6dKB  %s/%s", mt, r.Size/1024, r.NS, filepath.Base(r.Path))
		if r.Kind != "session" {
			fmt.Fprintf(stdout, "  [%s]", r.Kind)
		}
		fmt.Fprintln(stdout)
	}
	return 0
}

func runSessionAuditAudit(stdout, stderr io.Writer, argv []string) int {
	fs, roots, sinceDays, nsPrefix, here, allNS, includeSubagents, max := sessionAuditCommonFlags("session-audit audit", stderr)
	jsonOut := fs.String("json", "", "write JSON payload to OUT")
	mdOut := fs.String("md", "", "write markdown report to OUT")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	if fs.NArg() != 0 {
		fs.Usage()
		return 2
	}
	opts := discoverOptions(*roots, *sinceDays, *nsPrefix, *here, *allNS, *includeSubagents)
	recs, err := sessionaudit.Discover(opts)
	if err != nil {
		fmt.Fprintf(stderr, "fak session-audit audit: discover: %v\n", err)
		return 1
	}
	totalDiscovered := len(recs)
	if *max > 0 && len(recs) > *max {
		recs = recs[:*max]
	}
	fmt.Fprintf(stderr, "analyzing %d transcripts ...\n", len(recs))
	if *max > 0 && totalDiscovered > *max {
		if opts.NamespacePrefix != "" {
			fmt.Fprintf(stderr, "warning: --max clipped scoped discovery to first %d of %d transcripts; raise --max before treating older sessions inside this scope as absent\n", len(recs), totalDiscovered)
		} else {
			fmt.Fprintf(stderr, "warning: --max clipped discovery to first %d of %d transcripts; use --ns-prefix or --here to target a namespace before --max\n", len(recs), totalDiscovered)
		}
	}
	sessions := make([]sessionaudit.Session, 0, len(recs))
	for _, rec := range recs {
		s := sessionaudit.Analyze(rec.Path)
		s.Kind = rec.Kind
		sessions = append(sessions, s)
	}
	var top, subs []sessionaudit.Session
	for _, s := range sessions {
		if s.Kind == "subagent" {
			subs = append(subs, s)
		} else {
			top = append(top, s)
		}
	}
	agg := sessionaudit.AggregateSessions(top)
	var excluded *sessionaudit.Summary
	if !*includeSubagents {
		allOpts := opts
		allOpts.IncludeSubagents = true
		allRecs, err := sessionaudit.Discover(allOpts)
		if err != nil {
			fmt.Fprintf(stderr, "fak session-audit audit: subagent scan: %v\n", err)
			return 1
		}
		var subRecs []sessionaudit.Transcript
		for _, rec := range allRecs {
			if rec.Kind == "subagent" {
				subRecs = append(subRecs, rec)
			}
		}
		if len(subRecs) > 0 {
			sum := sessionaudit.SummarizeTranscripts(subRecs)
			excluded = &sum
		}
	}
	reportSince := opts.SinceDays
	reportNS := opts.NamespacePrefix
	md := sessionaudit.ReportMarkdown(top, agg, reportNS, reportSince, *includeSubagents, *max, totalDiscovered, excluded, time.Now())
	payload := sessionaudit.AuditPayload{
		Aggregate:         agg,
		ExcludedSubagents: excluded,
		Sessions:          sessions,
	}
	if len(subs) > 0 {
		sum := sessionaudit.SummarizeAnalyses(subs)
		payload.SubagentSummary = &sum
		payload.SubagentTranscript = len(subs)
		md += fmt.Sprintf("\n## Subagent / workflow spend (SEPARATE transcripts, usually uncounted)\n\n- **%d subagent transcripts**\n- Output tokens: %d  .  Cache-read: %d  .  Cache-creation: %d  .  Fresh input: %d\n- Est. cost: $%.2f  _(assumed pricing)_\n",
			len(subs), sum.Tokens.Output, sum.Tokens.CacheRead, sum.Tokens.CacheCreate, sum.Tokens.Input, sum.CostUSD)
	}
	if *mdOut != "" {
		if err := os.WriteFile(*mdOut, []byte(md), 0o666); err != nil {
			fmt.Fprintf(stderr, "fak session-audit audit: write md: %v\n", err)
			return 1
		}
		fmt.Fprintf(stderr, "wrote %s\n", *mdOut)
	}
	if *jsonOut != "" {
		f, err := os.Create(*jsonOut)
		if err != nil {
			fmt.Fprintf(stderr, "fak session-audit audit: write json: %v\n", err)
			return 1
		}
		err = sessionaudit.WriteJSON(f, payload)
		closeErr := f.Close()
		if err != nil {
			fmt.Fprintf(stderr, "fak session-audit audit: encode json: %v\n", err)
			return 1
		}
		if closeErr != nil {
			fmt.Fprintf(stderr, "fak session-audit audit: close json: %v\n", closeErr)
			return 1
		}
		fmt.Fprintf(stderr, "wrote %s\n", *jsonOut)
	}
	fmt.Fprint(stdout, md)
	return 0
}

func runSessionAuditDeep(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("session-audit deep", flag.ContinueOnError)
	fs.SetOutput(stderr)
	asJSON := fs.Bool("json", false, "emit analysis JSON")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	if fs.NArg() != 1 {
		fmt.Fprintln(stderr, "usage: fak session-audit deep [--json] <session.jsonl>")
		return 2
	}
	s := sessionaudit.Analyze(fs.Arg(0))
	if s.Error != "" {
		fmt.Fprintf(stderr, "fak session-audit deep: %v\n", s.Error)
		return 1
	}
	if *asJSON {
		if err := sessionaudit.WriteJSON(stdout, s); err != nil {
			fmt.Fprintf(stderr, "fak session-audit deep: encode json: %v\n", err)
			return 1
		}
		return 0
	}
	fmt.Fprint(stdout, sessionaudit.DeepMarkdown(s))
	return 0
}
