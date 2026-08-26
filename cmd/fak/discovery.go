package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"path/filepath"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/devindex"
	"github.com/anthony-chaudhary/fak/internal/discoveryrouter"
	"github.com/anthony-chaudhary/fak/internal/fleetsearch"
	"github.com/anthony-chaudhary/fak/internal/sessionjournal"
	"github.com/anthony-chaudhary/fak/internal/sessionregistry"
)

func runDiscovery(stdout, stderr io.Writer, args []string) int {
	fs := flag.NewFlagSet("search repository", flag.ContinueOnError)
	fs.SetOutput(stderr)
	root := fs.String("root", ".", "repository root")
	limit := fs.Int("limit", 10, "maximum combined results")
	asJSON := fs.Bool("json", false, "emit fak.discovery/1 JSON")
	omitSessions := fs.Bool("skip-sessions", false, "skip runtime session stores and report partial coverage")
	if err := fs.Parse(args); err != nil {
		return 2
	}
	if fs.NArg() != 1 || strings.TrimSpace(fs.Arg(0)) == "" {
		fmt.Fprintln(stderr, "fak search repository: exactly one QUERY is required")
		return 2
	}
	cat, err := devindex.Load(*root)
	if err != nil {
		fmt.Fprintf(stderr, "fak search repository: load docs: %v\n", err)
		return 1
	}
	plan := discoveryrouter.Plan{Adapters: []discoveryrouter.Adapter{
		discoveryrouter.DocsAdapter{Catalog: cat, Revision: "working-tree"},
		discoveryrouter.FleetAdapter{Config: fleetsearch.Config{LifecyclePath: sessionjournal.DefaultPath(), RegistrationPath: sessionregistry.DefaultPath(), ToolProcessPath: filepath.Join(*root, ".fak", "toolproc", "journal.jsonl")}, Watermark: "live-store"},
	}}
	skip := map[string]bool{}
	if *omitSessions {
		skip["sessions"] = true
	}
	report := plan.Run(fs.Arg(0), *limit, skip)
	if *asJSON {
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		if err := enc.Encode(report); err != nil {
			return 1
		}
		return 0
	}
	fmt.Fprintf(stdout, "DISCOVERY coverage_complete=%t results=%d query=%q\n", report.CoverageComplete, len(report.Results), report.Query)
	for _, c := range report.Coverage {
		fmt.Fprintf(stdout, "SOURCE %s %s %s\n", c.Source, c.Status, c.Reason)
	}
	for i, hit := range report.Results {
		fmt.Fprintf(stdout, "%d. %s %s score=%d reason=%s\n", i+1, hit.Source, hit.Owner, hit.Score, hit.Reason)
	}
	return 0
}
