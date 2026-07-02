package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"time"

	"github.com/anthony-chaudhary/fak/internal/issuecontractrepair"
)

func cmdIssueContractRepair(argv []string) {
	os.Exit(runIssueContractRepair(os.Stdout, os.Stderr, argv))
}

func runIssueContractRepair(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("issue-contract-repair", flag.ContinueOnError)
	fs.SetOutput(stderr)
	workspace := fs.String("workspace", "", "repo root (default: repo root)")
	lane := fs.String("lane", "", "restrict to one dispatch lane")
	limit := fs.Int("limit", 50, "max issues to examine, oldest issue number first")
	asJSON := fs.Bool("json", false, "emit JSON manifest")
	markdown := fs.Bool("markdown", false, "emit markdown manifest")
	actions := fs.Bool("actions", false, "emit review-only action list")
	out := fs.String("out", "", "write output to this path")
	asOf := fs.String("as-of", "", "date stamp (default: today UTC)")
	if code, done := parseFlagsRejectArgs(fs, argv, stderr); done {
		return code
	}
	root := *workspace
	if root == "" {
		root = resolveRoot("")
		if root == "" {
			root = "."
		}
	}
	stamp := *asOf
	if stamp == "" {
		stamp = time.Now().UTC().Format("2006-01-02")
	}
	issues, err := issuecontractrepair.FetchOpenIssues(root, issuecontractrepair.DefaultCap)
	if err != nil {
		fmt.Fprintf(stderr, "issue-contract-repair: %v\n", err)
		return 2
	}
	manifest := issuecontractrepair.BuildManifest(root, issues, issuecontractrepair.Options{
		Lane: *lane, Limit: *limit, AsOf: stamp,
	})
	var rendered string
	switch {
	case *actions:
		body := map[string]any{"as_of": stamp, "actions": issuecontractrepair.BuildActions(manifest)}
		b, err := json.MarshalIndent(body, "", "  ")
		if err != nil {
			fmt.Fprintf(stderr, "issue-contract-repair: encode json: %v\n", err)
			return 2
		}
		rendered = string(b)
	case *markdown:
		rendered = issuecontractrepair.RenderMarkdown(manifest)
	default:
		b, err := json.MarshalIndent(manifest, "", "  ")
		if err != nil {
			fmt.Fprintf(stderr, "issue-contract-repair: encode json: %v\n", err)
			return 2
		}
		rendered = string(b)
		_ = asJSON
	}
	if *out != "" {
		if err := os.MkdirAll(filepath.Dir(*out), 0o755); err != nil && filepath.Dir(*out) != "." {
			fmt.Fprintf(stderr, "issue-contract-repair: %v\n", err)
			return 2
		}
		if err := os.WriteFile(*out, []byte(rendered+"\n"), 0o644); err != nil {
			fmt.Fprintf(stderr, "issue-contract-repair: %v\n", err)
			return 2
		}
		fmt.Fprintf(stdout, "wrote %s\n", *out)
		return 0
	}
	fmt.Fprintln(stdout, rendered)
	return 0
}
