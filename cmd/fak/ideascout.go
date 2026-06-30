package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"time"

	"github.com/anthony-chaudhary/fak/internal/ideascout"
)

func cmdIdeaScout(argv []string) { os.Exit(runIdeaScout(os.Stdout, os.Stderr, argv)) }

func runIdeaScout(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("idea-scout", flag.ContinueOnError)
	fs.SetOutput(stderr)
	workspace := fs.String("workspace", ".", "repo root holding the .idea-scout cache")
	configPath := fs.String("config", "", "JSON file overriding topics/thresholds")
	maxIssues := fs.Int("max-issues", 0, "hard cap on issues filed")
	minScore := fs.Int("min-score", 0, "drop candidates below this")
	live := fs.Bool("live", false, "actually create issues and record them in the seen-cache")
	asJSON := fs.Bool("json", false, "emit machine-readable output")
	milestone := fs.String("milestone", "", "assign filed issues to this milestone title")
	project := fs.String("project", "", "ProjectsV2 number to add filed issues to")
	projectOwner := fs.String("project-owner", "", "owner login for --project")
	candidatesPath := fs.String("candidates", "", "fixture candidates JSON; skips live source fetching")
	issuesPath := fs.String("issues", "", "fixture existing issues JSON used with --candidates")
	today := fs.String("today", "", "override the report date (YYYY-MM-DD), primarily for tests")
	fs.Usage = func() { fmt.Fprint(stderr, ideaScoutUsage) }
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	seenFlag := map[string]bool{}
	fs.Visit(func(f *flag.Flag) { seenFlag[f.Name] = true })

	opts := ideascout.RunOptions{
		Workspace:    *workspace,
		ConfigPath:   *configPath,
		Live:         *live,
		JSON:         *asJSON,
		Today:        *today,
		Now:          time.Now().UTC(),
		UseFixtures:  *candidatesPath != "",
		ProjectOwner: optionalString(seenFlag["project-owner"], *projectOwner),
		Project:      optionalString(seenFlag["project"], *project),
		Milestone:    optionalString(seenFlag["milestone"], *milestone),
		MaxIssues:    optionalInt(seenFlag["max-issues"], *maxIssues),
		MinScore:     optionalInt(seenFlag["min-score"], *minScore),
	}
	if *candidatesPath != "" {
		cands, err := ideascout.ReadCandidates(*candidatesPath)
		if err != nil {
			fmt.Fprintf(stderr, "fak idea-scout: read --candidates: %v\n", err)
			return 2
		}
		opts.Candidates = cands
	}
	if *issuesPath != "" {
		issues, err := ideascout.ReadExistingIssues(*issuesPath)
		if err != nil {
			fmt.Fprintf(stderr, "fak idea-scout: read --issues: %v\n", err)
			return 2
		}
		opts.Existing = issues
	}

	result, err := ideascout.Run(opts)
	if err != nil {
		fmt.Fprintf(stderr, "fak idea-scout: %v\n", err)
		return 2
	}
	if *asJSON {
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		if err := enc.Encode(result); err != nil {
			fmt.Fprintf(stderr, "fak idea-scout: write JSON: %v\n", err)
			return 1
		}
		return 0
	}
	cfg, err := ideascout.ResultConfig(*configPath, opts.MaxIssues, opts.MinScore, opts.Milestone, opts.Project, opts.ProjectOwner)
	if err != nil {
		fmt.Fprintf(stderr, "fak idea-scout: config error: %v\n", err)
		return 2
	}
	ideascout.RenderHuman(stdout, result, cfg)
	return 0
}

func optionalInt(set bool, v int) *int {
	if !set {
		return nil
	}
	return &v
}

func optionalString(set bool, v string) *string {
	if !set {
		return nil
	}
	return &v
}

const ideaScoutUsage = `fak idea-scout - surface related arXiv/GitHub ideas as deduped issue plans.

usage:
  fak idea-scout [--json] [--workspace DIR] [--config FILE]
                 [--max-issues N] [--min-score N]
                 [--candidates FILE] [--issues FILE]
                 [--live] [--milestone TITLE] [--project N] [--project-owner OWNER]

Dry-run is the default and mutates nothing. --live creates issues through gh issue
create and records filed source IDs in .idea-scout/seen.json. --candidates supplies
fixture candidates and skips live arXiv/GitHub fetching for tests or replay.
`
