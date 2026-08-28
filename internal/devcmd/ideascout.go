package devcmd

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"time"

	"github.com/anthony-chaudhary/fak/internal/ideascout"
)

func RunIdeaScout(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("idea-scout", flag.ContinueOnError)
	fs.SetOutput(stderr)
	workspace := fs.String("workspace", ".", "repo root holding the .idea-scout cache")
	configPath := fs.String("config", "", "JSON file overriding topics/thresholds")
	maxIssues := fs.Int("max-issues", 0, "hard cap on issues filed")
	minScore := fs.Int("min-score", 0, "drop candidates below this")
	// BLAST RADIUS. --live is the only flag here with an effect outside this process:
	// it runs `gh issue create` against the CURRENT repo's REAL GitHub tracker, once per
	// kept candidate (up to --max-issues, default from the config thresholds), labels each
	// `idea-scout`+`research`, creates the `idea-scout` label if absent, optionally files
	// them into --milestone / --project, and records each filed source ID in
	// <workspace>/.idea-scout/seen.json. Those issues are public and are not rolled back
	// on a later error. Everything else -- the default dry-run, --json, and the
	// --candidates/--issues/--scout-issues fixture replay -- reads only and writes nothing.
	live := fs.Bool("live", false, "FILE REAL GITHUB ISSUES: run gh issue create for each kept candidate in the current repo (up to --max-issues) and record them in the seen-cache. Omit it and the run is a dry-run that mutates nothing")
	asJSON := fs.Bool("json", false, "emit machine-readable output")
	milestone := fs.String("milestone", "", "assign filed issues to this milestone title")
	project := fs.String("project", "", "ProjectsV2 number to add filed issues to")
	projectOwner := fs.String("project-owner", "", "owner login for --project")
	candidatesPath := fs.String("candidates", "", "fixture candidates JSON; skips live source fetching")
	issuesPath := fs.String("issues", "", "fixture existing issues JSON used with --candidates")
	scoutIssuesPath := fs.String("scout-issues", "", "fixture idea-scout-labelled issues JSON: the durable filed-stamp index (defaults to --issues)")
	today := fs.String("today", "", "override the report date (YYYY-MM-DD), primarily for tests")
	fs.Usage = func() { fmt.Fprint(stderr, ideaScoutUsage) }
	if !parseFlags(fs, argv) {
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
	if *scoutIssuesPath != "" {
		scoutIssues, err := ideascout.ReadExistingIssues(*scoutIssuesPath)
		if err != nil {
			fmt.Fprintf(stderr, "fak idea-scout: read --scout-issues: %v\n", err)
			return 2
		}
		opts.ScoutIssues = scoutIssues
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

const ideaScoutUsage = `fak idea-scout - surface related arXiv/GitHub/Hacker News/Reddit ideas as deduped issue plans.

usage:
  fak idea-scout [--json] [--workspace DIR] [--config FILE]
                 [--max-issues N] [--min-score N]
                 [--candidates FILE] [--issues FILE] [--scout-issues FILE]
                 [--live] [--milestone TITLE] [--project N] [--project-owner OWNER]

Dry-run is the default and mutates nothing. --live creates issues through gh issue
create and records filed source IDs in .idea-scout/seen.json. --candidates supplies
fixture candidates and skips live arXiv/GitHub/Hacker News/Reddit fetching for tests or replay.

Never filing the same source twice rests on the filed-stamp rung: the source ID
stamped into every issue the scout has ever filed, read back through a query
TARGETED at the idea-scout label rather than a window of recent issues. That index
is mandatory - if gh cannot build it, or it comes back saturated at
thresholds.scout_scan_limit (so it may be truncated), the run REFUSES with exit 2
rather than risk re-filing an already-triaged source. The seen-cache and the
issue_scan_limit recent-issue window are a fast path and a bonus catch; neither
carries the guarantee, and losing the cache does not weaken it.

Sources per topic (any subset): arxiv query, github search, hn (Hacker News Algolia
query) and reddit (Reddit search, sort=new) — the community feeds are newest-first,
catching trending items within moments of release.

GitHub is walked on two lanes from the same topic query: a stars lane (all-time
popular, floored at min_stars) and a fresh lane (fresh_per_topic repos sorted by
most-recently-updated, floored at the lower fresh_min_stars) so newly-created,
trending, and recently-pushed repos surface instead of only established incumbents.
Set fresh_per_topic: 0 in --config thresholds to disable the fresh lane.
`
