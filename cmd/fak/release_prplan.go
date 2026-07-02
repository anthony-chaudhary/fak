package main

// fak release prplan folds the promotion range (release branch .. release
// source) into PR-sized units grouped by the (fak <leaf>) ship-stamp, so a
// dev->main promotion can open as human-legible PRs whose bodies were managed
// in advance by the existing commit discipline. The fold is deterministic over
// git history — there is no plan file to go stale: every stamped commit is
// already a line item in the PR unit of the lane that owns it. --check turns
// the legibility invariant (no unstamped commits in the promotion range) into
// a gate a CI job or a pre-promotion hook can run.

import (
	"flag"
	"fmt"
	"io"
	"regexp"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/branchrole"
)

const releasePRPlanSchema = "fak.release.prplan.v1"

// releasePRPlanGit is the one git seam; tests override it.
var releasePRPlanGit = releaseStatusGitOutput

var (
	prPlanLeafRE  = regexp.MustCompile(`\(fak ([a-z0-9][a-z0-9-]*)\)\s*$`)
	prPlanTypeRE  = regexp.MustCompile(`^([a-z]+)[(!:]`)
	prPlanIssueRE = regexp.MustCompile(`#(\d+)\b`)
)

type prPlanCommit struct {
	SHA      string   `json:"sha"`
	Subject  string   `json:"subject"`
	Leaf     string   `json:"leaf,omitempty"`
	Type     string   `json:"type,omitempty"`
	Resolves []string `json:"resolves,omitempty"` // #N bound in the subject (closure-grade)
	Mentions []string `json:"mentions,omitempty"` // #N only in the body (safe mention)
	Files    []string `json:"files,omitempty"`
}

type prPlanUnit struct {
	Leaf     string         `json:"leaf"`
	Title    string         `json:"title"`
	Commits  []prPlanCommit `json:"commits"`
	Types    map[string]int `json:"types"`
	Resolves []string       `json:"resolves,omitempty"`
	Mentions []string       `json:"mentions,omitempty"`
	Files    []string       `json:"files"`
}

type prPlanOptions struct {
	AsJSON   bool
	Base     string
	Head     string
	Check    bool
	MaxFiles int
}

func runReleasePRPlan(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak release prplan", flag.ContinueOnError)
	fs.SetOutput(stderr)
	asJSON := fs.Bool("json", false, "emit machine-readable JSON")
	base := fs.String("base", "", "promotion base ref (default: origin/<release_branch>)")
	head := fs.String("head", "", "promotion head ref (default: <release_source> tip)")
	check := fs.Bool("check", false, "exit 1 if the range holds commits without a (fak <leaf>) ship-stamp")
	maxFiles := fs.Int("max-files", 20, "file paths listed per unit before folding to a count")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintf(stderr, "fak release prplan: unexpected argument %q\n", fs.Arg(0))
		return 2
	}
	if *maxFiles < 0 {
		fmt.Fprintln(stderr, "fak release prplan: --max-files must be >= 0")
		return 2
	}
	opts := prPlanOptions{AsJSON: *asJSON, Base: *base, Head: *head, Check: *check, MaxFiles: *maxFiles}

	root := repoRoot()
	plan, err := buildPRPlan(root, opts)
	if err != nil {
		fmt.Fprintf(stderr, "fak release prplan: %v\n", err)
		return 1
	}
	if opts.AsJSON {
		if err := writeIndentedJSON(stdout, plan); err != nil {
			fmt.Fprintf(stderr, "fak release prplan: encode json: %v\n", err)
			return 1
		}
	} else {
		fmt.Fprintln(stdout, renderPRPlanMarkdown(plan, opts.MaxFiles))
	}
	if opts.Check && !releaseStatusBool(plan["check_ok"]) {
		fmt.Fprintf(stderr, "fak release prplan: %d commit(s) in %s lack a (fak <leaf>) ship-stamp; stamp them so the promotion PR stays legible\n", releaseStatusInt(plan["unstamped_count"]), releaseStatusString(plan["range"]))
		return 1
	}
	return 0
}

func buildPRPlan(root string, opts prPlanOptions) (map[string]any, error) {
	roles, _ := branchrole.Load(root)
	baseRef, baseSHA, err := prPlanResolve(root, opts.Base, []string{"origin/" + roles.ReleaseBranch, roles.ReleaseBranch})
	if err != nil {
		return nil, fmt.Errorf("resolve base: %w", err)
	}
	headRef, headSHA, err := prPlanResolve(root, opts.Head, []string{roles.ReleaseSource, "origin/" + roles.ReleaseSource})
	if err != nil {
		return nil, fmt.Errorf("resolve head: %w", err)
	}
	var commits []prPlanCommit
	if baseSHA != headSHA {
		raw := releasePRPlanGit(root, "log", "--no-merges", "--name-only",
			"--format=%x1e%H%x1f%s%x1f%b%x1f", baseSHA+".."+headSHA)
		commits = parsePRPlanLog(raw)
	}
	units, unstamped := foldPRPlanUnits(commits)
	return map[string]any{
		"schema":             releasePRPlanSchema,
		"base":               baseRef,
		"base_sha":           baseSHA,
		"head":               headRef,
		"head_sha":           headSHA,
		"range":              baseRef + ".." + headRef,
		"development_branch": roles.DevelopmentBranch,
		"release_branch":     roles.ReleaseBranch,
		"release_source":     roles.ReleaseSource,
		"commit_count":       len(commits),
		"unit_count":         len(units),
		"unstamped_count":    len(unstamped),
		"units":              units,
		"unstamped":          unstamped,
		"check_ok":           len(unstamped) == 0,
	}, nil
}

// prPlanResolve resolves an explicit ref, or the first resolvable candidate.
// It returns the human ref name alongside the SHA so output stays readable.
func prPlanResolve(root, explicit string, candidates []string) (string, string, error) {
	if strings.TrimSpace(explicit) != "" {
		sha := prPlanRevParse(root, explicit)
		if sha == "" {
			return "", "", fmt.Errorf("ref %q does not resolve to a commit", explicit)
		}
		return explicit, sha, nil
	}
	for _, ref := range candidates {
		if strings.TrimSpace(ref) == "" || strings.TrimSpace(ref) == "origin/" {
			continue
		}
		if sha := prPlanRevParse(root, ref); sha != "" {
			return ref, sha, nil
		}
	}
	return "", "", fmt.Errorf("none of %v resolve to a commit", candidates)
}

func prPlanRevParse(root, ref string) string {
	out := releasePRPlanGit(root, "rev-parse", "--verify", "--quiet", ref+"^{commit}")
	if out == "" {
		return ""
	}
	return strings.Fields(out)[0]
}

// parsePRPlanLog parses `git log --no-merges --name-only
// --format=%x1e%H%x1f%s%x1f%b%x1f` output: records split on \x1e, fields on
// \x1f, with the touched-file list trailing the final field separator.
func parsePRPlanLog(raw string) []prPlanCommit {
	var commits []prPlanCommit
	for _, record := range strings.Split(raw, "\x1e") {
		if strings.TrimSpace(record) == "" {
			continue
		}
		fields := strings.SplitN(record, "\x1f", 4)
		if len(fields) < 4 {
			continue
		}
		sha := strings.TrimSpace(fields[0])
		subject := strings.TrimSpace(fields[1])
		body := fields[2]
		if sha == "" || subject == "" {
			continue
		}
		var files []string
		for _, line := range strings.Split(fields[3], "\n") {
			if line = strings.TrimSpace(line); line != "" {
				files = append(files, line)
			}
		}
		leaf := ""
		if m := prPlanLeafRE.FindStringSubmatch(subject); m != nil {
			leaf = m[1]
		}
		typ := ""
		if m := prPlanTypeRE.FindStringSubmatch(subject); m != nil {
			typ = m[1]
		}
		resolves := prPlanIssues(subject, nil)
		mentions := prPlanIssues(body, resolves)
		commits = append(commits, prPlanCommit{
			SHA: sha, Subject: subject, Leaf: leaf, Type: typ,
			Resolves: resolves, Mentions: mentions, Files: files,
		})
	}
	return commits
}

// prPlanIssues extracts deduplicated #N refs from text, excluding any already
// present in exclude (subject-bound refs outrank body mentions).
func prPlanIssues(text string, exclude []string) []string {
	seen := map[string]bool{}
	for _, ref := range exclude {
		seen[ref] = true
	}
	var out []string
	for _, m := range prPlanIssueRE.FindAllStringSubmatch(text, -1) {
		ref := "#" + m[1]
		if !seen[ref] {
			seen[ref] = true
			out = append(out, ref)
		}
	}
	sort.Strings(out)
	return out
}

// foldPRPlanUnits groups commits into one PR unit per (fak <leaf>) lane.
// Commits without a stamp are returned separately: they are the legibility
// debt --check gates on. Units are ordered biggest-first, then by leaf; the
// commits inside each unit read oldest-first, the way a PR body should.
func foldPRPlanUnits(commits []prPlanCommit) ([]prPlanUnit, []prPlanCommit) {
	byLeaf := map[string]*prPlanUnit{}
	var unstamped []prPlanCommit
	for _, c := range commits {
		if c.Leaf == "" {
			unstamped = append(unstamped, c)
			continue
		}
		unit, ok := byLeaf[c.Leaf]
		if !ok {
			unit = &prPlanUnit{Leaf: c.Leaf, Types: map[string]int{}}
			byLeaf[c.Leaf] = unit
		}
		unit.Commits = append(unit.Commits, c)
		if c.Type != "" {
			unit.Types[c.Type]++
		}
		unit.Resolves = prPlanMergeRefs(unit.Resolves, c.Resolves)
		unit.Mentions = prPlanMergeRefs(unit.Mentions, c.Mentions)
		unit.Files = prPlanMergeRefs(unit.Files, c.Files)
	}
	units := make([]prPlanUnit, 0, len(byLeaf))
	for _, unit := range byLeaf {
		// git log yields newest-first; a PR body reads oldest-first.
		for i, j := 0, len(unit.Commits)-1; i < j; i, j = i+1, j-1 {
			unit.Commits[i], unit.Commits[j] = unit.Commits[j], unit.Commits[i]
		}
		// A body mention that some commit subject-binds is already a closure.
		unit.Mentions = prPlanSubtractRefs(unit.Mentions, unit.Resolves)
		unit.Title = prPlanUnitTitle(*unit)
		units = append(units, *unit)
	}
	sort.Slice(units, func(i, j int) bool {
		if len(units[i].Commits) != len(units[j].Commits) {
			return len(units[i].Commits) > len(units[j].Commits)
		}
		return units[i].Leaf < units[j].Leaf
	})
	return units, unstamped
}

func prPlanMergeRefs(have, add []string) []string {
	seen := map[string]bool{}
	for _, v := range have {
		seen[v] = true
	}
	for _, v := range add {
		if !seen[v] {
			seen[v] = true
			have = append(have, v)
		}
	}
	sort.Strings(have)
	return have
}

func prPlanSubtractRefs(from, drop []string) []string {
	gone := map[string]bool{}
	for _, v := range drop {
		gone[v] = true
	}
	var out []string
	for _, v := range from {
		if !gone[v] {
			out = append(out, v)
		}
	}
	return out
}

func prPlanUnitTitle(unit prPlanUnit) string {
	if len(unit.Commits) == 1 {
		return unit.Commits[0].Subject
	}
	types := make([]string, 0, len(unit.Types))
	for t := range unit.Types {
		types = append(types, t)
	}
	sort.Slice(types, func(i, j int) bool {
		if unit.Types[types[i]] != unit.Types[types[j]] {
			return unit.Types[types[i]] > unit.Types[types[j]]
		}
		return types[i] < types[j]
	})
	parts := make([]string, 0, len(types))
	for _, t := range types {
		parts = append(parts, fmt.Sprintf("%s %d", t, unit.Types[t]))
	}
	detail := strings.Join(parts, ", ")
	if detail == "" {
		detail = "mixed"
	}
	return fmt.Sprintf("%s: %d commits (%s)", unit.Leaf, len(unit.Commits), detail)
}

func renderPRPlanMarkdown(plan map[string]any, maxFiles int) string {
	var b strings.Builder
	fmt.Fprintf(&b, "# Promotion PR plan — %s\n\n", releaseStatusString(plan["range"]))
	commitCount := releaseStatusInt(plan["commit_count"])
	if commitCount == 0 {
		fmt.Fprintf(&b, "The promotion range is empty: %s and %s point at the same history (base %s, head %s).\n",
			releaseStatusString(plan["base"]), releaseStatusString(plan["head"]),
			releaseStatusShortSHA(releaseStatusString(plan["base_sha"])), releaseStatusShortSHA(releaseStatusString(plan["head_sha"])))
		return strings.TrimRight(b.String(), "\n")
	}
	units, _ := plan["units"].([]prPlanUnit)
	unstamped, _ := plan["unstamped"].([]prPlanCommit)
	fmt.Fprintf(&b, "%d commit(s) across %d lane unit(s); base %s → head %s.\n",
		commitCount, len(units),
		releaseStatusShortSHA(releaseStatusString(plan["base_sha"])), releaseStatusShortSHA(releaseStatusString(plan["head_sha"])))
	b.WriteString("Each section below is a ready PR body; a single-PR promotion can use this whole document.\n")
	b.WriteString("The plan is managed in advance by the `(fak <leaf>)` ship-stamp discipline — regenerate any time with `fak release prplan`.\n")
	for _, unit := range units {
		fmt.Fprintf(&b, "\n## %s — %d commit(s)\n\n", unit.Leaf, len(unit.Commits))
		fmt.Fprintf(&b, "**Title:** `%s`\n", unit.Title)
		if len(unit.Resolves) > 0 {
			fmt.Fprintf(&b, "Closes %s.\n", strings.Join(unit.Resolves, ", "))
		}
		if len(unit.Mentions) > 0 {
			fmt.Fprintf(&b, "Mentions %s.\n", strings.Join(unit.Mentions, ", "))
		}
		b.WriteString("\n")
		for _, c := range unit.Commits {
			fmt.Fprintf(&b, "- `%s` %s\n", releaseStatusShortSHA(c.SHA), c.Subject)
		}
		b.WriteString("\n")
		fmt.Fprintf(&b, "Files touched (%d): %s\n", len(unit.Files), prPlanFileList(unit.Files, maxFiles))
	}
	if len(unstamped) > 0 {
		fmt.Fprintf(&b, "\n## ⚠ unstamped — %d commit(s) with no `(fak <leaf>)` ship-stamp\n\n", len(unstamped))
		b.WriteString("These commits cannot be routed to a lane unit; stamp future commits so the promotion PR stays legible.\n\n")
		for _, c := range unstamped {
			fmt.Fprintf(&b, "- `%s` %s\n", releaseStatusShortSHA(c.SHA), c.Subject)
		}
	}
	return strings.TrimRight(b.String(), "\n")
}

func prPlanFileList(files []string, maxFiles int) string {
	if len(files) == 0 {
		return "(none recorded)"
	}
	if maxFiles == 0 || len(files) <= maxFiles {
		return strings.Join(files, ", ")
	}
	return strings.Join(files[:maxFiles], ", ") + fmt.Sprintf(" (+%d more)", len(files)-maxFiles)
}
