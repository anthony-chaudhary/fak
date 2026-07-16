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
	"strings"

	"github.com/anthony-chaudhary/fak/internal/branchrole"
	"github.com/anthony-chaudhary/fak/internal/steerpr"
)

const releasePRPlanSchema = "fak.release.prplan.v1"

// releasePRPlanGit is the one git seam; tests override it.
var releasePRPlanGit = releaseStatusGitOutput

// The promotion plan's commit/unit vocabulary is the shared overlay fold's
// (internal/steerpr), kept under the release-time names these call sites read
// by. Aliases rather than wrapper structs: the two names denote ONE type, so a
// caller written against either compiles and no conversion can drift.
type (
	prPlanCommit = steerpr.Commit
	prPlanUnit   = steerpr.Unit
)

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
	if !parseFlags(fs, argv) {
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
// --format=%x1e%H%x1f%s%x1f%b%x1f` output into the shared overlay fold's
// commits. The promotion range and the continuous operator view read the same
// git format, so they parse it through the same code.
func parsePRPlanLog(raw string) []prPlanCommit { return steerpr.ParseLog(raw) }

// foldPRPlanUnits groups commits into one PR unit per (fak <leaf>) lane through
// the shared overlay fold. Commits without a stamp are returned separately:
// they are the legibility debt --check gates on. Units are ordered
// biggest-first, then by leaf; the commits inside each unit read oldest-first,
// the way a PR body should.
//
// The fold also bands each unit by where operator attention is owed. The
// promotion plan supplies no witness verdicts, so every unit here bands
// UNVERIFIABLE ("not yet graded") — honest, and inert for this caller, which
// renders the plan and never reads the band. The continuous operator view is
// the caller that supplies verdicts and reads them.
func foldPRPlanUnits(commits []prPlanCommit) ([]prPlanUnit, []prPlanCommit) {
	return steerpr.FoldUnits(commits)
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
