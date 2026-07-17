package main

// fak steer prs folds the PENDING dev->release delta into operator-legible,
// PR-sized units and renders them WORST-ATTENTION-FIRST, so an operator can see
// the coherent units forming on the trunk right now and which of them owe a
// look. It is the continuous, operator-facing twin of `fak release prplan` (the
// release-time promotion plan, ordered biggest-first): same range, same
// (fak <leaf>) ship-stamp fold via internal/steerpr, but ordered
// RESIDUAL -> UNVERIFIABLE -> CLEARED and banded by where operator attention is
// owed.
//
// It is READ-ONLY and gates NOTHING. --check reports a RESIDUAL unit for CI or
// an operator (exit 1), the same shape as `prplan --check` and dos_review's
// has_residual, but it must never sit in a commit or promotion path — the
// overlay's whole thesis is observability without a merge gate.
//
// The band is a VIEW over the kernel's existing witness oracle, never a second
// one: per-commit verdicts come from `dos commit-audit <base>..<head> --json`
// (one call over the whole range) mapped through the SAME keep-bit the dispatch
// sweep uses (dispatchtick.CommitWitnessed), so this view and the sweep can
// never disagree about whether a commit is witnessed. Grading is BEST-EFFORT:
// if dos is unavailable the commits stay VerdictUnknown -> UNVERIFIABLE ("not
// yet graded"), never fabricated as CLEARED.

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"os/exec"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/branchrole"
	"github.com/anthony-chaudhary/fak/internal/dispatchtick"
	"github.com/anthony-chaudhary/fak/internal/steerpr"
)

const steerPRsSchema = steerpr.Schema // "fak.steerpr.v1"

// steerPRsVerdicts grades base..head into per-SHA verdicts. Overridable in tests;
// the default shells `dos commit-audit`.
var steerPRsVerdicts = dosCommitAuditRange

func cmdSteer(argv []string) { os.Exit(runSteer(os.Stdout, os.Stderr, argv)) }

func runSteer(stdout, stderr io.Writer, argv []string) int {
	if len(argv) > 0 {
		switch strings.ToLower(strings.TrimSpace(argv[0])) {
		case "prs":
			return runSteerPRs(stdout, stderr, argv[1:])
		case "-h", "--help", "help":
			fmt.Fprintln(stdout, steerUsage)
			return 0
		}
	}
	fmt.Fprintln(stderr, steerUsage)
	return 2
}

const steerUsage = `fak steer prs — the forming operator PRs on the trunk, worst-attention-first

Usage:
  fak steer prs [--json] [--check] [--base REF] [--head REF] [--max-files N]

Folds the pending dev->release delta into PR-sized units per (fak <leaf>) stamp,
bands each by where attention is owed (RESIDUAL/UNVERIFIABLE/CLEARED), and lists
them worst-first. Read-only; --check reports RESIDUAL, it never gates a merge.`

func runSteerPRs(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak steer prs", flag.ContinueOnError)
	fs.SetOutput(stderr)
	asJSON := fs.Bool("json", false, "emit machine-readable JSON (schema fak.steerpr.v1)")
	base := fs.String("base", "", "range base ref (default: origin/<release_branch>)")
	head := fs.String("head", "", "range head ref (default: <release_source> tip)")
	check := fs.Bool("check", false, "exit 1 if any forming unit is RESIDUAL (reports; never blocks a merge)")
	maxFiles := fs.Int("max-files", 20, "file paths listed per unit before folding to a count")
	if !parseFlags(fs, argv) {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintf(stderr, "fak steer prs: unexpected argument %q\n", fs.Arg(0))
		return 2
	}
	if *maxFiles < 0 {
		fmt.Fprintln(stderr, "fak steer prs: --max-files must be >= 0")
		return 2
	}

	view, err := buildSteerPRsView(repoRoot(), *base, *head)
	if err != nil {
		fmt.Fprintf(stderr, "fak steer prs: %v\n", err)
		return 1
	}
	if *asJSON {
		if err := writeIndentedJSON(stdout, view); err != nil {
			fmt.Fprintf(stderr, "fak steer prs: encode json: %v\n", err)
			return 1
		}
	} else {
		fmt.Fprintln(stdout, writeSteerPRs(view, *maxFiles))
	}
	if *check && releaseStatusInt(view["residual_count"]) > 0 {
		fmt.Fprintf(stderr, "fak steer prs: %d unit(s) in %s are RESIDUAL — a claim the kernel could not witness; a human look buys something here\n",
			releaseStatusInt(view["residual_count"]), releaseStatusString(view["range"]))
		return 1
	}
	return 0
}

// buildSteerPRsView resolves the pending delta, grades it, and folds it into the
// worst-attention-first operator view. It reuses the release-plan range
// resolution (branchrole + prPlanResolve) and git seam so the continuous view
// and the promotion plan always fold the SAME range through the SAME parser.
func buildSteerPRsView(root, base, head string) (map[string]any, error) {
	roles, _ := branchrole.Load(root)
	baseRef, baseSHA, err := prPlanResolve(root, base, []string{"origin/" + roles.ReleaseBranch, roles.ReleaseBranch})
	if err != nil {
		return nil, fmt.Errorf("resolve base: %w", err)
	}
	headRef, headSHA, err := prPlanResolve(root, head, []string{roles.ReleaseSource, "origin/" + roles.ReleaseSource})
	if err != nil {
		return nil, fmt.Errorf("resolve head: %w", err)
	}

	var commits []steerpr.Commit
	if baseSHA != headSHA {
		raw := releasePRPlanGit(root, "log", "--no-merges", "--name-only",
			"--format=%x1e%H%x1f%s%x1f%b%x1f", baseSHA+".."+headSHA)
		commits = steerpr.ParseLog(raw)
		verdicts := steerPRsVerdicts(root, baseSHA, headSHA)
		for i := range commits {
			if v, ok := matchVerdict(commits[i].SHA, verdicts); ok {
				commits[i].Verdict = v
			}
		}
	}

	units, unstamped := steerpr.FoldUnits(commits)
	steerpr.SortWorstFirst(units)
	return map[string]any{
		"schema":             steerPRsSchema,
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
		"residual_count":     steerpr.Residual(units),
		"units":              units,
		"unstamped":          unstamped,
	}, nil
}

// matchVerdict finds a commit's verdict by SHA prefix: `dos commit-audit` returns
// abbreviated SHAs while git log yields full ones, so a stored short SHA that is
// a prefix of the full SHA is the same commit.
func matchVerdict(fullSHA string, verdicts map[string]steerpr.Verdict) (steerpr.Verdict, bool) {
	for short, v := range verdicts {
		if short != "" && strings.HasPrefix(fullSHA, short) {
			return v, true
		}
	}
	return "", false
}

// dosCommitAuditRange grades base..head in ONE `dos commit-audit A..B --json`
// call and maps each row through the dispatch keep-bit. It is best-effort: any
// failure (dos absent, non-zero exit with unreadable output, bad JSON) returns
// an empty map, and every commit stays ungraded (UNVERIFIABLE), which is the
// honest read — never a fabricated CLEARED.
func dosCommitAuditRange(root, baseSHA, headSHA string) map[string]steerpr.Verdict {
	out := map[string]steerpr.Verdict{}
	cmd := exec.Command("dos", "commit-audit", baseSHA+".."+headSHA, "--json")
	cmd.Dir = root
	configureDispatchHelperCommand(cmd)
	// dos exits 1 when it finds an unwitnessed claim — that is a real verdict,
	// not a tool failure, and it still prints the JSON on stdout. So read stdout
	// regardless of exit code and only bail if the payload does not parse.
	buf, _ := cmd.Output()
	var rows []struct {
		SHA     string `json:"sha"`
		Verdict string `json:"verdict"`
		Witness string `json:"witness"`
	}
	if json.Unmarshal(buf, &rows) != nil {
		return out
	}
	for _, r := range rows {
		if sha := strings.TrimSpace(r.SHA); sha != "" {
			out[sha] = mapAuditVerdict(r.Verdict, r.Witness)
		}
	}
	return out
}

// mapAuditVerdict maps a dos commit-audit row to the overlay's verdict vocabulary
// through the SAME keep-bit the dispatch sweep uses, so the band can never
// disagree with the sweep about whether a commit is witnessed.
func mapAuditVerdict(verdict, witness string) steerpr.Verdict {
	if dispatchtick.CommitWitnessed(verdict, witness) {
		return steerpr.VerdictWitnessed
	}
	if strings.EqualFold(strings.TrimSpace(verdict), string(steerpr.VerdictUnwitnessed)) {
		return steerpr.VerdictUnwitnessed
	}
	return steerpr.VerdictAbstain
}

func writeSteerPRs(view map[string]any, maxFiles int) string {
	var b strings.Builder
	fmt.Fprintf(&b, "# Forming operator PRs — %s\n\n", releaseStatusString(view["range"]))
	commitCount := releaseStatusInt(view["commit_count"])
	if commitCount == 0 {
		fmt.Fprintf(&b, "Nothing forming: %s and %s point at the same history (base %s, head %s).\n",
			releaseStatusString(view["base"]), releaseStatusString(view["head"]),
			releaseStatusShortSHA(releaseStatusString(view["base_sha"])), releaseStatusShortSHA(releaseStatusString(view["head_sha"])))
		return strings.TrimRight(b.String(), "\n")
	}
	units, _ := view["units"].([]steerpr.Unit)
	unstamped, _ := view["unstamped"].([]steerpr.Commit)
	residual := releaseStatusInt(view["residual_count"])
	fmt.Fprintf(&b, "%d commit(s) across %d unit(s); %d RESIDUAL. base %s → head %s.\n",
		commitCount, len(units), residual,
		releaseStatusShortSHA(releaseStatusString(view["base_sha"])), releaseStatusShortSHA(releaseStatusString(view["head_sha"])))
	b.WriteString("Worst-attention-first: RESIDUAL owes you a look; CLEARED the kernel already witnessed.\n")
	for _, unit := range units {
		fmt.Fprintf(&b, "\n## [%s] %s — %d commit(s)\n\n", unit.Band, unit.Leaf, len(unit.Commits))
		fmt.Fprintf(&b, "**Title:** `%s`\n", unit.Title)
		if len(unit.Resolves) > 0 {
			fmt.Fprintf(&b, "Closes %s.\n", strings.Join(unit.Resolves, ", "))
		}
		if len(unit.Mentions) > 0 {
			fmt.Fprintf(&b, "Mentions %s.\n", strings.Join(unit.Mentions, ", "))
		}
		b.WriteString("\n")
		for _, c := range unit.Commits {
			fmt.Fprintf(&b, "- `%s` [%s] %s\n", releaseStatusShortSHA(c.SHA), steerVerdictLabel(c.Verdict), c.Subject)
		}
		b.WriteString("\n")
		fmt.Fprintf(&b, "Files touched (%d): %s\n", len(unit.Files), prPlanFileList(unit.Files, maxFiles))
	}
	if len(unstamped) > 0 {
		fmt.Fprintf(&b, "\n## ⚠ unstamped — %d commit(s) with no `(fak <leaf>)` ship-stamp\n\n", len(unstamped))
		b.WriteString("These cannot be routed to a unit; an operator sees them, but they carry no attention band.\n\n")
		for _, c := range unstamped {
			fmt.Fprintf(&b, "- `%s` %s\n", releaseStatusShortSHA(c.SHA), c.Subject)
		}
	}
	return strings.TrimRight(b.String(), "\n")
}

// steerVerdictLabel renders a per-commit verdict compactly for the member list.
func steerVerdictLabel(v steerpr.Verdict) string {
	switch v {
	case steerpr.VerdictWitnessed:
		return "witnessed"
	case steerpr.VerdictUnwitnessed:
		return "UNWITNESSED"
	case steerpr.VerdictAbstain, steerpr.VerdictNoCommit:
		return "no-claim"
	default:
		return "ungraded"
	}
}
