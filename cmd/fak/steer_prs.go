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
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/branchrole"
	"github.com/anthony-chaudhary/fak/internal/dispatchtick"
	"github.com/anthony-chaudhary/fak/internal/ghexec"
	"github.com/anthony-chaudhary/fak/internal/issuecohort"
	"github.com/anthony-chaudhary/fak/internal/steerpr"
	"github.com/anthony-chaudhary/fak/internal/trajctl"
)

const steerPRsSchema = steerpr.Schema // "fak.steerpr.v1"

// steerPRsVerdicts grades base..head into per-SHA verdicts. Overridable in tests;
// the default shells `dos commit-audit`.
var steerPRsVerdicts = dosCommitAuditRange

// steerRoot resolves the repo root the steer verbs read the range and the ack
// ledger under. Overridable in tests so a test run never touches the real
// overlay ledger.
var steerRoot = repoRoot

// steerPRsTrajState folds the live objective ledger once per view. Tests replace
// this seam so the captured render never reads the repository ledger.
var steerPRsTrajState = func(root string) trajctl.State {
	return trajctl.Fold(trajctl.ReadLedgerFile(filepath.Join(root, trajctl.DefaultLedgerRel)))
}

func splitSteerUnitArg(argv []string) (string, []string) {
	if len(argv) > 0 && !strings.HasPrefix(argv[0], "-") {
		return strings.TrimSpace(argv[0]), argv[1:]
	}
	return "", argv
}

// steerActorCommand owns the common attributable flag contract while leaving each
// verb free to register its own flags and help wording.
type steerActorCommand struct {
	*flag.FlagSet
	argv    []string
	unitArg string
	label   string
	stderr  io.Writer
	actor   *string
}

func newSteerActorCommand(label string, stderr io.Writer, argv []string, unitArg, action string) *steerActorCommand {
	fs := flag.NewFlagSet(label, flag.ContinueOnError)
	fs.SetOutput(stderr)
	cmd := &steerActorCommand{FlagSet: fs, argv: argv, unitArg: unitArg, label: label, stderr: stderr}
	cmd.actor = fs.String("by", "", "who "+action+" (default: git config user.name; the row must be attributable)")
	return cmd
}

func (cmd *steerActorCommand) parseUnit(usage string) (string, int) {
	if !parseFlags(cmd.FlagSet, cmd.argv) {
		return "", 2
	}
	unitArg := cmd.unitArg
	if unitArg == "" && cmd.NArg() == 1 {
		unitArg = strings.TrimSpace(cmd.Arg(0))
	} else if cmd.NArg() != 0 {
		fmt.Fprintln(cmd.stderr, usage)
		return "", 2
	}
	if unitArg == "" {
		fmt.Fprintln(cmd.stderr, usage)
		return "", 2
	}
	return unitArg, 0
}

func steerActor(root, requested string) string {
	who := strings.TrimSpace(requested)
	if who == "" {
		// --get pins the invocation provably read-only for the architest steer-overlay floor.
		who = strings.TrimSpace(releasePRPlanGit(root, "config", "--get", "user.name"))
	}
	return who
}

func resolveSteerUnit(root, base, head, leaf, label string, stderr io.Writer) (*steerpr.Unit, bool) {
	view, err := buildSteerPRsView(root, base, head)
	if err != nil {
		fmt.Fprintf(stderr, "%s: %v\n", label, err)
		return nil, false
	}
	units, _ := view["units"].([]steerpr.Unit)
	for i := range units {
		if units[i].Leaf == leaf {
			return &units[i], true
		}
	}
	fmt.Fprintf(stderr, "%s: no forming unit %q in %s — see `fak steer prs` for the units forming now\n",
		label, leaf, releaseStatusString(view["range"]))
	return nil, false
}

func (cmd *steerActorCommand) resolveUnit(usage string, base, head *string, unbound string) (*steerpr.Unit, string, string, int) {
	unitArg, code := cmd.parseUnit(usage)
	if code != 0 {
		return nil, "", "", code
	}
	root := steerRoot()
	unit, ok := resolveSteerUnit(root, *base, *head, unitArg, cmd.label, cmd.stderr)
	if !ok {
		return nil, "", "", 1
	}
	if unbound != "" && len(unit.Resolves) == 0 {
		fmt.Fprintf(cmd.stderr, unbound, unitArg)
		return nil, "", "", 1
	}
	return unit, root, steerActor(root, *cmd.actor), 0
}

func (cmd *steerActorCommand) actorName(root string) string {
	return steerActor(root, *cmd.actor)
}

func appendAndWriteSteerRecord[T any](stdout, stderr io.Writer, label string, rec T, appendRecord func() error) (int, bool) {
	if err := appendRecord(); err != nil {
		fmt.Fprintf(stderr, "%s: append ledger row: %v\n", label, err)
		return 1, true
	}
	if err := writeIndentedJSON(stdout, rec); err != nil {
		fmt.Fprintf(stderr, "%s: %v\n", label, err)
		return 1, true
	}
	return 0, false
}

func cmdSteer(argv []string) { os.Exit(runSteer(os.Stdout, os.Stderr, argv)) }

func runSteer(stdout, stderr io.Writer, argv []string) int {
	if len(argv) > 0 {
		switch strings.ToLower(strings.TrimSpace(argv[0])) {
		case "prs":
			return runSteerPRs(stdout, stderr, argv[1:])
		case "ack":
			return runSteerAck(stdout, stderr, argv[1:])
		case "comment":
			return runSteerComment(stdout, stderr, argv[1:])
		case "redirect":
			return runSteerRedirect(stdout, stderr, argv[1:])
		case "pause":
			return runSteerPause(stdout, stderr, argv[1:])
		case "resume":
			return runSteerResume(stdout, stderr, argv[1:])
		case "-h", "--help", "help":
			fmt.Fprintln(stdout, steerUsage)
			return 0
		}
	}
	fmt.Fprintln(stderr, steerUsage)
	return 2
}

const steerUsage = `fak steer — the forming operator PRs on the trunk, and where a look lands

Usage:
  fak steer prs [--json] [--check] [--base REF] [--head REF] [--max-files N] [--cohort PLAN.json]
  fak steer ack <unit> [--by WHO] [--note TEXT] [--base REF] [--head REF]
  fak steer comment <unit> -m "<note>" [--by WHO] [--base REF] [--head REF]
  fak steer redirect <unit> -m "<steer note>" [--by WHO] [--base REF] [--head REF]
  fak steer pause <unit> [-m "<reason>"] [--by WHO] [--base REF] [--head REF]
  fak steer resume <unit> [--by WHO]

prs folds the pending dev->release delta into PR-sized units per (fak <leaf>)
stamp, bands each by where attention is owed (RESIDUAL/UNVERIFIABLE/CLEARED),
and lists them worst-first. Read-only; --check reports RESIDUAL, it never gates
a merge.

--cohort takes a "fak-dev issue cohort --json" plan and regroups the commits bound
to a planned WAVE into one unit per wave — the fleet dispatches by wave, so the
wave is the unit an operator can actually stop or redirect. Commits with no wave
binding keep folding by leaf, and every unit states which basis it used
(grouped_by: wave|leaf).

ack records that a human reviewed a unit: an append-only, attributable ledger
row bound to the unit's exact member SHA set. The unit then renders as
"RESIDUAL (acked by WHO)" — never CLEARED: an ack is a human's look, not a
witness, and it moves neither the machine band nor the residual count. A new
member commit invalidates the ack (it was a review of a different SHA set).

comment is the weakest steering rung — annotate. It posts a note to the unit's
closure-grade bound issue through the trusted gh seam, prefixed with the unit's
identity (leaf + the exact member SHA set and band that were read), then records
the annotation on the overlay ledger so the brief can see the unit got operator
attention. A unit that binds no issue is refused rather than posted somewhere
plausible: a mention is not a binding. It changes no band and no ack state.

redirect re-aims a unit's INTENT without touching what landed: it files (or
reopens) a steer follow-up through the trusted gh seam carrying the note, the
unit's exact member SHA set, and its band at redirect time, then appends a
countable redirect row to the overlay ledger. Advisory only: a redirect never
reverts, rewrites, force-pushes, or gates — the next tick changes, the merged
history does not.

pause stops the fleet spending on a unit's bound intent: the bound issue is
skipped with BLOCKED_BY_HUMAN (the dispatcher's existing backpressure token)
from the next dispatch tick until resume releases it. Pause is not a kill: an
in-flight worker still finishes and lands cleanly; the intent is simply not
picked up again while held. resume releases the hold — a pause with no resume
would silently starve the intent, so the verbs ship as a pair.`

func runSteerPRs(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak steer prs", flag.ContinueOnError)
	fs.SetOutput(stderr)
	asJSON := fs.Bool("json", false, "emit machine-readable JSON (schema fak.steerpr.v1)")
	base := fs.String("base", "", "range base ref (default: origin/<release_branch>)")
	head := fs.String("head", "", "range head ref (default: <release_source> tip)")
	check := fs.Bool("check", false, "exit 1 if any forming unit is RESIDUAL (reports; never blocks a merge)")
	maxFiles := fs.Int("max-files", 20, "file paths listed per unit before folding to a count")
	cohort := fs.String("cohort", "", "issue-cohort plan `file` (fak-dev issue cohort --json): fold commits bound to a planned wave into one unit per wave")
	demo := fs.String("demo", "", "deterministic commit-log fixture file (no git or network)")
	selfcheck := fs.Bool("selfcheck", false, "assert the demo fixture's expected worst-first fold")
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

	if *selfcheck && strings.TrimSpace(*demo) == "" {
		fmt.Fprintln(stderr, "fak steer prs: --selfcheck requires --demo")
		return 2
	}
	var view map[string]any
	if strings.TrimSpace(*demo) != "" {
		v, err := buildSteerPRsDemoView(*demo)
		if err != nil {
			fmt.Fprintf(stderr, "fak steer prs: demo: %v\n", err)
			return 1
		}
		view = v
	} else {
		// The wave bindings are read ONCE, here, and handed to the fold as data: an
		// unreadable or wave-less plan is a hard error rather than a silent fall back
		// to leaf grouping, because an operator who asked to watch waves must not be
		// shown a leaf view that looks the same.
		waves, err := steerPRsCohortWaves(*cohort)
		if err != nil {
			fmt.Fprintf(stderr, "fak steer prs: %v\n", err)
			return 2
		}

		v, err := buildSteerPRsViewWaves(steerRoot(), *base, *head, waves)
		if err != nil {
			fmt.Fprintf(stderr, "fak steer prs: %v\n", err)
			return 1
		}
		view = v
	}

	if code := renderSteerPRsView(stdout, stderr, view, *asJSON, *maxFiles); code != 0 {
		return code
	}
	if strings.TrimSpace(*demo) != "" {
		if *selfcheck {
			if err := checkSteerPRsDemo(view); err != nil {
				fmt.Fprintf(stderr, "fak steer prs: selfcheck FAILED: %v\n", err)
				return 1
			}
			fmt.Fprintln(stderr, "fak steer prs: selfcheck OK")
		}
		return 0
	}
	if *check && releaseStatusInt(view["residual_count"]) > 0 {
		fmt.Fprintf(stderr, "fak steer prs: %d unit(s) in %s are RESIDUAL — a claim the kernel could not witness; a human look buys something here\n",
			releaseStatusInt(view["residual_count"]), releaseStatusString(view["range"]))
		return 1
	}
	return 0
}

func renderSteerPRsView(stdout, stderr io.Writer, view map[string]any, asJSON bool, maxFiles int) int {
	if asJSON {
		if err := writeIndentedJSON(stdout, view); err != nil {
			fmt.Fprintf(stderr, "fak steer prs: encode json: %v\n", err)
			return 1
		}
	} else {
		fmt.Fprintln(stdout, writeSteerPRs(view, maxFiles))
	}
	return 0
}

// runSteerAck records a human's "I looked" against a forming unit (#5028): an
// append-only, attributable ledger row bound to the unit's exact member SHA
// set at ack time. It writes ONLY the ledger — never a Verdict, never a Band —
// so an ack cannot launder an unwitnessed commit into CLEARED (the #5036
// fence), and a member that lands later invalidates the ack by changing the
// SHA set the row was bound to.
func runSteerAck(stdout, stderr io.Writer, argv []string) int {
	// The unit name may come before the flags (`fak steer ack gateway --note x`)
	// or after them; accept both.
	unitArg, argv := splitSteerUnitArg(argv)
	ackFlags := newSteerActorCommand("fak steer ack", stderr, argv, unitArg, "looked")
	note := ackFlags.String("note", "", "optional note recorded with the ack")
	base := ackFlags.String("base", "", "range base ref (default: origin/<release_branch>)")
	head := ackFlags.String("head", "", "range head ref (default: <release_source> tip)")
	const usage = "usage: fak steer ack <unit> [--by WHO] [--note TEXT] [--base REF] [--head REF]"
	unit, root, who, code := ackFlags.resolveUnit(usage, base, head, "")
	if code != 0 {
		return code
	}
	ack, err := steerpr.NewAck(unit.Leaf, who, steerpr.UnitSHAs(*unit), *note, time.Now())
	if err != nil {
		fmt.Fprintf(stderr, "fak steer ack: %v\n", err)
		return 2
	}
	if code, done := appendAndWriteSteerRecord(stdout, stderr, "fak steer ack", ack, func() error {
		return steerpr.AppendAck(steerpr.AckLedgerPath(root), ack)
	}); done {
		return code
	}
	fmt.Fprintf(stdout, "acked %s (%d commit(s), band %s) as %s — the machine band is untouched, and a new member commit invalidates this ack\n",
		unit.Leaf, len(unit.Commits), unit.Band, who)
	return 0
}

// steerRedirectFile is the trusted `gh` seam the redirect files its follow-up
// through (#5030): overridable in tests so a test run never reaches the
// network. The default routes ONLY through internal/ghexec (the deadlined gh
// runner, the same trusted seam the comment affordance uses) — a redirect can
// move a GitHub issue, and can never move git.
var steerRedirectFile = ghSteerRedirectFollowUp

// runSteerRedirect records an operator redirect against a forming unit
// (#5030): re-aim the intent's NEXT tick without touching what already landed.
// It files (or reopens) a steer follow-up through the trusted gh seam carrying
// the operator's note plus the unit's exact member SHA set and current band,
// then appends an attributable, append-only redirect row to the overlay
// ledger so the steer is a first-class, countable event. ADVISORY by
// construction: no code path from here reaches a git mutation — the
// structural fence is TestRedirectNeverReachesGitMutation in
// internal/steerpr, and a redirect that could touch the trunk is a failed
// implementation of the affordance regardless of how useful it seems.
func runSteerRedirect(stdout, stderr io.Writer, argv []string) int {
	// The unit name may come before the flags or after them; accept both.
	unitArg, argv := splitSteerUnitArg(argv)
	redirectFlags := newSteerActorCommand("fak steer redirect", stderr, argv, unitArg, "is steering")
	note := redirectFlags.String("m", "", "the steer note: where the intent's next tick should aim (required)")
	base := redirectFlags.String("base", "", "range base ref (default: origin/<release_branch>)")
	head := redirectFlags.String("head", "", "range head ref (default: <release_source> tip)")
	const usage = `usage: fak steer redirect <unit> -m "<steer note>" [--by WHO] [--base REF] [--head REF]`
	unit, root, who, code := redirectFlags.resolveUnit(usage, base, head, "")
	if code != 0 {
		return code
	}
	bound := ""
	if len(unit.Resolves) > 0 {
		// The unit's closure-grade binding: the follow-up reopens/annotates it
		// rather than filing fresh, so the steer lands where the intent lives.
		bound = unit.Resolves[0]
	}
	rec, err := steerpr.NewRedirect(unit.Leaf, who, *note, steerpr.UnitSHAs(*unit), unit.Band, bound, time.Now())
	if err != nil {
		fmt.Fprintf(stderr, "fak steer redirect: %v\n", err)
		return 2
	}
	// File FIRST, ledger after: a follow-up that never landed is not a
	// steering event, and the ledger row records where the filed one went.
	followUp, err := steerRedirectFile(rec)
	if err != nil {
		fmt.Fprintf(stderr, "fak steer redirect: file follow-up via gh: %v\n", err)
		return 1
	}
	rec.FollowUp = strings.TrimSpace(followUp)
	if code, done := appendAndWriteSteerRecord(stdout, stderr, "fak steer redirect", rec, func() error {
		return steerpr.AppendRedirect(steerpr.RedirectLedgerPath(root), rec)
	}); done {
		return code
	}
	fmt.Fprintf(stdout, "redirected %s (%d commit(s), band %s) as %s — follow-up %s; the landed commits are untouched: a redirect re-aims the next tick, never the merge\n",
		unit.Leaf, len(unit.Commits), unit.Band, who, rec.FollowUp)
	return 0
}

// ghSteerRedirectFollowUp is the default trusted gh seam: with a bound issue
// it reopens best-effort (already-open is fine — the point is the note lands)
// and posts the anchored note as a comment; without one it files a fresh
// follow-up issue. Every invocation goes through internal/ghexec — deadlined,
// prompt-disabled, window-suppressed — and only ever the `gh issue` verb
// family: GitHub state moves, git never does.
func ghSteerRedirectFollowUp(r steerpr.Redirect) (string, error) {
	if r.Issue != "" {
		num := strings.TrimPrefix(r.Issue, "#")
		reopen, cancelReopen := ghexec.CommandTimeout(nil, ghexec.DefaultTimeout, "issue", "reopen", num)
		_, _ = reopen.CombinedOutput() // best-effort: an already-open issue is not an error
		cancelReopen()
		comment, cancelComment := ghexec.CommandTimeout(nil, ghexec.DefaultTimeout, "issue", "comment", num, "--body", r.FollowUpBody())
		defer cancelComment()
		if out, err := comment.CombinedOutput(); err != nil {
			return "", fmt.Errorf("gh issue comment %s: %v: %s", num, err, strings.TrimSpace(string(out)))
		}
		return r.Issue, nil
	}
	create, cancel := ghexec.CommandTimeout(nil, ghexec.DefaultTimeout, "issue", "create", "--title", r.FollowUpTitle(), "--body", r.FollowUpBody())
	defer cancel()
	out, err := create.Output()
	if err != nil {
		return "", fmt.Errorf("gh issue create: %v", err)
	}
	return strings.TrimSpace(string(out)), nil
}

// buildSteerPRsView resolves the pending delta, grades it, and folds it into the
// worst-attention-first operator view. It reuses the release-plan range
// resolution (branchrole + prPlanResolve) and git seam so the continuous view
// and the promotion plan always fold the SAME range through the SAME parser.
type steerPRsDemoFixture struct {
	Schema  string               `json:"schema"`
	Commits []steerPRsDemoCommit `json:"commits"`
}

type steerPRsDemoCommit struct {
	SHA     string          `json:"sha"`
	Subject string          `json:"subject"`
	Body    string          `json:"body,omitempty"`
	Files   []string        `json:"files,omitempty"`
	Verdict steerpr.Verdict `json:"verdict"`
}

func buildSteerPRsDemoView(path string) (map[string]any, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var fixture steerPRsDemoFixture
	if err := json.Unmarshal(b, &fixture); err != nil {
		return nil, fmt.Errorf("decode %s: %w", path, err)
	}
	if fixture.Schema != "fak.steerpr.demo.v1" {
		return nil, fmt.Errorf("schema = %q, want fak.steerpr.demo.v1", fixture.Schema)
	}
	commits := make([]steerpr.Commit, 0, len(fixture.Commits))
	for _, row := range fixture.Commits {
		raw := row.SHA + "\x1f" + row.Subject + "\x1f" + row.Body + "\x1f" + strings.Join(row.Files, "\n") + "\x1e"
		parsed := steerpr.ParseLog(raw)
		if len(parsed) != 1 {
			return nil, fmt.Errorf("commit %q did not parse", row.SHA)
		}
		parsed[0].Verdict = row.Verdict
		commits = append(commits, parsed[0])
	}
	units, unstamped := steerpr.FoldUnits(commits)
	steerpr.SortWorstFirst(units)
	return map[string]any{
		"schema": "fak.steerpr.v1", "base": "fixture-base", "base_sha": "fixture-base",
		"head": "fixture-head", "head_sha": "fixture-head", "range": "fixture",
		"development_branch": "fixture-dev", "release_branch": "fixture-release", "release_source": "fixture-dev",
		"commit_count": len(commits), "unit_count": len(units), "unstamped_count": len(unstamped),
		"residual_count": steerpr.Residual(units), "forming_count": 0, "unknown_expected_count": 0,
		"wave_unit_count": 0, "units": units, "unstamped": unstamped,
		"acks": map[string]steerpr.Ack{}, "pauses": map[string]steerpr.Pause{},
	}, nil
}

func checkSteerPRsDemo(view map[string]any) error {
	units, _ := view["units"].([]steerpr.Unit)
	unstamped, _ := view["unstamped"].([]steerpr.Commit)
	if len(units) != 3 || len(unstamped) != 1 {
		return fmt.Errorf("fold = %d unit(s), %d orphan(s); want 3, 1", len(units), len(unstamped))
	}
	wantBands := []steerpr.Band{steerpr.BandResidual, steerpr.BandUnverifiable, steerpr.BandCleared}
	wantLeaves := []string{"gateway", "model", "docs"}
	wantMembers := []int{1, 1, 2}
	for i := range wantBands {
		if units[i].Band != wantBands[i] || units[i].Leaf != wantLeaves[i] || len(units[i].Commits) != wantMembers[i] {
			return fmt.Errorf("unit[%d] = %s/%s/%d; want %s/%s/%d", i, units[i].Band, units[i].Leaf, len(units[i].Commits), wantBands[i], wantLeaves[i], wantMembers[i])
		}
	}
	if unstamped[0].Leaf != "" {
		return fmt.Errorf("orphan unexpectedly stamped %q", unstamped[0].Leaf)
	}
	return nil
}
func buildSteerPRsView(root, base, head string) (map[string]any, error) {
	return buildSteerPRsViewWaves(root, base, head, nil)
}

// steerPRsCohortWaves projects a `fak-dev issue cohort --json` plan into the overlay's
// wave bindings (#5040). It reads the EXISTING plan through the existing type —
// no second planner, no new grouping key: the wave index and its members are the
// cohort's own, and the join is the issue number a member already carries.
//
// An empty path means no wave grouping was asked for (the common case), so the
// overlay folds by leaf. A named path that cannot be read, cannot be parsed, or
// carries no issue-numbered wave member is an ERROR: those are the three ways an
// operator could ask for the wave view and silently be handed the leaf view
// instead, which is the confusion this issue exists to remove. A cohort planned
// over not-yet-filed candidates is exactly that case — its members have no issue
// numbers yet, so nothing could bind.
func steerPRsCohortWaves(path string) ([]steerpr.WaveBinding, error) {
	if strings.TrimSpace(path) == "" {
		return nil, nil
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read cohort plan: %w", err)
	}
	var plan issuecohort.Plan
	if err := json.Unmarshal(raw, &plan); err != nil {
		return nil, fmt.Errorf("parse cohort plan %s: %w", path, err)
	}
	var bindings []steerpr.WaveBinding
	for _, w := range plan.Waves {
		binding := steerpr.WaveBinding{Index: w.Index}
		for _, m := range w.Members {
			if m.IssueNumber > 0 {
				binding.Issues = append(binding.Issues, fmt.Sprintf("#%d", m.IssueNumber))
			}
		}
		if len(binding.Issues) > 0 {
			bindings = append(bindings, binding)
		}
	}
	if len(bindings) == 0 {
		return nil, fmt.Errorf("cohort plan %s carries no wave member with an issue number — plan the cohort over live issues (`fak-dev issue cohort --from-issues`) so a landed commit's #N can bind to its wave", path)
	}
	return bindings, nil
}

// buildSteerPRsViewWaves is buildSteerPRsView with optional cohort wave
// bindings: commits whose subject-bound issue belongs to a planned wave fold
// into one unit per wave, the rest keep folding by leaf (#5040).
func buildSteerPRsViewWaves(root, base, head string, waves []steerpr.WaveBinding) (map[string]any, error) {
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

	// The fleet dispatches by WAVE, not by leaf (#5040): when a commit's bound
	// issue belongs to a planned cohort wave, the wave is the unit an operator can
	// actually steer ("stop this wave"), so it is the unit of attention. Leaf
	// grouping stays the fallback for everything else — the common case — and each
	// unit states which basis it used.
	units, unstamped := steerpr.FoldUnitsByWave(commits, steerpr.WaveIndex(waves))
	attachSteerPRCurves(units, steerPRsTrajState(root))
	steerpr.SortWorstFirst(units)

	// The partial state rides beside the band as a third orthogonal axis (#5027):
	// N of M expected commits landed, where M is the bound intent's DECLARED
	// membership — its spine issue plus the fanout children carrying the
	// `fanout-<leaf>-` marker. The band says whether to look; this says whether it
	// is still cheap to act.
	//
	// The issue graph is gathered ONCE for the whole view (one bounded gh call,
	// through the existing ghexec seam — never a second GitHub client) and every
	// unit derives from that one set. Gathering is BEST-EFFORT in exactly the way
	// verdict grading is: if gh is unavailable the set is empty, no spine is
	// found, and every unit reports expected: unknown — never a fabricated
	// denominator, and never silently M = N.
	// The two declared denominator sources are routed by the unit's own grouping
	// basis, never blended. A WAVE unit's M is the cohort plan's declared wave
	// size; only a LEAF unit's M comes from the fanout marker graph.
	//
	// Routing a wave unit through the leaf derivation is the M = N trap in
	// disguise: a wave key ("wave:2") matches no "fanout-wave:2-" child, so the
	// derivation collapses to M = 1 and the unit reads COMPLETE on its first
	// commit — telling an operator a twelve-issue wave had finished at the exact
	// moment redirecting it was still cheap. See
	// steerpr.TestWaveUnitNeverRendersCompleteViaLeafDerivation.
	intentIssues := steerPRsIntentIssues(root)
	steerpr.AttachPartials(units, func(u steerpr.Unit) (steerpr.Expectation, bool) {
		if u.GroupedBy == steerpr.GroupedByWave {
			// No plan, or a wave nobody planned, yields unknown — never a
			// fabricated wave size.
			return steerpr.WaveExpectation(u.Leaf, waves)
		}
		if len(u.Resolves) == 0 {
			// No closure-grade binding: there is no intent to count. Unknown.
			return steerpr.Expectation{}, false
		}
		return steerpr.DeriveExpected(u.Leaf, u.Resolves[0], intentIssues)
	})

	// The acked state rides BESIDE the band as a separate field, never in it:
	// only a ledger row whose SHA set exactly matches the unit's CURRENT member
	// set still covers — a member that joined after the human looked drops the
	// unit back to unacked (#5028's SHA-set invalidation rule).
	acks := steerpr.LoadAcks(steerpr.AckLedgerPath(root))
	acked := map[string]steerpr.Ack{}
	for _, u := range units {
		if a, ok := steerpr.AckFor(acks, u.Leaf, steerpr.UnitSHAs(u)); ok {
			acked[u.Leaf] = a
		}
	}
	// The paused state rides BESIDE the band too (#5031): an active pause is an
	// operator's live hold on the unit's bound intent, shown with paused-since
	// so paused time is visible — a silently paused intent would be
	// indistinguishable from a finished one.
	pauses := steerpr.ActivePauses(steerpr.LoadPauses(steerpr.PauseLedgerPath(root)))
	paused := map[string]steerpr.Pause{}
	for _, u := range units {
		if p, ok := pauses[u.Leaf]; ok {
			paused[u.Leaf] = p
		}
	}
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
		// #5027: how much of the overlay is still forming (cheap to steer) versus
		// how much carries no denominator at all. The unknown count is posted
		// beside the forming count on purpose — an operator must be able to see how
		// much of the view is NOT carrying a steering signal, rather than reading
		// an unmeasured unit as a finished one.
		"forming_count":          len(steerpr.PartialUnits(units)),
		"unknown_expected_count": len(steerpr.UnknownExpectedUnits(units)),
		// #5040: how many units are grouped by cohort WAVE rather than by leaf,
		// posted up front so an operator sees that two bases are in play before
		// reading any single unit.
		"wave_unit_count": len(steerpr.WaveUnits(units)),
		"units":           units,
		"unstamped":       unstamped,
		"acks":            acked,
		"pauses":          paused,
	}, nil
}

// attachSteerPRCurves joins an overlay unit to the objective convention used by
// dispatch: a closure-grade #N binds issue-N first, then the unit leaf. The
// first live objective wins; units without one stay curve-free.
func attachSteerPRCurves(units []steerpr.Unit, state trajctl.State) {
	steerpr.AttachCurves(units, func(unit steerpr.Unit) (steerpr.Curve, bool) {
		ids := make([]string, 0, len(unit.Resolves)+1)
		for _, issue := range unit.Resolves {
			ids = append(ids, "issue-"+strings.TrimPrefix(issue, "#"))
		}
		ids = append(ids, unit.Leaf)
		for _, id := range ids {
			objective, ok := state.Objectives[id]
			if !ok || (objective.Status != trajctl.StatusActive && objective.Status != trajctl.StatusPaused) {
				continue
			}
			curve, ok := state.CurveFor(id)
			if !ok {
				continue
			}
			var rung steerpr.CurveRung
			for _, score := range state.ScoresFor(id) {
				if score.Method == trajctl.CommitScorerMethod {
					rung = steerpr.CurveRung(score.Witness)
				}
			}
			return steerpr.Curve{
				ObjectiveID: curve.ObjectiveID,
				Signal:      steerpr.CurveSignal(curve.Signal),
				Rung:        rung,
				Latest:      curve.Latest,
				Delta:       curve.Delta,
				Detail:      curve.Detail,
			}, true
		}
		return steerpr.Curve{}, false
	})
}

// steerPRsIntentGather is the bounded issue-graph scan the partial state derives
// its denominator from (#5027). Overridable in tests so a test run never reaches
// the network; the default shells the existing trusted gh seam.
var steerPRsIntentGather = ghSteerIntentIssues

// steerPRsIntentDedupeCap bounds the scan. An uncapped tracker walk is the
// failure mode the fanout marker-key contract already caps at 300; the overlay
// reads the same graph under the same bound.
const steerPRsIntentDedupeCap = 300

// steerPRsIntentIssues gathers the issue graph once per view build. Any failure
// returns an empty set, which makes every unit's denominator UNKNOWN rather than
// guessed — the same best-effort degradation the verdict grading uses, and for
// the same reason: an honest "not derivable" beats a fabricated number.
func steerPRsIntentIssues(root string) []steerpr.IntentIssue {
	return steerPRsIntentGather(root)
}

// ghSteerIntentIssues runs the bounded `gh issue list --state all` scan through
// internal/ghexec — deadlined, prompt-disabled, window-suppressed, and the SAME
// seam the redirect affordance and the fanout dedupe use. Read-only: only the
// `gh issue list` verb, so GitHub state never moves and git certainly never does.
//
// The scan is `--state all`, not `--state open`, deliberately. M is the intent's
// DECLARED membership and must stay stable as children close; counting only open
// children would shrink M every time work landed, so a unit would march toward
// "3 of 3" and report complete while children were still outstanding.
func ghSteerIntentIssues(root string) []steerpr.IntentIssue {
	cmd, cancel := ghexec.CommandTimeout(nil, ghexec.DefaultTimeout,
		"issue", "list", "--state", "all",
		"--limit", strconv.Itoa(steerPRsIntentDedupeCap),
		"--json", "number,body")
	defer cancel()
	cmd.Dir = root
	buf, err := cmd.Output()
	if err != nil {
		return nil
	}
	var rows []steerpr.IntentIssue
	if json.Unmarshal(buf, &rows) != nil {
		return nil
	}
	return rows
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
	// #5027: the forming/unknown split, posted up front so an operator sees how
	// much of the view is still cheap to redirect before reading any single unit.
	forming := releaseStatusInt(view["forming_count"])
	unknownExpected := releaseStatusInt(view["unknown_expected_count"])
	if forming > 0 || unknownExpected > 0 {
		fmt.Fprintf(&b, "%d unit(s) still FORMING (members outstanding — still cheap to steer); %d carry no derivable denominator (expected: unknown).\n",
			forming, unknownExpected)
	}
	// #5040: two grouping bases coexist, so the split is stated before any unit is
	// read. An operator who cannot tell why a unit holds what it holds has been
	// made worse off by the regrouping, not better.
	if waveUnits := releaseStatusInt(view["wave_unit_count"]); waveUnits > 0 {
		fmt.Fprintf(&b, "%d unit(s) grouped by cohort WAVE (the fleet's dispatch unit — stop/redirect the wave); the rest by (fak <leaf>) stamp.\n", waveUnits)
	}
	acked, _ := view["acks"].(map[string]steerpr.Ack)
	pausedNow, _ := view["pauses"].(map[string]steerpr.Pause)
	for _, unit := range units {
		// The acked state renders as a suffix beside the honest band — an acked
		// residual reads "RESIDUAL (acked by X)", never CLEARED. The grouping basis
		// rides in the header beside it: it is mandatory on every unit, so it is
		// printed for leaf units too, not only for the novel wave ones.
		a, ok := acked[unit.Leaf]
		fmt.Fprintf(&b, "\n## [%s] %s — %d commit(s) · grouped_by: %s\n\n", steerpr.BandLabel(unit.Band, a, ok), unit.Leaf, len(unit.Commits), steerpr.GroupingBasis(unit))
		if len(unit.Leaves) > 0 {
			fmt.Fprintf(&b, "Wave spans %d lane(s): %s.\n", len(unit.Leaves), strings.Join(unit.Leaves, ", "))
		}
		// A live hold renders with paused-since (#5031): paused time must be
		// visible, or a paused intent is indistinguishable from a finished one.
		if p, held := pausedNow[unit.Leaf]; held {
			fmt.Fprintf(&b, "**PAUSED** by %s since %s — dispatch skips %s (BLOCKED_BY_HUMAN); release with `fak steer resume %s`.\n", p.By, p.At, p.Issue, unit.Leaf)
		}
		// The partial state renders on its own line beside the band (#5027): a
		// forming unit ("3 of 12 landed") must not read like a finished one, and a
		// unit with no derivable denominator says so rather than going quiet —
		// silence reads as completeness to an operator scanning the overlay.
		if line := unit.Partial.Annotate(); line != "" {
			fmt.Fprintf(&b, "%s\n", line)
		}
		if line := unit.Curve.Annotate(); line != "" {
			if unit.Band == steerpr.BandCleared && len(steerpr.DriftHiddenByBand([]steerpr.Unit{unit})) > 0 {
				fmt.Fprintf(&b, "**DRIFT HIDDEN BY CLEARED BAND** - %s\n", line)
			} else {
				fmt.Fprintf(&b, "%s\n", line)
			}
		}
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
