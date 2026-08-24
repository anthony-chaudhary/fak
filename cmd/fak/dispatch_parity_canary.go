package main

// dispatch_parity_canary.go — `fak dispatch parity-canary`, the Go port of
// tools/dispatch_parity_canary.py (#420): the typed gate that holds the opencode
// worker population at 1 until glm-5.2 is *proven* to drive the dispatch loop the
// way Claude does. It does NOT run a worker — it grades an ALREADY-shipped canary
// commit against the parity bar, the closed set of behaviors a loop worker must
// exhibit no matter which model drives it, computed from the SAME non-forgeable
// surfaces the loop already trusts (git + `dos commit-audit`), never the worker's
// own transcript prose.
//
//	# grade an opencode canary against the bar (the population-raise gate)
//	fak dispatch parity-canary --commit 7528df3 --issue 545 --backend opencode \
//	    --log .dispatch-runs/resolve-545-20260623-162209.log --lane-tree internal/recall
//	# machine-readable verdict (schema fak-dispatch-parity-canary/1)
//	fak dispatch parity-canary --commit 7528df3 --issue 545 --json
//
// The parity bar, cheapest/most-load-bearing rung first so the first miss is the
// most informative:
//
//	1. shipped_a_unit   the canary produced a real commit (touched >=1 file), not an --allow-empty
//	2. witnessed        `dos commit-audit <sha>` grades it OK / diff|data-witnessed
//	3. issue_bound      the subject cites `#<issue>` so the closure auditor can bind it
//	4. by_pathspec      it committed by explicit `git commit -- <paths>`, never a blanket `git add -A`
//	5. signed_off       `git commit -s` (the DCO sign-off the hook requires)
//	6. lane_tree_clean  the lane's file tree carries no stranded edit
//
// All six hold ⇒ PARITY_PROVEN (exit 0). Any miss ⇒ a typed PARITY_UNPROVEN_*
// carrying the first failed rung (exit 1); an unobservable behavioral rung (no run
// log / unknown lane tree) ⇒ PARITY_UNOBSERVED — an evidence gap, not a refutation,
// so a thin answer never masquerades as a strong one. Read-only.
//
// Parity note (audited against the Python original): a behavioral cross-check runs
// both tools on real commits and produces identical verdicts. Two classes of
// difference are known, audited, and accepted — they never trigger on the real
// operating domain (positive integer issue numbers, captured ASCII shell run-logs):
//   - Go's RE2 \s/\d/\b are ASCII-only where Python's `re` is Unicode-aware, so the
//     regex rungs would grade a token separated by a vertical-tab/NBSP or an issue
//     number written in non-ASCII digits differently. Real git-command streams and
//     GitHub issue numbers are ASCII, so the accept/reject sets coincide there.
//   - The --workspace default is os.Getwd() (the fak-house convention shared by
//     `dispatch tick`/`scorecard`/`wave`), NOT the Python script's repo-root walk;
//     a fak subcommand follows fak's cwd/--workspace idiom. Pass --workspace or run
//     from the repo root, as with every other `fak dispatch` verb.

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"os/exec"
	"regexp"
	"strconv"
	"strings"
	"time"
)

const parityCanarySchema = "fak-dispatch-parity-canary/1"

// parityBar is the closed, ordered set of behaviors a dispatch-loop worker must
// exhibit regardless of which model drives it. Each rung names the typed
// PARITY_UNPROVEN_* reason emitted when it fails. Ordered cheapest/most-load-
// bearing first so the first miss is the most informative — a byte-for-byte
// mirror of dispatch_parity_canary.PARITY_BAR.
var parityBar = []struct{ rung, token string }{
	{"shipped_a_unit", "PARITY_UNPROVEN_NO_UNIT"},
	{"witnessed", "PARITY_UNPROVEN_UNWITNESSED"},
	{"issue_bound", "PARITY_UNPROVEN_UNBOUND"},
	{"by_pathspec", "PARITY_UNPROVEN_BLANKET_ADD"},
	{"signed_off", "PARITY_UNPROVEN_NO_SIGNOFF"},
	{"lane_tree_clean", "PARITY_UNPROVEN_TREE_DIRTY"},
}

// --------------------------------------------------------------------------
// Pure predicates over the commit subject and the worker's git-command stream.
// String-pure so a test drives every rung without git/subprocess/network.
// --------------------------------------------------------------------------

// A worker that ran `git add -A` / `git add .` / `git add --all` / `git add -u`
// staged the whole shared tree — the blanket-add that steals a sibling's WIP. The
// leading-anchor match ignores help-text echoes like `(use "git add <file>..." )`
// which carry a quote/paren and never sit at the start of a command.
var parityBlanketAdd = regexp.MustCompile(`(?m)^\s*git\s+add\s+(?:-A\b|--all\b|-u\b|\.(?:\s|$))`)

// The shared-tree discipline is satisfied two ways: a pathspec on the commit
// itself (`git commit -- <path>`), OR an explicit `git add <named-path>` (a real
// path, not a flag and not the `<file>` placeholder). Either proves it did not
// blanket-add. parityExplicitAdd rewrites the Python negative-lookahead
// `git add (?!-|<)[^\s"']+` into an RE2-safe first-char class: the first token
// after `git add ` must not start with a flag `-`, the `<file>` placeholder `<`,
// whitespace, or a quote. (parityBlanketAdd is checked FIRST, so a `git add .`
// that this class would also accept is already refuted upstream — matching the
// Python, where _EXPLICIT_ADD likewise matches `.` and relies on the same order.)
var parityCommitPathspec = regexp.MustCompile(`(?m)^\s*git\s+commit\b[^\n]*\s--\s+\S`)
var parityExplicitAdd = regexp.MustCompile(`(?m)^\s*git\s+add\s+[^\s"'<-][^\s"']*`)
var parityCommitSignoff = regexp.MustCompile(`(?m)^\s*git\s+commit\b[^\n]*(?:-s\b|--signoff\b)`)

// parityGitInLog pulls every `git ...` invocation out of a run log structurally
// (the command stream), never the surrounding prose. Mirrors _GIT_IN_LOG.
var parityGitInLog = regexp.MustCompile(`(?i)git\s+[a-z][\w-]*[^\n]*`)

// parityIssueBound is rung 5: the subject cites `#<issue>` (or any `#N` when the
// issue is unset). The `#\b<n>\b` anchoring is what stops `#54` matching a subject
// that only carries `#545`. Mirrors issue_bound.
func parityIssueBound(subject string, issue int, issueSet bool) bool {
	if issueSet {
		re := regexp.MustCompile(`#\b` + strconv.Itoa(issue) + `\b`)
		return re.MatchString(subject)
	}
	return regexp.MustCompile(`#\d+\b`).MatchString(subject)
}

// parityByPathspec is rung 3: staged explicitly (commit pathspec or named add),
// never blanket. Mirrors by_pathspec.
func parityByPathspec(gitCmds string) bool {
	if parityBlanketAdd.MatchString(gitCmds) {
		return false
	}
	return parityCommitPathspec.MatchString(gitCmds) || parityExplicitAdd.MatchString(gitCmds)
}

// paritySignedOff is rung 4: the commit carried a DCO sign-off. Mirrors signed_off.
func paritySignedOff(gitCmds string) bool {
	return parityCommitSignoff.MatchString(gitCmds)
}

// parityGitCmdsFromLog flattens a run log to the newline-joined `git ...` commands
// it ran (prose dropped). Mirrors git_cmds_from_log.
func parityGitCmdsFromLog(text string) string {
	matches := parityGitInLog.FindAllString(text, -1)
	out := make([]string, 0, len(matches))
	for _, m := range matches {
		out = append(out, strings.TrimRight(m, " \t\r"))
	}
	return strings.Join(out, "\n")
}

// --------------------------------------------------------------------------
// The fold: apply the parity bar, return the rung map + the first-failed verdict.
// --------------------------------------------------------------------------

// parityGradeInput is the tri-bool rung evidence. A nil *bool means the rung is
// UNOBSERVED (no run log / unknown lane tree) — an evidence gap, not a refutation.
type parityGradeInput struct {
	ShippedAUnit bool
	Witnessed    bool
	Subject      string
	Issue        int
	IssueSet     bool
	GitCmds      *string // nil ⇒ no run log ⇒ behavioral rungs unobserved
	LaneClean    *bool   // nil ⇒ unknown lane tree ⇒ unobserved
}

type parityGrade struct {
	Proven     bool             `json:"proven"`
	Verdict    string           `json:"verdict"`
	FailedRung *string          `json:"failed_rung"`
	Rungs      map[string]*bool `json:"rungs"`
}

func boolPtr(b bool) *bool { return &b }

// parityGradeFold applies the parity bar. Mirrors grade(): a nil rung stops the
// fold with the distinct PARITY_UNOBSERVED verdict; a false rung stops with that
// rung's typed token; all-true ⇒ PARITY_PROVEN.
func parityGradeFold(in parityGradeInput) parityGrade {
	haveLog := in.GitCmds != nil
	var byPath, signed *bool
	if haveLog {
		byPath = boolPtr(parityByPathspec(*in.GitCmds))
		signed = boolPtr(paritySignedOff(*in.GitCmds))
	}
	rungs := map[string]*bool{
		"shipped_a_unit":  boolPtr(in.ShippedAUnit),
		"witnessed":       boolPtr(in.Witnessed),
		"issue_bound":     boolPtr(parityIssueBound(in.Subject, in.Issue, in.IssueSet)),
		"by_pathspec":     byPath,
		"signed_off":      signed,
		"lane_tree_clean": in.LaneClean,
	}

	var failed *string
	var reason string
	for _, bar := range parityBar {
		value := rungs[bar.rung]
		if value == nil {
			// Unobserved behavioral rung: parity is not PROVEN, but this is an
			// evidence gap, not a refutation. Stop here with a distinct verdict.
			r := bar.rung
			failed = &r
			reason = "PARITY_UNOBSERVED"
			break
		}
		if !*value {
			r := bar.rung
			failed = &r
			reason = bar.token
			break
		}
	}

	proven := failed == nil
	verdict := reason
	if proven {
		verdict = "PARITY_PROVEN"
	}
	return parityGrade{Proven: proven, Verdict: verdict, FailedRung: failed, Rungs: rungs}
}

// --------------------------------------------------------------------------
// I/O layer: derive the rung inputs from git + `dos commit-audit` + the run log.
// The seams are package vars so an evaluate-level test drives them hermetically.
// --------------------------------------------------------------------------

var (
	parityGitRunner      = parityGitDefault
	parityWitnessRunner  = parityWitnessCommitAudit
	parityLaneCleanCheck = parityLaneTreeCleanGit
)

// parityGitDefault runs a read-only git command, returning (rc, stdout). Never
// panics — an exec error surfaces as (1, ""). Mirrors _git.
func parityGitDefault(root string, args ...string) (int, string) {
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, "git", args...)
	cmd.Dir = root
	configureDispatchHelperCommand(cmd)
	out, err := cmd.Output()
	if err != nil {
		if ee, ok := err.(*exec.ExitError); ok {
			return ee.ExitCode(), string(out)
		}
		return 1, ""
	}
	return 0, string(out)
}

// parityWitnessResult is rung 2's evidence: the commit-audit grade.
type parityWitnessResult struct {
	Witnessed bool    `json:"witnessed"`
	Verdict   *string `json:"verdict"`
	Witness   *string `json:"witness"`
	Error     string  `json:"error,omitempty"`
}

// parityWitnessCommitAudit grades sha via `dos commit-audit <sha> --workspace
// <root> --json`. OK ∧ diff|data-witnessed ⇒ witnessed. Fail-closed if dos is
// absent or the row is missing. Mirrors witness_commit.
func parityWitnessCommitAudit(root, sha string) parityWitnessResult {
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, "dos", "commit-audit", sha, "--workspace", root, "--json")
	cmd.Dir = root
	configureDispatchHelperCommand(cmd)
	out, err := cmd.Output()
	if err != nil {
		if _, ok := err.(*exec.ExitError); !ok {
			// could not start (dos absent) — fail-closed
			return parityWitnessResult{Verdict: strPtr("DOS_UNAVAILABLE"), Error: err.Error()}
		}
		// exit 1 from commit-audit is a real verdict row — fall through to parse.
	}
	trimmed := strings.TrimSpace(string(out))
	if trimmed == "" {
		return parityWitnessResult{Verdict: strPtr("DOS_UNAVAILABLE")}
	}
	var parsed any
	if err := json.Unmarshal([]byte(trimmed), &parsed); err != nil {
		return parityWitnessResult{Verdict: strPtr("DOS_UNPARSEABLE")}
	}
	row := map[string]any{}
	switch doc := parsed.(type) {
	case map[string]any:
		row = doc
	case []any:
		if len(doc) > 0 {
			if m, ok := doc[0].(map[string]any); ok {
				row = m
			}
		}
	}
	verdict := dispatchMapString(row, "verdict")
	witness := dispatchMapString(row, "witness")
	ok := strings.EqualFold(strings.TrimSpace(verdict), "OK") &&
		(witness == "diff-witnessed" || witness == "data-witnessed")
	return parityWitnessResult{Witnessed: ok, Verdict: strPtr(verdict), Witness: strPtr(witness)}
}

// parityLaneTreeCleanGit is rung 6: the lane's file tree carries no uncommitted
// change right now. laneTree "" ⇒ nil (unobserved). Mirrors lane_tree_clean.
func parityLaneTreeCleanGit(root, laneTree string) *bool {
	if laneTree == "" {
		return nil
	}
	rc, out := parityGitRunner(root, "status", "--porcelain", "--", laneTree)
	if rc != 0 {
		return nil
	}
	return boolPtr(strings.TrimSpace(out) == "")
}

// parityCommitSubject reads the commit subject (%s). Mirrors commit_subject.
func parityCommitSubject(root, ref string) string {
	rc, out := parityGitRunner(root, "show", "-s", "--format=%s", ref)
	if rc != 0 {
		return ""
	}
	return strings.TrimSpace(out)
}

// parityIsRealUnit is rung 1: the commit touched at least one file (not empty).
// Uses diff-tree to avoid the --name-only/-s clash `git show` trips on. Mirrors
// is_real_unit.
func parityIsRealUnit(root, ref string) bool {
	rc, out := parityGitRunner(root, "diff-tree", "--no-commit-id", "--name-only", "-r", ref)
	if rc != 0 {
		return false
	}
	for _, line := range strings.Split(out, "\n") {
		if strings.TrimSpace(line) != "" {
			return true
		}
	}
	return false
}

func strPtr(s string) *string { return &s }

// --------------------------------------------------------------------------
// Evaluate: assemble the fak-dispatch-parity-canary/1 payload.
// --------------------------------------------------------------------------

type parityPayload struct {
	Schema         string              `json:"schema"`
	Workspace      string              `json:"workspace"`
	Backend        string              `json:"backend"`
	Commit         string              `json:"commit"`
	Issue          *int                `json:"issue"`
	Subject        string              `json:"subject"`
	Log            *string             `json:"log"`
	LaneTree       *string             `json:"lane_tree"`
	ShipWitness    parityWitnessResult `json:"ship_witness"`
	Proven         bool                `json:"proven"`
	Verdict        string              `json:"verdict"`
	FailedRung     *string             `json:"failed_rung"`
	Rungs          map[string]*bool    `json:"rungs"`
	OK             bool                `json:"ok"`
	Interpretation string              `json:"interpretation"`
}

type parityEvalArgs struct {
	Commit   string
	Issue    int
	IssueSet bool
	Backend  string
	Log      string // "" ⇒ no run log
	LaneTree string // "" ⇒ unknown lane tree
}

func parityEvaluate(root string, a parityEvalArgs) parityPayload {
	shaRC, shaOut := parityGitRunner(root, "rev-parse", "--short", a.Commit)
	sha := a.Commit
	if shaRC == 0 && strings.TrimSpace(shaOut) != "" {
		sha = strings.TrimSpace(shaOut)
	}
	subject := parityCommitSubject(root, a.Commit)
	shipped := parityIsRealUnit(root, a.Commit)
	wit := parityWitnessRunner(root, a.Commit)

	var gitCmds *string
	var logField *string
	if a.Log != "" {
		lp := a.Log
		logField = &lp
		if data, err := os.ReadFile(a.Log); err == nil {
			flat := parityGitCmdsFromLog(string(data))
			gitCmds = &flat
		}
	}

	laneClean := parityLaneCleanCheck(root, a.LaneTree)

	graded := parityGradeFold(parityGradeInput{
		ShippedAUnit: shipped,
		Witnessed:    wit.Witnessed,
		Subject:      subject,
		Issue:        a.Issue,
		IssueSet:     a.IssueSet,
		GitCmds:      gitCmds,
		LaneClean:    laneClean,
	})

	var issuePtr *int
	if a.IssueSet {
		i := a.Issue
		issuePtr = &i
	}
	var laneTreePtr *string
	if a.LaneTree != "" {
		lt := a.LaneTree
		laneTreePtr = &lt
	}

	return parityPayload{
		Schema:         parityCanarySchema,
		Workspace:      root,
		Backend:        a.Backend,
		Commit:         sha,
		Issue:          issuePtr,
		Subject:        subject,
		Log:            logField,
		LaneTree:       laneTreePtr,
		ShipWitness:    wit,
		Proven:         graded.Proven,
		Verdict:        graded.Verdict,
		FailedRung:     graded.FailedRung,
		Rungs:          graded.Rungs,
		OK:             graded.Proven,
		Interpretation: parityInterpret(graded, a.Backend),
	}
}

func parityInterpret(g parityGrade, backend string) string {
	if g.Proven {
		return fmt.Sprintf("PARITY_PROVEN — the %s canary cleared every rung of the bar "+
			"(shipped a witnessed unit, bound to its issue, committed by pathspec with "+
			"sign-off, lane tree clean). Safe to gate the population raise ON this verdict.", backend)
	}
	failed := ""
	if g.FailedRung != nil {
		failed = *g.FailedRung
	}
	if g.Verdict == "PARITY_UNOBSERVED" {
		return fmt.Sprintf("PARITY_UNOBSERVED — rung '%s' has no evidence (no run log / "+
			"unknown lane tree). Parity is not refuted, but it is not proven either; supply "+
			"--log / --lane-tree before raising the population.", failed)
	}
	return fmt.Sprintf("%s — rung '%s' FAILED. The %s worker did not match the reference loop "+
		"discipline; keep opencode population at 1 and fix the gap before re-running the canary.",
		g.Verdict, failed, backend)
}

func renderParityCanary(p parityPayload) string {
	head := fmt.Sprintf("%s  (%s canary on %s", p.Verdict, p.Backend, p.Commit)
	if p.Issue != nil {
		head += fmt.Sprintf(", #%d", *p.Issue)
	}
	head += ")"
	marks := make([]string, 0, len(parityBar))
	for _, bar := range parityBar {
		v := p.Rungs[bar.rung]
		glyph := "?"
		if v != nil {
			if *v {
				glyph = "ok"
			} else {
				glyph = "MISS"
			}
		}
		marks = append(marks, fmt.Sprintf("%s=%s", bar.rung, glyph))
	}
	return head + "\n  " + strings.Join(marks, "  ") + "\n  " + p.Interpretation
}

func resolveCommandWorkspace(stderr io.Writer, label, workspace string) (string, bool) {
	if workspace != "" {
		return workspace, true
	}
	wd, err := os.Getwd()
	if err != nil {
		fmt.Fprintf(stderr, "%s: getwd: %v\n", label, err)
		return "", false
	}
	return wd, true
}

// runDispatchParityCanary is the CLI entry: exit 0 iff PARITY_PROVEN, else 1;
// usage errors exit 2. Mirrors dispatch_parity_canary.main.
func runDispatchParityCanary(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("dispatch parity-canary", flag.ContinueOnError)
	fs.SetOutput(stderr)
	commit := fs.String("commit", "", "the shipped canary commit (sha / ref) to grade")
	issue := fs.Int("issue", 0, "the issue the canary was supposed to resolve (for the #N-bound rung)")
	backend := fs.String("backend", "opencode", "which backend drove the canary")
	logPath := fs.String("log", "", "the dispatch run log (.dispatch-runs/resolve-*.log); without it the behavioral rungs report unobserved")
	laneTree := fs.String("lane-tree", "", "the lane's file-tree prefix (e.g. docs) for the tree-clean rung")
	workspace := fs.String("workspace", "", "workspace root (default: current directory)")
	asJSON := fs.Bool("json", false, "emit the fak-dispatch-parity-canary/1 JSON payload")
	if !parseFlags(fs, argv) {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintf(stderr, "fak dispatch parity-canary: unexpected argument %q\n", fs.Arg(0))
		return 2
	}
	if strings.TrimSpace(*commit) == "" {
		fmt.Fprintln(stderr, "fak dispatch parity-canary: --commit is required")
		return 2
	}

	root, ok := resolveCommandWorkspace(stderr, "fak dispatch parity-canary", *workspace)
	if !ok {
		return 1
	}

	// Match the Python's issue: int | None semantics faithfully: the #N-bound rung
	// keys off whether --issue was PROVIDED on the command line, not its value. So
	// `--issue 0` / a negative bind to that literal `#N` (exactly as Python does),
	// and only an omitted flag falls back to the any-`#N` branch. fs.Visit reports
	// only the flags actually set, which is the Go idiom for "was this passed?".
	issueSet := false
	fs.Visit(func(f *flag.Flag) {
		if f.Name == "issue" {
			issueSet = true
		}
	})

	payload := parityEvaluate(root, parityEvalArgs{
		Commit:   *commit,
		Issue:    *issue,
		IssueSet: issueSet,
		Backend:  *backend,
		Log:      *logPath,
		LaneTree: *laneTree,
	})

	if *asJSON {
		if err := writeIndentedJSON(stdout, payload); err != nil {
			fmt.Fprintf(stderr, "fak dispatch parity-canary: encode json: %v\n", err)
			return 1
		}
	} else {
		fmt.Fprintln(stdout, renderParityCanary(payload))
	}
	if payload.OK {
		return 0
	}
	return 1
}
