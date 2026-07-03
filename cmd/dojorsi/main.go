// Command dojorsi is the dojo-RSI loop's WORKTREE ARM — Phase 2 of
// docs/fak/dojo-rsi-loop.md (issue #1024). Where `fak dojo-rsi loop` is the
// pure-core proposer + self-scoring replay (Phase 1, mutates nothing), this
// binary DERIVES every keep-bit witness from a real measurement it runs itself:
// it forks a detached git worktree off the pinned baseline SHA, rewrites one
// claim(...) literal in internal/dojo/claims.go via the anchored rewriter, and
// re-measures FoldCalibrable by actually running `fak dojo run --json` over a
// real corpus, split into two disjoint shards for the anti-overfitting gate.
//
// It is the dojo twin of cmd/rsiloop: the same non-forgeable keep-bit
// (internal/shipgate) folded through the same rsiloop engine, pointed at the
// dojo's calibration metric instead of an LRU-hit-rate probe.
//
// Modes:
//
//	-mode improve  one RECALIBRATE cycle: read a dojo report, pick the worst
//	               RECALIBRATE cell, measure it in a worktree, keep-or-revert on
//	               the non-forgeable keep-bit (strict full-corpus gain AND both
//	               shards drop AND a real go suite is green AND truth-clean).
//	-mode track    record ONE baseline FoldCalibrable measurement on main.
//
//	-apply         after an improve cycle banks a KEEP, land the recalibrated
//	               literal on the shared trunk by explicit path
//	               (`git commit -- internal/dojo/claims.go`, never `git add -A`).
//	               Refuses with a structured reason unless the cycle banked a
//	               real KEEP — the non-forgeable gate: a keep requires a real
//	               SuiteGreen, so an apply can never land an unwitnessed swap.
//
// Exit codes: 0 = normal, 1 = error, 3 = ESCALATE (breaker tripped) or, in track
// mode, a detected regression on main.
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/dojo"
	"github.com/anthony-chaudhary/fak/internal/dojocal"
	"github.com/anthony-chaudhary/fak/internal/rsiloop"
	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

func main() {
	mode := flag.String("mode", "improve", "improve | track")
	repo := flag.String("repo", ".", "the fak module root (where go.mod lives)")
	reportPath := flag.String("report", "", "dojo report JSON from `fak dojo run --json` (required for improve)")
	corpus := flag.String("corpus", "", "directory of .jsonl transcripts for `fak dojo run` (required for improve/track)")
	journalPath := flag.String("journal", "-", "append-only JSONL journal path ('-' = stdout)")
	baselineRef := flag.String("baseline-ref", "main", "the ref the baseline + candidate fork from")
	cellSel := flag.String("cell", "", "target a specific RECALIBRATE cell as lever/metric (default: the worst RECALIBRATE)")
	suitePkgs := flag.String("suite-pkgs", "./internal/dojo/... ./internal/dojocal/...", "package pattern the suite-green gate builds+vets / tests")
	wslTest := flag.Bool("wsl-test", false, "use a real `wsl go test` (not the native build+vet proxy) for SuiteGreen — required for a witness-grade KEEP on this Windows host")
	k := flag.Int("k", 3, "escalation breaker: stop after K consecutive non-keeps")
	apply := flag.Bool("apply", false, "land a KEPT recalibration by-path on internal/dojo/claims.go (never `git add -A`); refuses without a real KEEP")
	applySubject := flag.String("apply-subject", "", "override the apply commit subject (default: a Conventional-Commits recalibration subject)")
	ttl := flag.String("ttl", "", "extra `--ttl` (5m|1h) passed through to `fak dojo run`")
	leverSel := flag.String("lever", "", "extra `--lever a,b` passed through to `fak dojo run`")
	flag.Parse()

	h, herr := buildHarness(*mode, *repo, *baselineRef, *corpus, *reportPath, *cellSel, *suitePkgs, *wslTest, *ttl, *leverSel)
	if herr != nil {
		fmt.Fprintln(os.Stderr, "dojorsi:", herr)
		os.Exit(2)
	}

	j, err := rsiloop.NewJournal(*journalPath)
	if err != nil {
		fmt.Fprintln(os.Stderr, "dojorsi: journal:", err)
		os.Exit(1)
	}
	defer j.Close()

	switch *mode {
	case "track":
		os.Exit(runTrack(h, j, *journalPath))
	case "improve":
		code, keptRow := runImprove(h, j, *k)
		if *apply {
			os.Exit(doApply(keptRow, *repo, *applySubject))
		}
		os.Exit(code)
	default:
		fmt.Fprintf(os.Stderr, "dojorsi: unknown -mode %q (want improve|track)\n", *mode)
		os.Exit(2)
	}
}

// buildHarness constructs the worktree harness for improve, or a track-only
// harness (no candidate needed). It reads the report, picks the worst
// RECALIBRATE cell, and wires the suite.
func buildHarness(mode, repo, baselineRef, corpus, reportPath, cellSel, suitePkgs string, wslTest bool, ttl, leverSel string) (rsiloop.Harness, error) {
	if corpus == "" {
		return rsiloop.Harness{}, fmt.Errorf("-corpus DIR is required (a directory of .jsonl transcripts)")
	}
	suiteCmds := defaultSuiteCmds(suitePkgs)
	if wslTest {
		suiteCmds = wslTestSuiteCmds(suitePkgs)
	}
	cfg := dojocal.WorktreeConfig{
		Repo:        repo,
		BaselineRef: baselineRef,
		Corpus:      corpus,
		SuitePkgs:   suitePkgs,
		SuiteCmds:   suiteCmds,
	}
	if ttl != "" {
		cfg.DojoArgs = append(cfg.DojoArgs, "--ttl", ttl)
	}
	if leverSel != "" {
		cfg.DojoArgs = append(cfg.DojoArgs, "--lever", leverSel)
	}
	if mode == "track" {
		// Track measures only the baseline; no candidate is exercised.
		return dojocal.NewWorktreeHarness(cfg), nil
	}
	if reportPath == "" {
		return rsiloop.Harness{}, fmt.Errorf("-report FILE is required for -mode improve (run `fak dojo run --json --corpus %s` first)", corpus)
	}
	wc, err := pickCandidate(reportPath, cellSel)
	if err != nil {
		return rsiloop.Harness{}, err
	}
	cfg.Candidate = wc
	return dojocal.NewWorktreeHarness(cfg), nil
}

// pickCandidate reads the dojo report, proposes recalibrations, and returns the
// worst RECALIBRATE cell (or the --cell named one) as the WorktreeCandidate to
// measure. Non-RECALIBRATE cells are refused: the worktree arm measures only a
// mechanical recalibration; REPROJECT/HARVEST/floor route to the agent arm.
func pickCandidate(reportPath, cellSel string) (dojocal.WorktreeCandidate, error) {
	r, err := readReport(reportPath)
	if err != nil {
		return dojocal.WorktreeCandidate{}, err
	}
	payload := dojocal.ProposeRecals(r)
	for _, c := range payload.Candidates {
		if sel := strings.TrimSpace(cellSel); sel != "" {
			l, m, found := strings.Cut(sel, "/")
			if !found {
				return dojocal.WorktreeCandidate{}, fmt.Errorf("--cell %q must be lever/metric", sel)
			}
			if c.Lever == l && c.Metric == m {
				if c.Kind != dojocal.RecalibrateKind {
					return dojocal.WorktreeCandidate{}, fmt.Errorf("%s/%s is %s, not RECALIBRATE — %s", l, m, c.Kind, c.Reason)
				}
				return dojocal.WorktreeCandidate{Lever: c.Lever, Metric: c.Metric, NewClaimed: c.NewClaimed}, nil
			}
			continue
		}
		if c.Kind == dojocal.RecalibrateKind {
			return dojocal.WorktreeCandidate{Lever: c.Lever, Metric: c.Metric, NewClaimed: c.NewClaimed}, nil
		}
	}
	return dojocal.WorktreeCandidate{}, fmt.Errorf("no RECALIBRATE candidate in the report (the corpus has no over-claiming estimate lever to recalibrate; see `fak dojo-rsi propose`)")
}

func readReport(path string) (dojo.Report, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return dojo.Report{}, fmt.Errorf("read -report: %w", err)
	}
	var r dojo.Report
	if err := json.Unmarshal(b, &r); err == nil && r.Schema == dojo.Schema {
		return r, nil
	}
	var env struct {
		Report dojo.Report `json:"report"`
	}
	if err := json.Unmarshal(b, &env); err == nil && env.Report.Schema == dojo.Schema {
		return env.Report, nil
	}
	return dojo.Report{}, fmt.Errorf("-report is not a dojo report JSON")
}

// defaultSuiteCmds is the native build+vet proxy — sound, but NOT a witness-grade
// suite (it does not run tests). A KEEP taken against this suite lands a
// recalibration proven only to compile; a witness-grade KEEP requires -wsl-test.
func defaultSuiteCmds(suitePkgs string) [][]string {
	pkgs := strings.Fields(suitePkgs)
	if len(pkgs) == 0 {
		pkgs = []string{"./..."}
	}
	build := append([]string{"go", "build"}, pkgs...)
	vet := append([]string{"go", "vet"}, pkgs...)
	return [][]string{build, vet}
}

// wslTestSuiteCmds runs a real `wsl go test` over the suite packages — the
// witness-grade SuiteGreen source on this Windows host (native go test is blocked
// by OS app-control per AGENTS.md).
func wslTestSuiteCmds(suitePkgs string) [][]string {
	pkgs := strings.Fields(suitePkgs)
	if len(pkgs) == 0 {
		pkgs = []string{"./..."}
	}
	args := append([]string{"go", "test", "-short", "-count=1"}, pkgs...)
	return [][]string{args}
}

// runImprove drives ONE worktree cycle and prints the verdict. It returns the
// exit code and the kept row (zero value if nothing kept) so -apply can gate on it.
func runImprove(h rsiloop.Harness, j *rsiloop.Journal, k int) (int, rsiloop.Row) {
	res, err := rsiloop.Run(h, j, k, 1) // maxCycles=1: one RECALIBRATE
	if err != nil {
		fmt.Fprintln(os.Stderr, "dojorsi:", err)
		return 1, rsiloop.Row{}
	}
	if len(res.Rows) == 0 {
		fmt.Println("dojorsi: no candidate measured")
		return 0, rsiloop.Row{}
	}
	r := res.Rows[0]
	fmt.Printf("baseline %s@%s = %.6f\n", h.MetricName, res.BaselineRef, r.Baseline)
	fmt.Printf("  %-40s base=%.6f cand=%.6f improved=%v suite=%v truth=%v -> %s (kept=%v, breaker=%d)\n",
		r.Candidate, r.Baseline, r.Candidate_, r.Improved, r.SuiteGreen, r.TruthClean,
		r.Decision, r.Kept, r.BreakerCount)
	if r.Note != "" {
		fmt.Printf("  %s\n", r.Note)
	}
	if res.Escalated {
		return 3, rsiloop.Row{}
	}
	if r.Kept {
		return 0, r
	}
	return 0, rsiloop.Row{}
}

// runTrack records one main measurement and compares it to the last recorded one.
func runTrack(h rsiloop.Harness, j *rsiloop.Journal, journalPath string) int {
	var prev rsiloop.Row
	havePrev := false
	if journalPath != "-" && journalPath != "" {
		p, ok, err := rsiloop.LastTrack(journalPath)
		if err != nil {
			fmt.Fprintln(os.Stderr, "warning: could not read prior track point:", err)
		}
		prev, havePrev = p, ok
	}
	row, err := rsiloop.Track(h, j)
	if err != nil {
		fmt.Fprintln(os.Stderr, "dojorsi track:", err)
		return 1
	}
	fmt.Printf("track %s@%s = %.6f\n", row.MetricName, row.BaselineRef, row.Baseline)
	if !havePrev {
		fmt.Println("  (no prior track point — this is the first)")
		return 0
	}
	if prev.RefName != row.RefName {
		fmt.Printf("  prior point was measured @%s, this run @%s — refs differ, skipping regression verdict\n",
			prev.RefName, row.RefName)
		return 0
	}
	regressed := row.Baseline > prev.Baseline // LowerBetter: a rise is a regression
	delta := row.Baseline - prev.Baseline
	fmt.Printf("  vs last (%.6f): delta=%+.6f regressed=%v\n", prev.Baseline, delta, regressed)
	if regressed {
		return 3
	}
	return 0
}

// doApply lands a KEPT recalibration on the shared trunk by explicit path. It is
// the gated auto-land: it REFUSES unless the improve cycle banked a real KEEP
// (which itself required a real SuiteGreen + two-shard gate + truth-clean). It
// rewrites the one anchored literal in the real working tree's claims.go and
// commits ONLY that path (`git commit -- internal/dojo/claims.go`, never -A).
//
// The gate is non-forgeable: a KEEP cannot be banked without a green suite, so an
// apply can never land an unwitnessed swap. A kept row whose Candidate does not
// carry a recalibration payload is refused outright.
func doApply(kept rsiloop.Row, repo, subject string) int {
	if !kept.Kept {
		fmt.Fprintln(os.Stderr, "dojorsi -apply: REFUSE — the cycle did not bank a KEEP; nothing to land (a keep requires a strict two-shard gain + a green suite + a clean tree)")
		return 1
	}
	wc, err := candidateFromRow(kept)
	if err != nil {
		fmt.Fprintf(os.Stderr, "dojorsi -apply: REFUSE — %v\n", err)
		return 1
	}
	root, _ := filepath.Abs(repo)
	claimsPath := filepath.Join(root, filepath.FromSlash(dojocal.ClaimsRelPath))
	src, err := os.ReadFile(claimsPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "dojorsi -apply: read claims registry: %v\n", err)
		return 1
	}
	out, oldVal, err := dojocal.RewriteClaim(src, wc.Lever, wc.Metric, wc.NewClaimed)
	if err != nil {
		fmt.Fprintf(os.Stderr, "dojorsi -apply: %v\n", err)
		return 1
	}
	if err := os.WriteFile(claimsPath, out, 0o644); err != nil {
		fmt.Fprintf(os.Stderr, "dojorsi -apply: write claims registry: %v\n", err)
		return 1
	}
	if subject == "" {
		subject = fmt.Sprintf("fix(dojo): recalibrate %s/%s claim %.3g -> %.3g (#1024) (fak dojo)", wc.Lever, wc.Metric, oldVal, wc.NewClaimed)
	}
	if err := commitByPath(root, dojocal.ClaimsRelPath, subject); err != nil {
		fmt.Fprintf(os.Stderr, "dojorsi -apply: commit by-path failed (the literal IS written to %s but NOT committed; resolve manually): %v\n", dojocal.ClaimsRelPath, err)
		return 1
	}
	fmt.Printf("dojorsi -apply: LANDED %s/%s %.3g -> %.3g on %s\n", wc.Lever, wc.Metric, oldVal, wc.NewClaimed, dojocal.ClaimsRelPath)
	return 0
}

// candidateFromRow reconstructs the WorktreeCandidate from a kept journal row's
// Candidate label ("RECALIBRATE lever/metric -> value"). It refuses a row whose
// label is not a recalibration so an apply can never act on a routed verdict.
func candidateFromRow(r rsiloop.Row) (dojocal.WorktreeCandidate, error) {
	label := r.Candidate
	const prefix = "RECALIBRATE "
	if !strings.HasPrefix(label, prefix) {
		return dojocal.WorktreeCandidate{}, fmt.Errorf("kept row %q is not a RECALIBRATE — only a mechanical recalibration may be applied", label)
	}
	rest := strings.TrimPrefix(label, prefix)
	// "lever/metric -> value"
	left, value, found := strings.Cut(rest, " -> ")
	if !found {
		return dojocal.WorktreeCandidate{}, fmt.Errorf("could not parse value from kept row %q", label)
	}
	l, m, found := strings.Cut(left, "/")
	if !found {
		return dojocal.WorktreeCandidate{}, fmt.Errorf("could not parse lever/metric from kept row %q", label)
	}
	v, err := parseFloat(value)
	if err != nil {
		return dojocal.WorktreeCandidate{}, fmt.Errorf("kept row value %q: %w", value, err)
	}
	return dojocal.WorktreeCandidate{Lever: l, Metric: m, NewClaimed: v}, nil
}

// commitByPath stages exactly one path and commits it with -s (DCO), mirroring
// the repo's by-path discipline. It NEVER uses `git add -A`: on the shared
// multi-session trunk that would sweep a peer's uncommitted files into the
// commit. The fak hooks (trunk guard, file admission) run as normal git hooks.
func commitByPath(root, relPath, subject string) error {
	add := exec.Command("git", "-C", root, "add", "--", relPath)
	windowgate.ConfigureBackgroundCommand(add)
	if out, err := add.CombinedOutput(); err != nil {
		return fmt.Errorf("git add -- %s: %v: %s", relPath, err, out)
	}
	commit := exec.Command("git", "-C", root, "commit", "-s", "-m", subject, "--", relPath)
	windowgate.ConfigureBackgroundCommand(commit)
	if out, err := commit.CombinedOutput(); err != nil {
		return fmt.Errorf("git commit: %v: %s", err, out)
	}
	return nil
}

// parseFloat parses a float that may carry a trailing format suffix.
func parseFloat(s string) (float64, error) {
	s = strings.TrimSpace(s)
	var v float64
	if _, err := fmt.Sscanf(s, "%g", &v); err != nil {
		return 0, err
	}
	return v, nil
}
