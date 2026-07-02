package main

// `fak affected` -- the fast inner-loop test gate. It runs `go test` for ONLY the
// packages your working-tree change can affect -- the changed packages plus every
// package that (transitively, test imports included) imports one -- instead of the full
// `go test ./...`. For a fleet editing one leaf at a time, that turns the multi-minute
// green gate into a seconds-long pre-commit check you can afford to run on every change,
// without dropping coverage on what you touched.
//
//	fak affected                 run go test on the affected packages (working tree vs HEAD)
//	fak affected --list          print the affected import paths and exit (no tests run)
//	fak affected --json          print the selection plan as JSON and exit
//	fak affected --base origin/main   select everything changed since a base ref
//	fak affected --budget 30s --report .fak/verify-loop-affected.json
//	fak affected --file internal/foo/foo.go   test a representative one-file change
//	fak affected --short -- -run TestX   pass-through flags to go test (after --)
//	fak affected --blame --mine internal/foo/foo.go   attribute each red package
//	     mine | peer-wip | peer-preexisting (#2138); exit reflects only 'mine' reds
//
// It is the impure shell over internal/affectedtests: it gathers the changed files
// (`git diff`), the import graph (`go list -json ./...`), folds them through the pure
// affectedtests.Select, and execs `go test`. `make ci` still runs the full `go test
// ./...` as the authoritative oracle -- this is the fast inner loop, not a replacement
// (see the package doc's stated limit).

import (
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/affectedtests"
	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

func cmdAffected(argv []string) { os.Exit(runAffected(os.Stdout, os.Stderr, argv)) }

var (
	affectedChangedFiles = gitChangedFiles
	affectedListGraph    = goListGraph
	affectedRunGoTest    = runAffectedGoTest
	affectedBaselineRed  = runAffectedBaselineRed
	affectedNow          = time.Now
)

// affectedPlan is the JSON view (-json): the deterministic selection the shell derived
// before running any test, so a peer can reproduce it from the same tree.
type affectedPlan struct {
	Base             string   `json:"base"`
	ChangedFiles     []string `json:"changed_files"`
	ChangedPackages  []string `json:"changed_packages"`
	SelectedPackages []string `json:"selected_packages"`
	TotalPackages    int      `json:"total_packages"`
}

type affectedRunReport struct {
	Schema           string   `json:"schema"`
	Mode             string   `json:"mode"`
	Base             string   `json:"base,omitempty"`
	ChangedFiles     []string `json:"changed_files"`
	ChangedPackages  []string `json:"changed_packages"`
	SelectedPackages []string `json:"selected_packages"`
	TotalPackages    int      `json:"total_packages"`
	Command          []string `json:"command,omitempty"`
	BudgetMS         int64    `json:"budget_ms,omitempty"`
	ElapsedMS        int64    `json:"elapsed_ms"`
	Verdict          string   `json:"verdict"`
	Reason           string   `json:"reason,omitempty"`
	// BaselineRef + Blame are the #2138 attribution evidence (--blame): each failing
	// package tagged mine | peer-wip | peer-preexisting against the clean baseline.
	BaselineRef string                `json:"baseline_ref,omitempty"`
	Blame       []affectedtests.Blame `json:"blame,omitempty"`
}

func runAffected(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak affected", flag.ContinueOnError)
	fs.SetOutput(stderr)
	base := fs.String("base", "", "git ref to diff against (default: working tree vs HEAD). e.g. --base origin/main selects everything changed since the base")
	var explicitFiles pathList
	fs.Var(&explicitFiles, "file", "repo-relative changed file to test as the representative change (repeatable; bypasses git diff)")
	list := fs.Bool("list", false, "print the affected package import paths and exit without running tests")
	asJSON := fs.Bool("json", false, "print the selection plan as JSON and exit without running tests")
	short := fs.Bool("short", false, "pass -short to go test")
	verbose := fs.Bool("v", false, "pass -v to go test")
	run := fs.String("run", "", "pass -run <regexp> to go test")
	count := fs.Int("count", 0, "pass -count <n> to go test (0 = omit; -count=1 bypasses the test cache)")
	timeout := fs.String("timeout", "", "pass -timeout <dur> to go test (e.g. 120s)")
	budget := fs.Duration("budget", 0, "fail with GATE_LATENCY_REGRESSION if the affected go test run exceeds this duration (e.g. 30s)")
	reportPath := fs.String("report", "", "write a measured verify-loop report JSON after running tests")
	blame := fs.Bool("blame", false, "on a red run, attribute each failing package mine | peer-wip | peer-preexisting (clean-baseline rerun + --mine closure, #2138); the exit code then reflects only 'mine' reds")
	var mineFiles pathList
	fs.Var(&mineFiles, "mine", "repo-relative file YOU changed (repeatable; implies --blame — a red package outside these files' affected closure is attributed peer-wip)")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	if *budget < 0 {
		fmt.Fprintln(stderr, "fak affected: --budget must be non-negative")
		return 2
	}
	passthrough := fs.Args() // anything after -- (flag stops at the first non-flag/--)
	if len(mineFiles) > 0 {
		*blame = true // declaring your own files only means anything through the attribution rung
	}

	root := repoRoot()
	start := affectedNow()

	changedFiles := append([]string(nil), explicitFiles...)
	if len(changedFiles) == 0 {
		var err error
		changedFiles, err = affectedChangedFiles(root, *base)
		if err != nil {
			fmt.Fprintf(stderr, "fak affected: discovering changed files: %v\n", err)
			return 1
		}
	}
	for i := range changedFiles {
		changedFiles[i] = repoSlash(changedFiles[i])
	}
	sort.Strings(changedFiles)

	fileToPkg, edges, total, err := affectedListGraph(root)
	if err != nil {
		fmt.Fprintf(stderr, "fak affected: %v\n", err)
		return 1
	}

	changedPkgs := affectedtests.ChangedPackages(fileToPkg, changedFiles)
	selected := affectedtests.Select(edges, changedPkgs)

	if *asJSON {
		plan := affectedPlan{
			Base:             *base,
			ChangedFiles:     changedFiles,
			ChangedPackages:  changedPkgs,
			SelectedPackages: selected,
			TotalPackages:    total,
		}
		_ = writeIndentedJSONNoEscape(stdout, plan)
		return 0
	}

	if *list {
		for _, p := range selected {
			fmt.Fprintln(stdout, p)
		}
		return 0
	}

	if len(selected) == 0 {
		fmt.Fprintf(stderr, "fak affected: no Go packages affected by the change (%d changed file(s)) -- nothing to test\n", len(changedFiles))
		if *reportPath != "" {
			rep := newAffectedRunReport(*base, changedFiles, changedPkgs, selected, total, nil, *budget, affectedNow().Sub(start), "NOOP", "no Go packages affected")
			if err := writeAffectedRunReport(*reportPath, rep); err != nil {
				fmt.Fprintf(stderr, "fak affected: write report: %v\n", err)
				return 1
			}
		}
		return 0
	}

	fmt.Fprintf(stderr, "fak affected: testing %d/%d package(s) affected by %d changed file(s)%s\n",
		len(selected), total, len(changedFiles), baseNote(*base))

	args := []string{"test"}
	if *short {
		args = append(args, "-short")
	}
	if *verbose {
		args = append(args, "-v")
	}
	if *run != "" {
		args = append(args, "-run", *run)
	}
	if *count != 0 {
		args = append(args, fmt.Sprintf("-count=%d", *count))
	}
	if *timeout != "" {
		args = append(args, "-timeout", *timeout)
	}
	args = append(args, passthrough...)
	args = append(args, selected...)

	// With --blame the test stream is additionally teed into a buffer so the
	// per-package FAIL verdict lines can be parsed for attribution; the operator still
	// sees the full output live either way.
	testOut := stdout
	var teed bytes.Buffer
	if *blame {
		testOut = io.MultiWriter(stdout, &teed)
	}
	code, runErr := affectedRunGoTest(root, args, testOut, stderr)
	elapsed := affectedNow().Sub(start)
	verdict := "OK"
	reason := ""
	if runErr != nil {
		verdict = "TEST_RUN_ERROR"
		reason = runErr.Error()
	} else if code != 0 {
		verdict = "TEST_FAILED"
		reason = fmt.Sprintf("go test exited %d", code)
	} else if *budget > 0 && elapsed > *budget {
		verdict = "GATE_LATENCY_REGRESSION"
		reason = fmt.Sprintf("elapsed %s exceeded budget %s", roundDuration(elapsed), *budget)
	}

	exitCode := code
	var blames []affectedtests.Blame
	baselineRef := *base
	if baselineRef == "" {
		baselineRef = "HEAD"
	}
	if *blame && verdict == "TEST_FAILED" {
		blames = attributeAffectedFailures(stderr, root, baselineRef, teed.String(), mineFiles, changedFiles, fileToPkg, edges)
		mineCount := 0
		for _, b := range blames {
			if b.Class == affectedtests.BlameMine {
				mineCount++
			}
			fmt.Fprintf(stderr, "fak affected blame: %s — %s: %s\n", b.Package, b.Class, b.Evidence)
		}
		if len(blames) > 0 && mineCount == 0 {
			// Every red is positively a peer's: green for THIS diff. The witness the
			// issue asked for — no stash-and-rerun cycle to prove the red isn't yours.
			verdict = "PEER_RED_ONLY"
			reason = fmt.Sprintf("all %d failing package(s) attributed to peers (peer-wip / peer-preexisting); green for your diff — make ci on the merged tree stays the oracle", len(blames))
			fmt.Fprintf(stderr, "fak affected: %s\n", reason)
			exitCode = 0
			// Exoneration clears the red, not the clock: a budget breach the TEST_FAILED
			// verdict pre-empted still fails the gate.
			if *budget > 0 && elapsed > *budget {
				verdict = "GATE_LATENCY_REGRESSION"
				reason = fmt.Sprintf("elapsed %s exceeded budget %s (all reds peer-attributed)", roundDuration(elapsed), *budget)
			}
		} else if len(blames) > 0 {
			reason = fmt.Sprintf("go test exited %d; %d/%d failing package(s) attributed mine", code, mineCount, len(blames))
		}
	}

	if *reportPath != "" {
		rep := newAffectedRunReport(*base, changedFiles, changedPkgs, selected, total, append([]string{"go"}, args...), *budget, elapsed, verdict, reason)
		if len(blames) > 0 {
			rep.BaselineRef = baselineRef
			rep.Blame = blames
		}
		if err := writeAffectedRunReport(*reportPath, rep); err != nil {
			fmt.Fprintf(stderr, "fak affected: write report: %v\n", err)
			return 1
		}
	}
	if runErr != nil {
		fmt.Fprintf(stderr, "fak affected: running go test: %v\n", runErr)
		return 1
	}
	if exitCode != 0 {
		return exitCode
	}
	if verdict == "GATE_LATENCY_REGRESSION" {
		fmt.Fprintf(stderr, "GATE_LATENCY_REGRESSION: affected test run took %s over budget %s\n", roundDuration(elapsed), *budget)
		return 1
	}
	return 0
}

// attributeAffectedFailures gathers the two evidence rungs the pure Attribute fold
// consumes (#2138): the per-package FAIL lines parsed from the teed test output, the
// affected-set closure of the caller's declared --mine files, and the clean-baseline
// rerun via the affectedBaselineRed seam. Every degradation fails CLOSED and is
// narrated: unparseable output attributes nothing (the red exit stands), an unavailable
// baseline exonerates nothing, and a --mine file that is not among the changed files
// (a typo'd path would otherwise shrink the closure and exonerate everything as
// peer-wip) voids the closure rung entirely.
func attributeAffectedFailures(stderr io.Writer, root, baselineRef, testOutput string, mineFiles, changedFiles []string, fileToPkg map[string]string, edges map[string][]string) []affectedtests.Blame {
	failing := affectedtests.FailedPackages(testOutput)
	if len(failing) == 0 {
		fmt.Fprintln(stderr, "fak affected: --blame could not parse per-package FAIL lines from the test output; keeping the red exit unattributed")
		return nil
	}
	var mineClosure map[string]bool
	if len(mineFiles) > 0 {
		changed := make(map[string]bool, len(changedFiles))
		for _, f := range changedFiles {
			changed[f] = true
		}
		mine := make([]string, len(mineFiles))
		valid := true
		for i, f := range mineFiles {
			mine[i] = repoSlash(f)
			if !changed[mine[i]] {
				fmt.Fprintf(stderr, "fak affected: --mine %s is not among the changed files; the closure rung is voided (fail-closed — a mistyped declaration must not exonerate)\n", mine[i])
				valid = false
			}
		}
		if valid {
			mineClosure = map[string]bool{}
			for _, p := range affectedtests.Select(edges, affectedtests.ChangedPackages(fileToPkg, mine)) {
				mineClosure[p] = true
			}
		}
	}
	baselineRed, baselineSeen, err := affectedBaselineRed(root, baselineRef, failing)
	if err != nil {
		fmt.Fprintf(stderr, "fak affected: baseline rerun at %s unavailable (%v); no baseline exoneration — fail-closed\n", baselineRef, err)
		baselineRed, baselineSeen = nil, nil
	}
	return affectedtests.Attribute(failing, mineClosure, baselineRed, baselineSeen, baselineRef)
}

// runAffectedBaselineRed answers "was this package ALREADY red before any working-tree
// change?" by testing pkgs at a CLEAN checkout of ref in a throwaway DETACHED worktree
// (no branch is created, nothing is committed, the worktree is removed before
// returning — the trunk stays untouched). The rerun invokes `go test` directly rather
// than through the test.ps1/WSL routing: that routing mirrors the MAIN working tree, so
// it would re-test the very tree the baseline must exclude.
//
// It returns (red, seen): the packages whose baseline verdict was FAIL, and every
// package that produced a verdict at all (FAIL or ok) — the coverage evidence that
// keeps a mine row's sentence honest for a package that does not exist at the base
// ref. FAIL-CLOSED on every degraded outcome: a go binary that cannot run, output
// carrying a harness-execution-failure marker (a blocked/unexecutable test binary
// prints FAIL lines indistinguishable from real reds — they must not exonerate), or a
// run that produced NO verdict at all (a module-load abort) each return an error, and
// the caller exonerates nothing. A plain non-zero test exit is the signal being
// measured, not an error.
func runAffectedBaselineRed(root, ref string, pkgs []string) (red, seen map[string]bool, err error) {
	tmp, err := os.MkdirTemp("", "fak-affected-baseline-")
	if err != nil {
		return nil, nil, fmt.Errorf("temp dir: %w", err)
	}
	wt := filepath.Join(tmp, "wt")
	if _, err := gitOut(root, "worktree", "add", "--detach", wt, ref); err != nil {
		os.RemoveAll(tmp)
		return nil, nil, fmt.Errorf("worktree add %s: %w", ref, err)
	}
	defer func() {
		if _, rerr := gitOut(root, "worktree", "remove", "--force", wt); rerr != nil {
			// A locked test binary can hold the dir on Windows; prune the registration
			// so no stale .git/worktrees entry lingers, then best-effort delete.
			_, _ = gitOut(root, "worktree", "prune")
		}
		os.RemoveAll(tmp)
	}()

	cmd := exec.Command("go", append([]string{"test"}, pkgs...)...)
	windowgate.ConfigureBackgroundCommand(cmd)
	cmd.Dir = wt
	var out bytes.Buffer
	cmd.Stdout = &out
	cmd.Stderr = &out
	if err := cmd.Run(); err != nil {
		if _, ok := err.(*exec.ExitError); !ok {
			return nil, nil, fmt.Errorf("baseline go test: %w", err)
		}
	}
	if marker := baselineHarnessFailure(out.String()); marker != "" {
		return nil, nil, fmt.Errorf("baseline go test could not execute test binaries (%q in output) — its FAIL lines are harness failures, not evidence", marker)
	}
	red = map[string]bool{}
	seen = map[string]bool{}
	for _, p := range affectedtests.FailedPackages(out.String()) {
		red[p] = true
		seen[p] = true
	}
	for _, p := range affectedtests.PassedPackages(out.String()) {
		seen[p] = true
	}
	if len(seen) == 0 {
		return nil, nil, fmt.Errorf("baseline go test produced no package verdicts (module-load failure?)")
	}
	return red, seen, nil
}

// baselineHarnessFailure scans baseline go-test output for the markers of a test BINARY
// that could not run at all (OS exec policy, wrong arch, missing loader). go test still
// prints a package-level FAIL for these, so without this guard a broken harness would
// masquerade as "everything was already red" and wrongly exonerate. Returns the matched
// marker, or "" when none is present.
func baselineHarnessFailure(output string) string {
	low := strings.ToLower(output)
	for _, marker := range []string{
		"could not execute",
		"fork/exec",
		"exec format error",
		"permission denied",
		"access is denied",
		"blocked by group policy",
		"operation not permitted",
	} {
		if strings.Contains(low, marker) {
			return marker
		}
	}
	return ""
}

func runAffectedGoTest(root string, args []string, stdout, stderr io.Writer) (int, error) {
	name, cmdArgs := affectedTestCommand(runtime.GOOS, args)
	cmd := exec.Command(name, cmdArgs...)
	windowgate.ConfigureBackgroundCommand(cmd)
	cmd.Dir = root
	cmd.Stdout = stdout
	cmd.Stderr = stderr
	if err := cmd.Run(); err != nil {
		if ee, ok := err.(*exec.ExitError); ok {
			return ee.ExitCode(), nil
		}
		return 1, err
	}
	return 0, nil
}

// affectedTestCommand resolves how to run `go <args>` (args[0] == "test") on the host:
// directly on non-Windows, or routed through the repo-root test.ps1 -> WSL on Windows,
// where native `go test` is OS-policy-blocked (the SAME host knowledge `fak test`
// encodes). Without this, the fast inner-loop gate execs a `go test` that the OS blocks
// on the primary dev platform, so it never ran where agents actually edit. test.ps1
// forwards its args verbatim to `go test` inside WSL, so the leading "test" verb is
// dropped (test.ps1 re-adds it). Pure so the routing is unit-testable.
func affectedTestCommand(goos string, args []string) (name string, cmdArgs []string) {
	if goos == "windows" {
		goArgs := args
		if len(goArgs) > 0 && goArgs[0] == "test" {
			goArgs = goArgs[1:]
		}
		return "powershell", append([]string{"-NoProfile", "-ExecutionPolicy", "Bypass", "-File", "test.ps1"}, goArgs...)
	}
	return "go", args
}

func newAffectedRunReport(base string, changedFiles, changedPkgs, selected []string, total int, command []string, budget, elapsed time.Duration, verdict, reason string) affectedRunReport {
	rep := affectedRunReport{
		Schema:           "fak.verify_loop.v1",
		Mode:             "incremental",
		Base:             base,
		ChangedFiles:     append([]string(nil), changedFiles...),
		ChangedPackages:  append([]string(nil), changedPkgs...),
		SelectedPackages: append([]string(nil), selected...),
		TotalPackages:    total,
		Command:          append([]string(nil), command...),
		ElapsedMS:        elapsed.Milliseconds(),
		Verdict:          verdict,
		Reason:           reason,
	}
	if budget > 0 {
		rep.BudgetMS = budget.Milliseconds()
	}
	return rep
}

func writeAffectedRunReport(path string, rep affectedRunReport) error {
	return writeIndentedJSONFile(path, rep)
}

func roundDuration(d time.Duration) time.Duration {
	if d < time.Millisecond {
		return d
	}
	return d.Round(time.Millisecond)
}

func repoSlash(path string) string {
	return strings.ReplaceAll(filepath.ToSlash(path), "\\", "/")
}

func baseNote(base string) string {
	if base == "" {
		return " (working tree vs HEAD)"
	}
	return " (since " + base + ")"
}

// gitChangedFiles returns the repo-relative, slash-separated paths changed relative to
// the base. With base == "" it is the working tree vs HEAD (staged + unstaged tracked
// changes) plus untracked files; with a base ref it is everything in the working tree
// that differs from that ref. Git already emits forward-slash paths on every platform,
// so the result keys directly into the go-list file index.
func gitChangedFiles(root, base string) ([]string, error) {
	set := map[string]bool{}
	add := func(out string) {
		for _, line := range strings.Split(out, "\n") {
			line = strings.TrimSpace(line)
			if line != "" {
				set[filepath.ToSlash(line)] = true
			}
		}
	}

	ref := base
	if ref == "" {
		ref = "HEAD"
	}
	// `git diff --name-only <ref>` compares <ref> to the WORKING TREE, so it covers both
	// committed and uncommitted differences from the ref -- "everything I'd be
	// introducing", which is what the affected set should cover.
	diff, err := gitOut(root, "diff", "--name-only", ref)
	if err != nil {
		return nil, err
	}
	add(diff)
	// Untracked-but-not-ignored files (a brand-new package the diff misses). New
	// relative to ANY ref, so include them whether diffing HEAD or an explicit base.
	others, err := gitOut(root, "ls-files", "--others", "--exclude-standard")
	if err != nil {
		return nil, err
	}
	add(others)

	out := make([]string, 0, len(set))
	for f := range set {
		out = append(out, f)
	}
	sort.Strings(out)
	return out, nil
}

func gitOut(root string, args ...string) (string, error) {
	cmd := exec.Command("git", append([]string{"-C", root}, args...)...)
	windowgate.ConfigureBackgroundCommand(cmd)
	var buf bytes.Buffer
	cmd.Stdout = &buf
	cmd.Stderr = io.Discard
	if err := cmd.Run(); err != nil {
		return "", fmt.Errorf("git %s: %w", strings.Join(args, " "), err)
	}
	return buf.String(), nil
}

// goPkg is the subset of `go list -json` fields the selector needs. The file-list
// fields (relative base names within Dir) are what map a changed FILE to its package;
// the import lists build the dependency edges.
type goPkg struct {
	ImportPath string
	Name       string
	Dir        string
	Module     *struct{ Path, Dir string }

	GoFiles           []string
	CgoFiles          []string
	CFiles            []string
	TestGoFiles       []string
	XTestGoFiles      []string
	EmbedFiles        []string
	TestEmbedFiles    []string
	XTestEmbedFiles   []string
	IgnoredGoFiles    []string // files excluded by build constraints -- still belong to the package
	IgnoredOtherFiles []string

	Imports      []string
	TestImports  []string
	XTestImports []string
}

// goListGraph runs `go list -e -json ./...` and folds it into (fileToPkg, edges, total):
//   - fileToPkg maps each package's repo-relative slash SOURCE/EMBED file to its import path.
//   - edges maps each import path to the intra-module packages it imports, INCLUDING
//     test imports (so a package whose _test.go imports the changed one is selected).
//   - total is the package count (for the "N/M selected" saving line).
//
// `-e` tolerates a package that does not compile (a peer's mid-edit tree) so the gate
// still selects sensibly instead of erroring out.
func goListGraph(root string) (fileToPkg map[string]string, edges map[string][]string, total int, err error) {
	cmd := exec.Command("go", "list", "-e", "-json", "./...")
	windowgate.ConfigureBackgroundCommand(cmd)
	cmd.Dir = root
	var out bytes.Buffer
	cmd.Stdout = &out
	cmd.Stderr = io.Discard
	// A non-zero exit with `-e` still emits valid JSON for the packages it could load;
	// only treat it as fatal if we parsed nothing.
	runErr := cmd.Run()

	fileToPkg, edges, total, err = parseGoList(&out)
	if err != nil {
		return nil, nil, 0, err
	}
	if total == 0 {
		if runErr != nil {
			return nil, nil, 0, fmt.Errorf("go list produced no packages: %w", runErr)
		}
		return nil, nil, 0, fmt.Errorf("go list produced no packages")
	}
	return fileToPkg, edges, total, nil
}

// parseGoList folds a `go list -json` object stream into (fileToPkg, edges, total). It is
// the pure half of goListGraph (no exec), so the fold is unit-testable against a fixture
// stream. The first package carrying a Module fixes the module path/dir used to make
// file paths repo-relative and to keep only intra-module import edges. Every category of
// a package's files (incl. build-constrained ones) is indexed so a change to any of them
// selects the package; a file that belongs to no package (a doc, a top-level Makefile) is
// simply absent from the index and selects nothing.
func parseGoList(r io.Reader) (fileToPkg map[string]string, edges map[string][]string, total int, err error) {
	fileToPkg = map[string]string{}
	edges = map[string][]string{}
	var modPath, modDir string

	dec := json.NewDecoder(r)
	for {
		var p goPkg
		if decErr := dec.Decode(&p); decErr != nil {
			if decErr == io.EOF {
				break
			}
			return nil, nil, 0, fmt.Errorf("parsing go list json: %w", decErr)
		}
		if p.Module != nil && modPath == "" {
			modPath, modDir = p.Module.Path, p.Module.Dir
		}
		total++
		if p.Dir != "" && modDir != "" {
			if rel, relErr := filepath.Rel(modDir, p.Dir); relErr == nil {
				relSlash := filepath.ToSlash(rel)
				for _, group := range [][]string{
					p.GoFiles, p.CgoFiles, p.CFiles, p.TestGoFiles, p.XTestGoFiles,
					p.EmbedFiles, p.TestEmbedFiles, p.XTestEmbedFiles,
					p.IgnoredGoFiles, p.IgnoredOtherFiles,
				} {
					for _, f := range group {
						key := f
						if relSlash != "." {
							key = relSlash + "/" + f
						}
						fileToPkg[key] = p.ImportPath
					}
				}
			}
		}
		var deps []string
		for _, group := range [][]string{p.Imports, p.TestImports, p.XTestImports} {
			for _, imp := range group {
				if modPath != "" && strings.HasPrefix(imp, modPath) {
					deps = append(deps, imp)
				}
			}
		}
		if len(deps) > 0 {
			edges[p.ImportPath] = deps
		}
	}
	return fileToPkg, edges, total, nil
}
