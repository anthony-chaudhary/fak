// fak validate answers the shared-trunk question that neither a live-tree build nor
// ci-preflight can answer: does the committed tip plus only my explicit uncommitted
// delta pass affected-package build/vet and tests? Examples:
//
//	fak validate --mine internal/gitgate/gate.go --mine internal/gitgate/gate_test.go
//	fak validate --ref origin/main --mine cmd/fak/new_verb.go --json
//
// Ownership is deliberately explicit and repeatable; the verb never guesses from git
// status because this checkout contains concurrent peers' tracked and untracked WIP.
package main

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/anthony-chaudhary/fak/internal/affectedtests"
	"github.com/anthony-chaudhary/fak/internal/interspersedflags"
	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

const (
	defaultValidateTimeout      = 4 * time.Minute
	validateWSLPreflightTimeout = 14500 * time.Millisecond
)

var (
	validateNow             = time.Now
	validatePhaseHook       = func(context.Context, string) {}
	validateWSLLookPath     = exec.LookPath
	validateWSLCommand      = runValidateWSLCapabilityCommand
	validateWSLCapabilities = struct {
		sync.Mutex
		byIdentity         map[string]validateWSLCapabilityVerdict
		identityByLauncher map[string]string
	}{
		byIdentity:         make(map[string]validateWSLCapabilityVerdict),
		identityByLauncher: make(map[string]string),
	}
)

type validatePhase struct {
	Name      string `json:"name"`
	Status    string `json:"status"`
	ElapsedMS int64  `json:"elapsed_ms"`
	Detail    string `json:"detail,omitempty"`
}

type validateOverlayProgress struct {
	Checked []string `json:"checked"`
	Skipped []string `json:"skipped"`
}

type validateWSLCapabilityVerdict struct {
	Status   string   `json:"status"`
	Identity string   `json:"identity,omitempty"`
	Required []string `json:"required"`
	Missing  []string `json:"missing"`
	Detail   string   `json:"detail,omitempty"`
	Cached   bool     `json:"cached"`
}

type validateResult struct {
	Schema         string                        `json:"schema"`
	Mode           string                        `json:"mode"`
	Ref            string                        `json:"ref"`
	Tip            string                        `json:"tip"`
	Mine           []string                      `json:"mine"`
	Tested         []string                      `json:"tested,omitempty"`
	Runner         string                        `json:"runner,omitempty"`
	TestRun        string                        `json:"test_run,omitempty"`
	TestScope      string                        `json:"test_scope,omitempty"`
	OK             bool                          `json:"ok"`
	Partial        bool                          `json:"partial"`
	TimedOut       bool                          `json:"timed_out"`
	Reason         string                        `json:"reason,omitempty"`
	TimeoutMS      int64                         `json:"timeout_ms"`
	ElapsedMS      int64                         `json:"elapsed_ms"`
	Phases         []validatePhase               `json:"phases"`
	SkippedPhases  []string                      `json:"skipped_phases"`
	Overlays       validateOverlayProgress       `json:"overlays"`
	WSLPreflight   *validateWSLCapabilityVerdict `json:"wsl_preflight,omitempty"`
	Failures       []ciPreflightFailure          `json:"failures"`
	SelectionAudit *validateSelectionAudit       `json:"selection_audit,omitempty"`
}

type validateSelectionAudit struct {
	Base             string   `json:"base"`
	Head             string   `json:"head"`
	SelectedPackages []string `json:"selected_packages"`
	affectedtests.SelectionAudit
}

type validateRecorder struct {
	ctx        context.Context
	stderr     io.Writer
	progress   bool
	started    time.Time
	phaseOrder []string
	res        *validateResult
}

type validateActivePhase struct {
	recorder *validateRecorder
	name     string
	started  time.Time
}

func (r *validateRecorder) start(name string) validateActivePhase {
	started := validateNow()
	if r.progress {
		fmt.Fprintf(r.stderr, "fak validate: phase=%s status=start elapsed=%s\n", name, started.Sub(r.started).Round(time.Millisecond))
	}
	validatePhaseHook(r.ctx, name)
	return validateActivePhase{recorder: r, name: name, started: started}
}

func (p validateActivePhase) finish(err error) {
	status := "ok"
	detail := ""
	switch {
	case p.recorder.ctx.Err() != nil:
		status = "timeout"
		detail = p.recorder.ctx.Err().Error()
	case err != nil:
		status = "failed"
		detail = err.Error()
	}
	p.finishAs(status, detail)
}

func (p validateActivePhase) finishAs(status, detail string) {
	elapsed := validateNow().Sub(p.started)
	if elapsed < 0 {
		elapsed = 0
	}
	p.recorder.res.Phases = append(p.recorder.res.Phases, validatePhase{
		Name: p.name, Status: status, ElapsedMS: elapsed.Milliseconds(), Detail: detail,
	})
	if p.recorder.progress {
		fmt.Fprintf(p.recorder.stderr, "fak validate: phase=%s status=%s phase_elapsed=%s total_elapsed=%s\n",
			p.name, status, elapsed.Round(time.Millisecond), validateNow().Sub(p.recorder.started).Round(time.Millisecond))
	}
}

func (r *validateRecorder) skip(name, detail string) {
	if r.progress {
		fmt.Fprintf(r.stderr, "fak validate: phase=%s status=skipped detail=%q elapsed=%s\n",
			name, detail, validateNow().Sub(r.started).Round(time.Millisecond))
	}
	r.res.Phases = append(r.res.Phases, validatePhase{Name: name, Status: "skipped", Detail: detail})
}

func (r *validateRecorder) finish() {
	elapsed := validateNow().Sub(r.started)
	if elapsed < 0 {
		elapsed = 0
	}
	r.res.ElapsedMS = elapsed.Milliseconds()
}

func cmdValidate(argv []string) { os.Exit(runValidate(os.Stdout, os.Stderr, argv)) }

// runValidate checks committed ref plus only explicitly-owned working-tree paths.
func runValidate(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("validate", flag.ContinueOnError)
	fs.SetOutput(stderr)
	root := fs.String("root", "", "repo root (default: git toplevel from cwd)")
	ref := fs.String("ref", "HEAD", "committed base ref or sha")
	asJSON := fs.Bool("json", false, "emit the result as JSON")
	timeout := fs.Duration("timeout", defaultValidateTimeout, "maximum total validation time")
	progress := fs.Bool("progress", validateWriterIsTerminal(stderr), "emit phase progress to stderr (default on when stderr is a TTY)")
	testOnly := fs.Bool("test-only", false, "skip affected-package build/vet and run only affected tests in the isolated checkout")
	wslTests := fs.Bool("wsl-tests", defaultValidateWSLTests(runtime.GOOS), "run isolated affected tests through WSL (default on Windows hosts)")
	testRun := fs.String("test-run", "", "go test -run expression for isolated affected tests")
	auditSelection := fs.Bool("audit-selection", false, "compare affected tests with a full-suite truth run")
	var mine pathList
	fs.Var(&mine, "mine", "owned changed path to overlay (repeatable; files and directories accepted)")
	positional, parseErr := interspersedflags.Parse(fs, argv)
	if parseErr != nil {
		return 2
	}
	for _, p := range positional {
		if p = strings.TrimSpace(p); p != "" {
			mine = append(mine, p)
		}
	}
	if len(mine) == 0 {
		fmt.Fprintln(stderr, "fak validate: at least one --mine path is required; ownership is never inferred from a peer-dirty tree")
		return 2
	}
	if *timeout <= 0 {
		fmt.Fprintln(stderr, "fak validate: --timeout must be greater than zero")
		return 2
	}
	mode := "full"
	if *testOnly {
		mode = "test-only"
	}
	started := validateNow()
	ctx, cancel := context.WithTimeout(context.Background(), *timeout)
	defer cancel()
	res := validateResult{
		Schema: "fak-validate/1", Mode: mode, Ref: *ref, Mine: requestedMinePaths(mine), OK: true,
		TimeoutMS: timeout.Milliseconds(), Phases: []validatePhase{}, SkippedPhases: []string{},
		Overlays: validateOverlayProgress{Checked: []string{}, Skipped: requestedMinePaths(mine)},
		Failures: []ciPreflightFailure{},
	}
	phaseOrder := validatePhaseOrder(*testOnly, *auditSelection)
	recorder := validateRecorder{ctx: ctx, stderr: stderr, progress: *progress, started: started, phaseOrder: phaseOrder, res: &res}

	phase := recorder.start("resolve_root")
	r := resolveRootWithin(ctx, *root)
	phase.finish(ctx.Err())
	if ctx.Err() != nil {
		return finishValidateTimeout(stdout, &res, &recorder, "resolve_root", *asJSON)
	}
	if r == "" {
		fmt.Fprintln(stderr, "fak validate: not in a git repo (or git unavailable)")
		return 2
	}
	phase = recorder.start("resolve_ref")
	tip, err := gitRevParseWithin(ctx, r, *ref)
	if code, failed := finishValidateRequiredPhase(stdout, stderr, &res, &recorder, phase, "resolve_ref", err, *asJSON,
		fmt.Sprintf("fak validate: cannot resolve ref %q: %v", *ref, err)); failed {
		return code
	}
	res.Tip = tip
	if runtime.GOOS == "windows" && *wslTests {
		phase = recorder.start("wsl_preflight")
		verdict := preflightValidateWSLCapabilitiesWithin(ctx)
		res.WSLPreflight = &verdict
		if ctx.Err() != nil {
			phase.finish(ctx.Err())
			return finishValidateTimeout(stdout, &res, &recorder, "wsl_preflight", *asJSON)
		}
		if verdict.Status != "ready" {
			phase.finishAs("failed", verdict.Detail)
			return finishValidateWSLCapabilityRefusal(stdout, stderr, &res, &recorder, verdict, *asJSON)
		}
		phase.finish(nil)
	} else {
		recorder.skip("wsl_preflight", "native workspace selected")
	}
	prep, code, failed := prepareValidateWorkspace(validateWorkspaceRequest{
		stdout: stdout, stderr: stderr, ctx: ctx, result: &res, recorder: &recorder,
		root: r, tip: tip, mine: mine, testRun: *testRun, wslTests: *wslTests, asJSON: *asJSON,
	})
	if failed {
		return code
	}
	paths, effectiveTestRun := prep.paths, prep.testRun
	dir, wslWorkspace := prep.dir, prep.wslWorkspace
	defer prep.cleanup()
	// Keep the base graph as well as the overlaid graph: a deleted file/package no longer
	// appears in `go list`, but its importers are still affected and must be built and vetted.
	baseFileToPkg := map[string]string{}
	baseEdges := map[string][]string{}
	if hasDeletedMinePath(r, paths) {
		phase = recorder.start("base_graph")
		baseFileToPkg, baseEdges, _, err = validateGoListGraphWithin(ctx, dir, wslWorkspace)
		if code, timedOut := finishValidatePhaseOrTimeout(stdout, &res, &recorder, phase, "base_graph", err, *asJSON); timedOut {
			return code
		}
		// Preserve the old fail-toward-running behavior: the post-overlay graph remains the
		// authoritative error, while a base graph failure merely loses deletion coverage.
		if err != nil {
			baseFileToPkg = map[string]string{}
			baseEdges = map[string][]string{}
		}
	} else {
		recorder.skip("base_graph", "no deleted owned paths")
	}
	phase = recorder.start("overlay")
	checked := func(path string) {
		res.Overlays.Checked = append(res.Overlays.Checked, path)
		res.Overlays.Skipped = subtractValidatePaths(paths, res.Overlays.Checked)
	}
	if wslWorkspace {
		err = overlayMinePathsWSLWithin(ctx, r, dir, paths, checked)
	} else {
		err = overlayMinePathsWithin(ctx, r, dir, paths, checked)
	}
	if code, failed := finishValidateRequiredPhase(stdout, stderr, &res, &recorder, phase, "overlay", err, *asJSON,
		fmt.Sprintf("fak validate: cannot overlay owned paths: %v", err)); failed {
		return code
	}
	if !*testOnly {
		if code, timedOut := runValidateGofmtPhase(ctx, stdout, &res, &recorder, r, dir, paths, wslWorkspace, *asJSON); timedOut {
			return code
		}
	}
	phase = recorder.start("list_graph")
	fileToPkg, edges, _, graphErr := validateGoListGraphWithin(ctx, dir, wslWorkspace)
	phase.finish(graphErr)
	if ctx.Err() != nil {
		return finishValidateTimeout(stdout, &res, &recorder, "list_graph", *asJSON)
	}
	if graphErr != nil {
		res.OK = false
		res.Failures = append(res.Failures, ciPreflightFailure{Step: "test-select", Detail: graphErr.Error()})
	} else {
		buildTargets := selectValidatePackages(&res, &recorder, dir, paths, fileToPkg, edges, baseFileToPkg, baseEdges)
		if !*testOnly {
			if code, timedOut := runValidateBuildAndVet(ctx, stdout, &res, &recorder, dir, wslWorkspace, *asJSON, buildTargets); timedOut {
				return code
			}
		}
		selectedObservation, code, timedOut := runValidateTestsPhase(ctx, stdout, &res, &recorder, r, dir, tip, effectiveTestRun, fileToPkg, *wslTests, wslWorkspace, *auditSelection, *asJSON)
		if timedOut {
			return code
		}
		if *auditSelection {
			if code, timedOut := runValidateAuditSelectionPhase(ctx, stdout, &res, &recorder, r, dir, tip, paths, fileToPkg, selectedObservation, *wslTests, wslWorkspace, *asJSON); timedOut {
				return code
			}
		}
	}
	recorder.finish()
	emitValidateResult(stdout, res, *asJSON)
	if !res.OK {
		return 1
	}
	return 0
}

func runValidateGofmtPhase(ctx context.Context, stdout io.Writer, res *validateResult, recorder *validateRecorder, r, dir string, paths []string, wslWorkspace, asJSON bool) (int, bool) {
	phase := recorder.start("gofmt")
	files, ferr := validateGofmtOwnedPathsWithin(ctx, r, dir, paths, wslWorkspace)
	if code, timedOut := finishValidateContextPhase(stdout, res, recorder, phase, "gofmt", asJSON); timedOut {
		return code, true
	}
	if ferr != nil {
		recordValidateFailure(res, phase, "gofmt", ferr.Error(), ferr)
	} else if len(files) > 0 {
		phase.finishAs("failed", "owned Go files are not gofmt-clean")
		res.OK = false
		res.Failures = append(res.Failures, ciPreflightFailure{Step: "gofmt", Files: files})
	} else {
		phase.finish(nil)
	}
	return 0, false
}

func runValidateBuildAndVet(ctx context.Context, stdout io.Writer, res *validateResult, recorder *validateRecorder, dir string, wslWorkspace, asJSON bool, buildTargets []string) (int, bool) {
	if len(buildTargets) == 0 {
		recorder.skip("build", "no affected package")
		recorder.skip("vet", "no affected package")
		return 0, false
	}
	// The base is a committed tip; only changed packages and their importer closure
	// can become newly red. Rebuilding ./... made two-file checks scale with the
	// entire repository and was the dominant #6568 timeout signature.
	phase := recorder.start("build")
	if code, timedOut := runValidateCheckPhase(stdout, res, recorder, phase, "build", errors.New("affected package build failed"), asJSON, func() (string, bool) {
		return validateRunGoCheckWithin(ctx, dir, wslWorkspace, validateGoCheckArgs("build", buildTargets)...)
	}); timedOut {
		return code, true
	}

	phase = recorder.start("vet")
	if code, timedOut := runValidateCheckPhase(stdout, res, recorder, phase, "vet", errors.New("affected package vet failed"), asJSON, func() (string, bool) {
		return validateRunGoCheckWithin(ctx, dir, wslWorkspace, validateGoCheckArgs("vet", buildTargets)...)
	}); timedOut {
		return code, true
	}
	return 0, false
}

func runValidateTestsPhase(ctx context.Context, stdout io.Writer, res *validateResult, recorder *validateRecorder, r, dir, tip, effectiveTestRun string, fileToPkg map[string]string, wslTests, wslWorkspace, auditSelection, asJSON bool) (affectedtests.TestObservation, int, bool) {
	if len(res.Tested) == 0 {
		recorder.skip("test", "no affected test-bearing package")
		return affectedtests.TestObservation{Complete: true, Packages: []affectedtests.PackageObservation{}}, 0, false
	}
	res.Runner = validateTestRunner(runtime.GOOS, wslTests)
	testTargets := packagePatternsForRoot(dir, res.Tested, fileToPkg)
	args := validateTestArgs(effectiveTestRun, testTargets)
	if auditSelection {
		args = validateJSONTestArgs(args)
	}
	phase := recorder.start("test")
	detail, ok := runValidateTestCommand(ctx, r, dir, tip, args, wslTests, wslWorkspace)
	var obs affectedtests.TestObservation
	if auditSelection {
		obs = parseValidateTestObservation(detail, res.Tested, ctx.Err() == nil)
	}
	if code, timedOut := finishValidateContextPhase(stdout, res, recorder, phase, "test", asJSON); timedOut {
		return obs, code, true
	}
	if ok {
		phase.finish(nil)
	} else {
		recordValidateFailure(res, phase, "test", detail, errors.New("affected tests failed"))
	}
	return obs, 0, false
}

func runValidateAuditSelectionPhase(ctx context.Context, stdout io.Writer, res *validateResult, recorder *validateRecorder, r, dir, tip string, paths []string, fileToPkg map[string]string, selectedObservation affectedtests.TestObservation, wslTests, wslWorkspace, asJSON bool) (int, bool) {
	fullPackages := validateAllPackages(fileToPkg)
	phase := recorder.start("test_audit_full")
	fullArgs := validateJSONTestArgs(validateTestArgs("", packagePatternsForRoot(dir, fullPackages, fileToPkg)))
	detail, _ := runValidateTestCommand(ctx, r, dir, tip, fullArgs, wslTests, wslWorkspace)
	fullObservation := parseValidateTestObservation(detail, fullPackages, ctx.Err() == nil)
	if code, timedOut := finishValidateContextPhase(stdout, res, recorder, phase, "test_audit_full", asJSON); timedOut {
		return code, true
	}
	audit := affectedtests.AuditSelection(selectedObservation, fullObservation)
	selectedPackages := append([]string(nil), res.Tested...)
	sort.Strings(selectedPackages)
	res.SelectionAudit = &validateSelectionAudit{
		Base: tip, Head: validateAuditHead(r, tip, paths),
		SelectedPackages: selectedPackages, SelectionAudit: audit,
	}
	if audit.Sound {
		phase.finish(nil)
	} else {
		detail := "affected-test selection was not sound"
		if !audit.Complete {
			detail = "affected-test selection audit was incomplete"
		}
		phase.finishAs("failed", detail)
		res.OK = false
		res.Failures = append(res.Failures, ciPreflightFailure{Step: "test-audit-selection", Detail: detail})
	}
	return 0, false
}

// selectValidatePackages restores deleted-path graph context, then records live
// changed packages for tests and returns the live importer closure for build/vet.
func selectValidatePackages(res *validateResult, recorder *validateRecorder, dir string, paths []string, fileToPkg map[string]string, edges map[string][]string, baseFileToPkg map[string]string, baseEdges map[string][]string) []string {
	for file, pkg := range baseFileToPkg {
		if _, exists := fileToPkg[file]; !exists {
			fileToPkg[file] = pkg
		}
	}
	for pkg, imports := range baseEdges {
		edges[pkg] = appendUniqueStrings(edges[pkg], imports...)
	}
	phase := recorder.start("test_select")
	changedPkgs := affectedtests.ChangedPackages(fileToPkg, paths)
	selected := affectedtests.Select(edges, changedPkgs)
	livePkgs := make(map[string]bool, len(fileToPkg))
	for _, pkg := range fileToPkg {
		livePkgs[pkg] = true
	}
	for _, pkg := range changedPkgs {
		if livePkgs[pkg] { // omit a package deleted by this delta
			res.Tested = append(res.Tested, pkg)
		}
	}
	var buildPkgs []string
	for _, pkg := range selected {
		if livePkgs[pkg] { // omit a package deleted by this delta
			buildPkgs = append(buildPkgs, pkg)
		}
	}
	phase.finish(nil)
	return packagePatternsForRoot(dir, buildPkgs, fileToPkg)
}

func normalizeMinePaths(root string, raw []string) ([]string, error) {
	return normalizeMinePathsWithin(context.Background(), root, raw)
}

func normalizeMinePathsWithin(ctx context.Context, root string, raw []string) ([]string, error) {
	seen := map[string]bool{}
	var out []string
	rootAbs, err := filepath.Abs(root)
	if err != nil {
		return nil, err
	}
	realRoot := rootAbs
	if resolved, rootErr := filepath.EvalSymlinks(rootAbs); rootErr == nil {
		realRoot = resolved
	}
	for _, value := range raw {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		if strings.TrimSpace(value) == "" {
			return nil, fmt.Errorf("empty --mine path")
		}
		p := value
		if !filepath.IsAbs(p) {
			p = filepath.Join(rootAbs, p)
		}
		p, err = filepath.Abs(p)
		if err != nil {
			return nil, err
		}
		rel, err := filepath.Rel(rootAbs, p)
		if err != nil || rel == ".." || strings.HasPrefix(rel, ".."+string(filepath.Separator)) {
			// If containment fails with raw paths, try with canonicalized paths
			// (handles symlinked roots such as macOS /var -> /private/var).
			realP := p
			if resolved, evalErr := filepath.EvalSymlinks(p); evalErr == nil {
				realP = resolved
			} else if parent, evalErr := filepath.EvalSymlinks(filepath.Dir(p)); evalErr == nil {
				realP = filepath.Join(parent, filepath.Base(p))
			}
			inside, relErr := filepath.Rel(realRoot, realP)
			if relErr != nil || inside == ".." || strings.HasPrefix(inside, ".."+string(filepath.Separator)) {
				return nil, fmt.Errorf("--mine path %q escapes repo root", value)
			}
			rel = inside
		}
		rel = filepath.ToSlash(filepath.Clean(rel))
		if rel == "." {
			return nil, fmt.Errorf("--mine cannot name the repo root; list owned paths explicitly")
		}
		info, statErr := os.Stat(p)
		if statErr == nil && info.IsDir() {
			walkErr := filepath.WalkDir(p, func(child string, d os.DirEntry, walkErr error) error {
				if walkErr != nil {
					return walkErr
				}
				if err := ctx.Err(); err != nil {
					return err
				}
				if d.IsDir() {
					// An explicitly named hidden/scratch directory is owned input. Hidden or
					// underscore descendants of a broader directory request are generated
					// workspace state, not source, and can dwarf the actual overlay.
					if child != p && validateSkipWalkDir(d.Name()) {
						return filepath.SkipDir
					}
					return nil
				}
				childRel, err := filepath.Rel(rootAbs, child)
				if (err != nil || childRel == ".." || strings.HasPrefix(childRel, ".."+string(filepath.Separator))) && realRoot != rootAbs {
					childRel, err = filepath.Rel(realRoot, child)
				}
				if err != nil {
					return err
				}
				childRel = filepath.ToSlash(childRel)
				if !seen[childRel] {
					seen[childRel] = true
					out = append(out, childRel)
				}
				return nil
			})
			if walkErr != nil {
				return nil, walkErr
			}
			continue
		}
		if statErr != nil && !os.IsNotExist(statErr) {
			return nil, statErr
		}
		if !seen[rel] {
			seen[rel] = true
			out = append(out, rel)
		}
	}
	sort.Strings(out)
	return out, nil
}

// overlayMinePaths copies each owned working-tree path onto the materialized tip.
//
// The containment check canonicalizes both sides, the same both-sides discipline
// dispatchWitnessSamePath uses. EvalSymlinks(src) returns a fully resolved path, so
// measuring it against an unresolved srcRoot refuses honest owned paths on every host
// whose repo root is merely reachable through a symlink — macOS puts TMPDIR under /var,
// a symlink to /private/var, and the resolved file then reads as outside its own root.
//
// Resolving the root is best-effort on purpose: when EvalSymlinks cannot canonicalize it
// the raw spelling is kept rather than the check being skipped. A raw root can only
// refuse more than a canonical one — no canonical path lies under a symlinked spelling of
// a directory — so the fallback stays on the strict side, where an uncertain containment
// check belongs. Containment stays on filepath.Rel rather than a string prefix: Rel is
// separator-aware, so /a/bc reads as outside /a/b, and it is case-insensitive on Windows.
func overlayMinePaths(srcRoot, dstRoot string, paths []string) error {
	return overlayMinePathsWithin(context.Background(), srcRoot, dstRoot, paths, nil)
}

func overlayMinePathsWithin(ctx context.Context, srcRoot, dstRoot string, paths []string, checked func(string)) error {
	realRoot := srcRoot
	if resolved, rootErr := filepath.EvalSymlinks(srcRoot); rootErr == nil {
		realRoot = resolved
	}
	for _, rel := range paths {
		if err := ctx.Err(); err != nil {
			return err
		}
		src := filepath.Join(srcRoot, filepath.FromSlash(rel))
		dst := filepath.Join(dstRoot, filepath.FromSlash(rel))
		realSrc, evalErr := filepath.EvalSymlinks(src)
		if os.IsNotExist(evalErr) {
			if removeErr := os.RemoveAll(dst); removeErr != nil {
				return removeErr
			}
			if checked != nil {
				checked(rel)
			}
			continue
		}
		if evalErr != nil {
			return evalErr
		}
		inside, relErr := filepath.Rel(realRoot, realSrc)
		if relErr != nil || inside == ".." || strings.HasPrefix(inside, ".."+string(filepath.Separator)) {
			return fmt.Errorf("owned path %q resolves outside repo root", rel)
		}
		info, err := os.Stat(src)
		if err != nil {
			return err
		}
		if err := os.MkdirAll(filepath.Dir(dst), 0o755); err != nil {
			return err
		}
		if err := copyValidateFileWithin(ctx, realSrc, dst, info.Mode().Perm()); err != nil {
			return err
		}
		if checked != nil {
			checked(rel)
		}
	}
	return nil
}

func copyValidateFileWithin(ctx context.Context, src, dst string, mode os.FileMode) error {
	in, err := os.Open(src)
	if err != nil {
		return err
	}
	defer in.Close()
	out, err := os.OpenFile(dst, os.O_CREATE|os.O_TRUNC|os.O_WRONLY, mode)
	if err != nil {
		return err
	}
	closed := false
	defer func() {
		if !closed {
			_ = out.Close()
		}
	}()
	buf := make([]byte, 64*1024)
	for {
		if err := ctx.Err(); err != nil {
			return err
		}
		n, readErr := in.Read(buf)
		if n > 0 {
			if _, err := out.Write(buf[:n]); err != nil {
				return err
			}
		}
		if readErr == io.EOF {
			break
		}
		if readErr != nil {
			return readErr
		}
	}
	if err := out.Close(); err != nil {
		return err
	}
	closed = true
	return nil
}

func gofmtOwnedPathsWithin(ctx context.Context, root string, paths []string) ([]string, error) {
	var goFiles []string
	for _, rel := range paths {
		if filepath.Ext(rel) != ".go" {
			continue
		}
		if _, err := os.Stat(filepath.Join(root, filepath.FromSlash(rel))); err == nil {
			goFiles = append(goFiles, rel)
		}
	}
	if len(goFiles) == 0 {
		return nil, nil
	}
	args := append([]string{"-l"}, goFiles...)
	cmd := windowgate.CommandContext(ctx, "gofmt", args...)
	cmd.Dir = root
	windowgate.ConfigureBackgroundCommand(cmd)
	out, err := cmd.Output()
	if err != nil {
		return nil, err
	}
	var files []string
	for _, line := range strings.Split(string(out), "\n") {
		if line = strings.TrimSpace(line); line != "" {
			files = append(files, filepath.ToSlash(line))
		}
	}
	return files, nil
}

var validateWSLCommandSurface = []struct {
	phase    string
	commands []string
}{
	{phase: "launcher", commands: []string{"wsl.exe", "bash"}},
	{phase: "test_materialize", commands: []string{"rm", "mkdir", "tar", "go"}},
	{phase: "extract_tip", commands: []string{"rm", "mkdir", "git", "tar", "mv", "ls", "tail", "xargs"}},
	{phase: "cleanup", commands: []string{"rm"}},
	{phase: "overlay", commands: []string{"pwd", "mkdir", "cp", "rm"}},
	{phase: "go_checks", commands: []string{"go", "gofmt"}},
}

func validateWSLRequiredCommandNames() []string {
	seen := make(map[string]bool)
	for _, surface := range validateWSLCommandSurface {
		for _, command := range surface.commands {
			seen[command] = true
		}
	}
	commands := make([]string, 0, len(seen))
	for command := range seen {
		commands = append(commands, command)
	}
	sort.Strings(commands)
	return commands
}

func runValidateWSLCapabilityCommand(ctx context.Context, args ...string) ([]byte, error) {
	cmd := windowgate.CommandContext(ctx, "wsl.exe", args...)
	windowgate.ConfigureBackgroundCommand(cmd)
	out, err := cmd.Output()
	if exitErr, ok := err.(*exec.ExitError); ok && len(exitErr.Stderr) > 0 {
		out = append(out, exitErr.Stderr...)
	}
	return out, err
}

func preflightValidateWSLCapabilitiesWithin(ctx context.Context) validateWSLCapabilityVerdict {
	required := validateWSLRequiredCommandNames()
	verdict := validateWSLCapabilityVerdict{Status: "ready", Required: required, Missing: []string{}}
	wslPath, err := validateWSLLookPath("wsl.exe")
	if err != nil {
		verdict.Status = "missing"
		verdict.Missing = []string{"wsl.exe"}
		verdict.Detail = "WSL capability preflight: host command wsl.exe is unavailable; repair WSL and rerun; no fallback selected"
		return verdict
	}
	validateWSLCapabilities.Lock()
	if identityKey, ok := validateWSLCapabilities.identityByLauncher[wslPath]; ok {
		if cached, found := validateWSLCapabilities.byIdentity[wslPath+"\x00"+identityKey]; found {
			validateWSLCapabilities.Unlock()
			cached.Cached = true
			cached.Required = append([]string(nil), cached.Required...)
			cached.Missing = append([]string(nil), cached.Missing...)
			return cached
		}
	}
	validateWSLCapabilities.Unlock()

	probeCtx, cancel := context.WithTimeout(ctx, validateWSLPreflightTimeout)
	defer cancel()
	out, err := validateWSLCommand(probeCtx, "bash", "-lc", validateWSLCapabilityScript(required))
	if err != nil {
		if errors.Is(probeCtx.Err(), context.DeadlineExceeded) && ctx.Err() == nil {
			verdict.Status = "unavailable"
			verdict.Detail = fmt.Sprintf("WSL capability preflight exceeded %s before workspace allocation; repair WSL and rerun; no fallback selected", validateWSLPreflightTimeout)
		} else {
			verdict.Status = "missing"
			verdict.Missing = []string{"bash"}
			verdict.Detail = fmt.Sprintf("WSL capability preflight: bash is unavailable in the selected WSL environment: %s; repair WSL and rerun; no fallback selected", validateCommandDetail(out, err))
		}
		return verdict
	}
	identityOut, missingOut, err := parseValidateWSLCapabilityOutput(out)
	if err != nil {
		verdict.Status = "unavailable"
		verdict.Detail = fmt.Sprintf("WSL capability preflight returned an invalid verdict: %v; repair WSL and rerun; no fallback selected", err)
		return verdict
	}
	identity, identityKey := validateWSLEnvironmentIdentity(identityOut)
	verdict.Identity = identity
	cacheKey := wslPath + "\x00" + identityKey
	seen := make(map[string]bool)
	for _, line := range strings.Split(string(missingOut), "\n") {
		if command := strings.TrimSpace(line); command != "" && !seen[command] {
			seen[command] = true
			verdict.Missing = append(verdict.Missing, command)
		}
	}
	sort.Strings(verdict.Missing)
	if len(verdict.Missing) > 0 {
		verdict.Status = "missing"
		verdict.Detail = fmt.Sprintf("WSL capability preflight: environment %q is missing required command(s): %s; repair the selected WSL distribution and rerun; no fallback selected", verdict.Identity, strings.Join(verdict.Missing, ", "))
	}
	validateWSLCapabilities.Lock()
	validateWSLCapabilities.byIdentity[cacheKey] = verdict
	validateWSLCapabilities.identityByLauncher[wslPath] = identityKey
	validateWSLCapabilities.Unlock()
	return verdict
}

const (
	validateWSLDistroPrefix = "__FAK_VALIDATE_DISTRO__="
	validateWSLPathPrefix   = "__FAK_VALIDATE_PATH__="
)

func parseValidateWSLCapabilityOutput(out []byte) (identity, missing []byte, err error) {
	lines := bytes.SplitN(bytes.ReplaceAll(out, []byte("\r\n"), []byte("\n")), []byte("\n"), 3)
	if len(lines) < 2 || !bytes.HasPrefix(lines[0], []byte(validateWSLDistroPrefix)) || !bytes.HasPrefix(lines[1], []byte(validateWSLPathPrefix)) {
		return nil, nil, fmt.Errorf("missing environment identity header")
	}
	distro := bytes.TrimPrefix(lines[0], []byte(validateWSLDistroPrefix))
	path := bytes.TrimPrefix(lines[1], []byte(validateWSLPathPrefix))
	identity = append(append(append([]byte(nil), distro...), '\n'), path...)
	if len(lines) == 3 {
		missing = lines[2]
	}
	return identity, missing, nil
}

func validateWSLEnvironmentIdentity(out []byte) (display, key string) {
	parts := strings.SplitN(strings.TrimSpace(string(out)), "\n", 2)
	distro := strings.TrimSpace(parts[0])
	path := ""
	if len(parts) == 2 {
		path = strings.TrimSpace(parts[1])
	}
	digest := sha256.Sum256([]byte(path))
	return fmt.Sprintf("%s@path:%x", distro, digest[:6]), distro + "\x00" + fmt.Sprintf("%x", digest[:])
}

func validateWSLCapabilityScript(required []string) string {
	var checked []string
	for _, command := range required {
		if command != "wsl.exe" {
			checked = append(checked, posixQuote(command))
		}
	}
	return "set -u; printf '" + validateWSLDistroPrefix + "%s\\n" + validateWSLPathPrefix + "%s\\n' \"${WSL_DISTRO_NAME:-unknown}\" \"${PATH:-}\"; " +
		"for cmd in " + strings.Join(checked, " ") + "; do " +
		"command -v \"$cmd\" >/dev/null 2>&1 || printf '%s\\n' \"$cmd\"; done"
}

func runValidateTestsWithin(ctx context.Context, repo, dir, tip string, args []string, wsl bool) (string, bool) {
	// The archive checkout is isolated from peer WIP. Windows defaults to WSL so test
	// binaries execute under Linux rather than the host application-control boundary.
	useWSL := runtime.GOOS == "windows" && wsl
	if wsl {
		if err := prepareValidateGitIdentityWithin(ctx, repo, dir, tip); err != nil {
			return "prepare isolated Git identity: " + err.Error(), false
		}
	}
	var cmd *exec.Cmd
	if useWSL {
		// Stream the isolated archive into WSL-native /tmp. A tar stream is cheaper
		// than per-file NTFS copies and lets Go compile/test from ext4.
		wslDir := "/tmp/fak-validate-" + filepath.Base(dir)
		cleanup := "rm -rf -- " + posixQuote(wslDir)
		command := "set -euo pipefail; trap " + posixQuote(cleanup) + " EXIT; rm -rf " + posixQuote(wslDir) + "; mkdir -p " + posixQuote(wslDir) + "; tar -cf - . | tar -xf - -C " + posixQuote(wslDir) + "; cd " + posixQuote(wslDir) + "; go"
		for _, arg := range args {
			command += " " + posixQuote(arg)
		}
		command += " && printf '\n__FAK_VALIDATE_TEST_OK__\n'"
		cmd = windowgate.CommandContext(ctx, "wsl.exe", "--cd", dir, "bash", "-lc", command)
	} else {
		cmd = windowgate.CommandContext(ctx, "go", args...)
		cmd.Dir = dir
	}
	windowgate.ConfigureBackgroundCommand(cmd)
	out, err := cmd.CombinedOutput()
	detail := strings.TrimSpace(string(out))
	if useWSL && strings.Contains(detail, "__FAK_VALIDATE_TEST_OK__") {
		detail = strings.TrimSpace(strings.ReplaceAll(detail, "__FAK_VALIDATE_TEST_OK__", ""))
		return detail, true
	}
	return detail, err == nil
}

// prepareValidateGitIdentity gives repository-aware tests the requested commit and its
// tracked-file index without attaching the isolated checkout to the peer-dirty worktree.
// The fetched object database travels with the archive into WSL and is removed with the
// surrounding temporary checkout on every return path.
func prepareValidateGitIdentityWithin(ctx context.Context, repo, dir, tip string) error {
	commands := [][]string{
		{"init", "--quiet"},
		{"fetch", "--quiet", "--no-tags", "--depth=1", repo, tip},
		{"read-tree", tip},
		{"update-ref", "--no-deref", "HEAD", tip},
	}
	for _, args := range commands {
		cmd := windowgate.CommandContext(ctx, "git", args...)
		cmd.Dir = dir
		windowgate.ConfigureBackgroundCommand(cmd)
		out, err := cmd.CombinedOutput()
		if err != nil {
			detail := strings.TrimSpace(string(out))
			if detail == "" {
				detail = err.Error()
			}
			return fmt.Errorf("git %s: %s", args[0], detail)
		}
	}
	return nil
}

func runGoCheckWithin(ctx context.Context, dir string, args ...string) (string, bool) {
	cmd := windowgate.CommandContext(ctx, "go", args...)
	cmd.Dir = dir
	windowgate.ConfigureBackgroundCommand(cmd)
	out, err := cmd.CombinedOutput()
	if err != nil {
		return strings.TrimSpace(string(out)), false
	}
	return "", true
}

func goListGraphWithin(ctx context.Context, root string) (fileToPkg map[string]string, edges map[string][]string, total int, err error) {
	cmd := windowgate.CommandContext(ctx, "go", "list", "-e", "-json", "./...")
	windowgate.ConfigureBackgroundCommand(cmd)
	cmd.Dir = root
	var out bytes.Buffer
	cmd.Stdout = &out
	cmd.Stderr = io.Discard
	runErr := cmd.Run()
	if ctx.Err() != nil {
		return nil, nil, 0, ctx.Err()
	}
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

func gitRevParseWithin(ctx context.Context, repo, ref string) (string, error) {
	cmd := windowgate.CommandContext(ctx, "git", "-C", repo, "rev-parse", ref)
	windowgate.ConfigureBackgroundCommand(cmd)
	out, err := cmd.Output()
	if err != nil {
		return "", err
	}
	return strings.TrimSpace(string(out)), nil
}

func extractCommittedTipWithin(ctx context.Context, repo, tip string) (string, error) {
	type result struct {
		dir string
		err error
	}
	done := make(chan result, 1)
	go func() {
		dir, err := extractCommittedTip(repo, tip)
		if ctx.Err() != nil && dir != "" {
			_ = os.RemoveAll(dir)
			dir = ""
		}
		done <- result{dir: dir, err: err}
	}()
	select {
	case got := <-done:
		if got.err != nil && strings.Contains(strings.ToLower(got.err.Error()), "timed out") {
			return "", context.DeadlineExceeded
		}
		return got.dir, got.err
	case <-ctx.Done():
		return "", ctx.Err()
	}
}

func wslMountPath(winPath string) (string, bool) {
	winPath = strings.TrimSpace(winPath)
	if len(winPath) < 3 || winPath[1] != ':' || (winPath[2] != '\\' && winPath[2] != '/') {
		return "", false
	}
	drive := winPath[0]
	if drive >= 'A' && drive <= 'Z' {
		drive += 'a' - 'A'
	}
	if drive < 'a' || drive > 'z' {
		return "", false
	}
	rest := strings.ReplaceAll(winPath[2:], "\\", "/")
	if !strings.HasPrefix(rest, "/") {
		rest = "/" + rest
	}
	return "/mnt/" + string(drive) + rest, true
}

func readWSLWorktreeGitDir(repo string) (string, bool) {
	dotGit := filepath.Join(repo, ".git")
	data, err := os.ReadFile(dotGit)
	if err != nil {
		return "", false
	}
	text := strings.TrimSpace(string(data))
	rest, ok := strings.CutPrefix(text, "gitdir:")
	if !ok {
		return "", false
	}
	rawDir := strings.TrimSpace(rest)
	if rawDir == "" {
		return "", false
	}
	if wsl, ok := wslMountPath(rawDir); ok {
		return wsl, true
	}
	if !filepath.IsAbs(rawDir) && !strings.Contains(rawDir, ":") {
		abs := filepath.Join(repo, rawDir)
		if wsl, ok := wslMountPath(abs); ok {
			return wsl, true
		}
	}
	return "", false
}

func extractCommittedTipWSLWithin(ctx context.Context, repo, tip string) (string, error) {
	wslDir := fmt.Sprintf("/tmp/fak-validate-%d-%d", os.Getpid(), validateNow().UnixNano())
	quotedDir := posixQuote(wslDir)
	cacheDir := "/tmp/fak-validate-cache"
	archive := cacheDir + "/" + tip + ".tar"
	temporaryArchive := archive + fmt.Sprintf(".%d.tmp", os.Getpid())
	gitFlags := ""
	if gitDir, ok := readWSLWorktreeGitDir(repo); ok {
		gitFlags = "--git-dir=" + posixQuote(gitDir) + " "
	}
	script := "set -euo pipefail; " +
		"rm -rf -- " + quotedDir + "; mkdir -p -- " + quotedDir + "; " +
		"mkdir -p -- " + posixQuote(cacheDir) + "; " +
		"trap 'rm -rf -- " + quotedDir + "; rm -f -- " + posixQuote(temporaryArchive) + "' ERR INT TERM; " +
		"if [ ! -s " + posixQuote(archive) + " ]; then " +
		"git " + gitFlags + "archive --format=tar -o " + posixQuote(temporaryArchive) + " " + posixQuote(tip) + "; " +
		"if [ ! -s " + posixQuote(archive) + " ]; then mv -- " + posixQuote(temporaryArchive) + " " + posixQuote(archive) + "; else rm -f -- " + posixQuote(temporaryArchive) + "; fi; " +
		"fi; " +
		"tar -xf " + posixQuote(archive) + " -C " + quotedDir + "; " +
		"git -C " + quotedDir + " init --quiet; " +
		"mkdir -p -- " + quotedDir + "/.git/objects/info; " +
		"git " + gitFlags + "rev-parse --path-format=absolute --git-path objects > " + quotedDir + "/.git/objects/info/alternates; " +
		"git -C " + quotedDir + " read-tree " + posixQuote(tip) + "; " +
		"git -C " + quotedDir + " update-ref --no-deref HEAD " + posixQuote(tip) + "; " +
		"ls -1t " + posixQuote(cacheDir) + "/*.tar 2>/dev/null | tail -n +4 | xargs -r rm -f --; " +
		"trap - ERR INT TERM"
	cmd := windowgate.CommandContext(ctx, "wsl.exe", "--cd", repo, "bash", "-lc", script)
	windowgate.ConfigureBackgroundCommand(cmd)
	out, err := cmd.CombinedOutput()
	if err != nil {
		cleanupValidateWSLDir(wslDir)
		detail := strings.TrimSpace(string(out))
		if ctx.Err() != nil {
			return "", ctx.Err()
		}
		if detail == "" {
			detail = err.Error()
		}
		return "", fmt.Errorf("materialize committed tip in WSL: %s", detail)
	}
	return wslDir, nil
}

func cleanupValidateWSLDir(dir string) {
	prefix := "/tmp/fak-validate-"
	rest := strings.TrimPrefix(dir, prefix)
	if rest == dir || rest == "" || strings.Contains(rest, "/") {
		return
	}
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	cmd := windowgate.CommandContext(ctx, "wsl.exe", "bash", "-lc", "rm -rf -- "+posixQuote(dir))
	windowgate.ConfigureBackgroundCommand(cmd)
	_ = cmd.Run()
}

func overlayMinePathsWSLWithin(ctx context.Context, srcRoot, wslRoot string, paths []string, checked func(string)) error {
	stage, err := os.MkdirTemp("", "fak-validate-overlay-*")
	if err != nil {
		return err
	}
	defer os.RemoveAll(stage)
	if err := overlayMinePathsWithin(ctx, srcRoot, stage, paths, nil); err != nil {
		return err
	}
	stageWSL, err := validateWSLPwdWithin(ctx, stage)
	if err != nil {
		return err
	}
	commands := []string{"set -euo pipefail"}
	for _, rel := range paths {
		dst := strings.TrimSuffix(wslRoot, "/") + "/" + filepath.ToSlash(rel)
		staged := filepath.Join(stage, filepath.FromSlash(rel))
		if _, statErr := os.Stat(staged); statErr == nil {
			parent := strings.TrimSuffix(wslRoot, "/") + "/" + filepath.ToSlash(filepath.Dir(filepath.FromSlash(rel)))
			src := strings.TrimSuffix(stageWSL, "/") + "/" + filepath.ToSlash(rel)
			commands = append(commands, "mkdir -p -- "+posixQuote(parent), "cp -- "+posixQuote(src)+" "+posixQuote(dst))
		} else if os.IsNotExist(statErr) {
			commands = append(commands, "rm -rf -- "+posixQuote(dst))
		} else {
			return statErr
		}
	}
	cmd := windowgate.CommandContext(ctx, "wsl.exe", "bash", "-lc", strings.Join(commands, "; "))
	windowgate.ConfigureBackgroundCommand(cmd)
	out, err := cmd.CombinedOutput()
	if err != nil {
		if ctx.Err() != nil {
			return ctx.Err()
		}
		return fmt.Errorf("overlay owned paths in WSL: %s", strings.TrimSpace(string(out)))
	}
	if checked != nil {
		for _, rel := range paths {
			checked(rel)
		}
	}
	return nil
}

func validateWSLPwdWithin(ctx context.Context, windowsDir string) (string, error) {
	cmd := windowgate.CommandContext(ctx, "wsl.exe", "--cd", windowsDir, "bash", "-lc", "pwd")
	windowgate.ConfigureBackgroundCommand(cmd)
	out, err := cmd.Output()
	if err != nil {
		if ctx.Err() != nil {
			return "", ctx.Err()
		}
		return "", err
	}
	return strings.TrimSpace(string(out)), nil
}

func runValidateWSLCommandWithin(ctx context.Context, root string, args ...string) ([]byte, error) {
	command := "set -euo pipefail; cd " + posixQuote(root) + "; exec"
	for _, arg := range args {
		command += " " + posixQuote(arg)
	}
	cmd := windowgate.CommandContext(ctx, "wsl.exe", "bash", "-lc", command)
	windowgate.ConfigureBackgroundCommand(cmd)
	return cmd.CombinedOutput()
}

func validateGoListGraphWithin(ctx context.Context, root string, wsl bool) (fileToPkg map[string]string, edges map[string][]string, total int, err error) {
	if !wsl {
		return goListGraphWithin(ctx, root)
	}
	out, runErr := runValidateWSLCommandWithin(ctx, root, "go", "list", "-e", "-json", "./...")
	if ctx.Err() != nil {
		return nil, nil, 0, ctx.Err()
	}
	fileToPkg, edges, total, err = parseGoList(bytes.NewReader(out))
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

func validateGofmtOwnedPathsWithin(ctx context.Context, sourceRoot, root string, paths []string, wsl bool) ([]string, error) {
	if !wsl {
		return gofmtOwnedPathsWithin(ctx, root, paths)
	}
	args := []string{"gofmt", "-l"}
	for _, rel := range paths {
		if filepath.Ext(rel) != ".go" {
			continue
		}
		if _, err := os.Stat(filepath.Join(sourceRoot, filepath.FromSlash(rel))); err == nil {
			args = append(args, filepath.ToSlash(rel))
		}
	}
	if len(args) == 2 {
		return nil, nil
	}
	gorootOut, err := runValidateWSLCommandWithin(ctx, root, "go", "env", "GOROOT")
	if err != nil {
		if ctx.Err() != nil {
			return nil, ctx.Err()
		}
		return nil, err
	}
	gofmtPath := strings.TrimSpace(string(gorootOut)) + "/bin/gofmt"
	out, err := runValidateWSLCommandWithin(ctx, root, append([]string{gofmtPath, "-l"}, args[2:]...)...)
	if err != nil {
		if ctx.Err() != nil {
			return nil, ctx.Err()
		}
		return nil, err
	}
	var files []string
	for _, line := range strings.Split(string(out), "\n") {
		if line = strings.TrimSpace(line); line != "" {
			files = append(files, filepath.ToSlash(line))
		}
	}
	return files, nil
}

func validateRunGoCheckWithin(ctx context.Context, root string, wsl bool, args ...string) (string, bool) {
	if !wsl {
		return runGoCheckWithin(ctx, root, args...)
	}
	out, err := runValidateWSLCommandWithin(ctx, root, append([]string{"go"}, args...)...)
	if err != nil {
		return strings.TrimSpace(string(out)), false
	}
	return "", true
}

func runValidateTestsInWSLWorkspaceWithin(ctx context.Context, root string, args []string) (string, bool) {
	out, err := runValidateWSLCommandWithin(ctx, root, append([]string{"go"}, args...)...)
	return strings.TrimSpace(string(out)), err == nil
}

func runValidateTestCommand(ctx context.Context, repo, dir, tip string, args []string, wsl, wslWorkspace bool) (string, bool) {
	if wslWorkspace {
		return runValidateTestsInWSLWorkspaceWithin(ctx, dir, args)
	}
	return runValidateTestsWithin(ctx, repo, dir, tip, args, wsl)
}

func finishValidateTimeout(stdout io.Writer, res *validateResult, recorder *validateRecorder, phase string, asJSON bool) int {
	res.OK = false
	res.Partial = true
	res.TimedOut = true
	res.Reason = "TIMEOUT"
	res.Failures = append(res.Failures, ciPreflightFailure{
		Step: phase, Detail: fmt.Sprintf("validation timeout after %s", time.Duration(res.TimeoutMS)*time.Millisecond),
	})
	res.Overlays.Skipped = subtractValidatePaths(res.Mine, res.Overlays.Checked)
	completed := make(map[string]bool, len(res.Phases))
	for _, timing := range res.Phases {
		completed[timing.Name] = true
	}
	for _, name := range recorder.phaseOrder {
		if !completed[name] {
			res.SkippedPhases = append(res.SkippedPhases, name)
		}
	}
	recorder.finish()
	emitValidateResult(stdout, *res, asJSON)
	return 1
}

func finishValidateWSLCapabilityRefusal(stdout, stderr io.Writer, res *validateResult, recorder *validateRecorder, verdict validateWSLCapabilityVerdict, asJSON bool) int {
	reason := "WSL_CAPABILITY_MISSING"
	if verdict.Status == "unavailable" {
		reason = "WSL_CAPABILITY_PREFLIGHT_FAILED"
	}
	res.OK = false
	res.Partial = true
	res.Reason = reason
	res.Failures = append(res.Failures, ciPreflightFailure{Step: "wsl_preflight", Detail: verdict.Detail})
	completed := make(map[string]bool, len(res.Phases))
	for _, timing := range res.Phases {
		completed[timing.Name] = true
	}
	for _, name := range recorder.phaseOrder {
		if !completed[name] {
			res.SkippedPhases = append(res.SkippedPhases, name)
		}
	}
	recorder.finish()
	emitValidateResult(stdout, *res, asJSON)
	fmt.Fprintln(stderr, "fak validate: "+verdict.Detail)
	return 2
}

func emitValidateResult(stdout io.Writer, res validateResult, asJSON bool) {
	if asJSON {
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		_ = enc.Encode(res)
		return
	}
	renderValidate(stdout, res)
}

func renderValidate(w io.Writer, res validateResult) {
	if res.TimedOut {
		fmt.Fprintf(w, "PARTIAL: validation timed out after %s during %s\n",
			time.Duration(res.ElapsedMS)*time.Millisecond, validateTimeoutPhase(res))
		fmt.Fprintf(w, "  overlays checked: %d; skipped: %d\n", len(res.Overlays.Checked), len(res.Overlays.Skipped))
		if len(res.SkippedPhases) > 0 {
			fmt.Fprintf(w, "  phases not run: %s\n", strings.Join(res.SkippedPhases, ", "))
		}
		return
	}
	if res.OK {
		writeValidateTestContext(w, res)
		if res.Mode == "test-only" {
			fmt.Fprintf(w, "OK: committed tip %s + %d owned path(s) changed-package tests clean (isolated test-only mode)\n", short(res.Tip), len(res.Mine))
		} else {
			fmt.Fprintf(w, "OK: committed tip %s + %d owned path(s) importer build/vet and changed-package tests clean\n", short(res.Tip), len(res.Mine))
		}
		return
	}
	writeValidateTestContext(w, res)
	fmt.Fprintf(w, "RED: committed tip %s + owned delta failed\n", short(res.Tip))
	for _, f := range res.Failures {
		fmt.Fprintf(w, "  %s", f.Step)
		if f.Detail != "" {
			fmt.Fprintf(w, ": %s", f.Detail)
		}
		if len(f.Files) > 0 {
			fmt.Fprintf(w, ": %s", strings.Join(f.Files, ", "))
		}
		fmt.Fprintln(w)
	}
}
