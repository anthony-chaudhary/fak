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
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"runtime"
	"sort"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/affectedtests"
	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

const defaultValidateTimeout = 4 * time.Minute

var (
	validateNow       = time.Now
	validatePhaseHook = func(context.Context, string) {}
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

type validateResult struct {
	Schema        string                  `json:"schema"`
	Mode          string                  `json:"mode"`
	Ref           string                  `json:"ref"`
	Tip           string                  `json:"tip"`
	Mine          []string                `json:"mine"`
	Tested        []string                `json:"tested,omitempty"`
	Runner        string                  `json:"runner,omitempty"`
	TestRun       string                  `json:"test_run,omitempty"`
	TestScope     string                  `json:"test_scope,omitempty"`
	OK            bool                    `json:"ok"`
	Partial       bool                    `json:"partial"`
	TimedOut      bool                    `json:"timed_out"`
	Reason        string                  `json:"reason,omitempty"`
	TimeoutMS     int64                   `json:"timeout_ms"`
	ElapsedMS     int64                   `json:"elapsed_ms"`
	Phases        []validatePhase         `json:"phases"`
	SkippedPhases []string                `json:"skipped_phases"`
	Overlays      validateOverlayProgress `json:"overlays"`
	Failures      []ciPreflightFailure    `json:"failures"`
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
	var mine pathList
	fs.Var(&mine, "mine", "owned changed path to overlay (repeatable; files and directories accepted)")
	if !parseFlags(fs, argv) {
		return 2
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
	phaseOrder := validatePhaseOrder(*testOnly)
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
	// appears in `go list`, but its importers are still affected and must be tested.
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
		phase = recorder.start("gofmt")
		files, ferr := validateGofmtOwnedPathsWithin(ctx, r, dir, paths, wslWorkspace)
		if code, timedOut := finishValidateContextPhase(stdout, &res, &recorder, phase, "gofmt", *asJSON); timedOut {
			return code
		}
		if ferr != nil {
			recordValidateFailure(&res, phase, "gofmt", ferr.Error(), ferr)
		} else if len(files) > 0 {
			phase.finishAs("failed", "owned Go files are not gofmt-clean")
			res.OK = false
			res.Failures = append(res.Failures, ciPreflightFailure{Step: "gofmt", Files: files})
		} else {
			phase.finish(nil)
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
		for file, pkg := range baseFileToPkg {
			if _, exists := fileToPkg[file]; !exists {
				fileToPkg[file] = pkg
			}
		}
		for pkg, imports := range baseEdges {
			edges[pkg] = appendUniqueStrings(edges[pkg], imports...)
		}
		phase = recorder.start("test_select")
		changedPkgs := affectedtests.ChangedPackages(fileToPkg, paths)
		selected := affectedtests.Select(edges, changedPkgs)
		livePkgs := make(map[string]bool, len(fileToPkg))
		for _, pkg := range fileToPkg {
			livePkgs[pkg] = true
		}
		for _, pkg := range selected {
			if livePkgs[pkg] { // omit a package deleted by this delta
				res.Tested = append(res.Tested, pkg)
			}
		}
		phase.finish(nil)
		testTargets := packagePatternsForRoot(dir, res.Tested, fileToPkg)
		if !*testOnly && len(testTargets) > 0 {
			// The base is a committed tip; only changed packages and their importer closure
			// can become newly red. Rebuilding ./... made two-file checks scale with the
			// entire repository and was the dominant #6568 timeout signature.
			phase = recorder.start("build")
			if code, timedOut := runValidateCheckPhase(stdout, &res, &recorder, phase, "build", errors.New("affected package build failed"), *asJSON, func() (string, bool) {
				return validateRunGoCheckWithin(ctx, dir, wslWorkspace, append([]string{"build"}, testTargets...)...)
			}); timedOut {
				return code
			}

			phase = recorder.start("vet")
			if code, timedOut := runValidateCheckPhase(stdout, &res, &recorder, phase, "vet", errors.New("affected package vet failed"), *asJSON, func() (string, bool) {
				return validateRunGoCheckWithin(ctx, dir, wslWorkspace, append([]string{"vet"}, testTargets...)...)
			}); timedOut {
				return code
			}
		} else if !*testOnly {
			recorder.skip("build", "no affected package")
			recorder.skip("vet", "no affected package")
		}
		if len(res.Tested) > 0 {
			res.Runner = validateTestRunner(runtime.GOOS, *wslTests)
			args := []string{"test"}
			if effectiveTestRun != "" {
				args = append(args, "-run", effectiveTestRun)
			}
			args = append(args, testTargets...)
			phase = recorder.start("test")
			if code, timedOut := runValidateCheckPhase(stdout, &res, &recorder, phase, "test", errors.New("affected tests failed"), *asJSON, func() (string, bool) {
				if wslWorkspace {
					return runValidateTestsInWSLWorkspaceWithin(ctx, dir, args)
				}
				return runValidateTestsWithin(ctx, r, dir, tip, args, *wslTests)
			}); timedOut {
				return code
			}
		} else {
			recorder.skip("test", "no affected test-bearing package")
		}
	}
	recorder.finish()
	emitValidateResult(stdout, res, *asJSON)
	if !res.OK {
		return 1
	}
	return 0
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
			return nil, fmt.Errorf("--mine path %q escapes repo root", value)
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

func validateSkipWalkDir(name string) bool {
	return strings.HasPrefix(name, ".") || strings.HasPrefix(name, "_")
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

func packagePatternsForRoot(root string, packages []string, fileToPkg map[string]string) []string {
	pkgDirs := make(map[string]string)
	for file, pkg := range fileToPkg {
		dir := filepath.ToSlash(filepath.Dir(file))
		if dir == "." {
			dir = ""
		}
		if old, ok := pkgDirs[pkg]; !ok || len(dir) < len(old) {
			pkgDirs[pkg] = dir
		}
	}
	out := make([]string, 0, len(packages))
	for _, pkg := range packages {
		if dir, ok := pkgDirs[pkg]; ok {
			if dir == "" {
				out = append(out, ".")
			} else {
				out = append(out, "./"+dir)
			}
			continue
		}
		out = append(out, pkg)
	}
	return out
}

func appendUniqueStrings(dst []string, values ...string) []string {
	seen := make(map[string]bool, len(dst)+len(values))
	for _, value := range dst {
		seen[value] = true
	}
	for _, value := range values {
		if !seen[value] {
			seen[value] = true
			dst = append(dst, value)
		}
	}
	return dst
}

func defaultValidateWSLTests(goos string) bool { return goos == "windows" }

func validateTestRunner(goos string, wsl bool) string {
	if goos == "windows" && wsl {
		return "wsl.exe bash -lc go test"
	}
	return "go test"
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

func posixQuote(s string) string {
	return "'" + strings.ReplaceAll(s, "'", "'\"'\"'") + "'"
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

func extractCommittedTipWSLWithin(ctx context.Context, repo, tip string) (string, error) {
	wslDir := fmt.Sprintf("/tmp/fak-validate-%d-%d", os.Getpid(), validateNow().UnixNano())
	quotedDir := posixQuote(wslDir)
	cacheDir := "/tmp/fak-validate-cache"
	archive := cacheDir + "/" + tip + ".tar"
	temporaryArchive := archive + fmt.Sprintf(".%d.tmp", os.Getpid())
	script := "set -euo pipefail; " +
		"rm -rf -- " + quotedDir + "; mkdir -p -- " + quotedDir + "; " +
		"mkdir -p -- " + posixQuote(cacheDir) + "; " +
		"trap 'rm -rf -- " + quotedDir + "; rm -f -- " + posixQuote(temporaryArchive) + "' ERR INT TERM; " +
		"if [ ! -s " + posixQuote(archive) + " ]; then " +
		"git archive --format=tar -o " + posixQuote(temporaryArchive) + " " + posixQuote(tip) + "; " +
		"if [ ! -s " + posixQuote(archive) + " ]; then mv -- " + posixQuote(temporaryArchive) + " " + posixQuote(archive) + "; else rm -f -- " + posixQuote(temporaryArchive) + "; fi; " +
		"fi; " +
		"tar -xf " + posixQuote(archive) + " -C " + quotedDir + "; " +
		"git -C " + quotedDir + " init --quiet; " +
		"mkdir -p -- " + quotedDir + "/.git/objects/info; " +
		"git rev-parse --path-format=absolute --git-path objects > " + quotedDir + "/.git/objects/info/alternates; " +
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

func hasDeletedMinePath(root string, paths []string) bool {
	for _, rel := range paths {
		if _, err := os.Stat(filepath.Join(root, filepath.FromSlash(rel))); os.IsNotExist(err) {
			return true
		}
	}
	return false
}

func ownedTestRunExpression(root string, paths []string) (string, error) {
	seen := map[string]bool{}
	var names []string
	for _, rel := range paths {
		if !strings.HasSuffix(strings.ToLower(rel), "_test.go") {
			continue
		}
		file, err := parser.ParseFile(token.NewFileSet(), filepath.Join(root, filepath.FromSlash(rel)), nil, 0)
		if err != nil {
			return "", err
		}
		for _, decl := range file.Decls {
			fn, ok := decl.(*ast.FuncDecl)
			if !ok || fn.Recv != nil || fn.Name == nil {
				continue
			}
			name := fn.Name.Name
			if !(strings.HasPrefix(name, "Test") || strings.HasPrefix(name, "Example") || strings.HasPrefix(name, "Fuzz")) || seen[name] {
				continue
			}
			seen[name] = true
			names = append(names, regexp.QuoteMeta(name))
		}
	}
	if len(names) == 0 {
		return "", nil
	}
	sort.Strings(names)
	return "^(" + strings.Join(names, "|") + ")$", nil
}

func requestedMinePaths(raw []string) []string {
	out := make([]string, 0, len(raw))
	for _, path := range raw {
		out = append(out, filepath.ToSlash(filepath.Clean(path)))
	}
	return out
}

func subtractValidatePaths(all, checked []string) []string {
	done := make(map[string]bool, len(checked))
	for _, path := range checked {
		done[path] = true
	}
	left := make([]string, 0, len(all)-len(done))
	for _, path := range all {
		if !done[path] {
			left = append(left, path)
		}
	}
	return left
}

func validateWriterIsTerminal(w io.Writer) bool {
	type statter interface {
		Stat() (os.FileInfo, error)
	}
	f, ok := w.(statter)
	if !ok {
		return false
	}
	info, err := f.Stat()
	return err == nil && info.Mode()&os.ModeCharDevice != 0
}

func validatePhaseOrder(testOnly bool) []string {
	phases := []string{"resolve_root", "resolve_ref", "normalize_mine", "extract_tip", "base_graph", "overlay"}
	if !testOnly {
		phases = append(phases, "gofmt")
	}
	phases = append(phases, "list_graph", "test_select")
	if !testOnly {
		phases = append(phases, "build", "vet")
	}
	return append(phases, "test")
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
			fmt.Fprintf(w, "OK: committed tip %s + %d owned path(s) affected-test clean (isolated test-only mode)\n", short(res.Tip), len(res.Mine))
		} else {
			fmt.Fprintf(w, "OK: committed tip %s + %d owned path(s) build, vet, and affected-test clean\n", short(res.Tip), len(res.Mine))
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

func writeValidateTestContext(w io.Writer, res validateResult) {
	if res.Runner != "" {
		fmt.Fprintf(w, "runner: %s\n", res.Runner)
	}
	if res.TestScope != "" {
		fmt.Fprintf(w, "tests: %s (%s)\n", res.TestScope, res.TestRun)
	}
}

func recordValidateFailure(res *validateResult, phase validateActivePhase, step, detail string, cause error) {
	phase.finishAs("failed", cause.Error())
	res.OK = false
	res.Failures = append(res.Failures, ciPreflightFailure{Step: step, Detail: detail})
}

func finishValidatePhaseOrTimeout(stdout io.Writer, res *validateResult, recorder *validateRecorder, phase validateActivePhase, name string, err error, asJSON bool) (int, bool) {
	phase.finish(err)
	if recorder.ctx.Err() == nil {
		return 0, false
	}
	return finishValidateTimeout(stdout, res, recorder, name, asJSON), true
}

func finishValidateRequiredPhase(stdout, stderr io.Writer, res *validateResult, recorder *validateRecorder, phase validateActivePhase, name string, err error, asJSON bool, failureMessage string) (int, bool) {
	if code, timedOut := finishValidatePhaseOrTimeout(stdout, res, recorder, phase, name, err, asJSON); timedOut {
		return code, true
	}
	if err == nil {
		return 0, false
	}
	fmt.Fprintln(stderr, failureMessage)
	return 2, true
}

func finishValidateContextPhase(stdout io.Writer, res *validateResult, recorder *validateRecorder, phase validateActivePhase, name string, asJSON bool) (int, bool) {
	if recorder.ctx.Err() == nil {
		return 0, false
	}
	phase.finish(recorder.ctx.Err())
	return finishValidateTimeout(stdout, res, recorder, name, asJSON), true
}

func runValidateCheckPhase(stdout io.Writer, res *validateResult, recorder *validateRecorder, phase validateActivePhase, name string, failure error, asJSON bool, run func() (string, bool)) (int, bool) {
	detail, ok := run()
	if code, timedOut := finishValidateContextPhase(stdout, res, recorder, phase, name, asJSON); timedOut {
		return code, true
	}
	if ok {
		phase.finish(nil)
	} else {
		recordValidateFailure(res, phase, name, detail, failure)
	}
	return 0, false
}

func validateTimeoutPhase(res validateResult) string {
	for i := len(res.Phases) - 1; i >= 0; i-- {
		if res.Phases[i].Status == "timeout" {
			return res.Phases[i].Name
		}
	}
	return "unknown"
}
