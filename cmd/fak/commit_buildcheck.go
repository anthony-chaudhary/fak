package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path"
	"path/filepath"
	"reflect"
	"sort"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/buildwitness"
	"github.com/anthony-chaudhary/fak/internal/procguard"
	"github.com/anthony-chaudhary/fak/internal/safecommit"
	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

// cmd/fak/commit_buildcheck.go — the COMMITTED_RED commit-boundary gate (#4152): refuse a
// `fak commit` whose PROSPECTIVE COMMITTED TREE (HEAD's committed bytes + exactly this commit's
// staged/working content, every other working-tree file masked) would not compile under DEFAULT
// build tags. This promotes the internal/buildwitness CI invariant — "the default build of
// cmd/fak is green" — from a post-hoc CI test to an invariant enforced at the moment the red
// would be created: the partial commit that lands a caller while its callee file stays
// uncommitted (`undefined: X`).
//
// Contract:
//   - REPORT, never hide, an inability to check (#6006). The gate returns a
//     safecommit.BuildCheckOutcome, and safecommit.DecideBuildCheck turns it into the
//     admit/refuse verdict and the build_check object in --json. A static infra failure (git or
//     the go toolchain unavailable) still fails open; a TIMEOUT no longer does silently — see
//     internal/safecommit/buildcheck.go for why the two are not the same shrug.
//   - Never block a PRE-EXISTING trunk red: when the prospective tree is red, the SAME packages
//     are built at HEAD too; if HEAD is also red the red was not introduced by this commit, so
//     the gate warns and admits (differential attribution, one extra build on the rare red path).
//   - The common green path first preserves the differential build witness, then delegates to
//     validate for owned gofmt, importer build/vet, and uncached changed-package tests. Both read
//     private materializations; the shared repo's real index and HEAD are never touched.

// commitBuildCheckPackages maps the commit's repo-relative pathspecs to the sorted, de-duplicated
// `go build` package patterns the gate must compile: always the buildwitness target ./cmd/fak,
// plus ./<dir> for every changed Go or native package source file ("." for a root-level file).
// The extensions mirror the source families indexed by validate's go-list graph; this keeps an
// Objective-C/header/assembly-only commit from bypassing the prospective native build.
func commitBuildCheckPackages(changedPaths []string) []string {
	set := make(map[string]struct{})
	sawSource := false
	for _, p := range changedPaths {
		slash := filepath.ToSlash(strings.TrimSpace(p))
		if !commitValidationSourcePath(slash) {
			continue
		}
		sawSource = true
		if dir := path.Dir(slash); dir == "." {
			set["."] = struct{}{}
		} else {
			set["./"+dir] = struct{}{}
		}
	}
	if !sawSource {
		return nil
	}
	set[buildwitness.TargetPackage] = struct{}{}
	pkgs := make([]string, 0, len(set))
	for p := range set {
		pkgs = append(pkgs, p)
	}
	sort.Strings(pkgs)
	return pkgs
}

func commitValidationSourcePath(path string) bool {
	switch strings.ToLower(filepath.Ext(path)) {
	case ".go", ".c", ".cc", ".cpp", ".cxx", ".m", ".mm", ".h", ".hh", ".hpp", ".hxx",
		".f", ".for", ".f90", ".f95", ".f03", ".f08", ".s", ".sx", ".swig", ".swigcxx", ".syso":
		return true
	default:
		return false
	}
}

// extractUndefinedSymbol scans compiler output for the first `undefined: <symbol>` occurrence and
// returns the symbol (empty when none): it headlines the refusal so the fix — commit the file
// that defines it — is visible without re-reading the whole build transcript.
func extractUndefinedSymbol(buildOutput string) string {
	const marker = "undefined: "
	idx := strings.Index(buildOutput, marker)
	if idx < 0 {
		return ""
	}
	sym := buildOutput[idx+len(marker):]
	if end := strings.IndexAny(sym, " \t\r\n"); end >= 0 {
		sym = sym[:end]
	}
	return strings.TrimSpace(sym)
}

// commitBuildCheckGate runs the gate and reports WHAT IT DID as a safecommit.BuildCheckOutcome
// plus the diagnostic detail (the compiler transcript, or the error that stopped it). It does
// not decide the commit's fate: safecommit.DecideBuildCheck owns that, so "the check could not
// run" is a first-class state on the wire instead of a stderr line the caller never sees
// (#6006).
var (
	defaultCommitBuildCheckGateWithTimeout = func(stderr io.Writer, root string, paths []string, timeout time.Duration) (safecommit.BuildCheckOutcome, string) {
		return runCommitBuildCheck(stderr, root, paths, timeout)
	}
	commitBuildCheckGateWithTimeout = defaultCommitBuildCheckGateWithTimeout

	defaultCommitBuildCheckGate = func(stderr io.Writer, root string, paths []string) (safecommit.BuildCheckOutcome, string) {
		return commitBuildCheckGateWithTimeout(stderr, root, paths, defaultValidateTimeout)
	}
	commitBuildCheckGate = defaultCommitBuildCheckGate
)

// executeCommitBuildCheck routes prospective validation to the active gate function: honoring
// an explicit timeout while preserving existing test mocks on commitBuildCheckGate or
// commitBuildCheckGateWithTimeout.
func executeCommitBuildCheck(stderr io.Writer, root string, paths []string, timeout time.Duration) (safecommit.BuildCheckOutcome, string) {
	if commitBuildCheckGateWithTimeout != nil && reflect.ValueOf(commitBuildCheckGateWithTimeout).Pointer() != reflect.ValueOf(defaultCommitBuildCheckGateWithTimeout).Pointer() {
		return commitBuildCheckGateWithTimeout(stderr, root, paths, timeout)
	}
	if commitBuildCheckGate != nil && reflect.ValueOf(commitBuildCheckGate).Pointer() != reflect.ValueOf(defaultCommitBuildCheckGate).Pointer() {
		return commitBuildCheckGate(stderr, root, paths)
	}
	if commitBuildCheckGateWithTimeout != nil {
		return commitBuildCheckGateWithTimeout(stderr, root, paths, timeout)
	}
	return runCommitBuildCheck(stderr, root, paths, timeout)
}

func runCommitBuildCheck(stderr io.Writer, root string, paths []string, timeout time.Duration) (safecommit.BuildCheckOutcome, string) {
	pkgs := commitBuildCheckPackages(paths)
	if len(pkgs) == 0 {
		return safecommit.BuildCheckNotApplicable, "" // non-Go commit: nothing to gate
	}
	// couldNotRun names the skip on stderr AND hands it back typed. Classification is
	// safecommit's: a missing toolchain is a static property of the host (fails open), an
	// expired deadline is not.
	couldNotRun := func(err error) (safecommit.BuildCheckOutcome, string) {
		outcome := safecommit.ClassifyBuildCheckError(err)
		fmt.Fprintf(stderr, "fak commit: build-check %s: %v\n", outcome, err)
		return outcome, err.Error()
	}
	if _, err := exec.LookPath("go"); err != nil {
		return couldNotRun(err) // toolchain unavailable: cannot check
	}
	headSHA, err := gitRevParse(root, "HEAD")
	if err != nil {
		return couldNotRun(err)
	}
	prospectiveTree, err := commitProspectiveTree(root, paths)
	if err != nil {
		return couldNotRun(err)
	}
	headTree, err := gitRevParse(root, "HEAD^{tree}")
	if err != nil {
		return couldNotRun(err)
	}
	if prospectiveTree == headTree {
		// No effective change: this commit cannot introduce a red, so there is nothing to
		// compile — not a skipped check.
		return safecommit.BuildCheckNotApplicable, "prospective tree is identical to HEAD"
	}

	propDir, err := extractCommittedTip(root, prospectiveTree)
	if err != nil {
		return couldNotRun(err)
	}
	defer os.RemoveAll(propDir)
	propPkgs := commitBuildCheckExistingPackages(propDir, pkgs)
	if len(propPkgs) == 0 {
		// e.g. the commit deletes the package outright: nothing left to compile.
		return safecommit.BuildCheckNotApplicable, "no buildable package remains in the prospective tree"
	}
	buildDetail, buildOK := goBuildPackages(propDir, propPkgs)
	if buildOK || commitBuildCheckOnlyUnbuildable(buildDetail) {
		return commitValidateOwnedPaths(stderr, root, paths, timeout)
	}

	// The prospective tree is RED. Differential attribution: build the SAME packages at HEAD's
	// committed bytes. A red already present at HEAD was not introduced by this commit — blocking
	// it here would wedge every commit behind someone else's break, so warn and admit.
	headDir, err := extractCommittedTip(root, headSHA)
	if err != nil {
		return couldNotRun(err)
	}
	defer os.RemoveAll(headDir)
	if headPkgs := commitBuildCheckExistingPackages(headDir, pkgs); len(headPkgs) > 0 {
		if headDetail, headOK := goBuildPackages(headDir, headPkgs); !headOK && !commitBuildCheckOnlyUnbuildable(headDetail) {
			fmt.Fprint(stderr, formatPreexistingRedAdvisory(headDetail))
			// Best-effort fleet witness: fold this per-clone shrug onto a shared class so the
			// fleet converges on ONE break instead of each clone re-discovering it. Fail-open —
			// the commit is already admitted; recording never changes that.
			w := emitTrunkRedWitness(stderr, root, "commit", headSHA, failingPackagesFromBuild(headDetail), extractUndefinedSymbol(headDetail))
			fmt.Fprint(stderr, trunkRedWitnessNote(w))
			return safecommit.BuildCheckHeadRed, headDetail
		}
	}
	if sym := extractUndefinedSymbol(buildDetail); sym != "" {
		buildDetail = "undefined: " + sym + "\n" + buildDetail
	}
	buildDetail += "\n" + commitValidateCommand(paths)
	return safecommit.BuildCheckFailed, buildDetail
}

// commitValidateOwnedPaths lifts the existing isolated validator into the commit admission
// gate. runValidate materializes HEAD plus only paths in a private checkout, so this runs before
// safecommit can stage anything in the caller's real index. Its JSON result gives this older
// build-check wire vocabulary a typed failed/timeout/infra outcome without inventing a second
// refusal protocol.
func commitValidateOwnedPaths(stderr io.Writer, root string, paths []string, timeout time.Duration) (safecommit.BuildCheckOutcome, string) {
	if timeout <= 0 {
		timeout = defaultValidateTimeout
	}
	args := []string{"--root", root, "--ref", "HEAD", "--json", "--progress=false", "--timeout", timeout.String()}
	for _, path := range paths {
		args = append(args, "--mine", path)
	}
	var stdout, diagnostics bytes.Buffer
	code := runValidate(&stdout, &diagnostics, args)
	var res validateResult
	if err := json.Unmarshal(stdout.Bytes(), &res); err != nil {
		detail := strings.TrimSpace(diagnostics.String())
		if detail == "" {
			detail = fmt.Sprintf("decode prospective validation result: %v", err)
		}
		fmt.Fprintf(stderr, "fak commit: prospective validation could not run: %s\n", detail)
		return safecommit.BuildCheckSkippedInfra, detail
	}
	if code == 0 && res.OK {
		return safecommit.BuildCheckPassed, ""
	}
	detail := formatCommitValidationFailure(res, paths)
	if res.TimedOut || res.Reason == "TIMEOUT" {
		return safecommit.BuildCheckSkippedTimeout, detail
	}
	if code == 1 {
		return safecommit.BuildCheckFailed, detail
	}
	if diag := strings.TrimSpace(diagnostics.String()); diag != "" {
		detail += "\nvalidator: " + diag
	}
	return safecommit.BuildCheckSkippedInfra, detail
}

func formatCommitValidationFailure(res validateResult, paths []string) string {
	var b strings.Builder
	b.WriteString("prospective validation failed")
	for _, failure := range res.Failures {
		b.WriteString("\n  ")
		b.WriteString(failure.Step)
		if failure.Detail != "" {
			b.WriteString(": ")
			b.WriteString(failure.Detail)
		}
		if len(failure.Files) > 0 {
			b.WriteString(": ")
			b.WriteString(strings.Join(failure.Files, ", "))
		}
	}
	b.WriteString("\n")
	b.WriteString(commitValidateCommand(paths))
	return b.String()
}

func commitValidateCommand(paths []string) string {
	var b strings.Builder
	b.WriteString("next: fak validate --ref HEAD")
	for _, path := range paths {
		b.WriteString(" --mine ")
		b.WriteString(path)
	}
	return b.String()
}

// refuseCommitBuildCheck renders a build-gate refusal on both channels and returns the process
// exit code. The --json branch is the point of #6006: a worker that parses the result object
// sees the refusal AND the gate's outcome, instead of a stderr line no exit code reflected.
func refuseCommitBuildCheck(stdout, stderr io.Writer, paths []string, bc safecommit.BuildCheckResult, reason string, asJSON bool) int {
	code, ok := safecommit.BuildCheckExitCode(reason)
	if !ok {
		code = safecommit.ExitRefused
	}
	fmt.Fprintf(stderr, "fak commit: %s\n", reason)
	if d := strings.TrimSpace(bc.Detail); d != "" {
		fmt.Fprintln(stderr, d)
	}
	fmt.Fprintln(stderr, commitBuildCheckAdvice(reason))
	if asJSON {
		res := safecommit.ScoreResult(safecommit.Result{Paths: paths, Reason: reason, Detail: bc.Detail, BuildCheck: &bc})
		if err := writeIndentedJSON(stdout, res); err != nil {
			fmt.Fprintf(stderr, "fak commit: %v\n", err)
			return safecommit.ExitPostCommitFailure
		}
	}
	return code
}

// commitBuildCheckAdvice is the operator sentence for each gate refusal: what it means and the
// named ways out. Pure, so the wording is testable without git or the go toolchain.
func commitBuildCheckAdvice(reason string) string {
	switch reason {
	case safecommit.ReasonBuildCheckTimeout:
		return "fak commit: prospective validation did not finish, so nothing here says this commit is green. This is retryable (exit 3): run the `fak validate --mine ...` command above, or pass --allow-build-check-timeout (env FAK_COMMIT_BUILD_CHECK=allow-timeout) to land it UNCHECKED on purpose. --allow-build-check-timeout is recorded in --json as build_check.failed_open; --no-build-check disables the admission gate."
	default:
		return "fak commit: the exact owned delta failed prospective build, vet, formatting, or affected tests before the real index changed. Run the `fak validate --mine ...` command above and fix the named phase; use --no-build-check only for an intentional unchecked landing."
	}
}

// formatPreexistingRedAdvisory renders the advisory the gate prints when differential attribution
// proves the red already exists at HEAD (a peer's break, not this commit) — so the commit is
// admitted, never wedged behind someone else's break. It does NOT stay silent: it NAMES the shared
// break (the failing package(s) and the first undefined symbol, parsed from the HEAD build output)
// and frames it as a trunk red that still needs a fix AT ITS SOURCE — not an island for this agent
// to route around in isolation. This aligns the commit gate with the pre-push gate's richer
// TRUNK_ALREADY_RED render (prepushBuild's renderPrePushBuild): a detected break is a first-class,
// actionable signal the fleet converges on, per
// docs/notes/CONCEPT-AGENTIC-OPERATOR-QUESTIONS-2026-07-10.md. Pure over the compiler output so it
// is unit-testable without git/go. Keeps the "ALREADY red" phrase the pre-existing-red case is
// recognized by. headDetail is the trimmed `go build` output of the same packages at HEAD.
func formatPreexistingRedAdvisory(headDetail string) string {
	var b strings.Builder
	b.WriteString("fak commit: build-check: the committed trunk is ALREADY red at HEAD (pre-existing, not introduced by this commit) — your commit is allowed.\n")
	if pkgs := failingPackagesFromBuild(headDetail); len(pkgs) > 0 {
		b.WriteString("  shared trunk red in: " + strings.Join(pkgs, " ") + "\n")
	}
	if sym := extractUndefinedSymbol(headDetail); sym != "" {
		b.WriteString("  first break: undefined: " + sym + "\n")
	}
	b.WriteString("  this is a SHARED break that still needs a fix at its source — until HEAD builds, every peer inherits it.\n")
	return b.String()
}

// commitProspectiveTree writes the tree object this commit WOULD produce: a THROWAWAY index
// (GIT_INDEX_FILE under a temp dir — the shared repo's real index is never read or written) is
// seeded from HEAD, exactly the commit's paths are staged from the working tree over it, and the
// result is written as a tree. Every other dirty/untracked working-tree file is thereby masked,
// which is the whole point on a permanently peer-dirty checkout.
func commitProspectiveTree(root string, paths []string) (string, error) {
	tmp, err := os.MkdirTemp("", "fak-commit-buildcheck-*")
	if err != nil {
		return "", err
	}
	defer os.RemoveAll(tmp)
	env := "GIT_INDEX_FILE=" + filepath.Join(tmp, "index")
	if _, err := gitWithEnv(root, env, "read-tree", "HEAD"); err != nil {
		return "", err
	}
	if _, err := gitWithEnv(root, env, append([]string{"add", "--"}, paths...)...); err != nil {
		return "", err
	}
	out, err := gitWithEnv(root, env, "write-tree")
	if err != nil {
		return "", err
	}
	return strings.TrimSpace(out), nil
}

// gitWithEnv runs `git -C root args...` with one extra environment entry (the throwaway
// GIT_INDEX_FILE); on failure the error carries the git output so the fail-open warn is
// diagnosable.
func gitWithEnv(root, envKV string, args ...string) (string, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, "git", append([]string{"-C", root}, args...)...)
	cmd.Env = append(os.Environ(), envKV)
	windowgate.ConfigureBackgroundCommand(cmd)
	cmd.WaitDelay = 5 * time.Second
	cmd.Cancel = func() error {
		if cmd.Process != nil && cmd.Process.Pid > 0 {
			procguard.KillPID(cmd.Process.Pid)
		}
		return nil
	}
	out, err := cmd.CombinedOutput()
	if err != nil {
		return "", fmt.Errorf("git %s: %v: %s", strings.Join(args, " "), err, strings.TrimSpace(string(out)))
	}
	return string(out), nil
}

// commitBuildCheckExistingPackages keeps only the patterns whose directory exists under dir.
// `fak commit` runs in arbitrary repos, so the always-included ./cmd/fak (absent outside this
// repo), a brand-new package absent at HEAD, or a package this commit deletes must not fabricate
// a "directory not found" build failure — a pattern with no directory has nothing to compile.
func commitBuildCheckExistingPackages(dir string, pkgs []string) []string {
	out := make([]string, 0, len(pkgs))
	for _, p := range pkgs {
		rel := strings.TrimPrefix(p, "./")
		if rel == "." || rel == "" {
			out = append(out, p)
			continue
		}
		if st, err := os.Stat(filepath.Join(dir, filepath.FromSlash(rel))); err == nil && st.IsDir() {
			out = append(out, p)
		}
	}
	return out
}

// commitBuildCheckOnlyUnbuildable reports whether a go build failure consists ONLY of the
// "nothing participates in the default build" family — a directory whose files are all excluded
// (fully //go:build wip_<feature>-fenced, test-only, or emptied of Go files by this commit).
// Under the invariant this gate enforces (the DEFAULT build compiles) such a package has nothing
// to compile; refusing on it would false-block the documented wip-fence escape hatch.
func commitBuildCheckOnlyUnbuildable(buildOutput string) bool {
	sawAny := false
	for _, ln := range strings.Split(buildOutput, "\n") {
		ln = strings.TrimSpace(ln)
		if ln == "" {
			continue
		}
		sawAny = true
		if strings.Contains(ln, "build constraints exclude all Go files") ||
			strings.Contains(ln, "no Go files in") ||
			strings.Contains(ln, "no non-test Go files in") {
			continue
		}
		return false
	}
	return sawAny
}
