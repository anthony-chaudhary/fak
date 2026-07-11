package main

import (
	"fmt"
	"io"
	"os"
	"os/exec"
	"path"
	"path/filepath"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/buildwitness"
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
//   - FAIL OPEN on infra error (git or the go toolchain unavailable, archive/extract failure):
//     the gate warns and admits — it never refuses on an inability to check.
//   - Never block a PRE-EXISTING trunk red: when the prospective tree is red, the SAME packages
//     are built at HEAD too; if HEAD is also red the red was not introduced by this commit, so
//     the gate warns and admits (differential attribution, one extra build on the rare red path).
//   - The common green path costs ONE build of only the touched packages (plus ./cmd/fak, the
//     buildwitness target) against an archive of the prospective tree — the shared repo's real
//     index and working tree are never touched (throwaway GIT_INDEX_FILE).

// commitBuildCheckPackages maps the commit's repo-relative pathspecs to the sorted, de-duplicated
// `go build` package patterns the gate must compile: always the buildwitness target ./cmd/fak,
// plus ./<dir> for every changed .go file ("." for a root-level file). A commit with NO .go path
// returns nil — a non-Go commit cannot red the build, so the caller skips the gate entirely.
func commitBuildCheckPackages(changedPaths []string) []string {
	set := make(map[string]struct{})
	sawGo := false
	for _, p := range changedPaths {
		slash := filepath.ToSlash(strings.TrimSpace(p))
		if !strings.HasSuffix(slash, ".go") {
			continue
		}
		sawGo = true
		if dir := path.Dir(slash); dir == "." {
			set["."] = struct{}{}
		} else {
			set["./"+dir] = struct{}{}
		}
	}
	if !sawGo {
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

// commitBuildCheckGate decides whether the commit described by paths may land. It returns
// ok=true to admit (including every fail-open case) or ok=false with reason "COMMITTED_RED" and
// the compiler detail when THIS commit would turn the committed trunk red.
func commitBuildCheckGate(stderr io.Writer, root string, paths []string) (ok bool, reason, detail string) {
	pkgs := commitBuildCheckPackages(paths)
	if len(pkgs) == 0 {
		return true, "", "" // non-Go commit: nothing to gate
	}
	failOpen := func(err error) (bool, string, string) {
		fmt.Fprintf(stderr, "fak commit: build-check skipped: %v\n", err)
		return true, "", ""
	}
	if _, err := exec.LookPath("go"); err != nil {
		return failOpen(err) // toolchain unavailable: cannot check, never refuse
	}
	headSHA, err := gitRevParse(root, "HEAD")
	if err != nil {
		return failOpen(err)
	}
	prospectiveTree, err := commitProspectiveTree(root, paths)
	if err != nil {
		return failOpen(err)
	}
	headTree, err := gitRevParse(root, "HEAD^{tree}")
	if err != nil {
		return failOpen(err)
	}
	if prospectiveTree == headTree {
		return true, "", "" // no effective change: this commit cannot introduce a red
	}

	propDir, err := extractCommittedTip(root, prospectiveTree)
	if err != nil {
		return failOpen(err)
	}
	defer os.RemoveAll(propDir)
	propPkgs := commitBuildCheckExistingPackages(propDir, pkgs)
	if len(propPkgs) == 0 {
		return true, "", "" // e.g. the commit deletes the package outright: nothing left to compile
	}
	buildDetail, buildOK := goBuildPackages(propDir, propPkgs)
	if buildOK || commitBuildCheckOnlyUnbuildable(buildDetail) {
		return true, "", "" // the common fast path: ONE build, green
	}

	// The prospective tree is RED. Differential attribution: build the SAME packages at HEAD's
	// committed bytes. A red already present at HEAD was not introduced by this commit — blocking
	// it here would wedge every commit behind someone else's break, so warn and admit.
	headDir, err := extractCommittedTip(root, headSHA)
	if err != nil {
		return failOpen(err)
	}
	defer os.RemoveAll(headDir)
	if headPkgs := commitBuildCheckExistingPackages(headDir, pkgs); len(headPkgs) > 0 {
		if headDetail, headOK := goBuildPackages(headDir, headPkgs); !headOK && !commitBuildCheckOnlyUnbuildable(headDetail) {
			fmt.Fprint(stderr, formatPreexistingRedAdvisory(headDetail))
			// Best-effort fleet witness: fold this per-clone shrug onto a shared class so the
			// fleet converges on ONE break instead of each clone re-discovering it. Fail-open —
			// the commit is already admitted; recording never changes that.
			w := emitTrunkRedWitness(stderr, "commit", headSHA, failingPackagesFromBuild(headDetail), extractUndefinedSymbol(headDetail))
			fmt.Fprint(stderr, trunkRedWitnessNote(w))
			return true, "", ""
		}
	}
	if sym := extractUndefinedSymbol(buildDetail); sym != "" {
		buildDetail = "undefined: " + sym + "\n" + buildDetail
	}
	return false, "COMMITTED_RED", buildDetail
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
	cmd := exec.Command("git", append([]string{"-C", root}, args...)...)
	cmd.Env = append(os.Environ(), envKV)
	windowgate.ConfigureBackgroundCommand(cmd)
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
