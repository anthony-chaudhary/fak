// fak validate answers the shared-trunk question that neither a live-tree build nor
// ci-preflight can answer: does the committed tip plus only my explicit uncommitted
// delta pass the full build/vet and affected tests? Examples:
//
//	fak validate --mine internal/gitgate/gate.go --mine internal/gitgate/gate_test.go
//	fak validate --ref origin/main --mine cmd/fak/new_verb.go --json
//
// Ownership is deliberately explicit and repeatable; the verb never guesses from git
// status because this checkout contains concurrent peers' tracked and untracked WIP.
package main

import (
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

	"github.com/anthony-chaudhary/fak/internal/affectedtests"
	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

type validateResult struct {
	Schema   string               `json:"schema"`
	Mode     string               `json:"mode"`
	Ref      string               `json:"ref"`
	Tip      string               `json:"tip"`
	Mine     []string             `json:"mine"`
	Tested   []string             `json:"tested,omitempty"`
	Runner   string               `json:"runner,omitempty"`
	OK       bool                 `json:"ok"`
	Failures []ciPreflightFailure `json:"failures"`
}

func cmdValidate(argv []string) { os.Exit(runValidate(os.Stdout, os.Stderr, argv)) }

// runValidate checks committed ref plus only explicitly-owned working-tree paths.
func runValidate(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("validate", flag.ContinueOnError)
	fs.SetOutput(stderr)
	root := fs.String("root", "", "repo root (default: git toplevel from cwd)")
	ref := fs.String("ref", "HEAD", "committed base ref or sha")
	asJSON := fs.Bool("json", false, "emit the result as JSON")
	testOnly := fs.Bool("test-only", false, "skip full build/vet and run only affected tests in the isolated checkout")
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
	r := resolveRoot(*root)
	if r == "" {
		fmt.Fprintln(stderr, "fak validate: not in a git repo (or git unavailable)")
		return 2
	}
	tip, err := gitRevParse(r, *ref)
	if err != nil {
		fmt.Fprintf(stderr, "fak validate: cannot resolve ref %q: %v\n", *ref, err)
		return 2
	}
	paths, err := normalizeMinePaths(r, mine)
	if err != nil {
		fmt.Fprintf(stderr, "fak validate: %v\n", err)
		return 2
	}
	dir, err := extractCommittedTip(r, tip)
	if err != nil {
		fmt.Fprintf(stderr, "fak validate: cannot materialize tip %s: %v\n", short(tip), err)
		return 2
	}
	defer os.RemoveAll(dir)
	// Keep the base graph as well as the overlaid graph: a deleted file/package no longer
	// appears in `go list`, but its importers are still affected and must be tested.
	baseFileToPkg, baseEdges, _, _ := goListGraph(dir)
	if err := overlayMinePaths(r, dir, paths); err != nil {
		fmt.Fprintf(stderr, "fak validate: cannot overlay owned paths: %v\n", err)
		return 2
	}

	mode := "full"
	if *testOnly {
		mode = "test-only"
	}
	res := validateResult{Schema: "fak-validate/1", Mode: mode, Ref: *ref, Tip: tip, Mine: paths, OK: true}
	if !*testOnly {
		if files, ferr := gofmtOwnedPaths(dir, paths); ferr != nil {
			res.OK = false
			res.Failures = append(res.Failures, ciPreflightFailure{Step: "gofmt", Detail: ferr.Error()})
		} else if len(files) > 0 {
			res.OK = false
			res.Failures = append(res.Failures, ciPreflightFailure{Step: "gofmt", Files: files})
		}
		if detail, ok := runGoCheck(dir, "build", "./..."); !ok {
			res.OK = false
			res.Failures = append(res.Failures, ciPreflightFailure{Step: "build", Detail: detail})
		}
		if detail, ok := runGoCheck(dir, "vet", "./..."); !ok {
			res.OK = false
			res.Failures = append(res.Failures, ciPreflightFailure{Step: "vet", Detail: detail})
		}
	}
	fileToPkg, edges, _, graphErr := goListGraph(dir)
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
		if len(res.Tested) > 0 {
			res.Runner = validateTestRunner(runtime.GOOS, *wslTests)
			testTargets := packagePatternsForRoot(dir, res.Tested, fileToPkg)
			args := []string{"test"}
			if strings.TrimSpace(*testRun) != "" {
				args = append(args, "-run", *testRun)
			}
			args = append(args, testTargets...)
			if detail, ok := runValidateTests(r, dir, tip, args, *wslTests); !ok {
				res.OK = false
				res.Failures = append(res.Failures, ciPreflightFailure{Step: "test", Detail: detail})
			}
		}
	}
	if *asJSON {
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		_ = enc.Encode(res)
	} else {
		renderValidate(stdout, res)
	}
	if !res.OK {
		return 1
	}
	return 0
}

func normalizeMinePaths(root string, raw []string) ([]string, error) {
	seen := map[string]bool{}
	var out []string
	rootAbs, err := filepath.Abs(root)
	if err != nil {
		return nil, err
	}
	for _, value := range raw {
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
				if d.IsDir() {
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
	realRoot := srcRoot
	if resolved, rootErr := filepath.EvalSymlinks(srcRoot); rootErr == nil {
		realRoot = resolved
	}
	for _, rel := range paths {
		src := filepath.Join(srcRoot, filepath.FromSlash(rel))
		dst := filepath.Join(dstRoot, filepath.FromSlash(rel))
		realSrc, evalErr := filepath.EvalSymlinks(src)
		if os.IsNotExist(evalErr) {
			if removeErr := os.RemoveAll(dst); removeErr != nil {
				return removeErr
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
		data, err := os.ReadFile(realSrc)
		if err != nil {
			return err
		}
		info, err := os.Stat(src)
		if err != nil {
			return err
		}
		if err := os.MkdirAll(filepath.Dir(dst), 0o755); err != nil {
			return err
		}
		if err := os.WriteFile(dst, data, info.Mode().Perm()); err != nil {
			return err
		}
	}
	return nil
}

func gofmtOwnedPaths(root string, paths []string) ([]string, error) {
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
	cmd := windowgate.Command("gofmt", args...)
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

func runValidateTests(repo, dir, tip string, args []string, wsl bool) (string, bool) {
	// The archive checkout is isolated from peer WIP. Windows defaults to WSL so test
	// binaries execute under Linux rather than the host application-control boundary.
	useWSL := runtime.GOOS == "windows" && wsl
	if wsl {
		if err := prepareValidateGitIdentity(repo, dir, tip); err != nil {
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
		cmd = windowgate.Command("wsl.exe", "--cd", dir, "bash", "-lc", command)
	} else {
		cmd = windowgate.Command("go", args...)
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
func prepareValidateGitIdentity(repo, dir, tip string) error {
	commands := [][]string{
		{"init", "--quiet"},
		{"fetch", "--quiet", "--no-tags", "--depth=1", repo, tip},
		{"read-tree", tip},
		{"update-ref", "--no-deref", "HEAD", tip},
	}
	for _, args := range commands {
		cmd := windowgate.Command("git", args...)
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

func runGoCheck(dir string, args ...string) (string, bool) {
	cmd := windowgate.Command("go", args...)
	cmd.Dir = dir
	windowgate.ConfigureBackgroundCommand(cmd)
	out, err := cmd.CombinedOutput()
	if err != nil {
		return strings.TrimSpace(string(out)), false
	}
	return "", true
}

func renderValidate(w io.Writer, res validateResult) {
	if res.OK {
		if res.Runner != "" {
			fmt.Fprintf(w, "runner: %s\n", res.Runner)
		}
		if res.Mode == "test-only" {
			fmt.Fprintf(w, "OK: committed tip %s + %d owned path(s) affected-test clean (isolated test-only mode)\n", short(res.Tip), len(res.Mine))
		} else {
			fmt.Fprintf(w, "OK: committed tip %s + %d owned path(s) build, vet, and affected-test clean\n", short(res.Tip), len(res.Mine))
		}
		return
	}
	if res.Runner != "" {
		fmt.Fprintf(w, "runner: %s\n", res.Runner)
	}
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
