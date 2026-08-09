package main

// `fak buildcheck` -- a lightweight, concurrency-safe compile check for a fleet of
// agents editing ONE shared trunk tree. It removes the two ways one agent's build
// blocks or reds another's:
//
//  1. NEVER drops a binary in the repo tree. A bare `go build ./cmd/fak` writes
//     `fak.exe` into the repo root; while a running fleet process holds that binary
//     open, a peer's build fails on Windows with "The process cannot access the file
//     because it is being used by another process" (#2373). And the documented hand
//     recipe (`go build -o $env:TEMP\fak-verify.exe`) uses a FIXED name, so two agents
//     doing it collide on the same temp file. buildcheck instead discards the output to
//     the null device (`-o os.DevNull`), which compiles-and-writes-nothing universally
//     -- lib, main, or multi-package -- so there is no in-tree write and no cross-agent
//     lock war. `--out DIR` isolates produced binaries in a caller-named dir instead.
//
//  2. Masks untracked SIBLING breakage via `go build -overlay`. On a shared tree a
//     peer's in-flight, not-yet-compiling `.go` file reds YOUR `go build ./...` even
//     though your change is clean. buildcheck generates a -overlay that hides untracked
//     .go files so the compile sees the committed tree plus your declared WIP -- immune to
//     peers' untracked WIP. Masking is SCOPED to packages with no in-flight tracked edit:
//     an untracked .go in a dir whose tracked .go you're already editing is KEPT, because
//     it is almost always the matched new file that edit references (e.g. a new
//     `compact.go` supplying symbols an edited `drain.go` calls) -- masking it would red
//     the edit's own compile, a false red. --mine still force-keeps a specific file (the
//     escape hatch for a brand-new untracked PACKAGE wired from an edit elsewhere). As a
//     backstop for that cross-package case, if the masked build STILL reds, buildcheck
//     re-runs once against the live tree (no overlay); if the live tree compiles, the red
//     was purely mask-induced and it reports OK (`live_cross_checked`), so a false red never
//     survives. This is the LIGHT path: no full-tree copy, unlike the heavy git-archive /
//     detached-worktree isolation (`fak worktree witness`). NOTE it does not REVERT peers'
//     modifications to TRACKED files; for a true committed-bytes view use the worktree.
//
//	fak buildcheck                     compile-check ./... , masking untracked siblings
//	fak buildcheck ./cmd/fak           compile one package, output discarded (never in-tree)
//	fak buildcheck --mine internal/foo/new.go   keep your own new untracked file in the build
//	fak buildcheck --out /tmp/bins ./cmd/fak    isolate the produced binary in a dir
//	fak buildcheck --isolate=false     build the live tree as-is (no overlay)
//	fak buildcheck --vet ./cmd/...     run go vet instead of go build
//	fak buildcheck --json              emit a machine-readable report
//
// It is the impure shell over a handful of pure folds (selectMaskedFiles, buildOverlay,
// buildCheckArgs), each unit-tested. `make ci` still runs the full suite as the
// authoritative gate -- this is the fast, collision-free inner-loop compile check.

import (
	"bufio"
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"go/parser"
	"go/token"
	"io"
	"os"
	"os/exec"
	"path"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

func cmdBuildCheck(argv []string) { os.Exit(runBuildCheck(os.Stdout, os.Stderr, argv)) }

var (
	buildCheckUntracked    = untrackedFiles
	buildCheckModifiedDirs = trackedModifiedDirs
	buildCheckLoadBearing  = loadBearingUntrackedFiles
	buildCheckRun          = runGoBuildCheck
	buildCheckNow          = time.Now
)

// goOverlay is the JSON shape `go build -overlay <file>` consumes: a single Replace
// map from an on-disk file path to its backing file. An EMPTY backing path makes the
// go command treat the disk file as if it does not exist (see `go help build`), which
// is exactly how we hide an untracked sibling from the compile.
type goOverlay struct {
	Replace map[string]string `json:"Replace"`
}

type buildCheckReport struct {
	Schema           string   `json:"schema"`
	Mode             string   `json:"mode"`
	Packages         []string `json:"packages"`
	Isolate          bool     `json:"isolate"`
	MaskedFiles      []string `json:"masked_files,omitempty"`
	MaskedCount      int      `json:"masked_count"`
	KeptFiles        []string `json:"kept_files,omitempty"`
	KeptCount        int      `json:"kept_count"`
	LiveCrossChecked bool     `json:"live_cross_checked,omitempty"`
	OverlayPath      string   `json:"overlay_path,omitempty"`
	Output           string   `json:"output"`
	Command          []string `json:"command"`
	ElapsedMS        int64    `json:"elapsed_ms"`
	Verdict          string   `json:"verdict"`
	ExitCode         int      `json:"exit_code"`
	Reason           string   `json:"reason,omitempty"`
}

func runBuildCheck(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak buildcheck", flag.ContinueOnError)
	fs.SetOutput(stderr)
	verbFlagUsage(fs, "buildcheck")
	isolate := fs.Bool("isolate", true, "generate a go -overlay that hides untracked sibling .go files (except --mine) for a tracked-tree-equivalent compile immune to peers' WIP; --isolate=false builds the live tree as-is")
	var mine pathList
	fs.Var(&mine, "mine", "repo-relative untracked .go file to KEEP in the build (your own new file; repeatable)")
	vet := fs.Bool("vet", false, "run go vet instead of go build (compiles and vets; stricter, slower)")
	outDir := fs.String("out", "", "directory to write produced binaries into, isolated from the tree (needs a main package); default: discard output via the null device (a pure compile check)")
	asJSON := fs.Bool("json", false, "print a machine-readable report instead of streaming go output")
	if !parseFlags(fs, argv) {
		return 2
	}

	pkgs := fs.Args()
	if len(pkgs) == 0 {
		pkgs = []string{"./..."}
	}
	mode := "build"
	if *vet {
		mode = "vet"
	}
	root := repoRoot()
	start := buildCheckNow()

	// A per-process scratch dir holds only the generated overlay file. os.MkdirTemp
	// gives a unique name, so two agents never collide, and it is always removed.
	scratch, err := os.MkdirTemp("", "fak-buildcheck-")
	if err != nil {
		fmt.Fprintf(stderr, "fak buildcheck: temp dir: %v\n", err)
		return 1
	}
	defer os.RemoveAll(scratch)

	var masked, kept, staleMine []string
	overlayPath := ""
	if *isolate {
		untracked, uerr := buildCheckUntracked(root)
		if uerr != nil {
			fmt.Fprintf(stderr, "fak buildcheck: listing untracked files: %v\n", uerr)
			return 1
		}
		// A package with an in-flight tracked .go edit keeps its untracked .go siblings in the
		// build: they are almost always the matched new file that edit references, and masking
		// them would red the edit's own compile. Fail OPEN -- if git can't answer, mask all as
		// before rather than block the check.
		modifiedDirs, merr := buildCheckModifiedDirs(root)
		if merr != nil {
			if !*asJSON {
				fmt.Fprintf(stderr, "fak buildcheck: cannot read in-flight edits (%v); masking all untracked siblings\n", merr)
			}
			modifiedDirs = nil
		}
		masked, kept, staleMine = selectMaskedFiles(untracked, mine, modifiedDirs)
		for _, m := range staleMine {
			fmt.Fprintf(stderr, "fak buildcheck: --mine %s is not an untracked file; ignoring (it is already in the build)\n", m)
		}
		if !*asJSON && len(kept) > 0 {
			fmt.Fprintf(stderr, "fak buildcheck: keeping %d untracked .go file(s) whose package has in-flight tracked edits (matched new files, kept so the edit compiles):\n", len(kept))
			for _, f := range kept {
				fmt.Fprintf(stderr, "  - %s\n", f)
			}
		}
		if len(masked) > 0 {
			overlayPath = filepath.Join(scratch, "overlay.json")
			if werr := writeOverlayFile(overlayPath, buildOverlay(root, masked)); werr != nil {
				fmt.Fprintf(stderr, "fak buildcheck: writing overlay: %v\n", werr)
				return 1
			}
			if !*asJSON {
				fmt.Fprintf(stderr, "fak buildcheck: masking %d untracked sibling .go file(s) in packages with no in-flight edits so peers' WIP cannot red this compile:\n", len(masked))
				for _, f := range masked {
					fmt.Fprintf(stderr, "  - %s\n", f)
				}
			}
		}
	}

	// go build always writes SOMEWHERE for a main package; target the null device so a
	// compile check never drops a binary in the tree (the #2373 lock war), universally
	// across lib / main / multi-package -- unlike `-o <dir>`, which errors on a lib. An
	// explicit --out DIR isolates produced binaries in a dir instead (needs a main pkg).
	outTarget := os.DevNull
	if *outDir != "" {
		outTarget = *outDir
	}
	args := buildCheckArgs(mode, overlayPath, outTarget, pkgs)

	if !*asJSON {
		fmt.Fprintf(stderr, "fak buildcheck: go %s\n", strings.Join(args, " "))
	}

	// The primary build's output is CAPTURED (not streamed live) whenever we masked files,
	// so an isolate red can be adjudicated against the live tree BEFORE any scary "undefined"
	// errors reach the user -- a mask-induced false red is then suppressed entirely. In JSON
	// mode output is always captured (stdout carries only the report); otherwise it streams.
	overlayMasked := *isolate && len(masked) > 0
	capture := *asJSON || overlayMasked
	runOut, runErr := stdout, stderr
	var buf bytes.Buffer
	if capture {
		runOut, runErr = &buf, &buf
	}
	code, execErr := buildCheckRun(root, args, runOut, runErr)

	verdict, reason := "OK", ""
	if execErr != nil {
		verdict = "RUN_ERROR"
		reason = execErr.Error()
	} else if code != 0 {
		if mode == "vet" {
			verdict = "VET_FAILED"
		} else {
			verdict = "BUILD_FAILED"
		}
		reason = fmt.Sprintf("go %s exited %d", mode, code)
	}

	// Live-tree cross-check: an isolate red can be a FALSE red -- the overlay hid an untracked
	// file that kept/tracked code needs (a brand-new package imported from an edited file, a
	// cross-package matched pair the dir-scoped keep cannot infer). Re-run once against the
	// live tree (no overlay); if THAT compiles, the tree the fleet sees is green, so report OK.
	// A live failure means the breakage is in tracked/kept code -> keep the failure verdict.
	liveCrossChecked := false
	if overlayMasked && execErr == nil && code != 0 {
		var liveBuf bytes.Buffer
		liveArgs := buildCheckArgs(mode, "", outTarget, pkgs)
		if code2, execErr2 := buildCheckRun(root, liveArgs, &liveBuf, &liveBuf); execErr2 == nil && code2 == 0 {
			liveCrossChecked = true
			verdict, reason, code = "OK", "", 0
			buf.Reset() // the captured masked-build errors were a false red; do not surface them
		}
	}
	elapsed := buildCheckNow().Sub(start)

	// Flush the captured primary-build output in non-JSON mode now that the verdict is settled:
	// a real red shows its errors; a cross-checked false red shows only a one-line explanation.
	if !*asJSON && capture {
		if liveCrossChecked {
			fmt.Fprintf(stderr, "fak buildcheck: isolate build red only because masked untracked deps were hidden; the live tree compiles -- reporting OK.\n")
			fmt.Fprintf(stderr, "fak buildcheck: `git add` the new package(s) your tracked/kept files import before committing so a peer's build stays green.\n")
		} else {
			io.Copy(stderr, &buf)
		}
	}

	if *asJSON {
		rep := buildCheckReport{
			Schema:           "fak.buildcheck.v1",
			Mode:             mode,
			Packages:         pkgs,
			Isolate:          *isolate,
			MaskedFiles:      masked,
			MaskedCount:      len(masked),
			KeptFiles:        kept,
			KeptCount:        len(kept),
			LiveCrossChecked: liveCrossChecked,
			OverlayPath:      overlayPath,
			Output:           outTarget,
			Command:          append([]string{"go"}, args...),
			ElapsedMS:        elapsed.Milliseconds(),
			Verdict:          verdict,
			ExitCode:         code,
			Reason:           reason,
		}
		if verdict != "OK" {
			rep.Reason = joinReason(reason, buildCheckTail(buf.String(), 40))
		}
		_ = writeIndentedJSONNoEscape(stdout, rep)
	}

	if execErr != nil {
		if !*asJSON {
			fmt.Fprintf(stderr, "fak buildcheck: running go %s: %v\n", mode, execErr)
		}
		return 1
	}
	return code
}

// selectMaskedFiles is the pure selection behind --isolate: from the repo-relative
// untracked paths, the .go files to HIDE are every untracked .go that is NOT declared as
// your own via --mine AND does NOT live in a package with an in-flight tracked edit
// (modifiedDirs). An untracked .go in an edited dir is KEPT (returned in `kept`) because it
// is almost always the matched new half of that edit -- masking it would red the edit's own
// compile, the exact false-red this scoping fixes. A nil/empty modifiedDirs masks every
// non-mine untracked .go (the original behavior). It also returns the --mine entries that
// are not actually untracked (a stale/typo'd declaration) so the shell can narrate them;
// those fail OPEN -- a tracked file named by --mine is in the build regardless, so ignoring
// it is safe (the opposite of `fak affected --mine`, where a bad declaration must fail
// closed because it could exonerate a real red). Inputs/outputs slash-normalized; sorted.
func selectMaskedFiles(untracked []string, mine []string, modifiedDirs map[string]bool) (masked, kept, staleMine []string) {
	untrackedSet := make(map[string]bool, len(untracked))
	for _, f := range untracked {
		untrackedSet[repoSlash(f)] = true
	}
	mineSet := make(map[string]bool, len(mine))
	for _, m := range mine {
		m = repoSlash(m)
		if mineSet[m] {
			continue
		}
		mineSet[m] = true
		if !untrackedSet[m] {
			staleMine = append(staleMine, m)
		}
	}
	for _, f := range untracked {
		f = repoSlash(f)
		if !strings.HasSuffix(f, ".go") || mineSet[f] {
			continue
		}
		if modifiedDirs[path.Dir(f)] {
			kept = append(kept, f)
			continue
		}
		masked = append(masked, f)
	}
	sort.Strings(masked)
	sort.Strings(kept)
	sort.Strings(staleMine)
	return masked, kept, staleMine
}

// loadBearingUntrackedFiles returns untracked Go files in local packages reachable
// from tracked Go source. Those packages are dependencies of the tree under test, not
// sibling poison, so the isolation overlay must retain their complete package contents.
func loadBearingUntrackedFiles(root string, untracked []string) ([]string, error) {
	modulePath, err := readModulePath(filepath.Join(root, "go.mod"))
	if err != nil {
		return nil, err
	}
	untrackedByDir := map[string][]string{}
	for _, name := range untracked {
		name = filepath.ToSlash(filepath.Clean(name))
		if strings.HasSuffix(name, ".go") {
			dir := path.Dir(name)
			untrackedByDir[dir] = append(untrackedByDir[dir], name)
		}
	}
	if len(untrackedByDir) == 0 {
		return nil, nil
	}
	cmd := windowgate.Command("git", "ls-files", "--", "*.go")
	cmd.Dir = root
	windowgate.ConfigureBackgroundCommand(cmd)
	out, err := cmd.Output()
	if err != nil {
		return nil, err
	}
	queue := localImportDirs(root, modulePath, strings.Fields(string(out)))
	seen := map[string]bool{}
	var kept []string
	for len(queue) > 0 {
		dir := path.Clean(queue[0])
		queue = queue[1:]
		if seen[dir] {
			continue
		}
		seen[dir] = true
		files := untrackedByDir[dir]
		if len(files) == 0 {
			continue
		}
		kept = append(kept, files...)
		queue = append(queue, localImportDirs(root, modulePath, files)...)
	}
	sort.Strings(kept)
	return kept, nil
}

func readModulePath(goMod string) (string, error) {
	f, err := os.Open(goMod)
	if err != nil {
		return "", err
	}
	defer f.Close()
	s := bufio.NewScanner(f)
	for s.Scan() {
		fields := strings.Fields(s.Text())
		if len(fields) == 2 && fields[0] == "module" {
			return strings.TrimSpace(fields[1]), nil
		}
	}
	if err := s.Err(); err != nil {
		return "", err
	}
	return "", fmt.Errorf("module directive not found in %s", goMod)
}

func localImportDirs(root, modulePath string, files []string) []string {
	prefix := strings.TrimSuffix(modulePath, "/") + "/"
	seen := map[string]bool{}
	var dirs []string
	for _, name := range files {
		parsed, err := parser.ParseFile(token.NewFileSet(), filepath.Join(root, filepath.FromSlash(name)), nil, parser.ImportsOnly)
		if err != nil {
			continue // the compiler will report malformed source; this fold only preserves dependencies
		}
		for _, spec := range parsed.Imports {
			importPath, err := strconv.Unquote(spec.Path.Value)
			if err != nil || !strings.HasPrefix(importPath, prefix) {
				continue
			}
			dir := path.Clean(strings.TrimPrefix(importPath, prefix))
			if dir != "." && !seen[dir] {
				seen[dir] = true
				dirs = append(dirs, dir)
			}
		}
	}
	return dirs
}

// buildOverlay maps each masked repo-relative file to an EMPTY backing path (absolute,
// OS-native — how the go command keys the overlay), which hides it from the compile.
func buildOverlay(root string, masked []string) goOverlay {
	rep := make(map[string]string, len(masked))
	for _, f := range masked {
		abs := filepath.Clean(filepath.Join(root, filepath.FromSlash(f)))
		rep[abs] = ""
	}
	return goOverlay{Replace: rep}
}

// buildCheckArgs assembles the argv after the "go" binary: the mode (build|vet), an
// optional -overlay file, the -o output target for a build (the null device by default,
// so nothing lands in the tree; a --out DIR when the caller wants the binaries), then
// the package patterns. `go vet` writes nothing, so it never takes -o.
func buildCheckArgs(mode, overlayPath, outTarget string, pkgs []string) []string {
	args := []string{mode}
	if overlayPath != "" {
		args = append(args, "-overlay", overlayPath)
	}
	if mode == "build" {
		args = append(args, "-o", outTarget)
	}
	return append(args, pkgs...)
}

// untrackedFiles returns the repo-relative, slash-separated untracked-but-not-ignored
// paths (`git ls-files --others --exclude-standard`) -- the peers'-WIP surface the
// overlay hides. Git emits forward-slash paths on every platform.
func untrackedFiles(root string) ([]string, error) {
	out, err := gitOut(root, "ls-files", "--others", "--exclude-standard")
	if err != nil {
		return nil, err
	}
	var files []string
	for _, line := range strings.Split(out, "\n") {
		line = strings.TrimSpace(line)
		if line != "" {
			files = append(files, filepath.ToSlash(line))
		}
	}
	sort.Strings(files)
	return files, nil
}

// trackedModifiedDirs returns the set of repo-relative, slash-separated directories that
// hold a TRACKED .go file differing from HEAD (staged or unstaged) -- the packages with
// in-flight edits. --isolate uses it to KEEP, not mask, an untracked .go sibling in such a
// dir: on this shared trunk that sibling is almost always the matched other half of the
// edit (a new file the edited file references), so masking it would red the edit's own
// compile. Restricted to .go changes so a stray doc/config edit does not un-mask a
// genuinely-independent untracked .go. Fails to the caller, which then masks all as before.
func trackedModifiedDirs(root string) (map[string]bool, error) {
	out, err := gitOut(root, "diff", "--name-only", "HEAD")
	if err != nil {
		return nil, err
	}
	dirs := map[string]bool{}
	for _, line := range strings.Split(out, "\n") {
		line = strings.TrimSpace(line)
		if line == "" || !strings.HasSuffix(line, ".go") {
			continue
		}
		dirs[path.Dir(filepath.ToSlash(line))] = true
	}
	return dirs, nil
}

func writeOverlayFile(path string, ov goOverlay) error {
	var buf bytes.Buffer
	enc := json.NewEncoder(&buf)
	enc.SetEscapeHTML(false) // paths, not HTML
	enc.SetIndent("", "  ")
	if err := enc.Encode(ov); err != nil {
		return err
	}
	return os.WriteFile(path, buf.Bytes(), 0o644)
}

func runGoBuildCheck(root string, args []string, stdout, stderr io.Writer) (int, error) {
	cmd := windowgate.Command("go", args...)
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

// buildCheckTail returns the last n non-empty lines of s, joined -- the actionable tail
// of a captured go build/vet failure for the JSON report's reason.
func buildCheckTail(s string, n int) string {
	var lines []string
	for _, ln := range strings.Split(s, "\n") {
		if strings.TrimSpace(ln) != "" {
			lines = append(lines, ln)
		}
	}
	if len(lines) > n {
		lines = lines[len(lines)-n:]
	}
	return strings.Join(lines, "\n")
}

func joinReason(a, b string) string {
	switch {
	case a == "":
		return b
	case b == "":
		return a
	default:
		return a + ": " + b
	}
}
