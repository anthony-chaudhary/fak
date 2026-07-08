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
//     though your change is clean. buildcheck generates a -overlay that hides every
//     untracked .go file (except the ones you declare with --mine), so the compile sees
//     the committed tree plus your declared WIP -- immune to peers' untracked WIP. This
//     is the LIGHT path: no full-tree copy, unlike the heavy git-archive / detached-
//     worktree isolation (`fak worktree witness`). NOTE it does not REVERT peers'
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
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

func cmdBuildCheck(argv []string) { os.Exit(runBuildCheck(os.Stdout, os.Stderr, argv)) }

var (
	buildCheckUntracked = untrackedFiles
	buildCheckRun       = runGoBuildCheck
	buildCheckNow       = time.Now
)

// goOverlay is the JSON shape `go build -overlay <file>` consumes: a single Replace
// map from an on-disk file path to its backing file. An EMPTY backing path makes the
// go command treat the disk file as if it does not exist (see `go help build`), which
// is exactly how we hide an untracked sibling from the compile.
type goOverlay struct {
	Replace map[string]string `json:"Replace"`
}

type buildCheckReport struct {
	Schema      string   `json:"schema"`
	Mode        string   `json:"mode"`
	Packages    []string `json:"packages"`
	Isolate     bool     `json:"isolate"`
	MaskedFiles []string `json:"masked_files,omitempty"`
	MaskedCount int      `json:"masked_count"`
	OverlayPath string   `json:"overlay_path,omitempty"`
	Output      string   `json:"output"`
	Command     []string `json:"command"`
	ElapsedMS   int64    `json:"elapsed_ms"`
	Verdict     string   `json:"verdict"`
	ExitCode    int      `json:"exit_code"`
	Reason      string   `json:"reason,omitempty"`
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

	var masked, staleMine []string
	overlayPath := ""
	if *isolate {
		untracked, uerr := buildCheckUntracked(root)
		if uerr != nil {
			fmt.Fprintf(stderr, "fak buildcheck: listing untracked files: %v\n", uerr)
			return 1
		}
		masked, staleMine = selectMaskedFiles(untracked, mine)
		for _, m := range staleMine {
			fmt.Fprintf(stderr, "fak buildcheck: --mine %s is not an untracked file; ignoring (it is already in the build)\n", m)
		}
		if len(masked) > 0 {
			overlayPath = filepath.Join(scratch, "overlay.json")
			if werr := writeOverlayFile(overlayPath, buildOverlay(root, masked)); werr != nil {
				fmt.Fprintf(stderr, "fak buildcheck: writing overlay: %v\n", werr)
				return 1
			}
			if !*asJSON {
				fmt.Fprintf(stderr, "fak buildcheck: masking %d untracked sibling .go file(s) so peers' WIP cannot red this compile:\n", len(masked))
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

	// In JSON mode the go output is captured (a trimmed tail lands in the report's
	// reason on failure) so stdout carries only the report; otherwise it streams live.
	runOut, runErr := stdout, stderr
	var buf bytes.Buffer
	if *asJSON {
		runOut, runErr = &buf, &buf
	}
	code, execErr := buildCheckRun(root, args, runOut, runErr)
	elapsed := buildCheckNow().Sub(start)

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

	if *asJSON {
		rep := buildCheckReport{
			Schema:      "fak.buildcheck.v1",
			Mode:        mode,
			Packages:    pkgs,
			Isolate:     *isolate,
			MaskedFiles: masked,
			MaskedCount: len(masked),
			OverlayPath: overlayPath,
			Output:      outTarget,
			Command:     append([]string{"go"}, args...),
			ElapsedMS:   elapsed.Milliseconds(),
			Verdict:     verdict,
			ExitCode:    code,
			Reason:      reason,
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
// untracked paths, the .go files to HIDE are every untracked .go NOT declared as your
// own via --mine. It also returns the --mine entries that are not actually untracked
// (a stale/typo'd declaration) so the shell can narrate them; those fail OPEN -- a
// tracked file named by --mine is in the build regardless, so ignoring it is safe (the
// opposite of `fak affected --mine`, where a bad declaration must fail closed because it
// could exonerate a real red). Inputs and outputs are slash-normalized; results sorted.
func selectMaskedFiles(untracked []string, mine []string) (masked, staleMine []string) {
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
		if strings.HasSuffix(f, ".go") && !mineSet[f] {
			masked = append(masked, f)
		}
	}
	sort.Strings(masked)
	sort.Strings(staleMine)
	return masked, staleMine
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
	cmd := exec.Command("go", args...)
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
