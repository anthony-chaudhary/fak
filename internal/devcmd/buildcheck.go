package devcmd

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
// It is the impure shell over a handful of pure folds (buildoverlay.SelectMaskedFiles, buildoverlay.Build,
// buildCheckArgs), each unit-tested. `make ci` still runs the full suite as the
// authoritative gate -- this is the fast, collision-free inner-loop compile check.

import (
	"bytes"
	"context"
	"flag"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"sort"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/buildoverlay"
	"github.com/anthony-chaudhary/fak/internal/windowgate"
	"github.com/anthony-chaudhary/fak/internal/wipfence"
	"github.com/anthony-chaudhary/fak/internal/workdelivery"
)

var (
	buildCheckUntracked    = buildoverlay.UntrackedGoFiles
	buildCheckModifiedDirs = buildoverlay.ModifiedDirs
	buildCheckLoadBearing  = buildoverlay.LoadBearingUntrackedFiles
	buildCheckRun          = runGoBuildCheck
	buildCheckNow          = time.Now
	buildCheckAcquireSlot  = AcquireBuildSlot
	buildCheckSlotTimeout  = 5 * time.Minute
)

type isolateWIPFlag struct {
	value string
	isSet bool
}

func (f *isolateWIPFlag) String() string {
	return f.value
}

func (f *isolateWIPFlag) Set(s string) error {
	f.isSet = true
	f.value = s
	return nil
}

func (f *isolateWIPFlag) IsBoolFlag() bool {
	return true
}

type buildCheckReport struct {
	Schema           string                           `json:"schema"`
	Mode             string                           `json:"mode"`
	Packages         []string                         `json:"packages"`
	Isolate          bool                             `json:"isolate"`
	IsolateWIP       string                           `json:"isolate_wip,omitempty"`
	MaskedFiles      []string                         `json:"masked_files,omitempty"`
	MaskedCount      int                              `json:"masked_count"`
	KeptFiles        []string                         `json:"kept_files,omitempty"`
	KeptCount        int                              `json:"kept_count"`
	LiveCrossChecked bool                             `json:"live_cross_checked,omitempty"`
	OverlayPath      string                           `json:"overlay_path,omitempty"`
	Output           string                           `json:"output"`
	Command          []string                         `json:"command"`
	ElapsedMS        int64                            `json:"elapsed_ms"`
	Verdict          string                           `json:"verdict"`
	ExitCode         int                              `json:"exit_code"`
	Reason           string                           `json:"reason,omitempty"`
	CompileManifests []string                         `json:"compile_manifests,omitempty"`
	AdmittedFiles    []string                         `json:"admitted_files,omitempty"`
	ExcludedFiles    []string                         `json:"excluded_files,omitempty"`
	Delivery         *workdelivery.AdapterObservation `json:"delivery,omitempty"`
}

func RunBuildCheck(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak buildcheck", flag.ContinueOnError)
	fs.SetOutput(stderr)
	verbFlagUsage(fs, "buildcheck")
	isolate := fs.Bool("isolate", true, "generate a go -overlay that hides untracked sibling .go files (except --mine) for a tracked-tree-equivalent compile immune to peers' WIP; --isolate=false builds the live tree as-is")
	var isolateWIP isolateWIPFlag
	fs.Var(&isolateWIP, "isolate-wip", "evaluate only tagged or trunk files (e.g. --isolate-wip=<tag> or empty/trunk)")
	var mine pathList
	var compileManifests pathList
	fs.Var(&mine, "mine", "repo-relative untracked .go file to KEEP in the build (your own new file; repeatable)")
	fs.Var(&compileManifests, "compile-manifest", "work-delivery JSON record declaring admitted/excluded compile artifacts (repeatable)")
	vet := fs.Bool("vet", false, "run go vet instead of go build (compiles and vets; stricter, slower)")
	outDir := fs.String("out", "", "directory to write produced binaries into, isolated from the tree (needs a main package); default: discard output via the null device (a pure compile check)")
	asJSON := fs.Bool("json", false, "print a machine-readable report instead of streaming go output")
	if !parseFlags(fs, argv) {
		return 2
	}

	pkgs := fs.Args()
	if isolateWIP.isSet && isolateWIP.value == "true" && len(pkgs) > 0 {
		first := pkgs[0]
		if !strings.HasPrefix(first, ".") && !strings.Contains(first, "/") && first != "trunk" {
			isolateWIP.value = first
			pkgs = pkgs[1:]
		}
	}
	if len(pkgs) == 0 {
		pkgs = []string{"./..."}
	}

	wipTag := ""
	if isolateWIP.isSet && isolateWIP.value != "" && isolateWIP.value != "true" && isolateWIP.value != "trunk" {
		wipTag = isolateWIP.value
		if !strings.HasPrefix(wipTag, "wip_") {
			wipTag = "wip_" + wipfence.SlugFromPath(wipTag)
		}
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

	isolation, ok := prepareBuildCheckIsolation(stdout, stderr, root, scratch, *isolate, *asJSON, mine, compileManifests, mode, wipTag, pkgs)
	if !ok {
		return 1
	}
	masked, kept := isolation.masked, isolation.kept
	compileSet, overlayPath := isolation.compileSet, isolation.overlayPath

	// go build always writes SOMEWHERE for a main package; target the null device so a
	// compile check never drops a binary in the tree (the #2373 lock war), universally
	// across lib / main / multi-package -- unlike `-o <dir>`, which errors on a lib. An
	// explicit --out DIR isolates produced binaries in a dir instead (needs a main pkg).
	outTarget := os.DevNull
	if *outDir != "" {
		outTarget = *outDir
	}
	args := buildCheckArgs(mode, overlayPath, outTarget, wipTag, pkgs)

	if !*asJSON {
		fmt.Fprintf(stderr, "fak buildcheck: go %s\n", strings.Join(args, " "))
	}

	// The primary build's output is CAPTURED (not streamed live) whenever we masked files,
	// so an isolate red can be adjudicated against the live tree BEFORE any scary "undefined"
	// errors reach the user -- a mask-induced false red is then suppressed entirely. In JSON
	// mode output is always captured (stdout carries only the report); otherwise it streams.
	overlayMasked := *isolate && len(masked) > 0
	capture := *asJSON || overlayMasked || len(kept) > 0
	runOut, runErr := stdout, stderr
	var buf bytes.Buffer
	if capture {
		runOut, runErr = &buf, &buf
	}

	releaseSlot, slotErr := buildCheckAcquireSlot(context.Background(), buildCheckSlotTimeout)
	if slotErr != nil {
		elapsed := buildCheckNow().Sub(start)
		if *asJSON {
			rep := buildCheckReport{
				Schema:           "fak.buildcheck.v1",
				Mode:             mode,
				Packages:         pkgs,
				Isolate:          *isolate,
				Output:           outTarget,
				Command:          append([]string{"go"}, args...),
				ElapsedMS:        elapsed.Milliseconds(),
				Verdict:          "BUILD_SLOT_UNAVAILABLE",
				ExitCode:         1,
				Reason:           fmt.Sprintf("host-wide build concurrency governor: %v", slotErr),
				CompileManifests: compileManifests,
			}
			_ = writeIndentedJSONNoEscape(stdout, rep)
		} else {
			fmt.Fprintf(stderr, "fak buildcheck: host-wide build concurrency governor: %v\n", slotErr)
		}
		return 1
	}
	defer releaseSlot()

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
		liveArgs := buildCheckArgs(mode, "", outTarget, wipTag, pkgs)
		if code2, execErr2 := buildCheckRun(root, liveArgs, &liveBuf, &liveBuf); execErr2 == nil && code2 == 0 {
			liveCrossChecked = true
			verdict, reason, code = "OK", "", 0
			buf.Reset() // the captured masked-build errors were a false red; do not surface them
		}
	}
	if code != 0 {
		labelUntrackedBuildFailures(&buf, append(append([]string{}, masked...), kept...))
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
		isolateWIPRep := ""
		if isolateWIP.isSet {
			if wipTag != "" {
				isolateWIPRep = wipTag
			} else {
				isolateWIPRep = "trunk"
			}
		}
		rep := buildCheckReport{
			Schema:           "fak.buildcheck.v1",
			Mode:             mode,
			Packages:         pkgs,
			Isolate:          *isolate,
			IsolateWIP:       isolateWIPRep,
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
			CompileManifests: compileManifests,
			AdmittedFiles:    compileSet.Admitted,
			ExcludedFiles:    compileSet.Excluded,
		}
		unitID := "buildcheck"
		if len(compileManifests) == 1 {
			unitID = compileManifests[0]
		}
		unit := workdelivery.WorkUnit{Schema: workdelivery.Schema, ID: unitID, Axes: workdelivery.InitialAxes()}
		unit.Axes.Authoring = workdelivery.AuthoringRecorded
		unit.Axes.Admission = workdelivery.AdmissionAdmitted
		if delivery, deliveryErr := workdelivery.VerificationObservation(unit, verdict == "OK", mode, strings.Join(rep.Command, " "), "fak buildcheck", time.Now()); deliveryErr == nil {
			rep.Delivery = &delivery
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

type buildCheckIsolation struct {
	masked      []string
	kept        []string
	compileSet  workdelivery.CompileSet
	overlayPath string
}

func prepareBuildCheckIsolation(stdout, stderr io.Writer, root, scratch string, isolate, asJSON bool, mine, compileManifests pathList, mode, wipTag string, pkgs []string) (buildCheckIsolation, bool) {
	var result buildCheckIsolation
	if !isolate {
		return result, true
	}
	untracked, err := buildCheckUntracked(root)
	if err != nil {
		fmt.Fprintf(stderr, "fak buildcheck: listing untracked files: %v\n", err)
		return result, false
	}
	// A package with an in-flight tracked .go edit keeps its untracked .go siblings in the
	// build: they are almost always the matched new file that edit references, and masking
	// them would red the edit's own compile. Fail OPEN -- if git can't answer, mask all as
	// before rather than block the check.
	modifiedDirs, err := buildCheckModifiedDirs(root)
	if err != nil {
		if !asJSON {
			fmt.Fprintf(stderr, "fak buildcheck: cannot read in-flight edits (%v); masking all untracked siblings\n", err)
		}
		modifiedDirs = nil
	}
	var staleMine []string
	result.masked, result.kept, staleMine = buildoverlay.SelectMaskedFiles(untracked, mine, modifiedDirs)
	if wipTag != "" {
		var newMasked []string
		for _, m := range result.masked {
			full := filepath.Join(root, filepath.FromSlash(m))
			b, rerr := os.ReadFile(full)
			if rerr == nil {
				if tag, ok := wipfence.IsFenced(string(b)); ok && tag == wipTag {
					result.kept = append(result.kept, m)
					continue
				}
			}
			newMasked = append(newMasked, m)
		}
		result.masked = newMasked
		sort.Strings(result.kept)
	}
	if len(compileManifests) > 0 {
		result.compileSet, err = workdelivery.LoadCompileSet(compileManifests...)
		if err != nil {
			if asJSON {
				_ = writeIndentedJSONNoEscape(stdout, buildCheckReport{Schema: "fak.buildcheck.v1", Verdict: "COMPILE_ADMISSION_BLOCKED", ExitCode: 1, Mode: mode, Packages: pkgs, Isolate: isolate, CompileManifests: compileManifests, Reason: err.Error()})
			} else {
				fmt.Fprintf(stderr, "fak buildcheck: compile admission: %v\n", err)
			}
			return result, false
		}
		mine = append(mine, result.compileSet.Admitted...)
		result.masked, result.kept, staleMine = buildoverlay.SelectMaskedFiles(untracked, mine, modifiedDirs)
		result.masked = append(result.masked, result.compileSet.Excluded...)
		sort.Strings(result.masked)
		result.masked = slices.Compact(result.masked)
	}
	for _, path := range staleMine {
		fmt.Fprintf(stderr, "fak buildcheck: --mine %s is not an untracked file; ignoring (it is already in the build)\n", path)
	}
	if !asJSON && len(result.kept) > 0 {
		fmt.Fprintf(stderr, "fak buildcheck: keeping %d untracked .go file(s) whose package has in-flight tracked edits (matched new files, kept so the edit compiles):\n", len(result.kept))
		for _, path := range result.kept {
			fmt.Fprintf(stderr, "  - %s\n", path)
		}
	}
	if len(result.masked) == 0 {
		return result, true
	}
	result.overlayPath = filepath.Join(scratch, "overlay.json")
	if err := buildoverlay.Write(result.overlayPath, buildoverlay.Build(root, result.masked)); err != nil {
		fmt.Fprintf(stderr, "fak buildcheck: writing overlay: %v\n", err)
		return result, false
	}
	if !asJSON {
		fmt.Fprintf(stderr, "fak buildcheck: masking %d untracked sibling .go file(s) in packages with no in-flight edits so peers' WIP cannot red this compile:\n", len(result.masked))
		for _, path := range result.masked {
			fmt.Fprintf(stderr, "  - %s\n", path)
		}
	}
	return result, true
}

// buildoverlay.SelectMaskedFiles is the pure selection behind --isolate: from the repo-relative
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
// buildoverlay.Build maps each masked repo-relative file to an EMPTY backing path (absolute,
// OS-native — how the go command keys the overlay), which hides it from the compile.
func buildCheckArgs(mode, overlayPath, outTarget, wipTag string, pkgs []string) []string {
	// Every buildcheck checkout is disposable and may live under a different absolute
	// root. Keep that root out of compile action identities so Go's shared content-addressed
	// cache can reuse sound artifacts. This is verification-only: the debuggable developer
	// build deliberately keeps host paths.
	args := []string{mode, "-trimpath"}
	if overlayPath != "" {
		args = append(args, "-overlay", overlayPath)
	}
	if wipTag != "" {
		args = append(args, "-tags", wipTag)
	}
	if mode == "build" {
		args = append(args, "-o", outTarget)
	}
	return append(args, pkgs...)
}

// buildoverlay.UntrackedGoFiles returns the repo-relative, slash-separated untracked-but-not-ignored
// paths (`git ls-files --others --exclude-standard`) -- the peers'-WIP surface the
// overlay hides. Git emits forward-slash paths on every platform.
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

func labelUntrackedBuildFailures(output *bytes.Buffer, untracked []string) {
	normalizedOutput := filepath.ToSlash(output.String())
	for _, path := range untracked {
		normalizedPath := filepath.ToSlash(path)
		if !strings.Contains(normalizedOutput, normalizedPath) {
			continue
		}
		fmt.Fprintf(output, "fak buildcheck: %s is UNTRACKED -- this red is not from the committed base\n", normalizedPath)
	}
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
