package main

// `fak go <build|vet|test|...> [args...]` — a poison-free `go` passthrough for the
// shared, permanently peer-dirty trunk (#4151, epic #4142). A bare `go build ./...` /
// `go vet` / `go test` run in place answers a FALSE question here: a peer's half-wired
// untracked `.go` fabricates a red, and a peer's uncommitted fix masks a real one. The
// fix — `fak buildcheck`'s untracked-mask overlay — already exists and is correct, but it
// is OPT-IN: every muscle-memory `go build` still hits the poison. This verb makes the
// DEFAULT developer build answer the clean question: it regenerates the untracked-mask
// overlay per-invocation (the untracked set changes constantly, so no static overlay or
// GOFLAGS could work — that is why this is a shim, not a config file), injects it as
// `-overlay` into the real `go` invocation, and forwards every other arg + the exit code +
// streaming output verbatim. It is a genuine drop-in:
//
//	fak go build ./...              build ./... , masking peers' untracked siblings
//	fak go vet ./cmd/...            vet, same overlay
//	fak go test ./internal/foo      test — hides peers' untracked _test.go poison
//	fak go --mine internal/x/new.go build ./...   keep your own new untracked file
//	fak go env                      pure passthrough (env/mod/etc. take no overlay)
//
// It does NOT fork the masking logic: the untracked set, the edited-dir keeps, the mask
// selection, and the overlay bytes all come from `cmd/fak/buildcheck.go`'s shipped folds
// (untrackedFiles/trackedModifiedDirs/selectMaskedFiles/buildOverlay/writeOverlayFile) via
// the SAME package seams `fak buildcheck` uses, so `fak go build` and `fak buildcheck`
// always mask the identical file set for a given tree state (asserted in go_shim_test.go).
//
// Like buildcheck, it preserves the live-tree cross-check for build/vet: a mask-induced
// false red (the overlay hid an untracked dep that tracked/kept code needs) is re-checked
// once against the live tree and reported OK, so a false red never survives.

import (
	"bytes"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/buildoverlay"
	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

func cmdGoShim(argv []string) { os.Exit(runGoShim(os.Stdout, os.Stderr, argv)) }

// goShimRun is the exec seam (stubbed in tests); it defaults to the same runner
// `fak buildcheck` uses so a real invocation execs the real `go`.
var (
	buildCheckUntracked    = buildoverlay.UntrackedGoFiles
	buildCheckModifiedDirs = buildoverlay.ModifiedDirs
	buildCheckLoadBearing  = buildoverlay.LoadBearingUntrackedFiles
	goShimRun              = runGoShimCommand
)

func runGoShimCommand(root string, args []string, stdout, stderr io.Writer) (int, error) {
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

// goShimOverlaySubcommands are the `go` subcommands that accept build flags and therefore
// `-overlay`. Everything else (env, mod, version, …) is a pure passthrough — injecting
// `-overlay` there is an error, so we don't.
var goShimOverlaySubcommands = map[string]bool{
	"build": true, "vet": true, "test": true, "run": true, "list": true, "install": true,
}

func runGoShim(stdout, stderr io.Writer, argv []string) int {
	if len(argv) == 0 {
		fmt.Fprintln(stderr, "fak go: a go subcommand is required (usage: fak go <build|vet|test|...> [args...])")
		return 2
	}
	switch argv[0] {
	case "-h", "--help", "help":
		if len(argv) == 1 {
			goShimUsage(stdout)
			return 0
		}
	}

	mine, rest := extractMineFlags(argv)
	if len(rest) == 0 {
		fmt.Fprintln(stderr, "fak go: a go subcommand is required after --mine (usage: fak go <build|vet|test|...> [args...])")
		return 2
	}
	sub := rest[0]
	root := repoRoot()

	// The overlay is only meaningful for the build-flag subcommands; for env/mod/version/etc.
	// pass argv straight through so `fak go` is a total drop-in for any `go` invocation.
	if !goShimOverlaySubcommands[sub] {
		code, err := goShimRun(root, rest, stdout, stderr)
		if err != nil {
			fmt.Fprintf(stderr, "fak go: running go %s: %v\n", sub, err)
			return 1
		}
		return code
	}

	scratch, err := os.MkdirTemp("", "fak-go-")
	if err != nil {
		fmt.Fprintf(stderr, "fak go: temp dir: %v\n", err)
		return 1
	}
	defer os.RemoveAll(scratch)

	masked, overlayPath, code := goShimOverlay(root, mine, scratch, stderr)
	if code != 0 {
		return code
	}
	args := goShimArgs(overlayPath, rest)

	if len(masked) > 0 {
		fmt.Fprintf(stderr, "fak go: masking %d untracked sibling .go file(s) so peers' WIP cannot red this %s\n", len(masked), sub)
	}
	fmt.Fprintf(stderr, "fak go: go %s\n", strings.Join(args, " "))

	// Live-tree cross-check applies to build/vet (a compile-only question): when we masked
	// files and the masked build reds, the red may be purely mask-induced (the overlay hid an
	// untracked dep that tracked/kept code imports across packages). Capture the masked run so
	// that false red is adjudicated against the live tree BEFORE its "undefined:" errors reach
	// the user. `test` and the rest stream verbatim — a test failure is not a mask false-red.
	crossCheckable := (sub == "build" || sub == "vet") && len(masked) > 0
	if !crossCheckable {
		out, runErr := goShimRun(root, args, stdout, stderr)
		if runErr != nil {
			fmt.Fprintf(stderr, "fak go: running go %s: %v\n", sub, runErr)
			return 1
		}
		return out
	}

	var buf bytes.Buffer
	out, runErr := goShimRun(root, args, &buf, &buf)
	if runErr != nil {
		io.Copy(stderr, &buf)
		fmt.Fprintf(stderr, "fak go: running go %s: %v\n", sub, runErr)
		return 1
	}
	if out != 0 {
		// Re-run once against the live tree (no overlay). If THAT compiles, the red was purely
		// the mask hiding a needed untracked dep — the tree the fleet sees is green, report OK.
		var liveBuf bytes.Buffer
		if code2, err2 := goShimRun(root, rest, &liveBuf, &liveBuf); err2 == nil && code2 == 0 {
			fmt.Fprintf(stderr, "fak go: masked %s red only because masked untracked deps were hidden; the live tree compiles -- reporting OK.\n", sub)
			fmt.Fprintf(stderr, "fak go: `git add` the new package(s) your tracked/kept files import before committing so a peer's build stays green.\n")
			return 0
		}
	}
	io.Copy(stderr, &buf)
	return out
}

// goShimOverlay computes the untracked-mask overlay for the current tree using the SAME
// folds `fak buildcheck` uses (no fork), writes it into scratch, and returns the masked
// file set + the overlay path ("" when nothing is masked). A non-zero code is a hard
// failure the caller should return. It fails OPEN on a missing in-flight-edit answer
// (mask all, as buildcheck does) but hard-fails if it cannot list untracked files at all.
func goShimOverlay(root string, mine []string, scratch string, stderr io.Writer) (masked []string, overlayPath string, code int) {
	untracked, uerr := buildCheckUntracked(root)
	if uerr != nil {
		fmt.Fprintf(stderr, "fak go: listing untracked files: %v\n", uerr)
		return nil, "", 1
	}
	modifiedDirs, merr := buildCheckModifiedDirs(root)
	if merr != nil {
		fmt.Fprintf(stderr, "fak go: cannot read in-flight edits (%v); masking all untracked siblings\n", merr)
		modifiedDirs = nil
	}
	loadBearing, lerr := buildCheckLoadBearing(root, untracked)
	if lerr != nil {
		fmt.Fprintf(stderr, "fak go: inspect local import closure: %v\n", lerr)
		return nil, "", 1
	}
	mine = append(mine, loadBearing...)
	masked, _, staleMine := buildoverlay.SelectMaskedFiles(untracked, mine, modifiedDirs)
	for _, m := range staleMine {
		fmt.Fprintf(stderr, "fak go: --mine %s is not an untracked file; ignoring (it is already in the build)\n", m)
	}
	if len(masked) == 0 {
		return nil, "", 0
	}
	overlayPath = filepath.Join(scratch, "overlay.json")
	if werr := buildoverlay.Write(overlayPath, buildoverlay.Build(root, masked)); werr != nil {
		fmt.Fprintf(stderr, "fak go: writing overlay: %v\n", werr)
		return nil, "", 1
	}
	return masked, overlayPath, 0
}

// goShimArgs injects `-overlay <path>` immediately after the go subcommand and forwards
// every other user arg untouched. With no overlay it returns the user's args verbatim, so
// `fak go` is byte-for-byte `go` when there is nothing to mask.
func goShimArgs(overlayPath string, rest []string) []string {
	if overlayPath == "" {
		return append([]string(nil), rest...)
	}
	out := make([]string, 0, len(rest)+2)
	out = append(out, rest[0], "-overlay", overlayPath)
	out = append(out, rest[1:]...)
	return out
}

// extractMineFlags pulls fak's own repeatable `--mine <file>` (also `-mine`, `--mine=…`)
// out of the argv wherever it appears, returning the collected mine paths and the
// remaining args to forward to `go` verbatim. Everything else — including go's own flags —
// is left in rest untouched so the shim never has to understand go's flag grammar.
func extractMineFlags(argv []string) (mine []string, rest []string) {
	for i := 0; i < len(argv); i++ {
		a := argv[i]
		switch {
		case a == "--mine" || a == "-mine":
			if i+1 < len(argv) {
				mine = append(mine, argv[i+1])
				i++
			}
		case strings.HasPrefix(a, "--mine="):
			mine = append(mine, strings.TrimPrefix(a, "--mine="))
		case strings.HasPrefix(a, "-mine="):
			mine = append(mine, strings.TrimPrefix(a, "-mine="))
		default:
			rest = append(rest, a)
		}
	}
	return mine, rest
}

func goShimUsage(w io.Writer) {
	fmt.Fprint(w, `fak go <build|vet|test|...> [args...] — poison-free go passthrough (#4151)

  Runs the real `+"`go`"+` with an untracked-mask -overlay injected per-invocation, so a
  peer's half-wired untracked .go cannot fabricate a red (and a peer's uncommitted fix
  cannot mask a real one). Forwards all other args, streaming output, and the exit code.

  fak go build ./...                 build ./... , masking peers' untracked siblings
  fak go vet ./cmd/...               vet with the same overlay
  fak go test ./internal/foo         test (hides peers' untracked _test.go poison)
  fak go --mine internal/x/new.go build ./...   keep your own new untracked file in the build

  --mine <file>   (repeatable) keep a specific untracked .go in the build — your own new file.
  Non-build subcommands (env, mod, version, …) pass straight through with no overlay.

  Optional per-developer shim so even a literal `+"`go`"+` is clean inside this repo:
    bash/zsh:  go() { if git rev-parse --show-toplevel >/dev/null 2>&1; then fak go "$@"; else command go "$@"; fi; }
    escape hatch: `+"`command go build ./...`"+` bypasses the shim.
`)
}
