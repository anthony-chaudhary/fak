// Package selfinstall rebuilds the fak binary from the current checkout and atomically
// swaps it into a target path — but ONLY after the freshly-built binary passes a gate, so
// a tree that does not compile, fails vet, or produces a binary that cannot even print its
// version is NEVER installed over a running fleet.
//
// This is the "make the latest verified fak available" half of keeping an always-on guard
// fleet converged (binstamp is the "am I stale?" detection half). The hard rule it exists
// to enforce: convergence must mean "converge on the latest GOOD commit," never "converge
// on the latest commit." A broken fak.exe swapped under N guards would break every guard at
// once; the gate is therefore not optional polish — it is the whole point.
//
// The flow (Install):
//  1. build   `go build -o <tmp> ./cmd/fak`     — a tree that won't compile stops here.
//  2. vet     `go vet ./cmd/fak`                — a vet failure stops here.
//  3. smoke   `<tmp> version`                   — the built binary must run + self-report.
//  4. swap    atomic replace of target by <tmp> — only reached when 1–3 all pass.
//
// Every effect goes through an injected Runner and an injected swap, so the gate ladder is
// testable with no toolchain and no filesystem race. A failed gate returns a Result with
// the failing Stage and the captured output, and leaves the target binary untouched.
package selfinstall

import (
	"context"
	"fmt"
	"path/filepath"
	"strings"
)

// Runner runs a command and returns combined output + whether it succeeded. ok=false means
// the command ran and failed OR could not be executed — either way the gate has not passed.
type Runner func(ctx context.Context, dir, name string, args ...string) (out string, ok bool)

// Swapper atomically replaces dst with the file at src (which is consumed/moved). It must be
// atomic enough that a concurrent reader sees either the old or the new binary, never a
// truncated one. On Windows replacing a mapped .exe requires renaming the old aside first;
// the production swap (osSwap) handles that.
type Swapper func(src, dst string) error

// Stage names the step a run reached (the last one attempted).
type Stage string

const (
	StageBuild   Stage = "build"
	StageVet     Stage = "vet"
	StageSmoke   Stage = "smoke"
	StageSwap    Stage = "swap"
	StageSkipped Stage = "skipped" // nothing to do (e.g. target already current)
)

// Result reports the outcome of an Install attempt.
type Result struct {
	Installed bool   // the target was replaced with a freshly-verified binary
	Stage     Stage  // the last stage attempted
	Detail    string // captured output / error context for the failing or final stage
}

// Options configures Install.
type Options struct {
	// RepoRoot is the checkout to build from (the dir `go build` runs in).
	RepoRoot string
	// Target is the binary path to replace (e.g. the guard's os.Executable()).
	Target string
	// BuildTmp is where the candidate binary is built before the swap. Empty => a sibling
	// of Target with a ".new" suffix (same volume, so the swap is a cheap rename).
	BuildTmp string
}

// Install runs the gated build→vet→smoke→swap ladder. It installs IFF every gate passes; on
// any gate failure it returns the failing Stage and leaves Target untouched.
func Install(ctx context.Context, run Runner, swap Swapper, opts Options) Result {
	tmp := strings.TrimSpace(opts.BuildTmp)
	if tmp == "" {
		tmp = opts.Target + ".new"
	}

	// 1. build the candidate.
	if out, ok := run(ctx, opts.RepoRoot, "go", "build", "-o", tmp, "./cmd/fak"); !ok {
		return Result{Stage: StageBuild, Detail: trim(out)}
	}
	// 2. vet the package (catches a compiling-but-suspect tree).
	if out, ok := run(ctx, opts.RepoRoot, "go", "vet", "./cmd/fak"); !ok {
		return Result{Stage: StageVet, Detail: trim(out)}
	}
	// 3. smoke: the freshly-built binary must run and self-report its version. This catches
	//    a binary that builds but cannot start (bad cgo link, missing data file, panic on
	//    init) BEFORE it ever replaces the one guarding live sessions.
	if out, ok := run(ctx, opts.RepoRoot, tmp, "version"); !ok {
		return Result{Stage: StageSmoke, Detail: trim(out)}
	}
	// 4. swap: only now is the candidate trusted over the running fleet.
	if err := swap(tmp, opts.Target); err != nil {
		return Result{Stage: StageSwap, Detail: err.Error()}
	}
	return Result{Installed: true, Stage: StageSwap, Detail: "installed " + filepath.Base(opts.Target)}
}

func trim(s string) string {
	s = strings.TrimSpace(s)
	const max = 2000
	if len(s) > max {
		return s[:max] + "…(truncated)"
	}
	if s == "" {
		return "(no output)"
	}
	return s
}

// FormatResult renders a Result as a single human line for the CLI / logs.
func FormatResult(r Result) string {
	if r.Installed {
		return "self-install: OK — " + r.Detail
	}
	return fmt.Sprintf("self-install: NOT installed — failed at %s gate: %s", r.Stage, r.Detail)
}
