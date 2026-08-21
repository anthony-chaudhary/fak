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
//  1. build   `go build [-ldflags -X …BuildVersion=<VERSION>] -o <tmp> ./cmd/fak` — a tree
//     that won't compile stops here; the ldflags bake the tree's VERSION into the
//     binary so its reported version does not depend on a guard's runtime cwd.
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
	"io"
	"os"
	"path/filepath"
	"strings"
	"sync"
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
	StageCopy    Stage = "copy"
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
	cleanupBuildArtifact := true
	defer func() {
		if cleanupBuildArtifact {
			_ = os.Remove(tmp)
		}
	}()

	// 1. build the candidate, baking the tree's VERSION into it as appversion.BuildVersion
	//    (see versionLDFlags for why this is load-bearing, not cosmetic). -buildvcs=true
	//    FORCES the VCS stamp: under Go's default -buildvcs=auto the detached-worktree build
	//    (PrepareOrigin) can silently drop the stamp when VCS detection can't resolve the
	//    repo, shipping a binary that cannot attest which commit it is (#3350, epic #2218 gap
	//    G2). With =true the build FAILS instead of emitting an unstamped binary.
	buildArgs := []string{"build", "-buildvcs=true"}
	commit := ""
	if out, ok := run(ctx, opts.RepoRoot, "git", "status", "--porcelain"); ok && strings.TrimSpace(out) == "" {
		if out, ok := run(ctx, opts.RepoRoot, "git", "rev-parse", "HEAD"); ok {
			candidate := strings.TrimSpace(out)
			if validCommit(candidate) {
				commit = candidate
			}
		}
	}
	if ld := versionLDFlags(opts.RepoRoot, commit); ld != "" {
		buildArgs = append(buildArgs, "-ldflags", ld)
	}
	buildArgs = append(buildArgs, "-o", tmp, "./cmd/fak")
	if out, ok := run(ctx, opts.RepoRoot, "go", buildArgs...); !ok {
		return Result{Stage: StageBuild, Detail: trim(out)}
	}
	// 2. vet the package (catches a compiling-but-suspect tree).
	if out, ok := run(ctx, opts.RepoRoot, "go", "vet", "./cmd/fak"); !ok {
		return Result{Stage: StageVet, Detail: trim(out)}
	}
	// 3. smoke: the freshly-built binary must run, self-report its version, AND carry a real
	//    VCS stamp. Running catches a binary that builds but cannot start (bad cgo link,
	//    missing data file, panic on init). The stamp check is fail-CLOSED on provenance
	//    (#3350, epic #2218 gap G2): an unstamped binary still exits 0 on `version`, so
	//    without it a stampless candidate would swap in indistinguishable from a good one and
	//    blind every downstream "which commit is this guard?" / version-skew check.
	//    -buildvcs=true already makes the build FAIL when it cannot stamp; this second gate
	//    refuses to swap a binary that somehow still reports no stamp.
	out, ok := run(ctx, opts.RepoRoot, tmp, "version")
	if !ok {
		return Result{Stage: StageSmoke, Detail: trim(out)}
	}
	if !versionOutputStamped(out) {
		return Result{Stage: StageSmoke, Detail: "candidate binary is UNSTAMPED — `version` reports no VCS provenance; refusing to swap an unattestable binary over the fleet: " + trim(out)}
	}
	// 4. swap: only now is the candidate trusted over the running fleet.
	cleanupBuildArtifact = false
	if err := swap(tmp, opts.Target); err != nil {
		return Result{Stage: StageSwap, Detail: err.Error()}
	}
	return Result{Installed: true, Stage: StageSwap, Detail: "installed " + filepath.Base(opts.Target)}
}

// InstallVerifiedCopy installs an already-gated fak binary at another hot-copy path.
// Self-update uses this after its first build/vet/smoke ladder succeeds, so converging N
// host copies pays that expensive ladder once rather than rebuilding identical bytes N times.
func InstallVerifiedCopy(swap Swapper, source, target string) Result {
	if strings.TrimSpace(source) == "" || strings.TrimSpace(target) == "" {
		return Result{Stage: StageCopy, Detail: "source and target are required"}
	}

	in, err := os.Open(source)
	if err != nil {
		return Result{Stage: StageCopy, Detail: "open verified source: " + err.Error()}
	}
	defer in.Close()

	info, err := in.Stat()
	if err != nil {
		return Result{Stage: StageCopy, Detail: "stat verified source: " + err.Error()}
	}
	tmp := fmt.Sprintf("%s.new.%d", target, os.Getpid())
	_ = os.Remove(tmp)
	out, err := os.OpenFile(tmp, os.O_WRONLY|os.O_CREATE|os.O_EXCL, info.Mode().Perm())
	if err != nil {
		return Result{Stage: StageCopy, Detail: "create candidate copy: " + err.Error()}
	}
	_, copyErr := io.Copy(out, in)
	if copyErr == nil {
		copyErr = out.Sync()
	}
	closeErr := out.Close()
	if copyErr == nil {
		copyErr = closeErr
	}
	if copyErr != nil {
		_ = os.Remove(tmp)
		return Result{Stage: StageCopy, Detail: "copy verified candidate: " + copyErr.Error()}
	}
	defer os.Remove(tmp)

	if err := swap(tmp, target); err != nil {
		return Result{Stage: StageSwap, Detail: err.Error()}
	}
	return Result{Installed: true, Stage: StageSwap, Detail: "installed " + filepath.Base(target) + " from verified candidate"}
}

// PrepareOrigin checks out a PRISTINE detached copy of a ref (e.g. "origin/main") into a
// fresh temp worktree under the repo, and returns its path plus a cleanup func. Building
// from this — instead of the live working tree — is what makes self-update viable on a
// permanently-dirty shared trunk: a build from the live tree always stamps vcs.modified=true
// (because peers are mid-edit), which would make every binary look "dirty" and defeat the
// staleness check; worse, it would bake peer work-in-progress INTO the installed binary.
// A detached origin worktree gives a clean VCS stamp AND installs exactly verified
// origin/main, never a contaminated local build.
//
// It is best-effort and self-cleaning: the cleanup removes the worktree (git worktree
// remove --force), prunes the admin entry, and removes the out-of-tree owner stamp.
// The cleanup is idempotent so callers may invoke it explicitly before os.Exit (which
// skips deferred functions) while still deferring it for ordinary returns. A partial
// `worktree add` failure is cleaned immediately, but only after proving the target did
// not exist before the add attempt.
func PrepareOrigin(ctx context.Context, run Runner, repoRoot, ref, dir string) (string, func(), error) {
	noop := func() {}
	if strings.TrimSpace(ref) == "" {
		ref = "origin/main"
	}
	dir = filepath.Clean(strings.TrimSpace(dir))
	if dir == "." || dir == "" {
		return "", noop, fmt.Errorf("prepare-origin: empty worktree path")
	}
	if _, err := os.Lstat(dir); err == nil {
		return "", noop, fmt.Errorf("prepare-origin: refusing to reuse existing path %s", dir)
	} else if !os.IsNotExist(err) {
		return "", noop, fmt.Errorf("prepare-origin: cannot inspect worktree path %s: %v", dir, err)
	}
	// Make sure the ref is current before we detach onto it.
	_, _ = run(ctx, repoRoot, "git", "fetch", "origin", "--quiet")
	if out, ok := run(ctx, repoRoot, "git", "worktree", "add", "--detach", dir, ref); !ok {
		// Git may have materialized part of the directory/admin entry before returning
		// failure. The path was proven absent above, so removing that partial result
		// cannot touch a pre-existing checkout.
		cleanupOriginWorktree(ctx, run, repoRoot, dir)
		return "", noop, fmt.Errorf("prepare-origin: git worktree add %s @ %s failed: %s", dir, ref, trim(out))
	}
	if err := writeBuildOwnerStamp(dir, defaultBuildOwnerStamp()); err != nil {
		cleanupOriginWorktree(ctx, run, repoRoot, dir)
		return "", noop, fmt.Errorf("prepare-origin: owner stamp %s: %v", BuildOwnerStampPath(dir), err)
	}
	var once sync.Once
	cleanup := func() { once.Do(func() { cleanupOriginWorktree(ctx, run, repoRoot, dir) }) }
	return dir, cleanup, nil
}

// cleanupOriginWorktree is the one source-cleanup path for PrepareOrigin. Git removal
// is preferred because it removes the administrative record and directory together.
// If it fails, direct removal is safe here because PrepareOrigin proved the target was
// absent before creating it; prune then clears any dangling admin entry.
func cleanupOriginWorktree(ctx context.Context, run Runner, repoRoot, dir string) {
	if _, ok := run(ctx, repoRoot, "git", "worktree", "remove", "--force", dir); !ok {
		_ = os.RemoveAll(dir)
	}
	_, _ = run(ctx, repoRoot, "git", "worktree", "prune")
	removeBuildOwnerStamp(dir)
}

// versionLDFlags returns the `-ldflags` value that bakes RepoRoot's VERSION marker into the
// built binary as appversion.BuildVersion, or "" when the tree has no readable VERSION file
// (in which case the build stays exactly as before — no ldflags).
//
// Why this is load-bearing, not polish: a bare `go build ./cmd/fak` produces a binary with
// NO embedded version, so at RUN time appversion.Current() falls through to walking the
// filesystem upward for a VERSION file — which makes a guard's reported version depend on
// its working directory. A guard launched with a cwd under a PARENT checkout inherits that
// parent's marker: on the fleet host, guards launched under C:\work reported the workspace's
// stale C:\work\VERSION ("0.1.1") instead of the fleet binary's actual version, and the same
// binary reported "dev" / the real version / the stale one purely by where it was launched.
// BuildVersion wins over the VERSION walk in appversion.Current() precisely so a
// shipped/installed binary "reports the version it was built with instead of inheriting a
// parent checkout's marker" — self-update simply never set it. Baking it here (the same
// -X the release-artifacts workflow uses) pins the installed binary's version to the commit
// it was built from, independent of wherever a guard is later launched.
func versionLDFlags(repoRoot, commit string) string {
	var flags []string
	if b, err := os.ReadFile(filepath.Join(repoRoot, "VERSION")); err == nil {
		if v := strings.TrimSpace(string(b)); v != "" {
			flags = append(flags, "-X github.com/anthony-chaudhary/fak/internal/appversion.BuildVersion="+v)
		}
	}
	if validCommit(commit) {
		flags = append(flags, "-X github.com/anthony-chaudhary/fak/internal/appversion.BuildCommit="+commit)
	}
	return strings.Join(flags, " ")
}

func validCommit(s string) bool {
	if len(s) != 40 {
		return false
	}
	for _, r := range s {
		if !((r >= '0' && r <= '9') || (r >= 'a' && r <= 'f') || (r >= 'A' && r <= 'F')) {
			return false
		}
	}
	return true
}

// versionOutputStamped reports whether `fak version` output carries a REAL VCS stamp: a
// `build: <rev>` line whose revision is an actual commit, not the "(no VCS stamp …)" or
// "module vX" sentinels an unstamped / `go install …@vX` build prints. It mirrors the parse
// cmd/fak's stampOfBinary uses, so the self-update smoke gate reads the exact same provenance
// signal a fleet version-skew witness does (#3350). An unstamped candidate yields false and
// the gate refuses the swap.
func versionOutputStamped(out string) bool {
	for _, line := range strings.Split(out, "\n") {
		line = strings.TrimSpace(line)
		if !strings.HasPrefix(line, "build:") {
			continue
		}
		fields := strings.Fields(strings.TrimSpace(strings.TrimPrefix(line, "build:")))
		if len(fields) == 0 {
			return false // a bare "build:" line — no revision to attest.
		}
		switch fields[0] {
		case "module", "(no": // "module vX" (go install …@vX) or "(no VCS stamp …)".
			return false
		default:
			return true
		}
	}
	return false // no build line at all — treat as unstamped, not as a pass.
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
