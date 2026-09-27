package artifact

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"runtime"
	"strconv"
	"strings"

	"github.com/anthony-chaudhary/fak/pkg/deploykit"
)

// BuildPinned builds a deployable from a clean-HEAD commit tuple across one or more
// repositories (for example fak-private then fak). Every repository of the tuple must be clean
// and at its pinned commit before the build and again after the smoke. The go children run
// under a generated go.work that uses exactly ModuleDirs, with GOFLAGS cleared and
// GOTOOLCHAIN=local, so neither the operator's ambient workspace nor ambient go flags nor a
// toolchain download can change what is built.
type BuildPinned struct {
	// ModuleDirs are the module roots of the tuple, one per repository. The first is the main
	// module: the build and vet run there, and the smoked binary must attest its pinned commit.
	ModuleDirs []string `json:"module_dirs"`
	// Commits pins the HEAD of each ModuleDirs repository, index for index, to a full 40-hex
	// commit.
	Commits []string `json:"commits"`
	// GoWorkPin is the go directive of the generated go.work (for example "1.26.7"). Empty pins
	// it to the local toolchain's own GOVERSION.
	GoWorkPin string `json:"go_work_pin,omitempty"`
	// Package is the main package built into the candidate, relative to the main module or an
	// import path (for example "./cmd/fak" or ".../cmd/fak-server").
	Package string `json:"package"`
	// Vet are the `go vet` targets. Empty vets Package.
	Vet []string `json:"vet,omitempty"`
	// Ldflags is passed verbatim as -ldflags, for example to stamp the version variables a
	// deployable's `version --json` reports. Empty passes no -ldflags.
	Ldflags string `json:"ldflags,omitempty"`
	// GOOS and GOARCH pin the target platform; empty means this host's. Another platform is
	// refused with ErrCrossBuildNotWired, because this host cannot smoke the result.
	GOOS   string `json:"goos,omitempty"`
	GOARCH string `json:"goarch,omitempty"`
	// Generation is the slot generation the candidate is stored under. It must be positive.
	Generation uint64 `json:"generation"`
}

// Kind reports KindBuild.
func (BuildPinned) Kind() string { return KindBuild }

var goVersionPattern = regexp.MustCompile(`^[0-9]+\.[0-9]+(\.[0-9]+)?((rc|beta)[0-9]+)?$`)

func (b BuildPinned) preflight(x deploykit.Executor) error {
	if _, err := b.modules(); err != nil {
		return err
	}
	if _, err := b.commits(); err != nil {
		return err
	}
	if !goTarget(b.Package) {
		return fmt.Errorf("%w: package %q is empty or looks like a flag", ErrInvalidSource, b.Package)
	}
	for _, v := range b.Vet {
		if !goTarget(v) {
			return fmt.Errorf("%w: vet target %q is empty or looks like a flag", ErrInvalidSource, v)
		}
	}
	if b.GoWorkPin != "" && !goVersionPattern.MatchString(b.GoWorkPin) {
		return fmt.Errorf("%w: go.work pin %q is not a Go version", ErrInvalidSource, b.GoWorkPin)
	}
	if b.Generation == 0 {
		return fmt.Errorf("%w: generation must be positive", ErrInvalidSource)
	}
	if goos, goarch := b.platform(); goos != runtime.GOOS || goarch != runtime.GOARCH {
		return fmt.Errorf("%w: target %s/%s, host %s/%s", ErrCrossBuildNotWired, goos, goarch, runtime.GOOS, runtime.GOARCH)
	}
	if _, ok := x.(EnvExecutor); !ok {
		return fmt.Errorf("%w: got %T", ErrEnvExecutorRequired, x)
	}
	return nil
}

func (b BuildPinned) acquire(ctx context.Context, x deploykit.Executor, dir string) (Candidate, error) {
	ex := x.(EnvExecutor) // preflight proved it
	mods, _ := b.modules()
	commits, _ := b.commits()
	if err := checkTuple(ctx, x, mods, commits, "before build"); err != nil {
		return Candidate{}, err
	}
	env, cleanup, err := b.pinnedEnv(ctx, ex, mods)
	if err != nil {
		return Candidate{}, err
	}
	defer cleanup()

	goos, _ := b.platform()
	out := filepath.Join(dir, candidateName(goos))
	args := []string{"build", "-trimpath"}
	if b.Ldflags != "" {
		args = append(args, "-ldflags", b.Ldflags)
	}
	args = append(args, "-o", out, b.Package)
	if o, ok := ex.RunEnv(ctx, mods[0], env, "go", args...); !ok {
		return Candidate{}, fmt.Errorf("%w: go build %s: %s", ErrBuildFailed, b.Package, clip(o))
	}
	vet := b.Vet
	if len(vet) == 0 {
		vet = []string{b.Package}
	}
	if o, ok := ex.RunEnv(ctx, mods[0], env, "go", append([]string{"vet"}, vet...)...); !ok {
		return Candidate{}, fmt.Errorf("%w: go vet %s: %s", ErrVetFailed, strings.Join(vet, " "), clip(o))
	}
	id, err := smoke(ctx, x, out, dir, commits[0])
	if err != nil {
		return Candidate{}, err
	}
	if err := checkTuple(ctx, x, mods, commits, "after build"); err != nil {
		return Candidate{}, err
	}
	digest, size, err := hashFile(out)
	if err != nil {
		return Candidate{}, fmt.Errorf("%w: hash candidate: %v", ErrBuildFailed, err)
	}
	return Candidate{
		Kind: KindBuild, Path: out, Generation: b.Generation, Commit: commits[0],
		Digest: digest, Size: size, AppVersion: id.AppVersion,
	}, nil
}

// modules returns ModuleDirs as distinct absolute paths.
func (b BuildPinned) modules() ([]string, error) {
	if len(b.ModuleDirs) == 0 {
		return nil, fmt.Errorf("%w: no module directories", ErrInvalidSource)
	}
	mods := make([]string, 0, len(b.ModuleDirs))
	seen := make(map[string]bool, len(b.ModuleDirs))
	for _, d := range b.ModuleDirs {
		if strings.TrimSpace(d) == "" {
			return nil, fmt.Errorf("%w: empty module directory", ErrInvalidSource)
		}
		abs, err := filepath.Abs(d)
		if err != nil {
			return nil, fmt.Errorf("%w: module directory %q: %v", ErrInvalidSource, d, err)
		}
		key := abs
		if runtime.GOOS == "windows" {
			key = strings.ToLower(abs)
		}
		if seen[key] {
			return nil, fmt.Errorf("%w: module directory %q is listed twice", ErrInvalidSource, abs)
		}
		seen[key] = true
		mods = append(mods, abs)
	}
	return mods, nil
}

// commits returns Commits lower-cased, one full commit per module directory.
func (b BuildPinned) commits() ([]string, error) {
	if len(b.Commits) != len(b.ModuleDirs) {
		return nil, fmt.Errorf("%w: %d commits for %d module directories", ErrInvalidSource, len(b.Commits), len(b.ModuleDirs))
	}
	out := make([]string, len(b.Commits))
	for i, c := range b.Commits {
		c = strings.ToLower(strings.TrimSpace(c))
		if !fullCommit(c) {
			return nil, fmt.Errorf("%w: pinned commit %q is not a full 40-hex object ID", ErrInvalidSource, clip(b.Commits[i]))
		}
		out[i] = c
	}
	return out, nil
}

func (b BuildPinned) platform() (string, string) {
	goos, goarch := b.GOOS, b.GOARCH
	if goos == "" {
		goos = runtime.GOOS
	}
	if goarch == "" {
		goarch = runtime.GOARCH
	}
	return goos, goarch
}

// pinnedEnv returns the environment every go child of the build runs with, plus the cleanup
// for the generated go.work.
func (b BuildPinned) pinnedEnv(ctx context.Context, ex EnvExecutor, mods []string) ([]string, func(), error) {
	goos, goarch := b.platform()
	base := []string{"GOFLAGS=", "GOTOOLCHAIN=local", "GOOS=" + goos, "GOARCH=" + goarch}
	version := b.GoWorkPin
	if version == "" {
		out, ok := ex.RunEnv(ctx, mods[0], append(append([]string{}, base...), "GOWORK=off"), "go", "env", "GOVERSION")
		fields := strings.Fields(out)
		if !ok || len(fields) == 0 || !goVersionPattern.MatchString(strings.TrimPrefix(fields[0], "go")) {
			return nil, nil, fmt.Errorf("%w: read the local Go toolchain version: %s", ErrBuildFailed, clip(out))
		}
		version = strings.TrimPrefix(fields[0], "go")
	}
	wdir, err := os.MkdirTemp("", "fak-deploykit-gowork-*")
	if err != nil {
		return nil, nil, fmt.Errorf("%w: create pinned workspace: %v", ErrBuildFailed, err)
	}
	cleanup := func() { _ = os.RemoveAll(wdir) }
	var work strings.Builder
	fmt.Fprintf(&work, "go %s\n\nuse (\n", version)
	for _, m := range mods {
		fmt.Fprintf(&work, "\t%s\n", strconv.Quote(filepath.ToSlash(m)))
	}
	work.WriteString(")\n")
	file := filepath.Join(wdir, "go.work")
	if err := os.WriteFile(file, []byte(work.String()), 0o600); err != nil {
		cleanup()
		return nil, nil, fmt.Errorf("%w: write pinned workspace: %v", ErrBuildFailed, err)
	}
	return append(base, "GOWORK="+file), cleanup, nil
}

// checkTuple proves every repository of the tuple is clean (no tracked or untracked changes)
// and at its pinned commit.
func checkTuple(ctx context.Context, x deploykit.Executor, mods, commits []string, when string) error {
	for i, m := range mods {
		out, ok := x.Run(ctx, m, "git", "rev-parse", "--verify", "HEAD")
		head := strings.ToLower(strings.TrimSpace(out))
		if !ok || !fullCommit(head) {
			return fmt.Errorf("%w: %s %s: cannot resolve HEAD: %s", ErrPinnedCommitMismatch, when, m, clip(out))
		}
		if head != commits[i] {
			return fmt.Errorf("%w: %s %s is at %s, pinned %s", ErrPinnedCommitMismatch, when, m, head, commits[i])
		}
		out, ok = x.Run(ctx, m, "git", "status", "--porcelain", "--untracked-files=normal")
		if !ok {
			return fmt.Errorf("%w: %s %s: cannot inspect the worktree: %s", ErrDirtySource, when, m, clip(out))
		}
		if strings.TrimSpace(out) != "" {
			return fmt.Errorf("%w: %s %s has changes: %s", ErrDirtySource, when, m, clip(out))
		}
	}
	return nil
}

// goTarget reports whether s can be passed to go build/vet as a package argument.
func goTarget(s string) bool {
	s = strings.TrimSpace(s)
	return s != "" && !strings.HasPrefix(s, "-")
}
