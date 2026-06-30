package nightrun

import (
	"context"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"time"

	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

// prebuild.go — collapse the per-task `go run` compile cost. A full --loop pass was
// dominated by compile time, not measurement: every benchmark Run is `go run ./cmd/<x>`,
// and `go run` recompiles the dependency tree PER task (~37s each on a cold box, #965).
// Pre-building each ./cmd/<x> to a binary ONCE — sharing the build cache across all
// benches so the heavy std/deps compile is paid once — turns an N-task offline pass from
// ~N×37s into one compile + N fast binary runs.
//
// The rewrite is conservative: it fires ONLY for a Run that begins exactly `go run ./cmd/<x>`
// with a clean package path, and falls back to the original `go run` (never errors the task)
// if the build fails. A `fak <verb>` Run or any non-go-run shape is returned unchanged.
//
// The cache is PER RunLoop invocation (newGoRunCache + defer cleanup): the compiled binaries
// must not outlive the run — TMPDIR is NOT cleared on process exit, so a package global would
// leak a dir of binaries per run.

// buildPrebuildBudget bounds a single `go build` of one ./cmd/<x> package, so a hung or
// pathologically slow compile cannot stall an unattended --loop. On expiry the build is killed
// and memoized as a failure, and the task falls back to the timeout-wrapped `go run`.
const buildPrebuildBudget = 5 * time.Minute

// goRunCache memoizes one built binary per `./cmd/<x>` package path for the lifetime of ONE
// RunLoop. The build dir is a single temp dir, removed by cleanup when the run ends. The loop
// is sequential (one task at a time), so the mutex is defensive, not a contended hot path.
type goRunCache struct {
	mu   sync.Mutex
	dir  string            // temp output dir (created lazily)
	bins map[string]string // "./cmd/radixbench" -> "<dir>/radixbench[.exe]" ("" = build failed, do not retry)
	// gocache is the GOCACHE the prebuild `go build` runs under, resolved once by the
	// detached-run preflight: "" means the ambient environment is fine (a derivable default),
	// a non-empty path is a per-run default this cache owns when the box has neither GOCACHE
	// nor a HOME to derive one from (the detached/unattended `setsid` case, #991).
	gocache string
}

// newGoRunCache returns an empty per-run build cache.
func newGoRunCache() *goRunCache { return &goRunCache{bins: map[string]string{}} }

// gocacheStatus is the verdict of the detached-run build-cache preflight (#991): whether
// `go build` can locate a cache, and — when it cannot be derived from the ambient
// environment — the per-run default we will set instead, plus a one-line remediation an
// operator/agent can act on. It distinguishes a missing cache (fixable: we set a default)
// from a real compile error, so a detached run's artifact says which one happened.
type gocacheStatus struct {
	// Usable is true when `go build` will find a cache: either GOCACHE is set, or HOME (the
	// source of the default os.UserCacheDir-derived GOCACHE) is set, or we provisioned a default.
	Usable bool
	// Default is the GOCACHE path this run will export when the ambient env has none, or "" when
	// the ambient environment already resolves a cache (nothing to override).
	Default string
	// Remediation names the env var an operator should export for a durable fix (HOME or GOCACHE),
	// "" when the ambient cache is already usable.
	Remediation string
}

// preflightGoCache decides whether the prebuild `go build` can locate a build cache, and when
// it cannot be derived from the ambient environment (no GOCACHE AND no HOME / no per-OS user
// cache dir — the minimal-environment detached run), names a per-run default under buildDir so
// the offline go-run benches still build instead of every one failing with "build cache is
// required, but could not be located" (#991). It is pure over its getenv/cachedir seams so a
// test drives every branch without touching the real environment.
//
// Resolution order mirrors cmd/go's own: an explicit GOCACHE wins; else a derivable default
// (os.UserCacheDir, which needs HOME on unix / LocalAppData on Windows); else our per-run
// default. The remediation always names the durable fix (export HOME or GOCACHE) so a detached
// run's artifact is actionable rather than a bare build failure.
func preflightGoCache(getenv func(string) string, userCacheDir func() (string, error), buildDir string) gocacheStatus {
	if strings.TrimSpace(getenv("GOCACHE")) != "" {
		return gocacheStatus{Usable: true} // explicit GOCACHE — nothing to do
	}
	if _, err := userCacheDir(); err == nil {
		return gocacheStatus{Usable: true} // a default is derivable (HOME / LocalAppData present)
	}
	// Neither set nor derivable: the detached/minimal-environment case. Provision a per-run
	// GOCACHE under the build dir so the benches build, and still surface the durable fix.
	def := filepath.Join(buildDir, "gocache")
	return gocacheStatus{
		Usable:      true,
		Default:     def,
		Remediation: "set HOME or GOCACHE for a durable build cache (nightrun used a per-run default at " + def + ")",
	}
}

// resolveGoCache runs the preflight once per cache lifetime (guarded by the caller holding mu)
// and memoizes the chosen per-run GOCACHE. It is a no-op after the first call. dir must already
// be created so the per-run default lives under the run's temp tree (removed by cleanup).
func (c *goRunCache) resolveGoCache() {
	if c.gocache != "" || c.dir == "" {
		return
	}
	st := preflightGoCache(os.Getenv, os.UserCacheDir, c.dir)
	if st.Default != "" {
		_ = os.MkdirAll(st.Default, 0o755)
		c.gocache = st.Default
	}
}

// buildEnv returns the environment for the prebuild `go build` subprocess: the ambient
// environment, with GOCACHE forced to the per-run default when the preflight provisioned one
// (the detached-run case) and otherwise inherited unchanged. nil means "inherit os.Environ()".
func (c *goRunCache) buildEnv() []string {
	if c.gocache == "" {
		return nil
	}
	return append(os.Environ(), "GOCACHE="+c.gocache)
}

// cleanup removes the build dir and resets the cache. RunLoop defers it so the compiled
// binaries are explicitly removed at the end of the run (the OS does not reclaim TMPDIR on exit).
func (c *goRunCache) cleanup() {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.dir != "" {
		_ = os.RemoveAll(c.dir)
		c.dir = ""
	}
	c.bins = map[string]string{}
}

// maybePrebuildRun rewrites `go run ./cmd/<x> <args...>` into `<built-binary> <args...>` when
// the package builds, so the executor runs a compiled binary instead of paying `go run`'s
// compile every task. Any other Run shape (a `fak` verb, a shell pipeline, a build failure)
// is returned unchanged — this can only make a go-run task faster, never break one.
func (c *goRunCache) maybePrebuildRun(ctx context.Context, root, run string) string {
	pkg, args, ok := parseGoRun(run)
	if !ok {
		return run
	}
	bin := c.binaryFor(ctx, root, pkg)
	if bin == "" {
		return run // build failed (or no toolchain) — fall back to the original go run
	}
	if args == "" {
		return quoteIfNeeded(bin)
	}
	return quoteIfNeeded(bin) + " " + args
}

// parseGoRun matches `go run ./cmd/<x>[ <args>]` and returns the package path, the remaining
// args, and whether it matched. It is deliberately strict: only a single `./cmd/<x>` package
// spec (no extra build flags, no multiple packages) is rewritten; anything else is left to go run.
func parseGoRun(run string) (pkg, args string, ok bool) {
	fields := strings.Fields(run)
	if len(fields) < 3 || fields[0] != "go" || fields[1] != "run" {
		return "", "", false
	}
	p := fields[2]
	if !strings.HasPrefix(p, "./cmd/") {
		return "", "", false
	}
	if strings.ContainsAny(p, " \t") || strings.Contains(p, "...") {
		return "", "", false
	}
	return p, strings.Join(fields[3:], " "), true
}

func (c *goRunCache) binaryFor(ctx context.Context, root, pkg string) string {
	c.mu.Lock()
	defer c.mu.Unlock()
	if bin, seen := c.bins[pkg]; seen {
		return bin // cached path, or "" for a prior build failure (do not rebuild)
	}
	if c.dir == "" {
		d, err := os.MkdirTemp("", "fak-nightrun-bin-")
		if err != nil {
			c.bins[pkg] = ""
			return ""
		}
		c.dir = d
	}
	// Detached-run preflight (#991): if the environment has neither GOCACHE nor a derivable
	// default (HOME / per-OS user cache dir), `go build` would abort with "build cache is
	// required, but could not be located" and EVERY go-run bench would be lost. Provision a
	// per-run GOCACHE under the build dir once so the benches build anyway.
	c.resolveGoCache()
	name := filepath.Base(pkg)
	if runtime.GOOS == "windows" {
		name += ".exe"
	}
	out := filepath.Join(c.dir, name)
	// Bound the compile: a hung/slow `go build` is killed at buildPrebuildBudget and memoized
	// as a failure, so the loop falls back to the timeout-wrapped `go run` rather than stalling.
	buildCtx, cancel := context.WithTimeout(ctx, buildPrebuildBudget)
	defer cancel()
	cmd := exec.CommandContext(buildCtx, "go", "build", "-o", out, pkg)
	windowgate.ConfigureBackgroundCommand(cmd)
	cmd.Dir = root
	cmd.Env = c.buildEnv() // nil => inherit os.Environ(); set only when the preflight provisioned a default
	if err := cmd.Run(); err != nil {
		c.bins[pkg] = "" // remember the failure so the loop falls back to go run, once
		return ""
	}
	c.bins[pkg] = out
	return out
}

// quoteIfNeeded wraps a path containing a space in double quotes so the shell re-parses it as
// one token (the temp dir can contain spaces on Windows). A space-free path is returned as-is.
func quoteIfNeeded(p string) string {
	if strings.ContainsAny(p, " \t") {
		return `"` + p + `"`
	}
	return p
}
