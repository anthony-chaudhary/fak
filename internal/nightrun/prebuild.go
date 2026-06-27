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
}

// newGoRunCache returns an empty per-run build cache.
func newGoRunCache() *goRunCache { return &goRunCache{bins: map[string]string{}} }

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
	cmd.Dir = root
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
