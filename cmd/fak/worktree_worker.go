package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"os/exec"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/windowgate"
	"github.com/anthony-chaudhary/fak/internal/workerworktree"
)

// cmdWorktreeVerb fronts `fak worktree <sub>`. Today it hosts the `worker`
// subcommand (the CLI face of internal/workerworktree, #3182). The sibling
// `witness` subcommand is authored separately (cmd/fak/worktree.go) and folds in
// here once it lands on trunk — until then this dispatcher owns the `worktree`
// verb so `fak worktree worker` is reachable. Kept a distinct symbol from that
// in-flight file's cmdWorktree so the shared, peer-dirty working tree still
// builds while both land.
func cmdWorktreeVerb(argv []string) {
	if len(argv) == 0 {
		worktreeWorkerUsage()
		os.Exit(2)
	}
	switch argv[0] {
	case "worker":
		cmdWorktreeWorker(argv[1:])
	case "-h", "--help", "help":
		worktreeWorkerUsage()
	default:
		fmt.Fprintf(os.Stderr, "fak worktree: unknown subcommand %q\n", argv[0])
		worktreeWorkerUsage()
		os.Exit(2)
	}
}

func worktreeWorkerUsage() {
	fmt.Fprintln(os.Stderr, strings.TrimSpace(`
fak worktree <subcommand>

  worker <op>   Per-worker git worktree isolation (#3182). Ops:
      prepare --lane <l> --key <k> [--base-sha S] [--wt-root D]
                   Create ONE worker's DETACHED worktree pinned at trunk HEAD
                   (or --base-sha). Prints {ok, path, base_sha, reused, env, ...}.
      land --worktree D [--base-sha S] [--msg-file F] [--paths p ...] [--verify go-build]
                   Apply the worktree's diff-since-base onto the trunk as one
                   signed-off commit. Prints {ok, applied, committed, ...}.
      reap --worktree D
                   Force-remove a finished worker worktree. Prints {ok, removed, ...}.
      list         List the live per-worker worktrees. Prints {count, paths}.
`))
}

// cmdWorktreeWorker routes `fak worktree worker <op>` to the matching
// internal/workerworktree primitive and prints exactly one JSON object, mirroring
// the tools/worker_worktree.py CLI contract. Production drives the real git via the
// package's default runner (nil GitRunner). Every op fails open: a git error is a
// JSON result with ok=false, never a crash.
func cmdWorktreeWorker(argv []string) {
	if len(argv) == 0 {
		worktreeWorkerUsage()
		os.Exit(2)
	}
	switch argv[0] {
	case "prepare":
		worktreeWorkerPrepare(argv[1:])
	case "land":
		worktreeWorkerLand(argv[1:])
	case "reap":
		worktreeWorkerReap(argv[1:])
	case "list":
		worktreeWorkerList(argv[1:])
	case "-h", "--help", "help":
		worktreeWorkerUsage()
	default:
		fmt.Fprintf(os.Stderr, "fak worktree worker: unknown op %q\n", argv[0])
		worktreeWorkerUsage()
		os.Exit(2)
	}
}

// worktreeWorkerRoot resolves the repo root a worker op runs against: the explicit
// --root, else discovered from cwd. An empty result is a usage error (mirrors the
// Python CLI, which requires a resolvable repo).
func worktreeWorkerRoot(flagVal string) string {
	root := strings.TrimSpace(flagVal)
	if root == "" {
		root = discoverRepoRoot()
	}
	if root == "" {
		fmt.Fprintln(os.Stderr, "fak worktree worker: could not resolve a git repo root (pass --root)")
		os.Exit(2)
	}
	return root
}

// worktreeWorkerEmit prints one JSON object (compact, one line) — the single
// machine-readable result the caller parses. Uses SetEscapeHTML(false) so paths
// with `&` etc. render verbatim.
func worktreeWorkerEmit(v any) {
	enc := json.NewEncoder(os.Stdout)
	enc.SetEscapeHTML(false)
	_ = enc.Encode(v)
}

// worktreePrepareOut is the prepare JSON: the primitive's Result plus the child
// env the caller needs to spawn the worker in the isolated worktree (the Python
// CLI adds `env` on a successful prepare the same way). Embedding flattens Result's
// fields to the top level.
type worktreePrepareOut struct {
	workerworktree.Result
	Env map[string]string `json:"env,omitempty"`
}

func worktreeWorkerPrepare(argv []string) {
	fs := flag.NewFlagSet("worktree worker prepare", flag.ExitOnError)
	lane := fs.String("lane", "", "worker's lane (e.g. cmd, gateway) — a segment of the worktree dir name")
	key := fs.String("key", "", "worker's unique key (issue number, wave id, pid) — hashed into the dir name")
	baseSHA := fs.String("base-sha", "", "commit to pin the detached worktree at (default: trunk HEAD)")
	wtRoot := fs.String("wt-root", "", "parent dir for the worktree (default: FLEET_WORKER_WORKTREE_ROOT or per-OS scratch)")
	root := fs.String("root", "", "repo root (default: discover from cwd)")
	fs.Parse(argv)

	repoRoot := worktreeWorkerRoot(*root)
	res := workerworktree.Prepare(repoRoot, *lane, *key, strings.TrimSpace(*baseSHA), strings.TrimSpace(*wtRoot), nil)
	out := worktreePrepareOut{Result: res}
	if res.OK && res.Path != "" {
		out.Env = workerworktree.WorktreeEnv(nil, res.Path)
	}
	worktreeWorkerEmit(out)
	if !res.OK {
		os.Exit(1)
	}
}

func worktreeWorkerLand(argv []string) {
	fs := flag.NewFlagSet("worktree worker land", flag.ExitOnError)
	worktree := fs.String("worktree", "", "the worker's worktree dir to land from (required)")
	baseSHA := fs.String("base-sha", "", "the sha the worktree was pinned at — the diff ref (default: HEAD)")
	msgFile := fs.String("msg-file", "", "commit message file for `git commit -s -F` (default: derive from the worktree tip)")
	verify := fs.String("verify", "off", "pre-land witness run IN the worktree: off | go-build")
	root := fs.String("root", "", "repo root the change lands on (default: discover from cwd)")
	var paths repeatedString
	fs.Var(&paths, "paths", "path to scope the commit to (repeatable); omit to commit the whole applied diff")
	fs.Parse(argv)

	if strings.TrimSpace(*worktree) == "" {
		fmt.Fprintln(os.Stderr, "fak worktree worker land: --worktree is required")
		os.Exit(2)
	}
	repoRoot := worktreeWorkerRoot(*root)

	var hook workerworktree.VerifyHook
	switch strings.ToLower(strings.TrimSpace(*verify)) {
	case "", "off", "none":
		hook = nil
	case "go-build", "gobuild", "build":
		hook = worktreeWorkerGoBuildVerify
	default:
		fmt.Fprintf(os.Stderr, "fak worktree worker land: unknown --verify %q (want off|go-build)\n", *verify)
		os.Exit(2)
	}

	res := workerworktree.Land(repoRoot, strings.TrimSpace(*worktree), strings.TrimSpace(*baseSHA), strings.TrimSpace(*msgFile), []string(paths), hook, nil)
	worktreeWorkerEmit(res)
	if !res.OK {
		os.Exit(1)
	}
}

func worktreeWorkerReap(argv []string) {
	fs := flag.NewFlagSet("worktree worker reap", flag.ExitOnError)
	worktree := fs.String("worktree", "", "the worker's worktree dir to force-remove (required)")
	root := fs.String("root", "", "repo root (default: discover from cwd)")
	fs.Parse(argv)

	if strings.TrimSpace(*worktree) == "" {
		fmt.Fprintln(os.Stderr, "fak worktree worker reap: --worktree is required")
		os.Exit(2)
	}
	repoRoot := worktreeWorkerRoot(*root)
	res := workerworktree.Reap(repoRoot, strings.TrimSpace(*worktree), nil)
	worktreeWorkerEmit(res)
	if !res.OK {
		os.Exit(1)
	}
}

// worktreeWorkerListOut is the list JSON: a count and the sorted live-worktree
// paths (never null — an empty slice renders `[]`), mirroring the Python CLI.
type worktreeWorkerListOut struct {
	Count int      `json:"count"`
	Paths []string `json:"paths"`
}

func worktreeWorkerList(argv []string) {
	fs := flag.NewFlagSet("worktree worker list", flag.ExitOnError)
	root := fs.String("root", "", "repo root (default: discover from cwd)")
	fs.Parse(argv)

	repoRoot := worktreeWorkerRoot(*root)
	n, paths := workerworktree.Count(repoRoot, nil)
	if paths == nil {
		paths = []string{}
	}
	worktreeWorkerEmit(worktreeWorkerListOut{Count: n, Paths: paths})
}

// worktreeWorkerGoBuildVerify is the `--verify go-build` witness: run `go build
// ./...` INSIDE the worker's worktree (with the isolated GOCACHE/GOTMPDIR) before
// its diff lands on the trunk, so a worktree that does not compile refuses the
// land. FAIL-OPEN: if the go toolchain is missing it returns ok (the pre-land
// gate never wedges a land just because `go` is absent), mirroring the Python
// module's _go_build_verify.
func worktreeWorkerGoBuildVerify(wtPath string) (bool, string) {
	if _, err := exec.LookPath("go"); err != nil {
		return true, "go toolchain not found — skipping build verify (fail open)"
	}
	cmd := exec.Command("go", "build", "./...")
	cmd.Dir = wtPath
	windowgate.ConfigureBackgroundCommand(cmd)
	env := workerworktree.WorktreeEnv(nil, wtPath)
	cmd.Env = append(os.Environ(), "GOCACHE="+env["GOCACHE"], "GOTMPDIR="+env["GOTMPDIR"])
	out, err := cmd.CombinedOutput()
	if err == nil {
		return true, ""
	}
	detail := strings.TrimSpace(string(out))
	if len(detail) > 500 {
		detail = detail[len(detail)-500:]
	}
	return false, "go build ./... failed: " + detail
}
