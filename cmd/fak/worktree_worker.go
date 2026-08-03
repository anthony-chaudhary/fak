package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io/fs"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/leaseref"
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
           [--core-lock-maintenance-witness CLAIM]
                   Apply the worktree's diff-since-base onto the trunk as one
                   signed-off commit. Prints {ok, applied, committed, ...}.
                   A diff touching a hard-self core-locked path is REFUSED with
                   CORE_SELF_MODIFY unless the witness claim (flag, or a
                   Core-lock-maintenance-witness: trailer in the commit message)
                   resolves CONFIRMED — the same lock fak commit enforces.
      reap --worktree D
                   Force-remove ONE finished worker worktree. Prints {ok, removed, ...}.
      reap --all-cold [--apply] [--age-floor-min N] [--even-if-unlanded]
                   Bulk cold sweep: enumerate every worker worktree and reap only the
                   COLD ones — lane lease dead, past the age floor, AND working tree
                   clean. One still holding uncommitted work is KEPT and reported as
                   held_by_work: land or abandon that diff to reclaim its disk.
                   DRY-RUN by default — reports the would-reap set and deletes nothing;
                   pass --apply (or FAK_WORKTREE_COLD_COLLECT=apply) to actually collect.
                   --even-if-unlanded also collects the held ones, DESTROYING that work.
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
	// Same flag name and same semantics as `fak commit --core-lock-maintenance-witness`
	// (#5392): the claim is RESOLVED against independent evidence, and only a CONFIRMED
	// resolution clears a hard-self core-lock pathset. Without it the land is refused
	// with CORE_SELF_MODIFY. A worker that has no CLI to pass a flag through carries the
	// same claim as a workerworktree.CoreLockWitnessTrailer line in its commit message.
	coreLockWitness := fs.String("core-lock-maintenance-witness", "",
		"independent witness claim that clears a hard-self core-lock land (same claim vocabulary as fak commit)")
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

	res := workerworktree.Land(repoRoot, strings.TrimSpace(*worktree), strings.TrimSpace(*baseSHA), strings.TrimSpace(*msgFile), []string(paths), hook, nil,
		workerworktree.WithCoreLockWitness(*coreLockWitness))
	worktreeWorkerEmit(res)
	if !res.OK {
		os.Exit(1)
	}
}

func worktreeWorkerReap(argv []string) {
	flags := flag.NewFlagSet("worktree worker reap", flag.ExitOnError)
	worktree := flags.String("worktree", "", "the worker's worktree dir to force-remove (single-worktree mode)")
	allCold := flags.Bool("all-cold", false, "bulk mode: enumerate ALL worker worktrees and reap only the COLD ones (dead lane lease, past the age floor, and clean). DRY-RUN unless --apply")
	apply := flags.Bool("apply", false, "with --all-cold, actually delete the cold worktrees (default: dry-run report only). Env "+worktreeColdApplyEnv+"=apply is equivalent")
	ageFloorMin := flags.Int("age-floor-min", int(workerworktree.DefaultColdAgeFloor/time.Minute), "with --all-cold, the age grace floor in minutes — a dead-lease worktree younger than this is kept")
	evenIfUnlanded := flags.Bool("even-if-unlanded", false, "with --all-cold, ALSO reap worktrees kept only because they still hold uncommitted work. DESTROYS that work — for reclaiming disk once the diffs are known to be abandoned")
	root := flags.String("root", "", "repo root (default: discover from cwd)")
	flags.Parse(argv)

	repoRoot := worktreeWorkerRoot(*root)

	if *allCold {
		worktreeWorkerReapAllCold(repoRoot, *apply, time.Duration(*ageFloorMin)*time.Minute, *evenIfUnlanded)
		return
	}

	if strings.TrimSpace(*worktree) == "" {
		fmt.Fprintln(os.Stderr, "fak worktree worker reap: --worktree is required (or pass --all-cold for the bulk cold sweep)")
		os.Exit(2)
	}
	res := workerworktree.Reap(repoRoot, strings.TrimSpace(*worktree), nil)
	worktreeWorkerEmit(res)
	if !res.OK {
		os.Exit(1)
	}
}

// worktreeColdApplyEnv opts the bulk cold sweep into DELETING rather than reporting.
// The sweep DEFAULTS to a dry-run — it deletes only under an explicit apply opt-in
// (this env set to "apply", or the --apply flag) — mirroring the census keep-side
// default the sibling deleters land (#5079 grace-prune, #5349 growthgate's
// FAK_GARDEN_GROWTH_COLLECT=apply). A false reap of a live worker's worktree corrupts
// an in-flight land, so the collect side is never the default.
const worktreeColdApplyEnv = "FAK_WORKTREE_COLD_COLLECT"

// worktreeColdReapItem is one worktree's line in the bulk-sweep ledger: the pure
// cold decision (path, age, lease-live, eligible, reason) plus its on-disk bytes and,
// under --apply, whether it was actually removed.
type worktreeColdReapItem struct {
	workerworktree.ColdWorktree
	Bytes   int64 `json:"bytes"`
	Removed bool  `json:"removed,omitempty"`
}

// worktreeColdReapOut is the single JSON object the bulk cold sweep prints: the mode
// (dry-run|apply), the age floor, the per-worktree decision ledger, and the roll-up
// counts/bytes. In dry-run Reaped is always 0 and Bytes is the reclaimable total.
type worktreeColdReapOut struct {
	Mode        string                 `json:"mode"`
	AgeFloorMin int                    `json:"age_floor_min"`
	Worktrees   []worktreeColdReapItem `json:"worktrees"`
	WouldReap   int                    `json:"would_reap"`
	Reaped      int                    `json:"reaped"`
	Bytes       int64                  `json:"bytes"`
	ReapedBytes int64                  `json:"reaped_bytes"`
	// HeldByWork counts the worktrees that cleared the lease and age gates and were
	// kept only because they still carry uncommitted work, with their reclaimable
	// bytes. Reported separately from the generic keep because it is a triage queue,
	// not a wait: that disk comes back only once each diff is landed or abandoned.
	HeldByWork      int   `json:"held_by_work"`
	HeldByWorkBytes int64 `json:"held_by_work_bytes"`
}

// worktreeWorkerReapAllCold is the bulk cold sweep (#5351): enumerate every worker
// worktree, decide which are COLD (dead lane lease, past the age floor, and clean) via
// the pure workerworktree.ColdReapList plan, and — only under an explicit apply opt-in —
// Reap each cold one with the SAME per-id workerworktree.Reap the single mode uses.
// The default is a DRY-RUN that ledgers the would-reap set and deletes nothing.
func worktreeWorkerReapAllCold(repoRoot string, apply bool, ageFloor time.Duration, evenIfUnlanded bool) {
	if !apply && strings.EqualFold(strings.TrimSpace(os.Getenv(worktreeColdApplyEnv)), "apply") {
		apply = true
	}
	out := worktreeColdReapReport(repoRoot, apply, ageFloor, time.Now(), evenIfUnlanded)
	worktreeWorkerEmit(out)
	// The human one-liner goes to stderr so stdout stays exactly one JSON object.
	kept := len(out.Worktrees) - out.WouldReap
	if apply {
		fmt.Fprintf(os.Stderr, "reaped %d/%d cold worktrees (%s), %d kept (apply)\n",
			out.Reaped, out.WouldReap, humanBytes(out.ReapedBytes), kept)
	} else {
		fmt.Fprintf(os.Stderr, "would reap %d cold worktrees (%s), 0 deleted (dry-run; pass --apply to collect), %d kept\n",
			out.WouldReap, humanBytes(out.Bytes), kept)
	}
	// Unlanded work is called out on its own line: it is the one keep-reason the
	// operator must ACT on, and folding it into the "kept" tally is what let 17
	// worktrees of unlanded diffs read as ordinary reclaimable disk.
	if out.HeldByWork > 0 {
		verb := "held"
		if evenIfUnlanded {
			verb = "REAPED DESPITE"
		}
		fmt.Fprintf(os.Stderr, "%s %d worktree(s) by unlanded work (%s) — land or abandon each before reclaiming\n",
			verb, out.HeldByWork, humanBytes(out.HeldByWorkBytes))
	}
}

// worktreeColdReapReport is the core of the bulk cold sweep, split out so a test drives
// it directly against a real repo without going through argv/stdout. It enumerates the
// worker worktrees, decides the cold set via workerworktree.ColdReapList under the
// lease-liveness gate, and — only when apply is true — Reaps each cold one. It NEVER
// deletes in dry-run: Reaped/ReapedBytes stay 0 and Bytes is the reclaimable total.
func worktreeColdReapReport(repoRoot string, apply bool, ageFloor time.Duration, now time.Time, evenIfUnlanded bool) worktreeColdReapOut {
	if ageFloor <= 0 {
		ageFloor = workerworktree.DefaultColdAgeFloor
	}
	oracle := worktreeLiveLeaseOracle(repoRoot, now)
	plan := workerworktree.ColdReapList(repoRoot, nil, now, ageFloor, oracle)

	out := worktreeColdReapOut{
		Mode:        "dry-run",
		AgeFloorMin: int(ageFloor / time.Minute),
		Worktrees:   make([]worktreeColdReapItem, 0, len(plan)),
	}
	if apply {
		out.Mode = "apply"
	}
	for _, c := range plan {
		item := worktreeColdReapItem{ColdWorktree: c, Bytes: worktreeDirBytes(c.Path)}
		// The override promotes ONLY the unlanded-work keeps. A live lease or a
		// worktree under the age floor stays kept either way: those protect an
		// in-flight land, which no disk-reclamation flag should be able to override.
		reap := c.Eligible || (evenIfUnlanded && c.HeldByWork)
		if c.HeldByWork {
			out.HeldByWork++
			out.HeldByWorkBytes += item.Bytes
		}
		if reap {
			out.WouldReap++
			out.Bytes += item.Bytes
			if apply {
				if res := workerworktree.Reap(repoRoot, c.Path, nil); res.Removed {
					item.Removed = true
					out.Reaped++
					out.ReapedBytes += item.Bytes
				}
			}
		}
		out.Worktrees = append(out.Worktrees, item)
	}
	return out
}

// worktreeLiveLeaseOracle builds the lease-liveness gate the bulk cold sweep keys on:
// a worktree is protected (treated as live) when a LIVE lane lease shares its lane. It
// reads leaseref's live records — the SAME TTL-based liveness the dispatcher's own lane
// admission uses (acquireDispatchLaneLease) — so it keys on the lease's own expiry, not
// a dead pid (a dead pid on a lease is EXPECTED and does not alone prove staleness).
// FAIL TOWARD KEEPING: if the lease store cannot be read, every worktree is protected,
// so an unreadable store never causes a false reap.
func worktreeLiveLeaseOracle(root string, now time.Time) workerworktree.LeaseLiveFn {
	live, _, err := leaseref.NewInDir(root).Live(context.Background(), now)
	if err != nil {
		return func(string) bool { return true }
	}
	lanes := map[string]bool{}
	for _, rec := range live {
		for _, lane := range dispatchLeaseLanes(rec.ID) {
			lanes[strings.ToLower(lane)] = true
		}
	}
	return func(wtPath string) bool {
		lane := workerworktree.LaneOf(wtPath)
		if lane == "" {
			return true // unclassifiable worktree -> keep
		}
		return lanes[strings.ToLower(lane)]
	}
}

// dispatchLeaseLanes returns the candidate lane token(s) a dispatch lease id binds to.
// Lease ids are "resolve-<lane>" (a lane lease) or "resolve-<lane>-<issue>" (an issue
// lease) — see dispatchLaneLeaseID / dispatchIssueLeaseID. Both the full tail and the
// issue-suffix-stripped tail are returned so a worktree lane matches either form; a
// non-resolve id yields nothing.
func dispatchLeaseLanes(id string) []string {
	rest := strings.TrimPrefix(id, "resolve-")
	if rest == id || rest == "" {
		return nil
	}
	out := []string{rest}
	if i := strings.LastIndex(rest, "-"); i > 0 && isDigitsOnly(rest[i+1:]) {
		out = append(out, rest[:i])
	}
	return out
}

func isDigitsOnly(s string) bool {
	if s == "" {
		return false
	}
	for _, r := range s {
		if r < '0' || r > '9' {
			return false
		}
	}
	return true
}

// worktreeDirBytes is the on-disk size of a worktree, for the reclaimable-bytes ledger.
// Best-effort: an unreadable dir or entry is skipped (counts 0), never an error — the
// bytes total is a report field, not a gate.
func worktreeDirBytes(dir string) int64 {
	var total int64
	_ = filepath.WalkDir(dir, func(_ string, d fs.DirEntry, err error) error {
		if err != nil || d.IsDir() {
			return nil
		}
		if info, ierr := d.Info(); ierr == nil {
			total += info.Size()
		}
		return nil
	})
	return total
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
