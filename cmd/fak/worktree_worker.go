package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/leaseref"
	"github.com/anthony-chaudhary/fak/internal/procguard"
	"github.com/anthony-chaudhary/fak/internal/windowgate"
	"github.com/anthony-chaudhary/fak/internal/witness"
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
              [--lease-id ID] [--owner-pid PID] [--capacity-reason WHY]
                   Create ONE worker's DETACHED worktree pinned at trunk HEAD
                   (or --base-sha), stamped with owner PID, lease, and timestamp.
                   Above the advisory setpoint of 50, --capacity-reason records
                   why growth is needed; omission warns but never blocks prepare.
                   Prints {ok, path, base_sha, reused, env, ...}.
      land --worktree D [--base-sha S] [--msg-file F] [--paths p ...] [--verify go-build]
           [--core-lock-maintenance-witness CLAIM] [--recovery-remote R]
           [--require-remote-recovery] [--disambiguation-timeout-ms N]
                   Apply the worktree's diff-since-base onto the trunk as one
                   signed-off commit. Prints {ok, applied, committed, ...}.
                   The optional disambiguation deadline is 1..900000 ms and uses
                   the oracle's existing resolver; omitted means 120000 ms.
                   Managed issue lands re-verify the Top-5 comment. Roll back with
                   FAK_THOUGHT_CHECK_MODE=observe|off (default: enforce).
                   A diff touching a hard-self core-locked path is REFUSED with
                   CORE_SELF_MODIFY unless the witness claim (flag, or a
                   Core-lock-maintenance-witness: trailer in the commit message)
                   resolves CONFIRMED — the same lock fak commit enforces.
      reap --worktree D [--superseded-by SHA] [--max-wait D]
                   Release ONE clean worker worktree within a shared deadline. A dirty
                   worktree is preserved by default. --superseded-by authorizes force
                   removal only when SHA is on trunk and exactly matches its bytes.
                   Prints a typed {ok, code, removed|preserved, ...} receipt.
      reap --all-cold [--apply] [--age-floor-min N] [--even-if-unlanded]
                   Bulk cold sweep: enumerate every worker worktree and reap only the
                   COLD ones — lane lease dead, past the age floor, AND working tree
                   clean. One still holding uncommitted work is KEPT and reported as
                   held_by_work: land or abandon that diff to reclaim its disk.
                   DRY-RUN by default — reports the would-reap set and deletes nothing;
                   pass --apply (or FAK_WORKTREE_COLD_COLLECT=apply) to actually collect.
                   --even-if-unlanded also collects the held ones, DESTROYING that work.
      gc [--max-age D] [--dry-run|--apply]
                   Owner-stamped leak GC. Selects only old, clean worktrees whose
                   owner PID is dead AND stamped lease is released. DRY-RUN by default;
                   --apply force-removes selected worktrees and prunes git admin entries.
      list [--json] [--capacity-reason WHY] [--remote R] [--fetch]
                   List the live per-worker worktrees. The default preserves the
                   existing {count, paths, inventory} output; --json emits the
                   typed association/liveness/cleanliness/lifecycle inventory.
      publish --remote R [--dry-run|--apply]
                   Explicitly publish one bounded path-scrubbed per-host snapshot.
      recover [--remote R] [--fetch] [--cleanup REF] [--force]
              [--cleanup-remote REF] [--apply] [--allow-peer] [--worktree-name NAME]
                   List durable off-branch land candidates and their LANDED or
                   RECOVERABLE state. Cleanup refuses unlanded refs unless forced.
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
	case "gc":
		worktreeWorkerGC(argv[1:])
	case "list":
		worktreeWorkerList(argv[1:])
	case "publish":
		worktreeWorkerPublish(argv[1:])
	case "recover":
		worktreeWorkerRecover(argv[1:])
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

func worktreeWorkerProgressEmitter(w io.Writer) func(workerworktree.LandProgressEvent) {
	enc := json.NewEncoder(w)
	enc.SetEscapeHTML(false)
	return func(event workerworktree.LandProgressEvent) { _ = enc.Encode(event) }
}

// worktreePrepareOut is the prepare JSON: the primitive's Result plus the child
// env the caller needs to spawn the worker in the isolated worktree (the Python
// CLI adds `env` on a successful prepare the same way). Embedding flattens Result's
// fields to the top level.
type worktreePrepareOut struct {
	workerworktree.Result
	Env      map[string]string               `json:"env,omitempty"`
	Capacity workerworktree.CapacityAdvisory `json:"capacity"`
}

func worktreeWorkerPrepare(argv []string) {
	fs := flag.NewFlagSet("worktree worker prepare", flag.ExitOnError)
	lane := fs.String("lane", "", "worker's lane (e.g. cmd, gateway) — a segment of the worktree dir name")
	key := fs.String("key", "", "worker's unique key (issue number, wave id, pid) — hashed into the dir name")
	baseSHA := fs.String("base-sha", "", "commit to pin the detached worktree at (default: trunk HEAD)")
	leaseID := fs.String("lease-id", "", "lease identity to retain in the owner stamp (default: FAK_LEASE_ID or resolve-<lane>)")
	ownerPID := fs.Int("owner-pid", os.Getpid(), "owner process PID to retain in the owner stamp")
	capacityReason := fs.String("capacity-reason", "", "why worker-worktree growth above the advisory setpoint is needed (advisory; never blocks)")
	message := fs.String("message", "", "intended signed commit message retained for lifecycle recovery")
	var paths repeatedString
	fs.Var(&paths, "path", "explicit intended land path (repeatable; required with --message for LAND_READY inventory)")
	wtRoot := fs.String("wt-root", "", "parent dir for the worktree (default: FLEET_WORKER_WORKTREE_ROOT or per-OS scratch)")
	root := fs.String("root", "", "repo root (default: discover from cwd)")
	fs.Parse(argv)

	repoRoot := worktreeWorkerRoot(*root)
	capacityCensus := workerworktree.CapacityCensusFor(repoRoot, nil)
	owner := workerworktree.OwnerStamp{PID: *ownerPID, LeaseID: strings.TrimSpace(*leaseID), CreatedAt: time.Now().UTC()}
	if owner.LeaseID == "" {
		owner.LeaseID = strings.TrimSpace(os.Getenv("FAK_LEASE_ID"))
	}
	if owner.LeaseID == "" && strings.TrimSpace(*lane) != "" {
		owner.LeaseID = "resolve-" + strings.TrimSpace(*lane)
	}
	res := workerworktree.PrepareOwnedBounded(repoRoot, *lane, *key, strings.TrimSpace(*baseSHA), strings.TrimSpace(*wtRoot), owner, 2*time.Minute)
	prospectiveCount := len(capacityCensus.Paths)
	if res.OK && !res.Reused {
		prospectiveCount++
	}
	capacity := worktreeWorkerCapacityAdvisory(repoRoot, capacityCensus, prospectiveCount, *capacityReason, nil)
	out := worktreePrepareOut{Result: res, Capacity: capacity}
	if res.OK && res.Path != "" {
		out.Env = workerworktree.WorktreeEnv(nil, res.Path)
		if strings.TrimSpace(*message) != "" || len(paths) > 0 {
			if strings.TrimSpace(*message) == "" || len(paths) == 0 {
				res.OK, res.Reason = false, "--message and at least one --path must be supplied together"
				out.Result = res
			} else if err := workerworktree.SaveIntent(res.Path, res.BaseSHA, *message, paths); err != nil {
				res.OK, res.Reason = false, "save worker land intent: "+err.Error()
				out.Result = res
			}
		}
	}
	worktreeWorkerWriteCapacityHuman(os.Stderr, capacity)
	worktreeWorkerEmit(out)
	if !res.OK {
		os.Exit(1)
	}
}

func worktreeCommitSubject(wtPath, msgFile string) string {
	if strings.TrimSpace(msgFile) != "" {
		if b, err := os.ReadFile(msgFile); err == nil {
			for _, line := range strings.Split(string(b), "\n") {
				line = strings.TrimSpace(line)
				if line != "" {
					return line
				}
			}
		}
	}
	cmd := windowgate.Command("git", "-C", wtPath, "log", "-1", "--format=%s")
	windowgate.ConfigureBackgroundCommand(cmd)
	if out, err := cmd.Output(); err == nil {
		return strings.TrimSpace(string(out))
	}
	return ""
}

func isFixSubject(s string) bool {
	lower := strings.ToLower(strings.TrimSpace(s))
	return strings.HasPrefix(lower, "fix(") || strings.HasPrefix(lower, "fix:")
}

func verifyWorkerLandSymptom(wtPath string) workerworktree.Result {
	resolver := witness.NewWithRunner(nil, wtPath)
	outcome := resolver.ResolveSymptom(context.Background(), "HEAD", true)
	switch outcome {
	case abi.WitnessConfirmed:
		return workerworktree.Result{OK: true}
	default:
		return workerworktree.Result{
			OK:     false,
			Code:   "SYMPTOM_UNWITNESSED",
			Reason: "SYMPTOM_UNWITNESSED: fix commit must include a test that fails on parent and passes on fix (red-then-green); bypass only with --unsafe-skip-symptom-witness",
		}
	}
}

func runWorktreeWorkerLand(stdout, stderr io.Writer, argv []string) (workerworktree.Result, int) {
	fs := flag.NewFlagSet("worktree worker land", flag.ContinueOnError)
	fs.SetOutput(stderr)
	worktree := fs.String("worktree", "", "the worker's worktree dir to land from (required)")
	baseSHA := fs.String("base-sha", "", "the sha the worktree was pinned at — the diff ref (default: HEAD)")
	msgFile := fs.String("msg-file", "", "commit message file for `git commit -s -F` (default: derive from the worktree tip)")
	verify := fs.String("verify", "off", "pre-land witness run IN the worktree: off | go-build")
	root := fs.String("root", "", "repo root the change lands on (default: discover from cwd)")
	disambiguationTimeoutMS := fs.String("disambiguation-timeout-ms", "", "one shared whole-tree disambiguation deadline in milliseconds (1..900000; default 120000; no retries)")
	coreLockWitness := fs.String("core-lock-maintenance-witness", "",
		"independent witness claim that clears a hard-self core-lock land (same claim vocabulary as fak commit)")
	recoveryRemote := fs.String("recovery-remote", "", "publish/read-back candidate on this git remote before trunk CAS")
	requireRemote := fs.Bool("require-remote-recovery", false, "refuse trunk CAS unless remote recovery read-back succeeds")
	unsafeSkipSymptomWitness := fs.Bool("unsafe-skip-symptom-witness", false,
		"bypass mandatory fail-to-pass symptom witness for fix(*) commits")
	var paths repeatedString
	fs.Var(&paths, "paths", "path to scope the commit to (repeatable); omit to commit the whole applied diff")
	if err := fs.Parse(argv); err != nil {
		return workerworktree.Result{OK: false, Reason: err.Error()}, 2
	}

	worktreeDir := strings.TrimSpace(*worktree)
	if worktreeDir == "" {
		fmt.Fprintln(stderr, "fak worktree worker land: --worktree is required")
		return workerworktree.Result{OK: false, Reason: "--worktree is required"}, 2
	}
	repoRoot := worktreeWorkerRoot(*root)

	// Mandatory fail-to-pass symptom witness for fix(*) commits (#10926)
	subj := worktreeCommitSubject(worktreeDir, strings.TrimSpace(*msgFile))
	if isFixSubject(subj) && !*unsafeSkipSymptomWitness {
		symptomRes := verifyWorkerLandSymptom(worktreeDir)
		if !symptomRes.OK {
			return symptomRes, 1
		}
	}

	var hook workerworktree.VerifyHook
	switch strings.ToLower(strings.TrimSpace(*verify)) {
	case "", "off", "none":
		hook = nil
	case "go-build", "gobuild", "build":
		hook = worktreeWorkerGoBuildVerify
	default:
		fmt.Fprintf(stderr, "fak worktree worker land: unknown --verify %q (want off|go-build)\n", *verify)
		return workerworktree.Result{OK: false, Reason: fmt.Sprintf("unknown --verify %q", *verify)}, 2
	}

	opts := []workerworktree.LandOption{
		workerworktree.WithCoreLockWitness(*coreLockWitness),
		workerworktree.WithLandProgress(worktreeWorkerProgressEmitter(stderr)),
	}
	if strings.TrimSpace(*recoveryRemote) != "" || *requireRemote {
		remote := strings.TrimSpace(*recoveryRemote)
		if remote == "" {
			remote = "origin"
		}
		opts = append(opts, workerworktree.WithRecoveryRemote(remote, *requireRemote))
	}
	timeoutSet := flagWasSet(fs, "disambiguation-timeout-ms")
	res, err := withWorkerLandDisambiguationTimeout(*disambiguationTimeoutMS, timeoutSet, func() workerworktree.Result {
		return workerworktree.Land(repoRoot, worktreeDir, strings.TrimSpace(*baseSHA), strings.TrimSpace(*msgFile), []string(paths), hook, nil, opts...)
	})
	if err != nil {
		res = workerworktree.Result{
			OK: false, Code: workerworktree.DisambiguationTimeoutCode,
			Reason: "configure worker land disambiguation timeout: " + err.Error(),
		}
	}
	return res, 0
}

func worktreeWorkerLand(argv []string) {
	res, code := runWorktreeWorkerLand(os.Stdout, os.Stderr, argv)
	worktreeWorkerEmit(res)
	if !res.OK {
		if code != 0 {
			os.Exit(code)
		}
		os.Exit(1)
	}
}

// withWorkerLandDisambiguationTimeout bridges the explicit CLI spelling onto the
// workerworktree resolver's bootstrap input for exactly one Land call. The CLI
// deliberately does not parse or clamp the value: the existing resolver remains
// the single authority for the inclusive 1..900000 ms bound and writes the same
// requested/effective/default fields into the disambiguation receipt. An omitted
// flag leaves the environment untouched, preserving the 120000 ms default (or an
// explicitly supplied bootstrap environment for older callers).
func withWorkerLandDisambiguationTimeout(raw string, explicit bool, land func() workerworktree.Result) (workerworktree.Result, error) {
	if !explicit {
		return land(), nil
	}
	previous, hadPrevious := os.LookupEnv(workerworktree.DisambiguationTimeoutEnv)
	if err := os.Setenv(workerworktree.DisambiguationTimeoutEnv, raw); err != nil {
		return workerworktree.Result{}, err
	}
	defer func() {
		if hadPrevious {
			_ = os.Setenv(workerworktree.DisambiguationTimeoutEnv, previous)
		} else {
			_ = os.Unsetenv(workerworktree.DisambiguationTimeoutEnv)
		}
	}()
	return land(), nil
}

func worktreeWorkerReap(argv []string) {
	flags := flag.NewFlagSet("worktree worker reap", flag.ExitOnError)
	worktree := flags.String("worktree", "", "the worker's managed worktree dir (single-worktree mode)")
	supersededBy := flags.String("superseded-by", "", "single-worktree mode: authorize dirty cleanup only when this commit is on trunk and byte-equivalent to the worktree")
	maxWait := flags.Duration("max-wait", 10*time.Second, "single-worktree mode: shared wall-clock deadline for inspection, verification, and removal")
	allCold := flags.Bool("all-cold", false, "bulk mode: enumerate ALL worker worktrees and reap only the COLD ones (dead lane lease, past the age floor, and clean). DRY-RUN unless --apply")
	apply := flags.Bool("apply", false, "with --all-cold, actually delete the cold worktrees (default: dry-run report only). Env "+worktreeColdApplyEnv+"=apply is equivalent")
	ageFloorMin := flags.Int("age-floor-min", int(workerworktree.DefaultColdAgeFloor/time.Minute), "with --all-cold, the age grace floor in minutes — a dead-lease worktree younger than this is kept")
	evenIfUnlanded := flags.Bool("even-if-unlanded", false, "with --all-cold, ALSO reap worktrees kept only because they still hold uncommitted work. DESTROYS that work — for reclaiming disk once the diffs are known to be abandoned")
	root := flags.String("root", "", "repo root (default: discover from cwd)")
	flags.Parse(argv)

	repoRoot := worktreeWorkerRoot(*root)
	if *allCold {
		effectiveApply := *apply || strings.EqualFold(strings.TrimSpace(os.Getenv(worktreeColdApplyEnv)), "apply")
		// This command's typed decision ledger is already the before/apply receipt.
		// The generic lifecycle wrapper would run a second serial git-status census
		// across every worktree before this bounded classifier can start.
		worktreeWorkerReapAllCold(repoRoot, effectiveApply, time.Duration(*ageFloorMin)*time.Minute, *evenIfUnlanded)
		return
	}
	if strings.TrimSpace(*worktree) == "" {
		fmt.Fprintln(os.Stderr, "fak worktree worker reap: --worktree is required (or pass --all-cold for the bulk cold sweep)")
		os.Exit(2)
	}
	if *maxWait <= 0 {
		worktreeWorkerEmit(workerworktree.Result{
			OK: false, Code: workerworktree.ReapCodeTimeout, Path: strings.TrimSpace(*worktree), Preserved: true,
			Reason: "--max-wait must be positive",
		})
		os.Exit(2)
	}
	fmt.Fprintf(os.Stderr, "REAP_PROGRESS code=REAP_STARTED max_wait=%s\n", maxWait.String())
	ctx, cancel := context.WithTimeout(context.Background(), *maxWait)
	defer cancel()
	git := workerworktree.BoundedGitRunner(ctx)
	finishLifecycle := beginAutomaticWIPLifecycleWithGit(repoRoot, "worker-reap", os.Stderr, git)
	defer finishLifecycle()
	res := workerworktree.ReapChecked(repoRoot, strings.TrimSpace(*worktree), strings.TrimSpace(*supersededBy), git)
	worktreeWorkerEmit(res)
	if !res.OK {
		os.Exit(1)
	}
}

func worktreeWorkerGC(argv []string) {
	flags := flag.NewFlagSet("worktree worker gc", flag.ExitOnError)
	maxAge := flags.Duration("max-age", workerworktree.DefaultColdAgeFloor, "minimum owner-stamp age before a dead-owner/released-lease worktree is eligible (for example 30m, 24h)")
	dryRun := flags.Bool("dry-run", false, "list candidates and delete nothing (this is already the default)")
	apply := flags.Bool("apply", false, "force-remove selected worktrees and prune git administrative entries")
	root := flags.String("root", "", "repo root (default: discover from cwd)")
	flags.Parse(argv)
	if *dryRun && *apply {
		fmt.Fprintln(os.Stderr, "fak worktree worker gc: --dry-run and --apply are mutually exclusive")
		os.Exit(2)
	}
	repoRoot := worktreeWorkerRoot(*root)
	report := workerworktree.GarbageCollect(repoRoot, nil, workerworktree.GCOptions{
		Now:          time.Now(),
		MaxAge:       *maxAge,
		Apply:        *apply,
		ProcessAlive: dispatchPIDAlive,
		LeaseLive:    worktreeStampedLeaseOracle(repoRoot, time.Now()),
	})
	worktreeWorkerEmit(report)
	if *apply {
		fmt.Fprintf(os.Stderr, "reaped %d/%d owner-dead, lease-released worktrees (apply)\n", report.Reaped, report.WouldReap)
	} else {
		fmt.Fprintf(os.Stderr, "would reap %d owner-dead, lease-released worktrees, 0 deleted (dry-run; pass --apply to collect)\n", report.WouldReap)
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
	// BytesKnown and Bytes are retained for existing JSON consumers. They mirror
	// the provenance-honest reclaim fields on ColdWorktree.
	BytesKnown bool  `json:"bytes_known"`
	Bytes      int64 `json:"bytes"`
	Removed    bool  `json:"removed,omitempty"`
}

// worktreeColdReapFailure records a worktree that was reapable in the planning
// snapshot but failed one of the apply-time revalidation or removal checks.
type worktreeColdReapFailure struct {
	Path   string `json:"path"`
	Reason string `json:"reason"`
}

// worktreeColdReapOut is the single JSON object the bulk cold sweep prints: the mode
// (dry-run|apply), the age floor, the per-worktree decision ledger, and the roll-up
// counts/bytes. In dry-run Reaped is always 0 and Bytes is the reclaimable total.
type worktreeColdReapOut struct {
	Mode                 string                    `json:"mode"`
	AgeFloorMin          int                       `json:"age_floor_min"`
	Worktrees            []worktreeColdReapItem    `json:"worktrees"`
	Failures             []worktreeColdReapFailure `json:"failures"`
	WouldReap            int                       `json:"would_reap"`
	Reaped               int                       `json:"reaped"`
	Bytes                int64                     `json:"bytes"`
	EligibleBytes        int64                     `json:"eligible_bytes"`
	EligibleBytesKnown   int                       `json:"eligible_bytes_known"`
	EligibleBytesUnknown int                       `json:"eligible_bytes_unknown"`
	ReapedBytes          int64                     `json:"reaped_bytes"`
	ReapedBytesKnown     int                       `json:"reaped_bytes_known"`
	ReapedBytesUnknown   int                       `json:"reaped_bytes_unknown"`
	// HeldByWork counts the worktrees that cleared the lease and age gates and were
	// kept only because they still carry uncommitted work, with their reclaimable
	// bytes. Reported separately from the generic keep because it is a triage queue,
	// not a wait: that disk comes back only once each diff is landed or abandoned.
	HeldByWork             int                          `json:"held_by_work"`
	HeldByWorkBytes        int64                        `json:"held_by_work_bytes"`
	HeldByWorkBytesUnknown int                          `json:"held_by_work_bytes_unknown"`
	UnregisteredResidue    []workerworktree.ResidueItem `json:"unregistered_residue,omitempty"`
}

// worktreeWorkerReapAllCold is the bulk cold sweep (#5351): enumerate every worker
// worktree, decide which are COLD (dead lane lease, past the age floor, and clean) via
// the pure workerworktree.ColdReapList plan, and — only under an explicit apply opt-in —
// Reap each cold one with the SAME per-id workerworktree.Reap the single mode uses.
// The default is a DRY-RUN that ledgers the would-reap set and deletes nothing.
func worktreeWorkerReapAllCold(repoRoot string, apply bool, ageFloor time.Duration, evenIfUnlanded bool) {
	worktreeWorkerReapAllColdTo(repoRoot, apply, ageFloor, evenIfUnlanded, os.Stdout, os.Stderr)
}

func worktreeWorkerReapAllColdTo(repoRoot string, apply bool, ageFloor time.Duration, evenIfUnlanded bool, stdout, stderr io.Writer) {
	if !apply && strings.EqualFold(strings.TrimSpace(os.Getenv(worktreeColdApplyEnv)), "apply") {
		apply = true
	}
	now := time.Now()
	progress := json.NewEncoder(stderr)
	progress.SetEscapeHTML(false)
	out := worktreeColdReapReportWithOptions(repoRoot, apply, ageFloor, now, evenIfUnlanded, workerworktree.ColdReapOptions{
		Concurrency: worktreeColdStatusConcurrency,
		Progress: func(event workerworktree.ColdReapProgress) {
			_ = progress.Encode(event)
		},
	})
	residueOpts := workerworktree.ResidueOptions{Repo: repoRoot, Now: now, AgeFloor: ageFloor}
	residue, residueErr := workerworktree.CollectUnregisteredResidue(repoRoot, residueOpts)
	if residueErr == nil && apply {
		residue, residueErr = workerworktree.ApplyUnregisteredResidue(residue, residueOpts)
	}
	out.UnregisteredResidue = residue
	if residueErr != nil {
		out.Failures = append(out.Failures, worktreeColdReapFailure{Path: "unregistered-residue", Reason: residueErr.Error()})
	}
	final := json.NewEncoder(stdout)
	final.SetEscapeHTML(false)
	_ = final.Encode(out)
	// Progress and human summaries stay on stderr so stdout remains the single
	// backward-compatible final JSON receipt.
	kept := len(out.Worktrees) - out.WouldReap
	if apply {
		fmt.Fprintf(stderr, "reaped %d/%d cold worktrees (%s), %d kept (apply)\n",
			out.Reaped, out.WouldReap, coldReapBytesSummary(out.ReapedBytes, out.ReapedBytesUnknown), kept)
	} else {
		fmt.Fprintf(stderr, "would reap %d cold worktrees (%s), 0 deleted (dry-run; pass --apply to collect), %d kept\n",
			out.WouldReap, coldReapBytesSummary(out.EligibleBytes, out.EligibleBytesUnknown), kept)
	}
	// Unlanded work is called out on its own line: it is the one keep-reason the
	// operator must ACT on, and folding it into the "kept" tally is what let 17
	// worktrees of unlanded diffs read as ordinary reclaimable disk.
	if out.HeldByWork > 0 {
		verb := "held"
		if evenIfUnlanded {
			verb = "REAPED DESPITE"
		}
		fmt.Fprintf(stderr, "%s %d worktree(s) by unlanded work (%s) — land or abandon each before reclaiming\n",
			verb, out.HeldByWork, coldReapBytesSummary(out.HeldByWorkBytes, out.HeldByWorkBytesUnknown))
	}
}

func coldReapBytesSummary(known int64, unknown int) string {
	switch {
	case unknown == 0:
		return humanBytes(known)
	case known == 0:
		return fmt.Sprintf("unknown for %d worktree(s)", unknown)
	default:
		return fmt.Sprintf("%s known + %d worktree(s) unknown", humanBytes(known), unknown)
	}
}

// worktreeColdReapReport is the core of the bulk cold sweep, split out so a test drives
// it directly against a real repo without going through argv/stdout. It enumerates the
// worker worktrees, decides the cold set via workerworktree.ColdReapList under the
// lease-liveness gate, and — only when apply is true — Reaps each cold one. It NEVER
// deletes in dry-run: Reaped/ReapedBytes stay 0 and Bytes is the reclaimable total.
func worktreeColdReapReport(repoRoot string, apply bool, ageFloor time.Duration, now time.Time, evenIfUnlanded bool) worktreeColdReapOut {
	return worktreeColdReapReportWithOptions(repoRoot, apply, ageFloor, now, evenIfUnlanded, workerworktree.ColdReapOptions{})
}

func worktreeColdReapReportWithOptions(repoRoot string, apply bool, ageFloor time.Duration, now time.Time, evenIfUnlanded bool, coldOpts workerworktree.ColdReapOptions) worktreeColdReapOut {
	return worktreeColdReapReportWithOptionsAndProbes(
		repoRoot, apply, ageFloor, now, evenIfUnlanded, coldOpts,
		worktreeColdProcessSnapshot,
		worktreeColdProcessLive,
		func(root, path string) workerworktree.Result {
			return workerworktree.ForceReap(root, path, nil)
		},
	)
}

// worktreeColdReapReportWithProbes separates the two destructive-seam probes so
// tests can make process liveness change between planning and apply and can prove
// that a refused reap never reaches workerworktree.Reap.
func worktreeColdReapReportWithProbes(
	repoRoot string,
	apply bool,
	ageFloor time.Duration,
	now time.Time,
	evenIfUnlanded bool,
	processSnapshot func(paths []string) (map[string]bool, error),
	processLive func(path string) (bool, error),
	reap func(root, path string) workerworktree.Result,
) worktreeColdReapOut {
	return worktreeColdReapReportWithOptionsAndProbes(repoRoot, apply, ageFloor, now, evenIfUnlanded, workerworktree.ColdReapOptions{}, processSnapshot, processLive, reap)
}

func worktreeColdReapReportWithOptionsAndProbes(
	repoRoot string,
	apply bool,
	ageFloor time.Duration,
	now time.Time,
	evenIfUnlanded bool,
	coldOpts workerworktree.ColdReapOptions,
	processSnapshot func(paths []string) (map[string]bool, error),
	processLive func(path string) (bool, error),
	reap func(root, path string) workerworktree.Result,
) worktreeColdReapOut {
	if ageFloor <= 0 {
		ageFloor = workerworktree.DefaultColdAgeFloor
	}
	if processSnapshot == nil {
		processSnapshot = func([]string) (map[string]bool, error) {
			return nil, fmt.Errorf("process liveness snapshot is unavailable")
		}
	}
	if processLive == nil {
		processLive = func(string) (bool, error) {
			return true, fmt.Errorf("process liveness probe is unavailable")
		}
	}
	if reap == nil {
		reap = func(_, path string) workerworktree.Result {
			return workerworktree.Result{Path: path, Reason: "reap function is unavailable"}
		}
	}
	oracle := worktreeLiveLeaseOracle(repoRoot, now)
	if coldOpts.Concurrency <= 0 {
		coldOpts.Concurrency = worktreeColdStatusConcurrency
	}
	plan := workerworktree.ColdReapListWithOptions(repoRoot, nil, now, ageFloor, oracle, coldOpts)
	return worktreeColdReapReportFromPlan(
		repoRoot,
		apply,
		ageFloor,
		evenIfUnlanded,
		plan,
		processSnapshot,
		processLive,
		reap,
	)
}

// worktreeColdReapReportFromPlan reduces one immutable eligibility snapshot into
// either a dry-run or apply receipt. Apply may refuse a planned item during its
// final safety revalidation, but it never re-enumerates or substitutes a second
// eligible set.
func worktreeColdReapReportFromPlan(
	repoRoot string,
	apply bool,
	ageFloor time.Duration,
	evenIfUnlanded bool,
	plan []workerworktree.ColdWorktree,
	processSnapshot func(paths []string) (map[string]bool, error),
	processLive func(path string) (bool, error),
	reap func(root, path string) workerworktree.Result,
) worktreeColdReapOut {
	out := worktreeColdReapOut{
		Mode:        "dry-run",
		AgeFloorMin: int(ageFloor / time.Minute),
		Worktrees:   make([]worktreeColdReapItem, 0, len(plan)),
		Failures:    make([]worktreeColdReapFailure, 0),
	}
	if apply {
		out.Mode = "apply"
	}
	processesByPath, processSnapshotErr := batchColdProcessRefs(plan, evenIfUnlanded, processSnapshot)
	for _, c := range plan {
		item := worktreeColdReapItem{
			ColdWorktree: c,
			BytesKnown:   c.ReclaimBytesKnown,
			Bytes:        c.ReclaimBytes,
		}
		// The override promotes ONLY the unlanded-work keeps. A live lease or a
		// worktree under the age floor stays kept either way: those protect an
		// in-flight land, which no disk-reclamation flag should be able to override.
		shouldReap := c.Eligible || (evenIfUnlanded && c.HeldByWork)
		if c.HeldByWork {
			out.HeldByWork++
			if item.BytesKnown {
				out.HeldByWorkBytes += item.Bytes
			} else {
				out.HeldByWorkBytesUnknown++
			}
		}
		if shouldReap {
			switch {
			case processSnapshotErr != nil:
				shouldReap = false
				item.Eligible = false
				item.Reason = "kept: process liveness probe failed: " + processSnapshotErr.Error()
			case processesByPath[c.Path]:
				shouldReap = false
				item.Eligible = false
				item.Reason = "kept: an active process still references this worktree"
			}
		}
		var beforeModTime time.Time
		if shouldReap {
			fi, err := os.Stat(c.Path)
			if err != nil {
				shouldReap = false
				item.Eligible = false
				item.Reason = "kept: worktree modtime could not be read: " + err.Error()
			} else {
				beforeModTime = fi.ModTime()
			}
		}
		if shouldReap {
			out.WouldReap++
			if item.ReclaimBytesKnown {
				out.Bytes += item.ReclaimBytes
				out.EligibleBytes += item.ReclaimBytes
				out.EligibleBytesKnown++
			} else {
				out.EligibleBytesUnknown++
			}
			if apply {
				applyOracle := worktreeLiveLeaseOracle(repoRoot, time.Now())
				leaseLive := applyOracle(c.Path)
				unlanded := workerworktree.UnlandedCount(c.Path, nil)
				applyProcessLive, applyProcessErr := processLive(c.Path)
				currentInfo, modErr := os.Stat(c.Path)

				failureReason := ""
				switch {
				case leaseLive:
					failureReason = "lease_live"
				case unlanded != 0:
					failureReason = "unlanded_work"
				case applyProcessErr != nil:
					failureReason = "process_probe_error"
				case applyProcessLive:
					failureReason = "process_live"
				case modErr != nil:
					failureReason = "modtime_unreadable"
				case !currentInfo.ModTime().Equal(beforeModTime):
					failureReason = "modtime_changed"
				}
				if failureReason != "" {
					out.Failures = append(out.Failures, worktreeColdReapFailure{Path: c.Path, Reason: failureReason})
					out.Worktrees = append(out.Worktrees, item)
					continue
				}

				res := reap(repoRoot, c.Path)
				_, statErr := os.Stat(c.Path)
				switch {
				case !res.Removed:
					out.Failures = append(out.Failures, worktreeColdReapFailure{Path: c.Path, Reason: "remove_failed"})
				case !os.IsNotExist(statErr):
					out.Failures = append(out.Failures, worktreeColdReapFailure{Path: c.Path, Reason: "directory_remains"})
				default:
					item.Removed = true
					out.Reaped++
					if item.BytesKnown {
						out.ReapedBytes += item.Bytes
						out.ReapedBytesKnown++
					} else {
						out.ReapedBytesUnknown++
					}
				}
			}
		}
		out.Worktrees = append(out.Worktrees, item)
	}
	return out
}

const worktreeColdStatusConcurrency = 8

func batchColdProcessRefs(plan []workerworktree.ColdWorktree, evenIfUnlanded bool, snapshot func([]string) (map[string]bool, error)) (map[string]bool, error) {
	paths := make([]string, 0, len(plan))
	for _, c := range plan {
		if c.Eligible || (evenIfUnlanded && c.HeldByWork) {
			paths = append(paths, c.Path)
		}
	}
	if len(paths) == 0 {
		return map[string]bool{}, nil
	}
	return snapshot(paths)
}

// worktreeColdProcessSnapshot performs the expensive OS process census once for
// the entire planning set. Destructive apply still calls worktreeColdProcessLive
// per path immediately before removal so a process that starts after planning is
// preserved.
func worktreeColdProcessSnapshot(paths []string) (map[string]bool, error) {
	result := make(map[string]bool, len(paths))
	if len(paths) == 0 {
		return result, nil
	}
	procs, collectErr := procguard.CollectRelations()
	if collectErr != "" {
		return nil, fmt.Errorf("process census: %s", collectErr)
	}
	cleanPaths := make(map[string]string, len(paths))
	for _, path := range paths {
		cleanPaths[path] = filepath.Clean(path)
	}
	self := os.Getpid()
	for _, proc := range procs {
		if proc.PID == self {
			continue
		}
		for path, cleanPath := range cleanPaths {
			if !result[path] && strings.Contains(proc.Cmdline, cleanPath) {
				result[path] = true
			}
		}
	}
	return result, nil
}

// worktreeColdProcessLive is the production process-liveness gate for a worker
// worktree. A census failure is not an empty host: it fails closed by reporting
// the path live and returning the census error. The current process is excluded
// because its own argv may name the target while it performs the sweep.
func worktreeColdProcessLive(path string) (bool, error) {
	procs, collectErr := procguard.CollectRelations()
	if collectErr != "" {
		return true, fmt.Errorf("process census: %s", collectErr)
	}
	cleanPath := filepath.Clean(path)
	self := os.Getpid()
	for _, proc := range procs {
		if proc.PID == self {
			continue
		}
		if strings.Contains(proc.Cmdline, cleanPath) {
			return true, nil
		}
	}
	return false, nil
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

// worktreeStampedLeaseOracle resolves a stamp's exact lease id while preserving the
// later cold sweep's lane-level protection: a coarse resolve-<lane> stamp is live when
// any live issue lease on that lane exists, and an exact issue stamp is live when its
// exact record or a coarse lane record exists. Read failures fail toward LIVE.
func worktreeStampedLeaseOracle(root string, now time.Time) workerworktree.LeaseIDLiveFn {
	live, _, err := leaseref.NewInDir(root).Live(context.Background(), now)
	if err != nil {
		return func(string) bool { return true }
	}
	ids := map[string]bool{}
	lanes := map[string]bool{}
	for _, rec := range live {
		ids[strings.ToLower(strings.TrimSpace(rec.ID))] = true
		for _, lane := range dispatchLeaseLanes(rec.ID) {
			lanes[strings.ToLower(lane)] = true
		}
	}
	return func(leaseID string) bool {
		leaseID = strings.ToLower(strings.TrimSpace(leaseID))
		if leaseID == "" {
			return true
		}
		if ids[leaseID] {
			return true
		}
		for _, lane := range dispatchLeaseLanes(leaseID) {
			if lanes[strings.ToLower(lane)] {
				return true
			}
		}
		return false
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

// worktreeWorkerListOut is the list JSON: a count and the sorted live-worktree
// paths (never null — an empty slice renders `[]`), mirroring the Python CLI.
type worktreeWorkerListOut struct {
	Count     int                             `json:"count"`
	Paths     []string                        `json:"paths"`
	Inventory []workerworktree.InventoryRow   `json:"inventory"`
	Capacity  workerworktree.CapacityAdvisory `json:"capacity"`
}

type worktreeWorkerRecoverOut struct {
	OK            bool                                `json:"ok"`
	Count         int                                 `json:"count"`
	Candidates    []workerworktree.RecoveryEntry      `json:"candidates"`
	Cleaned       string                              `json:"cleaned,omitempty"`
	Reason        string                              `json:"reason,omitempty"`
	RemoteCleanup *workerworktree.RemoteCleanupReport `json:"remote_cleanup,omitempty"`
}

// worktreeWorkerRecover is the crash-resume inventory for isolated lands. With
// no mutation flags it is read-only. Cleanup is guarded by HEAD reachability;
// --force is deliberately required to discard an unlanded candidate.
func worktreeWorkerRecover(argv []string) {
	fs := flag.NewFlagSet("worktree worker recover", flag.ExitOnError)
	root := fs.String("root", "", "repo root (default: discover from cwd)")
	cleanup := fs.String("cleanup", "", "delete one landed recovery ref")
	force := fs.Bool("force", false, "allow cleanup of an unlanded recovery ref")
	remote := fs.String("remote", "origin", "remote whose worker-land mirror is inspected")
	fetch := fs.Bool("fetch", false, "refresh the read-only remote mirror before listing")
	cleanupRemote := fs.String("cleanup-remote", "", "report/delete one remote recovery ref after default-branch ancestry proof")
	apply := fs.Bool("apply", false, "apply remote cleanup; otherwise report-only")
	allowPeer := fs.Bool("allow-peer", false, "permit cleanup of a peer-named recovery ref")
	worktreeName := fs.String("worktree-name", "", "local worktree identity used for remote cleanup ownership guard")
	fs.Parse(argv)

	repoRoot := worktreeWorkerRoot(*root)
	finishLifecycle := beginAutomaticWIPLifecycle(repoRoot, "crash-recovery", os.Stderr)
	defer finishLifecycle()
	out := worktreeWorkerRecoverOut{Candidates: []workerworktree.RecoveryEntry{}}
	if *fetch {
		if err := workerworktree.FetchRecoveryMirror(repoRoot, *remote, nil); err != nil {
			out.Reason = err.Error()
			worktreeWorkerEmit(out)
			finishLifecycle()
			os.Exit(1)
		}
	}
	if *cleanupRemote != "" {
		plan := workerworktree.CleanupRemoteRecoveryRef(repoRoot, *remote, *cleanupRemote, *worktreeName, *allowPeer, *apply, nil)
		out.RemoteCleanup = &plan
		if *apply && !plan.Applied {
			out.Reason = plan.Reason
			worktreeWorkerEmit(out)
			finishLifecycle()
			os.Exit(1)
		}
	}
	if *cleanup != "" {
		if err := workerworktree.DeleteRecoveryRef(repoRoot, *cleanup, *force, nil); err != nil {
			out.Reason = err.Error()
			worktreeWorkerEmit(out)
			finishLifecycle()
			os.Exit(1)
		}
		out.Cleaned = *cleanup
	}
	items, err := workerworktree.RecoveryEntries(repoRoot, nil)
	if err != nil {
		out.Reason = err.Error()
		worktreeWorkerEmit(out)
		finishLifecycle()
		os.Exit(1)
	}
	out.OK = true
	out.Count = len(items)
	out.Candidates = items
	worktreeWorkerEmit(out)
}
