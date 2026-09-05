package main

// fak garden -- the garden bundle: one default-on, read-only fold over the
// repo's self-maintenance passes (the scorecard control pane + fresh status), so
// "run the gardening" is one command instead of three. It runs each member
// (grandfathered Python tools), reads its control-pane JSON, and folds one
// schema/ok/verdict/finding/reason/next_action envelope. It mutates nothing.
// --check is the CI gate (exit non-zero only when a gating member regressed or a
// pass failed to run); --deep adds the slower fleet loop-audit member. Skipped
// when FAK_GARDEN is off (the env-side governor brake).

import (
	"bytes"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/commitlifecycle"
	"github.com/anthony-chaudhary/fak/internal/gardenbudget"
	"github.com/anthony-chaudhary/fak/internal/gardenbundle"
	"github.com/anthony-chaudhary/fak/internal/growthgate"
	"github.com/anthony-chaudhary/fak/internal/leaseref"
	"github.com/anthony-chaudhary/fak/internal/loopmgr"
	"github.com/anthony-chaudhary/fak/internal/pathutil"
	"github.com/anthony-chaudhary/fak/internal/procguard"
	"github.com/anthony-chaudhary/fak/internal/windowgate"
	"github.com/anthony-chaudhary/fak/internal/witness"
)

func cmdGarden(argv []string) { os.Exit(runGarden(os.Stdout, os.Stderr, argv)) }

func runGarden(stdout, stderr io.Writer, argv []string) int {
	// `fak garden tick` is the LIVE, ACTING pass (#1386): it remediates the
	// surfaced stale-work conditions on a recurring cadence rather than only
	// reporting. The default `fak garden` (no subcommand) stays the read-only fold.
	if len(argv) > 0 && argv[0] == "tick" {
		return runGardenTick(stdout, stderr, argv[1:])
	}
	// `fak garden watchdog` is the Go-native scheduled stale-work janitor. It owns
	// the hard outer deadline around `garden tick`, emits one typed JSON envelope
	// even when the child times out, and reaps the child's whole process tree.
	if len(argv) > 0 && argv[0] == "watchdog" {
		return runGardenWatchdog(stdout, stderr, argv[1:])
	}
	// `fak garden walk` is the ITEM pass (#item-walk): one resource-aware fold over
	// the HUNDREDS of individual garden items (issues today) a member surfaces — it
	// cheaply skips the live/healthy ones and proposes a budget-bounded, worst-first
	// worklist for the rest. Complements `tick` (which acts at the member level).
	if len(argv) > 0 && argv[0] == "walk" {
		return runGardenWalk(stdout, stderr, argv[1:])
	}
	// `fak garden dispatch` is the BRIDGE (#1791) between garden walk's propose-only
	// worklist and the guarded worker-spawn machinery in `fak dispatch tick`: it loads
	// the same candidate set garden walk would propose and hands each one to
	// evaluateDispatchTick (the same admission pipeline `dispatch tick`/`sweep`/`wave`
	// already use — loop governor, seat/weekly-cap preflight, lane-lease/DOS
	// arbitration, issue worker prompt/picker), never a second implementation of any
	// of them. --dry-run (default) spawns nothing; --apply attempts only admitted
	// candidates. See garden_dispatch.go.
	if len(argv) > 0 && argv[0] == "dispatch" {
		return runGardenDispatch(stdout, stderr, argv[1:])
	}
	fs := flag.NewFlagSet("garden", flag.ContinueOnError)
	fs.SetOutput(stderr)
	workspace := fs.String("workspace", "", "workspace root (default: repo root)")
	asJSON := fs.Bool("json", false, "emit machine-readable JSON")
	check := fs.Bool("check", false, "CI gate: exit non-zero if a gating member regressed or failed to run")
	deep := fs.Bool("deep", false, "also run the fleet loop-audit member (slower; non-gating advisory)")
	timeout := fs.Int("timeout", 240, "per-member timeout seconds")
	if !parseFlags(fs, argv) {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintf(stderr, "fak garden: unexpected argument %q\n", fs.Arg(0))
		return 2
	}

	root := *workspace
	if root == "" {
		root = repoRoot()
	} else if abs, err := filepath.Abs(root); err == nil {
		root = abs
	}
	commit := gardenbundle.HeadCommit(root)

	var payload gardenbundle.Payload
	if gardenbundle.GardenOff() {
		payload = gardenbundle.SkippedPayload(root, commit)
	} else {
		results := gardenbundle.Collect(root, "", time.Duration(*timeout)*time.Second, *deep)
		payload = gardenbundle.Fold(results, root, commit)
	}

	if *check {
		code, message := gardenbundle.CheckGate(payload)
		if *asJSON {
			gated := payload.WithGate(code, message)
			emitGardenJSON(stdout, gated)
		} else {
			fmt.Fprintln(stdout, message)
		}
		return code
	}

	if *asJSON {
		emitGardenJSON(stdout, payload)
	} else {
		fmt.Fprintln(stdout, gardenbundle.Render(payload))
	}
	if payload.OK {
		return 0
	}
	return 1
}

func emitGardenJSON(w io.Writer, p gardenbundle.Payload) {
	_ = writeIndentedJSONNoEscape(w, p)
}

// gardenTickLoopID is the durable loop id under which the acting tick records its
// run in the loop ledger and registers itself in the loop registry, so the tick
// is visible in `fak loop health` and re-arms at boot (the #1281 durable-loop
// registration precedent: a schedule definition that survives a restart).
var (
	gardenReclaimInspect = inspectGardenReclaim
	gardenCollectBounded = gardenbundle.CollectBounded
)

const gardenTickLoopID = "garden-stale-work-tick"

// gardenTickIntervalSeconds is the registered cadence: hourly. The stale-work
// conditions (orphaned runs 4 days old, expired leases) move on the order of
// hours, so an hourly act-pass keeps them bounded without churn.
const gardenTickIntervalSeconds = 3600

// runGardenTick runs ONE act-pass over the garden's stale-work members and takes
// the documented, idempotent remediation for each surfaced condition:
//
//   - stale_leases (expired refs/fak/locks/*) -> reap the expired records
//     in-process (the `fak leaseref reap` remediation), bounding a crashed
//     holder's lapsed lease. Idempotent: reaping an already-gone lease is a no-op.
//   - orphaned_runs -> surface the recovery worklist as a WITNESSED tick event in
//     the loop ledger so the operator can re-dispatch/re-verify (re-dispatch stays
//     gated, never automatic).
//   - release_staleness -> advisory only (acting needs the release path, #1367).
//
// --dry-run runs the SAME measurement but performs no side effect — preserving the
// read-only behavior behind the flag. --register installs the durable loop unit and
// returns. Every run (act or dry-run) appends a witnessed run-end to the loop ledger,
// so `fak loop health` shows the tick living.
func runGardenTick(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("garden tick", flag.ContinueOnError)
	fs.SetOutput(stderr)
	workspace := fs.String("workspace", "", "workspace root (default: repo root)")
	asJSON := fs.Bool("json", false, "emit machine-readable JSON")
	dryRun := fs.Bool("dry-run", false, "report what the tick WOULD act on, but perform no side effect (preserves report-only behavior)")
	growthApplyFlag := fs.Bool("growth-apply", false, "growthgate collect: actually delete the reapable set (default: ledger-only soak, delete nothing; also honored: FAK_GARDEN_GROWTH_COLLECT=apply)")
	register := fs.Bool("register", false, "register the durable garden-tick loop unit in the loop registry and return")
	timeout := fs.Int("timeout", 240, "per-member timeout seconds")
	budget := fs.Int("budget", gardenTickBudgetSeconds, "wall-clock budget seconds for the WHOLE tick; phases past the budget are deferred to the next tick via the resume checkpoint (0 = unbounded)")
	cursor := fs.String("cursor", "", "resume-checkpoint path for the bounded collection/action cycle (default: <workspace>/.dos/garden/tick-cursor.json)")
	ledger := fs.String("ledger", "", "loop JSONL ledger path (default: the loop ledger)")
	registry := fs.String("registry", "", "loop registry JSON path (default: the loop registry)")
	dir := fs.String("dir", "", "lease store repo dir (default: git discovery from cwd)")
	if !parseFlags(fs, argv) {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintf(stderr, "fak garden tick: unexpected argument %q\n", fs.Arg(0))
		return 2
	}
	*dir = pathutil.ExpandTilde(*dir)

	root := *workspace
	if root == "" {
		root = repoRoot()
	} else if abs, err := filepath.Abs(root); err == nil {
		root = abs
	}
	ledgerPath := firstNonEmpty(*ledger, defaultLoopLedger())
	registryPath := firstNonEmpty(*registry, defaultLoopRegistry())

	if *register {
		if err := registerGardenTickLoop(registryPath); err != nil {
			fmt.Fprintf(stderr, "fak garden tick: register: %v\n", err)
			return 1
		}
		fmt.Fprintf(stdout, "registered durable loop %q (interval %ds) in %s\n", gardenTickLoopID, gardenTickIntervalSeconds, registryPath)
		return 0
	}

	if gardenbundle.GardenOff() {
		fmt.Fprintln(stdout, "garden tick skipped: FAK_GARDEN is off")
		return 0
	}

	tickStart := time.Now()
	tickBudget := time.Duration(*budget) * time.Second
	cursorPath := firstNonEmpty(*cursor, defaultGardenTickCursor(root))
	resume, cerr := gardenbudget.LoadCursor(cursorPath)
	if cerr != nil {
		fmt.Fprintf(stderr, "fak garden tick: read resume checkpoint %s: %v (restarting the cycle)\n", cursorPath, cerr)
	}
	// A report-only invocation must not consume or mutate the acting checkpoint.
	if *dryRun {
		resume = gardenbudget.Cursor{}
	}
	resume = gardenbudget.Stamp(resume, gardenTickStageCollect, tickStart)
	state := decodeGardenTickState(resume.Payload)
	if !validGardenTickStage(resume.Stage) {
		resume.Stage, resume.Next = gardenTickStageCollect, ""
		state = gardenTickState{}
	}

	save := func(cur gardenbudget.Cursor) error {
		cur.Payload = encodeGardenTickState(state)
		cur.UpdatedUnix = time.Now().Unix()
		resume = cur
		if *dryRun {
			return nil
		}
		return gardenbudget.SaveCursor(cursorPath, cur)
	}
	finisher := gardenTickFinisher{
		stdout: stdout, stderr: stderr, asJSON: *asJSON, budget: *budget,
		started: tickStart, root: root, cursorPath: cursorPath, state: &state, resume: &resume,
	}
	finish := finisher.finish

	var collection gardenbundle.CollectProgress
	var action gardenbudget.Result
	plan := gardenbundle.PlanTick(state.Results, *dryRun)

	// Stage 1: member collection. It is a durable prefix, not one unbounded
	// prelude: every returned member advances the checkpoint and each member gets
	// at most the global time remaining.
	if resume.Stage == gardenTickStageCollect {
		results, progress := gardenCollectBounded(root, gardenbundle.CollectOptions{
			PerMemberTimeout: time.Duration(*timeout) * time.Second,
			Budget:           tickBudget,
			Start:            tickStart,
			Next:             string(resume.Next),
			Prior:            state.Results,
			Checkpoint: func(next string, results []gardenbundle.MemberResult) error {
				state.Results = results
				resume.Stage = gardenTickStageCollect
				resume.Next = gardenbudget.Phase(next)
				return save(resume)
			},
		})
		state.Results, collection = results, progress
		plan = gardenbundle.PlanTick(state.Results, *dryRun)
		if progress.CheckpointError != "" {
			return finish("error", "GARDEN_TICK_CHECKPOINT_FAILED", false, plan, collection, action, 1)
		}
		if !progress.Complete {
			return finish("partial", "GARDEN_TICK_BUDGET_EXHAUSTED", false, plan, collection, action, 0)
		}
		resume.Stage, resume.Next = gardenTickStageReclaim, gardenbudget.Phase(gardenTickReclaimKey)
		if err := save(resume); err != nil {
			fmt.Fprintf(stderr, "fak garden tick: checkpoint collection: %v\n", err)
			return finish("error", "GARDEN_TICK_CHECKPOINT_FAILED", false, plan, collection, action, 1)
		}
	}

	// Stage 2: the in-process commit-lifecycle inspection. It has its own
	// checkpoint so a hard outer kill retries it rather than replaying collection.
	if resume.Stage == gardenTickStageReclaim {
		if gardenbudget.Remaining(tickBudget, tickStart, time.Now) == 0 {
			return finish("partial", "GARDEN_TICK_BUDGET_EXHAUSTED", false, plan, collection, action, 0)
		}
		state.Results = append(state.Results, gardenReclaimInspect(stderr, root))
		plan = gardenbundle.PlanTick(state.Results, *dryRun)
		resume.Stage, resume.Next = gardenTickStageAct, gardenActionPhases(*dryRun)[0]
		if err := save(resume); err != nil {
			fmt.Fprintf(stderr, "fak garden tick: checkpoint reclaim inspection: %v\n", err)
			return finish("error", "GARDEN_TICK_CHECKPOINT_FAILED", false, plan, collection, action, 1)
		}
	}

	plan = gardenbundle.PlanTick(state.Results, *dryRun)
	if !*dryRun && resume.Stage == gardenTickStageAct && resume.Next != "" {
		var checkpointErr error
		state.Counts, action = performGardenTickBounded(stdout, stderr, plan, *dir, root,
			false, growthApplyEnabled(*growthApplyFlag), state.Counts, resume,
			gardenbudget.Options{Budget: tickBudget, Start: tickStart},
			func(cur gardenbudget.Cursor, counts gardenTickCounts) error {
				state.Counts = counts
				resume = cur
				checkpointErr = save(cur)
				return checkpointErr
			})
		if checkpointErr != nil || action.CheckpointError != "" {
			fmt.Fprintf(stderr, "fak garden tick: checkpoint action phase: %v\n", checkpointErr)
			return finish("error", "GARDEN_TICK_CHECKPOINT_FAILED", false, plan, collection, action, 1)
		}
		if !action.Complete {
			return finish("partial", "GARDEN_TICK_BUDGET_EXHAUSTED", false, plan, collection, action, 0)
		}
	}

	// A complete action suffix checkpoints Next="" before this reset. If a
	// process dies in the tiny gap, the next invocation sees act+empty and comes
	// straight here instead of replaying mutations.
	completedCounts := state.Counts
	if !*dryRun {
		resume.Stage, resume.Next = gardenTickStageCollect, ""
		state = gardenTickState{}
		if err := save(resume); err != nil {
			fmt.Fprintf(stderr, "fak garden tick: reset completed cycle: %v\n", err)
			return finish("error", "GARDEN_TICK_CHECKPOINT_FAILED", false, plan, collection, action, 1)
		}
	}
	// The durable reset intentionally clears the next cycle's payload; keep the
	// just-completed counts in memory for this invocation's renderer and witness.
	state.Counts = completedCounts
	witnessGardenTick(ledgerPath, plan, state.Counts.Reaped, state.Counts.Sessions,
		state.Counts.Surfaced, state.Counts.LockFiles, state.Counts.Collected,
		state.Counts.Intents, state.Counts.Folded)
	return finish("complete", "GARDEN_TICK_COMPLETE", true, plan, collection, action, 0)
}

// gardenTickBudgetSeconds bounds ONE tick's wall clock. It is deliberately far below
// the hourly cadence (gardenTickIntervalSeconds): a maintenance pass that outlives its
// own interval overlaps the next scheduled tick and monopolizes the process tree — the
// exact failure #6493 reports, where an hourly janitor sat blocked for 17 minutes.
// Nothing is dropped when the budget runs out: the phases that did not fit are deferred
// to the next tick through the resume checkpoint, so a large first drain converges
// across ticks instead of inside one. 0 (via --budget 0) restores unbounded ticks.
const gardenTickBudgetSeconds = 45

const (
	gardenTickStageCollect = "collect"
	gardenTickStageReclaim = "reclaim"
	gardenTickStageAct     = "act"
	gardenTickReclaimKey   = "commit_lifecycle"
)

// The garden tick's resumable phases, in rotation order. Each name is one of the
// tick's independent, idempotent, best-effort sweeps: splitting them this way is what
// lets a budgeted tick stop between sweeps and resume at the next one (#6493).
const (
	gardenPhaseLeases   gardenbudget.Phase = "leases"
	gardenPhaseLocks    gardenbudget.Phase = "lock-files"
	gardenPhaseIntents  gardenbudget.Phase = "intents"
	gardenPhaseGrowth   gardenbudget.Phase = "growth-logs"
	gardenPhaseSentinel gardenbudget.Phase = "sentinel-fold"
)

// defaultGardenTickCursor is where the budgeted rotation checkpoints its resume point:
// a gitignored per-workspace state file next to the rest of the .dos ephemera.
func defaultGardenTickCursor(root string) string {
	if root == "" {
		return ""
	}
	return filepath.Join(root, ".dos", "garden", "tick-cursor.json")
}

// writeGardenTickBudgetLine prints which phases this tick could not afford and
// where the next scheduled pass resumes.
func writeGardenTickBudgetLine(w io.Writer, spend gardenbudget.Result, cursorPath string) {
	if !spend.Exhausted {
		return
	}
	fmt.Fprintf(w, "  -> budget spent after %d phase(s) in %dms; deferred %v to the next tick (resume at %q, checkpoint %s)\n",
		len(spend.Ran), spend.Millis, spend.Deferred, spend.Next.Next, cursorPath)
}

type gardenTickState struct {
	Results []gardenbundle.MemberResult `json:"results,omitempty"`
	Counts  gardenTickCounts            `json:"counts,omitempty"`
}

func validGardenTickStage(stage string) bool {
	return stage == gardenTickStageCollect || stage == gardenTickStageReclaim || stage == gardenTickStageAct
}

func decodeGardenTickState(raw json.RawMessage) gardenTickState {
	var state gardenTickState
	if len(raw) > 0 {
		_ = json.Unmarshal(raw, &state)
	}
	return state
}

func encodeGardenTickState(state gardenTickState) json.RawMessage {
	b, _ := json.Marshal(state)
	return b
}

func gardenActionPhases(dryRun bool) []gardenbudget.Phase {
	if dryRun {
		return []gardenbudget.Phase{gardenPhaseLeases}
	}
	return []gardenbudget.Phase{
		gardenPhaseLeases,
		gardenPhaseLocks,
		gardenPhaseIntents,
		gardenPhaseGrowth,
		gardenPhaseSentinel,
	}
}

// performGardenTick executes the side effects the plan calls for. Under dry-run no
// decision has Perform=true, so this is a pure report. It returns the count of
// reaped leases / sessions, the count of orphan-run worklists surfaced, the count
// of orphan .lock files swept, the count of oversized disposable logs the growthgate
// collect reaped (0 in the ledger-only default), the count of lapsed intent leases
// reaped (the reap-parity sweep, #5345), and the count of decision-note lines folded
// off the empty-tree sentinel note (census row 12, #5361).
func performGardenTick(stdout, stderr io.Writer, plan gardenbundle.TickPlan, dir, root string, dryRun, growthApply bool) (reaped, sessions, surfaced, lockFiles, collected, intents, folded int) {
	c, _ := performGardenTickBounded(stdout, stderr, plan, dir, root, dryRun, growthApply,
		gardenTickCounts{}, gardenbudget.Cursor{}, gardenbudget.Options{}, nil)
	return c.Reaped, c.Sessions, c.Surfaced, c.LockFiles, c.Collected, c.Intents, c.Folded
}

// gardenTickCounts is what one tick actually did, one field per reaper. It exists so
// the budgeted rotation can hand each phase a shared accumulator instead of threading
// seven return values through every closure.
type gardenTickCounts struct {
	Reaped    int `json:"reaped_leases,omitempty"`
	Sessions  int `json:"reaped_sessions,omitempty"`
	Surfaced  int `json:"surfaced_runs,omitempty"`
	LockFiles int `json:"reaped_lock_files,omitempty"`
	Collected int `json:"reaped_growth_logs,omitempty"`
	Intents   int `json:"reaped_intents,omitempty"`
	Folded    int `json:"folded_sentinel_lines,omitempty"`
}

func (c gardenTickCounts) acted() bool {
	return c.Reaped+c.Sessions+c.Surfaced+c.LockFiles+c.Collected+c.Intents+c.Folded > 0
}

// performGardenTickBounded executes the durable action suffix. initial carries
// counts from phases completed by earlier ticks; checkpoint persists both the
// next phase and the updated cumulative counts after every returned phase.
func performGardenTickBounded(stdout, stderr io.Writer, plan gardenbundle.TickPlan, dir, root string, dryRun, growthApply bool, initial gardenTickCounts, resume gardenbudget.Cursor, opt gardenbudget.Options, checkpoint func(gardenbudget.Cursor, gardenTickCounts) error) (gardenTickCounts, gardenbudget.Result) {
	c := initial
	work := map[gardenbudget.Phase]func() error{
		gardenPhaseLeases:   func() error { return gardenPhaseReapLeases(stderr, plan, dir, &c) },
		gardenPhaseLocks:    func() error { return gardenPhaseReapLockFiles(stderr, dir, &c) },
		gardenPhaseIntents:  func() error { return gardenPhaseReapIntents(stderr, dir, &c) },
		gardenPhaseGrowth:   func() error { return gardenPhaseCollectGrowth(stderr, root, growthApply, &c) },
		gardenPhaseSentinel: func() error { return gardenPhaseFoldSentinel(stderr, root, &c) },
	}
	if checkpoint != nil {
		opt.Checkpoint = func(cur gardenbudget.Cursor) error {
			return checkpoint(cur, c)
		}
	}
	res := gardenbudget.Execute(gardenActionPhases(dryRun), resume, opt, func(p gardenbudget.Phase) error { return work[p]() })
	_ = stdout
	return c, res
}

// gardenPhaseReapLeases walks the plan's decisions: the expired lease/session reap and
// the orphan-worklist surfacing. Under dry-run no decision has Perform=true, so this
// is a pure report.
func gardenPhaseReapLeases(stderr io.Writer, plan gardenbundle.TickPlan, dir string, c *gardenTickCounts) error {
	for _, d := range plan.Decisions {
		if !d.Perform {
			continue
		}
		switch d.Act {
		case gardenbundle.ActReap:
			// Idempotent reap: leaseref.Store.Reap deletes only ALREADY-expired
			// records; a live lease is never touched, and a second pass over a
			// gone lease is a no-op. Mirrors `fak leaseref reap` exactly.
			store := leaseref.NewInDir(dir)
			now := time.Now()
			ctx := context.Background()
			leases, lerr := store.Reap(ctx, now)
			if lerr != nil {
				fmt.Fprintf(stderr, "fak garden tick: reap leases: %v\n", lerr)
			} else {
				c.Reaped += len(leases)
			}
			sess, serr := store.ReapSessions(ctx, now)
			if serr != nil {
				fmt.Fprintf(stderr, "fak garden tick: reap sessions: %v\n", serr)
			} else {
				c.Sessions += len(sess)
			}
		case gardenbundle.ActSurface:
			// The worklist is already in the member's surfaced detail; surfacing
			// it means recording the witnessed condition (the ledger event, below)
			// — re-dispatch stays the gated operator action, never automatic here.
			c.Surfaced++
		}
	}
	return nil
}

// gardenPhaseReapLockFiles is the orphan .lock sweep (#5348): run ONCE PER TICK,
// unconditionally — independent of whether any lease RECORD is expired. A ghost
// <git-common-dir>/refs/fak/locks/*.lock (a holder killed mid-CAS) can outlive its
// lease even when no lease record is stale, so gating this on the stale_leases ActReap
// decision would leave it uncollected. ReapLockFiles is fail-safe by construction
// (namespace-confined, age-bounded at the lease TTL, future-mtime kept). This phase is
// not in the dry-run rotation, mirroring the reap decisions (which never Perform under
// dry-run).
func gardenPhaseReapLockFiles(stderr io.Writer, dir string, c *gardenTickCounts) error {
	locks, _, lerr := leaseref.NewInDir(dir).ReapLockFiles(context.Background(), time.Now(), 0)
	if lerr != nil {
		fmt.Fprintf(stderr, "fak garden tick: reap orphan locks: %v\n", lerr)
		return lerr
	}
	c.LockFiles += len(locks)
	return nil
}

// gardenPhaseReapIntents is the intent-lease reap (#5345 reap parity): run ONCE PER
// TICK, unconditionally — closing the same reaper asymmetry that bit sessions (#5344).
// A lapsed refs/fak/locks/intent-<key> accretes independently of whether any lease
// RECORD is expired, so gating this on the ActReap decision (as the lease/session reaps
// are) would leave the intent namespace uncollected on the automatic loop — the CLI's
// `fak leaseref reap` already sweeps all three kinds. Same best-effort, idempotent
// contract as the ReapLockFiles sweep.
func gardenPhaseReapIntents(stderr io.Writer, dir string, c *gardenTickCounts) error {
	ints, ierr := leaseref.NewInDir(dir).ReapIntents(context.Background(), time.Now())
	if ierr != nil {
		fmt.Fprintf(stderr, "fak garden tick: reap intents: %v\n", ierr)
		return ierr
	}
	c.Intents += len(ints)
	return nil
}

// gardenPhaseCollectGrowth is the growthgate collect (#5349) over the repo + Fleet
// trees, the acting half of the ActGrowthReap edge. DELETE-SAFE by default: it appends
// the would-reap set to the reap ledger every tick (the soak evidence) but removes
// NOTHING unless the apply opt-in is set. Not in the dry-run rotation, mirroring the
// lock sweep (which never deletes under dry-run).
func gardenPhaseCollectGrowth(stderr io.Writer, root string, growthApply bool, c *gardenTickCounts) error {
	c.Collected += collectGrowthLogs(stderr, growthCensusRoots(root), growthApply, growthReapLedgerPath())
	return nil
}

// gardenPhaseFoldSentinel is the decisions-note fold (#5361, census row 12), bounding
// the ONE unbounded producer on refs/notes/fak/decisions — the empty-tree sentinel note
// that every pre-commit refusal (OFF_TRUNK / PATHSPEC_RACE / LEASE_HELD) appends to
// forever. CompactSentinelNote keeps the most-recent sentinelNoteKeepLines lines
// (recency = forensic value) and force-overwrites ONLY that one sentinel note on the
// side ref; commit-anchored notes are bounded per-object evidence and are NEVER
// touched, and it never touches main / HEAD / refs/heads and never pushes. The recorder
// binds to `root` (the gardened repo) so the fold hits its decisions ref, not the
// process cwd's. Best-effort + idempotent, same contract as the sweeps above: a fold
// error is logged and swallowed so it never fails the tick. Not in the dry-run rotation.
func gardenPhaseFoldSentinel(stderr io.Writer, root string, c *gardenTickCounts) error {
	fold, ferr := witness.NewRecorderForDir(root).CompactSentinelNote(context.Background(), sentinelKeepLines())
	if ferr != nil {
		fmt.Fprintf(stderr, "fak garden tick: fold decisions sentinel note: %v\n", ferr)
		return ferr
	}
	c.Folded += fold
	return nil
}

// witnessGardenTick records the tick's run in the loop ledger as the claim+verdict
// PAIR the rest of the fleet emits (`fak loop drive`, `fak dispatch progress`): an
// EventEnd carrying the tick's own claim, then an EventWitness carrying the verdict
// the folded member envelopes prove — so `fak loop health` shows the loop alive AND
// witnessed. Putting the verdict on the END channel instead (as this did until
// #5341) is counted by loopmgr as an UNWITNESSED run, because only EventWitness
// increments Witnessed: health then reported this loop 0-of-N witnessed with
// witness_collapse=true while its own last_state read "witnessed_done" — the exact
// opposite of what this function exists to do. A ledger append failure is non-fatal:
// the remediation already happened; losing a ledger line shouldn't fail the tick.
func witnessGardenTick(ledgerPath string, plan gardenbundle.TickPlan, reaped, sessions, surfaced, lockFiles, collected, intents, folded int) {
	if ledgerPath == "" {
		return
	}
	summary := fmt.Sprintf("garden tick clean: %d member(s), nothing to act on", len(plan.Decisions))
	if plan.DryRun {
		summary = fmt.Sprintf("garden tick dry-run: would reap %d, surface %d (no side effect)", plan.ToReap, plan.ToSurface)
	} else if plan.Acted() || lockFiles > 0 || collected > 0 || intents > 0 || folded > 0 {
		summary = fmt.Sprintf("garden tick acted: reaped %d lease(s) + %d session(s) + %d intent(s) + %d orphan lock(s) + %d growth log(s) + %d folded sentinel line(s), surfaced %d orphan worklist(s)", reaped, sessions, intents, lockFiles, collected, folded, surfaced)
	}
	metrics := map[string]int64{
		"reaped_leases":         int64(reaped),
		"reaped_sessions":       int64(sessions),
		"reaped_lock_files":     int64(lockFiles),
		"reaped_intents":        int64(intents),
		"reaped_growth_logs":    int64(collected),
		"folded_sentinel_lines": int64(folded),
		"surfaced_runs":         int64(surfaced),
		"advisory":              int64(plan.Advisory),
	}
	runID := firstNonEmpty(os.Getenv("FAK_LOOP_RUN_ID"), fmt.Sprintf("garden-tick-%d", time.Now().UnixNano()))
	_, _ = loopmgr.Append(ledgerPath, loopmgr.Event{
		LoopID:  gardenTickLoopID,
		RunID:   runID,
		Kind:    loopmgr.EventEnd,
		Status:  loopmgr.StatusClaimedDone,
		Source:  "fak garden tick",
		Summary: summary,
		Metrics: metrics,
	})

	// The verdict channel. The folded member envelopes ARE the witness: each one is a
	// machine-checked payload, not the tick's own narration. A member that ERRORED
	// produced no usable payload, so the tick cannot prove it swept everything — that
	// is witness_unavailable, not a done verdict. A red/action member is the opposite:
	// a MEASURED finding the tick surfaced honestly, so it still witnesses done.
	status, reason := loopmgr.StatusWitnessedDone, "GARDEN_TICK_FOLDED"
	witnessSummary := summary
	if unmeasured := gardenTickUnmeasured(plan); unmeasured > 0 {
		status, reason = loopmgr.StatusWitnessUnavailable, "GARDEN_TICK_MEMBER_ERRORED"
		witnessSummary = fmt.Sprintf("%s; %d member(s) errored — sweep completeness unproven", summary, unmeasured)
	}
	_, _ = loopmgr.Append(ledgerPath, loopmgr.Event{
		LoopID:  gardenTickLoopID,
		RunID:   runID,
		Kind:    loopmgr.EventWitness,
		Status:  status,
		Reason:  reason,
		Source:  "fak garden tick",
		Summary: witnessSummary,
		Metrics: metrics,
		EvidenceRefs: []loopmgr.EvidenceRef{{
			Kind:    "control_pane",
			Ref:     "fak garden tick",
			Summary: fmt.Sprintf("%d member envelope(s) folded", len(plan.Decisions)),
		}},
	})
}

// gardenTickUnmeasured counts the members whose envelope produced no usable payload
// ("errored"). They are the reason a tick cannot claim a done verdict: the sweep may
// have missed exactly the condition an unreadable member would have surfaced.
func gardenTickUnmeasured(plan gardenbundle.TickPlan) int {
	n := 0
	for _, d := range plan.Decisions {
		if d.State == "errored" {
			n++
		}
	}
	return n
}

// registerGardenTickLoop installs the durable garden-tick loop in the loop registry
// (the #1281 precedent): an armed Schedule that survives a restart and re-arms at
// boot, so the act-pass recurs. Idempotent: re-registering keeps the original
// CreatedUnixNano and re-arms the same job id.
func registerGardenTickLoop(registryPath string) error {
	return registerGardenLoop(registryPath, gardenTickLoopID, gardenTickIntervalSeconds, 300)
}

// registerGardenLoop is the shared registrar behind the durable garden loops: load the
// registry, Put an ARMED MissedSkip schedule, save. Idempotent, because Put keeps the
// original CreatedUnixNano and re-arms the same job id. jitterSeconds is a parameter rather
// than a constant because the loops deliberately spread differently — the hourly tick jitters
// by 5 minutes, the 6-hourly walk by 10 — so each caller passes the value it already used.
func registerGardenLoop(registryPath, jobID string, intervalSeconds, jitterSeconds int64) error {
	reg, err := loopmgr.LoadRegistry(registryPath)
	if err != nil {
		return err
	}
	if err := reg.Put(loopmgr.Job{
		Schedule: loopmgr.Schedule{
			JobID:           jobID,
			IntervalSeconds: intervalSeconds,
			MissedRun:       loopmgr.MissedSkip,
			JitterSeconds:   jitterSeconds,
		},
		State: loopmgr.JobArmed,
	}, time.Now()); err != nil {
		return err
	}
	return loopmgr.SaveRegistry(registryPath, reg)
}

// renderGardenTick prints the act-pass as an aligned snapshot: one row per member
// decision, then the summary of what the tick actually did.
func renderGardenTick(w io.Writer, plan gardenbundle.TickPlan, reaped, sessions, surfaced, lockFiles, collected, intents, folded int) {
	mode := "act"
	if plan.DryRun {
		mode = "dry-run"
	}
	fmt.Fprintf(w, "garden tick (%s)\n\n", mode)
	for _, d := range plan.Decisions {
		mark := "."
		if d.Perform {
			mark = "*"
		} else if d.State == "ok" {
			mark = "+"
		}
		fmt.Fprintf(w, "  %s %-18s %-8s %-10s %s\n", mark, d.Label, d.State, d.Act, d.Reason)
	}
	fmt.Fprintln(w, "")
	if plan.DryRun {
		fmt.Fprintf(w, "  -> dry-run: would reap %d expired lease(s), surface %d orphan worklist(s); nothing performed\n", plan.ToReap, plan.ToSurface)
		return
	}
	if plan.Acted() || lockFiles > 0 || collected > 0 || intents > 0 || folded > 0 {
		fmt.Fprintf(w, "  -> acted: reaped %d lease(s) + %d session(s) + %d intent(s) + %d orphan lock(s) + %d growth log(s) + %d folded sentinel line(s), surfaced %d orphan worklist(s)\n", reaped, sessions, intents, lockFiles, collected, folded, surfaced)
	} else {
		fmt.Fprintln(w, "  -> garden tick clean: nothing to act on")
	}
}

// growthCollectApplyEnv is the apply opt-in env for the schedule-driven growthgate
// collect. Absent (or any value other than "apply") keeps the collect in its
// soaked default: ledger-only, delete nothing. Set it to "apply" (or pass
// --growth-apply) to actually delete the reapable set — the one-line follow-on
// after the reap ledger has shown a correct set over a soak window (#5079
// grace-prune precedent: delete-on-schedule stays opt-in until soaked).
const growthCollectApplyEnv = "FAK_GARDEN_GROWTH_COLLECT"

// sentinelNoteKeepLines bounds the empty-tree sentinel decisions note (the one
// unbounded producer on refs/notes/fak/decisions — every pre-commit refusal piles
// line-by-line onto it fleet-lifetime). The tick folds it to the most-recent N lines
// (recency = forensic value); commit-anchored notes are per-object bounded evidence
// and are never folded. 2000 is a deliberately generous forensic tail — a deep window
// of recent refusals while still bounding unbounded growth. The keep-N policy lives
// here at the wiring, not in the witness library (which takes maxLines as a parameter).
const sentinelNoteKeepLines = 2000

// sentinelKeepEnv lets an operator (or a test) tighten or widen the sentinel-note
// keep-N bound without a rebuild. A positive integer overrides sentinelNoteKeepLines;
// an absent / non-positive / unparseable value falls back to the default rather than
// the library's fail-safe no-op, so a fat-fingered env can never silently DISABLE the
// collector (only re-tune it).
const sentinelKeepEnv = "FAK_GARDEN_SENTINEL_KEEP"

// sentinelKeepLines resolves the keep-last-N bound for the tick's sentinel-note fold:
// FAK_GARDEN_SENTINEL_KEEP when it parses to a positive int, else sentinelNoteKeepLines.
func sentinelKeepLines() int {
	if v := strings.TrimSpace(os.Getenv(sentinelKeepEnv)); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			return n
		}
	}
	return sentinelNoteKeepLines
}

// growthApplyEnabled reports whether the tick's growthgate collect may os.Remove
// files this run. Default-off: the first landing appends the would-reap set to the
// reap ledger every tick as soak evidence but deletes nothing until an operator
// sets FAK_GARDEN_GROWTH_COLLECT=apply or passes --growth-apply.
func growthApplyEnabled(flagApply bool) bool {
	if flagApply {
		return true
	}
	return strings.EqualFold(strings.TrimSpace(os.Getenv(growthCollectApplyEnv)), "apply")
}

// fleetTreeRoot resolves the Fleet tree the growth collect censuses beyond the
// repo, mirroring defaultStallLogPath's idiom: an env override, else
// %LOCALAPPDATA%/Fleet on a Windows fleet host. It returns "" when neither is set
// (an empty LOCALAPPDATA), so the caller skips the extra root cleanly rather than
// falling back to a home dir that holds no fleet logs.
func fleetTreeRoot() string {
	if d := os.Getenv("FAK_FLEET_DIR"); d != "" {
		return d
	}
	if la := os.Getenv("LOCALAPPDATA"); la != "" {
		return filepath.Join(la, "Fleet")
	}
	return ""
}

// growthCensusRoots is the census scope for the tick's growth collect: the repo
// root always, plus the Fleet tree when it resolves to a real directory. The
// growthgate member alone censuses only the repo root; adding the Fleet tree
// exposes the oversized single disposable logs that live under it.
func growthCensusRoots(repoRoot string) []string {
	roots := []string{repoRoot}
	if fr := fleetTreeRoot(); fr != "" {
		if info, err := os.Stat(fr); err == nil && info.IsDir() {
			roots = append(roots, fr)
		}
	}
	return roots
}

// growthReapLedgerPath resolves where the tick's growth collect appends its
// would-reap / reaped ledger (the soak evidence). Env override first, else
// %LOCALAPPDATA%/Fleet on a fleet host, else ~/.fak — mirroring defaultStallLogPath
// so the soak trail lands beside the stall log.
func growthReapLedgerPath() string {
	if d := os.Getenv("FAK_GARDEN_GROWTH_LEDGER"); d != "" {
		return d
	}
	if la := os.Getenv("LOCALAPPDATA"); la != "" {
		return filepath.Join(la, "Fleet", "growthgate-reap.jsonl")
	}
	home, _ := os.UserHomeDir()
	return filepath.Join(home, ".fak", "growthgate-reap.jsonl")
}

// inspectGardenReclaim reads the existing RECLAIM worklist and surfaces only its
// actionable head. It deliberately does not land, delete, or update checkpoint refs.
func inspectGardenReclaim(stderr io.Writer, root string) gardenbundle.MemberResult {
	rows, err := commitLifecycleQueue(context.Background(), root)
	if err != nil {
		fmt.Fprintf(stderr, "fak garden tick: commit lifecycle queue: %v\n", err)
		return gardenbundle.MemberResult{Key: "commit_lifecycle", Label: "Commit lifecycle queue", State: "errored", Detail: err.Error()}
	}
	counts := make(map[string]int)
	var actionable []commitlifecycle.Row
	for _, row := range rows {
		counts[string(row.State)]++
		if row.State != commitlifecycle.Shipped {
			actionable = append(actionable, row)
		}
	}
	if len(actionable) == 0 {
		return gardenbundle.MemberResult{Key: "commit_lifecycle", Label: "Commit lifecycle queue", State: "ok", Counts: counts}
	}
	head := actionable[0]
	detail := fmt.Sprintf("%d non-terminal; head=%s next=%s", len(actionable), head.State, commitLifecycleActionText(head.Action))
	return gardenbundle.MemberResult{Key: "commit_lifecycle", Label: "Commit lifecycle queue", State: "action", Detail: detail, Counts: counts}
}

// collectGrowthLogs is the schedule-driven growthgate collector wired to the
// tick's ActGrowthReap edge (#5349). It censuses the growth-prone files under
// roots, classifies them with the shared budget, and partitions the report with
// growthgate.ReapPlan into the COLD/over-budget/disposable reapable set versus the
// protected set (HOT files, WALs, chained ledgers — never touched). It ALWAYS
// appends the would-reap set to the reap ledger (the soak evidence); it hard-
// deletes a reapable file ONLY when the apply opt-in is set. The default is
// ledger-only, so the first landing records what it WOULD reap over a soak window
// without deleting anything. Returns the count of files actually reaped (0 in the
// ledger-only default).
//
// It writes to the ledger on EVERY tick, not only a reaping one: when the reapable
// set is empty it appends one census-clean heartbeat instead, so the soak window
// distinguishes a clean collect from a dead one (see appendGrowthCensusHeartbeat).
func collectGrowthLogs(stderr io.Writer, roots []string, apply bool, ledgerPath string) (reaped int) {
	now := time.Now()
	var arts []growthgate.Artifact
	for _, root := range roots {
		gathered, gerr := gatherGrowthArtifacts(root, now)
		if gerr != nil {
			fmt.Fprintf(stderr, "fak garden tick: growth census %s: %v\n", root, gerr)
		}
		arts = append(arts, gathered...)
	}
	rep := growthgate.Classify(arts, growthgate.DefaultBudget())
	toReap, protected := growthgate.ReapPlan(rep)

	rows := make([]growthReapDecision, 0, len(toReap))
	for _, f := range toReap {
		row := growthReapDecision{Path: f.Path, Class: string(f.Class), Size: f.Size, Action: "would-reap"}
		if apply {
			if err := os.Remove(f.Path); err != nil {
				row.Action, row.Err = "reap-failed", err.Error()
				fmt.Fprintf(stderr, "fak garden tick: growth reap %s: %v\n", f.Path, err)
			} else {
				row.Action = "reaped"
				reaped++
			}
		}
		rows = append(rows, row)
	}
	if ledgerPath == "" {
		return reaped
	}
	if len(rows) > 0 {
		if err := appendGrowthgateReapLedger(ledgerPath, rows); err != nil {
			fmt.Fprintf(stderr, "fak garden tick: growth ledger %s: %v\n", ledgerPath, err)
		}
		return reaped
	}
	// Nothing reapable — still witness the tick (#5349 step 5). A zero-reapable tick
	// used to write NOTHING, which made the soak window unfalsifiable: an empty
	// ledger read the same whether the collect was clean or dead. The apply flip is
	// gated on that window, so the clean case has to leave a row too.
	if err := appendGrowthCensusHeartbeat(ledgerPath, roots, rep.Scanned, len(protected), string(rep.Verdict)); err != nil {
		fmt.Fprintf(stderr, "fak garden tick: growth census heartbeat %s: %v\n", ledgerPath, err)
	}
	return reaped
}

const (
	// The child gets 35 seconds and the supervisor deadline is 45 seconds,
	// leaving a ten-second process-tree reap/JSON flush margin while the entire
	// default live watchdog remains below the issue's 60-second ceiling.
	gardenWatchdogTimeoutSeconds    = 45
	gardenWatchdogTickBudgetSeconds = 35
)

type gardenWatchdogEphemeralFile struct {
	Path    string  `json:"path"`
	Rel     string  `json:"rel"`
	Label   string  `json:"label"`
	AgeDays float64 `json:"age_days"`
	Size    int64   `json:"size"`
}

type gardenWatchdogStuck struct {
	Session     string `json:"session"`
	Rel         string `json:"rel"`
	Consecutive int    `json:"consecutive"`
	Total       int    `json:"total"`
}

type gardenWatchdogWIP struct {
	Count          int     `json:"count"`
	OldestAgeHours float64 `json:"oldest_age_hours"`
	OldestPath     string  `json:"oldest_path,omitempty"`
	Stale          bool    `json:"stale"`
	ThresholdHours int     `json:"threshold_hours"`
}

type gardenWatchdogSweep struct {
	Files int   `json:"files"`
	Bytes int64 `json:"bytes"`
}

type gardenWatchdogGarden struct {
	Invoked       bool   `json:"invoked"`
	Status        string `json:"status"`
	Reason        string `json:"reason"`
	TimedOut      bool   `json:"timed_out"`
	BudgetSeconds int64  `json:"budget_seconds"`
	ElapsedMillis int64  `json:"elapsed_millis"`
	ExitCode      *int   `json:"exit_code,omitempty"`
	Error         string `json:"error,omitempty"`
	Progress      any    `json:"progress,omitempty"`
}

type gardenWatchdogEnvelope struct {
	Schema         string                `json:"schema"`
	Status         string                `json:"status"`
	Reason         string                `json:"reason"`
	Repo           string                `json:"repo"`
	Live           bool                  `json:"live"`
	MaxAgeDays     int                   `json:"max_age_days"`
	AgeGC          gardenWatchdogAgeGC   `json:"age_gc"`
	Stuck          []gardenWatchdogStuck `json:"stuck"`
	WIP            gardenWatchdogWIP     `json:"wip"`
	HasStale       bool                  `json:"has_stale"`
	Garden         gardenWatchdogGarden  `json:"garden"`
	OverlapRefused bool                  `json:"overlap_refused,omitempty"`
}

type gardenWatchdogAgeGC struct {
	Files      int                           `json:"files"`
	Bytes      int64                         `json:"bytes"`
	Swept      gardenWatchdogSweep           `json:"swept"`
	Candidates []gardenWatchdogEphemeralFile `json:"candidates"`
}

type gardenWatchdogConfig struct {
	Repo           string
	MaxAgeDays     int
	StuckThreshold int
	WIPStaleHours  int
	Live           bool
	FailOnStale    bool
	AsJSON         bool
	Timeout        time.Duration
	TickBudget     time.Duration
	Now            func() time.Time
}

type gardenWatchdogCommandFactory func(root, cursor string, tickBudget time.Duration) *exec.Cmd

var gardenWatchdogCommand gardenWatchdogCommandFactory = defaultGardenWatchdogCommand

func runGardenWatchdog(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("garden watchdog", flag.ContinueOnError)
	fs.SetOutput(stderr)
	repo := fs.String("repo", "", "repo root to garden (default: repo root)")
	maxAgeDays := fs.Int("max-age-days", 7, "GC gitignored ephemera older than this many days")
	stuckThreshold := fs.Int("stuck-threshold", 3, "flag stop-failure sessions at/above this consecutive count")
	wipStaleHours := fs.Int("wip-stale-hours", 24, "flag shared-tree WIP older than this many hours")
	live := fs.Bool("live", false, "delete over-age ephemera and invoke the acting garden tick")
	failOnStale := fs.Bool("fail-on-stale", false, "exit 2 when stale work is found")
	asJSON := fs.Bool("json", false, "emit one machine-readable envelope")
	watchdogTimeout := fs.Int("watchdog-timeout", gardenWatchdogTimeoutSeconds, "hard outer watchdog bound in seconds")
	tickBudget := fs.Int("tick-budget", gardenWatchdogTickBudgetSeconds, "whole-child garden tick budget in seconds")
	if !parseFlags(fs, argv) {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintf(stderr, "fak garden watchdog: unexpected argument %q\n", fs.Arg(0))
		return 2
	}
	root := *repo
	if root == "" {
		root = repoRoot()
	} else if abs, err := filepath.Abs(root); err == nil {
		root = abs
	}
	return runGardenWatchdogConfigured(stdout, stderr, gardenWatchdogConfig{
		Repo: root, MaxAgeDays: *maxAgeDays, StuckThreshold: *stuckThreshold,
		WIPStaleHours: *wipStaleHours, Live: *live, FailOnStale: *failOnStale,
		AsJSON: *asJSON, Timeout: time.Duration(*watchdogTimeout) * time.Second,
		TickBudget: time.Duration(*tickBudget) * time.Second,
	})
}

func runGardenWatchdogConfigured(stdout, stderr io.Writer, cfg gardenWatchdogConfig) int {
	if cfg.Now == nil {
		cfg.Now = time.Now
	}
	if cfg.Timeout <= 0 {
		cfg.Timeout = gardenWatchdogTimeoutSeconds * time.Second
	}
	if cfg.TickBudget <= 0 {
		cfg.TickBudget = gardenWatchdogTickBudgetSeconds * time.Second
	}
	start := time.Now()
	lock, acquired, err := acquireGardenWatchdogLock(cfg.Repo, cfg.Now(), 2*cfg.Timeout)
	if err != nil {
		fmt.Fprintf(stderr, "fak garden watchdog: acquire overlap lock: %v\n", err)
		return 1
	}
	if !acquired {
		env := gardenWatchdogEnvelope{
			Schema: "fak.garden-watchdog.v2", Status: "refused", Reason: "SKIPPED_CONTENDED",
			Repo: cfg.Repo, Live: cfg.Live, MaxAgeDays: cfg.MaxAgeDays,
			Stuck: []gardenWatchdogStuck{}, OverlapRefused: true,
			Garden: gardenWatchdogGarden{Status: "skipped", Reason: "SKIPPED_CONTENDED"},
		}
		return emitGardenWatchdog(stdout, stderr, env, cfg.AsJSON, 0)
	}
	defer lock.release()

	candidates := scanGardenWatchdogEphemera(cfg.Repo, cfg.Now(), cfg.MaxAgeDays)
	stuck := scanGardenWatchdogStuck(cfg.Repo, cfg.StuckThreshold)
	if candidates == nil {
		candidates = []gardenWatchdogEphemeralFile{}
	}
	if stuck == nil {
		stuck = []gardenWatchdogStuck{}
	}
	wip := scanGardenWatchdogWIP(cfg.Repo, cfg.Now(), cfg.WIPStaleHours)
	swept := sweepGardenWatchdog(cfg.Repo, candidates, cfg.Live)
	var ageBytes int64
	for _, c := range candidates {
		ageBytes += c.Size
	}
	hasStale := len(candidates) > 0 || len(stuck) > 0 || wip.Stale

	garden := gardenWatchdogGarden{
		Invoked:       false,
		Status:        "skipped",
		Reason:        "DRY_RUN",
		BudgetSeconds: int64(cfg.Timeout.Seconds()),
	}
	if cfg.Live {
		remaining := cfg.Timeout - time.Since(start)
		cursorPath := defaultGardenTickCursor(cfg.Repo)
		if remaining <= 0 {
			garden = timedOutGardenWatchdog(cursorPath, cfg.Timeout, time.Since(start))
		} else {
			garden = runGardenWatchdogChild(cfg.Repo, cursorPath, cfg.TickBudget, remaining)
		}
	}

	env := gardenWatchdogEnvelope{
		Schema:     "fak.garden-watchdog.v2",
		Status:     "complete",
		Reason:     "GARDEN_WATCHDOG_COMPLETE",
		Repo:       cfg.Repo,
		Live:       cfg.Live,
		MaxAgeDays: cfg.MaxAgeDays,
		AgeGC: gardenWatchdogAgeGC{
			Files: len(candidates), Bytes: ageBytes, Swept: swept, Candidates: candidates,
		},
		Stuck:    stuck,
		WIP:      wip,
		HasStale: hasStale,
		Garden:   garden,
	}
	code := 0
	if cfg.FailOnStale && hasStale {
		code = 2
	}
	return emitGardenWatchdog(stdout, stderr, env, cfg.AsJSON, code)
}

func emitGardenWatchdog(stdout, stderr io.Writer, env gardenWatchdogEnvelope, asJSON bool, code int) int {
	if asJSON {
		if rc := encodeJSONOrFail(stdout, stderr, env, "fak garden watchdog"); rc != 0 {
			return rc
		}
		return code
	}
	mode := "dry-run"
	if env.Live {
		mode = "LIVE sweep"
	}
	fmt.Fprintf(stdout, "stale-work watchdog (%s)\n", mode)
	fmt.Fprintf(stdout, "  repo: %s\n", env.Repo)
	fmt.Fprintf(stdout, "  AGE-GC (> %dd): %d files, %.1f MB\n",
		env.MaxAgeDays, env.AgeGC.Files, float64(env.AgeGC.Bytes)/1e6)
	fmt.Fprintf(stdout, "  STUCK: %d session(s)\n", len(env.Stuck))
	fmt.Fprintf(stdout, "  WIP: %d uncommitted, oldest %.1fh, stale=%v\n",
		env.WIP.Count, env.WIP.OldestAgeHours, env.WIP.Stale)
	fmt.Fprintf(stdout, "  garden tick: %s (%s), %dms\n",
		env.Garden.Status, env.Garden.Reason, env.Garden.ElapsedMillis)
	fmt.Fprintf(stdout, "  result: %s\n", env.Reason)
	return code
}

func defaultGardenWatchdogCommand(root, cursor string, tickBudget time.Duration) *exec.Cmd {
	bin := "fak"
	if self, err := os.Executable(); err == nil && self != "" {
		bin = self
	}
	seconds := int64((tickBudget + time.Second - 1) / time.Second)
	if seconds < 1 {
		seconds = 1
	}
	return exec.Command(bin,
		"garden", "tick",
		"--workspace", root,
		"--cursor", cursor,
		"--budget", strconv.FormatInt(seconds, 10),
		"--timeout", strconv.FormatInt(seconds, 10),
		"--json",
	)
}

func runGardenWatchdogChild(root, cursor string, tickBudget, timeout time.Duration) gardenWatchdogGarden {
	start := time.Now()
	cmd := gardenWatchdogCommand(root, cursor, tickBudget)
	cmd.Dir = root
	windowgate.ConfigureBackgroundCommand(cmd)
	var stdout, stderr bytes.Buffer
	cmd.Stdout, cmd.Stderr = &stdout, &stderr
	job, err := windowgate.StartInNewJob(cmd)
	if err != nil {
		return gardenWatchdogGarden{
			Invoked: true, Status: "error", Reason: "GARDEN_TICK_SPAWN_FAILED",
			BudgetSeconds: int64(timeout.Seconds()), ElapsedMillis: time.Since(start).Milliseconds(),
			Error: err.Error(), Progress: gardenWatchdogCursorProgress(cursor),
		}
	}
	done := make(chan error, 1)
	go func() { done <- cmd.Wait() }()
	timer := time.NewTimer(timeout)
	defer timer.Stop()
	select {
	case <-timer.C:
		// Windows: closing the KILL_ON_JOB_CLOSE handle reaps every descendant.
		// POSIX: procguard walks/kills the descendant tree (or the process group).
		_ = job.Close()
		killed := make(chan struct{})
		go func() {
			_, _ = procguard.KillPID(cmd.Process.Pid)
			close(killed)
		}()
		select {
		case <-done:
		case <-killed:
		case <-time.After(5 * time.Second):
		}
		return timedOutGardenWatchdog(cursor, timeout, time.Since(start))
	case err := <-done:
		_ = job.Close()
		code := cmd.ProcessState.ExitCode()
		var progress any = gardenWatchdogCursorProgress(cursor)
		var child map[string]any
		if json.Unmarshal(stdout.Bytes(), &child) == nil {
			if p, ok := child["progress"]; ok {
				progress = p
			}
		}
		status, reason := "complete", "GARDEN_TICK_COMPLETE"
		if childStatus, _ := child["status"].(string); childStatus == "partial" {
			status, reason = "partial", "GARDEN_TICK_PARTIAL"
			if childReason, _ := child["reason"].(string); childReason != "" {
				reason = childReason
			}
		} else if code != 0 || err != nil {
			status, reason = "error", "GARDEN_TICK_FAILED"
		}
		result := gardenWatchdogGarden{
			Invoked: true, Status: status, Reason: reason,
			BudgetSeconds: int64(timeout.Seconds()), ElapsedMillis: time.Since(start).Milliseconds(),
			ExitCode: &code, Progress: progress,
		}
		if status == "error" {
			result.Error = strings.TrimSpace(stderr.String())
			if result.Error == "" && err != nil {
				result.Error = err.Error()
			}
		}
		return result
	}
}

func timedOutGardenWatchdog(cursor string, budget, elapsed time.Duration) gardenWatchdogGarden {
	return gardenWatchdogGarden{
		Invoked: true, Status: "timeout", Reason: "GARDEN_TICK_TIMEOUT",
		TimedOut: true, BudgetSeconds: int64(budget.Seconds()),
		ElapsedMillis: elapsed.Milliseconds(), Progress: gardenWatchdogCursorProgress(cursor),
	}
}

func gardenWatchdogCursorProgress(path string) map[string]any {
	cur, err := gardenbudget.LoadCursor(path)
	out := map[string]any{"cursor": path, "checkpoint": cur}
	if err != nil {
		out["error"] = err.Error()
	}
	return out
}

type gardenWatchdogLock struct{ path string }

func (l *gardenWatchdogLock) release() {
	if l != nil && l.path != "" {
		_ = os.Remove(l.path)
	}
}

func acquireGardenWatchdogLock(root string, now time.Time, staleAfter time.Duration) (*gardenWatchdogLock, bool, error) {
	path := filepath.Join(root, ".dos", "garden", "watchdog.lock")
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return nil, false, err
	}
	for attempt := 0; attempt < 2; attempt++ {
		f, err := os.OpenFile(path, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o644)
		if err == nil {
			_, _ = fmt.Fprintf(f, `{"pid":%d,"started_unix":%d}`+"\n", os.Getpid(), now.Unix())
			if cerr := f.Close(); cerr != nil {
				_ = os.Remove(path)
				return nil, false, cerr
			}
			return &gardenWatchdogLock{path: path}, true, nil
		}
		if !os.IsExist(err) {
			return nil, false, err
		}
		info, statErr := os.Stat(path)
		if statErr != nil {
			if os.IsNotExist(statErr) {
				continue
			}
			return nil, false, statErr
		}
		if staleAfter > 0 && now.Sub(info.ModTime()) > staleAfter {
			if err := os.Remove(path); err != nil && !os.IsNotExist(err) {
				return nil, false, err
			}
			continue
		}
		return nil, false, nil
	}
	return nil, false, nil
}

var gardenWatchdogEphemeralGlobs = []struct {
	label string
	glob  string
}{
	{"markers", filepath.Join(".dos", "markers", "*.jsonl")},
	{"streams", filepath.Join(".dos", "streams", "*.jsonl")},
	{"stop-failures", filepath.Join(".dos", "stop-failures", "*.json")},
	{"watchdog-logs", filepath.Join("tools", "_watchdog", "*.log")},
	{"watchdog-errs", filepath.Join("tools", "_watchdog", "*.err")},
	{"watchdog-jsonl", filepath.Join("tools", "_watchdog", "*.jsonl")},
}

var gardenWatchdogEphemeralDirs = []string{
	filepath.Join(".dos", "markers"),
	filepath.Join(".dos", "streams"),
	filepath.Join(".dos", "stop-failures"),
	filepath.Join("tools", "_watchdog"),
}

func scanGardenWatchdogEphemera(root string, now time.Time, maxAgeDays int) []gardenWatchdogEphemeralFile {
	var out []gardenWatchdogEphemeralFile
	for _, spec := range gardenWatchdogEphemeralGlobs {
		paths, _ := filepath.Glob(filepath.Join(root, spec.glob))
		for _, path := range paths {
			info, err := os.Stat(path)
			if err != nil || !info.Mode().IsRegular() {
				continue
			}
			age := now.Sub(info.ModTime()).Hours() / 24
			if age < 0 {
				age = 0
			}
			if age < float64(maxAgeDays) {
				continue
			}
			rel, _ := filepath.Rel(root, path)
			out = append(out, gardenWatchdogEphemeralFile{
				Path: path, Rel: filepath.ToSlash(rel), Label: spec.label,
				AgeDays: float64(int(age*10+0.5)) / 10, Size: info.Size(),
			})
		}
	}
	sort.Slice(out, func(i, j int) bool { return out[i].AgeDays > out[j].AgeDays })
	return out
}

func scanGardenWatchdogStuck(root string, threshold int) []gardenWatchdogStuck {
	paths, _ := filepath.Glob(filepath.Join(root, ".dos", "stop-failures", "*.json"))
	var out []gardenWatchdogStuck
	for _, path := range paths {
		b, err := os.ReadFile(path)
		if err != nil {
			continue
		}
		var row struct {
			Consecutive int `json:"consecutive"`
			Total       int `json:"total"`
		}
		if json.Unmarshal(b, &row) != nil || row.Consecutive < threshold {
			continue
		}
		rel, _ := filepath.Rel(root, path)
		out = append(out, gardenWatchdogStuck{
			Session: strings.TrimSuffix(filepath.Base(path), filepath.Ext(path)),
			Rel:     filepath.ToSlash(rel), Consecutive: row.Consecutive, Total: row.Total,
		})
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Consecutive > out[j].Consecutive })
	return out
}

func scanGardenWatchdogWIP(root string, now time.Time, thresholdHours int) gardenWatchdogWIP {
	out := gardenWatchdogWIP{ThresholdHours: thresholdHours}
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	cmd := windowgate.CommandContext(ctx, "git", "-C", root, "status", "--porcelain")
	b, err := cmd.Output()
	if err != nil {
		return out
	}
	for _, line := range strings.Split(string(b), "\n") {
		if strings.TrimSpace(line) == "" {
			continue
		}
		rel := ""
		if len(line) > 3 {
			rel = line[3:]
		}
		if i := strings.LastIndex(rel, " -> "); i >= 0 {
			rel = rel[i+4:]
		}
		rel = strings.Trim(strings.TrimSpace(rel), `"`)
		if rel == "" {
			continue
		}
		out.Count++
		info, err := os.Stat(filepath.Join(root, filepath.FromSlash(rel)))
		if err != nil {
			continue
		}
		age := now.Sub(info.ModTime()).Hours()
		if age < 0 {
			age = 0
		}
		if age > out.OldestAgeHours {
			out.OldestAgeHours, out.OldestPath = age, rel
		}
	}
	out.OldestAgeHours = float64(int(out.OldestAgeHours*10+0.5)) / 10
	out.Stale = out.Count > 0 && out.OldestAgeHours >= float64(thresholdHours)
	return out
}

func sweepGardenWatchdog(root string, candidates []gardenWatchdogEphemeralFile, live bool) gardenWatchdogSweep {
	var out gardenWatchdogSweep
	for _, candidate := range candidates {
		if !insideGardenWatchdogEphemera(root, candidate.Path) {
			continue
		}
		size := candidate.Size
		if live {
			info, err := os.Stat(candidate.Path)
			if err != nil {
				continue
			}
			size = info.Size()
			if err := os.Remove(candidate.Path); err != nil {
				continue
			}
		}
		out.Files++
		out.Bytes += size
	}
	return out
}

func insideGardenWatchdogEphemera(root, path string) bool {
	resolvedRoot, err := filepath.EvalSymlinks(root)
	if err != nil {
		resolvedRoot, err = filepath.Abs(root)
		if err != nil {
			return false
		}
	}
	resolved, err := filepath.EvalSymlinks(path)
	if err != nil {
		return false
	}
	rel, err := filepath.Rel(resolvedRoot, resolved)
	if err != nil || rel == ".." || strings.HasPrefix(rel, ".."+string(filepath.Separator)) {
		return false
	}
	for _, dir := range gardenWatchdogEphemeralDirs {
		if rel != dir && strings.HasPrefix(rel, dir+string(filepath.Separator)) {
			return true
		}
	}
	return false
}

// repoRoot resolves the repo root the way the Python tool did: the parent of
// tools/. It walks up from the cwd looking for the go.mod / tools marker, and
// falls back to the cwd.
func repoRoot() string {
	if ws := strings.TrimSpace(os.Getenv("FAK_WORKSPACE_ROOT")); ws != "" {
		if _, err := os.Stat(filepath.Join(ws, "go.mod")); err == nil {
			return filepath.Clean(ws)
		}
	}
	wd, err := os.Getwd()
	if err != nil {
		return "."
	}
	dir := wd
	for {
		if _, err := os.Stat(filepath.Join(dir, "go.mod")); err == nil {
			return dir
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			return wd
		}
		dir = parent
	}
}
