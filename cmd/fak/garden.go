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
	"context"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/commitlifecycle"
	"github.com/anthony-chaudhary/fak/internal/gardenbundle"
	"github.com/anthony-chaudhary/fak/internal/growthgate"
	"github.com/anthony-chaudhary/fak/internal/leaseref"
	"github.com/anthony-chaudhary/fak/internal/loopmgr"
	"github.com/anthony-chaudhary/fak/internal/pathutil"
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
var gardenReclaimInspect = inspectGardenReclaim

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

	results := gardenbundle.Collect(root, "", time.Duration(*timeout)*time.Second, false)
	results = append(results, gardenReclaimInspect(stderr, root))
	plan := gardenbundle.PlanTick(results, *dryRun)

	growthApply := growthApplyEnabled(*growthApplyFlag)
	reaped, sessions, surfaced, lockFiles, collected, intents, folded := performGardenTick(stdout, stderr, plan, *dir, root, *dryRun, growthApply)
	witnessGardenTick(ledgerPath, plan, reaped, sessions, surfaced, lockFiles, collected, intents, folded)

	if *asJSON {
		out := map[string]any{
			"schema":                "fak.garden-tick.v1",
			"workspace":             root,
			"commit":                gardenbundle.HeadCommit(root),
			"dry_run":               plan.DryRun,
			"acted":                 plan.Acted(),
			"reaped_leases":         reaped,
			"reaped_sessions":       sessions,
			"reaped_lock_files":     lockFiles,
			"reaped_intents":        intents,
			"reaped_growth_logs":    collected,
			"folded_sentinel_lines": folded,
			"surfaced_runs":         surfaced,
			"plan":                  plan,
		}
		return encodeJSONOrFail(stdout, stderr, out, "fak garden tick")
	}
	renderGardenTick(stdout, plan, reaped, sessions, surfaced, lockFiles, collected, intents, folded)
	return 0
}

// performGardenTick executes the side effects the plan calls for. Under dry-run no
// decision has Perform=true, so this is a pure report. It returns the count of
// reaped leases / sessions, the count of orphan-run worklists surfaced, the count
// of orphan .lock files swept, the count of oversized disposable logs the growthgate
// collect reaped (0 in the ledger-only default), the count of lapsed intent leases
// reaped (the reap-parity sweep, #5345), and the count of decision-note lines folded
// off the empty-tree sentinel note (census row 12, #5361).
func performGardenTick(stdout, stderr io.Writer, plan gardenbundle.TickPlan, dir, root string, dryRun, growthApply bool) (reaped, sessions, surfaced, lockFiles, collected, intents, folded int) {
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
				reaped += len(leases)
			}
			sess, serr := store.ReapSessions(ctx, now)
			if serr != nil {
				fmt.Fprintf(stderr, "fak garden tick: reap sessions: %v\n", serr)
			} else {
				sessions += len(sess)
			}
		case gardenbundle.ActSurface:
			// The worklist is already in the member's surfaced detail; surfacing
			// it means recording the witnessed condition (the ledger event, below)
			// — re-dispatch stays the gated operator action, never automatic here.
			surfaced++
		}
	}
	// Orphan .lock sweep (#5348): run ONCE PER TICK, unconditionally — independent
	// of whether any lease RECORD is expired. A ghost <git-common-dir>/refs/fak/locks/
	// *.lock (a holder killed mid-CAS) can outlive its lease even when no lease record
	// is stale, so gating this on the stale_leases ActReap decision above would leave
	// it uncollected. ReapLockFiles is fail-safe by construction (namespace-confined,
	// age-bounded at the lease TTL, future-mtime kept). Dry-run performs no delete,
	// mirroring the reap decisions above (which never Perform under dry-run).
	if !dryRun {
		store := leaseref.NewInDir(dir)
		locks, _, lerr := store.ReapLockFiles(context.Background(), time.Now(), 0)
		if lerr != nil {
			fmt.Fprintf(stderr, "fak garden tick: reap orphan locks: %v\n", lerr)
		} else {
			lockFiles += len(locks)
		}
		// Intent-lease reap (#5345 reap parity): run ONCE PER TICK, unconditionally —
		// closing the same reaper asymmetry that bit sessions (#5344). A lapsed
		// refs/fak/locks/intent-<key> accretes independently of whether any lease RECORD
		// is expired, so gating this on the ActReap decision above (as the lease/session
		// reaps are) would leave the intent namespace uncollected on the automatic loop —
		// the CLI's `fak leaseref reap` already sweeps all three kinds. Same best-effort,
		// idempotent contract as the ReapLockFiles sweep just above.
		ints, ierr := store.ReapIntents(context.Background(), time.Now())
		if ierr != nil {
			fmt.Fprintf(stderr, "fak garden tick: reap intents: %v\n", ierr)
		} else {
			intents += len(ints)
		}
		// growthgate collect (#5349): run ONCE PER TICK over the repo + Fleet trees,
		// the acting half of the ActGrowthReap edge. DELETE-SAFE by default: it appends
		// the would-reap set to the reap ledger every tick (the soak evidence) but
		// removes NOTHING unless the apply opt-in is set. Dry-run skips it entirely,
		// mirroring the lock sweep above (which never deletes under dry-run).
		collected += collectGrowthLogs(stderr, growthCensusRoots(root), growthApply, growthReapLedgerPath())
		// Decisions-note fold (#5361, census row 12): run ONCE PER TICK, bounding the ONE
		// unbounded producer on refs/notes/fak/decisions — the empty-tree sentinel note that
		// every pre-commit refusal (OFF_TRUNK / PATHSPEC_RACE / LEASE_HELD) appends to
		// forever. CompactSentinelNote keeps the most-recent sentinelNoteKeepLines lines
		// (recency = forensic value) and force-overwrites ONLY that one sentinel note on the
		// side ref; commit-anchored notes are bounded per-object evidence and are NEVER
		// touched, and it never touches main / HEAD / refs/heads and never pushes. Bind the
		// recorder to `root` (the gardened repo) so the fold hits its decisions ref, not the
		// process cwd's. Best-effort + idempotent, same contract as the sweeps above: a fold
		// error is logged and swallowed so it never fails the tick. Dry-run skips it.
		fold, ferr := witness.NewRecorderForDir(root).CompactSentinelNote(context.Background(), sentinelKeepLines())
		if ferr != nil {
			fmt.Fprintf(stderr, "fak garden tick: fold decisions sentinel note: %v\n", ferr)
		} else {
			folded += fold
		}
	}
	_ = stdout
	return reaped, sessions, surfaced, lockFiles, collected, intents, folded
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

// repoRoot resolves the repo root the way the Python tool did: the parent of
// tools/. It walks up from the cwd looking for the go.mod / tools marker, and
// falls back to the cwd.
func repoRoot() string {
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
