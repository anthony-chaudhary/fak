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
	"time"

	"github.com/anthony-chaudhary/fak/internal/gardenbundle"
	"github.com/anthony-chaudhary/fak/internal/leaseref"
	"github.com/anthony-chaudhary/fak/internal/loopmgr"
	"github.com/anthony-chaudhary/fak/internal/pathutil"
)

func cmdGarden(argv []string) { os.Exit(runGarden(os.Stdout, os.Stderr, argv)) }

func runGarden(stdout, stderr io.Writer, argv []string) int {
	// `fak garden tick` is the LIVE, ACTING pass (#1386): it remediates the
	// surfaced stale-work conditions on a recurring cadence rather than only
	// reporting. The default `fak garden` (no subcommand) stays the read-only fold.
	if len(argv) > 0 && argv[0] == "tick" {
		return runGardenTick(stdout, stderr, argv[1:])
	}
	fs := flag.NewFlagSet("garden", flag.ContinueOnError)
	fs.SetOutput(stderr)
	workspace := fs.String("workspace", "", "workspace root (default: repo root)")
	asJSON := fs.Bool("json", false, "emit machine-readable JSON")
	check := fs.Bool("check", false, "CI gate: exit non-zero if a gating member regressed or failed to run")
	deep := fs.Bool("deep", false, "also run the fleet loop-audit member (slower; non-gating advisory)")
	timeout := fs.Int("timeout", 240, "per-member timeout seconds")
	if err := fs.Parse(argv); err != nil {
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
	register := fs.Bool("register", false, "register the durable garden-tick loop unit in the loop registry and return")
	timeout := fs.Int("timeout", 240, "per-member timeout seconds")
	ledger := fs.String("ledger", "", "loop JSONL ledger path (default: the loop ledger)")
	registry := fs.String("registry", "", "loop registry JSON path (default: the loop registry)")
	dir := fs.String("dir", "", "lease store repo dir (default: git discovery from cwd)")
	if err := fs.Parse(argv); err != nil {
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
	plan := gardenbundle.PlanTick(results, *dryRun)

	reaped, sessions, surfaced := performGardenTick(stdout, stderr, plan, *dir, *dryRun)
	witnessGardenTick(ledgerPath, plan, reaped, sessions, surfaced)

	if *asJSON {
		out := map[string]any{
			"schema":          "fak.garden-tick.v1",
			"workspace":       root,
			"commit":          gardenbundle.HeadCommit(root),
			"dry_run":         plan.DryRun,
			"acted":           plan.Acted(),
			"reaped_leases":   reaped,
			"reaped_sessions": sessions,
			"surfaced_runs":   surfaced,
			"plan":            plan,
		}
		return encodeJSONOrFail(stdout, stderr, out, "fak garden tick")
	}
	renderGardenTick(stdout, plan, reaped, sessions, surfaced)
	return 0
}

// performGardenTick executes the side effects the plan calls for. Under dry-run no
// decision has Perform=true, so this is a pure report. It returns the count of
// reaped leases / sessions and the count of orphan-run worklists surfaced.
func performGardenTick(stdout, stderr io.Writer, plan gardenbundle.TickPlan, dir string, dryRun bool) (reaped, sessions, surfaced int) {
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
	_ = stdout
	_ = dryRun
	return reaped, sessions, surfaced
}

// witnessGardenTick records the tick's run-end in the loop ledger, so a witnessed
// run end (the tick + its findings) lives in the ledger and `fak loop health` shows
// the loop alive. A ledger append failure is non-fatal: the remediation already
// happened; losing the witness line shouldn't fail the tick.
func witnessGardenTick(ledgerPath string, plan gardenbundle.TickPlan, reaped, sessions, surfaced int) {
	if ledgerPath == "" {
		return
	}
	status := loopmgr.StatusWitnessedDone
	summary := fmt.Sprintf("garden tick clean: %d member(s), nothing to act on", len(plan.Decisions))
	if plan.DryRun {
		summary = fmt.Sprintf("garden tick dry-run: would reap %d, surface %d (no side effect)", plan.ToReap, plan.ToSurface)
	} else if plan.Acted() {
		summary = fmt.Sprintf("garden tick acted: reaped %d lease(s) + %d session(s), surfaced %d orphan worklist(s)", reaped, sessions, surfaced)
	}
	metrics := map[string]int64{
		"reaped_leases":   int64(reaped),
		"reaped_sessions": int64(sessions),
		"surfaced_runs":   int64(surfaced),
		"advisory":        int64(plan.Advisory),
	}
	_, _ = loopmgr.Append(ledgerPath, loopmgr.Event{
		LoopID:  gardenTickLoopID,
		RunID:   firstNonEmpty(os.Getenv("FAK_LOOP_RUN_ID"), fmt.Sprintf("garden-tick-%d", time.Now().UnixNano())),
		Kind:    loopmgr.EventEnd,
		Status:  status,
		Source:  "fak garden tick",
		Summary: summary,
		Metrics: metrics,
	})
}

// registerGardenTickLoop installs the durable garden-tick loop in the loop registry
// (the #1281 precedent): an armed Schedule that survives a restart and re-arms at
// boot, so the act-pass recurs. Idempotent: re-registering keeps the original
// CreatedUnixNano and re-arms the same job id.
func registerGardenTickLoop(registryPath string) error {
	reg, err := loopmgr.LoadRegistry(registryPath)
	if err != nil {
		return err
	}
	if err := reg.Put(loopmgr.Job{
		Schedule: loopmgr.Schedule{
			JobID:           gardenTickLoopID,
			IntervalSeconds: gardenTickIntervalSeconds,
			MissedRun:       loopmgr.MissedSkip,
			JitterSeconds:   300,
		},
		State: loopmgr.JobArmed,
	}, time.Now()); err != nil {
		return err
	}
	return loopmgr.SaveRegistry(registryPath, reg)
}

// renderGardenTick prints the act-pass as an aligned snapshot: one row per member
// decision, then the summary of what the tick actually did.
func renderGardenTick(w io.Writer, plan gardenbundle.TickPlan, reaped, sessions, surfaced int) {
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
	if plan.Acted() {
		fmt.Fprintf(w, "  -> acted: reaped %d lease(s) + %d session(s), surfaced %d orphan worklist(s)\n", reaped, sessions, surfaced)
	} else {
		fmt.Fprintln(w, "  -> garden tick clean: nothing to act on")
	}
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
