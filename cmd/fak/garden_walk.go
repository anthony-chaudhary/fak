package main

// garden_walk.go — the cmd wiring for `fak garden walk`: the resource-aware pass
// over the HUNDREDS of individual garden items a single member surfaces. Where
// `fak garden` folds the ~8 orchestrator members and `fak garden tick` acts at the
// member level, `walk` zooms IN to the items themselves — today the open-issue
// backlog (347+ live items) classified by the issue gardener.
//
// It is the answer to "one garden command that walks sets of 100s of garden items,
// aware of what to do or not, to save resources/time": it loads the set once,
// classifies each item with the existing deterministic gardener (no per-item
// network), cheaply SKIPS the live/healthy ones, and proposes a budget-bounded,
// worst-first worklist for the rest. PROPOSE-ONLY by design (the issue-garden /
// trajectory-garden discipline): it emits the exact command per item, it does not
// execute it — auto-apply is a later witness-gated rung.
//
// Durable like the tick: --register installs a loop unit and every run appends a
// witnessed run-end to the loop ledger, so `fak loop health` shows the walk living.

import (
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/gardenbundle"
	"github.com/anthony-chaudhary/fak/internal/loopmgr"
)

// gardenWalkLoopID / gardenWalkIntervalSeconds register the durable walk loop. The
// issue backlog moves on the order of a day (open/label/close), so a 6-hour cadence
// keeps the worklist warm without churn — slower than the hourly stale-work tick.
const gardenWalkLoopID = "garden-item-walk"
const gardenWalkIntervalSeconds = 21600 // 6h

// runGardenWalk loads the garden item set, folds it through the pure resource-aware
// planner, renders/ledgers, and (with --register) installs the durable loop unit.
func runGardenWalk(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("garden walk", flag.ContinueOnError)
	fs.SetOutput(stderr)
	workspace := fs.String("workspace", "", "workspace root (default: repo root)")
	source := fs.String("source", "issues", "garden item set to walk (currently: issues)")
	repo := fs.String("repo", "", "gh repo (owner/name); default: the current repo")
	state := fs.String("state", "open", "issue state filter (open|closed|all)")
	limit := fs.Int("limit", 400, "max items to load from the source")
	input := fs.String("input", "", "read the item set from a gh-issue-list JSON file instead of calling gh (offline/test)")
	budget := fs.Int("budget", 20, "cap the worklist to the worst N items needing attention; the rest defer to the next pass (<=0: unbounded)")
	skipActive := fs.Bool("skip-active", true, "skip items already in-progress (the cheap 'someone is on it' pre-filter; immune to timestamp churn)")
	skipFresh := fs.Int("skip-fresh", 0, "also skip any item idle fewer than this many days (off by default: on a bot-churned tracker the update timestamp bumps constantly, so this over-skips)")
	asJSON := fs.Bool("json", false, "emit machine-readable JSON")
	register := fs.Bool("register", false, "register the durable garden-walk loop unit in the loop registry and return")
	ledger := fs.String("ledger", "", "loop JSONL ledger path (default: the loop ledger)")
	registry := fs.String("registry", "", "loop registry JSON path (default: the loop registry)")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintf(stderr, "fak garden walk: unexpected argument %q\n", fs.Arg(0))
		return 2
	}

	root := *workspace
	if root == "" {
		root = repoRoot()
	} else if abs, err := filepath.Abs(root); err == nil {
		root = abs
	}

	ledgerPath := firstNonEmpty(*ledger, defaultLoopLedger())
	registryPath := firstNonEmpty(*registry, defaultLoopRegistry())

	if *register {
		if err := registerGardenWalkLoop(registryPath); err != nil {
			fmt.Fprintf(stderr, "fak garden walk: register: %v\n", err)
			return 1
		}
		fmt.Fprintf(stdout, "registered durable loop %q (interval %ds) in %s\n", gardenWalkLoopID, gardenWalkIntervalSeconds, registryPath)
		return 0
	}

	if gardenbundle.GardenOff() {
		fmt.Fprintln(stdout, "garden walk skipped: FAK_GARDEN is off")
		return 0
	}

	// Today the only wired source is issues. Reject an unknown source loudly rather
	// than silently walking nothing — the seam (WalkItem is source-agnostic) is ready
	// for trajectories/runs, but a source must be wired here to be honest about it.
	if *source != "issues" {
		fmt.Fprintf(stderr, "fak garden walk: source %q not wired yet (have: issues); the WalkItem seam is ready for it\n", *source)
		return 2
	}

	items, srcLabel, err := loadGardenWalkIssues(*input, *repo, *state, *limit)
	if err != nil {
		fmt.Fprintf(stderr, "fak garden walk: %v\n", err)
		return 1
	}

	// PROPOSE-ONLY: DryRun is always on for the issues source (the gardener emits the
	// command; the operator or a future witness-gated rung runs it). The Perform/!DryRun
	// path is exercised by the pure planner tests and reserved for that rung.
	plan := gardenbundle.PlanWalk("issue", items, gardenbundle.WalkPolicy{
		Budget:         *budget,
		SkipFreshDays:  *skipFresh,
		SkipInProgress: *skipActive,
		DryRun:         true,
	})
	plan.Source = srcLabel

	witnessGardenWalk(ledgerPath, plan)

	if *asJSON {
		return encodeJSONOrFail(stdout, stderr, plan, "fak garden walk")
	}
	renderGardenWalk(stdout, plan)
	return 0
}

// loadGardenWalkIssues loads the open-issue set and classifies each item with the
// existing issue gardener (classify + duplicate detection + per-item action), then
// reduces every row to a source-agnostic WalkItem. Closed rows are dropped.
func loadGardenWalkIssues(input, repo, state string, limit int) ([]gardenbundle.WalkItem, string, error) {
	issues, source, err := loadTUIIssues(input, repo, state, limit)
	if err != nil {
		return nil, "", err
	}
	report := buildTUIIssueReport(issues, source, time.Now().UTC(), 0)
	actByNum := make(map[int]tuiIssueAction, len(report.Actions))
	for _, a := range report.Actions {
		actByNum[a.Number] = a
	}
	items := make([]gardenbundle.WalkItem, 0, len(report.Rows))
	for _, row := range report.Rows {
		if strings.EqualFold(row.State, "closed") {
			continue
		}
		it := gardenbundle.WalkItem{
			ID:         row.Number,
			Title:      row.Title,
			Score:      row.Score,
			IdleDays:   row.IdleDays,
			InProgress: row.InProgress,
		}
		if a, ok := actByNum[row.Number]; ok {
			it.Action = a.Kind
			it.Reason = a.Reason
			switch a.Kind {
			case "close-dormant-question", "mark-stale":
				it.Disposition = gardenbundle.DispAct
				it.Command = a.Command
			default:
				// "review" and any future judgment-only kind: surfaced, never auto-acted.
				it.Disposition = gardenbundle.DispReview
			}
		} else {
			// No action from the gardener -> the item is healthy: nothing to do.
			it.Disposition = gardenbundle.DispSkip
			it.Reason = "no condition"
		}
		items = append(items, it)
	}
	return items, "issue", nil
}

// witnessGardenWalk appends the walk's run-end to the loop ledger, so a witnessed
// run end (the walk + its counts) lives in the ledger and `fak loop health` shows
// the loop alive. A ledger append failure is non-fatal: the proposal already
// printed; losing the witness line shouldn't fail the walk.
func witnessGardenWalk(ledgerPath string, plan gardenbundle.WalkPlan) {
	if ledgerPath == "" {
		return
	}
	summary := fmt.Sprintf("garden walk: %d %s walked; %d need attention (%d act, %d review), %d deferred, %d skipped",
		plan.Total, plan.Source, plan.Attention, plan.Acted, plan.Review, plan.Deferred, plan.Active+plan.Fresh+plan.Healthy)
	metrics := map[string]int64{
		"walked":    int64(plan.Total),
		"attention": int64(plan.Attention),
		"acted":     int64(plan.Acted),
		"review":    int64(plan.Review),
		"deferred":  int64(plan.Deferred),
		"skipped":   int64(plan.Active + plan.Fresh + plan.Healthy),
	}
	_, _ = loopmgr.Append(ledgerPath, loopmgr.Event{
		LoopID:  gardenWalkLoopID,
		RunID:   firstNonEmpty(os.Getenv("FAK_LOOP_RUN_ID"), fmt.Sprintf("garden-walk-%d", time.Now().UnixNano())),
		Kind:    loopmgr.EventEnd,
		Status:  loopmgr.StatusWitnessedDone,
		Source:  "fak garden walk",
		Summary: summary,
		Metrics: metrics,
	})
}

// registerGardenWalkLoop installs the durable garden-walk loop (the #1281 precedent):
// an armed Schedule that survives a restart and re-arms at boot. Idempotent.
func registerGardenWalkLoop(registryPath string) error {
	reg, err := loopmgr.LoadRegistry(registryPath)
	if err != nil {
		return err
	}
	if err := reg.Put(loopmgr.Job{
		Schedule: loopmgr.Schedule{
			JobID:           gardenWalkLoopID,
			IntervalSeconds: gardenWalkIntervalSeconds,
			MissedRun:       loopmgr.MissedSkip,
			JitterSeconds:   600,
		},
		State: loopmgr.JobArmed,
	}, time.Now()); err != nil {
		return err
	}
	return loopmgr.SaveRegistry(registryPath, reg)
}

// renderGardenWalk prints the walk as an aligned worklist: the envelope verdict,
// the resource counts, then one row per budgeted decision worst-first.
func renderGardenWalk(w io.Writer, plan gardenbundle.WalkPlan) {
	fmt.Fprintf(w, "garden walk -- %s (%s)  source=%s\n\n", plan.Verdict, plan.Finding, plan.Source)
	fmt.Fprintf(w, "  walked %d   attention %d   skipped %d (in-progress %d, fresh %d, healthy %d)   budget %d   deferred %d\n\n",
		plan.Total, plan.Attention, plan.Active+plan.Fresh+plan.Healthy, plan.Active, plan.Fresh, plan.Healthy, plan.Budget, plan.Deferred)
	if len(plan.Decisions) == 0 {
		fmt.Fprintf(w, "  -> %s\n", plan.NextAction)
		return
	}
	for _, d := range plan.Decisions {
		mark := "."
		if d.Disposition == gardenbundle.DispAct {
			mark = "*"
		}
		title := d.Title
		if len(title) > 48 {
			title = title[:47] + "…"
		}
		kind := string(d.Disposition)
		if d.Action != "" {
			kind += ":" + d.Action
		}
		fmt.Fprintf(w, "  %s #%-5d %5d  %-26s idle %3dd  %s\n", mark, d.ID, d.Score, kind, d.IdleDays, title)
		if d.Command != "" {
			fmt.Fprintf(w, "        $ %s\n", d.Command)
		}
	}
	fmt.Fprintf(w, "\n  -> %s\n", plan.NextAction)
}
