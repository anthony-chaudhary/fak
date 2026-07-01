package main

// garden_dispatch.go — `fak garden dispatch`: the bridge issue #1791 asks for between
// the propose-only `fak garden walk` worklist and the guarded worker-spawn machinery
// in `fak dispatch tick` (internal/dispatchtick).
//
// garden walk already produces a bounded, worst-first issue worklist but runs nothing
// (see garden_walk.go). dispatch tick already has the real admission pipeline: loop
// policy gating, seat/weekly-cap preflight, lane-lease/DOS-arbitration collision
// checks, and the issue worker prompt/picker semantics. Nothing wired the two
// together, so an operator had to carry the worklist over by hand.
//
// This bridge adds ONLY the wiring, never a second admission path:
//
//  1. Load the SAME candidate set `garden walk` would propose (loadGardenWalkIssues +
//     gardenbundle.PlanWalk with the same policy knobs) so the two commands never
//     disagree about what the worklist is.
//  2. Gate the WHOLE run on the loop governor (loopmgr.Admit against the bridge's own
//     loop id) before touching a single candidate — the same mechanism a scheduler
//     line gates on via `fak loop admit`.
//  3. For each candidate issue (worst-first, budget-bounded), route it to its lane via
//     the existing router and hand BOTH the lane and the issue to evaluateDispatchTick
//     (--lane X --target-issue N under the hood) — the exact function `fak dispatch
//     tick`, `dispatch sweep`, and `dispatch wave` already call. That function owns
//     preflight (seat/weekly-cap), lane-lease/DOS-arbitration collision checks, and the
//     issue worker prompt/picker build. This file never reimplements any of them.
//  4. Classify the tick verdict into admitted+spawned or skipped-with-a-typed-reason,
//     and record one witnessed run-end in the loop ledger carrying walked / considered
//     / admitted / spawned / skipped-by-reason counts.
//
// --dry-run (the default) never sets Live on the underlying tick, so nothing is ever
// spawned — it reports exactly what WOULD happen. --apply sets Live for each admitted
// candidate in turn and stops attempting further candidates once a tick reports a
// capacity-shaped refusal (REFUSE_AT_CAP / REFUSE_NO_ACCOUNT / WEEKLY_CAPPED /
// REFUSE_HOST_DIRTY / BACKEND_UNHEALTHY) since every later candidate would refuse for
// the identical fleet-wide reason.
//
// No worker is ever spawned from `fak garden` or default `fak garden walk` — only this
// explicit, separately-invoked subcommand can reach evaluateDispatchTick's Live path.

import (
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/dispatchorder"
	"github.com/anthony-chaudhary/fak/internal/dispatchtick"
	"github.com/anthony-chaudhary/fak/internal/gardenbundle"
	"github.com/anthony-chaudhary/fak/internal/loopmgr"
)

// gardenDispatchLoopID is the loop identity the bridge gates its OWN run on (via the
// loop governor, loopmgr.Admit) and witnesses its run-end under. A distinct id from
// gardenWalkLoopID / gardenTickLoopID: it is a third, independent durable loop, and an
// operator can pause it (loop admission policy) without touching the walk or tick.
const gardenDispatchLoopID = "garden-issue-dispatch"

// gardenDispatchCapacityVerdicts are tick verdicts that reflect a FLEET-WIDE boundary
// (not something specific to the one candidate issue): every later candidate this run
// would attempt refuses for the identical reason, so --apply stops trying more of them
// the instant one of these is seen rather than burning the rest of the budget on
// certain repeats of the same refusal.
var gardenDispatchCapacityVerdicts = map[string]bool{
	"REFUSE_AT_CAP":     true,
	"REFUSE_NO_ACCOUNT": true,
	"WEEKLY_CAPPED":     true,
	"REFUSE_HOST_DIRTY": true,
	"BACKEND_UNHEALTHY": true,
}

// gardenDispatchCandidateDecision is one garden-walk decision handed to the dispatch
// bridge: a DispSkip item never reaches here (loadGardenDispatchCandidates drops it).
type gardenDispatchCandidateResult struct {
	ID          int    `json:"id"`
	Title       string `json:"title,omitempty"`
	Score       int    `json:"score"`
	Disposition string `json:"disposition"`
	Lane        string `json:"lane,omitempty"`
	Admitted    bool   `json:"admitted"`
	Spawned     bool   `json:"spawned"`
	Verdict     string `json:"verdict,omitempty"`
	Reason      string `json:"reason,omitempty"`
	SkipReason  string `json:"skip_reason,omitempty"` // typed bucket: capped|no-seat|contended|unrouted|refused|""
}

type gardenDispatchPlan struct {
	Schema      string                          `json:"schema"`
	DryRun      bool                            `json:"dry_run"`
	Budget      int                             `json:"budget"`
	Walked      int                             `json:"walked"`
	Considered  int                             `json:"considered"` // candidates handed to dispatch (act+review, budgeted)
	Admitted    int                             `json:"admitted"`   // ticks that cleared every gate (spawned, or would_spawn under dry-run)
	Spawned     int                             `json:"spawned"`    // live spawns only (0 under dry-run)
	SkippedBy   map[string]int                  `json:"skipped_by,omitempty"`
	Results     []gardenDispatchCandidateResult `json:"results"`
	StopReason  string                          `json:"stop_reason,omitempty"`
	LoopAdmit   bool                            `json:"loop_admit"`
	LoopVerdict string                          `json:"loop_verdict,omitempty"`
	LoopReason  string                          `json:"loop_reason,omitempty"`
	Verdict     string                          `json:"verdict"`
	Reason      string                          `json:"reason"`
}

// runGardenDispatch is the testable core of `fak garden dispatch`.
func runGardenDispatch(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("garden dispatch", flag.ContinueOnError)
	fs.SetOutput(stderr)
	workspace := fs.String("workspace", "", "workspace root (default: repo root)")
	repo := fs.String("repo", "", "gh repo (owner/name); default: the current repo")
	state := fs.String("state", "open", "issue state filter (open|closed|all)")
	limit := fs.Int("limit", 400, "max items to load from the source")
	input := fs.String("input", "", "read the item set from a gh-issue-list JSON file instead of calling gh (offline/test) -- the same fixture shape garden walk --input accepts")
	budget := fs.Int("budget", 20, "cap the candidate set to the worst N items needing attention, same semantics as garden walk --budget")
	skipActive := fs.Bool("skip-active", true, "skip items already in-progress, same as garden walk --skip-active")
	skipFresh := fs.Int("skip-fresh", 0, "also skip items idle fewer than this many days, same as garden walk --skip-fresh")
	backend := fs.String("backend", "claude", "worker backend passed to each dispatch tick (claude|opencode|codex)")
	maxWorkers := fs.Int("max-workers", dispatchtick.DefaultMaxWorkers, "hard cap on live workers, enforced by dispatch preflight (same knob as dispatch tick)")
	apply := fs.Bool("apply", false, "attempt to actually spawn admitted candidates (default: dry-run, spawns nothing)")
	dryRunFlag := fs.Bool("dry-run", false, "explicit alias for the default: report the decision per candidate, spawn nothing")
	asJSON := fs.Bool("json", false, "emit machine-readable JSON")
	ledger := fs.String("ledger", "", "loop JSONL ledger path (default: the loop ledger)")
	policyPath := fs.String("policy", "", "loop admission policy JSON path (default: the loop policy)")
	noLoopLedger := fs.Bool("no-loop-ledger", false, "skip recording this run in the loop ledger (hermetic probes)")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintf(stderr, "fak garden dispatch: unexpected argument %q\n", fs.Arg(0))
		return 2
	}
	if *apply && *dryRunFlag {
		fmt.Fprintln(stderr, "fak garden dispatch: --apply and --dry-run are mutually exclusive")
		return 2
	}
	live := *apply

	root := *workspace
	if root == "" {
		root = repoRoot()
	} else if abs, err := filepath.Abs(root); err == nil {
		root = abs
	}

	if gardenbundle.GardenOff() {
		fmt.Fprintln(stdout, "garden dispatch skipped: FAK_GARDEN is off")
		return 0
	}

	ledgerPath := firstNonEmpty(*ledger, defaultLoopLedger())
	policyFile := firstNonEmpty(*policyPath, defaultLoopPolicy())

	plan := &gardenDispatchPlan{
		Schema: "fak.garden-dispatch.v1",
		DryRun: !live,
		Budget: *budget,
	}

	// Gate 1: the loop governor. Same mechanism `fak loop admit` exposes at the CLI --
	// called in-process here so the bridge honors an operator pause/backoff/cadence
	// floor on ITS OWN loop id before a single candidate is even loaded, exactly like a
	// scheduler line gating on `fak loop admit --loop garden-issue-dispatch`.
	admitDecision, admitErr := evaluateGardenDispatchLoopAdmit(ledgerPath, policyFile)
	if admitErr != nil {
		fmt.Fprintf(stderr, "fak garden dispatch: loop admit: %v\n", admitErr)
		return 1
	}
	plan.LoopAdmit = admitDecision.Admit
	plan.LoopVerdict = admitDecision.Reason
	plan.LoopReason = admitDecision.Summary
	if !admitDecision.Admit {
		plan.Verdict = "LOOP_REFUSED"
		plan.Reason = fmt.Sprintf("loop governor refused %s: %s", gardenDispatchLoopID, admitDecision.Summary)
		witnessGardenDispatch(ledgerPath, !*noLoopLedger, plan)
		return renderGardenDispatchResult(stdout, stderr, plan, *asJSON)
	}

	items, _, err := loadGardenWalkIssues(*input, *repo, *state, *limit)
	if err != nil {
		fmt.Fprintf(stderr, "fak garden dispatch: %v\n", err)
		return 1
	}
	walkPlan := gardenbundle.PlanWalk("issue", items, gardenbundle.WalkPolicy{
		Budget:         *budget,
		SkipFreshDays:  *skipFresh,
		SkipInProgress: *skipActive,
		DryRun:         true, // the walk fold itself always proposes; --apply governs the TICK below, not this fold
	})
	plan.Walked = walkPlan.Total

	candidates := gardenDispatchCandidates(walkPlan)
	plan.Considered = len(candidates)
	plan.SkippedBy = map[string]int{}

	router, routerErr := dispatchRouteIssues(root, stderr)
	laneByIssue := map[int]string{}
	if routerErr == nil {
		for _, ir := range router.Issues {
			laneByIssue[ir.Number] = ir.Lane
		}
	}

	for _, cand := range candidates {
		res := gardenDispatchCandidateResult{
			ID:          cand.ID,
			Title:       cand.Title,
			Score:       cand.Score,
			Disposition: string(cand.Disposition),
		}
		lane, laneOK := laneByIssue[cand.ID]
		if !laneOK || strings.TrimSpace(lane) == "" {
			res.SkipReason = "unrouted"
			res.Reason = "no dispatch lane routes to this issue (router has no entry for it)"
			plan.SkippedBy[res.SkipReason]++
			plan.Results = append(plan.Results, res)
			continue
		}
		res.Lane = lane

		payload, tickErr := evaluateDispatchTick(dispatchTickOptions{
			Workspace:      root,
			MaxWorkers:     *maxWorkers,
			WorkKind:       dispatchtickWorkKind(*backend),
			Lane:           lane,
			TargetIssue:    cand.ID,
			Backend:        *backend,
			Live:           live,
			Refresh:        true,
			CooldownMin:    dispatchtick.DefaultCooldownMinutes,
			WorkerTimeoutS: dispatchtick.DefaultWorkerTimeoutS,
			SpawnProbeS:    dispatchtick.DefaultSpawnProbeS,
			RecordLoop:     !*noLoopLedger,
			LoopLedger:     ledgerPath,
		}, stderr)
		if tickErr != nil {
			res.SkipReason = "tick-error"
			res.Reason = tickErr.Error()
			plan.SkippedBy[res.SkipReason]++
			plan.Results = append(plan.Results, res)
			continue
		}

		action := dispatchMapString(payload, "action")
		verdict := dispatchMapString(payload, "verdict")
		res.Verdict = verdict
		res.Reason = dispatchMapString(payload, "reason")

		switch action {
		case "spawned":
			res.Admitted = true
			res.Spawned = true
			plan.Admitted++
			plan.Spawned++
		case "would_spawn":
			res.Admitted = true
			plan.Admitted++
		default:
			res.SkipReason = gardenDispatchSkipReason(verdict, action)
			plan.SkippedBy[res.SkipReason]++
		}
		plan.Results = append(plan.Results, res)

		if gardenDispatchCapacityVerdicts[verdict] {
			plan.StopReason = fmt.Sprintf("stopped after candidate #%d: %s is a fleet-wide boundary, every remaining candidate would refuse the same way", cand.ID, verdict)
			break
		}
	}

	switch {
	case plan.Considered == 0:
		plan.Verdict = "NO_CANDIDATES"
		plan.Reason = fmt.Sprintf("walked %d issue(s); none need attention (nothing to dispatch)", plan.Walked)
	case plan.Admitted == 0:
		plan.Verdict = "NONE_ADMITTED"
		plan.Reason = fmt.Sprintf("considered %d candidate(s); none admitted (see skipped_by for reasons)", plan.Considered)
	case live && plan.Spawned == plan.Admitted:
		plan.Verdict = "APPLIED"
		plan.Reason = fmt.Sprintf("spawned %d of %d considered candidate(s)", plan.Spawned, plan.Considered)
	case live:
		plan.Verdict = "PARTIAL"
		plan.Reason = fmt.Sprintf("admitted %d, spawned %d of %d considered candidate(s)", plan.Admitted, plan.Spawned, plan.Considered)
	default:
		plan.Verdict = "WOULD_APPLY"
		plan.Reason = fmt.Sprintf("dry-run: %d of %d considered candidate(s) would be admitted; re-run with --apply to spawn", plan.Admitted, plan.Considered)
	}

	witnessGardenDispatch(ledgerPath, !*noLoopLedger, plan)
	return renderGardenDispatchResult(stdout, stderr, plan, *asJSON)
}

// gardenDispatchCandidates reduces a walk plan's budgeted decisions to the ones the
// dispatch bridge should attempt: DispAct and DispReview only (a healthy/skip item
// never reaches here, matching garden walk's own worklist exactly).
func gardenDispatchCandidates(plan gardenbundle.WalkPlan) []gardenbundle.WalkDecision {
	out := make([]gardenbundle.WalkDecision, 0, len(plan.Decisions))
	for _, d := range plan.Decisions {
		if d.Disposition == gardenbundle.DispAct || d.Disposition == gardenbundle.DispReview {
			out = append(out, d)
		}
	}
	return out
}

// gardenDispatchSkipReason buckets a refusing tick verdict into the typed reason
// vocabulary the acceptance criteria ask for (capped / no-seat / contended / ...),
// without inventing a second verdict space -- it is a pure re-label of the SAME
// verdict evaluateDispatchTick already returns.
func gardenDispatchSkipReason(verdict, action string) string {
	switch verdict {
	case "REFUSE_AT_CAP":
		return "capped"
	case "WEEKLY_CAPPED":
		return "capped"
	case "REFUSE_NO_ACCOUNT":
		return "no-seat"
	case "REFUSE_HOST_DIRTY":
		return "host-dirty"
	case "BACKEND_UNHEALTHY":
		return "backend-unhealthy"
	case "LANE_BUSY", "LANE_LEASE_HELD", "IN_FLIGHT_DUPLICATE", dispatchorder.ReasonCollisionRisk:
		return "contended"
	case "SELF_MODIFY_HOLD":
		return "self-modify-hold"
	case "NO_LANE", "NO_ISSUE":
		return "no-candidate"
	case "SPAWN_FAILED":
		return "spawn-failed"
	default:
		if action == "refused" {
			return "refused"
		}
		return "unknown"
	}
}

// evaluateGardenDispatchLoopAdmit calls the SAME loop governor `fak loop admit`
// exposes at the CLI (loopmgr.Admit against a ledger snapshot + policy file), scoped
// to this bridge's own loop id, so an operator pause/cadence-floor/backoff policy
// gates the whole run before a single candidate is touched.
func evaluateGardenDispatchLoopAdmit(ledgerPath, policyPath string) (loopmgr.Decision, error) {
	policies, err := loopmgr.LoadPolicies(policyPath)
	if err != nil {
		return loopmgr.Decision{}, err
	}
	now := time.Now()
	st, err := loopmgr.SnapshotFile(ledgerPath, now)
	if err != nil {
		return loopmgr.Decision{}, err
	}
	snapshot := loopSnapshotForID(st, gardenDispatchLoopID)
	return loopmgr.Admit(snapshot, policies.PolicyFor(gardenDispatchLoopID), now), nil
}

// witnessGardenDispatch appends the bridge's run-end to the loop ledger with the
// counts the issue asks for: walked, considered, admitted, spawned, and skipped by
// reason (flattened to skipped_<reason> metrics, plus a total skipped_total).
func witnessGardenDispatch(ledgerPath string, record bool, plan *gardenDispatchPlan) {
	if !record || ledgerPath == "" {
		return
	}
	metrics := map[string]int64{
		"walked":     int64(plan.Walked),
		"considered": int64(plan.Considered),
		"admitted":   int64(plan.Admitted),
		"spawned":    int64(plan.Spawned),
	}
	skippedTotal := int64(0)
	for reason, n := range plan.SkippedBy {
		metrics["skipped_"+strings.ReplaceAll(reason, "-", "_")] = int64(n)
		skippedTotal += int64(n)
	}
	metrics["skipped_total"] = skippedTotal
	status := loopmgr.StatusWitnessedDone
	if plan.Verdict == "LOOP_REFUSED" {
		status = loopmgr.StatusWitnessRefused
	}
	_, _ = loopmgr.Append(ledgerPath, loopmgr.Event{
		LoopID:  gardenDispatchLoopID,
		RunID:   firstNonEmpty(os.Getenv("FAK_LOOP_RUN_ID"), fmt.Sprintf("garden-dispatch-%d", time.Now().UnixNano())),
		Kind:    loopmgr.EventEnd,
		Status:  status,
		Source:  "fak garden dispatch",
		Summary: fmt.Sprintf("garden dispatch %s: %s", plan.Verdict, plan.Reason),
		Metrics: metrics,
	})
}

func renderGardenDispatchResult(stdout, stderr io.Writer, plan *gardenDispatchPlan, asJSON bool) int {
	if asJSON {
		code := encodeJSONOrFail(stdout, stderr, plan, "fak garden dispatch")
		if code != 0 {
			return code
		}
	} else {
		fmt.Fprint(stdout, renderGardenDispatch(plan))
	}
	if plan.Verdict == "LOOP_REFUSED" {
		return 3 // mirrors `fak loop admit`'s exit-3 refused contract
	}
	return 0
}

// renderGardenDispatch prints the bridge run as an aligned worklist: mode, the loop
// governor's verdict, then one row per candidate's dispatch decision.
func renderGardenDispatch(plan *gardenDispatchPlan) string {
	var b strings.Builder
	mode := "dry-run"
	if !plan.DryRun {
		mode = "apply"
	}
	fmt.Fprintf(&b, "garden dispatch (%s) -- %s\n\n", mode, plan.Verdict)
	fmt.Fprintf(&b, "  loop admit: %v (%s)\n", plan.LoopAdmit, plan.LoopReason)
	if !plan.LoopAdmit {
		fmt.Fprintf(&b, "\n  -> %s\n", plan.Reason)
		return b.String()
	}
	fmt.Fprintf(&b, "  walked %d   considered %d   admitted %d   spawned %d\n\n",
		plan.Walked, plan.Considered, plan.Admitted, plan.Spawned)
	for _, r := range plan.Results {
		mark := "."
		switch {
		case r.Spawned:
			mark = "*"
		case r.Admitted:
			mark = "+"
		}
		title := r.Title
		if len(title) > 40 {
			title = title[:39] + "…"
		}
		status := "admitted"
		if !r.Admitted {
			status = "skip:" + r.SkipReason
		}
		fmt.Fprintf(&b, "  %s #%-5d %5d  lane=%-14s %-20s %s\n", mark, r.ID, r.Score, firstNonEmpty(r.Lane, "-"), status, title)
		if r.Reason != "" {
			fmt.Fprintf(&b, "        %s\n", r.Reason)
		}
	}
	if plan.StopReason != "" {
		fmt.Fprintf(&b, "\n  STOP: %s\n", plan.StopReason)
	}
	fmt.Fprintf(&b, "\n  -> %s\n", plan.Reason)
	return b.String()
}
