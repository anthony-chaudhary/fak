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
	"github.com/anthony-chaudhary/fak/internal/seatpark"
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

// seatParkReasonNoSeat is the ledger Reason token a garden-dispatch run-end records when
// a LIVE run ATTEMPTED a spawn and stopped on a seat refuse (#3523) — so a later run's
// Gate-1.5 park-and-retry counts it toward the bounded budget. A deferred run records
// seatpark.StatusParked (neutral in the tail); an exhausted one records
// seatpark.StatusExhausted (which ends the tail so the next cycle starts fresh).
const seatParkReasonNoSeat = "SEAT_NO_SEAT"

// gardenDispatchSeatRefuses are the tick verdicts #3523 treats as a "no free seat" refuse
// specifically — distinct from a drained queue (NO_LANE/NO_ISSUE) or a fault, so only a
// genuine seat wall arms the park-and-retry. WEEKLY_CAPPED rides along per the issue: it
// too is an account-budget wall a bounded wait respects.
var gardenDispatchSeatRefuses = map[string]bool{
	"REFUSE_NO_ACCOUNT": true,
	"WEEKLY_CAPPED":     true,
}

// deriveSeatParkState folds the loop ledger's garden-dispatch run-ends (newest→oldest)
// into the consecutive no-seat park count and the most-recent no-seat attempt time that
// seatpark.Decide keys on. A deferred run (seatpark.StatusParked) is NEUTRAL — it neither
// counts nor ends the tail (it is the consequence of parking, not a new no-seat failure);
// any run that made progress or ended for another reason (including an exhausted park)
// ends the tail, so a fresh cycle starts from zero. Pure: ledger in, counts out.
func deriveSeatParkState(events []loopmgr.Event) (parks int, lastParkUnix int64) {
	return deriveSeatParkStateForLoop(events, gardenDispatchLoopID)
}

// deriveSeatParkStateForLoop is the shared fold behind every seat-park tail. The loop id is
// the ONLY thing that ever differed between the per-loop copies — the park vocabulary
// (seatParkReasonNoSeat, seatpark.StatusParked) and the tail-boundary rule are shared on
// purpose, because seatpark.Decide is one policy and two loops reading the same ledger with
// different boundary rules would back off on different schedules for the same seat shortage.
func deriveSeatParkStateForLoop(events []loopmgr.Event, loopID string) (parks int, lastParkUnix int64) {
	for i := len(events) - 1; i >= 0; i-- {
		ev := events[i]
		if ev.LoopID != loopID || ev.Kind != loopmgr.EventEnd {
			continue
		}
		switch ev.Reason {
		case string(seatpark.StatusParked):
			continue // a chosen wait — transparent in the park tail
		case seatParkReasonNoSeat:
			parks++
			if lastParkUnix == 0 {
				lastParkUnix = ev.TSUnixNano / 1_000_000_000
			}
		default:
			return parks, lastParkUnix // tail boundary: progress, exhaustion, or another stop
		}
	}
	return parks, lastParkUnix
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
	backend := fs.String("backend", "codex", "worker backend passed to each dispatch tick (claude|opencode|codex); default codex")
	maxWorkers := fs.Int("max-workers", dispatchtick.DefaultMaxWorkers, "hard cap on live workers, enforced by dispatch preflight (same knob as dispatch tick)")
	apply := fs.Bool("apply", false, "attempt to actually spawn admitted candidates (default: dry-run, spawns nothing)")
	dryRunFlag := fs.Bool("dry-run", false, "explicit alias for the default: report the decision per candidate, spawn nothing")
	asJSON := fs.Bool("json", false, "emit machine-readable JSON")
	ledger := fs.String("ledger", "", "loop JSONL ledger path (default: the loop ledger)")
	policyPath := fs.String("policy", "", "loop admission policy JSON path (default: the loop policy)")
	noLoopLedger := fs.Bool("no-loop-ledger", false, "skip recording this run in the loop ledger (hermetic probes)")
	if !parseFlags(fs, argv) {
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
		witnessGardenDispatch(ledgerPath, !*noLoopLedger, plan, "")
		return renderGardenDispatchResult(stdout, stderr, plan, *asJSON)
	}

	// Gate 1.5: bounded no-seat park-and-retry (#3523). When recent LIVE runs stopped on
	// a seat refuse (REFUSE_NO_ACCOUNT / WEEKLY_CAPPED — no free Claude seat), re-driving
	// another spawn attempt just bursts against a wall only a peer finishing can move, and
	// the burst's preflight load can turn the clean seat-refuse into a REFUSE_INSPECT. So
	// park until a bounded backoff window elapses, then retry, up to a bounded budget —
	// the durable loop ledger this bridge already writes IS the park-state store. A parked
	// run returns HERE, before the candidate load and before any preflight probe, so it
	// adds no load. Only the LIVE path parks (dry-run is inspection and always runs).
	if live {
		parkEvents, _ := loopmgr.Load(ledgerPath)
		parks, lastParkUnix := deriveSeatParkState(parkEvents)
		seat := seatpark.Decide(seatpark.Input{
			TaskID:       gardenDispatchLoopID,
			Parks:        parks,
			LastParkUnix: lastParkUnix,
			NowUnix:      time.Now().Unix(),
		})
		if !seat.ShouldAttempt() {
			plan.Verdict = string(seat.Status)
			plan.Reason = seat.Detail
			witnessGardenDispatch(ledgerPath, !*noLoopLedger, plan, string(seat.Status))
			return renderGardenDispatchResult(stdout, stderr, plan, *asJSON)
		}
	}

	items, _, err := loadGardenWalkIssues(*input, *repo, *state, *limit)
	if err != nil {
		fmt.Fprintf(stderr, "fak garden dispatch: %v\n", err)
		return 1
	}
	walkPlan := gardenbundle.PlanWalk("issue", items, gardenWalkPolicy(*budget, *skipFresh, *skipActive))
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

	// stopVerdict captures the fleet-wide verdict that ended the run (if any), so the
	// witness can record a seat refuse (#3523) as a NoSeat park in the durable ledger.
	stopVerdict := dispatchGardenCandidates(stderr, plan, candidates, laneByIssue, root, *backend, ledgerPath, *maxWorkers, live, !*noLoopLedger)

	classifyGardenDispatchOutcome(plan, live)

	// Record a NoSeat park in the ledger when a LIVE run actually stopped on a seat
	// refuse, so the next run's Gate-1.5 park-and-retry (#3523) counts it. A dry-run
	// or a non-seat stop records no park token (the derivation ends its tail there).
	seatReason := ""
	if live && gardenDispatchSeatRefuses[stopVerdict] {
		seatReason = seatParkReasonNoSeat
	}
	witnessGardenDispatch(ledgerPath, !*noLoopLedger, plan, seatReason)
	return renderGardenDispatchResult(stdout, stderr, plan, *asJSON)
}

// dispatchGardenCandidates routes each budgeted candidate to its lane and hands BOTH
// to evaluateDispatchTick, folding every decision into plan (results, counters, and
// skipped-by-reason buckets). It returns the fleet-wide capacity verdict that stopped
// the run early ("" when the run walked every candidate).
func dispatchGardenCandidates(stderr io.Writer, plan *gardenDispatchPlan, candidates []gardenbundle.WalkDecision, laneByIssue map[int]string, root, backend, ledgerPath string, maxWorkers int, live, recordLoop bool) string {
	stopVerdict := ""
	// Refresh the fleet registry (the ~40s tools/fleet_sessions.py scan) only on the
	// FIRST tick of the drain, then reuse it -- the same cadence sweep (iter==0) and
	// wave (i==0) already use. The scan's result is surfaced only as the payload's
	// registry_refresh provenance; seat admission reads live pidfile leases and the
	// on-disk roster directly, so re-scanning before every candidate added (N-1)x~40s
	// of serial wall-clock to a garden run that walks EVERY budgeted candidate (up to
	// ~13 min at the default --budget 20) without changing a single admission decision.
	registryRefreshed := false
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
			Workspace:  root,
			MaxWorkers: maxWorkers,
			WorkKind:   dispatchtickWorkKind(backend),
			// Garden dispatch IS project-management / repo-maintenance work (triage, dedup,
			// close/relabel of stale-dormant issues) — routine coordination where a wrong
			// call costs a re-run, not a bad production write. Default its workers to fable
			// as a LOW-precedence work-class pin: an operator --worker-model, a lane pin, or
			// an explicit tier/T0 label on a genuinely-hard garden issue still escalates.
			WorkClassModel: dispatchtick.WorkerModelFable,
			Lane:           lane,
			TargetIssue:    cand.ID,
			Backend:        backend,
			Live:           live,
			Refresh:        !registryRefreshed,
			CooldownMin:    dispatchtick.DefaultCooldownMinutes,
			WorkerTimeoutS: dispatchtick.DefaultWorkerTimeoutS,
			SpawnProbeS:    dispatchtick.DefaultSpawnProbeS,
			RecordLoop:     recordLoop,
			LoopLedger:     ledgerPath,
		}, stderr)
		// The registry is refreshed inside evaluateDispatchTick at the top of the tick
		// (before any error), so one Refresh:true tick is enough for the whole drain.
		registryRefreshed = true
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
			stopVerdict = verdict
			plan.StopReason = fmt.Sprintf("stopped after candidate #%d: %s is a fleet-wide boundary, every remaining candidate would refuse the same way", cand.ID, verdict)
			break
		}
	}
	return stopVerdict
}

// classifyGardenDispatchOutcome folds the run's counters into the plan's closing
// verdict and reason -- the same closed vocabulary the bridge has always emitted
// (NO_CANDIDATES / NONE_ADMITTED / APPLIED / PARTIAL / WOULD_APPLY).
func classifyGardenDispatchOutcome(plan *gardenDispatchPlan, live bool) {
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

// The skip buckets that carry a CONTENTION meaning. They are named constants rather
// than inline literals because witnessGardenDispatch turns each one into a ledger
// metric key (`skipped_<bucket>`, hyphens folded to underscores), so these two strings
// ARE the ledger's category vocabulary for contention — the thing #4321 asks to keep
// un-conflatable.
//
// Two different mechanisms both get called a "collision" in this launcher, and folding
// them into one bucket is what made ~231 working-tree refusals unreadable as a class in
// ledger analysis:
//
//   - gardenSkipLaneContended is LANE-LEASE contention: a live peer holds the lane's
//     fenced lease (or the same issue is already in flight, or the ordering layer saw a
//     COLLISION_RISK). It clears BY ITSELF when that peer finishes or the TTL lapses.
//     The correct response is to wait, or to pick a disjoint lane.
//   - gardenSkipWorktreeCotenancy is WORKING-TREE co-tenancy: uncommitted peer WIP sits
//     in the one shared checkout and NO live lease owns it. Nothing clears it but a
//     human/peer commit or revert; waiting is exactly the wrong move. The correct
//     response is commit-by-path landing, or the sanctioned detached per-worker worktree
//     (#1334 / epic #3165).
//
// Same word, opposite remedy, so they must never share a bucket.
const (
	gardenSkipLaneContended     = "contended"
	gardenSkipWorktreeCotenancy = "worktree-cotenancy"
)

// gardenDispatchCotenancyVerdicts is the CLOSED set of tick verdicts that mean
// working-tree co-tenancy. It is deliberately an explicit table, never a substring rule
// over "COLLISION"/"WIP": a future refusal code must be classified on purpose rather
// than inherit a class from its spelling — DIRTY_PATH_COLLISION and
// dispatchorder.ReasonCollisionRisk both contain "COLLISION" and belong to DIFFERENT
// classes, which is precisely how the two got conflated in the first place.
//
// This mirrors the producer's own vocabulary: tools/issue_resolve_dispatch.py stamps
// REFUSAL_CLASS_WORKTREE_COTENANCY on exactly these verdicts (its _REFUSAL_CLASSES
// table) and REFUSAL_CLASS_LANE_LEASE on LANE_LEASE_HELD. Go cannot import Python, so
// garden_dispatch_refusalclass_test.go parses that table back OUT of the Python source
// and fails when either side drifts — without that pin, a third co-tenancy code added
// on the producer side would silently fall through to the untyped "refused" bucket
// here, which is the bug this constant exists to close.
var gardenDispatchCotenancyVerdicts = map[string]bool{
	"DIRTY_PATH_COLLISION": true,
	"SAME_ISSUE_WIP":       true,
}

// gardenDispatchSkipReason buckets a refusing tick verdict into the typed reason
// vocabulary the acceptance criteria ask for (capped / no-seat / contended /
// worktree-cotenancy / ...), without inventing a second verdict space -- it is a pure
// re-label of the SAME verdict evaluateDispatchTick already returns.
func gardenDispatchSkipReason(verdict, action string) string {
	// Checked before the switch so the closed table above is the single source of
	// truth for this class -- a verdict added there is bucketed without editing here.
	if gardenDispatchCotenancyVerdicts[verdict] {
		return gardenSkipWorktreeCotenancy
	}
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
		return gardenSkipLaneContended
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
	// The garden bridge runs unattended; with no operator policy it inherits the
	// embedded sane default (the garden/default 12h floor + the global storm cap).
	policies, err := loopmgr.LoadPoliciesOrDefault(policyPath)
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
// reason (flattened to skipped_<reason> metrics, plus a total skipped_total). seatReason
// stamps the event's Reason with the #3523 park token (SEAT_NO_SEAT / SEAT_PARKED /
// SEAT_EXHAUSTED) so a later run's Gate-1.5 derivation reads the park tail off the
// durable ledger; "" leaves Reason unset (a tail boundary), exactly as before.
func witnessGardenDispatch(ledgerPath string, record bool, plan *gardenDispatchPlan, seatReason string) {
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
		Reason:  seatReason,
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
