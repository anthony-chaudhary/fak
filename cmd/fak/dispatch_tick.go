package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"runtime"
	"strconv"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/branchrole"
	"github.com/anthony-chaudhary/fak/internal/dispatchorder"
	"github.com/anthony-chaudhary/fak/internal/dispatchtick"
	"github.com/anthony-chaudhary/fak/internal/leaseref"
	"github.com/anthony-chaudhary/fak/internal/loopmgr"
	"github.com/anthony-chaudhary/fak/internal/regionadmit"
	"github.com/anthony-chaudhary/fak/internal/workerworktree"
)

type dispatchTickOptions struct {
	Workspace    string
	MaxWorkers   int
	WorkKind     string
	Lane         string
	TargetIssue  int
	LeaseID      string
	LeaseTree    []string
	Backend      string
	Goal         string
	GoalProfile  string
	ExcludeLanes []string
	Live         bool
	Refresh      bool
	PreferNewest bool
	Generation   string
	// View scopes the tick's issue routing to a named issue-view from
	// .github/issue-views.json (#1411). Empty disables the scoping; the CLI
	// flag defaults it to the operator-marked `current` board/milestone focus.
	View           string
	CooldownMin    int
	WorkerTimeoutS int
	SpawnProbeS    float64
	LoopLedger     string
	RecordLoop     bool
	// WorkerModel un-blanks the claude worker model to an explicit id (highest-precedence
	// model pin); empty falls back to the lane_models pin, the benchmark gate, else the
	// seat default. PinWorkerModel turns on the benchmark gate (a model-accounting run that
	// pins the account/default model so it measures a KNOWN model, not a silently-switched
	// one). Both default off, so a normal claude tick stays on the seat default + fallback.
	WorkerModel    string
	PinWorkerModel bool
	// WorkClassModel is a LOW-precedence worker-model default for the tick's work class:
	// `fak garden dispatch` (the project-management / repo-maintenance dispatcher) sets it
	// to fable so routine coordination work runs on the cheap model by default. It sits
	// BELOW the per-issue tier profile (so an explicit tier/pm or tier/T0 label still
	// decides — a hard planning issue still escalates to opus) and ABOVE the seat default.
	// Empty for a normal tick, so nothing changes unless a work-class dispatcher sets it.
	WorkClassModel string
	// ModelDowngrade turns on Layer-2 in-tick re-dispatch: when the target issue's last
	// finished slot exited with a model-switchable no-commit reason (usage_cap /
	// model_unknown / rate_limit), re-dispatch it on the NEXT downgrade-chain model instead
	// of the same walled one. Default off, so the live claude fleet is byte-identical.
	ModelDowngrade          bool
	CodexLoopGate           string
	CodexLoopGateSinceHours float64
	CodexLoopGateLimit      int
	// FocusHold flips the focus WIP-breadth term (#3223) from its default WARN posture to
	// HOLD: when the fleet is at/over the focusscore WIP cap, a spawn that OPENS a new
	// objective is refused (FOCUS_WIP_SATURATED) instead of merely advised. Continuation of
	// an already-open objective is never held. Default off (warn), so the fleet is
	// byte-identical until an operator opts in via --focus-hold / FLEET_DISPATCH_FOCUS_HOLD.
	FocusHold  bool
	Account    *dispatchtick.Account
	Membership *dispatchtick.Membership
}

type dispatchLanePick struct {
	Lane             string
	Numbers          []int
	ByLaneCount      map[string]int
	ByLaneStepBudget map[string]int
	ExcludedLanes    []string
	Tree             []string
	// PathsByIssue carries each open issue's DECLARED file scope (route.Paths) so
	// a single-target tick can narrow its claim to the target issue -- a per-issue
	// lease over just those paths -- exactly as the wave path already does. Empty
	// for an issue that declared no scope; such an issue keeps the coarse lane tree.
	PathsByIssue map[int][]string
	// IssueByNumber caches each routed issue's already-fetched row (title/body/labels,
	// State=OPEN) so dispatchPrompt builds the worker prompt from it instead of a second
	// `gh issue view` on the hot path (#4167). A cache miss (an unrouted --target-issue)
	// yields the zero value, which dispatchPrompt treats as "no cache" and falls back to
	// the live fetch.
	IssueByNumber      map[int]dispatchIssueInfo
	View               string
	ViewQuery          string
	ViewDigest         string
	ViewFallback       bool
	ViewFallbackReason string
	RouterError        string
	// SelfSourceHeld names the lanes the guarded auto-pick SKIPPED because their
	// tree is fak's own running source (cmd/** or internal/**). It is populated only
	// on an auto-pick (no explicit lane) under guard, and is the witness behind the
	// honest all-self-source surface (#1397): when it is non-empty AND Lane is "" the
	// backlog was not empty -- every eligible lane was held as self-source -- so the
	// tick reports SELF_MODIFY_HOLD over the held set, not a silent/empty NO_LANE.
	SelfSourceHeld []string
}

type dispatchSpawnResult struct {
	PID        int            `json:"pid"`
	Log        string         `json:"log"`
	Issue      int            `json:"issue"`
	Lane       string         `json:"lane"`
	Backend    string         `json:"backend"`
	LeaseID    string         `json:"lease_id,omitempty"`
	Tree       []string       `json:"tree,omitempty"`
	Startup    string         `json:"startup_bundle,omitempty"`
	Account    map[string]any `json:"account,omitempty"`
	Membership any            `json:"membership,omitempty"`
	EarlyExit  map[string]any `json:"early_exit,omitempty"`
}

const dispatchLeaseTreeSidecarSuffix = ".lease-tree.json"
const dispatchLeaseIDSidecarSuffix = ".lease-id"

// dispatchWorktreeSidecarSuffix records the per-worker git worktree (#3168) a slot
// ran in, so the async witness sweep can land its diff onto the trunk and reap the
// worktree once the pid is provably dead — without trusting any live in-process map.
// Absent for a worker that ran in the shared trunk (isolation off / prepare failed).
const dispatchWorktreeSidecarSuffix = ".worktree"
const dispatchStartupBundleSidecarSuffix = ".startup.json"
const dispatchStartupBundleSchema = "fleet-worker-startup-bundle/1"
const dispatchCodexLoopGateDefaultSinceHours = 24
const dispatchCodexLoopGateDefaultLimit = 20

var dispatchResolvePIDRE = regexp.MustCompile(`^(?:resolve|repair)-\d+-\d{8}-\d{6}\.pid$`)
var dispatchGoalPIDRE = regexp.MustCompile(`^.+-\d{8}-\d{6}\.pid$`)

func runDispatchTick(stdout, stderr io.Writer, argv []string) int {
	opts, asJSON, code := parseDispatchTickFlags(stderr, argv)
	if code != 0 {
		return code
	}
	payload, err := evaluateDispatchTick(opts, stderr)
	if err != nil {
		fmt.Fprintf(stderr, "fak dispatch tick: %v\n", err)
		return 1
	}
	if asJSON {
		if err := writeIndentedJSON(stdout, payload); err != nil {
			fmt.Fprintf(stderr, "fak dispatch tick: encode json: %v\n", err)
			return 1
		}
	} else {
		fmt.Fprint(stdout, renderDispatchTick(payload))
	}
	return dispatchTickExitCode(payload)
}

var dispatchTickBenignActions = map[string]bool{
	"spawned":                 true,
	"would_spawn":             true,
	"would_enroll":            true,
	"enrolled":                true,
	"refused":                 true,
	"no_lane":                 true,
	"no_issue":                true,
	"self_modify_hold":        true,
	"in_flight_duplicate":     true,
	"collision_risk":          true,
	"lane_busy":               true,
	"lane_leased":             true,
	"broker_denied":           true,
	"codex_loop_gate_refused": true,
	"focus_hold":              true,
}

func dispatchTickExitCode(payload map[string]any) int {
	if payload == nil {
		return 1
	}
	action := dispatchMapString(payload, "action")
	if dispatchTickBenignActions[action] {
		return 0
	}
	return 1
}

func parseDispatchTickFlags(stderr io.Writer, argv []string) (dispatchTickOptions, bool, int) {
	fs := flag.NewFlagSet("dispatch tick", flag.ContinueOnError)
	fs.SetOutput(stderr)
	workspace := fs.String("workspace", "", "workspace root (default: current directory)")
	maxWorkers := fs.Int("max-workers", dispatchtick.DefaultMaxWorkers, "hard cap on live workers, enforced by dispatch preflight")
	workKind := fs.String("work-kind", "", "switcher work kind (default follows --backend)")
	lane := fs.String("lane", "", "explicit lane (default: lane with the largest routed step budget)")
	targetIssue := fs.Int("target-issue", 0, "explicit issue number for the selected lane")
	leaseID := fs.String("lease-id", "", "explicit lane/issue lease id")
	leaseTree := fs.String("lease-tree", "", "comma-separated lease tree globs for the explicit lease")
	backend := fs.String("backend", firstString(strings.TrimSpace(os.Getenv("FLEET_WORKER_BACKEND")), "claude"), "worker backend (claude|opencode|codex|micro); micro (#2030, opt-in) enrolls the routed lane into the in-process microagent host instead of a detached CLI — default follows $FLEET_WORKER_BACKEND, else claude")
	goal := fs.String("goal", "", "durable dispatch loop goal id (for example throughput or high-priority); known goal ids also select the default --goal-profile")
	goalProfile := fs.String("goal-profile", "", "dispatch picker profile: throughput|high-priority (default follows --goal, else throughput)")
	excludeLane := fs.String("exclude-lane", "", "comma-separated lanes to drop from the busiest pick")
	live := fs.Bool("live", false, "actually spawn the issue-resolution worker")
	noRefresh := fs.Bool("no-refresh", false, "skip the per-tick account-registry refresh")
	preferNewest := fs.Bool("prefer-newest", false, "pick the NEWEST open issue on the lane first (default: oldest first, to drain the backlog)")
	generationFlag := fs.String("generation", "", "generation horizon to admit: now|next|second-next|future|all (default: now+next; only engages when a candidate carries a gen/* label)")
	view := fs.String("view", dispatchDefaultView, "scope issue routing to this named issue-view from .github/issue-views.json (empty disables; an unresolvable or empty view fail-softs to the full open backlog)")
	cooldownMin := fs.Int("cooldown-min", dispatchtick.DefaultCooldownMinutes, "skip issues attempted within this many minutes (0 disables)")
	workerTimeoutS := fs.Int("worker-timeout-s", dispatchtick.DefaultWorkerTimeoutS, "worker lease TTL base in seconds (0 uses default)")
	spawnProbeS := fs.Float64("spawn-probe-s", dispatchtick.DefaultSpawnProbeS, "seconds to wait after spawn to catch immediate empty-log exits")
	loopLedger := fs.String("loop-ledger", "", "append this tick to a fak loop ledger (default: FAK_LOOP_LEDGER or .fak/loops.jsonl)")
	noLoopLedger := fs.Bool("no-loop-ledger", false, "disable loop-ledger append for this tick")
	workerModel := fs.String("worker-model", "", "pin the claude worker to this exact --model id (un-blanks the seat default; empty falls back to the lane_models pin/benchmark gate/seat default)")
	pinWorkerModel := fs.Bool("pin-worker-model", false, "benchmark gate: pin the claude worker to the account/default model (model-accounting run) instead of the seat default + fallback chain")
	modelDowngrade := fs.Bool("model-downgrade", false, "Layer-2 in-tick re-dispatch: when the target's last slot exited model-switchable (usage_cap/model_unknown/rate_limit), re-dispatch it on the next downgrade-chain model")
	focusHold := fs.Bool("focus-hold", false, "focus WIP backpressure (#3223): HOLD (refuse) a spawn that OPENS a new objective while the fleet is at/over the focusscore WIP cap, instead of the default WARN (advise + still spawn); continuation of an already-open objective is never held ($FLEET_DISPATCH_FOCUS_HOLD also enables)")
	codexLoopGate := fs.String("codex-loop-gate", dispatchCodexLoopGateDefaultThreshold(), "for live Codex workers, audit recent Codex sessions before spawn and refuse at threshold: loop|action|off (default: $FLEET_CODEX_LOOP_GATE or loop)")
	codexLoopGateSinceHours := fs.Float64("codex-loop-gate-since-hours", dispatchCodexLoopGateDefaultSinceHoursValue(), "with --codex-loop-gate, only scan Codex sessions modified within N hours (0 = all)")
	codexLoopGateLimit := fs.Int("codex-loop-gate-limit", dispatchCodexLoopGateDefaultLimitValue(), "with --codex-loop-gate, maximum newest Codex sessions to scan")
	asJSON := fs.Bool("json", false, "emit machine-readable JSON")

	accountTag := fs.String("account-tag", "", "internal: forced account tag (used by dispatch wave)")
	accountTier := fs.String("account-tier", "", "internal: forced account tier (used by dispatch wave)")
	accountModel := fs.String("account-model", "", "internal: forced account model (used by dispatch wave)")
	accountDir := fs.String("account-dir", "", "internal: forced account config dir (used by dispatch wave)")
	waveID := fs.String("wave-id", "", "internal: wave id sidecar")
	waveRank := fs.Int("wave-rank", -1, "internal: wave rank sidecar")
	waveSize := fs.Int("wave-size", 0, "internal: wave size sidecar")
	waveShortfall := fs.Int("wave-shortfall", 0, "internal: wave shortfall sidecar")
	if err := fs.Parse(argv); err != nil {
		return dispatchTickOptions{}, false, 2
	}

	root := *workspace
	if root == "" {
		wd, err := os.Getwd()
		if err != nil {
			fmt.Fprintf(stderr, "fak dispatch tick: getwd: %v\n", err)
			return dispatchTickOptions{}, false, 1
		}
		root = wd
	}
	b, err := dispatchtick.NormalizeBackend(*backend)
	if err != nil {
		fmt.Fprintf(stderr, "fak dispatch tick: %v\n", err)
		return dispatchTickOptions{}, false, 2
	}
	wk := strings.TrimSpace(*workKind)
	if wk == "" {
		wk = dispatchtick.DefaultWorkKind(b)
	}
	goalID, profile, goalErr := normalizeDispatchGoal(*goal, *goalProfile)
	if goalErr != nil {
		fmt.Fprintf(stderr, "fak dispatch tick: %v\n", goalErr)
		return dispatchTickOptions{}, false, 2
	}
	opts := dispatchTickOptions{
		Workspace:               root,
		MaxWorkers:              *maxWorkers,
		WorkKind:                wk,
		Lane:                    strings.TrimSpace(*lane),
		TargetIssue:             *targetIssue,
		LeaseID:                 strings.TrimSpace(*leaseID),
		LeaseTree:               dispatchSplitCSV(*leaseTree),
		Backend:                 b,
		Goal:                    goalID,
		GoalProfile:             profile,
		ExcludeLanes:            dispatchSplitCSV(*excludeLane),
		Live:                    *live,
		Refresh:                 !*noRefresh,
		PreferNewest:            *preferNewest,
		Generation:              strings.TrimSpace(*generationFlag),
		View:                    strings.TrimSpace(*view),
		CooldownMin:             *cooldownMin,
		WorkerTimeoutS:          *workerTimeoutS,
		SpawnProbeS:             maxFloat64(0, *spawnProbeS),
		LoopLedger:              *loopLedger,
		RecordLoop:              !*noLoopLedger,
		WorkerModel:             firstString(strings.TrimSpace(*workerModel), strings.TrimSpace(os.Getenv("FLEET_DISPATCH_WORKER_MODEL"))),
		PinWorkerModel:          *pinWorkerModel || dispatchBoolValue(os.Getenv("FLEET_DISPATCH_PIN_MODEL")),
		ModelDowngrade:          *modelDowngrade || dispatchBoolValue(os.Getenv("FLEET_DISPATCH_MODEL_DOWNGRADE")),
		FocusHold:               *focusHold || dispatchBoolValue(os.Getenv("FLEET_DISPATCH_FOCUS_HOLD")),
		CodexLoopGate:           strings.TrimSpace(*codexLoopGate),
		CodexLoopGateSinceHours: maxFloat64(0, *codexLoopGateSinceHours),
		CodexLoopGateLimit:      *codexLoopGateLimit,
	}
	if *accountTag != "" || *accountTier != "" || *accountModel != "" || *accountDir != "" {
		opts.Account = &dispatchtick.Account{
			Tag:   *accountTag,
			Tier:  parseAccountTier(*accountTier),
			Model: *accountModel,
			Dir:   *accountDir,
		}
	}
	if *waveID != "" {
		opts.Membership = &dispatchtick.Membership{
			Rank:      *waveRank,
			WaveID:    *waveID,
			Size:      *waveSize,
			Shortfall: *waveShortfall,
		}
	}
	return opts, *asJSON, 0
}

// dispatchTickPick bundles the resolved candidate lane and target issue for a tick
// together with the live/cooldown/held state that fed the decision, so
// evaluateDispatchTick can unpack them to the same local names it used before the
// resolveDispatchTickPick extraction.
type dispatchTickPick struct {
	pick             dispatchLanePick
	held             map[string]bool
	liveIssueDetails map[int]dispatchLiveScope
	// liveScopes is this tick's captured live-worker set (from the same runs-directory
	// snapshot that fed liveIssueDetails), reused for the per-pick tree-collision gate in
	// evaluateDispatchTick so that check projects off captured state instead of re-scanning.
	liveScopes     []dispatchLiveScope
	liveIssues     map[int]bool
	cooled         map[int]bool
	cooldownStatus []map[string]any
	target         int
	hasTarget      bool
	// leaseID is this tick's resolved lease id: the per-issue id
	// (resolve-<lane>-<issue>) when the claim narrowed to the target, else the
	// coarse lane id (resolve-<lane>) or an operator --lease-id pin. pick.Tree is
	// narrowed in lock-step, so every downstream gate (tree-collision, self-modify,
	// the lease acquire, the sidecars) sees the same id+tree.
	leaseID string
}

// resolveDispatchTickPick computes this tick's candidate lane and target issue under
// the live/cooldown/structurally-held skip set. Pure code motion out of
// evaluateDispatchTick; behavior is unchanged.
func resolveDispatchTickPick(root string, stderr io.Writer, opts dispatchTickOptions, runsDir string, heldNoCommit map[int]bool) (dispatchTickPick, error) {
	// One runs-directory scan feeds every live/cooldown/collision view this tick needs
	// (held lanes, live issue details, cooldown set + rows, and the per-pick tree-collision
	// gate below), instead of re-globbing/re-statting the sidecars once per view (#3593).
	snap := scanRunsSnapshot(runsDir, time.Now())
	scopes := snap.liveScopes()
	held := snap.liveLanes()
	exclude := map[string]bool{}
	for _, lane := range opts.ExcludeLanes {
		exclude[lane] = true
	}
	if opts.Lane == "" {
		for lane := range held {
			exclude[lane] = true
		}
		// #4285: also soft-exclude lanes the #2062 low-yield fold flagged (recent
		// finished sessions burned turns yet closed nothing). Auto-pick only -- an
		// explicit --lane still overrides. Fail-open (nil on any probe error).
		for lane := range dispatchLowYieldExcludes(root) {
			exclude[lane] = true
		}
	}
	pick, err := pickDispatchLane(root, stderr, opts.Lane, exclude, opts.PreferNewest, opts.Generation, opts.GoalProfile, opts.TargetIssue)
	if err != nil {
		return dispatchTickPick{}, err
	}
	liveIssueDetails := snap.liveIssueDetails()
	liveIssues := liveIssueSet(liveIssueDetails)
	cooled := snap.recentlyAttempted(opts.CooldownMin)
	cooldownStatus := snap.cooldownRowMaps(opts.CooldownMin)
	skip := map[int]bool{}
	for n := range liveIssues {
		skip[n] = true
	}
	for n := range cooled {
		skip[n] = true
	}
	// The pick-held-invariant rung (#1396): an issue whose last finished worker was
	// STRUCTURALLY guard-blocked (self_modify / policy_block) re-blocks identically
	// on re-dispatch, so hold it this tick instead of re-storming the same
	// un-landable drain. An auth wall is deliberately NOT held here — it re-probes
	// after the time cooldown window.
	for n := range heldNoCommit {
		skip[n] = true
	}
	target, hasTarget := dispatchtick.PickTargetIssue(pick.Numbers, skip)
	if opts.TargetIssue > 0 {
		target, hasTarget = opts.TargetIssue, true
		if liveIssues[target] || cooled[target] {
			hasTarget = false
		}
	}
	if len(opts.LeaseTree) > 0 {
		pick.Tree = append([]string(nil), opts.LeaseTree...)
	}
	// Per-issue claim narrowing (parity with the wave path). When the chosen target
	// declares its OWN file scope (route.Paths) and the operator pinned neither the
	// lease id nor the lease tree, narrow this tick's whole claim to that issue: a
	// per-issue lease id over the declared tree. pick.Tree is mutated here so EVERY
	// downstream gate -- the pre-acquire tree-collision check, the self-modify hold,
	// the lease acquire, and the sidecars -- adjudicates on the narrowed tree, so two
	// issues of one lane with disjoint declared scopes co-run (regionadmit admits
	// disjoint sub-regions) instead of colliding on the coarse lane tree. With no
	// declared scope, or an operator pin, the claim stays the coarse lane (byte-
	// identical to before this seam).
	leaseID := firstString(opts.LeaseID, dispatchLaneLeaseID(pick.Lane))
	if hasTarget && opts.LeaseID == "" && len(opts.LeaseTree) == 0 {
		if paths := pick.PathsByIssue[target]; len(paths) > 0 {
			pick.Tree = append([]string(nil), paths...)
			leaseID = dispatchIssueLeaseID(pick.Lane, target)
		}
	}
	return dispatchTickPick{
		pick:             pick,
		held:             held,
		liveIssueDetails: liveIssueDetails,
		liveScopes:       scopes,
		liveIssues:       liveIssues,
		cooled:           cooled,
		cooldownStatus:   cooldownStatus,
		target:           target,
		hasTarget:        hasTarget,
		leaseID:          leaseID,
	}, nil
}

// seedDispatchTickPayload builds the base tick payload (including the startup bundle)
// shared by every downstream verdict branch. Pure code motion out of
// evaluateDispatchTick; behavior is unchanged.
func seedDispatchTickPayload(root string, opts dispatchTickOptions, reg, pre map[string]any, account dispatchtick.Account, pr dispatchTickPick) map[string]any {
	startup := dispatchStartupBundle(root, opts, pre, account, pr.pick, pr.leaseID, pr.target, pr.hasTarget, pr.held, pr.liveIssues, pr.cooled, pr.cooldownStatus)
	preflight := map[string]any{
		"verdict":   dispatchMapString(pre, "verdict"),
		"reason":    dispatchMapString(pre, "reason"),
		"cap":       pre["cap"],
		"live":      pre["live"],
		"cap_terms": mapAt(pre, "cap_terms"),
	}
	payload := map[string]any{
		"schema":               dispatchtick.Schema,
		"workspace":            root,
		"live":                 opts.Live,
		"backend":              opts.Backend,
		"goal":                 opts.Goal,
		"goal_profile":         opts.GoalProfile,
		"max_workers":          opts.MaxWorkers,
		"view":                 pr.pick.View,
		"view_query":           pr.pick.ViewQuery,
		"view_digest":          pr.pick.ViewDigest,
		"view_fallback":        pr.pick.ViewFallback,
		"view_fallback_reason": pr.pick.ViewFallbackReason,
		"registry_refresh":     reg,
		"preflight":            preflight,
		"account":              dispatchtick.AccountSidecar(account),
		"lane":                 pr.pick.Lane,
		"lease_id":             pr.leaseID,
		"lease_tree":           append([]string(nil), pr.pick.Tree...),
		"lane_issue_count":     len(pr.pick.Numbers),
		"lane_step_budget":     pr.pick.ByLaneStepBudget[pr.pick.Lane],
		"cooled_recently":      sortedSet(pr.cooled),
		"cooldown_status":      pr.cooldownStatus,
		"target_issue":         nil,
		"already_live":         sortedSet(pr.liveIssues),
		"held_lanes":           sortedStringSet(pr.held),
		"startup_bundle":       startup,
		"stale_base":           mapAt(startup, "stale_base"),
	}
	if strings.TrimSpace(pr.pick.View) == "" {
		delete(payload, "view")
		delete(payload, "view_query")
		delete(payload, "view_digest")
		delete(payload, "view_fallback")
		delete(payload, "view_fallback_reason")
	}
	return payload
}

// prepareDispatchWorkerCommand resolves the worker model policy and builds the guarded
// worker command preview, mutating payload with the resolved command/model surface.
// Pure code motion out of evaluateDispatchTick (the CLI-only path after the micro-backend
// enroll); behavior is unchanged.
func prepareDispatchWorkerCommand(root string, opts dispatchTickOptions, pick dispatchLanePick, account dispatchtick.Account, target, promptChars int, labels []string, witnessRecords []dispatchtick.WitnessRecord, payload map[string]any) (launch dispatchtick.WorkerLaunch, launchPreview []string, guardedPreview bool, err error) {
	// Opt-in per-issue tier launch profile (FLEET_TIER_LAUNCH): the target's trusted tier
	// labels resolve to a {model, effort, ultracode} profile handed to the resolver as its
	// lowest-precedence un-blanking source. A coordination work kind (project_management /
	// gardening) additionally routes an UNLABELLED issue to the cheap PM bucket, so a PM
	// dispatch loop runs on fable by default without a per-issue tier/pm label. Nil (knob
	// off / non-claude / untagged on an implementation tick) keeps the seat default, so a
	// default fleet tick is byte-identical to before this seam.
	tierProfile, tierBucket := dispatchTierLaunchProfile(opts.Backend, labels, opts.WorkKind)
	modelPolicy := resolveWorkerModelPolicy(opts.Backend, pick.Lane, opts.WorkerModel, account, dispatchTickPolicy(root), opts.PinWorkerModel, tierProfile, opts.WorkClassModel)
	// Preventive placement gate (#3521). The tier table is DIFFICULTY-driven and happily puts
	// the cheap model on the MOST churning bucket (BucketUltra -> fable+ultracode). Before
	// launch, refuse a (work-shape × model-reliability) pairing the model cannot hold and
	// re-route it, instead of letting the worker starve on restart-amnesia and then blaming
	// the model. An untagged issue yields ShapeUnknown, so the gate is inert and a default
	// fleet tick is byte-identical; operator pins are never gated.
	shape := dispatchtick.WorkShapeForIssue(labels)
	if gp, fired := applyPlacementGate(modelPolicy, shape); fired {
		payload["placement_gate"] = map[string]any{
			"issue":  target,
			"shape":  string(shape),
			"from":   modelPolicy.Model,
			"to":     gp.Model,
			"reason": gp.PlacementReason,
		}
		modelPolicy = gp
	}
	// Layer 2: if the target's last slot walled on a model-switchable reason this tick,
	// re-dispatch it on the next downgrade-chain model instead of the resolved one. Live +
	// --model-downgrade only, so the default fleet is unaffected. A model wall is transient and
	// model-scoped, so the tier's effort/ultracode reasoning posture is carried across the
	// switch rather than stripped along with the walled model.
	if opts.Live && opts.ModelDowngrade {
		if dp, fired := applyModelDowngrade(opts.Backend, target, witnessRecords); fired {
			dp.Effort, dp.Ultracode = modelPolicy.Effort, modelPolicy.Ultracode
			modelPolicy = dp
			payload["model_downgrade"] = map[string]any{"issue": target, "model": dp.Model}
		}
	}
	fallbackModel := dispatchWorkerFallbackModel(opts.Backend)
	launch = dispatchtick.WorkerLaunch{
		Model:     modelPolicy.Model,
		Fallback:  fallbackModel,
		Effort:    modelPolicy.Effort,
		Ultracode: modelPolicy.Ultracode,
	}
	preview, err := dispatchtick.BuildWorkerCommand(opts.Backend, dispatchtick.PreviewPrompt(target, promptChars), launch)
	if err != nil {
		return dispatchtick.WorkerLaunch{}, nil, false, err
	}
	if fallbackModel != "" {
		payload["worker_fallback_model"] = fallbackModel
	}
	// Surface the resolved model only when the resolver UN-BLANKED it (an explicit/lane pin,
	// the benchmark gate, or a tier profile). A default claude tick keeps the seat default, so
	// its payload stays byte-identical to before this seam. dispatchWorkerModelMap folds in the
	// effort/ultracode knobs when the tier profile set them.
	if modelPolicy.pinned() {
		payload["worker_model"] = dispatchWorkerModelMap(modelPolicy)
	}
	// Attribute the uplift to the issue's tier bucket when the tier profile actually decided
	// the model (it can be out-competed by an operator pin or the bench gate above).
	if modelPolicy.Source == modelSourceTier && tierBucket != "" {
		payload["worker_tier"] = string(tierBucket)
	}
	launchPreview, guardedPreview = guardedDispatchCommand(root, pick.Lane, opts.Backend, preview)
	payload["command"] = dispatchtick.LaunchCommandShape(preview, root, account)
	payload["launch_command"] = dispatchtick.LaunchCommandShape(launchPreview, root, account)
	payload["guarded"] = guardedPreview
	return launch, launchPreview, guardedPreview, nil
}

// dispatchStampMs records the wall-clock elapsed since start into m under name, in
// integer milliseconds -- the shared stopwatch for evaluateDispatchTick's per-phase
// timing map (payload["timings_ms"]). Kept dependency-free (no new package) so any
// hot path can attribute its own cost the same way.
func dispatchStampMs(m map[string]int64, name string, start time.Time) {
	m[name] = time.Since(start).Milliseconds()
}

func evaluateDispatchTick(opts dispatchTickOptions, stderr io.Writer) (map[string]any, error) {
	root, err := filepath.Abs(opts.Workspace)
	if err != nil {
		return nil, err
	}
	runsDir := filepath.Join(root, dispatchtick.RunsDirName)
	if opts.WorkerTimeoutS <= 0 {
		opts.WorkerTimeoutS = dispatchtick.DefaultWorkerTimeoutS
	}
	// #1411: scope this tick's native issue routing to the named view. The CLI
	// flag defaults View to `current`; a programmatic tick (wave/sweep/garden)
	// that leaves View empty keeps today's full-backlog behavior.
	dispatchTickView = opts.View

	// Per-phase wall-clock attribution (observability): a slow tick is otherwise a
	// black box -- the dominant cost (the ~40s fleet_sessions.py registry scan) and
	// the per-tick subprocess fan-out (preflight PowerShell probes, router gh/dos
	// spawns) were only ever asserted in comments, never witnessed. `timings` holds
	// int64-millisecond durations, keyed per phase; a phase that did not run this
	// tick simply has no key (mirroring the omitempty provenance sub-maps). It is
	// attached as payload["timings_ms"] and stamped with `total` inside `finish`
	// (the single funnel every verdict return passes through), and folded into the
	// loop-ledger metrics under *_ms names by recordDispatchTickLoop so cross-tick
	// per-phase percentiles become a later fold. Purely additive: no decision reads it.
	t0 := time.Now()
	timings := map[string]int64{}
	var spawnStart time.Time

	reg := map[string]any{"skipped": true}
	if opts.Refresh {
		tReg := time.Now()
		reg = dispatchRefreshRegistry(root, stderr)
		dispatchStampMs(timings, "registry_refresh", tReg)
	}

	// Commit-time diff-witness binding (#1324 proposal #2), ported from the Python
	// dispatcher: grade each finished (dead-pid) worker's slot through `dos
	// commit-audit` and record the verdict in a .witness sidecar, so a bare `exit 0`
	// never silently counts as productive. Live ticks only (the audit + the sidecar
	// write are the side effects, and a dry run must stay byte-identical); fail-open.
	// The re-blockable guard refusals it surfaces (self_modify / policy_block) feed
	// the pick's hold set below (#1396).
	witnessedSlots := map[string]any{"skipped": true}
	heldNoCommit := map[int]bool{}
	var witnessRecords []dispatchtick.WitnessRecord
	if opts.Live {
		tWitness := time.Now()
		witnessedSlots, witnessRecords = witnessExitedWorkers(root, runsDir, true)
		heldNoCommit = dispatchtick.HeldNoCommitIssues(witnessRecords)
		dispatchStampMs(timings, "witness", tWitness)
	}

	tPreflight := time.Now()
	pre, preflightTimings, err := dispatchPreflightTimed(root, stderr, opts.MaxWorkers, opts.WorkKind, dispatchtick.ProductForBackend(opts.Backend))
	if err != nil {
		return nil, err
	}
	dispatchStampMs(timings, "preflight", tPreflight)
	for name, ms := range preflightTimings {
		timings["preflight_"+name] = ms
	}
	preOK := dispatchMapString(pre, "verdict") == "SPAWN_OK"
	account := accountFromMap(mapAt(pre, "account"))
	if opts.Account != nil {
		account = *opts.Account
	}

	tPick := time.Now()
	pickRes, err := resolveDispatchTickPick(root, stderr, opts, runsDir, heldNoCommit)
	if err != nil {
		return nil, err
	}
	dispatchStampMs(timings, "lane_pick", tPick)
	pick := pickRes.pick
	held := pickRes.held
	liveIssueDetails := pickRes.liveIssueDetails
	liveScopes := pickRes.liveScopes
	target := pickRes.target
	hasTarget := pickRes.hasTarget

	tSeed := time.Now()
	payload := seedDispatchTickPayload(root, opts, reg, pre, account, pickRes)
	dispatchStampMs(timings, "startup_bundle", tSeed)
	if hasTarget {
		payload["target_issue"] = target
	}
	// Surface the slot witness only on a live tick where the sweep ran, and the
	// structurally-held issues only when something is actually held, so the common
	// dry-run / nothing-held payloads stay byte-identical to before (#1396).
	if opts.Live {
		payload["witnessed_slots"] = witnessedSlots
	}
	if len(heldNoCommit) > 0 {
		payload["held_no_commit"] = sortedSet(heldNoCommit)
	}

	finish := func(p map[string]any) map[string]any {
		// The spawn phase (dispatchTickLiveSpawn / dispatchTickHostEnroll) calls finish
		// internally, so its duration is stamped here from the start captured just before
		// that tail call -- only on the live-spawn path where spawnStart was set.
		if !spawnStart.IsZero() {
			dispatchStampMs(timings, "spawn", spawnStart)
		}
		timings["total"] = time.Since(t0).Milliseconds()
		p["timings_ms"] = timings
		if opts.RecordLoop {
			p["loop_ledger"] = recordDispatchTickLoop(root, opts.LoopLedger, p)
		}
		return p
	}

	if !preOK {
		payload["ok"] = false
		payload["action"] = "refused"
		payload["verdict"] = firstString(dispatchMapString(pre, "verdict"), "REFUSE")
		payload["reason"] = "preflight refused: " + dispatchMapString(pre, "reason")
		// #3109 self-heal: if the refusal is (partly) driven by unattributed_live orphans,
		// preflight surfaced their exact PIDs (dispatch marker + no live lease). Reap them
		// here -- refuse THIS tick, but tree-kill the poison so the NEXT tick's count
		// recovers instead of staying wedged until a separately-scheduled janitor runs.
		// Live ticks only: a dry run stays observation-only (worklist surfaced, no kill).
		if worklist, ok := pre["janitor_worklist"].([]int); ok && len(worklist) > 0 {
			payload["janitor_worklist"] = worklist
			if opts.Live {
				payload["janitor_reaped"] = dispatchReapWorklist(worklist)
			}
		}
		return finish(payload), nil
	}
	if pick.Lane == "" {
		// All-self-source edge case (#1397): the auto-pick found candidate lanes but
		// every one was held as the trust-critical witness machinery (the adjudicator,
		// policy, kernel, shipgate/architest gates -- the referee's own trees) under
		// guard, so `chosen` stayed "". This is NOT an empty/error router -- the backlog
		// is real, it is just all the one narrow set a self-guarded RSI worker must never
		// SHIP an edit to (rewriting its own referee). Say so honestly with the
		// SELF_MODIFY_HOLD vocabulary (over the whole held set) instead of the misleading
		// "router empty/error" NO_LANE, so the operator routes the work to an unguarded/
		// operator or worktree-isolated path (#1334). (Merely-self-source lanes -- gateway,
		// agent, cmd, ... -- are NOT held: the worker guard permits shipping those.)
		if len(pick.SelfSourceHeld) > 0 {
			payload["ok"] = false
			payload["action"] = "self_modify_hold"
			payload["verdict"] = "SELF_MODIFY_HOLD"
			payload["self_modify_held_lanes"] = append([]string(nil), pick.SelfSourceHeld...)
			payload["reason"] = fmt.Sprintf("every candidate lane (%s) is rooted in fak's trust-critical witness machinery (the adjudicator/policy/kernel/shipgate the referee binds to): a guarded %s worker can investigate but must never SHIP an edit that rewrites its own referee (reason=SELF_MODIFY), so this narrow set is operator-gated -- route it to an unguarded/operator or worktree-isolated path (#1334), not a self-guarded worker", strings.Join(pick.SelfSourceHeld, ", "), opts.Backend)
			return finish(payload), nil
		}
		payload["ok"] = false
		payload["action"] = "no_lane"
		payload["verdict"] = "NO_LANE"
		payload["reason"] = "no lane has open issues (router empty/error)"
		return finish(payload), nil
	}
	if live, ok := inFlightDuplicateForPick(opts, pick.Numbers, hasTarget, liveIssueDetails); ok {
		payload["ok"] = false
		payload["action"] = "in_flight_duplicate"
		payload["verdict"] = "IN_FLIGHT_DUPLICATE"
		payload["target_issue"] = live.Issue
		payload["in_flight_duplicate"] = dispatchLiveScopeMap(live)
		payload["reason"] = fmt.Sprintf("issue #%d already has live worker %s (pid %d, lease %q)", live.Issue, live.Worker, live.PID, live.LeaseID)
		return finish(payload), nil
	}
	if hasTarget {
		if live, ok := treeCollisionFromScopes(liveScopes, pick.Tree); ok {
			payload["ok"] = false
			payload["action"] = "collision_risk"
			payload["verdict"] = dispatchorder.ReasonCollisionRisk
			payload["live_collision"] = map[string]any{
				"issue": live.Issue,
				"lane":  live.Lane,
				"tree":  append([]string(nil), live.Tree...),
				"log":   live.Log,
			}
			payload["reason"] = fmt.Sprintf("candidate issue #%d tree %v overlaps live worker issue #%d lane %q tree %v", target, pick.Tree, live.Issue, live.Lane, live.Tree)
			return finish(payload), nil
		}
	}
	if opts.Lane != "" && held[pick.Lane] && opts.TargetIssue == 0 {
		payload["ok"] = false
		payload["action"] = "lane_busy"
		payload["verdict"] = "LANE_BUSY"
		payload["reason"] = fmt.Sprintf("lane %q already has a live resolution worker", pick.Lane)
		return finish(payload), nil
	}
	if !hasTarget {
		payload["ok"] = false
		payload["action"] = "no_issue"
		payload["verdict"] = "NO_ISSUE"
		reason := fmt.Sprintf("every open issue on lane %q is live or cooling", pick.Lane)
		if len(heldNoCommit) > 0 {
			reason = fmt.Sprintf("every open issue on lane %q is live, cooling, or held after a structural guard refusal (held: %v)", pick.Lane, sortedSet(heldNoCommit))
		}
		payload["reason"] = reason
		return finish(payload), nil
	}

	tPrompt := time.Now()
	// #4167: hand dispatchPrompt the router-fetched row for the selected target so it
	// reuses the already-fetched body instead of a second `gh issue view`. A cache miss
	// (unrouted --target-issue) yields the zero value, and dispatchPrompt falls back.
	promptRec, err := dispatchPrompt(root, stderr, target, pick.Lane, pick.IssueByNumber[target])
	if err != nil {
		return nil, err
	}
	dispatchStampMs(timings, "prompt", tPrompt)
	promptChars := dispatchMapInt(promptRec, "prompt_chars")
	labels := dispatchStringSlice(promptRec["labels"])
	payload["prompt_chars"] = promptChars
	payload["issue_title"] = dispatchMapString(promptRec, "title")
	payload["development_branch"] = dispatchMapString(promptRec, "development_branch")
	if errText := dispatchMapString(promptRec, "branch_role_error"); errText != "" {
		payload["branch_role_error"] = errText
	}
	if warning := dispatchMapString(mapAt(payload, "stale_base"), "warning"); warning != "" {
		prompt := dispatchMapString(promptRec, "prompt") + "\n\nworker preflight warning:\n- " + warning + "\n"
		promptRec["prompt"] = prompt
		promptRec["prompt_chars"] = len(prompt)
		payload["worker_preflight_warning"] = warning
		promptChars = len(prompt)
		payload["prompt_chars"] = promptChars
	}
	// #2030 gen/second-next: the micro backend enrolls this routed issue into the
	// in-process microagent host (internal/microagent, M2) instead of exec-spawning a
	// detached guarded CLI. It runs AFTER the shared duplicate/collision/lane-busy/
	// no-issue gates (so tree-safety is decided identically) and BEFORE the CLI-only
	// model/guard/command machinery below (BuildWorkerCommand refuses micro). Opt-in
	// only — a default claude/opencode/codex tick never enters here.
	if dispatchtick.IsMicroBackend(opts.Backend) {
		spawnStart = time.Now()
		return dispatchTickHostEnroll(root, runsDir, opts, pick, pickRes.leaseID, account, target, payload, finish), nil
	}
	launch, launchPreview, guardedPreview, err := prepareDispatchWorkerCommand(root, opts, pick, account, target, promptChars, labels, witnessRecords, payload)
	if err != nil {
		return nil, err
	}

	// Self-modify pre-route (#1397): a GUARDED worker aimed at the trust-critical witness
	// machinery (the adjudicator/policy/kernel/shipgate -- the referee's own trees) can
	// investigate but must never SHIP -- rewriting its own referee is the RSI hazard, so
	// hold rather than let it burn turns. NOTE the correction to the original #1338 model:
	// the hold is NOT the whole cmd/**+internal/** module. `fak guard` (no --policy) runs
	// the embedded guard-default-policy.json, whose self_modify_globs are secrets/dotfiles
	// only -- it names no cmd/ or internal/ code tree, so a guarded worker DOES ship
	// gateway/agent/cmd/... normally. Only the trust-critical subset is held. The hold
	// fires on TWO signals: the lane tree is trust-critical (a correctly-routed
	// adjudicator/policy/... lane), OR the target issue's own text references that
	// machinery even though it routed to a SAFE lane -- the MIS-ROUTE the router's path
	// extractor hides (a `fix(policy):` issue whose real work is internal/policy aliases
	// to the tools lane carrying zero extracted paths). Hold BEFORE both the dry-run plan
	// and the live spawn so the loop honest-STOPs and the operator routes it to an
	// unguarded/operator or worktree-isolated path (#1334). An unguarded worker
	// (FLEET_DOGFOOD_GUARD=0, or a worktree-isolated path) never trips this.
	issueText := dispatchMapString(promptRec, "title") + "\n" + dispatchMapString(promptRec, "body")
	if held, tree := dispatchtick.SelfModifyHoldForPick(guardedPreview, pick.Tree, issueText); held {
		payload["ok"] = false
		payload["action"] = "self_modify_hold"
		payload["verdict"] = "SELF_MODIFY_HOLD"
		payload["self_modify_tree"] = tree
		payload["reason"] = fmt.Sprintf("issue #%d targets fak's trust-critical witness machinery (lane %q, tree %q -- the adjudicator/policy/kernel/shipgate the referee binds to): a guarded %s worker can investigate but must never SHIP an edit that rewrites its own referee (reason=SELF_MODIFY), so this work is operator-gated -- route it to an unguarded/operator or worktree-isolated path (#1334), not a self-guarded worker", target, pick.Lane, tree, opts.Backend)
		return finish(payload), nil
	}

	dryRunGrant := launchSpawnBroker(newLaunchBrokerAttempt("dispatch_tick", opts.Backend, launchPreview, nil, root))
	payload["spawn_broker"] = launchBrokerGrantMap(dryRunGrant)
	if !dryRunGrant.Allow {
		payload["ok"] = false
		payload["action"] = "broker_denied"
		payload["verdict"] = "SPAWN_BROKER_DENIED"
		payload["reason"] = "spawn broker denied dispatch worker launch: " + dryRunGrant.Reason
		return finish(payload), nil
	}

	if gate, refused, err := dispatchCodexLoopGateForTick(opts, account); err != nil {
		return nil, err
	} else if gate != nil {
		payload["codex_loop_gate"] = gate
		if refused {
			payload["ok"] = false
			payload["action"] = "codex_loop_gate_refused"
			payload["verdict"] = "CODEX_LOOP_GATE_REFUSED"
			payload["reason"] = fmt.Sprintf("Codex loop gate refused live dispatch: fail-on=%s verdict=%s reason=%s",
				dispatchMapString(gate, "fail_on"), dispatchMapString(gate, "verdict"), dispatchMapString(gate, "reason"))
			return finish(payload), nil
		}
	}

	// Focus WIP-breadth backpressure (#3223): warn-first. When the fleet is at/over the
	// focusscore WIP cap AND this spawn OPENS a new objective, attach the FOCUS_WIP_SATURATED
	// advisory so `fak dispatch status` / the tick JSON surface it distinctly from the
	// rate-limit and collision holds. Default posture WARN (advise + still spawn, so the
	// live fleet is byte-identical below cap and on a continuation); the --focus-hold /
	// FLEET_DISPATCH_FOCUS_HOLD posture instead REFUSES opening the new objective. It never
	// blocks a continuation of an already-open objective and runs AFTER every higher-precedence
	// gate, so it is the last, narrowest throttle before a genuinely new concurrent objective.
	if focusAdm := dispatchEvaluateFocus(root, dispatchFocusHoldPosture(opts), target); focusAdm.Advise {
		payload["focus"] = focusAdm.Map()
		if focusAdm.Hold {
			payload["ok"] = false
			payload["action"] = "focus_hold"
			payload["verdict"] = dispatchtick.FocusWIPSaturated
			payload["reason"] = focusAdm.Reason
			return finish(payload), nil
		}
	}

	if !opts.Live {
		payload["ok"] = true
		payload["action"] = "would_spawn"
		payload["verdict"] = "WOULD_SPAWN"
		payload["reason"] = fmt.Sprintf("safe to spawn 1 %s issue-resolution worker on #%d (lane %q) under account %q", opts.Backend, target, pick.Lane, account.Tag)
		return finish(payload), nil
	}

	spawnStart = time.Now()
	return dispatchTickLiveSpawn(root, runsDir, opts, pick, pickRes.leaseID, account, launch, target, promptRec, payload, finish)
}

// dispatchTickLiveSpawn performs the live spawn once every dry-run gate has passed: acquire
// the lane lease (refused → LANE_LEASE_HELD), build the guarded worker command + env, spawn
// the issue-resolution worker, and record the SPAWNED / SPAWN_FAILED payload. It mutates and
// returns the shared payload through finish, mirroring the dry-run return sites it splits off.
func dispatchTickLiveSpawn(root, runsDir string, opts dispatchTickOptions, pick dispatchLanePick, leaseID string, account dispatchtick.Account, launch dispatchtick.WorkerLaunch, target int, promptRec, payload map[string]any, finish func(map[string]any) map[string]any) (map[string]any, error) {
	lease := acquireDispatchLaneLease(root, leaseID, pick.Lane, pick.Tree, opts.WorkerTimeoutS+dispatchtick.LeaseTTLMarginS, opts.Goal)
	payload["lease"] = lease
	if bundle := mapAt(payload, "startup_bundle"); len(bundle) > 0 {
		bundle["lease"] = lease
	}
	if refused, _ := lease["refused"].(bool); refused {
		payload["ok"] = false
		payload["action"] = "lane_leased"
		payload["verdict"] = "LANE_LEASE_HELD"
		payload["reason"] = fmt.Sprintf("lane %q lease is held by a live peer", pick.Lane)
		recordDispatchPayload(runsDir, opts.Backend, payload)
		return finish(payload), nil
	}

	prompt := dispatchMapString(promptRec, "prompt")
	command, err := dispatchtick.BuildWorkerCommand(opts.Backend, prompt, launch)
	if err != nil {
		return nil, err
	}
	launchCommand, guarded := guardedDispatchCommand(root, pick.Lane, opts.Backend, command)
	if guarded {
		augmentGuardEnvDefaults()
	}
	env, err := dispatchWorkerEnv(opts.Backend, pick.Lane, root, runsDir, account, opts.Goal, opts.GoalProfile)
	if err != nil {
		return nil, err
	}
	env["FLEET_RESOLVE_ISSUE"] = strconv.Itoa(target)
	if opts.Membership != nil {
		for k, v := range dispatchtick.WaveMembershipEnv(*opts.Membership) {
			env[k] = v
		}
	}
	baseSHA := currentGitSHA(root)
	grant := launchSpawnBroker(newLaunchBrokerAttempt("dispatch_tick", opts.Backend, launchCommand, env, root))
	payload["spawn_broker"] = launchBrokerGrantMap(grant)
	if bundle := mapAt(payload, "startup_bundle"); len(bundle) > 0 {
		bundle["spawn_broker"] = launchBrokerMetadataMap(grant.Metadata)
	}
	if !grant.Allow {
		payload["ok"] = false
		payload["action"] = "broker_denied"
		payload["verdict"] = "SPAWN_BROKER_DENIED"
		payload["reason"] = "spawn broker denied dispatch worker launch: " + grant.Reason
		recordDispatchPayload(runsDir, opts.Backend, payload)
		return finish(payload), nil
	}
	launchCommand = grant.Argv
	env = grant.Env
	spawnCWD := firstString(grant.CWD, root)

	// #3168: opt-in per-worker git worktree isolation. When FLEET_WORKER_WORKTREE
	// is on, prepare a throwaway detached worktree pinned at baseSHA and run the
	// worker in it with GOCACHE/GOTMPDIR redirected inside — so a broken build in
	// one worker can't red another and two commits can't race the shared index.
	// FAIL-OPEN: any worktree-layer fault leaves spawnCWD/env untouched (the worker
	// runs in the shared trunk exactly as before). The .worktree sidecar the spawner
	// writes lets the witness sweep land+reap it on exit. Default-off restores
	// today's behavior byte-for-byte.
	if workerWorktreeEnabled() {
		if res := workerworktree.Prepare(root, pick.Lane, strconv.Itoa(target), baseSHA, "", nil); res.OK {
			spawnCWD = res.Path
			env = workerworktree.WorktreeEnv(env, res.Path)
			payload["worker_worktree"] = res.Path
		} else {
			payload["worker_worktree_failopen"] = res.Reason
		}
	}

	stdinPayload := dispatchtick.WorkerStdinPayload(opts.Backend, prompt)
	spawned, err := dispatchIssueWorkerSpawner(launchCommand, env, spawnCWD, runsDir, target, pick.Lane, opts.Backend, leaseID, pick.Tree, account, opts.Membership, baseSHA, stdinPayload, opts.SpawnProbeS)
	if err != nil {
		payload["ok"] = false
		payload["action"] = "spawn_failed"
		payload["verdict"] = "SPAWN_FAILED"
		payload["reason"] = err.Error()
		recordDispatchPayload(runsDir, opts.Backend, payload)
		return finish(payload), nil
	}
	payload["command"] = dispatchtick.LaunchCommandShape(command, root, account)
	payload["launch_command"] = dispatchtick.LaunchCommandShape(launchCommand, root, account)
	payload["guarded"] = guarded
	if bundle := mapAt(payload, "startup_bundle"); len(bundle) > 0 {
		spawned.Startup = writeDispatchStartupBundleSidecar(spawned.Log, bundle)
	}
	payload["spawned"] = dispatchSpawnMap(spawned)
	// Layer 5b: record the pinned model as a .model sidecar so the witness sweep can scrape
	// it back into WitnessRecord.Model (and Layer-2 downgrade can read what the slot ran on).
	// Written only when the model was un-blanked — a seat-default worker leaves no sidecar.
	writeDispatchModelSidecar(spawned.Log, launch.Model)
	if reason, failed := dispatchEarlyExitFailureReason(opts.Backend, spawned.PID, target, spawned.EarlyExit); failed {
		payload["ok"] = false
		payload["action"] = "spawn_failed"
		payload["verdict"] = "SPAWN_FAILED"
		payload["reason"] = reason
		recordDispatchPayload(runsDir, opts.Backend, payload)
		return finish(payload), nil
	}
	payload["ok"] = true
	payload["action"] = "spawned"
	payload["verdict"] = "SPAWNED"
	payload["reason"] = fmt.Sprintf("spawned %s issue-resolution worker pid %d on #%d (lane %q) under %q", opts.Backend, spawned.PID, target, pick.Lane, account.Tag)
	recordDispatchPayload(runsDir, opts.Backend, payload)
	return finish(payload), nil
}

var dispatchRunJSON = dispatchRunPythonJSON

func dispatchRunPythonJSON(root string, stderr io.Writer, timeout time.Duration, args ...string) (map[string]any, error) {
	interps := []string{}
	if p := strings.TrimSpace(os.Getenv("FAK_PYTHON")); p != "" {
		interps = append(interps, p)
	}
	interps = append(interps, "python3", "python")
	var lastErr error
	for _, py := range interps {
		ctx, cancel := context.WithTimeout(context.Background(), timeout)
		cmd := exec.CommandContext(ctx, py, args...)
		cmd.Dir = root
		// Bound the post-deadline pipe wait: on Windows both `python3` (the
		// WindowsApps alias) and pip console-script shims run the real
		// interpreter as a GRANDCHILD that inherits the stdout pipe. When the
		// context fires, Go kills only the direct child (the shim), and without
		// a WaitDelay cmd.Output() then blocks UNBOUNDEDLY until the surviving
		// grandchild exits -- the dispatch tick "hangs past its timeout" class.
		cmd.WaitDelay = 10 * time.Second
		configureDispatchHelperCommand(cmd)
		out, err := cmd.Output()
		cancel()
		if obj, perr := lastJSONObject(out); perr == nil {
			return obj, nil
		}
		if err == nil {
			// The helper RAN and exited 0 but printed no JSON object. That is a
			// semantic output mismatch, not a broken interpreter: re-running the
			// SAME helper under the next interpreter cannot print different
			// output, it only doubles the cost of the heaviest preflight step
			// (a fleet_sessions.py registry scan is ~40s per run on an idle
			// box, multiples under fleet load). Fail fast instead of retrying.
			return nil, fmt.Errorf("python helper %s (%s): no JSON object in helper output", strings.Join(args, " "), py)
		}
		lastErr = err
	}
	return nil, fmt.Errorf("python helper %s (tried %s): %w", strings.Join(args, " "), strings.Join(interps, ", "), lastErr)
}

func acquireDispatchLaneLease(root, id, lane string, tree []string, ttlS int, goal string) map[string]any {
	holder := dispatchLeaseHolderForGoal(goal)
	store := leaseref.NewInDir(root)
	now := time.Now()
	live, _, liveErr := store.Live(context.Background(), now)
	if liveErr != nil {
		return map[string]any{"acquired": false, "refused": false, "id": id, "holder": holder, "fail_open": true, "error": liveErr.Error(), "tree": tree}
	}
	// One admission contract for every surface (internal/regionadmit): the same
	// decision `fak loop drive` and `fak loop region` run — tree geometry PLUS
	// dos.toml lane semantics (a named lane serializes; an exclusive lane runs
	// alone). A missing taxonomy degrades to the historical geometry-only check.
	tax, taxErr := regionadmit.LoadTaxonomy(root)
	if taxErr != nil {
		tax = regionadmit.Taxonomy{}
	}
	// SelfID stays empty on purpose: a live lease under this very id (a previous
	// worker on this lane, still running) must refuse here exactly as it always
	// has — with a pinned FAK_LEASE_OWNER the fence would otherwise read the new
	// tick as the SAME holder and silently renew, double-spawning the lane.
	req := regionadmit.Request{Actor: holder, Lane: lane, Tree: tree}
	dec := regionadmit.Decide(req, regionLeases(live), tax)
	if !dec.Admit {
		return map[string]any{
			"acquired": false,
			"refused":  true,
			"id":       id,
			"holder":   holder,
			"reason":   dec.Reason,
			"rung":     dec.Rung,
			"detail":   dec.Detail,
			"tree":     tree,
		}
	}
	// Record the tree the decision was MADE on: with an empty requested tree
	// and a named lane, Decide admits on the lane's canonical taxonomy tree —
	// writing the raw (empty) tree instead would publish an unknown-blast-radius
	// lease that conservatively blocks every peer after a permissive admit.
	recTree := regionadmit.ResolveTree(req, tax)
	if len(recTree) == 0 {
		recTree = tree
	}
	rec := leaseref.Record{ID: id, TreeGlobs: recTree, Holder: holder, TTLSeconds: int64(ttlS)}
	written, verdict, err := store.AcquireFenced(context.Background(), rec, now)
	if err != nil {
		return map[string]any{"acquired": false, "refused": false, "id": id, "holder": holder, "fail_open": true, "error": err.Error(), "tree": tree}
	}
	if verdict.OK {
		return map[string]any{"acquired": true, "refused": false, "id": id, "holder": holder, "generation": written.Generation, "tree": tree}
	}
	return map[string]any{"acquired": false, "refused": true, "id": id, "holder": holder, "reason": string(verdict.Reason), "detail": verdict.Detail, "tree": tree}
}

func dispatchLeaseHolder() string {
	if v := strings.TrimSpace(os.Getenv("FAK_LEASE_OWNER")); v != "" {
		return v
	}
	if v := strings.TrimSpace(os.Getenv("CLAUDE_CODE_SESSION_ID")); v != "" {
		return v
	}
	host, _ := os.Hostname()
	if host == "" {
		host = runtime.GOOS
	}
	return fmt.Sprintf("%s:%d", host, os.Getpid())
}

func dispatchWorkerEnv(backend, lane, root, runsDir string, account dispatchtick.Account, goal, goalProfile string) (map[string]string, error) {
	env := envMap(os.Environ())
	env["DISPATCH_WORKSPACE"] = root
	env["DISPATCH_LANE"] = lane
	env["DISPATCH_BACKEND"] = backend
	if strings.TrimSpace(goal) != "" {
		env["DISPATCH_GOAL"] = strings.TrimSpace(goal)
		env["FLEET_DISPATCH_GOAL"] = strings.TrimSpace(goal)
	}
	if strings.TrimSpace(goalProfile) != "" {
		env["DISPATCH_GOAL_PROFILE"] = strings.TrimSpace(goalProfile)
	}
	switch backend {
	case "claude":
		if account.Dir != "" {
			env["CLAUDE_CONFIG_DIR"] = account.Dir
			delete(env, "CLAUDE_CODE_OAUTH_TOKEN")
		}
		env["FLEET_DISPATCH_WITNESS"] = "benchmark"
		env["FLEET_BENCH_WITNESS_CMD"] = "python tools/bench_witness.py --lane " + lane
		env["DISPATCH_OBSERVE"] = "1"
	case "opencode":
		delete(env, "CLAUDE_CONFIG_DIR")
		delete(env, "CLAUDE_CODE_OAUTH_TOKEN")
		if account.Dir != "" {
			env["XDG_CONFIG_HOME"] = opencodeConfigHome(account.Dir, runsDir)
		}
	case "codex":
		delete(env, "CLAUDE_CONFIG_DIR")
		delete(env, "CLAUDE_CODE_OAUTH_TOKEN")
		if account.Dir != "" {
			env["CODEX_HOME"] = account.Dir
		}
	default:
		return nil, fmt.Errorf("unknown backend %q", backend)
	}
	return env, nil
}

func opencodeConfigHome(accountDir, runsDir string) string {
	if filepath.Base(accountDir) == "opencode" {
		return filepath.Dir(accountDir)
	}
	// Best-effort, no shell: when a non-canonical account dir is supplied, use its parent.
	// The switcher normally hands the canonical dir; this fallback keeps the Go tick portable.
	return filepath.Dir(accountDir)
}

func guardedDispatchCommand(root, lane, backend string, command []string) ([]string, bool) {
	if guardDisabled() {
		return command, false
	}
	fakBin := resolveDispatchFakBin(root)
	baseURL := strings.TrimSpace(os.Getenv("FLEET_DOGFOOD_GUARD_BASEURL"))
	guarded, ok := dispatchtick.GuardedLaunchCommand(command, fakBin, lane, backend, root, baseURL)
	if !ok {
		return guarded, ok
	}
	// Resolve the fleet managed-cache posture (FAK_MANAGED_CACHE / FAK_GUARD_API_KEY_ENV) in
	// THIS tick process and splice it into the child's guard argv — the worker's guard reads
	// the flag, not the env, so a resumed child (whose gateway env is stripped, b2926823) still
	// carries it. A headless fleet turn must not die over a cache-posture typo: warn to the
	// worker log and fall back to auto (no flag) so the wave still launches.
	postureArgs, postureErr := fleetGuardCachePostureArgs()
	if postureErr != nil {
		fmt.Fprintf(os.Stderr, "fak dispatch: %v; using managed-cache auto\n", postureErr)
		postureArgs = nil
	}
	return spliceGuardPostureArgs(guarded, postureArgs), ok
}

// spliceGuardPostureArgs inserts extra `fak guard` flags immediately before the `--` that
// separates guard's own flags from the wrapped agent command, so guard parses them and the
// agent never sees them. A nil/empty posture, or an argv with no `--` (already-unguarded),
// returns the argv unchanged — an unconfigured fleet's command is byte-identical to before.
func spliceGuardPostureArgs(argv, postureArgs []string) []string {
	if len(postureArgs) == 0 {
		return argv
	}
	for i, a := range argv {
		if a == "--" {
			out := make([]string, 0, len(argv)+len(postureArgs))
			out = append(out, argv[:i]...)
			out = append(out, postureArgs...)
			out = append(out, argv[i:]...)
			return out
		}
	}
	return argv
}

// dispatchWorkerFallbackModel resolves the Claude fallback CHAIN a headless dispatch
// worker hands to `claude -p --fallback-model`, so an unattended fleet turn degrades to
// a backup model through a transient overload/unavailability window instead of dying and
// re-dispatching the same walled model. It is the background/headless counterpart of the
// interactive launcher's chain (accounts_launch.go's defaultLaunchFallbackModel), and it
// reuses that same default (Opus 4.8) so both fronts fall back the same way. The flag is
// Claude-specific and print-mode scoped, so it applies ONLY to the claude backend; codex
// and opencode pin their own model via -m and get "". FLEET_WORKER_FALLBACK_MODEL overrides
// the default (a comma-separated chain, e.g. "claude-opus-4-8,claude-sonnet-5"); an explicit
// empty/off/none/disable/0/false DISABLES it (restores the historical no-fallback command),
// so an operator who needs the worker pinned to exactly the seat model for a benchmark or a
// model-accounting run can turn it off without a rebuild.
func dispatchWorkerFallbackModel(backend string) string {
	if backend != "claude" {
		return ""
	}
	raw, ok := os.LookupEnv("FLEET_WORKER_FALLBACK_MODEL")
	if !ok {
		return defaultLaunchFallbackModel
	}
	switch strings.ToLower(strings.TrimSpace(raw)) {
	case "", "0", "off", "false", "no", "none", "disable", "disabled":
		return ""
	}
	return strings.TrimSpace(raw)
}

func guardDisabled() bool {
	raw, ok := os.LookupEnv("FLEET_DOGFOOD_GUARD")
	if !ok {
		return false
	}
	switch strings.ToLower(strings.TrimSpace(raw)) {
	case "", "0", "off", "false", "no", "disable", "disabled":
		return true
	}
	return false
}

// workerWorktreeEnabled reports whether #3168 per-worker git worktree isolation is
// switched on via FLEET_WORKER_WORKTREE. Default (unset / an off-ish value) is OFF,
// which restores the shared-trunk spawn behavior byte-for-byte; any other value
// turns isolation on. Mirrors guardDisabled's truthy/falsy grammar (inverted).
func workerWorktreeEnabled() bool {
	raw, ok := os.LookupEnv("FLEET_WORKER_WORKTREE")
	if !ok {
		return false
	}
	switch strings.ToLower(strings.TrimSpace(raw)) {
	case "", "0", "off", "false", "no", "disable", "disabled":
		return false
	}
	return true
}

// dispatchTierLaunchEnabled reports whether the opt-in per-issue tier launch profile
// (FLEET_TIER_LAUNCH) is switched on. Default (unset / an off-ish value) is OFF, which keeps
// every worker on the seat-default model with no effort/ultracode uplift — byte-identical to
// before this seam. Mirrors workerWorktreeEnabled's truthy/falsy grammar.
func dispatchTierLaunchEnabled() bool {
	raw, ok := os.LookupEnv("FLEET_TIER_LAUNCH")
	if !ok {
		return false
	}
	switch strings.ToLower(strings.TrimSpace(raw)) {
	case "", "0", "off", "false", "no", "disable", "disabled":
		return false
	}
	return true
}

// dispatchTierLaunchTable is the tier→launch-profile table the resolver consults. Today it is
// the built-in default (routine→fable+xhigh, normal→opus+xhigh, hard→opus+ultracode,
// ultra→fable+ultracode). An operator per-bucket override is a fail-open follow-on that merges
// onto this default, so a malformed override can only ever fall back to here.
func dispatchTierLaunchTable() dispatchtick.TierLaunchTable {
	return dispatchtick.DefaultTierLaunchTable()
}

// dispatchTierLaunchProfile resolves the opt-in launch profile for a target issue from its
// per-issue labels AND the tick-wide work kind, or nil to leave the seat-default posture. It
// returns nil when the FLEET_TIER_LAUNCH knob is off, the backend is not claude (the model
// uplift + effort/ultracode are Claude-only; opencode/codex pin their own seat model with -m
// and ignore both), or nothing resolves a profile. Per-issue labels win first (tier/ultra, a
// valid tier, or a bare tier/pm); only an UNLABELLED issue falls through to the work kind,
// where a coordination kind (project_management / gardening) routes to the cheap PM bucket —
// so a PM dispatch loop runs on fable by default — and any other kind (notably engineering)
// keeps the seat default. The bucket is returned alongside for the payload surface; it is ""
// whenever the profile is nil.
func dispatchTierLaunchProfile(backend string, labels []string, workKind string) (*dispatchtick.LaunchProfile, dispatchtick.LaunchBucket) {
	if backend != "claude" || !dispatchTierLaunchEnabled() {
		return nil, ""
	}
	profile, bucket, ok := dispatchtick.LaunchProfileForDispatch(labels, workKind, dispatchTierLaunchTable())
	if !ok {
		return nil, ""
	}
	return &profile, bucket
}

func resolveDispatchFakBin(root string) string {
	if v := strings.TrimSpace(os.Getenv("FAK_BIN")); v != "" {
		if _, err := os.Stat(v); err == nil {
			return v
		}
	}
	exe := "fak"
	if runtime.GOOS == "windows" {
		exe = "fak.exe"
	}
	intree := filepath.Join(root, "tools", ".bin", exe)
	if _, err := os.Stat(intree); err == nil {
		return intree
	}
	if self, err := os.Executable(); err == nil && self != "" {
		return self
	}
	if p, err := exec.LookPath("fak"); err == nil {
		return p
	}
	return ""
}

func augmentGuardEnvDefaults() {
	for _, key := range []string{"FAK_PLANNER_TIMEOUT_S", "FAK_HTTP_WRITE_TIMEOUT_S"} {
		if os.Getenv(key) == "" {
			_ = os.Setenv(key, "600")
		}
	}
}

var dispatchIssueWorkerSpawner = spawnDispatchIssueWorker

func spawnDispatchIssueWorker(command []string, env map[string]string, cwd, runsDir string, issue int, lane, backend, leaseID string, tree []string, account dispatchtick.Account, membership *dispatchtick.Membership, baseSHA, stdinPayload string, probeS float64) (dispatchSpawnResult, error) {
	if len(command) == 0 {
		return dispatchSpawnResult{}, errors.New("empty worker command")
	}
	if err := os.MkdirAll(runsDir, 0o755); err != nil {
		return dispatchSpawnResult{}, err
	}
	stamp := time.Now().UTC().Format("20060102-150405")
	outLog := filepath.Join(runsDir, fmt.Sprintf("resolve-%d-%s.log", issue, stamp))
	exe := resolveDispatchWorkerExecutable(backend, command[0])
	fh, err := os.OpenFile(outLog, os.O_CREATE|os.O_TRUNC|os.O_WRONLY, 0o644)
	if err != nil {
		return dispatchSpawnResult{}, err
	}
	fmt.Fprintf(fh, "# fak-spawn %s issue=%d lane=%s backend=%s argv0=%s\n", stamp, issue, lane, backend, filepath.Base(exe))
	_ = fh.Sync()
	stem := strings.TrimSuffix(outLog, filepath.Ext(outLog))
	if backend == "opencode" && stdinPayload != "" {
		promptPath := stem + ".prompt.txt"
		if err := os.WriteFile(promptPath, []byte(stdinPayload), 0o600); err != nil {
			_ = fh.Close()
			return dispatchSpawnResult{}, err
		}
		command = dispatchAttachOpencodePromptFile(command, promptPath)
		stdinPayload = ""
	}
	cmd := exec.Command(exe, command[1:]...)
	cmd.Dir = cwd
	cmd.Env = envSliceFromMap(env)
	if stdinPayload != "" {
		cmd.Stdin = strings.NewReader(stdinPayload)
	} else {
		devNull, _ := os.Open(os.DevNull)
		if devNull != nil {
			defer devNull.Close()
			cmd.Stdin = devNull
		}
	}
	cmd.Stdout = fh
	cmd.Stderr = fh
	configureDispatchSpawn(cmd)
	if err := cmd.Start(); err != nil {
		_ = fh.Close()
		return dispatchSpawnResult{}, err
	}
	_ = fh.Close()

	_ = os.WriteFile(stem+".pid", []byte(strconv.Itoa(cmd.Process.Pid)), 0o644)
	_ = os.WriteFile(stem+".backend", []byte(backend), 0o644)
	if leaseID != "" {
		_ = os.WriteFile(stem+dispatchLeaseIDSidecarSuffix, []byte(leaseID), 0o644)
	}
	tree = dispatchTrimTree(tree)
	if len(tree) > 0 {
		if b, err := json.Marshal(tree); err == nil {
			_ = os.WriteFile(stem+dispatchLeaseTreeSidecarSuffix, b, 0o644)
		}
	}
	if baseSHA != "" {
		_ = os.WriteFile(stem+dispatchtick.BaseSHASidecarSuffix, []byte(baseSHA), 0o644)
	}
	// #3168: when the worker ran in a per-worker git worktree (cwd carries the marker),
	// record it so the witness sweep can land+reap it after the pid dies.
	if workerworktree.IsWorkerWorktree(cwd) {
		_ = os.WriteFile(stem+dispatchWorktreeSidecarSuffix, []byte(cwd), 0o644)
	}
	acct := dispatchtick.AccountSidecar(account)
	if len(acct) > 0 {
		if b, err := json.Marshal(acct); err == nil {
			_ = os.WriteFile(stem+dispatchtick.AccountSidecarSuffix, b, 0o644)
		}
	}
	var mem any
	if membership != nil {
		mem = *membership
		if b, err := json.Marshal(membership); err == nil {
			_ = os.WriteFile(stem+dispatchtick.WaveSidecarSuffix, b, 0o644)
		}
	}
	res := dispatchSpawnResult{PID: cmd.Process.Pid, Log: outLog, Issue: issue, Lane: lane, Backend: backend, LeaseID: leaseID, Tree: tree, Account: acct, Membership: mem}
	if probeS > 0 {
		res.EarlyExit = probeDispatchSpawn(cmd, outLog, probeS)
	}
	return res, nil
}

func dispatchAttachOpencodePromptFile(command []string, promptPath string) []string {
	out := append([]string(nil), command...)
	if len(out) == 0 || strings.TrimSpace(promptPath) == "" {
		return out
	}
	if len(out) == 1 {
		return append(out, "--file", promptPath)
	}
	last := out[len(out)-1]
	out = out[:len(out)-1]
	out = append(out, "--file", promptPath, "--", last)
	return out
}

func resolveDispatchWorkerExecutable(backend, name string) string {
	exe := name
	if p, err := exec.LookPath(exe); err == nil {
		exe = p
	}
	if backend == "opencode" && runtime.GOOS == "windows" {
		if target := unwrapOpencodeNpmShim(exe); target != "" {
			return target
		}
	}
	return exe
}

func unwrapOpencodeNpmShim(exe string) string {
	switch strings.ToLower(filepath.Base(exe)) {
	case "opencode", "opencode.cmd", "opencode.bat", "opencode.ps1":
	default:
		return ""
	}
	dir := filepath.Dir(exe)
	if dir == "" || dir == "." {
		return ""
	}
	target := filepath.Join(dir, "node_modules", "opencode-ai", "bin", "opencode.exe")
	if st, err := os.Stat(target); err == nil && !st.IsDir() {
		return target
	}
	return ""
}

func probeDispatchSpawn(cmd *exec.Cmd, logPath string, waitS float64) map[string]any {
	done := make(chan error, 1)
	go func() { done <- cmd.Wait() }()
	select {
	case err := <-done:
		rec := map[string]any{"checked": true, "alive": false, "wait_s": waitS, "silent": true, "returncode": 0}
		if err != nil {
			if ee, ok := err.(*exec.ExitError); ok {
				rec["returncode"] = ee.ExitCode()
			} else {
				rec["error"] = err.Error()
			}
		}
		if st, statErr := os.Stat(logPath); statErr == nil {
			rec["log_bytes"] = st.Size()
			rec["silent"] = st.Size() == 0
			if st.Size() > 0 {
				class := dispatchEarlyExitClass(logPath)
				rec["class"] = class
				rec["summary"] = dispatchEarlyExitSummary(class)
			}
		}
		return rec
	case <-time.After(time.Duration(waitS * float64(time.Second))):
		return map[string]any{"checked": true, "alive": true, "wait_s": waitS}
	}
}

func dispatchEarlyExitClass(logPath string) string {
	tail, size := dispatchWitnessLogTail(logPath)
	if tail == "" && size <= 0 {
		return dispatchtick.NoCommitUnknown
	}
	return dispatchtick.ClassifyNoCommitReason(tail, size)
}

func dispatchEarlyExitSummary(class string) string {
	switch class {
	case dispatchtick.NoCommitAuthWall:
		return "login/auth wall"
	case dispatchtick.NoCommitUsageCap:
		return "usage/weekly cap (model-switchable)"
	case dispatchtick.NoCommitModelUnknown:
		return "model unavailable/unentitled (model-switchable)"
	case dispatchtick.NoCommitRateLimit:
		return "rate limit/overload (model-switchable)"
	case dispatchtick.NoCommitSelfModify:
		return "guard self-modify refusal"
	case dispatchtick.NoCommitPolicyBlock:
		return "guard policy refusal"
	case dispatchtick.NoCommitOffTrunk:
		return "guard off-trunk refusal"
	case dispatchtick.NoCommitBannerNoop:
		return "banner-only no-op"
	default:
		return "unclassified early process exit"
	}
}

func dispatchEarlyExitFailureReason(backend string, pid, issue int, early map[string]any) (string, bool) {
	if len(early) == 0 || dispatchMapBool(early, "alive") {
		return "", false
	}
	if dispatchMapBool(early, "silent") {
		return fmt.Sprintf("%s worker pid %d for #%d exited immediately and produced an empty log", backend, pid, issue), true
	}
	code := dispatchMapInt(early, "returncode")
	if code == 0 && dispatchMapString(early, "error") == "" {
		return "", false
	}
	waitS := dispatchMapFloat(early, "wait_s")
	reason := fmt.Sprintf("%s worker pid %d for #%d exited within %.1fs", backend, pid, issue, waitS)
	if code != 0 {
		reason += fmt.Sprintf(" with code %d", code)
	}
	if err := dispatchMapString(early, "error"); err != "" {
		reason += ": " + err
	}
	if class := dispatchMapString(early, "class"); class != "" && class != dispatchtick.NoCommitUnknown {
		reason += " (" + dispatchEarlyExitSummary(class) + ")"
	}
	return reason, true
}

func recordDispatchPayload(runsDir, backend string, payload map[string]any) {
	blob, err := json.MarshalIndent(payload, "", "  ")
	if err != nil {
		return
	}
	if err := os.MkdirAll(runsDir, 0o755); err != nil {
		return
	}
	_ = os.WriteFile(filepath.Join(runsDir, "last-resolve-tick-"+backend+".json"), blob, 0o644)
	_ = os.WriteFile(filepath.Join(runsDir, "last-resolve-tick.json"), blob, 0o644)
}

func dispatchStartupBundle(root string, opts dispatchTickOptions, pre map[string]any, account dispatchtick.Account, pick dispatchLanePick, leaseID string, target int, hasTarget bool, held map[string]bool, liveIssues map[int]bool, cooled map[int]bool, cooldownStatus []map[string]any) map[string]any {
	route := map[string]any{
		"lane":             pick.Lane,
		"target_issue":     nil,
		"candidate_issues": append([]int(nil), pick.Numbers...),
		"lane_issue_count": len(pick.Numbers),
		"lane_step_budget": pick.ByLaneStepBudget[pick.Lane],
		"tree":             append([]string(nil), pick.Tree...),
		"held_lanes":       sortedStringSet(held),
		"already_live":     sortedSet(liveIssues),
		"cooled_recently":  sortedSet(cooled),
		"cooldown_status":  cooldownStatus,
	}
	if hasTarget {
		route["target_issue"] = target
	}
	return map[string]any{
		"schema":    dispatchStartupBundleSchema,
		"workspace": root,
		"backend":   opts.Backend,
		"goal": map[string]any{
			"id":      opts.Goal,
			"profile": opts.GoalProfile,
		},
		"route": route,
		"cap": map[string]any{
			"cap":             pre["cap"],
			"live":            pre["live"],
			"headroom":        pre["headroom"],
			"max_workers":     pre["max_workers"],
			"host_cap":        pre["host_cap"],
			"host_capacity":   mapAt(pre, "host_capacity"),
			"cap_terms":       mapAt(pre, "cap_terms"),
			"kernel":          mapAt(pre, "kernel"),
			"os_worker_procs": pre["os_worker_procs"],
		},
		"seat": mapAt(pre, "seat"),
		"lease": map[string]any{
			"id":   leaseID,
			"tree": append([]string(nil), pick.Tree...),
		},
		"dirty_tree": dispatchDirtyTree(root),
		"stale_base": dispatchStaleBase(root, pick.Tree),
		"account":    dispatchtick.AccountSidecar(account),
		"preflight": map[string]any{
			"verdict": dispatchMapString(pre, "verdict"),
			"reason":  dispatchMapString(pre, "reason"),
		},
	}
}

func dispatchStaleBase(root string, tree []string) map[string]any {
	tree = dispatchTrimTree(tree)
	roles, roleErr := branchrole.Load(root)
	if roleErr != nil {
		roles = branchrole.Defaults()
	}
	upstreamBranch := strings.TrimSpace(roles.DevelopmentBranch)
	if upstreamBranch == "" {
		upstreamBranch = branchrole.Defaults().DevelopmentBranch
	}
	upstreamRef := fmt.Sprintf("origin/%s", upstreamBranch)
	out := map[string]any{
		"available": false,
		"stale":     false,
		"base":      "HEAD",
		"upstream":  upstreamRef,
		"tree":      append([]string(nil), tree...),
	}
	if len(tree) == 0 {
		out["available"] = true
		out["reason"] = "no target tree to compare"
		return out
	}
	args := []string{"diff", "--name-only", "HEAD.." + upstreamRef, "--"}
	args = append(args, tree...)
	// Bounded: this runs on every tick's startup bundle, before any spawn
	// decision. An unbounded git call (index lock contention under a loaded
	// fleet) would stall the whole tick; the bundle already fail-opens on error.
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, "git", args...)
	cmd.Dir = root
	cmd.WaitDelay = 10 * time.Second
	configureDispatchHelperCommand(cmd)
	raw, err := cmd.CombinedOutput()
	if err != nil {
		out["error"] = truncateString(strings.TrimSpace(string(raw)), 300)
		return out
	}
	changed := nonEmptyLines(string(raw))
	out["available"] = true
	out["changed"] = changed
	out["changed_count"] = len(changed)
	if len(changed) > 0 {
		out["stale"] = true
		out["warning"] = fmt.Sprintf("stale base: %s has newer changes in this target scope (%s). Before editing, refresh in place with `git fetch origin %s` and merge %s so these files include upstream work; the issue remains dispatchable after refresh.", upstreamRef, strings.Join(changed, ", "), upstreamBranch, upstreamRef)
	}
	return out
}

func dispatchDirtyTree(root string) map[string]any {
	// Bounded like dispatchStaleBase above: never let a wedged git stall the tick.
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, "git", "status", "--porcelain=v1")
	cmd.Dir = root
	cmd.WaitDelay = 10 * time.Second
	configureDispatchHelperCommand(cmd)
	out, err := cmd.CombinedOutput()
	if err != nil {
		return map[string]any{
			"available":   false,
			"clean":       nil,
			"dirty_total": nil,
			"error":       truncateString(strings.TrimSpace(string(out)), 300),
		}
	}
	rows := nonEmptyLines(string(out))
	sample := rows
	if len(sample) > 25 {
		sample = sample[:25]
	}
	return map[string]any{
		"available":     true,
		"clean":         len(rows) == 0,
		"dirty_total":   len(rows),
		"dirty_sample":  append([]string(nil), sample...),
		"dirty_omitted": len(rows) - len(sample),
	}
}

func writeDispatchStartupBundleSidecar(logPath string, bundle map[string]any) string {
	if strings.TrimSpace(logPath) == "" || len(bundle) == 0 {
		return ""
	}
	blob, err := json.MarshalIndent(bundle, "", "  ")
	if err != nil {
		return ""
	}
	stem := strings.TrimSuffix(logPath, filepath.Ext(logPath))
	path := stem + dispatchStartupBundleSidecarSuffix
	if err := os.WriteFile(path, blob, 0o644); err != nil {
		return ""
	}
	return path
}

func dispatchSpawnMap(s dispatchSpawnResult) map[string]any {
	out := map[string]any{
		"pid":     s.PID,
		"log":     s.Log,
		"issue":   s.Issue,
		"lane":    s.Lane,
		"backend": s.Backend,
	}
	if len(s.Account) > 0 {
		out["account"] = s.Account
	}
	if s.LeaseID != "" {
		out["lease_id"] = s.LeaseID
	}
	if s.Startup != "" {
		out["startup_bundle"] = s.Startup
	}
	if len(s.Tree) > 0 {
		out["tree"] = append([]string(nil), s.Tree...)
	}
	if s.Membership != nil {
		out["membership"] = s.Membership
	}
	if len(s.EarlyExit) > 0 {
		out["early_exit"] = s.EarlyExit
	}
	return out
}

func recordDispatchTickLoop(root, ledger string, payload map[string]any) map[string]any {
	if strings.TrimSpace(ledger) == "" {
		ledger = defaultLoopLedger()
	}
	runID := dispatchLoopRunID(payload)
	loopID := dispatchTickLoopID(dispatchMapString(payload, "backend"), dispatchMapString(payload, "goal"))
	pre := mapAt(payload, "preflight")
	metrics := map[string]int64{
		"live":             boolInt(payload["live"]),
		"lane_issue_count": int64(dispatchMapInt(payload, "lane_issue_count")),
		"lane_step_budget": int64(dispatchMapInt(payload, "lane_step_budget")),
		"max_workers":      int64(dispatchMapInt(payload, "max_workers")),
		"preflight_live":   int64(dispatchMapInt(pre, "live")),
		"preflight_cap":    int64(dispatchMapInt(pre, "cap")),
	}
	if n := dispatchMapInt(payload, "target_issue"); n != 0 {
		metrics["target_issue"] = int64(n)
	}
	if n := dispatchMapInt(payload, "prompt_chars"); n != 0 {
		metrics["prompt_chars"] = int64(n)
	}
	// Fold the per-phase wall-clock durations (payload["timings_ms"]) into the ledger
	// metrics under *_ms names, so every Fire/Admit/Start/End event carries them and a
	// later fold (mirroring turntaxmeter.FoldHookLatency) can compute cross-tick p50/p99
	// per phase -- the measurement a TICK_PHASE_REGRESSION budget would gate on.
	if tm, ok := payload["timings_ms"].(map[string]int64); ok {
		for phase, ms := range tm {
			key := phase + "_ms"
			if phase == "total" {
				key = "tick_total_ms"
			}
			metrics[key] = ms
		}
	}
	evidence := []loopmgr.EvidenceRef{}
	if n := dispatchMapInt(payload, "target_issue"); n != 0 {
		evidence = append(evidence, loopmgr.EvidenceRef{Kind: "issue", Ref: strconv.Itoa(n)})
	}
	if spawned := mapAt(payload, "spawned"); dispatchMapString(spawned, "log") != "" {
		evidence = append(evidence, loopmgr.EvidenceRef{Kind: "log", Ref: dispatchMapString(spawned, "log")})
	}
	if spawned := mapAt(payload, "spawned"); dispatchMapString(spawned, "startup_bundle") != "" {
		evidence = append(evidence, loopmgr.EvidenceRef{Kind: "startup_bundle", Ref: dispatchMapString(spawned, "startup_bundle")})
	}
	if goal := dispatchMapString(payload, "goal"); goal != "" {
		evidence = append(evidence, loopmgr.EvidenceRef{Kind: "goal", Ref: goal, Summary: "profile=" + dispatchMapString(payload, "goal_profile")})
	}
	account := mapAt(payload, "account")
	if tag := dispatchMapString(account, "tag"); tag != "" {
		evidence = append(evidence, loopmgr.EvidenceRef{Kind: "account", Ref: tag})
	}
	admitted := dispatchMapBool(payload, "ok") && (dispatchMapString(payload, "action") == "would_spawn" || dispatchMapString(payload, "action") == "spawned")
	events := []loopmgr.Event{
		{LoopID: loopID, RunID: runID, Kind: loopmgr.EventFire, Source: "fak dispatch tick", Principal: dispatchMapString(payload, "backend"), Summary: "issue dispatch tick lane=" + firstString(dispatchMapString(payload, "lane"), "-"), Metrics: metrics, EvidenceRefs: evidence},
		{LoopID: loopID, RunID: runID, Kind: loopmgr.EventAdmit, Source: "fak dispatch tick", Principal: dispatchMapString(payload, "backend"), Status: chooseStatus(admitted, loopmgr.StatusAdmitted, loopmgr.StatusRefused), Reason: dispatchMapString(payload, "verdict"), Summary: truncateString(dispatchMapString(payload, "reason"), 200), Metrics: metrics, EvidenceRefs: evidence},
	}
	if dispatchMapString(payload, "action") == "spawned" {
		events = append(events, loopmgr.Event{LoopID: loopID, RunID: runID, Kind: loopmgr.EventStart, Source: "fak dispatch tick", Principal: dispatchMapString(payload, "backend"), Status: loopmgr.StatusRunning, Reason: "SPAWNED", Summary: truncateString(dispatchMapString(payload, "reason"), 200), Metrics: metrics, EvidenceRefs: evidence})
	}
	if dispatchMapBool(payload, "ok") {
		events = append(events, loopmgr.Event{LoopID: loopID, RunID: runID, Kind: loopmgr.EventEnd, Source: "fak dispatch tick", Principal: dispatchMapString(payload, "backend"), Status: loopmgr.StatusClaimedDone, Reason: dispatchMapString(payload, "verdict"), Summary: truncateString(dispatchMapString(payload, "reason"), 200), Metrics: metrics, EvidenceRefs: evidence})
	}
	rows := []map[string]any{}
	ok := true
	for _, ev := range events {
		row, err := loopmgr.Append(filepath.Join(root, ledger), ev)
		if err != nil {
			ok = false
			rows = append(rows, map[string]any{"ok": false, "kind": string(ev.Kind), "error": err.Error()})
			continue
		}
		rows = append(rows, map[string]any{"ok": true, "kind": string(row.Kind), "seq": row.Seq, "hash": row.Hash})
	}
	return map[string]any{"ledger": filepath.Join(root, ledger), "loop_id": loopID, "run_id": runID, "events": rows, "ok": ok}
}

func dispatchLoopRunID(payload map[string]any) string {
	if spawned := mapAt(payload, "spawned"); dispatchMapInt(spawned, "pid") != 0 {
		return fmt.Sprintf("resolve-%d-%d", dispatchMapInt(payload, "target_issue"), dispatchMapInt(spawned, "pid"))
	}
	parts := []string{"resolve-tick", firstString(dispatchMapString(payload, "backend"), "claude")}
	if token := dispatchGoalToken(dispatchMapString(payload, "goal")); token != "" {
		parts = append(parts, token)
	}
	parts = append(parts, time.Now().UTC().Format("20060102T150405Z"))
	return strings.Join(parts, "-")
}

func chooseStatus(cond bool, yes, no loopmgr.RunStatus) loopmgr.RunStatus {
	if cond {
		return yes
	}
	return no
}

func currentGitSHA(root string) string {
	// Bounded like dispatchTickGitStatus above: never let a wedged git stall
	// the tick.
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, "git", "rev-parse", "HEAD")
	cmd.Dir = root
	cmd.WaitDelay = 10 * time.Second
	configureDispatchHelperCommand(cmd)
	out, err := cmd.Output()
	if err != nil {
		return ""
	}
	return strings.TrimSpace(string(out))
}

func renderDispatchTick(p map[string]any) string {
	a := mapAt(p, "account")
	pf := mapAt(p, "preflight")
	var b strings.Builder
	fmt.Fprintf(&b, "issue-resolve-dispatch: %s (%s)  backend=%s  live=%v\n",
		dispatchMapString(p, "verdict"), okWord(dispatchMapBool(p, "ok")), dispatchMapString(p, "backend"), p["live"])
	fmt.Fprintf(&b, "  preflight : %s (%v/%v live)\n", dispatchMapString(pf, "verdict"), pf["live"], pf["cap"])
	fmt.Fprintf(&b, "  account   : %s (t%v)  %s\n", firstString(dispatchMapString(a, "tag"), "-"), a["tier"], dispatchMapString(a, "model"))
	if goal := dispatchMapString(p, "goal"); goal != "" {
		fmt.Fprintf(&b, "  goal      : %s (%s)\n", goal, dispatchMapString(p, "goal_profile"))
	}
	fmt.Fprintf(&b, "  lane      : %s  (%d issues, %d steps)\n", firstString(dispatchMapString(p, "lane"), "-"), dispatchMapInt(p, "lane_issue_count"), dispatchMapInt(p, "lane_step_budget"))
	if n := dispatchMapInt(p, "target_issue"); n != 0 {
		fmt.Fprintf(&b, "  target    : #%d  %.54s\n", n, dispatchMapString(p, "issue_title"))
	}
	if rows := anySlice(p["cooldown_status"]); len(rows) > 0 {
		fmt.Fprintln(&b, "  cooldowns : issue age_s remaining_s next_eligible_utc state")
		for _, raw := range rows {
			row, _ := raw.(map[string]any)
			state := "ready"
			if dispatchMapBool(row, "cooling") {
				state = "cooling"
			}
			fmt.Fprintf(&b, "              #%d %d %d %s %s\n",
				dispatchMapInt(row, "issue"),
				dispatchMapInt(row, "last_attempt_age_seconds"),
				dispatchMapInt(row, "cooldown_remaining_seconds"),
				dispatchMapString(row, "next_eligible_utc"),
				state)
		}
	}
	if launch := stringSlice(p["launch_command"]); len(launch) > 0 {
		fmt.Fprintf(&b, "  launch    : %s\n", strings.Join(launch, " "))
	}
	fmt.Fprintf(&b, "  -> %s\n", dispatchMapString(p, "reason"))
	if spawned := mapAt(p, "spawned"); len(spawned) > 0 {
		fmt.Fprintf(&b, "  spawned pid=%d issue=#%d log=%s\n", dispatchMapInt(spawned, "pid"), dispatchMapInt(spawned, "issue"), dispatchMapString(spawned, "log"))
	}
	if !dispatchMapBool(p, "live") && dispatchMapString(p, "action") == "would_spawn" {
		fmt.Fprintln(&b, "  DRY-RUN - re-run with --live to spawn the issue worker")
	}
	return b.String()
}

func okWord(ok bool) string {
	if ok {
		return "ok"
	}
	return "refuse"
}

func boolInt(v any) int64 {
	if b, _ := v.(bool); b {
		return 1
	}
	return 0
}

func firstString(vals ...string) string {
	for _, v := range vals {
		if strings.TrimSpace(v) != "" {
			return v
		}
	}
	return ""
}

func firstInt(vals ...int) int {
	for _, v := range vals {
		if v != 0 {
			return v
		}
	}
	return 0
}

func maxFloat64(a, b float64) float64 {
	if a > b {
		return a
	}
	return b
}

func truncateString(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n]
}
