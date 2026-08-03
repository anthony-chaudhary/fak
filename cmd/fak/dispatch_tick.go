package main

import (
	"context"
	"encoding/json"
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

	"github.com/anthony-chaudhary/fak/internal/dispatchorder"
	"github.com/anthony-chaudhary/fak/internal/dispatchtick"
	"github.com/anthony-chaudhary/fak/internal/leasequeue"
	"github.com/anthony-chaudhary/fak/internal/leaseref"
	"github.com/anthony-chaudhary/fak/internal/modelroute"
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
	WorkerModel        string
	CapacityReason     string
	CapacityFrom       string
	CapacityTargets    []modelroute.CapacityTarget
	RequiredModelB     float64
	RequiredContext    int
	PinWorkerModel     bool
	AcceptanceArtifact string
	AcceptanceOverride string
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
	FocusHold bool
	// PlacementEvidence, RungPlacement, and AccountsRoster are the #5416 placement seams'
	// declarations — the tick's own config surface for three behavioral settings, rather than
	// three os.LookupEnv reads of the ambient process environment (internal/envconfiglint's
	// CONFIG_NOT_ENV rule: secrets belong in the environment, behavioral settings belong on the
	// config surface). PlacementEvidence records the work-class/zone sidecars and the graded
	// turn journal; RungPlacement arms the automatic placement ladder and its escalation half;
	// AccountsRoster names the roster consulted for zone ATTRIBUTION (never for dispatch), empty
	// meaning the conventional tools/model-accounts.json. All three zero values are the
	// pre-seam posture, so a tick that declares nothing is byte-identical to before they landed.
	// evaluateDispatchTick publishes them into the package seams the leaf helpers read.
	PlacementEvidence bool
	RungPlacement     bool
	AccountsRoster    string
	// HostProbeShellReuse declares the #3405 host-probe shell-reuse spine for this tick:
	// route every Windows host probe of the tick (process scan, free RAM, worker rows, codex
	// rows) through ONE warm racked PowerShell instead of one `powershell` process -- and one
	// ConPTY/conhost -- per probe. It is the tick's own config surface
	// (--host-probe-shell-reuse), not an environment read, for the same CONFIG_NOT_ENV reason
	// as the placement settings above. It is a *bool rather than a bool because the two
	// undeclared postures differ: nil means the caller declared nothing and takes the HOST
	// default (on for Windows, where the ConPTY cost is; off for every other GOOS, whose
	// probes never reach PowerShell), so a programmatic tick (wave / sweep / garden) that
	// fills no field still gets the dividend, while an explicit false is a real off switch
	// that puts every probe back on the historical one-shot spawn.
	HostProbeShellReuse *bool
	Account             *dispatchtick.Account
	Membership          *dispatchtick.Membership
	DiscoverySnapshot   *runsSnapshot
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
	capacityReason := fs.String("capacity-reason", "", "capacity block reason token")
	capacityFrom := fs.String("capacity-from", "", "blocked target name")
	capacityTargetsPath := fs.String("capacity-targets", "", "JSON file containing alternate capacity targets")
	requiredModelB := fs.Float64("required-model-b", 0, "minimum faithful model size in billions")
	requiredContext := fs.Int("required-context", 0, "required context tokens")
	pinWorkerModel := fs.Bool("pin-worker-model", false, "benchmark gate: pin the claude worker to the account/default model (model-accounting run) instead of the seat default + fallback chain")
	acceptanceArtifact := fs.String("model-acceptance", "", "require this exact-ID model acceptance artifact before provider launch")
	acceptanceOverride := fs.String("model-acceptance-override", "", "operator reason to override a model acceptance HOLD (audited)")
	modelDowngrade := fs.Bool("model-downgrade", false, "Layer-2 in-tick re-dispatch: when the target's last slot exited model-switchable (usage_cap/model_unknown/rate_limit), re-dispatch it on the next downgrade-chain model")
	focusHold := fs.Bool("focus-hold", false, "focus WIP backpressure (#3223): HOLD (refuse) a spawn that OPENS a new objective while the fleet is at/over the focusscore WIP cap, instead of the default WARN (advise + still spawn); continuation of an already-open objective is never held ($FLEET_DISPATCH_FOCUS_HOLD also enables)")
	codexLoopGate := fs.String("codex-loop-gate", dispatchCodexLoopGateDefaultThreshold(), "for live Codex workers, audit recent Codex sessions before spawn and refuse at threshold: loop|action|off (default: $FLEET_CODEX_LOOP_GATE or loop)")
	codexLoopGateSinceHours := fs.Float64("codex-loop-gate-since-hours", dispatchCodexLoopGateDefaultSinceHoursValue(), "with --codex-loop-gate, only scan Codex sessions modified within N hours (0 = all)")
	codexLoopGateLimit := fs.Int("codex-loop-gate-limit", dispatchCodexLoopGateDefaultLimitValue(), "with --codex-loop-gate, maximum newest Codex sessions to scan")
	placementEvidence := fs.Bool("placement-evidence", false, "record #5416 placement evidence: write the .workclass/.zone sidecars beside each worker log and append the witness sweep's graded turn outcomes to the runs-directory journal (default off: no extra sidecars, no extra payload keys, no journal)")
	rungPlacement := fs.Bool("rung-placement", false, "arm the automatic placement ladder: grade the account roster's bound models from the turn journal and start an UNPINNED worker on the cheapest rung the evidence supports, and re-dispatch an underpowered attempt one rung up (default off; also requires $FLEET_DISPATCH_RUNG_ACCOUNTS to declare which accounts this backend can dial)")
	accountsRoster := fs.String("accounts-roster", "", "model-account roster consulted for placement zone ATTRIBUTION and grading only, never for dispatch (default: tools/model-accounts.json when it exists; with no roster nothing is attributed rather than defaulted to a rung)")
	hostProbeShellReuse := fs.Bool("host-probe-shell-reuse", dispatchHostProbeShellReuseDefault(), "run this tick's Windows host probes (process scan, free RAM, worker rows, codex rows) on ONE warm racked PowerShell instead of one process -- and one ConPTY/conhost -- per probe (#3405/#3153); defaults on for Windows and off for every other GOOS, whose probes never reach PowerShell; =false puts every probe back on the historical one-shot spawn")
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
	var capacityTargets []modelroute.CapacityTarget
	if strings.TrimSpace(*capacityTargetsPath) != "" {
		b, err := os.ReadFile(*capacityTargetsPath)
		if err != nil {
			fmt.Fprintln(stderr, "fak dispatch tick:", err)
			return dispatchTickOptions{}, false, 1
		}
		if err := json.Unmarshal(b, &capacityTargets); err != nil {
			fmt.Fprintln(stderr, "fak dispatch tick: capacity targets:", err)
			return dispatchTickOptions{}, false, 2
		}
	}
	opts := dispatchTickOptions{
		Workspace:               root,
		MaxWorkers:              *maxWorkers,
		WorkKind:                wk,
		Lane:                    strings.TrimSpace(*lane),
		TargetIssue:             *targetIssue,
		LeaseID:                 strings.TrimSpace(*leaseID),
		LeaseTree:               splitCommaList(*leaseTree),
		Backend:                 b,
		Goal:                    goalID,
		GoalProfile:             profile,
		ExcludeLanes:            splitCommaList(*excludeLane),
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
		CapacityReason:          strings.TrimSpace(*capacityReason),
		CapacityFrom:            strings.TrimSpace(*capacityFrom),
		CapacityTargets:         capacityTargets,
		RequiredModelB:          *requiredModelB,
		RequiredContext:         *requiredContext,
		PinWorkerModel:          *pinWorkerModel || dispatchBoolValue(os.Getenv("FLEET_DISPATCH_PIN_MODEL")),
		AcceptanceArtifact:      firstString(strings.TrimSpace(*acceptanceArtifact), strings.TrimSpace(os.Getenv("FLEET_MODEL_ACCEPTANCE"))),
		AcceptanceOverride:      strings.TrimSpace(*acceptanceOverride),
		ModelDowngrade:          *modelDowngrade || dispatchBoolValue(os.Getenv("FLEET_DISPATCH_MODEL_DOWNGRADE")),
		FocusHold:               *focusHold || dispatchBoolValue(os.Getenv("FLEET_DISPATCH_FOCUS_HOLD")),
		CodexLoopGate:           strings.TrimSpace(*codexLoopGate),
		CodexLoopGateSinceHours: maxFloat64(0, *codexLoopGateSinceHours),
		CodexLoopGateLimit:      *codexLoopGateLimit,
		PlacementEvidence:       *placementEvidence,
		RungPlacement:           *rungPlacement,
		AccountsRoster:          strings.TrimSpace(*accountsRoster),
	}
	// The CLI always DECLARES the shell-reuse setting, because the flag's own default is
	// already the host default -- so a bare `fak dispatch tick` and a programmatic tick that
	// leaves the field nil resolve to the same posture, and an explicit
	// --host-probe-shell-reuse=false survives as a real declaration rather than reading as
	// "undeclared". Copied out of the flag set so opts owns the value.
	reuseHostProbeShell := *hostProbeShellReuse
	opts.HostProbeShellReuse = &reuseHostProbeShell
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
	snap := opts.DiscoverySnapshot
	if snap == nil {
		snap = scanRunsSnapshot(runsDir, time.Now())
	}
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
func dispatchTickSeatSelection(root, workKind, product, selectedTag string) (dispatchtick.SeatSelection, bool) {
	rows, err := dispatchReadAccountRoster(root)
	if err != nil {
		return dispatchtick.SeatSelection{}, false
	}
	route := dispatchtick.RouteAccount(dispatchtick.AccountRouteInput{Rows: rows, Product: product, WorkKind: workKind})
	selection := route.SeatSelection
	// An explicit --account override remains authoritative. Keep the ranked roster,
	// but say plainly when the launched winner differs from RouteAccount's winner.
	if selectedTag != "" && selectedTag != selection.WinnerTag {
		selection.WinnerTag = selectedTag
		selection.WinnerReason = "explicit account override"
		selection.Summary = fmt.Sprintf("picked %s over %d (explicit account override)", selectedTag, max(len(selection.Candidates)-1, 0))
	}
	return selection, len(selection.Candidates) > 0
}

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
	// Opt-in automatic placement ladder (--rung-placement, #5416 track E): grade
	// the roster's bound models from the turn journal and start this worker on the cheapest
	// rung the evidence supports, instead of the seat's vendor default. Lowest precedence of
	// all — it can only fill a seat default — and it runs BEFORE the capacity reroute and the
	// preventive placement gate below, so a live wall and a shape mismatch both still overrule
	// it. Seam off adds no payload key, so a default tick is byte-identical. The result is
	// adopted only on the no-skip branch, so a named refusal cannot move a worker's model no
	// matter what a future edit to the resolver returns alongside its reason.
	//
	// Turning the seam on is not sufficient: the operator must ALSO declare which roster
	// accounts these seats can dial (FLEET_DISPATCH_RUNG_ACCOUNTS), because a model id is the
	// whole launch instruction and a rung this backend cannot reach is a walled slot rather
	// than a cheaper one. Undeclared places nothing, and says so.
	if pinned, rungSkip := applyRungPlacement(root, labels, modelPolicy); rungSkip != "" {
		payload["rung_pin_skipped"] = rungSkip
	} else {
		modelPolicy = pinned
	}
	// The ladder's other direction (#5416 track D): when the target's last finished attempt RAN
	// the work on its rung and failed its own tests, re-dispatch it one rung up instead of
	// re-running the same underpowered attempt. Only an underpowered outcome earns a rung — a
	// guard refusal never does, and a transport wall is Layer-2's business below — and the climb
	// is bounded by an operator-declared ceiling and a per-item budget counted from a durable
	// ledger. It runs immediately after the placement above because it raises THAT decision (and
	// an unpinned seat default), and before the capacity reroute and the shape gate because a
	// live wall and a shape mismatch still overrule a rung that was bought.
	//
	// Nothing here is speculative: the debit is on disk before the pin exists, so an escalated
	// launch always has a recorded purchase behind it. Seam off adds no payload key.
	if raised, esc, escSkip := applyRungEscalation(root, opts.Live, target, labels, witnessRecords, modelPolicy); escSkip != "" {
		payload["rung_escalation_skipped"] = escSkip
	} else if esc != nil {
		payload["rung_escalation"] = esc.Map()
		modelPolicy = raised
	}
	if opts.CapacityReason != "" {
		reroute := modelroute.RerouteCapacity(opts.CapacityFrom, modelroute.CapacitySignal{Blocked: true, Reason: opts.CapacityReason, RequiredModelB: opts.RequiredModelB, RequiredContext: opts.RequiredContext}, opts.CapacityTargets)
		payload["capacity_reroute"] = reroute
		if reroute.Rerouted {
			modelPolicy.Model = reroute.To.Model
			modelPolicy.Source = "capacity_reroute"
		}
	}
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
	// #5416 tracks E+F: resolve the two facts a later witness sweep cannot reconstruct —
	// what class of work an operator DECLARED this issue to be, and which placement rung
	// serves the model this slot is about to run. Both are point-in-time (labels get
	// re-tagged, rosters get re-bound), so they are resolved here, beside the decision they
	// describe. Opt-in: an unconfigured tick adds no payload keys at all.
	if dispatchPlacementEvidenceEnabled() {
		recordDispatchPlacementEvidence(root, labels, launch.Model, payload)
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
	// #5416 tracks D/E/F: publish this tick's placement settings to the package seams the
	// placement, escalation, sidecar, and journal helpers read. They are DECLARED on the tick's
	// own config surface (--placement-evidence / --rung-placement / --accounts-roster) rather
	// than read out of the process environment, which is the CONFIG_NOT_ENV rule
	// internal/envconfiglint ratchets: an env var is for a secret, a behavioral setting belongs
	// where `--help` can name it and a caller can set it per invocation. Written here, beside
	// dispatchTickView and for the same reason — the readers hang off several call chains whose
	// many stubs keep their signature. A programmatic tick (wave / sweep / garden) that leaves
	// them zero gets the pre-seam posture: both switches off, no roster override.
	dispatchPlacementEvidence = opts.PlacementEvidence
	dispatchRungPlacement = opts.RungPlacement
	dispatchAccountsRoster = opts.AccountsRoster

	// #3405: publish this tick's host-probe shell-reuse setting into the probe seam BEFORE
	// preflight, which is what actually runs the probes. Without this call the whole reuse
	// spine is inert code -- landed, tested, and never reached -- so every Windows tick keeps
	// paying one `powershell` process, and one ConPTY/conhost, per probe (#3153). The arm
	// hands back the teardown, deferred here rather than folded into `finish` so that the
	// early error returns above the funnel cannot leak a warm console either: no console
	// outlives the tick that opened it.
	closeHostProbeShells := dispatchArmHostProbeShellReuse(opts.HostProbeShellReuse)
	defer closeHostProbeShells()

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
		// #5416 tracks E+F: fold this sweep's finished slots into durable capability
		// evidence. This is the caller the producer and the journal were missing — without
		// it every pure piece was correct and nothing on a live fleet ever wrote a row.
		// Opt-in, live-only, and fail-open: the accounting lands in the payload either way,
		// so an operator sees how much of the sweep became evidence and how much could not.
		if dispatchPlacementEvidenceEnabled() {
			if ev := appendDispatchTurnOutcomes(runsDir, witnessRecords); len(ev) > 0 {
				witnessedSlots["turn_evidence"] = ev
			}
		}
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
	if selection, ok := dispatchTickSeatSelection(root, opts.WorkKind, dispatchtick.ProductForBackend(opts.Backend), account.Tag); ok {
		payload["seat_selection"] = selection
	}
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
		// #3405: report the reuse dividend for the tick that earned it, read here (inside the
		// funnel) while the rack is still alive -- the deferred teardown drops it on the way
		// out. tasks_run is the no-reuse cost this tick would have paid (one process per
		// probe, the before), cold_spawns is what it actually paid (the after), and
		// spawns_avoided is the difference: the ConPTY/conhost creations that did not happen.
		// A tick that ran no warm task at all -- the spine off, or any non-Windows GOOS, whose
		// probes never touch PowerShell -- adds no key, so those payloads stay byte-identical.
		if st := dispatchHostProbeRackStats(); st.TasksRun > 0 {
			p["host_probe_shells"] = map[string]any{
				"cold_spawns":       st.ColdSpawns,
				"warm_reuses":       st.WarmReuses,
				"tasks_run":         st.TasksRun,
				"unhealthy_retired": st.UnhealthyRetired,
				"spawns_avoided":    st.SpawnsAvoided(),
			}
		}
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
	contract := dispatchtick.ParseObjectiveContract(dispatchMapString(promptRec, "body"))
	if contract.Refusal != "" {
		payload["ok"] = false
		payload["action"] = "objective_contract_refused"
		payload["verdict"] = contract.Refusal
		payload["reason"] = "issue objective has no attached Witness scorer"
		payload["objective_contract"] = contract
		return finish(payload), nil
	}
	dispatchStampMs(timings, "prompt", tPrompt)
	promptChars := dispatchMapInt(promptRec, "prompt_chars")
	labels := dispatchStringSlice(promptRec["labels"])
	payload["prompt_chars"] = promptChars
	payload["issue_title"] = dispatchMapString(promptRec, "title")
	payload["development_branch"] = dispatchMapString(promptRec, "development_branch")
	payload["child_curve"] = dispatchtick.ChildCurve(root, target)
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

	if !applyDispatchModelAcceptance(opts.AcceptanceArtifact, launch.Model, labels, time.Now().UTC(), opts.AcceptanceOverride, payload) {
		return finish(payload), nil
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
	// #4324: persist the fencing token this lane lease was acquired under, so the async
	// witness sweep — a LATER tick process that never saw the acquire — can prove the
	// lease is still this worker's and hand the lane back the moment the worker exits
	// normally, instead of stranding it for the whole TTL. Nothing is written for a
	// refused/fail-open acquire or a zero generation; such a lease keeps its TTL.
	writeDispatchLeaseFenceSidecar(spawned.Log, lease)
	// #5416 tracks E+F: persist the work class and placement rung resolved at prepare time
	// beside the log, so the witness sweep reads what was true AT LAUNCH rather than
	// re-deriving a present-tense answer about a finished slot. No-op when either could not
	// be named, and when the seam is off nothing was resolved and nothing is written.
	writeDispatchPlacementSidecars(spawned.Log, payload)
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
		return map[string]any{"acquired": false, "refused": false, "id": id, "holder": holder, "fail_open": true, "error": liveErr.Error(), "tree": tree, "lane": lane, "lane_kind": leaseref.ArbiterLaneKind}
	}
	// One admission contract for every surface (internal/regionadmit): the same
	// decision `fak loop drive` and `fak loop region` run — tree geometry PLUS
	// dos.toml lane semantics (a named lane serializes; an exclusive lane runs
	// alone). A missing taxonomy degrades to the historical geometry-only check.
	tax, taxErr := regionadmit.LoadTaxonomy(root)
	if taxErr != nil {
		tax = regionadmit.Taxonomy{}
	}
	// Structured lane fields for the refusal ledger (#4322): the canonical lane,
	// its kind (a refs/fak/locks lease is a tree-scoped CLUSTER lease — exactly
	// leaseref's arbiter projection), and its serialization mode. Stamped onto
	// every result map so per-lane collision rate and the WIP-vs-lease split are
	// computable straight from loops.jsonl instead of being regex-scraped out of
	// the summary prose. Additive: existing readers of the lease map are unaffected.
	laneMode := "shared"
	if tax.Exclusive[lane] {
		laneMode = "exclusive"
	}
	// SelfID stays empty on purpose: a live lease under this very id (a previous
	// worker on this lane, still running) must refuse here exactly as it always
	// has — with a pinned FAK_LEASE_OWNER the fence would otherwise read the new
	// tick as the SAME holder and silently renew, double-spawning the lane.
	req := regionadmit.Request{Actor: holder, Lane: lane, Tree: tree}
	dec := regionadmit.Decide(req, regionLeases(live), tax)
	if !dec.Admit {
		refusal := map[string]any{
			"acquired":  false,
			"refused":   true,
			"id":        id,
			"holder":    holder,
			"reason":    dec.Reason,
			"rung":      dec.Rung,
			"detail":    dec.Detail,
			"tree":      tree,
			"lane":      lane,
			"lane_kind": leaseref.ArbiterLaneKind,
			"mode":      laneMode,
		}
		// #5505: the tick's refusal takes a PLACE IN LINE instead of evaporating. Until now a
		// LANE_LEASE_HELD tick recorded that it lost and nothing else, so the next tick re-raced
		// from scratch and whoever polled first after a release won — a tick that had been
		// refused for four hours had exactly the same odds as one that arrived 200ms ago. The
		// ticket's id is stable across retries (actor+lane+resolved tree), so a returning tick
		// REFRESHES its ticket and keeps the enqueue clock it earned.
		//
		// The same plane `fak loop region` mints on, through the same helper — one waiter list,
		// so an operator's `fak loop region --lane X` and this tick queue in ONE line rather than
		// two private ones. Class is `loop` because this IS the background dispatch driver; that
		// is what keeps it from ranking ahead of a waiting operator.
		//
		// It runs AFTER the verdict and can only ADD a report key: the decision above and the
		// refused:true the caller branches on (dispatchTickLiveSpawn's LANE_LEASE_HELD, the host
		// enroll's) are already computed and are never read back from here. Every failure inside
		// the helper is swallowed into a nil report, so an unwritable queue degrades the payload
		// and never the decision.
		//
		// No `--no-queue` twin is wired here, unlike the loop-region verb: that flag exists for a
		// PURE QUERY (an operator asking "may I?" who wants to leave no trace), and this function
		// has no query-shaped caller — the dry-run WOULD_SPAWN/WOULD_ENROLL paths return before
		// the acquire, and the scorecard's deliberate-collision probe runs against an isolated
		// temp repo that is removed with its queue. Every caller that reaches here genuinely
		// wants the region, so every caller genuinely wants its place in line.
		if q := loopRegionEnqueue(root, req, tax, live, leasequeue.ClassLoop, now).payload(); q != nil {
			refusal["queue"] = q
		}
		return refusal
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
		return map[string]any{"acquired": false, "refused": false, "id": id, "holder": holder, "fail_open": true, "error": err.Error(), "tree": tree, "lane": lane, "lane_kind": leaseref.ArbiterLaneKind, "mode": laneMode}
	}
	if verdict.OK {
		// The waiter got in, so it gives up the place it was holding (#5505) — otherwise a tick
		// that finally acquired would keep a reservation that ranks ahead of the peers still
		// waiting behind it for the whole ticket TTL. Best-effort and silent, exactly like the
		// enqueue: the acquisition already succeeded and nothing below re-reads the queue.
		loopRegionDequeue(root, req, tax)
		return map[string]any{"acquired": true, "refused": false, "id": id, "holder": holder, "generation": written.Generation, "tree": tree, "lane": lane, "lane_kind": leaseref.ArbiterLaneKind, "mode": laneMode}
	}
	return map[string]any{"acquired": false, "refused": true, "id": id, "holder": holder, "reason": string(verdict.Reason), "detail": verdict.Detail, "tree": tree, "lane": lane, "lane_kind": leaseref.ArbiterLaneKind, "mode": laneMode}
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
