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
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/branchrole"
	"github.com/anthony-chaudhary/fak/internal/dispatchorder"
	"github.com/anthony-chaudhary/fak/internal/dispatchtick"
	"github.com/anthony-chaudhary/fak/internal/leaseref"
	"github.com/anthony-chaudhary/fak/internal/loopmgr"
)

type dispatchTickOptions struct {
	Workspace      string
	MaxWorkers     int
	WorkKind       string
	Lane           string
	TargetIssue    int
	LeaseID        string
	LeaseTree      []string
	Backend        string
	ExcludeLanes   []string
	Live           bool
	Refresh        bool
	PreferNewest   bool
	Generation     string
	CooldownMin    int
	WorkerTimeoutS int
	SpawnProbeS    float64
	LoopLedger     string
	RecordLoop     bool
	Account        *dispatchtick.Account
	Membership     *dispatchtick.Membership
}

type dispatchLanePick struct {
	Lane             string
	Numbers          []int
	ByLaneCount      map[string]int
	ByLaneStepBudget map[string]int
	ExcludedLanes    []string
	Tree             []string
	RouterError      string
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
const dispatchStartupBundleSidecarSuffix = ".startup.json"
const dispatchStartupBundleSchema = "fleet-worker-startup-bundle/1"

var dispatchResolvePIDRE = regexp.MustCompile(`^resolve-\d+-\d{8}-\d{6}\.pid$`)

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
	if ok, _ := payload["ok"].(bool); ok {
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
	backend := fs.String("backend", "claude", "worker backend (claude|opencode|codex)")
	excludeLane := fs.String("exclude-lane", "", "comma-separated lanes to drop from the busiest pick")
	live := fs.Bool("live", false, "actually spawn the issue-resolution worker")
	noRefresh := fs.Bool("no-refresh", false, "skip the per-tick account-registry refresh")
	preferNewest := fs.Bool("prefer-newest", false, "pick the NEWEST open issue on the lane first (default: oldest first, to drain the backlog)")
	generationFlag := fs.String("generation", "", "generation horizon to admit: now|next|second-next|future|all (default: now+next; only engages when a candidate carries a gen/* label)")
	cooldownMin := fs.Int("cooldown-min", dispatchtick.DefaultCooldownMinutes, "skip issues attempted within this many minutes (0 disables)")
	workerTimeoutS := fs.Int("worker-timeout-s", dispatchtick.DefaultWorkerTimeoutS, "worker lease TTL base in seconds (0 uses default)")
	spawnProbeS := fs.Float64("spawn-probe-s", dispatchtick.DefaultSpawnProbeS, "seconds to wait after spawn to catch immediate empty-log exits")
	loopLedger := fs.String("loop-ledger", "", "append this tick to a fak loop ledger (default: FAK_LOOP_LEDGER or .fak/loops.jsonl)")
	noLoopLedger := fs.Bool("no-loop-ledger", false, "disable loop-ledger append for this tick")
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
	opts := dispatchTickOptions{
		Workspace:      root,
		MaxWorkers:     *maxWorkers,
		WorkKind:       wk,
		Lane:           strings.TrimSpace(*lane),
		TargetIssue:    *targetIssue,
		LeaseID:        strings.TrimSpace(*leaseID),
		LeaseTree:      dispatchSplitCSV(*leaseTree),
		Backend:        b,
		ExcludeLanes:   dispatchSplitCSV(*excludeLane),
		Live:           *live,
		Refresh:        !*noRefresh,
		PreferNewest:   *preferNewest,
		Generation:     strings.TrimSpace(*generationFlag),
		CooldownMin:    *cooldownMin,
		WorkerTimeoutS: *workerTimeoutS,
		SpawnProbeS:    maxFloat64(0, *spawnProbeS),
		LoopLedger:     *loopLedger,
		RecordLoop:     !*noLoopLedger,
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

func evaluateDispatchTick(opts dispatchTickOptions, stderr io.Writer) (map[string]any, error) {
	root, err := filepath.Abs(opts.Workspace)
	if err != nil {
		return nil, err
	}
	runsDir := filepath.Join(root, dispatchtick.RunsDirName)
	if opts.WorkerTimeoutS <= 0 {
		opts.WorkerTimeoutS = dispatchtick.DefaultWorkerTimeoutS
	}

	reg := map[string]any{"skipped": true}
	if opts.Refresh {
		reg = dispatchRefreshRegistry(root, stderr)
	}

	pre, err := dispatchPreflight(root, stderr, opts.MaxWorkers, opts.WorkKind, dispatchtick.ProductForBackend(opts.Backend))
	if err != nil {
		return nil, err
	}
	preOK := dispatchMapString(pre, "verdict") == "SPAWN_OK"
	account := accountFromMap(mapAt(pre, "account"))
	if opts.Account != nil {
		account = *opts.Account
	}

	held := liveResolutionLanes(runsDir)
	exclude := map[string]bool{}
	for _, lane := range opts.ExcludeLanes {
		exclude[lane] = true
	}
	if opts.Lane == "" {
		for lane := range held {
			exclude[lane] = true
		}
	}
	pick, err := pickDispatchLane(root, stderr, opts.Lane, exclude, opts.PreferNewest, opts.Generation)
	if err != nil {
		return nil, err
	}
	liveIssueDetails := liveResolutionIssueDetails(runsDir)
	liveIssues := liveIssueSet(liveIssueDetails)
	cooled := recentlyAttemptedIssues(runsDir, opts.CooldownMin)
	cooldownStatus := cooldownIssueRows(runsDir, opts.CooldownMin)
	skip := map[int]bool{}
	for n := range liveIssues {
		skip[n] = true
	}
	for n := range cooled {
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

	startup := dispatchStartupBundle(root, opts, pre, account, pick, target, hasTarget, held, liveIssues, cooled, cooldownStatus)
	payload := map[string]any{
		"schema":           dispatchtick.Schema,
		"workspace":        root,
		"live":             opts.Live,
		"backend":          opts.Backend,
		"max_workers":      opts.MaxWorkers,
		"registry_refresh": reg,
		"preflight": map[string]any{
			"verdict":   dispatchMapString(pre, "verdict"),
			"reason":    dispatchMapString(pre, "reason"),
			"cap":       pre["cap"],
			"live":      pre["live"],
			"cap_terms": mapAt(pre, "cap_terms"),
		},
		"account":          dispatchtick.AccountSidecar(account),
		"lane":             pick.Lane,
		"lease_id":         firstString(opts.LeaseID, dispatchLaneLeaseID(pick.Lane)),
		"lease_tree":       append([]string(nil), pick.Tree...),
		"lane_issue_count": len(pick.Numbers),
		"lane_step_budget": pick.ByLaneStepBudget[pick.Lane],
		"cooled_recently":  sortedSet(cooled),
		"cooldown_status":  cooldownStatus,
		"target_issue":     nil,
		"already_live":     sortedSet(liveIssues),
		"held_lanes":       sortedStringSet(held),
		"startup_bundle":   startup,
		"stale_base":       mapAt(startup, "stale_base"),
	}
	if hasTarget {
		payload["target_issue"] = target
	}

	finish := func(p map[string]any) map[string]any {
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
		return finish(payload), nil
	}
	if pick.Lane == "" {
		// All-self-source edge case (#1397): the auto-pick found candidate lanes but
		// every one was held as fak's own running source (cmd/** + internal/**) under
		// guard, so `chosen` stayed "". This is NOT an empty/error router -- the backlog
		// is real, it is just all structurally unshippable by a self-guarded worker. Say
		// so honestly with the SELF_MODIFY_HOLD vocabulary (over the whole held set)
		// instead of the misleading "router empty/error" NO_LANE, so the operator routes
		// the work to an unguarded/operator or worktree-isolated path (#1334).
		if len(pick.SelfSourceHeld) > 0 {
			payload["ok"] = false
			payload["action"] = "self_modify_hold"
			payload["verdict"] = "SELF_MODIFY_HOLD"
			payload["self_modify_held_lanes"] = append([]string(nil), pick.SelfSourceHeld...)
			payload["reason"] = fmt.Sprintf("every candidate lane (%s) is rooted in fak's own running source (cmd/** or internal/**): a guarded %s worker can investigate but never SHIP such an edit (the guard refuses with reason=SELF_MODIFY), so the whole eligible backlog is operator-gated -- route it to an unguarded/operator or worktree-isolated path (#1334), not a self-guarded worker", strings.Join(pick.SelfSourceHeld, ", "), opts.Backend)
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
		if live, ok := liveResolutionTreeCollision(runsDir, pick.Tree); ok {
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
		payload["reason"] = fmt.Sprintf("every open issue on lane %q is live or cooling", pick.Lane)
		return finish(payload), nil
	}

	promptRec, err := dispatchPrompt(root, stderr, target, pick.Lane)
	if err != nil {
		return nil, err
	}
	promptChars := dispatchMapInt(promptRec, "prompt_chars")
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
	model := account.Model
	if opts.Backend != "opencode" && opts.Backend != "codex" {
		model = ""
	}
	preview, err := dispatchtick.BuildWorkerCommand(opts.Backend, dispatchtick.PreviewPrompt(target, promptChars), model)
	if err != nil {
		return nil, err
	}
	launchPreview, guardedPreview := guardedDispatchCommand(root, pick.Lane, opts.Backend, preview)
	payload["command"] = dispatchtick.LaunchCommandShape(preview, root, account)
	payload["launch_command"] = dispatchtick.LaunchCommandShape(launchPreview, root, account)
	payload["guarded"] = guardedPreview

	// Self-modify pre-route (#1397): a GUARDED worker aimed at fak's own running source
	// (cmd/** or internal/**) can investigate but never SHIP -- the guard refuses an edit
	// to the binary adjudicating it (reason=SELF_MODIFY), so the worker burns turns and
	// lands 0 commits (#1338's evidence). The hold fires on TWO signals: the lane tree is
	// self-source (a correctly-routed cmd/internal lane), OR the target issue's own text
	// references cmd/** or internal/** even though it routed to a SAFE lane -- the
	// MIS-ROUTE the router's path extractor hides, because it only sees fak/-prefixed
	// paths, so a `fix(dispatch):` issue whose real work is in cmd/fak aliases to the
	// tools lane carrying zero extracted paths (#1338/#1397 are themselves this case).
	// Hold BEFORE both the dry-run plan and the live spawn so the loop honest-STOPs and
	// the operator routes it to an unguarded/operator or worktree-isolated path (#1334)
	// instead. The guard wrapper and account are already resolved above, so the witness
	// names exactly why -- this is a pre-route, not a guard/account failure. An unguarded
	// worker (FLEET_DOGFOOD_GUARD=0, or a worktree-isolated path) never trips this.
	issueText := dispatchMapString(promptRec, "title") + "\n" + dispatchMapString(promptRec, "body")
	if held, tree := dispatchtick.SelfModifyHoldForPick(guardedPreview, pick.Tree, issueText); held {
		payload["ok"] = false
		payload["action"] = "self_modify_hold"
		payload["verdict"] = "SELF_MODIFY_HOLD"
		payload["self_modify_tree"] = tree
		payload["reason"] = fmt.Sprintf("issue #%d targets fak's own running source (lane %q, tree %q): a guarded %s worker can investigate but never SHIP an edit to cmd/** or internal/** (the guard refuses with reason=SELF_MODIFY), so this work is operator-gated -- route it to an unguarded/operator or worktree-isolated path (#1334), not a self-guarded worker", target, pick.Lane, tree, opts.Backend)
		return finish(payload), nil
	}

	if !opts.Live {
		payload["ok"] = true
		payload["action"] = "would_spawn"
		payload["verdict"] = "WOULD_SPAWN"
		payload["reason"] = fmt.Sprintf("safe to spawn 1 %s issue-resolution worker on #%d (lane %q) under account %q", opts.Backend, target, pick.Lane, account.Tag)
		return finish(payload), nil
	}

	return dispatchTickLiveSpawn(root, runsDir, opts, pick, account, model, target, promptRec, payload, finish)
}

func liveIssueSet(details map[int]dispatchLiveScope) map[int]bool {
	out := map[int]bool{}
	for issue := range details {
		out[issue] = true
	}
	return out
}

func inFlightDuplicateForPick(opts dispatchTickOptions, numbers []int, hasTarget bool, details map[int]dispatchLiveScope) (dispatchLiveScope, bool) {
	if opts.TargetIssue > 0 {
		live, ok := details[opts.TargetIssue]
		return live, ok
	}
	if hasTarget {
		return dispatchLiveScope{}, false
	}
	for _, issue := range numbers {
		if live, ok := details[issue]; ok {
			return live, true
		}
	}
	return dispatchLiveScope{}, false
}

func dispatchLiveScopeMap(live dispatchLiveScope) map[string]any {
	return map[string]any{
		"issue":    live.Issue,
		"lane":     live.Lane,
		"tree":     append([]string(nil), live.Tree...),
		"log":      live.Log,
		"pid":      live.PID,
		"worker":   live.Worker,
		"lease_id": live.LeaseID,
	}
}

// dispatchTickLiveSpawn performs the live spawn once every dry-run gate has passed: acquire
// the lane lease (refused → LANE_LEASE_HELD), build the guarded worker command + env, spawn
// the issue-resolution worker, and record the SPAWNED / SPAWN_FAILED payload. It mutates and
// returns the shared payload through finish, mirroring the dry-run return sites it splits off.
func dispatchTickLiveSpawn(root, runsDir string, opts dispatchTickOptions, pick dispatchLanePick, account dispatchtick.Account, model string, target int, promptRec, payload map[string]any, finish func(map[string]any) map[string]any) (map[string]any, error) {
	leaseID := firstString(opts.LeaseID, dispatchLaneLeaseID(pick.Lane))
	lease := acquireDispatchLaneLease(root, leaseID, pick.Lane, pick.Tree, opts.WorkerTimeoutS+dispatchtick.LeaseTTLMarginS)
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
	command, err := dispatchtick.BuildWorkerCommand(opts.Backend, prompt, model)
	if err != nil {
		return nil, err
	}
	launchCommand, guarded := guardedDispatchCommand(root, pick.Lane, opts.Backend, command)
	if guarded {
		augmentGuardEnvDefaults()
	}
	env, err := dispatchWorkerEnv(opts.Backend, pick.Lane, root, runsDir, account)
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
	spawned, err := spawnDispatchIssueWorker(launchCommand, env, root, runsDir, target, pick.Lane, opts.Backend, leaseID, pick.Tree, account, opts.Membership, baseSHA, opts.SpawnProbeS)
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
	if early, ok := spawned.EarlyExit["silent"].(bool); ok && early {
		payload["ok"] = false
		payload["action"] = "spawn_failed"
		payload["verdict"] = "SPAWN_FAILED"
		payload["reason"] = fmt.Sprintf("%s worker pid %d for #%d exited immediately and produced an empty log", opts.Backend, spawned.PID, target)
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
		configureDispatchHelperCommand(cmd)
		out, err := cmd.Output()
		cancel()
		if obj, perr := lastJSONObject(out); perr == nil {
			return obj, nil
		}
		if err != nil {
			lastErr = err
		} else {
			lastErr = errors.New("no JSON object in helper output")
		}
	}
	return nil, fmt.Errorf("python helper %s (tried %s): %w", strings.Join(args, " "), strings.Join(interps, ", "), lastErr)
}

func acquireDispatchLaneLease(root, id, lane string, tree []string, ttlS int) map[string]any {
	holder := dispatchLeaseHolder()
	store := leaseref.NewInDir(root)
	now := time.Now()
	live, _, liveErr := store.Live(context.Background(), now)
	if liveErr != nil {
		return map[string]any{"acquired": false, "refused": false, "id": id, "holder": holder, "fail_open": true, "error": liveErr.Error(), "tree": tree}
	}
	for _, held := range live {
		if dispatchorder.TreesOverlap(tree, held.TreeGlobs) {
			return map[string]any{
				"acquired": false,
				"refused":  true,
				"id":       id,
				"holder":   holder,
				"reason":   dispatchorder.ReasonCollisionRisk,
				"detail":   fmt.Sprintf("requested tree %v overlaps live lease %s tree %v", tree, held.ID, held.TreeGlobs),
				"tree":     tree,
			}
		}
	}
	rec := leaseref.Record{ID: id, TreeGlobs: tree, Holder: holder, TTLSeconds: int64(ttlS)}
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

func dispatchWorkerEnv(backend, lane, root, runsDir string, account dispatchtick.Account) (map[string]string, error) {
	env := envMap(os.Environ())
	env["DISPATCH_WORKSPACE"] = root
	env["DISPATCH_LANE"] = lane
	env["DISPATCH_BACKEND"] = backend
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
	return dispatchtick.GuardedLaunchCommand(command, fakBin, lane, backend, root, baseURL)
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

func spawnDispatchIssueWorker(command []string, env map[string]string, cwd, runsDir string, issue int, lane, backend, leaseID string, tree []string, account dispatchtick.Account, membership *dispatchtick.Membership, baseSHA string, probeS float64) (dispatchSpawnResult, error) {
	if len(command) == 0 {
		return dispatchSpawnResult{}, errors.New("empty worker command")
	}
	if err := os.MkdirAll(runsDir, 0o755); err != nil {
		return dispatchSpawnResult{}, err
	}
	stamp := time.Now().UTC().Format("20060102-150405")
	outLog := filepath.Join(runsDir, fmt.Sprintf("resolve-%d-%s.log", issue, stamp))
	exe := command[0]
	if p, err := exec.LookPath(exe); err == nil {
		exe = p
	}
	fh, err := os.OpenFile(outLog, os.O_CREATE|os.O_TRUNC|os.O_WRONLY, 0o644)
	if err != nil {
		return dispatchSpawnResult{}, err
	}
	fmt.Fprintf(fh, "# fak-spawn %s issue=%d lane=%s backend=%s argv0=%s\n", stamp, issue, lane, backend, filepath.Base(exe))
	_ = fh.Sync()
	devNull, _ := os.Open(os.DevNull)
	cmd := exec.Command(exe, command[1:]...)
	cmd.Dir = cwd
	cmd.Env = envSliceFromMap(env)
	if devNull != nil {
		defer devNull.Close()
		cmd.Stdin = devNull
	}
	cmd.Stdout = fh
	cmd.Stderr = fh
	configureDispatchSpawn(cmd)
	if err := cmd.Start(); err != nil {
		_ = fh.Close()
		return dispatchSpawnResult{}, err
	}
	_ = fh.Close()

	stem := strings.TrimSuffix(outLog, filepath.Ext(outLog))
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
		}
		return rec
	case <-time.After(time.Duration(waitS * float64(time.Second))):
		return map[string]any{"checked": true, "alive": true, "wait_s": waitS}
	}
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

func dispatchStartupBundle(root string, opts dispatchTickOptions, pre map[string]any, account dispatchtick.Account, pick dispatchLanePick, target int, hasTarget bool, held map[string]bool, liveIssues map[int]bool, cooled map[int]bool, cooldownStatus []map[string]any) map[string]any {
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
		"route":     route,
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
			"id":   firstString(opts.LeaseID, dispatchLaneLeaseID(pick.Lane)),
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
	cmd := exec.Command("git", args...)
	cmd.Dir = root
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
	cmd := exec.Command("git", "status", "--porcelain=v1")
	cmd.Dir = root
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
	loopID := "issue-resolve-dispatch/" + firstString(dispatchMapString(payload, "backend"), "claude")
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
	return fmt.Sprintf("resolve-tick-%s-%s", firstString(dispatchMapString(payload, "backend"), "claude"), time.Now().UTC().Format("20060102T150405Z"))
}

func chooseStatus(cond bool, yes, no loopmgr.RunStatus) loopmgr.RunStatus {
	if cond {
		return yes
	}
	return no
}

func currentGitSHA(root string) string {
	cmd := exec.Command("git", "rev-parse", "HEAD")
	cmd.Dir = root
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

func accountFromMap(m map[string]any) dispatchtick.Account {
	return dispatchtick.Account{
		Tag:   dispatchMapString(m, "tag"),
		Tier:  m["tier"],
		Model: dispatchMapString(m, "model"),
		Dir:   firstString(dispatchMapString(m, "dir"), dispatchMapString(m, "config_dir")),
	}
}

func dispatchSplitCSV(s string) []string {
	var out []string
	for _, part := range strings.Split(s, ",") {
		part = strings.TrimSpace(part)
		if part != "" {
			out = append(out, part)
		}
	}
	return out
}

func dispatchtickWorkKind(backend string) string {
	b, err := dispatchtick.NormalizeBackend(backend)
	if err != nil {
		return dispatchtick.DefaultWorkKind("claude")
	}
	return dispatchtick.DefaultWorkKind(b)
}

func stringSlice(v any) []string {
	var out []string
	if arr, ok := v.([]any); ok {
		for _, x := range arr {
			if s, ok := x.(string); ok {
				out = append(out, s)
			}
		}
	}
	return out
}

func envMap(kvs []string) map[string]string {
	out := map[string]string{}
	for _, kv := range kvs {
		if i := strings.IndexByte(kv, '='); i >= 0 {
			out[kv[:i]] = kv[i+1:]
		}
	}
	return out
}

func envSliceFromMap(env map[string]string) []string {
	out := make([]string, 0, len(env))
	for k, v := range env {
		out = append(out, k+"="+v)
	}
	sort.Strings(out)
	return out
}

func mapAt(m map[string]any, key string) map[string]any {
	if v, ok := m[key].(map[string]any); ok {
		return v
	}
	return map[string]any{}
}

func dispatchMapString(m map[string]any, key string) string {
	if v, ok := m[key].(string); ok {
		return v
	}
	return ""
}

func dispatchMapBool(m map[string]any, key string) bool {
	v, _ := m[key].(bool)
	return v
}

func dispatchStringValue(v any) string {
	if s, ok := v.(string); ok {
		return strings.TrimSpace(s)
	}
	return ""
}

func dispatchBoolValue(v any) bool {
	switch x := v.(type) {
	case bool:
		return x
	case string:
		switch strings.ToLower(strings.TrimSpace(x)) {
		case "1", "true", "yes", "y", "on":
			return true
		}
	}
	return false
}

func dispatchIntValue(v any) int {
	if n := intPtrFromAny(v); n != nil {
		return *n
	}
	return 0
}

func dispatchMapInt(m map[string]any, key string) int {
	switch v := m[key].(type) {
	case int:
		return v
	case int64:
		return int(v)
	case float64:
		return int(v)
	case json.Number:
		n, _ := v.Int64()
		return int(n)
	}
	return 0
}

func intPtrFromAny(v any) *int {
	switch x := v.(type) {
	case int:
		return &x
	case int64:
		n := int(x)
		return &n
	case float64:
		n := int(x)
		return &n
	case json.Number:
		if n, err := x.Int64(); err == nil {
			i := int(n)
			return &i
		}
	case string:
		if n, err := strconv.Atoi(strings.TrimSpace(x)); err == nil {
			return &n
		}
	}
	return nil
}

func anySlice(v any) []any {
	if arr, ok := v.([]any); ok {
		return arr
	}
	if arr, ok := v.([]map[string]any); ok {
		out := make([]any, 0, len(arr))
		for _, item := range arr {
			out = append(out, item)
		}
		return out
	}
	return nil
}

func nonEmptyLines(s string) []string {
	rows := []string{}
	for _, line := range strings.Split(s, "\n") {
		line = strings.TrimRight(line, "\r")
		if strings.TrimSpace(line) != "" {
			rows = append(rows, line)
		}
	}
	return rows
}

func sortedSet(in map[int]bool) []int {
	out := make([]int, 0, len(in))
	for n := range in {
		out = append(out, n)
	}
	sort.Ints(out)
	return out
}

func sortedStringSet(in map[string]bool) []string {
	out := make([]string, 0, len(in))
	for s := range in {
		out = append(out, s)
	}
	sort.Strings(out)
	return out
}

func dispatchAnyOSBase(path string) string {
	path = strings.TrimRight(strings.ReplaceAll(strings.TrimSpace(path), "\\", "/"), "/")
	if path == "" {
		return ""
	}
	if i := strings.LastIndex(path, "/"); i >= 0 {
		return path[i+1:]
	}
	return path
}

func parseAccountTier(s string) any {
	if s == "" {
		return nil
	}
	if n, err := strconv.Atoi(s); err == nil {
		return n
	}
	return s
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
