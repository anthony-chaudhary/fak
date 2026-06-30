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

	"github.com/anthony-chaudhary/fak/internal/dispatchtick"
	"github.com/anthony-chaudhary/fak/internal/leaseref"
	"github.com/anthony-chaudhary/fak/internal/loopmgr"
)

type dispatchTickOptions struct {
	Workspace      string
	MaxWorkers     int
	WorkKind       string
	Lane           string
	Backend        string
	ExcludeLanes   []string
	Live           bool
	Refresh        bool
	PreferNewest   bool
	CooldownMin    int
	WorkerTimeoutS int
	SpawnProbeS    float64
	LoopLedger     string
	RecordLoop     bool
	Account        *dispatchtick.Account
	Membership     *dispatchtick.Membership
}

type dispatchLanePick struct {
	Lane          string
	Numbers       []int
	ByLaneCount   map[string]int
	ExcludedLanes []string
	Tree          []string
	RouterError   string
}

type dispatchSpawnResult struct {
	PID        int            `json:"pid"`
	Log        string         `json:"log"`
	Issue      int            `json:"issue"`
	Lane       string         `json:"lane"`
	Backend    string         `json:"backend"`
	Account    map[string]any `json:"account,omitempty"`
	Membership any            `json:"membership,omitempty"`
	EarlyExit  map[string]any `json:"early_exit,omitempty"`
}

var dispatchResolveLogRE = regexp.MustCompile(`^resolve-(\d+)-.*\.log$`)
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
	lane := fs.String("lane", "", "explicit lane (default: busiest lane with an open issue)")
	backend := fs.String("backend", "claude", "worker backend (claude|opencode|codex)")
	excludeLane := fs.String("exclude-lane", "", "comma-separated lanes to drop from the busiest pick")
	live := fs.Bool("live", false, "actually spawn the issue-resolution worker")
	noRefresh := fs.Bool("no-refresh", false, "skip the per-tick account-registry refresh")
	preferNewest := fs.Bool("prefer-newest", false, "pick the NEWEST open issue on the lane first (default: oldest first, to drain the backlog)")
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
		Backend:        b,
		ExcludeLanes:   dispatchSplitCSV(*excludeLane),
		Live:           *live,
		Refresh:        !*noRefresh,
		PreferNewest:   *preferNewest,
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
	pick, err := pickDispatchLane(root, stderr, opts.Lane, exclude, opts.PreferNewest)
	if err != nil {
		return nil, err
	}
	liveIssues := liveResolutionIssues(runsDir)
	cooled := recentlyAttemptedIssues(runsDir, opts.CooldownMin)
	skip := map[int]bool{}
	for n := range liveIssues {
		skip[n] = true
	}
	for n := range cooled {
		skip[n] = true
	}
	target, hasTarget := dispatchtick.PickTargetIssue(pick.Numbers, skip)

	payload := map[string]any{
		"schema":           dispatchtick.Schema,
		"workspace":        root,
		"live":             opts.Live,
		"backend":          opts.Backend,
		"max_workers":      opts.MaxWorkers,
		"registry_refresh": reg,
		"preflight": map[string]any{
			"verdict": dispatchMapString(pre, "verdict"),
			"reason":  dispatchMapString(pre, "reason"),
			"cap":     pre["cap"],
			"live":    pre["live"],
		},
		"account":          dispatchtick.AccountSidecar(account),
		"lane":             pick.Lane,
		"lane_issue_count": len(pick.Numbers),
		"cooled_recently":  sortedSet(cooled),
		"target_issue":     nil,
		"already_live":     sortedSet(liveIssues),
		"held_lanes":       sortedStringSet(held),
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
		payload["ok"] = false
		payload["action"] = "no_lane"
		payload["verdict"] = "NO_LANE"
		payload["reason"] = "no lane has open issues (router empty/error)"
		return finish(payload), nil
	}
	if opts.Lane != "" && held[pick.Lane] {
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
	model := account.Model
	if opts.Backend != "opencode" && opts.Backend != "codex" {
		model = ""
	}
	preview, err := dispatchtick.BuildWorkerCommand(opts.Backend, dispatchtick.PreviewPrompt(target, promptChars), model)
	if err != nil {
		return nil, err
	}
	launchPreview, guardedPreview := guardedDispatchCommand(root, pick.Lane, opts.Backend, preview)
	payload["command"] = preview
	payload["launch_command"] = launchPreview
	payload["guarded"] = guardedPreview

	if !opts.Live {
		payload["ok"] = true
		payload["action"] = "would_spawn"
		payload["verdict"] = "WOULD_SPAWN"
		payload["reason"] = fmt.Sprintf("safe to spawn 1 %s issue-resolution worker on #%d (lane %q) under account %q", opts.Backend, target, pick.Lane, account.Tag)
		return finish(payload), nil
	}

	lease := acquireDispatchLaneLease(root, pick.Lane, pick.Tree, opts.WorkerTimeoutS+dispatchtick.LeaseTTLMarginS)
	payload["lease"] = lease
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
	spawned, err := spawnDispatchIssueWorker(launchCommand, env, root, runsDir, target, pick.Lane, opts.Backend, account, opts.Membership, baseSHA, opts.SpawnProbeS)
	if err != nil {
		payload["ok"] = false
		payload["action"] = "spawn_failed"
		payload["verdict"] = "SPAWN_FAILED"
		payload["reason"] = err.Error()
		recordDispatchPayload(runsDir, opts.Backend, payload)
		return finish(payload), nil
	}
	payload["command"] = command
	payload["launch_command"] = launchCommand
	payload["guarded"] = guarded
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

type dispatchIssueInfo struct {
	Number     int
	Title      string
	Body       string
	Labels     []string
	FetchError string
}

var dispatchFetchIssue = dispatchFetchIssueGH
var dispatchRouteIssues = dispatchRouteIssuesNative

func dispatchPrompt(root string, _ io.Writer, issue int, lane string) (map[string]any, error) {
	inf := dispatchFetchIssue(root, issue)
	rec := dispatchtick.BuildIssuePrompt(dispatchtick.IssuePromptInput{
		Number:     firstInt(inf.Number, issue),
		Title:      inf.Title,
		Body:       inf.Body,
		Labels:     inf.Labels,
		Lane:       lane,
		Workspace:  root,
		FetchError: inf.FetchError,
	})
	return map[string]any{
		"schema":       rec.Schema,
		"issue":        rec.Issue,
		"lane":         rec.Lane,
		"title":        rec.Title,
		"fetch_error":  rec.FetchError,
		"prompt":       rec.Prompt,
		"prompt_chars": rec.PromptChars,
	}, nil
}

func dispatchFetchIssueGH(root string, issue int) dispatchIssueInfo {
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, "gh", "issue", "view", strconv.Itoa(issue), "--json", "number,title,body,labels,state")
	cmd.Dir = root
	configureDispatchHelperCommand(cmd)
	out, err := cmd.CombinedOutput()
	if err != nil {
		return dispatchIssueInfo{Number: issue, FetchError: truncateString(strings.TrimSpace(string(out)), 300)}
	}
	var doc map[string]any
	if err := json.Unmarshal(out, &doc); err != nil {
		return dispatchIssueInfo{Number: issue, FetchError: "gh issue view produced no JSON"}
	}
	n := dispatchMapInt(doc, "number")
	if n == 0 {
		n = issue
	}
	return dispatchIssueInfo{
		Number: n,
		Title:  dispatchMapString(doc, "title"),
		Body:   dispatchMapString(doc, "body"),
		Labels: dispatchIssueLabels(doc["labels"]),
	}
}

func dispatchIssueLabels(raw any) []string {
	out := []string{}
	for _, item := range anySlice(raw) {
		if m, ok := item.(map[string]any); ok {
			if name := dispatchMapString(m, "name"); name != "" {
				out = append(out, name)
			}
		}
	}
	return out
}

func pickDispatchLane(root string, stderr io.Writer, explicit string, exclude map[string]bool, preferNewest bool) (dispatchLanePick, error) {
	router, err := dispatchRouteIssues(root, stderr)
	if err != nil {
		return dispatchLanePick{}, err
	}
	numsByLane := map[string][]int{}
	treesByLane := map[string][]string{}
	counts := map[string]int{}
	for lane, info := range router.Lanes {
		nums := append([]int(nil), info.Issues...)
		treesByLane[lane] = append([]string(nil), info.Tree...)
		// Order the lane's open issues PRIORITY-first, then by recency (#1395), so
		// PickTargetIssue (which takes the first not-skipped) drains the heaviest
		// priority/P* work before newer unlabeled noise: an old priority/P1 outranks
		// a fresh unlabeled filing. Ties fall back to the by-number recency order --
		// oldest-first by default (GitHub issue numbers are monotonic in creation
		// time, so the dispatcher drains the oldest backlog instead of forever
		// chasing the newest filing), newest-first under --prefer-newest. When no
		// candidate carries a priority/* label every weight is equal and the order
		// is byte-for-byte the old by-number order. This is safe ("when reasonable")
		// because the anti-churn cooldown (recentlyAttemptedIssues) advances past an
		// issue a worker could not land rather than re-storming it every tick.
		cands := make([]dispatchtick.LaneCandidate, len(nums))
		for i, n := range nums {
			weight := dispatchtick.PriorityWeightDefault
			if w, ok := info.Priority[n]; ok {
				weight = w
			}
			cands[i] = dispatchtick.LaneCandidate{Number: n, Weight: weight}
		}
		numsByLane[lane] = dispatchtick.OrderLaneCandidates(cands, preferNewest)
		counts[lane] = len(nums)
	}
	chosen := strings.TrimSpace(explicit)
	if chosen == "" {
		bestCount := -1
		for lane, nums := range numsByLane {
			if exclude[lane] {
				continue
			}
			if len(nums) > bestCount || (len(nums) == bestCount && lane < chosen) {
				chosen = lane
				bestCount = len(nums)
			}
		}
	}
	excluded := make([]string, 0, len(exclude))
	for lane := range exclude {
		excluded = append(excluded, lane)
	}
	sort.Strings(excluded)
	tree := treesByLane[chosen]
	if len(tree) == 0 && chosen != "" {
		tree = []string{fmt.Sprintf("internal/%s/**", chosen)}
	}
	return dispatchLanePick{
		Lane:          chosen,
		Numbers:       numsByLane[chosen],
		ByLaneCount:   counts,
		ExcludedLanes: excluded,
		Tree:          tree,
		RouterError:   dispatchRouterError(router),
	}, nil
}

func dispatchRouteIssuesNative(root string, _ io.Writer) (dispatchtick.RouterPayload, error) {
	const issueLimit = 1000
	taxonomy, taxErr := dispatchLaneTaxonomy(root)
	issues, issueErr := dispatchFetchOpenIssues(root, issueLimit)
	fetchErrs := []string{}
	if taxErr != nil {
		fetchErrs = append(fetchErrs, taxErr.Error())
	}
	if issueErr != nil {
		fetchErrs = append(fetchErrs, issueErr.Error())
	}
	return dispatchtick.RouteIssues(dispatchtick.RouterInput{
		Workspace:  root,
		Taxonomy:   taxonomy,
		Issues:     issues,
		IssueLimit: issueLimit,
		FetchError: strings.Join(fetchErrs, "; "),
	}), nil
}

func dispatchLaneTaxonomy(root string) (dispatchtick.LaneTaxonomy, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, "dos", "doctor", "--workspace", root, "--json")
	cmd.Dir = root
	configureDispatchHelperCommand(cmd)
	out, err := cmd.CombinedOutput()
	doc, perr := lastJSONObject(out)
	if perr != nil {
		if err != nil {
			return dispatchtick.LaneTaxonomy{}, fmt.Errorf("dos doctor: %w (%s)", err, strings.TrimSpace(string(out)))
		}
		return dispatchtick.LaneTaxonomy{}, fmt.Errorf("dos doctor produced no JSON")
	}
	lanes := mapAt(doc, "lanes")
	taxonomy := dispatchtick.LaneTaxonomy{
		Concurrent: stringSlice(lanes["concurrent"]),
		Trees:      map[string][]string{},
	}
	if raw, ok := lanes["trees"].(map[string]any); ok {
		for lane, globs := range raw {
			taxonomy.Trees[lane] = stringSlice(globs)
		}
	}
	return taxonomy, nil
}

func dispatchFetchOpenIssues(root string, limit int) ([]dispatchtick.Issue, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 90*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, "gh", "issue", "list", "--state", "open", "--limit", strconv.Itoa(limit), "--json", "number,title,labels,body")
	cmd.Dir = root
	configureDispatchHelperCommand(cmd)
	out, err := cmd.CombinedOutput()
	var issues []dispatchtick.Issue
	if uerr := json.Unmarshal(out, &issues); uerr != nil {
		if err != nil {
			return nil, fmt.Errorf("gh issue list: %w (%s)", err, strings.TrimSpace(string(out)))
		}
		return nil, fmt.Errorf("gh issue list produced invalid JSON: %w", uerr)
	}
	return issues, nil
}

func dispatchRouterError(router dispatchtick.RouterPayload) string {
	if router.OK {
		return ""
	}
	return router.Reason
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

func liveResolutionIssues(runsDir string) map[int]bool {
	out := map[int]bool{}
	for _, log := range resolveLogs(runsDir) {
		issue, ok := issueFromResolveLog(filepath.Base(log))
		if !ok {
			continue
		}
		pid, ok := readPID(strings.TrimSuffix(log, filepath.Ext(log)) + ".pid")
		if ok && dispatchPIDAlive(pid) {
			out[issue] = true
		}
	}
	return out
}

func liveResolutionLanes(runsDir string) map[string]bool {
	out := map[string]bool{}
	for _, log := range resolveLogs(runsDir) {
		pid, ok := readPID(strings.TrimSuffix(log, filepath.Ext(log)) + ".pid")
		if !ok || !dispatchPIDAlive(pid) {
			continue
		}
		if lane := laneFromSpawnHeader(log); lane != "" {
			out[lane] = true
		}
	}
	return out
}

func recentlyAttemptedIssues(runsDir string, cooldownMin int) map[int]bool {
	out := map[int]bool{}
	if cooldownMin <= 0 {
		return out
	}
	cutoff := time.Now().Add(-time.Duration(cooldownMin) * time.Minute)
	for _, log := range resolveLogs(runsDir) {
		st, err := os.Stat(log)
		if err != nil || st.ModTime().Before(cutoff) {
			continue
		}
		issue, ok := issueFromResolveLog(filepath.Base(log))
		if ok {
			out[issue] = true
		}
	}
	return out
}

func resolveLogs(runsDir string) []string {
	matches, _ := filepath.Glob(filepath.Join(runsDir, "resolve-*.log"))
	sort.Strings(matches)
	return matches
}

func issueFromResolveLog(name string) (int, bool) {
	m := dispatchResolveLogRE.FindStringSubmatch(name)
	if m == nil {
		return 0, false
	}
	n, err := strconv.Atoi(m[1])
	return n, err == nil
}

func laneFromSpawnHeader(path string) string {
	b, err := os.ReadFile(path)
	if err != nil {
		return ""
	}
	line := strings.SplitN(string(b), "\n", 2)[0]
	for _, field := range strings.Fields(line) {
		if strings.HasPrefix(field, "lane=") {
			return strings.TrimPrefix(field, "lane=")
		}
	}
	return ""
}

func readPID(path string) (int, bool) {
	b, err := os.ReadFile(path)
	if err != nil {
		return 0, false
	}
	pid, err := strconv.Atoi(strings.TrimSpace(string(b)))
	return pid, err == nil && pid > 0
}

func acquireDispatchLaneLease(root, lane string, tree []string, ttlS int) map[string]any {
	holder := dispatchLeaseHolder()
	id := "resolve-" + lane
	rec := leaseref.Record{ID: id, TreeGlobs: tree, Holder: holder, TTLSeconds: int64(ttlS)}
	written, verdict, err := leaseref.NewInDir(root).AcquireFenced(context.Background(), rec, time.Now())
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

func spawnDispatchIssueWorker(command []string, env map[string]string, cwd, runsDir string, issue int, lane, backend string, account dispatchtick.Account, membership *dispatchtick.Membership, baseSHA string, probeS float64) (dispatchSpawnResult, error) {
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
	res := dispatchSpawnResult{PID: cmd.Process.Pid, Log: outLog, Issue: issue, Lane: lane, Backend: backend, Account: acct, Membership: mem}
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
	fmt.Fprintf(&b, "  lane      : %s  (%d issues)\n", firstString(dispatchMapString(p, "lane"), "-"), dispatchMapInt(p, "lane_issue_count"))
	if n := dispatchMapInt(p, "target_issue"); n != 0 {
		fmt.Fprintf(&b, "  target    : #%d  %.54s\n", n, dispatchMapString(p, "issue_title"))
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

func issueNumbers(raw any) []int {
	var out []int
	switch v := raw.(type) {
	case []any:
		for _, it := range v {
			switch x := it.(type) {
			case float64:
				out = append(out, int(x))
			case map[string]any:
				if n := dispatchMapInt(x, "number"); n != 0 {
					out = append(out, n)
				}
			}
		}
	}
	return out
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
	return nil
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
