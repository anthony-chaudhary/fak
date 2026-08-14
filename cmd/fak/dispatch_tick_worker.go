package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/branchrole"
	"github.com/anthony-chaudhary/fak/internal/dispatchtick"
	"github.com/anthony-chaudhary/fak/internal/loopmgr"
	"github.com/anthony-chaudhary/fak/internal/windowgate"
	"github.com/anthony-chaudhary/fak/internal/workerworktree"
)

func dispatchWorkerEnv(backend, lane, root, runsDir string, account dispatchtick.Account, goal, goalProfile string) (map[string]string, error) {
	env := envMap(os.Environ())
	env["DISPATCH_WORKSPACE"] = root
	env["DISPATCH_LANE"] = lane
	env["DISPATCH_BACKEND"] = backend
	// Name the seat this worker actually runs on. Without it every downstream
	// consumer of DISPATCH_ACCOUNT (guard's goal-park record, dispatchworker's
	// env echo) saw "" and could not attribute a wall to an account — which is
	// why every live park record carried a blank account and therefore walled
	// EVERY account on the lane instead of the one that was rate-limited.
	if id := dispatchAccountID(account); id != "" {
		env["DISPATCH_ACCOUNT"] = id
	} else {
		// An unattributable spawn must not inherit THIS process's stale identity
		// and mislabel someone else's wall as its own.
		delete(env, "DISPATCH_ACCOUNT")
	}
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

// dispatchAccountID is the stable, human-legible identity of the seat a worker
// was dispatched onto, as stamped into DISPATCH_ACCOUNT. The chooser's tag is
// the identity of record (it is what the account sidecar and `fak dispatch tick`
// render); the config dir's base name is the fallback for a seat that carries no
// tag, so a seat is attributable whenever anything at all distinguishes it.
// Returns "" only when the account is genuinely anonymous.
func dispatchAccountID(account dispatchtick.Account) string {
	if tag := strings.TrimSpace(account.Tag); tag != "" {
		return tag
	}
	if dir := strings.TrimSpace(account.Dir); dir != "" {
		return filepath.Base(filepath.Clean(dir))
	}
	return ""
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

// fleetEnvSwitchOn is the ONE truthy/falsy grammar every FLEET_* on/off switch below
// shares, so the switches can never drift apart by copy-paste: an unset variable takes
// `whenUnset`, any of the off-ish words is OFF, and every other value is ON. The Python
// mirrors (tools/worker_worktree.py, tools/tier_launch.py) reproduce exactly this
// grammar. Note `dispatchWorkerFallbackModel` deliberately does NOT use it — its
// vocabulary carries an extra "none", which means a model name, not a switch.
func fleetEnvSwitchOn(name string, whenUnset bool) bool {
	raw, ok := os.LookupEnv(name)
	if !ok {
		return whenUnset
	}
	switch strings.ToLower(strings.TrimSpace(raw)) {
	case "", "0", "off", "false", "no", "disable", "disabled":
		return false
	}
	return true
}

func guardDisabled() bool {
	// Inverted: the switch turns the guard ON, so "off-ish" means the guard is disabled.
	// Unset leaves the guard ON, hence not disabled.
	return !fleetEnvSwitchOn("FLEET_DOGFOOD_GUARD", true)
}

// workerWorktreeEnabled reports whether #3168 per-worker git worktree isolation is
// switched on via FLEET_WORKER_WORKTREE. Default (unset / an off-ish value) is OFF,
// which restores the shared-trunk spawn behavior byte-for-byte; any other value
// turns isolation on.
func workerWorktreeEnabled() bool {
	return fleetEnvSwitchOn("FLEET_WORKER_WORKTREE", false)
}

// dispatchTierLaunchEnabled reports whether the opt-in per-issue tier launch profile
// (FLEET_TIER_LAUNCH) is switched on. Default (unset / an off-ish value) is OFF, which keeps
// every worker on the seat-default model with no effort/ultracode uplift — byte-identical to
// before this seam.
func dispatchTierLaunchEnabled() bool {
	return fleetEnvSwitchOn("FLEET_TIER_LAUNCH", false)
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
	// #3597: a DISPATCHED worker is unattended by construction — its stdout and stderr
	// are bound to the transcript `fh` just above, and the monitor reads liveness from
	// that transcript, never from a console. configureDispatchSpawn only hides the
	// console WINDOW; the console (and its conhost.exe host process) is still allocated,
	// which is the cost #2340 measured. Decline the console outright here. Scoped to this
	// spawn deliberately: configureDispatchSpawn's other callers include the operator-
	// attended `fak dispatch canary` foreground run, which must keep its console.
	windowgate.ConfigureDetachedCommand(cmd)
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
	if receipt, ok := payload["repo_pulse_receipt"].(map[string]any); ok && dispatchMapString(receipt, "schema") == "fak-dispatch-repo-pulse-receipt/1" {
		issue, pid := dispatchMapInt(payload, "issue"), dispatchMapInt(payload, "pid")
		if spawned := mapAt(payload, "spawned"); issue <= 0 || pid <= 0 {
			issue, pid = dispatchMapInt(spawned, "issue"), dispatchMapInt(spawned, "pid")
		}
		if issue > 0 && pid > 0 {
			name := fmt.Sprintf("repo-pulse-launch-%d-%d.json", issue, pid)
			_ = os.WriteFile(filepath.Join(runsDir, name), blob, 0o644)
			if durableDir := dispatchRepoPulseLedgerDir(runsDir); durableDir != "" {
				if os.MkdirAll(durableDir, 0o755) == nil {
					_ = os.WriteFile(filepath.Join(durableDir, name), blob, 0o644)
				}
			}
		}
	}
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
