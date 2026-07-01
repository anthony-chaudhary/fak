package main

import (
	"encoding/json"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/dispatchorder"
	"github.com/anthony-chaudhary/fak/internal/dispatchtick"
)

func withDispatchJSONHelper(t *testing.T, fn func(root string, args ...string) (map[string]any, error)) {
	t.Helper()
	old := dispatchRunJSON
	oldExternal := dispatchRunExternalJSON
	oldResources := dispatchProbeHostResources
	oldWorkerCount := dispatchProbeWorkerCount
	oldProcesses := dispatchProbeProcesses
	oldReadAccountRoster := dispatchReadAccountRoster
	oldFetchIssue := dispatchFetchIssue
	oldRouteIssues := dispatchRouteIssues
	dispatchRunJSON = func(root string, _ io.Writer, _ time.Duration, args ...string) (map[string]any, error) {
		return fn(root, args...)
	}
	dispatchRunExternalJSON = func(root string, _ time.Duration, name string, args ...string) (map[string]any, error) {
		if name == "dos" {
			return map[string]any{"alive": float64(0), "target": float64(3), "verdict": "FILLING"}, nil
		}
		return nil, fmt.Errorf("unexpected external helper %q %v", name, args)
	}
	dispatchProbeHostResources = func() dispatchtick.HostResources {
		return dispatchtick.HostResources{Cores: dispatchtick.IntPtr(64), FreeRAMMB: dispatchtick.IntPtr(128000), TotalThreads: dispatchtick.IntPtr(1000)}
	}
	dispatchProbeWorkerCount = func(root, product string) int { return 0 }
	dispatchProbeProcesses = func() dispatchtick.ProcGuardInput {
		return dispatchtick.ProcGuardInput{
			Processes:  []dispatchtick.ProcInfo{{PID: 1, Name: "worker", Threads: dispatchtick.IntPtr(10)}},
			Thresholds: dispatchtick.DefaultProcGuardThresholds(),
		}
	}
	dispatchReadAccountRoster = func(root string) ([]dispatchtick.AccountRow, error) {
		return []dispatchtick.AccountRow{
			{Account: ".claude-preflight", Tag: "acct-preflight", Product: "claude", Dir: filepath.Join(root, "acct"), Available: true, ModelTier: 1, Model: "claude"},
			{Account: ".claude-wave-b", Tag: "acct-wave-b", Product: "claude", Dir: filepath.Join(root, "acct-b"), Available: true, ModelTier: 1, Model: "claude"},
			{Account: ".claude-blocked", Tag: "blocked", Product: "claude", Dir: filepath.Join(root, "blocked"), Available: false, ModelTier: 1, Model: "claude", BlockReason: "usage limit"},
		}, nil
	}
	dispatchFetchIssue = func(root string, issue int) dispatchIssueInfo {
		return dispatchIssueInfo{
			Number: issue,
			Title:  "first-class fak dispatch verb",
			Body:   "Resolve the issue and keep literal braces like {\"ok\":true} intact.",
			Labels: []string{"cmd"},
		}
	}
	dispatchRouteIssues = func(root string, _ io.Writer) (dispatchtick.RouterPayload, error) {
		return dispatchtick.RouterPayload{
			Schema: dispatchtick.RouterSchema,
			OK:     true,
			Lanes: map[string]dispatchtick.RouterLaneGroup{
				"cmd": {
					Tree: []string{"cmd/**"},
					// 1338 is the OLDEST (lowest) cmd issue, so the default oldest-first
					// pick selects it; a newer 1400 sits behind it on the same lane.
					Issues: []int{1400, 1338},
					Count:  2,
				},
				"docs": {
					Tree:   []string{"docs/**"},
					Issues: []int{12},
					Count:  1,
				},
			},
		}, nil
	}
	t.Cleanup(func() {
		dispatchRunJSON = old
		dispatchRunExternalJSON = oldExternal
		dispatchProbeHostResources = oldResources
		dispatchProbeWorkerCount = oldWorkerCount
		dispatchProbeProcesses = oldProcesses
		dispatchReadAccountRoster = oldReadAccountRoster
		dispatchFetchIssue = oldFetchIssue
		dispatchRouteIssues = oldRouteIssues
	})
}

func dispatchHappyHelper(t *testing.T) func(root string, args ...string) (map[string]any, error) {
	t.Helper()
	return func(root string, args ...string) (map[string]any, error) {
		t.Helper()
		if len(args) == 0 {
			return nil, fmt.Errorf("missing helper argv")
		}
		switch filepath.Base(args[0]) {
		case "fleet_sessions.py":
			return map[string]any{"ok": true}, nil
		case "fleet_accounts.py":
			return nil, fmt.Errorf("dispatch should use native account routing, not fleet_accounts.py %v", args)
		default:
			return nil, fmt.Errorf("unexpected helper %q (argv %v)", args[0], args)
		}
	}
}

func TestDispatchPromptCarriesDevelopmentBranchRole(t *testing.T) {
	root := t.TempDir()
	if err := os.WriteFile(filepath.Join(root, "dos.toml"), []byte("[branch_roles]\ndevelopment_branch = \"dev\"\nrelease_branch = \"main\"\nrelease_source = \"dev\"\npublic_front_door = \"main\"\n"), 0o644); err != nil {
		t.Fatalf("write dos.toml: %v", err)
	}
	oldFetchIssue := dispatchFetchIssue
	dispatchFetchIssue = func(root string, issue int) dispatchIssueInfo {
		return dispatchIssueInfo{
			Number: issue,
			Title:  "branch-aware worker prompt",
			Body:   "Make ordinary issue workers use the development branch role.",
			Labels: []string{"dispatch"},
		}
	}
	t.Cleanup(func() { dispatchFetchIssue = oldFetchIssue })

	got, err := dispatchPrompt(root, io.Discard, 1699, "dispatchtick")
	if err != nil {
		t.Fatalf("dispatchPrompt: %v", err)
	}
	if got["development_branch"] != "dev" || got["branch_role_error"] != nil {
		t.Fatalf("branch role fields = %#v", got)
	}
	prompt := dispatchMapString(got, "prompt")
	for _, want := range []string{"configured development branch `dev`", "Just commit on `dev`."} {
		if !strings.Contains(prompt, want) {
			t.Fatalf("prompt missing %q:\n%s", want, prompt)
		}
	}
	for _, stale := range []string{"Work on `main` ONLY", "Just commit on main"} {
		if strings.Contains(prompt, stale) {
			t.Fatalf("prompt contains stale branch wording %q:\n%s", stale, prompt)
		}
	}
}

func TestDispatchCodexProcessPIDsCollapseNodeWrapperAndNativeChild(t *testing.T) {
	rows := []dispatchCodexProcessRow{
		{
			PID:     10,
			PPID:    1,
			Name:    "node.exe",
			Cmdline: `C:\Program Files\nodejs\node.exe C:\Users\USER\AppData\Roaming\npm\node_modules\@openai\codex\bin\codex.js`,
		},
		{PID: 11, PPID: 10, Name: "codex.exe", Cmdline: `C:\...\codex.exe`},
		{
			PID:     20,
			PPID:    1,
			Name:    "node",
			Cmdline: "/usr/bin/node /x/node_modules/@openai/codex/bin/codex.js",
		},
		{PID: 30, PPID: 1, Name: "node", Cmdline: "/usr/bin/node /x/not-codex.js"},
	}
	got := dispatchCodexProcessPIDs(rows)
	if len(got) != 2 || !got[11] || !got[20] {
		t.Fatalf("codex pids = %v, want native child 11 and orphan wrapper 20", got)
	}
}

func TestDispatchCodexSeatIsSingleAmbientLogin(t *testing.T) {
	old := dispatchProbeCodexProcessRows
	dispatchProbeCodexProcessRows = func() ([]dispatchCodexProcessRow, error) {
		return []dispatchCodexProcessRow{{PID: 42, Name: "codex.exe"}}, nil
	}
	t.Cleanup(func() { dispatchProbeCodexProcessRows = old })

	seat := dispatchPreflightSeat(t.TempDir(), io.Discard, "codex")
	if seat.Total == nil || *seat.Total != 1 {
		t.Fatalf("codex seat total = %v, want 1", seat.Total)
	}
	if seat.Free == nil || *seat.Free != 0 || seat.Leased == nil || *seat.Leased != 1 || !seat.Depleted {
		t.Fatalf("codex seat = %+v, want free=0 leased=1 depleted", seat)
	}
	if got := dispatchProductWorkerCount(t.TempDir(), "codex"); got != 1 {
		t.Fatalf("codex product worker count = %d, want ambient process to consume cap", got)
	}
}

func TestDispatchCodexBusyAmbientSeatRefusesSpawn(t *testing.T) {
	withDispatchJSONHelper(t, func(root string, args ...string) (map[string]any, error) {
		return map[string]any{"ok": true}, nil
	})
	old := dispatchProbeCodexProcessRows
	dispatchProbeCodexProcessRows = func() ([]dispatchCodexProcessRow, error) {
		return []dispatchCodexProcessRow{{PID: 42, Name: "codex.exe"}}, nil
	}
	t.Cleanup(func() { dispatchProbeCodexProcessRows = old })

	got, err := dispatchPreflight(t.TempDir(), io.Discard, 4, "engineering", "codex")
	if err != nil {
		t.Fatalf("dispatchPreflight: %v", err)
	}
	if got["verdict"] != dispatchtick.PreflightRefuseNoSeat {
		t.Fatalf("verdict = %v, want REFUSE_NO_SEAT; payload=%v", got["verdict"], got)
	}
	if got["cap"] != 1 {
		t.Fatalf("cap = %v, want 1", got["cap"])
	}
}

func TestDispatchReadAccountRosterNativeLoadsRegistryAndPolicyWeights(t *testing.T) {
	root := t.TempDir()
	t.Setenv("FLEET_REG_DIR", "")
	t.Setenv("FLEET_POLICY_PATH", "")
	t.Setenv("FLEET_POLICY_DIR", "")
	writeDispatchJSONFixture(t, filepath.Join(root, "tools", "_registry", "sessions.json"), map[string]any{
		"accounts": []any{
			map[string]any{
				"account":         ".claude-day26",
				"tag":             "day26",
				"config_dir":      "C:\\Users\\U\\.claude-day26",
				"available":       true,
				"active_sessions": 8,
				"live_sessions":   2,
			},
			map[string]any{
				"account":    "opencode-zai2",
				"config_dir": "C:\\Users\\U\\opencode-zai2",
				"available":  true,
			},
			map[string]any{
				"account":    ".claude-hidden",
				"tag":        "hidden",
				"kind":       "excluded",
				"config_dir": "C:\\Users\\U\\.claude-hidden",
				"available":  true,
			},
			map[string]any{
				"account":      ".claude-needslogin",
				"tag":          "needslogin",
				"config_dir":   "C:\\Users\\U\\.claude-needslogin",
				"available":    true,
				"login_status": "needs_login",
				"can_serve":    false,
			},
		},
	})
	writeDispatchJSONFixture(t, filepath.Join(root, "tools", "_registry", "accounts_policy.json"), map[string]any{
		"route_weights": map[string]any{
			"claude:day26":  20,
			"opencode:zai2": 5,
		},
	})

	rows, err := dispatchReadAccountRosterNative(root)
	if err != nil {
		t.Fatalf("dispatchReadAccountRosterNative: %v", err)
	}
	byTag := map[string]dispatchtick.AccountRow{}
	for _, row := range rows {
		byTag[row.Tag] = row
	}
	if byTag["day26"].RouteWeight != 20 || byTag["day26"].Product != "claude" || byTag["day26"].ModelTier != 1 {
		t.Fatalf("day26 row = %+v, want policy-weighted tier-1 claude", byTag["day26"])
	}
	if byTag["zai2"].RouteWeight != 5 || byTag["zai2"].Product != "opencode" || byTag["zai2"].ModelTier != 2 {
		t.Fatalf("zai2 row = %+v, want policy-weighted inferred opencode tier 2", byTag["zai2"])
	}
	if byTag["needslogin"].Available || byTag["needslogin"].LoginStatus != "needs_login" ||
		byTag["needslogin"].BlockReason == "" {
		t.Fatalf("needslogin row = %+v, want login-gated blocked row", byTag["needslogin"])
	}
	route := dispatchtick.RouteAccount(dispatchtick.AccountRouteInput{Rows: rows, Product: "claude", WorkKind: "engineering"})
	if !route.OK || route.Account.Tag != "day26" {
		t.Fatalf("route = %+v, want day26 and not excluded hidden row", route)
	}
}

func TestDispatchLiveSeatLeasesReadsLiveAccountSidecars(t *testing.T) {
	runsDir := t.TempDir()
	liveStem := filepath.Join(runsDir, "resolve-99-20000101-000000")
	if err := os.WriteFile(liveStem+".pid", []byte(fmt.Sprint(os.Getpid())), 0o644); err != nil {
		t.Fatalf("write live pid: %v", err)
	}
	writeDispatchJSONFixture(t, liveStem+dispatchtick.AccountSidecarSuffix, map[string]any{
		"tag": "acct",
		"dir": "C:\\Users\\U\\.claude-acct",
	})
	if err := os.WriteFile(filepath.Join(runsDir, "resolve-100-20000101-000000.pid"), []byte("0"), 0o644); err != nil {
		t.Fatalf("write stale pid: %v", err)
	}

	leases := dispatchLiveSeatLeases(runsDir)
	if len(leases) != 1 {
		t.Fatalf("leases = %+v, want one live lease", leases)
	}
	if leases[0].Worker != "resolve-99-20000101-000000" || leases[0].PID != os.Getpid() || leases[0].Tag != "acct" {
		t.Fatalf("lease = %+v, want live acct sidecar", leases[0])
	}
}

// TestLiveResolutionLanesDropsDeadBannerNoopWorker is the #1398 witness: a docs lane
// "held" only by an exited opencode banner-no-op worker must report FREE, not busy.
// An exited opencode worker runs as a `node` image, so AFTER it exits a recycled
// `node` pid that lands in the spawn window passes the weak liveness gate
// (dispatchPIDAlive) and would pin `docs` at LANE_BUSY forever behind a dead
// 122-byte no-op. The banner-no-op log (under the 512-byte stub floor, "> build ·
// glm-…") must be dropped so the lane reports FREE and `fak dispatch tick --lane docs`
// returns WOULD_SPAWN rather than LANE_BUSY.
func TestLiveResolutionLanesDropsDeadBannerNoopWorker(t *testing.T) {
	runsDir := t.TempDir()
	stem := filepath.Join(runsDir, "resolve-1398-20260629-101010")
	// header + the documented opencode/glm banner-only no-op, well under the 512-byte
	// stub floor (the real #1398 holders were 122 bytes).
	if err := os.WriteFile(stem+".log",
		[]byte("# fak-spawn 20260629-101010 issue=1398 lane=docs backend=opencode argv0=node\n> build · glm-4.5-air\n"),
		0o644); err != nil {
		t.Fatalf("write banner log: %v", err)
	}
	// a recycled, still-LIVE pid (our own) -- exactly what passes dispatchPIDAlive.
	if err := os.WriteFile(stem+".pid", []byte(fmt.Sprint(os.Getpid())), 0o644); err != nil {
		t.Fatalf("write live pid: %v", err)
	}
	if held := liveResolutionLanes(runsDir); len(held) != 0 {
		t.Fatalf("held lanes = %v, want none (a dead banner no-op holds no lane)", held)
	}
}

// TestLiveResolutionLanesKeepsLiveStreamingWorker is the safety side of #1398: a
// genuinely live worker streams past the 512-byte stub floor within seconds (even
// though its log opens with the same banner), so it never classifies as a no-op and
// its lane stays held -- the banner-no-op reap must not free a lane with real work.
func TestLiveResolutionLanesKeepsLiveStreamingWorker(t *testing.T) {
	runsDir := t.TempDir()
	stem := filepath.Join(runsDir, "resolve-1398-20260629-101011")
	body := "# fak-spawn 20260629-101011 issue=1398 lane=docs backend=opencode argv0=node\n> build · glm-4.5-air\n" +
		strings.Repeat("streaming real work output line\n", 40) // well past the 512-byte floor
	if err := os.WriteFile(stem+".log", []byte(body), 0o644); err != nil {
		t.Fatalf("write streaming log: %v", err)
	}
	if err := os.WriteFile(stem+".pid", []byte(fmt.Sprint(os.Getpid())), 0o644); err != nil {
		t.Fatalf("write live pid: %v", err)
	}
	if held := liveResolutionLanes(runsDir); !held["docs"] {
		t.Fatalf("held lanes = %v, want docs held (a live streaming worker holds its lane)", held)
	}
}

// TestDispatchTickDryRunHoldsGuardedSelfModifyLane is the #1397 witness: a dry-run
// dispatch that lands on a lane rooted in fak's own running source (here the cmd lane,
// tree cmd/**) reports the SELF_MODIFY_HOLD rather than would_spawn -- a guarded worker
// can investigate but never SHIP an edit to cmd/** or internal/**. The guard wrapper and
// account are still resolved (the hold is a pre-route, not a guard/account failure), so
// the operator sees exactly why the doomed worker was not launched.
//
// The lane is named EXPLICITLY (--lane cmd): the #1397 fix skips self-source lanes from the
// guarded AUTO-pick (so a default tick on this fixture lands on the shippable docs lane), but
// an operator who explicitly names a self-source lane must still reach the post-pick
// SELF_MODIFY hold -- that is exactly the safety-net this test pins.
func TestDispatchTickDryRunHoldsGuardedSelfModifyLane(t *testing.T) {
	withDispatchJSONHelper(t, dispatchHappyHelper(t))
	root := t.TempDir()

	out, errb, code := runDispatchAt("tick", "--workspace", root, "--lane", "cmd", "--no-refresh", "--no-loop-ledger", "--json")
	if code != 1 {
		t.Fatalf("exit = %d, want 1 (a self-modify hold is a refuse) (stderr: %s)", code, errb)
	}

	var got map[string]any
	if err := json.Unmarshal([]byte(out), &got); err != nil {
		t.Fatalf("bad json: %v\n%s", err, out)
	}
	if got["action"] != "self_modify_hold" || got["verdict"] != "SELF_MODIFY_HOLD" || got["ok"] != false {
		t.Fatalf("dispatch tick result = action %v verdict %v ok %v, want self_modify_hold/SELF_MODIFY_HOLD/false", got["action"], got["verdict"], got["ok"])
	}
	if got["lane"] != "cmd" || got["self_modify_tree"] != "cmd/**" || got["target_issue"] != float64(1338) {
		t.Fatalf("lane/tree/target = %v/%v/%v, want cmd/cmd-tree/1338", got["lane"], got["self_modify_tree"], got["target_issue"])
	}
	if guarded, _ := got["guarded"].(bool); !guarded {
		t.Fatalf("the hold must still resolve the guard wrapper (that is WHY it holds): %#v", got)
	}
	launch := stringAnySlice(got["launch_command"])
	if len(launch) < 6 || launch[1] != "guard" || !containsString(launch, "--audit") || !containsString(launch, "claude") {
		t.Fatalf("launch command is not guarded claude argv: %#v", launch)
	}
}

// TestDispatchTickDryRunHoldsGuardedMisroutedSelfSourceIssue is the second #1397
// witness: an issue that ROUTED to a safe lane (tools, tree tools/**) but whose own text
// targets fak's own running source (cmd/** + internal/**) still reports SELF_MODIFY_HOLD
// rather than would_spawn. This is the exact failure #1338/#1397 name -- a
// `fix(dispatch):` title aliases to the tools lane carrying ZERO extracted paths, so the
// lane tree alone never reveals the self-modify hazard; the issue-text arm catches it.
func TestDispatchTickDryRunHoldsGuardedMisroutedSelfSourceIssue(t *testing.T) {
	withDispatchJSONHelper(t, dispatchHappyHelper(t))
	oldRoute := dispatchRouteIssues
	oldFetch := dispatchFetchIssue
	dispatchRouteIssues = func(root string, _ io.Writer) (dispatchtick.RouterPayload, error) {
		return dispatchtick.RouterPayload{
			Schema: dispatchtick.RouterSchema,
			OK:     true,
			Lanes: map[string]dispatchtick.RouterLaneGroup{
				"tools": {Tree: []string{"tools/**", "scripts/**"}, Issues: []int{1397}, Count: 1},
			},
		}, nil
	}
	dispatchFetchIssue = func(root string, issue int) dispatchIssueInfo {
		return dispatchIssueInfo{
			Number: issue,
			Title:  "fix(dispatch): pre-route fak-own-code issues away from self-guarded workers",
			Body:   "most of the backlog lives in `cmd/**` + `internal/**` -- structurally unshippable by a self-guarded worker.",
			Labels: nil,
		}
	}
	t.Cleanup(func() { dispatchRouteIssues = oldRoute; dispatchFetchIssue = oldFetch })
	root := t.TempDir()

	out, errb, code := runDispatchAt("tick", "--workspace", root, "--lane", "tools", "--no-refresh", "--no-loop-ledger", "--json")
	if code != 1 {
		t.Fatalf("exit = %d, want 1 (a self-modify hold is a refuse) (stderr: %s)", code, errb)
	}
	var got map[string]any
	if err := json.Unmarshal([]byte(out), &got); err != nil {
		t.Fatalf("bad json: %v\n%s", err, out)
	}
	if got["action"] != "self_modify_hold" || got["verdict"] != "SELF_MODIFY_HOLD" || got["ok"] != false {
		t.Fatalf("mis-routed tools issue = action %v verdict %v ok %v, want self_modify_hold/SELF_MODIFY_HOLD/false", got["action"], got["verdict"], got["ok"])
	}
	if got["lane"] != "tools" || got["self_modify_tree"] != "cmd/**" || got["target_issue"] != float64(1397) {
		t.Fatalf("lane/tree/target = %v/%v/%v, want tools/cmd-glob/1397", got["lane"], got["self_modify_tree"], got["target_issue"])
	}
	if guarded, _ := got["guarded"].(bool); !guarded {
		t.Fatalf("the hold must still resolve the guard wrapper (that is WHY it holds): %#v", got)
	}
}

// TestDispatchTickDryRunPlansGuardedWorkerOnShippableLane pins the OTHER half of #1397:
// the self-modify hold is SELECTIVE. A guarded worker pinned to a non-self-modify lane
// (docs) still plans normally -- would_spawn -- because docs/** is shippable under guard.
func TestDispatchTickDryRunPlansGuardedWorkerOnShippableLane(t *testing.T) {
	withDispatchJSONHelper(t, dispatchHappyHelper(t))
	root := t.TempDir()
	initDispatchGit(t, root)
	if err := os.WriteFile(filepath.Join(root, "dirty.txt"), []byte("peer work in progress\n"), 0o644); err != nil {
		t.Fatalf("write dirty fixture: %v", err)
	}

	out, errb, code := runDispatchAt("tick", "--workspace", root, "--lane", "docs", "--no-refresh", "--no-loop-ledger", "--json")
	if code != 0 {
		t.Fatalf("exit = %d, want 0 (stderr: %s)", code, errb)
	}

	var got map[string]any
	if err := json.Unmarshal([]byte(out), &got); err != nil {
		t.Fatalf("bad json: %v\n%s", err, out)
	}
	if got["action"] != "would_spawn" || got["verdict"] != "WOULD_SPAWN" || got["lane"] != "docs" || got["target_issue"] != float64(12) {
		t.Fatalf("docs lane result = action %v verdict %v lane %v target %v, want would_spawn/docs/12", got["action"], got["verdict"], got["lane"], got["target_issue"])
	}
	if guarded, _ := got["guarded"].(bool); !guarded {
		t.Fatalf("dry-run launch should still be fak guard-fronted on a shippable lane: %#v", got)
	}
	launch := stringAnySlice(got["launch_command"])
	if len(launch) < 6 || launch[1] != "guard" || !containsString(launch, "--audit") || !containsString(launch, "claude") {
		t.Fatalf("launch command is not guarded claude argv: %#v", launch)
	}
	launchLine := strings.Join(launch, " ")
	if strings.Contains(launchLine, root) || strings.Contains(launchLine, "acct-preflight") {
		t.Fatalf("launch command shape leaked workspace/account detail: %#v", launch)
	}
	if !strings.Contains(launchLine, "<workspace>") {
		t.Fatalf("launch command shape did not preserve redacted workspace marker: %#v", launch)
	}
	acct, _ := got["account"].(map[string]any)
	if acct["tag"] != "acct-preflight" {
		t.Fatalf("account = %#v, want acct-preflight", acct)
	}
	bundle := mapAt(got, "startup_bundle")
	if bundle["schema"] != dispatchStartupBundleSchema {
		t.Fatalf("startup bundle schema = %v, want %s", bundle["schema"], dispatchStartupBundleSchema)
	}
	route := mapAt(bundle, "route")
	if route["lane"] != "docs" || route["target_issue"] != float64(12) {
		t.Fatalf("startup route = %#v, want docs/#12", route)
	}
	capFact := mapAt(bundle, "cap")
	if dispatchMapInt(capFact, "cap") != 3 || dispatchMapInt(capFact, "live") != 0 {
		t.Fatalf("startup cap = %#v, want cap=3 live=0", capFact)
	}
	terms := mapAt(capFact, "cap_terms")
	if dispatchMapInt(terms, "configured_cap") != 4 || dispatchMapInt(terms, "lease_cap") != 3 ||
		dispatchMapInt(terms, "host_cap") != 32 || dispatchMapInt(terms, "seat_cap") != 3 ||
		dispatchMapInt(terms, "effective_cap") != 3 || dispatchMapString(terms, "limiting") != "lease" {
		t.Fatalf("startup cap terms = %#v, want configured=4 lease=3 host=32 seat=3 effective=3 limiting=lease", terms)
	}
	preflight := mapAt(got, "preflight")
	if dispatchMapString(mapAt(preflight, "cap_terms"), "limiting") != "lease" {
		t.Fatalf("tick preflight cap terms = %#v, want limiting=lease", preflight)
	}
	seat := mapAt(bundle, "seat")
	if dispatchMapInt(seat, "total") != 3 || dispatchMapInt(seat, "free") != 2 {
		t.Fatalf("startup seat = %#v, want total/free 3/2", seat)
	}
	lease := mapAt(bundle, "lease")
	if dispatchMapString(lease, "id") != "resolve-docs" {
		t.Fatalf("startup lease = %#v, want resolve-docs id", lease)
	}
	if tree := stringAnySlice(lease["tree"]); len(tree) != 1 || tree[0] != "docs/**" {
		t.Fatalf("startup lease tree = %#v, want docs/**", tree)
	}
	dirty := mapAt(bundle, "dirty_tree")
	if dirty["available"] != true || dirty["clean"] != false || dispatchMapInt(dirty, "dirty_total") == 0 {
		t.Fatalf("startup dirty tree = %#v, want available dirty tree fact", dirty)
	}
}

func TestDispatchTickDryRunShowsStaleBaseWarningAndFreshAfterRefresh(t *testing.T) {
	withDispatchJSONHelper(t, dispatchHappyHelper(t))
	root := t.TempDir()
	initDispatchGit(t, root)
	if err := os.MkdirAll(filepath.Join(root, "docs"), 0o755); err != nil {
		t.Fatalf("mkdir docs: %v", err)
	}
	if err := os.WriteFile(filepath.Join(root, "docs", "shared.md"), []byte("base\n"), 0o644); err != nil {
		t.Fatalf("write base docs file: %v", err)
	}
	runDispatchGit(t, root, "add", "docs/shared.md")
	commitDispatchGit(t, root, "base")
	runDispatchGit(t, root, "update-ref", "refs/remotes/origin/main", "HEAD")

	if err := os.WriteFile(filepath.Join(root, "docs", "shared.md"), []byte("origin change\n"), 0o644); err != nil {
		t.Fatalf("write upstream docs file: %v", err)
	}
	runDispatchGit(t, root, "add", "docs/shared.md")
	commitDispatchGit(t, root, "origin change")
	runDispatchGit(t, root, "update-ref", "refs/remotes/origin/main", "HEAD")
	runDispatchGit(t, root, "update-ref", "refs/heads/main", "HEAD~1")

	out, errb, code := runDispatchAt("tick", "--workspace", root, "--lane", "docs", "--no-refresh", "--no-loop-ledger", "--json")
	if code != 0 {
		t.Fatalf("stale-base dry run exit = %d, want 0 (stderr: %s)\n%s", code, errb, out)
	}
	var staleGot map[string]any
	if err := json.Unmarshal([]byte(out), &staleGot); err != nil {
		t.Fatalf("bad stale json: %v\n%s", err, out)
	}
	if staleGot["action"] != "would_spawn" || staleGot["verdict"] != "WOULD_SPAWN" {
		t.Fatalf("stale-base tick = action %v verdict %v, want dispatchable WOULD_SPAWN", staleGot["action"], staleGot["verdict"])
	}
	stale := mapAt(staleGot, "stale_base")
	if stale["available"] != true || stale["stale"] != true || dispatchMapInt(stale, "changed_count") != 1 {
		t.Fatalf("stale_base = %#v, want available stale changed_count=1", stale)
	}
	if !strings.Contains(dispatchMapString(staleGot, "worker_preflight_warning"), "stale base") {
		t.Fatalf("worker preflight warning missing stale-base text: %#v", staleGot)
	}
	bundleStale := mapAt(mapAt(staleGot, "startup_bundle"), "stale_base")
	if bundleStale["stale"] != true {
		t.Fatalf("startup bundle stale_base = %#v, want stale=true", bundleStale)
	}

	runDispatchGit(t, root, "update-ref", "refs/heads/main", "refs/remotes/origin/main")
	out, errb, code = runDispatchAt("tick", "--workspace", root, "--lane", "docs", "--no-refresh", "--no-loop-ledger", "--json")
	if code != 0 {
		t.Fatalf("fresh dry run exit = %d, want 0 (stderr: %s)\n%s", code, errb, out)
	}
	var freshGot map[string]any
	if err := json.Unmarshal([]byte(out), &freshGot); err != nil {
		t.Fatalf("bad fresh json: %v\n%s", err, out)
	}
	fresh := mapAt(freshGot, "stale_base")
	if fresh["available"] != true || fresh["stale"] != false || dispatchMapInt(fresh, "changed_count") != 0 {
		t.Fatalf("fresh stale_base = %#v, want available stale=false changed_count=0", fresh)
	}
	if dispatchMapString(freshGot, "worker_preflight_warning") != "" {
		t.Fatalf("fresh run should not carry stale-base warning: %#v", freshGot)
	}
}

// dispatchWriteResolveWorker writes a resolver worker's .log + .pid sidecars into a
// workspace's .dispatch-runs dir so an end-to-end `fak dispatch tick` sees it exactly as a
// live host would. The pid is the test process's own (alive) -- the recycled-pid case the
// #1398 reap must survive: an exited opencode worker runs as a `node` image, so its pid can
// be reused by another live process and pass the weak liveness gate.
func dispatchWriteResolveWorker(t *testing.T, root, stem, header, body string) {
	t.Helper()
	runsDir := filepath.Join(root, dispatchtick.RunsDirName)
	if err := os.MkdirAll(runsDir, 0o755); err != nil {
		t.Fatalf("mkdir runs dir: %v", err)
	}
	if err := os.WriteFile(filepath.Join(runsDir, stem+".log"), []byte(header+body), 0o644); err != nil {
		t.Fatalf("write worker log: %v", err)
	}
	if err := os.WriteFile(filepath.Join(runsDir, stem+".pid"), []byte(fmt.Sprint(os.Getpid())), 0o644); err != nil {
		t.Fatalf("write worker pid: %v", err)
	}
}

func TestDispatchTickRendersCooldownStatusForCoolingAndReadyIssues(t *testing.T) {
	root := t.TempDir()
	runsDir := filepath.Join(root, dispatchtick.RunsDirName)
	if err := os.MkdirAll(runsDir, 0o755); err != nil {
		t.Fatalf("mkdir runs dir: %v", err)
	}
	now := time.Unix(1700000000, 0).UTC()
	writeAttempt := func(issue int, stamp string, mod time.Time) {
		t.Helper()
		path := filepath.Join(runsDir, fmt.Sprintf("resolve-%d-%s.log", issue, stamp))
		if err := os.WriteFile(path, []byte("# fak-spawn\n"), 0o644); err != nil {
			t.Fatalf("write attempt log: %v", err)
		}
		if err := os.Chtimes(path, mod, mod); err != nil {
			t.Fatalf("chtime attempt log: %v", err)
		}
	}
	writeAttempt(1775, "20260701-010000", now.Add(-30*time.Minute))
	writeAttempt(1776, "20260701-000000", now.Add(-3*time.Hour))

	rows := cooldownIssueRowsAt(runsDir, 120, now)
	if len(rows) != 2 || !rows[0].Cooling || rows[1].Cooling {
		t.Fatalf("cooldown rows = %+v, want first cooling and second ready", rows)
	}
	cooled := recentlyAttemptedIssuesAt(runsDir, 120, now)
	if !cooled[1775] || cooled[1776] {
		t.Fatalf("recently attempted = %#v, want only #1775 cooling", cooled)
	}
	status := make([]map[string]any, 0, len(rows))
	for _, row := range rows {
		status = append(status, row.Map())
	}
	out := renderDispatchTick(map[string]any{
		"verdict":         "WOULD_SPAWN",
		"ok":              true,
		"backend":         "claude",
		"live":            false,
		"preflight":       map[string]any{"verdict": "SPAWN_OK", "live": 0, "cap": 4},
		"account":         map[string]any{"tag": "acct", "tier": 1, "model": "claude"},
		"lane":            "docs",
		"target_issue":    1777,
		"issue_title":     "cooldown render",
		"reason":          "safe to spawn",
		"cooldown_status": status,
	})
	for _, want := range []string{
		"cooldowns : issue age_s remaining_s next_eligible_utc state",
		"#1775 1800 5400 2023-11-14T23:43:20Z cooling",
		"#1776 10800 0 2023-11-14T21:13:20Z ready",
	} {
		if !strings.Contains(out, want) {
			t.Fatalf("rendered cooldown status missing %q:\n%s", want, out)
		}
	}
}

// TestDispatchTickDocsLaneHeldOnlyByDeadNoopReturnsWouldSpawn is the #1398 END-TO-END
// witness: the full `fak dispatch tick --lane docs` verb -- not just the liveResolutionLanes
// helper -- must return WOULD_SPAWN, NOT LANE_BUSY, when the docs lane is "held" only by an
// exited opencode banner-no-op worker whose recycled `node` pid still passes the weak
// liveness gate. This is the exact repro in the issue ("docs stayed LANE_BUSY behind dead
// 122-byte no-ops while real docs work could not dispatch"): the helper-level test
// (TestLiveResolutionLanesDropsDeadBannerNoopWorker) proves the reap drops the lane, this
// pins that the reap actually clears the LANE_BUSY gate through the real dispatch verb.
func TestDispatchTickDocsLaneHeldOnlyByDeadNoopReturnsWouldSpawn(t *testing.T) {
	withDispatchJSONHelper(t, dispatchHappyHelper(t))
	root := t.TempDir()
	// header carries lane=docs (what laneFromSpawnHeader keys on); the body is the
	// documented opencode/glm banner-only no-op, well under the 512-byte stub floor (the
	// real #1398 holders were 122 bytes). Numbered 1398 (not the docs issue 12) so the
	// no-op is a lane holder, not the lane's pickable work.
	dispatchWriteResolveWorker(t, root, "resolve-1398-20260629-101010",
		"# fak-spawn 20260629-101010 issue=1398 lane=docs backend=opencode argv0=node\n",
		"> build · glm-4.5-air\n")

	out, errb, code := runDispatchAt("tick", "--workspace", root, "--lane", "docs", "--no-refresh", "--no-loop-ledger", "--json")
	if code != 0 {
		t.Fatalf("exit = %d, want 0 (a dead no-op holder must not wedge the lane) (stderr: %s)\n%s", code, errb, out)
	}
	var got map[string]any
	if err := json.Unmarshal([]byte(out), &got); err != nil {
		t.Fatalf("bad json: %v\n%s", err, out)
	}
	if got["verdict"] != "WOULD_SPAWN" || got["action"] != "would_spawn" || got["lane"] != "docs" || got["target_issue"] != float64(12) {
		t.Fatalf("dead-no-op docs tick = verdict %v action %v lane %v target %v, want WOULD_SPAWN/would_spawn/docs/12 (not LANE_BUSY)", got["verdict"], got["action"], got["lane"], got["target_issue"])
	}
	if held := stringAnySlice(got["held_lanes"]); len(held) != 0 {
		t.Fatalf("held_lanes = %v, want none (a dead banner no-op holds no lane)", held)
	}
}

// TestDispatchTickDocsLaneHeldByLiveStreamingWorkerStaysLaneBusy is the selectivity side of
// the #1398 end-to-end witness: the reap must NOT free a lane that carries real work. A
// genuinely live worker streams past the 512-byte stub floor (even though its log opens with
// the same banner), so the full `fak dispatch tick --lane docs` still returns LANE_BUSY --
// proving the WOULD_SPAWN above comes from the no-op reap, not from the lane never being held.
func TestDispatchTickDocsLaneHeldByLiveStreamingWorkerStaysLaneBusy(t *testing.T) {
	withDispatchJSONHelper(t, dispatchHappyHelper(t))
	root := t.TempDir()
	dispatchWriteResolveWorker(t, root, "resolve-1398-20260629-101011",
		"# fak-spawn 20260629-101011 issue=1398 lane=docs backend=opencode argv0=node\n",
		"> build · glm-4.5-air\n"+strings.Repeat("streaming real work output line\n", 40))

	out, errb, code := runDispatchAt("tick", "--workspace", root, "--lane", "docs", "--no-refresh", "--no-loop-ledger", "--json")
	if code != 1 {
		t.Fatalf("exit = %d, want 1 (a live worker keeps the lane busy) (stderr: %s)\n%s", code, errb, out)
	}
	var got map[string]any
	if err := json.Unmarshal([]byte(out), &got); err != nil {
		t.Fatalf("bad json: %v\n%s", err, out)
	}
	if got["verdict"] != "LANE_BUSY" || got["action"] != "lane_busy" {
		t.Fatalf("live-streaming docs tick = verdict %v action %v, want LANE_BUSY/lane_busy (a live worker holds the lane)", got["verdict"], got["action"])
	}
	if held := stringAnySlice(got["held_lanes"]); len(held) != 1 || held[0] != "docs" {
		t.Fatalf("held_lanes = %v, want [docs] (a live streaming worker holds its lane)", held)
	}
}

// TestDispatchTickRefusesSameFileLiveWorkerCollision is the #1763 witness: the tick must
// compare the candidate issue's file scope with already-live resolver workers before spawn.
// A generic lane-busy answer is not enough for the fleet scheduler; it needs the named
// COLLISION_RISK reason and the overlapping live worker evidence so the second candidate is
// left unspawned for the right reason.
func TestDispatchTickRefusesSameFileLiveWorkerCollision(t *testing.T) {
	withDispatchJSONHelper(t, dispatchHappyHelper(t))
	oldRoute := dispatchRouteIssues
	dispatchRouteIssues = func(root string, _ io.Writer) (dispatchtick.RouterPayload, error) {
		return dispatchtick.RouterPayload{
			Schema: dispatchtick.RouterSchema,
			OK:     true,
			Lanes: map[string]dispatchtick.RouterLaneGroup{
				"docs": {Tree: []string{"docs/shared.md"}, Issues: []int{1763}, Count: 1},
			},
		}, nil
	}
	t.Cleanup(func() { dispatchRouteIssues = oldRoute })
	root := t.TempDir()
	runsDir := filepath.Join(root, dispatchtick.RunsDirName)
	stem := "resolve-1762-20260701-101011"
	dispatchWriteResolveWorker(t, root, stem,
		"# fak-spawn 20260701-101011 issue=1762 lane=docs backend=claude argv0=claude\n",
		strings.Repeat("streaming real work output line\n", 40))
	writeDispatchJSONFixture(t, filepath.Join(runsDir, stem+dispatchLeaseTreeSidecarSuffix), []string{"docs/shared.md"})

	out, errb, code := runDispatchAt("tick", "--workspace", root, "--lane", "docs", "--no-refresh", "--no-loop-ledger", "--json")
	if code != 1 {
		t.Fatalf("exit = %d, want 1 (same-file live worker collision must refuse) (stderr: %s)\n%s", code, errb, out)
	}
	var got map[string]any
	if err := json.Unmarshal([]byte(out), &got); err != nil {
		t.Fatalf("bad json: %v\n%s", err, out)
	}
	if got["verdict"] != dispatchorder.ReasonCollisionRisk || got["action"] != "collision_risk" || got["ok"] != false {
		t.Fatalf("collision tick = verdict %v action %v ok %v, want COLLISION_RISK/collision_risk/false", got["verdict"], got["action"], got["ok"])
	}
	if got["target_issue"] != float64(1763) || got["lane"] != "docs" {
		t.Fatalf("target/lane = %v/%v, want 1763/docs", got["target_issue"], got["lane"])
	}
	collision := mapAt(got, "live_collision")
	if dispatchMapInt(collision, "issue") != 1762 || dispatchMapString(collision, "lane") != "docs" {
		t.Fatalf("live_collision = %#v, want issue 1762 on docs", collision)
	}
	if tree := stringAnySlice(collision["tree"]); len(tree) != 1 || tree[0] != "docs/shared.md" {
		t.Fatalf("live_collision.tree = %v, want [docs/shared.md]", tree)
	}
}

// TestDispatchTickRefusesInFlightDuplicateIssue is the #1766 witness: when the
// selected issue already has a live resolver worker, the tick must refuse the duplicate
// issue by name with worker/lease evidence instead of falling through to generic
// LANE_BUSY or NO_ISSUE.
func TestDispatchTickRefusesInFlightDuplicateIssue(t *testing.T) {
	withDispatchJSONHelper(t, dispatchHappyHelper(t))
	oldRoute := dispatchRouteIssues
	dispatchRouteIssues = func(root string, _ io.Writer) (dispatchtick.RouterPayload, error) {
		return dispatchtick.RouterPayload{
			Schema: dispatchtick.RouterSchema,
			OK:     true,
			Lanes: map[string]dispatchtick.RouterLaneGroup{
				"docs": {Tree: []string{"docs/**"}, Issues: []int{1766}, Count: 1},
			},
		}, nil
	}
	t.Cleanup(func() { dispatchRouteIssues = oldRoute })
	root := t.TempDir()
	runsDir := filepath.Join(root, dispatchtick.RunsDirName)
	stem := "resolve-1766-20260701-111111"
	dispatchWriteResolveWorker(t, root, stem,
		"# fak-spawn 20260701-111111 issue=1766 lane=docs backend=claude argv0=claude\n",
		strings.Repeat("streaming real work output line\n", 40))
	if err := os.WriteFile(filepath.Join(runsDir, stem+dispatchLeaseIDSidecarSuffix), []byte("resolve-docs"), 0o644); err != nil {
		t.Fatalf("write lease-id sidecar: %v", err)
	}

	out, errb, code := runDispatchAt("tick", "--workspace", root, "--lane", "docs", "--no-refresh", "--no-loop-ledger", "--json")
	if code != 1 {
		t.Fatalf("exit = %d, want 1 (in-flight duplicate must refuse) (stderr: %s)\n%s", code, errb, out)
	}
	var got map[string]any
	if err := json.Unmarshal([]byte(out), &got); err != nil {
		t.Fatalf("bad json: %v\n%s", err, out)
	}
	if got["verdict"] != "IN_FLIGHT_DUPLICATE" || got["action"] != "in_flight_duplicate" || got["target_issue"] != float64(1766) {
		t.Fatalf("duplicate tick = verdict %v action %v target %v, want IN_FLIGHT_DUPLICATE/in_flight_duplicate/1766", got["verdict"], got["action"], got["target_issue"])
	}
	dup := mapAt(got, "in_flight_duplicate")
	if dispatchMapInt(dup, "issue") != 1766 || dispatchMapString(dup, "worker") != stem || dispatchMapString(dup, "lease_id") != "resolve-docs" {
		t.Fatalf("in_flight_duplicate = %#v, want issue 1766 worker %s lease resolve-docs", dup, stem)
	}
	if dispatchMapInt(dup, "pid") == 0 || dispatchMapString(dup, "log") == "" {
		t.Fatalf("in_flight_duplicate missing pid/log evidence: %#v", dup)
	}
}

// TestDispatchTickDefaultPicksOldestIssue / WithPreferNewestPicksNewest pin the
// backlog-draining policy: by default the tick picks the OLDEST open issue on the
// busiest lane (#1338 here), and --prefer-newest restores the historical newest-first
// pick (#1400). The fixture's cmd lane carries {1400, 1338}.
func TestDispatchTickDefaultPicksOldestIssue(t *testing.T) {
	withDispatchJSONHelper(t, dispatchHappyHelper(t))
	// Disable the guard so this test isolates the pick-ORDER policy from the #1397
	// self-modify hold (cmd/** is fak's own source, so a GUARDED tick would hold here).
	t.Setenv("FLEET_DOGFOOD_GUARD", "0")
	root := t.TempDir()
	out, errb, code := runDispatchAt("tick", "--workspace", root, "--no-refresh", "--no-loop-ledger", "--json")
	if code != 0 {
		t.Fatalf("exit = %d, want 0 (stderr: %s)", code, errb)
	}
	var got map[string]any
	if err := json.Unmarshal([]byte(out), &got); err != nil {
		t.Fatalf("bad json: %v\n%s", err, out)
	}
	if got["target_issue"] != float64(1338) {
		t.Fatalf("default target = %v, want 1338 (oldest on the lane)", got["target_issue"])
	}
}

func TestDispatchTickPreferNewestPicksNewest(t *testing.T) {
	withDispatchJSONHelper(t, dispatchHappyHelper(t))
	// Disable the guard so this test isolates the pick-ORDER policy from the #1397
	// self-modify hold (cmd/** is fak's own source, so a GUARDED tick would hold here).
	t.Setenv("FLEET_DOGFOOD_GUARD", "0")
	root := t.TempDir()
	out, errb, code := runDispatchAt("tick", "--workspace", root, "--no-refresh", "--no-loop-ledger", "--prefer-newest", "--json")
	if code != 0 {
		t.Fatalf("exit = %d, want 0 (stderr: %s)", code, errb)
	}
	var got map[string]any
	if err := json.Unmarshal([]byte(out), &got); err != nil {
		t.Fatalf("bad json: %v\n%s", err, out)
	}
	if got["target_issue"] != float64(1400) {
		t.Fatalf("--prefer-newest target = %v, want 1400 (newest on the lane)", got["target_issue"])
	}
}

func TestDispatchWaveDryRunAllocatesAccountsAndPlansFirstTick(t *testing.T) {
	withDispatchJSONHelper(t, dispatchHappyHelper(t))
	// Disable the guard so this test exercises account allocation + pricing without the
	// #1397 self-modify hold firing on the cmd lane (cmd/** is fak's own source); the
	// hold itself is witnessed by TestDispatchTickDryRunHoldsGuardedSelfModifyLane.
	t.Setenv("FLEET_DOGFOOD_GUARD", "0")
	root := t.TempDir()

	out, errb, code := runDispatchAt("wave", "--workspace", root, "--count", "2", "--no-loop-ledger", "--json")
	if code != 0 {
		t.Fatalf("exit = %d, want 0 (stderr: %s)", code, errb)
	}

	var got map[string]any
	if err := json.Unmarshal([]byte(out), &got); err != nil {
		t.Fatalf("bad json: %v\n%s", err, out)
	}
	if !strings.HasPrefix(fmt.Sprint(got["wave_id"]), "wave-") || got["granted"] != float64(2) || got["spawned"] != float64(0) {
		t.Fatalf("wave header = %#v", got)
	}
	if strings.Contains(out, "should-not-render") || strings.Contains(out, "oauth_token") {
		t.Fatalf("wave output leaked allocator token material:\n%s", out)
	}
	price, _ := got["price"].(map[string]any)
	runLanes := stringAnySlice(price["run_lanes"])
	if len(runLanes) != 2 || runLanes[0] != "cmd" || runLanes[1] != "docs" {
		t.Fatalf("priced run lanes = %#v, want cmd/docs full dry-run plan", runLanes)
	}
	if price["collisions_avoided"] != float64(0) || price["lanes_utilized"] != float64(2) {
		t.Fatalf("price metrics = %#v, want zero collisions and two utilized lanes", price)
	}
	if price["action"] != "LAUNCH_ALL" || price["safe_concurrency_pct"] != float64(100) ||
		price["scope_coverage_pct"] != float64(0) || price["same_lane_parallelism"] != float64(0) {
		t.Fatalf("price action/score = %#v, want launch-all with full safe concurrency and no scoped same-lane work", price)
	}
	ticks, _ := got["ticks"].([]any)
	if len(ticks) != 1 {
		t.Fatalf("dry-run wave should plan exactly one tick, got %d", len(ticks))
	}
	tick, _ := ticks[0].(map[string]any)
	acct, _ := tick["account"].(map[string]any)
	if tick["action"] != "would_spawn" || acct["tag"] != "acct-preflight" {
		t.Fatalf("tick/account = %#v / %#v, want would_spawn/acct-preflight", tick, acct)
	}
}

func TestDispatchWavePlannerSelectsNextFiveNonCollidingIssues(t *testing.T) {
	router := dispatchtick.RouterPayload{
		Schema: dispatchtick.RouterSchema,
		OK:     true,
		Lanes: map[string]dispatchtick.RouterLaneGroup{
			"bench":   {Tree: []string{"internal/bench/**"}, Issues: []int{105}, Count: 1},
			"docs":    {Tree: []string{"docs/**"}, Issues: []int{101, 90}, Count: 2},
			"gateway": {Tree: []string{"internal/gateway/**"}, Issues: []int{102}, Count: 1},
			"model":   {Tree: []string{"internal/model/**"}, Issues: []int{103}, Count: 1},
			"policy":  {Tree: []string{"internal/policy/**"}, Issues: []int{104}, Count: 1},
		},
		Issues: []dispatchtick.IssueRoute{
			{Number: 101, Title: "docs alpha", Lane: "docs", Confidence: "path-confirmed", Paths: []string{"docs/alpha.md"}},
			{Number: 102, Title: "gateway http", Lane: "gateway", Confidence: "path-confirmed", Paths: []string{"internal/gateway/http.go"}},
			{Number: 103, Title: "model planner", Lane: "model", Confidence: "path-confirmed", Paths: []string{"internal/model/planner.go"}},
			{Number: 104, Title: "policy rules", Lane: "policy", Confidence: "path-confirmed", Paths: []string{"internal/policy/rules.go"}},
			{Number: 105, Title: "bench run", Lane: "bench", Confidence: "path-confirmed", Paths: []string{"internal/bench/run.go"}},
			{Number: 90, Title: "older docs alpha", Lane: "docs", Confidence: "path-confirmed", Paths: []string{"docs/alpha.md"}},
		},
	}

	price, err := priceDispatchWavePayload(t.TempDir(), router, 5, 5, "", nil, 0)
	if err != nil {
		t.Fatalf("priceDispatchWavePayload: %v", err)
	}
	if price.EffectiveCap != 5 || len(price.RunTargets) != 5 || price.SafeConcurrency != 5 {
		t.Fatalf("wave cap/run/safe = %d/%d/%d, want 5/5/5; price=%+v",
			price.EffectiveCap, len(price.RunTargets), price.SafeConcurrency, price)
	}

	selected := map[int]dispatchWaveCandidate{}
	for _, target := range price.RunTargets {
		selected[target.Issue] = target
		if !target.Selected || !target.Scoped || len(target.Tree) != 1 {
			t.Fatalf("run target = %+v, want selected issue-scoped one-path target", target)
		}
	}
	for _, want := range []int{101, 102, 103, 104, 105} {
		if _, ok := selected[want]; !ok {
			t.Fatalf("selected issues = %#v, missing #%d", selected, want)
		}
	}
	if _, ok := selected[90]; ok {
		t.Fatalf("selected colliding older issue #90: %#v", selected[90])
	}
	for a, left := range price.RunTargets {
		for b, right := range price.RunTargets {
			if a >= b {
				continue
			}
			if dispatchorder.TreesOverlap(left.Tree, right.Tree) {
				t.Fatalf("selected targets overlap: %+v and %+v", left, right)
			}
		}
	}

	var skipped dispatchWaveCandidate
	for _, cand := range price.Candidates {
		if cand.Issue == 90 {
			skipped = cand
			break
		}
	}
	if skipped.Issue != 90 || skipped.Selected || skipped.Reason != dispatchorder.ReasonCollisionRisk {
		t.Fatalf("skipped #90 = %+v, want unselected collision-risk candidate", skipped)
	}

	out := renderDispatchWave(map[string]any{
		"live":      false,
		"requested": 5,
		"granted":   5,
		"spawned":   0,
		"backend":   "claude",
		"price":     price,
		"ok":        true,
	})
	for _, want := range []string{
		"effective_cap=5",
		"selected_targets:",
		"issue=#105 lane=bench",
		"issue=#101 lane=docs",
		"skipped_candidates:",
		"issue=#90 lane=docs",
		"reason=COLLISION_RISK",
	} {
		if !strings.Contains(out, want) {
			t.Fatalf("rendered planner missing %q:\n%s", want, out)
		}
	}
}

func TestDispatchWavePriceUsesStepBudgetBeforeIssueCount(t *testing.T) {
	router := dispatchtick.RouterPayload{
		Schema: dispatchtick.RouterSchema,
		OK:     true,
		Lanes: map[string]dispatchtick.RouterLaneGroup{
			"docs": {
				Tree:       []string{"docs/**"},
				Issues:     []int{10, 11, 12},
				Count:      3,
				StepBudget: 3,
			},
			"gateway": {
				Tree:       []string{"internal/gateway/**"},
				Issues:     []int{20, 21},
				Count:      2,
				StepBudget: 9,
			},
		},
	}
	price, err := priceDispatchWavePayload(t.TempDir(), router, 1, 1, "", nil, 0)
	if err != nil {
		t.Fatalf("priceDispatchWavePayload: %v", err)
	}
	if strings.Join(price.RunLanes, ",") != "gateway" {
		t.Fatalf("run lanes = %#v, want gateway because it has the larger step budget", price.RunLanes)
	}
	if price.RunStepBudget != 9 || price.CandidateStepBudget != 12 {
		t.Fatalf("step budgets = run %d candidate %d, want 9/12", price.RunStepBudget, price.CandidateStepBudget)
	}
	if len(price.RunTargets) != 1 || price.RunTargets[0].StepBudget != 9 {
		t.Fatalf("run targets = %+v, want one gateway target with step_budget=9", price.RunTargets)
	}
	if len(price.Candidates) < 2 || price.Candidates[0].Lane != "gateway" {
		t.Fatalf("candidate order = %+v, want gateway first by step budget", price.Candidates)
	}
}

func TestDispatchWavePriceSerializesCollidingLaneBeforeLaunch(t *testing.T) {
	oldRoute := dispatchRouteIssues
	dispatchRouteIssues = func(root string, _ io.Writer) (dispatchtick.RouterPayload, error) {
		return dispatchtick.RouterPayload{
			Schema: dispatchtick.RouterSchema,
			OK:     true,
			Lanes: map[string]dispatchtick.RouterLaneGroup{
				"gateway": {
					Tree:   []string{"internal/gateway/**"},
					Issues: []int{10},
					Count:  3,
				},
				"gateway-http": {
					Tree:   []string{"internal/gateway/http.go"},
					Issues: []int{11},
					Count:  2,
				},
				"docs": {
					Tree:   []string{"docs/**"},
					Issues: []int{12},
					Count:  1,
				},
			},
		}, nil
	}
	t.Cleanup(func() { dispatchRouteIssues = oldRoute })

	price, err := priceDispatchWave(t.TempDir(), io.Discard, 3, 3, "", nil, 0)
	if err != nil {
		t.Fatalf("priceDispatchWave: %v", err)
	}
	if price.CollisionsAvoided != 1 || price.SerializationWasted != 1 || price.SafeConcurrency != 2 {
		t.Fatalf("price metrics = avoided %d wasted %d safe %d, want 1/1/2",
			price.CollisionsAvoided, price.SerializationWasted, price.SafeConcurrency)
	}
	if price.Action != "LAUNCH_SAFE_SET" || price.SafeConcurrencyPct != 67 ||
		price.ScopeCoveragePct != 0 || price.SameLaneParallelism != 0 {
		t.Fatalf("price action/score = action %s safe_pct %d scope_pct %d same_lane %d, want safe-set 67/0/0",
			price.Action, price.SafeConcurrencyPct, price.ScopeCoveragePct, price.SameLaneParallelism)
	}
	if strings.Join(price.RunLanes, ",") != "gateway,docs" {
		t.Fatalf("run lanes = %#v, want gateway/docs safe set", price.RunLanes)
	}
	if len(price.Repartition) != 1 || price.Repartition[0].Action != "narrow_to_issue_paths" {
		t.Fatalf("repartition advice = %+v, want one narrow_to_issue_paths row", price.Repartition)
	}
	var serialized dispatchWaveCandidate
	for _, cand := range price.Candidates {
		if cand.Lane == "gateway-http" {
			serialized = cand
		}
	}
	if serialized.Disposition != dispatchorder.DispCollisionRisk || serialized.Reason != dispatchorder.ReasonCollisionRisk {
		t.Fatalf("gateway-http candidate = %+v, want collision-risk", serialized)
	}
}

func TestDispatchWavePriceAllowsDisjointIssueScopesInsideOneLane(t *testing.T) {
	oldRoute := dispatchRouteIssues
	dispatchRouteIssues = func(root string, _ io.Writer) (dispatchtick.RouterPayload, error) {
		return dispatchtick.RouterPayload{
			Schema: dispatchtick.RouterSchema,
			OK:     true,
			Lanes: map[string]dispatchtick.RouterLaneGroup{
				"gateway": {
					Tree:   []string{"internal/gateway/**"},
					Issues: []int{10, 11},
					Count:  2,
				},
			},
			Issues: []dispatchtick.IssueRoute{
				{Number: 10, Title: "gateway http", Lane: "gateway", Confidence: "path-confirmed", Paths: []string{"internal/gateway/http.go"}},
				{Number: 11, Title: "gateway mcp", Lane: "gateway", Confidence: "path-confirmed", Paths: []string{"internal/gateway/mcp.go"}},
			},
		}, nil
	}
	t.Cleanup(func() { dispatchRouteIssues = oldRoute })

	price, err := priceDispatchWave(t.TempDir(), io.Discard, 2, 2, "", nil, 0)
	if err != nil {
		t.Fatalf("priceDispatchWave: %v", err)
	}
	if price.CollisionsAvoided != 0 || price.SerializationWasted != 0 || price.SafeConcurrency != 2 {
		t.Fatalf("price metrics = avoided %d wasted %d safe %d, want 0/0/2",
			price.CollisionsAvoided, price.SerializationWasted, price.SafeConcurrency)
	}
	if price.Action != "LAUNCH_ALL" || price.SafeConcurrencyPct != 100 ||
		price.ScopeCoveragePct != 100 || price.SameLaneParallelism != 1 {
		t.Fatalf("price action/score = action %s safe_pct %d scope_pct %d same_lane %d, want launch-all 100/100/1",
			price.Action, price.SafeConcurrencyPct, price.ScopeCoveragePct, price.SameLaneParallelism)
	}
	if len(price.RunTargets) != 2 || price.RunTargets[0].Lane != "gateway" || price.RunTargets[1].Lane != "gateway" {
		t.Fatalf("run targets = %+v, want two gateway issue targets", price.RunTargets)
	}
	if len(price.Repartition) != 0 {
		t.Fatalf("repartition advice = %+v, want none for disjoint scoped targets", price.Repartition)
	}
	if price.RunTargets[0].LeaseID == price.RunTargets[1].LeaseID {
		t.Fatalf("scoped targets share lease id: %+v", price.RunTargets)
	}
	for _, target := range price.RunTargets {
		if !target.Scoped || len(target.Tree) != 1 {
			t.Fatalf("target = %+v, want scoped one-path target", target)
		}
	}
}

func TestDispatchRouteJSONUsesNativeRouterPayload(t *testing.T) {
	withDispatchJSONHelper(t, dispatchHappyHelper(t))
	root := t.TempDir()

	out, errb, code := runDispatchAt("route", "--workspace", root, "--json")
	if code != 0 {
		t.Fatalf("exit = %d, want 0 (stderr: %s)", code, errb)
	}
	var got dispatchtick.RouterPayload
	if err := json.Unmarshal([]byte(out), &got); err != nil {
		t.Fatalf("bad json: %v\n%s", err, out)
	}
	if got.Schema != dispatchtick.RouterSchema || !got.OK {
		t.Fatalf("route header = schema %q ok %v, want native router payload ok", got.Schema, got.OK)
	}
	if got.Lanes["cmd"].Count != 2 || len(got.Lanes["cmd"].Issues) != 2 {
		t.Fatalf("cmd lane = %+v, want two routed issues", got.Lanes["cmd"])
	}
	if strings.Contains(out, "issue_lane_router.py") {
		t.Fatalf("route output should not mention legacy Python router:\n%s", out)
	}
}

func TestDispatchProgressSnapshotWritesBaselineLogAndLedger(t *testing.T) {
	withDispatchProgressStubs(t, 483, nil, map[string]any{
		"issues": []any{
			map[string]any{"number": 491, "bucket": "OPEN_WITNESSED"},
			map[string]any{"number": 492, "bucket": "OPEN"},
			map[string]any{"number": 493, "bucket": "OPEN_WITNESSED"},
		},
	})
	root := t.TempDir()

	out, errb, code := runDispatchAt("progress", "--workspace", root, "--target", "50", "--json")
	if code != 0 {
		t.Fatalf("exit = %d, want 0 (stderr: %s)", code, errb)
	}
	var got map[string]any
	if err := json.Unmarshal([]byte(out), &got); err != nil {
		t.Fatalf("bad json: %v\n%s", err, out)
	}
	if got["schema"] != dispatchProgressSchema || got["open_now"] != float64(483) || got["baseline_open"] != float64(483) {
		t.Fatalf("progress header = %#v", got)
	}
	if got["resolved_toward_target"] != float64(0) || got["target_remaining"] != float64(50) || got["witnessed_open"] != float64(2) {
		t.Fatalf("progress math = resolved %v remaining %v witnessed %v", got["resolved_toward_target"], got["target_remaining"], got["witnessed_open"])
	}
	assertJSONField(t, filepath.Join(root, dispatchProgressRunsDir, dispatchProgressBaseline), "baseline_open", float64(483))
	assertFileContains(t, filepath.Join(root, dispatchProgressRunsDir, dispatchProgressLogName), dispatchProgressSchema)
	ledger, _ := got["loop_ledger"].(map[string]any)
	events, _ := ledger["events"].([]any)
	if ledger["loop_id"] != dispatchProgressLoopID || len(events) != 4 {
		t.Fatalf("loop ledger = %#v, want four native progress events", ledger)
	}
}

func TestDispatchProgressBaselineDropAndAuditError(t *testing.T) {
	withDispatchProgressStubs(t, 479, nil, map[string]any{"_error": "dos not found"})
	root := t.TempDir()
	writeDispatchJSONFixture(t, filepath.Join(root, dispatchProgressRunsDir, dispatchProgressBaseline), map[string]any{"baseline_open": 483})
	if err := os.WriteFile(filepath.Join(root, dispatchProgressRunsDir, dispatchProgressLogName), []byte("{\"closed_now\":2}\n{\"closed_now\":3}\n"), 0o644); err != nil {
		t.Fatalf("write progress log: %v", err)
	}

	out, errb, code := runDispatchAt("progress", "--workspace", root, "--target", "50", "--no-loop-ledger", "--json")
	if code != 0 {
		t.Fatalf("exit = %d, want 0 (stderr: %s)", code, errb)
	}
	var got map[string]any
	if err := json.Unmarshal([]byte(out), &got); err != nil {
		t.Fatalf("bad json: %v\n%s", err, out)
	}
	if got["resolved_toward_target"] != float64(4) || got["target_remaining"] != float64(46) {
		t.Fatalf("resolved/remaining = %v/%v, want 4/46", got["resolved_toward_target"], got["target_remaining"])
	}
	if got["closed_by_loop_total"] != float64(5) || got["audit_error"] != "dos not found" || got["ok"] != true {
		t.Fatalf("history/audit/ok = %v/%v/%v, want 5/audit-error/true", got["closed_by_loop_total"], got["audit_error"], got["ok"])
	}
}

func TestDispatchProgressOpenCountFailureFailsSnapshot(t *testing.T) {
	withDispatchProgressStubs(t, 0, fmt.Errorf("gh down"), map[string]any{"issues": []any{}})
	root := t.TempDir()

	out, _, code := runDispatchAt("progress", "--workspace", root, "--no-loop-ledger", "--json")
	if code != 1 {
		t.Fatalf("exit = %d, want 1", code)
	}
	var got map[string]any
	if err := json.Unmarshal([]byte(out), &got); err != nil {
		t.Fatalf("bad json: %v\n%s", err, out)
	}
	if got["ok"] != false || !strings.Contains(fmt.Sprint(got["open_error"]), "gh down") {
		t.Fatalf("failure payload = %#v", got)
	}
}

func withDispatchProgressStubs(t *testing.T, openNow int, openErr error, audit map[string]any) {
	t.Helper()
	oldOpen := dispatchProgressOpenCount
	oldAudit := dispatchProgressAudit
	oldNow := dispatchProgressNow
	dispatchProgressOpenCount = func(root string) (int, error) { return openNow, openErr }
	dispatchProgressAudit = func(root string, _ io.Writer, maxCommits int, auditJSON string) (map[string]any, error) {
		return audit, nil
	}
	dispatchProgressNow = func() time.Time {
		return time.Date(2026, 6, 29, 12, 0, 0, 0, time.UTC)
	}
	t.Cleanup(func() {
		dispatchProgressOpenCount = oldOpen
		dispatchProgressAudit = oldAudit
		dispatchProgressNow = oldNow
	})
}

func TestLastJSONObjectIgnoresBracesInsideStrings(t *testing.T) {
	got, err := lastJSONObject([]byte("prefix log\n{\"old\":true}\nnoise\n{\"ok\":true,\"prompt\":\"keep {braces} and {\\\"json\\\":true}\"}\n"))
	if err != nil {
		t.Fatalf("lastJSONObject: %v", err)
	}
	if got["ok"] != true || got["prompt"] == "" || got["old"] != nil {
		t.Fatalf("got %#v, want the final object with the prompt string", got)
	}
}

func TestSpawnDispatchIssueWorkerWritesAuditSidecars(t *testing.T) {
	exe, err := os.Executable()
	if err != nil {
		t.Fatalf("os.Executable: %v", err)
	}
	runsDir := t.TempDir()
	env := envMap(os.Environ())
	env["FAK_DISPATCH_SPAWN_HELPER"] = "1"
	membership := dispatchtick.Membership{Rank: 1, WaveID: "wave-test", Size: 2, Shortfall: 0}
	account := dispatchtick.Account{Tag: "acct-a", Tier: float64(1), Model: "claude", Dir: filepath.Join(t.TempDir(), "acct-a")}

	spawned, err := spawnDispatchIssueWorker(
		[]string{exe, "-test.run=TestDispatchSpawnHelper"},
		env,
		t.TempDir(),
		runsDir,
		1338,
		"cmd",
		"claude",
		"resolve-cmd",
		[]string{"cmd/fak/dispatch_tick.go"},
		account,
		&membership,
		"abc123",
		5,
	)
	if err != nil {
		t.Fatalf("spawnDispatchIssueWorker: %v", err)
	}
	stem := strings.TrimSuffix(spawned.Log, filepath.Ext(spawned.Log))
	assertFileContains(t, stem+".backend", "claude")
	assertFileContains(t, stem+dispatchLeaseIDSidecarSuffix, "resolve-cmd")
	assertFileContains(t, stem+dispatchtick.BaseSHASidecarSuffix, "abc123")
	assertFileContains(t, stem+dispatchLeaseTreeSidecarSuffix, "cmd/fak/dispatch_tick.go")
	assertJSONField(t, stem+dispatchtick.AccountSidecarSuffix, "tag", "acct-a")
	assertJSONField(t, stem+dispatchtick.WaveSidecarSuffix, "wave_id", "wave-test")
	if early := spawned.EarlyExit; early["checked"] != true || early["silent"] == true {
		t.Fatalf("early-exit probe = %#v, want checked and non-silent", early)
	}
}

func TestWriteDispatchStartupBundleSidecar(t *testing.T) {
	dir := t.TempDir()
	logPath := filepath.Join(dir, "resolve-1784-20260701-120000.log")
	if err := os.WriteFile(logPath, []byte("# fak-spawn\n"), 0o644); err != nil {
		t.Fatalf("write log fixture: %v", err)
	}
	path := writeDispatchStartupBundleSidecar(logPath, map[string]any{
		"schema":     dispatchStartupBundleSchema,
		"route":      map[string]any{"lane": "docs", "target_issue": 1784},
		"cap":        map[string]any{"cap": 3, "live": 0},
		"seat":       map[string]any{"total": 3, "free": 2},
		"lease":      map[string]any{"id": "resolve-docs", "tree": []string{"docs/**"}},
		"dirty_tree": map[string]any{"available": true, "clean": false, "dirty_total": 1},
	})
	if path == "" || !strings.HasSuffix(path, dispatchStartupBundleSidecarSuffix) {
		t.Fatalf("startup sidecar path = %q, want *%s", path, dispatchStartupBundleSidecarSuffix)
	}
	assertFileContains(t, path, dispatchStartupBundleSchema)
	assertFileContains(t, path, "dirty_tree")
}

func TestDispatchSpawnHelper(t *testing.T) {
	if os.Getenv("FAK_DISPATCH_SPAWN_HELPER") != "1" {
		return
	}
	fmt.Println("worker helper wrote output")
	os.Exit(0)
}

func stringAnySlice(v any) []string {
	raw, _ := v.([]any)
	out := make([]string, 0, len(raw))
	for _, item := range raw {
		if s, ok := item.(string); ok {
			out = append(out, s)
		}
	}
	return out
}

func containsString(values []string, needle string) bool {
	for _, v := range values {
		if v == needle {
			return true
		}
	}
	return false
}

func initDispatchGit(t *testing.T, root string) {
	t.Helper()
	runDispatchGit(t, root, "init")
	runDispatchGit(t, root, "checkout", "-B", "main")
}

func commitDispatchGit(t *testing.T, root, message string) {
	t.Helper()
	runDispatchGit(t, root, "-c", "user.email=fak@example.invalid", "-c", "user.name=fak test", "commit", "-m", message)
}

func runDispatchGit(t *testing.T, root string, args ...string) string {
	t.Helper()
	cmd := exec.Command("git", args...)
	cmd.Dir = root
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("git %s: %v\n%s", strings.Join(args, " "), err, out)
	}
	return string(out)
}

func writeDispatchJSONFixture(t *testing.T, path string, doc any) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatalf("mkdir %s: %v", filepath.Dir(path), err)
	}
	b, err := json.MarshalIndent(doc, "", "  ")
	if err != nil {
		t.Fatalf("marshal fixture %s: %v", path, err)
	}
	if err := os.WriteFile(path, b, 0o644); err != nil {
		t.Fatalf("write fixture %s: %v", path, err)
	}
}

func assertFileContains(t *testing.T, path, want string) {
	t.Helper()
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read %s: %v", path, err)
	}
	if !strings.Contains(string(b), want) {
		t.Fatalf("%s = %q, want to contain %q", path, string(b), want)
	}
}

func assertJSONField(t *testing.T, path, key string, want any) {
	t.Helper()
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read %s: %v", path, err)
	}
	var got map[string]any
	if err := json.Unmarshal(b, &got); err != nil {
		t.Fatalf("decode %s: %v\n%s", path, err, b)
	}
	if got[key] != want {
		t.Fatalf("%s[%s] = %#v, want %#v", path, key, got[key], want)
	}
}
