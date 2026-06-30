package main

import (
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

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

func TestDispatchTickDryRunPlansGuardedIssueWorker(t *testing.T) {
	withDispatchJSONHelper(t, dispatchHappyHelper(t))
	root := t.TempDir()

	out, errb, code := runDispatchAt("tick", "--workspace", root, "--no-refresh", "--no-loop-ledger", "--json")
	if code != 0 {
		t.Fatalf("exit = %d, want 0 (stderr: %s)", code, errb)
	}

	var got map[string]any
	if err := json.Unmarshal([]byte(out), &got); err != nil {
		t.Fatalf("bad json: %v\n%s", err, out)
	}
	if got["action"] != "would_spawn" || got["verdict"] != "WOULD_SPAWN" || got["target_issue"] != float64(1338) {
		t.Fatalf("dispatch tick result = action %v verdict %v target %v", got["action"], got["verdict"], got["target_issue"])
	}
	if got["lane"] != "cmd" || got["issue_title"] != "first-class fak dispatch verb" {
		t.Fatalf("lane/title = %v/%v, want cmd/title", got["lane"], got["issue_title"])
	}
	if guarded, _ := got["guarded"].(bool); !guarded {
		t.Fatalf("dry-run launch should be fak guard-fronted: %#v", got)
	}
	launch := stringAnySlice(got["launch_command"])
	if len(launch) < 6 || launch[1] != "guard" || !containsString(launch, "--audit") || !containsString(launch, "claude") {
		t.Fatalf("launch command is not guarded claude argv: %#v", launch)
	}
	acct, _ := got["account"].(map[string]any)
	if acct["tag"] != "acct-preflight" {
		t.Fatalf("account = %#v, want acct-preflight", acct)
	}
}

// TestDispatchTickDefaultPicksOldestIssue / WithPreferNewestPicksNewest pin the
// backlog-draining policy: by default the tick picks the OLDEST open issue on the
// busiest lane (#1338 here), and --prefer-newest restores the historical newest-first
// pick (#1400). The fixture's cmd lane carries {1400, 1338}.
func TestDispatchTickDefaultPicksOldestIssue(t *testing.T) {
	withDispatchJSONHelper(t, dispatchHappyHelper(t))
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
	assertFileContains(t, stem+dispatchtick.BaseSHASidecarSuffix, "abc123")
	assertJSONField(t, stem+dispatchtick.AccountSidecarSuffix, "tag", "acct-a")
	assertJSONField(t, stem+dispatchtick.WaveSidecarSuffix, "wave_id", "wave-test")
	if early := spawned.EarlyExit; early["checked"] != true || early["silent"] == true {
		t.Fatalf("early-exit probe = %#v, want checked and non-silent", early)
	}
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
