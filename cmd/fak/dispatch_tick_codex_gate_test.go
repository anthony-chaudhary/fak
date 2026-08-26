package main

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/dispatchtick"
	"github.com/anthony-chaudhary/fak/internal/sessionregistry"
)

func TestDispatchCommandExecutedLiveCodexAllowsGuardedSubscriptionChildFromUnguardedParent(t *testing.T) {
	root, threadID := dispatchCodexGateFixture(t, false)
	t.Setenv("FLEET_DOGFOOD_GUARD_BASEURL", "")

	got, spawned, command := runDispatchCodexGateTick(t, root)
	t.Cleanup(func() { releaseInProcessLaneLease(root, mapAt(got, "lease")) })
	if !spawned {
		t.Fatalf("guarded Codex child did not reach the live spawner: %#v", got)
	}
	if len(command) < 2 || command[1] != "guard" {
		t.Fatalf("spawned child command = %#v, want fak guard front", command)
	}
	if slices.Contains(command, "--base-url") {
		t.Fatalf("subscription child command must let guard select the matched OAuth upstream: %#v", command)
	}

	if got["action"] != "spawned" || got["verdict"] != "SPAWNED" || got["ok"] != true {
		t.Fatalf("dispatch result = action %v verdict %v ok %v, want spawned/SPAWNED/true", got["action"], got["verdict"], got["ok"])
	}
	if got["command_executed"] != true {
		t.Fatalf("spawned dispatch command_executed = %#v, want true", got["command_executed"])
	}
	gate := mapAt(got, "codex_loop_gate")
	parent := mapAt(gate, "parent")
	launch := mapAt(gate, "launch")
	if dispatchMapString(parent, "source") != "current_thread" ||
		dispatchMapString(parent, "session_id") != threadID ||
		parent["guard_witnessed"] != false ||
		launch["guarded"] != true ||
		dispatchMapString(gate, "action") != "spawned" ||
		dispatchMapString(gate, "next_action") != "spawn the prepared child through fak guard" {
		t.Fatalf("codex loop gate receipt = %#v", gate)
	}
}

func TestDispatchTickCodexLoopGateDefaultOffSkipsAudit(t *testing.T) {
	root, _ := dispatchCodexGateFixture(t, true)
	t.Setenv("FLEET_CODEX_LOOP_GATE", "")
	t.Setenv("FLEET_DOGFOOD_GUARD_BASEURL", healthyDispatchProvider(t)+"/v1")

	oldRecent := diagnoseRecentCodexLoopsForGate
	diagnoseRecentCodexLoopsForGate = func(string, float64, int) (codexLoopRecentReport, error) {
		t.Fatal("default-off dispatch ran the Codex transcript audit")
		return codexLoopRecentReport{}, nil
	}
	t.Cleanup(func() { diagnoseRecentCodexLoopsForGate = oldRecent })

	got, spawned, _ := runDispatchCodexGateTickModeArgs(t, root, true)
	t.Cleanup(func() { releaseInProcessLaneLease(root, mapAt(got, "lease")) })
	if !spawned || got["action"] != "spawned" || got["ok"] != true {
		t.Fatalf("default-off Codex dispatch did not spawn: spawned=%v receipt=%#v", spawned, got)
	}
	if gate, evaluated := got["codex_loop_gate"]; evaluated {
		t.Fatalf("default-off Codex dispatch emitted a loop-gate receipt: %#v", gate)
	}
}

func TestDispatchTickCodexLoopGateEnvironmentOptInRefuses(t *testing.T) {
	root, _ := dispatchCodexGateFixture(t, true)
	t.Setenv("FLEET_CODEX_LOOP_GATE", "loop")
	t.Setenv("FLEET_DOGFOOD_GUARD_BASEURL", healthyDispatchProvider(t)+"/v1")

	got, spawned, _ := runDispatchCodexGateTickModeArgs(t, root, true)
	if spawned || got["action"] != "codex_loop_gate_refused" || got["verdict"] != "CODEX_LOOP_GATE_REFUSED" {
		t.Fatalf("environment-opted-in loop gate did not refuse: spawned=%v receipt=%#v", spawned, got)
	}
	if got["command_executed"] != false {
		t.Fatalf("refused dispatch command_executed = %#v, want false", got["command_executed"])
	}
	gate := mapAt(got, "codex_loop_gate")
	if gate["evaluated"] != true || dispatchMapString(gate, "fail_on") != "loop" || dispatchMapString(gate, "verdict") != "LOOP" {
		t.Fatalf("environment-opted-in gate receipt = %#v", gate)
	}
}

func TestDispatchTickCodexLoopGateInvalidEnvironmentFailsClosed(t *testing.T) {
	t.Setenv("FLEET_CODEX_LOOP_GATE", "urgent")
	_, refused, err := dispatchCodexLoopGateForTick(dispatchTickOptions{Backend: "codex"}, dispatchtick.Account{Dir: t.TempDir()}, true)
	if err == nil || refused || !strings.Contains(err.Error(), "invalid --codex-loop-gate") {
		t.Fatalf("invalid environment threshold refused=%v err=%v, want validation error", refused, err)
	}
}

func TestDispatchCodexLoopGateHelpSaysOptInDefaultOff(t *testing.T) {
	for _, verb := range []string{"tick", "wave"} {
		_, errb, code := runDispatchAt(verb, "--help")
		if code != 2 {
			t.Fatalf("fak dispatch %s --help exit=%d, want 2", verb, code)
		}
		for _, want := range []string{"opt in to a pre-spawn audit", "else off", "loop|action"} {
			if !strings.Contains(errb, want) {
				t.Fatalf("fak dispatch %s help missing %q:\n%s", verb, want, errb)
			}
		}
	}
}

func TestDispatchTickLiveCodexStillRefusesGenuineLoopWithGuardedChild(t *testing.T) {
	root, _ := dispatchCodexGateFixture(t, true)
	t.Setenv("FLEET_DOGFOOD_GUARD_BASEURL", healthyDispatchProvider(t)+"/v1")

	got, spawned, _ := runDispatchCodexGateTick(t, root)
	if spawned {
		t.Fatal("guarded child spawned despite genuine repeated-output loop evidence")
	}

	if got["action"] != "codex_loop_gate_refused" || got["verdict"] != "CODEX_LOOP_GATE_REFUSED" || got["ok"] != false {
		t.Fatalf("dispatch result = action %v verdict %v ok %v, want codex_loop_gate_refused/CODEX_LOOP_GATE_REFUSED/false", got["action"], got["verdict"], got["ok"])
	}
	gate := mapAt(got, "codex_loop_gate")
	launch := mapAt(gate, "launch")
	if dispatchMapString(gate, "fail_on") != "loop" ||
		dispatchMapString(gate, "verdict") != "LOOP" ||
		launch["guarded"] != true ||
		dispatchMapString(gate, "action") != "refused" ||
		dispatchMapString(gate, "next_action") == "" {
		t.Fatalf("codex loop gate receipt = %#v", gate)
	}
}

func TestDispatchTickDryRunPredictsLiveCodexLoopRefusal(t *testing.T) {
	root, _ := dispatchCodexGateFixture(t, true)
	t.Setenv("FLEET_DOGFOOD_GUARD_BASEURL", healthyDispatchProvider(t)+"/v1")

	dry, drySpawned, _ := runDispatchCodexGateTickMode(t, root, false)
	if lease, ok := dry["lease"].(map[string]any); ok {
		if path := dispatchMapString(lease, "path"); path != "" {
			_ = os.Remove(path)
		}
	}
	live, liveSpawned, _ := runDispatchCodexGateTickMode(t, root, true)
	if drySpawned || liveSpawned {
		t.Fatalf("spawned dry=%v live=%v, want refusal before side effects", drySpawned, liveSpawned)
	}
	for name, got := range map[string]map[string]any{"dry": dry, "live": live} {
		if got["action"] != "codex_loop_gate_refused" || got["verdict"] != "CODEX_LOOP_GATE_REFUSED" {
			t.Fatalf("%s result = action %v verdict %v, want matching typed refusal", name, got["action"], got["verdict"])
		}
		gate, _ := got["codex_loop_gate"].(map[string]any)
		if dispatchMapString(gate, "id") != "codex_loop" || gate["evaluated"] != true || dispatchMapString(gate, "verdict") != "LOOP" {
			t.Fatalf("%s gate = %#v, want evaluated codex_loop/LOOP", name, gate)
		}
		gates, _ := got["launch_checks"].(map[string]any)
		provider, _ := gates["provider_reachability"].(map[string]any)
		if dispatchMapString(provider, "id") != "provider_reachability" || provider["evaluated"] != true || provider["ok"] != true {
			t.Fatalf("%s provider check = %#v, want evaluated healthy route", name, provider)
		}
	}
}

func TestDispatchTickTerminalHistoricalLoopIsVisibleButAdmitted(t *testing.T) {
	old := readSessionRows
	readSessionRows = func() ([]sessionregistry.Record, error) {
		return []sessionregistry.Record{{State: sessionregistry.StateFailed, Identity: sessionregistry.Identity{Runtime: "codex", SessionID: "terminal"}}}, nil
	}
	t.Cleanup(func() { readSessionRows = old })
	opts := dispatchTickOptions{Backend: "codex", Live: false, CodexLoopGate: "loop"}
	account := dispatchtick.Account{Dir: t.TempDir()}
	diagnosis := loopDiagnosis("terminal")
	diagnosis.Path = filepath.Join(account.Dir, "sessions", "rollout-terminal.jsonl")
	rep := codexLaunchReport(account.Dir, 24, diagnosis)
	oldRecent := diagnoseRecentCodexLoopsForGate
	diagnoseRecentCodexLoopsForGate = func(string, float64, int) (codexLoopRecentReport, error) { return rep, nil }
	t.Cleanup(func() { diagnoseRecentCodexLoopsForGate = oldRecent })
	gate, refused, err := dispatchCodexLoopGateForTick(opts, account, true)
	if err != nil || refused {
		t.Fatalf("terminal history refused=%v err=%v gate=%#v", refused, err, gate)
	}
	life := mapAt(gate, "lifecycle")
	if life["terminal_count"] != 1 || life["live_count"] != 0 || dispatchMapString(gate, "verdict") != "OK" {
		t.Fatalf("terminal lifecycle receipt = %#v", gate)
	}
	cleanup := dispatchMapString(gate, "cleanup_command")
	if !strings.Contains(cleanup, "fak sessions codex-loop archive --path") || !strings.Contains(cleanup, "--dry-run") {
		t.Fatalf("cleanup_command = %q", cleanup)
	}
}

func TestDispatchTickAmbiguousLoopFailsSafeWithCleanupAction(t *testing.T) {
	root, _ := dispatchCodexGateFixture(t, true)
	t.Setenv("FLEET_DOGFOOD_GUARD_BASEURL", healthyDispatchProvider(t)+"/v1")
	readSessionRows = func() ([]sessionregistry.Record, error) { return nil, nil }
	got, spawned, _ := runDispatchCodexGateTick(t, root)
	if spawned || got["action"] != "codex_loop_gate_refused" {
		t.Fatalf("ambiguous lifecycle did not fail safe: spawned=%v receipt=%#v", spawned, got)
	}
	gate := mapAt(got, "codex_loop_gate")
	life := mapAt(gate, "lifecycle")
	nextAction := dispatchMapString(gate, "next_action")
	if life["ambiguous_count"] != float64(1) ||
		dispatchMapString(gate, "verdict") != "AMBIGUOUS" ||
		!strings.Contains(nextAction, "fak sessions codex-loop --recent --json") ||
		!strings.Contains(nextAction, "reconcile or clean terminal session registrations") {
		t.Fatalf("ambiguous lifecycle receipt = %#v", gate)
	}
}

func TestDispatchTickLiveCodexStillRefusesUnguardedChild(t *testing.T) {
	root, threadID := dispatchCodexGateFixture(t, false)
	t.Setenv("FLEET_DOGFOOD_GUARD_BASEURL", "")
	t.Setenv("FLEET_DOGFOOD_GUARD", "0")

	got, spawned, _ := runDispatchCodexGateTick(t, root)
	if spawned {
		t.Fatal("unguarded Codex child reached the live spawner")
	}

	if got["action"] != "worker_preflight_refused" || got["verdict"] != dispatchWorkerPreflightRouteMisconfigured || got["ok"] != false {
		t.Fatalf("dispatch result = action %v verdict %v ok %v, want worker_preflight_refused/%s/false", got["action"], got["verdict"], got["ok"], dispatchWorkerPreflightRouteMisconfigured)
	}
	preflight := mapAt(got, "worker_preflight")
	reason := dispatchMapString(preflight, "reason")
	if preflight["guarded"] != false || !strings.Contains(reason, "FLEET_DOGFOOD_GUARD") || !strings.Contains(reason, "fak guard -- codex") {
		t.Fatalf("worker preflight receipt = %#v", preflight)
	}
	if _, acquired := got["lease"]; acquired {
		t.Fatalf("unguarded worker acquired a lane before refusal: %#v", got["lease"])
	}
	_ = threadID
}

func healthyDispatchProvider(t *testing.T) string {
	t.Helper()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"ok":true,"provider_reachability":{"evaluated":true,"ok":true,"status":405}}`))
	}))
	t.Cleanup(server.Close)
	return server.URL
}

func dispatchCodexGateFixture(t *testing.T, loop bool) (string, string) {
	t.Helper()
	withDispatchJSONHelper(t, dispatchHappyHelper(t))

	root := t.TempDir()
	home := t.TempDir()
	codexHome := filepath.Join(home, ".codex")
	if err := os.MkdirAll(filepath.Join(codexHome, "sessions", "2026", "08", "17"), 0o755); err != nil {
		t.Fatalf("mkdir Codex sessions: %v", err)
	}
	if err := os.WriteFile(filepath.Join(codexHome, "auth.json"), []byte(`{"ok":true}`), 0o644); err != nil {
		t.Fatalf("write Codex auth: %v", err)
	}

	threadID := "01a0102a-420f-7741-b170-4ea13eded940"
	path := filepath.Join(codexHome, "sessions", "2026", "08", "17", "rollout-2026-08-17T14-00-00-"+threadID+".jsonl")
	lines := []string{
		`{"timestamp":"2026-08-17T14:00:00.000Z","type":"session_meta","payload":{"session_id":"` + threadID + `","originator":"codex-tui","cli_version":"0.142.5","model_provider":"openai","git":{"commit_hash":"111ff04","branch":"main"}}}`,
	}
	if loop {
		lines = append(lines,
			`{"timestamp":"2026-08-17T14:00:03.000Z","type":"response_item","payload":{"type":"function_call","name":"update_plan","arguments":"{\"plan\":[{\"step\":\"one\",\"status\":\"in_progress\"}]}","call_id":"plan_1"}}`,
			`{"timestamp":"2026-08-17T14:00:04.000Z","type":"response_item","payload":{"type":"function_call_output","call_id":"plan_1","output":"Plan updated"}}`,
			`{"timestamp":"2026-08-17T14:00:15.000Z","type":"response_item","payload":{"type":"function_call","name":"update_plan","arguments":"{\"plan\":[{\"step\":\"one\",\"status\":\"in_progress\"}]}","call_id":"plan_2"}}`,
			`{"timestamp":"2026-08-17T14:00:16.000Z","type":"response_item","payload":{"type":"function_call_output","call_id":"plan_2","output":"Plan updated"}}`,
			`{"timestamp":"2026-08-17T14:00:27.000Z","type":"response_item","payload":{"type":"function_call","name":"update_plan","arguments":"{\"plan\":[{\"step\":\"one\",\"status\":\"in_progress\"}]}","call_id":"plan_3"}}`,
			`{"timestamp":"2026-08-17T14:00:28.000Z","type":"response_item","payload":{"type":"function_call_output","call_id":"plan_3","output":"Plan updated"}}`,
		)
	} else {
		lines = append(lines,
			`{"timestamp":"2026-08-17T14:00:02.000Z","type":"response_item","payload":{"type":"function_call","name":"shell_command","arguments":"{\"command\":\"git status --short\"}","call_id":"shell_1"}}`,
			`{"timestamp":"2026-08-17T14:00:03.000Z","type":"response_item","payload":{"type":"function_call_output","call_id":"shell_1","output":"## main"}}`,
		)
	}
	writeCodexLoopFixture(t, path, lines)

	t.Setenv("HOME", home)
	t.Setenv("USERPROFILE", home)
	t.Setenv("CODEX_HOME", codexHome)
	t.Setenv("CODEX_THREAD_ID", threadID)
	t.Setenv("FAK_CODEX_OAUTH_SESSIONS", "10")
	t.Setenv("FLEET_DOGFOOD_GUARD", "1")
	t.Setenv("FAK_LEASE_OWNER", "codex-gate-"+strings.ReplaceAll(t.Name(), "/", "-"))

	oldRows := dispatchProbeCodexProcessRows
	dispatchProbeCodexProcessRows = func() ([]dispatchCodexProcessRow, error) { return nil, nil }
	oldRegistry := readSessionRows
	readSessionRows = func() ([]sessionregistry.Record, error) {
		state := sessionregistry.StateCompleted
		if loop {
			state = sessionregistry.StateActive
		}
		return []sessionregistry.Record{{State: state, Identity: sessionregistry.Identity{Runtime: "codex", SessionID: threadID}}}, nil
	}
	t.Cleanup(func() {
		dispatchProbeCodexProcessRows = oldRows
		readSessionRows = oldRegistry
	})
	return root, threadID
}

func runDispatchCodexGateTick(t *testing.T, root string) (map[string]any, bool, []string) {
	t.Helper()
	return runDispatchCodexGateTickMode(t, root, true)
}

func runDispatchCodexGateTickMode(t *testing.T, root string, live bool) (map[string]any, bool, []string) {
	t.Helper()
	return runDispatchCodexGateTickModeArgs(t, root, live, "--codex-loop-gate", "loop")
}

func runDispatchCodexGateTickModeArgs(t *testing.T, root string, live bool, extraArgs ...string) (map[string]any, bool, []string) {
	t.Helper()
	oldBroker := launchSpawnBroker
	oldSpawner := dispatchIssueWorkerSpawner
	spawned := false
	var command []string
	launchSpawnBroker = func(a launchBrokerAttempt) launchBrokerGrant {
		return allowLaunchBrokerGrant(a, "unit-test-allow")
	}
	dispatchIssueWorkerSpawner = func(argv []string, env map[string]string, cwd, runsDir string, issue int, lane, backend, leaseID string, tree []string, account dispatchtick.Account, membership *dispatchtick.Membership, baseSHA, stdinPayload string, probeS float64) (dispatchSpawnResult, error) {
		spawned = true
		command = append([]string(nil), argv...)
		logPath := filepath.Join(runsDir, "resolve-12-20260817-140000.log")
		if err := os.MkdirAll(filepath.Dir(logPath), 0o755); err != nil {
			t.Fatalf("mkdir runs dir: %v", err)
		}
		if err := os.WriteFile(logPath, []byte("# fak-spawn\nworking\n"), 0o644); err != nil {
			t.Fatalf("write spawn log: %v", err)
		}
		return dispatchSpawnResult{PID: 7073, Log: logPath, Issue: issue, Lane: lane, Backend: backend, LeaseID: leaseID, Tree: tree}, nil
	}
	t.Cleanup(func() {
		launchSpawnBroker = oldBroker
		dispatchIssueWorkerSpawner = oldSpawner
	})

	args := []string{"tick", "--workspace", root, "--backend", "codex", "--lane", "docs", "--no-refresh", "--no-loop-ledger"}
	args = append(args, extraArgs...)
	if live {
		args = append(args, "--live")
	}
	args = append(args, "--json")
	out, errb, code := runDispatchAt(args...)
	if code != 0 && code != 1 {
		t.Fatalf("dispatch tick exit = %d, want a typed receipt (stderr: %s)\n%s", code, errb, out)
	}
	if !json.Valid([]byte(out)) {
		t.Fatalf("dispatch tick emitted invalid JSON:\n%s", out)
	}
	var got map[string]any
	if err := json.Unmarshal([]byte(out), &got); err != nil {
		t.Fatalf("decode dispatch JSON: %v\n%s", err, out)
	}
	return got, spawned, command
}
