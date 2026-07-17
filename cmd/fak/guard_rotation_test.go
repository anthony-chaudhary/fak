package main

import (
	"encoding/json"
	"errors"
	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/accounts"
	"github.com/anthony-chaudhary/fak/internal/journal"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/toolprocgate"
)

func TestGuardRotateModeDefaults(t *testing.T) {
	cases := []struct {
		raw              string
		set, interactive bool
		want             string
	}{{"", false, true, "off"}, {"", false, false, "auto"}, {"auto", true, true, "auto"}, {"off", true, false, "off"}, {"seat-b", true, true, "seat-b"}}
	for _, c := range cases {
		got, err := normalizeGuardRotateMode(c.raw, c.set, c.interactive)
		if err != nil || got != c.want {
			t.Fatalf("%+v got %q %v", c, got, err)
		}
	}
}
func TestGuardNextRotationDistinctAndFailLoud(t *testing.T) {
	ready := func(name, key string) accounts.Home {
		return accounts.Home{Name: name, Dir: "/" + name, Identity: accounts.Identity{Exists: true, HasCreds: true, AccountUUID: key}}
	}
	reg := accounts.Registry{Homes: []accounts.Home{ready("a", "acct-a"), ready("b", "acct-b")}}
	r, ok := guardNextRotation(reg, "a", "auto", "stale", "CLAUDE_CONFIG_DIR")
	if !ok || r.Seat != "b" || r.Dir != "/b" {
		t.Fatalf("rotation=%+v ok=%v", r, ok)
	}
	if !strings.Contains(guardRotationBanner(r), "account b (stale)") {
		t.Fatalf("banner=%q", guardRotationBanner(r))
	}
	one := accounts.Registry{Homes: []accounts.Home{ready("a", "acct-a")}}
	if _, ok := guardNextRotation(one, "a", "auto", "stale", "CLAUDE_CONFIG_DIR"); ok {
		t.Fatal("one bucket re-handed walled account")
	}
	if _, ok := guardNextRotation(reg, "a", "off", "stale", "CLAUDE_CONFIG_DIR"); ok {
		t.Fatal("rotate=off changed behavior")
	}
}
func TestGuardNextRotationExplicitCodexSeat(t *testing.T) {
	ready := func(name, key string) accounts.Home {
		return accounts.Home{Name: name, Dir: "/codex-" + name, Identity: accounts.Identity{Exists: true, HasCreds: true, AccountUUID: key}}
	}
	reg := accounts.Registry{Homes: []accounts.Home{ready("chatgpt-a", "acct-a"), ready("chatgpt-b", "acct-b")}}
	r, ok := guardNextRotation(reg, "chatgpt-a", "chatgpt-b", "walled", "CODEX_HOME")
	if !ok || r.EnvKey != "CODEX_HOME" || r.Seat != "chatgpt-b" {
		t.Fatalf("rotation=%+v ok=%v", r, ok)
	}
}

func TestGuardApplyRotationRepointsChildEnvironment(t *testing.T) {
	cmd := []string{"codex", "exec"}
	gotCmd, env := guardApplyRotation(cmd, [][2]string{{"CODEX_HOME", "/old"}, {"X", "1"}}, guardRotation{Seat: "b", Dir: "/new", EnvKey: "CODEX_HOME"})
	if strings.Join(gotCmd, " ") != "codex exec" || len(env) != 2 || env[0] != [2]string{"CODEX_HOME", "/new"} {
		t.Fatalf("command=%v env=%v", gotCmd, env)
	}
}

func TestGuardNextRotationHeadroomSkipsWalled(t *testing.T) {
	ready := func(name, key string) accounts.Home {
		return accounts.Home{Name: name, Dir: "/" + name, Identity: accounts.Identity{Exists: true, HasCreds: true, AccountUUID: key}}
	}
	reg := accounts.Registry{Homes: []accounts.Home{ready("a", "acct-a"), ready("b", "acct-b"), ready("c", "acct-c")}}
	hr := accounts.RotationHeadroom{"uuid:acct-b": -1, "uuid:acct-c": 1}
	r, ok := guardNextRotationWithHeadroom(reg, "a", "auto", "walled", "CLAUDE_CONFIG_DIR", hr)
	if !ok || r.Seat != "c" {
		t.Fatalf("rotation=%+v ok=%v", r, ok)
	}
}

func TestGuardRotationRuntimeRotateWritesAuditAndRepoints(t *testing.T) {
	ready := func(name, key string) accounts.Home {
		return accounts.Home{Name: name, Dir: "/" + name, Identity: accounts.Identity{Exists: true, HasCreds: true, AccountUUID: key}}
	}
	rt := &guardRotationRuntime{Mode: "auto", CurrentSeat: "a", EnvKey: "CLAUDE_CONFIG_DIR", Registry: accounts.Registry{Homes: []accounts.Home{ready("a", "acct-a"), ready("b", "acct-b")}}}
	cmd, env, ok := rt.rotate([]string{"claude", "-p"}, nil, "stale", nil, "trace-1", nil)
	if !ok || rt.CurrentSeat != "b" || len(env) != 1 || env[0] != [2]string{"CLAUDE_CONFIG_DIR", "/b"} || len(cmd) != 2 {
		t.Fatalf("runtime=%+v cmd=%v env=%v ok=%v", rt, cmd, env, ok)
	}
}

func TestGuardRotationDoesNotRelaunchSuccessfulChild(t *testing.T) {
	ready := func(name, key string) accounts.Home {
		return accounts.Home{Name: name, Dir: "/" + name, Identity: accounts.Identity{Exists: true, HasCreds: true, AccountUUID: key}}
	}
	rt := &guardRotationRuntime{Mode: "auto", CurrentSeat: "a", EnvKey: "CODEX_HOME", Registry: accounts.Registry{Homes: []accounts.Home{ready("a", "acct-a"), ready("b", "acct-b")}}}
	cmd := []string{"codex", "exec", "do-once"}
	env := [][2]string{{"CODEX_HOME", "/a"}}
	gotCmd, gotEnv, ok := rt.rotateAfterExit(nil, guardRotationEvidence{}, cmd, env, nil, "trace", nil)
	if ok || rt.CurrentSeat != "a" || strings.Join(gotCmd, " ") != strings.Join(cmd, " ") || gotEnv[0] != env[0] {
		t.Fatalf("clean exit rotated/relaunched: ok=%v runtime=%+v command=%v env=%v", ok, rt, gotCmd, gotEnv)
	}
}

func TestGuardRotationLauncherWitnessesFirstFailureThenRotatedEnv(t *testing.T) {
	ready := func(name, key string) accounts.Home {
		return accounts.Home{Name: name, Dir: "/" + name, Identity: accounts.Identity{Exists: true, HasCreds: true, AccountUUID: key}}
	}
	var launches []map[string]string
	rt := &guardRotationRuntime{Mode: "auto", CurrentSeat: "a", EnvKey: "CLAUDE_CONFIG_DIR", Registry: accounts.Registry{Homes: []accounts.Home{ready("a", "acct-a"), ready("b", "acct-b")}}}
	rt.Launcher = func(grant toolprocgate.SpawnGrant) (*exec.Cmd, error) {
		m := map[string]string{}
		for _, e := range grant.Env {
			m[e.Name] = e.Value
		}
		launches = append(launches, m)
		if runtime.GOOS == "windows" {
			return exec.Command("cmd.exe", "/c", "exit 0"), nil
		}
		return exec.Command("sh", "-c", "exit 0"), nil
	}
	cmd, env, ok := rt.rotate([]string{"claude", "-p"}, [][2]string{{"CLAUDE_CONFIG_DIR", "/a"}}, "walled", nil, "trace", nil)
	if !ok {
		t.Fatal("walled account did not rotate")
	}
	broker := toolprocgate.NewSpawnBroker()
	meta := guardChildSpawnMetadata{AgentRunID: "run", ToolCallID: "guard-child:run", PolicyDigest: "sha256:test-policy", Backend: "anthropic", Envelope: toolprocgate.CapabilityEnvelope{Capabilities: []abi.Capability{toolprocgate.CapAgentRunSpawn}}}
	_, child, err := launchGuardChildWithBroker(cmd, env, false, meta, broker, rt.launcher())
	if err != nil {
		t.Fatal(err)
	}
	if err := child.Run(); err != nil {
		t.Fatal(err)
	}
	if len(launches) != 1 || launches[0]["CLAUDE_CONFIG_DIR"] != "/b" {
		t.Fatalf("launches=%v", launches)
	}
}

func TestGuardRotationWritesDurableAuditRow(t *testing.T) {
	path := filepath.Join(t.TempDir(), "guard-audit.jsonl")
	j, err := journal.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	ready := func(name, key string) accounts.Home {
		return accounts.Home{Name: name, Dir: "/" + name, Identity: accounts.Identity{Exists: true, HasCreds: true, AccountUUID: key}}
	}
	rt := &guardRotationRuntime{Mode: "auto", CurrentSeat: "a", EnvKey: "CODEX_HOME", Registry: accounts.Registry{Homes: []accounts.Home{ready("a", "acct-a"), ready("b", "acct-b")}}}
	if _, _, ok := rt.rotate([]string{"codex"}, nil, "walled", j, "trace-rotation", nil); !ok {
		t.Fatal("rotation refused")
	}
	if err := j.Close(); err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var row journal.Row
	if err := json.Unmarshal([]byte(strings.TrimSpace(string(data))), &row); err != nil {
		t.Fatal(err)
	}
	if row.Kind != "ACCOUNT_ROTATION" || row.TraceID != "trace-rotation" || row.Reason != "b:walled" || row.Hash == "" {
		t.Fatalf("row=%+v", row)
	}
}

func TestGuardRotationRequiresPositiveAuthEvidence(t *testing.T) {
	ready := func(name, key string) accounts.Home {
		return accounts.Home{Name: name, Dir: "/" + name, Identity: accounts.Identity{Exists: true, HasCreds: true, AccountUUID: key}}
	}
	newRT := func() *guardRotationRuntime {
		return &guardRotationRuntime{Mode: "auto", CurrentSeat: "a", EnvKey: "CODEX_HOME", Registry: accounts.Registry{Homes: []accounts.Home{ready("a", "acct-a"), ready("b", "acct-b")}}}
	}
	failures := []struct {
		name string
		err  error
	}{
		{"generic_exit_1", errors.New("exit status 1")},
		{"sandbox_process", errors.New("CreateProcessAsUserW failed with error 1312")},
		{"oom_signal", errors.New("signal: killed (out of memory)")},
	}
	for _, tc := range failures {
		t.Run(tc.name, func(t *testing.T) {
			rt := newRT()
			cmd := []string{"codex", "exec"}
			got, _, ok := rt.rotateAfterExit(tc.err, guardRotationEvidence{}, cmd, nil, nil, "trace", nil)
			if ok || rt.CurrentSeat != "a" || strings.Join(got, " ") != "codex exec" {
				t.Fatalf("rotated generic failure: ok=%v seat=%s cmd=%v", ok, rt.CurrentSeat, got)
			}
		})
	}
	for _, evidence := range []guardRotationEvidence{{Kind: "provider_auth", Detail: "401"}, {Kind: "stale_credential", Detail: "expired"}, {Kind: "provider_rate_limited", Detail: "429"}} {
		rt := newRT()
		_, _, ok := rt.rotateAfterExit(errors.New("exit status 1"), evidence, []string{"codex"}, nil, nil, "trace", nil)
		if !ok || rt.CurrentSeat != "b" {
			t.Fatalf("evidence=%+v did not rotate: ok=%v seat=%s", evidence, ok, rt.CurrentSeat)
		}
	}
}

func TestGuardRotationEvidenceReasonIsTyped(t *testing.T) {
	got, ok := (guardRotationEvidence{Kind: "provider_auth", Detail: "401 invalid token"}).reason()
	if !ok || got != "provider_auth:401 invalid token" {
		t.Fatalf("reason=%q ok=%v", got, ok)
	}
	if _, ok := (guardRotationEvidence{Kind: "sandbox", Detail: "1312"}).reason(); ok {
		t.Fatal("sandbox evidence admitted")
	}
}

func TestGuardRotationEvidenceSinceRequiresProviderDelta(t *testing.T) {
	if got := guardRotationEvidenceSince(map[string]uint64{"auth": 1}, map[string]uint64{"auth": 1}); got.Kind != "" {
		t.Fatalf("no delta evidence=%+v", got)
	}
	got := guardRotationEvidenceSince(map[string]uint64{"auth": 1}, map[string]uint64{"auth": 2})
	if got.Kind != "provider_auth" || !strings.Contains(got.Detail, "delta=1") {
		t.Fatalf("auth evidence=%+v", got)
	}
	got = guardRotationEvidenceSince(nil, map[string]uint64{"rate_limited": 2})
	if got.Kind != "provider_rate_limited" || !strings.Contains(got.Detail, "delta=2") {
		t.Fatalf("rate evidence=%+v", got)
	}
}
