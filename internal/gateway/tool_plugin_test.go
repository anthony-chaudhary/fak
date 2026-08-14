package gateway

import (
	"context"
	"encoding/json"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/engine"
	"github.com/anthony-chaudhary/fak/internal/toolcatalog"
	"github.com/anthony-chaudhary/fak/internal/toolplugin"
)

type gatewayTestPlugin struct {
	profile toolplugin.Profile
	apply   func(context.Context, toolplugin.Input) (toolplugin.Decision, error)
	calls   int
}

func (p *gatewayTestPlugin) Profile() toolplugin.Profile { return p.profile }
func (p *gatewayTestPlugin) Apply(ctx context.Context, in toolplugin.Input) (toolplugin.Decision, error) {
	p.calls++
	return p.apply(ctx, in)
}

func gatewayPinned(id string, stage toolplugin.Stage, precedence int) toolplugin.Profile {
	return toolplugin.Profile{ID: id, Version: "1.0.0", Digest: "sha256:0123456789abcdef", Stage: stage, Timeout: toolplugin.Duration(time.Second), Fallback: toolplugin.ActionDeny, Precedence: precedence}
}

func newPluginTestServer(t *testing.T, plugins []toolplugin.Plugin, preferences toolplugin.PreferenceLayers) *Server {
	t.Helper()
	abi.ResetForTest()
	abi.RegisterRegionBackend(inlineBackend{})
	abi.RegisterEngine("test", echoEngine{})
	abi.RegisterEngine("mock", engine.MockEngine)
	abi.RegisterAdjudicator(0, toolAdj{})
	srv, err := New(Config{EngineID: "test", Model: "test-model", VDSO: true, ToolPlugins: plugins, ToolPreferences: preferences})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	t.Cleanup(srv.Close)
	return srv
}

func TestGatewayPluginLiveTraceTransformReadjudicateExecuteAdmit(t *testing.T) {
	transform := &gatewayTestPlugin{profile: gatewayPinned("canonical", toolplugin.StageCanonicalize, 1), apply: func(_ context.Context, in toolplugin.Input) (toolplugin.Decision, error) {
		return toolplugin.Decision{Action: toolplugin.ActionTransform, Proposal: &toolplugin.Proposal{Tool: in.Proposal.Tool, Args: json.RawMessage(`{"normalized":true}`)}, Reason: "NORMALIZED"}, nil
	}}
	admit := &gatewayTestPlugin{profile: gatewayPinned("admit", toolplugin.StageResultAdmit, 2), apply: func(_ context.Context, in toolplugin.Input) (toolplugin.Decision, error) {
		if len(in.Result) == 0 {
			return toolplugin.Decision{Action: toolplugin.ActionQuarantine, Reason: "EMPTY_RESULT"}, nil
		}
		return toolplugin.Decision{Action: toolplugin.ActionNarrow, Reason: "RESULT_OK"}, nil
	}}
	srv := newPluginTestServer(t, []toolplugin.Plugin{admit, transform}, toolplugin.PreferenceLayers{Organization: toolplugin.Preference{Disclosure: "minimal"}, User: toolplugin.Preference{TransformMode: "auto"}})
	wv, env, trace, pref, err := srv.syscallWithPlugins(context.Background(), "allow_echo", `{"raw":"yes"}`, false, "", "trace-plugin-live", toolplugin.Preference{WaitMode: "background"})
	if err != nil {
		t.Fatal(err)
	}
	if wv.Kind != "ALLOW" || env == nil || env.Content != `{"normalized":true}` {
		t.Fatalf("verdict/result = %+v %+v", wv, env)
	}
	stages := make([]string, len(trace))
	for i, event := range trace {
		stages[i] = event.Stage
	}
	if want := []string{"canonicalize", "floor", "execute", "result_admit"}; !reflect.DeepEqual(stages, want) {
		t.Fatalf("trace=%v, want=%v", stages, want)
	}
	if pref.Disclosure != "minimal" || pref.TransformMode != "auto" || pref.WaitMode != "background" {
		t.Fatalf("preference provenance/value lost: %+v", pref)
	}
	if transform.calls != 1 || admit.calls != 1 {
		t.Fatalf("plugin calls transform=%d admit=%d", transform.calls, admit.calls)
	}
}

func TestGatewayPluginPolicyDenialNeverInvokesExecutorOrWideningPlugin(t *testing.T) {
	plugin := &gatewayTestPlugin{profile: gatewayPinned("would-widen", toolplugin.StageAdjudicate, 1), apply: func(context.Context, toolplugin.Input) (toolplugin.Decision, error) {
		return toolplugin.Decision{Action: toolplugin.ActionNarrow, Reason: "USER_ALLOW"}, nil
	}}
	srv := newPluginTestServer(t, []toolplugin.Plugin{plugin}, toolplugin.PreferenceLayers{})
	wv, env, trace, _, err := srv.syscallWithPlugins(context.Background(), "deny_destroy", `{}`, false, "", "trace-denied", toolplugin.Preference{})
	if err != nil {
		t.Fatal(err)
	}
	if wv.Kind != "DENY" || env != nil || plugin.calls != 0 {
		t.Fatalf("floor widened: verdict=%+v env=%+v calls=%d trace=%+v", wv, env, plugin.calls, trace)
	}
	if len(trace) != 1 || trace[0].Stage != "floor" {
		t.Fatalf("denied call crossed later stages: %+v", trace)
	}
}

func TestGatewayNoPluginSyscallWireRemainsLegacyShape(t *testing.T) {
	srv := newPluginTestServer(t, nil, toolplugin.PreferenceLayers{})
	params := json.RawMessage(`{"name":"fak_syscall","arguments":{"tool":"allow_echo","arguments":{"x":1},"trace_id":"legacy-shape"}}`)
	got, rpcErr := srv.callTool(context.Background(), params)
	if rpcErr != nil {
		t.Fatal(rpcErr)
	}
	body, err := json.Marshal(got)
	if err != nil {
		t.Fatal(err)
	}
	if json.Valid(body) == false {
		t.Fatalf("invalid wire: %s", body)
	}
	if strings.Contains(string(body), `"plugin_trace"`) || strings.Contains(string(body), `"effective_preferences"`) {
		t.Fatalf("zero-config response shape changed: %s", body)
	}
}

func TestGatewayPluginMCPResponseExposesTraceAndPreferenceProvenance(t *testing.T) {
	transform := &gatewayTestPlugin{profile: gatewayPinned("canonical", toolplugin.StageCanonicalize, 1), apply: func(_ context.Context, in toolplugin.Input) (toolplugin.Decision, error) {
		return toolplugin.Decision{Action: toolplugin.ActionTransform, Proposal: &toolplugin.Proposal{Tool: in.Proposal.Tool, Args: json.RawMessage(`{"mcp":true}`)}}, nil
	}}
	srv := newPluginTestServer(t, []toolplugin.Plugin{transform}, toolplugin.PreferenceLayers{Organization: toolplugin.Preference{Disclosure: "minimal"}})
	params := json.RawMessage(`{"name":"fak_syscall","arguments":{"tool":"allow_echo","arguments":{"raw":1},"trace_id":"plugin-wire","preferences":{"wait_mode":"background"}}}`)
	got, rpcErr := srv.callTool(context.Background(), params)
	if rpcErr != nil {
		t.Fatal(rpcErr)
	}
	body, err := json.Marshal(got)
	if err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{`plugin_trace`, `effective_preferences`, `canonicalize`, `wait_mode`, `background`, `disclosure`, `minimal`} {
		if !strings.Contains(string(body), want) {
			t.Errorf("wire missing %s: %s", want, body)
		}
	}
}

func gatewayCatalogRegistration(t *testing.T) (toolcatalog.Registration, toolcatalog.Snapshot) {
	t.Helper()
	skill := `---
name: allow_echo
description: echo through the canonical gateway tool
---
` + "```fak-program" + `
{"version":"fak.skill-program/v1","name":"allow_echo","input_schema":{"type":"object"},"executor":{"argv":["fak","echo"]},"aliases":{"codex":"functions.shell_command"}}
` + "```"
	registration, err := toolcatalog.CompileSkill([]byte(skill), "skills/allow-echo/SKILL.md")
	if err != nil {
		t.Fatal(err)
	}
	snapshot, err := toolcatalog.Expose([]toolcatalog.Registration{registration}, []string{"allow_echo"}, "codex")
	if err != nil {
		t.Fatal(err)
	}
	return registration, snapshot
}

func TestCatalogAliasRunsCanonicalPolicyPluginAndExecutionPath(t *testing.T) {
	srv := newTestServer(t)
	registration, snapshot := gatewayCatalogRegistration(t)
	wv, env, trace, _, err := srv.syscallCatalogTool(context.Background(), snapshot, []toolcatalog.Registration{registration}, "functions.shell_command", `{"raw":"yes"}`, false, "", "trace-catalog", toolplugin.Preference{})
	if err != nil {
		t.Fatal(err)
	}
	if wv.Kind != "ALLOW" || env == nil || env.Content != `{"raw":"yes"}` {
		t.Fatalf("verdict=%+v env=%+v", wv, env)
	}
	if len(trace) != 2 || trace[0].Stage != "floor" || trace[1].Stage != "execute" {
		t.Fatalf("plugin trace = %+v", trace)
	}
}

func TestCatalogAliasUnknownAndStaleFailBeforeExecution(t *testing.T) {
	srv := newTestServer(t)
	registration, snapshot := gatewayCatalogRegistration(t)
	for _, tc := range []struct {
		name          string
		visible       string
		registrations []toolcatalog.Registration
	}{
		{name: "unknown alias", visible: "allow_echo", registrations: []toolcatalog.Registration{registration}},
		{name: "missing registration", visible: "functions.shell_command"},
		{name: "stale registration", visible: "functions.shell_command", registrations: []toolcatalog.Registration{func() toolcatalog.Registration { r := registration; r.Program.Description = "mutated"; return r }()}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			wv, env, trace, _, err := srv.syscallCatalogTool(context.Background(), snapshot, tc.registrations, tc.visible, `{}`, false, "", "trace-refused", toolplugin.Preference{})
			if err != nil || wv.Kind != "DENY" || wv.By != "toolcatalog" || env != nil || trace != nil {
				t.Fatalf("verdict=%+v env=%+v trace=%+v err=%v", wv, env, trace, err)
			}
		})
	}
}
