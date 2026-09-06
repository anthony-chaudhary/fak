package gateway

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/ctxplan"
	"github.com/anthony-chaudhary/fak/internal/engine"
	"github.com/anthony-chaudhary/fak/internal/memq"
	"github.com/anthony-chaudhary/fak/internal/modelroute"
	"github.com/anthony-chaudhary/fak/internal/toolplugin"
)

type nestedNativeEngine struct{ calls atomic.Int64 }

func (*nestedNativeEngine) Caps() []abi.Capability { return nil }
func (e *nestedNativeEngine) Complete(ctx context.Context, c *abi.ToolCall) (*abi.Result, error) {
	e.calls.Add(1)
	return echoEngine{}.Complete(ctx, c)
}

type nestedNativeFloor struct{ deny atomic.Bool }

func (*nestedNativeFloor) Caps() []abi.Capability { return nil }
func (*nestedNativeFloor) Admit(context.Context, *abi.ToolCall, *abi.Result) abi.Verdict {
	return abi.Verdict{Kind: abi.VerdictDeny, Reason: abi.ReasonPolicyBlock, By: "native-result-test"}
}
func (f *nestedNativeFloor) Adjudicate(context.Context, *abi.ToolCall) abi.Verdict {
	if f.deny.Load() {
		return abi.Verdict{Kind: abi.VerdictDeny, Reason: abi.ReasonPolicyBlock, By: "native-test"}
	}
	return abi.Verdict{Kind: abi.VerdictAllow, By: "native-test"}
}

// Both transport paths must use the native implementation after policy/plugin
// admission, preserving bounded bytes and the caller's trace ownership.
func TestNestedNativeMCP(t *testing.T) {
	for _, plugins := range []bool{false, true} {
		t.Run(map[bool]string{false: "legacy", true: "plugins"}[plugins], func(t *testing.T) {
			s := newTestServer(t)
			eng := &nestedNativeEngine{}
			floor := &nestedNativeFloor{}
			abi.RegisterEngine("test", eng)
			abi.RegisterAdjudicator(-1, floor)
			engine.RegisterResidencyGate()
			s.route = modelroute.NewLive(&modelroute.Manifest{Default: modelroute.Plan{Members: []modelroute.Member{{Model: "guard-a"}}}})
			s.roster = parityRoster("remote-only-owner")
			if _, err := s.buildCall(WithPrincipal(context.Background(), "owner"), "allow_echo", `{}`, true, "", ""); err == nil {
				t.Fatal("test remote account must reject this principal")
			}
			localCall, err := s.buildCall(WithPrincipal(context.Background(), "owner"), "fak_context_restore", `{}`, true, "", "")
			if err != nil || localCall.Engine != "local:fak-mcp" {
				t.Fatalf("native call inherited model route: %+v %v", localCall, err)
			}
			localCall.Args.Scope = abi.ScopeTenant
			if v := s.k.Decide(context.Background(), localCall); v.Kind != abi.VerdictAllow {
				t.Fatalf("local native call failed tenant residency: %+v", v)
			}
			plugin := &gatewayTestPlugin{profile: gatewayPinned("native-check", toolplugin.StageAdjudicate, 1), apply: func(context.Context, toolplugin.Input) (toolplugin.Decision, error) {
				return toolplugin.Decision{Action: toolplugin.ActionNarrow}, nil
			}}
			if plugins {
				s.toolPlugins = []toolplugin.Plugin{plugin}
			}
			const trace = "native-owned"
			payload := []byte("bounded native context recovery")
			id := ctxplan.Digest(payload)
			s.bindTraceOwner(trace, "owner")
			s.stashRestore(trace, id, "context", payload)
			ts := httptest.NewServer(s.Handler())
			defer ts.Close()
			call := func(tool, principal string, args any) rpcDecoded {
				t.Helper()
				frame, err := json.Marshal(map[string]any{"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": map[string]any{"name": "fak_syscall", "arguments": map[string]any{"tool": tool, "arguments": args, "trace_id": "outer-" + principal, "read_only": true}}})
				if err != nil {
					t.Fatal(err)
				}
				req, err := http.NewRequest(http.MethodPost, ts.URL+"/mcp", bytes.NewReader(frame))
				if err != nil {
					t.Fatal(err)
				}
				req.Header.Set("Content-Type", "application/json")
				req.Header.Set("X-Fak-Principal", principal)
				resp, err := ts.Client().Do(req)
				if err != nil {
					t.Fatal(err)
				}
				defer resp.Body.Close()
				var rpc rpcDecoded
				if err := json.NewDecoder(resp.Body).Decode(&rpc); err != nil {
					t.Fatal(err)
				}
				return rpc
			}
			restore := ContextRestoreRequest{ID: id, TraceID: trace, Offset: 8, Limit: 6}
			for _, tool := range []string{"fak_context_restore", "functions.fak_context_restore", "mcp__fak__fak_context_restore", "mcp__fak_guard__fak_context_restore"} {
				sc := unwrapToolResult(t, call(tool, "owner", restore))
				var got CtxRestoreResult
				if sc.Result != nil && sc.Result.Meta["route_account"] != "" {
					t.Errorf("native result falsely attributed to model account: %+v", sc.Result.Meta)
				}
				if sc.Result == nil || json.Unmarshal([]byte(sc.Result.Content), &got) != nil || got.Bytes != string(payload[8:14]) || got.TotalBytes != len(payload) || !got.HasMore || got.NextOffset != 14 {
					t.Errorf("%s returned no bounded native restore: %+v", tool, sc)
				}
			}
			sc := unwrapToolResult(t, call("fak_memory_run", "owner", MemoryRequest{Driver: "render"}))
			var memory memq.Result
			if sc.Result == nil || json.Unmarshal([]byte(sc.Result.Content), &memory) != nil || memory.Stats.Rendered == 0 || memory.Stats.EffectsApplied != 0 {
				t.Errorf("memory returned no read-only native result: %+v", sc)
			}
			if rpc := call("fak_context_restore", "other", restore); rpc.Error == nil || !strings.Contains(rpc.Error.Message, "READ_SCOPE_DENIED") {
				t.Errorf("cross-principal read not refused: %+v", rpc)
			}
			if rpc := call("fak_memory_run", "owner", MemoryRequest{Driver: "clean", Apply: true}); rpc.Error == nil {
				t.Error("read-only call accepted apply")
			}
			floor.deny.Store(true)
			for _, tool := range []string{"fak_context_restore", "fak_memory_run"} {
				sc := unwrapToolResult(t, call(tool, "owner", map[string]any{"id": "missing", "driver": "unknown"}))
				if sc.Verdict.Kind != "DENY" {
					t.Errorf("denied input executed: %+v", sc)
				}
			}
			floor.deny.Store(false)
			if plugins {
				for _, action := range []toolplugin.Action{toolplugin.ActionDeny, toolplugin.ActionQuarantine} {
					admitter := &gatewayTestPlugin{profile: gatewayPinned("withhold-native", toolplugin.StageResultAdmit, 2), apply: func(context.Context, toolplugin.Input) (toolplugin.Decision, error) {
						return toolplugin.Decision{Action: action, Reason: "WITHHELD"}, nil
					}}
					s.toolPlugins = []toolplugin.Plugin{plugin, admitter}
					sc := unwrapToolResult(t, call("fak_context_restore", "owner", restore))
					if sc.Verdict.Kind != strings.ToUpper(string(action)) || sc.Result != nil || admitter.calls != 1 {
						t.Errorf("plugin result admission leaked native bytes: %+v", sc)
					}
				}
				s.toolPlugins = []toolplugin.Plugin{plugin}
			}
			abi.RegisterResultAdmitter(0, floor)
			sc = unwrapToolResult(t, call("fak_context_restore", "owner", restore))
			if sc.Verdict.Kind != "DENY" || sc.Result == nil || sc.Result.Status != "ERROR" || strings.Contains(sc.Result.Content, string(payload[8:14])) {
				t.Errorf("result admission leaked native bytes: %+v", sc)
			}
			if eng.calls.Load() != 0 {
				t.Errorf("fallback engine executed %d times", eng.calls.Load())
			}
			if plugins && plugin.calls == 0 {
				t.Error("plugin control skipped")
			}
		})
	}
}
