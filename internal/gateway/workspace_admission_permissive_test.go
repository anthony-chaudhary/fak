package gateway

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"strconv"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/toolplugin"
)

type permissiveAdmissionTestAdj struct{}

func (permissiveAdmissionTestAdj) Caps() []abi.Capability { return nil }
func (permissiveAdmissionTestAdj) Adjudicate(ctx context.Context, call *abi.ToolCall) abi.Verdict {
	if call.Tool == "bash" {
		return abi.Verdict{Kind: abi.VerdictAllow, By: "test"}
	}
	return (toolAdj{}).Adjudicate(ctx, call)
}

func newPermissiveAdmissionTestServer(t *testing.T, permissive bool) *Server {
	t.Helper()
	// Reuse the fixture's engine and adjudicator registration, but construct the
	// subject with the operator setting rather than mutating a live server.
	newTestServer(t)
	abi.RegisterAdjudicator(-1, permissiveAdmissionTestAdj{})
	srv, err := New(Config{EngineID: "test", Model: "test-model", VDSO: true, WorkspaceAdmissionPermissive: permissive})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(srv.Close)
	return srv
}

// fak-test:runtime fast est=1s
func TestWorkspaceAdmissionPermissiveRetainsDefaultDenyEvidence(t *testing.T) {
	for _, authority := range []string{"absent", "unavailable", "malformed", "unknown-shell"} {
		t.Run(authority, func(t *testing.T) {
			root := t.TempDir()
			SetLeasePlaneProviders(nil, func(context.Context) (LeasePresenceView, error) {
				return LeasePresenceView{ClassifiedLeases: json.RawMessage(`[]`)}, nil
			})
			SetWorkspaceLeaseAdmissionProvider(func(context.Context) (WorkspaceLeaseAdmissionView, error) {
				view := WorkspaceLeaseAdmissionView{WorkspaceRoot: root}
				if authority == "unavailable" {
					return view, errors.New("test authority unavailable")
				}
				if authority == "malformed" {
					view.Leases = []WorkspaceAdmissionLease{{SessionID: "peer"}}
				}
				return view, nil
			})
			if authority == "absent" {
				SetLeasePlaneProviders(nil, nil)
			}
			t.Cleanup(func() {
				SetLeasePlaneProviders(nil, nil)
				SetWorkspaceLeaseAdmissionProvider(nil)
			})
			args := `{"tool":"allow_write","arguments":{"file_path":` + strconv.Quote(filepath.Join(root, "probe.txt")) + `}}`
			if authority == "unknown-shell" {
				args = `{"tool":"bash","arguments":{"command":"python -c 'print(1)'"}}`
			}
			strict, rpcErr := callFakAdjudicate(t, newPermissiveAdmissionTestServer(t, false), args)
			if rpcErr != nil || strict.Verdict.Kind != "DENY" || strict.Verdict.Reason != "DEFAULT_DENY" || strict.Verdict.By != "lease-admission" {
				t.Fatalf("default admission response=%+v error=%+v", strict, rpcErr)
			}
			soft, rpcErr := callFakAdjudicate(t, newPermissiveAdmissionTestServer(t, true), args)
			if rpcErr != nil || soft.Verdict.Kind != "ALLOW" || soft.Verdict.By != "lease-admission-permissive" {
				t.Fatalf("permissive response=%+v error=%+v", soft, rpcErr)
			}
			for key, want := range map[string]string{
				"workspace_admission": "permissive",
				"would_deny":          "DEFAULT_DENY",
				"would_deny_by":       "lease-admission",
				"would_deny_claim":    strict.Verdict.Detail["claim"],
			} {
				if got := soft.Verdict.Detail[key]; got != want || got == "" {
					t.Errorf("evidence %s=%q, want %q", key, got, want)
				}
			}
			assertAdjudicateReceipt(t, soft.Receipt, "allowed")
		})
	}
}

// fak-test:runtime fast est=1s
func TestWorkspaceAdmissionPermissivePreservesPeerAndKernelDenials(t *testing.T) {
	root := t.TempDir()
	if err := os.MkdirAll(filepath.Join(root, "protected"), 0o755); err != nil {
		t.Fatal(err)
	}
	SetLeasePlaneProviders(nil, func(context.Context) (LeasePresenceView, error) {
		return LeasePresenceView{ClassifiedLeases: json.RawMessage(`[]`)}, nil
	})
	SetWorkspaceLeaseAdmissionProvider(func(context.Context) (WorkspaceLeaseAdmissionView, error) {
		return WorkspaceLeaseAdmissionView{WorkspaceRoot: root, Leases: []WorkspaceAdmissionLease{{SessionID: "peer", TreeGlobs: []string{"protected/**"}}}}, nil
	})
	t.Cleanup(func() {
		SetLeasePlaneProviders(nil, nil)
		SetWorkspaceLeaseAdmissionProvider(nil)
	})
	srv := newPermissiveAdmissionTestServer(t, true)
	for _, tc := range []struct{ tool, reason, by string }{
		{"allow_write", "LEASE_HELD", "lease-admission"},
		{"deny_write", abi.ReasonName(abi.ReasonPolicyBlock), "test"},
		{"unknown_tool", "DEFAULT_DENY", "kernel"},
	} {
		t.Run(tc.tool, func(t *testing.T) {
			response, rpcErr := callFakAdjudicate(t, srv, `{"tool":"`+tc.tool+`","arguments":{"file_path":`+strconv.Quote(filepath.Join(root, "protected", "probe.txt"))+`}}`)
			if rpcErr != nil || response.Verdict.Kind != "DENY" || response.Verdict.Reason != tc.reason {
				t.Fatalf("response=%+v error=%+v", response, rpcErr)
			}
			if tc.tool != "unknown_tool" && response.Verdict.By != tc.by {
				t.Errorf("deciding authority=%q, want %q", response.Verdict.By, tc.by)
			}
			if response.Verdict.Detail["workspace_admission"] == "permissive" {
				t.Fatal("hard denial acquired permissive evidence")
			}
			assertAdjudicateReceipt(t, response.Receipt, "denied")
		})
	}
}

// fak-test:runtime fast est=1s
func TestWorkspaceAdmissionPermissiveRetainsKernelTransform(t *testing.T) {
	root := t.TempDir()
	SetLeasePlaneProviders(nil, func(context.Context) (LeasePresenceView, error) {
		return LeasePresenceView{ClassifiedLeases: json.RawMessage(`[]`)}, nil
	})
	SetWorkspaceLeaseAdmissionProvider(func(context.Context) (WorkspaceLeaseAdmissionView, error) {
		return WorkspaceLeaseAdmissionView{WorkspaceRoot: root, Leases: []WorkspaceAdmissionLease{{SessionID: "peer"}}}, nil
	})
	t.Cleanup(func() {
		SetLeasePlaneProviders(nil, nil)
		SetWorkspaceLeaseAdmissionProvider(nil)
	})
	srv := newPermissiveAdmissionTestServer(t, true)
	abi.RegisterAdjudicator(-2, protectedWriteTransformAdj{target: "repaired/probe.txt"})
	response, rpcErr := callFakAdjudicate(t, srv, `{"tool":"allow_write","arguments":{"file_path":"original/probe.txt"}}`)
	if rpcErr != nil || response.Verdict.Kind != "TRANSFORM" {
		t.Fatalf("kernel transform response=%+v error=%+v", response, rpcErr)
	}
	var repaired map[string]any
	if err := json.Unmarshal(response.RepairedArguments, &repaired); err != nil {
		t.Fatal(err)
	}
	if repaired["file_path"] != "repaired/probe.txt" || repaired["content"] != "must-not-run" {
		t.Fatalf("kernel repair lost: %s", response.RepairedArguments)
	}
	if response.Verdict.Detail["would_deny"] != "DEFAULT_DENY" || response.Verdict.Detail["workspace_admission"] != "permissive" {
		t.Fatalf("missing softened admission evidence: %+v", response.Verdict)
	}
	assertAdjudicateReceipt(t, response.Receipt, "transformed")
}

// fak-test:runtime fast est=1s
func TestWorkspaceAdmissionPermissiveSyscallExecution(t *testing.T) {
	for _, plugins := range []bool{false, true} {
		for _, tc := range []struct {
			name               string
			permissive, peer   bool
			tool, kind, reason string
			calls              int64
		}{
			{name: "strict missing atomic guard", tool: "allow_write", kind: "DENY", reason: "DEFAULT_DENY"},
			{name: "permissive missing atomic guard", permissive: true, tool: "allow_write", kind: "ALLOW", calls: 1},
			{name: "permissive peer protected", permissive: true, peer: true, tool: "allow_write", kind: "DENY", reason: "LEASE_HELD"},
			{name: "permissive kernel policy", permissive: true, tool: "deny_write", kind: "DENY", reason: "POLICY_BLOCK"},
		} {
			name := tc.name
			if plugins {
				name = "plugin/" + name
			}
			t.Run(name, func(t *testing.T) {
				root := t.TempDir()
				setPermissiveAdmissionTestCWD(t, root)
				SetLeasePlaneProviders(nil, func(context.Context) (LeasePresenceView, error) {
					return LeasePresenceView{ClassifiedLeases: json.RawMessage(`[]`)}, nil
				})
				SetWorkspaceLeaseAdmissionProvider(func(context.Context) (WorkspaceLeaseAdmissionView, error) {
					view := WorkspaceLeaseAdmissionView{WorkspaceRoot: root}
					if tc.peer {
						view.Leases = []WorkspaceAdmissionLease{{SessionID: "peer", TreeGlobs: []string{"protected/**"}}}
					}
					return view, nil
				})
				t.Cleanup(func() { SetLeasePlaneProviders(nil, nil); SetWorkspaceLeaseAdmissionProvider(nil) })
				engine := &pluginWriteFenceEngine{}
				abi.ResetForTest()
				abi.RegisterRegionBackend(inlineBackend{})
				abi.RegisterEngine("permissive-execution", engine)
				abi.RegisterAdjudicator(0, toolAdj{})
				cfg := Config{EngineID: "permissive-execution", Model: "test-model", VDSO: true, WorkspaceAdmissionPermissive: tc.permissive}
				if plugins {
					cfg.ToolPlugins = []toolplugin.Plugin{&gatewayTestPlugin{
						profile: gatewayPinned("permissive-execution-pass", toolplugin.StageCanonicalize, 1),
						apply: func(context.Context, toolplugin.Input) (toolplugin.Decision, error) {
							return toolplugin.Decision{Action: toolplugin.ActionNarrow, Reason: "UNCHANGED"}, nil
						},
					}}
				}
				srv, err := New(cfg)
				if err != nil {
					t.Fatal(err)
				}
				t.Cleanup(srv.Close)
				response := callMCPTool[SyscallResponse](t, srv, "fak_syscall", map[string]any{
					"tool":      tc.tool,
					"arguments": map[string]any{"file_path": filepath.Join(root, "protected", "probe.txt"), "content": "fake-engine-only"},
					"trace_id":  "permissive-execution-test",
				})
				if response.Verdict.Kind != tc.kind || response.Verdict.Reason != tc.reason {
					t.Fatalf("verdict=%+v, want %s/%s", response.Verdict, tc.kind, tc.reason)
				}
				if got := engine.calls.Load(); got != tc.calls {
					t.Fatalf("engine calls=%d, want %d", got, tc.calls)
				}
				if tc.calls == 1 {
					if response.Verdict.Detail["workspace_admission"] != "permissive" || response.Verdict.Detail["would_deny"] != "DEFAULT_DENY" || response.Verdict.Detail["would_deny_claim"] == "" {
						t.Fatalf("execution lost no-atomic-guard evidence: %+v", response.Verdict)
					}
				}
			})
		}
	}
}

func setPermissiveAdmissionTestCWD(t *testing.T, root string) {
	t.Helper()
	previous, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Join(root, "protected"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.Chdir(root); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := os.Chdir(previous); err != nil {
			t.Error(err)
		}
	})
}

type permissiveEffectiveArgsEngine struct {
	pluginWriteFenceEngine
	args []byte
}

type permissiveShellTransformAdj struct{ command string }

func (permissiveShellTransformAdj) Caps() []abi.Capability { return nil }
func (a permissiveShellTransformAdj) Adjudicate(ctx context.Context, call *abi.ToolCall) abi.Verdict {
	args, _ := json.Marshal(map[string]string{"command": a.command})
	ref, _ := abi.ActiveResolver().Put(ctx, args)
	return abi.Verdict{Kind: abi.VerdictTransform, By: "test-shell-transform", Payload: abi.TransformPayload{NewArgs: ref}}
}

func (e *permissiveEffectiveArgsEngine) Complete(ctx context.Context, call *abi.ToolCall) (*abi.Result, error) {
	args, err := abi.ActiveResolver().Resolve(ctx, call.Args)
	if err != nil {
		return nil, err
	}
	e.args = append([]byte(nil), args...)
	return e.pluginWriteFenceEngine.Complete(ctx, call)
}

// fak-test:runtime fast est=1s
func TestWorkspaceAdmissionPermissiveEffectiveExecutionTargets(t *testing.T) {
	for _, tc := range []struct {
		name         string
		peer, plugin bool
	}{
		{name: "kernel transform into peer", peer: true},
		{name: "kernel transform executed args"},
		{name: "plugin rewrite into peer", peer: true, plugin: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			root := t.TempDir()
			setPermissiveAdmissionTestCWD(t, root)
			SetLeasePlaneProviders(nil, func(context.Context) (LeasePresenceView, error) {
				return LeasePresenceView{ClassifiedLeases: json.RawMessage(`[]`)}, nil
			})
			SetWorkspaceLeaseAdmissionProvider(func(context.Context) (WorkspaceLeaseAdmissionView, error) {
				view := WorkspaceLeaseAdmissionView{WorkspaceRoot: root}
				if tc.peer {
					view.Leases = []WorkspaceAdmissionLease{{SessionID: "peer", TreeGlobs: []string{"protected/**"}}}
				}
				return view, nil
			})
			t.Cleanup(func() { SetLeasePlaneProviders(nil, nil); SetWorkspaceLeaseAdmissionProvider(nil) })
			engine := &permissiveEffectiveArgsEngine{}
			abi.ResetForTest()
			abi.RegisterRegionBackend(inlineBackend{})
			abi.RegisterEngine("effective-args", engine)
			target := filepath.Join(root, "protected", "effective.txt")
			command := "touch " + strconv.Quote(target)
			cfg := Config{EngineID: "effective-args", Model: "test-model", VDSO: true, WorkspaceAdmissionPermissive: true}
			if tc.plugin {
				abi.RegisterAdjudicator(0, permissiveAdmissionTestAdj{})
				cfg.ToolPlugins = []toolplugin.Plugin{&gatewayTestPlugin{
					profile: gatewayPinned("effective-target", toolplugin.StageCanonicalize, 1),
					apply: func(context.Context, toolplugin.Input) (toolplugin.Decision, error) {
						args, _ := json.Marshal(map[string]any{"command": command})
						return toolplugin.Decision{Action: toolplugin.ActionTransform, Proposal: &toolplugin.Proposal{Tool: "bash", Args: args}, Reason: "WRITE_TARGET"}, nil
					},
				}}
			} else {
				abi.RegisterAdjudicator(0, permissiveShellTransformAdj{command: command})
			}
			srv, err := New(cfg)
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(srv.Close)
			response := callMCPTool[SyscallResponse](t, srv, "fak_syscall", map[string]any{
				"tool": "bash", "read_only": true,
				"arguments": map[string]any{"command": "wc -c " + strconv.Quote(filepath.Join(root, "original.txt"))},
			})
			if tc.peer {
				if response.Verdict.Kind != "DENY" || response.Verdict.Reason != "LEASE_HELD" || engine.calls.Load() != 0 {
					t.Fatalf("effective peer write response=%+v engine=%d", response, engine.calls.Load())
				}
			} else {
				if engine.calls.Load() != 1 {
					t.Fatalf("engine calls=%d, want 1", engine.calls.Load())
				}
				var executed map[string]any
				if err := json.Unmarshal(engine.args, &executed); err != nil {
					t.Fatal(err)
				}
				if executed["command"] != command || len(executed) != 1 {
					t.Fatalf("executed original/stale arguments: %s", engine.args)
				}
			}
		})
	}
}
