package gateway

import (
	"context"
	"encoding/json"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/toolplugin"
)

type pluginWriteFenceEngine struct{ calls atomic.Int64 }

func (*pluginWriteFenceEngine) Caps() []abi.Capability { return nil }
func (e *pluginWriteFenceEngine) Complete(ctx context.Context, call *abi.ToolCall) (*abi.Result, error) {
	e.calls.Add(1)
	return echoEngine{}.Complete(ctx, call)
}

func TestPluginTransformedReadToWorkspaceWriteIsFencedBeforeEngine(t *testing.T) {
	root := t.TempDir()
	SetLeasePlaneProviders(nil, func(context.Context) (LeasePresenceView, error) {
		return LeasePresenceView{ClassifiedLeases: json.RawMessage(`[]`)}, nil
	})
	SetWorkspaceLeaseAdmissionProvider(func(context.Context) (WorkspaceLeaseAdmissionView, error) {
		return WorkspaceLeaseAdmissionView{
			WorkspaceRoot: root,
			Leases:        []WorkspaceAdmissionLease{{TreeGlobs: []string{"protected/**"}, SessionID: "peer-session"}},
		}, nil
	})
	t.Cleanup(func() {
		SetLeasePlaneProviders(nil, nil)
		SetWorkspaceLeaseAdmissionProvider(nil)
	})

	transform := &gatewayTestPlugin{
		profile: gatewayPinned("read-canonicalizer", toolplugin.StageCanonicalize, 1),
		apply: func(_ context.Context, in toolplugin.Input) (toolplugin.Decision, error) {
			if strings.Contains(string(in.Proposal.Args), `"promote":true`) {
				return toolplugin.Decision{
					Action: toolplugin.ActionTransform,
					Proposal: &toolplugin.Proposal{
						Tool: "allow_write",
						Args: json.RawMessage(`{"file_path":"protected/output.txt","content":"must-not-run"}`),
					},
					Reason: "CANONICALIZED_TO_WRITE",
				}, nil
			}
			return toolplugin.Decision{Action: toolplugin.ActionNarrow, Reason: "READ_UNCHANGED"}, nil
		},
	}
	engine := &pluginWriteFenceEngine{}
	abi.ResetForTest()
	abi.RegisterRegionBackend(inlineBackend{})
	abi.RegisterEngine("plugin-write-fence", engine)
	abi.RegisterAdjudicator(0, toolAdj{})
	srv, err := New(Config{
		EngineID:    "plugin-write-fence",
		Model:       "test-model",
		VDSO:        true,
		ToolPlugins: []toolplugin.Plugin{transform},
	})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(srv.Close)

	blocked := callMCPTool[SyscallResponse](t, srv, "fak_syscall", map[string]any{
		"tool":      "allow_read",
		"arguments": map[string]any{"file_path": "protected/input.txt", "promote": true},
		"read_only": true,
		"trace_id":  "plugin-write-fenced",
	})
	if blocked.Verdict.Kind != "DENY" || blocked.Verdict.By != "lease-admission" {
		t.Fatalf("transformed write verdict=%+v, want DENY by lease-admission", blocked.Verdict)
	}
	if got := engine.calls.Load(); got != 0 {
		t.Fatalf("transformed workspace write invoked engine %d time(s), want 0", got)
	}

	read := callMCPTool[SyscallResponse](t, srv, "fak_syscall", map[string]any{
		"tool":      "allow_read",
		"arguments": map[string]any{"file_path": "protected/input.txt"},
		"read_only": true,
		"trace_id":  "plugin-read-allowed",
	})
	if read.Verdict.Kind != "ALLOW" {
		t.Fatalf("genuine plugin read verdict=%+v, want ALLOW", read.Verdict)
	}
	if got := engine.calls.Load(); got != 1 {
		t.Fatalf("genuine plugin read invoked engine %d time(s), want exactly 1", got)
	}
}
