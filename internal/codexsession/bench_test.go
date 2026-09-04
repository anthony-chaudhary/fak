package codexsession

import (
	"encoding/json"
	"path/filepath"
	"testing"

	"github.com/anthony-chaudhary/fak/pkg/harnesskit"
)

// BenchmarkCodexAdapter measures adapter construction and notification projection
// across core Codex JSON-RPC protocol events.
func BenchmarkCodexAdapter(b *testing.B) {
	workspace := b.TempDir()
	absWorkspace, err := filepath.Abs(workspace)
	if err != nil {
		b.Fatal(err)
	}

	sink := func(harnesskit.Envelope) error { return nil }
	cfg := Config{
		Command:   "codex",
		Args:      []string{"app-server"},
		Workspace: absWorkspace,
		Version:   "codex-cli 1.2.3",
		RunID:     "bench-run-1",
		Sink:      sink,
	}

	adapter, err := New(cfg)
	if err != nil {
		b.Fatal(err)
	}

	emit := func(harnesskit.EventType, string, string, any) error {
		return nil
	}

	deltaMsg := rpcMessage{
		Method: "item/agentMessage/delta",
		Params: json.RawMessage(`{"threadId":"t1","turnId":"turn-1","itemId":"msg-1","delta":"streaming chunk"}`),
	}
	toolStartedMsg := rpcMessage{
		Method: "item/started",
		Params: json.RawMessage(`{"threadId":"t1","turnId":"turn-1","item":{"id":"cmd-1","type":"commandExecution","command":"git status","status":"inProgress"}}`),
	}
	toolCompletedMsg := rpcMessage{
		Method: "item/completed",
		Params: json.RawMessage(`{"threadId":"t1","turnId":"turn-1","item":{"id":"cmd-1","type":"commandExecution","command":"git status","status":"completed","aggregatedOutput":"clean"}}`),
	}
	usageMsg := rpcMessage{
		Method: "thread/tokenUsage/updated",
		Params: json.RawMessage(`{"threadId":"t1","turnId":"turn-1","tokenUsage":{"inputTokens":1024,"outputTokens":256}}`),
	}

	b.Run("New", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			a, err := New(cfg)
			if err != nil || a == nil {
				b.Fatalf("New failed: %v", err)
			}
		}
	})

	b.Run("NotificationDelta", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			done, err := adapter.notification(emit, deltaMsg)
			if err != nil || done {
				b.Fatalf("unexpected delta notification result: done=%v err=%v", done, err)
			}
		}
	})

	b.Run("NotificationToolLifecycle", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			if _, err := adapter.notification(emit, toolStartedMsg); err != nil {
				b.Fatal(err)
			}
			if _, err := adapter.notification(emit, toolCompletedMsg); err != nil {
				b.Fatal(err)
			}
		}
	})

	b.Run("NotificationUsage", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			done, err := adapter.notification(emit, usageMsg)
			if err != nil || done {
				b.Fatalf("unexpected usage notification result: done=%v err=%v", done, err)
			}
		}
	})
}

func TestBenchmarkCodexAdapterSanity(t *testing.T) {
	sink := func(harnesskit.Envelope) error { return nil }
	cfg := Config{
		Command:   "codex",
		Args:      []string{"app-server"},
		Workspace: t.TempDir(),
		Version:   "codex-cli 1.2.3",
		RunID:     "sanity-run-1",
		Sink:      sink,
	}
	adapter, err := New(cfg)
	if err != nil {
		t.Fatalf("New failed: %v", err)
	}

	var emitted []harnesskit.EventType
	emit := func(typ harnesskit.EventType, _, _ string, _ any) error {
		emitted = append(emitted, typ)
		return nil
	}

	deltaMsg := rpcMessage{
		Method: "item/agentMessage/delta",
		Params: json.RawMessage(`{"threadId":"t1","turnId":"turn-1","itemId":"msg-1","delta":"chunk"}`),
	}
	done, err := adapter.notification(emit, deltaMsg)
	if err != nil || done {
		t.Fatalf("notification delta failed: done=%v err=%v", done, err)
	}
	if len(emitted) != 1 || emitted[0] != harnesskit.EventMessageDelta {
		t.Fatalf("unexpected emitted events: %v", emitted)
	}
}
