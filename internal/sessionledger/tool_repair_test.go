package sessionledger

import (
	"encoding/json"
	"strings"
	"testing"
)

func TestToolRepairCrashCutsDoNotDuplicateMutatingEffects(t *testing.T) {
	tests := []struct {
		name          string
		cut           func(*testing.T, *Ledger, *int)
		wantState     ToolRepairState
		wantReason    ToolRepairReason
		wantAutoRetry bool
		wantEffects   int
	}{
		{
			name:          "pre-start is safe to retry",
			cut:           func(*testing.T, *Ledger, *int) {},
			wantState:     ToolNeverStarted,
			wantAutoRetry: true,
			wantEffects:   0,
		},
		{
			name: "post-start pre-result fences the unknown outcome",
			cut: func(t *testing.T, l *Ledger, effects *int) {
				t.Helper()
				if _, err := l.MarkToolDispatchStarted("trace", "call-1"); err != nil {
					t.Fatal(err)
				}
				*effects++
			},
			wantState:     ToolStartedOutcomeUnknown,
			wantReason:    ToolOutcomeUnknown,
			wantAutoRetry: false,
			wantEffects:   1,
		},
		{
			name: "post-result reuses the completed receipt",
			cut: func(t *testing.T, l *Ledger, effects *int) {
				t.Helper()
				if _, err := l.MarkToolDispatchStarted("trace", "call-1"); err != nil {
					t.Fatal(err)
				}
				*effects++
				if _, err := l.RecordToolTerminal("trace", ToolTerminal{
					CallID:  "call-1",
					Status:  ToolResult,
					Content: json.RawMessage(`{"updated":true}`),
				}); err != nil {
					t.Fatal(err)
				}
			},
			wantState:     ToolCompleted,
			wantAutoRetry: false,
			wantEffects:   1,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dir := t.TempDir()
			l, err := Open(dir)
			if err != nil {
				t.Fatal(err)
			}
			call := ToolCall{
				ID:        "call-1",
				Name:      "increment_counter",
				Arguments: json.RawMessage(`{"amount":1}`),
			}
			if _, err := l.PrepareToolCall("trace", call); err != nil {
				t.Fatal(err)
			}

			effects := 0
			tt.cut(t, l, &effects)

			reopened, err := Open(dir)
			if err != nil {
				t.Fatal(err)
			}
			receipts, err := reopened.RepairToolCalls("trace")
			if err != nil {
				t.Fatal(err)
			}
			if len(receipts) != 1 {
				t.Fatalf("repair receipts = %d, want 1", len(receipts))
			}
			receipt := receipts[0]
			if receipt.State != tt.wantState || receipt.Reason != tt.wantReason || receipt.AutoRetry != tt.wantAutoRetry {
				t.Fatalf("repair receipt = %+v, want state=%q reason=%q auto_retry=%v", receipt, tt.wantState, tt.wantReason, tt.wantAutoRetry)
			}
			if receipt.Call.ID != call.ID || receipt.Call.Name != call.Name || string(receipt.Call.Arguments) != string(call.Arguments) {
				t.Fatalf("call identity/arguments not restored: %+v", receipt.Call)
			}

			// A resume executor is permitted to run only the explicitly retryable cut.
			if receipt.AutoRetry {
				if _, err := reopened.MarkToolDispatchStarted("trace", receipt.Call.ID); err != nil {
					t.Fatal(err)
				}
				effects++
			}
			if effects != tt.wantEffects+boolInt(tt.wantAutoRetry) {
				t.Fatalf("mutating effect count = %d, want %d; ambiguous call was replayed", effects, tt.wantEffects+boolInt(tt.wantAutoRetry))
			}
			t.Logf("repair state=%s reason=%s auto_retry=%v mutating_effects=%d", receipt.State, receipt.Reason, receipt.AutoRetry, effects)
		})
	}
}

func TestToolRepairTreatsEveryTerminalKindAsCompleted(t *testing.T) {
	for _, status := range []ToolTerminalStatus{ToolResult, ToolRefusal, ToolSkipped} {
		t.Run(string(status), func(t *testing.T) {
			l := Memory()
			if _, err := l.PrepareToolCall("trace", ToolCall{ID: "call-1", Name: "tool", Arguments: json.RawMessage(`{}`)}); err != nil {
				t.Fatal(err)
			}
			terminal := ToolTerminal{CallID: "call-1", Status: status, Content: json.RawMessage(`{"detail":"done"}`)}
			if _, err := l.RecordToolTerminal("trace", terminal); err != nil {
				t.Fatal(err)
			}
			receipts, err := l.RepairToolCalls("trace")
			if err != nil {
				t.Fatal(err)
			}
			if len(receipts) != 1 || receipts[0].State != ToolCompleted || receipts[0].Terminal == nil || receipts[0].Terminal.Status != status {
				t.Fatalf("terminal %q repair = %+v", status, receipts)
			}
		})
	}
}

func TestToolRepairRejectsInvalidTransitions(t *testing.T) {
	l := Memory()
	if _, err := l.MarkToolDispatchStarted("missing", "call-1"); err == nil || !strings.Contains(err.Error(), "trace") {
		t.Fatalf("start without prepare error = %v", err)
	}
	if _, err := l.PrepareToolCall("trace", ToolCall{ID: "call-1", Name: "tool", Arguments: json.RawMessage(`{}`)}); err != nil {
		t.Fatal(err)
	}
	if _, err := l.PrepareToolCall("trace", ToolCall{ID: "call-1", Name: "tool", Arguments: json.RawMessage(`{}`)}); err == nil || !strings.Contains(err.Error(), "already prepared") {
		t.Fatalf("duplicate prepare error = %v", err)
	}
	if _, err := l.MarkToolDispatchStarted("trace", "call-1"); err != nil {
		t.Fatal(err)
	}
	if _, err := l.MarkToolDispatchStarted("trace", "call-1"); err == nil || !strings.Contains(err.Error(), "cannot start") {
		t.Fatalf("duplicate start error = %v", err)
	}
	terminal := ToolTerminal{CallID: "call-1", Status: ToolResult, Content: json.RawMessage(`{"ok":true}`)}
	if _, err := l.RecordToolTerminal("trace", terminal); err != nil {
		t.Fatal(err)
	}
	if _, err := l.RecordToolTerminal("trace", terminal); err == nil || !strings.Contains(err.Error(), "already has") {
		t.Fatalf("duplicate terminal error = %v", err)
	}
}

func boolInt(v bool) int {
	if v {
		return 1
	}
	return 0
}
