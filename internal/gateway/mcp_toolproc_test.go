package gateway

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/toolproc"
	"github.com/anthony-chaudhary/fak/internal/toolprocgate"
)

func mcpToolprocTestJournal(t *testing.T) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "journal.jsonl")
	t.Setenv(mcpToolprocEnvJournal, path)
	mcpToolprocReset()
	toolprocgate.Reset()
	t.Cleanup(func() {
		mcpToolprocReset()
		toolprocgate.Reset()
	})
	return path
}

func foldTestJournal(t *testing.T, path string) toolproc.Table {
	t.Helper()
	f, err := os.Open(path)
	if err != nil {
		t.Fatalf("open journal: %v", err)
	}
	defer f.Close()
	events, err := toolproc.ParseEvents(f)
	if err != nil {
		t.Fatalf("parse journal: %v", err)
	}
	tab, err := toolproc.Fold(events, 1<<62, toolproc.Config{})
	if err != nil {
		t.Fatalf("fold journal: %v", err)
	}
	return tab
}

// A brokered fak_syscall becomes a DONE tool process in the journal — the
// seam-3 observation half, driven through the REAL dispatch path.
func TestMCPSyscallJournalsSpawnExit(t *testing.T) {
	journal := mcpToolprocTestJournal(t)
	srv := newTestServer(t)
	resp := srv.dispatchRPC(context.Background(), []byte(
		`{"jsonrpc":"2.0","id":"req-1","method":"tools/call",`+
			`"params":{"name":"fak_syscall","arguments":{"tool":"search_kb","arguments":{},"trace_id":"tp-trace-1"}}}`))
	if resp == nil || resp.Error != nil {
		t.Fatalf("fak_syscall round-trip failed: %+v", resp)
	}
	tab := foldTestJournal(t, journal)
	if len(tab.Procs) != 1 {
		t.Fatalf("journal procs = %+v, want exactly one", tab.Procs)
	}
	p := tab.Procs[0]
	if p.CallID != "tp-trace-1" || p.Tool != "search_kb" || p.State != toolproc.StateDone {
		t.Errorf("proc = %+v, want tp-trace-1/search_kb DONE", p)
	}
}

// notifications/cancelled kills the correlated call: journaled AND armed in
// the seam-2 revocation table (a racing completion quarantines).
func TestMCPCancelledKillsAndArmsRevocation(t *testing.T) {
	journal := mcpToolprocTestJournal(t)
	srv := newTestServer(t)

	ctx := mcpWithRequestID(context.Background(), json.RawMessage(`"req-9"`))
	callID := mcpToolprocSpawn(ctx, "tp-trace-9", "slow_tool")
	if callID != "tp-trace-9" {
		t.Fatalf("spawn call id = %q", callID)
	}
	// The real dispatch path routes the id-less frame to the notify handler.
	if r := srv.dispatchRPC(context.Background(), []byte(
		`{"jsonrpc":"2.0","method":"notifications/cancelled","params":{"requestId":"req-9"}}`)); r != nil {
		t.Fatalf("notification must produce no response, got %+v", r)
	}
	reason, killed := toolprocgate.KilledReason("tp-trace-9")
	if !killed || reason != mcpToolprocKillReason {
		t.Fatalf("revocation table: killed=%t reason=%q, want MCP_CANCELLED", killed, reason)
	}
	tab := foldTestJournal(t, journal)
	p := tab.Procs[0]
	if p.State != toolproc.StateKilled || p.KillReason != mcpToolprocKillReason {
		t.Errorf("proc = %+v, want KILLED citing MCP_CANCELLED", p)
	}
}

// An unmatched cancel is a silent no-op: nothing killed, nothing journaled.
func TestMCPCancelledUnknownRequestIsNoOp(t *testing.T) {
	journal := mcpToolprocTestJournal(t)
	srv := newTestServer(t)
	if r := srv.dispatchRPC(context.Background(), []byte(
		`{"jsonrpc":"2.0","method":"notifications/cancelled","params":{"requestId":"ghost"}}`)); r != nil {
		t.Fatalf("notification must produce no response, got %+v", r)
	}
	if _, err := os.Stat(journal); !os.IsNotExist(err) {
		t.Errorf("unmatched cancel must journal nothing (stat err=%v)", err)
	}
}

// notifications/progress pulses a known call and drops an unknown token.
func TestMCPProgressPulsesKnownCallOnly(t *testing.T) {
	journal := mcpToolprocTestJournal(t)
	srv := newTestServer(t)
	mcpToolprocSpawn(context.Background(), "tp-trace-5", "train")
	srv.mcpToolprocNotify("notifications/progress",
		json.RawMessage(`{"progressToken":"tp-trace-5","progress":3,"total":10}`))
	srv.mcpToolprocNotify("notifications/progress",
		json.RawMessage(`{"progressToken":"nobody","progress":1}`))
	tab := foldTestJournal(t, journal)
	p := tab.Procs[0]
	if p.Pulses != 1 || p.State != toolproc.StateRunning {
		t.Errorf("proc = %+v, want RUNNING with exactly 1 pulse", p)
	}
}

// A client-reused trace id becomes a new generation (trace@2), so the shared
// journal still folds instead of refusing on duplicate spawn.
func TestMCPReusedTraceGetsGeneration(t *testing.T) {
	journal := mcpToolprocTestJournal(t)
	ctx := context.Background()
	first := mcpToolprocSpawn(ctx, "tp-trace-7", "x")
	mcpToolprocExit(first, nil)
	second := mcpToolprocSpawn(ctx, "tp-trace-7", "x")
	mcpToolprocExit(second, nil)
	if first != "tp-trace-7" || second != "tp-trace-7@2" {
		t.Fatalf("generations = %q, %q", first, second)
	}
	tab := foldTestJournal(t, journal)
	if len(tab.Procs) != 2 || tab.Counts.Done != 2 {
		t.Errorf("table = %+v, want two DONE generations", tab.Counts)
	}
}
