package gateway

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strconv"
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

// A brokered MCP call runs in-process: it exposes no PID/cancel handle, so its
// spawn row is stamped advice-only coverage — the leak boundary can observe and
// quarantine it but cannot kill it. Absence of a PID must be VISIBLE on the
// folded proc, never silently treated as full coverage (#2363).
func TestMCPSpawnStampsAdviceOnlyCoverage(t *testing.T) {
	journal := mcpToolprocTestJournal(t)
	callID := mcpToolprocSpawn(context.Background(), "tp-trace-cov", "search_kb")
	mcpToolprocExit(callID, nil)
	tab := foldTestJournal(t, journal)
	p := tab.Procs[0]
	if p.Coverage != toolproc.CoverageAdviceOnly {
		t.Errorf("proc coverage = %q, want %q (no PID handle bound)", p.Coverage, toolproc.CoverageAdviceOnly)
	}
}

// The spawn row folds under the AgentRun envelope this server runs beneath, so
// a brokered tool-call process shares the SAME run-boundary session as ordinary
// spawned agents — not an opaque "mcp" tag that the supervisor can't correlate.
func TestMCPSpawnFoldsUnderAgentRunSession(t *testing.T) {
	journal := mcpToolprocTestJournal(t)
	t.Setenv(toolprocgate.EnvAgentRunID, "run-42")
	callID := mcpToolprocSpawn(context.Background(), "tp-trace-run", "search_kb")
	mcpToolprocExit(callID, nil)
	tab := foldTestJournal(t, journal)
	if p := tab.Procs[0]; p.Session != "run-42" {
		t.Errorf("proc session = %q, want the AgentRun id run-42", p.Session)
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

// The respawn-counter map is bounded like its byRPC sibling: a gateway brokering
// fak_syscall across many unique trace ids (a fresh process-unique trace per call
// when no default trace is pinned) must not grow mcpTP.spawned for the process
// lifetime (#3287). Spawning well past the cap holds the map at mcpToolprocMaxLive,
// evicting oldest-first — so a re-used-after-eviction trace restarts its generation
// suffix while a still-live recent trace keeps incrementing. Journal is off: this
// pins the in-memory bound only, with no per-spawn disk I/O.
func TestMCPSpawnCounterMapIsBounded(t *testing.T) {
	t.Setenv(mcpToolprocEnvJournal, "off")
	mcpToolprocReset()
	t.Cleanup(mcpToolprocReset)
	ctx := context.Background()

	// The very first trace is the eviction victim once the cap overflows.
	const oldest = "trace-oldest"
	mcpToolprocSpawn(ctx, oldest, "tool")
	for i := 1; i < mcpToolprocMaxLive; i++ { // fill exactly to the cap
		mcpToolprocSpawn(ctx, "trace-"+strconv.Itoa(i), "tool")
	}
	mcpTP.mu.Lock()
	atCap := len(mcpTP.spawned)
	mcpTP.mu.Unlock()
	if atCap != mcpToolprocMaxLive {
		t.Fatalf("map size at fill = %d, want the cap %d", atCap, mcpToolprocMaxLive)
	}

	// Overflow with a full cap of fresh traces: the map must stay pinned, never grow.
	for i := 0; i < mcpToolprocMaxLive; i++ {
		mcpToolprocSpawn(ctx, "overflow-"+strconv.Itoa(i), "tool")
	}
	mcpTP.mu.Lock()
	bounded, fifo := len(mcpTP.spawned), len(mcpTP.spawnFIFO)
	_, oldestLive := mcpTP.spawned[oldest]
	mcpTP.mu.Unlock()
	if bounded != mcpToolprocMaxLive {
		t.Fatalf("map size after overflow = %d, want it pinned at %d", bounded, mcpToolprocMaxLive)
	}
	if fifo != mcpToolprocMaxLive {
		t.Fatalf("FIFO length = %d, want == cap %d (one live entry per key)", fifo, mcpToolprocMaxLive)
	}
	if oldestLive {
		t.Errorf("oldest trace %q survived; the FIFO must evict oldest-first", oldest)
	}

	// The evicted oldest, re-spawned, restarts at generation 1 (bare id); a still-live
	// recent trace advances to @2 — the counter semantics survive the bounding.
	if got := mcpToolprocSpawn(ctx, oldest, "tool"); got != oldest {
		t.Errorf("re-spawned evicted trace id = %q, want bare %q (counter reset)", got, oldest)
	}
	recent := "overflow-" + strconv.Itoa(mcpToolprocMaxLive-1)
	if got := mcpToolprocSpawn(ctx, recent, "tool"); got != recent+"@2" {
		t.Errorf("re-spawned live trace id = %q, want %q (counter advanced)", got, recent+"@2")
	}
}
