package gateway

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/anthony-chaudhary/fak/internal/toolproc"
	"github.com/anthony-chaudhary/fak/internal/toolprocgate"
)

// Seam 3 of the tool process table (docs/notes/CONCEPT-TOOL-PROCESS-TABLE-
// 2026-07-02.md): the MCP wire's native lifecycle semantics mapped onto the
// toolproc vocabulary. A brokered `fak_syscall` becomes a tool process
// (spawn/exit rows in the same journal the guard hooks feed), a client
// `notifications/cancelled` becomes a KILL — journaled AND armed in the
// toolprocgate revocation table, so a completion that races the cancel is
// quarantined by the shipped rank-2 ResultAdmitter — and a client
// `notifications/progress` becomes a liveness pulse.
//
// Everything here is FAIL-OPEN observation except the revocation arm:
// journaling errors are swallowed (the wire must never fault because a log
// append failed), while toolprocgate.Kill is a pure in-memory mark.
const (
	// mcpToolprocEnvJournal overrides the journal path; "off" disables
	// journaling (the cancel revocation stays armed regardless).
	mcpToolprocEnvJournal = "FAK_TOOLPROC_JOURNAL"

	// mcpToolprocJournalRel matches the guard hook installer's default, so a
	// workspace's hook-fed and gateway-fed rows land in ONE table. The id
	// namespaces cannot collide: hook rows are toolu_*/hk:*, gateway rows are
	// kernel trace ids.
	mcpToolprocJournalRel = ".fak/toolproc/journal.jsonl"

	// mcpToolprocKillReason is the closed operator-class token a client
	// cancellation cites (the toolproc kill event requires a non-empty token;
	// TOOL_* names are reserved for fold verdicts).
	mcpToolprocKillReason = "MCP_CANCELLED"

	// mcpToolprocMaxLive bounds the rpc-id -> call-id correlation table
	// (FIFO eviction, mirroring toolprocgate's revocation bound).
	mcpToolprocMaxLive = 4096
)

// mcpRequestIDKey carries the JSON-RPC request id from dispatchRPC to the
// tools/call arms without widening every handler signature.
type mcpRequestIDKey struct{}

func mcpWithRequestID(ctx context.Context, id json.RawMessage) context.Context {
	rid := mcpCompactID(id)
	if rid == "" {
		return ctx
	}
	return context.WithValue(ctx, mcpRequestIDKey{}, rid)
}

// mcpCompactID renders a JSON-RPC id (string or number) as a stable key.
func mcpCompactID(id json.RawMessage) string {
	s := strings.TrimSpace(string(id))
	s = strings.Trim(s, `"`)
	return s
}

// mcpToolprocState correlates live calls: JSON-RPC request id -> call id (for
// cancelled), call id -> seen count (for respawn-suffix uniqueness within this
// process's journal writes).
type mcpToolprocState struct {
	mu      sync.Mutex
	byRPC   map[string]string
	rpcFIFO []string
	spawned map[string]int
}

var mcpTP = &mcpToolprocState{byRPC: map[string]string{}, spawned: map[string]int{}}

// mcpToolprocReset clears the correlation state (tests only).
func mcpToolprocReset() {
	mcpTP.mu.Lock()
	defer mcpTP.mu.Unlock()
	mcpTP.byRPC = map[string]string{}
	mcpTP.rpcFIFO = nil
	mcpTP.spawned = map[string]int{}
}

// mcpToolprocSpawn journals the spawn row for a brokered kernel call and
// registers the rpc-id correlation. It returns the journal call id: the trace
// id itself, or trace@N when this process already journaled a spawn for the
// same trace (a client-reused trace id is a NEW process, not a duplicate-spawn
// fold refusal).
func mcpToolprocSpawn(ctx context.Context, traceID, tool string) string {
	if traceID == "" || tool == "" {
		return ""
	}
	mcpTP.mu.Lock()
	mcpTP.spawned[traceID]++
	n := mcpTP.spawned[traceID]
	callID := traceID
	if n > 1 {
		callID = traceID + "@" + strconv.Itoa(n)
	}
	if rid, ok := ctx.Value(mcpRequestIDKey{}).(string); ok && rid != "" {
		if len(mcpTP.rpcFIFO) >= mcpToolprocMaxLive {
			delete(mcpTP.byRPC, mcpTP.rpcFIFO[0])
			mcpTP.rpcFIFO = mcpTP.rpcFIFO[1:]
		}
		mcpTP.byRPC[rid] = callID
		mcpTP.rpcFIFO = append(mcpTP.rpcFIFO, rid)
	}
	mcpTP.mu.Unlock()
	mcpToolprocAppend(toolproc.Event{
		Kind: toolproc.EvSpawn, CallID: callID, Tool: tool,
		Session: "mcp", AtMS: time.Now().UnixMilli(),
	})
	return callID
}

// mcpToolprocExit journals the exit row. A deny verdict is still a completed
// call (deny-as-value); only a transport/build fault is status=error.
func mcpToolprocExit(callID string, callErr error) {
	if callID == "" {
		return
	}
	status := "ok"
	if callErr != nil {
		status = "error"
	}
	mcpToolprocAppend(toolproc.Event{
		Kind: toolproc.EvExit, CallID: callID, AtMS: time.Now().UnixMilli(), Status: status,
	})
}

// mcpToolprocNotify routes the lifecycle notifications the RPC loop previously
// dropped. Unknown methods and unmatched correlations are silently ignored —
// a notification is best-effort by protocol contract.
func (s *Server) mcpToolprocNotify(method string, params json.RawMessage) {
	switch method {
	case "notifications/cancelled":
		var p struct {
			RequestID json.RawMessage `json:"requestId"`
		}
		if json.Unmarshal(params, &p) != nil {
			return
		}
		rid := mcpCompactID(p.RequestID)
		if rid == "" {
			return
		}
		mcpTP.mu.Lock()
		callID, ok := mcpTP.byRPC[rid]
		mcpTP.mu.Unlock()
		if !ok {
			return
		}
		// The enforcement half: any completion for this call that still folds
		// through result admission is quarantined (TOOL_RESULT_AFTER_KILL).
		toolprocgate.Kill(callID, mcpToolprocKillReason)
		mcpToolprocAppend(toolproc.Event{
			Kind: toolproc.EvKill, CallID: callID, AtMS: time.Now().UnixMilli(),
			Reason: mcpToolprocKillReason,
		})
	case "notifications/progress":
		var p struct {
			ProgressToken json.RawMessage `json:"progressToken"`
			Progress      float64         `json:"progress"`
			Total         float64         `json:"total"`
		}
		if json.Unmarshal(params, &p) != nil {
			return
		}
		token := mcpCompactID(p.ProgressToken)
		if token == "" {
			return
		}
		// A token is only meaningful when it names a call this process spawned
		// (clients echo the trace id); anything else is dropped, never guessed.
		mcpTP.mu.Lock()
		callID, ok := mcpTP.byRPC[token]
		if !ok {
			if _, seen := mcpTP.spawned[token]; seen {
				callID, ok = token, true
			}
		}
		mcpTP.mu.Unlock()
		if !ok {
			return
		}
		mcpToolprocAppend(toolproc.Event{
			Kind: toolproc.EvPulse, CallID: callID, AtMS: time.Now().UnixMilli(),
			Done: p.Progress, Total: p.Total, Via: "mcp-progress",
		})
	}
}

// mcpToolprocAppend best-effort-appends one journal row. Failures are
// swallowed by design: observation must never fault the wire.
func mcpToolprocAppend(ev toolproc.Event) {
	path := strings.TrimSpace(os.Getenv(mcpToolprocEnvJournal))
	switch strings.ToLower(path) {
	case "off", "0", "none":
		return
	case "":
		path = mcpToolprocJournalRel
	}
	if toolproc.ValidateEvent(ev) != nil {
		return
	}
	line, err := json.Marshal(ev)
	if err != nil {
		return
	}
	if dir := filepath.Dir(path); dir != "" && dir != "." {
		if os.MkdirAll(dir, 0o755) != nil {
			return
		}
	}
	f, err := os.OpenFile(path, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644)
	if err != nil {
		return
	}
	defer f.Close()
	_, _ = f.Write(append(line, '\n'))
}
