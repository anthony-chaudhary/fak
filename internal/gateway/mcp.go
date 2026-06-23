package gateway

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
)

// MCP transport. The kernel is exposed as an MCP server speaking JSON-RPC 2.0,
// hand-rolled on the stdlib (the repo is zero-dependency by design). The default
// transport is stdio with newline-delimited frames — one JSON-RPC message per
// line, the MCP stdio convention — which needs no listener and no auth surface.
// /mcp serves the same dispatch over a single-request POST for an HTTP MCP client.
//
// Methods: initialize, tools/list, tools/call, ping (notifications/initialized is
// accepted and ignored). The primary tools are:
//
//	fak_adjudicate — pre-execution verdict only (k.Decide): the production path for
//	                 a client that executes its own tools.
//	fak_syscall    — adjudicate + execute through the kernel engine (self-contained).
//
// A DENY is a valid adjudication RESULT (deny-as-value), never a JSON-RPC error;
// JSON-RPC errors are reserved for protocol/internal faults.

// JSON-RPC 2.0 error codes (the standard reserved set).
const (
	rpcParseError     = -32700
	rpcInvalidRequest = -32600
	rpcMethodNotFound = -32601
	rpcInvalidParams  = -32602
	rpcInternalError  = -32603
)

type rpcRequest struct {
	JSONRPC string          `json:"jsonrpc"`
	ID      json.RawMessage `json:"id"` // absent => notification (no response)
	Method  string          `json:"method"`
	Params  json.RawMessage `json:"params"`
}

type rpcResponse struct {
	JSONRPC string          `json:"jsonrpc"`
	ID      json.RawMessage `json:"id"` // echoed; null on a parse error
	Result  any             `json:"result,omitempty"`
	Error   *rpcError       `json:"error,omitempty"`
}

type rpcError struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
	Data    any    `json:"data,omitempty"`
}

// ServeStdio serves MCP over newline-delimited JSON-RPC on in/out until EOF or ctx
// cancellation. Each input line is one request; each response is one line
// (json.Encoder appends the newline). Notifications (no id) get no response. An
// oversized frame is rejected PER-FRAME (an Invalid Request response) and the
// session continues — one bad frame never tears down the loop.
func (s *Server) ServeStdio(ctx context.Context, in io.Reader, out io.Writer) error {
	br := bufio.NewReader(in)
	enc := json.NewEncoder(out)
	enc.SetEscapeHTML(false)
	// The MCP-over-stdio loop is ready to serve frames; close the boot timeline.
	s.MarkReady()
	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}
		line, tooLong, err := readFrame(br, maxBody)
		if err != nil {
			if err == io.EOF {
				return nil
			}
			return err
		}
		if tooLong {
			_ = enc.Encode(&rpcResponse{JSONRPC: "2.0",
				Error: &rpcError{Code: rpcInvalidRequest, Message: "frame exceeds maximum size"}})
			continue
		}
		if len(bytes.TrimSpace(line)) == 0 {
			continue
		}
		if resp := s.dispatchRPC(ctx, line); resp != nil {
			if err := enc.Encode(resp); err != nil {
				return err
			}
		}
	}
}

// readFrame reads one newline-delimited frame, capping growth at max bytes. If the
// frame exceeds max it keeps draining to the newline (without growing the buffer)
// and returns tooLong=true, so the caller can reject that one frame and continue.
func readFrame(br *bufio.Reader, max int) (line []byte, tooLong bool, err error) {
	var buf []byte
	over := false
	for {
		b, e := br.ReadByte()
		if e != nil {
			if len(buf) > 0 && e == io.EOF {
				return buf, over, nil // final line with no trailing newline
			}
			return nil, false, e
		}
		if b == '\n' {
			return buf, over, nil
		}
		if len(buf) < max {
			buf = append(buf, b)
		} else {
			over = true // stop growing; keep draining to the newline
		}
	}
}

// handleMCPHTTP serves a single JSON-RPC request/response over POST /mcp.
func (s *Server) handleMCPHTTP(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeErr(w, http.StatusMethodNotAllowed, "use POST")
		return
	}
	body, err := io.ReadAll(http.MaxBytesReader(w, r.Body, maxBody))
	if err != nil {
		writeErr(w, http.StatusBadRequest, "request too large or unreadable")
		return
	}
	resp := s.dispatchRPC(r.Context(), body)
	if resp == nil { // a notification
		w.WriteHeader(http.StatusAccepted)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	enc := json.NewEncoder(w)
	enc.SetEscapeHTML(false)
	_ = enc.Encode(resp)
}

// dispatchRPC parses one JSON-RPC frame and routes it. It returns nil for a
// notification (no id) so the caller writes no response.
func (s *Server) dispatchRPC(ctx context.Context, raw []byte) *rpcResponse {
	var req rpcRequest
	if err := json.Unmarshal(raw, &req); err != nil {
		return &rpcResponse{JSONRPC: "2.0", Error: &rpcError{Code: rpcParseError, Message: "parse error"}}
	}
	if len(bytes.TrimSpace(req.ID)) == 0 {
		// Notification (e.g. notifications/initialized): accept, no response.
		return nil
	}
	if req.JSONRPC != "2.0" {
		return &rpcResponse{JSONRPC: "2.0", ID: req.ID,
			Error: &rpcError{Code: rpcInvalidRequest, Message: `jsonrpc must be "2.0"`}}
	}
	result, rerr := s.handleMethod(ctx, req.Method, req.Params)
	if rerr != nil {
		return &rpcResponse{JSONRPC: "2.0", ID: req.ID, Error: rerr}
	}
	return &rpcResponse{JSONRPC: "2.0", ID: req.ID, Result: result}
}

func (s *Server) handleMethod(ctx context.Context, method string, params json.RawMessage) (any, *rpcError) {
	switch method {
	case "initialize":
		return s.initializeResult(params), nil
	case "ping":
		return map[string]any{}, nil
	case "tools/list":
		return map[string]any{"tools": toolDescriptors()}, nil
	case "tools/call":
		return s.callTool(ctx, params)
	default:
		return nil, &rpcError{Code: rpcMethodNotFound, Message: "method not found: " + method}
	}
}

// mcpProtocolVersions is the SINGLE source of truth for the MCP revisions whose
// initialize/tools shape this hand-rolled server is wire-compatible with. The
// negotiator (initializeResult) consults nothing else: supportedProtocols (the
// fast membership set) and defaultProtocol (the answer for an unsupported
// request) are both DERIVED from this list, so adding/removing a revision is a
// one-line edit here. The first entry is the default — what we answer with when
// the client requests a revision we do not support (so we never falsely claim
// support for an arbitrary/future revision with different framing).
var mcpProtocolVersions = []string{"2024-11-05", "2025-03-26", "2025-06-18"}

// defaultProtocol and supportedProtocols are derived from mcpProtocolVersions —
// never edit them directly; edit the list above.
var defaultProtocol = mcpProtocolVersions[0]

var supportedProtocols = func() map[string]bool {
	m := make(map[string]bool, len(mcpProtocolVersions))
	for _, v := range mcpProtocolVersions {
		m[v] = true
	}
	return m
}()

func (s *Server) initializeResult(params json.RawMessage) map[string]any {
	proto := defaultProtocol
	if len(params) > 0 {
		var p struct {
			ProtocolVersion string `json:"protocolVersion"`
		}
		// Negotiate: adopt the client's version only if WE support it, else answer
		// with our own — never echo an unknown revision back as if implemented.
		if err := json.Unmarshal(params, &p); err == nil && supportedProtocols[p.ProtocolVersion] {
			proto = p.ProtocolVersion
		}
	}
	return map[string]any{
		"protocolVersion": proto,
		"capabilities":    map[string]any{"tools": map[string]any{}},
		"serverInfo":      map[string]any{"name": "fak-gateway", "version": s.version},
	}
}

// callTool handles tools/call. The MCP `arguments` object IS a SyscallRequest
// ({tool, arguments, read_only}). A deny returns a normal tool result (deny-as-
// value); only a protocol/build fault is a JSON-RPC error.
func (s *Server) callTool(ctx context.Context, params json.RawMessage) (any, *rpcError) {
	var p struct {
		Name      string          `json:"name"`
		Arguments json.RawMessage `json:"arguments"`
	}
	if err := json.Unmarshal(params, &p); err != nil {
		return nil, &rpcError{Code: rpcInvalidParams, Message: "invalid tools/call params: " + err.Error()}
	}
	switch p.Name {
	case "fak_syscall":
		req := decodeSyscallArgs(p.Arguments)
		req.TraceID = s.traceFor(req.TraceID)
		wv, env, err := s.syscall(ctx, req.Tool, rawArgs(req.Arguments), req.ReadOnly, req.Witness, req.TraceID)
		if err != nil {
			return nil, &rpcError{Code: rpcInvalidParams, Message: err.Error()}
		}
		return mcpToolResult(SyscallResponse{Verdict: wv, Result: env, TraceID: req.TraceID}), nil
	case "fak_adjudicate":
		req := decodeSyscallArgs(p.Arguments)
		req.TraceID = s.traceFor(req.TraceID)
		wv, repaired, err := s.adjudicate(ctx, req.Tool, rawArgs(req.Arguments), req.ReadOnly, req.Witness, req.TraceID)
		if err != nil {
			return nil, &rpcError{Code: rpcInvalidParams, Message: err.Error()}
		}
		resp := SyscallResponse{Verdict: wv, TraceID: req.TraceID}
		if repaired != "" {
			resp.RepairedArguments = json.RawMessage(repaired)
		}
		return mcpToolResult(resp), nil
	case "fak_admit":
		var req AdmitRequest
		_ = json.Unmarshal(p.Arguments, &req)
		req.TraceID = s.traceFor(req.TraceID)
		wv, env, err := s.admit(ctx, req.Tool, rawArgs(req.Result), req.Witness, req.TraceID)
		if err != nil {
			return nil, &rpcError{Code: rpcInvalidParams, Message: err.Error()}
		}
		return mcpToolResult(SyscallResponse{Verdict: wv, Result: env, TraceID: req.TraceID}), nil
	case "fak_changes":
		var req ChangesRequest
		_ = json.Unmarshal(p.Arguments, &req)
		events, cursor := s.changes(req.Since)
		return mcpToolResult(ChangesResponse{Events: events, Cursor: cursor}), nil
	case "fak_revoke":
		var req RevokeRequest
		if err := json.Unmarshal(p.Arguments, &req); err != nil || req.Witness == "" {
			return nil, &rpcError{Code: rpcInvalidParams, Message: "fak_revoke requires a non-empty witness"}
		}
		evicted, te := s.revoke(req.Witness)
		return mcpToolResult(RevokeResponse{Witness: req.Witness, Evicted: evicted, TrustEpoch: te}), nil
	case "fak_context_change":
		var req ContextChangeRequest
		if err := json.Unmarshal(p.Arguments, &req); err != nil {
			return nil, &rpcError{Code: rpcInvalidParams, Message: "invalid fak_context_change arguments: " + err.Error()}
		}
		resp, err := s.contextChange(ctx, req)
		if err != nil {
			return nil, &rpcError{Code: rpcInvalidParams, Message: err.Error()}
		}
		return mcpToolResult(resp), nil
	case "fak_memory_drivers":
		return mcpToolResult(map[string]any{"drivers": s.memoryDrivers()}), nil
	case "fak_memory_explain":
		var req MemoryRequest
		if err := json.Unmarshal(p.Arguments, &req); err != nil {
			return nil, &rpcError{Code: rpcInvalidParams, Message: "invalid fak_memory_explain arguments: " + err.Error()}
		}
		plan, err := s.memoryExplain(req)
		if err != nil {
			return nil, &rpcError{Code: rpcInvalidParams, Message: err.Error()}
		}
		return mcpToolResult(plan), nil
	case "fak_memory_run":
		var req MemoryRequest
		if err := json.Unmarshal(p.Arguments, &req); err != nil {
			return nil, &rpcError{Code: rpcInvalidParams, Message: "invalid fak_memory_run arguments: " + err.Error()}
		}
		res, err := s.memoryRun(ctx, req)
		if err != nil {
			return nil, &rpcError{Code: rpcInvalidParams, Message: err.Error()}
		}
		return mcpToolResult(res), nil
	default:
		return nil, &rpcError{Code: rpcInvalidParams, Message: "unknown tool: " + p.Name}
	}
}

// decodeSyscallArgs parses the MCP tools/call `arguments` object into a SyscallRequest.
// A malformed object yields the zero request (an empty tool name), which the kernel
// rejects downstream — never a panic.
func decodeSyscallArgs(raw json.RawMessage) SyscallRequest {
	var req SyscallRequest
	_ = json.Unmarshal(raw, &req)
	return req
}

// mcpToolResult wraps a SyscallResponse as an MCP tool result: a single text
// content block carrying the JSON. isError stays false — a deny is a successful
// adjudication, surfaced in the verdict, not a tool failure.
func mcpToolResult(v any) map[string]any {
	b, _ := json.Marshal(v)
	return map[string]any{
		"content": []map[string]any{{"type": "text", "text": string(b)}},
		"isError": false,
	}
}

// toolDescriptors is the tools/list payload. The inputSchema is a JSON Schema for
// the {tool, arguments, read_only} shape both tools accept.
func toolDescriptors() []map[string]any {
	schema := json.RawMessage(`{
  "type": "object",
  "properties": {
    "tool": {"type": "string", "description": "the logical tool name to route through the kernel"},
    "arguments": {"description": "the tool arguments: a JSON object, or a JSON-encoded string (the OpenAI function.arguments convention)"},
    "read_only": {"type": "boolean", "description": "hint that the tool is read-only/idempotent (enables vDSO dedup)"},
    "trace_id": {"type": "string", "description": "optional session trace id; omitted means the gateway mints one and returns it"},
    "witness": {"type": "string", "description": "optional external world-state token the call is reading at"}
  },
  "required": ["tool"]
}`)
	return []map[string]any{
		{
			"name":        "fak_adjudicate",
			"description": "Adjudicate a proposed tool call through the fak kernel WITHOUT executing it. Returns the verdict (ALLOW/DENY/TRANSFORM/REQUIRE_WITNESS) and, for a denial, a disposition (RETRYABLE/WAIT/ESCALATE/TERMINAL); for a TRANSFORM, the repaired canonical arguments. Call this before running a tool your own client executes.",
			"inputSchema": schema,
		},
		{
			"name":        "fak_syscall",
			"description": "Adjudicate AND execute a tool call through the fak kernel (dispatch to the registered engine + context-MMU result admission). Returns the verdict and the admitted result. Use when fak should run the tool.",
			"inputSchema": schema,
		},
		{
			"name":        "fak_admit",
			"description": "Submit a tool RESULT your own client executed, to run it through the fak kernel's result-side stack (context-MMU quarantine + IFC source-stamp / per-trace taint ledger) BEFORE you admit it to context. A poisoned/secret-shaped result comes back QUARANTINE with the bytes paged out; the session's taint high-water mark is raised so a later egress is gated. Pass {tool, result, trace_id}. This arms the exfil floor on the path where YOU run the tool (the complement of fak_adjudicate).",
			"inputSchema": json.RawMessage(`{
  "type": "object",
  "properties": {
    "tool": {"type": "string", "description": "the tool name that produced this result (its source class keys the provenance taint)"},
    "result": {"description": "the tool result content: a JSON object, or a JSON-encoded string"},
    "trace_id": {"type": "string", "description": "the session trace this result belongs to (keys the IFC taint ledger)"},
    "witness": {"type": "string", "description": "optional external world-state token the result was read at"}
  },
  "required": ["tool"]
}`),
		},
		{
			"name":        "fak_changes",
			"description": "Drain the cross-agent 'what changed' feed: the typed write Mutations and Revocations observed since your cursor, so you can re-plan or evict your own cache when another agent changed or refuted shared data. Pass {since: <cursor>} (0 = everything retained); returns the events and your next cursor.",
			"inputSchema": json.RawMessage(`{"type":"object","properties":{"since":{"type":"integer","description":"the Seq cursor of the last event you saw (0 = from the start of the retained window)"}}}`),
		},
		{
			"name":        "fak_revoke",
			"description": "Refute an external world-state witness (a git commit / blob hash / lease epoch) found poisoned or stale: every pooled tier-2 entry admitted under it is causally evicted fleet-wide, future re-admission under it is refused, and the eviction is broadcast on the change feed. Pass {witness: <token>}; returns the local eviction count and the new trust epoch.",
			"inputSchema": json.RawMessage(`{"type":"object","properties":{"witness":{"type":"string","description":"the external world-state witness to refute"}},"required":["witness"]}`),
		},
		{
			"name":        "fak_context_change",
			"description": "Request a safe negative-only context mutation against a persisted recall core image. Today this records a tombstone for one page: future Resolve/Recall/working-set assembly skips it, while the original page row and CAS bytes remain available for audit. Pass {image_dir, step, reason, requested_by?, digest?, witness?, action?}; action may be omitted, 'tombstone', or 'tombstone_page'.",
			"inputSchema": json.RawMessage(`{
  "type": "object",
  "properties": {
    "image_dir": {"type": "string", "description": "path to the persisted recall core image directory"},
    "action": {"type": "string", "description": "optional; omit or use tombstone/tombstone_page"},
    "step": {"type": "integer", "description": "page step to suppress from future model-visible recall"},
    "digest": {"type": "string", "description": "optional CAS digest guard; mismatch refuses the request"},
    "reason": {"type": "string", "description": "why the page should be absent from future context"},
    "requested_by": {"type": "string", "description": "agent/operator identity requesting the tombstone"},
    "witness": {"type": "string", "description": "optional external witness supporting the request"}
  },
  "required": ["image_dir", "step", "reason"]
}`),
		},
		{
			"name":        "fak_memory_drivers",
			"description": "List the built-in memory STRATEGIES (recall/render/clean/compact/dream). Each is a composable query in the memq algebra (scan|filter|rank|limit|budget|render|tombstone|consolidate|reclassify|prune), not a hardcoded function — 'build SQL, not a specific query'. Returns each driver's name, doc, and compiled plan so you can see the pipeline and author your own.",
			"inputSchema": json.RawMessage(`{"type":"object","properties":{}}`),
		},
		{
			"name":        "fak_memory_explain",
			"description": "EXPLAIN a memory query as a plan WITHOUT executing it — every step, which steps are effects, and which mutate durable state (and so are proposal-only). Pass {driver} for a built-in, or {query} with an inline authored memq Query ({intent, ops:[{kind,...}]}). This is the 'step through it before you run it' surface.",
			"inputSchema": memoryInputSchema,
		},
		{
			"name":        "fak_memory_run",
			"description": "RUN a memory query against a backend: pick a built-in {driver} or supply an inline {query}; parameterize with {intent,k,budget}; point at a recall core image with {image_dir} (default: an in-memory demo corpus). Effects default to PROPOSED — set {apply:true} to enact the safe negative-only/storage mutations (tombstone, prune). Sealed spans are never rendered (the trust gate); consolidate/reclassify never persist this rung. Returns the per-step trace, the rendered set, proposed/applied effects, refusals, and stats.",
			"inputSchema": memoryInputSchema,
		},
	}
}

// memoryInputSchema is the {driver|query, intent, k, budget, image_dir, apply} shape
// shared by fak_memory_explain and fak_memory_run.
var memoryInputSchema = json.RawMessage(`{
  "type": "object",
  "properties": {
    "driver": {"type": "string", "description": "a built-in strategy name (see fak_memory_drivers); omit if you supply an inline query"},
    "query": {"type": "object", "description": "an inline authored memq Query: {intent, ops:[{kind, pred?, by?, desc?, k?, bytes?, reason?}]}. Ops: scan|filter|rank|limit|budget|render|tombstone|consolidate|reclassify|prune"},
    "intent": {"type": "string", "description": "the task intent (drives relevance ranking and default match terms)"},
    "k": {"type": "integer", "description": "limit (driver-specific; 0 = driver default)"},
    "budget": {"type": "integer", "description": "byte budget for the rendered/selected set (0 = unbounded)"},
    "image_dir": {"type": "string", "description": "run (not explain): path to a recall core image; omit for the in-memory demo corpus"},
    "apply": {"type": "boolean", "description": "run only: APPLY the safe negative-only/storage mutations (tombstone, prune). Default false = propose only (fail-closed)"}
  }
}`)
