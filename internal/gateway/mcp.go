package gateway

import (
	"bufio"
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"path"
	"reflect"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/archcheck"
	"github.com/anthony-chaudhary/fak/internal/toolplugin"
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
	if !requireMethod(w, r, http.MethodPost) {
		return
	}
	body, err := io.ReadAll(http.MaxBytesReader(w, r.Body, maxBody))
	if err != nil {
		writeErr(w, http.StatusBadRequest, "request too large or unreadable")
		return
	}
	// Lower the request's isolation principal (the auth proxy's X-Fak-Principal header) onto the
	// context so read-self tool arms (fak_context_restore / fak_context_spans) can apply the C1
	// scope floor. Absent a proxy (the no-RequireKey loopback) the principal is "" — the
	// single-tenant default the floor reads as a self-read.
	ctx := WithPrincipal(r.Context(), principalFor(r, ""))
	if r.URL.Query().Get("strict") == "true" || r.Header.Get("X-Fak-Strict") == "true" {
		ctx = withMCPStrict(ctx, true)
	}
	resp := s.dispatchRPC(ctx, body)
	if resp == nil { // a notification
		w.WriteHeader(http.StatusAccepted)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	enc := json.NewEncoder(w)
	enc.SetEscapeHTML(false)
	_ = enc.Encode(resp)
}

type mcpStrictReqKey struct{}

func withMCPStrict(ctx context.Context, strict bool) context.Context {
	return context.WithValue(ctx, mcpStrictReqKey{}, strict)
}

func isMCPStrictRequested(ctx context.Context) bool {
	if ctx == nil {
		return false
	}
	v, _ := ctx.Value(mcpStrictReqKey{}).(bool)
	return v
}

// dispatchRPC parses one JSON-RPC frame and routes it. It returns nil for a
// notification (no id) so the caller writes no response.
func (s *Server) dispatchRPC(ctx context.Context, raw []byte) *rpcResponse {
	var req rpcRequest
	if err := json.Unmarshal(raw, &req); err != nil {
		return &rpcResponse{JSONRPC: "2.0", Error: &rpcError{Code: rpcParseError, Message: "parse error"}}
	}
	if len(bytes.TrimSpace(req.ID)) == 0 {
		// Notification: accept, no response. Lifecycle notifications feed the
		// tool process table (seam 3): cancelled -> kill + revocation arm,
		// progress -> pulse. Everything else stays a silent accept.
		s.mcpToolprocNotify(req.Method, req.Params)
		return nil
	}
	if req.JSONRPC != "2.0" {
		return &rpcResponse{JSONRPC: "2.0", ID: req.ID,
			Error: &rpcError{Code: rpcInvalidRequest, Message: `jsonrpc must be "2.0"`}}
	}
	// Carry the request id so a tools/call arm can correlate a later
	// notifications/cancelled back to the kernel call it spawned (seam 3).
	ctx = mcpWithRequestID(ctx, req.ID)
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
		return s.handleMCPToolsList(ctx, params)
	case "tools/call":
		return s.callTool(ctx, params)
	case "resources/list":
		return mcpCacheHint(map[string]any{"resources": s.resourceDescriptors()}, mcpCatalogTTLMillis, mcpCacheScopePublic), nil
	case "resources/templates/list":
		return mcpCacheHint(map[string]any{"resourceTemplates": s.resourceTemplateDescriptors()}, mcpCatalogTTLMillis, mcpCacheScopePublic), nil
	case "resources/read":
		return s.readResource(params)
	case "prompts/list":
		return mcpCacheHint(map[string]any{"prompts": promptDescriptors()}, mcpCatalogTTLMillis, mcpCacheScopePublic), nil
	case "prompts/get":
		return s.getPrompt(params)
	default:
		return nil, &rpcError{Code: rpcMethodNotFound, Message: "method not found: " + method}
	}
}

type mcpToolsListParams struct {
	Cursor string `json:"cursor,omitempty"`
	Strict *bool  `json:"strict,omitempty"`
}

func (s *Server) handleMCPToolsList(ctx context.Context, params json.RawMessage) (any, *rpcError) {
	strict := envEnabled("FAK_STRICT_MCP_TOOLS") || isMCPStrictRequested(ctx)
	if len(params) > 0 {
		var p mcpToolsListParams
		if err := json.Unmarshal(params, &p); err == nil && p.Strict != nil {
			strict = *p.Strict
		}
	}
	tools, filter := s.toolsListView()
	if strict {
		tools = ToStrictToolDescriptors(tools)
	}
	s.metrics.observeToolFilter(filter)
	return mcpCacheHint(map[string]any{
		"tools": tools,
		"_meta": map[string]any{"fak/tool_filter": filter},
	}, mcpCatalogTTLMillis, mcpCacheScopePublic), nil
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
		// Advertise all three MCP primitives this server implements so a spec-
		// compliant client knows it may call resources/* and prompts/* (#213),
		// not just tools/* — an unadvertised capability is one a client won't probe.
		"capabilities": map[string]any{
			"tools":     map[string]any{},
			"resources": map[string]any{},
			"prompts":   map[string]any{},
		},
		"serverInfo": map[string]any{"name": "fak-gateway", "version": s.version},
	}
}

// callTool handles tools/call. The MCP `arguments` object IS a SyscallRequest
// ({tool, arguments, read_only}). A deny returns a normal tool result (deny-as-
// value); only a protocol/build fault is a JSON-RPC error.
// mcpUnmarshalParams decodes a JSON-RPC method's params into the given pointer,
// returning the uniform "invalid <method> params: ..." rpcInvalidParams fault on a
// malformed body. Shared by tools/call, resources/read, and prompts/get.
func mcpUnmarshalParams(params json.RawMessage, into any, method string) *rpcError {
	if err := json.Unmarshal(params, into); err != nil {
		return &rpcError{Code: rpcInvalidParams, Message: "invalid " + method + " params: " + err.Error()}
	}
	return nil
}

// mcpDecodeCall is the shared body of the tools/call arms that decode the
// arguments into a typed request, run a server handler, and wrap the result as an
// MCP tool result — a malformed body is rpcInvalidParams ("invalid <tool>
// arguments: ..."), a handler error is rpcInvalidParams with the handler's own
// message, and success is mcpToolResult(resp).
func mcpDecodeCall[Req any](raw json.RawMessage, tool string, fn func(Req) (any, error)) (any, *rpcError) {
	var req Req
	if err := json.Unmarshal(raw, &req); err != nil {
		return nil, &rpcError{Code: rpcInvalidParams, Message: "invalid " + tool + " arguments: " + err.Error()}
	}
	resp, err := fn(req)
	if err != nil {
		return nil, &rpcError{Code: rpcInvalidParams, Message: err.Error()}
	}
	return mcpToolResult(resp), nil
}

func (s *Server) callTool(ctx context.Context, params json.RawMessage) (any, *rpcError) {
	var p struct {
		Name      string          `json:"name"`
		Arguments json.RawMessage `json:"arguments"`
	}
	if e := mcpUnmarshalParams(params, &p, "tools/call"); e != nil {
		return nil, e
	}
	// The --expose allowlist is authoritative for INVOCATION too, not just
	// discovery: a tool the operator hid is answered "unknown tool" — the SAME
	// message the default arm gives a name that does not exist — so a client
	// cannot distinguish a hidden tool from a non-existent one (no existence leak).
	if s.exposeAllow != nil && !s.exposeAllow(p.Name) {
		return nil, &rpcError{Code: rpcInvalidParams, Message: "unknown tool: " + p.Name}
	}
	// Count every admitted fak-verb call (the #3093 unused-substrate signal). Placed after
	// the --expose gate so a hidden tool answered "unknown" does not count as a real use.
	s.metrics.observeFakVerbCall()
	switch p.Name {
	case "fak_syscall":
		req := decodeSyscallArgs(p.Arguments)
		if (req.Tool == "view_image" || req.Tool == "functions.view_image") && !s.supportsVision() {
			return nil, &rpcError{Code: rpcInvalidParams, Message: "view_image is not allowed because active model does not support image inputs"}
		}
		req.TraceID = s.traceFor(req.TraceID)
		ctx = WithPrincipal(ctx, req.Principal)
		// The brokered call is a tool process (seam 3): spawn/exit rows in the
		// same journal the guard hooks feed, keyed by the kernel trace id the
		// seam-2 revocation gate also keys on.
		tpID := mcpToolprocSpawn(ctx, req.TraceID, req.Tool)
		var trace []toolplugin.TraceEvent
		var pref *toolplugin.ResolvedPreference
		var wv WireVerdict
		var env *ResultEnvelope
		var err error
		if len(s.toolPlugins) == 0 && reflect.DeepEqual(s.toolPreferences, toolplugin.PreferenceLayers{}) && reflect.DeepEqual(req.Preferences, toolplugin.Preference{}) {
			// Exact legacy path: no allocations, no response fields, no behavior delta.
			wv, env, err = s.syscall(ctx, req.Tool, rawArgs(req.Arguments), req.ReadOnly, req.Witness, req.TraceID)
		} else {
			wv, env, trace, pref, err = s.syscallWithPlugins(ctx, req.Tool, rawArgs(req.Arguments), req.ReadOnly, req.Witness, req.TraceID, req.Preferences)
		}
		mcpToolprocExit(tpID, err)
		if err != nil {
			return nil, &rpcError{Code: rpcInvalidParams, Message: err.Error()}
		}
		return mcpToolResult(SyscallResponse{Verdict: wv, Result: env, TraceID: req.TraceID, PluginTrace: trace, EffectivePreferences: pref}), nil
	case "fak_read":
		// The vToolcall serve seam (#795): a real, kernel-mediated file read the model can
		// call INSTEAD of the harness's built-in Read. Routing through k.Syscall means the
		// vDSO fast path serves a FRESH cached read with no disk I/O (the #795 per-path
		// invalidator proves freshness), and only a genuine miss reaches the confined
		// readEngine. No Claude Code change is needed — the model opts in via the MCP tool.
		var rr struct {
			FilePath    string   `json:"file_path"`
			FilePaths   []string `json:"file_paths"`
			Path        string   `json:"path"`
			Offset      int      `json:"offset"`
			Limit       int      `json:"limit"`
			LineNumbers bool     `json:"line_numbers"`
			TraceID     string   `json:"trace_id"`
			Witness     string   `json:"witness"`
		}
		_ = json.Unmarshal(p.Arguments, &rr)
		if rr.FilePaths != nil {
			if len(rr.FilePaths) == 0 {
				return nil, &rpcError{Code: rpcInvalidParams, Message: "fak_read requires a non-empty file_paths array"}
			}
			traceID := s.traceFor(rr.TraceID)
			return mcpToolResult(s.fakReadBatch(ctx, rr.FilePaths, traceID, rr.Witness)), nil
		}
		path := rr.FilePath
		if path == "" {
			path = rr.Path
		}
		wv, env, err := s.fakReadWithOptions(ctx, path, rr.Offset, rr.Limit, rr.LineNumbers, s.traceFor(rr.TraceID), rr.Witness)
		if err != nil {
			return nil, &rpcError{Code: rpcInvalidParams, Message: err.Error()}
		}
		return mcpToolResult(SyscallResponse{Verdict: wv, Result: env, TraceID: s.traceFor(rr.TraceID)}), nil
	case "fak_adjudicate":
		req := decodeSyscallArgs(p.Arguments)
		req.TraceID = s.traceFor(req.TraceID)
		started := time.Now()
		wv, repaired, err := s.adjudicate(ctx, req.Tool, rawArgs(req.Arguments), req.ReadOnly, req.Witness, req.TraceID)
		receipt := adjudicateReceipt(wv, err, time.Since(started))
		if err != nil {
			return nil, &rpcError{Code: rpcInvalidParams, Message: "fak_adjudicate failed", Data: receipt}
		}
		resp := SyscallResponse{Verdict: wv, TraceID: req.TraceID, Receipt: &receipt}
		if repaired != "" {
			resp.RepairedArguments = json.RawMessage(repaired)
		}
		return mcpToolResult(resp), nil
	case "fak_admit":
		var req struct {
			AdmitRequest
			Items []AdmitRequest `json:"items"`
		}
		_ = json.Unmarshal(p.Arguments, &req)
		if req.Items != nil {
			if len(req.Items) == 0 {
				return nil, &rpcError{Code: rpcInvalidParams, Message: "fak_admit requires a non-empty items array"}
			}
			traceID := s.traceFor(req.TraceID)
			return mcpToolResult(s.fakAdmitBatch(ctx, req.Items, traceID, req.Witness)), nil
		}
		req.TraceID = s.traceFor(req.TraceID)
		wv, env, err := s.admit(ctx, req.Tool, rawArgs(req.Result), req.Witness, req.TraceID)
		if err != nil {
			return nil, &rpcError{Code: rpcInvalidParams, Message: err.Error()}
		}
		return mcpToolResult(SyscallResponse{Verdict: wv, Result: env, TraceID: req.TraceID}), nil
	case "fak_changes":
		var req ChangesRequest
		_ = json.Unmarshal(p.Arguments, &req)
		events, cursor := s.changes(req.Principal, req.Since)
		return mcpToolResult(ChangesResponse{Events: events, Cursor: cursor}), nil
	case "fak_revoke":
		var req RevokeRequest
		if err := json.Unmarshal(p.Arguments, &req); err != nil || req.Witness == "" {
			return nil, &rpcError{Code: rpcInvalidParams, Message: "fak_revoke requires a non-empty witness"}
		}
		evicted, te := s.revoke(req.Witness)
		return mcpToolResult(RevokeResponse{Witness: req.Witness, Evicted: evicted, TrustEpoch: te}), nil
	case "fak_session_reset":
		return mcpDecodeCall[SessionResetRequest](p.Arguments, "fak_session_reset", func(req SessionResetRequest) (any, error) {
			return s.sessionReset(ctx, req)
		})
	case "fak_context_change":
		return mcpDecodeCall[ContextChangeRequest](p.Arguments, "fak_context_change", func(req ContextChangeRequest) (any, error) {
			return s.contextChange(ctx, req)
		})
	case "fak_memory_drivers":
		return mcpToolResult(map[string]any{"drivers": s.memoryDrivers()}), nil
	case "fak_memory_explain":
		return mcpDecodeCall[MemoryRequest](p.Arguments, "fak_memory_explain", func(req MemoryRequest) (any, error) {
			return s.memoryExplain(req)
		})
	case "fak_memory_run":
		return mcpDecodeCall[MemoryRequest](p.Arguments, "fak_memory_run", func(req MemoryRequest) (any, error) {
			return s.memoryRun(ctx, req)
		})
	case "fak_tools_search":
		return mcpDecodeCall[ToolsSearchRequest](p.Arguments, "fak_tools_search", func(req ToolsSearchRequest) (any, error) {
			return s.toolsSearch(req)
		})
	case "fak_trajquery":
		return mcpDecodeCall[TrajQueryRequest](p.Arguments, "fak_trajquery", func(req TrajQueryRequest) (any, error) {
			return s.trajQuery(req)
		})
	case "fak_feature_query":
		return mcpDecodeCall[FeatureQueryRequest](p.Arguments, "fak_feature_query", func(req FeatureQueryRequest) (any, error) {
			return s.featureQuery(req)
		})
	case "fak_capabilities":
		return mcpDecodeCall[CapabilitiesRequest](p.Arguments, "fak_capabilities", func(req CapabilitiesRequest) (any, error) {
			return s.capabilities(req)
		})
	case "fak_context_value":
		return mcpDecodeCall[ContextValueRequest](p.Arguments, "fak_context_value", func(req ContextValueRequest) (any, error) {
			return s.CtxValueReportFor(s.traceFor(req.TraceID)), nil
		})
	case "fak_context_restore":
		return mcpDecodeCall[ContextRestoreRequest](p.Arguments, "fak_context_restore", func(req ContextRestoreRequest) (any, error) {
			return s.restoreContext(principalFromContext(ctx), req)
		})
	case "fak_context_spans":
		return mcpDecodeCall[ContextSpansRequest](p.Arguments, "fak_context_spans", func(req ContextSpansRequest) (any, error) {
			return s.contextSpans(principalFromContext(ctx), req)
		})
	case "fak_resume_history":
		return mcpDecodeCall[ResumeHistoryRequest](p.Arguments, "fak_resume_history", func(req ResumeHistoryRequest) (any, error) {
			return s.ResumeHistoryFor(req), nil
		})
	case "view_image", "functions.view_image":
		if !s.supportsVision() {
			return nil, &rpcError{Code: rpcInvalidParams, Message: "view_image is not allowed because active model does not support image inputs"}
		}
		var args struct {
			Path string `json:"path"`
		}
		_ = json.Unmarshal(p.Arguments, &args)
		return mcpToolResult(map[string]any{"status": "ok", "path": args.Path}), nil
	case "fak_arch_check":
		var args struct {
			Package string `json:"package"`
			Mine    bool   `json:"mine"`
		}
		_ = json.Unmarshal(p.Arguments, &args)
		root := "."
		var res *archcheck.CheckResult
		var err error
		if args.Package != "" {
			res, err = archcheck.CheckPackage(root, args.Package)
		} else if args.Mine {
			res, err = archcheck.CheckMine(root)
		} else {
			res, err = archcheck.CheckAll(root)
		}
		if err != nil {
			return nil, &rpcError{Code: rpcInternalError, Message: err.Error()}
		}
		return mcpToolResult(res), nil
	default:
		return nil, &rpcError{Code: rpcInvalidParams, Message: "unknown tool: " + p.Name}
	}
}

func (s *Server) sessionReset(ctx context.Context, req SessionResetRequest) (SessionResetResponse, error) {
	if req.ContextTokens < 0 {
		return SessionResetResponse{}, errors.New("fak_session_reset context_tokens must be non-negative")
	}
	trace := s.traceFor(req.TraceID)
	resp := SessionResetResponse{TraceID: trace, FromTraceID: trace}

	var st SessionState
	switch {
	case req.ContextTokens > 0:
		if s.debitSession == nil {
			resp.Note = "session debit hook unavailable; cannot apply reported context_tokens"
			return resp, nil
		}
		st = s.debitSession(ctx, trace, SessionUsage{ContextTokens: req.ContextTokens})
	case s.observeSession != nil:
		st = s.observeSession(ctx, trace)
	default:
		st = SessionState{TraceID: trace, Run: "running"}
	}
	if st.TraceID == "" {
		st.TraceID = trace
	}
	resp.Session = st
	resp.Reason = st.Reason

	if s.resetOnBudget == nil {
		resp.Note = "reset_on_budget hook unavailable; start fak with --reset-on-budget to build a carryover seed"
		return resp, nil
	}
	if !isBudgetResetReason(st) {
		resp.Note = "session is not budget-drained; reset refused"
		return resp, nil
	}
	newTrace, seed, ok := s.resetOnBudget(ctx, trace, req.Messages)
	if !ok || newTrace == "" {
		resp.Note = "reset_on_budget hook declined the reset"
		return resp, nil
	}
	resp.Reset = true
	resp.ToTraceID = newTrace
	resp.TraceID = newTrace
	resp.Seed = seed
	resp.Directive = &SessionResetDirective{
		Action:        "restart_fresh_session",
		FromTraceID:   trace,
		ToTraceID:     newTrace,
		Reason:        st.Reason,
		CacheAffinity: st.CacheAffinity,
		Required: []string{
			"dump_session_image",
			"start_fresh_process",
			"rehydrate_planned_view",
			"reuse_provider_cache_when_legal",
		},
		Note: "context budget exhausted; prepend seed_messages in a fresh model window",
	}
	if s.observeSession != nil {
		fresh := s.observeSession(ctx, newTrace)
		if fresh.TraceID == "" {
			fresh.TraceID = newTrace
		}
		resp.Session = fresh
	}
	return resp, nil
}

// decodeSyscallArgs parses the MCP tools/call `arguments` object into a SyscallRequest.
// A malformed object yields the zero request (an empty tool name), which the kernel
// rejects downstream — never a panic.
func decodeSyscallArgs(raw json.RawMessage) SyscallRequest {
	var req SyscallRequest
	_ = json.Unmarshal(raw, &req)
	return req
}

// fakRead runs a kernel-mediated file read for the fak_read MCP tool (#795). It builds a
// read-only Read call (so the vDSO tier path is armed and the #795 per-path tag binds),
// PINS the engine to the confined readEngine (agent.FakReadEngineID) so a cache MISS reads
// the real file regardless of the gateway's configured chat engine, and runs the full
// syscall boundary. On a vDSO hit k.Syscall serves the cached bytes with no engine
// dispatch and no disk read; the per-path invalidator guarantees that hit is fresh.
func (s *Server) fakRead(ctx context.Context, path, traceID, witness string) (WireVerdict, *ResultEnvelope, error) {
	return s.fakReadWithOptions(ctx, path, 0, 0, false, traceID, witness)
}

func (s *Server) fakReadWithOptions(ctx context.Context, path string, offset, limit int, lineNumbers bool, traceID, witness string) (WireVerdict, *ResultEnvelope, error) {
	argsMap := map[string]any{"file_path": path}
	if offset > 0 {
		argsMap["offset"] = offset
	}
	if limit > 0 {
		argsMap["limit"] = limit
	}
	if lineNumbers {
		argsMap["line_numbers"] = true
	}
	args, _ := json.Marshal(argsMap)
	tc, err := s.buildCall(ctx, "fak_read", string(args), true, witness, traceID)
	if err != nil {
		return WireVerdict{}, nil, err
	}
	// Pin the confined read engine: routeEngine left Engine "" (kernel default = the chat
	// engine) or a model id, neither of which can read a file. k.Syscall honors a non-empty
	// c.Engine, so this is what dispatches a miss to readEngine.
	tc.Engine = agent.FakReadEngineID
	started := time.Now()
	r, v := s.k.Syscall(ctx, tc)
	duration := time.Since(started)
	wv := renderVerdict(v, resultMeta(r))
	var env *ResultEnvelope
	if r != nil {
		payload := resolveBytes(ctx, r.Payload)
		payload = unpageFakReadPayload(ctx, payload)
		payload = fakReadReceipt(payload, r, duration)
		env = &ResultEnvelope{
			Status:  statusName(r.Status),
			Content: string(payload),
			Meta:    r.Meta,
		}
	}
	return wv, env, nil
}

// unpageFakReadPayload resolves a paged-out MMU pointer back to the original payload bytes.
// If ctxmmu paged out an oversize read result to a pointer stub ({"_paged":true,"ref":...}),
// unpageFakReadPayload faults the original file-read JSON back in from CAS so the MCP client
// receives the intended file content rather than an empty stub.
func unpageFakReadPayload(ctx context.Context, payload []byte) []byte {
	if len(payload) == 0 || !bytes.Contains(payload, []byte(`"_paged"`)) {
		return payload
	}
	var m map[string]any
	if err := json.Unmarshal(payload, &m); err != nil {
		return payload
	}
	if paged, _ := m["_paged"].(bool); !paged {
		return payload
	}
	ref, _ := m["ref"].(string)
	if ref == "" {
		return payload
	}
	cleanRef := strings.TrimPrefix(ref, "sha256:")
	if b, ok := abi.PageOut("blob"); ok {
		handle := abi.Ref{Kind: abi.RefBlob, Digest: cleanRef}
		if res, err := b.PageIn(ctx, handle); err == nil && len(res.Inline) > 0 {
			return res.Inline
		}
	}
	if res := abi.ActiveResolver(); res != nil {
		handle := abi.Ref{Kind: abi.RefBlob, Digest: cleanRef}
		if raw, err := res.Resolve(ctx, handle); err == nil && len(raw) > 0 {
			return raw
		}
	}
	return payload
}

// fakReadReceipt adds the additive fak_read/1 receipt to the legacy JSON payload. Existing
// file_path/content/error fields are preserved. The receipt is rebuilt after every syscall so
// a cached payload can never falsely retain the cold-read outcome or its latency.
func fakReadReceipt(payload []byte, r *abi.Result, duration time.Duration) []byte {
	var body map[string]any
	if json.Unmarshal(payload, &body) != nil {
		return payload
	}
	outcome := "executed_cold_read"
	witness := "filesystem_read"
	if r != nil && r.Meta != nil && r.Meta["served_by"] == "vdso" {
		outcome = "verified_fresh_reuse"
		witness = "vdso"
	}
	byteCount := 0
	if encoded, ok := body["content_base64"].(string); ok {
		if decoded, err := base64.StdEncoding.DecodeString(encoded); err == nil {
			byteCount = len(decoded)
		}
	} else if content, ok := body["content"].(string); ok {
		byteCount = len([]byte(content))
	}
	receipt := map[string]any{
		"schema":             "fak-read-receipt/1",
		"outcome":            outcome,
		"bytes":              byteCount,
		"duration_ns":        duration.Nanoseconds(),
		"freshness_verified": true,
		"witness":            witness,
	}
	if code, ok := body["error_code"].(string); ok {
		receipt["freshness_verified"] = false
		receipt["witness"] = "filesystem_read_attempt"
		receipt["error"] = map[string]any{"code": code, "source": body["error_source"]}
	}
	body["receipt"] = receipt
	out, err := json.Marshal(body)
	if err != nil {
		return payload
	}
	return out
}

// fakReadBatch is the additive batch form of fak_read. Each path crosses fakRead on its
// own, so every item gets an independent buildCall + k.Syscall adjudication and its own
// cache hit/miss decision. Results retain request order; an item-level fault is captured
// on that row instead of aborting the remaining independent reads.
func (s *Server) fakReadBatch(ctx context.Context, paths []string, traceID, witness string) FakReadBatchResponse {
	resp := FakReadBatchResponse{
		TraceID:   traceID,
		ItemCount: len(paths),
		Results:   make([]FakReadBatchItem, 0, len(paths)),
	}
	for _, path := range paths {
		wv, env, err := s.fakRead(ctx, path, traceID, witness)
		item := FakReadBatchItem{FilePath: path, Verdict: wv, Result: env}
		if err != nil {
			item.Error = err.Error()
		} else if env != nil && env.Status == "ERROR" {
			item.Error = resultEnvelopeError(env)
		}
		resp.Results = append(resp.Results, item)
	}
	return resp
}

// fakAdmitBatch gives the other per-item MCP verb the same additive batch axis. Top-level
// trace/witness values are defaults; an item may override either. As with fak_read, one
// admission fault is data on that row and never silently drops the remaining items.
func (s *Server) fakAdmitBatch(ctx context.Context, items []AdmitRequest, traceID, witness string) FakAdmitBatchResponse {
	resp := FakAdmitBatchResponse{
		TraceID:   traceID,
		ItemCount: len(items),
		Results:   make([]FakAdmitBatchItem, 0, len(items)),
	}
	for _, req := range items {
		if req.TraceID == "" {
			req.TraceID = traceID
		} else {
			req.TraceID = s.traceFor(req.TraceID)
		}
		if req.Witness == "" {
			req.Witness = witness
		}
		wv, env, err := s.admit(ctx, req.Tool, rawArgs(req.Result), req.Witness, req.TraceID)
		item := FakAdmitBatchItem{Tool: req.Tool, TraceID: req.TraceID, Verdict: wv, Result: env}
		if err != nil {
			item.Error = err.Error()
		} else if env != nil && env.Status == "ERROR" {
			item.Error = resultEnvelopeError(env)
		}
		resp.Results = append(resp.Results, item)
	}
	return resp
}

// resultEnvelopeError lifts the read/admit engine's stable {"error":"..."} payload onto
// the batch row. The full envelope remains present for callers that already understand it.
func resultEnvelopeError(env *ResultEnvelope) string {
	if env == nil {
		return ""
	}
	var payload struct {
		Error string `json:"error"`
	}
	if json.Unmarshal([]byte(env.Content), &payload) == nil && payload.Error != "" {
		return payload.Error
	}
	return env.Content
}

// FakReadBatchResponse is the MCP wire shape for {file_paths:[...]}. ItemCount makes
// fusion visible to width observers: one MCP call doing five reads is not mistaken for
// one call doing one read.
type FakReadBatchResponse struct {
	Results   []FakReadBatchItem `json:"results"`
	TraceID   string             `json:"trace_id,omitempty"`
	ItemCount int                `json:"item_count"`
}

type FakReadBatchItem struct {
	FilePath string          `json:"file_path"`
	Verdict  WireVerdict     `json:"verdict"`
	Result   *ResultEnvelope `json:"result,omitempty"`
	Error    string          `json:"error,omitempty"`
}

// FakAdmitBatchResponse is the MCP wire shape for {items:[{tool,result},...]}; rows are
// deliberately independent because each result must cross the result-side safety floor.
type FakAdmitBatchResponse struct {
	Results   []FakAdmitBatchItem `json:"results"`
	TraceID   string              `json:"trace_id,omitempty"`
	ItemCount int                 `json:"item_count"`
}

type FakAdmitBatchItem struct {
	Tool    string          `json:"tool"`
	TraceID string          `json:"trace_id,omitempty"`
	Verdict WireVerdict     `json:"verdict"`
	Result  *ResultEnvelope `json:"result,omitempty"`
	Error   string          `json:"error,omitempty"`
}

// mcpToolResult wraps a SyscallResponse as an MCP tool result: a single text
// content block carrying the JSON. isError stays false — a deny is a successful
// adjudication, surfaced in the verdict, not a tool failure.
func mcpToolResult(v any) map[string]any {
	return mcpToolResultSpan(v, toonCacheResident(v))
}

// mcpToolResultSpan is mcpToolResult with the span's cache-residency made explicit —
// the single wrap point every tool result leaves through. Behind FAK_TOON_WIRE
// (default off) the text block may carry the payload TOON-encoded instead of JSON,
// but ONLY when every toon.Decide gate proves a net win (#3067); on any skip the
// text is the canonical JSON, byte-identical to what shipped before the wire existed.
func mcpToolResultSpan(v any, cacheResident bool) map[string]any {
	b, _ := json.Marshal(v)
	text := string(b)
	if enc, ok := toonWireText(b, cacheResident); ok {
		text = enc
	}
	return map[string]any{
		"content": []map[string]any{{"type": "text", "text": text}},
		"isError": false,
	}
}

// mcpToolAnnotations defines standard MCP tool annotations (e.g. MCP 2024-11-05+).
type mcpToolAnnotations struct {
	ReadOnly            *bool `json:"readOnly,omitempty"`
	Idempotent          *bool `json:"idempotent,omitempty"`
	Consequential       *bool `json:"consequential,omitempty"`
	ReadOnlyHint        bool  `json:"readOnlyHint,omitempty"`
	ReadOnlyHintSnake   bool  `json:"read_only_hint,omitempty"`
	IdempotentHint      bool  `json:"idempotentHint,omitempty"`
	IdempotentHintSnake bool  `json:"idempotent_hint,omitempty"`
}

func (a *mcpToolAnnotations) toMap() map[string]any {
	if a == nil {
		return nil
	}
	m := make(map[string]any)
	if a.ReadOnly != nil {
		m["readOnly"] = *a.ReadOnly
	}
	if a.Idempotent != nil {
		m["idempotent"] = *a.Idempotent
	}
	if a.Consequential != nil {
		m["consequential"] = *a.Consequential
	}
	if a.ReadOnlyHint {
		m["readOnlyHint"] = true
	}
	if a.ReadOnlyHintSnake {
		m["read_only_hint"] = true
	}
	if a.IdempotentHint {
		m["idempotentHint"] = true
	}
	if a.IdempotentHintSnake {
		m["idempotent_hint"] = true
	}
	return m
}

func boolPtr(b bool) *bool {
	return &b
}

func readOnlyToolAnnotations() *mcpToolAnnotations {
	return &mcpToolAnnotations{
		ReadOnly:            boolPtr(true),
		Idempotent:          boolPtr(true),
		Consequential:       boolPtr(false),
		ReadOnlyHint:        true,
		ReadOnlyHintSnake:   true,
		IdempotentHint:      true,
		IdempotentHintSnake: true,
	}
}

func mutatingToolAnnotations() *mcpToolAnnotations {
	return &mcpToolAnnotations{
		ReadOnly:      boolPtr(false),
		Consequential: boolPtr(true),
	}
}

// mcpToolDescriptor represents an MCP tool descriptor with schema and annotations.
type mcpToolDescriptor struct {
	Name        string              `json:"name"`
	Description string              `json:"description,omitempty"`
	InputSchema json.RawMessage     `json:"inputSchema"`
	Annotations *mcpToolAnnotations `json:"annotations,omitempty"`
	Strict      bool                `json:"strict,omitempty"`
}

func (td mcpToolDescriptor) toMap() map[string]any {
	m := map[string]any{
		"name":        td.Name,
		"description": td.Description,
		"inputSchema": td.InputSchema,
	}
	if td.Annotations != nil {
		m["annotations"] = td.Annotations.toMap()
	}
	if td.Strict {
		m["strict"] = true
	}
	return m
}

// toolDescriptors is the tools/list payload. The inputSchema is a JSON Schema for
// the {tool, arguments, read_only} shape both tools accept.
func toolDescriptors() []map[string]any {
	schema := json.RawMessage(`{"type":"object","properties":{"tool":{"type":"string","description":"the logical tool name to route through the kernel"},"arguments":{"anyOf":[{"type":"object","additionalProperties":false},{"type":"string"}],"description":"the tool arguments: a JSON object, or a JSON-encoded string"},"read_only":{"type":"boolean","description":"hint that the tool is read-only/idempotent (enables vDSO dedup)"},"trace_id":{"type":"string","description":"optional session trace id; omitted means the gateway mints one and returns it"},"witness":{"type":"string","description":"optional external world-state token the call is reading at"}},"required":["tool"],"additionalProperties":false}`)
	tools := []map[string]any{
		mcpToolDescriptor{
			Name:        "fak_adjudicate",
			Description: "Adjudicate a proposed tool call through the fak kernel WITHOUT executing it. Returns verdict and a fak-adjudicate-receipt/1 receipt with outcome, duration, execution=not_executed, and kernel_decide provenance. Repaired arguments appear only for TRANSFORM. Use when client executes tools out-of-band.",
			InputSchema: schema,
			Annotations: readOnlyToolAnnotations(),
		}.toMap(),
		mcpToolDescriptor{
			Name:        "fak_syscall",
			Description: "Adjudicate AND execute a tool call through the fak kernel (dispatch to the registered engine + context-MMU result admission). Returns the verdict and the admitted result. Use when fak should run the tool.",
			InputSchema: schema,
			Annotations: mutatingToolAnnotations(),
		}.toMap(),
		mcpToolDescriptor{
			Name:        "fak_read",
			Description: "Read files with verified-fresh cache reuse or cold read. Preserves file_path/content/error format; adds receipt {schema,outcome,bytes,duration_ns,witness,error?}. Outcome is executed_cold_read or verified_fresh_reuse; typed errors expose code/source. Prefer {file_paths:[...]} for multiple files.",
			InputSchema: json.RawMessage(`{"type":"object","properties":{"file_path":{"type":"string","description":"the path of the file to read (absolute, or relative to the working tree)"},"file_paths":{"type":"array","minItems":1,"items":{"type":"string","minLength":1},"description":"independent file paths to read in one call; preferred when reading more than one file"},"offset":{"type":"integer","description":"optional 1-based line number to start reading from"},"limit":{"type":"integer","description":"optional maximum number of lines to read"},"line_numbers":{"type":"boolean","description":"optional flag to prefix lines with 1-based line numbers (<line>: <content>)"},"trace_id":{"type":"string","description":"optional session trace id; omitted means the gateway mints one and returns it"},"witness":{"type":"string","description":"optional external world-state token (a git commit / blob hash) the read is taken at"}},"additionalProperties":false}`),
			Annotations: readOnlyToolAnnotations(),
		}.toMap(),
		mcpToolDescriptor{
			Name:        "fak_admit",
			Description: "Submit tool RESULTS your own client executed, to run them through the fak kernel's result-side stack (context-MMU quarantine + IFC source-stamp / per-trace taint ledger) BEFORE admitting them to context. Wire-proxied sessions are gated automatically; fak_admit is for out-of-band client tool results. A poisoned/secret-shaped result comes back QUARANTINE with the bytes paged out; the session's taint high-water mark is raised so a later egress is gated. Prefer {items:[{tool,result},...]} for independent results; {tool,result,trace_id} remains the unchanged single-result form. Every batch item crosses the safety floor independently and returns in request order, so one refusal never drops its peers. This arms the exfil floor on the path where YOU run the tool (the complement of fak_adjudicate).",
			InputSchema: json.RawMessage(`{
  "type": "object",
  "properties": {
    "tool": {"type": "string", "description": "the tool name that produced this result (its source class keys the provenance taint)"},
    "result": {
      "anyOf": [{"type": "object", "additionalProperties": false}, {"type": "string"}],
      "description": "the tool result content: a JSON object, or a JSON-encoded string"
    },
	"items": {"type": "array", "minItems": 1, "description": "independent tool results to admit in one call", "items": {"type": "object", "properties": {"tool": {"type": "string"}, "result": {"anyOf": [{"type": "object", "additionalProperties": false}, {"type": "string"}], "description": "the tool result content: a JSON object, or a JSON-encoded string"}, "trace_id": {"type": "string"}, "witness": {"type": "string"}}, "required": ["tool"], "additionalProperties": false}},
    "trace_id": {"type": "string", "description": "the session trace this result belongs to (keys the IFC taint ledger)"},
    "witness": {"type": "string", "description": "optional external world-state token the result was read at"}
  },
  "additionalProperties": false
}`),
			Annotations: mutatingToolAnnotations(),
		}.toMap(),
		mcpToolDescriptor{
			Name:        "fak_changes",
			Description: "Drain the cross-agent 'what changed' feed: the typed write Mutations and Revocations observed since your cursor, so you can re-plan or evict your own cache when another agent changed or refuted shared data. Pass {since: <cursor>} (0 = everything retained); returns the events and your next cursor.",
			InputSchema: json.RawMessage(`{"type":"object","properties":{"since":{"type":"integer","description":"the Seq cursor of the last event you saw (0 = from the start of the retained window)"}},"additionalProperties":false}`),
			Annotations: readOnlyToolAnnotations(),
		}.toMap(),
		mcpToolDescriptor{
			Name:        "fak_revoke",
			Description: "Refute an external world-state witness (a git commit / blob hash / lease epoch) found poisoned or stale: every pooled tier-2 entry admitted under it is causally evicted fleet-wide, future re-admission under it is refused, and the eviction is broadcast on the change feed. Pass {witness: <token>}; returns the local eviction count and the new trust epoch.",
			InputSchema: json.RawMessage(`{"type":"object","properties":{"witness":{"type":"string","description":"the external world-state witness to refute"}},"required":["witness"],"additionalProperties":false}`),
			Annotations: mutatingToolAnnotations(),
		}.toMap(),
		mcpToolDescriptor{
			Name:        "fak_session_reset",
			Description: "Cooperatively reset a budget-drained served session from an MCP client. Pass {trace_id?, context_tokens?, messages?}; context_tokens is first debited against the session budget, then fak reuses the same --reset-on-budget carryover builder to mint a fresh continuation trace and seed_messages for a new model window. Returns reset=false when the session is not budget-drained or the host did not wire --reset-on-budget.",
			InputSchema: json.RawMessage(`{
  "type": "object",
  "properties": {
    "trace_id": {"type": "string", "description": "session trace id; omitted uses the gateway default trace when configured"},
    "context_tokens": {"type": "integer", "description": "optional provider/model context-token count to debit before checking the reset boundary"},
    "messages": {"type": "array", "items": {"type": "object", "additionalProperties": false}, "description": "optional transcript messages to distill into the fresh-window carryover seed"}
  },
  "additionalProperties": false
}`),
			Annotations: mutatingToolAnnotations(),
		}.toMap(),
		mcpToolDescriptor{
			Name:        "fak_context_change",
			Description: "Request a safe negative-only context mutation against a persisted recall core image. Today this records a tombstone for one page: future Resolve/Recall/working-set assembly skips it, while the original page row and CAS bytes remain available for audit. Pass {image_dir, step, reason, requested_by?, digest?, witness?, action?}; action may be omitted, 'tombstone', or 'tombstone_page'.",
			InputSchema: json.RawMessage(`{
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
  "required": ["image_dir", "step", "reason"],
  "additionalProperties": false
}`),
			Annotations: mutatingToolAnnotations(),
		}.toMap(),
		mcpToolDescriptor{
			Name:        "fak_memory_drivers",
			Description: "List the built-in memory STRATEGIES (recall/render/clean/compact/dream). Each is a composable query in the memq algebra (scan|filter|rank|limit|budget|render|tombstone|consolidate|reclassify|prune), not a hardcoded function — 'build SQL, not a specific query'. Returns each driver's name, doc, and compiled plan so you can see the pipeline and author your own.",
			InputSchema: json.RawMessage(`{"type":"object","properties":{},"additionalProperties":false}`),
			Annotations: readOnlyToolAnnotations(),
		}.toMap(),
		mcpToolDescriptor{
			Name:        "fak_memory_explain",
			Description: "EXPLAIN a memory query as a plan WITHOUT executing it — every step, which steps are effects, and which mutate durable state (and so are proposal-only). Pass {driver} for a built-in, or {query} with an inline authored memq Query ({intent, ops:[{kind,...}]}). This is the 'step through it before you run it' surface.",
			InputSchema: memoryInputSchema,
			Annotations: readOnlyToolAnnotations(),
		}.toMap(),
		mcpToolDescriptor{
			Name:        "fak_memory_run",
			Description: "RUN a memory query against a backend: pick a built-in {driver} or supply an inline {query}; parameterize with {intent,k,budget}; point at a recall core image with {image_dir} (default: an in-memory demo corpus). Effects default to PROPOSED — set {apply:true} to enact the safe negative-only/storage mutations (tombstone, prune). Sealed spans are never rendered (the trust gate); consolidate/reclassify never persist this rung. Returns the per-step trace, the rendered set, proposed/applied effects, refusals, and stats.",
			InputSchema: memoryInputSchema,
			Annotations: mutatingToolAnnotations(),
		}.toMap(),
		mcpToolDescriptor{
			Name:        "fak_tools_search",
			Description: "Search and retrieve tool schemas with progressive disclosure. Filter tools by query; detail_level selects 'name', 'description', or 'full' schemas.",
			InputSchema: json.RawMessage(`{"type":"object","properties":{"query":{"type":"string","description":"optional filter string; matches tools whose name or description contains this substring (case-insensitive)"},"detail_level":{"type":"string","enum":["name","description","full"],"description":"level of detail to return: 'name' = just tool names, 'description' = names + descriptions, 'full' = complete schemas including inputSchema"}},"additionalProperties":false}`),
			Annotations: readOnlyToolAnnotations(),
		}.toMap(),
		mcpToolDescriptor{
			Name:        "fak_feature_query",
			Description: "Query fak's unified self-feature catalog: dev facts, live MCP tools, memory drivers, and capability cards. Returns lightweight FeatureCards with guarded request shapes; pass detail to fault only one selected schema, doc snippet, or memory explain plan.",
			InputSchema: featureQueryInputSchema,
			Annotations: readOnlyToolAnnotations(),
		}.toMap(),
		mcpToolDescriptor{
			Name:        "fak_capabilities",
			Description: "The task-scoped toolbelt: memory drivers (memq recall/render/clean/compact/dream), the fak index * self-index verbs, and the kernel shared-path verbs (fak_changes, dos_arbitrate), ranked by an optional intent, each with the exact call to make (a memory-driver card carries a ready fak_memory_run call). Narrower and memory-forward compared to fak_feature_query.",
			InputSchema: capabilitiesInputSchema,
			Annotations: readOnlyToolAnnotations(),
		}.toMap(),
		mcpToolDescriptor{
			Name:        "view_image",
			Description: "Inspect an image file or visual artifact when the active model supports multimodal vision inputs. Omitted from tool declarations when model vision is unsupported.",
			InputSchema: json.RawMessage(`{"type":"object","properties":{"path":{"type":"string","description":"path to the image file to inspect"}},"required":["path"],"additionalProperties":false}`),
			Annotations: readOnlyToolAnnotations(),
		}.toMap(),
		mcpToolDescriptor{
			Name:        "fak_arch_check",
			Description: "Preflight architectural validity: check whether Go package imports violate layered DAG tiers or primitive leaf constraints in <50ms.",
			InputSchema: json.RawMessage(`{"type":"object","properties":{"package":{"type":"string","description":"repo-relative package path, e.g. internal/agentquery"},"mine":{"type":"boolean","description":"check only packages touched by uncommitted or staged changes"}},"additionalProperties":false}`),
			Annotations: readOnlyToolAnnotations(),
		}.toMap(),
	}
	// The scoped trajectory-query surface (#3550): trajquery over MCP with the same
	// validate -> rewrite -> execute scope enforcement the CLI runs.
	tj := trajqueryToolDescriptor()
	tj["annotations"] = readOnlyToolAnnotations().toMap()
	tools = append(tools, tj)
	return append(tools, contextIntrospectionToolDescriptors()...)
}

// contextIntrospectionToolDescriptors is the tools/list tail covering the
// managed-context introspection surface (value report, compacted-task restore,
// restorable-span discovery). Split out of toolDescriptors as a cohesive concern.
func contextIntrospectionToolDescriptors() []map[string]any {
	return []map[string]any{
		mcpToolDescriptor{
			Name:        "fak_context_value",
			Description: "Inspect managed-context pressure for a session. Returns observed token headroom and growth, a turn/compaction forecast, lifecycle volumes, and step_advice (any | bounded | checkpoint | rebuild | unknown). Read-only; optional trace_id defaults to the current guarded session.",
			InputSchema: json.RawMessage(`{
  "type": "object",
  "properties": {
    "trace_id": {"type": "string", "description": "session trace id; omitted uses the gateway default trace when configured (your own session under fak guard)"}
  },
  "additionalProperties": false
}`),
			Annotations: readOnlyToolAnnotations(),
		}.toMap(),
		mcpToolDescriptor{
			Name:        "fak_context_restore",
			Description: "Restore dropped context by content-addressed sha256 id. Returns verbatim stashed bytes plus orientation; optional trace_id defaults to the current guarded session, and image_dir may page a recall-image digest. Read-only and trust-gated: sealed or tombstoned spans are refused, while unknown or evicted ids return a miss.",
			InputSchema: json.RawMessage(`{
  "type": "object",
  "required": ["id"],
  "properties": {
    "id": {"type": "string", "description": "the content-address handle (sha256 hex) a compaction tombstone embedded as id=<hex>, or a recall page digest"},
    "trace_id": {"type": "string", "description": "session trace id; omitted uses the gateway default trace (your own session under fak guard)"},
    "image_dir": {"type": "string", "description": "optional persisted recall core image dir; when the compaction stash misses the id, a recall page at that digest is paged back in under the image's trust gate"}
  },
  "additionalProperties": false
}`),
			Annotations: readOnlyToolAnnotations(),
		}.toMap(),
		mcpToolDescriptor{
			Name:        "fak_context_spans",
			Description: "List dropped context spans available to fak_context_restore. Returns safe metadata (id, excerpt, bytes, evidence edges, suppression, restorable) without content or paging; sealed or tombstoned spans remain listed but not restorable. Optional trace_id defaults to the current guarded session; unknown traces return Count 0.",
			InputSchema: json.RawMessage(`{
  "type": "object",
  "properties": {
    "trace_id": {"type": "string", "description": "session trace id; omitted uses the gateway default trace (your own session under fak guard)"}
  },
  "additionalProperties": false
}`),
			Annotations: readOnlyToolAnnotations(),
		}.toMap(),
		mcpToolDescriptor{
			Name:        "fak_resume_history",
			Description: "Inspect the current session's durable resume and heal history. Returns attempts, closed resume_state, retry block and reason, earned budget, operator settlement, and next hint; no history is explicit and unresolved ledgers fail closed. Read-only and advice-only. Optional session, ledger, and max_attempts default from the guarded fleet environment.",
			InputSchema: json.RawMessage(`{
  "type": "object",
  "properties": {
    "session": {"type": "string", "description": "session id to observe; omitted resolves $CLAUDE_SESSION_ID, then the gateway default trace (your own session under fak guard)"},
    "ledger": {"type": "string", "description": "explicit resume ledger path; omitted resolves the fleet default from the environment ($FLEET_REG_DIR, then the Fleet registry conventions)"},
    "max_attempts": {"type": "integer", "description": "give-up cap; omitted or <= 0 uses the progress-earned budget"}
  },
  "additionalProperties": false
}`),
			Annotations: readOnlyToolAnnotations(),
		}.toMap(),
	}
}

// ToolDescriptorsForResolver returns the tool descriptors for use by the
// protocol-generic capindex MCP resolver. This function is exported to allow
// the resolver to index MCP tools as generic Capabilities, proving the loader
// is protocol-blind (issue #1108, C5).
func ToolDescriptorsForResolver() []map[string]any {
	return toolDescriptors()
}

// validateToolDescriptors verifies that every registered MCP tool provides a
// well-formed, provider-conforming OpenAPI 3.0 schema at startup (#10769).
// Fails loud rather than discover schema dialect incompatibilities at runtime.
func validateToolDescriptors() error {
	for _, td := range toolDescriptors() {
		name, _ := td["name"].(string)
		if name == "" {
			return errors.New("gateway: tool descriptor missing name")
		}
		desc, _ := td["description"].(string)
		if desc == "" {
			return fmt.Errorf("gateway: tool %q missing description", name)
		}
		raw, ok := td["inputSchema"].(json.RawMessage)
		if !ok || len(raw) == 0 {
			continue
		}
		var schemaObj map[string]any
		if err := json.Unmarshal(raw, &schemaObj); err != nil {
			return fmt.Errorf("gateway: tool %q inputSchema is not valid JSON: %w", name, err)
		}
		if err := validateOpenAPISchemaNode(name, schemaObj); err != nil {
			return err
		}
	}
	return nil
}

func validateOpenAPISchemaNode(path string, schema map[string]any) error {
	stype, _ := schema["type"].(string)
	if req, ok := schema["required"].([]any); ok {
		if !strings.EqualFold(stype, "object") {
			return fmt.Errorf("gateway: schema %s.required only allowed for type object (got %q)", path, stype)
		}
		props, _ := schema["properties"].(map[string]any)
		for _, r := range req {
			rStr, _ := r.(string)
			if props == nil || props[rStr] == nil {
				return fmt.Errorf("gateway: schema %s.required property %q not declared in properties", path, rStr)
			}
		}
	}
	for _, key := range []string{"anyOf", "any_of", "oneOf", "one_of", "allOf", "all_of"} {
		if alts, ok := schema[key].([]any); ok {
			for i, alt := range alts {
				if altMap, ok := alt.(map[string]any); ok {
					subPath := fmt.Sprintf("%s.%s[%d]", path, key, i)
					if err := validateOpenAPISchemaNode(subPath, altMap); err != nil {
						return err
					}
				}
			}
		}
	}
	if props, ok := schema["properties"].(map[string]any); ok {
		for k, child := range props {
			if childMap, ok := child.(map[string]any); ok {
				if err := validateOpenAPISchemaNode(path+".properties."+k, childMap); err != nil {
					return err
				}
			}
		}
	}
	if items, ok := schema["items"].(map[string]any); ok {
		if err := validateOpenAPISchemaNode(path+".items", items); err != nil {
			return err
		}
	}
	return nil
}

// toolRegistryNames returns the bare names of every tool in the built-in
// registry, in registry order. It is the universe compileToolExposeAllow
// validates --expose patterns against and exposedToolDescriptors filters.
func toolRegistryNames() []string {
	all := toolDescriptors()
	names := make([]string, 0, len(all))
	for _, td := range all {
		if n, _ := td["name"].(string); n != "" {
			names = append(names, n)
		}
	}
	return names
}

// compileToolExposeAllow turns the raw --expose patterns (Config.ExposeTools)
// into an allowlist predicate over tool names. Each raw entry may be
// comma-separated; entries are trimmed and empties dropped, so `--expose a,b`
// and `--expose a --expose b` compile identically. An empty result set returns
// (nil, nil) — the full-surface default. Otherwise every pattern is validated
// as a path.Match glob that matches ≥1 registered tool; a malformed glob or a
// zero-match pattern is a fail-loud error (a typo must never silently hide the
// whole surface). The returned predicate reports whether a tool NAME matches
// any pattern.
func compileToolExposeAllow(raw []string) (func(string) bool, error) {
	var patterns []string
	for _, entry := range raw {
		for _, p := range strings.Split(entry, ",") {
			if p = strings.TrimSpace(p); p != "" {
				patterns = append(patterns, p)
			}
		}
	}
	if len(patterns) == 0 {
		return nil, nil
	}
	names := toolRegistryNames()
	for _, p := range patterns {
		matched := false
		for _, n := range names {
			ok, err := path.Match(p, n)
			if err != nil {
				return nil, fmt.Errorf("gateway: --expose pattern %q is not a valid glob: %w", p, err)
			}
			if ok {
				matched = true
			}
		}
		if !matched {
			return nil, fmt.Errorf("gateway: --expose pattern %q matches no known tool (have: %s)", p, strings.Join(names, ", "))
		}
	}
	return func(name string) bool {
		for _, p := range patterns {
			if ok, _ := path.Match(p, name); ok {
				return true
			}
		}
		return false
	}, nil
}

func (s *Server) supportsVision() bool {
	if s == nil {
		return false
	}
	if s.modelVision {
		return true
	}
	if envEnabled("FAK_MODEL_VISION") {
		return true
	}
	m := strings.ToLower(s.model)
	if m != "" && (strings.Contains(m, "vision") || strings.Contains(m, "-vl") || strings.Contains(m, "4o") || strings.Contains(m, "claude") || strings.Contains(m, "gemini")) {
		return true
	}
	return false
}

// exposedToolDescriptors is toolDescriptors filtered by the optional --expose
// allowlist and model capability constraints (e.g. vision support for view_image).
// It is the SINGLE choke point every discovery view routes through
// (tools/list, fak_tools_search, the capabilities resource + self-feature
// catalog, fak_feature_query, fak_capabilities), so a hidden tool never appears
// in any of them. When no allowlist is in force (s.exposeAllow == nil) it
// returns the eligible registry filtered by model capabilities.
func (s *Server) exposedToolDescriptors() []map[string]any {
	all := toolDescriptors()
	hasVision := s.supportsVision()
	out := make([]map[string]any, 0, len(all))
	for _, td := range all {
		name, _ := td["name"].(string)
		if !hasVision && (name == "view_image" || name == "functions.view_image") {
			continue
		}
		if s != nil && s.exposeAllow != nil && !s.exposeAllow(name) {
			continue
		}
		out = append(out, td)
	}
	return out
}

// memoryInputSchema is the {driver|query, intent, k, budget, image_dir, apply} shape
// shared by fak_memory_explain and fak_memory_run.
var memoryInputSchema = json.RawMessage(`{
  "type": "object",
  "properties": {
    "driver": {"type": "string", "description": "a built-in strategy name (see fak_memory_drivers); omit if you supply an inline query"},
    "query": {"type": "object", "description": "an inline authored memq Query: {intent, near_dup_threshold?, ops:[{kind, pred?, by?, desc?, k?, bytes?, reason?}]}. Ops: scan|filter|rank|limit|budget|dedup|render|tombstone|consolidate|reclassify|prune"},
    "intent": {"type": "string", "description": "the task intent (drives relevance ranking and default match terms)"},
    "k": {"type": "integer", "description": "limit (driver-specific; 0 = driver default)"},
    "budget": {"type": "integer", "description": "byte budget for the rendered/selected set (0 = unbounded)"},
    "image_dir": {"type": "string", "description": "run (not explain): path to a recall core image; omit for the in-memory demo corpus"},
    "apply": {"type": "boolean", "description": "run only: APPLY the safe negative-only/storage mutations (tombstone, prune). Default false = propose only (fail-closed)"},
    "backend": {"type": "string", "description": "run only: recall source. \"\" (default) = recall image at image_dir else demo; \"codex\" = read the external Codex memories home as a READ-ONLY generated recall layer (every cell external/untrusted, gated — not an AGENTS.md replacement)"},
    "codex_home": {"type": "string", "description": "run only, backend=codex: the Codex memories home to read (default: $CODEX_HOME; never silently ~/.codex over MCP)"},
    "include_chronicle": {"type": "boolean", "description": "run only, backend=codex: also include the higher-risk screen-generated chronicle memories. Default false"}
  },
  "additionalProperties": false
}`)

// ToolsSearchRequest is the request shape for fak_tools_search.
type ToolsSearchRequest struct {
	Query       string `json:"query"`        // optional filter string
	DetailLevel string `json:"detail_level"` // "name" | "description" | "full"
}

// ToolsSearchResponse is the response shape for fak_tools_search.
type ToolsSearchResponse struct {
	Tools []map[string]any `json:"tools"` // filtered tool descriptors at requested detail level
}

// toolsSearch is the tool_search_tool: lazy/on-demand tool-schema loading with
// progressive disclosure. As of #3231 it ranks the FULL exposed registry
// (including the cold tools absent from a deferred tools/list) through the
// selfquery hybrid catalog (#3235) rather than a flat substring scan, so a
// deferred schema is re-findable by intent — recall is the documented failure
// mode of deferral. Results are returned best-first at the requested detail
// level. This is the "full retrieval view" half of the two-view split: tools/list
// may be schema-light, but the search tool always sees every exposed tool.
func (s *Server) toolsSearch(req ToolsSearchRequest) (ToolsSearchResponse, error) {
	level := req.DetailLevel
	if level == "" {
		level = "description" // default to middle ground
	}
	if level != "name" && level != "description" && level != "full" {
		return ToolsSearchResponse{}, fmt.Errorf("invalid detail_level: %s (must be name, description, or full)", level)
	}

	// Index the full registry by name so ranked names map back to descriptors.
	byName := make(map[string]map[string]any)
	for _, t := range s.exposedToolDescriptors() {
		if n, _ := t["name"].(string); n != "" {
			byName[n] = t
		}
	}

	ranked := s.rankToolNamesByIntent(req.Query)
	result := make([]map[string]any, 0, len(ranked))
	for _, name := range ranked {
		t, ok := byName[name]
		if !ok {
			continue
		}
		desc, _ := t["description"].(string)
		tool := map[string]any{"name": name}
		if level == "description" || level == "full" {
			tool["description"] = desc
		}
		if level == "full" {
			if schema, ok := t["inputSchema"]; ok {
				tool["inputSchema"] = schema
			}
			if ann, ok := t["annotations"]; ok {
				tool["annotations"] = ann
			}
		}
		result = append(result, tool)
	}

	return ToolsSearchResponse{Tools: result}, nil
}
