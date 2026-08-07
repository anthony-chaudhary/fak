package gateway

// http_fak_endpoints.go — the /v1/fak/* control-plane HTTP handlers, moved out of
// http.go verbatim as behavior-preserving code motion (same package, same decls) so the
// gateway's front-door file stays under the god-file growth ceiling (#2898). These are
// the single-shot fak-native verbs: syscall / admit / adjudicate, the changes and
// audit-journal feeds, revoke, context-change, the policy and route reloads, and the
// trace reset/observe pair. The /v1/fak/session/ drive-state subtree stays in http.go,
// and routeTable() remains the single source of truth for what is mounted where.

import (
	"encoding/json"
	"errors"
	"net/http"
	"strconv"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/journal"
)

// ---------------------------------------------------------------------------
// fak-native surface — the simplest non-Go integration: one POST, one verdict.
// ---------------------------------------------------------------------------

// handleFakSyscall adjudicates AND executes a single tool call through the kernel
// (the self-contained / CI path).
func (s *Server) handleFakSyscall(w http.ResponseWriter, r *http.Request) {
	req, ok := s.decodeSyscall(w, r)
	if !ok {
		return
	}
	ctx := WithPrincipal(r.Context(), principalFor(r, req.Principal))
	wv, env, err := s.syscall(ctx, req.Tool, rawArgs(req.Arguments), req.ReadOnly, req.Witness, req.TraceID)
	if err != nil {
		writeErr(w, http.StatusBadRequest, err.Error())
		return
	}
	writeJSON(w, http.StatusOK, SyscallResponse{Verdict: wv, Result: env, TraceID: req.TraceID})
}

// principalFor resolves a request's isolation principal: the X-Fak-Principal header
// (set by an auth proxy / tenant router in front of the gateway) takes precedence, else
// the request body's principal field. Empty => single-tenant (every caller shares).
func principalFor(r *http.Request, bodyPrincipal string) string {
	// A keyset-authenticated caller's principal is bound at the front door (withAuth →
	// WithPrincipal, #5332) and is AUTHORITATIVE: it outranks the X-Fak-Principal header
	// and the body field so a caller cannot present one key and then claim another
	// tenant via a spoofed header or body. On the no-keyset path the context carries no
	// principal, so this falls through to the header/body exactly as before.
	if p := strings.TrimSpace(principalFromContext(r.Context())); p != "" {
		return p
	}
	if h := strings.TrimSpace(r.Header.Get("X-Fak-Principal")); h != "" {
		return h
	}
	return strings.TrimSpace(bodyPrincipal)
}

// handleFakAdmit runs a CLIENT-PRODUCED tool result through the kernel's
// result-side stack (context-MMU quarantine + IFC source-stamp). This is the
// served-path complement to handleFakAdjudicate: adjudicate gates the CALL before
// the client runs it; admit contains the RESULT after. A poisoned result is
// quarantined and the session's taint ledger raised before it is admitted.
func (s *Server) handleFakAdmit(w http.ResponseWriter, r *http.Request) {
	if !requireMethod(w, r, http.MethodPost) {
		return
	}
	var req AdmitRequest
	if !decodeRequestBody(w, r, &req) {
		return
	}
	req.TraceID = s.useHTTPTrace(w, r, req.TraceID)
	wv, env, err := s.admit(r.Context(), req.Tool, rawArgs(req.Result), req.Witness, req.TraceID)
	if err != nil {
		// A REMOTE engine-cache reset failure is a gateway/upstream fault — surface it
		// as a 502 (the same fail-closed signal the proxy returns), with a generic
		// message so the upstream error body never crosses the trust boundary. Any
		// other admit error is a client-side 400.
		if errors.Is(err, errEngineCacheReset) {
			s.logf("gateway: native admit engine cache reset failed: %v", err)
			writeErr(w, http.StatusBadGateway, "upstream cache invalidation failed")
			return
		}
		writeErr(w, http.StatusBadRequest, err.Error())
		return
	}
	writeJSON(w, http.StatusOK, SyscallResponse{Verdict: wv, Result: env, TraceID: req.TraceID})
}

// handleFakAdjudicate returns the pre-execution verdict only (the production path
// for a client that runs its own tools): no dispatch, no engine, no pending state.
func (s *Server) handleFakAdjudicate(w http.ResponseWriter, r *http.Request) {
	req, ok := s.decodeSyscall(w, r)
	if !ok {
		return
	}
	wv, repaired, err := s.adjudicate(r.Context(), req.Tool, rawArgs(req.Arguments), req.ReadOnly, req.Witness, req.TraceID)
	if err != nil {
		writeErr(w, http.StatusBadRequest, err.Error())
		return
	}
	resp := SyscallResponse{Verdict: wv, TraceID: req.TraceID}
	if repaired != "" {
		resp.RepairedArguments = json.RawMessage(repaired)
	}
	writeJSON(w, http.StatusOK, resp)
}

// handleFakChanges drains the cross-agent "what changed" feed for events after the
// client's ?since= (or {"since":N}) cursor. GET or POST.
func (s *Server) handleFakChanges(w http.ResponseWriter, r *http.Request) {
	var since uint64
	var bodyPrincipal string
	if r.Method == http.MethodPost {
		var req ChangesRequest
		if !decodeRequestBody(w, r, &req) {
			return
		}
		since = req.Since
		bodyPrincipal = req.Principal
	} else if v := r.URL.Query().Get("since"); v != "" {
		var n uint64
		for _, c := range v {
			if c < '0' || c > '9' {
				writeErr(w, http.StatusBadRequest, "since must be a non-negative integer")
				return
			}
			n = n*10 + uint64(c-'0')
		}
		since = n
	}
	events, cursor := s.changes(principalFor(r, bodyPrincipal), since)
	writeJSON(w, http.StatusOK, ChangesResponse{Events: events, Cursor: cursor})
}

// activeJournal returns the process-global durable audit journal, or nil if
// FAK_AUDIT_JOURNAL was unset at boot. Indirected through a var so a test can
// inject an in-memory journal without process-global env setup.
var activeJournal = journal.Active

// EventsResponse is the drained durable audit-journal tail plus the client's next
// cursor (mirrors ChangesResponse for the coherence feed).
type EventsResponse struct {
	Events []journal.Row `json:"events"`
	Cursor uint64        `json:"cursor"`
	// Principals maps a row's X-Trace-Id to the tenant ISOLATION principal that owns
	// that trace (the keyset-bound org/project, #5332), letting a reader attribute each
	// audit row to the tenant that produced it WITHOUT the hash-chained journal.Row
	// schema carrying — or persisting — a principal field. Populated at read time from
	// the live traceOwner map; omitted when no drained row has a named owner (the
	// single-tenant loopback), so the single-key wire shape is unchanged.
	Principals map[string]string `json:"principals,omitempty"`
}

// decodeEventsRequest decodes the /v1/fak/events POST body STRICTLY — an unknown field
// is a 400, the body-side twin of the unknown-query-key rejection. It is a local
// variant of decodeRequestBody rather than a change to that shared helper because
// every other fak-native verb tolerates forward-compatible extra fields; only the
// events drain treats an unrecognized key as a filter the caller believes is applied.
func decodeEventsRequest(w http.ResponseWriter, r *http.Request, req *EventsRequest) bool {
	r.Body = http.MaxBytesReader(w, r.Body, maxTranscriptBody)
	dec := json.NewDecoder(r.Body)
	dec.DisallowUnknownFields()
	if err := dec.Decode(req); err != nil {
		writeErr(w, http.StatusBadRequest, "malformed request body: "+err.Error())
		return false
	}
	return true
}

// EventsRequest is the POST body shape of /v1/fak/events: the ?since= cursor plus
// the optional row predicates (#3979). It is deliberately its OWN type rather than a
// reuse of ChangesRequest — the two feeds narrow by different columns (the coherence
// feed by isolation principal, the audit journal by verdict/tool/trace/kind) — and it
// is decoded STRICTLY, so a misspelled predicate is a 400 rather than a silently
// unfiltered full-tail drain that reads like "nothing matched".
type EventsRequest struct {
	Since   uint64 `json:"since,omitempty"`
	Verdict string `json:"verdict,omitempty"`
	Tool    string `json:"tool,omitempty"`
	TraceID string `json:"trace_id,omitempty"`
	Kind    string `json:"kind,omitempty"`
}

// eventsFilterKeys is the closed set of query parameters /v1/fak/events accepts. Any
// other key is a 400: an unrecognized predicate that were merely IGNORED would drain
// the whole tail and look to the caller like a filter that matched everything.
var eventsFilterKeys = map[string]bool{
	"since": true, "verdict": true, "tool": true, "trace_id": true, "kind": true,
}

// eventsFilter is a pure conjunctive row predicate over columns journal.Row already
// carries. An empty field is "don't care", so the zero value matches every row and the
// unfiltered drain keeps its exact previous behavior. Matching is exact equality — the
// journal writes Kind and Verdict as closed upper-case enums (DECIDE/DENY/..., the
// abi.Verdict kinds), and Tool/TraceID are opaque identifiers whose case is meaningful.
type eventsFilter struct {
	verdict string
	tool    string
	traceID string
	kind    string
}

func (f eventsFilter) match(row journal.Row) bool {
	if f.verdict != "" && row.Verdict != f.verdict {
		return false
	}
	if f.tool != "" && row.Tool != f.tool {
		return false
	}
	if f.traceID != "" && row.TraceID != f.traceID {
		return false
	}
	if f.kind != "" && row.Kind != f.kind {
		return false
	}
	return true
}

// handleFakEvents drains the durable, hash-chained audit journal
// (internal/journal) after the client's ?since= cursor — the Seq of the last row
// it saw; 0 returns the whole retained tail. It mirrors the /v1/fak/changes
// cursor protocol but over the persisted verdict ledger rather than the live
// coherence bus. It serves the bounded in-memory tail without re-reading disk;
// the full tamper-evident history is the on-disk JSONL. Returns 404 if no journal
// is configured (FAK_AUDIT_JOURNAL unset at boot). GET or POST {"since":N}.
//
// The drain narrows by the combinable row predicates ?verdict= / ?tool= / ?trace_id= /
// ?kind= (POST body equivalents, #3979), so a remote consumer debugging a deny-storm
// pays for the rows it asked for instead of the whole tail plus a client-side jq. The
// predicates are PURE row filters and do not touch the cursor protocol: the cursor
// still advances over every row the drain walked, including the ones the filter
// dropped, so a narrowed poll neither re-delivers a filtered-out row on the next call
// nor skips the head — and a drain that matches nothing still returns the advanced
// cursor rather than pinning the caller to a stale one.
func (s *Server) handleFakEvents(w http.ResponseWriter, r *http.Request) {
	j := activeJournal()
	if j == nil {
		writeErr(w, http.StatusNotFound, "audit journal not enabled (set FAK_AUDIT_JOURNAL to a path)")
		return
	}
	var since uint64
	var filter eventsFilter
	if r.Method == http.MethodPost {
		var req EventsRequest
		if !decodeEventsRequest(w, r, &req) {
			return
		}
		since = req.Since
		filter = eventsFilter{verdict: req.Verdict, tool: req.Tool, traceID: req.TraceID, kind: req.Kind}
	} else {
		q := r.URL.Query()
		for key := range q {
			if !eventsFilterKeys[key] {
				writeErr(w, http.StatusBadRequest, "unknown query parameter "+strconv.Quote(key)+
					"; supported: since, verdict, tool, trace_id, kind")
				return
			}
		}
		if v := q.Get("since"); v != "" {
			n, err := strconv.ParseUint(v, 10, 64)
			if err != nil {
				writeErr(w, http.StatusBadRequest, "since must be a non-negative integer")
				return
			}
			since = n
		}
		filter = eventsFilter{
			verdict: q.Get("verdict"), tool: q.Get("tool"),
			traceID: q.Get("trace_id"), kind: q.Get("kind"),
		}
	}
	rows := j.Recent(0)
	out := make([]journal.Row, 0, len(rows))
	cursor := since
	for _, row := range rows {
		if row.Seq > since && filter.match(row) {
			out = append(out, row)
		}
		// Advance over EVERY walked row, not just the delivered ones: the cursor is the
		// caller's "I have seen up to here" watermark over the journal's Seq line, not a
		// count of what it was handed.
		if row.Seq > cursor {
			cursor = row.Seq
		}
	}
	resp := EventsResponse{Events: out, Cursor: cursor}
	// Attribute each drained row to the tenant principal that owns its trace, so a
	// reader joins the audit journal to the org/project that produced it by X-Trace-Id.
	// Read-time only: the on-disk hash-chained row schema is untouched. Only NAMED
	// owners are emitted (an unbound / single-tenant "" owner is skipped), so the map is
	// absent on the single-key loopback and present exactly for keyset-attributed rows.
	principals := make(map[string]string, len(out))
	for _, row := range out {
		if row.TraceID == "" {
			continue
		}
		if _, seen := principals[row.TraceID]; seen {
			continue
		}
		if owner, ok := s.traceOwnerOf(row.TraceID); ok && owner != "" {
			principals[row.TraceID] = owner
		}
	}
	if len(principals) > 0 {
		resp.Principals = principals
	}
	writeJSON(w, http.StatusOK, resp)
}

// handleFakRevoke triggers a fleet-wide refutation of an external world-state witness.
func (s *Server) handleFakRevoke(w http.ResponseWriter, r *http.Request) {
	if !requireMethod(w, r, http.MethodPost) {
		return
	}
	var req RevokeRequest
	if !decodeRequestBody(w, r, &req) {
		return
	}
	if req.Witness == "" {
		writeErr(w, http.StatusBadRequest, "revoke requires a non-empty witness")
		return
	}
	evicted, te := s.revoke(req.Witness)
	writeJSON(w, http.StatusOK, RevokeResponse{Witness: req.Witness, Evicted: evicted, TrustEpoch: te})
}

// handleFakContextChange records a safe requester-initiated mutation against a
// persisted recall core image. The only shipped mutation is a tombstone that
// suppresses one page from future model-visible recall without deleting evidence.
func (s *Server) handleFakContextChange(w http.ResponseWriter, r *http.Request) {
	if !requireMethod(w, r, http.MethodPost) {
		return
	}
	var req ContextChangeRequest
	if !decodeRequestBody(w, r, &req) {
		return
	}
	resp, err := s.contextChange(r.Context(), req)
	if err != nil {
		writeErr(w, http.StatusBadRequest, err.Error())
		return
	}
	writeJSON(w, http.StatusOK, resp)
}

// handleFakPolicyReload reloads the configured policy manifest in-place. The
// actual loader is injected by cmd/fak so this package stays policy-schema blind.
func (s *Server) handleFakPolicyReload(w http.ResponseWriter, r *http.Request) {
	if !requireMethod(w, r, http.MethodPost) {
		return
	}
	if s.reloadPolicy == nil {
		writeErr(w, http.StatusNotFound, "policy reload is not configured")
		return
	}
	resp, err := s.reloadPolicy(r.Context())
	if err != nil {
		s.logf("gateway: policy reload failed: %v", err)
		writeErr(w, http.StatusBadRequest, "policy reload failed: "+err.Error())
		return
	}
	if resp.Source != "" {
		s.logf("gateway: reloaded capability floor from %s", resp.Source)
	} else {
		s.logf("gateway: reloaded capability floor")
	}
	writeJSON(w, http.StatusOK, resp)
}

// handleFakRouteReload forces a reload of the installed model-routing manifest in
// place (#4003) — the route-plane twin of handleFakPolicyReload. It drives the SAME
// modelroute.Watcher the background poll loop uses (installed by the host via
// SetRouteWatcher), so a manual reload and the poll loop share the last-good gate and
// the atomic Live swap. A malformed edit is REJECTED (last-good kept) and reported as
// a 400, never a silent success; a byte-identical file is a 200 no-op (reloaded:false).
func (s *Server) handleFakRouteReload(w http.ResponseWriter, r *http.Request) {
	if !requireMethod(w, r, http.MethodPost) {
		return
	}
	watcher := s.currentRouteWatcher()
	if watcher == nil {
		writeErr(w, http.StatusNotFound, "route reload is not configured")
		return
	}
	ev := watcher.Reload()
	if ev.Err != nil {
		s.logf("gateway: route reload failed: %v", ev.Err)
		writeErr(w, http.StatusBadRequest, "route reload failed: "+ev.Err.Error())
		return
	}
	if ev.Reloaded {
		s.logf("gateway: hot-reloaded model-routing policy from %s (reload #%d)", ev.Path, ev.Reloads)
	} else {
		s.logf("gateway: route reload: no change (%s)", ev.Path)
	}
	writeJSON(w, http.StatusOK, RouteReloadResponse{
		Reloaded: ev.Reloaded,
		Source:   ev.Path,
		Changed:  ev.Changed,
		Reloads:  ev.Reloads,
		Rejects:  ev.Rejects,
	})
}

// handleFakTraceReset clears the per-trace IFC taint high-water mark for a live
// served session. The reset implementation is injected by cmd/fak.
func (s *Server) handleFakTraceReset(w http.ResponseWriter, r *http.Request) {
	if !requireMethod(w, r, http.MethodPost) {
		return
	}
	if s.resetTrace == nil {
		writeErr(w, http.StatusNotFound, "trace reset is not configured")
		return
	}
	var req TraceResetRequest
	if !decodeRequestBody(w, r, &req) {
		return
	}
	traceID := strings.TrimSpace(req.TraceID)
	if traceID == "" {
		writeErr(w, http.StatusBadRequest, "trace_id is required")
		return
	}
	if err := s.resetTrace(r.Context(), traceID); err != nil {
		s.logf("gateway: trace reset failed: %v", err)
		writeErr(w, http.StatusBadRequest, "trace reset failed: "+err.Error())
		return
	}
	s.logf("gateway: reset trace %s", traceID)
	writeJSON(w, http.StatusOK, TraceResetResponse{Reset: true, TraceID: traceID})
}

// handleFakTraceObserve is the read-only complement of /v1/fak/trace/reset (#411):
// GET /v1/fak/trace/{trace_id} returns the current IFC taint high-water mark for a
// live/recent served session, so an operator can see whether a session's taint is
// rising WITHOUT parsing stderr. It is mounted on the /v1/fak/trace/ subtree; the
// exact /v1/fak/trace/reset route (POST) is matched first by the mux, so only the
// observe id-path lands here. The observe implementation is injected by cmd/fak so
// this package stays IFC-internals blind, mirroring resetTrace.
func (s *Server) handleFakTraceObserve(w http.ResponseWriter, r *http.Request) {
	if !requireMethod(w, r, http.MethodGet) {
		return
	}
	if s.observeTrace == nil {
		writeErr(w, http.StatusNotFound, "trace observe is not configured")
		return
	}
	traceID := strings.TrimSpace(strings.TrimPrefix(r.URL.Path, "/v1/fak/trace/"))
	if traceID == "" {
		writeErr(w, http.StatusBadRequest, "trace_id is required")
		return
	}
	level, dangerous := s.observeTrace(r.Context(), traceID)
	writeJSON(w, http.StatusOK, TraceObserveResponse{TraceID: traceID, Taint: level, Dangerous: dangerous})
}
