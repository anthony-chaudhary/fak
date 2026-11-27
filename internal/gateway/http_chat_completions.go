package gateway

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/ctxmmu"
	"github.com/anthony-chaudhary/fak/pkg/turncost"
)

// ---------------------------------------------------------------------------
// OpenAI-compatible surface.
// ---------------------------------------------------------------------------

// handleChatCompletions is the adjudication PROXY. It forwards the chat to the
// configured model (upstream HTTPPlanner or the offline mock), then runs each
// PROPOSED tool_call through k.Decide BEFORE the caller sees it: denied calls are
// dropped, grammar-repaired calls have their arguments rewritten to the canonical
// form, and a fak-aware client gets the full per-call adjudication in the `fak`
// extension. It NEVER executes the client's tools — the client does.
func (s *Server) handleChatCompletions(w http.ResponseWriter, r *http.Request) {
	if !requireMethod(w, r, http.MethodPost) {
		return
	}
	turnCostBegan := time.Now()
	if s.checkWarmupPending(w) {
		return
	}
	if dl := s.ScalarConfig().CompletionDeadlineMs; dl > 0 {
		ctx, cancel := context.WithTimeout(r.Context(), time.Duration(dl)*time.Millisecond)
		defer cancel()
		r = r.WithContext(ctx)
	}
	releaseEPFanout, waitEPFanout, ok := s.prepareChatEPFanout(w, r, epRouteChatCompletions)
	if !ok {
		return
	}
	defer waitEPFanout()
	var req ChatRequest
	if !decodeRequestBody(w, r, &req) {
		return
	}
	normalizeChatMaxTokens(&req)
	req.AffinityKey = extractAffinityKey(r, req.AffinityKey)
	// Dialect tool-normalization parity: fold the OpenAI chat tool catalog onto the
	// same serving-input contract the Responses wire folds through, so every mounted
	// dialect hands admission + the served core one normalized catalog (empty-name
	// drop, first-wins dedupe, MCP legacy/canonical collapse). Capability checks stay
	// per-dialect; only the serving input is shared.
	req.Tools = normalizeServingToolDefs(req.Tools)
	if !validateChatRequestIngress(w, req) {
		return
	}
	loopDepth, ok := resolveChatLoopDepth(w, req, s.loopDepthRule.Load())
	if !ok {
		return
	}
	req.AffinityKey = depthAffinityKey(req.AffinityKey, loopDepth)
	s.bindUnkeyedChatConversation(w, r, req.Model, req.Tools, req.Messages)
	// Bind the client deadline before routing or preparation. Cache credit is
	// selected only after the execution route is fixed below.
	r, cancelDeadline := bindClientDeadline(r, turnCostBegan)
	defer cancelDeadline()
	// Retain served-prefix history as observation only. Admission below uses
	// current request-owned native state or measured residency evidence.
	r = withDeadlineCacheObservation(r, req.Model, req.Tools, req.Messages)
	deadlineCtx := r.Context()
	// Stamp the causal input on the untouched wire envelope before admission
	// transforms, request routing, planner selection, or model execution.
	inputTriggerRoute, routedModel, err := s.admitAndRouteChatInputTriggerWithContext(r.Context(), req)
	if err != nil {
		if errors.Is(err, errInvalidExplicitInputTrigger) {
			writeErr(w, http.StatusBadRequest, "invalid input_trigger")
			return
		}
		s.logf("gateway: input-trigger request route failed: %v", err)
		writeErr(w, http.StatusInternalServerError, "input-trigger request routing failed")
		return
	}
	if routedModel != "" {
		req.Model = routedModel
	}
	// Request-invariant system+tools head so parent and subagent requests share a prefix-KV hit.
	req.Messages, req.Tools = normalizeHarnessPrefix(req.Messages, req.Tools)
	r, ok = s.prepareChatRoute(w, r, req.Model)
	if !ok {
		return
	}
	if s.refuseNativeModelMismatch(w, r, req.Model, routedModel) {
		return
	}
	r, releaseDeadline, ok := s.admitRoutedClientDeadlineModel(w, r, turnCostBegan, req.Model, req.Messages, req.MaxTokens)
	if !ok {
		return
	}
	defer releaseDeadline()
	if chatBeamParams(req) != nil && !s.chatBeamSearchSupported(req.Model) {
		writeErrCode(w, http.StatusBadRequest, "native_beam_search_unsupported", "beam search requires the native dense CPU route")
		return
	}
	if !releaseEPFanout(r) {
		return
	}
	receiptRequested := req.Fak != nil && req.Fak.NativeInferenceReceipt
	decodeTraceRequested := req.FakDecodeTrace
	decodeTokenIDsRequested := req.Fak != nil && req.Fak.NativeDecodeTokenIDs
	// Request-model pass-through (#82): forward the client's requested model to the
	// upstream verbatim, falling back to the gateway's configured model only when the
	// client omitted one. This stops the gateway silently serving a DIFFERENT model
	// than the client asked for — an unknown model now reaches the upstream and
	// surfaces its 404 instead of a misleading 200. --model stays the advertised
	// /v1/models id and the default. reqModel is also the response-model fallback
	// when the upstream omits a served-model field.
	reqModel := req.Model
	if reqModel == "" {
		reqModel = s.model
	}
	applyChatCompletionSpeculativeHeaders(w, s, reqModel)
	if req.Stream && s.isVulkanMTPEnabled(reqModel) && chatRequestVulkanMTPEligible(req) {
		// The buffered in-kernel stream opens before decode completes. Declare
		// trailers now, then populate them from the request-local execution
		// receipt after completion; configuration alone never claims MTP ran.
		w.Header().Add("Trailer", HeaderSpeculative)
		w.Header().Add("Trailer", HeaderSpeculativeDowngrade)
	}
	if decodeTraceRequested && !s.chatDecodeTraceSupported(req.Model) {
		writeErr(w, http.StatusBadRequest, "fak_decode_trace requires a fak-native model route")
		return
	}

	// Thread one request TraceID across every proposed call in this chat so the IFC
	// ledger, plan-CFI, response header, and access log all correlate. The
	// middleware honors a client-supplied X-Trace-Id or mints one.
	ctx, reqTrace, messages, sessionTurn, admitted := s.admitServedRequest(w, r, req.Messages)
	if !admitted {
		return
	}
	// Count a text-form tool call promoted into structured calls as a parse
	// outcome, not just a log line. Installed on the handler ctx so both the
	// buffered path and the live chat stream (called with this same ctx) inherit it.
	ctx = s.withToolCallParseObserver(ctx)
	sessionTurn.turnCost = newTurnCostRecord(req.Stream)
	defer func() {
		sessionTurn.complete()
		s.finishTurnCost(sessionTurn, reqTrace, reqModel, turnCostBegan)
	}()
	req.Messages = messages
	// Oversize tool results page out to restore stubs only for a client that can call
	// the restore tool; the decision rides ctx into both admission passes. The
	// server's resolved CtxPaging posture (serve: OFF) sits under the per-request
	// header and the process env.
	paging := s.resolveChatCtxPaging(r, req.Tools)
	w.Header().Set(HeaderCtxPaging, paging.header)
	ctx = ctxmmu.WithOversizePagingSuppressed(ctx, paging.suppressReason)
	resultAdmissions, err := s.admitInboundResults(ctx, req.Messages, req.Tools, reqTrace)
	if err != nil {
		writeErr(w, http.StatusBadGateway, "upstream cache invalidation failed")
		return
	}

	// Microcontext tool result elision (#11648). The chat-tool-result-elide
	// posture (serve: OFF) declines the rewrite entirely — no marker, no stash,
	// no restore-tool injection; the client's tools pass through untouched.
	restoreToolName, restoreToolPresent, autoAdvertise := determineChatRestoreTool(req.Tools)
	if !s.chatToolResultElideOff {
		req.Messages = s.maybeElideResponsesToolResults(reqTrace, req.Messages, restoreToolName)
	}
	// The restore tool is auto-advertised only while a context-plane affordance
	// that PRODUCES restore handles is armed (elision or oversize paging), or
	// when the history already carries an elided marker an earlier-armed
	// posture (or the client) left behind — the defensive recovery path. With
	// both postures off (the serving-router default) a request's declared tools
	// pass through byte-identical.
	if len(req.Tools) > 0 && !restoreToolPresent &&
		((autoAdvertise && s.restoreAffordanceArmed(paging)) || hasElidedToolResult(req.Messages)) {
		req.Tools = append(req.Tools, agent.ToolDef{
			Type: "function",
			Function: agent.ToolDefFunction{
				Name:        restoreToolName,
				Description: "Restore dropped context by content-addressed sha256 id. Returns verbatim stashed bytes plus orientation; optional trace_id defaults to the current guarded session. Read-only and trust-gated.",
				Parameters:  json.RawMessage(`{"type":"object","properties":{"id":{"type":"string","description":"the content-address handle (sha256 hex) a compaction tombstone embedded as id=<hex>, or a recall page digest"},"trace_id":{"type":"string","description":"session trace id; omitted uses the gateway default trace"}},"required":["id"]}`),
			},
		})
	}
	ctx = withClientToolSchemas(ctx, req.Tools)

	// True streaming fast path: when the client asked to stream AND the planner can
	// stream this wire, forward the upstream tokens live for a real time-to-first-token
	// instead of synthesizing the SSE from a fully-buffered turn. Tool-bearing requests
	// take this path too: CompleteStream HOLDS every proposed call off-wire for
	// adjudication and the lift-guard keeps a text-form call from leaking into the live
	// content, so the buffered path's trust posture is preserved (see streamChatLive). A
	// non-streaming-wire request falls through to the buffered path below, whose tail
	// still synthesizes a stream for stream=true. streamChatLive returns false having
	// written nothing when it cannot stream, so the fall-through is safe.
	if req.Stream {
		if s.streamChatLive(ctx, w, req, reqModel, reqTrace, sessionTurn, resultAdmissions, inputTriggerRoute, loopDepth) {
			return
		}
	}

	// #5399 (the remaining half of #4855): the buffered path below blocks in
	// completeServed for the WHOLE decode, and used to write its first byte only after
	// that returned — so a stream:true request served by a Complete-only planner (every
	// in-kernel serve: agent.InKernelPlanner is not an agent.StreamingPlanner) emitted no
	// status line, no headers and no SSE byte for the entire multi-rank decode. Open the
	// stream NOW instead: 200 + SSE headers + the opening role chunk, flushed, so the
	// client can tell an accepted streaming request from a dead socket.
	//
	// Placement is load-bearing. Every PRE-decode refusal — the method/body/sampling
	// 400s, writeSessionRefusal, the inbound-result 502 — is already behind us and kept
	// its real HTTP status. Everything that can still fail below (the upstream error,
	// the tool-call conformance fail-closed) has to report in-band now, as an SSE error
	// event + [DONE]; see chatStreamWriter.fail.
	var stream *chatStreamWriter
	if req.Stream && req.Seed == nil {
		stream = newChatStreamWriter(w, reqModel, req.DeclaredStreamUsage())
		if err := stream.open(); err != nil {
			// The client is already gone; do not spend a decode on a socket nobody reads.
			s.logf("gateway: client vanished before the streamed preamble landed: %v", err)
			return
		}
	}

	// Forward the client's per-request sampling params to the upstream model. Each
	// option is a no-op when its field is absent (max_tokens 0, nil temperature/top_p,
	// empty stop), so an OpenAI client that omits them gets the planner default —
	// identical to the pre-seam behavior — while one asking for a long completion is
	// no longer hard-capped at the planner's 1024-token floor (#62).
	began := time.Now()
	serviceTier, _ := parseClientServiceTier(req.ServiceTier)
	chatOpts := append(chatWireSampleOpts(req, sessionTurn, loopDepth),
		agent.WithNativeBeamSearch(chatBeamParams(req)),
		agent.WithNativeInferenceReceipt(receiptRequested),
		agent.WithDecodeTrace(decodeTraceRequested),
		agent.WithNativeDecodeTokenIDs(decodeTokenIDsRequested),
	)
	comp, err := s.completeServed(ctx, sessionTurn, req.Messages, req.Tools, serviceTier.appendSampleOpt(chatOpts)...)
	if err != nil {
		if comp != nil {
			applyVulkanMTPExecutionHeaders(w, comp.VulkanMTP)
		}
		s.renderTurnDebugError(reqTrace, "openai_chat_completions", err, time.Since(began))
		// Map the upstream failure to an honest status. Log the detail for the operator
		// but return a GENERIC message — the planner error embeds up to 400 bytes of the
		// upstream provider's raw body, which must not cross the trust boundary to a
		// (possibly unauthenticated) downstream caller.
		s.logf("gateway: upstream model error: %v", err)
		if stream != nil {
			// The 200 + SSE headers went out before the decode, so the status line is
			// spent: report the SAME classified failure in-band as an SSE error event +
			// [DONE] rather than truncating the stream. plannerErrorStatus carries the
			// identical metric/observation side effects writeUpstreamErr would have run,
			// and msg is the same client-facing string (never the upstream's raw body).
			status, code, msg := s.plannerErrorStatus(err)
			stream.failFields(status, code, msg, s.upstreamErrorFields(err, code))
			return
		}
		s.writeUpstreamErr(w, err)
		return
	}
	applyVulkanMTPExecutionHeaders(w, comp.VulkanMTP)
	if receiptRequested {
		applyNativeBatchReceiptHeaders(w, comp.InKernelBatch)
	}
	if req.Stream && stream == nil {
		stream = newChatStreamWriter(w, reqModel, req.DeclaredStreamUsage())
		if err := stream.open(); err != nil {
			s.logf("gateway: client vanished before the streamed response landed: %v", err)
			return
		}
	}

	asst := comp.Message
	if chatBeamParams(req) != nil && comp.NativeBeamSearch == nil {
		writeErr(w, http.StatusBadGateway, "native planner did not produce the requested beam-search result")
		// A missing beam result still consumed the completed decode (#13562).
		s.logInferenceTurnAs(reqTrace, "openai_chat_completions", req.Stream, comp.Usage, toolCallLogFinishReason(comp.FinishReason), time.Since(began), false, s.consumeDecodedCtxViewEvent(reqTrace), principalFromContext(ctx))
		return
	}
	asst.Role = agent.RoleAssistant

	if !s.validateChatCompletionConformance(w, stream, comp, asst, receiptRequested, decodeTraceRequested, decodeTokenIDsRequested, inputTriggerRoute) {
		// #13562: the refused turn still spent a full decode, so it keeps its
		// per-turn inference log row — finish reason and token usage included —
		// instead of vanishing behind the typed 502 / in-band SSE error. The
		// refusal response itself is unchanged; the finish reason uses the same
		// closed vocabulary as the sibling refusal log line.
		s.logInferenceTurnAs(reqTrace, "openai_chat_completions", req.Stream, comp.Usage, toolCallLogFinishReason(comp.FinishReason), time.Since(began), false, s.consumeDecodedCtxViewEvent(reqTrace), principalFromContext(ctx))
		return
	}

	finish, adjs := s.AdjudicateAssistantTurn(ctx, &asst, nil, comp.FinishReason, reqTrace)

	// Echo the model the UPSTREAM reported it served (#82); fall back to the client's
	// requested model (or, if it omitted one, the configured model) when the upstream
	// did not name a served model. Never just s.model — that is the silent-substitution
	// this fix removes.
	respModel := s.responseModel(comp.Model, reqModel, chatStreamModel(stream), "#5399")
	s.logInferenceTurnAs(reqTrace, "openai_chat_completions", req.Stream, comp.Usage, finish, time.Since(began), false, s.consumeDecodedCtxViewEvent(reqTrace), principalFromContext(ctx))
	s.recordDeadlineWarmPrefix(deadlineCtx, comp.Usage)
	stampTurnCost(sessionTurn.turnCost, reqTrace, respModel, turnCostBegan)
	resp := s.buildChatResponse(comp, asst, finish, respModel, adjs, resultAdmissions, inputTriggerRoute, decodeTraceRequested, decodeTokenIDsRequested, sessionTurn.turnCost)
	stampLoopDepthReceipt(&resp, loopDepth)
	resp.ServiceTier = serviceTier.echo(comp.ServiceTier)
	if stream != nil {
		timePhase(sessionTurn.turnCost, turncost.PhaseStream, func() {
			writeChatCompletionStream(stream, resp)
		})
		return
	}
	writeJSON(w, http.StatusOK, resp)
}

func (s *Server) validateChatCompletionConformance(w http.ResponseWriter, stream *chatStreamWriter, comp *agent.Completion, asst agent.Message, receiptRequested, decodeTraceRequested, decodeTokenIDsRequested bool, inputTriggerRoute *InputTriggerRouteReceipt) bool {
	if comp.ToolCallsDropped && len(asst.ToolCalls) == 0 {
		s.metrics.observeToolCallParse(toolCallParseOutcomeFailClosed, "", toolCallDropReasonName(comp))
		s.logf("gateway: upstream announced tool_calls but none parsed (conformance fail-closed); model=%s reason=%s finish_reason=%s completion_tokens=%d", s.model, toolCallDropReasonName(comp), toolCallLogFinishReason(comp.FinishReason), comp.Usage.CompletionTokens)
		if msg, code, typed := toolCallDropRefusal(comp); typed {
			if stream != nil {
				stream.fail(http.StatusBadGateway, code, msg)
				return false
			}
			writeErrCode(w, http.StatusBadGateway, code, msg)
			return false
		}
		const conformanceMsg = "upstream tool-call format not recognized; refusing to skip adjudication"
		if stream != nil {
			stream.fail(http.StatusBadGateway, "", conformanceMsg)
			return false
		}
		writeErr(w, http.StatusBadGateway, conformanceMsg)
		return false
	} else if len(asst.ToolCalls) > 0 {
		s.metrics.observeToolCallParse(toolCallParseOutcomeStructured, "", "")
	}
	if receiptRequested && comp.NativeInference == nil {
		writeErr(w, http.StatusBadGateway, "configured planner cannot produce a native inference receipt")
		return false
	}
	if inputTriggerRoute != nil && comp.NativeInference != nil &&
		(comp.NativeInference.Engine != TurnIngressEngine ||
			comp.NativeInference.Model != TurnIngressModel ||
			comp.NativeInference.FallbackActive) {
		s.logf("gateway: input-trigger route execution identity mismatch")
		writeErr(w, http.StatusBadGateway, "fak-native route execution identity mismatch")
		return false
	}
	if decodeTraceRequested && comp.DecodeTrace == nil {
		writeErr(w, http.StatusBadGateway, "fak-native planner did not produce a decode trace")
		return false
	}
	if decodeTokenIDsRequested && comp.NativeDecodeTokenIDs == nil {
		writeErr(w, http.StatusBadGateway, "fak-native planner did not produce native decode token IDs")
		return false
	}
	return true
}

// applyNativeBatchReceiptHeaders emits the additive native-cohort headers from
// the coordinator's own batch receipt (#13317). The set is atomic: it is
// written only when the cohort id is non-empty and header-safe, the cohort
// size is positive, and the shared-panel count is non-negative. An absent or
// malformed receipt writes NO header, so a downstream benchmark always sees
// either the complete evidence triple or "unknown" — never a partial set that
// requested concurrency could be mistaken for. SharedPanels is written even
// when zero: a present 0 is authoritative serial/no-shared-panel evidence,
// distinct from an absent header. CohortSize is coordinator membership; this
// function never labels it a GPU batch width.
func applyNativeBatchReceiptHeaders(w http.ResponseWriter, receipt *agent.InKernelBatchReceipt) {
	if w == nil || receipt == nil {
		return
	}
	cohortID := strconv.FormatUint(receipt.CohortID, 10)
	if receipt.CohortID == 0 || !headerSafeValue(cohortID) {
		return
	}
	if receipt.CohortSize <= 0 || receipt.SharedPanels < 0 {
		return
	}
	h := w.Header()
	h.Set(HeaderNativeCohortID, cohortID)
	h.Set(HeaderNativeCohortSize, strconv.Itoa(receipt.CohortSize))
	h.Set(HeaderNativeSharedPanels, strconv.Itoa(receipt.SharedPanels))
}

// headerSafeValue reports whether a value is safe to place in a response
// header: non-empty, ASCII printable, and free of control characters. The
// coordinator's cohort id is numeric today, so this is a fail-closed guard
// against a future non-numeric or injected value reaching the wire.
func headerSafeValue(v string) bool {
	if v == "" {
		return false
	}
	for i := 0; i < len(v); i++ {
		if c := v[i]; c < 0x20 || c > 0x7e {
			return false
		}
	}
	return true
}

func extractAffinityKey(r *http.Request, explicit string) string {
	if s := strings.TrimSpace(explicit); s != "" {
		return s
	}
	if r != nil {
		if s := strings.TrimSpace(r.Header.Get("X-Session-ID")); s != "" {
			return s
		}
	}
	return ""
}

func (s *Server) buildChatResponse(comp *agent.Completion, asst agent.Message, finish, respModel string, adjs []ToolAdjudication, resultAdmissions []ResultAdmission, inputTriggerRoute *InputTriggerRouteReceipt, decodeTraceRequested, decodeTokenIDsRequested bool, turnCost *turncost.TurnCostRecord) ChatResponse {
	resp := ChatResponse{
		ID:      "chatcmpl-fak-" + itoa(uint64(time.Now().UnixNano())),
		Object:  "chat.completion",
		Created: time.Now().Unix(),
		Model:   respModel,
		Choices: []ChatChoice{{Index: 0, Message: asst, FinishReason: finish}},
		Usage:   comp.Usage,
		Timings: comp.Timings,
	}
	redactions := wireRedactionsFrom(comp.PreSendRedactionRecords)
	if len(adjs) > 0 || len(resultAdmissions) > 0 || len(redactions) > 0 || inputTriggerRoute != nil {
		resp.Fak = &FakExt{Adjudications: adjs, ResultAdmissions: resultAdmissions, Redactions: redactions}
	}
	if inputTriggerRoute != nil {
		resp.Fak.InputTriggerRoute = inputTriggerRoute
	}
	if comp.NativeInference != nil {
		if s.nativeReceiptMetrics != nil {
			s.nativeReceiptMetrics.Observe(comp.NativeInference, time.Now())
		}
		if resp.Fak == nil {
			resp.Fak = &FakExt{}
		}
		resp.Fak.NativeInferenceReceipt = comp.NativeInference
		// The per-turn cost record rides the native receipt (#924): opt-in, so an
		// ordinary turn's response bytes are unchanged whether the surface is on or off.
		if turncost.Enabled() && turnCost != nil {
			resp.Fak.TurnCost = turnCost
		}
	}
	if decodeTraceRequested {
		if resp.Fak == nil {
			resp.Fak = &FakExt{}
		}
		resp.Fak.DecodeTrace = comp.DecodeTrace
	}
	if decodeTokenIDsRequested {
		if resp.Fak == nil {
			resp.Fak = &FakExt{}
		}
		resp.Fak.NativeDecodeTokenIDs = comp.NativeDecodeTokenIDs
	}
	if receipt := nativeBeamWireReceipt(comp.NativeBeamSearch); receipt != nil {
		if resp.Fak == nil {
			resp.Fak = &FakExt{}
		}
		resp.Fak.BeamSearch = receipt
	}
	return resp
}

func (s *Server) chatDecodeTraceSupported(reqModel string) bool {
	p := s.planner
	if dual, ok := p.(*DualPlanner); ok {
		if !dual.RoutesLocal(reqModel) {
			return false
		}
		p = dual.Local()
	}
	capable, ok := p.(agent.NativeDecodeTracePlanner)
	return ok && capable.NativeDecodeTraceSupported()
}

func (s *Server) responseModel(served, requested, streamModel, issue string) string {
	if served == "" {
		served = requested
	}
	if streamModel != "" && served != streamModel {
		s.logf("gateway: streamed turn announced model %q in its preamble; upstream served %q (constant-model SSE, %s)", streamModel, served, issue)
		return streamModel
	}
	return served
}

func chatStreamModel(stream *chatStreamWriter) string {
	if stream == nil {
		return ""
	}
	return stream.model
}

// applyAdjudicatedTurn folds one adjudicated proposal set into the assistant
// message the OpenAI wire will carry — refused-body blanking, vDSO served-inline
// text, the livelock advisory note — and returns the finish_reason the choice
// reports: "tool_calls" while calls survive, "stop" when every proposal was
// refused or served inline (with the deny summary in-band when the content is
// otherwise empty), else the upstream's own finish.
func (s *Server) applyAdjudicatedTurn(asst *agent.Message, adjs []ToolAdjudication, kept []agent.ToolCall, dropped, servedHits int, servedText string, bodyRefused bool, upstreamFinish string) string {
	if len(kept) == 0 {
		if asst.ToolCalls != nil && len(asst.ToolCalls) == 0 {
			kept = []agent.ToolCall{}
		} else {
			kept = nil
		}
	}
	asst.ToolCalls = kept
	// #3567 output-side shadow: classify the MODEL's own outbound prose (sampled,
	// observe-only) before fak blanks/appends anything, folding the negative-framing
	// finding count into fak_negframe_output_negatives_total. Pure telemetry.
	outputNegframeAudit.observe(asst.Content)
	if bodyRefused {
		asst.Content = ""
	}
	// vDSO served-inline (vDSO live in the hot path): a re-proposed read-only call the
	// vDSO already holds fresh is answered LOCALLY and folded into the assistant content
	// (the OpenAI assistant message carries tool_calls, never results), the call dropped
	// from kept so the client never re-runs it — the engine round-trip is saved. The fold
	// keys on servedText, not servedHits: the wire-affordance gate's (#2498) declined
	// restore rides the served slice as prose with servedHits flat, and its correction
	// must still reach the model.
	if servedText != "" {
		if asst.Content != "" {
			asst.Content += "\n" + servedText
		} else {
			asst.Content = servedText
		}
	}
	// Strictly the served-inline hit count: a declined restore's correction is prose,
	// not a served tool result, and must not inflate cache_attribution.fak_inline_served_calls.
	if servedHits > 0 {
		s.metrics.recordServedInline(servedHits)
	}
	// A refused OpenAI call can coexist with model prose. Preserve that prose,
	// but make the refusal visible to clients that ignore the fak extension;
	// otherwise a no-tool stop looks like completed work to their harness.
	if anyLivelock(adjs) || (dropped > 0 && len(kept) == 0 && asst.Content != "") {
		asst.Content = prependAdjudicationContentNote(asst.Content, adjs)
	}
	finish := upstreamFinish
	if len(kept) > 0 {
		finish = "tool_calls"
	} else if dropped > 0 || servedHits > 0 || servedText != "" {
		// Every proposed call was refused OR served inline. A pure-served turn has
		// dropped==0, so broaden the guard: the turn ends (no surviving tool_calls
		// array for the client to act on). servedText is in the guard for the same
		// reason: a declined restore (#2498) is a zero-hit, zero-drop turn whose only
		// content is the correction, and it must still collapse to stop rather than
		// report the upstream's tool_calls finish over an empty tool_calls array.
		// On a deny, surface the reason in-band.
		finish = "stop"
		if asst.Content == "" {
			asst.Content = denySummary(adjs)
		}
	}
	return finish
}

// validateSampling enforces the OpenAI sampling-param contract on an inbound chat
// request, returning a client-facing 400 message for the first invalid field (or ""
// when every present field is in range). It catches the unambiguous wire-contract
// violations the proxy otherwise forwarded verbatim — a negative max_tokens, a
// temperature outside [0, 2], a top_p outside [0, 1] — so bad client input surfaces
// as a 400 instead of the model silently answering it (#326).
//
// max_tokens == 0 is deliberately NOT rejected. The wire field is an omitempty int,
// so an explicit "max_tokens":0 and an omitted field both decode to Go 0 and are
// indistinguishable here; 0 therefore falls through to the planner default (the
// documented semantics). Only values that cannot be a zero-value default — negatives
// and out-of-band floats — are caught, which keeps the check free of false positives
// on a client that simply omitted a field.
func validateSampling(req ChatRequest) string {
	if req.MaxTokens < 0 {
		return "max_tokens: must be a positive integer"
	}
	if req.SpeculativeNMax != nil && *req.SpeculativeNMax < 0 {
		return "speculative.n_max: must be a non-negative integer"
	}
	return validateSamplingRanges(req.Temperature, req.TopP)
}

// validateSamplingRanges enforces the temperature/top_p range contract shared by
// every inbound wire (chat, completions, responses): a present temperature must be
// in [0, 2] and a present top_p in [0, 1]. A nil pointer (the field was omitted) is
// always valid. Returns the first client-facing 400 message, or "" when both are in
// range. Each wire keeps its own max-tokens check inline because the field name
// differs (max_tokens vs max_output_tokens).
func validateSamplingRanges(temperature, topP *float64) string {
	if temperature != nil && (*temperature < 0 || *temperature > 2) {
		return "temperature: must be in [0, 2]"
	}
	if topP != nil && (*topP < 0 || *topP > 1) {
		return "top_p: must be in [0, 1]"
	}
	return ""
}

// maxAbsSamplingPenalty is the OpenAI and vLLM bound on frequency_penalty and
// presence_penalty.
const maxAbsSamplingPenalty = 2.0

const codeSamplingParamOutOfRange = "sampling_param_out_of_range"

// validateSamplingPenalties returns the 400 message for a present
// frequency_penalty or presence_penalty outside [-2, 2], or "" when both are in
// range. An unbounded penalty pushes a logit to +-Inf in the native sampler,
// which would surface a client error as a model fault.
func validateSamplingPenalties(frequency, presence *float64) string {
	if !samplingPenaltyInRange(frequency) {
		return "frequency_penalty: must be in [-2, 2]"
	}
	if !samplingPenaltyInRange(presence) {
		return "presence_penalty: must be in [-2, 2]"
	}
	return ""
}

func samplingPenaltyInRange(p *float64) bool {
	return p == nil || (*p >= -maxAbsSamplingPenalty && *p <= maxAbsSamplingPenalty)
}

// segmentContent splits assistant content into incremental streaming fragments at
// word boundaries (each fragment keeps its trailing space), so concatenating the
// fragments in order reproduces the content byte-for-byte. Empty content yields no
// fragments: a pure tool-call turn streams no content delta, matching OpenAI, which
// emits a content delta only when there is content to deliver.
func segmentContent(content string) []string {
	if content == "" {
		return nil
	}
	segs := strings.SplitAfter(content, " ")
	out := segs[:0]
	for _, s := range segs {
		if s != "" {
			out = append(out, s)
		}
	}
	return out
}

func streamToolCalls(calls []agent.ToolCall) []ChatDeltaToolCall {
	if len(calls) == 0 {
		return nil
	}
	out := make([]ChatDeltaToolCall, 0, len(calls))
	for i, tc := range calls {
		out = append(out, ChatDeltaToolCall{
			Index:    i,
			ID:       tc.ID,
			Type:     tc.Type,
			Function: tc.Function,
		})
	}
	return out
}

func writeSSEData(w http.ResponseWriter, v any) error {
	raw, err := json.Marshal(v)
	if err != nil {
		return err
	}
	if _, err := fmt.Fprintf(w, "data: %s\n\n", raw); err != nil {
		return err
	}
	if f, ok := w.(http.Flusher); ok {
		f.Flush()
	}
	return nil
}
