package gateway

// native_serve.go — the native-harness keystone (#1316/#1837). When
// `fak serve --native` is set, /v1/messages turns are driven by fak's OWN agent loop
// (agent.RunArm / RunArmStream) instead of the single-shot proxy turn at gateway.go's
// complete(). This is the FIRST live, non-test serve-path caller of RunArm +
// WithSessionGate + WithRouteManifest + the operator steer bus — the options that, per
// the program survey, were fully built and tested but had zero live callers.
//
// The thesis (docs/notes/native-harness-progress-tracking-1315.md): on the proxy path
// the external harness (Claude Code, codex) owns the turn loop and consumes tool calls
// outside fak. The native loop is fak OWNING dispatch: fak's loop drives the turns and
// the in-kernel syscall boundary is the only tool path. This handler is that ownership,
// reachable from the wire.
//
// Scope of THIS child (honest fence): the loop drives a served turn to a final answer —
// the AgentDojo-shaped run the program's definition-of-done names ("an AgentDojo run
// driven entirely by fak serve --native"). The operator console and full session control
// remain a tracked follow-on (#1320/#1321).
//
// #6657 closed the wire seam this file used to narrow: the loop was seeded with
// lastUserText(req.Messages) alone and req.Tools was never read, so a served turn ran the
// built-in fixture catalog against a one-line reconstruction of the conversation. Both
// entry points now run ONE conversion (native_wire.go's newNativeWireSeed) that carries
// the ordered transcript and the request-scoped catalog across the seam, and refuse a
// declaration they cannot honor with a typed 400 before the loop runs.

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/adjudicator"
	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/codetools"
	"github.com/anthony-chaudhary/fak/internal/grammar"
	"github.com/anthony-chaudhary/fak/internal/promptaudit"
	"github.com/anthony-chaudhary/fak/internal/sessionledger"
	"github.com/anthony-chaudhary/fak/internal/vdso"
)

var openNativeModelRequestLedger = sessionledger.OpenDefault

// nativeMaxTurnsOr resolves the configured native loop turn cap, defaulting a
// non-positive value to DefaultNativeMaxTurns.
func nativeMaxTurnsOr(n int) int {
	if n <= 0 {
		return DefaultNativeMaxTurns
	}
	return n
}

// serveNativeMessages handles a buffered /v1/messages turn by driving fak's owned
// agent loop and rendering its final answer (plus the per-turn ArmMetrics witness) back
// on the Anthropic wire. It is the native counterpart to completeAnthropicTurn.
func (s *Server) serveNativeMessages(w http.ResponseWriter, r *http.Request, req *agent.AnthropicMessagesRequest, reqTrace string) {
	began := time.Now()
	// Convert the served request BEFORE the loop runs (P3 fail-closed): a tools[]
	// declaration this seam cannot honor is refused with its closed reason token, so the
	// caller learns its catalog was rejected instead of watching a turn run against a
	// catalog it never declared.
	seed, wireErr := newNativeWireSeed(req)
	if wireErr != nil {
		s.logf("gateway: native serve refused request (trace %s): %v", reqTrace, wireErr)
		writeNativeWireErr(w, wireErr)
		return
	}
	m, err := s.runNativeArmSeed(r.Context(), seed, reqTrace)
	if err != nil {
		// An owned-loop failure is classified like any served turn error: a device OOM
		// becomes an actionable 503, a genuine model failure stays a 502 with the raw
		// provider body kept off the wire.
		s.logf("gateway: native serve loop error (trace %s): %v", reqTrace, err)
		s.writeUpstreamErr(w, err)
		return
	}

	// Render the loop's final answer as the assistant turn. The kernel already mediated
	// every tool call INSIDE the loop (vDSO-served, adjudicated, quarantined as it
	// decided), so there are no proposed-call adjudications to fold here — the owned loop
	// consumed them itself. That is exactly the "fak owns dispatch" distinction.
	asst := agent.Message{Role: agent.RoleAssistant, Content: m.FinalAnswer}
	blocks := agent.AnthropicResponseBlocks(asst)
	// A session boundary that ended the loop early (PAUSED/DRAINING/STOPPED/budget) is a
	// clean stop, not a model end-of-turn; the closed reason rides the ArmMetrics witness.
	stop := agent.AnthropicStopReason(nativeFinishReason(m), false)
	usage := anthropicUsage{InputTokens: m.PromptTokens, OutputTokens: m.CompletionTokens}

	s.logInferenceTurn(reqTrace, "anthropic_messages_native", false, agent.Usage{
		PromptTokens:     m.PromptTokens,
		CompletionTokens: m.CompletionTokens,
	}, stop, time.Since(began), false)

	arm := m // copy so the response holds a stable address, not a loop-local
	writeJSON(w, http.StatusOK, anthropicMessageResponse{
		ID:           "msg_fak_" + itoa(uint64(began.UnixNano())),
		Type:         "message",
		Role:         "assistant",
		Model:        s.modelOr(req.Model),
		Content:      blocks,
		StopReason:   stop,
		StopSequence: nil,
		Usage:        usage,
		Fak:          &FakExt{NativeArm: &arm},
	})
}

func (s *Server) serveNativeMessagesStream(w http.ResponseWriter, r *http.Request, req *agent.AnthropicMessagesRequest, reqTrace string) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		s.serveNativeMessages(w, r, req, reqTrace)
		return
	}
	cursorRaw := r.Header.Get("Last-Event-ID")
	if cursorRaw == "" {
		cursorRaw = r.URL.Query().Get("since")
	}
	var replay []agent.ProgressEvent
	if cursorRaw != "" {
		cursor, err := nativeProgressCursor(cursorRaw)
		if err != nil {
			writeErrCode(w, http.StatusBadRequest, "PROGRESS_CURSOR_INVALID", "progress cursor must be an unsigned integer")
			return
		}
		replay, err = s.nativeProgress.after(reqTrace, cursor)
		if err != nil {
			writeErrCode(w, http.StatusConflict, "PROGRESS_CURSOR_TOO_OLD", err.Error())
			return
		}
	}
	sp, ok := s.planner.(agent.StreamingPlanner)
	if cursorRaw == "" && (!ok || !sp.StreamingSupported()) {
		s.serveNativeMessages(w, r, req, reqTrace)
		return
	}

	// The SAME conversion the buffered handler runs, and — critically — run BEFORE the
	// 200 and the SSE headers go out. A refusal discovered after the stream opened could
	// only be an error frame inside a successful response; refusing here keeps it a real
	// 400 the client can act on. The two fallbacks above delegate to the buffered handler,
	// which does its own conversion, so no request reaches the loop unconverted.
	seed, wireErr := newNativeWireSeed(req)
	if wireErr != nil {
		s.logf("gateway: native stream refused request (trace %s): %v", reqTrace, wireErr)
		writeNativeWireErr(w, wireErr)
		return
	}

	began := time.Now()
	id := "msg_fak_" + itoa(uint64(began.UnixNano()))
	h := w.Header()
	h.Set("Content-Type", "text/event-stream")
	h.Set("Cache-Control", "no-cache")
	h.Set("Connection", "keep-alive")
	h.Set("X-Accel-Buffering", "no")
	w.WriteHeader(http.StatusOK)
	send := anthropicSSESender(w, flusher)
	sendProgress := func(ev agent.ProgressEvent) {
		payload := map[string]any{"type": string(ev.Kind), "session": reqTrace, "turn": ev.Turn, "seq": ev.Seq}
		for k, v := range map[string]string{"call_id": ev.CallID, "tool": ev.Tool, "verdict": ev.Verdict, "reason": ev.Reason, "taint": ev.Taint, "summary": ev.Summary} {
			if v != "" {
				payload[k] = v
			}
		}
		b, _ := json.Marshal(payload)
		_, _ = fmt.Fprintf(w, "id: %d\nevent: %s\ndata: %s\n\n", ev.Seq, ev.Kind, b)
		flusher.Flush()
	}
	if cursorRaw != "" {
		for _, ev := range replay {
			sendProgress(ev)
		}
		return
	}
	send("message_start", map[string]any{
		"type": "message_start",
		"message": map[string]any{
			"id": id, "type": "message", "role": "assistant", "model": s.modelOr(req.Model),
			"content": []any{}, "stop_reason": nil, "stop_sequence": nil,
			"usage": map[string]int{"input_tokens": agent.EstimateAnthropicTokens(req), "output_tokens": 0},
		},
	})

	outIdx := 0
	stopBuf := NewStopHoldbackBuffer(req.StopSequences)
	textOpen := false
	textIdx := -1
	closeText := func() {
		if !textOpen {
			return
		}
		if tail := stopBuf.Flush(); tail != "" {
			send("content_block_delta", map[string]any{
				"type": "content_block_delta", "index": textIdx,
				"delta": map[string]any{"type": "text_delta", "text": tail},
			})
		}
		send("content_block_stop", map[string]any{"type": "content_block_stop", "index": textIdx})
		textOpen = false
		textIdx = -1
	}
	emitText := func(text string) error {
		safe := stopBuf.Append(text)
		if safe == "" {
			return nil
		}
		if !textOpen {
			textIdx = outIdx
			outIdx++
			textOpen = true
			send("content_block_start", map[string]any{
				"type": "content_block_start", "index": textIdx,
				"content_block": map[string]any{"type": "text", "text": ""},
			})
		}
		send("content_block_delta", map[string]any{
			"type": "content_block_delta", "index": textIdx,
			"delta": map[string]any{"type": "text_delta", "text": safe},
		})
		return nil
	}

	// Typed loop-progress → structured native SSE (#5148). Each witnessed lifecycle
	// transition of the OWNED loop becomes a custom SSE event named for its kind,
	// interleaved with the text deltas above and tagged with this request's session
	// trace + the originating call id. The observer runs on the same request goroutine
	// as the text sink (the loop is synchronous within CompleteStream), so no
	// cross-goroutine send race is introduced.
	onProgress := func(ev agent.ProgressEvent) {
		s.nativeProgress.append(reqTrace, ev)
		sendProgress(ev)
	}
	m, err := s.runNativeArmStreamSeed(r.Context(), seed, reqTrace, emitText, onProgress)
	if err != nil {
		s.logf("gateway: native stream loop error (trace %s): %v", reqTrace, err)
		closeText()
		send("error", nativeTerminationError(err))
		return
	}
	closeText()

	stop := agent.AnthropicStopReason(nativeFinishReason(m), false)
	usage := anthropicUsage{InputTokens: m.PromptTokens, OutputTokens: m.CompletionTokens}
	s.logInferenceTurn(reqTrace, "anthropic_messages_native", false, agent.Usage{
		PromptTokens:     m.PromptTokens,
		CompletionTokens: m.CompletionTokens,
	}, stop, time.Since(began), false)

	arm := m
	sendAnthropicTerminalWithNativeArm(send, stop, usage, &arm)
}

// runNativeArm drives agent.RunArm(fak=true) for one served request, wiring the
// already-built-but-uncalled loop options to live serve-path sources:
//
//   - WithSessionGate: the SAME injected DecideSession/DebitSession hooks the proxy
//     request boundary uses (serveSessions in cmd/fak), so the owned loop gates each turn
//     boundary on the live drive state — run-state, budget, pace — and reports usage back.
//     Wiring the trace also arms drainSteer, so an operator POST .../steer is folded into
//     the next turn of THIS loop (the consumer half that had no live caller).
//   - WithRouteManifest: the live, hot-reloadable routing policy (s.route), so a per-call
//     model route is bound before each in-loop k.Syscall, exactly as the proxy path does.
//   - WithRouteAccounts + WithRoutePrincipal: the model-ACCOUNT roster (s.roster) and the
//     request's isolation principal, so the routed id the manifest PICKs is resolved to
//     the account-bound Target.EngineRoute() and gated on the caller's tenancy — the same
//     resolveRoute the proxy path runs, rather than a manifest-only route (#5644).
//
// The loop is seeded with the request's ordered conversation and its request-scoped tool
// catalog (#6657); a request that declares no tools still drives the kernel-owned
// agent.ToolCatalog(). It returns the per-turn ArmMetrics — the witness that the loop,
// not an external harness, drove the turn.
//
// This is the conversion-owning entry point for callers that hold only a request (the
// route-parity and stop-gate tests, and any future non-HTTP caller): it fails closed on a
// declaration the seam cannot honor rather than running the loop on a partial catalog.
// The HTTP handlers convert once themselves so they can render that refusal as a typed
// 400, and call runNativeArmSeed below with the seed they already hold.
func (s *Server) runNativeArm(ctx context.Context, req *agent.AnthropicMessagesRequest, reqTrace string) (agent.ArmMetrics, error) {
	return withNativeWireSeed(req, func(seed nativeWireSeed) (agent.ArmMetrics, error) {
		return s.runNativeArmSeed(ctx, seed, reqTrace)
	})
}

// runNativeArmSeed drives the owned loop for one ALREADY-CONVERTED request. Splitting the
// conversion out is what lets the wire handlers refuse before they commit a status code
// while both entry points still share exactly one conversion.
func (s *Server) runNativeArmSeed(ctx context.Context, seed nativeWireSeed, reqTrace string) (agent.ArmMetrics, error) {
	ensureGovernedRungs()
	opts, release := s.nativeRunOptions(ctx, reqTrace)
	defer release()
	opts = append(opts, s.nativeSeedOptions(seed)...)
	return agent.RunArm(ctx, s.planner, seed.Task, true, s.nativeMaxTurns, nil, opts...)
}

// runNativeArmStream drives the owned loop for one STREAMED request. onProgress (may be
// nil) receives the loop's typed lifecycle transitions (#5148) so the caller can render
// them as structured SSE alongside the text deltas; a nil observer leaves the loop
// byte-for-byte the historical one.
// It converts the request itself and fails closed on a declaration the seam cannot honor,
// for the same reason runNativeArm does; the SSE handler converts up front instead so its
// refusal is a 400 rather than an error frame, and calls runNativeArmStreamSeed.
func (s *Server) runNativeArmStream(ctx context.Context, req *agent.AnthropicMessagesRequest, reqTrace string, sink agent.StreamSink, onProgress agent.ProgressObserver) (agent.ArmMetrics, error) {
	return withNativeWireSeed(req, func(seed nativeWireSeed) (agent.ArmMetrics, error) {
		return s.runNativeArmStreamSeed(ctx, seed, reqTrace, sink, onProgress)
	})
}

func withNativeWireSeed(req *agent.AnthropicMessagesRequest, run func(nativeWireSeed) (agent.ArmMetrics, error)) (agent.ArmMetrics, error) {
	seed, wireErr := newNativeWireSeed(req)
	if wireErr != nil {
		return agent.ArmMetrics{}, wireErr
	}
	return run(seed)
}

// runNativeArmStreamSeed is the streamed twin of runNativeArmSeed: one already-converted
// request, driven with the same wired conversation and request-scoped catalog.
func nativeTerminationError(err error) map[string]any {
	termination := agent.ClassifyTermination(err)
	return map[string]any{
		"type":  "error",
		"error": map[string]any{"type": "terminated_run", "message": "run terminated", "cause": termination.Cause, "evidence": termination.Evidence},
	}
}

func (s *Server) runNativeArmStreamSeed(ctx context.Context, seed nativeWireSeed, reqTrace string, sink agent.StreamSink, onProgress agent.ProgressObserver) (agent.ArmMetrics, error) {
	ensureGovernedRungs()
	opts, release := s.nativeRunOptions(ctx, reqTrace)
	defer release()
	opts = append(opts, s.nativeSeedOptions(seed)...)
	if onProgress != nil {
		opts = append(opts, agent.WithProgressObserver(onProgress))
	}
	if s.stopGate != nil {
		// A rejected final answer must never leak as an SSE delta. Buffer stop-gated
		// turns through the non-streaming planner path and emit only the final answer
		// whose declared witness passed.
		m, err := agent.RunArm(ctx, s.planner, seed.Task, true, s.nativeMaxTurns, nil, opts...)
		if err == nil && m.FinalAnswer != "" && sink != nil {
			err = sink(m.FinalAnswer)
		}
		return m, err
	}
	return agent.RunArmStream(ctx, s.planner, seed.Task, true, s.nativeMaxTurns, sink, nil, opts...)
}

func (s *Server) nativeSeedOptions(seed nativeWireSeed) []agent.RunOption {
	opts := make([]agent.RunOption, 0, 3)
	if len(seed.Conversation) > 0 {
		opts = append(opts, agent.WithConversation(seed.Conversation))
	}
	catalog := append([]agent.ToolDef(nil), seed.Tools...)
	seen := map[string]bool{}
	for _, d := range catalog {
		seen[d.Function.Name] = true
	}
	for _, d := range s.nativeCodeCatalog {
		if !seen[d.Function.Name] {
			catalog = append(catalog, d)
		}
	}
	if len(catalog) > 0 {
		opts = append(opts, agent.WithToolCatalog(catalog))
	}
	if len(s.nativeCodeCatalog) == 0 || !s.nativeSpeculate {
		return opts
	}
	spec := abi.NewSpeculator(0)
	spec.Learn(abi.SpecPattern{
		Signature: codetools.ToolRead, PredictTool: codetools.ToolGlob, SuccessProb: 1,
		Meta: codetools.CallMeta(codetools.ToolGlob, ""),
		DeriveArgs: func([]*abi.Result) (abi.Ref, bool) {
			b := []byte(`{"pattern":"**/*.go"}`)
			return abi.Ref{Kind: abi.RefInline, Inline: b, Len: int64(len(b))}, true
		},
	})
	return append(opts, agent.WithSpeculator(spec))
}

// ensureAgentPolicyRung guarantees the agent policy adjudicator is present in the
// kernel's adjudication chain before the owned loop drives a turn. agent.Configure()
// (which RunArm calls at the start of every run) only SetPolicy's the process-global
// adjudicator.Default INSTANCE; the RUNG itself is placed in the abi chain by
// adjudicator's package init, which runs once. On the live serve path that init has
// already registered the rung, so the loop below finds it and this is a no-op. In a
// test binary a sibling's abi.ResetForTest wipes every registered rung and Configure
// does NOT restore the registration — so without this a policy-denied call folds
// through the emptied chain to the fail-closed DEFAULT_DENY instead of the policy's
// POLICY_BLOCK, and every tool would deny, starving the owned loop of a final answer.
// Re-registering here at the rung's canonical rank (100, matching adjudicator.init)
// self-heals a reset-wiped chain; the presence check keeps it idempotent.
func ensureAgentPolicyRung() {
	for _, a := range abi.Adjudicators() {
		if a == abi.Adjudicator(adjudicator.Default) {
			return
		}
	}
	abi.RegisterAdjudicator(100, adjudicator.Default)
}

// ensureGovernedRungs restores the whole driver set the owned loop is measured
// against, not just the policy rung. The same reset that strips the rank-100
// monitor also strips grammar's rank-5 rung and vDSO's fast paths — and those two
// fail QUIETLY where a missing monitor fails loudly: with grammar gone an
// alias-shaped call is no longer repaired in-syscall (it just errors and burns a
// turn, Repairs stays 0), and with the fast paths gone a duplicate read-only call
// re-dispatches to the engine (VDSOHits stays 0). The loop still produces an
// answer, so nothing looks broken; only the kernel's own counters — the numbers a
// served session reports as its witness — silently read zero. Healing all three
// together keeps "the endpoint ran under the real kernel" true of the metrics and
// not merely of the verdicts.
func ensureGovernedRungs() {
	ensureAgentPolicyRung()
	ensureGrammarRung()
	vdso.EnsureRegistered()
}

// ensureGrammarRung restores grammar's rank-5 rung at its canonical rank, mirroring
// grammar's package init. Presence-checked so repeat calls stay idempotent.
func ensureGrammarRung() {
	for _, a := range abi.Adjudicators() {
		if a == abi.Adjudicator(grammar.Default) {
			return
		}
	}
	abi.RegisterAdjudicator(5, grammar.Default)
}

// The returned release closes this run's mid-flight mailbox and MUST be deferred by the
// caller: the registry entry is what a control-plane verb looks the live run up by, so
// leaving it behind would let a later verb address a finished run.
func (s *Server) nativeRunOptions(ctx context.Context, reqTrace string) ([]agent.RunOption, func()) {
	opts := make([]agent.RunOption, 0, 6)
	// #2403 write half: open this run's mid-flight mailbox under the request trace — the
	// same id the typed progress events carry — so POST /v1/fak/session/{trace}/{interrupt|
	// drop-pending-call|set-budget} reaches THIS live run and lands at its next clean turn
	// boundary. Wired here rather than per-entrypoint so the buffered and streamed native
	// /v1/messages turns and the agent-sessions run are all addressable by the same verbs.
	// An empty trace registers nothing and yields a nil mailbox, which WithMidflightVerbs
	// accepts as the historical (mailbox-free) loop.
	mfOpt, release := s.midflightRunOption(reqTrace)
	opts = append(opts, mfOpt)
	opts = append(opts, agent.WithInputClaimLifecycle(agent.InputClaimLifecycle{
		Claim: func(claim agent.AdmittedInputClaim) (agent.InputClaimBinding, error) {
			return appendNativeInputClaim(reqTrace, claim)
		},
		Release: func(binding agent.InputClaimBinding, reason string) error {
			return releaseNativeInputClaim(reqTrace, binding, reason)
		},
	}))
	// The receipt observes the exact planned message slice immediately before the
	// Planner call. Persistence is synchronous and fail-closed: a native request
	// never reaches the model when its durable reconstruction witness did not land.
	opts = append(opts, agent.WithModelRequestObserver(func(boundary agent.ModelRequestBoundary) error {
		return appendNativeModelRequest(reqTrace, boundary)
	}))
	opts = append(opts, agent.WithInterruptedTurnObserver(func(interrupted agent.InterruptedTurn) error {
		return appendNativeInterruptedTurn(reqTrace, interrupted)
	}))
	if s.decideSession != nil {
		opts = append(opts, agent.WithSessionGate(agent.SessionGate{
			Decide: func(trace string) (int, bool, int, string) {
				v := s.decideSession(ctx, trace)
				return v.MaxTokens, v.Proceed, v.MinGapMs, v.Reason
			},
			Debit: func(trace string, out, cx int) {
				if s.debitSession == nil {
					return
				}
				s.debitSession(ctx, trace, SessionUsage{CompletionTokens: out, ContextTokens: cx})
			},
		}, reqTrace))
	}
	if s.stopGate != nil {
		opts = append(opts, agent.WithFinalGate(func() (bool, string) {
			result := s.stopGate(ctx, reqTrace)
			if !result.Satisfied {
				s.recordStopGateHold(reqTrace, result.Witness)
			}
			return result.Satisfied, result.Witness
		}))
	}
	if s.route != nil {
		if mfst := s.route.Manifest(); mfst != nil {
			opts = append(opts, agent.WithRouteManifest(mfst))
		}
	}
	// #5644: hang the roster mirror the manifest option only half-wired. Until now the
	// owned loop resolved a tool call's engine through the MANIFEST alone while the
	// proxying path resolved the same call through the manifest AND the account roster
	// (buildCall -> routeEngine -> resolveRoute -> Target.EngineRoute()), so two serving
	// modes of one binary bound two different routes for the same call and the residency
	// PDP adjudicating a native turn saw a manifest route where the proxy path would have
	// shown it an account-resolved one. Passed unconditionally: a nil roster leaves the
	// abstract routed id verbatim, which is byte-for-byte the pre-#5644 native loop, so a
	// gateway with no --route-accounts is unchanged.
	//
	// The principal rides along because the roster's residency arm needs BOTH halves
	// (#5332): resolveRoute refuses an account whose principals allowlist does not name
	// this caller, so wiring the roster without the principal would make the native path
	// resolve a route the proxy path REFUSES — trading one divergence for a worse one.
	// withAuth already stamped the request principal onto ctx, and an unattributed caller
	// yields "", which Target.Admits fails closed against a restricted account.
	opts = append(opts,
		agent.WithRouteAccounts(s.roster),
		agent.WithRoutePrincipal(principalFromContext(ctx)),
	)
	return opts, release
}

func appendNativeModelRequest(trace string, boundary agent.ModelRequestBoundary) error {
	ledger, err := openNativeModelRequestLedger()
	if err != nil {
		return fmt.Errorf("open session ledger: %w", err)
	}
	segments, err := nativeModelRequestSegments(boundary.Messages, boundary.Injected)
	if err != nil {
		return err
	}
	tools, err := json.Marshal(boundary.Tools)
	if err != nil {
		return fmt.Errorf("marshal model request tools: %w", err)
	}
	request := sessionledger.ModelRequest{
		Identity: sessionledger.ModelRequestIdentity{
			Model: boundary.Model, Turn: boundary.Turn, Stream: boundary.Stream,
			MaxTokens: boundary.MaxTokens,
		},
		Segments: segments,
		Tools:    tools,
	}
	if boundary.InputClaim != nil {
		request.InputClaim = &sessionledger.InputClaimBinding{
			ClaimID: boundary.InputClaim.ID, InputSHA256: boundary.InputClaim.SHA256,
			InputCount: boundary.InputClaim.Count,
		}
	}
	_, err = ledger.AppendModelRequest(trace, request)
	return err
}

func appendNativeInterruptedTurn(trace string, interrupted agent.InterruptedTurn) error {
	ledger, err := openNativeModelRequestLedger()
	if err != nil {
		return fmt.Errorf("open session ledger: %w", err)
	}
	_, err = ledger.AppendInterruptedTurn(trace, sessionledger.InterruptedTurn{
		Turn:   interrupted.Turn,
		Chunks: append([]string(nil), interrupted.Chunks...),
		Reason: sessionledger.TerminalReason{
			Cause: interrupted.Reason.Cause, Evidence: interrupted.Reason.Evidence,
		},
	})
	return err
}

func appendNativeInputClaim(trace string, claim agent.AdmittedInputClaim) (agent.InputClaimBinding, error) {
	ledger, err := openNativeModelRequestLedger()
	if err != nil {
		return agent.InputClaimBinding{}, fmt.Errorf("open session ledger: %w", err)
	}
	inputs := make([]json.RawMessage, 0, len(claim.Inputs))
	for i, input := range claim.Inputs {
		raw, err := json.Marshal(input)
		if err != nil {
			return agent.InputClaimBinding{}, fmt.Errorf("marshal admitted input %d: %w", i, err)
		}
		inputs = append(inputs, raw)
	}
	receipt, err := ledger.AppendInputClaim(trace, sessionledger.InputClaim{Turn: claim.Turn, Inputs: inputs})
	if err != nil {
		return agent.InputClaimBinding{}, err
	}
	return agent.InputClaimBinding{ID: receipt.ClaimID, SHA256: receipt.InputSHA256, Count: receipt.InputCount}, nil
}

func releaseNativeInputClaim(trace string, binding agent.InputClaimBinding, reason string) error {
	ledger, err := openNativeModelRequestLedger()
	if err != nil {
		return fmt.Errorf("open session ledger: %w", err)
	}
	_, current, err := ledger.ReconstructInputClaim(trace, binding.ID)
	if err != nil {
		return err
	}
	if current.InputSHA256 != binding.SHA256 || current.InputCount != binding.Count {
		return errors.New("native input claim binding changed before release")
	}
	_, err = ledger.ReleaseInputClaim(trace, binding.ID, reason)
	return err
}

func nativeModelRequestSegments(messages, injected []agent.Message) ([]sessionledger.ModelRequestSegment, error) {
	injectedJSON := make([][]byte, len(injected))
	for i, message := range injected {
		raw, err := json.Marshal(message)
		if err != nil {
			return nil, fmt.Errorf("marshal injected model request segment %d: %w", i, err)
		}
		injectedJSON[i] = raw
	}
	usedInjected := make([]bool, len(injectedJSON))
	segments := make([]sessionledger.ModelRequestSegment, 0, len(messages))
	for i, message := range messages {
		raw, err := json.Marshal(message)
		if err != nil {
			return nil, fmt.Errorf("marshal model request segment %d: %w", i, err)
		}
		kind, source := nativeModelRequestAttribution(i, message)
		for j, candidate := range injectedJSON {
			if !usedInjected[j] && bytes.Equal(raw, candidate) {
				kind = "injected_directive"
				source = promptaudit.SourceUserConfig
				usedInjected[j] = true
				break
			}
		}
		segments = append(segments, sessionledger.ModelRequestSegment{
			Kind: kind, Source: source, Content: raw,
		})
	}
	return segments, nil
}

func nativeModelRequestAttribution(index int, message agent.Message) (string, promptaudit.Source) {
	switch message.Role {
	case agent.RoleSystem:
		if index == 0 {
			return "system", promptaudit.SourceFakPolicy
		}
		return "system", promptaudit.SourceUserConfig
	case agent.RoleUser:
		return "user_input", promptaudit.SourceUserConfig
	case agent.RoleAssistant:
		return "assistant", promptaudit.SourceUnknown
	case agent.RoleTool:
		return "tool_result", promptaudit.SourceIntegration
	default:
		return "message", promptaudit.SourceUnknown
	}
}

func sendAnthropicTerminalWithNativeArm(send func(string, any), stop string, usage anthropicUsage, arm *agent.ArmMetrics) {
	finalUsage := map[string]int{
		"input_tokens":  usage.InputTokens,
		"output_tokens": usage.OutputTokens,
	}
	send("message_delta", map[string]any{
		"type":  "message_delta",
		"delta": map[string]any{"stop_reason": stop, "stop_sequence": nil},
		"usage": finalUsage,
	})
	send("message_stop", map[string]any{
		"type": "message_stop",
		"fak":  &FakExt{NativeArm: arm},
	})
}

// nativeFinishReason maps the owned loop's outcome to a planner-style finish reason for
// the Anthropic stop-reason projection: a session-boundary stop is reported as "stop"
// (a clean end_turn carrying the reason on the ArmMetrics witness), as is a normal final
// answer; only a turn-cap hit reports "length".
func nativeFinishReason(m agent.ArmMetrics) string {
	if m.HitTurnCap {
		return "length"
	}
	return "stop"
}

// lastUserText returns the content of the last user-role message in the canonical
// transcript — the task the owned loop is seeded with. DecodeAnthropicMessagesRequest
// has already flattened each inbound user content block into a single text Content, so a
// plain scan from the end is sufficient. Returns "" when there is no user message (the
// loop then runs from its system prompt alone).
func lastUserText(messages []agent.Message) string {
	for i := len(messages) - 1; i >= 0; i-- {
		if messages[i].Role == agent.RoleUser {
			return messages[i].Content
		}
	}
	return ""
}
