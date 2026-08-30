package gateway

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"sync/atomic"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/cachemeta"
	"github.com/anthony-chaudhary/fak/internal/guardrsi"
	"github.com/anthony-chaudhary/fak/internal/vdso"
)

// admit runs a CLIENT-PRODUCED tool result through the kernel's result-side stack
// (k.AdmitResult: the context-MMU quarantine + the IFC source-stamp that raises the
// per-trace taint ledger). This is what arms the exfil floor on the served path: a
// client that executes its own tool sends the RESULT back here, and a poisoned
// result is quarantined (paged out) + the session's taint high-water mark is raised
// before the result is admitted to context. The verdict + the admitted (possibly
// paged-out) result are rendered for the wire. It is the explicit fak_admit verb;
// admitOp is the shared core the auto-proxy also drives under its own op label.
func (s *Server) admit(ctx context.Context, tool, rawResult, witness, traceID string) (wv WireVerdict, env *ResultEnvelope, err error) {
	wv, env, err = s.admitOp(ctx, "admit", tool, rawResult, witness, traceID)
	if err != nil {
		return wv, env, err
	}
	// Native-path parity with the proxy (#411). The proxy fires the remote
	// engine-cache reset from admitInboundResults; the native admit routes
	// (POST /v1/fak/admit, the fak_admit MCP tool) quarantined locally but never
	// reset the upstream serving-engine cache, so a poisoned token-sequence could
	// survive in the provider KV/prefix cache when an agent drives fak natively
	// instead of through /v1/chat/completions. resetEngineCacheAfterQuarantine is
	// the SAME reset the proxy fires (a no-op when no engine cache is configured);
	// a remote-reset failure is surfaced fail-closed, wrapped so the HTTP handler
	// maps it to a 502 rather than a client 400.
	if wv.Kind == "QUARANTINE" {
		if rerr := s.resetEngineCacheAfterQuarantine(ctx, []ResultAdmission{{Verdict: wv}}); rerr != nil {
			return wv, env, fmt.Errorf("%w: %v", errEngineCacheReset, rerr)
		}
	}
	return wv, env, nil
}

// errEngineCacheReset marks an admit failure that originated in the REMOTE
// engine-cache reset (not the local admission). handleFakAdmit maps it to a 502 —
// the same fail-closed signal the proxy returns on a reset failure — while a local
// build/resolver error stays a 400 client error.
var errEngineCacheReset = errors.New("engine cache reset failed")

// admitOp is the shared result-side admission core: it builds an agent-scoped call,
// puts the result bytes into a tainted Ref, and folds the kernel's ResultAdmitter
// chain over them (context-MMU quarantine + IFC source-stamp/taint ledger), tagged
// with the caller's op label for metrics/logs. Both the explicit fak_admit verb
// (op "admit") and the auto /v1/chat/completions proxy (op "proxy_admit") route
// through it, so the result-side floor is identical on every served topology.
func (s *Server) admitOp(ctx context.Context, operation, tool, rawResult, witness, traceID string) (wv WireVerdict, env *ResultEnvelope, err error) {
	return s.admitOpWithSeq(ctx, operation, tool, rawResult, witness, traceID, 0)
}

func (s *Server) admitOpWithSeq(ctx context.Context, operation, tool, rawResult, witness, traceID string, callSeq uint64) (wv WireVerdict, env *ResultEnvelope, err error) {
	start := time.Now()
	opTrace, opTool := traceID, tool
	defer func() {
		dur := time.Since(start)
		s.metrics.observeOperation(operation, wv, err, dur)
		s.logGatewayOperation(operation, opTrace, opTool, wv, err, dur)
	}()
	tc, err := s.buildCall(ctx, tool, "", false, witness, traceID)
	if err != nil {
		return WireVerdict{}, nil, err
	}
	if callSeq != 0 {
		tc.SeqNo = callSeq
	}
	opTrace, opTool = tc.TraceID, tc.Tool
	body := []byte(rawResult)
	if len(body) == 0 {
		body = []byte("{}")
	}
	ref, err := abi.ActiveResolver().Put(ctx, body)
	if err != nil {
		return WireVerdict{}, nil, fmt.Errorf("resolver: %w", err)
	}
	r := &abi.Result{Call: tc, Payload: ref, Status: abi.StatusOK, Meta: map[string]string{}}
	v := s.k.AdmitResult(ctx, tc, r)
	env = &ResultEnvelope{
		Status:  statusName(r.Status),
		Content: string(resolveBytes(ctx, r.Payload)),
		Meta:    r.Meta,
	}
	wv = renderVerdict(v, r.Meta)
	return wv, env, nil
}

func (s *Server) rememberOriginSeq(traceID, tool, rawArgs string, seq uint64) {
	if seq == 0 || traceID == "" || tool == "" {
		return
	}
	s.originSeqMu.Lock()
	if s.originSeq == nil {
		s.originSeq = map[string]uint64{}
	}
	if len(s.originSeq) >= maxResetHealthSessions {
		for k := range s.originSeq {
			delete(s.originSeq, k)
			break
		}
	}
	s.originSeq[originSeqKey(traceID, tool, rawArgs)] = seq
	s.originSeqMu.Unlock()
}

const gatewayOriginSeqBase uint64 = 1 << 63

func (s *Server) nextOriginSeq() uint64 {
	return gatewayOriginSeqBase | atomic.AddUint64(&s.originSeqNext, 1)
}

func (s *Server) rememberOriginSeqID(traceID, callID string, seq uint64) {
	if seq == 0 || traceID == "" || callID == "" {
		return
	}
	s.originSeqMu.Lock()
	if s.originSeqByID == nil {
		s.originSeqByID = map[string]uint64{}
	}
	if len(s.originSeqByID) >= maxResetHealthSessions {
		for k := range s.originSeqByID {
			delete(s.originSeqByID, k)
			break
		}
	}
	s.originSeqByID[originSeqIDKey(traceID, callID)] = seq
	s.originSeqMu.Unlock()
}

func (s *Server) originSeqFor(traceID string, call agent.ToolCall) uint64 {
	if traceID == "" {
		return 0
	}
	s.originSeqMu.Lock()
	if call.ID != "" {
		if seq := s.originSeqByID[originSeqIDKey(traceID, call.ID)]; seq != 0 {
			s.originSeqMu.Unlock()
			return seq
		}
	}
	seq := s.originSeq[originSeqKey(traceID, call.Function.Name, call.Function.Arguments)]
	s.originSeqMu.Unlock()
	return seq
}

func originSeqKey(traceID, tool, rawArgs string) string {
	if rawArgs == "" {
		rawArgs = "{}"
	}
	return traceID + "\x00" + tool + "\x00" + rawArgs
}

func originSeqIDKey(traceID, callID string) string {
	return traceID + "\x00id\x00" + callID
}

// admitInboundResults arms the RESULT-side floor on the auto /v1/chat/completions
// proxy (#7). In the OpenAI tool protocol a tool RESULT the client executed comes
// back on the NEXT turn as a role="tool" message; before this, those results flowed
// straight to the upstream model, so the result-side containment (context-MMU
// quarantine + IFC source-stamp/taint ledger + eviction) was inert on the proxy —
// armed only on the in-process Syscall/Reap path and the explicit fak_admit verb.
//
// Each inbound tool result is routed through k.AdmitResult keyed on the per-session
// traceID BEFORE it reaches the model: a poisoned/secret-bearing result is PAGED
// OUT (its forwarded content replaced with the quarantine stub, so the upstream
// model's KV never ingests the poison), and an untrusted-source result RAISES the
// trace's IFC taint high-water mark. That high-water mark is exactly what the
// already-wired sink-gate (k.Decide, keyed on the SAME traceID) reads when it
// adjudicates the calls the model then proposes — so an exfil call on a tainted
// session is refused. messages is mutated in place (request-local). The per-result
// admissions are returned for the fak response extension.
func (s *Server) admitInboundResults(ctx context.Context, messages []agent.Message, tools []agent.ToolDef, traceID string) ([]ResultAdmission, error) {
	// Snapshot each message's ORIGINAL content before admission rewrites any quarantined
	// payload in place. The in-kernel poison-eviction hook needs the original (poisoned)
	// bytes to render the token path that was actually cached, not the paged-out form.
	origContent := make([]string, len(messages))
	for i := range messages {
		origContent[i] = messages[i].Content
	}
	// Pair each inbound tool_result to its originating call's (tool, args): the result
	// block carries only ToolCallID + Content, but the args live on the prior assistant
	// tool_use whose ID == ToolCallID (decoded into Message.ToolCalls). The same index
	// feeds optional vDSO fill and, when a real kernel sequence was recorded, the journal
	// call_seq on result-side quarantines.
	callByID := make(map[string]agent.ToolCall)
	for _, m := range messages {
		if m.Role != agent.RoleAssistant {
			continue
		}
		for _, tcc := range m.ToolCalls {
			if tcc.ID == "" {
				continue
			}
			callByID[tcc.ID] = tcc
		}
	}
	var admissions []ResultAdmission
	var observed []ObservedResult
	var quarantinedIdx []int
	for i := range messages {
		if messages[i].Role != agent.RoleTool {
			continue
		}
		tool := messages[i].Name
		var origin agent.ToolCall
		var hasOrigin bool
		if messages[i].ToolCallID != "" {
			origin, hasOrigin = callByID[messages[i].ToolCallID]
		}
		if tool == "" && hasOrigin {
			tool = origin.Function.Name
		}
		if tool == "" {
			// A nameless tool result is still untrusted cross-boundary input. Admit it
			// under a placeholder so the content screen + fail-closed taint still fire
			// (provenance treats an unregistered tool as Untrusted).
			tool = "tool_result"
		}
		var originSeq uint64
		if hasOrigin {
			originSeq = s.originSeqFor(traceID, origin)
		}
		resultShape, shapeVerdict, shapeContent := resultContractAdmission(tool, tools, messages[i].Content)
		resultDigest := guardrsi.ArgsDigest(messages[i].Content)
		// Screen this result EXACTLY ONCE per (trace, origin call ID, content) (#2417). On first arrival
		// the ledger runs the closure — the real result-side stack — and records the
		// verdict; on a later replay of the same content it returns the recorded verdict
		// without re-screening, so the kernel work, the vDSO fill, the proxy_admit metric,
		// and the eviction/reset below all happen once per unique result, not once per turn.
		rec, fresh := s.admitLedger.admit(traceID, messages[i].ToolCallID, resultDigest, func() (WireVerdict, string, bool) {
			// Exhaustive result contracts are a pre-consumer boundary: a shape mutant
			// is replaced before ordinary admission, cache fill, observers, or the
			// upstream model can consume its values.
			if shapeVerdict != nil {
				return *shapeVerdict, shapeContent, true
			}
			wv, envlp, aerr := s.admitOpWithSeq(ctx, "proxy_admit", tool, messages[i].Content, "", traceID, originSeq)
			if aerr != nil {
				// A result we cannot even admit is held out fail-closed rather than
				// forwarded raw to the model.
				return WireVerdict{Kind: "QUARANTINE", Reason: "ADMIT_ERROR", Disposition: "TERMINAL"},
					`{"_quarantined":true,"boundary":"proxy","reason":"ADMIT_ERROR"}`, true
			}
			// On a quarantine/transform the kernel paged the bytes out and rewrote the
			// payload in place; record the paged-out form so every replay forwards it and
			// the poison never reaches the model. A plain Allow leaves the content untouched.
			content, rewrote := messages[i].Content, false
			if envlp != nil && (wv.Kind == "QUARANTINE" || wv.Kind == "TRANSFORM") {
				content, rewrote = envlp.Content, true
			}
			// Subagent-boundary witness (#2438): a child terminal result whose prose CLAIMS
			// ship/create/fix but carries no artifact witness (commit SHA / file hash) is
			// folded RESIDUAL, not clean — the loop-body-witness discipline
			// (ReasonLoopBodyUnwitnessed) at the subagent fold. Only a plain ALLOW is demoted;
			// the content is still forwarded (the parent sees it) but the admission records it
			// visibly unverified. Placed before the vDSO fill (ALLOW-only) so a demoted, unbacked
			// claim can never warm the cache either.
			if demoted, ok := subagentDoneVerdict(tool, messages[i].Content, wv); ok {
				wv = demoted
			}
			// Warm the vDSO from this ADMITTED result (opt-in, default off): only a plain
			// Allow (never QUARANTINE/TRANSFORM/DENY), paired to its originating read-only
			// call, fills (tool,args)->result so a later identical read is served inline.
			// All the soundness/security guards live in fillVDSOFromResult.
			if messages[i].RefutesWitness != "" {
				s.revokeVDSOWitness(messages[i].RefutesWitness)
			}
			if s.vdsoProxyFill && wv.Kind == "ALLOW" {
				if orig, ok := callByID[messages[i].ToolCallID]; ok {
					s.fillVDSOFromResult(ctx, orig, messages[i].Content, messages[i].Witness, traceID)
				}
			}
			// Deposit the SAME admitted result into the toolproc reuse cache (#5119),
			// the other half of the loop whose serve side sits in
			// adjudicateProposedServed. Same ALLOW-only gate as the vDSO fill above —
			// a quarantined or transformed result never becomes servable bytes — but
			// carried on its own opt-in (SetToolprocReuse), because the two probes key
			// on different things: the vDSO on the tool NAME, this on the command
			// CONTENT. Unarmed, this is one lock-free map probe.
			if wv.Kind == "ALLOW" {
				if orig, ok := callByID[messages[i].ToolCallID]; ok {
					s.reuseOffer(orig.Function.Name, orig.Function.Arguments, messages[i].Content)
				}
			}
			return wv, content, rewrote
		})
		wv := rec.verdict
		if rec.rewrote {
			messages[i].Content = rec.content
		}
		// Only a FIRST-arrival quarantine drives the in-kernel eviction + engine-cache
		// reset below; a replay already paged the bytes out and evicted on its first turn,
		// so re-firing them per replay is exactly the redundant work #2417 removes.
		if wv.Kind == "QUARANTINE" && fresh {
			quarantinedIdx = append(quarantinedIdx, i)
		}
		adm := ResultAdmission{
			ToolCallID:   messages[i].ToolCallID,
			Tool:         tool,
			ResultDigest: resultDigest,
			Verdict:      wv,
			ResultShape:  resultShape,
			fresh:        fresh,
		}
		admissions = append(admissions, adm)
		// Capture a read-only snapshot for the async observer stratum (#2434). messages[i].Content
		// is the SETTLED admitted content at this point (the blocking chain rewrote it in place on a
		// quarantine/transform; unchanged on a plain allow), so an observer sees exactly the bytes
		// the turn forwarded — as a value copy it can neither block nor mutate them.
		observed = append(observed, ObservedResult{
			TraceID:      traceID,
			ToolCallID:   messages[i].ToolCallID,
			Tool:         tool,
			ResultDigest: resultDigest,
			Verdict:      wv.Kind,
			Content:      messages[i].Content,
		})
	}
	s.annotateResultLivelock(traceID, admissions)
	// Both defenses fire on a FIRST-arrival quarantine only (quarantinedIdx holds the fresh
	// ones): a replay was already paged out and evicted on its first turn, so re-firing them
	// per replay is the redundant work admit-once (#2417) removes. evictInKernelPoison already
	// no-ops on an empty index; gate the engine-cache reset the same way so a replayed
	// quarantine verdict in `admissions` does not re-reset the remote cache every turn.
	s.evictInKernelPoison(messages, origContent, quarantinedIdx, tools)
	if len(quarantinedIdx) > 0 {
		if err := s.resetEngineCacheAfterQuarantine(ctx, admissions); err != nil {
			return admissions, err
		}
	}
	// The blocking chain has settled; hand the observer stratum read-only copies async, OFF the
	// turn path (#2434). This is fire-and-forget: a slow or failing observer degrades against its
	// own budget/health, never against this result or this turn.
	s.observers.dispatch(ctx, observed)
	return admissions, nil
}

// fillVDSOFromResult warms the vDSO tier-2 cache from one ADMITTED inbound tool_result
// (the opt-in proxy-fill path) so a LATER re-proposed identical read is served inline
// instead of bounced back to the client. The caller has already confirmed the result's
// admission verdict was a plain Allow; this function applies the remaining soundness +
// security guards that the generic vdso.Emit fill gate (built for fak-authored
// completions) does not enforce against a client-supplied producer:
//
//   - read-only-shaped tool ONLY (readOnlyPrefix); IsWriteShaped is the un-bypassable
//     backstop. A write tool's result must never become a cached "answer".
//   - NAMED principal ONLY: an empty principal lands the entry in the shared global
//     slice, letting one client seed bytes an unrelated tenant reads. A client fill must
//     be attributable to the principal that produced it.
//   - never a Shareable tool: a Shareable entry drops the principal dimension (shared
//     across all tenants), so a client fill into one would be a cross-tenant poison.
//
// On a hit the LATER read serves these bytes; ctxmmu.ScreenBytes on the serve side
// (adjudicateProposedServed) remains the backstop, but a quarantined result never
// reaches here because the caller gates on wv.Kind=="ALLOW". The fill is built via the
// SAME buildCall(readOnly=true) the served probe uses, so the key matches exactly.
func (s *Server) fillVDSOFromResult(ctx context.Context, orig agent.ToolCall, result, witness, traceID string) {
	if strings.TrimSpace(witness) == "" {
		return
	}
	tool := orig.Function.Name
	// Trust the assistant-side tool NAME (the result block drops it on the Anthropic
	// wire). Eligibility mirrors the served probe; IsWriteShaped is the hard backstop.
	if !readOnlyPrefix(tool) || vdso.IsWriteShaped(tool) {
		return
	}
	// A client fill must be principal-attributed (empty principal => shared global slice).
	if principalFromContext(ctx) == "" {
		return
	}
	args := orig.Function.Arguments
	if strings.TrimSpace(args) == "" {
		args = "{}"
	}
	// Build the call the SAME way the served probe does (readOnly=true => readOnlyHint+
	// idempotentHint, principal scoping), so the fill key == the later Lookup key.
	c, err := s.buildCall(ctx, tool, args, true /*readOnly*/, witness, traceID)
	if err != nil {
		return
	}
	ref, err := abi.ActiveResolver().Put(ctx, []byte(result))
	if err != nil {
		return
	}
	// Meta must NOT carry served_by=vdso (vdso.Emit refuses to re-store an already-served
	// entry). Emit ONLY to the registered vDSO observers — NOT every EvComplete emitter,
	// which would feed a phantom completion to the journal/rungobs counters. In production
	// and in tests the wired *vdso.VDSO is the same instance the served probe reads via
	// abi.FastPaths(), so the fill lands where a later Lookup will find it. vdso.Emit's own
	// gates (Status OK, !destructive, both hints, resourceMisnamed) are the final backstop.
	r := &abi.Result{Call: c, Payload: ref, Status: abi.StatusOK, Meta: map[string]string{}}
	ev := abi.Event{Kind: abi.EvComplete, Call: c, Result: r}
	for _, em := range abi.EmittersFor(abi.EvComplete) {
		v, ok := em.(*vdso.VDSO)
		if !ok {
			continue
		}
		// Per-instance Shareable guard: a Shareable entry drops the principal dimension
		// (shared across all tenants), so a client-supplied result must never fill one —
		// that would let one client poison every tenant. Checked on the SAME instance the
		// fill targets (Shareable is registered per-vDSO), not a global default.
		if v.Shareable(tool) {
			continue
		}
		v.Emit(ev)
	}
}

// evictInKernelPoison drives the in-kernel poison eviction when the chat backend is the
// in-kernel planner. It drives TWO complementary seams on the SAME quarantine event, each a
// no-op on a planner that does not implement it (proxy/mock, or the seam left off):
//
//   - agent.PoisonEvictor — drops the reusable RadixAttention PREFIX node along the poisoned
//     path so a later turn re-prefills instead of replaying the poisoned KV (candidate #14).
//   - agent.KVSpanEvictor — the model-side KV-quarantine eviction BRIDGE (issue #579): it
//     rebuilds the transcript's per-message K/V SPANS on a fresh model.Session over the loaded
//     model and evicts the quarantined result's span via the proven model.KVCache.Evict
//     (re-RoPE + renumber), so the live session's attention state is bit-identical to a run
//     that never saw the poison — the flagship guarantee, now fired by a LIVE request and not
//     only the synthetic-model unit witness. DEFAULT OFF (FAK_INKERNEL_KVMMU opts in).
//
// The transcript is rendered with each message's ORIGINAL content AND the request's tool
// schemas (tools) so the evicted token path matches what the cache actually prefilled before
// the verdict paged the bytes out — generation rendered renderChatMLTools(messages, tools)
// with the tool-spec folded into the leading system block, so a tools-less eviction render
// would not be a token-prefix of the cached tool-using turn and would reclaim nothing (#612).
func (s *Server) evictInKernelPoison(messages []agent.Message, origContent []string, quarantinedIdx []int, tools []agent.ToolDef) {
	if len(quarantinedIdx) == 0 {
		return
	}
	prefixEv, hasPrefix := s.planner.(agent.PoisonEvictor)
	spanEv, hasSpan := s.planner.(agent.KVSpanEvictor)
	if !hasPrefix && !hasSpan {
		return
	}
	restored := make([]agent.Message, len(messages))
	copy(restored, messages)
	for i := range restored {
		if i < len(origContent) {
			restored[i].Content = origContent[i]
		}
	}
	for _, idx := range quarantinedIdx {
		if hasPrefix {
			if freed := prefixEv.EvictPoisoned(restored, idx, tools); freed > 0 {
				s.logf("gateway: in-kernel KV prefix evicted on tool-result quarantine msg=%d freed=%dtok", idx, freed)
			}
		}
		if hasSpan {
			// Default-off bridge: a no-op (0,false) unless FAK_INKERNEL_KVMMU opted it in, so the
			// served path is unchanged by default. When on, a non-zero freed span proves the live
			// KVCache.Evict fired; reposition_exact records the bit-exact never-saw invariant.
			if freed, exact := spanEv.EvictKVSpan(restored, idx, tools); freed > 0 {
				s.logf("gateway: in-kernel KV span evicted on tool-result quarantine msg=%d freed=%dpos reposition_exact=%v", idx, freed, exact)
			}
		}
	}
}

func (s *Server) resetEngineCacheAfterQuarantine(ctx context.Context, admissions []ResultAdmission) error {
	if s.engineCache == nil {
		return nil
	}
	for _, a := range admissions {
		if a.Verdict.Kind != "QUARANTINE" {
			continue
		}
		dirs := []cachemeta.ExternalInvalidationDirective{{
			Kind:      cachemeta.ExternalInvalidateKVSpan,
			Plane:     cachemeta.PlaneKVPrefix,
			Residency: cachemeta.Residency{Tier: cachemeta.TierRemote, Owner: string(s.engineCache.Engine)},
			Provider:  string(s.engineCache.Engine),
			Engine:    string(s.engineCache.Engine),
			Reason:    "proxy_tool_result_quarantine",
		}}
		res, err := s.engineCache.Invalidate(ctx, dirs)
		if err != nil {
			s.logf("gateway: engine cache reset failed after quarantined tool result: %v", err)
			return err
		}
		s.logf("gateway: engine cache reset engine=%s scope=%s directives=%d endpoint=%s",
			res.Engine, res.Scope, res.Directives, res.Endpoint)
		return nil
	}
	return nil
}

func (s *Server) revokeVDSOWitness(witness string) {
	witness = strings.TrimSpace(witness)
	if witness == "" {
		return
	}
	for _, em := range abi.EmittersFor(abi.EvComplete) {
		if v, ok := em.(*vdso.VDSO); ok {
			v.Revoke(witness)
		}
	}
}
