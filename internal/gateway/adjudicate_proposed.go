package gateway

import (
	"context"
	"encoding/json"
	"fmt"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/ctxmmu"
	"github.com/anthony-chaudhary/fak/internal/guardrsi"
)

// fastPathLookup probes the registered vDSO fast paths (the same abi.FastPaths()
// registry kernel.Submit consults) for a fresh cached answer, returning the first hit.
// It is the served-turn analogue of the kernel's fast-path loop — Lookup-only, so a
// miss executes nothing — and respects whichever vDSO instance the gateway wired.
func fastPathLookup(ctx context.Context, c *abi.ToolCall) (*abi.Result, bool) {
	for _, fp := range abi.FastPaths() {
		if r, ok := fp.Lookup(ctx, c); ok {
			return r, true
		}
	}
	return nil, false
}

const ReasonLoopBodyUnwitnessed = "LOOP_DONE_UNWITNESSED"

// ReasonLivelockFuse is the refusal reason stamped when the livelock hard fuse
// converts a repeated admitted call into a denial. It is RETRYABLE per-tool feedback
// (the model can make progress by changing approach), never a deny-all session stop.
const ReasonLivelockFuse = "LIVELOCK_FUSE"

// adjudicateProposedServed is the served-turn vDSO fast path (issue: vDSO live in
// the hot path). It is adjudicateProposed plus a vDSO Lookup probe FIRST for every
// read-only-shaped proposed call: on a fresh cache hit the answer is served LOCALLY
// (no engine round-trip, no client re-execution) and folded into servedText, and the
// call is dropped from kept. On a miss it is byte-identical to adjudicateProposed —
// the call falls through to the normal adjudicate -> k.Decide -> return-to-client path.
//
// Why a direct vdso.Default.Lookup and not k.Syscall: Submit would dispatch to an
// engine (which the proxy does not own for a client's arbitrary tools), store a
// pending call, and bump the kernel VDSOHits counter. We want ONLY the fast-path
// probe: Lookup executes nothing on a miss (vdso.go) and a hit equals a fresh call
// (world-versioned + integrity-gated). Served bytes are screened through
// ctxmmu.ScreenBytes before folding, so a poisoned cache entry can never enter the
// model-visible transcript as prose.
//
// The three result buckets are DISJOINT: a call is in exactly one of kept (survives
// to the wire as tool_use), dropped (denied), or servedHits (answered inline). A
// served hit is NOT a surviving tool call, so the caller's stop-reason logic (keyed
// on len(kept)) collapses a fully-served turn to end_turn/stop correctly, and the
// deny-all guard must exclude servedHits (a served turn is a SUCCESS, not a deny).
func (s *Server) adjudicateProposedServed(ctx context.Context, calls []agent.ToolCall, reqTrace string) (kept []agent.ToolCall, adjs []ToolAdjudication, dropped int, servedText string, servedHits int) {
	// PRINCIPAL AUTHORITY FLOOR (#2412): before any adjudication (or vDSO fast-path
	// probe), type-check the turn's authority-consuming acts against the inbound
	// principal. A confirmation token is a consent-shaped act, honored only under the
	// human principal; a peer-agent / timer / network-tool / unknown turn has it
	// stripped here, so the underlying irreversible call falls back to its
	// REQUIRE_WITNESS hold instead of being waved through by a relayed approval. The
	// human principal (the common direct-wire default) is a no-op passthrough.
	if p := s.tracePrincipalOf(reqTrace); !p.IsHuman() {
		calls, _ = gateInboundAuthority(p, calls)
	}
	pass := make([]agent.ToolCall, 0, len(calls))
	var served []string
	for _, tc := range calls {
		tool := tc.Function.Name
		argsDigest := guardrsi.ArgsDigest(tc.Function.Arguments)
		// Force-fresh escape hatch: a re-proposed read carrying the advertised _fak_fresh
		// marker skips the cache probe and passes through to the client to actually run.
		// Sound: this only ever turns a would-be served hit into a normal tool_use —
		// byte-identical to a cache MISS, the already-tested fall-through. It can never
		// create an effect or relax a gate; it only declines the optimization. The model
		// reaches for this when the served age (below) says the cached read is too stale.
		//
		// It sits above BOTH probes so the affordance the served line advertises beats
		// the toolproc reuse cache as well as the vDSO one. Hoisting it past the
		// read-only name gate is behavior-preserving: a marker-carrying call that is not
		// vDSO-eligible takes the same `pass` fall-through that gate would have given it.
		if callRequestsFresh(tc.Function.Arguments) {
			pass = append(pass, tc)
			continue
		}
		// TOOLPROC REUSE PROBE (#5119) — the content-keyed twin of the vDSO name probe
		// below, and the live caller internal/gateway/toolproc_reuse.go was written to
		// have. It keys on the command CONTENT (`read:<path>@<digest>` / `query:<canon>`)
		// rather than the tool NAME, so it reaches exactly the family readOnlyPrefix
		// cannot match — native Read/Bash/Grep, where the name gate serves 0 for a
		// structural reason (served_inline_guardtrace_test.go measures it).
		//
		// It runs BEFORE the vDSO probe because it is the stricter of the two where they
		// overlap (a `read_file`-shaped name): it serves an immutable read only under a
		// content digest witnessed RIGHT NOW, whereas a vDSO tier-2 hit may be stale by
		// up to its max-age ceiling. Unarmed — the default — reuseServe is one lock-free
		// map probe and a nil check, so the pre-#5119 path stays byte-identical.
		if body, meta, ok := s.reuseServe(ctx, tool, tc.Function.Arguments); ok {
			served = append(served, servedToolLine(tool, body, meta))
			servedHits++
			adjs = append(adjs, ToolAdjudication{ToolCallID: tc.ID, Tool: tool, ArgsDigest: argsDigest, Admitted: true,
				Verdict: WireVerdict{Kind: "ALLOW", Reason: "SERVED_INLINE", By: ReuseServedByVerdict}})
			continue
		}
		if isRestoreTool(tool) {
			var req ContextRestoreRequest
			if err := json.Unmarshal([]byte(tc.Function.Arguments), &req); err == nil && req.ID != "" {
				if req.TraceID == "" {
					req.TraceID = reqTrace
				}
				caller, _ := s.traceOwnerOf(reqTrace)
				res, err := s.restoreContext(caller, req)
				if err == nil {
					served = append(served, fmt.Sprintf("[fak: restored context id=%s]\n%s", req.ID, res.Bytes))
					servedHits++
					adjs = append(adjs, ToolAdjudication{
						ToolCallID: tc.ID,
						Tool:       tool,
						ArgsDigest: argsDigest,
						Admitted:   true,
						Verdict:    WireVerdict{Kind: "ALLOW", Reason: "SERVED_INLINE", By: "restoreContext"},
					})
					continue
				}
			}
		}
		// vDSO-eligible iff the tool name is read-only-shaped (the same readOnlyPrefix
		// gate buildCall uses to stamp readOnlyHint+idempotentHint). A write-shaped tool
		// is never probed; vdso.Lookup's own destructive gate is the backstop.
		if !readOnlyPrefix(tool) {
			pass = append(pass, tc)
			continue
		}
		c2, err := s.buildCall(ctx, tool, tc.Function.Arguments, true, "", reqTrace)
		if err != nil {
			pass = append(pass, tc)
			continue
		}
		// Probe the SAME registered fast paths the kernel consults in Submit
		// (abi.FastPaths() -> the wired vDSO), not vdso.Default directly, so the seam
		// respects whatever instance the gateway wired (production vdso.Default, or a
		// fresh per-test vDSO). A miss returns ok=false and executes nothing.
		res, ok := fastPathLookup(ctx, c2)
		if !ok || res == nil {
			pass = append(pass, tc) // miss -> normal adjudication path, unchanged
			continue
		}
		// Operator TTL ceiling (#1349): if this tool has a configured max-age and the
		// tier-2 hit is OLDER than it, decline the inline serve and let the call run fresh.
		// This is the deterministic counterpart to the model-driven _fak_fresh above — the
		// operator sets a hard per-tool freshness bound; the model judges per-call. Sound
		// for the same reason: it only ever turns a would-be served hit into a normal
		// tool_use, byte-identical to a cache MISS (the already-tested fall-through). It is
		// the first ENFORCED consumer of abi.ConsistencyBoundedStale's staleness bound.
		if s.servedHitOverMaxAge(tool, res.Meta) {
			pass = append(pass, tc)
			continue
		}
		body := resolveBytes(ctx, res.Payload)
		// Never fold a poisoned cache entry into context as prose. If the served bytes
		// trip the screen, drop the served hit and let the call go through normal
		// adjudication instead (fail-safe: behave as a miss).
		if _, held := ctxmmu.ScreenBytes(body); held {
			pass = append(pass, tc)
			continue
		}
		served = append(served, servedToolLine(tool, body, res.Meta))
		servedHits++
		adjs = append(adjs, ToolAdjudication{ToolCallID: tc.ID, Tool: tool, ArgsDigest: argsDigest, Admitted: true,
			Verdict: WireVerdict{Kind: "ALLOW", Reason: "SERVED_INLINE", By: "vdso"}})
	}
	kept, adjs2, dropped := s.adjudicateProposed(ctx, pass, reqTrace)
	adjs = append(adjs, adjs2...)
	fusedIDs := s.annotateToolLivelock(reqTrace, adjs)
	// A fused call had its adjudication flipped to DENY above, but it is still sitting
	// in `kept` (adjudicateProposed admitted it before the fuse ran). Drop it here so it
	// never reaches the wire — the fuse is only real if the call actually stops.
	if len(fusedIDs) > 0 {
		filtered := kept[:0]
		for _, tc := range kept {
			if _, fused := fusedIDs[tc.ID]; fused {
				dropped++
				continue
			}
			filtered = append(filtered, tc)
		}
		kept = filtered
	}
	if len(served) > 0 {
		servedText = strings.Join(served, "\n")
	}
	return kept, adjs, dropped, servedText, servedHits
}

// fakFreshMarker is the reserved arg key the served-cache line tells the model to set
// to force a fresh read. It is namespaced under "_fak_" so it cannot collide with a real
// tool argument (no real tool defines a leading-underscore _fak_-prefixed param).
const fakFreshMarker = "_fak_fresh"

// servedToolLine renders one vDSO-served tool result as an assistant-text line — the
// only wire-valid surface for a locally-served answer (a tool_result block is a
// user-turn block, illegal in an assistant response). The model's next turn reads it
// as the assistant's own statement of the tool's result. When the hit carries an age
// (tier-2 only), the line names how stale the read is AND how to force a fresh one, so
// the model can decide for itself whether the cached value is good enough.
func servedToolLine(tool string, body []byte, meta map[string]string) string {
	suffix := "served from cache"
	if age, ok := cacheAgeLabel(meta); ok {
		suffix += ", ~" + age + " old; to force a fresh read, re-call with \"" + fakFreshMarker + "\": true"
	}
	return "[fak] " + tool + " (" + suffix + "): " + strings.TrimSpace(string(body))
}

// cacheAgeLabel turns a tier-2 hit's age_ms into a coarse human/model label ("3m", "45s").
// ok=false when no age_ms is present (tier-1 pure / tier-3 static hits, or a non-vdso
// caller), so the line renders WITHOUT an age clause — byte-identical to the pre-age text.
func cacheAgeLabel(meta map[string]string) (string, bool) {
	if meta == nil {
		return "", false
	}
	ms, err := strconv.ParseInt(meta["age_ms"], 10, 64)
	if err != nil || ms < 0 {
		return "", false
	}
	d := time.Duration(ms) * time.Millisecond
	switch {
	case d < time.Minute:
		return strconv.Itoa(int(d.Seconds())) + "s", true
	case d < time.Hour:
		return strconv.Itoa(int(d.Minutes())) + "m", true
	default:
		return strconv.Itoa(int(d.Hours())) + "h", true
	}
}

// SetToolMaxAge sets the operator's served-read freshness ceiling for one tool (#1349):
// a vDSO tier-2 hit older than d is declined inline and runs fresh (see
// servedHitOverMaxAge). A zero/negative d removes the ceiling. It is a wiring-time seam
// (the same inject-after-New posture as the other host-set knobs) — set it before Serve
// accepts turns; the served path reads maxAgeByTool without a lock.
func (s *Server) SetToolMaxAge(tool string, d time.Duration) {
	if s == nil || tool == "" {
		return
	}
	if d <= 0 {
		delete(s.maxAgeByTool, tool)
		return
	}
	if s.maxAgeByTool == nil {
		s.maxAgeByTool = map[string]time.Duration{}
	}
	s.maxAgeByTool[tool] = d
}

// servedHitOverMaxAge reports whether a tier-2 served hit for tool exceeds the tool's
// configured max-age ceiling (#1349) — the deterministic gate adjudicateProposedServed
// uses to decline an over-age inline serve. It is fail-open by construction: no config
// for the tool, a non-positive ceiling, or a hit that carries no age (tier-1 pure /
// tier-3 static, which recompute every hit and so have no staleness) all return false,
// leaving the served path byte-identical to today. Only a tool WITH a positive ceiling
// AND a hit whose age_ms strictly exceeds it declines the serve.
func (s *Server) servedHitOverMaxAge(tool string, meta map[string]string) bool {
	if s == nil || len(s.maxAgeByTool) == 0 {
		return false
	}
	max, ok := s.maxAgeByTool[tool]
	if !ok || max <= 0 {
		return false
	}
	ms, err := strconv.ParseInt(meta["age_ms"], 10, 64)
	if err != nil || ms < 0 {
		return false // no age surfaced (tier-1/3, or a non-vdso hit) ⇒ no ceiling applies
	}
	return time.Duration(ms)*time.Millisecond > max
}

// callRequestsFresh reports whether the proposed call's JSON args set _fak_fresh truthy —
// the model's signal to bypass the served-inline cache and run the tool for real. A
// non-JSON or unparseable args blob is treated as NO marker (today's behavior), so a model
// that ignores the affordance gets exactly today's served-inline path.
func callRequestsFresh(args string) bool {
	if !strings.Contains(args, fakFreshMarker) {
		return false // fast reject: no substring, no parse
	}
	var m map[string]json.RawMessage
	if err := json.Unmarshal([]byte(args), &m); err != nil {
		return false
	}
	raw, ok := m[fakFreshMarker]
	if !ok {
		return false
	}
	var b bool
	return json.Unmarshal(raw, &b) == nil && b
}

// livelockObservationOf builds the livelock observation for one adjudicated tool
// call — the identical field-fill the admitted and failed observation paths both
// record.
func livelockObservationOf(trace string, a ToolAdjudication) guardrsi.LivelockObservation {
	return guardrsi.LivelockObservation{
		TraceID:     trace,
		Tool:        a.Tool,
		ArgsDigest:  a.ArgsDigest,
		Verdict:     a.Verdict.Kind,
		Reason:      a.Verdict.Reason,
		Disposition: a.Verdict.Disposition,
	}
}

// reusableCapabilityDiscovery reports the exact tool identities through
// which fak's MCP capabilities operation reaches adjudication. That operation is
// side-effect-free and its MCP implementation bounds successful-result reuse, so
// applying the generic call-level livelock advisory as well only spams the model.
// Exact identities are intentional: another discovery tool, another MCP server,
// or any stateful call keeps the generic detector.
func reusableCapabilityDiscovery(a ToolAdjudication) bool {
	if !a.Admitted || a.Verdict.Kind != "ALLOW" {
		return false
	}
	return a.Tool == "fak_capabilities" ||
		a.Tool == "mcp__fak__fak_capabilities" ||
		a.Tool == "mcp__fak_guard__fak_capabilities"
}

// annotateToolLivelock attaches the livelock envelope to each repeated adjudication
// and, when an admitted call's envelope has armed the hard Fuse, converts that call
// into a retryable DENY in place. It returns the set of tool-call IDs it fused so the
// caller can drop them from the kept slice (they were admitted before the fuse ran).
func (s *Server) annotateToolLivelock(trace string, adjs []ToolAdjudication) map[string]struct{} {
	if s == nil || trace == "" {
		return nil
	}
	s.livelockMu.Lock()
	if s.livelock == nil {
		s.livelock = guardrsi.NewLivelockDetector(guardrsi.DefaultLivelockThreshold)
	}
	type hit struct {
		idx int
		env guardrsi.LivelockEnvelope
	}
	var hits []hit
	sawObservation := false
	sawReusableDiscovery := false
	for i := range adjs {
		a := adjs[i]
		if reusableCapabilityDiscovery(a) {
			// Ignore rather than clear: interleaving a cached discovery call must
			// not reset a stateful tool's generic livelock run.
			sawReusableDiscovery = true
			continue
		}
		switch {
		case a.Admitted && (a.Verdict.Kind == "ALLOW" || a.Verdict.Kind == "TRANSFORM") && a.Verdict.Reason != "SERVED_INLINE":
			sawObservation = true
			env, ok := s.livelock.ObserveAdmitted(livelockObservationOf(trace, a))
			if ok {
				hits = append(hits, hit{idx: i, env: env})
			}
			continue
		case a.Admitted || a.Verdict.Kind == "ALLOW" || a.Verdict.Kind == "TRANSFORM":
			continue
		}
		sawObservation = true
		env, ok := s.livelock.ObserveFailure(livelockObservationOf(trace, a))
		if ok {
			hits = append(hits, hit{idx: i, env: env})
		}
	}
	if !sawObservation && !sawReusableDiscovery {
		s.livelock.Clear(trace)
	}
	s.livelockMu.Unlock()

	var fused map[string]struct{}
	for _, h := range hits {
		env := h.env
		adjs[h.idx].Livelock = &env
		// Hard fuse: an ADMITTED call that has repeated identically past the fuse count
		// has ignored every advisory note. Admitting it again just burns tokens on a loop
		// the model won't leave on its own, so convert it into a real refusal here. The
		// refusal is retryable per-tool feedback (not a deny-all session stop): the
		// existing adjudicationNote machinery renders LIVELOCK_FUSE as "fak refused this
		// repeated call", which is the structural break the advisory note alone never was.
		//
		// Terminal escalation: once the run passes the ABORT count (env.Escalate), even
		// the retryable fuse has been ignored turn after turn — the model keeps
		// re-proposing the identical call and the auto-continue burns tokens without end
		// (worker #2704: an identical Bash call fused at 6, then spun to ~125k tokens
		// because RETRYABLE feedback is auto-continued). Stamp the refusal TERMINAL so
		// toolRejectionIsRetryableFeedback returns false and the all-refused turn becomes
		// a deny-all — the bounded give-up path that actually ends the session. This only
		// engages far above the fuse, so the advisory + retryable fuse always fire first.
		switch {
		case env.Fuse && adjs[h.idx].Admitted:
			disposition := "RETRYABLE"
			by := "livelock-fuse"
			if env.Escalate {
				disposition = "TERMINAL"
				by = "livelock-abort"
			}
			adjs[h.idx].Admitted = false
			adjs[h.idx].RepairedArguments = nil
			adjs[h.idx].Verdict = WireVerdict{
				Kind:        "DENY",
				Reason:      ReasonLivelockFuse,
				By:          by,
				Disposition: disposition,
			}
			if fused == nil {
				fused = map[string]struct{}{}
			}
			if id := adjs[h.idx].ToolCallID; id != "" {
				fused[id] = struct{}{}
			}
		case env.Escalate && !adjs[h.idx].Admitted:
			// An ALREADY-denied call the model keeps re-proposing past the abort count:
			// its per-tool refusal is being ignored just like the admitted loop, so lift
			// its retryable disposition to TERMINAL and let the all-refused turn escalate
			// to a deny-all stop. (MALFORMED/MISROUTE reasons stay retryable by classifier
			// design — those are genuinely fixable, not a stuck loop.)
			adjs[h.idx].Verdict.Disposition = "TERMINAL"
			adjs[h.idx].Verdict.By = "livelock-abort"
		}
	}
	return fused
}

func adjudicationOutcomeForTurn(adjs []ToolAdjudication, keptTools, servedTools int) adjudicationOutcomeSignal {
	if len(adjs) == 0 || keptTools > 0 || servedTools > 0 {
		return adjudicationOutcomeReset
	}
	sawRejected := false
	allRetryableFeedback := true
	for _, adj := range adjs {
		if adj.Admitted || adj.Verdict.Kind == "ALLOW" || adj.Verdict.Kind == "TRANSFORM" {
			return adjudicationOutcomeReset
		}
		sawRejected = true
		if !toolRejectionIsRetryableFeedback(adj.Verdict) {
			allRetryableFeedback = false
		}
	}
	if !sawRejected {
		return adjudicationOutcomeReset
	}
	if allRetryableFeedback {
		return adjudicationOutcomeToolFeedback
	}
	return adjudicationOutcomeDenyAll
}

func toolRejectionIsRetryableFeedback(v WireVerdict) bool {
	reason := strings.ToUpper(strings.TrimSpace(v.Reason))
	if reason == "MALFORMED" || reason == "MISROUTE" {
		return true
	}
	return strings.EqualFold(strings.TrimSpace(v.Disposition), "RETRYABLE")
}

// denyAllFingerprint builds a stable identity for a deny-all turn: the sorted, distinct set of
// (tool, reason) pairs the capability floor hard-refused. Two deny-all turns share a fingerprint
// iff they refused the SAME tools for the SAME reasons — the "same issue" test the guard Stop
// hook keys its bounded give-up on (a session that re-proposes the identical refused action turn
// after turn is genuinely spinning; one hitting a fresh block each turn is exploring and must not
// be stopped). Order- and args-insensitive by construction, mirroring session.DenyAllBreaker's
// tool+reason identity. Returns "" when nothing is fingerprintable (no refused call carried a
// tool name or reason), which the fold treats as fail-open — an unidentifiable turn never
// accumulates toward a stop.
func denyAllFingerprint(adjs []ToolAdjudication) string {
	seen := make(map[string]struct{}, len(adjs))
	pairs := make([]string, 0, len(adjs))
	for _, a := range adjs {
		// A deny-all turn admitted nothing, but guard against a stray admitted entry so the
		// fingerprint only ever reflects the refusals that define the issue.
		if a.Admitted || a.Verdict.Kind == "ALLOW" || a.Verdict.Kind == "TRANSFORM" {
			continue
		}
		tool := strings.TrimSpace(a.Tool)
		reason := strings.ToUpper(strings.TrimSpace(a.Verdict.Reason))
		if tool == "" && reason == "" {
			continue
		}
		key := tool + "\x1f" + reason
		if _, ok := seen[key]; ok {
			continue
		}
		seen[key] = struct{}{}
		pairs = append(pairs, key)
	}
	sort.Strings(pairs)
	return strings.Join(pairs, "\x1e")
}

func (s *Server) adjudicateProposed(ctx context.Context, calls []agent.ToolCall, reqTrace string) ([]agent.ToolCall, []ToolAdjudication, int) {
	kept := make([]agent.ToolCall, 0, len(calls))
	adjs := make([]ToolAdjudication, 0, len(calls))
	dropped := 0
	// One clock for the whole turn's admitted-activity stamps (#2627): the calls in a
	// single adjudicateProposed are effectively simultaneous, so a shared `now` keeps the
	// agents pane's last_tool/idle deterministic and spares a time.Now() per call.
	now := time.Now()
	for _, tc := range calls {
		tool := tc.Function.Name
		s.observePrunedToolProposal(reqTrace, tool)
		argsDigest := guardrsi.ArgsDigest(tc.Function.Arguments)
		seq := s.nextOriginSeq()
		wv, repaired, aerr := s.adjudicateWithSeq(ctx, tool, tc.Function.Arguments, false, "", reqTrace, seq)
		if aerr != nil {
			dropped++
			adjs = append(adjs, ToolAdjudication{ToolCallID: tc.ID, Tool: tool, ArgsDigest: argsDigest, Admitted: false,
				Verdict: WireVerdict{Kind: "DENY", Reason: "MALFORMED", Disposition: "RETRYABLE"}})
			continue
		}
		adj := ToolAdjudication{ToolCallID: tc.ID, Tool: tool, ArgsDigest: argsDigest, Verdict: wv}
		switch wv.Kind {
		case "ALLOW":
			adj.Admitted = true
			s.rememberOriginSeqID(reqTrace, tc.ID, seq)
			s.rememberOriginSeq(reqTrace, tool, tc.Function.Arguments, seq)
			kept = append(kept, tc)
		case "TRANSFORM":
			adj.Admitted = true
			if repaired != "" {
				tc.Function.Arguments = repaired
				adj.RepairedArguments = json.RawMessage(repaired)
			}
			s.rememberOriginSeqID(reqTrace, tc.ID, seq)
			s.rememberOriginSeq(reqTrace, tool, tc.Function.Arguments, seq)
			kept = append(kept, tc)
		default:
			dropped++
		}
		adjs = append(adjs, adj)
		// Stamp the agents pane's live-status cell (#2627) for admitted calls only: an
		// admitted call is what the trace is actually DOING. last_tool takes the last
		// admitted tool of the turn; spawn_count counts admitted subagent-shaped calls.
		if adj.Admitted {
			s.activity.stampProposed(reqTrace, tool, now)
		}
	}
	return kept, adjs, dropped
}

func (s *Server) adjudicateProposedTurn(ctx context.Context, asst agent.Message, reqTrace string) (kept []agent.ToolCall, adjs []ToolAdjudication, dropped int, servedText string, servedHits int, bodyRefused bool) {
	kept, adjs, dropped, servedText, servedHits = s.adjudicateProposedServed(ctx, asst.ToolCalls, reqTrace)
	if !turnBodyClaimsCompletedEdit(asst.Content) || !turnHasAdmittedCall(adjs) || turnHasEffectCapableCall(kept) {
		return kept, adjs, dropped, servedText, servedHits, false
	}
	residual := WireVerdict{
		Kind:        "RESIDUAL",
		Reason:      ReasonLoopBodyUnwitnessed,
		By:          "loop-body-witness",
		Disposition: "RETRYABLE",
	}
	for i := range adjs {
		if !adjs[i].Admitted {
			continue
		}
		if adjs[i].Verdict.Reason == "SERVED_INLINE" {
			servedHits--
		}
		adjs[i].Admitted = false
		adjs[i].Verdict = residual
		adjs[i].RepairedArguments = nil
		dropped++
	}
	servedText = ""
	if servedHits < 0 {
		servedHits = 0
	}
	return nil, adjs, dropped, servedText, servedHits, true
}

func turnBodyClaimsCompletedEdit(content string) bool {
	body := " " + strings.ToLower(strings.Join(strings.Fields(content), " ")) + " "
	if strings.TrimSpace(body) == "" {
		return false
	}
	if containsAnySubstring(body, []string{
		" changes are complete ",
		" edits are complete ",
		" file has been updated ",
		" file was updated ",
	}) {
		return true
	}
	if !turnBodyNamesEditableArtifact(body) {
		return false
	}
	return containsAnySubstring(body, []string{
		" i edited ",
		" i've edited ",
		" i have edited ",
		" i modified ",
		" i've modified ",
		" i have modified ",
		" i updated ",
		" i've updated ",
		" i have updated ",
		" i created ",
		" i deleted ",
		" i removed ",
		" i replaced ",
		" i renamed ",
		" i wrote ",
	})
}

// containsAnySubstring reports whether s contains at least one of subs.
func containsAnySubstring(s string, subs []string) bool {
	for _, sub := range subs {
		if strings.Contains(s, sub) {
			return true
		}
	}
	return false
}

func turnBodyNamesEditableArtifact(body string) bool {
	return containsAnySubstring(body, []string{
		" file ", " files ", " folder ", " directory ", " repo ", " code ",
		" readme", " doc ", " docs ", ".go", ".md", ".json", ".yaml", ".yml",
		".txt", ".ps1", ".sh", ".py", ".ts", ".tsx", ".js", ".jsx",
	})
}

func turnHasAdmittedCall(adjs []ToolAdjudication) bool {
	for _, adj := range adjs {
		if adj.Admitted {
			return true
		}
	}
	return false
}

func turnHasEffectCapableCall(calls []agent.ToolCall) bool {
	for _, tc := range calls {
		name := strings.ToLower(tc.Function.Name)
		if containsAnySubstring(name, []string{
			"write", "edit", "patch", "apply", "create", "delete", "remove",
			"replace", "rename", "move", "commit", "bash", "shell", "exec",
			"command", "python",
		}) {
			return true
		}
	}
	return false
}

func isRestoreTool(tool string) bool {
	return tool == "fak_context_restore" ||
		tool == "mcp__fak__fak_context_restore" ||
		tool == "mcp__fak_guard__fak_context_restore" ||
		tool == "functions.fak_context_restore"
}
