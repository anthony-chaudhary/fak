package gateway

import (
	"context"
	"encoding/json"
	"strconv"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/ctxmmu"
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
	pass := make([]agent.ToolCall, 0, len(calls))
	var served []string
	for _, tc := range calls {
		tool := tc.Function.Name
		// vDSO-eligible iff the tool name is read-only-shaped (the same readOnlyPrefix
		// gate buildCall uses to stamp readOnlyHint+idempotentHint). A write-shaped tool
		// is never probed; vdso.Lookup's own destructive gate is the backstop.
		if !readOnlyPrefix(tool) {
			pass = append(pass, tc)
			continue
		}
		// Force-fresh escape hatch: a re-proposed read carrying the advertised _fak_fresh
		// marker skips the cache probe and passes through to the client to actually run.
		// Sound: this only ever turns a would-be served hit into a normal tool_use —
		// byte-identical to a cache MISS, the already-tested fall-through. It can never
		// create an effect or relax a gate; it only declines the optimization. The model
		// reaches for this when the served age (below) says the cached read is too stale.
		if callRequestsFresh(tc.Function.Arguments) {
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
		adjs = append(adjs, ToolAdjudication{ToolCallID: tc.ID, Tool: tool, Admitted: true,
			Verdict: WireVerdict{Kind: "ALLOW", Reason: "SERVED_INLINE", By: "vdso"}})
	}
	kept, adjs2, dropped := s.adjudicateProposed(ctx, pass, reqTrace)
	adjs = append(adjs, adjs2...)
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

func (s *Server) adjudicateProposed(ctx context.Context, calls []agent.ToolCall, reqTrace string) ([]agent.ToolCall, []ToolAdjudication, int) {
	kept := make([]agent.ToolCall, 0, len(calls))
	adjs := make([]ToolAdjudication, 0, len(calls))
	dropped := 0
	for _, tc := range calls {
		tool := tc.Function.Name
		seq := s.nextOriginSeq()
		wv, repaired, aerr := s.adjudicateWithSeq(ctx, tool, tc.Function.Arguments, false, "", reqTrace, seq)
		if aerr != nil {
			dropped++
			adjs = append(adjs, ToolAdjudication{ToolCallID: tc.ID, Tool: tool, Admitted: false,
				Verdict: WireVerdict{Kind: "DENY", Reason: "MALFORMED", Disposition: "RETRYABLE"}})
			continue
		}
		adj := ToolAdjudication{ToolCallID: tc.ID, Tool: tool, Verdict: wv}
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
	for _, phrase := range []string{
		" changes are complete ",
		" edits are complete ",
		" file has been updated ",
		" file was updated ",
	} {
		if strings.Contains(body, phrase) {
			return true
		}
	}
	if !turnBodyNamesEditableArtifact(body) {
		return false
	}
	for _, phrase := range []string{
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
	} {
		if strings.Contains(body, phrase) {
			return true
		}
	}
	return false
}

func turnBodyNamesEditableArtifact(body string) bool {
	for _, marker := range []string{
		" file ", " files ", " folder ", " directory ", " repo ", " code ",
		" readme", " doc ", " docs ", ".go", ".md", ".json", ".yaml", ".yml",
		".txt", ".ps1", ".sh", ".py", ".ts", ".tsx", ".js", ".jsx",
	} {
		if strings.Contains(body, marker) {
			return true
		}
	}
	return false
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
		for _, token := range []string{
			"write", "edit", "patch", "apply", "create", "delete", "remove",
			"replace", "rename", "move", "commit", "bash", "shell", "exec",
			"command", "python",
		} {
			if strings.Contains(name, token) {
				return true
			}
		}
	}
	return false
}
