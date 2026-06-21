package cachemeta

import (
	"sort"
	"strings"
)

// prefix_coherence.go — GLM52-HOSTED-CACHE-COHERENCE §A4 (witness-gated provider-prefix
// breaking, CPU-only).
//
// A3 (prefix_stability.go) maximizes provider-prefix STABILITY so the hosted model
// caches the longest prefix. A4 is its load-bearing counterpart: when the WORLD the
// prefix described has changed, fak must FORCE the next GLM turn to break the prefix
// rather than let the provider serve a cached, now-stale prefix — refusal rules 1 + 3
// made mechanical (a cached_tokens hit is never freshness evidence; a span fak knows is
// stale must not be re-served).
//
// A provider-cached prefix can depend on world witnesses — a git SHA, a blob hash, a
// lease epoch — captured when it was built. When a witness is REFUTED (its current
// value diverges, or can no longer be confirmed) the dependent span is stale, and the
// directive breaks the prefix at the EARLIEST stale span by injecting a volatile marker
// the provider cache cannot match. The marker is a SegVolatile segment, so A3's analysis
// accounts for the intentional break. fak breaks the prefix on ITS side; it cannot
// exact-evict the hosted provider cache.

// PrefixDependency ties a span of a provider-cached prefix to a world witness it
// depends on. TokenStart is the token offset where the dependent span begins.
type PrefixDependency struct {
	Witness    string // "git_sha:path" | "blob_hash:id" | "lease_epoch:id"
	Value      string // value witnessed when the prefix was built
	TokenStart int64
}

// PrefixBreakDirective is the typed coherence answer for the next GLM turn.
type PrefixBreakDirective struct {
	Break        bool
	RefutedKeys  []string      // witnesses whose current value diverged or is unconfirmable
	BreakAtToken int64         // token offset of the earliest stale span
	Marker       PromptSegment // the volatile marker to inject ahead of the stale span
	Reason       string
}

// EvaluatePrefixCoherence checks a provider-cached prefix's dependencies against the
// current world (witness-key -> current value). A dependency is REFUTED when its
// current value differs from the recorded value OR is absent — fail-closed: an
// unconfirmable witness is treated as stale, because a coherence layer must never serve
// a prefix it cannot prove fresh. When any dependency is refuted the directive breaks at
// the EARLIEST refuted span and supplies a volatile marker derived from the refuted
// (key,newValue) pairs: deterministic for a given refutation, but guaranteed to differ
// from the cached prefix so the provider cannot serve the stale span.
func EvaluatePrefixCoherence(deps []PrefixDependency, current map[string]string) PrefixBreakDirective {
	var refs []refutation
	for _, d := range deps {
		cur, ok := current[d.Witness]
		switch {
		case !ok:
			refs = append(refs, refutation{d.Witness, "<unconfirmed>", d.TokenStart})
		case cur != d.Value:
			refs = append(refs, refutation{d.Witness, cur, d.TokenStart})
		}
	}
	return buildBreakDirective(refs)
}

// EvaluatePrefixCoherenceRevoked is the REVOCATION-PREDICATE form of the coherence
// check — the shape that wires directly to the live vdso / gateway revocation bus. A
// dependency is refuted when `revoked(witness)` reports its trust witness has been
// revoked (the recall layer's Page.Witness + vdso.Default.Revoked model). This is what
// the live OpenAI-proxy path uses; EvaluatePrefixCoherence is the value-comparison form
// for offline transcript analysis. Both break at the earliest refuted span with a
// world-sensitive volatile marker. A nil predicate refutes nothing (fail-OPEN is the
// caller's explicit choice when there is no revocation bus to consult).
func EvaluatePrefixCoherenceRevoked(deps []PrefixDependency, revoked func(witness string) bool) PrefixBreakDirective {
	if revoked == nil {
		return PrefixBreakDirective{Break: false, Reason: "no revocation bus"}
	}
	var refs []refutation
	for _, d := range deps {
		if revoked(d.Witness) {
			refs = append(refs, refutation{d.Witness, "<revoked>", d.TokenStart})
		}
	}
	return buildBreakDirective(refs)
}

// EvaluateSegmentCoherence is the SEGMENT-LEVEL §A4 check — the form that dissolves the
// page-vs-prompt token-coordinate problem by carrying the trust witness ON each segment
// (PromptSegment.Witness) instead of in a separate offset-indexed deps list. It refutes
// every segment whose witness is revoked and breaks at the earliest one (its cumulative
// token offset). This is the form the LIVE proxy uses: when converting request messages
// to segments, attach each segment's witness from the recorded page that produced its
// content, then call this — no coordinate alignment between two byte sources required.
func EvaluateSegmentCoherence(segs []PromptSegment, revoked func(witness string) bool) PrefixBreakDirective {
	if revoked == nil {
		return PrefixBreakDirective{Break: false, Reason: "no revocation bus"}
	}
	var refs []refutation
	var off int64
	for _, s := range segs {
		if s.Witness != "" && revoked(s.Witness) {
			refs = append(refs, refutation{s.Witness, "<revoked>", off})
		}
		off += s.Tokens
	}
	return buildBreakDirective(refs)
}

type refutation struct {
	key, newVal string
	at          int64
}

// buildBreakDirective turns a refutation set into the typed directive: no refutations
// => no break; otherwise break at the earliest stale span with a deterministic,
// world-sensitive marker over the refuted (key,newValue) pairs.
func buildBreakDirective(refs []refutation) PrefixBreakDirective {
	if len(refs) == 0 {
		return PrefixBreakDirective{Break: false, Reason: "all witnesses hold"}
	}
	sort.Slice(refs, func(i, j int) bool {
		if refs[i].at != refs[j].at {
			return refs[i].at < refs[j].at
		}
		return refs[i].key < refs[j].key
	})
	keys := make([]string, len(refs))
	var marker strings.Builder
	marker.WriteString("coherence-break:")
	for i, r := range refs {
		keys[i] = r.key
		marker.WriteString(r.key)
		marker.WriteByte('=')
		marker.WriteString(r.newVal)
		marker.WriteByte(';')
	}
	return PrefixBreakDirective{
		Break:        true,
		RefutedKeys:  keys,
		BreakAtToken: refs[0].at,
		Marker: PromptSegment{
			Kind:    SegVolatile,
			Tokens:  1,
			Content: []byte(marker.String()),
		},
		Reason: "world witness refuted; break provider prefix at earliest stale span",
	}
}

// InjectBreakMarker APPLIES a PrefixBreakDirective to a turn's segments: it inserts the
// volatile break marker immediately AHEAD of the stale span (the first segment whose
// cumulative start offset reaches BreakAtToken), so the provider cache misses the stale
// prefix while the fresh prefix before the break still hits. A non-break directive
// returns the turn unchanged. This is the actionable output a live gateway / OpenAI
// proxy applies to the next GLM request after a world witness is refuted — the bridge
// between the A4 decision and the wire.
func InjectBreakMarker(turn []PromptSegment, d PrefixBreakDirective) []PromptSegment {
	if !d.Break {
		return turn
	}
	var off int64
	idx := len(turn)
	for i, s := range turn {
		if off >= d.BreakAtToken {
			idx = i
			break
		}
		off += s.Tokens
	}
	out := make([]PromptSegment, 0, len(turn)+1)
	out = append(out, turn[:idx]...)
	out = append(out, d.Marker)
	out = append(out, turn[idx:]...)
	return out
}

// ShapeGLMTurn is the hosted-GLM coherence front-end's single entry point — the one
// call a gateway / OpenAI-proxy makes per GLM turn. It runs the §A4 coherence check
// over the turn's provider-prefix dependencies against the current world and, if a
// witness is refuted, returns the turn with the break marker injected ahead of the
// stale span (so the provider cannot serve a now-stale cached prefix); otherwise it
// returns the turn unchanged. The directive records WHAT was done (for metrics/audit);
// the report is the §A3 stability of the SHAPED turn — what the provider will actually
// be able to cache after coherence shaping. This is the seam the live gateway plugs
// into; everything below it (decision, injection, stability) is already witnessed.
func ShapeGLMTurn(turn []PromptSegment, deps []PrefixDependency, world map[string]string) (shaped []PromptSegment, directive PrefixBreakDirective, report StabilityReport) {
	directive = EvaluatePrefixCoherence(deps, world)
	shaped = InjectBreakMarker(turn, directive)
	report = AnalyzeStability([][]PromptSegment{shaped})
	return shaped, directive, report
}

// ShapeGLMTurnRevoked is ShapeGLMTurn wired to the live revocation bus: the gateway
// passes `vdso.Default.Revoked` (or any revocation predicate over the recall layer's
// per-page trust witnesses) and gets back the coherence-shaped turn. This is the exact
// call the live OpenAI-proxy makes per GLM turn — deps come from the recorded prefix's
// witnessed pages, `revoked` from the vdso/gateway bus. The cachemeta layer stays
// decoupled (it takes a predicate, never imports vdso).
func ShapeGLMTurnRevoked(turn []PromptSegment, deps []PrefixDependency, revoked func(witness string) bool) (shaped []PromptSegment, directive PrefixBreakDirective, report StabilityReport) {
	directive = EvaluatePrefixCoherenceRevoked(deps, revoked)
	shaped = InjectBreakMarker(turn, directive)
	report = AnalyzeStability([][]PromptSegment{shaped})
	return shaped, directive, report
}

// ShapeGLMTurnSegmentWitnessed is the simplest live-path form: the turn's segments carry
// their own trust witnesses (PromptSegment.Witness), so no separate deps list and no
// page-vs-prompt coordinate alignment is needed. The live OpenAI-proxy builds segments
// from the request messages (attaching each tool-result segment's witness from the
// recorded page that produced it) and calls this with vdso.Default.Revoked. A revoked
// witness breaks the prefix at that segment.
func ShapeGLMTurnSegmentWitnessed(turn []PromptSegment, revoked func(witness string) bool) (shaped []PromptSegment, directive PrefixBreakDirective, report StabilityReport) {
	directive = EvaluateSegmentCoherence(turn, revoked)
	shaped = InjectBreakMarker(turn, directive)
	report = AnalyzeStability([][]PromptSegment{shaped})
	return shaped, directive, report
}
