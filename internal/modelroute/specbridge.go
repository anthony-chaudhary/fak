package modelroute

// Bridge ensemble roles to speculative decoding (#604, epic #595).
//
// Member.Role (drafter / verifier) and internal/polymodel (residency +
// PickDrafter + AcceptGreedy/AcceptTree) are complementary: a drafter+verifier
// ensemble Plan is a natural front-end for speculative decoding on co-resident
// models. This file bridges them — it maps an ensemble Plan's roles onto the
// drafter/verifier pair, so an "ensemble" can mean THROUGHPUT (spec-decode) as
// well as answer-level reduce.
//
// LANE PURITY (lane rule + #604 wording): the tier-1 leaf stays stdlib-only — it
// does NOT import internal/polymodel. The pass-through to polymodel.AcceptGreedy
// takes the polymodel accept call as a CLOSURE the caller binds:
//
//	res := plan.SpecAccept(draft, targetArgmax, func(d, t []int) modelroute.SpecAccept {
//	    r := polymodel.AcceptGreedy(d, t)
//	    return modelroute.SpecAccept{Accepted: r.Accepted, Advance: r.Advance, KeepKV: r.KeepKV, EvictKV: r.EvictKV}
//	})
//
// so polymodel's types cross the seam as plain ints, never as an import. The LIVE
// engine verify pass (running the verifier model to PRODUCE targetArgmax on a real
// engine, and the model.KVCache.Evict rollback) is DEFERRED — that is engine
// wiring above this bridge, out of lane.

import "fmt"

// SpecRoles is the drafter/verifier split a roles-tagged ensemble Plan maps onto
// for speculative decoding: one cheap DRAFTER proposes tokens, one VERIFIER
// accepts the longest correct prefix in a single pass. It is the answer to "which
// member drafts and which verifies?" derived purely from Member.Role.
type SpecRoles struct {
	Drafter  string `json:"drafter"`  // the member model id tagged role=drafter
	Verifier string `json:"verifier"` // the member model id tagged role=verifier
}

// SpecRoles extracts the drafter/verifier pair from a Plan's member roles for a
// spec-decode bridge. It requires EXACTLY one member tagged role=="drafter" and
// EXACTLY one tagged role=="verifier" (a verifier may also be the Plan's primary;
// roles are read from Member.Role, case-insensitive on the two reserved labels).
// Any other shape — zero or multiple drafters/verifiers — is not a spec-decode
// ensemble and fails loud, so a vote/best-of ensemble is never silently treated
// as a draft/verify pair.
func (p Plan) SpecRoles() (SpecRoles, error) {
	var draft, verify string
	var nDraft, nVerify int
	for _, m := range p.Members {
		switch normalizeRole(m.Role) {
		case "drafter":
			draft = m.Model
			nDraft++
		case "verifier":
			verify = m.Model
			nVerify++
		}
	}
	if nDraft != 1 || nVerify != 1 {
		return SpecRoles{}, fmt.Errorf("modelroute: plan is not a drafter/verifier spec ensemble (drafters=%d verifiers=%d)", nDraft, nVerify)
	}
	return SpecRoles{Drafter: draft, Verifier: verify}, nil
}

// IsSpecEnsemble reports whether the Plan carries exactly one drafter and one
// verifier — the shape SpecRoles accepts. A dispatcher uses this to decide whether
// to run an ensemble as spec-decode (throughput) rather than answer-level reduce.
func (p Plan) IsSpecEnsemble() bool {
	_, err := p.SpecRoles()
	return err == nil
}

// SpecAccept is the polymodel.SpecResult shape carried across the lane seam as
// plain ints, so internal/modelroute never imports internal/polymodel. The caller
// adapts polymodel.AcceptGreedy (or AcceptTree) into this struct; the fields are
// the same accounting: Accepted leading draft tokens matched, Advance real tokens
// committed, KeepKV speculative positions kept, EvictKV rolled back.
type SpecAccept struct {
	Accepted int `json:"accepted"`
	Advance  int `json:"advance"`
	KeepKV   int `json:"keep_kv"`
	EvictKV  int `json:"evict_kv"`
}

// AcceptFunc is the bound speculative-accept call — the caller passes
// polymodel.AcceptGreedy adapted to SpecAccept. It takes the drafter's proposed
// token ids and the verifier's argmax at each position (from the single verify
// pass) and returns the accept accounting. This is the closure seam that keeps
// polymodel out of the leaf's import graph.
type AcceptFunc func(draft, targetArgmax []int) SpecAccept

// SpecAccept runs the spec-decode accept pass for a roles-tagged ensemble Plan: it
// first asserts the Plan IS a drafter/verifier ensemble (else refuses — a non-spec
// ensemble must not be run as spec-decode), then passes the drafter's proposed
// tokens and the verifier's argmax through the bound accept call (the caller binds
// polymodel.AcceptGreedy). The pass-through is read-only: it computes the accept
// accounting and the resolved roles, but never runs an engine. The live verify
// pass that PRODUCES targetArgmax, and the KVCache.Evict rollback of EvictKV
// positions, are the deferred engine wiring above this bridge.
func (p Plan) SpecAccept(draft, targetArgmax []int, accept AcceptFunc) (SpecRoles, SpecAccept, error) {
	roles, err := p.SpecRoles()
	if err != nil {
		return SpecRoles{}, SpecAccept{}, err
	}
	if accept == nil {
		return SpecRoles{}, SpecAccept{}, fmt.Errorf("modelroute: SpecAccept needs a bound accept func (pass polymodel.AcceptGreedy)")
	}
	return roles, accept(draft, targetArgmax), nil
}

// normalizeRole lowercases and trims a Member.Role so "Drafter"/"DRAFTER"/
// "drafter " all map to the reserved "drafter"/"verifier" labels. Any other role
// (primary, judge, …) returns unchanged-lowercased and is ignored by SpecRoles.
func normalizeRole(role string) string {
	out := make([]byte, 0, len(role))
	// trim surrounding spaces
	i, j := 0, len(role)
	for i < j && role[i] == ' ' {
		i++
	}
	for j > i && role[j-1] == ' ' {
		j--
	}
	for ; i < j; i++ {
		c := role[i]
		if c >= 'A' && c <= 'Z' {
			c += 'a' - 'A'
		}
		out = append(out, c)
	}
	return string(out)
}
