package gateway

// Policy mutation as an adjudicated, journaled, expiring SYSCALL (#2406).
//
// The session control plane already applies typed verbs (run|budget|pace|priority)
// via POST /v1/fak/session/{id}/{verb} (handleFakSession, http.go). This is the
// policy analogue: three policy-op verbs carried as DATA — add_rules / remove_rules
// / set_regime — each ADJUDICATED like any tool call before it applies, JOURNALED
// into the hash chain (internal/journal), and SCOPED:
//
//   - A session-scoped op expires automatically with its trace (EndSession) — its
//     grant is bounded, so it never becomes durable residue (`fak policy --dump`
//     shows nothing after the session ends).
//   - A DURABLE op survives the session, so a WIDENING durable op requires a WITNESS
//     (an operator token or a human-principal event). Without one it is refused with
//     the closed reason UNWITNESSED, routed through abi.VerdictRequireWitness — the
//     same gate the kernel uses for an unwitnessed widening tool call.
//   - A TIGHTEN-ONLY op (adding a deny, removing an allow) applies immediately at
//     either scope: narrowing the floor never needs corroboration.
//
// Every applied op mints a monotonic rule EPOCH; an admission records the epoch that
// allowed it, so Revoke can causally EVICT everything admitted at or after a
// later-refuted epoch (the fak_revoke primitive) without touching unrelated
// admissions.
//
// This is the reachable Go seam. Wiring a "policy" verb into handleFakSession's
// dispatch (a nil PolicyRegime ⇒ 404, mirroring steer/control) and injecting a live
// regime from cmd/fak is the named PROMOTION step; until then the mechanism is
// exercised by its contract tests and kept off the live wire (gen/next: gated).

import (
	"fmt"
	"sort"
	"strings"
	"sync"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/journal"
	"github.com/anthony-chaudhary/fak/internal/maputil"
)

// PolicyOpKind is the closed set of policy-mutation verbs carried as data over the
// control plane — the policy analogue of the run|budget|pace|priority control verbs.
type PolicyOpKind string

const (
	// PolicyAddRules installs one or more rules. A rule with Allow=true WIDENS the
	// floor (grants a tool it did not permit); an Allow=false rule is an explicit deny
	// that TIGHTENS it.
	PolicyAddRules PolicyOpKind = "add_rules"
	// PolicyRemoveRules removes rules by tool. Removing an allow TIGHTENS the floor;
	// removing a deny WIDENS it (the tool falls back to whatever the floor says).
	PolicyRemoveRules PolicyOpKind = "remove_rules"
	// PolicySetRegime pivots the named regime. A pivot may widen, so a durable pivot
	// is treated as widening (witness-gated); a session-scoped pivot expires with the
	// trace.
	PolicySetRegime PolicyOpKind = "set_regime"
)

// PolicyScope is WHERE an applied op lives — and how long. A session-scoped op
// expires with its trace; a durable op survives the session.
type PolicyScope string

const (
	// ScopeSession is the default: the op lives only until EndSession(trace) and
	// leaves no durable residue.
	ScopeSession PolicyScope = "session"
	// ScopeDurable promotes the op past the session; a widening durable op requires a
	// witness.
	ScopeDurable PolicyScope = "durable"
)

// PolicyRule is one capability grant/denial carried by a policy op. The tighten/widen
// split is read structurally off Allow, not by convention, so the adjudicator can
// classify an op without trusting the caller's intent.
type PolicyRule struct {
	Tool  string `json:"tool"`
	Allow bool   `json:"allow"`
}

// PolicyOp is one typed, adjudicated policy mutation — the body of a policy syscall.
type PolicyOp struct {
	Kind    PolicyOpKind `json:"kind"`
	Scope   PolicyScope  `json:"scope,omitempty"`   // "" ⇒ session
	Rules   []PolicyRule `json:"rules,omitempty"`   // add_rules / remove_rules
	Regime  string       `json:"regime,omitempty"`  // set_regime
	Witness string       `json:"witness,omitempty"` // operator token / human-principal event id
}

// scope resolves the op's scope, defaulting to session (the safe, self-expiring one).
func (op PolicyOp) scope() PolicyScope {
	if op.Scope == ScopeDurable {
		return ScopeDurable
	}
	return ScopeSession
}

// widens reports whether the op could GRANT authority the floor did not already
// permit — the property that, at durable scope, forces a witness. It is fail-closed:
// an op whose direction is ambiguous (a regime pivot) counts as widening.
func (op PolicyOp) widens() bool {
	switch op.Kind {
	case PolicyAddRules:
		for _, r := range op.Rules {
			if r.Allow {
				return true // adding a grant widens
			}
		}
		return false // adding only denies tightens
	case PolicyRemoveRules:
		for _, r := range op.Rules {
			if !r.Allow {
				return true // removing a deny widens
			}
		}
		return false // removing only allows tightens
	case PolicySetRegime:
		return true // a pivot may widen — fail closed
	default:
		return true // unknown kind: fail closed
	}
}

// PolicyRegime is the runtime policy-mutation control plane. It holds a durable
// allow/deny floor plus per-session overlays that expire with the trace, mints a
// monotonic rule epoch on every applied op, records admissions against the epoch that
// allowed them, and journals every op into the hash chain. The zero value is not
// usable; build one with NewPolicyRegime.
type PolicyRegime struct {
	mu      sync.Mutex
	regime  string
	durable map[string]bool            // tool -> allowed (durable floor)
	session map[string]map[string]bool // traceID -> tool -> allowed (session overlay)
	epoch   uint64
	admits  []policyAdmission
	jnl     *journal.Journal
}

// policyAdmission records that a tool call was admitted under a rule epoch, so a
// later Revoke of that epoch can causally evict it.
type policyAdmission struct {
	traceID string
	tool    string
	epoch   uint64
	evicted bool
}

// NewPolicyRegime builds an empty regime that journals every applied op to jnl (nil ⇒
// journaling is inert, but adjudication and scoping still hold). regime names the
// initial named regime ("" ⇒ "default").
func NewPolicyRegime(regime string, jnl *journal.Journal) *PolicyRegime {
	if strings.TrimSpace(regime) == "" {
		regime = "default"
	}
	return &PolicyRegime{
		regime:  regime,
		durable: map[string]bool{},
		session: map[string]map[string]bool{},
		jnl:     jnl,
	}
}

// Apply adjudicates and, if admitted, applies one policy op scoped to traceID. It
// returns the adjudication verdict and a non-nil error ONLY when the op is refused
// (so a caller can branch on err) — a refused durable widen returns a
// VerdictRequireWitness carrying the closed UNWITNESSED reason. An applied op mints a
// fresh epoch and is journaled; a refused one is journaled as a DENY so the audit
// trail shows the gated attempt.
func (pr *PolicyRegime) Apply(traceID string, op PolicyOp) (abi.Verdict, error) {
	pr.mu.Lock()
	defer pr.mu.Unlock()

	switch op.Kind {
	case PolicyAddRules, PolicyRemoveRules, PolicySetRegime:
	default:
		return abi.Verdict{Kind: abi.VerdictDeny, Reason: abi.ReasonMalformed, By: "policy-regime"},
			fmt.Errorf("policy: unknown op kind %q (want add_rules|remove_rules|set_regime)", op.Kind)
	}

	scope := op.scope()
	// A durable WIDENING without a witness is refused, routed through the same
	// require-witness gate the kernel uses for an unwitnessed widening call.
	if scope == ScopeDurable && op.widens() && strings.TrimSpace(op.Witness) == "" {
		v := abi.Verdict{
			Kind:    abi.VerdictRequireWitness,
			Reason:  abi.ReasonUnwitnessed,
			By:      "policy-regime",
			Payload: abi.WitnessPayload{Claim: "durable policy widening requires a witness (operator token or human-principal event)"},
		}
		pr.journal(string(op.Kind), "", "DENY", abi.ReasonName(abi.ReasonUnwitnessed), pr.epoch)
		return v, fmt.Errorf("policy: durable widening op %q refused: %s", op.Kind, abi.ReasonName(abi.ReasonUnwitnessed))
	}

	// Admitted: mint the epoch that will be cited by everything this op newly allows.
	pr.epoch++
	verdictLabel := "ALLOW"
	if scope == ScopeDurable && op.widens() {
		verdictLabel = "WITNESS" // a witness-corroborated durable promotion
	}

	switch op.Kind {
	case PolicyAddRules:
		dst := pr.dst(traceID, scope)
		for _, r := range op.Rules {
			dst[r.Tool] = r.Allow
		}
	case PolicyRemoveRules:
		dst := pr.dst(traceID, scope)
		for _, r := range op.Rules {
			delete(dst, r.Tool)
		}
	case PolicySetRegime:
		if scope == ScopeDurable {
			pr.regime = op.Regime
		}
		// A session-scoped pivot is recorded for the trace only via the overlay
		// bookkeeping; the durable regime name is untouched so it cannot leak past
		// the session.
	}

	journalTrace := traceID
	if scope == ScopeDurable {
		journalTrace = "" // a durable op is not bound to a session
	}
	pr.journal(string(op.Kind), journalTrace, verdictLabel, "", pr.epoch)
	return abi.Verdict{Kind: abi.VerdictAllow, By: "policy-regime"}, nil
}

// dst returns the rule map an op at scope should mutate for traceID.
func (pr *PolicyRegime) dst(traceID string, scope PolicyScope) map[string]bool {
	if scope == ScopeDurable {
		return pr.durable
	}
	m := pr.session[traceID]
	if m == nil {
		m = map[string]bool{}
		pr.session[traceID] = m
	}
	return m
}

// IsAllowed reports whether tool is permitted for traceID under the effective floor:
// a live session overlay for the trace wins, else the durable floor, else fail-closed
// default-deny. It is the read side the admission path consults.
func (pr *PolicyRegime) IsAllowed(traceID, tool string) bool {
	pr.mu.Lock()
	defer pr.mu.Unlock()
	if m := pr.session[traceID]; m != nil {
		if allow, ok := m[tool]; ok {
			return allow
		}
	}
	if allow, ok := pr.durable[tool]; ok {
		return allow
	}
	return false
}

// EndSession expires every session-scoped rule for traceID — the automatic lifetime
// bound that makes a session-scoped widening safe without a witness. After it returns,
// the trace's overlay is gone and Dump shows no residue.
func (pr *PolicyRegime) EndSession(traceID string) {
	pr.mu.Lock()
	defer pr.mu.Unlock()
	delete(pr.session, traceID)
}

// RecordAdmission notes that a tool call for traceID was admitted under the current
// rule epoch, so a later Revoke of that epoch can causally evict it. It returns the
// epoch cited, so a caller can bind the admission to it.
func (pr *PolicyRegime) RecordAdmission(traceID, tool string) uint64 {
	pr.mu.Lock()
	defer pr.mu.Unlock()
	pr.admits = append(pr.admits, policyAdmission{traceID: traceID, tool: tool, epoch: pr.epoch})
	return pr.epoch
}

// Revoke causally evicts every admission cited under epoch or any LATER epoch — the
// fak_revoke primitive: when a rule epoch is refuted, everything admitted under it or
// under a rule minted after it is walked forward and evicted, while admissions from
// before the refuted epoch are untouched. It returns the number evicted.
func (pr *PolicyRegime) Revoke(epoch uint64) int {
	pr.mu.Lock()
	defer pr.mu.Unlock()
	n := 0
	for i := range pr.admits {
		if !pr.admits[i].evicted && pr.admits[i].epoch >= epoch {
			pr.admits[i].evicted = true
			n++
		}
	}
	return n
}

// LiveAdmissions returns the tools of admissions not yet evicted (sorted), the
// residue an operator sees after a Revoke.
func (pr *PolicyRegime) LiveAdmissions() []string {
	pr.mu.Lock()
	defer pr.mu.Unlock()
	var out []string
	for _, a := range pr.admits {
		if !a.evicted {
			out = append(out, a.tool)
		}
	}
	sort.Strings(out)
	return out
}

// Epoch returns the current rule epoch (the last minted), so a test or an admission
// path can cite it.
func (pr *PolicyRegime) Epoch() uint64 {
	pr.mu.Lock()
	defer pr.mu.Unlock()
	return pr.epoch
}

// Dump renders the effective floor in a deterministic, `fak policy --dump`-shaped
// form: the named regime, the durable allow/deny floor, and every LIVE session
// overlay. After a session ends its rules are absent — the residue check the
// acceptance names.
func (pr *PolicyRegime) Dump() string {
	pr.mu.Lock()
	defer pr.mu.Unlock()
	var b strings.Builder
	fmt.Fprintf(&b, "regime: %s\n", pr.regime)
	fmt.Fprintf(&b, "durable:\n")
	for _, tool := range policySortedKeys(pr.durable) {
		fmt.Fprintf(&b, "  %s -> %s\n", tool, allowWord(pr.durable[tool]))
	}
	for _, trace := range policySortedTraces(pr.session) {
		fmt.Fprintf(&b, "session[%s]:\n", trace)
		for _, tool := range policySortedKeys(pr.session[trace]) {
			fmt.Fprintf(&b, "  %s -> %s\n", tool, allowWord(pr.session[trace][tool]))
		}
	}
	return b.String()
}

// journal appends one policy-op row to the hash chain (a no-op when journaling is
// off). The caller holds pr.mu.
func (pr *PolicyRegime) journal(verb, traceID, verdict, reason string, epoch uint64) {
	if pr.jnl == nil {
		return
	}
	pr.jnl.AppendPolicyOp(verb, traceID, verdict, reason, epoch)
}

func allowWord(allow bool) string {
	if allow {
		return "allow"
	}
	return "deny"
}

func policySortedKeys(m map[string]bool) []string {
	out := maputil.SortedKeys(m)
	return out
}

func policySortedTraces(m map[string]map[string]bool) []string {
	out := maputil.SortedKeys(m)
	return out
}
