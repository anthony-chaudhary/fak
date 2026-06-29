package syspromptmmu

import (
	"crypto/sha256"
	"encoding/hex"

	"github.com/anthony-chaudhary/fak/internal/cachemeta"
)

// Tier classifies a base-context segment by its mutability and residency contract. The
// order is the head→tail layout order the attention geometry forces: the spine leads,
// the policy floor follows, and the overlay (this rung does not emit it) is appended
// after the Rung-2 cache breakpoint.
type Tier int

const (
	// TierSpine is fak's irreducible concepts — immutable, always resident, the
	// attention-sink anchor that is byte-identical every turn.
	TierSpine Tier = iota
	// TierPolicy is the deny/allow rules + safety-critical instructions — resident and
	// versioned, changing only at a marked cache breakpoint, never mid-prefix.
	TierPolicy
	// TierOverlay is the queried harness layer — paged capability cards filled by
	// Rung 3 (#1261) after the breakpoint. This rung emits no overlay segments.
	TierOverlay
)

// String renders a Tier as its stable lowercase token (used in PlanDigest and the
// Rung-6 observability surface).
func (t Tier) String() string {
	switch t {
	case TierSpine:
		return "spine"
	case TierPolicy:
		return "policy"
	case TierOverlay:
		return "overlay"
	default:
		return "unknown"
	}
}

// Version stamps for the two authored tiers. The spine and policy floor are
// version-stamped, never templated per-turn; a content edit is a deliberate version
// bump, and the content-derived Witness (see WitnessFor) detects any drift
// independently of the stamp.
const (
	SpineVersion  = "v1"
	PolicyVersion = "v1"
)

// witnessPrefix labels the Witness as a content blob hash (not a git object id), so a
// reader knows how to re-derive it: WitnessFor(seg.Content) == seg.Witness.
const witnessPrefix = "blob-sha256:"

// Segment is one tier-classified piece of the base-context plan: the wire-neutral
// cachemeta.PromptSegment (embedded, so Kind/Content/Witness promote) plus the Tier
// that governs its residency. BaseContextPlan returns the flat
// []cachemeta.PromptSegment projection a splicer (Rung 2) consumes; BaseContext
// returns these tier-tagged segments so Rung 4 can enforce residency without
// re-deriving the tier.
type Segment struct {
	Tier Tier
	cachemeta.PromptSegment
}

// authored is one base-context block as written here: the canonical fak text plus its
// tier. The witness and token estimate are derived deterministically from the content.
type authored struct {
	tier    Tier
	content string
}

// baseContext is the ordered, authored base context: fak's spine then the policy
// floor. This is the one genuinely new authorship surface — fak's own system prompt.
// The order is load-bearing (head→tail) and the contents are fixed literals, so the
// emitted plan is byte-identical every call (invariant 1).
var baseContext = []authored{
	{TierSpine, spineIdentity},
	{TierSpine, spineGate},
	{TierSpine, spineJournal},
	{TierSpine, spineCapability},
	{TierPolicy, policyDenyFloor},
	{TierPolicy, policySafetyResident},
}

// The spine: fak's irreducible concepts (the gate, the journal, what a capability is),
// authored once and never templated per-turn.
const (
	spineIdentity = "You operate inside fak, an agent kernel: one process that sits between you and the " +
		"tools you call and adjudicates every tool call before it runs. fak is the irreducible " +
		"head of your context — these concepts are always resident and never change per turn."

	spineGate = "The gate: every tool call is adjudicated before it executes. fak denies by structure " +
		"(a default-deny capability floor you cannot talk past), repairs malformed calls, and " +
		"quarantines poisoned results. A denied call is refused by structure, not by persuasion."

	spineJournal = "The journal: every decision is appended to a hash-chained, tamper-evident decision " +
		"journal. A claim counts as true only when a witness in the journal corroborates it; a " +
		"self-reported success with no witness is not yet done."

	spineCapability = "A capability is a named, versioned affordance — a skill, an MCP tool, or an A2A " +
		"agent — that the gate may admit. Capabilities are queried by intent, not menu-dumped; " +
		"their bodies are paged in on demand and evicted under pressure."
)

// The policy floor: the deny/allow rules + safety-critical instructions. Resident and
// versioned; it changes only at a marked cache breakpoint, never mid-prefix.
const (
	policyDenyFloor = "Policy floor: the deployed capability manifest is default-deny. Any tool call " +
		"outside the granted allow set is refused. The floor is versioned and resident; it is " +
		"never relaxed mid-session and never paged out."

	policySafetyResident = "Safety-critical instructions are always resident — never paged, never " +
		"compressed, never summarized. Anything load-bearing for the deny floor or fak's identity " +
		"stays in the spine or policy tier and is excluded from the evictable set by construction."
)

// WitnessFor returns the content-derived trust witness for a segment's bytes: a labeled
// blob hash a later turn re-derives to prove the segment is byte-unchanged (the
// spine-unchanged proof invariant 1 relies on). Deterministic by construction.
func WitnessFor(content []byte) string {
	sum := sha256.Sum256(content)
	return witnessPrefix + hex.EncodeToString(sum[:])
}

// estTokens is a deterministic, length-proportional token estimate (≈4 chars/token, the
// house heuristic in cachemeta.estTokens). The real provider-billed count is the
// tokenizer's job at splice time (Rung 2); this rung produces a plan, not wire bytes,
// so it carries an estimate, not a measured count.
func estTokens(content []byte) int64 {
	if n := int64(len(content)) / 4; n > 0 {
		return n
	}
	if len(content) > 0 {
		return 1
	}
	return 0
}

// NonEvictable reports whether a tier is pinned resident (never paged out). The spine
// and policy tiers are non-evictable substrate (invariant 3); only the overlay tier is
// evictable. This rung only FLAGS the contract — Rung 4 (#1262) enforces it live.
func NonEvictable(t Tier) bool {
	return t == TierSpine || t == TierPolicy
}

// BaseContext returns the ordered, tier-classified base-context plan: fak's SegStable
// spine (TierSpine) followed by the versioned policy floor (TierPolicy). The overlay
// tier is not emitted here — it is the queried harness layer Rung 3 (#1261) appends
// after the Rung-2 cache breakpoint. Deterministic: same inputs → byte-identical
// segment contents (invariant 1).
func BaseContext() []Segment {
	out := make([]Segment, len(baseContext))
	for i, a := range baseContext {
		content := []byte(a.content)
		out[i] = Segment{
			Tier: a.tier,
			PromptSegment: cachemeta.PromptSegment{
				Kind:    cachemeta.SegStable,
				Tokens:  estTokens(content),
				Content: content,
				Witness: WitnessFor(content),
			},
		}
	}
	return out
}

// BaseContextPlan returns the base-context plan as the flat, wire-neutral
// []cachemeta.PromptSegment a splicer (Rung 2, #1260) realizes into wire bytes. It is
// the embedded-PromptSegment projection of BaseContext. No wire mutation, no provider
// call — a plan only, byte-identical every call.
func BaseContextPlan() []cachemeta.PromptSegment {
	segs := BaseContext()
	out := make([]cachemeta.PromptSegment, len(segs))
	for i, s := range segs {
		out[i] = s.PromptSegment
	}
	return out
}

// PlanDigest returns a stable digest over the whole base-context plan — the ordered
// (tier, kind, witness) tuple of every segment, NUL-separated so no concatenation
// aliases another. It is the spine-unchanged proof at the plan level (Rung 6
// observability re-derives it and compares; any drift is an alarm) and the golden the
// determinism test pins.
func PlanDigest() string {
	h := sha256.New()
	for _, s := range BaseContext() {
		h.Write([]byte(s.Tier.String()))
		h.Write([]byte{0})
		h.Write([]byte(s.Kind))
		h.Write([]byte{0})
		h.Write([]byte(s.Witness))
		h.Write([]byte{0})
	}
	return witnessPrefix + hex.EncodeToString(h.Sum(nil))
}
