// Durability is the memory-write half of the grammar rung: the domain-free
// "context is not memory" contract. Where Grammar/MessageType name the SHAPE of a
// payload (arguments to a tool, fields to a peer), DurabilityClass names how LONG a
// write may be believed — and the promotion predicate decides whether a write may
// advance to a longer-lived storage tier. It is the standalone, package-vocabulary-
// free lift of fak's recall promotion gate (internal/recall PromotionWarn ->
// PromotionEnforce over a ctxmmu-stamped Page.Durability): any persistent-agent
// runtime can satisfy THIS contract without importing fak's recall/ctxmmu types.
//
// Two pieces, both pure:
//   - A CLOSED durability vocabulary {turn, session, durable, bounded}, validated in
//     the dos_check_reason style: a known/unknown membership check, so an
//     unrecognized class is REJECTED, not silently tolerated (the TypeRegistry
//     fail-closed discipline, pointed at a class instead of a type).
//   - A promotion PREDICATE: given a write's STAMPED durability class, may it advance
//     to a longer-lived tier? FAIL-CLOSED — an unclassified or non-`durable` write
//     does NOT promote by default. It mirrors recall's audit-only -> enforce split
//     with a PromotionMode flag (warn vs enforce), so callers can be audited before
//     the boundary bites.
//
// Evidence-bound (acceptance #3): the class is an INPUT, stamped at write time by a
// declared classifier and handed to MayPromote. The predicate NEVER infers a class
// from the write's content — it cannot upgrade an unclassified write to `durable` by
// reading the bytes. A missing/unknown stamp fails closed exactly like a refused
// reason, never like a believed self-report.
package grammar

// DurabilityClass is one member of the closed durability vocabulary — how long a
// memory write may be believed. It is a stamped INPUT to the promotion predicate,
// never a value the predicate infers from a write's content.
type DurabilityClass string

// The closed durability vocabulary. These four values are the ONLY recognized
// classes; anything else is unknown and fails the membership check (KnownDurability),
// the way a reason outside the closed refusal set fails dos_check_reason.
const (
	// DurabilityTurn: true only for this turn — the fail-closed default. Not promotable.
	DurabilityTurn DurabilityClass = "turn"
	// DurabilitySession: true for this session only. Not promotable across sessions.
	DurabilitySession DurabilityClass = "session"
	// DurabilityDurable: true across sessions until revised — the ONLY promotable
	// class. A `durable` stamp is the earned exception the gate promotes.
	DurabilityDurable DurabilityClass = "durable"
	// DurabilityBounded: true until an explicit validity boundary (a TTL / as-of
	// interval) elapses. A KNOWN member of the vocabulary, but NOT promotable by
	// default: without an evaluated validity interval it is treated like turn/session
	// at the durable boundary (it has no validity home that the predicate can check),
	// so it expires rather than crossing into the persisted image.
	DurabilityBounded DurabilityClass = "bounded"
)

// durabilityVocabulary is the closed set, in most-durable-first order. It is the
// single source of truth both KnownDurability and DurabilityVocabulary read, so the
// "closed set" claim is one list, not two that can drift.
var durabilityVocabulary = []DurabilityClass{
	DurabilityDurable,
	DurabilitySession,
	DurabilityBounded,
	DurabilityTurn,
}

// DurabilityVocabulary returns a copy of the closed durability vocabulary — the four
// recognized classes. Callers can enumerate the legal set (e.g. to render a help
// page) without being able to mutate it.
func DurabilityVocabulary() []DurabilityClass {
	out := make([]DurabilityClass, len(durabilityVocabulary))
	copy(out, durabilityVocabulary)
	return out
}

// KnownDurability reports whether class is a member of the closed durability
// vocabulary — the dos_check_reason-style membership check. An unrecognized class
// returns false (rejected, not tolerated): a caller MUST treat known=false as a
// refusal, never as a believed class. This is the companion to MayPromote the way
// dos_check_reason is the companion to a refusal — emit/promote only what this
// returns true for.
func KnownDurability(class DurabilityClass) bool {
	switch class {
	case DurabilityTurn, DurabilitySession, DurabilityDurable, DurabilityBounded:
		return true
	default:
		return false
	}
}

// PromotionMode is the promotion gate's posture, mirroring recall's
// PromotionWarn -> PromotionEnforce honesty split. The same predicate is evaluated in
// both modes; the mode only changes whether a non-promotable verdict BITES (enforce)
// or is merely recorded for audit while the write still persists (warn). This lets a
// runtime classify and count would-refusals before the durable boundary actually
// blocks a write.
type PromotionMode uint8

const (
	// PromotionWarn is the audit-only, non-behavior-changing default (the zero value):
	// the predicate computes whether a write WOULD promote, but the verdict's Promote
	// bit does not block — a caller in warn mode persists every write and uses the
	// verdict only to count would-refusals. Safe while callers are audited for the
	// enforce flip.
	PromotionWarn PromotionMode = iota
	// PromotionEnforce makes the gate bite: only a write stamped `durable` promotes to
	// the longer-lived tier. A turn / session / bounded / unclassified / unknown write
	// is refused, so a transient or unvalidated write can never silently become a
	// persistent fact (the benign over-promotion arm of OWASP Memory-Poisoning T1).
	PromotionEnforce
)

// String renders a PromotionMode for diagnostics.
func (m PromotionMode) String() string {
	switch m {
	case PromotionEnforce:
		return "enforce"
	case PromotionWarn:
		return "warn"
	default:
		return "warn"
	}
}

// PromotionVerdict is the typed result of the promotion predicate — the closed,
// peer-readable answer to "may this write advance to a longer-lived tier?". It is a
// pure function of (stamped class, mode): no field is inferred from a write's content.
type PromotionVerdict struct {
	// Class is the durability class AS STAMPED on the write (echoed back verbatim,
	// including an unknown value), so the verdict is self-describing for audit.
	Class DurabilityClass `json:"class"`
	// Known is the membership check: false iff Class is outside the closed vocabulary.
	// An unknown class is conservatively non-promotable regardless of mode.
	Known bool `json:"known"`
	// Promote is the gate's decision in ENFORCE terms: whether the write is allowed to
	// advance to the longer-lived tier. True ONLY for a known `durable` class. In warn
	// mode a caller MAY still persist a non-promoting write, but Promote reports what
	// enforce WOULD do, so the would-refusal is always countable.
	Promote bool `json:"promote"`
	// Mode is the posture the verdict was computed under (echoed for audit).
	Mode PromotionMode `json:"mode"`
	// Reason is a short, stable token explaining a non-promotion, drawn from a closed
	// set: "" when Promote is true, else one of unknown_class / non_durable. It is the
	// structured-refusal half (a verifiable token, not free-text prose).
	Reason string `json:"reason,omitempty"`
}

// Promotion reason tokens — the closed set Reason draws from. A non-promotion always
// names exactly one; a promotion carries none.
const (
	// PromoteReasonUnknownClass: the stamped class is outside the closed vocabulary.
	PromoteReasonUnknownClass = "unknown_class"
	// PromoteReasonNonDurable: a KNOWN class, but not `durable` (turn/session/bounded).
	PromoteReasonNonDurable = "non_durable"
)

// MayPromote is the promotion predicate: given a write's STAMPED durability class and
// the gate mode, decide whether the write may advance to a longer-lived tier. It is a
// pure function — the class is an evidence-bound INPUT, never inferred from content.
//
// FAIL-CLOSED contract:
//   - An UNCLASSIFIED write (the empty class "") or any value outside the closed
//     vocabulary is unknown -> NEVER promotes (Reason unknown_class), in either mode.
//   - A KNOWN but non-`durable` class (turn / session / bounded) -> does not promote
//     (Reason non_durable).
//   - ONLY a known `durable` class promotes (Promote true, no Reason).
//
// The mode does NOT change WHICH writes are promotable — it is recorded on the verdict
// so a warn-mode caller can persist a non-promoting write while still counting the
// would-refusal, and an enforce-mode caller can block it. Promote reflects the
// enforce decision in both modes; ShouldPersist folds the mode in for callers that
// want the single persist bit.
func MayPromote(class DurabilityClass, mode PromotionMode) PromotionVerdict {
	v := PromotionVerdict{Class: class, Mode: mode, Known: KnownDurability(class)}
	switch {
	case !v.Known:
		// Unclassified / unrecognized: fail closed, never promote, in any mode.
		v.Promote = false
		v.Reason = PromoteReasonUnknownClass
	case class == DurabilityDurable:
		v.Promote = true
	default:
		// Known but not durable (turn / session / bounded): the earned exception was
		// not earned.
		v.Promote = false
		v.Reason = PromoteReasonNonDurable
	}
	return v
}

// ShouldPersist folds the mode into a single persist bit for callers that want one
// answer to "does this write reach the longer-lived tier?". In PromotionEnforce only a
// promoting write persists (the boundary bites). In PromotionWarn the write ALWAYS
// persists (audit-only, non-behavior-changing) — the verdict's Promote bit and Reason
// still record what enforce WOULD have done, so the would-refusal is countable without
// changing behavior.
func (v PromotionVerdict) ShouldPersist() bool {
	if v.Mode == PromotionEnforce {
		return v.Promote
	}
	return true
}

// WouldRefuse reports whether the gate refused (or, in warn mode, WOULD have refused)
// to promote this write — i.e. a non-promotion. It is the auditable signal of the
// default-expire inversion: the count of WouldRefuse verdicts is the would-refusal
// tally a warn-mode caller accrues before flipping to enforce.
func (v PromotionVerdict) WouldRefuse() bool { return !v.Promote }
