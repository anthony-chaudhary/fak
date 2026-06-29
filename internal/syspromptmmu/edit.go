package syspromptmmu

import (
	"github.com/anthony-chaudhary/fak/internal/cachemeta"
)

// edit.go — Rung 5 of the system-prompt MMU (#1263, epic #1258): the live base. The
// base context becomes a first-class object fak rewrites at runtime — under a witness
// gate, append-mostly, never edit-in-place, and with the spine + edit-governing
// meta-rules OFF-LIMITS to the agent's own edits (invariant 5).
//
// This file is the SAFETY GATE core: the pure mechanism that decides whether a proposed
// self-edit may be applied, and applies it append-mostly over a COPY so the prior plan
// is a bit-for-bit rollback. It enforces, structurally:
//
//   - The agent never rewrites its own spine. Any edit whose target tier is TierSpine,
//     or whose target block is one of the immutable spine/policy-floor blocks (the
//     edit-governing meta-rules), is HARD-refused regardless of witness.
//   - The agent never grades its own edit. The success signal is an INJECTED witness
//     predicate (the caller runs `dos verify` / the guard witness and passes the
//     verdict) — syspromptmmu stays tier-2 and never imports the guard. A nil witness is
//     fail-closed: a resident-affecting edit with no witness bus is refused.
//   - Edits are versioned deltas, never a full rewrite (a full rewrite causes context
//     collapse + brevity bias that erodes safety text). ApplyEdit never mutates its
//     input; the previous plan IS the rollback.
//
// FENCE — deferred to the integrator follow-on (cmd/fak, tier 4): the `fak prompt`
// verbs, the persistent versioned store, the next-prefix-rebuild landing via
// internal/sessionreset, and the auto-demote loop that feeds outcome correlations from
// the guard-RSI signal. This file ships the gate those wire into; an edit applied here
// flows ApplyEdit -> BuildSystemValue -> the wire, and AuditRealizedPrefix (Rung 6)
// still reads AuditOK because the immutable spine is untouched.

// EditOp is the kind of base-context delta.
type EditOp int

const (
	// EditAdd appends a NEW resident/overlay block (a learned rule).
	EditAdd EditOp = iota
	// EditPromote brings a paged overlay block into the resident tier (modeled here as
	// an Add of the promoted content; the source overlay is the Rung-3 layer).
	EditPromote
	// EditDemote removes a learned (non-floor) block, making it non-resident.
	EditDemote
	// EditVersion replaces a learned block's content with a new version (the prior
	// version is preserved by the caller — ApplyEdit never mutates its input).
	EditVersion
)

// String renders an EditOp as a stable token (for journals and the Rung-6 view).
func (o EditOp) String() string {
	switch o {
	case EditAdd:
		return "add"
	case EditPromote:
		return "promote"
	case EditDemote:
		return "demote"
	case EditVersion:
		return "version"
	default:
		return "unknown"
	}
}

// BaseEdit is one versioned, append-mostly delta proposed against the live base.
type BaseEdit struct {
	// Op is the delta kind.
	Op EditOp
	// Tier is the TARGET tier. TierSpine is always refused; only TierPolicy (a learned
	// resident rule) and TierOverlay are editable.
	Tier Tier
	// Target identifies the existing block for Demote/Version — its content Witness
	// (WitnessFor of its bytes). Empty for Add/Promote (a new block).
	Target string
	// Content is the new block bytes (Add/Promote/Version). Ignored for Demote.
	Content []byte
	// Version is the version stamp the edit carries (informational; the content Witness
	// is the byte-level identity).
	Version string
}

// Closed set of edit verdicts. An un-applied edit ALWAYS names one, so a refusal is
// auditable — never a silent no-op.
const (
	EditOK               = "ok"                 // applied
	EditRefusedSpine     = "refused-spine"      // targets the spine — off-limits to the agent
	EditRefusedMetaRule  = "refused-meta-rule"  // targets an immutable policy-floor / edit-governing block
	EditRefusedNoWitness = "refused-no-witness" // no witness passed the change (fail-closed)
	EditRefusedNotFound  = "refused-not-found"  // Demote/Version target block is absent
	EditRefusedUnknownOp = "refused-unknown-op" // not a known EditOp
	EditRefusedNoContent = "refused-no-content" // Add/Promote/Version with empty content
)

// EditVerdict is the gated result of a proposed edit.
type EditVerdict struct {
	// Applied reports whether the edit was admitted.
	Applied bool
	// Reason names the verdict (a closed-set constant above).
	Reason string
}

// protectedWitnesses is the set of content witnesses that are OFF-LIMITS to a
// self-edit: every spine and policy-floor block fak authored (BaseContext). A learned
// rule (added via EditAdd) is NOT in this set, so it stays editable; the immutable
// substrate is not.
func protectedWitnesses() map[string]Tier {
	out := make(map[string]Tier)
	for _, s := range BaseContext() {
		out[s.Witness] = s.Tier
	}
	return out
}

// GateEdit is the pure admission decision for a proposed edit — structural refusals
// FIRST (they are absolute, witness or not), then the witness gate. It never looks at
// the live plan; ApplyEdit composes it with the apply step.
func GateEdit(edit BaseEdit, witness func(BaseEdit) bool) EditVerdict {
	switch edit.Op {
	case EditAdd, EditPromote, EditDemote, EditVersion:
	default:
		return EditVerdict{Reason: EditRefusedUnknownOp}
	}
	// The spine is never a legitimate target of a self-edit.
	if edit.Tier == TierSpine {
		return EditVerdict{Reason: EditRefusedSpine}
	}
	// Demote/Version target an existing block by witness; an immutable spine/floor block
	// is off-limits (the edit-governing meta-rules).
	if edit.Op == EditDemote || edit.Op == EditVersion {
		if t, protected := protectedWitnesses()[edit.Target]; protected {
			if t == TierSpine {
				return EditVerdict{Reason: EditRefusedSpine}
			}
			return EditVerdict{Reason: EditRefusedMetaRule}
		}
	}
	// Content-bearing ops need content.
	if edit.Op != EditDemote && len(edit.Content) == 0 {
		return EditVerdict{Reason: EditRefusedNoContent}
	}
	// The agent never grades its own edit: an independent witness must pass. A nil bus is
	// fail-closed.
	if witness == nil || !witness(edit) {
		return EditVerdict{Reason: EditRefusedNoWitness}
	}
	return EditVerdict{Applied: true, Reason: EditOK}
}

// ApplyEdit gates `edit` and, if admitted, applies it append-mostly to a COPY of `base`
// (the resident spine+policy plan as tier-tagged Segments). It NEVER mutates `base`, so
// the caller's prior value is a bit-for-bit rollback. On any refusal it returns `base`
// unchanged with the verdict.
//
// EditAdd/EditPromote append a new block at the edit's tier. EditVersion replaces the
// matching learned block's content (new content Witness). EditDemote drops the matching
// learned block. A Version/Demote whose target is absent is EditRefusedNotFound.
func ApplyEdit(base []Segment, edit BaseEdit, witness func(BaseEdit) bool) ([]Segment, EditVerdict) {
	v := GateEdit(edit, witness)
	if !v.Applied {
		return base, v
	}

	switch edit.Op {
	case EditAdd, EditPromote:
		seg := Segment{
			Tier: edit.Tier,
			PromptSegment: cachemeta.PromptSegment{
				Kind:    cachemeta.SegStable,
				Tokens:  estTokens(edit.Content),
				Content: append([]byte(nil), edit.Content...),
				Witness: WitnessFor(edit.Content),
			},
		}
		out := make([]Segment, 0, len(base)+1)
		out = append(out, base...)
		out = append(out, seg)
		return out, v

	case EditVersion:
		idx := indexByWitness(base, edit.Target)
		if idx < 0 {
			return base, EditVerdict{Reason: EditRefusedNotFound}
		}
		out := make([]Segment, len(base))
		copy(out, base)
		out[idx] = Segment{
			Tier: out[idx].Tier,
			PromptSegment: cachemeta.PromptSegment{
				Kind:    cachemeta.SegStable,
				Tokens:  estTokens(edit.Content),
				Content: append([]byte(nil), edit.Content...),
				Witness: WitnessFor(edit.Content),
			},
		}
		return out, v

	case EditDemote:
		idx := indexByWitness(base, edit.Target)
		if idx < 0 {
			return base, EditVerdict{Reason: EditRefusedNotFound}
		}
		out := make([]Segment, 0, len(base)-1)
		out = append(out, base[:idx]...)
		out = append(out, base[idx+1:]...)
		return out, v
	}
	return base, EditVerdict{Reason: EditRefusedUnknownOp}
}

// PlanOf projects tier-tagged base segments to the flat []cachemeta.PromptSegment a
// splicer (BuildSystemValue / SpliceSystemOverlay) and the auditor (AuditRealizedPrefix)
// consume — the general form of BaseContextPlan for an EDITED base.
func PlanOf(base []Segment) []cachemeta.PromptSegment {
	out := make([]cachemeta.PromptSegment, len(base))
	for i, s := range base {
		out[i] = s.PromptSegment
	}
	return out
}

// indexByWitness returns the index of the first segment whose content Witness matches,
// or -1.
func indexByWitness(base []Segment, witness string) int {
	for i, s := range base {
		if s.Witness == witness {
			return i
		}
	}
	return -1
}
