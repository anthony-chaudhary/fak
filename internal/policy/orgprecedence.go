// orgprecedence.go is the org/operator PRECEDENCE FOLD (W4 of epic #5315, issue
// #5322) — the rule that decides what the floor actually is once three channels
// have each had an opinion about it, and the record of WHICH ONE won.
//
// The three preceding children built the plumbing and stopped short of this:
// #5319 declared the `central` channel in the amendment registry, #5320 proved an
// envelope authentic, #5321 pulled and aged one, #5323 pinned the anchor that
// authorizes it. Every one of them answers "may this document speak?" None of
// them answers the question an operator actually has once it has spoken: given a
// compiled-in floor, a signed org overlay and a local operator overlay that all
// touch the same knob, WHICH VALUE RUNS — and who do I go to if I want it
// changed?
//
// The lattice, most authority first:
//
//	compiled-in FROZEN floor  >  central org overlay  >  operator overlay  >  agent-self
//
// It is NOT a simple override chain, and reading it as one is the mistake this
// file exists to prevent. Authority here means "caps", not "wins":
//
//   - A RATCHET knob resolves DENY if ANY channel denies. Authority does not let
//     a higher channel UN-deny a lower one — an operator who has locked their own
//     box down further than the org requires keeps that lock, because the org's
//     job is a floor, not a ceiling on caution.
//
//   - A GATED_WIDEN cap resolves to the MINIMUM of every ceiling set, each
//     clamped by the channel above it. So central may raise a device up to (never
//     past) the compiled cap — the IT-enable-more path — and the operator may
//     lower it further but cannot climb back past what central granted.
//
//   - A FROZEN knob resolves to its compiled-in value and nothing else moves it.
//     Central contributing to a FROZEN knob is not an error to swallow: it is
//     recorded as a no-authority note, because a central plane trying to move the
//     floor is exactly the event an auditor wants surfaced.
//
// Two questions the R3 research note (#5318) singled out, both answered NO by the
// min-fold above and both pinned as their own cases in org_precedence_test.go:
// central cannot raise a cap the operator lowered, and the operator cannot widen
// past a central grant.
//
// The file has two layers, and they are deliberately separate:
//
//	ResolveOrgPrecedence  — the pure per-knob lattice over abstract channel
//	                        contributions. This is the SEMANTIC authority; the
//	                        executable spec in org_precedence_test.go enumerates
//	                        it row by row.
//	FoldOrgProvenance     — attribution over three REAL assembled floors, built on
//	                        DiffAmendment so it cannot disagree with the engine
//	                        that governs admission everywhere else.
//
// FoldOrgProvenance ATTRIBUTES and REPORTS; it does not assemble. It is handed
// the stage snapshots a real assembly produced and says who moved what. That
// split is on purpose: guard_policy_diff.go already learned that a report which
// re-derives its own copy of the layering rules can disagree with the guard it
// claims to describe. The refusals it surfaces (a central FROZEN move, an
// operator widening past a central grant) are for the ASSEMBLY SITE to act on —
// this layer names them, the assembler enforces them.
package policy

import "github.com/anthony-chaudhary/fak/internal/adjudicator"

// OrgAmendVerdict is the resolved boolean posture of a RATCHET or FROZEN knob.
type OrgAmendVerdict string

const (
	// OrgVerdictAllow: no channel refuses this knob.
	OrgVerdictAllow OrgAmendVerdict = "ALLOW"
	// OrgVerdictDeny: at least one channel with authority refuses it.
	OrgVerdictDeny OrgAmendVerdict = "DENY"
)

// Note effects — the CLOSED vocabulary for what a channel asked for and did not
// get. Every one of them is a thing the fold silently absorbed, which is exactly
// why it must be reported rather than dropped: an org admin who sets a ceiling of
// 300 against a compiled cap of 200 gets 200, and needs to know their grant did
// not land whole.
const (
	// OrgNoteClamped: the channel asked for a wider ceiling than the channel
	// above it permits, and was reduced to that ceiling.
	OrgNoteClamped = "clamped"
	// OrgNoteRefusedWiden: the channel tried to loosen a knob that only ever
	// tightens (RATCHET). The attempt changed nothing.
	OrgNoteRefusedWiden = "refused-widen"
	// OrgNoteNoAuthority: the channel contributed to a knob it may not move at
	// all (FROZEN, whose registry entry lists only compiled-in).
	OrgNoteNoAuthority = "no-authority"
)

// OrgKnobContribution is ONE channel's opinion about one knob.
//
// The two halves are used by different amendment classes and never both:
//
//   - RATCHET / FROZEN boolean knobs read Deny ("this channel adds a refusal")
//     and WidenAttempt ("this channel tried to remove one").
//   - GATED_WIDEN cap knobs read Set and Cap. Set=false means INHERIT — the
//     channel expressed no ceiling and drops out of the min-fold entirely. It
//     does NOT mean "a ceiling of zero"; a channel that stays quiet must not
//     tighten a knob by accident.
type OrgKnobContribution struct {
	Deny         bool
	WidenAttempt bool
	Set          bool
	Cap          int
}

// contributed reports whether this channel said anything at all — the test for
// whether a FROZEN knob should record a no-authority note against it.
func (c OrgKnobContribution) contributed() bool {
	return c.Deny || c.WidenAttempt || c.Set
}

// OrgPrecedenceInput is one knob's full picture: its declared amendment class
// plus the three reachable channels' contributions.
//
// agent-self is absent by construction, not by omission: AmendSelfAmendable is
// empty in the registry today, so there is no knob the wrapped agent may move on
// its own and nothing for a fourth contribution to carry.
type OrgPrecedenceInput struct {
	Class      AmendmentClass
	CompiledIn OrgKnobContribution
	Central    OrgKnobContribution
	Operator   OrgKnobContribution
}

// OrgPrecedenceNote is one absorbed ask: which channel, what the fold did to it,
// and — for a clamp — the numbers, so a report can say "asked 300, got 200"
// instead of the useless "clamped".
type OrgPrecedenceNote struct {
	Channel string
	Effect  string
	Asked   int
	Got     int
}

// OrgPrecedenceResolution is the resolved knob plus its provenance.
//
// Channel is the load-bearing field for #5322: it names WHICH channel's value is
// the one running, which is the difference between an operator who can fix their
// own problem and one who has to file a ticket with IT.
type OrgPrecedenceResolution struct {
	Verdict OrgAmendVerdict
	Cap     int
	Channel string
	Notes   []OrgPrecedenceNote
}

// orgChannelOrder is the fold order: most authority first. Every walk in this
// file uses it, so "authority" has exactly one definition in one place.
var orgChannelOrder = []string{ChannelCompiledIn, ChannelCentral, ChannelOperatorOverlay}

// contributionsInOrder pairs each channel name with its contribution, in
// authority order.
func (in OrgPrecedenceInput) contributionsInOrder() []OrgKnobContribution {
	return []OrgKnobContribution{in.CompiledIn, in.Central, in.Operator}
}

// ResolveOrgPrecedence folds one knob's three channel contributions into the
// value that runs, plus the provenance and the absorbed asks.
//
// An unrecognized class — including AmendSelfAmendable, which is declared but
// empty — falls through to the FROZEN arm. That is deliberate fail-closed
// behavior: a knob whose class this fold does not understand must resolve to the
// compiled-in floor, never to whichever channel spoke last. A new amendment class
// that wants different handling has to come here and say so.
func ResolveOrgPrecedence(in OrgPrecedenceInput) OrgPrecedenceResolution {
	switch in.Class {
	case AmendRatchet:
		return resolveOrgRatchet(in)
	case AmendGatedWiden:
		return resolveOrgGatedWiden(in)
	default:
		return resolveOrgFrozen(in)
	}
}

// resolveOrgRatchet: DENY iff ANY channel denies, and no channel may un-deny a
// peer. Provenance names the HIGHEST-authority denier — that is the one an
// operator has to go through to change it, and naming a lower one would send them
// to someone who cannot help.
func resolveOrgRatchet(in OrgPrecedenceInput) OrgPrecedenceResolution {
	out := OrgPrecedenceResolution{Verdict: OrgVerdictAllow, Channel: ChannelCompiledIn}
	for i, c := range in.contributionsInOrder() {
		name := orgChannelOrder[i]
		if c.Deny && out.Verdict != OrgVerdictDeny {
			out.Verdict = OrgVerdictDeny
			out.Channel = name
		}
		// A widen attempt on a tighten-only knob is always refused, whether or not
		// anything is currently denying: a channel asking to loosen a ratchet has
		// misunderstood the knob, and swallowing that silently is how a fleet ends
		// up believing a grant landed.
		if c.WidenAttempt {
			out.Notes = append(out.Notes, OrgPrecedenceNote{Channel: name, Effect: OrgNoteRefusedWiden})
		}
	}
	return out
}

// resolveOrgGatedWiden: the cap is the MINIMUM ceiling any channel set, walked in
// authority order so each channel is clamped by the one above it.
//
// The walk, not a plain min(), is what produces honest provenance. A channel that
// asks for MORE than the running ceiling does not lower it and does not own it —
// it gets a clamp note. A channel that restates the ceiling exactly does not steal
// ownership from the higher-authority channel that set it. Only a channel that
// genuinely lowers the ceiling becomes its source.
//
// With NO channel setting a ceiling at all the result is 0 attributed to
// compiled-in: a GATED_WIDEN knob's zero value is already its tightest posture
// (DirectionWidenOnly), so "nobody asked to widen" resolves to "not widened".
func resolveOrgGatedWiden(in OrgPrecedenceInput) OrgPrecedenceResolution {
	out := OrgPrecedenceResolution{Verdict: OrgVerdictAllow, Channel: ChannelCompiledIn}
	bounded := false
	for i, c := range in.contributionsInOrder() {
		name := orgChannelOrder[i]
		// A cap knob can still carry a refusal from a channel that denies the tool
		// outright. Folding it in keeps the resolution fail-closed rather than
		// reporting ALLOW for a knob some channel has shut.
		if c.Deny && out.Verdict != OrgVerdictDeny {
			out.Verdict = OrgVerdictDeny
			out.Channel = name
		}
		if !c.Set {
			continue
		}
		switch {
		case !bounded:
			out.Cap, bounded, out.Channel = c.Cap, true, name
		case c.Cap < out.Cap:
			out.Cap, out.Channel = c.Cap, name
		case c.Cap > out.Cap:
			out.Notes = append(out.Notes, OrgPrecedenceNote{
				Channel: name, Effect: OrgNoteClamped, Asked: c.Cap, Got: out.Cap,
			})
		}
	}
	return out
}

// resolveOrgFrozen: the compiled-in value, full stop.
//
// Every other channel's contribution is recorded as no-authority rather than
// dropped. A central plane pushing a manifest that touches the FROZEN floor is
// not a harmless no-op — it is either a misconfigured org or an attempt on the
// one part of the floor that is supposed to be unreachable, and both are things
// an auditor has to be able to see happened.
func resolveOrgFrozen(in OrgPrecedenceInput) OrgPrecedenceResolution {
	out := OrgPrecedenceResolution{
		Verdict: OrgVerdictAllow,
		Cap:     in.CompiledIn.Cap,
		Channel: ChannelCompiledIn,
	}
	if in.CompiledIn.Deny {
		out.Verdict = OrgVerdictDeny
	}
	for i, c := range in.contributionsInOrder() {
		if i == 0 || !c.contributed() {
			continue
		}
		out.Notes = append(out.Notes, OrgPrecedenceNote{Channel: orgChannelOrder[i], Effect: OrgNoteNoAuthority})
	}
	return out
}

// ---------------------------------------------------------------------------
// Attribution over three real assembled floors
// ---------------------------------------------------------------------------

// OrgStages are the three floor SNAPSHOTS a real assembly passes through, each
// one the previous plus that channel's overlay:
//
//	CompiledIn — the shipped floor, before any overlay
//	Central    — CompiledIn with the verified org envelope applied
//	Operator   — Central with the local operator overlay applied
//
// The fold is handed the snapshots rather than the overlays because it must
// describe the floor that actually assembled. An attribution layer that applied
// its own copy of the overlays could disagree with the guard it claims to
// describe — the same trap guard_policy_diff.go avoids by reusing the launch
// loader instead of re-deriving the layering.
//
// CentralApplied distinguishes "the org overlay ran and changed nothing" from
// "there is no org authority on this box". They produce identical snapshots and
// mean opposite things: with no central authority an operator widening is
// ordinary local configuration (today's behavior, which the epic requires stay
// byte-for-byte unchanged), and under one it is a widening past the org grant.
type OrgStages struct {
	CompiledIn     adjudicator.Policy
	Central        adjudicator.Policy
	Operator       adjudicator.Policy
	CentralApplied bool
}

// OrgKnobProvenance is one knob's answer to "who set this?".
//
// Changes carries the rendered field-level movement from the owning channel, so a
// report can show what moved without the caller re-diffing. Widened records the
// DIRECTION the owning channel moved it, because "central set this" reads very
// differently depending on whether central tightened or loosened.
type OrgKnobProvenance struct {
	Field   string
	Class   AmendmentClass
	Channel string
	Widened bool
	Changes []AmendmentChange
}

// OrgFold is the attributed posture of one assembly.
//
// The three Central* buckets are the audit surface: CentralWiden is what #5322
// requires journaled with issuer and envelope version, CentralTighten is the
// fleet-wide ratchet landing, and CentralRefused is a central manifest reaching
// for a knob no central manifest may move.
//
// OperatorPastCentral is populated ONLY when CentralApplied — see OrgStages for
// why the distinction is load-bearing. It is a REPORT for the assembly site to
// refuse on; this layer cannot un-apply an overlay it was handed after the fact.
type OrgFold struct {
	Knobs               []OrgKnobProvenance
	CentralWiden        []AmendmentChange
	CentralTighten      []AmendmentChange
	CentralRefused      []AmendmentChange
	OperatorPastCentral []AmendmentChange
}

// CentralMoved reports whether the org overlay changed the floor at all —
// including a refused reach at a FROZEN knob, which moved nothing but is still
// the org plane having acted.
func (f OrgFold) CentralMoved() bool {
	return len(f.CentralWiden) > 0 || len(f.CentralTighten) > 0 || len(f.CentralRefused) > 0
}

// FoldOrgProvenance attributes each field-backed knob in PolicyKnobRegistry to
// the channel whose overlay last moved it, and buckets the central-channel
// movement for the audit trail.
//
// Attribution is LAST-WRITER, walking down the authority order: if the operator
// overlay moved a knob, the operator owns it; else if the central overlay did,
// central owns it; else it stands at the compiled-in floor. That is the honest
// reading of a snapshot chain — the value running is the one the last stage left
// — and it is why the fold reports OperatorPastCentral separately rather than
// pretending the clamp already happened.
//
// Every field-backed knob is returned, including untouched ones. A report that
// listed only the moved knobs would make "this knob is at the shipped floor" and
// "this knob is not in the registry" look identical, and the second is a bug.
func FoldOrgProvenance(st OrgStages) OrgFold {
	central := DiffAmendment(st.CompiledIn, st.Central)
	operator := DiffAmendment(st.Central, st.Operator)

	var out OrgFold
	out.CentralWiden = central.Widen
	out.CentralTighten = central.Tighten
	// DiffAmendment routes a change to an unknown OR FROZEN field into Frozen —
	// fail closed, so an unclassified knob can never be read as an ordinary
	// amendment. Both readings are a refusal here for the same reason.
	out.CentralRefused = central.Frozen
	if st.CentralApplied {
		out.OperatorPastCentral = operator.Widen
	}

	centralByField := changesByField(central)
	operatorByField := changesByField(operator)
	centralWidened := widenedFields(central)
	operatorWidened := widenedFields(operator)

	for _, knob := range PolicyKnobRegistry {
		if knob.Field == "" {
			// A non-field floor element (the hardwired egress SSRF floor and its
			// siblings) has no struct field to diff, so no snapshot comparison can
			// speak to it. It is compiled-in by definition and is listed in the
			// registry precisely so the FROZEN set is enumerated in one place —
			// there is nothing to attribute.
			continue
		}
		p := OrgKnobProvenance{Field: knob.Field, Class: knob.Class, Channel: ChannelCompiledIn}
		switch {
		case len(operatorByField[knob.Field]) > 0:
			p.Channel, p.Changes, p.Widened = ChannelOperatorOverlay, operatorByField[knob.Field], operatorWidened[knob.Field]
		case len(centralByField[knob.Field]) > 0:
			p.Channel, p.Changes, p.Widened = ChannelCentral, centralByField[knob.Field], centralWidened[knob.Field]
		}
		out.Knobs = append(out.Knobs, p)
	}
	return out
}

// changesByField indexes every bucket of a delta by the field that moved. All
// three buckets are folded in: a FROZEN-bucket change still moved the field, and
// leaving it out would attribute a refused central reach to compiled-in — the one
// channel that definitely did not make it.
func changesByField(d AmendmentDelta) map[string][]AmendmentChange {
	out := map[string][]AmendmentChange{}
	for _, bucket := range [][]AmendmentChange{d.Tighten, d.Widen, d.Frozen} {
		for _, c := range bucket {
			out[c.Field] = append(out[c.Field], c)
		}
	}
	return out
}

// widenedFields reports which fields a delta LOOSENED. A field carrying both a
// widen and a tighten (a rule edited into a different rule reads as
// removed+added) counts as widened: the loosening half is the part that needs a
// grant, so the mixed case must not round down to "tightened".
func widenedFields(d AmendmentDelta) map[string]bool {
	out := map[string]bool{}
	for _, c := range d.Widen {
		out[c.Field] = true
	}
	return out
}
