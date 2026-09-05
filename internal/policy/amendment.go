// amendment.go is the PolicyKnob amendment-class registry (#5171, epic #5170,
// Track A): the single machine-checked source of truth stating, for every
// exported adjudicator.Policy field (and each non-field compiled-in floor
// element), its amendment class — who, if anyone, may move that knob, and in
// which direction. The per-knob mutability discipline itself is enforced
// elsewhere (sanitizeProfile, sanitizeAdvisoryReasons, AdvisoryEligible,
// mustRun, diffPolicyWidening, protectGuardPolicyConfig); this table makes the
// model explicit so a new knob cannot land without declaring how it may be
// amended — the reflection conformance test in amendment_test.go fails on any
// unclassified field.
package policy

// AmendmentClass is the closed vocabulary of who may amend a policy surface.
type AmendmentClass string

const (
	// AmendFrozen: no channel may move it — the compiled-in floor.
	AmendFrozen AmendmentClass = "FROZEN"
	// AmendRatchet: any authorized channel may TIGHTEN it, none may widen.
	AmendRatchet AmendmentClass = "RATCHET"
	// AmendGatedWiden: a gated operator channel may widen it (overlay, reload,
	// operator escalation) — never the agent on its own.
	AmendGatedWiden AmendmentClass = "GATED_WIDEN"
	// AmendSelfAmendable: the agent-writable frontier. Empty today — no knob
	// is self-amendable; the class exists so the frontier is declared, not
	// implied by omission.
	AmendSelfAmendable AmendmentClass = "SELF_AMENDABLE"
)

// AmendmentDirection is which way a knob is allowed to move under its
// authorized channels.
type AmendmentDirection string

const (
	// DirectionFrozen: the knob cannot move at all.
	DirectionFrozen AmendmentDirection = "frozen"
	// DirectionTightenOnly: changes may only narrow what is admitted.
	DirectionTightenOnly AmendmentDirection = "tighten-only"
	// DirectionWidenOnly: the knob exists to (gatedly) loosen the floor;
	// its zero value is already the tightest posture.
	DirectionWidenOnly AmendmentDirection = "widen-only"
	// DirectionBidirectional: reserved for a future SELF_AMENDABLE knob that
	// may move either way under its declared channels. Unused today.
	DirectionBidirectional AmendmentDirection = "bidirectional"
)

// Amendment channel names — the closed set of mechanisms through which a
// policy surface can change. FROZEN knobs may declare only ChannelCompiledIn.
const (
	// ChannelCompiledIn: shipped default, changeable only by a code change.
	ChannelCompiledIn = "compiled-in"
	// ChannelOperatorOverlay: .fak/guard/{allow,deny}.json overlays / --policy.
	ChannelOperatorOverlay = "operator-overlay"
	// ChannelLiveReload: the gated live policy reload (reload-widen path).
	ChannelLiveReload = "live-reload"
	// ChannelOperatorEscalation: an explicit operator escalation grant.
	ChannelOperatorEscalation = "operator-escalation"
	// ChannelCentral: the org policy plane (epic #5315) — a signed, out-of-band
	// manifest pulled from a company endpoint that the wrapped agent cannot
	// reach. Its authority sits ABOVE the operator overlay and BELOW the
	// compiled-in FROZEN floor: it may RATCHET the floor fleet-wide and
	// GATED_WIDEN a knob per enrolled device up to (never past) its FROZEN cap,
	// so it appears on RATCHET and GATED_WIDEN knobs but NEVER on a FROZEN one.
	ChannelCentral = "central"
)

// PolicyKnob is one classified policy surface. Field is the exported
// adjudicator.Policy struct field name when the knob is field-backed, or ""
// for a non-field compiled-in floor element (e.g. the hardwired egress SSRF
// floor), which the registry still lists so the FROZEN set is enumerated in
// one place.
type PolicyKnob struct {
	Field     string
	Class     AmendmentClass
	Direction AmendmentDirection
	Channels  []string
	Doc       string
}

// PolicyKnobRegistry classifies EVERY exported adjudicator.Policy field
// (conformance-tested by reflection in amendment_test.go) plus the non-field
// FROZEN floor elements. Epic #5170 mapping:
//
//	FROZEN       — egress SSRF floor, reversibility gate, structural danger
//	               arg-rules, write-class self-modify rungs, AdvisoryEligible clamp
//	RATCHET      — tighten-only additive knobs (extra deny hosts, block lists,
//	               secret patterns, arg predicates, added self-modify globs, deny overlay)
//	GATED_WIDEN  — operator-gated loosening (allow overlay, reload-widen,
//	               Complain, AdvisoryReasons, Posture, RungProfile, AutoRepairSidestep)
//	SELF_AMENDABLE — empty today.
var PolicyKnobRegistry = []PolicyKnob{
	// ---- non-field FROZEN floor elements (Field == "", compiled-in only) ----
	{Field: "", Class: AmendFrozen, Direction: DirectionFrozen,
		Channels: []string{ChannelCompiledIn},
		Doc:      "Hardwired egress SSRF floor: the cloud-metadata / link-local refusal class no allow rule can un-block."},
	{Field: "", Class: AmendFrozen, Direction: DirectionFrozen,
		Channels: []string{ChannelCompiledIn},
		Doc:      "Reversibility preview-confirm gate on irreversible/outward calls; policy cannot disable it."},
	{Field: "", Class: AmendFrozen, Direction: DirectionFrozen,
		Channels: []string{ChannelCompiledIn},
		Doc:      "Structural danger argument rules (rm -rf / RCE-pipe / out-of-tree-write class)."},
	{Field: "", Class: AmendFrozen, Direction: DirectionFrozen,
		Channels: []string{ChannelCompiledIn},
		Doc:      "Write-class self-modify refusal rungs; sanitizeProfile/mustRun clamp any profile that tries to drop them."},
	{Field: "", Class: AmendFrozen, Direction: DirectionFrozen,
		Channels: []string{ChannelCompiledIn},
		Doc:      "AdvisoryEligible clamp: POLICY_BLOCK / SECRET_EXFIL / EGRESS_BLOCK can never be blanket-softened to advisory."},

	// ---- field-backed knobs (Field == the adjudicator.Policy field name) ----
	// ChannelCentral (epic #5315) is added to every RATCHET and GATED_WIDEN knob:
	// the org plane may tighten the floor fleet-wide (RATCHET) and gate-widen a
	// knob per enrolled device up to its FROZEN cap (GATED_WIDEN). It is never
	// added to a FROZEN knob — the compiled-in floor caps every central grant.
	{Field: "Posture", Class: AmendGatedWiden, Direction: DirectionWidenOnly,
		Channels: []string{ChannelOperatorOverlay, ChannelLiveReload, ChannelOperatorEscalation, ChannelCentral},
		Doc:      "Default-deny posture; fail_closed may loosen to admit_and_log for read-shaped default denies."},
	{Field: "Allow", Class: AmendGatedWiden, Direction: DirectionWidenOnly,
		Channels: []string{ChannelOperatorOverlay, ChannelLiveReload, ChannelCentral},
		Doc:      "Affirmative per-tool allowlist (allow overlay); adding a name widens what is admitted."},
	{Field: "AllowPrefix", Class: AmendGatedWiden, Direction: DirectionWidenOnly,
		Channels: []string{ChannelOperatorOverlay, ChannelLiveReload, ChannelCentral},
		Doc:      "Prefix allow family (read_/get_/search_/list_); adding a prefix widens the admitted set."},
	{Field: "Deny", Class: AmendRatchet, Direction: DirectionTightenOnly,
		Channels: []string{ChannelOperatorOverlay, ChannelLiveReload, ChannelCentral},
		Doc:      "Per-tool provable refusal map (deny overlay); adding an entry only ever tightens the floor."},
	{Field: "SelfModifyGlobs", Class: AmendRatchet, Direction: DirectionTightenOnly,
		Channels: []string{ChannelOperatorOverlay, ChannelCentral},
		Doc:      "SELF_MODIFY target globs; the compiled base set is frozen and the field only ever adds globs."},
	{Field: "BlockedPathGlobs", Class: AmendRatchet, Direction: DirectionTightenOnly,
		Channels: []string{ChannelOperatorOverlay, ChannelCentral},
		Doc:      "Credential and host configuration target globs; adding an entry tightens the floor."},
	{Field: "RedactFields", Class: AmendRatchet, Direction: DirectionTightenOnly,
		Channels: []string{ChannelOperatorOverlay, ChannelCentral},
		Doc:      "Arg keys stripped by TRANSFORM before dispatch; adding a key only tightens secret hygiene."},
	{Field: "ArgPredicates", Class: AmendRatchet, Direction: DirectionTightenOnly,
		Channels: []string{ChannelOperatorOverlay, ChannelLiveReload, ChannelCentral},
		Doc:      "Per-tool argument-value constraints; restrict-only — a rule can turn an allow into a deny, never grant one."},
	{Field: "LintWrites", Class: AmendRatchet, Direction: DirectionTightenOnly,
		Channels: []string{ChannelOperatorOverlay, ChannelCentral},
		Doc:      "Opt-in in-process write-lint rung (MALFORMED on unparseable Go/JSON); turning it on only tightens."},
	{Field: "Profile", Class: AmendGatedWiden, Direction: DirectionWidenOnly,
		Channels: []string{ChannelOperatorOverlay, ChannelCentral},
		Doc:      "RungProfile eliding convenience rungs per risk class; sanitizeProfile keeps write-class rungs mandatory."},
	{Field: "Complain", Class: AmendGatedWiden, Direction: DirectionWidenOnly,
		Channels: []string{ChannelOperatorOverlay, ChannelCentral},
		Doc:      "Per-tool admit-and-log trial set; downgrades only the default-deny rung to a logged allow."},
	{Field: "AdvisoryReasons", Class: AmendGatedWiden, Direction: DirectionWidenOnly,
		Channels: []string{ChannelOperatorOverlay, ChannelCentral},
		Doc:      "Per-reason advisory (warn) downgrade for heuristic rungs; clamped to AdvisoryEligible."},
	{Field: "SecretPosture", Class: AmendGatedWiden, Direction: DirectionWidenOnly,
		Channels: []string{ChannelOperatorOverlay, ChannelCentral},
		Doc:      "On-discovery secret rung posture (quarantine default); may loosen to admit_and_log for read-shaped results."},
	{Field: "SecretPatterns", Class: AmendRatchet, Direction: DirectionTightenOnly,
		Channels: []string{ChannelOperatorOverlay, ChannelLiveReload, ChannelCentral},
		Doc:      "Policy-declared extra secret shapes unioned with the canon floor; extend-only, never replace."},
	{Field: "InlineEval", Class: AmendRatchet, Direction: DirectionTightenOnly,
		Channels: []string{ChannelOperatorOverlay, ChannelLiveReload, ChannelCentral},
		Doc:      "Supplemental interpreter specs unioned with the compiled inline-eval write floor; extend-only."},
	{Field: "EgressExtraDenyHosts", Class: AmendRatchet, Direction: DirectionTightenOnly,
		Channels: []string{ChannelOperatorOverlay, ChannelCentral},
		Doc:      "Operator-declared extra egress deny hosts on top of the hardwired metadata class; tighten-only."},
	{Field: "ResearchEgressAllowHosts", Class: AmendRatchet, Direction: DirectionTightenOnly,
		Channels: []string{ChannelOperatorOverlay, ChannelCentral},
		Doc:      "Positive WebFetch allowlist for research sub-agents; a non-empty list forces the strict allowlist, tightening WebFetch."},
	{Field: "EgressAllowHosts", Class: AmendGatedWiden, Direction: DirectionWidenOnly,
		Channels: []string{ChannelOperatorOverlay, ChannelCentral},
		Doc:      "Adblock-style exception rules; un-blocks a host under a block list (never the hardwired SSRF floor)."},
	{Field: "EgressBlockHosts", Class: AmendRatchet, Direction: DirectionTightenOnly,
		Channels: []string{ChannelOperatorOverlay, ChannelCentral},
		Doc:      "Operator host block rules with subdomain matching; additive tighten over the egress floor."},
	{Field: "EgressBlockLists", Class: AmendRatchet, Direction: DirectionTightenOnly,
		Channels: []string{ChannelOperatorOverlay, ChannelCentral},
		Doc:      "Bundled community filter lists compiled into the block set; additive tighten."},
	{Field: "EgressRestrict", Class: AmendGatedWiden, Direction: DirectionWidenOnly,
		Channels: []string{ChannelOperatorOverlay, ChannelCentral},
		Doc:      "Selects the WebFetch default posture; the zero value is the looser default-allowed stance, so the knob is gate-widened deliberately."},
	{Field: "AutoRepairSidestep", Class: AmendGatedWiden, Direction: DirectionWidenOnly,
		Channels: []string{ChannelOperatorOverlay, ChannelCentral},
		Doc:      "Opts the reversibility rung into in-flight safe-subset TRANSFORM instead of the preview-confirm HOLD (AutoRepairSidestep)."},
	{Field: "TestLanes", Class: AmendRatchet, Direction: DirectionTightenOnly,
		Channels: []string{ChannelOperatorOverlay, ChannelCentral},
		Doc:      "Test lane names explicitly designated as test lanes."},
	{Field: "ExemptLanes", Class: AmendGatedWiden, Direction: DirectionWidenOnly,
		Channels: []string{ChannelOperatorOverlay, ChannelCentral},
		Doc:      "Lanes exempt from test immunity."},
	{Field: "DisableTestImmunity", Class: AmendGatedWiden, Direction: DirectionWidenOnly,
		Channels: []string{ChannelOperatorOverlay, ChannelCentral},
		Doc:      "Disables the test-immunity gate entirely."},
	{Field: "Lane", Class: AmendRatchet, Direction: DirectionTightenOnly,
		Channels: []string{ChannelOperatorOverlay, ChannelCentral},
		Doc:      "Active policy lane name."},
}

// KnobByField returns the registry entry backed by the named
// adjudicator.Policy struct field. The empty field name never matches
// (non-field floor elements have no field key).
func KnobByField(field string) (PolicyKnob, bool) {
	if field == "" {
		return PolicyKnob{}, false
	}
	for _, k := range PolicyKnobRegistry {
		if k.Field == field {
			return k, true
		}
	}
	return PolicyKnob{}, false
}

// KnobsByClass returns every registry entry of the given amendment class, in
// registry order.
func KnobsByClass(c AmendmentClass) []PolicyKnob {
	var out []PolicyKnob
	for _, k := range PolicyKnobRegistry {
		if k.Class == c {
			out = append(out, k)
		}
	}
	return out
}
