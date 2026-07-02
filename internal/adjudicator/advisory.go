package adjudicator

// Advisory posture — the false-positive escape hatch for the HEURISTIC rungs.
//
// The capability floor's hard rungs are deliberately fail-closed, but two of them
// decide by heuristic (the self-modify glob/shell/synth-tool family, the lint
// gate) and one by absence (default-deny), so they carry a real false-positive
// rate — witnessed live 2026-07-01 when `python -c "…os.environ…"` was denied
// SELF_MODIFY on the ".env" glob. Before this file, a false positive on a hard
// rung blocked with no softer setting: Posture/Complain gate only the
// default-deny rung, so the only unblock was editing the rule itself.
//
// AdvisoryReasons is the operator-declared, REASON-KEYED advisory (warn) mode:
// a monitor refusal citing an advisory reason is downgraded to an admit-and-log
// Allow that carries the full forensic record (posture=advisory, the would-deny
// reason, the bounded witness claim) in Verdict.Meta — so the decision journal
// still shows every would-deny and the guard-verdict RSI / complaint loops can
// fold them. Enforcement stays the default; advisory is opt-in per policy load
// (manifest `advisory_reasons`, or the FAK_ADVISORY_REASONS dev-session env
// overlay — see internal/policy).
//
// The set is CLAMPED to the heuristic reasons (AdvisoryEligible): SELF_MODIFY,
// MALFORMED, DEFAULT_DENY. The genuine-danger classes can never be blanket-
// softened — POLICY_BLOCK guards the destructive-Bash / privilege-escalation /
// RCE-pipe rules (per-rule softening exists instead: ArgPredicate.Advisory),
// SECRET_EXFIL guards exfil shapes, and the hardwired egress floor
// (EGRESS_BLOCK) is never a false positive by construction. Two more bounds:
//
//   - soften applies only to the base monitor's own refusals (By == "monitor").
//     The research-egress sub-rung ("monitor/research-egress") is a POSITIVE
//     allowlist, so its MALFORMED (unparseable WebFetch URL) must keep failing
//     closed — softening it would let a deliberately malformed URL launder past
//     the host allowlist. Other links in the kernel fold (shipgate, ratelimit)
//     are separate authorities and are never softened by this policy.
//   - the downgrade produces an Allow, which the kernel folds by the
//     restrictiveness lattice — any peer link's Deny still wins, so advisory
//     can only ever soften THIS link's verdict, never override another's.
import (
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// AdvisoryEligible reports whether a refusal reason may be declared advisory.
// Only the heuristic rungs' reasons qualify; everything else is a genuine-danger
// or positive-allowlist floor that stays fail-closed under every posture. The
// policy loader validates against this predicate (fail-loud at load); New /
// SetPolicy clamp against it too (fail-safe for programmatic Policy values).
func AdvisoryEligible(r abi.ReasonCode) bool {
	switch r {
	case abi.ReasonSelfModify, abi.ReasonMalformed, abi.ReasonDefaultDeny:
		return true
	}
	return false
}

// AdvisoryEligibleNames returns the sorted stable names of the reasons
// AdvisoryEligible admits — the closed vocabulary a loader error message cites.
func AdvisoryEligibleNames() []string {
	names := []string{
		abi.ReasonName(abi.ReasonSelfModify),
		abi.ReasonName(abi.ReasonMalformed),
		abi.ReasonName(abi.ReasonDefaultDeny),
	}
	sort.Strings(names)
	return names
}

// sanitizeAdvisoryReasons clamps an advisory set to the eligible reasons — the
// same floor invariant move as sanitizeProfile: a Policy constructed in code
// (bypassing the loader's loud validation) still cannot mark a genuine-danger
// reason advisory. Returns nil for an effectively empty set so the zero Policy
// stays byte-for-byte unchanged.
func sanitizeAdvisoryReasons(m map[abi.ReasonCode]bool) map[abi.ReasonCode]bool {
	if len(m) == 0 {
		return nil
	}
	out := make(map[abi.ReasonCode]bool, len(m))
	for r, on := range m {
		if on && AdvisoryEligible(r) {
			out[r] = true
		}
	}
	if len(out) == 0 {
		return nil
	}
	return out
}

// soften is the advisory downgrade, applied at every monitor deny site: a Deny
// whose reason the policy declares advisory becomes an admit-and-log Allow whose
// Meta carries the would-deny record — posture=advisory, the refusal name, the
// bounded witness claim (which glob/rule fired — the false-positive diagnostic),
// and any advisory arg-rule violations noted earlier in the fold. Everything
// else passes through untouched. Gated on By == "monitor" so a sub-rung with its
// own authority (research-egress) is never softened.
func (p Policy) soften(v abi.Verdict, notes []string) abi.Verdict {
	if v.Kind != abi.VerdictDeny || v.By != "monitor" || !p.AdvisoryReasons[v.Reason] {
		return v
	}
	meta := map[string]string{
		"posture":    "advisory",
		"would_deny": abi.ReasonName(v.Reason),
	}
	if wp, ok := v.Payload.(abi.WitnessPayload); ok && wp.Claim != "" {
		meta["claim"] = wp.Claim
	}
	if len(notes) > 0 {
		meta["advisory_violations"] = strings.Join(notes, "; ")
	}
	return abi.Verdict{Kind: abi.VerdictAllow, By: v.By, Meta: meta}
}

// allowWithNotes is the affirmative Allow, carrying any advisory arg-rule
// violation notes so a rule the operator put on logged trial still leaves its
// record on the admitted call.
func allowWithNotes(by string, notes []string) abi.Verdict {
	v := abi.Verdict{Kind: abi.VerdictAllow, By: by}
	if len(notes) > 0 {
		v.Meta = map[string]string{"advisory_violations": strings.Join(notes, "; ")}
	}
	return v
}
