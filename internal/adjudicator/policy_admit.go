package adjudicator

import (
	"strings"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// complainFor reports whether a tool is in the per-tool complain set (#670). A nil
// set admits nothing, so the zero Policy is unaffected.
func (p *Policy) complainFor(tool string) bool {
	return p.Complain[tool]
}

// admitAndLog reports whether a DEFAULT-denied tool should be downgraded to an
// admit-and-log Allow: the global read-shaped posture, OR the per-tool complain set
// (#670). It gates ONLY the default-deny rung — the hard-refusal rungs (explicit Deny,
// self-modify, arg violations) return before defaultDeny, so neither path can admit one.
func (p *Policy) admitAndLog(tool string) bool {
	return p.admitAndLogLower(tool, strings.ToLower(tool))
}

// admitAndLogLower is admitAndLog with the case fold already paid (#4007). The
// complain set stays keyed on the RAW name as authored; only the read-shape
// predicate is case-insensitive.
func (p *Policy) admitAndLogLower(tool, lowerTool string) bool {
	return (p.Posture == PostureAdmitAndLog && lowRiskReadShapedLower(lowerTool)) || p.complainFor(tool)
}

func wouldAdmit(p Policy, tool, lowerTool string) bool {
	if p.Posture == PostureDefaultOpen {
		return true
	}
	if p.Allow[tool] || p.admitAndLogLower(tool, lowerTool) || p.AdvisoryReasons[abi.ReasonDefaultDeny] {
		return true
	}
	for _, pre := range p.AllowPrefix {
		if strings.HasPrefix(tool, pre) {
			return true
		}
	}
	return false
}

// NeverAdmits (on the live Adjudicator) is the locked read of the installed floor's
// Policy.NeverAdmits — the args-independent "this name can never be Allowed" query the
// inbound tool-def compactor asks. Reads the current policy under the lock so it is safe
// to call per request from the serving path (where the floor lives in adjudicator.Default
// rather than a host-held Policy value). A pure read: it never mutates run-state.
func (a *Adjudicator) NeverAdmits(tool string) bool {
	p := a.state.Load().policy
	return p.NeverAdmits(tool)
}

// normalizeDenyMap returns a copy of deny with every tool-name key folded to
// lowercase so explicit Deny lookups are consistently case-insensitive (#14512).
// Keys that collide after folding keep the strongest entry (denyEntryWins), so
// map iteration order can never let a softenable reason shadow a hard one.
// advisory must already be clamped by sanitizeAdvisoryReasons.
func normalizeDenyMap(deny map[string]abi.ReasonCode, advisory map[abi.ReasonCode]bool) map[string]abi.ReasonCode {
	if len(deny) == 0 {
		return deny
	}
	out := make(map[string]abi.ReasonCode, len(deny))
	winner := make(map[string]string, len(deny))
	for k, r := range deny {
		lk := strings.ToLower(k)
		if prev, exists := winner[lk]; !exists || denyEntryWins(advisory, k, r, prev, deny[prev]) {
			winner[lk] = k
			out[lk] = r
		}
	}
	return out
}

// denySoftened reports whether this policy softens a deny citing r. It reads the
// clamped view New installs, so a raw Policy and an installed one agree.
func denySoftened(advisory map[abi.ReasonCode]bool, r abi.ReasonCode) bool {
	return advisory[r] && AdvisoryEligible(r)
}

// denyEntryWins orders case-colliding Deny entries, strongest first: a reason
// this policy does not soften, then one an operator approval cannot lift, then
// one that can never be declared advisory, then the exact-lowercase key, then
// the byte-smallest key. Keys are unique, so the order is total.
func denyEntryWins(advisory map[abi.ReasonCode]bool, k string, r abi.ReasonCode, prevK string, prevR abi.ReasonCode) bool {
	if soft, prevSoft := denySoftened(advisory, r), denySoftened(advisory, prevR); soft != prevSoft {
		return !soft
	}
	if lift, prevLift := operatorApprovalLiftable(r), operatorApprovalLiftable(prevR); lift != prevLift {
		return !lift
	}
	if hard, prevHard := !AdvisoryEligible(r), !AdvisoryEligible(prevR); hard != prevHard {
		return hard
	}
	lk := strings.ToLower(k)
	if exact, prevExact := k == lk, prevK == lk; exact != prevExact {
		return exact
	}
	return k < prevK
}

// lookupDeny looks up lowerTool in deny case-insensitively. It scans rather than
// indexing so a raw Policy (e.g. Policy.NeverAdmits without Adjudicator.New)
// resolves case collisions exactly as normalizeDenyMap does.
func lookupDeny(deny map[string]abi.ReasonCode, advisory map[abi.ReasonCode]bool, lowerTool string) (abi.ReasonCode, bool) {
	if len(deny) == 0 {
		return 0, false
	}
	var best string
	var bestR abi.ReasonCode
	found := false
	for k, r := range deny {
		if strings.ToLower(k) != lowerTool {
			continue
		}
		if !found || denyEntryWins(advisory, k, r, best, bestR) {
			best, bestR, found = k, r, true
		}
	}
	return bestR, found
}

// HasNameDeny reports whether this policy carries an explicit name-level Deny for
// tool under the same case folding Adjudicate applies, so routing that resolves
// wire names against the raw policy agrees with the adjudicator on what is denied.
func (p Policy) HasNameDeny(tool string) bool {
	_, ok := lookupDeny(p.Deny, p.AdvisoryReasons, strings.ToLower(tool))
	return ok
}

// NeverAdmits reports whether the floor can NEVER produce an Allow for this tool
// NAME, for ANY argument value — the pure, args-independent question the inbound
// tool-def compactor (promptmmu) asks before it may safely drop a tool DEFINITION.
//
// True ⇔ the name is not affirmatively allowed (absent from Allow, matching no
// AllowPrefix) AND it would not be admitted-and-logged (so a read-shaped name under
// PostureAdmitAndLog, or a complain-set name, is NOT droppable — it can still be
// Allowed). Arg predicates can only RESTRICT an otherwise-allow, never grant one, so
// a never-allowed name stays never-allowed under every argument: dropping its
// advertisement is behavior-preserving. A pure read — no run-state mutation, no lock,
// safe to call per request — so the gateway can build its drop set without folding a
// real adjudication. Hard-refusal names (explicit Deny / self-modify globs) are ALSO
// never admitted, so they report true too; the inbound compactor only ever needs the
// "model can't reach it" guarantee, which both classes satisfy.
func (p Policy) NeverAdmits(tool string) bool {
	if r, denied := lookupDeny(p.Deny, p.AdvisoryReasons, strings.ToLower(tool)); denied {
		// A name-deny whose cited reason is advisory can still be admitted (the
		// soften downgrade), so its tool-def must not be pruned.
		return !denySoftened(p.AdvisoryReasons, r)
	}
	if p.Posture == PostureDefaultOpen {
		return false
	}
	// Fail-safe against an UNCONFIGURED floor: a Policy with no affirmative-allow
	// surface at all (empty Allow, empty AllowPrefix, fail-closed posture) denies
	// EVERY tool — true by the rule below, but as a DROP signal that is almost always
	// "the floor was never installed" rather than "deliberately deny all advertised
	// tools." Pruning every tool-def against a zero floor would be a catastrophic
	// over-drop, so we refuse to prune anything when there is nothing to admit. A real
	// floor (any Allow entry, AllowPrefix, or research WebFetch allowlist) re-enables
	// pruning of the names it genuinely never admits.
	if len(p.Allow) == 0 && len(p.AllowPrefix) == 0 && len(p.ResearchEgressAllowHosts) == 0 {
		return false
	}
	if p.Allow[tool] {
		return false
	}
	for _, pre := range p.AllowPrefix {
		if strings.HasPrefix(tool, pre) {
			return false
		}
	}
	if strings.EqualFold(tool, "WebFetch") && len(p.ResearchEgressAllowHosts) > 0 {
		return false
	}
	// Advisory DEFAULT_DENY admits any name (with the would-deny record), so
	// nothing is never-admitted under it.
	return !p.admitAndLog(tool) && !p.AdvisoryReasons[abi.ReasonDefaultDeny]
}

func defaultDeny(p Policy, tool, lowerTool string, advisoryNotes []string) abi.Verdict {
	if p.Posture == PostureDefaultOpen {
		meta := map[string]string{
			"posture": "default_open",
		}
		if len(advisoryNotes) > 0 {
			meta["advisory_violations"] = strings.Join(advisoryNotes, "; ")
		}
		return abi.Verdict{
			Kind: abi.VerdictAllow,
			By:   "monitor",
			Meta: meta,
		}
	}
	if p.admitAndLogLower(tool, lowerTool) {
		// Admit-and-log record (#671): the default-deny rung is the refusal being
		// suppressed, so the record carries would_deny = its reason name via
		// abi.ReasonName — the forensic field the promotion ledger (#672) folds. Both
		// the complain-set and the global read-shaped path carry it identically.
		meta := map[string]string{
			"posture":    "admit_and_log",
			"would_deny": abi.ReasonName(abi.ReasonDefaultDeny),
		}
		if len(advisoryNotes) > 0 {
			meta["advisory_violations"] = strings.Join(advisoryNotes, "; ")
		}
		return abi.Verdict{
			Kind: abi.VerdictAllow,
			By:   "monitor",
			Meta: meta,
		}
	}
	// Advisory DEFAULT_DENY (Policy.AdvisoryReasons) admits ANY tool with the
	// would-deny record — the strictly-wider dev dual of admit_and_log, carrying
	// posture=advisory so the promotion ledger's admit_and_log fold is untouched.
	return p.soften(abi.Verdict{Kind: abi.VerdictDeny, Reason: abi.ReasonDefaultDeny, By: "monitor"}, advisoryNotes)
}
