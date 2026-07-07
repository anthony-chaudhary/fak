package rsiloop

// skillvalue.go — #2842: the self-fulfilling-SKILL detector (part of #2834,
// Track A). Hermes' curator promotes/keeps agent-created skills on a raw
// use_count telemetry sidecar, so a skill the loop learns to invoke reflexively
// looks "valuable" by use-count even when it changes no outcome — a classic
// self-fulfilling metric (the same reward-hack shape fak already names on the
// cache surface, #2818, and the memory/learning surface via
// memvaluescore.DetectSelfFulfilling, #2914).
//
// This points that detector at PER-SKILL keep-gate telemetry: a skill's value
// must be net-of-invocation — the counterfactual value measured against the
// world where the skill is absent — not raw use_count. A skill whose value
// COLLAPSES when you subtract its own invocations (net counterfactual <= 0
// despite a positive use_count) is flagged and routed to the curator's
// per-decision revert path (#2841) with a structured self_fulfilling reason,
// instead of being kept because it "looks used."

import "fmt"

// SkillValue is one agent-created skill's keep-gate telemetry: the gameable
// use_count the curator would keep on, paired with the value net of the skill's
// own invocations — the counterfactual value column beside use_count that #2842
// adds.
//
//   - UseCount is the raw invocation count: the self-fulfilling metric a loop can
//     raise just by learning to call the skill reflexively.
//   - GrossValue is the skill's value BEFORE the counterfactual subtraction — the
//     value a use_count-crediting curator would attribute to it.
//   - ValuePerInvocation is the per-call credit that value grants for merely being
//     invoked: the self-referential component the counterfactual subtracts out.
//
// The counterfactual value is GrossValue net of that per-invocation credit: what
// survives when the skill's own invocations are removed. A skill that changed no
// outcome has GrossValue == UseCount*ValuePerInvocation, so its counterfactual
// value is 0 — flagged.
type SkillValue struct {
	Skill              string  `json:"skill"`
	UseCount           int     `json:"use_count"`
	GrossValue         float64 `json:"gross_value"`
	ValuePerInvocation float64 `json:"value_per_invocation"`
}

// CounterfactualValue is the skill's value net of its own invocations: GrossValue
// minus the per-invocation credit the self-fulfilling metric grants for merely
// being called. This is the "counterfactual value column beside use_count" #2842
// adds — the value that survives the world where the skill is absent.
func (s SkillValue) CounterfactualValue() float64 {
	return s.GrossValue - float64(s.UseCount)*s.ValuePerInvocation
}

// The skill-value detector verdicts.
const (
	// SkillSelfFulfilling — use_count is positive but counterfactual value is not:
	// the skill's only "improvement" is being invoked. Flagged, routed to revert.
	SkillSelfFulfilling = "self_fulfilling"
	// SkillWitnessed — use_count is backed by real value net of invocations. Kept.
	SkillWitnessed = "witnessed"
	// SkillUnused — the skill was never invoked, so there is no self-fulfilling
	// metric to check.
	SkillUnused = "unused"
)

// SkillValueVerdict is the keep-gate decision for one skill: its telemetry, the
// computed counterfactual value, and whether the skill is a self-fulfilling
// metric artifact that should route to revert.
type SkillValueVerdict struct {
	Value               SkillValue `json:"value"`
	CounterfactualValue float64    `json:"counterfactual_value"`
	Flagged             bool       `json:"flagged"`
	Verdict             string     `json:"verdict"`
	Reason              string     `json:"reason"`
}

// DetectSelfFulfillingSkill applies the self-fulfilling-metric detector to one
// skill's keep-gate telemetry. It computes the skill's value NET of its own
// invocations and flags the skill when its use_count is positive (it "looks
// used") but that counterfactual value has collapsed to zero or negative — a
// skill whose only "improvement" is being invoked. A flagged verdict carries a
// structured curator reason (SelfFulfillingReason) that routes the skill to the
// #2841 per-decision revert path. A skill with real net value, or one that was
// never invoked, is never flagged: raising use_count alone is exactly what the
// detector refuses to reward.
func DetectSelfFulfillingSkill(s SkillValue) SkillValueVerdict {
	cf := s.CounterfactualValue()
	v := SkillValueVerdict{Value: s, CounterfactualValue: cf}
	switch {
	case s.UseCount <= 0:
		v.Verdict = SkillUnused
		v.Reason = fmt.Sprintf("skill %q was never invoked (use_count=%d); no self-fulfilling metric to check", s.Skill, s.UseCount)
	case cf > 0:
		v.Verdict = SkillWitnessed
		v.Reason = fmt.Sprintf("skill %q kept: use_count=%d is backed by counterfactual value %.3g net of its own invocations", s.Skill, s.UseCount, cf)
	default:
		v.Flagged = true
		v.Verdict = SkillSelfFulfilling
		v.Reason = fmt.Sprintf("skill %q flagged self-fulfilling: use_count=%d but counterfactual value collapses to %.3g once its own invocations are subtracted — the only 'improvement' is being invoked", s.Skill, s.UseCount, cf)
	}
	return v
}

// SelfFulfillingReason renders a flagged verdict as the structured curator reason
// that routes the skill to the per-decision revert path (#2841). It returns
// ok=false for an unflagged verdict, so only a genuinely self-fulfilling skill
// (positive use_count, collapsed counterfactual value) enters the keep-gate
// revert path — an operator asking "why is this skill gone?" then reads the
// use_count that made it look valuable and the counterfactual value that did not.
func (v SkillValueVerdict) SelfFulfillingReason() (CuratorReason, bool) {
	if !v.Flagged {
		return CuratorReason{}, false
	}
	return CuratorReason{
		Kind:                ReasonSelfFulfilling,
		UseCount:            v.Value.UseCount,
		CounterfactualValue: v.CounterfactualValue,
	}, true
}
