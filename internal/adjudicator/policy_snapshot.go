package adjudicator

import "github.com/anthony-chaudhary/fak/internal/abi"

// PolicySnapshot returns an isolated copy of the currently installed capability
// floor. Callers may diff or retain the result without racing a later SetPolicy.
func (a *Adjudicator) PolicySnapshot() Policy {
	p := a.state.Load().policy
	p.Allow = cloneBoolMap(p.Allow)
	p.AllowPrefix = append([]string(nil), p.AllowPrefix...)
	p.Deny = cloneReasonMap(p.Deny)
	p.SelfModifyGlobs = append([]string(nil), p.SelfModifyGlobs...)
	p.BlockedPathGlobs = append([]string(nil), p.BlockedPathGlobs...)
	p.ArgPredicates = append([]ArgPredicate(nil), p.ArgPredicates...)
	if p.Profile != nil {
		profile := *p.Profile
		p.Profile = &profile
	}
	p.AdvisoryReasons = cloneReasonBoolMap(p.AdvisoryReasons)
	return p
}

func cloneBoolMap(src map[string]bool) map[string]bool {
	if src == nil {
		return nil
	}
	dst := make(map[string]bool, len(src))
	for k, v := range src {
		dst[k] = v
	}
	return dst
}

func cloneReasonMap(src map[string]abi.ReasonCode) map[string]abi.ReasonCode {
	if src == nil {
		return nil
	}
	dst := make(map[string]abi.ReasonCode, len(src))
	for k, v := range src {
		dst[k] = v
	}
	return dst
}

func cloneReasonBoolMap(src map[abi.ReasonCode]bool) map[abi.ReasonCode]bool {
	if src == nil {
		return nil
	}
	dst := make(map[abi.ReasonCode]bool, len(src))
	for k, v := range src {
		dst[k] = v
	}
	return dst
}
