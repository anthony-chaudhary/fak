package guard

import (
	"fmt"
	"strings"
)

// Nomos action-review gate.
//
// Inspired by NVIDIA/Nomos, which reviews each proposed agent action before
// execution and refuses the ones that are too aggressive ("overeager"). This is
// a concepts-only, host-side port: a pure-Go gate that classifies a proposed
// tool action into a risk category, applies a native tool-action guard policy,
// and emits an allow/deny verdict with a closed-vocabulary reason token.
//
// Invariant: the gate is fail-closed. An action whose category is not in the
// known vocabulary is refused (DENY / UNKNOWN_CATEGORY), never allowed. A
// verdict is advisory input to the operator's pre-flight ladder; it does not,
// on its own, grant authority. No guard-efficacy claim is made here — that
// requires a native eval, which this file deliberately does not provide.

// ActionCategory is the closed-vocabulary risk classification of a proposed
// tool action.
type ActionCategory string

const (
	// CategoryRead is a read-only observation (file read, list, query).
	CategoryRead ActionCategory = "READ"
	// CategoryWrite is an in-scope, reversible write (create/modify a file).
	CategoryWrite ActionCategory = "WRITE"
	// CategoryExecute is running a command within the declared scope.
	CategoryExecute ActionCategory = "EXECUTE"
	// CategoryDestructive is an irreversible or hard-to-reverse action
	// (delete, drop, force-overwrite, format).
	CategoryDestructive ActionCategory = "DESTRUCTIVE"
	// CategoryPrivilege is a privilege or security-configuration change
	// (sudo, chmod 777, disable a guard, edit auth config).
	CategoryPrivilege ActionCategory = "PRIVILEGE"
	// CategoryExfiltration is outbound transfer of data out of the host
	// (upload, POST of local data, exfil to a remote endpoint).
	CategoryExfiltration ActionCategory = "EXFILTRATION"
)

// knownCategories is the closed set of categories the gate understands.
var knownCategories = map[ActionCategory]struct{}{
	CategoryRead:         {},
	CategoryWrite:        {},
	CategoryExecute:      {},
	CategoryDestructive:  {},
	CategoryPrivilege:    {},
	CategoryExfiltration: {},
}

// IsKnown reports whether c is a recognized action category.
func (c ActionCategory) IsKnown() bool {
	_, ok := knownCategories[c]
	return ok
}

// OvereagerCategories is the set of categories the native tool-action guard
// policy refuses by default. These are the "overeager" actions Nomos would
// flag: they are irreversible, escalate authority, or move data out of the
// host, so they must not be auto-approved.
//
//enumlint:exempt the overeager set names only the categories refused by default; READ, WRITE and EXECUTE are benign by policy, so this set is partial by definition
var OvereagerCategories = map[ActionCategory]struct{}{
	CategoryDestructive:  {},
	CategoryPrivilege:    {},
	CategoryExfiltration: {},
}

// IsOvereager reports whether c is an overeager category.
func (c ActionCategory) IsOvereager() bool {
	_, ok := OvereagerCategories[c]
	return ok
}

// Verdict is the allow/deny outcome of an action review.
type Verdict string

const (
	VerdictAllow Verdict = "ALLOW"
	VerdictDeny  Verdict = "DENY"
)

// Reason tokens (closed vocabulary) that accompany a verdict.
const (
	// ReasonBenign marks an ALLOW: the action is within the allowed policy.
	ReasonBenign = "BENIGN"
	// ReasonOvereager marks a DENY: the category is in the overeager set.
	ReasonOvereager = "OVEREAGER"
	// ReasonNotAllowed marks a DENY: a known category the policy does not allow.
	ReasonNotAllowed = "NOT_ALLOWED"
	// ReasonUnknownCategory marks a fail-closed DENY: unrecognized category.
	ReasonUnknownCategory = "UNKNOWN_CATEGORY"
)

// ReviewVerdict is the structured outcome of reviewing a proposed action.
type ReviewVerdict struct {
	Verdict  Verdict        `json:"verdict"`
	Reason   string         `json:"reason"`
	Category ActionCategory `json:"category"`
	Detail   string         `json:"detail,omitempty"`
}

// Render returns the canonical verdict-grammar string: "VERDICT:REASON".
func (v ReviewVerdict) Render() string {
	return string(v.Verdict) + ":" + v.Reason
}

// ParseVerdict parses a canonical verdict-grammar string "VERDICT:REASON" back
// into a ReviewVerdict. It returns an error if the string is not in the closed
// grammar: missing separator, unknown verdict, unknown reason, or a verdict and
// reason that cannot legally pair (ALLOW only pairs with BENIGN; DENY never
// pairs with BENIGN).
func ParseVerdict(s string) (ReviewVerdict, error) {
	parts := strings.SplitN(s, ":", 2)
	if len(parts) != 2 {
		return ReviewVerdict{}, fmt.Errorf("verdict grammar: missing ':' separator in %q", s)
	}
	v := Verdict(parts[0])
	reason := parts[1]
	switch v {
	case VerdictAllow, VerdictDeny:
	default:
		return ReviewVerdict{}, fmt.Errorf("verdict grammar: unknown verdict %q", parts[0])
	}
	switch reason {
	case ReasonBenign, ReasonOvereager, ReasonNotAllowed, ReasonUnknownCategory:
	default:
		return ReviewVerdict{}, fmt.Errorf("verdict grammar: unknown reason %q", reason)
	}
	if v == VerdictAllow && reason != ReasonBenign {
		return ReviewVerdict{}, fmt.Errorf("verdict grammar: ALLOW must pair with %s, got %s", ReasonBenign, reason)
	}
	if v == VerdictDeny && reason == ReasonBenign {
		return ReviewVerdict{}, fmt.Errorf("verdict grammar: DENY cannot pair with %s", ReasonBenign)
	}
	return ReviewVerdict{Verdict: v, Reason: reason}, nil
}

// ReviewPolicy is the native tool-action guard policy. It decides, per
// category, whether an action is auto-allowed. The default policy allows the
// benign categories and denies the overeager ones.
type ReviewPolicy struct {
	// Allowed is the set of categories the policy auto-approves. Any known
	// category not in Allowed is denied.
	Allowed map[ActionCategory]struct{}
}

// DefaultReviewPolicy returns the built-in tool-action guard policy: allow
// read/write/execute, deny destructive/privilege/exfiltration.
func DefaultReviewPolicy() ReviewPolicy {
	return ReviewPolicy{
		Allowed: map[ActionCategory]struct{}{ //enumlint:exempt the default policy auto-allows only READ, WRITE and EXECUTE; DESTRUCTIVE, PRIVILEGE and EXFILTRATION are denied by being absent from the allow set
			CategoryRead:    {},
			CategoryWrite:   {},
			CategoryExecute: {},
		},
	}
}

// ProposedAction is a tool action proposed for execution, to be reviewed
// before it runs.
type ProposedAction struct {
	Tool     string         `json:"tool,omitempty"`
	Category ActionCategory `json:"category"`
	Scope    string         `json:"scope,omitempty"`
}

// Review applies the policy to a proposed action and returns a verdict.
//
// The gate is fail-closed:
//   - an unknown category is DENY / UNKNOWN_CATEGORY (never allowed);
//   - a known overeager category is DENY / OVEREAGER;
//   - a known category the policy allows is ALLOW / BENIGN;
//   - a known category the policy does not allow is DENY / NOT_ALLOWED.
func (p ReviewPolicy) Review(a ProposedAction) ReviewVerdict {
	if !a.Category.IsKnown() {
		return ReviewVerdict{
			Verdict:  VerdictDeny,
			Reason:   ReasonUnknownCategory,
			Category: a.Category,
			Detail:   fmt.Sprintf("action category %q is not in the known vocabulary; refusing fail-closed", a.Category),
		}
	}
	if a.Category.IsOvereager() {
		return ReviewVerdict{
			Verdict:  VerdictDeny,
			Reason:   ReasonOvereager,
			Category: a.Category,
			Detail:   fmt.Sprintf("action category %q is overeager and refused by the tool-action guard", a.Category),
		}
	}
	if _, ok := p.Allowed[a.Category]; ok {
		return ReviewVerdict{
			Verdict:  VerdictAllow,
			Reason:   ReasonBenign,
			Category: a.Category,
			Detail:   fmt.Sprintf("action category %q is within the allowed tool-action policy", a.Category),
		}
	}
	return ReviewVerdict{
		Verdict:  VerdictDeny,
		Reason:   ReasonNotAllowed,
		Category: a.Category,
		Detail:   fmt.Sprintf("action category %q is not in the allowed set; refusing fail-closed", a.Category),
	}
}

// ReviewAction reviews a single proposed action under the default policy.
func ReviewAction(a ProposedAction) ReviewVerdict {
	return DefaultReviewPolicy().Review(a)
}

// BatchReviewResult summarizes a review of a sequence of proposed actions.
type BatchReviewResult struct {
	Verdicts   []ReviewVerdict `json:"verdicts"`
	Allowed    int             `json:"allowed"`
	Denied     int             `json:"denied"`
	AllAllowed bool            `json:"all_allowed"`
}

// ReviewBatch reviews each proposed action in order and summarizes the batch.
// Nomos reviews each proposed action before execution; this mirrors that by
// producing one verdict per action plus a batch summary.
func (p ReviewPolicy) ReviewBatch(actions []ProposedAction) BatchReviewResult {
	res := BatchReviewResult{Verdicts: make([]ReviewVerdict, 0, len(actions))}
	for _, a := range actions {
		v := p.Review(a)
		res.Verdicts = append(res.Verdicts, v)
		if v.Verdict == VerdictAllow {
			res.Allowed++
		} else {
			res.Denied++
		}
	}
	res.AllAllowed = res.Denied == 0
	return res
}
