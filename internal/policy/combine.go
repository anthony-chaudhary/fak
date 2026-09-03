package policy

import (
	"errors"
	"fmt"
	"sort"
)

// PrivacyPolicy specifies data retention, privacy, and logging rules.
type PrivacyPolicy struct {
	ZeroRetention   bool `json:"zero_retention"`
	MaskPII         bool `json:"mask_pii"`
	LogContent      bool `json:"log_content"`
	StoreTranscript bool `json:"store_transcript"`
}

// Budget specifies resource consumption caps for multi-tenant layers.
// Positive values represent active caps; zero or negative values indicate unconstrained dimensions.
type Budget struct {
	MaxTokens  int64   `json:"max_tokens,omitempty"`
	MaxCostUSD float64 `json:"max_cost_usd,omitempty"`
	MaxCalls   int64   `json:"max_calls,omitempty"`
}

// Usage captures recorded resource utilization.
type Usage struct {
	Tokens  int64   `json:"tokens"`
	CostUSD float64 `json:"cost_usd"`
	Calls   int64   `json:"calls"`
}

// CombinedPolicy represents a merged policy across organizational hierarchy layers.
type CombinedPolicy struct {
	Allow   []string      `json:"allow,omitempty"`
	Deny    []string      `json:"deny,omitempty"`
	Privacy PrivacyPolicy `json:"privacy"`
	Budgets []Budget      `json:"budgets,omitempty"`
}

// ErrBudgetExceeded is returned when recorded usage exceeds any defined positive budget cap.
var ErrBudgetExceeded = errors.New("budget exceeded")

// dedupeAndSort returns a deduplicated and lexicographically sorted slice.
func dedupeAndSort(items []string) []string {
	if items == nil {
		return nil
	}
	if len(items) == 0 {
		return []string{}
	}
	seen := make(map[string]struct{}, len(items))
	res := make([]string, 0, len(items))
	for _, s := range items {
		if _, ok := seen[s]; !ok {
			seen[s] = struct{}{}
			res = append(res, s)
		}
	}
	sort.Strings(res)
	return res
}

// CombineAllowlists calculates the deterministic set intersection of parent and child allowlists.
// In hierarchical enterprise multi-tenancy, a child layer cannot expand what a parent layer restricted.
// If parent is nil, child's list is returned (parent unconstrained).
// If child is nil, parent's list is returned.
// If either is an empty slice ([]string{}), the intersection is empty ([]string{}).
// The resulting list is deduplicated and stably sorted.
func CombineAllowlists(parent, child []string) []string {
	if parent == nil {
		return dedupeAndSort(child)
	}
	if child == nil {
		return dedupeAndSort(parent)
	}
	if len(parent) == 0 || len(child) == 0 {
		return []string{}
	}

	parentSet := make(map[string]struct{}, len(parent))
	for _, s := range parent {
		parentSet[s] = struct{}{}
	}

	seen := make(map[string]struct{})
	var common []string
	for _, s := range child {
		if _, inParent := parentSet[s]; inParent {
			if _, inSeen := seen[s]; !inSeen {
				seen[s] = struct{}{}
				common = append(common, s)
			}
		}
	}

	if len(common) == 0 {
		return []string{}
	}
	sort.Strings(common)
	return common
}

// CombineDenylists calculates the deterministic set union of parent and child denylists.
// In hierarchical enterprise multi-tenancy, if either parent or child denies a tool/pattern, it is denied.
// The resulting list is deduplicated and stably sorted.
func CombineDenylists(parent, child []string) []string {
	if parent == nil && child == nil {
		return nil
	}
	seen := make(map[string]struct{}, len(parent)+len(child))
	var res []string
	for _, s := range parent {
		if _, ok := seen[s]; !ok {
			seen[s] = struct{}{}
			res = append(res, s)
		}
	}
	for _, s := range child {
		if _, ok := seen[s]; !ok {
			seen[s] = struct{}{}
			res = append(res, s)
		}
	}
	if len(res) == 0 {
		return []string{}
	}
	sort.Strings(res)
	return res
}

// CombinePrivacy combines parent and child privacy policies using least-privilege rules.
// Restrictive flags (ZeroRetention, MaskPII) use logical OR (most restrictive wins).
// Permissive logging flags (LogContent, StoreTranscript) use logical AND (least permissive wins).
func CombinePrivacy(parent, child PrivacyPolicy) PrivacyPolicy {
	return PrivacyPolicy{
		ZeroRetention:   parent.ZeroRetention || child.ZeroRetention,
		MaskPII:         parent.MaskPII || child.MaskPII,
		LogContent:      parent.LogContent && child.LogContent,
		StoreTranscript: parent.StoreTranscript && child.StoreTranscript,
	}
}

// EvaluateBudgets checks usage against all budgets in the slice.
// If any budget's positive cap is exceeded (Tokens > MaxTokens, CostUSD > MaxCostUSD, Calls > MaxCalls),
// it returns an error wrapping ErrBudgetExceeded.
func EvaluateBudgets(budgets []Budget, usage Usage) error {
	for _, b := range budgets {
		if b.MaxTokens > 0 && usage.Tokens > b.MaxTokens {
			return fmt.Errorf("%w: tokens %d exceeds max %d", ErrBudgetExceeded, usage.Tokens, b.MaxTokens)
		}
		if b.MaxCostUSD > 0 && usage.CostUSD > b.MaxCostUSD {
			return fmt.Errorf("%w: cost %.4f exceeds max %.4f", ErrBudgetExceeded, usage.CostUSD, b.MaxCostUSD)
		}
		if b.MaxCalls > 0 && usage.Calls > b.MaxCalls {
			return fmt.Errorf("%w: calls %d exceeds max %d", ErrBudgetExceeded, usage.Calls, b.MaxCalls)
		}
	}
	return nil
}

// CombinePolicies combines parent and child CombinedPolicy into a unified CombinedPolicy.
// Allowlists are intersected, denylists are unioned, privacy policies are combined
// with least-privilege rules, and budgets are concatenated.
func CombinePolicies(parent, child CombinedPolicy) CombinedPolicy {
	var budgets []Budget
	if len(parent.Budgets) > 0 || len(child.Budgets) > 0 {
		budgets = make([]Budget, 0, len(parent.Budgets)+len(child.Budgets))
		budgets = append(budgets, parent.Budgets...)
		budgets = append(budgets, child.Budgets...)
	}

	return CombinedPolicy{
		Allow:   CombineAllowlists(parent.Allow, child.Allow),
		Deny:    CombineDenylists(parent.Deny, child.Deny),
		Privacy: CombinePrivacy(parent.Privacy, child.Privacy),
		Budgets: budgets,
	}
}
