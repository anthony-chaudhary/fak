package executionroute

import (
	"encoding/json"
	"fmt"
	"sort"
	"strings"
)

// ExecutionSurface identifies the compute execution target (#10687).
type ExecutionSurface string

const (
	SurfaceNPU    ExecutionSurface = "npu"
	SurfaceGPU    ExecutionSurface = "gpu"
	SurfaceCPU    ExecutionSurface = "cpu"
	SurfaceHosted ExecutionSurface = "hosted"
)

// ValidExecutionSurfaces maps all valid surfaces to true.
var ValidExecutionSurfaces = map[ExecutionSurface]bool{
	SurfaceNPU:    true,
	SurfaceGPU:    true,
	SurfaceCPU:    true,
	SurfaceHosted: true,
}

// PlacementRequest encapsulates structured request attributes used for deterministic
// placement decisions without learned classifiers.
type PlacementRequest struct {
	Model          string            `json:"model"`
	Role           string            `json:"role,omitempty"`             // e.g. "auxiliary", "tool_screen", "scout", "primary", "judge"
	ParameterSizeB float64           `json:"parameter_size_b,omitempty"` // Model parameter magnitude in billions (e.g. 1.5, 7.0, 70.0)
	ContextTokens  int               `json:"context_tokens,omitempty"`
	RequireLocal   bool              `json:"require_local,omitempty"` // Hard fence: must not route to hosted
	OfflineOnly    bool              `json:"offline_only,omitempty"`  // Hard fence: no external egress
	MaxLatencyMS   int               `json:"max_latency_ms,omitempty"`
	Tags           []string          `json:"tags,omitempty"`
	Attributes     map[string]string `json:"attributes,omitempty"`
}

// PlacementRule defines structured match criteria mapping to an execution surface.
type PlacementRule struct {
	Name          string            `json:"name"`
	Surface       ExecutionSurface  `json:"surface"`
	Priority      int               `json:"priority,omitempty"` // Higher values evaluate first
	Models        []string          `json:"models,omitempty"`   // Exact or prefix matches
	Roles         []string          `json:"roles,omitempty"`    // E.g. ["auxiliary", "tool_screen", "scout"]
	MinParamSizeB float64           `json:"min_param_size_b,omitempty"`
	MaxParamSizeB float64           `json:"max_param_size_b,omitempty"`
	MinContext    int               `json:"min_context,omitempty"`
	MaxContext    int               `json:"max_context,omitempty"`
	RequireLocal  *bool             `json:"require_local,omitempty"`
	OfflineOnly   *bool             `json:"offline_only,omitempty"`
	RequiredTags  []string          `json:"required_tags,omitempty"`
	Attributes    map[string]string `json:"attributes,omitempty"`
}

// RuleAudit captures the deterministic evaluation outcome for one rule.
type RuleAudit struct {
	RuleName string           `json:"rule_name"`
	Surface  ExecutionSurface `json:"surface"`
	Matched  bool             `json:"matched"`
	Reason   string           `json:"reason"`
}

// PlacementDecision reports the selected surface, winning rule, and full audit trail.
type PlacementDecision struct {
	Surface        ExecutionSurface `json:"surface"`
	RuleName       string           `json:"rule_name"`
	Reason         string           `json:"reason"`
	AuditLog       []RuleAudit      `json:"audit_log"`
	EvaluatedRules int              `json:"evaluated_rules"`
}

// PlacementManifest defines a deterministic policy manifest containing structured rules.
type PlacementManifest struct {
	Version        string           `json:"version"`
	DefaultSurface ExecutionSurface `json:"default_surface"`
	Rules          []PlacementRule  `json:"rules"`
}

// DefaultPlacementManifest constructs a standard deterministic placement policy.
// It encodes the proven hardware trade-offs:
// - Small auxiliary / screening models (<=3B) -> NPU
// - Primary local models (>3B, local) -> GPU
// - Massive cloud models (>70B, not offline/local) -> Hosted
// - Fallback default -> GPU (or CPU if offline/local constraints dictate)
func DefaultPlacementManifest() PlacementManifest {
	offlineTrue := true
	return PlacementManifest{
		Version:        "fak-placement-policy/1",
		DefaultSurface: SurfaceGPU,
		Rules: []PlacementRule{
			{
				Name:          "npu-auxiliary-small",
				Surface:       SurfaceNPU,
				Priority:      100,
				Roles:         []string{"auxiliary", "tool_screen", "speculative_draft", "scout"},
				MaxParamSizeB: 3.5,
			},
			{
				Name:          "hosted-massive-cloud",
				Surface:       SurfaceHosted,
				Priority:      80,
				MinParamSizeB: 70.0,
			},
			{
				Name:          "gpu-primary-local",
				Surface:       SurfaceGPU,
				Priority:      60,
				MinParamSizeB: 3.0,
			},
			{
				Name:     "cpu-offline-fallback",
				Surface:  SurfaceCPU,
				Priority: 40,
				Roles:    []string{"embed", "classifier"},
			},
			{
				Name:        "cpu-strict-offline-embed",
				Surface:     SurfaceCPU,
				Priority:    30,
				OfflineOnly: &offlineTrue,
				Roles:       []string{"embed"},
			},
		},
	}
}

// RoutePlacement deterministically evaluates req against the manifest rules in priority order.
func (m PlacementManifest) RoutePlacement(req PlacementRequest) (PlacementDecision, error) {
	if !ValidExecutionSurfaces[m.DefaultSurface] {
		m.DefaultSurface = SurfaceGPU
	}

	// Sort rules deterministically by priority descending, keeping declared order on tie.
	type indexedRule struct {
		index int
		rule  PlacementRule
	}
	indexed := make([]indexedRule, len(m.Rules))
	for i, r := range m.Rules {
		indexed[i] = indexedRule{index: i, rule: r}
	}
	sort.SliceStable(indexed, func(i, j int) bool {
		if indexed[i].rule.Priority != indexed[j].rule.Priority {
			return indexed[i].rule.Priority > indexed[j].rule.Priority
		}
		return indexed[i].index < indexed[j].index
	})

	var auditLog []RuleAudit
	for _, ir := range indexed {
		r := ir.rule
		matched, reason := evaluateRule(r, req)
		auditLog = append(auditLog, RuleAudit{
			RuleName: r.Name,
			Surface:  r.Surface,
			Matched:  matched,
			Reason:   reason,
		})

		if matched {
			return PlacementDecision{
				Surface:        r.Surface,
				RuleName:       r.Name,
				Reason:         reason,
				AuditLog:       auditLog,
				EvaluatedRules: len(auditLog),
			}, nil
		}
	}

	// Fallback to DefaultSurface, ensuring hard safety constraints are respected.
	fallbackSurface := m.DefaultSurface
	fallbackReason := "fallback to manifest default surface"
	if (req.RequireLocal || req.OfflineOnly) && fallbackSurface == SurfaceHosted {
		fallbackSurface = SurfaceGPU
		fallbackReason = "default surface hosted refused by require_local/offline_only; defaulted to gpu"
	}

	auditLog = append(auditLog, RuleAudit{
		RuleName: "default-fallback",
		Surface:  fallbackSurface,
		Matched:  true,
		Reason:   fallbackReason,
	})

	return PlacementDecision{
		Surface:        fallbackSurface,
		RuleName:       "default-fallback",
		Reason:         fallbackReason,
		AuditLog:       auditLog,
		EvaluatedRules: len(auditLog),
	}, nil
}

func evaluateRule(r PlacementRule, req PlacementRequest) (bool, string) {
	// Surface validity
	if !ValidExecutionSurfaces[r.Surface] {
		return false, fmt.Sprintf("invalid surface %q in rule", r.Surface)
	}

	// Hard constraints: SurfaceHosted refused if RequireLocal or OfflineOnly
	if r.Surface == SurfaceHosted && (req.RequireLocal || req.OfflineOnly) {
		return false, "hosted surface disqualified by require_local or offline_only constraint"
	}

	// Model filter
	if len(r.Models) > 0 {
		matchedModel := false
		reqModelNorm := strings.ToLower(strings.TrimSpace(req.Model))
		for _, m := range r.Models {
			mNorm := strings.ToLower(strings.TrimSpace(m))
			if mNorm == "*" || mNorm == reqModelNorm || strings.HasPrefix(reqModelNorm, mNorm) {
				matchedModel = true
				break
			}
		}
		if !matchedModel {
			return false, fmt.Sprintf("model %q does not match models filter %v", req.Model, r.Models)
		}
	}

	// Role filter
	if len(r.Roles) > 0 {
		matchedRole := false
		reqRoleNorm := strings.ToLower(strings.TrimSpace(req.Role))
		for _, role := range r.Roles {
			rNorm := strings.ToLower(strings.TrimSpace(role))
			if rNorm == "*" || rNorm == reqRoleNorm {
				matchedRole = true
				break
			}
		}
		if !matchedRole {
			return false, fmt.Sprintf("role %q does not match roles filter %v", req.Role, r.Roles)
		}
	}

	// Parameter size bounds
	if r.MinParamSizeB > 0 && req.ParameterSizeB < r.MinParamSizeB {
		return false, fmt.Sprintf("parameter size %.2fB is below minimum %.2fB", req.ParameterSizeB, r.MinParamSizeB)
	}
	if r.MaxParamSizeB > 0 && req.ParameterSizeB > r.MaxParamSizeB {
		return false, fmt.Sprintf("parameter size %.2fB exceeds maximum %.2fB", req.ParameterSizeB, r.MaxParamSizeB)
	}

	// Context bounds
	if r.MinContext > 0 && req.ContextTokens > 0 && req.ContextTokens < r.MinContext {
		return false, fmt.Sprintf("context %d is below minimum %d", req.ContextTokens, r.MinContext)
	}
	if r.MaxContext > 0 && req.ContextTokens > 0 && req.ContextTokens > r.MaxContext {
		return false, fmt.Sprintf("context %d exceeds maximum %d", req.ContextTokens, r.MaxContext)
	}

	// RequireLocal filter
	if r.RequireLocal != nil && *r.RequireLocal != req.RequireLocal {
		return false, fmt.Sprintf("require_local requirement %t does not match request %t", *r.RequireLocal, req.RequireLocal)
	}

	// OfflineOnly filter
	if r.OfflineOnly != nil && *r.OfflineOnly != req.OfflineOnly {
		return false, fmt.Sprintf("offline_only requirement %t does not match request %t", *r.OfflineOnly, req.OfflineOnly)
	}

	// RequiredTags
	if len(r.RequiredTags) > 0 {
		reqTags := make(map[string]bool, len(req.Tags))
		for _, t := range req.Tags {
			reqTags[strings.ToLower(strings.TrimSpace(t))] = true
		}
		for _, rt := range r.RequiredTags {
			if !reqTags[strings.ToLower(strings.TrimSpace(rt))] {
				return false, fmt.Sprintf("missing required tag %q", rt)
			}
		}
	}

	// Attributes match
	if len(r.Attributes) > 0 {
		if req.Attributes == nil {
			return false, "request has nil attributes"
		}
		for k, v := range r.Attributes {
			if req.Attributes[k] != v {
				return false, fmt.Sprintf("attribute %q=%q does not match %q", k, req.Attributes[k], v)
			}
		}
	}

	return true, fmt.Sprintf("matched rule %q for surface %s", r.Name, r.Surface)
}

// ParsePlacementManifest unmarshals a JSON policy manifest.
func ParsePlacementManifest(data []byte) (PlacementManifest, error) {
	var m PlacementManifest
	if err := json.Unmarshal(data, &m); err != nil {
		return PlacementManifest{}, fmt.Errorf("executionroute: unmarshal placement manifest: %w", err)
	}
	return m, nil
}
