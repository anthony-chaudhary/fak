package harnesshint

import (
	"strings"
)

// Posture represents the operational posture and scoping discipline appropriate
// for an AI model driving an agent harness.
type Posture string

const (
	// PostureSupportHeavy applies to small and fast/flash models that require
	// heavy scaffolding, low turn limits, and atomic S0/S1 task decomposition to
	// prevent confabulation and token inflation.
	PostureSupportHeavy Posture = "support_heavy"

	// PostureBalanced applies to mid-tier frontier models suited for standard
	// multi-step coordination with moderate context budgets.
	PostureBalanced Posture = "balanced"

	// PostureCostHeavy applies to frontier and expensive reasoning models where
	// turn counts must be tightly bounded to prevent runaway thinking costs.
	PostureCostHeavy Posture = "cost_heavy"

	// PostureNeutral is the fallback posture for unknown models, applying
	// balanced baseline defaults without special constraints.
	PostureNeutral Posture = "neutral"
)

// String returns the string representation of the Posture.
func (p Posture) String() string {
	return string(p)
}

// Valid returns true if the posture is one of the recognized posture constants.
func (p Posture) Valid() bool {
	switch p {
	case PostureSupportHeavy, PostureBalanced, PostureCostHeavy, PostureNeutral:
		return true
	default:
		return false
	}
}

// Provenance constants representing the origin of a ScopeHint.
const (
	ProvenanceBuiltinAlias     = "builtin_alias"
	ProvenanceUnknownDefault   = "unknown_default"
	ProvenanceExplicitOverride = "explicit_override"
)

// ScopeHint contains model-relative scoping recommendations and operational
// bounds to guide harness planning and execution.
type ScopeHint struct {
	// Model is the original model identifier requested by the caller.
	Model string `json:"model"`

	// CanonicalModel is the resolved model alias or canonical identifier.
	CanonicalModel string `json:"canonical_model"`

	// Posture classifies the model into a scoping category.
	Posture Posture `json:"posture"`

	// MaxTurnsRecommended is the recommended upper bound on turns for a task.
	MaxTurnsRecommended int `json:"max_turns_recommended"`

	// DecompositionRecommended indicates whether tasks should be subdivided into
	// atomic units (e.g. S0/S1 leaves) to ensure reliability.
	DecompositionRecommended bool `json:"decomposition_recommended"`

	// ContextBudgetRecommended is the suggested context token budget hint.
	ContextBudgetRecommended int `json:"context_budget_recommended"`

	// Advisory provides compact text guidance on planning, decomposition, and context.
	Advisory string `json:"advisory"`

	// Provenance indicates how the hint was resolved ("builtin_alias", "unknown_default", or "explicit_override").
	Provenance string `json:"provenance"`
}

type modelSpec struct {
	canonical string
	posture   Posture
}

// builtins maps canonical model identifiers and common aliases to modelSpec.
var builtins = map[string]modelSpec{
	// Small/Flash models -> PostureSupportHeavy
	"gemini-1.5-flash":         {canonical: "gemini-1.5-flash", posture: PostureSupportHeavy},
	"gemini-2.0-flash":         {canonical: "gemini-2.0-flash", posture: PostureSupportHeavy},
	"gemini-2.5-flash":         {canonical: "gemini-2.5-flash", posture: PostureSupportHeavy},
	"gemini-3.8-flash":         {canonical: "gemini-3.8-flash", posture: PostureSupportHeavy},
	"gemini-3.8-flash-cyber":   {canonical: "gemini-3.8-flash", posture: PostureSupportHeavy},
	"gemini-3.8-flash-high":    {canonical: "gemini-3.8-flash", posture: PostureSupportHeavy},
	"gemini-3.8-flash-medium":  {canonical: "gemini-3.8-flash", posture: PostureSupportHeavy},
	"gemini-3.8-flash-low":     {canonical: "gemini-3.8-flash", posture: PostureSupportHeavy},
	"gemini-3.8-flash-minimal": {canonical: "gemini-3.8-flash", posture: PostureSupportHeavy},
	"gemini-3.8-pro":           {canonical: "gemini-3.8-pro", posture: PostureBalanced},
	"llama-3-8b":               {canonical: "llama-3-8b", posture: PostureSupportHeavy},
	"llama-3.1-8b":             {canonical: "llama-3.1-8b", posture: PostureSupportHeavy},
	"llama-3.2-3b":             {canonical: "llama-3.2-3b", posture: PostureSupportHeavy},
	"qwen-2.5-7b":              {canonical: "qwen-2.5-7b", posture: PostureSupportHeavy},
	"qwen-2.5-3b":              {canonical: "qwen-2.5-3b", posture: PostureSupportHeavy},
	"claude-3-haiku":           {canonical: "claude-3-haiku", posture: PostureSupportHeavy},
	"claude-3.5-haiku":         {canonical: "claude-3.5-haiku", posture: PostureSupportHeavy},
	"gpt-4o-mini":              {canonical: "gpt-4o-mini", posture: PostureSupportHeavy},

	// Aliases for Small/Flash models
	"claude-3-5-haiku":      {canonical: "claude-3.5-haiku", posture: PostureSupportHeavy},
	"llama-3-8b-instruct":   {canonical: "llama-3-8b", posture: PostureSupportHeavy},
	"llama-3.1-8b-instruct": {canonical: "llama-3.1-8b", posture: PostureSupportHeavy},
	"llama-3.2-3b-instruct": {canonical: "llama-3.2-3b", posture: PostureSupportHeavy},
	"llama3-8b":             {canonical: "llama-3-8b", posture: PostureSupportHeavy},
	"llama3.1-8b":           {canonical: "llama-3.1-8b", posture: PostureSupportHeavy},
	"llama3.2-3b":           {canonical: "llama-3.2-3b", posture: PostureSupportHeavy},
	"qwen-2.5-7b-instruct":  {canonical: "qwen-2.5-7b", posture: PostureSupportHeavy},
	"qwen-2.5-3b-instruct":  {canonical: "qwen-2.5-3b", posture: PostureSupportHeavy},
	"qwen2.5-7b":            {canonical: "qwen-2.5-7b", posture: PostureSupportHeavy},
	"qwen2.5-3b":            {canonical: "qwen-2.5-3b", posture: PostureSupportHeavy},

	// Balanced models -> PostureBalanced
	"gpt-4o":            {canonical: "gpt-4o", posture: PostureBalanced},
	"claude-3-5-sonnet": {canonical: "claude-3-5-sonnet", posture: PostureBalanced},
	"claude-3.7-sonnet": {canonical: "claude-3.7-sonnet", posture: PostureBalanced},
	"gemini-1.5-pro":    {canonical: "gemini-1.5-pro", posture: PostureBalanced},
	"gemini-2.5-pro":    {canonical: "gemini-2.5-pro", posture: PostureBalanced},
	"qwen-2.5-72b":      {canonical: "qwen-2.5-72b", posture: PostureBalanced},
	"deepseek-v3":       {canonical: "deepseek-v3", posture: PostureBalanced},

	// Aliases for Balanced models
	"claude-3.5-sonnet":     {canonical: "claude-3-5-sonnet", posture: PostureBalanced},
	"claude-3-7-sonnet":     {canonical: "claude-3.7-sonnet", posture: PostureBalanced},
	"qwen-2.5-72b-instruct": {canonical: "qwen-2.5-72b", posture: PostureBalanced},
	"qwen2.5-72b":           {canonical: "qwen-2.5-72b", posture: PostureBalanced},
	"deepseek-chat":         {canonical: "deepseek-v3", posture: PostureBalanced},

	// CostHeavy/Reasoning models -> PostureCostHeavy
	"o1":            {canonical: "o1", posture: PostureCostHeavy},
	"o3":            {canonical: "o3", posture: PostureCostHeavy},
	"o3-mini":       {canonical: "o3-mini", posture: PostureCostHeavy},
	"claude-3-opus": {canonical: "claude-3-opus", posture: PostureCostHeavy},
	"deepseek-r1":   {canonical: "deepseek-r1", posture: PostureCostHeavy},

	// Aliases for CostHeavy models
	"o1-preview":        {canonical: "o1", posture: PostureCostHeavy},
	"o1-mini":           {canonical: "o1-mini", posture: PostureCostHeavy},
	"claude-3.0-opus":   {canonical: "claude-3-opus", posture: PostureCostHeavy},
	"deepseek-reasoner": {canonical: "deepseek-r1", posture: PostureCostHeavy},

	// Astra GPT-6 Frontier Reasoning Models -> PostureCostHeavy
	"gpt-6-astra": {canonical: "gpt-6-astra", posture: PostureCostHeavy},
	"gpt-6":       {canonical: "gpt-6-astra", posture: PostureCostHeavy},
	"astra":       {canonical: "gpt-6-astra", posture: PostureCostHeavy},
	"astra-gpt-6": {canonical: "gpt-6-astra", posture: PostureCostHeavy},
	"astra-gpt6":  {canonical: "gpt-6-astra", posture: PostureCostHeavy},
	"gpt6-astra":  {canonical: "gpt-6-astra", posture: PostureCostHeavy},
	"gpt6":        {canonical: "gpt-6-astra", posture: PostureCostHeavy},
	"astra gpt 6": {canonical: "gpt-6-astra", posture: PostureCostHeavy},
	"gpt 6 astra": {canonical: "gpt-6-astra", posture: PostureCostHeavy},
	"gpt-6 astra": {canonical: "gpt-6-astra", posture: PostureCostHeavy},
	"gpt6astra":   {canonical: "gpt-6-astra", posture: PostureCostHeavy},
}

func defaultTurnsForPosture(p Posture) int {
	switch p {
	case PostureSupportHeavy:
		return 8
	case PostureBalanced:
		return 24
	case PostureCostHeavy:
		return 12
	default:
		return 16
	}
}

func defaultDecompositionForPosture(p Posture) bool {
	return p == PostureSupportHeavy
}

func defaultContextForPosture(p Posture) int {
	switch p {
	case PostureSupportHeavy:
		return 16384
	case PostureBalanced:
		return 65536
	case PostureCostHeavy:
		return 131072
	default:
		return 32768
	}
}

func defaultAdvisoryForPosture(p Posture) string {
	switch p {
	case PostureSupportHeavy:
		return "Small/flash model: decompose tasks into atomic S0/S1 units, enforce strict single-turn verification, and constrain turns to avoid repetition loops."
	case PostureBalanced:
		return "Balanced capability: standard multi-step coordination with moderate context runway; decomposes as needed for multi-package tasks."
	case PostureCostHeavy:
		return "High-cost reasoning model: bound turn counts to control reasoning spend; leverage single-turn deep inference."
	default:
		return "Unknown model posture: neutral baseline posture applied; calibrate turn limits and context bounds to observed execution."
	}
}

func isDigits(s string) bool {
	if len(s) == 0 {
		return false
	}
	for i := 0; i < len(s); i++ {
		if s[i] < '0' || s[i] > '9' {
			return false
		}
	}
	return true
}

func trimDateSuffix(s string) string {
	parts := strings.Split(s, "-")
	n := len(parts)
	if n >= 2 {
		last := parts[n-1]
		if len(last) == 8 && isDigits(last) {
			return strings.TrimSuffix(s, "-"+last)
		}
		if n >= 4 {
			p1, p2, p3 := parts[n-3], parts[n-2], parts[n-1]
			if len(p1) == 4 && len(p2) == 2 && len(p3) == 2 && isDigits(p1) && isDigits(p2) && isDigits(p3) {
				return strings.TrimSuffix(s, "-"+p1+"-"+p2+"-"+p3)
			}
		}
	}
	return s
}

func stripVersionSuffix(s string) string {
	if idx := strings.Index(s, ":"); idx >= 0 {
		s = s[:idx]
	}
	s = strings.TrimSuffix(s, "-latest")
	return s
}

func stripReasoningEffortSuffix(s string) string {
	for _, suffix := range []string{"-minimal", "-low", "-medium", "-high"} {
		if strings.HasSuffix(s, suffix) {
			return strings.TrimSuffix(s, suffix)
		}
	}
	return s
}

func normalizeLookupKey(id string) (string, string) {
	trimmed := strings.TrimSpace(id)
	lower := strings.ToLower(trimmed)

	slashIdx := strings.LastIndex(lower, "/")
	var stripped string
	if slashIdx >= 0 && slashIdx < len(lower)-1 {
		stripped = lower[slashIdx+1:]
	}
	return lower, stripped
}

func lookupModel(id string) (modelSpec, bool) {
	lower, stripped := normalizeLookupKey(id)
	if lower == "" {
		return modelSpec{}, false
	}
	// 1. Direct lookup with full normalized string
	if spec, ok := builtins[lower]; ok {
		return spec, true
	}
	// 2. Lookup stripped provider prefix if present
	if stripped != "" {
		if spec, ok := builtins[stripped]; ok {
			return spec, true
		}
	}
	// 3. Strip date suffix (e.g. -20241022 or -2024-08-06)
	candidate := stripped
	if candidate == "" {
		candidate = lower
	}
	if trimmedDate := trimDateSuffix(candidate); trimmedDate != candidate {
		if spec, ok := builtins[trimmedDate]; ok {
			return spec, true
		}
	}
	// 4. Strip version/tag suffixes (e.g. :latest, -latest)
	if trimmedTag := stripVersionSuffix(candidate); trimmedTag != candidate {
		if spec, ok := builtins[trimmedTag]; ok {
			return spec, true
		}
	}
	// 5. Strip reasoning/thinking effort suffix (e.g. -high, -medium, -low, -minimal)
	if trimmedEffort := stripReasoningEffortSuffix(candidate); trimmedEffort != candidate {
		if spec, ok := builtins[trimmedEffort]; ok {
			return spec, true
		}
	}
	return modelSpec{}, false
}

func resolveBaseHint(modelID string) ScopeHint {
	trimmed := strings.TrimSpace(modelID)
	if trimmed == "" {
		return ScopeHint{
			Model:                    modelID,
			CanonicalModel:           "",
			Posture:                  PostureNeutral,
			MaxTurnsRecommended:      defaultTurnsForPosture(PostureNeutral),
			DecompositionRecommended: defaultDecompositionForPosture(PostureNeutral),
			ContextBudgetRecommended: defaultContextForPosture(PostureNeutral),
			Advisory:                 defaultAdvisoryForPosture(PostureNeutral),
			Provenance:               ProvenanceUnknownDefault,
		}
	}

	if spec, ok := lookupModel(trimmed); ok {
		return ScopeHint{
			Model:                    modelID,
			CanonicalModel:           spec.canonical,
			Posture:                  spec.posture,
			MaxTurnsRecommended:      defaultTurnsForPosture(spec.posture),
			DecompositionRecommended: defaultDecompositionForPosture(spec.posture),
			ContextBudgetRecommended: defaultContextForPosture(spec.posture),
			Advisory:                 defaultAdvisoryForPosture(spec.posture),
			Provenance:               ProvenanceBuiltinAlias,
		}
	}

	return ScopeHint{
		Model:                    modelID,
		CanonicalModel:           trimmed,
		Posture:                  PostureNeutral,
		MaxTurnsRecommended:      defaultTurnsForPosture(PostureNeutral),
		DecompositionRecommended: defaultDecompositionForPosture(PostureNeutral),
		ContextBudgetRecommended: defaultContextForPosture(PostureNeutral),
		Advisory:                 defaultAdvisoryForPosture(PostureNeutral),
		Provenance:               ProvenanceUnknownDefault,
	}
}

// ResolveHint emits a ScopeHint for the given modelID. If override is non-nil,
// its fields overlay the resolved default values and the resulting hint carries
// ProvenanceExplicitOverride.
func ResolveHint(modelID string, override *ScopeHint) ScopeHint {
	trimmedID := strings.TrimSpace(modelID)
	base := resolveBaseHint(trimmedID)

	if override == nil {
		return base
	}

	res := base
	res.Provenance = ProvenanceExplicitOverride

	if override.Model != "" {
		res.Model = override.Model
	}
	if override.CanonicalModel != "" {
		res.CanonicalModel = override.CanonicalModel
	}
	if override.Posture != "" {
		res.Posture = override.Posture
		if override.MaxTurnsRecommended == 0 {
			res.MaxTurnsRecommended = defaultTurnsForPosture(override.Posture)
		}
		if override.ContextBudgetRecommended == 0 {
			res.ContextBudgetRecommended = defaultContextForPosture(override.Posture)
		}
		if override.Advisory == "" {
			res.Advisory = defaultAdvisoryForPosture(override.Posture)
		}
		res.DecompositionRecommended = defaultDecompositionForPosture(override.Posture)
	}
	if override.MaxTurnsRecommended > 0 {
		res.MaxTurnsRecommended = override.MaxTurnsRecommended
	}
	if override.ContextBudgetRecommended > 0 {
		res.ContextBudgetRecommended = override.ContextBudgetRecommended
	}
	if override.Advisory != "" {
		res.Advisory = override.Advisory
	}
	if override.Provenance != "" {
		res.Provenance = override.Provenance
	}
	if override.DecompositionRecommended {
		res.DecompositionRecommended = true
	}

	return res
}
