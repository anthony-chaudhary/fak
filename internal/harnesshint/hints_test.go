package harnesshint

import (
	"go/parser"
	"go/token"
	"path/filepath"
	"strings"
	"testing"
)

func TestResolveHint_SmallFlashModels(t *testing.T) {
	models := []string{
		"gemini-1.5-flash",
		"gemini-2.0-flash",
		"gemini-2.5-flash",
		"gemini-3.8-flash",
		"llama-3-8b",
		"llama-3.1-8b",
		"llama-3.2-3b",
		"qwen-2.5-7b",
		"qwen-2.5-3b",
		"claude-3-haiku",
		"claude-3.5-haiku",
		"gpt-4o-mini",
	}

	for _, model := range models {
		t.Run(model, func(t *testing.T) {
			hint := ResolveHint(model, nil)

			if hint.Model != model {
				t.Errorf("expected Model=%q, got %q", model, hint.Model)
			}
			if hint.CanonicalModel != model {
				t.Errorf("expected CanonicalModel=%q, got %q", model, hint.CanonicalModel)
			}
			if hint.Posture != PostureSupportHeavy {
				t.Errorf("expected Posture=%v, got %v", PostureSupportHeavy, hint.Posture)
			}
			if !hint.DecompositionRecommended {
				t.Errorf("expected DecompositionRecommended=true for small/flash model %q", model)
			}
			if hint.Provenance != ProvenanceBuiltinAlias {
				t.Errorf("expected Provenance=%q, got %q", ProvenanceBuiltinAlias, hint.Provenance)
			}
			if hint.MaxTurnsRecommended <= 0 {
				t.Errorf("expected positive MaxTurnsRecommended, got %d", hint.MaxTurnsRecommended)
			}
			if hint.ContextBudgetRecommended <= 0 {
				t.Errorf("expected positive ContextBudgetRecommended, got %d", hint.ContextBudgetRecommended)
			}
			if hint.Advisory == "" {
				t.Errorf("expected non-empty Advisory")
			}
		})
	}
}

func TestResolveHint_BalancedModels(t *testing.T) {
	models := []string{
		"gpt-4o",
		"claude-3-5-sonnet",
		"claude-3.7-sonnet",
		"gemini-1.5-pro",
		"gemini-2.5-pro",
		"qwen-2.5-72b",
		"deepseek-v3",
	}

	for _, model := range models {
		t.Run(model, func(t *testing.T) {
			hint := ResolveHint(model, nil)

			if hint.Model != model {
				t.Errorf("expected Model=%q, got %q", model, hint.Model)
			}
			if hint.CanonicalModel != model {
				t.Errorf("expected CanonicalModel=%q, got %q", model, hint.CanonicalModel)
			}
			if hint.Posture != PostureBalanced {
				t.Errorf("expected Posture=%v, got %v", PostureBalanced, hint.Posture)
			}
			if hint.DecompositionRecommended {
				t.Errorf("expected DecompositionRecommended=false for balanced model %q", model)
			}
			if hint.Provenance != ProvenanceBuiltinAlias {
				t.Errorf("expected Provenance=%q, got %q", ProvenanceBuiltinAlias, hint.Provenance)
			}
			if hint.MaxTurnsRecommended <= 0 {
				t.Errorf("expected positive MaxTurnsRecommended, got %d", hint.MaxTurnsRecommended)
			}
			if hint.ContextBudgetRecommended <= 0 {
				t.Errorf("expected positive ContextBudgetRecommended, got %d", hint.ContextBudgetRecommended)
			}
			if hint.Advisory == "" {
				t.Errorf("expected non-empty Advisory")
			}
		})
	}
}

func TestResolveHint_CostHeavyReasoningModels(t *testing.T) {
	models := []string{
		"o1",
		"o3",
		"o3-mini",
		"claude-3-opus",
		"deepseek-r1",
	}

	for _, model := range models {
		t.Run(model, func(t *testing.T) {
			hint := ResolveHint(model, nil)

			if hint.Model != model {
				t.Errorf("expected Model=%q, got %q", model, hint.Model)
			}
			if hint.CanonicalModel != model {
				t.Errorf("expected CanonicalModel=%q, got %q", model, hint.CanonicalModel)
			}
			if hint.Posture != PostureCostHeavy {
				t.Errorf("expected Posture=%v, got %v", PostureCostHeavy, hint.Posture)
			}
			if hint.DecompositionRecommended {
				t.Errorf("expected DecompositionRecommended=false for cost-heavy model %q", model)
			}
			if hint.Provenance != ProvenanceBuiltinAlias {
				t.Errorf("expected Provenance=%q, got %q", ProvenanceBuiltinAlias, hint.Provenance)
			}
			if hint.MaxTurnsRecommended <= 0 {
				t.Errorf("expected positive MaxTurnsRecommended, got %d", hint.MaxTurnsRecommended)
			}
			if hint.ContextBudgetRecommended <= 0 {
				t.Errorf("expected positive ContextBudgetRecommended, got %d", hint.ContextBudgetRecommended)
			}
			if hint.Advisory == "" {
				t.Errorf("expected non-empty Advisory")
			}
		})
	}
}

func TestResolveHint_NormalizationAndPrefixes(t *testing.T) {
	tests := []struct {
		input             string
		expectedCanonical string
		expectedPosture   Posture
	}{
		{"openai/gpt-4o", "gpt-4o", PostureBalanced},
		{"anthropic/claude-3.5-haiku", "claude-3.5-haiku", PostureSupportHeavy},
		{"anthropic/claude-3-5-sonnet", "claude-3-5-sonnet", PostureBalanced},
		{"google/gemini-3.8-flash", "gemini-3.8-flash", PostureSupportHeavy},
		{"deepseek/deepseek-r1", "deepseek-r1", PostureCostHeavy},
		{"meta-llama/Llama-3.1-8B", "llama-3.1-8b", PostureSupportHeavy},
		{"meta-llama/Llama-3.1-8B-Instruct", "llama-3.1-8b", PostureSupportHeavy},
		{"qwen/qwen-2.5-7b", "qwen-2.5-7b", PostureSupportHeavy},
		{"deepseek-ai/deepseek-v3", "deepseek-v3", PostureBalanced},
		{"models/gemini-1.5-pro", "gemini-1.5-pro", PostureBalanced},
		{"azure/openai/o3-mini", "o3-mini", PostureCostHeavy},
		{"  OPENAI/GPT-4O  ", "gpt-4o", PostureBalanced},
		{"  GEMINI-2.5-FLASH  ", "gemini-2.5-flash", PostureSupportHeavy},
		{"gpt-4o-2024-08-06", "gpt-4o", PostureBalanced},
		{"claude-3-5-sonnet-20241022", "claude-3-5-sonnet", PostureBalanced},
		{"gemini-1.5-flash-latest", "gemini-1.5-flash", PostureSupportHeavy},
		{"claude-3.5-sonnet", "claude-3-5-sonnet", PostureBalanced},
		{"deepseek-chat", "deepseek-v3", PostureBalanced},
		{"deepseek-reasoner", "deepseek-r1", PostureCostHeavy},
	}

	for _, tc := range tests {
		t.Run(tc.input, func(t *testing.T) {
			hint := ResolveHint(tc.input, nil)
			if hint.CanonicalModel != tc.expectedCanonical {
				t.Errorf("input %q: expected CanonicalModel=%q, got %q", tc.input, tc.expectedCanonical, hint.CanonicalModel)
			}
			if hint.Posture != tc.expectedPosture {
				t.Errorf("input %q: expected Posture=%v, got %v", tc.input, tc.expectedPosture, hint.Posture)
			}
			if hint.Provenance != ProvenanceBuiltinAlias {
				t.Errorf("input %q: expected Provenance=%q, got %q", tc.input, ProvenanceBuiltinAlias, hint.Provenance)
			}
		})
	}
}

func TestResolveHint_UnknownModels(t *testing.T) {
	tests := []string{
		"unknown-experimental-model",
		"my-org/custom-fine-tuned",
		"llama-4-99b-super",
		"random-agent-kernel",
		"",
		"   ",
	}

	for _, input := range tests {
		t.Run("unknown_"+input, func(t *testing.T) {
			hint := ResolveHint(input, nil)

			if hint.Posture != PostureNeutral {
				t.Errorf("input %q: expected Posture=%v, got %v", input, PostureNeutral, hint.Posture)
			}
			if hint.Provenance != ProvenanceUnknownDefault {
				t.Errorf("input %q: expected Provenance=%q, got %q", input, ProvenanceUnknownDefault, hint.Provenance)
			}
			if hint.DecompositionRecommended {
				t.Errorf("input %q: expected DecompositionRecommended=false for unknown model", input)
			}
			if hint.MaxTurnsRecommended <= 0 {
				t.Errorf("input %q: expected positive MaxTurnsRecommended", input)
			}
			if hint.ContextBudgetRecommended <= 0 {
				t.Errorf("input %q: expected positive ContextBudgetRecommended", input)
			}
			if hint.Advisory == "" {
				t.Errorf("input %q: expected non-empty Advisory", input)
			}
		})
	}
}

func TestResolveHint_Overrides(t *testing.T) {
	t.Run("nil override returns base", func(t *testing.T) {
		base := ResolveHint("gemini-3.8-flash", nil)
		if base.Provenance != ProvenanceBuiltinAlias {
			t.Errorf("expected Provenance=%q, got %q", ProvenanceBuiltinAlias, base.Provenance)
		}
		if base.Posture != PostureSupportHeavy {
			t.Errorf("expected Posture=%v, got %v", PostureSupportHeavy, base.Posture)
		}
	})

	t.Run("explicit full override", func(t *testing.T) {
		override := &ScopeHint{
			Model:                    "custom-override-model",
			CanonicalModel:           "custom-canonical",
			Posture:                  PostureCostHeavy,
			MaxTurnsRecommended:      5,
			DecompositionRecommended: true,
			ContextBudgetRecommended: 8192,
			Advisory:                 "Custom test advisory",
			Provenance:               ProvenanceExplicitOverride,
		}

		hint := ResolveHint("gemini-3.8-flash", override)
		if hint.Model != "custom-override-model" {
			t.Errorf("expected Model=%q, got %q", "custom-override-model", hint.Model)
		}
		if hint.CanonicalModel != "custom-canonical" {
			t.Errorf("expected CanonicalModel=%q, got %q", "custom-canonical", hint.CanonicalModel)
		}
		if hint.Posture != PostureCostHeavy {
			t.Errorf("expected Posture=%v, got %v", PostureCostHeavy, hint.Posture)
		}
		if hint.MaxTurnsRecommended != 5 {
			t.Errorf("expected MaxTurnsRecommended=5, got %d", hint.MaxTurnsRecommended)
		}
		if !hint.DecompositionRecommended {
			t.Errorf("expected DecompositionRecommended=true")
		}
		if hint.ContextBudgetRecommended != 8192 {
			t.Errorf("expected ContextBudgetRecommended=8192, got %d", hint.ContextBudgetRecommended)
		}
		if hint.Advisory != "Custom test advisory" {
			t.Errorf("expected Advisory=%q, got %q", "Custom test advisory", hint.Advisory)
		}
		if hint.Provenance != ProvenanceExplicitOverride {
			t.Errorf("expected Provenance=%q, got %q", ProvenanceExplicitOverride, hint.Provenance)
		}
	})

	t.Run("partial override inherits base values", func(t *testing.T) {
		override := &ScopeHint{
			MaxTurnsRecommended: 3,
		}

		hint := ResolveHint("gpt-4o", override)
		if hint.CanonicalModel != "gpt-4o" {
			t.Errorf("expected CanonicalModel=gpt-4o, got %q", hint.CanonicalModel)
		}
		if hint.Posture != PostureBalanced {
			t.Errorf("expected Posture=%v, got %v", PostureBalanced, hint.Posture)
		}
		if hint.MaxTurnsRecommended != 3 {
			t.Errorf("expected overridden MaxTurnsRecommended=3, got %d", hint.MaxTurnsRecommended)
		}
		if hint.ContextBudgetRecommended != defaultContextForPosture(PostureBalanced) {
			t.Errorf("expected base context budget, got %d", hint.ContextBudgetRecommended)
		}
		if hint.Provenance != ProvenanceExplicitOverride {
			t.Errorf("expected Provenance=%q, got %q", ProvenanceExplicitOverride, hint.Provenance)
		}
	})

	t.Run("override posture adapts defaults", func(t *testing.T) {
		override := &ScopeHint{
			Posture: PostureSupportHeavy,
		}

		hint := ResolveHint("gpt-4o", override)
		if hint.Posture != PostureSupportHeavy {
			t.Errorf("expected Posture=%v, got %v", PostureSupportHeavy, hint.Posture)
		}
		if !hint.DecompositionRecommended {
			t.Errorf("expected DecompositionRecommended=true for overridden PostureSupportHeavy")
		}
		if hint.MaxTurnsRecommended != defaultTurnsForPosture(PostureSupportHeavy) {
			t.Errorf("expected turns default for PostureSupportHeavy, got %d", hint.MaxTurnsRecommended)
		}
		if hint.Provenance != ProvenanceExplicitOverride {
			t.Errorf("expected Provenance=%q, got %q", ProvenanceExplicitOverride, hint.Provenance)
		}
	})
}

func TestZeroNetworkAndHermeticImports(t *testing.T) {
	// Parse hints.go and doc.go to verify zero network, io, or OS runtime dependencies.
	fset := token.NewFileSet()
	files := []string{"hints.go", "doc.go"}
	for _, f := range files {
		path := filepath.Join(".", f)
		node, err := parser.ParseFile(fset, path, nil, parser.ImportsOnly)
		if err != nil {
			t.Fatalf("failed to parse %s: %v", f, err)
		}
		for _, imp := range node.Imports {
			val := strings.Trim(imp.Path.Value, `"`)
			if strings.HasPrefix(val, "net") || strings.HasPrefix(val, "os") || strings.HasPrefix(val, "syscall") {
				t.Errorf("%s illegally imports non-hermetic package %q", f, val)
			}
		}
	}
}

func TestPosture_ValidAndString(t *testing.T) {
	postures := []Posture{
		PostureSupportHeavy,
		PostureBalanced,
		PostureCostHeavy,
		PostureNeutral,
	}

	for _, p := range postures {
		if !p.Valid() {
			t.Errorf("expected posture %v to be valid", p)
		}
		if p.String() == "" {
			t.Errorf("expected non-empty String() for posture %v", p)
		}
	}

	invalid := Posture("invalid_posture")
	if invalid.Valid() {
		t.Errorf("expected invalid_posture to be invalid")
	}
}

func BenchmarkResolveHint(b *testing.B) {
	models := []string{
		"gemini-3.8-flash",
		"openai/gpt-4o",
		"anthropic/claude-3-5-sonnet",
		"deepseek/deepseek-r1",
		"meta-llama/Llama-3.1-8B-Instruct",
		"unknown-model-benchmark",
	}

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		_ = ResolveHint(models[i%len(models)], nil)
	}
}
