package agent

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/model"
)

// TestResolveInKernelRadixBudgetTokens pins the precedence the resident-server
// leak fix relies on: an operator env always wins, an explicit negative field
// means keep-unbounded, an explicit positive field is honored, and otherwise a
// finite context ceiling bounds the tree to one context of cached prefix.
func TestResolveInKernelRadixBudgetTokens(t *testing.T) {
	tests := []struct {
		name    string
		env     string
		envSet  bool
		cfg     int
		context int
		want    int
	}{
		{name: "env wins over explicit field", env: "12345", envSet: true, cfg: 999, context: 20480, want: 12345},
		{name: "env zero is respected (unbounded)", env: "0", envSet: true, cfg: 999, context: 20480, want: 0},
		{name: "negative field keeps unbounded", cfg: -1, context: 20480, want: 0},
		{name: "positive field wins over derived", cfg: 4096, context: 20480, want: 4096},
		{name: "derived from context when field unset", cfg: 0, context: 20480, want: 20480},
		{name: "unbounded when no ceiling known", cfg: 0, context: 0, want: 0},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.envSet {
				t.Setenv("FAK_INKERNEL_RADIX_BUDGET", tt.env)
			} else {
				// A set-but-empty value is not a valid int, so resolution falls
				// through to the field/context arms exactly like an unset var.
				t.Setenv("FAK_INKERNEL_RADIX_BUDGET", "")
			}
			got := resolveInKernelRadixBudgetTokens(InKernelPlannerConfig{RadixBudgetTokens: tt.cfg}, tt.context)
			if got != tt.want {
				t.Fatalf("resolveInKernelRadixBudgetTokens(cfg=%d, ctx=%d) = %d, want %d", tt.cfg, tt.context, got, tt.want)
			}
		})
	}
}

// TestContextTokensForRadixBudget mirrors the ContextWindow min-rule: the
// smaller of the configured cap and the declared window wins, and either alone
// supplies a bound.
func TestContextTokensForRadixBudget(t *testing.T) {
	tests := []struct {
		name       string
		configured int
		declared   int
		want       int
	}{
		{name: "configured narrows declared", configured: 20480, declared: 262144, want: 20480},
		{name: "declared when configured zero", configured: 0, declared: 32768, want: 32768},
		{name: "configured when declared zero", configured: 8192, declared: 0, want: 8192},
		{name: "declared when configured wider", configured: 999999, declared: 32768, want: 32768},
		{name: "zero when neither known", configured: 0, declared: 0, want: 0},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			m := &model.Model{Cfg: model.Config{MaxPositionEmbeddings: tt.declared}}
			got := contextTokensForRadixBudget(InKernelPlannerConfig{ContextTokens: tt.configured}, m)
			if got != tt.want {
				t.Fatalf("contextTokensForRadixBudget(cfg=%d, declared=%d) = %d, want %d", tt.configured, tt.declared, got, tt.want)
			}
		})
	}
}
