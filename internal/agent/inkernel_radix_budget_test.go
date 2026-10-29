package agent

import (
	"errors"
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

// TestResolveInKernelRadixHostL2Bytes pins the host-DRAM L2 KV reserve default
// (perf-parity-20261029 umem TICKET-02): an env byte count wins exactly (0 is the
// OFF arm), otherwise 15% of MemAvailable, unknown host memory stays at today's 0,
// and a refused env fails closed to 0 with ErrInKernelRadixHostL2Config.
// fak-test:runtime fast est=1ms lane=default
func TestResolveInKernelRadixHostL2Bytes(t *testing.T) {
	const gib = int64(1) << 30
	host := func(avail int64, known bool) func() (int64, int64, bool) {
		return func() (int64, int64, bool) { return 2 * avail, avail, known }
	}
	env := func(kv map[string]string) func(string) (string, bool) {
		return func(k string) (string, bool) { v, ok := kv[k]; return v, ok }
	}
	tests := []struct {
		name    string
		env     map[string]string
		avail   int64
		known   bool
		want    int64
		source  string
		refused bool
	}{
		{name: "unset 64GiB avail takes 15%", avail: 64 * gib, known: true, want: (64 * gib * 15 / 100) &^ (1<<20 - 1), source: InKernelRadixHostL2SourceDefault},
		{name: "env zero is the OFF arm", env: map[string]string{inKernelRadixHostL2BytesEnv: "0"}, avail: 64 * gib, known: true, want: 0, source: InKernelRadixHostL2SourceEnv},
		{name: "env bytes honoured exactly", env: map[string]string{inKernelRadixHostL2BytesEnv: "123456789"}, avail: 64 * gib, known: true, want: 123456789, source: InKernelRadixHostL2SourceEnv},
		{name: "env bytes win on unknown host", env: map[string]string{inKernelRadixHostL2BytesEnv: "4096"}, want: 4096, source: InKernelRadixHostL2SourceEnv},
		{name: "unknown host stays off", avail: 64 * gib, known: false, want: 0, source: InKernelRadixHostL2SourceUnknown},
		{name: "small host clamps to margin", avail: 6 * gib, known: true, want: 0, source: InKernelRadixHostL2SourceDefault},
		{name: "fraction 0.20 honoured", env: map[string]string{inKernelRadixHostL2FractionEnv: "0.20"}, avail: 100 * gib, known: true, want: 20 * gib, source: inKernelRadixHostL2FractionEnv + "=0.2 x MemAvailable"},
		{name: "fraction above range refused", env: map[string]string{inKernelRadixHostL2FractionEnv: "0.5"}, avail: 64 * gib, known: true, want: 0, source: InKernelRadixHostL2SourceRefused, refused: true},
		{name: "fraction below range refused", env: map[string]string{inKernelRadixHostL2FractionEnv: "0.05"}, avail: 64 * gib, known: true, want: 0, source: InKernelRadixHostL2SourceRefused, refused: true},
		{name: "malformed bytes refused", env: map[string]string{inKernelRadixHostL2BytesEnv: "lots"}, avail: 64 * gib, known: true, want: 0, source: InKernelRadixHostL2SourceRefused, refused: true},
		{name: "negative bytes refused", env: map[string]string{inKernelRadixHostL2BytesEnv: "-1"}, avail: 64 * gib, known: true, want: 0, source: InKernelRadixHostL2SourceRefused, refused: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := resolveInKernelRadixHostL2Bytes(env(tt.env), host(tt.avail, tt.known))
			if got.Bytes != tt.want || got.Source != tt.source || errors.Is(got.Err, ErrInKernelRadixHostL2Config) != tt.refused {
				t.Fatalf("resolve = %+v, want bytes=%d source=%q refused=%v", got, tt.want, tt.source, tt.refused)
			}
		})
	}
}
