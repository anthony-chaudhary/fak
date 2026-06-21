package ctxmmu_test

// env_test.go — proves the sensible-default + escape-hatch contract: New() honors the
// FAK_CTXMMU_MAX_HELD env override, and a bad/empty value fails safe to DefaultMaxHeld.

import (
	"context"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/ctxmmu"
)

func TestEnvOverridesLedgerCap(t *testing.T) {
	t.Setenv("FAK_CTXMMU_MAX_HELD", "4")
	ctx := context.Background()
	m := ctxmmu.New() // reads FAK_CTXMMU_MAX_HELD at construction
	for i := 0; i < 100; i++ {
		c := call("read_file")
		m.Admit(ctx, c, result(c, poison(i)))
	}
	if hl := m.HeldLen(); hl != 4 {
		t.Fatalf("env cap FAK_CTXMMU_MAX_HELD=4 not honored: HeldLen=%d, want 4", hl)
	}
}

func TestEnvBadValueFallsBackToDefault(t *testing.T) {
	t.Setenv("FAK_CTXMMU_MAX_HELD", "not-a-number")
	ctx := context.Background()
	m := ctxmmu.New() // bad value must fail safe to DefaultMaxHeld (8192), never to 0
	for i := 0; i < 20; i++ {
		c := call("read_file")
		m.Admit(ctx, c, result(c, poison(i)))
	}
	if m.Evicted() != 0 {
		t.Fatalf("bad env value should fall back to the large default (no eviction at 20 < 8192), evicted=%d", m.Evicted())
	}
}
