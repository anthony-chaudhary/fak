package normgate_test

// env_test.go — proves New() honors the FAK_NORMGATE_MAX_HELD env override (sensible
// default + escape hatch, matching FAK_NORMGATE=off).

import (
	"context"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/normgate"
)

func TestEnvOverridesLedgerCap(t *testing.T) {
	t.Setenv("FAK_NORMGATE_MAX_HELD", "4")
	ctx := context.Background()
	g := normgate.New() // reads FAK_NORMGATE_MAX_HELD at construction
	for i := 0; i < 100; i++ {
		g.Admit(ctx, untrusted("read_file"), result(secretBody(i)))
	}
	if hl := g.HeldLen(); hl != 4 {
		t.Fatalf("env cap FAK_NORMGATE_MAX_HELD=4 not honored: HeldLen=%d, want 4", hl)
	}
}
