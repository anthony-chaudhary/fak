package vdso

import (
	"context"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// TestTier2HitCarriesAge proves a tier-2 cache hit surfaces age_ms = (hit time - fill
// time), read off the injectable clock so the assertion never depends on wall-clock.
// This is the staleness signal the served-inline renderer turns into "~Nm old" so the
// model can judge whether to force a fresh read.
func TestTier2HitCarriesAge(t *testing.T) {
	v := New(DefaultCacheSize)
	clk := time.Unix(1000, 0)
	v.now = func() time.Time { return clk } // fill stamps t=1000

	ctx := context.Background()
	c := roCall("get_doc", `{"id":"x"}`)
	v.Emit(completeEvent(c, `{"answer":42}`)) // fills tier-2 at t=1000

	clk = clk.Add(3 * time.Minute) // advance to t=1180 before the hit
	res, ok := v.Lookup(ctx, roCall("get_doc", `{"id":"x"}`))
	if !ok {
		t.Fatal("expected a tier-2 hit")
	}
	if res.Meta["served_by"] != "vdso" || res.Meta["tier"] != "2" {
		t.Fatalf("expected a tier-2 vdso hit, got meta=%v", res.Meta)
	}
	if got := res.Meta["age_ms"]; got != "180000" {
		t.Fatalf("age_ms=%q, want 180000 (3 minutes between fill and hit)", got)
	}
}

// TestTier1And3HitsCarryNoAge confirms a pure (tier-1) and a static (tier-3) hit carry
// NO age_ms — they are recomputed/canned every hit, so a fill age is meaningless and the
// served line must render without a staleness clause.
func TestTier1And3HitsCarryNoAge(t *testing.T) {
	v := New(DefaultCacheSize)
	v.RegisterPure("calculate", func(args []byte) ([]byte, bool) { return []byte(`{"sum":2}`), true })
	v.RegisterStatic("list_all", []byte(`["a","b"]`))
	ctx := context.Background()

	pure, ok := v.Lookup(ctx, roCall("calculate", `{"a":1,"b":1}`))
	if !ok {
		t.Fatal("expected a tier-1 pure hit")
	}
	if _, present := pure.Meta["age_ms"]; present {
		t.Fatalf("tier-1 pure hit must carry no age_ms, got meta=%v", pure.Meta)
	}

	static, ok := v.Lookup(ctx, &abi.ToolCall{Tool: "list_all",
		Args: abi.Ref{Kind: abi.RefInline, Inline: []byte(`{}`)},
		Meta: map[string]string{"readOnlyHint": "true", "idempotentHint": "true"}})
	if !ok {
		t.Fatal("expected a tier-3 static hit")
	}
	if _, present := static.Meta["age_ms"]; present {
		t.Fatalf("tier-3 static hit must carry no age_ms, got meta=%v", static.Meta)
	}
}
