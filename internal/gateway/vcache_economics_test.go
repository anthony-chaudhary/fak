package gateway

import (
	"math"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/vcachegov"
	"github.com/anthony-chaudhary/fak/internal/vcacheobserve"
)

func approxEq(a, b float64) bool { return math.Abs(a-b) < 1e-6 }

// TestProviderCacheNetSavingsAccumulatesAndReconciles proves the session retains BOTH
// the read and write axes and that ProviderCacheNetSavings() reproduces a hand-built
// vcachegov proof over the same totals — methodology parity, not just a number.
func TestProviderCacheNetSavingsAccumulatesAndReconciles(t *testing.T) {
	m := newGatewayMetrics(time.Now())
	// turn 1 warms (creation, no read); turns 2-3 read it back.
	m.observeInference(100, 10, 0, 40000, "end_turn", time.Second)
	m.observeInference(50, 10, 40000, 500, "end_turn", time.Second)
	m.observeInference(50, 10, 40000, 500, "end_turn", time.Second)

	sum := m.adjudicationSummary()
	if sum.InputTokens != 200 {
		t.Fatalf("InputTokens: got %d want 200", sum.InputTokens)
	}
	if sum.CachedPromptTokens != 80000 {
		t.Fatalf("CachedPromptTokens (read): got %d want 80000", sum.CachedPromptTokens)
	}
	if sum.CacheCreationTokens != 41000 {
		t.Fatalf("CacheCreationTokens (write): got %d want 41000", sum.CacheCreationTokens)
	}

	got := sum.ProviderCacheNetSavings()
	// The hand-built reference MUST set ReadMult explicitly: vcachegov defaults the write
	// multipliers but NOT the read one, so an unset ReadMult would price reads at 0x.
	want := vcachegov.ProveTelemetrySavings(vcachegov.TelemetrySavingsInput{
		Rows:        []vcachegov.TelemetryRow{{InputTokens: 200, CacheReadInputTokens: 80000, CacheCreationInputTokens: 41000}},
		ReadMult:    CacheReadMultiplier,
		Write5mMult: CacheWrite5mMultiplier,
		Write1hMult: CacheWrite1hMultiplier,
	})
	if !approxEq(got.SavedTokenEquiv, want.SavedTokenEquiv) || got.Status != want.Status {
		t.Fatalf("ProviderCacheNetSavings drifted from the engine: got %+v want %+v", got, want)
	}
	// NET = read rebate (0.9*80000=72000) minus write premium (0.25*41000=10250) = 61750.
	if !approxEq(got.SavedTokenEquiv, 61750) {
		t.Fatalf("net saving: got %.1f want 61750", got.SavedTokenEquiv)
	}
}

// TestProviderCacheNetSavingsIsNetNotGross proves the surface accounts for the write
// premium: a cold-write-only session reads NEGATIVE and REFUTED (the read-only
// ProviderCacheSavingsUSD would show a non-negative number here), while a warm sequence
// turns PROVEN and positive.
func TestProviderCacheNetSavingsIsNetNotGross(t *testing.T) {
	cold := newGatewayMetrics(time.Now())
	cold.observeInference(100, 10, 0, 40000, "end_turn", time.Second) // wrote, never read
	cs := cold.adjudicationSummary().ProviderCacheNetSavings()
	if cs.Status != vcachegov.ProofRefuted {
		t.Fatalf("cold-write-only should REFUTE (net negative), got %s", cs.Status)
	}
	if cs.SavedTokenEquiv >= 0 {
		t.Fatalf("cold-write-only net saving should be negative, got %.1f", cs.SavedTokenEquiv)
	}

	warm := newGatewayMetrics(time.Now())
	warm.observeInference(100, 10, 0, 40000, "end_turn", time.Second)
	for i := 0; i < 4; i++ {
		warm.observeInference(50, 10, 40000, 0, "end_turn", time.Second)
	}
	ws := warm.adjudicationSummary().ProviderCacheNetSavings()
	if ws.Status != vcachegov.ProofProven {
		t.Fatalf("warm sequence should PROVE, got %s", ws.Status)
	}
	if ws.SavedTokenEquiv <= 0 {
		t.Fatalf("warm net saving should be positive, got %.1f", ws.SavedTokenEquiv)
	}
}

// TestGatewayNetSavingsMatchesObserve proves the live gateway number equals what
// `fak vcache observe` computes offline on the same totals — the "same engine, same
// number" guarantee that keeps the live and offline surfaces from disagreeing.
func TestGatewayNetSavingsMatchesObserve(t *testing.T) {
	m := newGatewayMetrics(time.Now())
	turns := []vcacheobserve.Turn{
		{Family: "s", UnixMillis: 0, InputTokens: 100, CacheCreation: 40000},
		{Family: "s", UnixMillis: 1000, InputTokens: 50, CacheRead: 40000, CacheCreation: 500},
		{Family: "s", UnixMillis: 2000, InputTokens: 50, CacheRead: 40000, CacheCreation: 500},
	}
	for _, tn := range turns {
		m.observeInference(int(tn.InputTokens), 10, int(tn.CacheRead), int(tn.CacheCreation), "end_turn", time.Second)
	}
	live := m.adjudicationSummary().ProviderCacheNetSavings()
	off := vcacheobserve.Observe(turns, vcacheobserve.DefaultMultipliers()).Aggregate

	if !approxEq(live.BaselineTokenEquiv, off.BaselineTokenEquiv) ||
		!approxEq(live.ActualTokenEquiv, off.ActualTokenEquiv) ||
		!approxEq(live.SavedTokenEquiv, off.SavedTokenEquiv) {
		t.Fatalf("live gateway economics != offline observe aggregate:\n live=%+v\n off =%+v", live, off)
	}
}

// TestWriteVCacheMetricsProvenanceAndZeroGuard proves the fak_vcache_* family is present
// with the OBSERVED/NET provenance discipline after activity, and that an idle gateway
// emits NO fak_vcache_* series (no phantom) and a nil debug block.
func TestWriteVCacheMetricsProvenanceAndZeroGuard(t *testing.T) {
	idle := newGatewayMetrics(time.Now())
	var zb strings.Builder
	idle.writeVCacheMetrics(&zb)
	if strings.Contains(zb.String(), "fak_vcache_") {
		t.Fatalf("idle gateway must emit no fak_vcache_* series, got:\n%s", zb.String())
	}
	if vcacheVarsFromSnapshot(idle.inferenceSnapshotData()) != nil {
		t.Fatal("idle gateway debug vcache block must be nil")
	}

	m := newGatewayMetrics(time.Now())
	m.observeInference(100, 10, 0, 40000, "end_turn", time.Second)
	m.observeInference(50, 10, 40000, 0, "end_turn", time.Second)
	var b strings.Builder
	m.writeVCacheMetrics(&b)
	out := b.String()
	for _, want := range []string{
		"fak_vcache_saved_token_equiv", "fak_vcache_hit_rate", "fak_vcache_multiplier",
		"fak_vcache_proven", "fak_vcache_cache_creation_tokens_total", "OBSERVED", "NET",
	} {
		if !strings.Contains(out, want) {
			t.Fatalf("metrics missing %q:\n%s", want, out)
		}
	}
	if vb := vcacheVarsFromSnapshot(m.inferenceSnapshotData()); vb == nil {
		t.Fatal("active gateway debug vcache block must be populated")
	} else if vb.Status != string(vcachegov.ProofProven) {
		t.Fatalf("debug vcache status: got %q want PROVEN", vb.Status)
	}
}
