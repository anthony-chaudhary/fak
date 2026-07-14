package bench

import (
	"bytes"
	"math"
	"os"
	"path/filepath"
	"testing"
)

// TestFloorReloadCacheBench is the re-runnable witness for issue #3971: it produces
// the mid-session-floor-reload cache-cost report and asserts the acceptance
// criteria — a forfeited-cached-tokens number per reload at 3 session lengths, the
// reused/fresh split derived with and without the reload, and the net-true
// assumptions block naming #2916 as the live-run promotion path. Regenerate the
// golden with UPDATE_GOLDEN=1.
func TestFloorReloadCacheBench(t *testing.T) {
	m := DefaultFloorReloadModel()
	r := BuildFloorReloadCacheReport()

	if r.Schema != FloorReloadCacheSchema {
		t.Fatalf("schema = %q; want %q", r.Schema, FloorReloadCacheSchema)
	}
	if r.Provenance.Kind != ProvenanceSimulated {
		t.Fatalf("provenance = %q; want %q (hermetic model)", r.Provenance.Kind, ProvenanceSimulated)
	}
	// Acceptance: forfeited-cached-tokens reported at 3 session lengths.
	if len(r.Sweep) != 3 { //boundarylint:ignore CHANGE_DETECTOR_TEST the sweep is a fixed 3-length design set {50,100,200}
		t.Fatalf("sweep points = %d; want 3 session lengths (50,100,200)", len(r.Sweep))
	}

	for _, p := range r.Sweep {
		// The reload must be mid-session: after the cache is warm (turn > 0) and
		// before the session ends. A turn-0 reload would forfeit nothing.
		if !(p.ReloadAtTurn > 0 && p.ReloadAtTurn < p.SessionTurns) {
			t.Errorf("%d turns: reload at %d not mid-session (0 < k < n)", p.SessionTurns, p.ReloadAtTurn)
		}

		// Analytic identity: the reload forfeits the tools block plus every
		// conversation turn resident before it — ToolsBlock + (k-1)*Growth.
		wantForfeit := m.ToolsBlockTokens + (p.ReloadAtTurn-1)*m.GrowthPerTurn
		if p.ForfeitedCachedTokens != wantForfeit {
			t.Errorf("%d turns: forfeited = %d; want ToolsBlock+(k-1)*Growth = %d",
				p.SessionTurns, p.ForfeitedCachedTokens, wantForfeit)
		}

		// The forfeit is exactly the reused->fresh flip between the arms, measured
		// from BOTH sides (fresh gained == reused lost). This is the "compute
		// reused/fresh split with and without the reload" acceptance, made a gate.
		if got := p.Reload.FreshTokens - p.Baseline.FreshTokens; got != p.ForfeitedCachedTokens {
			t.Errorf("%d turns: fresh gained %d != forfeited %d", p.SessionTurns, got, p.ForfeitedCachedTokens)
		}
		if got := p.Baseline.ReusedTokens - p.Reload.ReusedTokens; got != p.ForfeitedCachedTokens {
			t.Errorf("%d turns: reused lost %d != forfeited %d", p.SessionTurns, got, p.ForfeitedCachedTokens)
		}

		// Total input is conserved: a reload moves tokens reused->fresh, it does not
		// create or destroy any (same resident bytes, different cache split).
		if bt, rt := p.Baseline.ReusedTokens+p.Baseline.FreshTokens, p.Reload.ReusedTokens+p.Reload.FreshTokens; bt != rt {
			t.Errorf("%d turns: total input not conserved: baseline %d != reload %d", p.SessionTurns, bt, rt)
		}

		// The surcharge is the forfeited tokens re-priced from read to write, and it
		// equals the billed difference between the arms. Both routes must agree.
		wantExtra := int(math.Round(float64(p.ForfeitedCachedTokens) * (m.CacheWriteMult - m.CacheReadMult)))
		if p.ReloadExtraBilled != wantExtra {
			t.Errorf("%d turns: extra billed = %d; want forfeited*(write-read) = %d",
				p.SessionTurns, p.ReloadExtraBilled, wantExtra)
		}
		if got := p.Reload.BilledTokens - p.Baseline.BilledTokens; got != p.ReloadExtraBilled {
			t.Errorf("%d turns: billed delta %d != reload_extra_billed %d", p.SessionTurns, got, p.ReloadExtraBilled)
		}

		// A reload strictly costs more than the same session without one.
		if !(p.Reload.BilledTokens > p.Baseline.BilledTokens) {
			t.Errorf("%d turns: reload billed %d not > baseline %d", p.SessionTurns, p.Reload.BilledTokens, p.Baseline.BilledTokens)
		}

		// The three headline numbers are well-formed.
		if !(p.FractionOfSessionSpend > 0 && p.FractionOfSessionSpend < 1) {
			t.Errorf("%d turns: fraction_of_session_spend = %.4f; want in (0,1)", p.SessionTurns, p.FractionOfSessionSpend)
		}
		if p.TurnsToAmortize <= 0 {
			t.Errorf("%d turns: turns_to_amortize = %.4f; want > 0", p.SessionTurns, p.TurnsToAmortize)
		}
	}

	// Trend witness (not a single point): a single reload's share of session spend
	// SHRINKS as the session length that amortizes it grows.
	for i := 1; i < len(r.Sweep); i++ {
		prev, cur := r.Sweep[i-1], r.Sweep[i]
		if !(cur.FractionOfSessionSpend < prev.FractionOfSessionSpend) {
			t.Errorf("fraction of spend should shrink with session length: %d turns %.4f not < %d turns %.4f",
				cur.SessionTurns, cur.FractionOfSessionSpend, prev.SessionTurns, prev.FractionOfSessionSpend)
		}
	}

	// Headline is the deepest swept session (the worst case in absolute forfeit).
	deepest := r.Sweep[len(r.Sweep)-1]
	if r.Headline.SessionTurns != deepest.SessionTurns || r.Headline.ForfeitedCachedTokens != deepest.ForfeitedCachedTokens {
		t.Errorf("headline = %d turns / %d forfeited; want deepest sweep point %d / %d",
			r.Headline.SessionTurns, r.Headline.ForfeitedCachedTokens, deepest.SessionTurns, deepest.ForfeitedCachedTokens)
	}

	// Net-true hygiene: the report names its assumptions + promotion/demotion +
	// invalidating unknown, and the promotion path is #2916's live metric (the
	// acceptance criterion that the assumptions block name the promotion path).
	if len(r.Assumptions) == 0 || r.Promotion == "" || r.DemotionRetirement == "" || r.InvalidatingUnknown == "" {
		t.Errorf("report must name assumptions + promotion + demotion + invalidating unknown")
	}
	if !bytes.Contains([]byte(r.Promotion), []byte("#2916")) {
		t.Errorf("promotion path must name the #2916 live-run metric; got %q", r.Promotion)
	}

	// The report marshals to stable JSON (the re-derivable artifact).
	got, err := r.JSON()
	if err != nil {
		t.Fatalf("JSON: %v", err)
	}
	if len(got) == 0 {
		t.Fatal("empty report JSON")
	}

	// Golden: the committed benchmark result artifact. Regenerate with UPDATE_GOLDEN=1.
	golden := filepath.Join("testdata", "floorreloadcache_report.json")
	if os.Getenv("UPDATE_GOLDEN") != "" {
		if err := os.WriteFile(golden, append(got, '\n'), 0o644); err != nil {
			t.Fatalf("write golden: %v", err)
		}
		t.Logf("updated golden %s", golden)
		return
	}
	want, err := os.ReadFile(golden)
	if err != nil {
		t.Fatalf("read golden (run with UPDATE_GOLDEN=1 to create): %v", err)
	}
	if !bytes.Equal(bytes.TrimRight(want, "\n"), bytes.TrimRight(got, "\n")) {
		t.Errorf("report drifted from golden %s; re-run with UPDATE_GOLDEN=1 if intended", golden)
	}
}
