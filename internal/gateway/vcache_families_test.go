package gateway

import (
	"encoding/json"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/vcachegov"
	"github.com/anthony-chaudhary/fak/internal/vcacheobserve"
)

// TestLogInferenceTurnPopulatesVCacheWindow proves the LIVE wiring: a served turn flowing
// through the per-turn chokepoint logInferenceTurn records into the per-family window even
// with both sinks (--log, --debug-stats) off, so /debug/vars exposes the per-family view
// on real traffic. The family is the trace; the axes are the provider's own usage.
func TestLogInferenceTurnPopulatesVCacheWindow(t *testing.T) {
	s := newResetShadowServer() // debugStatsf nil, logf nil — the bare live path
	s.logInferenceTurn("trace-A", "anthropic_messages", true,
		agent.Usage{PromptTokens: 100, CacheCreationInputTokens: 40000}, "end_turn", time.Millisecond, false)
	s.logInferenceTurn("trace-A", "anthropic_messages", true,
		agent.Usage{PromptTokens: 50, CacheReadInputTokens: 40000}, "end_turn", time.Millisecond, false)
	s.logInferenceTurn("trace-B", "anthropic_messages", false,
		agent.Usage{PromptTokens: 80, CacheReadInputTokens: 8000}, "end_turn", time.Millisecond, false)

	turns, capped := s.metrics.vcacheTurnsSnapshot()
	if len(turns) != 3 {
		t.Fatalf("live window retained %d turns, want 3", len(turns))
	}
	block := vcacheFamiliesVars(turns, capped)
	if block == nil {
		t.Fatal("served turns with cache activity must populate the live per-family block")
	}
	if block.FamilyCount != 2 {
		t.Fatalf("expected 2 prefix families (trace-A, trace-B), got %d", block.FamilyCount)
	}
}

func TestVCacheTurnsSnapshotCarriesContextEvidence(t *testing.T) {
	m := newGatewayMetrics(time.Now())
	m.observeVCacheTurn("trace-A", 1, 100, 900, 0)
	m.observeCompaction(agent.CompactOutcome{
		Reason:     agent.CompactReasonNone,
		Dropped:    3,
		ShedTokens: 1200,
	}, false)

	turns, _ := m.vcacheTurnsSnapshot()
	if len(turns) != 1 {
		t.Fatalf("snapshot retained %d turns, want 1", len(turns))
	}
	got := turns[0]
	if got.ContextEvents != 1 || got.ContextShedTokens != 1200 || got.ContextDroppedTurns != 3 {
		t.Fatalf("context evidence = events:%d shed:%d dropped:%d, want 1/1200/3",
			got.ContextEvents, got.ContextShedTokens, got.ContextDroppedTurns)
	}
}

func TestVCacheTurnsSnapshotCarriesContextOnlyEvidence(t *testing.T) {
	m := newGatewayMetrics(time.Now())
	m.observeCompaction(agent.CompactOutcome{
		Reason:     agent.CompactReasonNone,
		Dropped:    4,
		ShedTokens: 1500,
	}, false)

	turns, capped := m.vcacheTurnsSnapshot()
	if capped {
		t.Fatal("context-only evidence must not mark the provider-cache window capped")
	}
	if len(turns) != 1 {
		t.Fatalf("snapshot retained %d turns, want one context-only witness row", len(turns))
	}
	got := turns[0]
	if got.Family != "context" {
		t.Fatalf("context-only family = %q, want context", got.Family)
	}
	if got.InputTokens != 0 || got.CacheRead != 0 || got.CacheCreation != 0 {
		t.Fatalf("context-only row must not invent provider telemetry: %+v", got)
	}
	if got.ContextEvents != 1 || got.ContextShedTokens != 1500 || got.ContextDroppedTurns != 4 {
		t.Fatalf("context evidence = events:%d shed:%d dropped:%d, want 1/1500/4",
			got.ContextEvents, got.ContextShedTokens, got.ContextDroppedTurns)
	}
}

func TestServerVCacheTurnsSnapshotCarriesContextEconomics(t *testing.T) {
	m := newGatewayMetrics(time.Now())
	m.observeVCacheTurn("trace-A", 1, 100, 900, 0)
	m.observeCompaction(agent.CompactOutcome{
		Reason:     agent.CompactReasonNone,
		Dropped:    3,
		ShedTokens: 800,
	}, false)
	s := &Server{metrics: m, compactHistoryBudget: 1200}

	turns, _ := s.VCacheTurnsSnapshot()
	if len(turns) != 1 {
		t.Fatalf("snapshot retained %d turns, want 1", len(turns))
	}
	got := turns[0]
	if got.ContextEvents != 1 || got.ContextShedTokens != 800 || got.ContextDroppedTurns != 3 {
		t.Fatalf("context evidence = events:%d shed:%d dropped:%d, want 1/800/3",
			got.ContextEvents, got.ContextShedTokens, got.ContextDroppedTurns)
	}
	if got.ContextBaselineTokens != 2000 || got.ContextCostTokens != 1200 {
		t.Fatalf("context economics = baseline:%d cost:%d, want 2000/1200",
			got.ContextBaselineTokens, got.ContextCostTokens)
	}
}

func TestServerVCacheTurnsSnapshotCarriesContextOnlyEconomics(t *testing.T) {
	m := newGatewayMetrics(time.Now())
	m.observeCompaction(agent.CompactOutcome{
		Reason:     agent.CompactReasonNone,
		Dropped:    2,
		ShedTokens: 900,
	}, false)
	s := &Server{metrics: m, compactHistoryBudget: 1100}

	turns, _ := s.VCacheTurnsSnapshot()
	if len(turns) != 1 {
		t.Fatalf("snapshot retained %d turns, want one context-only witness row", len(turns))
	}
	got := turns[0]
	if got.InputTokens != 0 || got.CacheRead != 0 || got.CacheCreation != 0 {
		t.Fatalf("context-only server row must not invent provider telemetry: %+v", got)
	}
	if got.ContextEvents != 1 || got.ContextShedTokens != 900 || got.ContextDroppedTurns != 2 {
		t.Fatalf("context evidence = events:%d shed:%d dropped:%d, want 1/900/2",
			got.ContextEvents, got.ContextShedTokens, got.ContextDroppedTurns)
	}
	if got.ContextBaselineTokens != 2000 || got.ContextCostTokens != 1100 {
		t.Fatalf("context economics = baseline:%d cost:%d, want 2000/1100",
			got.ContextBaselineTokens, got.ContextCostTokens)
	}
}

// TestVCacheFamiliesReconcilesWithObserve proves the live per-family block the gateway
// exposes over its rolling window is byte-identical to what `fak vcache observe` computes
// offline on the same traffic — the #935 acceptance ("reconciling with `fak vcache
// observe` on the same traffic"). The live block is built by feeding the retained turns to
// the SAME vcacheobserve.Observe engine, so this also guards against a field being
// dropped or mis-mapped on the way to the surface.
func TestVCacheFamiliesReconcilesWithObserve(t *testing.T) {
	m := newGatewayMetrics(time.Now())
	// Two prefix families. s1 warms then reads back (a proven, concentrated family);
	// s2 is a sparse second session. Distinct timestamps so the warmth-belief sequencing
	// and arrival rate are real.
	turns := []vcacheobserve.Turn{
		{Family: "s1", UnixMillis: 0, InputTokens: 100, CacheCreation: 40000},
		{Family: "s1", UnixMillis: 1000, InputTokens: 50, CacheRead: 40000, CacheCreation: 500},
		{Family: "s1", UnixMillis: 2000, InputTokens: 50, CacheRead: 40000, CacheCreation: 500},
		{Family: "s2", UnixMillis: 1500, InputTokens: 80, CacheCreation: 8000},
		{Family: "s2", UnixMillis: 3000, InputTokens: 40, CacheRead: 8000},
	}
	for _, tn := range turns {
		m.observeVCacheTurn(tn.Family, tn.UnixMillis, int(tn.InputTokens), int(tn.CacheRead), int(tn.CacheCreation))
	}

	snap, capped := m.vcacheTurnsSnapshot()
	if len(snap) != len(turns) {
		t.Fatalf("snapshot retained %d turns, want %d", len(snap), len(turns))
	}
	if capped {
		t.Fatal("window must not be capped under the cap")
	}
	block := vcacheFamiliesVars(snap, capped)
	if block == nil {
		t.Fatal("active per-family block must be populated")
	}

	want := vcacheobserve.Observe(turns, vcacheobserve.DefaultMultipliers())

	if block.TurnsObserved != want.Turns {
		t.Fatalf("turns_observed: got %d want %d", block.TurnsObserved, want.Turns)
	}
	if block.FamilyCount != want.FamilyCount || block.FamilyCount != 2 {
		t.Fatalf("family_count: got %d want %d", block.FamilyCount, want.FamilyCount)
	}
	if !approxEq(block.SavedTokenEquiv, want.Aggregate.SavedTokenEquiv) {
		t.Fatalf("window saved: got %.3f want %.3f", block.SavedTokenEquiv, want.Aggregate.SavedTokenEquiv)
	}
	if block.Status != string(want.Aggregate.Status) {
		t.Fatalf("window status: got %q want %q", block.Status, want.Aggregate.Status)
	}
	if block.GradeMeasured != want.GradeMeasured || block.GradeSynthetic != want.GradeSynthetic {
		t.Fatalf("grades: got measured=%q synthetic=%q want measured=%q synthetic=%q",
			block.GradeMeasured, block.GradeSynthetic, want.GradeMeasured, want.GradeSynthetic)
	}
	if block.Concentration.ZipfS != want.Concentration.ZipfS ||
		block.Concentration.Measured != want.Concentration.Measured ||
		block.Concentration.Defeated != want.Concentration.Defeated {
		t.Fatalf("concentration: got %+v want zipfS=%.3f measured=%v defeated=%v",
			block.Concentration, want.Concentration.ZipfS, want.Concentration.Measured, want.Concentration.Defeated)
	}

	// Every family row must equal the offline family it came from — keyed, so order is
	// not assumed.
	wantFam := map[string]vcacheobserve.Family{}
	for _, f := range want.Families {
		wantFam[f.Key] = f
	}
	if len(block.Families) != len(want.Families) {
		t.Fatalf("family rows: got %d want %d", len(block.Families), len(want.Families))
	}
	for _, got := range block.Families {
		wf, ok := wantFam[got.Key]
		if !ok {
			t.Fatalf("live block has family %q absent from offline observe", got.Key)
		}
		if got.Turns != wf.Turns {
			t.Fatalf("family %q turns: got %d want %d", got.Key, got.Turns, wf.Turns)
		}
		if !approxEq(got.HitRate, wf.HitRate) {
			t.Fatalf("family %q hit_rate: got %.4f want %.4f", got.Key, got.HitRate, wf.HitRate)
		}
		if !approxEq(got.SavedTokenEquiv, wf.Economics.SavedTokenEquiv) {
			t.Fatalf("family %q saved: got %.3f want %.3f", got.Key, got.SavedTokenEquiv, wf.Economics.SavedTokenEquiv)
		}
		if got.Status != string(wf.Economics.Status) {
			t.Fatalf("family %q status: got %q want %q", got.Key, got.Status, wf.Economics.Status)
		}
		if got.GovernorDecision != string(wf.GovernorDecision) {
			t.Fatalf("family %q governor: got %q want %q", got.Key, got.GovernorDecision, wf.GovernorDecision)
		}
		if !approxEq(got.ArrivalRatePerSec, wf.ArrivalRatePerSec) {
			t.Fatalf("family %q arrival: got %.4f want %.4f", got.Key, got.ArrivalRatePerSec, wf.ArrivalRatePerSec)
		}
		if got.WarmthTrueWarm != wf.Prediction.TrueWarm || got.WarmthFalseWarm != wf.Prediction.FalseWarm ||
			got.WarmthTrueCold != wf.Prediction.TrueCold || got.WarmthFalseCold != wf.Prediction.FalseCold {
			t.Fatalf("family %q warmth: got tw=%d fw=%d tc=%d fc=%d want tw=%d fw=%d tc=%d fc=%d",
				got.Key, got.WarmthTrueWarm, got.WarmthFalseWarm, got.WarmthTrueCold, got.WarmthFalseCold,
				wf.Prediction.TrueWarm, wf.Prediction.FalseWarm, wf.Prediction.TrueCold, wf.Prediction.FalseCold)
		}
	}

	// The window aggregate must also reconcile with the cumulative-counter engine on the
	// same totals — the same number the headline `vcache` block and fak_vcache_* family
	// report — so the per-family and aggregate surfaces never disagree.
	var sumIn, sumRead, sumCreate uint64
	for _, tn := range turns {
		sumIn += uint64(tn.InputTokens)
		sumRead += uint64(tn.CacheRead)
		sumCreate += uint64(tn.CacheCreation)
	}
	cumulative := vcacheProofFromCounters(sumIn, sumRead, sumCreate)
	if !approxEq(block.SavedTokenEquiv, cumulative.SavedTokenEquiv) {
		t.Fatalf("window saved %.3f != cumulative-counter saved %.3f", block.SavedTokenEquiv, cumulative.SavedTokenEquiv)
	}
}

// TestVCacheFamiliesNoPhantomAndProvenance proves the no-phantom zero guard (idle, and a
// no-cache workload, emit no block) and that an active block carries explicit OBSERVED /
// DECISION provenance labels on every value class plus the governor + warmth view.
func TestVCacheFamiliesNoPhantomAndProvenance(t *testing.T) {
	idle := newGatewayMetrics(time.Now())
	turns, capped := idle.vcacheTurnsSnapshot()
	if vcacheFamiliesVars(turns, capped) != nil {
		t.Fatal("idle gateway must emit no per-family block")
	}

	// A served turn with NO provider cache activity must still produce no block — the
	// block tracks the cache, not raw traffic (no phantom).
	noCache := newGatewayMetrics(time.Now())
	noCache.observeVCacheTurn("s1", 0, 900, 0, 0)
	turns, capped = noCache.vcacheTurnsSnapshot()
	if vcacheFamiliesVars(turns, capped) != nil {
		t.Fatal("a no-cache turn must not mint a per-family block")
	}

	// Cache activity → block present with provenance + governor + warmth.
	m := newGatewayMetrics(time.Now())
	m.observeVCacheTurn("s1", 0, 100, 0, 40000)
	m.observeVCacheTurn("s1", 1000, 50, 40000, 0)
	turns, capped = m.vcacheTurnsSnapshot()
	block := vcacheFamiliesVars(turns, capped)
	if block == nil {
		t.Fatal("active gateway must populate the per-family block")
	}
	if block.Provenance["hit_rate"] != "OBSERVED" || block.Provenance["governor_decision"] != "DECISION" {
		t.Fatalf("provenance labels missing/wrong: %+v", block.Provenance)
	}
	if len(block.Families) != 1 || block.Families[0].GovernorDecision == "" {
		t.Fatalf("expected one family with a governor decision, got %+v", block.Families)
	}

	// The block must serialize with both provenance labels so an operator scraping
	// /debug/vars sees who owns each value.
	raw, err := json.Marshal(block)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	for _, want := range []string{"OBSERVED", "DECISION", "governor_decision", "warmth_false_warm", "concentration", "grade_measured"} {
		if !strings.Contains(string(raw), want) {
			t.Fatalf("serialized block missing %q:\n%s", want, raw)
		}
	}
}

// TestVCacheWarmthBeliefMetrics proves the M1 warmth-belief estimator is visible on
// live gateway metrics without phantom idle/no-cache series. The fixture deliberately
// creates both error classes: first read is false-cold (belief started cold but the
// provider read cache), then a within-TTL miss is false-warm (the Law A1 signal).
func TestVCacheWarmthBeliefMetrics(t *testing.T) {
	idle := newGatewayMetrics(time.Now())
	var idleOut strings.Builder
	idle.writeVCacheWarmthMetrics(&idleOut)
	if idleOut.Len() != 0 {
		t.Fatalf("idle gateway must not emit warmth phantom metrics:\n%s", idleOut.String())
	}

	noCache := newGatewayMetrics(time.Now())
	noCache.observeVCacheTurn("s1", 0, 900, 0, 0)
	var noCacheOut strings.Builder
	noCache.writeVCacheWarmthMetrics(&noCacheOut)
	if noCacheOut.Len() != 0 {
		t.Fatalf("no-cache workload must not emit warmth metrics:\n%s", noCacheOut.String())
	}

	srv := newTestServer(t)
	srv.metrics.observeVCacheTurn("warm", 0, 100, 40000, 0)    // predicted cold, actual warm => false_cold
	srv.metrics.observeVCacheTurn("warm", 1000, 100, 0, 0)     // predicted warm, actual cold => false_warm
	srv.metrics.observeVCacheTurn("warm", 2000, 100, 40000, 0) // predicted cold after demote, actual warm => false_cold

	text := srv.renderMetrics()
	for _, want := range []string{
		"# TYPE fak_vcache_warmth_prediction_outcomes gauge",
		`fak_vcache_warmth_prediction_outcomes{class="true_warm"} 0`,
		`fak_vcache_warmth_prediction_outcomes{class="false_warm"} 1`,
		`fak_vcache_warmth_prediction_outcomes{class="true_cold"} 0`,
		`fak_vcache_warmth_prediction_outcomes{class="false_cold"} 2`,
		`fak_vcache_warmth_predictions_total 3`,
		`fak_vcache_warmth_false_warm_rate 1`,
		`fak_vcache_warmth_false_cold_rate 1`,
		`fak_vcache_warmth_demotions_total 1`,
	} {
		if !strings.Contains(text, want) {
			t.Fatalf("warmth scrape missing %q\n--- metrics ---\n%s", want, text)
		}
	}

	vars := srv.debugVars(time.Now())
	records := vars.VCacheWarmth
	if len(records) != 1 {
		t.Fatalf("warmth demotion journal recorded %d rows, want 1: %+v", len(records), records)
	}
	r := records[0]
	if r.Schema != vcacheWarmthDemotionJournalSchema {
		t.Fatalf("schema = %q, want %q", r.Schema, vcacheWarmthDemotionJournalSchema)
	}
	if r.Family != "warm" || r.Action != "mark_belief_cold" || r.Reason != "false_warm" {
		t.Fatalf("unexpected demotion record identity: %+v", r)
	}
	if !r.PredictedWarm || r.ActualWarm || r.StateBefore != "resident" || r.StateAfter != "expired" {
		t.Fatalf("unexpected demotion state: %+v", r)
	}
	if r.InputTokens != 100 || r.CacheReadTokens != 0 || r.CacheCreationTokens != 0 {
		t.Fatalf("unexpected demotion token counters: %+v", r)
	}
	if r.DivergenceProbe != "not_wired" {
		t.Fatalf("divergence probe = %q, want explicit not_wired scope", r.DivergenceProbe)
	}
	if r.PrevHash != "" || r.Hash == "" {
		t.Fatalf("bad demotion hash chain genesis: %+v", r)
	}
	if got := hashVCacheWarmthDemotion(r.PrevHash, r); got != r.Hash {
		t.Fatalf("demotion hash mismatch: got %q want %q", got, r.Hash)
	}
}

// TestVCacheGovernorDecisionMetrics proves the M5 Governor verdict is default-visible
// on live gateway traffic without minting per-family Prometheus labels or phantom idle
// series. The decisions are re-derived from the same rolling turns as /debug/vars, so a
// scrape gives a low-cardinality witness that the Governor classified the active window.
func TestVCacheGovernorDecisionMetrics(t *testing.T) {
	idle := newGatewayMetrics(time.Now())
	var idleOut strings.Builder
	idle.writeVCacheGovernorMetrics(&idleOut)
	if idleOut.Len() != 0 {
		t.Fatalf("idle gateway must not emit governor phantom metrics:\n%s", idleOut.String())
	}

	noCache := newGatewayMetrics(time.Now())
	noCache.observeVCacheTurn("s1", 0, 900, 0, 0)
	var noCacheOut strings.Builder
	noCache.writeVCacheGovernorMetrics(&noCacheOut)
	if noCacheOut.Len() != 0 {
		t.Fatalf("no-cache workload must not emit governor metrics:\n%s", noCacheOut.String())
	}

	srv := newTestServer(t)
	srv.metrics.observeVCacheTurn("hot", 0, 100, 0, 40000)
	srv.metrics.observeVCacheTurn("hot", 1000, 50, 40000, 0)
	srv.metrics.observeVCacheTurn("sparse", 0, 100, 0, 8000)
	srv.metrics.observeVCacheTurn("sparse", 10*vcachegov.TTL5MinutesMillis, 50, 8000, 0)

	text := srv.renderMetrics()
	for _, want := range []string{
		"# TYPE fak_vcache_governor_decision_families gauge",
		`fak_vcache_governor_decision_families{decision="ride_natural"} 1`,
		`fak_vcache_governor_decision_families{decision="lazy_rebuild"} 1`,
		`fak_vcache_governor_decision_families{decision="heartbeat_pin"} 0`,
		`fak_vcache_governor_decision_families{decision="evict"} 0`,
		`fak_vcache_governor_decision_families{decision="no_cache"} 0`,
		`fak_vcache_governor_decision_families{decision="explicit_cache"} 0`,
	} {
		if !strings.Contains(text, want) {
			t.Fatalf("governor decision scrape missing %q\n--- metrics ---\n%s", want, text)
		}
	}

	vars := srv.debugVars(time.Now())
	records := vars.VCacheGovernor
	if len(records) < 4 {
		t.Fatalf("governor journal recorded %d rows, want at least 4: %+v", len(records), records)
	}
	seen := map[string]bool{}
	for i, r := range records {
		if r.Schema != vcacheGovernorDecisionJournalSchema {
			t.Fatalf("record %d schema = %q, want %q", i, r.Schema, vcacheGovernorDecisionJournalSchema)
		}
		if r.Hash == "" {
			t.Fatalf("record %d has empty hash: %+v", i, r)
		}
		if got := hashVCacheGovernorDecision(r.PrevHash, r); got != r.Hash {
			t.Fatalf("record %d hash mismatch: got recomputed %q want %q", i, got, r.Hash)
		}
		if i == 0 {
			if r.PrevHash != "" {
				t.Fatalf("first journal row prev_hash = %q, want genesis", r.PrevHash)
			}
		} else if r.PrevHash != records[i-1].Hash {
			t.Fatalf("record %d prev_hash = %q, want previous hash %q", i, r.PrevHash, records[i-1].Hash)
		}
		seen[r.Family+"/"+r.Decision] = true
	}
	if !seen["hot/ride_natural"] || !seen["sparse/lazy_rebuild"] {
		t.Fatalf("governor journal missing expected family decisions, seen=%v records=%+v", seen, records)
	}
}

// TestVCacheTurnWindowBounded proves the live window stays flat under a long-running
// gateway: past the cap it keeps the most-recent vcacheTurnCap turns and flags the trim.
func TestVCacheTurnWindowBounded(t *testing.T) {
	m := newGatewayMetrics(time.Now())
	total := vcacheTurnCap + 250
	for i := 0; i < total; i++ {
		// Tag the family with the index so the most-recent-kept turns are identifiable.
		m.observeVCacheTurn("s", int64(i), 10, 5, 0)
	}
	snap, capped := m.vcacheTurnsSnapshot()
	if len(snap) != vcacheTurnCap {
		t.Fatalf("window length: got %d want cap %d", len(snap), vcacheTurnCap)
	}
	if !capped {
		t.Fatal("window_capped must be true once drop-oldest has trimmed the window")
	}
	// The oldest survivor is turn index (total-cap); the head was dropped.
	if snap[0].UnixMillis != int64(total-vcacheTurnCap) {
		t.Fatalf("oldest retained turn millis: got %d want %d", snap[0].UnixMillis, total-vcacheTurnCap)
	}
}
