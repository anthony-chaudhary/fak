package cacheobs

import (
	"fmt"
	"sync"
	"testing"
)

// labeled_test.go — the #3391 eligibility-filtered denominator and the (model, tenant)
// series breakdown: the fair hit-rate excludes the always-cold head, clamps keep it
// honest, legacy taps degrade (never inflate), and label rows always reconcile with the
// global counters.

// #3391 (a): the eligibility-filtered denominator lifts the always-cold first prefill
// out of the hit-rate. A cold 1000-token turn-1 plus a warm 1000-token turn-2 (800
// reused) reads 0.4 on the raw ratio — the cache blamed for tokens it could never have
// served — but 0.8 on the filtered ratio. The raw ratio itself must stay untouched.
func TestEligibleDenominatorExcludesColdHead(t *testing.T) {
	o := New()
	if s := o.Snapshot(); s.EligibleReuseRatio != 0 || s.EligibleTokens != 0 {
		t.Fatalf("idle observer reported phantom eligibility: %+v", s)
	}
	o.ObserveLabeled(Labels{Model: "m", Tenant: "t"}, 1000, 0, 0, 0)        // always-cold first prefill
	o.ObserveLabeled(Labels{Model: "m", Tenant: "t"}, 1000, 800, 800, 1000) // whole prompt in play
	s := o.Snapshot()
	if s.EligibleTokens != 1000 {
		t.Fatalf("eligible = %d, want 1000 (the cold head must not count)", s.EligibleTokens)
	}
	if s.ReuseRatio != 0.4 {
		t.Fatalf("raw reuse ratio = %v, want 0.4 (the split must not redefine it)", s.ReuseRatio)
	}
	if s.EligibleReuseRatio != 0.8 {
		t.Fatalf("eligible reuse ratio = %v, want 0.8 (reused/eligible, not reused/prompt)", s.EligibleReuseRatio)
	}
}

// #3391 (b): the eligibility witness is clamped into [cacheable, prompt] AFTER the
// #3390 clamps — a stale zero on a turn that demonstrably matched (a prewarmed tree
// serving the "first" prefill) is raised to the lookup match, an over-claim is capped
// at the prompt, and a negative is treated as no better than the lookup match — so
// reused/eligible can never exceed 1.
func TestEligibleClampedBetweenCacheableAndPrompt(t *testing.T) {
	o := New()
	o.ObserveLabeled(Labels{}, 1000, 300, 200, 0) // stale zero: raised to cacheable (300)
	o.ObserveLabeled(Labels{}, 100, 0, 0, 5000)   // over-claim: capped at prompt (100)
	o.ObserveLabeled(Labels{}, 50, 0, 0, -7)      // negative: raised to cacheable (0)
	s := o.Snapshot()
	if want := uint64(300 + 100 + 0); s.EligibleTokens != want {
		t.Fatalf("eligible = %d, want %d (clamped into [cacheable, prompt] per turn)", s.EligibleTokens, want)
	}
	if s.EligibleReuseRatio > 1 {
		t.Fatalf("eligible reuse ratio = %v, must never exceed 1", s.EligibleReuseRatio)
	}
	if s.EligibleTokens < s.CacheableTokens || s.EligibleTokens > s.PromptTokens {
		t.Fatalf("invariant broken: cacheable=%d <= eligible=%d <= prompt=%d",
			s.CacheableTokens, s.EligibleTokens, s.PromptTokens)
	}
}

// #3391 (c): taps with no eligibility witness (Observe / ObserveSplit /
// ObservePreempted) book their whole prompt as eligible, so the filtered ratio degrades
// to exactly the raw ratio — an over-counted denominator can only understate the fair
// hit-rate, never inflate it — and they land on the ("unknown","unknown") series.
func TestLegacyTapsBookWholePromptEligible(t *testing.T) {
	o := New()
	o.Observe(1000, 400)
	o.ObserveSplit(500, 300, 100)
	o.ObservePreempted(300, 0, 0, 100)
	s := o.Snapshot()
	if s.EligibleTokens != s.PromptTokens {
		t.Fatalf("legacy taps must book the whole prompt eligible: eligible=%d prompt=%d",
			s.EligibleTokens, s.PromptTokens)
	}
	if s.EligibleReuseRatio != s.ReuseRatio {
		t.Fatalf("with no witness the filtered ratio must equal the raw one: %v vs %v",
			s.EligibleReuseRatio, s.ReuseRatio)
	}
	rows := o.LabeledSnapshot()
	if len(rows) != 1 || rows[0].Labels != (Labels{Model: "unknown", Tenant: "unknown", Phase: PhaseOther}) {
		t.Fatalf("legacy rows = %+v, want one (unknown, unknown) series", rows)
	}
}

// #3391 (d): the never-desync invariant — summing any column across LabeledSnapshot
// rows reconciles exactly with the global counter, labeled and legacy taps mixed —
// and rows come back sorted by model then tenant with per-row bookkeeping intact.
func TestLabeledRowsReconcileWithGlobals(t *testing.T) {
	o := New()
	o.ObserveLabeled(Labels{Model: "qwen", Tenant: "acme"}, 1000, 0, 0, 0)
	o.ObserveLabeled(Labels{Model: "qwen", Tenant: "acme"}, 1000, 900, 900, 1000)
	o.ObserveLabeled(Labels{Model: "qwen", Tenant: "globex"}, 400, 100, 50, 400)
	o.Observe(200, 0) // unlabeled legacy turn → (unknown, unknown)
	rows := o.LabeledSnapshot()
	if len(rows) != 3 {
		t.Fatalf("rows = %+v, want 3 series", rows)
	}
	if rows[0].Labels != (Labels{Model: "qwen", Tenant: "acme", Phase: PhaseOther}) ||
		rows[1].Labels != (Labels{Model: "qwen", Tenant: "globex", Phase: PhaseOther}) ||
		rows[2].Labels != (Labels{Model: "unknown", Tenant: "unknown", Phase: PhaseOther}) {
		t.Fatalf("rows out of (model, tenant) order: %+v", rows)
	}
	if r := rows[0]; r.Turns != 2 || r.PromptTokens != 2000 || r.EligibleTokens != 1000 || r.ReusedTokens != 900 {
		t.Fatalf("acme row = %+v, want turns=2 prompt=2000 eligible=1000 reused=900", r)
	}
	var turns, prompt, eligible, reused uint64
	for _, r := range rows {
		turns += r.Turns
		prompt += r.PromptTokens
		eligible += r.EligibleTokens
		reused += r.ReusedTokens
	}
	s := o.Snapshot()
	if turns != s.Turns || prompt != s.PromptTokens || eligible != s.EligibleTokens || reused != s.ReusedTokens {
		t.Fatalf("label sums desync from globals: turns=%d/%d prompt=%d/%d eligible=%d/%d reused=%d/%d",
			turns, s.Turns, prompt, s.PromptTokens, eligible, s.EligibleTokens, reused, s.ReusedTokens)
	}
}

// #3391 (e): every unlabeled spelling — empty, whitespace, or a legacy tap — folds into
// the single canonical ("unknown","unknown") series, so a renderer can never emit two
// Prometheus series for the same identity.
func TestLabelNormalizationCannotMintDuplicateSeries(t *testing.T) {
	o := New()
	o.ObserveLabeled(Labels{Model: "  ", Tenant: ""}, 100, 0, 0, 100)
	o.Observe(100, 0)
	rows := o.LabeledSnapshot()
	if len(rows) != 1 {
		t.Fatalf("rows = %+v, want both unlabeled spellings folded into one series", rows)
	}
	if rows[0].Labels != (Labels{Model: "unknown", Tenant: "unknown", Phase: PhaseOther}) || rows[0].PromptTokens != 200 {
		t.Fatalf("folded row = %+v, want (unknown, unknown) with prompt=200", rows[0])
	}
}

// #3391 (f): nil-observer safety, matching Observe/Snapshot — a tap on a nil observer
// is a no-op, never a panic, and the labeled snapshot is nil.
func TestLabeledNilObserverSafe(t *testing.T) {
	var o *Observer
	o.ObserveLabeled(Labels{Model: "m"}, 10, 0, 0, 10) // must not panic
	if rows := o.LabeledSnapshot(); rows != nil {
		t.Fatalf("nil observer rows = %+v, want nil", rows)
	}
}

// TestLabeledAttributionConcurrentSessions is the race witness for #3638. Each
// tenant label represents one gateway session. Distinct, non-overlapping token
// counts make any shared-counter contamination visible, while the aggregate
// reconciliation proves no update disappeared under contention.
func TestLabeledAttributionConcurrentSessions(t *testing.T) {
	const sessions = 32
	const turnsPerSession = 200

	o := New()
	start := make(chan struct{})
	var wg sync.WaitGroup
	for session := 1; session <= sessions; session++ {
		session := session
		wg.Add(1)
		go func() {
			defer wg.Done()
			labels := Labels{Model: "synthetic-upstream", Tenant: fmt.Sprintf("session-%02d", session)}
			<-start
			for turn := 0; turn < turnsPerSession; turn++ {
				prompt := session*1000 + turn + 1
				reused := session
				o.ObserveLabeled(labels, prompt, prompt, reused, prompt)
			}
		}()
	}
	close(start)
	wg.Wait()

	rows := o.LabeledSnapshot()
	if len(rows) != sessions {
		t.Fatalf("labeled rows = %d, want %d", len(rows), sessions)
	}

	var wantPrompt, wantReused uint64
	for i, row := range rows {
		session := i + 1
		wantTenant := fmt.Sprintf("session-%02d", session)
		if row.Labels != (Labels{Model: "synthetic-upstream", Tenant: wantTenant, Phase: PhaseOther}) {
			t.Fatalf("row %d labels = %+v, want tenant %q", i, row.Labels, wantTenant)
		}
		wantSessionPrompt := uint64(turnsPerSession*(session*1000+1) + turnsPerSession*(turnsPerSession-1)/2)
		wantSessionReused := uint64(turnsPerSession * session)
		if row.Turns != turnsPerSession || row.PromptTokens != wantSessionPrompt || row.EligibleTokens != wantSessionPrompt || row.ReusedTokens != wantSessionReused {
			t.Fatalf("%s attribution = %+v, want turns=%d prompt=eligible=%d reused=%d", wantTenant, row, turnsPerSession, wantSessionPrompt, wantSessionReused)
		}
		wantPrompt += wantSessionPrompt
		wantReused += wantSessionReused
	}

	global := o.Snapshot()
	if global.Turns != sessions*turnsPerSession || global.PromptTokens != wantPrompt || global.EligibleTokens != wantPrompt || global.ReusedTokens != wantReused {
		t.Fatalf("global attribution = %+v, want turns=%d prompt=eligible=%d reused=%d", global, sessions*turnsPerSession, wantPrompt, wantReused)
	}
}

func TestPipelinePhasesReconcileWithGlobalTotals(t *testing.T) {
	o := New()
	o.ObserveLabeled(Labels{Model: "qwen", Tenant: "acme", Phase: PhasePrefill}, 1000, 800, 700, 900)
	o.ObserveLabeled(Labels{Model: "qwen", Tenant: "acme", Phase: PhaseDecode}, 200, 100, 50, 150)

	rows := o.LabeledSnapshot()
	if len(rows) != 2 {
		t.Fatalf("labeled rows = %d, want 2: %+v", len(rows), rows)
	}

	phases := make(map[string]bool, len(rows))
	var turns, prompt, eligible, reused uint64
	for _, row := range rows {
		phases[row.Labels.Phase] = true
		turns += row.Turns
		prompt += row.PromptTokens
		eligible += row.EligibleTokens
		reused += row.ReusedTokens
	}
	if !phases[PhasePrefill] || !phases[PhaseDecode] {
		t.Fatalf("phases = %v, want %q and %q", phases, PhasePrefill, PhaseDecode)
	}

	global := o.Snapshot()
	if turns != global.Turns || prompt != global.PromptTokens || eligible != global.EligibleTokens || reused != global.ReusedTokens {
		t.Fatalf("phase totals = turns:%d prompt:%d eligible:%d reused:%d, global = %+v", turns, prompt, eligible, reused, global)
	}
}

func TestUnknownPipelinePhaseMapsToOther(t *testing.T) {
	o := New()
	for _, phase := range []string{"", "   ", "prefill-v2", "DECODE", "tenant-controlled"} {
		o.ObserveLabeled(Labels{Model: "qwen", Tenant: "acme", Phase: phase}, 10, 5, 2, 8)
	}

	rows := o.LabeledSnapshot()
	if len(rows) != 1 {
		t.Fatalf("labeled rows = %d, want one bounded fallback row: %+v", len(rows), rows)
	}
	if rows[0].Labels.Phase != PhaseOther {
		t.Fatalf("phase = %q, want %q", rows[0].Labels.Phase, PhaseOther)
	}
	if rows[0].Turns != 5 {
		t.Fatalf("fallback turns = %d, want 5", rows[0].Turns)
	}
}

func TestPipelinePhaseCardinalityCappedUnderAdversarialInput(t *testing.T) {
	o := New()
	known := []string{PhasePrefill, PhaseDecode}
	for _, phase := range known {
		o.ObserveLabeled(Labels{Model: "qwen", Tenant: "acme", Phase: phase}, 1, 0, 0, 1)
	}
	for i := 0; i < 1000; i++ {
		phase := fmt.Sprintf("attacker-phase-%d", i)
		o.ObserveLabeled(Labels{Model: "qwen", Tenant: "acme", Phase: phase}, 1, 0, 0, 1)
	}

	rows := o.LabeledSnapshot()
	if len(rows) != 3 {
		t.Fatalf("labeled rows = %d, want closed cardinality 3: %+v", len(rows), rows)
	}
	seen := map[string]uint64{}
	for _, row := range rows {
		seen[row.Labels.Phase] = row.Turns
	}
	for _, phase := range []string{PhasePrefill, PhaseDecode, PhaseOther} {
		if _, ok := seen[phase]; !ok {
			t.Fatalf("phase set = %v, missing %q", seen, phase)
		}
	}
	if seen[PhaseOther] != 1000 {
		t.Fatalf("other turns = %d, want 1000", seen[PhaseOther])
	}
}
