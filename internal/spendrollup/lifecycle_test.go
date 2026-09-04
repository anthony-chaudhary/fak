package spendrollup

import (
	"encoding/json"
	"fmt"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/fleetaccounts"
)

// TestSpendRollupLifecycle_StateTransitions verifies the complete lifecycle transitions
// of a fleet account pool from cold zero-state, through active session scaling,
// provider telemetry reporting, and eventual worker draining.
func TestSpendRollupLifecycle_StateTransitions(t *testing.T) {
	// 1. Cold state: empty account roster
	emptyRoster := []fleetaccounts.Account{}
	r0 := Build(emptyRoster)
	if r0.Schema != Schema {
		t.Fatalf("lifecycle 0: schema = %q, want %q", r0.Schema, Schema)
	}
	if len(r0.Figures) != 0 || len(r0.Subtotals) != 0 {
		t.Fatalf("lifecycle 0: expected 0 figures and 0 subtotals, got %d and %d", len(r0.Figures), len(r0.Subtotals))
	}
	if err := r0.Gate(); err != nil {
		t.Fatalf("lifecycle 0: empty rollup must pass gate, got: %v", err)
	}
	rendered0 := Render(r0)
	if !strings.Contains(rendered0, "(no routable worker accounts)") {
		t.Fatalf("lifecycle 0: render missing empty notice: %s", rendered0)
	}

	// 2. Provisioning state: active workers dispatching live sessions
	worker1 := fleetaccounts.Account{
		Account:      "worker-claude-primary",
		Product:      "claude",
		Kind:         fleetaccounts.KindWorker,
		LiveSessions: ip(4),
	}
	worker2 := fleetaccounts.Account{
		Account:      "worker-claude-secondary",
		Product:      "claude",
		Kind:         fleetaccounts.KindWorker,
		LiveSessions: ip(6),
	}
	r1 := Build([]fleetaccounts.Account{worker1, worker2})
	if len(r1.Figures) != 2 {
		t.Fatalf("lifecycle 1: expected 2 figures, got %d", len(r1.Figures))
	}
	if len(r1.Subtotals) != 1 {
		t.Fatalf("lifecycle 1: expected 1 subtotal, got %d", len(r1.Subtotals))
	}
	if r1.Subtotals[0].Amount != 10 {
		t.Fatalf("lifecycle 1: expected subtotal amount 10, got %v", r1.Subtotals[0].Amount)
	}
	// Warning must be present because no OBSERVED figures have been reported yet
	if len(r1.Warnings) == 0 || !strings.Contains(r1.Warnings[0], "no provider-relayed figure present") {
		t.Fatalf("lifecycle 1: expected warning about unobserved provider figures, got %v", r1.Warnings)
	}
	if err := r1.Gate(); err != nil {
		t.Fatalf("lifecycle 1: active rollup must pass Gate, got %v", err)
	}

	// 3. Telemetry state: provider reports rate-limiting / weekly window posture
	worker2Throttled := worker2
	worker2Throttled.Throttled = bp(true)
	worker2Throttled.Weekly = sp("resets 2026-09-07T00:00:00Z")

	r2 := Build([]fleetaccounts.Account{worker1, worker2Throttled})
	if len(r2.Figures) != 3 { // 2 WITNESSED + 1 OBSERVED
		t.Fatalf("lifecycle 2: expected 3 figures, got %d: %+v", len(r2.Figures), r2.Figures)
	}
	if len(r2.Subtotals) != 2 {
		t.Fatalf("lifecycle 2: expected 2 subtotals (witnessed + observed), got %d", len(r2.Subtotals))
	}
	if len(r2.Warnings) != 0 {
		t.Fatalf("lifecycle 2: warning should be cleared once observed figure is recorded, got %v", r2.Warnings)
	}
	if err := r2.Gate(); err != nil {
		t.Fatalf("lifecycle 2: gate must pass on mixed figures, got %v", err)
	}

	// 4. Drained state: workers decommissioned or marked excluded
	worker1Drained := worker1
	worker1Drained.Kind = fleetaccounts.KindExcluded
	worker2Drained := worker2Throttled
	worker2Drained.LiveSessions = ip(0)
	worker2Drained.Throttled = bp(false)
	worker2Drained.Weekly = sp("")

	r3 := Build([]fleetaccounts.Account{worker1Drained, worker2Drained})
	// worker1 is excluded (0 figures). worker2 has 0 live sessions and no provider posture (1 WITNESSED figure of amount 0).
	if len(r3.Figures) != 1 {
		t.Fatalf("lifecycle 3: expected 1 figure for drained worker, got %d", len(r3.Figures))
	}
	if r3.Figures[0].Amount != 0 {
		t.Fatalf("lifecycle 3: expected amount 0 for drained session, got %v", r3.Figures[0].Amount)
	}
	if err := r3.Gate(); err != nil {
		t.Fatalf("lifecycle 3: gate must pass, got %v", err)
	}
}

// TestSpendRollupLifecycle_JSONSerialization verifies that a Rollup preserves all
// invariants and passes Gate across JSON serialization and deserialization cycles.
func TestSpendRollupLifecycle_JSONSerialization(t *testing.T) {
	acc := fleetaccounts.Account{
		Account:      "worker-roundtrip",
		Product:      "claude",
		Kind:         fleetaccounts.KindWorker,
		LiveSessions: ip(8),
		Throttled:    bp(true),
	}
	original := Build([]fleetaccounts.Account{acc})

	data, err := json.Marshal(original)
	if err != nil {
		t.Fatalf("failed to marshal Rollup to JSON: %v", err)
	}

	var restored Rollup
	if err := json.Unmarshal(data, &restored); err != nil {
		t.Fatalf("failed to unmarshal JSON into Rollup: %v", err)
	}

	if restored.Schema != original.Schema {
		t.Errorf("schema mismatch: got %q, want %q", restored.Schema, original.Schema)
	}
	if len(restored.Figures) != len(original.Figures) {
		t.Fatalf("figures length mismatch: got %d, want %d", len(restored.Figures), len(original.Figures))
	}
	for i := range original.Figures {
		of := original.Figures[i]
		rf := restored.Figures[i]
		if of.Account != rf.Account || of.Provenance != rf.Provenance || of.Basis != rf.Basis || of.Amount != rf.Amount || of.Unit != rf.Unit {
			t.Errorf("figure[%d] mismatch:\nwant: %+v\ngot:  %+v", i, of, rf)
		}
	}
	if len(restored.Subtotals) != len(original.Subtotals) {
		t.Fatalf("subtotals length mismatch: got %d, want %d", len(restored.Subtotals), len(original.Subtotals))
	}
	if err := restored.Gate(); err != nil {
		t.Fatalf("restored Rollup must pass Gate, got: %v", err)
	}
}

// TestSpendRollupAggregation_PartitionIntegrity verifies that subtotals partition
// figures strictly by (Provenance, Unit, Basis) and never collapse distinct bases or units.
func TestSpendRollupAggregation_PartitionIntegrity(t *testing.T) {
	figures := []Figure{
		{
			Account:    "acc-1",
			Provider:   "anthropic",
			Label:      "full prompt tokens",
			Amount:     1000,
			Unit:       "tokens",
			Basis:      BasisFullInput,
			Provenance: Witnessed,
		},
		{
			Account:    "acc-2",
			Provider:   "anthropic",
			Label:      "cached prompt tokens",
			Amount:     5000,
			Unit:       "tokens",
			Basis:      BasisCacheReadMarginal,
			Provenance: Witnessed,
		},
		{
			Account:    "acc-3",
			Provider:   "anthropic",
			Label:      "additional full prompt tokens",
			Amount:     2000,
			Unit:       "tokens",
			Basis:      BasisFullInput,
			Provenance: Witnessed,
		},
		{
			Account:    "acc-1",
			Provider:   "anthropic",
			Label:      "provider billed tokens",
			Amount:     8000,
			Unit:       "tokens",
			Basis:      BasisProviderBilled,
			Provenance: Observed,
		},
		{
			Account:    "acc-2",
			Provider:   "anthropic",
			Label:      "provider billed dollars",
			Amount:     25.50,
			Unit:       "usd",
			Basis:      BasisProviderBilled,
			Provenance: Observed,
		},
	}

	subs := subtotals(figures)

	// Expect 4 distinct subtotals:
	// 1. Observed / tokens / BasisProviderBilled (amount: 8000, figures: 1)
	// 2. Observed / usd / BasisProviderBilled (amount: 25.50, figures: 1)
	// 3. Witnessed / tokens / BasisFullInput (amount: 3000, figures: 2)
	// 4. Witnessed / tokens / BasisCacheReadMarginal (amount: 5000, figures: 1)
	if len(subs) != 4 {
		t.Fatalf("expected 4 partitioned subtotals, got %d: %+v", len(subs), subs)
	}

	// Verify deterministic sorting: OBSERVED < WITNESSED; then Unit ascending
	if subs[0].Provenance != Observed || subs[0].Unit != "tokens" || subs[0].Basis != BasisProviderBilled {
		t.Errorf("subtotal[0] unexpected: %+v", subs[0])
	}
	if subs[0].Amount != 8000 || subs[0].Figures != 1 {
		t.Errorf("subtotal[0] amount/count unexpected: got amount=%v count=%d", subs[0].Amount, subs[0].Figures)
	}

	if subs[1].Provenance != Observed || subs[1].Unit != "usd" || subs[1].Basis != BasisProviderBilled {
		t.Errorf("subtotal[1] unexpected: %+v", subs[1])
	}
	if subs[1].Amount != 25.50 || subs[1].Figures != 1 {
		t.Errorf("subtotal[1] amount/count unexpected: got amount=%v count=%d", subs[1].Amount, subs[1].Figures)
	}

	// For Witnessed, unit is "tokens" for both, but Basis is different.
	// Both subs[2] and subs[3] must have Provenance=Witnessed and Unit="tokens".
	if subs[2].Provenance != Witnessed || subs[2].Unit != "tokens" {
		t.Errorf("subtotal[2] unexpected: %+v", subs[2])
	}
	if subs[3].Provenance != Witnessed || subs[3].Unit != "tokens" {
		t.Errorf("subtotal[3] unexpected: %+v", subs[3])
	}

	// Find the BasisFullInput subtotal
	var fullInputSub, cacheReadSub *Subtotal
	for i := range subs {
		if subs[i].Basis == BasisFullInput {
			fullInputSub = &subs[i]
		}
		if subs[i].Basis == BasisCacheReadMarginal {
			cacheReadSub = &subs[i]
		}
	}
	if fullInputSub == nil || fullInputSub.Amount != 3000 || fullInputSub.Figures != 2 {
		t.Errorf("fullInputSub mismatch: %+v, want amount 3000, figures 2", fullInputSub)
	}
	if cacheReadSub == nil || cacheReadSub.Amount != 5000 || cacheReadSub.Figures != 1 {
		t.Errorf("cacheReadSub mismatch: %+v, want amount 5000, figures 1", cacheReadSub)
	}
}

// TestSpendRollupBudgeting_CapacityEvaluation evaluates whether a fleet's aggregated
// live-session load respects a defined capacity budget and enforces fail-closed checks.
func TestSpendRollupBudgeting_CapacityEvaluation(t *testing.T) {
	evalBudget := func(r Rollup, maxSessions float64) (float64, bool, error) {
		// Guard: must pass Gate before evaluating budget
		if err := r.Gate(); err != nil {
			return 0, false, fmt.Errorf("gate refusal: %w", err)
		}
		var liveSessions float64
		for _, s := range r.Subtotals {
			if s.Provenance == Witnessed && s.Unit == "live-sessions" && s.Basis == BasisObservedNet {
				liveSessions += s.Amount
			}
		}
		return liveSessions, liveSessions <= maxSessions, nil
	}

	accounts := []fleetaccounts.Account{
		worker("worker-a", 3),
		worker("worker-b", 4),
	}
	rollup := Build(accounts)

	// Within budget (7 <= 10)
	live, ok, err := evalBudget(rollup, 10.0)
	if err != nil {
		t.Fatalf("unexpected budget error: %v", err)
	}
	if !ok || live != 7.0 {
		t.Fatalf("expected within budget (live=7, ok=true), got live=%v, ok=%v", live, ok)
	}

	// Exceeds budget (7 > 5)
	live, ok, err = evalBudget(rollup, 5.0)
	if err != nil {
		t.Fatalf("unexpected budget error: %v", err)
	}
	if ok || live != 7.0 {
		t.Fatalf("expected over budget (live=7, ok=false), got live=%v, ok=%v", live, ok)
	}

	// Poisoned rollup: figure with missing basis must fail-closed during budget evaluation
	poisoned := rollup
	poisoned.Figures = append(poisoned.Figures, Figure{
		Account:    "malformed",
		Provenance: Witnessed,
		Basis:      ValuationBasis(""), // Defect
	})
	_, _, err = evalBudget(poisoned, 10.0)
	if err == nil {
		t.Fatal("expected fail-closed error on poisoned rollup during budget evaluation")
	}
	if !strings.Contains(err.Error(), "gate refusal") {
		t.Fatalf("expected gate refusal, got: %v", err)
	}
}

// TestSpendRollupBudgeting_FailClosedGateDefects ensures Gate detects all variations
// of missing or invalid labels across accounts and positional indexes.
func TestSpendRollupBudgeting_FailClosedGateDefects(t *testing.T) {
	cases := []struct {
		name       string
		figures    []Figure
		wantErrors []string
	}{
		{
			name: "unlabeled provenance with known account",
			figures: []Figure{
				{Account: "acc-bad-prov", Label: "load", Basis: BasisObservedNet, Provenance: "UNKNOWN"},
			},
			wantErrors: []string{"acc-bad-prov", `spend figure has no WITNESSED/OBSERVED provenance label (got "UNKNOWN")`},
		},
		{
			name: "unlabeled basis with positional index",
			figures: []Figure{
				{Account: "", Label: "orphan", Provenance: Witnessed, Basis: "INVALID_BASIS"},
			},
			wantErrors: []string{"figure[0]", `spend figure names no valuation basis (got "INVALID_BASIS")`},
		},
		{
			name: "multiple defects across multiple figures",
			figures: []Figure{
				{Account: "first", Label: "f1", Provenance: "", Basis: BasisObservedNet},
				{Account: "second", Label: "f2", Provenance: Witnessed, Basis: ""},
			},
			wantErrors: []string{"2 unlabeled spend figure(s)", "first", "second"},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			r := Rollup{Figures: tc.figures}
			err := r.Gate()
			if err == nil {
				t.Fatalf("expected Gate to return error, got nil")
			}
			for _, exp := range tc.wantErrors {
				if !strings.Contains(err.Error(), exp) {
					t.Errorf("error %q missing expected text %q", err.Error(), exp)
				}
			}
		})
	}
}

// BenchmarkSpendRollupAggregate benchmarks the Build and aggregation pipeline
// across a fleet of accounts with varied states.
func BenchmarkSpendRollupAggregate(b *testing.B) {
	accounts := make([]fleetaccounts.Account, 50)
	for i := 0; i < 50; i++ {
		accName := fmt.Sprintf("worker-%02d", i)
		live := (i % 5) + 1
		accounts[i] = worker(accName, live)
		if i%3 == 0 {
			accounts[i].Throttled = bp(true)
		}
		if i%7 == 0 {
			accounts[i].Weekly = sp("resets Monday 00:00")
		}
		if i%10 == 0 {
			accounts[i].Kind = fleetaccounts.KindExcluded
		}
	}

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		r := Build(accounts)
		if len(r.Figures) == 0 {
			b.Fatal("unexpected empty figures in benchmark")
		}
	}
}

// BenchmarkSpendRollupGate benchmarks validation throughput of the fail-closed Gate.
func BenchmarkSpendRollupGate(b *testing.B) {
	accounts := make([]fleetaccounts.Account, 100)
	for i := 0; i < 100; i++ {
		accounts[i] = worker(fmt.Sprintf("bench-worker-%d", i), 2)
		if i%2 == 0 {
			accounts[i].Throttled = bp(true)
		}
	}
	r := Build(accounts)
	if err := r.Gate(); err != nil {
		b.Fatalf("setup rollup failed Gate: %v", err)
	}

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		if err := r.Gate(); err != nil {
			b.Fatalf("Gate failed in benchmark: %v", err)
		}
	}
}

// BenchmarkSpendRollupRender benchmarks the human-readable formatting throughput.
func BenchmarkSpendRollupRender(b *testing.B) {
	accounts := make([]fleetaccounts.Account, 20)
	for i := 0; i < 20; i++ {
		accounts[i] = worker(fmt.Sprintf("render-worker-%d", i), i+1)
		if i%2 == 0 {
			accounts[i].Throttled = bp(true)
		}
	}
	r := Build(accounts)

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		out := Render(r)
		if len(out) == 0 {
			b.Fatal("unexpected empty render output")
		}
	}
}
