package metrics

// Proof-witness tests for OPEN math-proof obligations touching internal/metrics.
//
// OPEN [ab-kpi-fold-correct] — mechanism cited at fak/internal/bench/bench.go:237
// (plus metrics.go:175 Validate, metrics.go:162 ComputeGate). The FIVE KPI field
// definitions the theorem asserts (vdso_hit_rate = VDSOHits/Calls,
// context_pollution_rate = Quarantines/Calls, tokens_per_task =
// (InTokens+OutTokens)/Calls, tool_call p50/p99 = the On arm's percentiles,
// token_delta_pct = 100*(offTok-onTok)/offTok) are NOT computed anywhere in
// package metrics. metrics only DECLARES the Arm/KPIs/Report structs; the fold
// itself lives inline in bench.Run (package bench), which imports metrics.
//
// VERDICT for [ab-kpi-fold-correct] from this file's vantage: SKIP. A sound,
// non-vacuous witness of that fold must call the actual fold (bench.Run). package
// bench imports package metrics, so a metrics-package test CANNOT import bench
// (import cycle), and re-implementing the formula here to compare it against
// itself would be vacuous. The honest home for that witness is
// internal/bench/proofs_witness_test.go (a different package's new file, owned by
// the bench-side agent). This file therefore does NOT claim to close the fold.
//
// What this file DOES witness soundly, in-package and non-vacuously, is the
// metrics-side contract the fold's output depends on: the KPIs/Arm/Report structs
// carry exactly the JSON keys the theorem's definitions name, and those fields
// survive a real marshal -> unmarshal round-trip byte/value-identically. If a
// rename or tag-drift broke a key the theorem references, these tests fail.

import (
	"encoding/json"
	"testing"
)

// TestKPIJSONKeysMatchTheoremDefinitions asserts that the serialized Report
// exposes EXACTLY the JSON keys the [ab-kpi-fold-correct] definitions name, on
// the structs that carry the fold's inputs (Arm) and outputs (KPIs / Report).
// This is a real guard: it marshals a populated Report and checks the literal
// key set, so a tag rename (e.g. vdso_hit_rate -> hit_rate) that would silently
// invalidate the theorem's field-definition mapping fails here. It does NOT
// assert the fold arithmetic (that lives in bench.Run) — see SKIP note above.
func TestKPIJSONKeysMatchTheoremDefinitions(t *testing.T) {
	r := &Report{}
	// Populate the fold inputs (Arm counters) and outputs (KPIs / token delta) so
	// every key is present and non-zero in the serialized form.
	r.On = Arm{
		Label: "vdso_on", Calls: 4,
		P50Ns: 111, P99Ns: 222,
		VDSOHits: 3, Quarantines: 1,
		InTokens: 800, OutTokens: 200,
	}
	r.Off = Arm{
		Label: "vdso_off", Calls: 4,
		InTokens: 1000, OutTokens: 300,
	}
	r.KPIs = KPIs{
		ToolCallP50Ns:        r.On.P50Ns,
		ToolCallP99Ns:        r.On.P99Ns,
		VDSOHitRate:          0.75,
		ContextPollutionRate: 0.25,
		TokensPerTask:        250,
	}
	r.TokenDeltaPct = -7.5

	b := r.JSON()
	var generic map[string]any
	if err := json.Unmarshal(b, &generic); err != nil {
		t.Fatalf("JSON did not unmarshal: %v", err)
	}

	// The fold OUTPUT keys named by the theorem live under "kpis" and on the
	// top-level report (token_delta_pct). Assert each by its theorem name.
	kpis, ok := generic["kpis"].(map[string]any)
	if !ok {
		t.Fatalf("report missing \"kpis\" object")
	}
	for _, k := range []string{
		"tool_call_p50_ns",       // tool_call p50 = On arm
		"tool_call_p99_ns",       // tool_call p99 = On arm
		"vdso_hit_rate",          // VDSOHits/Calls
		"context_pollution_rate", // Quarantines/Calls
		"tokens_per_task",        // (InTokens+OutTokens)/Calls
	} {
		if _, present := kpis[k]; !present {
			t.Errorf("kpis JSON missing theorem field %q", k)
		}
	}
	if _, present := generic["token_delta_pct"]; !present {
		t.Errorf("report JSON missing theorem field \"token_delta_pct\"")
	}

	// The fold INPUT keys named by the theorem live on each arm.
	for _, armKey := range []string{"vdso_on", "vdso_off"} {
		arm, ok := generic[armKey].(map[string]any)
		if !ok {
			t.Fatalf("report missing arm object %q", armKey)
		}
		for _, k := range []string{
			"calls",         // denominator of every rate
			"vdso_hits",     // vdso_hit_rate numerator
			"quarantines",   // context_pollution_rate numerator
			"input_tokens",  // tokens_per_task / token_delta numerator part
			"output_tokens", // tokens_per_task / token_delta numerator part
			"p50_ns",        // tool_call p50 source (On arm)
			"p99_ns",        // tool_call p99 source (On arm)
		} {
			if _, present := arm[k]; !present {
				t.Errorf("arm %q JSON missing theorem input field %q", armKey, k)
			}
		}
	}
}

// TestReportFoldFieldsRoundTrip asserts that the exact field VALUES the fold
// reads (Arm counters) and writes (KPIs + token_delta_pct) survive a real
// marshal -> unmarshal round-trip identically. This is the metrics-side
// invariant the [ab-kpi-fold-correct] mapping relies on: the report a consumer
// ingests carries the same numbers the fold placed. Non-vacuous: it compares a
// decoded value to the encoded one, exact for the integer counters and within a
// tight tolerance for the float KPIs. It does NOT compute the fold.
func TestReportFoldFieldsRoundTrip(t *testing.T) {
	const eps = 1e-9
	r := &Report{}
	r.On = Arm{
		Label: "vdso_on", Calls: 5,
		P50Ns: 1234, P99Ns: 5678,
		VDSOHits: 4, Quarantines: 2,
		InTokens: 1500, OutTokens: 500,
	}
	r.Off = Arm{
		Label: "vdso_off", Calls: 5,
		InTokens: 2000, OutTokens: 800,
	}
	r.KPIs = KPIs{
		ToolCallP50Ns:        r.On.P50Ns,
		ToolCallP99Ns:        r.On.P99Ns,
		VDSOHitRate:          float64(r.On.VDSOHits) / float64(r.On.Calls),
		ContextPollutionRate: float64(r.On.Quarantines) / float64(r.On.Calls),
		TokensPerTask:        float64(r.On.InTokens+r.On.OutTokens) / float64(r.On.Calls),
	}
	offTok := r.Off.InTokens + r.Off.OutTokens
	onTok := r.On.InTokens + r.On.OutTokens
	r.TokenDeltaPct = 100 * float64(offTok-onTok) / float64(offTok)

	var round Report
	if err := json.Unmarshal(r.JSON(), &round); err != nil {
		t.Fatalf("round-trip unmarshal failed: %v", err)
	}

	// Integer fold inputs: exact identity through the round trip.
	if round.On.Calls != r.On.Calls ||
		round.On.VDSOHits != r.On.VDSOHits ||
		round.On.Quarantines != r.On.Quarantines ||
		round.On.InTokens != r.On.InTokens ||
		round.On.OutTokens != r.On.OutTokens ||
		round.On.P50Ns != r.On.P50Ns ||
		round.On.P99Ns != r.On.P99Ns {
		t.Errorf("On-arm fold inputs did not round-trip identically: got %+v want %+v", round.On, r.On)
	}
	if round.Off.Calls != r.Off.Calls ||
		round.Off.InTokens != r.Off.InTokens ||
		round.Off.OutTokens != r.Off.OutTokens {
		t.Errorf("Off-arm fold inputs did not round-trip identically: got %+v want %+v", round.Off, r.Off)
	}

	// tool_call p50/p99 are sourced from the On arm: the KPI value must equal the
	// On arm's percentile, and must survive the round trip exactly.
	if round.KPIs.ToolCallP50Ns != round.On.P50Ns {
		t.Errorf("tool_call_p50_ns = %d, want On arm p50 %d", round.KPIs.ToolCallP50Ns, round.On.P50Ns)
	}
	if round.KPIs.ToolCallP99Ns != round.On.P99Ns {
		t.Errorf("tool_call_p99_ns = %d, want On arm p99 %d", round.KPIs.ToolCallP99Ns, round.On.P99Ns)
	}

	// Float fold outputs: identity within tolerance.
	checkFloat := func(name string, got, want float64) {
		d := got - want
		if d < 0 {
			d = -d
		}
		if d > eps {
			t.Errorf("%s = %v, want %v (|d|=%v > %v)", name, got, want, d, eps)
		}
	}
	checkFloat("vdso_hit_rate", round.KPIs.VDSOHitRate, r.KPIs.VDSOHitRate)
	checkFloat("context_pollution_rate", round.KPIs.ContextPollutionRate, r.KPIs.ContextPollutionRate)
	checkFloat("tokens_per_task", round.KPIs.TokensPerTask, r.KPIs.TokensPerTask)
	checkFloat("token_delta_pct", round.TokenDeltaPct, r.TokenDeltaPct)
}
