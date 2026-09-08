package modelroute

import (
	"encoding/json"
	"math"
	"testing"
	"time"
)

/*
Astra Formal Packet Derivation: fak-formal-packet/1
Kind: numerical_correctness_derivation
Target: internal/modelroute/** (Issue #12112)

1. INPUT DOMAIN:
   - Outcome.Cost: float64 in [0, +inf)
   - Outcome.Quality: float64 in [0.0, 1.0]
   - Objective.{QualityWeight, CostWeight, LatencyWeight}: float64 in [0, +inf)
   - Objective.MaxMeanCost: float64 in [0, +inf)
   - Objective.MinMeanQuality: float64 in [0.0, 1.0]
   - Member.Weight: float64 in (0, +inf)
   - AllReduce outputs: string-encoded float64 in (-inf, +inf)

2. DATAFLOW:
   - Telemetry -> OutcomeJournal.Append -> OutcomeJournal.Aggregate -> AspectRuleStats (MeanCost, MeanQuality)
   - OutcomeRecords -> EvaluateManifest / evaluateRuleRecords -> EvaluationScore (MeanCost, MeanQuality, Score, Feasible)
   - EvaluationScore -> computeScoreDelta -> ScoreDelta (Delta, Improved) -> JSON emission
   - Member outputs -> Combine(ReduceVote / ReduceBestOf / ReduceAllReduce) -> Result (Output, Winner, Tally)

3. LEMMAS:
   - Lemma 1 (Comparison Invalidation): Under IEEE-754, for any comparison op in {<, <=, >, >=, ==}, if either operand
     is NaN, the result is false. Therefore, naive constraint checks (`eval.MeanCost > o.MaxMeanCost` or `eval.MeanQuality < o.MinMeanQuality`)
     evaluate to false when metrics are NaN, causing `Feasible()` to return true (failing open).
   - Lemma 2 (JSON Non-Representability): The JSON specification (RFC 8259) does not admit NaN, +Infinity, or -Infinity.
     Any struct containing non-finite floats causes `json.Marshal` to fail with an unsupported value error.
   - Lemma 3 (Reduction Poisoning): For any arithmetic reduction sum = sum + x, if x is NaN or +Inf with opposite sign,
     the accumulator becomes NaN, poisoning all downstream averages and comparisons.

4. PROOF / MINIMAL VALIDATION FRONTIER:
   - Frontier 1: In `Outcome.IsFinite()`, reject Cost/Quality that are NaN, infinite, or out-of-domain before addition in `Aggregate()`.
   - Frontier 2: In `Objective.IsFinite()`, validate weights and operational bounds before computing `Score()` and `Feasible()`.
   - Frontier 3: In `Objective.Feasible()`, assert `!o.IsFinite() || !eval.IsFinite() => false` before checking constraint thresholds.
   - Frontier 4: In `computeScoreDelta()`, sanitize non-finite floats to 0, ensuring `delta` is finite and `improved` requires `after.Feasible`.
   - Frontier 5: In `Combine()`, validate parsed numbers in `ReduceAllReduce`, sanitizing non-finite weights in `weightOf()`.

5. WITNESS MATRIX:
   - Outcome.IsFinite: rejects NaN, Inf, negative cost, out-of-bound quality.
   - OutcomeJournal.Aggregate: filters non-finite records; yields finite sums, means, and exact admitted count.
   - Objective.Feasible: returns false on NaN/Inf inputs without failing open.
   - computeScoreDelta: never yields Improved=true on NaN; marshals to valid JSON.
   - Combine: rejects "NaN" in all_reduce; handles NaN weight in vote.
*/

func TestNonFiniteTelemetryFailsClosed(t *testing.T) {
	t.Run("Outcome_IsFinite", func(t *testing.T) {
		valid := Outcome{Cost: 0.05, Latency: 100 * time.Millisecond, Quality: 0.95}
		if !valid.IsFinite() {
			t.Fatal("expected valid outcome to be finite")
		}

		invalidCases := []struct {
			name string
			o    Outcome
		}{
			{"NaN_Cost", Outcome{Cost: math.NaN(), Quality: 0.9}},
			{"PosInf_Cost", Outcome{Cost: math.Inf(1), Quality: 0.9}},
			{"NegInf_Cost", Outcome{Cost: math.Inf(-1), Quality: 0.9}},
			{"Negative_Cost", Outcome{Cost: -0.01, Quality: 0.9}},
			{"NaN_Quality", Outcome{Cost: 0.05, Quality: math.NaN()}},
			{"PosInf_Quality", Outcome{Cost: 0.05, Quality: math.Inf(1)}},
			{"NegInf_Quality", Outcome{Cost: 0.05, Quality: math.Inf(-1)}},
			{"Negative_Quality", Outcome{Cost: 0.05, Quality: -0.01}},
			{"Excessive_Quality", Outcome{Cost: 0.05, Quality: 1.01}},
		}

		for _, tc := range invalidCases {
			t.Run(tc.name, func(t *testing.T) {
				if tc.o.IsFinite() {
					t.Fatalf("case %s: expected IsFinite=false, got true", tc.name)
				}
			})
		}
	})

	t.Run("Aggregate_RejectsNonFinite", func(t *testing.T) {
		m := manifestForOutcomes()
		tool := m.Route(Subject{Aspect: AspectToolCall, Tool: "write_file"})

		var j OutcomeJournal
		j.Record(m.Version, tool, Outcome{Cost: 0.10, Latency: 100 * time.Millisecond, Quality: 0.8})
		// Non-finite poisoned records that must NOT enter aggregate reductions:
		j.Record(m.Version, tool, Outcome{Cost: math.NaN(), Latency: 200 * time.Millisecond, Quality: 0.9})
		j.Record(m.Version, tool, Outcome{Cost: math.Inf(1), Latency: 200 * time.Millisecond, Quality: 0.9})
		j.Record(m.Version, tool, Outcome{Cost: 0.20, Latency: 200 * time.Millisecond, Quality: math.NaN()})
		j.Record(m.Version, tool, Outcome{Cost: 0.30, Latency: 300 * time.Millisecond, Quality: 1.0})

		agg := j.Aggregate()
		if agg.Total != 2 {
			t.Fatalf("expected Total=2 valid records admitted, got %d", agg.Total)
		}
		toolKey := AspectRuleKey{Aspect: AspectToolCall, Rule: "tool-writes"}
		ts := agg.ByKey[toolKey]
		if ts.Count != 2 {
			t.Fatalf("expected Count=2, got %d", ts.Count)
		}
		if math.IsNaN(ts.MeanCost) || math.IsInf(ts.MeanCost, 0) || ts.MeanCost != 0.20 {
			t.Fatalf("expected MeanCost=0.20, got %v", ts.MeanCost)
		}
		if math.IsNaN(ts.MeanQuality) || math.IsInf(ts.MeanQuality, 0) || ts.MeanQuality != 0.90 {
			t.Fatalf("expected MeanQuality=0.90, got %v", ts.MeanQuality)
		}
	})

	t.Run("Objective_IsFinite_And_Feasible", func(t *testing.T) {
		validObj := Objective{
			QualityWeight:  1.0,
			CostWeight:     0.5,
			MaxMeanCost:    1.0,
			MinMeanQuality: 0.8,
		}
		if !validObj.IsFinite() {
			t.Fatal("expected validObj to be finite")
		}

		nanObj := Objective{
			QualityWeight: math.NaN(),
		}
		if nanObj.IsFinite() {
			t.Fatal("expected nanObj to not be finite")
		}

		// Proves that Feasible fails closed when eval contains NaN instead of failing open
		nanCostEval := EvaluationScore{
			Count:       10,
			MeanCost:    math.NaN(),
			MeanQuality: 0.95,
			Score:       1.0,
		}
		if validObj.Feasible(nanCostEval) {
			t.Fatal("Feasible must return false when MeanCost is NaN")
		}

		nanQualityEval := EvaluationScore{
			Count:       10,
			MeanCost:    0.10,
			MeanQuality: math.NaN(),
			Score:       1.0,
		}
		if validObj.Feasible(nanQualityEval) {
			t.Fatal("Feasible must return false when MeanQuality is NaN")
		}

		nanScoreEval := EvaluationScore{
			Count:       10,
			MeanCost:    0.10,
			MeanQuality: 0.90,
			Score:       math.NaN(),
		}
		if validObj.Feasible(nanScoreEval) {
			t.Fatal("Feasible must return false when Score is NaN")
		}

		// Feasible must return false when Objective contains NaN constraints
		objWithNanConstraint := Objective{
			MaxMeanCost: math.NaN(),
		}
		cleanEval := EvaluationScore{
			Count:       10,
			MeanCost:    0.10,
			MeanQuality: 0.90,
			Score:       1.0,
		}
		if objWithNanConstraint.Feasible(cleanEval) {
			t.Fatal("Feasible must return false when Objective constraints contain NaN")
		}
	})

	t.Run("EvaluateManifest_RejectsNonFinite", func(t *testing.T) {
		m := manifestForOutcomes()
		tool := m.Route(Subject{Aspect: AspectToolCall, Tool: "write_file"})

		records := []OutcomeRecord{
			RecordOutcome(m.Version, tool, Outcome{Cost: 0.10, Latency: 100 * time.Millisecond, Quality: 0.8}),
			RecordOutcome(m.Version, tool, Outcome{Cost: math.NaN(), Latency: 100 * time.Millisecond, Quality: 0.8}),
			RecordOutcome(m.Version, tool, Outcome{Cost: 0.10, Latency: 100 * time.Millisecond, Quality: math.NaN()}),
		}

		obj := Objective{QualityWeight: 1.0}
		score, err := EvaluateManifest(m, records, obj)
		if err != nil {
			t.Fatalf("EvaluateManifest unexpected error: %v", err)
		}
		if score.Count != 1 {
			t.Fatalf("expected Count=1 valid record, got %d", score.Count)
		}
		if math.IsNaN(score.MeanCost) || math.IsNaN(score.MeanQuality) || math.IsNaN(score.Score) {
			t.Fatal("EvaluateManifest produced NaN metrics")
		}

		// Non-finite objective returns error and fails closed
		_, errNan := EvaluateManifest(m, records, Objective{QualityWeight: math.NaN()})
		if errNan == nil {
			t.Fatal("expected error on non-finite Objective")
		}
	})

	t.Run("ComputeScoreDelta_FailsClosed_And_JSONSafe", func(t *testing.T) {
		before := EvaluationScore{Count: 10, Score: 0.5, Feasible: true}
		afterNaN := EvaluationScore{Count: 10, Score: math.NaN(), Feasible: true}

		delta := computeScoreDelta(before, afterNaN)
		if delta.Improved {
			t.Fatal("computeScoreDelta must not mark improved=true when after Score is NaN")
		}

		// Serialization to JSON must succeed without "json: unsupported value: NaN"
		data, err := json.Marshal(delta)
		if err != nil {
			t.Fatalf("json.Marshal(ScoreDelta) failed on non-finite input: %v", err)
		}
		if len(data) == 0 {
			t.Fatal("expected non-empty JSON output")
		}

		var roundtrip ScoreDelta
		if err := json.Unmarshal(data, &roundtrip); err != nil {
			t.Fatalf("json.Unmarshal failed: %v", err)
		}
		if math.IsNaN(roundtrip.Delta) || math.IsNaN(roundtrip.After.Score) {
			t.Fatal("roundtrip unmarshaled non-finite float")
		}
	})

	t.Run("Reductions_NonFiniteSafe", func(t *testing.T) {
		// weightOf returns 1 on NaN/Inf
		nanMember := Member{Model: "test", Weight: math.NaN()}
		if w := weightOf(nanMember); w != 1.0 {
			t.Fatalf("weightOf(NaN) want 1.0, got %v", w)
		}

		infMember := Member{Model: "test", Weight: math.Inf(1)}
		if w := weightOf(infMember); w != 1.0 {
			t.Fatalf("weightOf(Inf) want 1.0, got %v", w)
		}

		// ReduceVote with NaN weight does not poison vote tallies
		voteRes, err := Combine(ReduceVote, []Vote{
			{Member: Member{Model: "a", Weight: math.NaN()}, Output: "out_a"},
			{Member: Member{Model: "b", Weight: 2.0}, Output: "out_b"},
		})
		if err != nil {
			t.Fatalf("Combine(ReduceVote) error: %v", err)
		}
		if voteRes.Winner != "b" {
			t.Fatalf("expected winner 'b', got %q", voteRes.Winner)
		}
		for out, weight := range voteRes.Tally {
			if math.IsNaN(weight) || math.IsInf(weight, 0) {
				t.Fatalf("tally for %q contains non-finite weight: %v", out, weight)
			}
		}

		// ReduceAllReduce rejects "NaN" and "Inf" string member outputs
		_, errAllReduceNaN := Combine(ReduceAllReduce, []Vote{
			{Member: Member{Model: "a"}, Output: "NaN"},
			{Member: Member{Model: "b"}, Output: "10.0"},
		})
		if errAllReduceNaN == nil {
			t.Fatal("expected Combine(ReduceAllReduce) to reject 'NaN' output")
		}

		_, errAllReduceInf := Combine(ReduceAllReduce, []Vote{
			{Member: Member{Model: "a"}, Output: "+Inf"},
			{Member: Member{Model: "b"}, Output: "10.0"},
		})
		if errAllReduceInf == nil {
			t.Fatal("expected Combine(ReduceAllReduce) to reject '+Inf' output")
		}

		// ReduceBestOf ignores NaN scores
		bestOfRes, err := Combine(ReduceBestOf, []Vote{
			{Member: Member{Model: "a"}, Output: "out_a", Score: math.NaN()},
			{Member: Member{Model: "b"}, Output: "out_b", Score: 0.5},
		})
		if err != nil {
			t.Fatalf("Combine(ReduceBestOf) error: %v", err)
		}
		if bestOfRes.Winner != "b" {
			t.Fatalf("expected winner 'b', got %q", bestOfRes.Winner)
		}
	})
}
