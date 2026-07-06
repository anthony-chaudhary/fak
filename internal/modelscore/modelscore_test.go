package modelscore

import (
	"encoding/json"
	"strings"
	"testing"
)

// sampleEvidence is one fully-populated model row exercising every field the
// done condition names: benchmark name/version/native-score/unit/harness, cost,
// context, and provenance (source/as_of/confidence).
func sampleEvidence() ModelEvidence {
	return ModelEvidence{
		Model: "opus",
		Benchmarks: []BenchScore{
			{
				Benchmark: "terminal-bench", Version: "2.1", Score: 55.0, Unit: "pct-resolved", Harness: "example-agent",
				Provenance: Provenance{Source: "https://www.tbench.ai/leaderboard/terminal-bench/2.1", AsOf: "2026-07-06", Confidence: 0.2, Illustrative: true},
			},
			{
				Benchmark: "swe-bench-verified", Score: 70.0, Unit: "pct-resolved",
				Provenance: Provenance{Source: "https://www.swebench.com/", AsOf: "2026-07-06", Confidence: 0.2, Illustrative: true},
			},
			{
				Benchmark: "frontier-swe", Score: 12.0, Unit: "pct-resolved",
				Provenance: Provenance{Source: "https://www.frontierswe.com/", AsOf: "2026-07-06", Confidence: 0.2, Illustrative: true},
			},
		},
		Cost:    &Cost{In: 3, Out: 15, Provenance: Provenance{Source: "cost.go", Confidence: 0.5}},
		Context: &ContextWindow{Tokens: 200000, Provenance: Provenance{Source: "fixture", Confidence: 0.2}},
	}
}

// The done condition: registry rows preserve benchmark name/version/native
// score/unit/harness/source/confidence and cost fields through a JSON round-trip,
// with NO normalization of the native score.
func TestRoundTripPreservesRawEvidence(t *testing.T) {
	r := NewRegistry()
	if err := r.Add(sampleEvidence()); err != nil {
		t.Fatalf("add: %v", err)
	}

	data, err := json.Marshal(r)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	if !strings.Contains(string(data), Schema) {
		t.Fatalf("snapshot missing schema tag: %s", data)
	}

	back, err := Load(data)
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	got, ok := back.Profile("opus")
	if !ok {
		t.Fatalf("opus missing after round-trip")
	}

	tb, ok := got.Benchmark("terminal-bench")
	if !ok {
		t.Fatalf("terminal-bench row lost")
	}
	// Every field survives, and the native score is byte-for-byte unchanged — no
	// normalization onto a shared 0..100 axis happened anywhere.
	if tb.Version != "2.1" || tb.Score != 55.0 || tb.Unit != "pct-resolved" || tb.Harness != "example-agent" {
		t.Fatalf("terminal-bench row mangled: %+v", tb)
	}
	if tb.Provenance.Source != "https://www.tbench.ai/leaderboard/terminal-bench/2.1" ||
		tb.Provenance.AsOf != "2026-07-06" || tb.Provenance.Confidence != 0.2 || !tb.Provenance.Illustrative {
		t.Fatalf("provenance lost: %+v", tb.Provenance)
	}
	if got.Cost == nil || got.Cost.In != 3 || got.Cost.Out != 15 {
		t.Fatalf("cost lost: %+v", got.Cost)
	}
	if got.Context == nil || got.Context.Tokens != 200000 {
		t.Fatalf("context lost: %+v", got.Context)
	}
}

// A native score may exceed any prior maximum (unbounded); the registry must not
// clamp or reject it.
func TestUnboundedScorePreserved(t *testing.T) {
	r := NewRegistry()
	ev := ModelEvidence{Model: "future", Benchmarks: []BenchScore{
		{Benchmark: "frontier-swe", Score: 250.0, Unit: "tasks", Provenance: Provenance{Source: "x", Confidence: 1}},
	}}
	if err := r.Add(ev); err != nil {
		t.Fatalf("add: %v", err)
	}
	p, _ := r.Profile("future")
	b, _ := p.Benchmark("frontier-swe")
	if b.Score != 250.0 {
		t.Fatalf("unbounded score clamped: %v", b.Score)
	}
}

// The Profile keeps raw evidence side by side and computes NO blended ranking:
// the highest-scoring benchmark row does not become a tier, and score/cost/
// context stay distinct. We assert the shape carries only raw rows.
func TestProfileKeepsEvidenceSeparateFromPolicy(t *testing.T) {
	r := NewRegistry()
	_ = r.Add(sampleEvidence())
	p, _ := r.Profile("opus")

	if len(p.Benchmarks) != 3 {
		t.Fatalf("expected 3 raw benchmark rows, got %d", len(p.Benchmarks))
	}
	// A profile is raw evidence, not a decision: its top-level shape carries only
	// raw-evidence fields — no blended score, tier, ranking, or admission verdict.
	blob, _ := json.Marshal(p)
	var top map[string]json.RawMessage
	if err := json.Unmarshal(blob, &top); err != nil {
		t.Fatalf("unmarshal profile: %v", err)
	}
	allowed := map[string]bool{"model": true, "benchmarks": true, "cost": true, "context": true, "notes": true}
	for k := range top {
		if !allowed[k] {
			t.Fatalf("profile leaked a non-evidence field %q: %s", k, blob)
		}
	}
	for _, banned := range []string{"tier", "blended", "normalized", "rank", "score_total", "admit"} {
		if _, bad := top[banned]; bad {
			t.Fatalf("profile leaked a policy field %q", banned)
		}
	}
}

// Mutating a returned profile must not reach back into registry state.
func TestProfileIsACopy(t *testing.T) {
	r := NewRegistry()
	_ = r.Add(sampleEvidence())
	p, _ := r.Profile("opus")
	p.Benchmarks[0].Score = -999
	p.Cost.Out = -999

	fresh, _ := r.Profile("opus")
	if fresh.Benchmarks[0].Score == -999 || fresh.Cost.Out == -999 {
		t.Fatalf("registry state mutated through returned profile")
	}
}

// Validation fails closed: a row with missing witness/units or an out-of-range
// confidence is refused, so bad evidence never enters the registry.
func TestValidationFailsClosed(t *testing.T) {
	cases := []struct {
		name string
		ev   ModelEvidence
	}{
		{"no model id", ModelEvidence{Benchmarks: []BenchScore{{Benchmark: "x", Unit: "pct", Provenance: Provenance{Source: "s"}}}}},
		{"benchmark no name", ModelEvidence{Model: "m", Benchmarks: []BenchScore{{Unit: "pct", Provenance: Provenance{Source: "s"}}}}},
		{"benchmark no unit", ModelEvidence{Model: "m", Benchmarks: []BenchScore{{Benchmark: "x", Provenance: Provenance{Source: "s"}}}}},
		{"benchmark no source", ModelEvidence{Model: "m", Benchmarks: []BenchScore{{Benchmark: "x", Unit: "pct"}}}},
		{"confidence over 1", ModelEvidence{Model: "m", Benchmarks: []BenchScore{{Benchmark: "x", Unit: "pct", Provenance: Provenance{Source: "s", Confidence: 1.5}}}}},
		{"negative cost", ModelEvidence{Model: "m", Cost: &Cost{In: -1, Provenance: Provenance{Source: "s"}}}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if err := NewRegistry().Add(tc.ev); err == nil {
				t.Fatalf("expected refusal for %q", tc.name)
			}
		})
	}
}

// Load rejects a foreign / missing schema tag rather than silently dropping rows.
func TestLoadRejectsWrongSchema(t *testing.T) {
	if _, err := Load([]byte(`{"schema":"nope","models":[]}`)); err == nil {
		t.Fatalf("expected schema mismatch refusal")
	}
	if _, err := Load([]byte(`{"models":[],"surprise":1}`)); err == nil {
		t.Fatalf("expected unknown-field refusal")
	}
}

// The embedded fixture loads, carries the three named benchmarks, and marks every
// score illustrative — fak never ships an invented benchmark number as measured.
func TestBuiltinFixture(t *testing.T) {
	r, err := Builtin()
	if err != nil {
		t.Fatalf("builtin: %v", err)
	}
	p, ok := r.Profile("opus")
	if !ok {
		t.Fatalf("builtin missing opus")
	}
	for _, name := range []string{"terminal-bench", "swe-bench-verified", "frontier-swe"} {
		b, ok := p.Benchmark(name)
		if !ok {
			t.Fatalf("builtin opus missing %s", name)
		}
		if !b.Provenance.Illustrative {
			t.Fatalf("builtin %s score not marked illustrative: %+v", name, b)
		}
		if b.Provenance.Source == "" {
			t.Fatalf("builtin %s score has no source", name)
		}
	}
	// Every fixture score across every model must be illustrative.
	for _, m := range r.Models() {
		mp, _ := r.Profile(m)
		for _, b := range mp.Benchmarks {
			if !b.Provenance.Illustrative {
				t.Fatalf("fixture %s/%s is not illustrative", m, b.Benchmark)
			}
		}
	}
}
