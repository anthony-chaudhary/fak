package metrics

import (
	"encoding/json"
	"testing"
)

// Unit 85: Hist.RecordNs over a known set; P50/P99/Mean sane and monotonic.
func TestHistPercentilesMonotonic(t *testing.T) {
	var h Hist
	// Known, sorted, evenly spaced set: 100,200,...,1000 (10 samples).
	for i := int64(1); i <= 10; i++ {
		h.RecordNs(i * 100)
	}
	if got := h.Count(); got != 10 {
		t.Fatalf("Count() = %d, want 10", got)
	}

	p50 := h.P50()
	p99 := h.P99()
	mean := h.Mean()

	// All values lie within [100,1000]; every statistic must be in range.
	for name, v := range map[string]int64{"P50": p50, "P99": p99, "Mean": mean} {
		if v < 100 || v > 1000 {
			t.Errorf("%s = %d, out of expected range [100,1000]", name, v)
		}
	}

	// Mean of 100..1000 step 100 is 550.
	if mean != 550 {
		t.Errorf("Mean() = %d, want 550", mean)
	}

	// Monotonic: P50 <= P99 (P99 is at the tail of the sorted sample set).
	if p50 > p99 {
		t.Errorf("expected P50 (%d) <= P99 (%d)", p50, p99)
	}

	// Index math: idx(50) = int(0.5*9) = 4 -> sorted[4] = 500.
	if p50 != 500 {
		t.Errorf("P50() = %d, want 500", p50)
	}
	// idx(99) = int(0.99*9) = int(8.91) = 8 -> sorted[8] = 900.
	if p99 != 900 {
		t.Errorf("P99() = %d, want 900", p99)
	}

	// Empty histogram returns zeroes (no panic).
	var empty Hist
	if empty.P50() != 0 || empty.P99() != 0 || empty.Mean() != 0 {
		t.Errorf("empty hist stats = (%d,%d,%d), want all 0", empty.P50(), empty.P99(), empty.Mean())
	}
}

// Unit 82: Buckets() — sum of bucket Counts equals number of samples.
func TestBucketsSumEqualsSamples(t *testing.T) {
	var h Hist
	// Sample across the full edge spread, including exact edge values and a
	// value that exceeds the largest edge (lands in the >=1s catch-all).
	samples := []int64{
		50,            // <100ns
		100,           // exactly an edge -> not < 100, lands <1µs
		999,           // <1µs
		1_000,         // edge -> <10µs
		50_000,        // <100µs
		500_000,       // <1ms
		5_000_000,     // <10ms
		50_000_000,    // <100ms
		500_000_000,   // <1s
		1_000_000_000, // edge of largest -> catch-all >=1s
		2_000_000_000, // >=1s
	}
	for _, s := range samples {
		h.RecordNs(s)
	}

	buckets := h.Buckets()
	var total int
	for _, b := range buckets {
		if b.Count < 0 {
			t.Errorf("bucket %q has negative count %d", b.Label, b.Count)
		}
		total += b.Count
	}
	if total != len(samples) {
		t.Errorf("sum of bucket counts = %d, want %d (every sample must be placed)", total, len(samples))
	}

	// There should be one bucket per edge plus a catch-all (9 buckets).
	if len(buckets) != 9 {
		t.Errorf("Buckets() len = %d, want 9", len(buckets))
	}

	// Empty histogram: sum is zero, buckets still present.
	var empty Hist
	eb := empty.Buckets()
	var esum int
	for _, b := range eb {
		esum += b.Count
	}
	if esum != 0 {
		t.Errorf("empty buckets sum = %d, want 0", esum)
	}
}

// Unit 81: Report.ComputeGate — subsystem check passes when in-process p50 beats baseline, fail otherwise.
func TestComputeGate(t *testing.T) {
	// Pass case: On p50 (1000ns) < baseline p50 (5ms).
	r := &Report{}
	r.On.P50Ns = 1000
	r.Baseline.P50Ns = 5_000_000
	r.ComputeGate()
	if r.GatePrimary != "pass" {
		t.Errorf("GatePrimary = %q, want \"pass\"", r.GatePrimary)
	}
	if r.PrimaryDetail == "" {
		t.Errorf("PrimaryDetail empty, want a populated detail string")
	}

	// Fail case: On p50 (10ms) does not beat baseline p50 (5ms).
	r.On.P50Ns = 10_000_000
	r.ComputeGate()
	if r.GatePrimary != "fail" {
		t.Errorf("GatePrimary = %q, want \"fail\"", r.GatePrimary)
	}

	// Defensive: baseline of 0 must not pass (no positive baseline to beat).
	z := &Report{}
	z.On.P50Ns = 1
	z.Baseline.P50Ns = 0
	z.ComputeGate()
	if z.GatePrimary != "fail" {
		t.Errorf("with zero baseline, GatePrimary = %q, want \"fail\"", z.GatePrimary)
	}
}

// Unit 80: Report.Validate — mismatched workload hashes error; matching returns nil.
func TestValidateWorkloadHash(t *testing.T) {
	r := &Report{}
	if err := r.Validate("h1", "h2"); err == nil {
		t.Errorf("Validate(\"h1\",\"h2\") = nil, want error on mismatched workload hashes")
	}
	if err := r.Validate("h", "h"); err != nil {
		t.Errorf("Validate(\"h\",\"h\") = %v, want nil", err)
	}
}

// Unit 83: Report.JSON unmarshals and carries provenance fields; token delta is structural.
func TestReportJSONAndTokenDelta(t *testing.T) {
	r := &Report{}
	r.Provenance = Provenance{
		Command:      "fak bench",
		SliceID:      "slice-42",
		WorkloadHash: "abc123",
	}
	// Structural token fields: set in/out tokens on both arms and a delta pct.
	r.On.InTokens = 800
	r.On.OutTokens = 200
	r.Off.InTokens = 1000
	r.Off.OutTokens = 300
	r.TokenDeltaPct = -23.0

	b := r.JSON()
	if len(b) == 0 {
		t.Fatalf("JSON() returned empty")
	}

	// Must unmarshal cleanly.
	var round Report
	if err := json.Unmarshal(b, &round); err != nil {
		t.Fatalf("JSON() did not round-trip unmarshal: %v", err)
	}

	// Provenance fields must survive.
	if round.Provenance.Command != "fak bench" {
		t.Errorf("provenance command = %q, want \"fak bench\"", round.Provenance.Command)
	}
	if round.Provenance.SliceID != "slice-42" {
		t.Errorf("provenance slice_id = %q, want \"slice-42\"", round.Provenance.SliceID)
	}
	if round.Provenance.WorkloadHash != "abc123" {
		t.Errorf("provenance workload_hash = %q, want \"abc123\"", round.Provenance.WorkloadHash)
	}

	// The serialized JSON must literally contain the provenance keys.
	var generic map[string]any
	if err := json.Unmarshal(b, &generic); err != nil {
		t.Fatalf("generic unmarshal failed: %v", err)
	}
	prov, ok := generic["provenance"].(map[string]any)
	if !ok {
		t.Fatalf("provenance object missing from JSON")
	}
	// The lineage triple (git_commit/utc/hostname) is the #9 convergence — every
	// bench artifact now carries the same four traceability axes.
	for _, k := range []string{"command", "slice_id", "workload_hash", "git_commit", "utc", "hostname"} {
		if _, present := prov[k]; !present {
			t.Errorf("provenance JSON missing field %q", k)
		}
	}

	// Token delta is carried structurally on the report.
	if round.TokenDeltaPct != -23.0 {
		t.Errorf("token_delta_pct = %v, want -23", round.TokenDeltaPct)
	}
	// And the per-arm token fields survive the round trip.
	if round.On.InTokens != 800 || round.On.OutTokens != 200 {
		t.Errorf("on-arm tokens = (%d,%d), want (800,200)", round.On.InTokens, round.On.OutTokens)
	}
	if round.Off.InTokens != 1000 || round.Off.OutTokens != 300 {
		t.Errorf("off-arm tokens = (%d,%d), want (1000,300)", round.Off.InTokens, round.Off.OutTokens)
	}
}
