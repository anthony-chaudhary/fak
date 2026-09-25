package agent

// inkernel_fault_clause_test.go — the witness for the V4.1 elapsed-split
// surfacing leaf (fak#13510).
//
// The #13299 ledger already separates routed-expert fault, dequantization and
// contraction DURATIONS, but the execution-summary clause that every physical
// strix3 receipt parses rendered only BYTES — so the 261.54 s prefill at
// fc7e86a7e could not be attributed to disk wait vs f32 materialization vs the
// scalar contraction. This test pins the appended clause: distinct nanosecond
// accumulators must each reach the rendered string, per phase, with the observed
// contraction backend, and a zero attribution must still render the empty clause
// so non-V4.1 turns are byte-for-byte unchanged.

import (
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/model"
)

// TestV41FaultAttributionClauseElapsed asserts the pure renderer carries the
// elapsed fault/dequant/contraction split for both phases plus the backend, and
// still returns the empty clause for a zero attribution.
func TestV41FaultAttributionClauseElapsed(t *testing.T) {
	if got := formatV41FaultClause(model.V41ExpertFaultAttribution{}); got != "" {
		t.Fatalf("zero attribution clause = %q, want empty (non-V4.1 turns unchanged)", got)
	}

	var fa model.V41ExpertFaultAttribution
	// Prefill: distinct durations so a copied/shared value would be caught.
	fa.Prefill.Tokens = 10
	fa.Prefill.Faults = 4
	fa.Prefill.ResidentHits = 6
	fa.Prefill.FaultedBytes = 4 << 20
	fa.Prefill.DequantBytes = 8 << 20
	fa.Prefill.FaultDoorNanos = 2_000_000_000        // 2.000 s
	fa.Prefill.DequantNanos = 3_000_000_000          // 3.000 s
	fa.Prefill.ContractionNanos = 500_000_000        // 0.500 s
	fa.Prefill.FaultNanosPerToken = 200_000_000      // 200 ms
	fa.Prefill.DequantNanosPerToken = 300_000_000    // 300 ms
	fa.Prefill.ContractionNanosPerToken = 50_000_000 // 50 ms
	fa.Prefill.ContractionBackend = "vulkan"
	// Decode: different magnitudes again.
	fa.Decode.Tokens = 1
	fa.Decode.Faults = 2
	fa.Decode.FaultDoorNanos = 1_000_000_000       // 1.000 s
	fa.Decode.DequantNanos = 1_500_000_000         // 1.500 s
	fa.Decode.ContractionNanos = 100_000_000       // 0.100 s
	fa.Decode.FaultNanosPerToken = 1_000_000_000   // 1000 ms
	fa.Decode.DequantNanosPerToken = 1_500_000_000 // 1500 ms
	fa.Decode.ContractionNanosPerToken = 100_000_000
	fa.Decode.ContractionBackend = "vulkan"

	got := formatV41FaultClause(fa)

	// The byte clause is preserved verbatim (existing receipts still parse).
	if !strings.Contains(got, "v41_faults prefill=[faults=4f/") {
		t.Fatalf("byte clause missing/reshaped: %q", got)
	}
	// Each distinct elapsed total and per-token rate must reach the string.
	for _, want := range []string{
		"elapsed prefill=[fault=2.000s/200.000mspt dequant=3.000s/300.000mspt contraction=0.500s/50.000mspt]",
		"decode=[fault=1.000s/1000.000mspt dequant=1.500s/1500.000mspt contraction=0.100s/100.000mspt]",
		"contraction_backend=vulkan",
	} {
		if !strings.Contains(got, want) {
			t.Fatalf("clause missing %q:\n%s", want, got)
		}
	}

	// No contraction ran on either phase: the backend reports a dash, not an
	// ambiguous empty token.
	var noContract model.V41ExpertFaultAttribution
	noContract.Prefill.Tokens = 1
	noContract.Prefill.Faults = 1
	noContract.Prefill.FaultDoorNanos = 1
	if c := formatV41FaultClause(noContract); !strings.Contains(c, "contraction_backend=-") {
		t.Fatalf("no-contraction clause backend = %q, want dash", c)
	}
}
