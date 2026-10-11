package compute

import (
	"encoding/json"
	"math"
	"testing"
)

// Each element of a Qwen3.5 sequence panel may differ from the CPU oracle by
// qwen35PanelAbsTol plus qwen35PanelRelTol of the expected magnitude.
const (
	qwen35PanelAbsTol = 2e-3
	qwen35PanelRelTol = 2e-4
)

// qwen35PanelBoundRatio is an element's |delta| over its own tolerance; above 1 fails.
func qwen35PanelBoundRatio(delta, expected float64) float64 {
	return delta / (qwen35PanelAbsTol + qwen35PanelRelTol*math.Abs(expected))
}

// qwen35PanelReceiptDelta rescales the worst ratio onto the receipt's absolute
// max_abs_delta bound, so the receipt passes exactly when no element exceeds its tolerance.
func qwen35PanelReceiptDelta(worstRatio float64) float64 {
	return worstRatio * qwen35PanelAbsTol
}

type qwen35SequenceParityOracleEvent struct {
	Schema         string                             `json:"schema"`
	Selector       string                             `json:"selector"`
	TestName       string                             `json:"test_name"`
	OracleKind     string                             `json:"oracle_kind"`
	Engine         string                             `json:"engine"`
	DeviceObserved bool                               `json:"device_observed"`
	CaseCount      int                                `json:"case_count"`
	Passed         bool                               `json:"passed"`
	Observed       qwen35SequenceParityOracleObserved `json:"observed"`
	Bounds         qwen35SequenceParityOracleBounds   `json:"bounds"`
}

type qwen35SequenceParityOracleObserved struct {
	MaxAbsDelta  float64 `json:"max_abs_delta"`
	FiniteOutput bool    `json:"finite_output"`
}

type qwen35SequenceParityOracleBounds struct {
	MaxAbsDelta float64 `json:"max_abs_delta"`
	// MaxAbsDeltaRel marks max_abs_delta as the floor of a per-element bound
	// max_abs_delta + max_abs_delta_rel*|want|, with the observation rescaled onto it.
	MaxAbsDeltaRel float64 `json:"max_abs_delta_rel,omitempty"`
	RequireFinite  bool    `json:"require_finite"`
}

// formatQwen35SequenceParityOracle takes maxAbsDelta already rescaled by
// qwen35PanelReceiptDelta, so passed agrees with the panel test's per-element assertion.
func formatQwen35SequenceParityOracle(maxAbsDelta float64, finiteOutput bool, caseCount int, fixturePassed bool) ([]byte, error) {
	// Passing metrics from an aborted or filtered fixture are not parity evidence.
	passed := fixturePassed && caseCount == 4 && finiteOutput && maxAbsDelta <= qwen35PanelAbsTol
	event := qwen35SequenceParityOracleEvent{
		Schema:         "fak.strix.subkernel-parity/v1",
		Selector:       "qwen35_sequence_prefill",
		TestName:       "TestVulkanQwen35SequenceQuantizedPanelsMatchCPU",
		OracleKind:     "max_abs",
		Engine:         "fak-native/vulkan",
		DeviceObserved: true,
		CaseCount:      caseCount,
		Passed:         passed,
		Observed: qwen35SequenceParityOracleObserved{
			MaxAbsDelta:  maxAbsDelta,
			FiniteOutput: finiteOutput,
		},
		Bounds: qwen35SequenceParityOracleBounds{
			MaxAbsDelta:    qwen35PanelAbsTol,
			MaxAbsDeltaRel: qwen35PanelRelTol,
			RequireFinite:  true,
		},
	}
	return json.Marshal(event)
}

// TestQwen35PanelReceiptPassMatchesPerElementBound pins that the parity receipt
// and the panel test assertion judge the same elements the same way (#13807): the
// RX 7600 q4_k panel passed its per-element bound but its receipt read passed=false.
// fak-test:runtime fast est=1ms
func TestQwen35PanelReceiptPassMatchesPerElementBound(t *testing.T) {
	for _, tc := range []struct {
		name          string
		delta, want   float64
		elementPasses bool
	}{
		{name: "rx7600_q4k_observed", delta: 0.00341796875, want: 8, elementPasses: true},
		{name: "exact", delta: 0, want: 1, elementPasses: true},
		{name: "floor_pass", delta: 1.9e-3, want: 0, elementPasses: true},
		{name: "floor_fail", delta: 2.5e-3, want: 0, elementPasses: false},
		{name: "relative_pass", delta: 3.9e-3, want: -10, elementPasses: true},
		{name: "relative_fail", delta: 4.1e-3, want: 10, elementPasses: false},
		{name: "q2k_race_red", delta: 4.5499, want: -1.4979788, elementPasses: false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			ratio := qwen35PanelBoundRatio(tc.delta, tc.want)
			if got := !(ratio > 1); got != tc.elementPasses {
				t.Fatalf("panel assertion passes=%v, want %v (ratio %g)", got, tc.elementPasses, ratio)
			}
			raw, err := formatQwen35SequenceParityOracle(qwen35PanelReceiptDelta(ratio), true, 4, true)
			if err != nil {
				t.Fatalf("format parity oracle: %v", err)
			}
			var receipt struct {
				Passed   bool `json:"passed"`
				Observed struct {
					MaxAbsDelta float64 `json:"max_abs_delta"`
				} `json:"observed"`
				Bounds struct {
					MaxAbsDelta    float64 `json:"max_abs_delta"`
					MaxAbsDeltaRel float64 `json:"max_abs_delta_rel"`
				} `json:"bounds"`
			}
			if err := json.Unmarshal(raw, &receipt); err != nil {
				t.Fatalf("unmarshal parity oracle: %v", err)
			}
			if receipt.Passed != tc.elementPasses {
				t.Errorf("receipt passed=%v, panel assertion passes=%v: %s", receipt.Passed, tc.elementPasses, raw)
			}
			if receipt.Bounds.MaxAbsDelta != 2e-3 || receipt.Bounds.MaxAbsDeltaRel != 2e-4 {
				t.Errorf("receipt bounds changed: %s", raw)
			}
			// A receipt pass must also clear the Strix consumer's absolute max_abs check.
			if receipt.Passed && receipt.Observed.MaxAbsDelta > receipt.Bounds.MaxAbsDelta {
				t.Errorf("receipt passes with observed %g above bound %g", receipt.Observed.MaxAbsDelta, receipt.Bounds.MaxAbsDelta)
			}
		})
	}
}
