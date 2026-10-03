package agent

import (
	"encoding/json"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/model"
)

// fak-test:runtime fast est=1ms lane=default
func TestNativeInferenceReceiptCarriesObservedHostFallback(t *testing.T) {
	planner := &InKernelPlanner{modelID: "gpu-only-receipt"}

	for _, observed := range []bool{false, true} {
		measurement := &nativeInferenceMeasurement{hostFallbackObserved: observed}
		raw, err := json.Marshal(planner.buildNativeInferenceReceipt(measurement, 0, 0))
		if err != nil {
			t.Fatal(err)
		}
		var receipt model.NativeInferenceReceipt
		if err := json.Unmarshal(raw, &receipt); err != nil {
			t.Fatal(err)
		}
		if receipt.FallbackActive != observed {
			t.Fatalf("fallback_active=%t, want observed session state %t; json=%s", receipt.FallbackActive, observed, raw)
		}
	}
}
