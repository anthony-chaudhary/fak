package harnessmodelsetconformance_test

import (
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/harnessmodelset"
	"github.com/anthony-chaudhary/fak/internal/modelinventory"
)

func TestTwoRoleModelSetConformanceEdgeAndAdversarialInputs(t *testing.T) {
	intentCases := []struct {
		name string
		raw  string
		want string
	}{
		{name: "empty", raw: "", want: "JSON_INVALID"},
		{name: "malformed", raw: `{"schema":`, want: "JSON_INVALID"},
		{name: "hostile unknown field", raw: `{"schema":"fak-harness-model-intent/1","roles":[],"__proto__":{"polluted":true}}`, want: "FIELD_UNKNOWN"},
		{name: "duplicate key", raw: `{"schema":"fak-harness-model-intent/1","schema":"fak-harness-model-intent/1","roles":[]}`, want: "FIELD_DUPLICATE"},
		{name: "oversized", raw: strings.Repeat(" ", 1<<20+1), want: "JSON_INVALID"},
	}
	for _, tc := range intentCases {
		t.Run("intent/"+tc.name, func(t *testing.T) {
			intent, err := harnessmodelset.ParseJSON([]byte(tc.raw))
			if err == nil {
				t.Fatalf("ParseJSON accepted hostile input: %+v", intent)
			}
			if !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("ParseJSON error = %q, want code %q", err, tc.want)
			}
		})
	}

	inventoryCases := []struct {
		name         string
		observations modelinventory.Observations
		wantPath     string
	}{
		{name: "empty", observations: modelinventory.Observations{}, wantPath: "$.candidates"},
		{name: "hostile provider id", observations: modelinventory.Observations{Providers: []modelinventory.ProviderObservation{{ID: "../../escape"}}}, wantPath: "$.candidates[0].id"},
		{name: "malformed digest", observations: modelinventory.Observations{Locals: []modelinventory.LocalObservation{{ID: "local-safe", Artifact: "models/safe.gguf", Digest: "not-a-sha256", Format: "gguf"}}}, wantPath: "$.candidates[0].digest"},
		{name: "oversized id", observations: modelinventory.Observations{Locals: []modelinventory.LocalObservation{{ID: "m" + strings.Repeat("x", 4096), Artifact: "models/safe.gguf", Digest: "sha256:" + strings.Repeat("a", 64), Format: "gguf"}}}, wantPath: "$.candidates[0].id"},
	}
	for _, tc := range inventoryCases {
		t.Run("inventory/"+tc.name, func(t *testing.T) {
			inventory, diagnostics := modelinventory.Normalize(tc.observations, conformanceTime)
			if len(diagnostics) == 0 {
				t.Fatalf("Normalize accepted hostile observations: %+v", inventory)
			}
			if !strings.Contains(diagnostics.Error(), tc.wantPath) {
				t.Fatalf("Normalize diagnostics = %q, want path %q", diagnostics, tc.wantPath)
			}
		})
	}
}
