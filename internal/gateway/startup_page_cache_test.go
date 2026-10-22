package gateway

import (
	"encoding/json"
	"testing"
)

// The boot's page-cache residency reaches /debug/vars startup.model_load.page_cache
// with its startup case, a cold zero stays visible, and an unmeasured boot omits it.
// fak-test:runtime fast est=10ms lane=default
func TestModelLoadResidencyDebugVars(t *testing.T) {
	wire := func(p *ModelLoadProfile) map[string]any {
		t.Helper()
		srv := newTestServer(t)
		srv.SetModelLoadProfile(p)
		if p != nil && p.StartupResidency != nil {
			p.StartupResidency.StartupCase = "mutated-after-set"
		}
		b, err := json.Marshal(debugModelLoadProfile(srv.modelLoadProfile()))
		if err != nil {
			t.Fatal(err)
		}
		var out map[string]any
		if err := json.Unmarshal(b, &out); err != nil {
			t.Fatal(err)
		}
		return out
	}
	cold := wire(&ModelLoadProfile{TotalSeconds: 1, StartupResidency: &ModelLoadResidency{StartupCase: "cold", FileBytes: 4096, Files: 2}})
	pc, ok := cold["page_cache"].(map[string]any)
	if !ok {
		t.Fatalf("page_cache missing from %v", cold)
	}
	if pc["startup_case"] != "cold" || pc["resident_bytes"] != float64(0) || pc["file_bytes"] != float64(4096) || pc["files"] != float64(2) {
		t.Fatalf("page_cache = %v, want the cold boot with resident_bytes 0", pc)
	}
	if _, ok := wire(&ModelLoadProfile{TotalSeconds: 1})["page_cache"]; ok {
		t.Fatal("an unmeasured boot reported page_cache")
	}
}
