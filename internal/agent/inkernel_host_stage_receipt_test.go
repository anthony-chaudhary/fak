package agent

import (
	"encoding/json"
	"strings"
	"testing"
)

func TestInKernelHostStageReceiptCopyAndReset(t *testing.T) {
	p := qwenQ4KPrefillPlanner(nil)
	m := &nativeInferenceMeasurement{hostPathBypass: "device-only-session", hostStages: []string{"v41-dense-projection"}}
	r := p.buildNativeInferenceReceipt(m, 0, 0)
	if r.HostPathBypass != m.hostPathBypass || len(r.HostStages) != 1 || !r.FallbackActive {
		t.Fatal("missing actual host observation")
	}
	m.hostStages[0] = "mutated"
	if r.HostStages[0] != "v41-dense-projection" {
		t.Fatal("receipt aliases mutable measurement")
	}
	m.reset()
	if m.hostPathBypass != "" || len(m.hostStages) != 0 {
		t.Fatal("measurement retained previous session")
	}
	empty := p.buildNativeInferenceReceipt(m, 0, 0)
	if empty.FallbackActive {
		t.Fatal("empty observation became fallback")
	}
	raw, err := json.Marshal(empty)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(raw), "host_stages") || strings.Contains(string(raw), "host_path_bypass") {
		t.Fatal("empty observation changed legacy JSON shape")
	}
}
