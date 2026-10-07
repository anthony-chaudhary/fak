package agent

import (
	"encoding/json"
	"github.com/anthony-chaudhary/fak/internal/model"
	"strings"
	"testing"
)

// fak-test:runtime fast est=1ms lane=default
func TestV41IncrementalDeviceExpertAttributionSummary(t *testing.T) {
	t.Parallel()
	var fa model.V41ExpertFaultAttribution
	if err := json.Unmarshal([]byte(`{"prefill":{"tokens":2,"incremental_device_gate_up_calls":17,"incremental_device_down_calls":19,"incremental_device_dispatch_nanos":23000000},"decode":{"tokens":1,"incremental_device_gate_up_calls":29,"incremental_device_down_calls":31,"incremental_device_dispatch_nanos":37000000}}`), &fa); err != nil {
		t.Fatal(err)
	}
	raw, err := json.Marshal(fa)
	if err != nil {
		t.Fatal(err)
	}
	var fields map[string]map[string]json.RawMessage
	if err := json.Unmarshal(raw, &fields); err != nil {
		t.Fatal(err)
	}
	for phase, expected := range map[string][]string{"prefill": {"17", "19", "23000000"}, "decode": {"29", "31", "37000000"}} {
		for i, key := range []string{"incremental_device_gate_up_calls", "incremental_device_down_calls", "incremental_device_dispatch_nanos"} {
			if string(fields[phase][key]) != expected[i] {
				t.Errorf("default attribution %s.%s lost: %s", phase, key, fields[phase][key])
			}
		}
	}
	got := formatV41FaultClause(fa)
	for _, key := range []string{"incremental_device", "gate_up", "down", "dispatch"} {
		if !strings.Contains(got, key) {
			t.Errorf("default summary omits %s: %s", key, got)
		}
	}
	for _, value := range []string{"17", "19", "29", "31", "23", "37"} {
		if !strings.Contains(got, value) {
			t.Errorf("phase value %s lost in default summary: %s", value, got)
		}
	}
	if len(got) > 1800 {
		t.Errorf("summary clause unbounded: %d bytes", len(got))
	}
	if got := formatV41FaultClause(model.V41ExpertFaultAttribution{}); got != "" {
		t.Error("zero attribution changed summary")
	}
}
