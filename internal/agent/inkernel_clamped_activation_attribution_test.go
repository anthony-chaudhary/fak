package agent

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/model"
)

// fak-test:runtime fast est=1ms lane=default
func TestV41ClampedDeviceSwiGLUAttributionSummary(t *testing.T) {
	t.Parallel()
	var fa model.V41ExpertFaultAttribution
	input := `{"prefill":{"tokens":2,"expert_activation_device_calls":17,"expert_activation_host_calls":19,"expert_activation_readback_bytes":8192,"expert_activation_nanos":23000000},"decode":{"tokens":1,"expert_activation_device_calls":29,"expert_activation_host_calls":31,"expert_activation_readback_bytes":16384,"expert_activation_nanos":37000000}}`
	if err := json.Unmarshal([]byte(input), &fa); err != nil {
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
	keys := []string{"expert_activation_device_calls", "expert_activation_host_calls", "expert_activation_readback_bytes", "expert_activation_nanos"}
	for phase, values := range map[string][]string{"prefill": {"17", "19", "8192", "23000000"}, "decode": {"29", "31", "16384", "37000000"}} {
		for i, key := range keys {
			if string(fields[phase][key]) != values[i] {
				t.Errorf("default %s.%s=%s want %s", phase, key, fields[phase][key], values[i])
			}
		}
	}
	got := formatV41FaultClause(fa)
	for _, clause := range []string{"expert_activation prefill=[device=17 host=19 readback=8192B elapsed=0.023s]", "decode=[device=29 host=31 readback=16384B elapsed=0.037s]"} {
		if !strings.Contains(got, clause) {
			t.Errorf("default summary lost selected activation clause %q: %s", clause, got)
		}
	}
	if len(got) > 2200 {
		t.Errorf("summary is unbounded: %d bytes", len(got))
	}
	if got := formatV41FaultClause(model.V41ExpertFaultAttribution{}); got != "" {
		t.Error("empty activation attribution changed zero summary")
	}
}
