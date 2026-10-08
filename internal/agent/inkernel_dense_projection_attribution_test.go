package agent

import (
	"encoding/json"
	"github.com/anthony-chaudhary/fak/internal/model"
	"strings"
	"testing"
)

// fak-test:runtime fast est=1ms lane=default
func TestV41DenseProjectionAttributionSummary(t *testing.T) {
	t.Parallel()
	input := `{"prefill":{"tokens":2,"dense_projection_device_calls":17,"dense_projection_device_rows":19,"dense_projection_host_calls":23,"dense_projection_host_rows":29,"dense_projection_activation_upload_bytes":4096,"dense_projection_readback_bytes":8192,"dense_projection_nanos":31000000},"decode":{"tokens":1,"dense_projection_device_calls":37,"dense_projection_device_rows":41,"dense_projection_host_calls":43,"dense_projection_host_rows":47,"dense_projection_activation_upload_bytes":16384,"dense_projection_readback_bytes":32768,"dense_projection_nanos":53000000}}`
	var fa model.V41ExpertFaultAttribution
	if err := json.Unmarshal([]byte(input), &fa); err != nil {
		t.Fatal(err)
	}
	raw, err := json.Marshal(fa)
	if err != nil {
		t.Fatal(err)
	}
	var phases map[string]map[string]json.RawMessage
	if err := json.Unmarshal(raw, &phases); err != nil {
		t.Fatal(err)
	}
	keys := []string{"dense_projection_device_calls", "dense_projection_device_rows", "dense_projection_host_calls", "dense_projection_host_rows", "dense_projection_activation_upload_bytes", "dense_projection_readback_bytes", "dense_projection_nanos"}
	for phase, expected := range map[string][]string{"prefill": {"17", "19", "23", "29", "4096", "8192", "31000000"}, "decode": {"37", "41", "43", "47", "16384", "32768", "53000000"}} {
		for i, key := range keys {
			if string(phases[phase][key]) != expected[i] {
				t.Errorf("default %s.%s=%s want %s", phase, key, phases[phase][key], expected[i])
			}
		}
	}
	got := formatV41FaultClause(fa)
	for _, clause := range []string{"dense_projection prefill=[device=17calls/19rows host=23calls/29rows upload=4096B readback=8192B elapsed=0.031s]", "decode=[device=37calls/41rows host=43calls/47rows upload=16384B readback=32768B elapsed=0.053s]"} {
		if !strings.Contains(got, clause) {
			t.Errorf("default clause omitted %q: %s", clause, got)
		}
	}
	if len(got) > 2600 {
		t.Errorf("default summary unbounded: %d bytes", len(got))
	}
	if formatV41FaultClause(model.V41ExpertFaultAttribution{}) != "" {
		t.Error("empty projection attribution changed zero summary")
	}
}
