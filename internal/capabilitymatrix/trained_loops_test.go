package capabilitymatrix

import (
	"encoding/json"
	"testing"
)

// fak-test:runtime fast est=2ms lane=default
func TestTrainedLoopsCapabilityJSONRoundTrip(t *testing.T) {
	in := Capabilities{MaxLoops: 8, TrainedLoops: 3}
	raw, err := json.Marshal(in)
	if err != nil {
		t.Fatal(err)
	}
	var out Capabilities
	if err := json.Unmarshal(raw, &out); err != nil {
		t.Fatal(err)
	}
	if out.TrainedLoops != 3 || out.MaxLoops != 8 {
		t.Fatalf("round trip = %+v, want trained 3 max 8 (raw %s)", out, raw)
	}
}
