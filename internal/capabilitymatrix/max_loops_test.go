package capabilitymatrix

import (
	"encoding/json"
	"strings"
	"testing"
)

// fak-test:runtime fast est=2ms lane=default
func TestCapabilitiesMaxLoopsWire(t *testing.T) {
	cases := []struct {
		name    string
		caps    Capabilities
		present bool
	}{
		{"unlooped omits", Capabilities{}, false},
		{"looped emits", Capabilities{MaxLoops: 5}, true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			raw, err := json.Marshal(tc.caps)
			if err != nil {
				t.Fatal(err)
			}
			if got := strings.Contains(string(raw), `"max_loops"`); got != tc.present {
				t.Fatalf("max_loops present=%v in %s", got, raw)
			}
			var back Capabilities
			if err := json.Unmarshal(raw, &back); err != nil {
				t.Fatal(err)
			}
			if back.MaxLoops != tc.caps.MaxLoops {
				t.Fatalf("round trip MaxLoops=%d, want %d", back.MaxLoops, tc.caps.MaxLoops)
			}
		})
	}
	t.Setenv("FAK_MODEL_CAPABILITIES", `{"looped-x":{"max_loops":4}}`)
	if got := Lookup("looped-x").MaxLoops; got != 4 {
		t.Fatalf("env override MaxLoops=%d, want 4", got)
	}
}
