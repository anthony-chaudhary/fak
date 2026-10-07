package agent

import (
	"encoding/json"
	"testing"
)

// fak-test:runtime fast est=5ms lane=default
func TestParseOptionalTimingsReadsLlamaDraftPair(t *testing.T) {
	got := parseOptionalTimings(json.RawMessage(`{"prompt_n":10,"predicted_n":40,"draft_n":32,"draft_n_accepted":24}`))
	if got == nil || got.DraftN != 32 || got.DraftNAccepted != 24 {
		t.Fatalf("timings = %+v, want draft 32/24", got)
	}
	raw, err := json.Marshal(NewTimings(100, 0, 10, 1, 1))
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]any
	if err := json.Unmarshal(raw, &m); err != nil {
		t.Fatal(err)
	}
	for _, k := range []string{"draft_n", "draft_n_accepted"} {
		if _, ok := m[k]; ok {
			t.Fatalf("undrafted timings carry %q: %s", k, raw)
		}
	}
}
