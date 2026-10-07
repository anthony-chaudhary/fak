package agent

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/model"
)

// fak-test:justify why=regression when=changed:internal/agent/**
// fak-test:runtime fast est=1ms lane=default
func TestV41IncrementalEngramAttributionSummary(t *testing.T) {
	t.Parallel()
	var fa model.V41ExpertFaultAttribution
	if err := json.Unmarshal([]byte(`{"prefill":{"tokens":3,"incremental_engram_injections":17,"incremental_engram_rows":19,"incremental_engram_hash_tokens":23,"incremental_engram_nanos":29000000},"decode":{"tokens":2,"incremental_engram_injections":31,"incremental_engram_rows":37,"incremental_engram_hash_tokens":41,"incremental_engram_nanos":43000000}}`), &fa); err != nil {
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
	for phase, want := range map[string][]int64{"prefill": {17, 19, 23, 29000000}, "decode": {31, 37, 41, 43000000}} {
		for i, key := range []string{"incremental_engram_injections", "incremental_engram_rows", "incremental_engram_hash_tokens", "incremental_engram_nanos"} {
			var got int64
			if err := json.Unmarshal(fields[phase][key], &got); err != nil {
				t.Errorf("%s.%s absent: %v", phase, key, err)
				continue
			}
			if got != want[i] {
				t.Errorf("%s.%s=%d want=%d", phase, key, got, want[i])
			}
		}
	}
	got := formatV41FaultClause(fa)
	for _, word := range []string{"incremental_engram", "injections", "rows", "hash_tokens"} {
		if !strings.Contains(got, word) {
			t.Errorf("default summary omits %s", word)
		}
	}
	for _, value := range []string{"17", "19", "23", "29", "31", "37", "41", "43"} {
		if !strings.Contains(got, value) {
			t.Errorf("default summary omits phase value %s", value)
		}
	}
	if len(got) > 2200 {
		t.Errorf("default summary unbounded: %d bytes", len(got))
	}
	if got := formatV41FaultClause(model.V41ExpertFaultAttribution{}); got != "" {
		t.Error("zero attribution changed summary")
	}
}
