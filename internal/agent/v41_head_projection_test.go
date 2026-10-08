package agent

import (
	"encoding/json"
	"strconv"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/model"
)

// fak-test:justify why=contract when=changed:internal/**
// fak-test:runtime fast est=1ms lane=default
func TestV41HeadProjectionDefaultReader(t *testing.T) {
	t.Parallel()
	keys := []string{"head_projection_device_calls", "head_projection_host_calls", "head_projection_device_rows", "head_projection_host_rows", "head_projection_activation_upload_bytes", "head_projection_readback_bytes", "head_projection_nanos"}
	input := map[string]map[string]int64{}
	for index, phase := range []string{"prefill", "decode"} {
		input[phase] = map[string]int64{"tokens": int64(3 - index)}
		for i, key := range keys {
			input[phase][key] = int64(1000*(index+1) + 17*i)
		}
		input[phase]["head_projection_nanos"] = int64(17000000 + index*14000000)
	}
	raw, err := json.Marshal(input)
	if err != nil {
		t.Fatal(err)
	}
	var attribution model.V41ExpertFaultAttribution
	if err := json.Unmarshal(raw, &attribution); err != nil {
		t.Fatal(err)
	}
	raw, err = json.Marshal(attribution)
	if err != nil {
		t.Fatal(err)
	}
	var roundtrip map[string]map[string]json.RawMessage
	if err := json.Unmarshal(raw, &roundtrip); err != nil {
		t.Fatal(err)
	}
	for phase, values := range input {
		for _, key := range keys {
			var n int64
			if err := json.Unmarshal(roundtrip[phase][key], &n); err != nil {
				t.Errorf("default %s.%s missing: %v", phase, key, err)
				continue
			}
			if n != values[key] {
				t.Errorf("default %s.%s=%d want %d", phase, key, n, values[key])
			}
		}
	}
	clause := formatV41FaultClause(attribution)
	start := strings.Index(clause, "head_projection")
	if start < 0 {
		t.Fatal("default reader omits head projection attribution")
	}
	projection := clause[start:]
	for phase, values := range input {
		if !strings.Contains(projection, phase) {
			t.Errorf("head projection reader omits %s", phase)
		}
		for _, key := range []string{"head_projection_device_calls", "head_projection_host_calls", "head_projection_device_rows", "head_projection_host_rows", "head_projection_activation_upload_bytes", "head_projection_readback_bytes"} {
			if !strings.Contains(projection, strconv.FormatInt(values[key], 10)) {
				t.Errorf("head projection reader omits observed %s.%s", phase, key)
			}
		}
	}
	for _, phase := range []string{"prefill", "decode"} {
		elapsed := input[phase]["head_projection_nanos"]
		input[phase]["head_projection_nanos"] = 0
		withoutElapsed, err := json.Marshal(input)
		if err != nil {
			t.Fatal(err)
		}
		var changed model.V41ExpertFaultAttribution
		if err := json.Unmarshal(withoutElapsed, &changed); err != nil {
			t.Fatal(err)
		}
		if formatV41FaultClause(changed) == clause {
			t.Errorf("head projection reader omits observed %s elapsed time", phase)
		}
		input[phase]["head_projection_nanos"] = elapsed
	}

	if len(clause) > 4200 {
		t.Errorf("default reader exceeds bounded budget: %d bytes", len(clause))
	}
	if formatV41FaultClause(model.V41ExpertFaultAttribution{}) != "" {
		t.Error("inert head projection changed empty summary")
	}
}
