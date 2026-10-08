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
func TestV41GroupedOutputAttributionDefaultSummary(t *testing.T) {
	t.Parallel()
	keys := []string{"grouped_output_device_calls", "grouped_output_host_calls", "grouped_output_device_rows", "grouped_output_host_rows", "grouped_output_matmul_calls", "grouped_output_activation_upload_bytes", "grouped_output_readback_bytes", "grouped_output_nanos", "grouped_output_host_weight_f32_bytes"}
	input := map[string]map[string]int64{}
	for index, phase := range []string{"prefill", "decode"} {
		input[phase] = map[string]int64{"tokens": int64(2 - index)}
		for i, key := range keys {
			input[phase][key] = int64(1000*(index+1) + 17*i)
		}
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
			var got int64
			if err := json.Unmarshal(roundtrip[phase][key], &got); err != nil {
				t.Errorf("default %s.%s is unavailable: %v", phase, key, err)
				continue
			}
			if got != values[key] {
				t.Errorf("default %s.%s=%d want %d", phase, key, got, values[key])
			}
		}
	}
	clause := formatV41FaultClause(attribution)
	start := strings.Index(clause, "grouped_output")
	if start < 0 {
		t.Fatal("default request summary omits grouped-output attribution")
	}
	grouped := clause[start:]
	for _, phase := range []string{"prefill", "decode"} {
		if !strings.Contains(grouped, phase) {
			t.Errorf("grouped summary omits %s phase", phase)
		}
		for _, key := range []string{"grouped_output_device_calls", "grouped_output_host_calls", "grouped_output_device_rows", "grouped_output_host_rows", "grouped_output_activation_upload_bytes", "grouped_output_readback_bytes", "grouped_output_host_weight_f32_bytes"} {
			if !strings.Contains(grouped, strconv.FormatInt(input[phase][key], 10)) {
				t.Errorf("grouped summary omits observed %s.%s", phase, key)
			}
		}
	}
	if len(clause) > 3500 {
		t.Errorf("default grouped summary exceeds bounded reader budget: %d bytes", len(clause))
	}
	if formatV41FaultClause(model.V41ExpertFaultAttribution{}) != "" {
		t.Error("inert grouped ledger changed empty summary")
	}
}
