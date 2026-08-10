package main

import (
	"bytes"
	"os"
	"strings"
	"testing"
)

func TestRunScheduleHeld(t *testing.T) {
	input := `{"schema":"fak-held-schedule-input/1","jobs":[{"id":"a","cached_input_tokens":1000,"decode_tokens":1,"service_ms":2},{"id":"b","uncached_input_tokens":5,"decode_tokens":100,"service_ms":100},{"id":"c","uncached_input_tokens":200,"decode_tokens":1,"service_ms":20}],"calibrated":{"name":"calibrated","prefill_rate_ms_per_token":0.1,"cache_read_rate_ms_per_token":0.001,"decode_rate_ms_per_token":1},"scalar_total":{"name":"scalar","prefill_rate_ms_per_token":0.05,"cache_read_rate_ms_per_token":0.05,"decode_rate_ms_per_token":0.05}}`
	p := t.TempDir() + "/held.json"
	if err := os.WriteFile(p, []byte(input), 0o600); err != nil {
		t.Fatal(err)
	}
	var out, stderr bytes.Buffer
	if rc := runScheduleHeld(&out, &stderr, []string{"--in", p, "--overhead-iterations", "10"}); rc != 0 {
		t.Fatalf("rc=%d stderr=%s", rc, stderr.String())
	}
	if !strings.Contains(out.String(), `"calibrated_beats_scalar_total": true`) {
		t.Fatalf("output=%s", out.String())
	}
}
