package perfledger

import (
	"encoding/json"
	"strings"
	"testing"
)

// fak-test:runtime fast est=5ms lane=default
func TestSummaryFoldsGPUBusyAndSlowestTurn(t *testing.T) {
	busy := func(v int) *int { return &v }
	recs := []Record{
		{RequestID: "a", InboundRequestID: "rt-a", E2EMS: 100, GPUBusyPercent: busy(20)},
		{RequestID: "b", InboundRequestID: "rt-b", E2EMS: 900, GPUBusyPercent: busy(90)},
		{RequestID: "c", E2EMS: 300},
		{RequestID: "probe", E2EMS: 5000, Synthetic: true},
	}
	s := Summarize(recs)
	if s.GPUBusyMeasured != 2 || s.GPUBusyP50 != 20 {
		t.Fatalf("gpu busy fold = measured %d p50 %v, want 2/20", s.GPUBusyMeasured, s.GPUBusyP50)
	}
	if s.Slowest == nil || s.Slowest.RequestID != "b" || s.Slowest.InboundRequestID != "rt-b" || s.Slowest.E2EMS != 900 {
		t.Fatalf("slowest = %+v, want b/rt-b/900", s.Slowest)
	}
	line := RenderCompact(BuildReport(recs, 0, false, 0))
	for _, want := range []string{"gpu busy p50=20% (measured 2/3)", "slowest e2e=", "request_id=b", "inbound_request_id=rt-b"} {
		if !strings.Contains(line, want) {
			t.Errorf("compact line missing %q: %s", want, line)
		}
	}
}

// fak-test:runtime fast est=5ms lane=default
func TestJoinFieldsOmittedWhenUnmeasured(t *testing.T) {
	b, err := json.Marshal(Record{Schema: Schema})
	if err != nil {
		t.Fatal(err)
	}
	for _, key := range []string{"gpu_busy_percent", "inbound_request_id"} {
		if strings.Contains(string(b), key) {
			t.Errorf("unmeasured row carries %q: %s", key, b)
		}
	}
	idle := 0
	b, _ = json.Marshal(Record{Schema: Schema, GPUBusyPercent: &idle})
	var back Record
	if err := json.Unmarshal(b, &back); err != nil || back.GPUBusyPercent == nil || *back.GPUBusyPercent != 0 {
		t.Fatalf("measured idle GPU did not round-trip as 0: %s", b)
	}
	if line := RenderCompact(BuildReport([]Record{{E2EMS: 10}}, 0, false, 0)); strings.Contains(line, "gpu busy") || strings.Contains(line, "slowest") {
		t.Errorf("compact line names unmeasured fields: %s", line)
	}
}
