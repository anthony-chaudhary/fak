package perfledger

import (
	"encoding/json"
	"math"
	"strings"
	"testing"
)

// fak-test:runtime fast est=10ms lane=default
func TestCPUPerTokenRoundTripsAndSummarizes(t *testing.T) {
	n := NewNativeTiming(128, 11, 10, 100).WithHostWork(2.5, true, 0.25, true)
	raw, err := json.Marshal(Record{NativeTiming: n})
	if err != nil {
		t.Fatal(err)
	}
	var back Record
	if err := json.Unmarshal(raw, &back); err != nil {
		t.Fatal(err)
	}
	got := back.NativeTiming
	if got == nil || got.CPUMSPerToken == nil || *got.CPUMSPerToken != 2.5 || got.HostShare == nil || *got.HostShare != 0.25 {
		t.Fatalf("round trip = %s", raw)
	}
	rep := BuildReport([]Record{{NativeTiming: got}, {NativeTiming: NewNativeTiming(0, 8, 0, 2)}}, 2, false, 0)
	s := rep.Summary.NativeTiming
	if s == nil || s.HostWorkMeasured != 1 || s.HostShareP50 != 0.25 || s.CPUMSPerTokenP50 != 2.5 {
		t.Fatalf("summary = %+v, want one host-work row at 0.25 / 2.5", s)
	}
	if line := RenderCompact(rep); !strings.Contains(line, "host share p50=0.250 cpu p50=2.50ms/tok (measured 1/2)") {
		t.Fatalf("compact line missing host share: %s", line)
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestCPUPerTokenAbsentIsNotZero(t *testing.T) {
	n := NewNativeTiming(0, 8, 0, 2).WithHostWork(math.NaN(), true, 0, false)
	if n.CPUMSPerToken != nil || n.HostShare != nil {
		t.Fatalf("unmeasured host work must stay absent: %+v", n)
	}
	raw, _ := json.Marshal(n)
	if strings.Contains(string(raw), "host_share") || strings.Contains(string(raw), "cpu_ms_per_token") {
		t.Fatalf("absent host work leaked onto the wire: %s", raw)
	}
	zero := NewNativeTiming(0, 8, 0, 2).WithHostWork(0, true, 0, true)
	if zero.HostShare == nil || *zero.HostShare != 0 {
		t.Fatal("a measured zero must stay a measurement")
	}
}
