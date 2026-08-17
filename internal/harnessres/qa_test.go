package harnessres

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestFoldFixturesAndRenderers(t *testing.T) {
	fixtures := []struct {
		name      string
		fold      func(*Sampler)
		wantCPU   float64
		wantPeak  uint64
		wantCPUOK bool
		wantRSSOK bool
	}{
		{name: "cpu percent", fold: func(s *Sampler) {
			s.foldProc(procSample{haveCPU: true, cpuUser: 500 * time.Millisecond}, time.Unix(2, 0), 1, 0)
		}, wantCPU: 25, wantCPUOK: true},
		{name: "peak tracking", fold: func(s *Sampler) {
			s.foldProc(procSample{haveRSS: true, rss: 100}, time.Unix(1, 0), 1, 0)
			s.foldProc(procSample{haveRSS: true, rss: 80}, time.Unix(2, 0), 1, 0)
		}, wantPeak: 100, wantRSSOK: true},
		{name: "missing axes", fold: func(s *Sampler) { s.foldProc(procSample{}, time.Unix(1, 0), 1, 0) }},
	}

	for _, fixture := range fixtures {
		t.Run(fixture.name, func(t *testing.T) {
			got := New()
			got.start = time.Unix(0, 0)
			got.nowFn = func() time.Time { return time.Unix(2, 0) }
			fixture.fold(got)
			snapshot := got.Snapshot()
			if cpuPct(snapshot) != fixture.wantCPU || snapshot.Kernel.PeakRSSBytes != fixture.wantPeak || snapshot.Kernel.HaveCPU != fixture.wantCPUOK || snapshot.Kernel.HaveRSS != fixture.wantRSSOK {
				t.Fatalf("snapshot = %+v, want cpu=%v peak=%d cpu_ok=%v rss_ok=%v", snapshot, fixture.wantCPU, fixture.wantPeak, fixture.wantCPUOK, fixture.wantRSSOK)
			}
		})
	}

	snapshot := Snapshot{Elapsed: 2 * time.Second, Samples: 2, NumCPU: 4, Kernel: Half{CPUUser: 500 * time.Millisecond, HaveCPU: true, RSSBytes: 80, PeakRSSBytes: 100, HaveRSS: true}}
	rendered := map[string]string{
		"report":     snapshot.Report(),
		"prometheus": snapshot.PrometheusText(),
		"jsonl":      string(mustLedgerRow(t, snapshot, time.Date(2026, 8, 16, 12, 0, 0, 0, time.UTC))),
	}
	for name, output := range rendered {
		if strings.TrimSpace(output) == "" {
			t.Fatalf("%s renderer returned no output", name)
		}
	}
	if !strings.Contains(rendered["report"], "25% avg") || !strings.Contains(rendered["prometheus"], "fak_harness_cpu_seconds_total{half=\"kernel\",mode=\"user\"} 0.5") {
		t.Fatalf("renderers omitted folded CPU: report=%q prometheus=%q", rendered["report"], rendered["prometheus"])
	}
}

func cpuPct(snapshot Snapshot) float64 {
	pct, _ := snapshot.Kernel.CPUPercentAvg(snapshot.Elapsed)
	return pct
}

func mustLedgerRow(t *testing.T, snapshot Snapshot, now time.Time) []byte {
	t.Helper()
	row, err := snapshot.MarshalLedgerRow("offline", "test", "qa", now)
	if err != nil {
		t.Fatal(err)
	}
	return row
}

func TestLedgerSmoke(t *testing.T) {
	ledger := filepath.Join(t.TempDir(), "harness-resources.jsonl")
	snapshot := Snapshot{Elapsed: time.Second, Samples: 1, NumCPU: 1, Kernel: Half{RSSBytes: 64, PeakRSSBytes: 64, HaveRSS: true}}
	encoded := mustLedgerRow(t, snapshot, time.Now())
	if err := os.WriteFile(ledger, append(encoded, '\n'), 0o600); err != nil {
		t.Fatal(err)
	}
	raw, err := os.ReadFile(ledger)
	if err != nil {
		t.Fatal(err)
	}
	var row map[string]any
	if err := json.Unmarshal(raw, &row); err != nil {
		t.Fatalf("ledger row is invalid JSON: %v", err)
	}
	for _, key := range []string{"schema", "ts", "mode", "elapsed_s", "samples", "kernel", "agent"} {
		if _, ok := row[key]; !ok {
			t.Errorf("ledger row missing required key %q: %s", key, raw)
		}
	}
	if row["schema"] != LedgerSchema {
		t.Errorf("schema = %v, want %q", row["schema"], LedgerSchema)
	}
}

func TestSamplerTickOverhead(t *testing.T) {
	sampler := New()
	const samples = 1000
	started := time.Now()
	for range samples {
		sampler.foldProc(readProcSelf(), time.Now(), 1, 0)
	}
	if perSample := time.Since(started) / samples; perSample > 10*time.Millisecond {
		t.Fatalf("sampler tick cost %s exceeds 10ms ceiling", perSample)
	}
}
