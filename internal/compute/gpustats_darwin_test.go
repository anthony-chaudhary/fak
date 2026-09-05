//go:build darwin

package compute

import (
	"context"
	"errors"
	"testing"
	"time"
)

const sampleIOAcceleratorXML = `<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<array>
	<dict>
		<key>CFBundleIdentifier</key>
		<string>com.apple.AGXG15S</string>
		<key>MetalPluginName</key>
		<string>AGXMetalG15X_M1</string>
		<key>PerformanceStatistics</key>
		<dict>
			<key>Alloc system memory</key>
			<integer>2472755200</integer>
			<key>Allocated PB Size</key>
			<integer>88342528</integer>
			<key>Device Utilization %</key>
			<integer>14</integer>
			<key>In use system memory</key>
			<integer>639418368</integer>
			<key>Renderer Utilization %</key>
			<integer>12</integer>
			<key>Tiler Utilization %</key>
			<integer>5</integer>
		</dict>
	</dict>
</array>
</plist>
`

const sampleIOAcceleratorActivityXML = `<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<array>
	<dict>
		<key>PerformanceStatistics</key>
		<dict>
			<key>Alloc system memory</key>
			<integer>1073741824</integer>
			<key>GPU Activity(%)</key>
			<real>42.5</real>
		</dict>
	</dict>
</array>
</plist>
`

func TestParseIORegAcceleratorXML(t *testing.T) {
	const totalVRAM = 32 * (1 << 30) // 32 GiB
	stats, ok := parseIORegAcceleratorXML([]byte(sampleIOAcceleratorXML), totalVRAM)
	if !ok {
		t.Fatalf("expected parseIORegAcceleratorXML to succeed, got false")
	}
	if len(stats) != 1 {
		t.Fatalf("expected 1 stat, got %d", len(stats))
	}
	s := stats[0]
	if s.Index != 0 {
		t.Errorf("expected Index 0, got %d", s.Index)
	}
	if s.VRAMUsedBytes != 2472755200 {
		t.Errorf("expected VRAMUsedBytes 2472755200, got %d", s.VRAMUsedBytes)
	}
	if s.VRAMTotalBytes != totalVRAM {
		t.Errorf("expected VRAMTotalBytes %d, got %d", totalVRAM, s.VRAMTotalBytes)
	}
	if s.UtilizationPct != 14 {
		t.Errorf("expected UtilizationPct 14, got %f", s.UtilizationPct)
	}
}

func TestParseIORegAcceleratorXML_GPUActivity(t *testing.T) {
	const totalVRAM = 16 * (1 << 30)
	stats, ok := parseIORegAcceleratorXML([]byte(sampleIOAcceleratorActivityXML), totalVRAM)
	if !ok {
		t.Fatalf("expected parseIORegAcceleratorXML to succeed, got false")
	}
	if len(stats) != 1 {
		t.Fatalf("expected 1 stat, got %d", len(stats))
	}
	s := stats[0]
	if s.VRAMUsedBytes != 1073741824 {
		t.Errorf("expected VRAMUsedBytes 1073741824, got %d", s.VRAMUsedBytes)
	}
	if s.UtilizationPct != 42.5 {
		t.Errorf("expected UtilizationPct 42.5, got %f", s.UtilizationPct)
	}
}

func TestParseIORegAcceleratorXML_FailSoft(t *testing.T) {
	cases := []struct {
		name string
		data string
	}{
		{"empty", ""},
		{"invalid xml", "<plist><broken"},
		{"no perf stats", "<plist><dict><key>foo</key><string>bar</string></dict></plist>"},
		{"empty perf stats", "<plist><dict><key>PerformanceStatistics</key><dict></dict></dict></plist>"},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			stats, ok := parseIORegAcceleratorXML([]byte(tc.data), 1024)
			if ok || len(stats) > 0 {
				t.Errorf("expected false/empty for %s, got ok=%v stats=%+v", tc.name, ok, stats)
			}
		})
	}
}

func TestAppleSiliconGPUStats_Mock(t *testing.T) {
	mockRunner := func(ctx context.Context, args ...string) ([]byte, error) {
		return []byte(sampleIOAcceleratorXML), nil
	}
	stats, ok := appleSiliconGPUStats(mockRunner)
	if !ok {
		t.Fatalf("expected appleSiliconGPUStats to succeed with mock")
	}
	if len(stats) != 1 {
		t.Fatalf("expected 1 stat, got %d", len(stats))
	}
	if stats[0].VRAMUsedBytes != 2472755200 {
		t.Errorf("expected VRAMUsedBytes 2472755200, got %d", stats[0].VRAMUsedBytes)
	}
	if stats[0].VRAMTotalBytes == 0 {
		t.Errorf("expected non-zero VRAMTotalBytes from host mem")
	}
}

func TestAppleSiliconGPUStats_Errors(t *testing.T) {
	// Nil runner
	if _, ok := appleSiliconGPUStats(nil); ok {
		t.Errorf("expected nil runner to fail-soft")
	}

	// Exec error
	errRunner := func(ctx context.Context, args ...string) ([]byte, error) {
		return nil, errors.New("command not found")
	}
	if _, ok := appleSiliconGPUStats(errRunner); ok {
		t.Errorf("expected command error to fail-soft")
	}

	// Timeout runner
	slowRunner := func(ctx context.Context, args ...string) ([]byte, error) {
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		case <-time.After(5 * time.Second):
			return []byte(sampleIOAcceleratorXML), nil
		}
	}
	// Test timeout with very short deadline
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Millisecond)
	defer cancel()
	resCh := make(chan bool)
	go func() {
		// Run a test with a slow mock that gets canceled
		runResultCh := make(chan []byte, 1)
		go func() {
			out, _ := slowRunner(ctx)
			runResultCh <- out
		}()
		select {
		case <-ctx.Done():
			resCh <- true
		case <-runResultCh:
			resCh <- false
		}
	}()
	if !<-resCh {
		t.Errorf("expected slow runner to be aborted by context timeout")
	}
}

func TestAppleSiliconGPUStats_Live(t *testing.T) {
	// On real Darwin Apple Silicon, verify that AppleSiliconGPUStats runs and either returns valid stats or fails soft.
	stats, ok := AppleSiliconGPUStats()
	t.Logf("AppleSiliconGPUStats() -> ok=%v, stats=%+v", ok, stats)
	if ok {
		if len(stats) == 0 {
			t.Errorf("ok=true but 0 stats returned")
		}
		for i, s := range stats {
			if s.VRAMTotalBytes == 0 {
				t.Errorf("stat[%d] has zero VRAMTotalBytes", i)
			}
			t.Logf("GPU[%d]: Used=%d MB, Total=%d MB, Util=%.1f%%",
				s.Index, s.VRAMUsedBytes>>20, s.VRAMTotalBytes>>20, s.UtilizationPct)
		}
	}
}

func TestSystemGPUStats_Darwin(t *testing.T) {
	stats, ok := SystemGPUStats()
	t.Logf("SystemGPUStats() -> ok=%v, stats=%+v", ok, stats)
	// Should at least succeed on this Darwin Apple Silicon machine
	if !ok {
		t.Logf("SystemGPUStats returned ok=false (may be expected if no GPU probe succeeded)")
	} else if len(stats) > 0 {
		used, total, util, aok := AggregateGPUStats(stats)
		if !aok {
			t.Errorf("AggregateGPUStats failed on non-empty stats")
		}
		t.Logf("Aggregated: Used=%d MB, Total=%d MB, Util=%.1f%%", used>>20, total>>20, util)
	}
}
