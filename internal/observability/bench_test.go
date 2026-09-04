package observability

import (
	"encoding/binary"
	"os"
	"path/filepath"
	"testing"
)

// Invariant: telemetry benchmarks provide allocation and throughput baselines without external I/O.

func makeBenchmarkDB(b *testing.B, totalBytes int, pageSize uint16, freelist uint32) string {
	b.Helper()
	dir := b.TempDir()
	p := filepath.Join(dir, "bench_test.db")
	data := make([]byte, totalBytes)
	copy(data[0:16], "SQLite format 3\x00")
	binary.BigEndian.PutUint16(data[16:18], pageSize)
	binary.BigEndian.PutUint32(data[36:40], freelist)
	if err := os.WriteFile(p, data, 0600); err != nil {
		b.Fatalf("failed to write benchmark sqlite file: %v", err)
	}
	return p
}

func makeTestDB(t *testing.T, totalBytes int, pageSize uint16, freelist uint32) string {
	t.Helper()
	dir := t.TempDir()
	p := filepath.Join(dir, "test_bench_smoke.db")
	data := make([]byte, totalBytes)
	copy(data[0:16], "SQLite format 3\x00")
	binary.BigEndian.PutUint16(data[16:18], pageSize)
	binary.BigEndian.PutUint32(data[36:40], freelist)
	if err := os.WriteFile(p, data, 0600); err != nil {
		t.Fatalf("failed to write test sqlite file: %v", err)
	}
	return p
}

// TestObservabilitySmoke verifies core telemetry health evaluation invariants across safe and alarm states.
func TestObservabilitySmoke(t *testing.T) {
	dbPath := makeTestDB(t, 4096, 4096, 0)

	// Clean baseline report
	repClean := EvaluateTelemetryHealth(1500, 1500, []float64{0.5, 0.8, 1.1}, dbPath)
	if !repClean.OK {
		t.Fatalf("expected clean telemetry report, got ok=false findings=%d", repClean.Findings)
	}
	if len(repClean.Alarms) != 3 {
		t.Fatalf("expected 3 alarm evaluations, got %d", len(repClean.Alarms))
	}

	// Triggered alarm report
	repAlarm := EvaluateTelemetryHealth(45000, 10000, []float64{1.0, 20.0}, dbPath)
	if repAlarm.OK {
		t.Fatalf("expected alarm report to fail OK gate, got ok=true")
	}
	if repAlarm.Findings < 2 {
		t.Fatalf("expected at least 2 findings, got %d", repAlarm.Findings)
	}
}

// TestTelemetryInvariants verifies that missing or corrupt databases fail-closed safely.
func TestTelemetryInvariants(t *testing.T) {
	repMissing := EvaluateTelemetryHealth(1000, 1000, []float64{1.0}, filepath.Join(t.TempDir(), "nonexistent.db"))
	if repMissing.OK {
		t.Fatalf("expected missing db to trigger warning finding, got ok=true")
	}
	if repMissing.DatabaseAlarm.Severity != SeverityWarn {
		t.Fatalf("expected database alarm SeverityWarn on missing file, got %v", repMissing.DatabaseAlarm.Severity)
	}
	if !repMissing.DatabaseAlarm.Triggered {
		t.Fatalf("expected database alarm to be triggered on missing file")
	}
}

// BenchmarkCheckPromptTokenAlarm measures prompt token threshold evaluation throughput.
func BenchmarkCheckPromptTokenAlarm(b *testing.B) {
	b.Run("Normal", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			alarm := CheckPromptTokenAlarm(5000, 5000)
			if alarm.Triggered {
				b.Fatal("unexpected alarm trigger")
			}
		}
	})

	b.Run("BreachHardCap", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			alarm := CheckPromptTokenAlarm(35000, 10000)
			if !alarm.Triggered {
				b.Fatal("expected alarm trigger")
			}
		}
	})
}

// BenchmarkCheckLatencyAlarm measures turn latency spike evaluation throughput.
func BenchmarkCheckLatencyAlarm(b *testing.B) {
	b.Run("Normal", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			alarm := CheckLatencyAlarm(2.0, 2.0, 0)
			if alarm.Triggered {
				b.Fatal("unexpected alarm trigger")
			}
		}
	})

	b.Run("Spike", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			alarm := CheckLatencyAlarm(16.5, 3.0, 2)
			if !alarm.Triggered {
				b.Fatal("expected alarm trigger")
			}
		}
	})
}

// BenchmarkCheckDatabaseBloatAlarm measures database size and freelist ratio evaluation throughput.
func BenchmarkCheckDatabaseBloatAlarm(b *testing.B) {
	b.Run("Normal", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			alarm := CheckDatabaseBloatAlarm(10*1024*1024, 5, 2560, 4096)
			if alarm.Triggered {
				b.Fatal("unexpected alarm trigger")
			}
		}
	})

	b.Run("Bloat", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			alarm := CheckDatabaseBloatAlarm(2*1024*1024*1024, 500, 1000, 4096)
			if !alarm.Triggered {
				b.Fatal("expected alarm trigger")
			}
		}
	})
}

// BenchmarkInspectSQLiteFileHeader measures pure Go SQLite header inspection performance.
func BenchmarkInspectSQLiteFileHeader(b *testing.B) {
	p := makeBenchmarkDB(b, 8192, 4096, 2)
	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		dbBytes, freelist, pageCount, pageSize, err := InspectSQLiteFileHeader(p)
		if err != nil || dbBytes == 0 || freelist == 0 || pageCount == 0 || pageSize == 0 {
			b.Fatalf("inspection failed: %v", err)
		}
	}
}

// BenchmarkEvaluateTelemetryHealth measures full telemetry health evaluation across all telemetry axes.
func BenchmarkEvaluateTelemetryHealth(b *testing.B) {
	p := makeBenchmarkDB(b, 4096, 4096, 0)
	latencies := []float64{1.0, 1.2, 0.9, 1.5, 1.1, 1.3, 0.8, 1.4}
	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		rep := EvaluateTelemetryHealth(2000, 2000, latencies, p)
		if !rep.OK {
			b.Fatalf("expected healthy report: %+v", rep)
		}
	}
}
