package trajectory

import (
	"encoding/binary"
	"os"
	"path/filepath"
	"testing"
)

func TestTelemetryAlarm_CheckPromptTokenAlarm(t *testing.T) {
	tests := []struct {
		name           string
		turn1Tokens    int
		baselineTokens int
		wantSeverity   TelemetryAlarmSeverity
		wantTriggered  bool
	}{
		{
			name:           "normal within bounds",
			turn1Tokens:    1000,
			baselineTokens: 1000,
			wantSeverity:   SeverityOK,
			wantTriggered:  false,
		},
		{
			name:           "exactly hard cap 25000",
			turn1Tokens:    25000,
			baselineTokens: 0,
			wantSeverity:   SeverityOK,
			wantTriggered:  false,
		},
		{
			name:           "hard cap breach 25001",
			turn1Tokens:    25001,
			baselineTokens: 0,
			wantSeverity:   SeverityWarn,
			wantTriggered:  true,
		},
		{
			name:           "doubling breach turn1 > 2.0*baseline",
			turn1Tokens:    2001,
			baselineTokens: 1000,
			wantSeverity:   SeverityWarn,
			wantTriggered:  true,
		},
		{
			name:           "exactly 2.0x baseline",
			turn1Tokens:    2000,
			baselineTokens: 1000,
			wantSeverity:   SeverityOK,
			wantTriggered:  false,
		},
		{
			name:           "zero baseline with normal tokens",
			turn1Tokens:    5000,
			baselineTokens: 0,
			wantSeverity:   SeverityOK,
			wantTriggered:  false,
		},
		{
			name:           "negative baseline with normal tokens",
			turn1Tokens:    5000,
			baselineTokens: -100,
			wantSeverity:   SeverityOK,
			wantTriggered:  false,
		},
		{
			name:           "both hard cap and doubling breach",
			turn1Tokens:    30000,
			baselineTokens: 10000,
			wantSeverity:   SeverityWarn,
			wantTriggered:  true,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			alarm := CheckPromptTokenAlarm(tc.turn1Tokens, tc.baselineTokens)
			if alarm.Type != AlarmPromptDoubling {
				t.Fatalf("expected Type %v, got %v", AlarmPromptDoubling, alarm.Type)
			}
			if alarm.Severity != tc.wantSeverity {
				t.Fatalf("expected Severity %v, got %v", tc.wantSeverity, alarm.Severity)
			}
			if alarm.Triggered != tc.wantTriggered {
				t.Fatalf("expected Triggered %v, got %v", tc.wantTriggered, alarm.Triggered)
			}
		})
	}
}

func TestTelemetryAlarm_CheckLatencyAlarm(t *testing.T) {
	tests := []struct {
		name              string
		currentLatency    float64
		medianLatency     float64
		consecutiveSpikes int
		wantSeverity      TelemetryAlarmSeverity
		wantTriggered     bool
	}{
		{
			name:              "normal within bounds",
			currentLatency:    2.0,
			medianLatency:     2.0,
			consecutiveSpikes: 0,
			wantSeverity:      SeverityOK,
			wantTriggered:     false,
		},
		{
			name:              "exactly hard cap 15.0s",
			currentLatency:    15.0,
			medianLatency:     10.0,
			consecutiveSpikes: 0,
			wantSeverity:      SeverityOK,
			wantTriggered:     false,
		},
		{
			name:              "hard cap breach 15.1s",
			currentLatency:    15.1,
			medianLatency:     10.0,
			consecutiveSpikes: 1,
			wantSeverity:      SeverityWarn,
			wantTriggered:     true,
		},
		{
			name:              "spike over 2.5x median",
			currentLatency:    5.1,
			medianLatency:     2.0,
			consecutiveSpikes: 1,
			wantSeverity:      SeverityWarn,
			wantTriggered:     true,
		},
		{
			name:              "exactly 2.5x median",
			currentLatency:    5.0,
			medianLatency:     2.0,
			consecutiveSpikes: 0,
			wantSeverity:      SeverityOK,
			wantTriggered:     false,
		},
		{
			name:              "zero median normal latency",
			currentLatency:    10.0,
			medianLatency:     0.0,
			consecutiveSpikes: 0,
			wantSeverity:      SeverityOK,
			wantTriggered:     false,
		},
		{
			name:              "both hard cap and 2.5x median breach",
			currentLatency:    20.0,
			medianLatency:     4.0,
			consecutiveSpikes: 2,
			wantSeverity:      SeverityWarn,
			wantTriggered:     true,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			alarm := CheckLatencyAlarm(tc.currentLatency, tc.medianLatency, tc.consecutiveSpikes)
			if alarm.Type != AlarmLatencySpike {
				t.Fatalf("expected Type %v, got %v", AlarmLatencySpike, alarm.Type)
			}
			if alarm.Severity != tc.wantSeverity {
				t.Fatalf("expected Severity %v, got %v", tc.wantSeverity, alarm.Severity)
			}
			if alarm.Triggered != tc.wantTriggered {
				t.Fatalf("expected Triggered %v, got %v", tc.wantTriggered, alarm.Triggered)
			}
		})
	}
}

func TestTelemetryAlarm_CheckDatabaseBloatAlarm(t *testing.T) {
	const fiveGB = 5 * 1024 * 1024 * 1024

	tests := []struct {
		name          string
		dbBytes       int64
		freelistPages int64
		pageCount     int64
		pageSize      int64
		wantSeverity  TelemetryAlarmSeverity
		wantTriggered bool
	}{
		{
			name:          "normal healthy database",
			dbBytes:       1024 * 1024,
			freelistPages: 5,
			pageCount:     256,
			pageSize:      4096,
			wantSeverity:  SeverityOK,
			wantTriggered: false,
		},
		{
			name:          "exactly 5GB",
			dbBytes:       fiveGB,
			freelistPages: 0,
			pageCount:     1000,
			pageSize:      4096,
			wantSeverity:  SeverityOK,
			wantTriggered: false,
		},
		{
			name:          "5GB breach",
			dbBytes:       fiveGB + 1,
			freelistPages: 0,
			pageCount:     1000,
			pageSize:      4096,
			wantSeverity:  SeverityWarn,
			wantTriggered: true,
		},
		{
			name:          "freelist ratio breach > 50%",
			dbBytes:       1024 * 1024,
			freelistPages: 51,
			pageCount:     100,
			pageSize:      4096,
			wantSeverity:  SeverityWarn,
			wantTriggered: true,
		},
		{
			name:          "exactly 50% freelist ratio",
			dbBytes:       1024 * 1024,
			freelistPages: 50,
			pageCount:     100,
			pageSize:      4096,
			wantSeverity:  SeverityOK,
			wantTriggered: false,
		},
		{
			name:          "zero page count",
			dbBytes:       0,
			freelistPages: 0,
			pageCount:     0,
			pageSize:      0,
			wantSeverity:  SeverityOK,
			wantTriggered: false,
		},
		{
			name:          "both 5GB and freelist ratio breach",
			dbBytes:       6 * 1024 * 1024 * 1024,
			freelistPages: 80,
			pageCount:     100,
			pageSize:      4096,
			wantSeverity:  SeverityWarn,
			wantTriggered: true,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			alarm := CheckDatabaseBloatAlarm(tc.dbBytes, tc.freelistPages, tc.pageCount, tc.pageSize)
			if alarm.Type != AlarmDatabaseBloat {
				t.Fatalf("expected Type %v, got %v", AlarmDatabaseBloat, alarm.Type)
			}
			if alarm.Severity != tc.wantSeverity {
				t.Fatalf("expected Severity %v, got %v", tc.wantSeverity, alarm.Severity)
			}
			if alarm.Triggered != tc.wantTriggered {
				t.Fatalf("expected Triggered %v, got %v", tc.wantTriggered, alarm.Triggered)
			}
		})
	}
}

func makeSyntheticSQLiteDB(t *testing.T, totalBytes int, pageSize uint16, freelistPages uint32) string {
	t.Helper()
	dir := t.TempDir()
	p := filepath.Join(dir, "test.db")

	data := make([]byte, totalBytes)
	copy(data[0:16], "SQLite format 3\x00")
	binary.BigEndian.PutUint16(data[16:18], pageSize)
	binary.BigEndian.PutUint32(data[36:40], freelistPages)

	if err := os.WriteFile(p, data, 0644); err != nil {
		t.Fatalf("failed to write synthetic sqlite file: %v", err)
	}
	return p
}

func TestTelemetryAlarm_InspectSQLiteFileHeader(t *testing.T) {
	t.Run("standard 4096 page size", func(t *testing.T) {
		p := makeSyntheticSQLiteDB(t, 8192, 4096, 1)
		dbBytes, freelist, pages, pageSize, err := InspectSQLiteFileHeader(p)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if dbBytes != 8192 {
			t.Errorf("dbBytes = %d, want 8192", dbBytes)
		}
		if freelist != 1 {
			t.Errorf("freelistPages = %d, want 1", freelist)
		}
		if pages != 2 {
			t.Errorf("pageCount = %d, want 2", pages)
		}
		if pageSize != 4096 {
			t.Errorf("pageSize = %d, want 4096", pageSize)
		}
	})

	t.Run("page size 1 represents 65536", func(t *testing.T) {
		p := makeSyntheticSQLiteDB(t, 131072, 1, 0)
		dbBytes, freelist, pages, pageSize, err := InspectSQLiteFileHeader(p)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if dbBytes != 131072 {
			t.Errorf("dbBytes = %d, want 131072", dbBytes)
		}
		if freelist != 0 {
			t.Errorf("freelistPages = %d, want 0", freelist)
		}
		if pages != 2 {
			t.Errorf("pageCount = %d, want 2", pages)
		}
		if pageSize != 65536 {
			t.Errorf("pageSize = %d, want 65536", pageSize)
		}
	})

	t.Run("empty path", func(t *testing.T) {
		_, _, _, _, err := InspectSQLiteFileHeader("")
		if err == nil {
			t.Fatal("expected error for empty path, got nil")
		}
	})

	t.Run("non-existent file", func(t *testing.T) {
		_, _, _, _, err := InspectSQLiteFileHeader(filepath.Join(t.TempDir(), "nonexistent.db"))
		if err == nil {
			t.Fatal("expected error for non-existent file, got nil")
		}
	})

	t.Run("directory path", func(t *testing.T) {
		_, _, _, _, err := InspectSQLiteFileHeader(t.TempDir())
		if err == nil {
			t.Fatal("expected error for directory path, got nil")
		}
	})

	t.Run("file too small", func(t *testing.T) {
		p := filepath.Join(t.TempDir(), "toosmall.db")
		if err := os.WriteFile(p, []byte("SQLite format 3\x00"), 0644); err != nil {
			t.Fatal(err)
		}
		_, _, _, _, err := InspectSQLiteFileHeader(p)
		if err == nil {
			t.Fatal("expected error for file < 100 bytes, got nil")
		}
	})

	t.Run("magic mismatch", func(t *testing.T) {
		p := filepath.Join(t.TempDir(), "badmagic.db")
		data := make([]byte, 100)
		copy(data, "Not SQLite at all")
		if err := os.WriteFile(p, data, 0644); err != nil {
			t.Fatal(err)
		}
		_, _, _, _, err := InspectSQLiteFileHeader(p)
		if err == nil {
			t.Fatal("expected error for bad magic string, got nil")
		}
	})

	t.Run("zero page size", func(t *testing.T) {
		p := makeSyntheticSQLiteDB(t, 1024, 0, 0)
		_, _, _, _, err := InspectSQLiteFileHeader(p)
		if err == nil {
			t.Fatal("expected error for zero page size, got nil")
		}
	})
}

func TestTelemetryAlarm_EvaluateTelemetryHealth(t *testing.T) {
	t.Run("all healthy", func(t *testing.T) {
		p := makeSyntheticSQLiteDB(t, 4096, 4096, 0)
		rep := EvaluateTelemetryHealth(1000, 1000, []float64{1.0, 1.2, 1.1}, p)
		if !rep.OK {
			t.Errorf("expected OK == true, got false (findings: %d)", rep.Findings)
		}
		if rep.Findings != 0 {
			t.Errorf("expected 0 findings, got %d", rep.Findings)
		}
		if rep.PromptAlarm.Severity != SeverityOK {
			t.Errorf("prompt alarm severity = %v, want OK", rep.PromptAlarm.Severity)
		}
		if rep.LatencyAlarm.Severity != SeverityOK {
			t.Errorf("latency alarm severity = %v, want OK", rep.LatencyAlarm.Severity)
		}
		if rep.DatabaseAlarm.Severity != SeverityOK {
			t.Errorf("database alarm severity = %v, want OK", rep.DatabaseAlarm.Severity)
		}
	})

	t.Run("prompt doubling breach", func(t *testing.T) {
		rep := EvaluateTelemetryHealth(26000, 1000, []float64{1.0}, "")
		if rep.OK {
			t.Error("expected OK == false")
		}
		if rep.Findings != 1 {
			t.Errorf("expected 1 finding, got %d", rep.Findings)
		}
		if rep.PromptAlarm.Severity != SeverityWarn {
			t.Errorf("expected PromptAlarm.Severity == WARN, got %v", rep.PromptAlarm.Severity)
		}
	})

	t.Run("latency spike breach", func(t *testing.T) {
		rep := EvaluateTelemetryHealth(1000, 1000, []float64{1.0, 1.0, 10.0}, "")
		if rep.OK {
			t.Error("expected OK == false")
		}
		if rep.Findings != 1 {
			t.Errorf("expected 1 finding, got %d", rep.Findings)
		}
		if rep.LatencyAlarm.Severity != SeverityWarn {
			t.Errorf("expected LatencyAlarm.Severity == WARN, got %v", rep.LatencyAlarm.Severity)
		}
	})

	t.Run("database bloat breach", func(t *testing.T) {
		p := makeSyntheticSQLiteDB(t, 4096*10, 4096, 6) // 6/10 = 60% freelist
		rep := EvaluateTelemetryHealth(1000, 1000, []float64{1.0}, p)
		if rep.OK {
			t.Error("expected OK == false")
		}
		if rep.Findings != 1 {
			t.Errorf("expected 1 finding, got %d", rep.Findings)
		}
		if rep.DatabaseAlarm.Severity != SeverityWarn {
			t.Errorf("expected DatabaseAlarm.Severity == WARN, got %v", rep.DatabaseAlarm.Severity)
		}
	})

	t.Run("missing database path triggers warning", func(t *testing.T) {
		rep := EvaluateTelemetryHealth(1000, 1000, nil, filepath.Join(t.TempDir(), "missing.db"))
		if rep.OK {
			t.Error("expected OK == false for missing db")
		}
		if rep.DatabaseAlarm.Severity != SeverityWarn {
			t.Errorf("expected DatabaseAlarm.Severity == WARN, got %v", rep.DatabaseAlarm.Severity)
		}
		if rep.DBError == "" {
			t.Error("expected non-empty DBError")
		}
	})

	t.Run("empty inputs are healthy", func(t *testing.T) {
		rep := EvaluateTelemetryHealth(0, 0, nil, "")
		if !rep.OK {
			t.Errorf("expected OK == true, got false (findings: %d)", rep.Findings)
		}
		if rep.Findings != 0 {
			t.Errorf("expected 0 findings, got %d", rep.Findings)
		}
	})
}
