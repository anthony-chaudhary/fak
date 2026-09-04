package observability

import (
	"encoding/binary"
	"os"
	"path/filepath"
	"testing"
)

func TestCheckPromptTokenAlarm(t *testing.T) {
	tests := []struct {
		name          string
		turnTokens    int
		baseline      int
		wantSeverity  TelemetryAlarmSeverity
		wantTriggered bool
	}{
		{
			name:          "normal within limits",
			turnTokens:    5000,
			baseline:      5000,
			wantSeverity:  SeverityOK,
			wantTriggered: false,
		},
		{
			name:          "hard cap 30k boundary ok",
			turnTokens:    30000,
			baseline:      20000,
			wantSeverity:  SeverityOK,
			wantTriggered: false,
		},
		{
			name:          "hard cap 30k exceeded",
			turnTokens:    30001,
			baseline:      25000,
			wantSeverity:  SeverityWarn,
			wantTriggered: true,
		},
		{
			name:          "prompt doubling > 2x baseline",
			turnTokens:    12001,
			baseline:      6000,
			wantSeverity:  SeverityWarn,
			wantTriggered: true,
		},
		{
			name:          "exactly 2x baseline",
			turnTokens:    12000,
			baseline:      6000,
			wantSeverity:  SeverityOK,
			wantTriggered: false,
		},
		{
			name:          "zero baseline with normal tokens",
			turnTokens:    10000,
			baseline:      0,
			wantSeverity:  SeverityOK,
			wantTriggered: false,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			alarm := CheckPromptTokenAlarm(tc.turnTokens, tc.baseline)
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

func TestCheckLatencyAlarm(t *testing.T) {
	tests := []struct {
		name          string
		current       float64
		median        float64
		spikes        int
		wantSeverity  TelemetryAlarmSeverity
		wantTriggered bool
	}{
		{
			name:          "normal latency",
			current:       2.0,
			median:        2.0,
			spikes:        0,
			wantSeverity:  SeverityOK,
			wantTriggered: false,
		},
		{
			name:          "15s hard cap boundary ok",
			current:       15.0,
			median:        10.0,
			spikes:        0,
			wantSeverity:  SeverityOK,
			wantTriggered: false,
		},
		{
			name:          "15s hard cap exceeded",
			current:       15.1,
			median:        10.0,
			spikes:        1,
			wantSeverity:  SeverityWarn,
			wantTriggered: true,
		},
		{
			name:          "latency spike > 2.5x median",
			current:       5.5,
			median:        2.0,
			spikes:        1,
			wantSeverity:  SeverityWarn,
			wantTriggered: true,
		},
		{
			name:          "exactly 2.5x median",
			current:       5.0,
			median:        2.0,
			spikes:        0,
			wantSeverity:  SeverityOK,
			wantTriggered: false,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			alarm := CheckLatencyAlarm(tc.current, tc.median, tc.spikes)
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

func TestCheckDatabaseBloatAlarm(t *testing.T) {
	const oneGB int64 = 1 * 1024 * 1024 * 1024

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
			name:          "normal small database",
			dbBytes:       10 * 1024 * 1024,
			freelistPages: 5,
			pageCount:     2560,
			pageSize:      4096,
			wantSeverity:  SeverityOK,
			wantTriggered: false,
		},
		{
			name:          "1GB hard cap exceeded",
			dbBytes:       oneGB + 1,
			freelistPages: 0,
			pageCount:     250000,
			pageSize:      4096,
			wantSeverity:  SeverityWarn,
			wantTriggered: true,
		},
		{
			name:          "large freelist ratio > 25%",
			dbBytes:       100 * 1024 * 1024,
			freelistPages: 26,
			pageCount:     100,
			pageSize:      4096,
			wantSeverity:  SeverityWarn,
			wantTriggered: true,
		},
		{
			name:          "freelist ratio exactly 25%",
			dbBytes:       100 * 1024 * 1024,
			freelistPages: 25,
			pageCount:     100,
			pageSize:      4096,
			wantSeverity:  SeverityOK,
			wantTriggered: false,
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

func makeTestSQLiteDB(t *testing.T, totalBytes int, pageSize uint16, freelist uint32) string {
	t.Helper()
	dir := t.TempDir()
	p := filepath.Join(dir, "obs_test.db")
	data := make([]byte, totalBytes)
	copy(data[0:16], "SQLite format 3\x00")
	binary.BigEndian.PutUint16(data[16:18], pageSize)
	binary.BigEndian.PutUint32(data[36:40], freelist)
	if err := os.WriteFile(p, data, 0644); err != nil {
		t.Fatalf("failed to write test sqlite file: %v", err)
	}
	return p
}

func TestInspectSQLiteFileHeader(t *testing.T) {
	p := makeTestSQLiteDB(t, 8192, 4096, 1)
	dbBytes, freelist, pages, pageSize, err := InspectSQLiteFileHeader(p)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if dbBytes != 8192 || freelist != 1 || pages != 2 || pageSize != 4096 {
		t.Fatalf("unexpected values: bytes=%d, freelist=%d, pages=%d, pageSize=%d", dbBytes, freelist, pages, pageSize)
	}
}

func TestEvaluateTelemetryHealth(t *testing.T) {
	p := makeTestSQLiteDB(t, 4096, 4096, 0)
	rep := EvaluateTelemetryHealth(1000, 1000, []float64{1.0, 1.2, 1.5}, p)
	if !rep.OK || rep.Findings != 0 {
		t.Fatalf("expected OK report, got %+v", rep)
	}

	// Breach evaluation
	repBreach := EvaluateTelemetryHealth(35000, 10000, []float64{1.0, 18.0}, p)
	if repBreach.OK || repBreach.Findings < 2 {
		t.Fatalf("expected multiple findings on breach, got %+v", repBreach)
	}
}
