package main

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/trajectory"
)

func createTestSQLiteFile(t *testing.T, size int, pageSize uint16, freelist uint32) string {
	t.Helper()
	dir := t.TempDir()
	p := filepath.Join(dir, "telemetry_test.db")
	buf := make([]byte, size)
	copy(buf[0:16], "SQLite format 3\x00")
	binary.BigEndian.PutUint16(buf[16:18], pageSize)
	binary.BigEndian.PutUint32(buf[36:40], freelist)
	if err := os.WriteFile(p, buf, 0644); err != nil {
		t.Fatalf("failed to write test sqlite file: %v", err)
	}
	return p
}

func TestDoctorTelemetry_JSON_Healthy(t *testing.T) {
	dbPath := createTestSQLiteFile(t, 4096, 4096, 0)
	var stdout, stderr bytes.Buffer
	argv := []string{
		"telemetry",
		"--db", dbPath,
		"--prompt-tokens", "1000",
		"--baseline-tokens", "1000",
		"--latency", "1.5",
		"--json",
	}

	rc := runDoctor(strings.NewReader(""), &stdout, &stderr, argv)
	if rc != 0 {
		t.Fatalf("expected exit code 0, got %d, stderr: %s", rc, stderr.String())
	}

	var rep trajectory.TelemetryHealthReport
	if err := json.Unmarshal(stdout.Bytes(), &rep); err != nil {
		t.Fatalf("failed to unmarshal JSON output: %v, raw: %s", err, stdout.String())
	}

	if !rep.OK {
		t.Errorf("expected rep.OK == true, got false (findings: %d)", rep.Findings)
	}
	if rep.Findings != 0 {
		t.Errorf("expected 0 findings, got %d", rep.Findings)
	}
	if rep.PromptAlarm.Severity != trajectory.SeverityOK {
		t.Errorf("expected prompt alarm OK, got %v", rep.PromptAlarm.Severity)
	}
	if rep.LatencyAlarm.Severity != trajectory.SeverityOK {
		t.Errorf("expected latency alarm OK, got %v", rep.LatencyAlarm.Severity)
	}
	if rep.DatabaseAlarm.Severity != trajectory.SeverityOK {
		t.Errorf("expected database alarm OK, got %v", rep.DatabaseAlarm.Severity)
	}
}

func TestDoctorTelemetry_HealthSubcommand_Warning(t *testing.T) {
	var stdout, stderr bytes.Buffer
	argv := []string{
		"health",
		"--prompt-tokens", "30000",
		"--baseline-tokens", "1000",
		"--latency", "20.0",
		"--json",
	}

	rc := runDoctor(strings.NewReader(""), &stdout, &stderr, argv)
	if rc != 1 {
		t.Fatalf("expected exit code 1, got %d, stderr: %s", rc, stderr.String())
	}

	var rep trajectory.TelemetryHealthReport
	if err := json.Unmarshal(stdout.Bytes(), &rep); err != nil {
		t.Fatalf("failed to unmarshal JSON output: %v, raw: %s", err, stdout.String())
	}

	if rep.OK {
		t.Errorf("expected rep.OK == false, got true")
	}
	if rep.Findings < 2 {
		t.Errorf("expected at least 2 findings, got %d", rep.Findings)
	}
	if rep.PromptAlarm.Severity != trajectory.SeverityWarn {
		t.Errorf("expected prompt alarm WARN, got %v", rep.PromptAlarm.Severity)
	}
	if rep.LatencyAlarm.Severity != trajectory.SeverityWarn {
		t.Errorf("expected latency alarm WARN, got %v", rep.LatencyAlarm.Severity)
	}
}

func TestDoctorTelemetry_Human_Healthy(t *testing.T) {
	dbPath := createTestSQLiteFile(t, 4096, 4096, 0)
	var stdout, stderr bytes.Buffer
	argv := []string{
		"telemetry",
		"--db", dbPath,
		"--prompt-tokens", "500",
		"--baseline-tokens", "500",
		"--latency", "0.8",
	}

	rc := runDoctor(strings.NewReader(""), &stdout, &stderr, argv)
	if rc != 0 {
		t.Fatalf("expected exit code 0, got %d, stderr: %s", rc, stderr.String())
	}

	out := stdout.String()
	if !strings.Contains(out, "== fak doctor: telemetry health ==") {
		t.Errorf("missing header in output: %s", out)
	}
	if !strings.Contains(out, "doctor: healthy (0 findings)") {
		t.Errorf("missing healthy line in output: %s", out)
	}
	if !strings.Contains(out, "[OK  ] PROMPT_DOUBLING") {
		t.Errorf("missing OK PROMPT_DOUBLING in output: %s", out)
	}
	if strings.Contains(out, "[WARN]") {
		t.Errorf("unexpected WARN in healthy output: %s", out)
	}
}

func TestDoctorTelemetry_Human_Warning(t *testing.T) {
	dbPath := createTestSQLiteFile(t, 4096, 4096, 0)
	var stdout, stderr bytes.Buffer
	argv := []string{
		"telemetry",
		"--db", dbPath,
		"--prompt-tokens", "26000",
	}

	rc := runDoctor(strings.NewReader(""), &stdout, &stderr, argv)
	if rc != 1 {
		t.Fatalf("expected exit code 1, got %d, stderr: %s", rc, stderr.String())
	}

	out := stdout.String()
	if !strings.Contains(out, "== fak doctor: telemetry health ==") {
		t.Errorf("missing header in output: %s", out)
	}
	if !strings.Contains(out, "[WARN] PROMPT_DOUBLING") {
		t.Errorf("missing WARN PROMPT_DOUBLING in output: %s", out)
	}
	if !strings.Contains(out, "doctor: 1 finding(s)") {
		t.Errorf("missing 1 finding(s) in output: %s", out)
	}
}

func TestDoctorTelemetry_AutoDiscover(t *testing.T) {
	var stdout, stderr bytes.Buffer
	argv := []string{
		"telemetry",
		"--prompt-tokens", "100",
		"--json",
	}

	_ = runDoctor(strings.NewReader(""), &stdout, &stderr, argv)
	var rep trajectory.TelemetryHealthReport
	if err := json.Unmarshal(stdout.Bytes(), &rep); err != nil {
		t.Fatalf("failed to unmarshal JSON: %v", err)
	}
	// If opencode.db exists on the host, DBPath will be discovered
	if rep.DBPath != "" {
		if !strings.HasSuffix(rep.DBPath, "opencode.db") {
			t.Errorf("unexpected discovered DBPath: %s", rep.DBPath)
		}
	}
}

func TestDoctorTelemetry_DatabaseBloat(t *testing.T) {
	dbPath := createTestSQLiteFile(t, 4096*10, 4096, 6) // 60% freelist
	var stdout, stderr bytes.Buffer
	argv := []string{
		"telemetry",
		"--db", dbPath,
		"--json",
	}

	rc := runDoctor(strings.NewReader(""), &stdout, &stderr, argv)
	if rc != 1 {
		t.Fatalf("expected exit code 1, got %d, stderr: %s", rc, stderr.String())
	}

	var rep trajectory.TelemetryHealthReport
	if err := json.Unmarshal(stdout.Bytes(), &rep); err != nil {
		t.Fatalf("failed to unmarshal JSON: %v", err)
	}

	if rep.DatabaseAlarm.Severity != trajectory.SeverityWarn {
		t.Errorf("expected database alarm WARN, got %v", rep.DatabaseAlarm.Severity)
	}
	if rep.Findings != 1 {
		t.Errorf("expected 1 finding, got %d", rep.Findings)
	}
}

func TestDoctorTelemetry_LatencySpikeList(t *testing.T) {
	var stdout, stderr bytes.Buffer
	argv := []string{
		"telemetry",
		"--latency", "1.0,1.2,16.5",
		"--json",
	}

	rc := runDoctor(strings.NewReader(""), &stdout, &stderr, argv)
	if rc != 1 {
		t.Fatalf("expected exit code 1, got %d, stderr: %s", rc, stderr.String())
	}

	var rep trajectory.TelemetryHealthReport
	if err := json.Unmarshal(stdout.Bytes(), &rep); err != nil {
		t.Fatalf("failed to unmarshal JSON: %v", err)
	}

	if rep.LatencyAlarm.Severity != trajectory.SeverityWarn {
		t.Errorf("expected latency alarm WARN, got %v", rep.LatencyAlarm.Severity)
	}
}

func TestDoctorTelemetry_UsageError(t *testing.T) {
	var stdout, stderr bytes.Buffer
	argv := []string{
		"telemetry",
		"--nonexistent-flag",
	}

	rc := runDoctor(strings.NewReader(""), &stdout, &stderr, argv)
	if rc != 2 {
		t.Fatalf("expected exit code 2 for bad flag, got %d", rc)
	}
}
