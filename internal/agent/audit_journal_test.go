package agent

import (
	"context"
	"os"
	"path/filepath"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/journal"
)

func TestAuditJournalRawArm(t *testing.T) {
	dir := t.TempDir()
	logPath := filepath.Join(dir, "raw-audit.jsonl")
	j, err := journal.Open(logPath)
	if err != nil {
		t.Fatalf("journal.Open: %v", err)
	}

	p := NewMockPlanner("test")
	m, err := RunArm(context.Background(), p, DefaultTask, false, 10, nil, WithAuditJournal(j))
	if err != nil {
		t.Fatalf("RunArm raw: %v", err)
	}
	if err := j.Close(); err != nil {
		t.Fatalf("journal.Close: %v", err)
	}

	if m.ToolCalls == 0 {
		t.Fatal("expected raw arm to make tool calls")
	}

	n, err := journal.Verify(logPath)
	if err != nil {
		t.Fatalf("journal.Verify: %v", err)
	}
	if n == 0 {
		t.Fatal("expected journal to contain rows")
	}

	rows, _, err := journal.ReadRowsFrom(logPath, 0)
	if err != nil {
		t.Fatalf("ReadRowsFrom: %v", err)
	}
	for _, r := range rows {
		if r.By != "raw-harness" {
			t.Errorf("expected By=raw-harness, got %q", r.By)
		}
		if r.ArgsDigest == "" {
			t.Error("expected non-empty ArgsDigest")
		}
		if r.ResultDigest == "" {
			t.Error("expected non-empty ResultDigest")
		}
	}

	// Verify tampering detection
	data, err := os.ReadFile(logPath)
	if err != nil {
		t.Fatalf("ReadFile: %v", err)
	}
	tampered := append([]byte(nil), data...)
	for i := range tampered {
		if tampered[i] != '\n' {
			tampered[i] ^= 0x01
			break
		}
	}
	tamperedPath := filepath.Join(dir, "raw-tampered.jsonl")
	if err := os.WriteFile(tamperedPath, tampered, 0o644); err != nil {
		t.Fatalf("WriteFile: %v", err)
	}
	if _, err := journal.Verify(tamperedPath); err == nil {
		t.Fatal("expected journal.Verify to fail on tampered file")
	}
}

func TestAuditJournalMediatedArm(t *testing.T) {
	dir := t.TempDir()
	logPath := filepath.Join(dir, "mediated-audit.jsonl")
	j, err := journal.Open(logPath)
	if err != nil {
		t.Fatalf("journal.Open: %v", err)
	}

	p := NewMockPlanner("test")
	m, err := RunArm(context.Background(), p, DefaultTask, true, 10, nil, WithAuditJournal(j))
	if err != nil {
		t.Fatalf("RunArm mediated: %v", err)
	}
	if err := j.Close(); err != nil {
		t.Fatalf("journal.Close: %v", err)
	}

	if m.ToolCalls == 0 {
		t.Fatal("expected mediated arm to make tool calls")
	}

	n, err := journal.Verify(logPath)
	if err != nil {
		t.Fatalf("journal.Verify: %v", err)
	}
	if n == 0 {
		t.Fatal("expected journal to contain rows")
	}

	rows, _, err := journal.ReadRowsFrom(logPath, 0)
	if err != nil {
		t.Fatalf("ReadRowsFrom: %v", err)
	}
	for _, r := range rows {
		if r.ArgsDigest == "" {
			t.Error("expected non-empty ArgsDigest")
		}
		if r.ResultDigest == "" {
			t.Error("expected non-empty ResultDigest")
		}
	}

	// Verify tampering detection
	data, err := os.ReadFile(logPath)
	if err != nil {
		t.Fatalf("ReadFile: %v", err)
	}
	tampered := append([]byte(nil), data...)
	for i := range tampered {
		if tampered[i] != '\n' {
			tampered[i] ^= 0x01
			break
		}
	}
	tamperedPath := filepath.Join(dir, "mediated-tampered.jsonl")
	if err := os.WriteFile(tamperedPath, tampered, 0o644); err != nil {
		t.Fatalf("WriteFile: %v", err)
	}
	if _, err := journal.Verify(tamperedPath); err == nil {
		t.Fatal("expected journal.Verify to fail on tampered file")
	}
}

func TestAuditJournalDualArm(t *testing.T) {
	dir := t.TempDir()
	logPath := filepath.Join(dir, "dual-audit.jsonl")
	j, err := journal.Open(logPath)
	if err != nil {
		t.Fatalf("journal.Open: %v", err)
	}

	p := NewMockPlanner("test")
	res, _, err := Run(context.Background(), p, DefaultTask, 12, WithAuditJournal(j))
	if err != nil {
		t.Fatalf("Run dual-arm: %v", err)
	}
	if err := j.Close(); err != nil {
		t.Fatalf("journal.Close: %v", err)
	}

	if res.Fak.ToolCalls == 0 || res.Baseline.ToolCalls == 0 {
		t.Fatal("expected both arms to make tool calls")
	}

	n, err := journal.Verify(logPath)
	if err != nil {
		t.Fatalf("journal.Verify: %v", err)
	}
	if n < res.Fak.ToolCalls+res.Baseline.ToolCalls {
		t.Fatalf("expected rows >= %d, got %d", res.Fak.ToolCalls+res.Baseline.ToolCalls, n)
	}

	rows, _, err := journal.ReadRowsFrom(logPath, 0)
	if err != nil {
		t.Fatalf("ReadRowsFrom: %v", err)
	}
	hasRaw := false
	hasFak := false
	for _, r := range rows {
		if r.By == "raw-harness" {
			hasRaw = true
		} else {
			hasFak = true
		}
	}
	if !hasRaw {
		t.Error("expected journal to contain By=raw-harness rows")
	}
	if !hasFak {
		t.Error("expected journal to contain mediated kernel rows")
	}
}
