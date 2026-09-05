package main

import (
	"bytes"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/journal"
)

func TestAgentAuditJournalCLI(t *testing.T) {
	t.Run("raw arm with jsonl log", func(t *testing.T) {
		dir := t.TempDir()
		logPath := filepath.Join(dir, "raw-cli.jsonl")
		outPath := filepath.Join(dir, "out.json")

		runAgent([]string{
			"--raw",
			"--offline",
			"--log", logPath,
			"--out", outPath,
			"--code-tools=false",
			"--sys-tools=false",
		})

		var stdout, stderr bytes.Buffer
		code := runCmdAuditVerify(&stdout, &stderr, []string{logPath})
		if code != 0 {
			t.Fatalf("runCmdAuditVerify exited %d, stderr: %s", code, stderr.String())
		}
		if !strings.Contains(stdout.String(), "OK") || !strings.Contains(stdout.String(), "chain intact") {
			t.Fatalf("unexpected stdout: %s", stdout.String())
		}

		// Tampering detection
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
		tamperedPath := filepath.Join(dir, "raw-cli-tampered.jsonl")
		if err := os.WriteFile(tamperedPath, tampered, 0o644); err != nil {
			t.Fatalf("WriteFile: %v", err)
		}

		stdout.Reset()
		stderr.Reset()
		code = runCmdAuditVerify(&stdout, &stderr, []string{tamperedPath})
		if code != 1 {
			t.Fatalf("expected code 1 on tampered file, got %d", code)
		}
		if !strings.Contains(stderr.String(), "TAMPERED/BROKEN") {
			t.Fatalf("expected TAMPERED/BROKEN in stderr, got: %s", stderr.String())
		}
		if _, err := journal.Verify(tamperedPath); err == nil {
			t.Fatal("expected journal.Verify to fail on tampered file")
		}
	})

	t.Run("native arm with jsonl log", func(t *testing.T) {
		dir := t.TempDir()
		logPath := filepath.Join(dir, "native-cli.jsonl")
		outPath := filepath.Join(dir, "out.json")

		runAgent([]string{
			"--native",
			"--offline",
			"--log", logPath,
			"--out", outPath,
			"--code-tools=false",
			"--sys-tools=false",
		})

		var stdout, stderr bytes.Buffer
		code := runCmdAuditVerify(&stdout, &stderr, []string{logPath})
		if code != 0 {
			t.Fatalf("runCmdAuditVerify exited %d, stderr: %s", code, stderr.String())
		}
		if !strings.Contains(stdout.String(), "OK") || !strings.Contains(stdout.String(), "chain intact") {
			t.Fatalf("unexpected stdout: %s", stdout.String())
		}

		// Tampering detection
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
		tamperedPath := filepath.Join(dir, "native-cli-tampered.jsonl")
		if err := os.WriteFile(tamperedPath, tampered, 0o644); err != nil {
			t.Fatalf("WriteFile: %v", err)
		}

		stdout.Reset()
		stderr.Reset()
		code = runCmdAuditVerify(&stdout, &stderr, []string{tamperedPath})
		if code != 1 {
			t.Fatalf("expected code 1 on tampered file, got %d", code)
		}
		if !strings.Contains(stderr.String(), "TAMPERED/BROKEN") {
			t.Fatalf("expected TAMPERED/BROKEN in stderr, got: %s", stderr.String())
		}
		if _, err := journal.Verify(tamperedPath); err == nil {
			t.Fatal("expected journal.Verify to fail on tampered file")
		}
	})

	t.Run("dual arm with jsonl log", func(t *testing.T) {
		dir := t.TempDir()
		logPath := filepath.Join(dir, "dual-cli.jsonl")
		outPath := filepath.Join(dir, "out.json")

		runAgent([]string{
			"--offline",
			"--log", logPath,
			"--out", outPath,
			"--code-tools=false",
			"--sys-tools=false",
		})

		var stdout, stderr bytes.Buffer
		code := runCmdAuditVerify(&stdout, &stderr, []string{logPath})
		if code != 0 {
			t.Fatalf("runCmdAuditVerify exited %d, stderr: %s", code, stderr.String())
		}
		if !strings.Contains(stdout.String(), "OK") || !strings.Contains(stdout.String(), "chain intact") {
			t.Fatalf("unexpected stdout: %s", stdout.String())
		}

		// Tampering detection
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
		tamperedPath := filepath.Join(dir, "dual-cli-tampered.jsonl")
		if err := os.WriteFile(tamperedPath, tampered, 0o644); err != nil {
			t.Fatalf("WriteFile: %v", err)
		}

		stdout.Reset()
		stderr.Reset()
		code = runCmdAuditVerify(&stdout, &stderr, []string{tamperedPath})
		if code != 1 {
			t.Fatalf("expected code 1 on tampered file, got %d", code)
		}
		if !strings.Contains(stderr.String(), "TAMPERED/BROKEN") {
			t.Fatalf("expected TAMPERED/BROKEN in stderr, got: %s", stderr.String())
		}
	})

	t.Run("legacy plaintext log in dual arm", func(t *testing.T) {
		dir := t.TempDir()
		logPath := filepath.Join(dir, "trace.txt")
		outPath := filepath.Join(dir, "out.json")

		runAgent([]string{
			"--offline",
			"--log", logPath,
			"--out", outPath,
			"--code-tools=false",
			"--sys-tools=false",
		})

		data, err := os.ReadFile(logPath)
		if err != nil {
			t.Fatalf("ReadFile: %v", err)
		}
		if len(data) == 0 {
			t.Fatal("expected plaintext trace to be non-empty")
		}
	})
}

func TestAgentAuditJournalLegacyRejection(t *testing.T) {
	if os.Getenv("TEST_AGENT_AUDIT_HELPER") == "1" {
		for i, arg := range os.Args {
			if arg == "--" && i+1 < len(os.Args) {
				runAgent(os.Args[i+1:])
				return
			}
		}
		return
	}

	t.Run("raw arm rejects plaintext log", func(t *testing.T) {
		cmd := exec.Command(os.Args[0], "-test.run=TestAgentAuditJournalLegacyRejection", "--", "--raw", "--offline", "--log", "trace.txt", "--out", "out.json", "--code-tools=false", "--sys-tools=false")
		cmd.Env = append(os.Environ(), "TEST_AGENT_AUDIT_HELPER=1")
		out, err := cmd.CombinedOutput()
		if err == nil {
			t.Fatal("expected error, got nil")
		}
		if !strings.Contains(string(out), "--raw does not support --log") {
			t.Fatalf("expected rejection message, got: %s", string(out))
		}
	})

	t.Run("native arm rejects plaintext log", func(t *testing.T) {
		cmd := exec.Command(os.Args[0], "-test.run=TestAgentAuditJournalLegacyRejection", "--", "--native", "--offline", "--log", "trace.txt", "--out", "out.json", "--code-tools=false", "--sys-tools=false")
		cmd.Env = append(os.Environ(), "TEST_AGENT_AUDIT_HELPER=1")
		out, err := cmd.CombinedOutput()
		if err == nil {
			t.Fatal("expected error, got nil")
		}
		if !strings.Contains(string(out), "--native does not support --log") {
			t.Fatalf("expected rejection message, got: %s", string(out))
		}
	})
}
