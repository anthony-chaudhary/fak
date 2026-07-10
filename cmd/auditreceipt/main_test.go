package main

import (
	"bytes"
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/modelroute"
)

func TestAuditReceiptBinaryExitCodes(t *testing.T) {
	binary := buildAuditReceiptBinary(t)
	clean := writeCleanLedgerFixture(t)
	cleanBytes, err := os.ReadFile(clean)
	if err != nil {
		t.Fatal(err)
	}

	tests := []struct {
		name    string
		data    []byte
		verdict string
		exit    int
	}{
		{name: "clean", data: cleanBytes, verdict: "CLEAN", exit: 0},
		{name: "one-byte-mutation", data: mutateOneByte(t, cleanBytes), verdict: "CORRUPT", exit: 1},
		{name: "torn-tail", data: bytes.TrimSuffix(cleanBytes, []byte{'\n'}), verdict: "TORN", exit: 1},
		{name: "corrupt-row", data: append(append([]byte(nil), cleanBytes...), []byte("not-json\n")...), verdict: "CORRUPT", exit: 1},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "ledger.jsonl")
			if err := os.WriteFile(path, tt.data, 0o600); err != nil {
				t.Fatal(err)
			}

			cmd := exec.Command(binary, path)
			output, runErr := cmd.Output()
			gotExit := processExitCode(runErr)
			if gotExit != tt.exit {
				t.Fatalf("process exit=%d, want %d; err=%v output=%s", gotExit, tt.exit, runErr, output)
			}
			t.Logf("process exit=%d output=%s", gotExit, bytes.TrimSpace(output))

			var got verificationOutput
			if err := json.Unmarshal(output, &got); err != nil {
				t.Fatalf("decode process output %q: %v", output, err)
			}
			if got.Schema != outputSchema || got.Verdict != tt.verdict {
				t.Fatalf("process output=%+v, want schema=%s verdict=%s", got, outputSchema, tt.verdict)
			}
			if tt.exit == 0 && (got.Rows != 1 || got.UniqueAudits != 1 || got.HeadHash == "" || got.Cursor == nil) {
				t.Fatalf("clean process output omitted strict verification facts: %+v", got)
			}
			if tt.exit != 0 && got.Error == "" {
				t.Fatalf("rejected process output omitted verifier error: %+v", got)
			}
		})
	}
}

func buildAuditReceiptBinary(t *testing.T) string {
	t.Helper()
	name := "auditreceipt"
	if runtime.GOOS == "windows" {
		name += ".exe"
	}
	path := filepath.Join(t.TempDir(), name)
	cmd := exec.Command("go", "build", "-o", path, ".")
	if output, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("build actual auditreceipt binary: %v\n%s", err, output)
	}
	return path
}

func writeCleanLedgerFixture(t *testing.T) string {
	t.Helper()
	receiptBytes, err := os.ReadFile(filepath.Join("..", "..", "internal", "modelroute", "testdata", "crossaudit_receipt_v1_3847.json"))
	if err != nil {
		t.Fatalf("read published receipt fixture: %v", err)
	}
	receipt, err := modelroute.ParseIssueAuditReceipt(receiptBytes)
	if err != nil {
		t.Fatalf("parse published receipt fixture: %v", err)
	}
	path := filepath.Join(t.TempDir(), "clean-ledger.jsonl")
	if _, err := modelroute.AppendAuditReceiptLedger(path, receipt); err != nil {
		t.Fatalf("write clean ledger fixture: %v", err)
	}
	return path
}

func mutateOneByte(t *testing.T, clean []byte) []byte {
	t.Helper()
	mutated := bytes.Replace(clean, []byte(`"verdict":"REFUTE"`), []byte(`"verdict":"XEFUTE"`), 1)
	if bytes.Equal(mutated, clean) || len(mutated) != len(clean) {
		t.Fatal("failed to make exactly one byte of semantic ledger corruption")
	}
	return mutated
}

func processExitCode(err error) int {
	if err == nil {
		return 0
	}
	if exitErr, ok := err.(*exec.ExitError); ok {
		return exitErr.ExitCode()
	}
	return -1
}
