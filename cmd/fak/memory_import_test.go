package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func copyImportFixture(t *testing.T, name string) string {
	t.Helper()
	src := filepath.Join("testdata", "memory-import", name)
	dst := t.TempDir()
	entries, err := os.ReadDir(src)
	if err != nil {
		t.Fatal(err)
	}
	for _, e := range entries {
		body, err := os.ReadFile(filepath.Join(src, e.Name()))
		if err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(dst, e.Name()), body, 0o600); err != nil {
			t.Fatal(err)
		}
	}
	return dst
}

func TestClaudeMemoryImportClassifiesAndAccountsFixture(t *testing.T) {
	source := copyImportFixture(t, "source")
	destination := copyImportFixture(t, "destination")
	receipt, decisions, err := inspectClaudeMemory(source, destination)
	if err != nil {
		t.Fatal(err)
	}
	if receipt.Unexplained != 0 || receipt.Accounted != receipt.Source.Files {
		t.Fatalf("incomplete receipt: %+v", receipt)
	}
	want := map[string]int{"importable": 1, "duplicate": 2, "private": 1, "session-specific": 1, "stale": 1, "unsupported": 1}
	for class, n := range want {
		if receipt.Counts[class] != n {
			t.Errorf("%s=%d want %d; all=%v", class, receipt.Counts[class], n, receipt.Counts)
		}
	}
	for _, d := range decisions {
		if d.class == "private" && strings.Contains(receiptJSON(t, receipt), "credential") {
			t.Fatal("receipt leaked a private filename/body")
		}
	}
}

func TestClaudeMemoryImportApplyRequiresAuditAndNeverOverwrites(t *testing.T) {
	source := t.TempDir()
	destination := t.TempDir()
	body := []byte("# Durable operator rule\n\nAlways validate the isolated committed overlay before claiming that shared-main changes are green and ready to push.\n")
	if err := os.WriteFile(filepath.Join(source, "rule.md"), body, 0o600); err != nil {
		t.Fatal(err)
	}
	var out, errOut bytes.Buffer
	if code := runMemoryImportClaude(&out, &errOut, []string{"--source", source, "--destination", destination, "--apply"}); code != 2 {
		t.Fatalf("missing audit code=%d stderr=%s", code, errOut.String())
	}
	out.Reset()
	errOut.Reset()
	args := []string{"--source", source, "--destination", destination, "--apply", "--consent-scope", "project-memory-6855", "--producer", "fixture-test", "--capture-time", "2026-08-14T00:00:00Z"}
	if code := runMemoryImportClaude(&out, &errOut, args); code != 0 {
		t.Fatalf("apply code=%d stderr=%s out=%s", code, errOut.String(), out.String())
	}
	entries, _ := os.ReadDir(destination)
	if len(entries) != 1 {
		t.Fatalf("applied files=%d", len(entries))
	}
	got, _ := os.ReadFile(filepath.Join(destination, entries[0].Name()))
	for _, field := range []string{"source_sha256:", "source_bytes:", "durability_class:", "consent_scope:", "producer:", "captured_at:"} {
		if !bytes.Contains(got, []byte(field)) {
			t.Errorf("missing audit %s", field)
		}
	}
	out.Reset()
	if code := runMemoryImportClaude(&out, &errOut, args); code != 0 {
		t.Fatalf("second apply should dedup, code=%d out=%s", code, out.String())
	}
	var receipt claudeImportReceipt
	_ = json.Unmarshal(out.Bytes(), &receipt)
	if receipt.Applied != 0 || receipt.Counts["duplicate"] != 1 {
		t.Fatalf("second apply=%+v", receipt)
	}
}

func TestClaudeMemoryImportRefusesConcurrentMutation(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "durable.md")
	if err := os.WriteFile(path, []byte("# Durable rule\n\nThis sufficiently long durable project rule is suitable for deterministic importer mutation testing.\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	old := claudeImportBeforeResnapshot
	defer func() { claudeImportBeforeResnapshot = old }()
	claudeImportBeforeResnapshot = func() { _ = os.WriteFile(path, []byte("changed while scanning"), 0o600); time.Sleep(time.Millisecond) }
	receipt, _, err := inspectClaudeMemory(dir, "")
	if err == nil || !strings.Contains(err.Error(), "SOURCE_CHANGED") {
		t.Fatalf("receipt=%+v err=%v", receipt, err)
	}
}

func receiptJSON(t *testing.T, v any) string {
	t.Helper()
	b, err := json.Marshal(v)
	if err != nil {
		t.Fatal(err)
	}
	return string(b)
}
