package main

import (
	"bytes"
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/sessionmine"
)

func TestSessionHistoryDefaultPathRefreshReadAndStatus(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)
	codex := filepath.Join(home, ".codex", "sessions")
	if err := os.MkdirAll(codex, 0755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(codex, "one.jsonl"), []byte(`{"type":"response_item","payload":{"type":"function_call","name":"view_image"}}`+"\n"), 0644); err != nil {
		t.Fatal(err)
	}

	var out, errOut bytes.Buffer
	if code := runSessionHistory(&out, &errOut, []string{"refresh", "--once", "--claude-root", "", "--days", "0", "--min-support", "1"}); code != 0 {
		t.Fatalf("refresh code=%d stderr=%s", code, errOut.String())
	}
	index := filepath.Join(home, ".fak", "session-history", "index.json")
	if _, err := os.Stat(index); err != nil {
		t.Fatalf("default index was not written at %s: %v", index, err)
	}

	out.Reset()
	errOut.Reset()
	if code := runSessionHistory(&out, &errOut, nil); code != 0 {
		t.Fatalf("read code=%d stderr=%s", code, errOut.String())
	}
	var report sessionmine.HistoryReport
	if err := json.Unmarshal(out.Bytes(), &report); err != nil {
		t.Fatal(err)
	}
	if report.Metrics.Sessions != 1 || len(report.Sessions) != 1 {
		t.Fatalf("default-path report=%+v", report)
	}

	out.Reset()
	errOut.Reset()
	if code := runSessionHistory(&out, &errOut, []string{"status"}); code != 0 {
		t.Fatalf("status code=%d stderr=%s", code, errOut.String())
	}
	if status := out.String(); !strings.Contains(status, "index: exists=true") || strings.Contains(status, "index_missing") {
		t.Fatalf("default-path status did not inspect refreshed index:\n%s", status)
	}
}

func TestRunSessionHistoryRefreshSpine(t *testing.T) {
	root := t.TempDir()
	codex := filepath.Join(root, "codex")
	os.MkdirAll(codex, 0755)
	os.WriteFile(filepath.Join(codex, "one.jsonl"), []byte(`{"type":"response_item","payload":{"type":"function_call","name":"view_image","arguments":"SECRET"}}`+"\n"), 0644)
	index := filepath.Join(root, "state", "index.json")
	var out, errOut bytes.Buffer
	args := []string{"refresh", "--once", "--index", index, "--codex-root", codex, "--claude-root", "", "--days", "0", "--min-support", "1"}
	if code := runSessionHistory(&out, &errOut, args); code != 0 {
		t.Fatalf("code=%d stderr=%s", code, errOut.String())
	}
	var receipt sessionmine.RefreshReceipt
	if err := json.Unmarshal(out.Bytes(), &receipt); err != nil {
		t.Fatal(err)
	}
	if receipt.Schema != "fak-session-history-refresh/1" || receipt.Run != 1 || receipt.ParsedFiles != 1 || receipt.Sessions != 1 {
		t.Fatalf("receipt=%+v", receipt)
	}
	out.Reset()
	if code := runSessionHistoryRefresh(context.Background(), &out, &errOut, []string{"--max-runs", "2", "--interval", "1ms", "--index", index, "--codex-root", codex, "--claude-root", "", "--days", "0"}); code != 0 {
		t.Fatalf("loop code=%d stderr=%s", code, errOut.String())
	}
	decoder := json.NewDecoder(bytes.NewReader(out.Bytes()))
	var receipts []sessionmine.RefreshReceipt
	for decoder.More() {
		var r sessionmine.RefreshReceipt
		if err := decoder.Decode(&r); err != nil {
			t.Fatal(err)
		}
		receipts = append(receipts, r)
	}
	if len(receipts) != 2 || receipts[0].ReusedFiles != 1 || receipts[1].ReusedFiles != 1 {
		t.Fatalf("receipts=%+v", receipts)
	}
}
