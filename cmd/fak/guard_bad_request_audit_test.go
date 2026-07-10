package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/journal"
)

func TestGuardUpstreamBadRequestAuditNotifyPersistsScrubbedReason(t *testing.T) {
	path := filepath.Join(t.TempDir(), "audit.jsonl")
	j, err := journal.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	notify := guardUpstreamBadRequestAuditNotify(j, "trace-400")
	if notify == nil {
		t.Fatal("nil notifier")
	}
	notify(`input[3].id must match call_id [redacted]`)
	if err := j.Close(); err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var row journal.Row
	if err := json.Unmarshal([]byte(strings.TrimSpace(string(data))), &row); err != nil {
		t.Fatal(err)
	}
	if row.Kind != "UPSTREAM_BAD_REQUEST" || row.TraceID != "trace-400" || !strings.Contains(row.Reason, "input[3].id") || row.Hash == "" {
		t.Fatalf("row=%+v", row)
	}
}
