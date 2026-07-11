package main

// nightrun_cut_test.go — the `fak nightrun cut` verb (#3490): dry-run by
// default, --apply rewrites, totals preserved. The fold mechanics themselves
// are proven in internal/gatewayusageledger/cut_test.go; this covers the door.

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/gatewayusageledger"
)

func TestNightrunCutDryRunByDefaultThenApply(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "gateway-usage.jsonl")
	for i := int64(1); i <= 4; i++ {
		row := gatewayusageledger.NewRow("exit", "serve", "test", "", 0, nil,
			gatewayusageledger.Counters{InputTokens: uint64(i)}, time.UnixMilli(i*1000))
		if err := gatewayusageledger.Append(path, row); err != nil {
			t.Fatalf("Append: %v", err)
		}
	}
	orig, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read: %v", err)
	}

	var out, errb bytes.Buffer
	if rc := runNightrun(&out, &errb, []string{"cut", "--keep", "1", "--usage-ledger", path}); rc != 0 {
		t.Fatalf("dry-run cut rc=%d stderr=%s", rc, errb.String())
	}
	if !strings.Contains(out.String(), "DRY-RUN") {
		t.Fatalf("default must be dry-run:\n%s", out.String())
	}
	if now, _ := os.ReadFile(path); string(now) != string(orig) {
		t.Fatalf("dry-run rewrote the ledger")
	}

	out.Reset()
	if rc := runNightrun(&out, &errb, []string{"cut", "--keep", "1", "--apply", "--usage-ledger", path}); rc != 0 {
		t.Fatalf("apply cut rc=%d stderr=%s", rc, errb.String())
	}
	if !strings.Contains(out.String(), "APPLIED") {
		t.Fatalf("apply must report APPLIED:\n%s", out.String())
	}
	rows := gatewayusageledger.ReadLedgerFile(path)
	if len(rows) != 2 { // 1 carryforward + 1 kept
		t.Fatalf("want 2 rows after cut, got %d", len(rows))
	}
	var total uint64
	for _, r := range rows {
		total += r.Counters.InputTokens
	}
	if total != 1+2+3+4 {
		t.Fatalf("input-token total not preserved across cut: %d", total)
	}
}
