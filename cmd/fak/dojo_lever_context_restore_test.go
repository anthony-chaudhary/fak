package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/dojo"
	"github.com/anthony-chaudhary/fak/internal/gateway"
	"github.com/anthony-chaudhary/fak/internal/gatewayusageledger"
)

func TestGatewayUsageCountersPopulatesCompactionRestoredTurns(t *testing.T) {
	src, err := os.ReadFile("serve.go")
	if err != nil {
		t.Fatalf("read serve.go: %v", err)
	}
	body, ok := funcBodyText(string(src), "func gatewayUsageCounters(")
	if !ok {
		t.Fatal("gatewayUsageCounters not found in serve.go")
	}
	if !strings.Contains(body, "CompactionRestoredTurns:   adj.CompactionRestoredTurns") &&
		!strings.Contains(body, "CompactionRestoredTurns: adj.CompactionRestoredTurns") {
		t.Fatal("gatewayUsageCounters does not assign CompactionRestoredTurns from adj.CompactionRestoredTurns")
	}

	srv, err := gateway.New(gateway.Config{})
	if err != nil {
		t.Fatalf("gateway.New: %v", err)
	}
	counters := gatewayUsageCounters(srv)
	if counters.CompactionRestoredTurns != srv.AdjudicationSummary().CompactionRestoredTurns {
		t.Fatalf("gatewayUsageCounters().CompactionRestoredTurns = %d, want %d",
			counters.CompactionRestoredTurns, srv.AdjudicationSummary().CompactionRestoredTurns)
	}
}

func TestLoadContextSpanLedgerProducesMeasuredWhenExitRowsExistWithDroppedSpans(t *testing.T) {
	root := t.TempDir()
	ledgerDir := filepath.Join(root, filepath.Dir(gatewayusageledger.DefaultLedgerRel))
	if err := os.MkdirAll(ledgerDir, 0o755); err != nil {
		t.Fatalf("mkdir ledger dir: %v", err)
	}
	ledgerPath := filepath.Join(root, gatewayusageledger.DefaultLedgerRel)

	row1 := gatewayusageledger.NewRow("exit", "guard", "claude", "sess-1", time.Minute, nil, gatewayusageledger.Counters{
		CompactionDroppedTurns:  10,
		CompactionRestoredTurns: 5,
	}, time.Unix(1000, 0))

	row2 := gatewayusageledger.NewRow("periodic", "guard", "claude", "sess-1", time.Minute, nil, gatewayusageledger.Counters{
		CompactionDroppedTurns:  100, // Should be skipped because kind != "exit"
		CompactionRestoredTurns: 50,
	}, time.Unix(1001, 0))

	row3 := gatewayusageledger.NewRow("exit", "guard", "claude", "sess-2", time.Minute, nil, gatewayusageledger.Counters{
		CompactionDroppedTurns:  6,
		CompactionRestoredTurns: 3,
	}, time.Unix(1002, 0))

	if err := gatewayusageledger.Append(ledgerPath, row1); err != nil {
		t.Fatalf("append row1: %v", err)
	}
	if err := gatewayusageledger.Append(ledgerPath, row2); err != nil {
		t.Fatalf("append row2: %v", err)
	}
	if err := gatewayusageledger.Append(ledgerPath, row3); err != nil {
		t.Fatalf("append row3: %v", err)
	}

	led := loadContextSpanLedger(root)
	if !led.RestoreRecorded {
		t.Fatalf("led.RestoreRecorded = false, want true")
	}
	if led.DroppedSpans != 16 {
		t.Fatalf("led.DroppedSpans = %d, want 16 (10 + 6)", led.DroppedSpans)
	}
	if led.RestoredSpans != 8 {
		t.Fatalf("led.RestoredSpans = %d, want 8 (5 + 3)", led.RestoredSpans)
	}

	episodes := dojo.ContextRestoreEpisodes(led)
	if len(episodes) != 1 {
		t.Fatalf("len(episodes) = %d, want 1", len(episodes))
	}
	ep := episodes[0]
	if !ep.Outcome.Measured {
		t.Fatalf("ep.Outcome.Measured = false, want true")
	}
	wantRecall := 8.0 / 16.0 // 0.5
	if ep.Outcome.Realized != wantRecall {
		t.Fatalf("ep.Outcome.Realized = %v, want %v", ep.Outcome.Realized, wantRecall)
	}
}
