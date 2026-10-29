package hwutil

import (
	"context"
	"strings"
	"testing"
	"time"
)

// TestTierCollectMergesRecorderDrain pins the collector contract: whatever
// the in-process emitter recorded since the last drain lands verbatim in
// Snapshot.Tiers/Transfers, and an idle recorder keeps both arrays empty
// (never zero-filled rows). The full family collection runs for real, so on
// Windows this costs a couple of seconds of WMI polling.
func TestTierCollectMergesRecorderDrain(t *testing.T) {
	rec := NewRecorder(16)
	rec.RecordPlacement("qwen3.8-14b", TierRAM, "host", 0, 35, 6291456000, 0)
	rec.RecordTransfer(TierSSD, TierRAM, 1<<30, 250*time.Millisecond, TransferLoad)

	snap, err := Collect(context.Background(), Options{Interval: time.Millisecond, Samples: 1, Recorder: rec})
	if err != nil {
		t.Fatalf("Collect: %v", err)
	}
	if len(snap.Tiers) != 1 || snap.Tiers[0].ModelID != "qwen3.8-14b" || snap.Tiers[0].Kind != TierRAM || snap.Tiers[0].Bytes != 6291456000 {
		t.Fatalf("tiers merged wrong: %+v", snap.Tiers)
	}
	if len(snap.Transfers) != 1 || snap.Transfers[0].FromTier != "ssd" || snap.Transfers[0].ToTier != "ram" || snap.Transfers[0].Reason != TransferLoad {
		t.Fatalf("transfers merged wrong: %+v", snap.Transfers)
	}
	if err := snap.Validate(); err != nil {
		t.Fatalf("merged snapshot violates the frozen schema: %v", err)
	}

	// Second pass: the drain consumed the rows, so an idle process reports
	// empty tier planes, not stale or invented ones.
	idle := NewRecorder(16)
	snap2, err := Collect(context.Background(), Options{Interval: time.Millisecond, Samples: 1, Recorder: idle})
	if err != nil {
		t.Fatalf("Collect (idle): %v", err)
	}
	if snap2.Tiers != nil || snap2.Transfers != nil {
		t.Fatalf("idle recorder must leave tiers/transfers nil: (%+v, %+v)", snap2.Tiers, snap2.Transfers)
	}
	raw, err := RenderJSON(snap2)
	if err != nil {
		t.Fatalf("RenderJSON: %v", err)
	}
	for _, needle := range []string{`"tiers":[]`, `"transfers":[]`} {
		if !strings.Contains(string(raw), needle) {
			t.Fatalf("wire lost empty array %s:\n%s", needle, raw)
		}
	}
}
