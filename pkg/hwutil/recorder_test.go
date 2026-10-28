package hwutil

import (
	"sync"
	"testing"
	"time"
)

func TestTierRecorderPlacementAccumulatesAndDrains(t *testing.T) {
	rec := NewRecorder(16)
	rec.RecordPlacement("qwen3.8-14b", TierRAM, "host", 0, 35, 6291456000, 0)
	rec.RecordPlacement("qwen3.8-14b", TierVRAM, "cuda:0", 0, 35, 1073741824, 0)
	rec.RecordPlacement("qwen3.8-14b", TierVRAM, "cuda:0", 0, 35, 536870912, 0) // same band grows

	tiers, transfers := rec.Drain()
	if len(transfers) != 0 {
		t.Fatalf("transfers = %v, want none", transfers)
	}
	if len(tiers) != 2 {
		t.Fatalf("tiers = %+v, want 2 bands", tiers)
	}
	ram, vram := tiers[0], tiers[1]
	if ram.Kind != TierRAM || ram.Device != "host" || ram.ModelID != "qwen3.8-14b" ||
		ram.LayersStart != 0 || ram.LayersEnd != 35 || ram.Bytes != 6291456000 || ram.HotnessRank != 0 {
		t.Fatalf("ram band = %+v", ram)
	}
	if vram.Kind != TierVRAM || vram.Device != "cuda:0" || vram.Bytes != 1610612736 {
		t.Fatalf("vram band must accumulate both placements: %+v", vram)
	}
	if ram.LastTransferAt == nil || vram.LastTransferAt == nil {
		t.Fatalf("placement events stamp LastTransferAt: ram=%+v vram=%+v", ram, vram)
	}
	if ram.TransferBytesPerS != nil || vram.TransferBytesPerS != nil {
		t.Fatalf("placement alone carries no rate: ram=%v vram=%v", ram.TransferBytesPerS, vram.TransferBytesPerS)
	}
	// Drain clears: a second drain is empty, and RenderJSON keeps arrays honest.
	againTiers, againTransfers := rec.Drain()
	if againTiers != nil || againTransfers != nil {
		t.Fatalf("second drain = (%v, %v), want nil/nil", againTiers, againTransfers)
	}
}

func TestTierRecorderTransferRingEvictsOldest(t *testing.T) {
	rec := NewRecorder(4)
	for i := int64(1); i <= 6; i++ {
		rec.RecordTransfer(TierSSD, TierRAM, i*1024, 100*time.Microsecond, TransferLoad)
	}
	_, got := rec.Drain()
	if len(got) != 4 {
		t.Fatalf("ring kept %d rows, want cap 4", len(got))
	}
	// Oldest two evicted; FIFO order preserved for the survivors.
	for i, want := range []int64{3072, 4096, 5120, 6144} {
		if got[i].Bytes != want {
			t.Fatalf("row %d bytes = %d, want %d (rows=%+v)", i, got[i].Bytes, want, got)
		}
	}
	for _, row := range got {
		if row.FromTier != "ssd" || row.ToTier != "ram" || row.Reason != TransferLoad {
			t.Fatalf("row identity drifted: %+v", row)
		}
		if row.RateGiBS < 0 {
			t.Fatalf("rate negative: %+v", row)
		}
	}
}

func TestTierRecorderTransferRateMath(t *testing.T) {
	rec := NewRecorder(8)
	rec.RecordTransfer(TierRAM, TierVRAM, 2<<30, time.Second, TransferLoad)
	rec.RecordTransfer(TierVRAM, TierRAM, 1<<30, 0, TransferEvict) // zero duration -> rate 0
	_, got := rec.Drain()
	if len(got) != 2 {
		t.Fatalf("got %d rows", len(got))
	}
	if got[0].RateGiBS != 2.0 {
		t.Fatalf("measured rate = %v GiB/s, want 2", got[0].RateGiBS)
	}
	if got[0].DurationS != 1.0 || got[0].Reason != TransferLoad {
		t.Fatalf("row 0 = %+v", got[0])
	}
	if got[1].RateGiBS != 0 || got[1].DurationS != 0 || got[1].Reason != TransferEvict {
		t.Fatalf("row 1 = %+v", got[1])
	}
}

func TestTierRecorderDropsMalformedRows(t *testing.T) {
	rec := NewRecorder(8)
	rec.RecordPlacement("", TierRAM, "host", 0, 4, 100, 0)                            // empty model
	rec.RecordPlacement("m", TierRAM, " ", 0, 4, 100, 0)                              // empty device
	rec.RecordPlacement("m", TierRAM, "host", 5, 2, 100, 0)                           // inverted layers
	rec.RecordPlacement("m", TierRAM, "host", 0, 4, -1, 0)                            // negative bytes
	rec.RecordPlacement("m", TierRAM, "host", 0, 4, 100, -1)                          // negative rank
	rec.RecordPlacement("m", TierKind("hbm"), "host", 0, 4, 100, 0)                   // open tier kind
	rec.RecordTransfer(TierRAM, TierRAM, 100, time.Second, TransferLoad)              // same tier
	rec.RecordTransfer(TierSSD, TierRAM, -1, time.Second, TransferLoad)               // negative bytes
	rec.RecordTransfer(TierSSD, TierRAM, 100, -time.Second, TransferLoad)             // negative duration
	rec.RecordTransfer(TierSSD, TierRAM, 100, time.Second, TransferReason("migrate")) // open reason -> unknown
	rec.RecordPlacement("control-model", TierRAM, "host", 0, 4, 100, 0)               // valid control row

	tiers, transfers := rec.Drain()
	if len(tiers) != 1 || tiers[0].Bytes != 100 {
		t.Fatalf("malformed placements leaked: %+v", tiers)
	}
	if len(transfers) != 1 {
		t.Fatalf("malformed transfers leaked: %+v", transfers)
	}
	if transfers[0].Reason != TransferUnknown {
		t.Fatalf("open reason must land as unknown, got %q", transfers[0].Reason)
	}
}

func TestTierRecorderConcurrentNoLoss(t *testing.T) {
	rec := NewRecorder(DefaultTransferCapacity)
	const workers, perWorker = 8, 100
	var wg sync.WaitGroup
	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func(w int) {
			defer wg.Done()
			for i := 0; i < perWorker; i++ {
				rec.RecordTransfer(TierSSD, TierRAM, int64(w*perWorker+i)+1, time.Millisecond, TransferLoad)
			}
		}(w)
	}
	wg.Wait()
	if got := rec.TransferLen(); got != workers*perWorker {
		t.Fatalf("buffered %d transfers, want %d", got, workers*perWorker)
	}
	seen := make(map[int64]bool, workers*perWorker)
	_, got := rec.Drain()
	for _, row := range got {
		seen[row.Bytes] = true
	}
	for w := 0; w < workers; w++ {
		for i := 0; i < perWorker; i++ {
			v := int64(w*perWorker+i) + 1
			if !seen[v] {
				t.Fatalf("transfer byte tag %d lost under concurrency", v)
			}
		}
	}
}

func TestTierRecorderNilReceiverIsNoop(t *testing.T) {
	var rec *Recorder
	rec.RecordPlacement("m", TierRAM, "host", 0, 4, 100, 0)
	rec.RecordTransfer(TierSSD, TierRAM, 100, time.Second, TransferLoad)
	if tiers, transfers := rec.Drain(); tiers != nil || transfers != nil {
		t.Fatalf("nil recorder must stay inert: (%v, %v)", tiers, transfers)
	}
	if rec.TransferLen() != 0 {
		t.Fatalf("nil recorder TransferLen = %d", rec.TransferLen())
	}
}

func TestTierNewRecorderClampsCapacity(t *testing.T) {
	if NewRecorder(0).transferCap != DefaultTransferCapacity {
		t.Fatal("zero capacity must select the default")
	}
	if NewRecorder(-5).transferCap != DefaultTransferCapacity {
		t.Fatal("negative capacity must select the default")
	}
	if NewRecorder(maxTransferCapacity+1).transferCap != maxTransferCapacity {
		t.Fatal("over-capacity must clamp down")
	}
}
