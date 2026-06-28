package session

import (
	"context"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/cachemeta"
	"github.com/anthony-chaudhary/fak/internal/model"
)

// warmCache builds a small kernel-owned KVCache for the splice tests. It is empty (Len 0)
// because a populated cache can only be produced by a prefill inside internal/model, which this
// lane must not edit; the splice mechanism this test proves — Clone + the cachemeta promote — is
// exercised identically on an empty or a populated cache (Clone deep-copies whatever it holds),
// so the assertions key on the SPLICE (a distinct reattached pointer, a KVRestore directive)
// rather than on cache contents.
func warmCache() *model.KVCache {
	return model.NewKVCache(model.Config{NumLayers: 2, NumKVHeads: 1, HeadDim: 4})
}

// TestWarmKVStoreSpliceRestoresWarm proves the CONCRETE warm splice: a parked cache is cloned
// (a DISTINCT reattached cache, not an alias) and the cachemeta lifecycle promote emits
// KVRestore (cold tier -> hot tier). This is the #916 splice the resume loop reports as warm —
// the resumed turn attends reattached KV instead of cold re-prefilling.
func TestWarmKVStoreSpliceRestoresWarm(t *testing.T) {
	store := NewWarmKVStore()
	const trace = "gw-warm"
	orig := warmCache()
	store.Park(trace, orig, cachemeta.TierDRAM)

	res := store.Splice(trace)
	if !res.Warm {
		t.Fatalf("splice = %+v, want Warm (a parked cache must reattach)", res)
	}
	if res.Restored == nil {
		t.Fatal("warm splice returned a nil restored cache")
	}
	if res.Restored == orig {
		t.Fatal("restored cache is the SAME pointer as the parked one; Clone must deep-copy, not alias")
	}
	if res.RestoredPositions != orig.Len() {
		t.Fatalf("restored positions = %d, want %d (Clone preserves Len)", res.RestoredPositions, orig.Len())
	}
	// The cachemeta promote: a span resident in DRAM moved back to HBM is a RESTORE.
	if res.Direction != cachemeta.KVRestore {
		t.Fatalf("splice direction = %q, want %q (promote DRAM->HBM is a restore)", res.Direction, cachemeta.KVRestore)
	}
	if res.FromTier != cachemeta.TierDRAM || res.ToTier != cachemeta.TierHBM {
		t.Fatalf("splice tiers = %s->%s, want dram->hbm", res.FromTier, res.ToTier)
	}

	// LastSplice records the move for an observability / supervisor read.
	got, ok := store.LastSplice(trace)
	if !ok || !got.Warm || got.Direction != cachemeta.KVRestore {
		t.Fatalf("LastSplice = (%+v, %v), want a recorded warm KVRestore", got, ok)
	}

	// The parked entry is consumed on a warm splice: a SECOND resume finds nothing and is cold.
	if again := store.Splice(trace); again.Warm {
		t.Fatalf("second splice = %+v, want cold (a resume reclaims the parked KV exactly once)", again)
	}
}

// TestWarmKVStoreColdMiss proves the degrade-safe path: a trace with no parked KV (never
// offloaded, or evicted while paused) splices nothing and reports cold — the resume loop then
// falls back to cold re-prefill, so correctness never depends on the warm path.
func TestWarmKVStoreColdMiss(t *testing.T) {
	store := NewWarmKVStore()

	// Never parked -> cold.
	if res := store.Splice("gw-never"); res.Warm || res.Restored != nil {
		t.Fatalf("unparked splice = %+v, want cold with no restored cache", res)
	}

	// Parked then EVICTED while paused -> cold.
	const trace = "gw-evicted"
	store.Park(trace, warmCache(), cachemeta.TierDRAM)
	store.Evict(trace)
	if res := store.Splice(trace); res.Warm {
		t.Fatalf("evicted splice = %+v, want cold (warm KV dropped while paused)", res)
	}
}

// TestWarmSpliceWiredIntoResumeLoop proves the END-TO-END acceptance: a WarmKVStore wired into
// a Table via WatchResumeSplice makes a Paused->Running resume return ResumeWarm AND drives the
// concrete KVCache.Clone + cachemeta.MoveTo(KVRestore) splice — not the bare bool seam, the real
// mover. A session with no parked KV resumes cold through the same wiring.
func TestWarmSpliceWiredIntoResumeLoop(t *testing.T) {
	tbl := NewTable()
	store := NewWarmKVStore()
	tbl.WatchResumeSplice(store.Splicer())

	const trace = "gw-e2e"
	// The session offloads its KV at pause...
	store.Park(trace, warmCache(), cachemeta.TierDRAM)
	if _, ok := tbl.Transition(trace, Paused, "operator-hold"); !ok {
		t.Fatal("pause rejected")
	}

	verdicts := make(chan ResumeVerdict, 1)
	go func() { verdicts <- tbl.WaitResume(context.Background(), trace) }()
	time.Sleep(10 * time.Millisecond)

	// ...and reclaims it warm on resume.
	if _, ok := tbl.Transition(trace, Running, ""); !ok {
		t.Fatal("resume rejected")
	}
	select {
	case v := <-verdicts:
		if !v.Resumed || v.Mode != ResumeWarm {
			t.Fatalf("verdict = %+v, want Resumed warm (the wired splicer reattached KV)", v)
		}
	case <-time.After(time.Second):
		t.Fatal("WaitResume did not wake warm on Paused->Running")
	}
	// The concrete mover ran: a KVRestore was recorded for the trace.
	res, ok := store.LastSplice(trace)
	if !ok || res.Direction != cachemeta.KVRestore || res.Restored == nil {
		t.Fatalf("LastSplice = (%+v, %v), want a recorded warm KVRestore with a reattached cache", res, ok)
	}

	// A DIFFERENT session with no parked KV resumes COLD through the same wiring (degrade-safe).
	const cold = "gw-cold-e2e"
	tbl.Transition(cold, Paused, "hold")
	go func() { verdicts <- tbl.WaitResume(context.Background(), cold) }()
	time.Sleep(10 * time.Millisecond)
	tbl.Transition(cold, Running, "")
	select {
	case v := <-verdicts:
		if !v.Resumed || v.Mode != ResumeCold {
			t.Fatalf("unparked verdict = %+v, want Resumed cold (no warm KV for this trace)", v)
		}
	case <-time.After(time.Second):
		t.Fatal("WaitResume did not wake cold for the unparked session")
	}
}
