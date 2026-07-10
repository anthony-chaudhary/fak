package session

import (
	"context"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
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

func TestWarmKVSpliceCarriesKVSpanPointer(t *testing.T) {
	store := NewWarmKVStore()
	first := warmCache()
	store.Park("gw-pointer-a", first, cachemeta.TierDRAM)
	got := store.Splice("gw-pointer-a")
	if !got.Warm || got.SpanPointer.Kind != KindKVSpan || got.SpanPointer.Ref == "" {
		t.Fatalf("warm splice pointer = %+v, want non-empty %q pointer", got.SpanPointer, KindKVSpan)
	}

	store.Park("gw-pointer-b", warmCache(), cachemeta.TierDRAM)
	again := store.Splice("gw-pointer-b")
	if again.SpanPointer != got.SpanPointer {
		t.Fatalf("same parked cache pointer = %+v, want stable %+v", again.SpanPointer, got.SpanPointer)
	}
	if cold := store.Splice("gw-missing"); cold.SpanPointer != (KVSpanPointer{}) {
		t.Fatalf("cold splice pointer = %+v, want zero", cold.SpanPointer)
	}
}

func TestWarmKVResumeVerdictCarriesSpanPointer(t *testing.T) {
	tbl := NewTable()
	store := NewWarmKVStore()
	tbl.WatchResumeSplice(store.Splicer())
	const trace = "gw-pointer-resume"
	store.Park(trace, warmCache(), cachemeta.TierDRAM)

	verdicts := make(chan ResumeVerdict, 1)
	tbl.Transition(trace, Paused, "hold")
	go func() { verdicts <- tbl.WaitResume(context.Background(), trace) }()
	time.Sleep(10 * time.Millisecond)
	tbl.Transition(trace, Running, "")
	verdict := <-verdicts
	if verdict.Mode != ResumeWarm || verdict.SpanPointer.Kind != KindKVSpan || verdict.SpanPointer.Ref == "" {
		t.Fatalf("resume verdict = %+v, want warm with kv_span pointer", verdict)
	}
}

type relaunchKVBackend struct {
	staged map[string]bool
}

func (b *relaunchKVBackend) Len() int                { return 0 }
func (b *relaunchKVBackend) Prefill([]int) []float32 { return nil }
func (b *relaunchKVBackend) Evict(int, int) int      { return 0 }
func (b *relaunchKVBackend) ModelID() string         { return "test" }
func (b *relaunchKVBackend) StageSpan(_ context.Context, digest string, _, _ int) (abi.KVResidency, error) {
	if b.staged == nil {
		b.staged = map[string]bool{}
	}
	b.staged[digest] = true
	return abi.KVResidency{Outcome: abi.KVResidencyOK, Digest: digest}, nil
}
func (b *relaunchKVBackend) RestoreSpan(_ context.Context, digest string) (abi.KVResidency, error) {
	if b.staged[digest] {
		return abi.KVResidency{Outcome: abi.KVResidencyOK, Digest: digest}, nil
	}
	return abi.KVResidency{Outcome: abi.KVResidencyMiss, Digest: digest}, nil
}

func TestWarmKVSurvivesRelaunchViaRestoreSpan(t *testing.T) {
	backend := &relaunchKVBackend{}
	origin := NewWarmKVStoreWithBackend(backend)
	origin.Park("origin", warmCache(), cachemeta.TierDRAM)
	parked := origin.Splice("origin")
	if parked.SpanPointer.Ref == "" || !backend.staged[parked.SpanPointer.Ref] {
		t.Fatalf("park stage = %+v staged=%v, want durable pointer", parked, backend.staged)
	}

	fresh := NewWarmKVStoreWithBackend(backend)
	fresh.CarrySpan("relaunched", parked.SpanPointer)
	restored := fresh.Splice("relaunched")
	if !restored.Warm || restored.Residency.Outcome != abi.KVResidencyOK {
		t.Fatalf("fresh-store restore = %+v, want warm residency OK", restored)
	}

	missing := NewWarmKVStoreWithBackend(&relaunchKVBackend{})
	missing.CarrySpan("missing", parked.SpanPointer)
	miss := missing.Splice("missing")
	if miss.Warm || miss.Residency.Outcome != abi.KVResidencyMiss {
		t.Fatalf("default/missing restore = %+v, want cold MISS", miss)
	}
}

// degradeKVBackend is a configurable KVBackend double for the cold-degrade proof (#4134): its
// RestoreSpan returns a fixed KVResidency outcome plus an optional transport error (standing in
// for a ctx-deadline / store FAULT), so the table below can drive the OK|MISS|FAULT trichotomy
// through Splice/WaitResume without a real off-box tier. StageSpan always succeeds so a span can
// be parked off-box before the restore is exercised.
type degradeKVBackend struct {
	restore abi.KVResidencyOutcome
	err     error
}

func (b *degradeKVBackend) Len() int                { return 0 }
func (b *degradeKVBackend) Prefill([]int) []float32 { return nil }
func (b *degradeKVBackend) Evict(int, int) int      { return 0 }
func (b *degradeKVBackend) ModelID() string         { return "degrade-test" }
func (b *degradeKVBackend) StageSpan(_ context.Context, digest string, _, _ int) (abi.KVResidency, error) {
	return abi.KVResidency{Outcome: abi.KVResidencyOK, Digest: digest}, nil
}
func (b *degradeKVBackend) RestoreSpan(_ context.Context, digest string) (abi.KVResidency, error) {
	return abi.KVResidency{Outcome: b.restore, Digest: digest}, b.err
}

// TestWarmKVResumeDegradesToColdOnUnavailableSpan pins the safety invariant the whole warm-KV
// cluster rests on (#4134, epic #1193): warm KV is an optimization, NEVER a correctness
// dependency. F-warmkv-relaunch-2 (#4133) added an off-box restore that can MISS (the tier
// evicted the span), FAULT (transport error / ctx deadline), or find no carried pointer at all;
// each MUST collapse to a correct COLD re-prefill — never a silent wrong-KV attend, never a hang.
// This proves it across the whole trichotomy at both levels: the store (Splice) and end-to-end
// (WaitResume), each guarded by a timeout so a FAULT that blocked would fail rather than wedge.
func TestWarmKVResumeDegradesToColdOnUnavailableSpan(t *testing.T) {
	span := KVSpanPointer{Kind: KindKVSpan, Ref: "deadbeef-span-digest"}

	cases := []struct {
		name    string
		carry   bool                   // install a carried off-box pointer (false = absent, no pointer)
		restore abi.KVResidencyOutcome // RestoreSpan outcome when the pointer is carried
		err     error                  // RestoreSpan transport error (a FAULT / ctx-deadline stand-in)
		want    abi.KVResidencyOutcome // residency expected on the (cold) SpliceResult
	}{
		{name: "miss_evicted", carry: true, restore: abi.KVResidencyMiss, want: abi.KVResidencyMiss},
		{name: "fault_transport", carry: true, restore: abi.KVResidencyFault, err: context.DeadlineExceeded, want: abi.KVResidencyFault},
		{name: "absent_no_pointer", carry: false},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			// Store level: Splice must report cold (Warm=false), surface the typed residency, and
			// RETURN — a FAULT / ctx deadline must degrade, not hang. The goroutine + timeout is the
			// non-blocking witness.
			store := NewWarmKVStoreWithBackend(&degradeKVBackend{restore: tc.restore, err: tc.err})
			if tc.carry {
				store.CarrySpan("relaunched", span)
			}
			done := make(chan SpliceResult, 1)
			go func() { done <- store.Splice("relaunched") }()
			var res SpliceResult
			select {
			case res = <-done:
			case <-time.After(time.Second):
				t.Fatal("Splice blocked on an unavailable off-box span; must degrade to cold, not hang")
			}
			if res.Warm {
				t.Fatalf("Splice = %+v, want cold (Warm=false) for an unavailable span", res)
			}
			if tc.carry && res.Residency.Outcome != tc.want {
				t.Fatalf("Splice residency = %s, want %s", res.Residency.Outcome, tc.want)
			}

			// End-to-end: a Paused->Running resume through the wired splicer must verdict ResumeCold,
			// never ResumeWarm, and must WAKE (not block) on the resume edge.
			tbl := NewTable()
			e2e := NewWarmKVStoreWithBackend(&degradeKVBackend{restore: tc.restore, err: tc.err})
			tbl.WatchResumeSplice(e2e.Splicer())
			if tc.carry {
				e2e.CarrySpan("e2e", span)
			}
			verdicts := make(chan ResumeVerdict, 1)
			tbl.Transition("e2e", Paused, "hold")
			go func() { verdicts <- tbl.WaitResume(context.Background(), "e2e") }()
			time.Sleep(10 * time.Millisecond)
			tbl.Transition("e2e", Running, "")
			select {
			case v := <-verdicts:
				if !v.Resumed {
					t.Fatalf("verdict = %+v, want Resumed", v)
				}
				if v.Mode != ResumeCold {
					t.Fatalf("verdict mode = %s, want cold for an unavailable span (warm KV is not a correctness precondition)", v.Mode)
				}
			case <-time.After(time.Second):
				t.Fatal("WaitResume did not wake on the resume edge; a MISS/FAULT must not block the loop")
			}
		})
	}

	// In-process-default parity: the default store (nil backend — the in-process default hosts no
	// off-box tier, so RestoreSpan is a guaranteed MISS) resumes byte-identically to today's cold
	// path. Even a carried pointer degrades to cold, so a default build is unchanged.
	t.Run("in_process_default_is_cold", func(t *testing.T) {
		store := NewWarmKVStore()
		store.CarrySpan("default", span)
		if res := store.Splice("default"); res.Warm {
			t.Fatalf("default-store splice = %+v, want cold (the in-process default never restores off-box)", res)
		}
	})
}
