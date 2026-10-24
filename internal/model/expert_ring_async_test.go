package model

import (
	"context"
	"errors"
	"sync"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/polymodel"
)

// expert_ring_async_test.go — the #5627 overlap primitive.
//
// R3/#5614 staged a layer's activated set before its first GEMM and could not report the one number
// it was asked for: the fraction of page-in latency overlapped with compute. Nothing in
// compute.Backend could say when an upload landed, so "issued early" was real and unobservable.
//
// These drive the REAL moeFFN.apply seam (as the R3 tests do) against a backend that advertises
// compute.AsyncUploader, so the fence discipline is exercised where it actually runs rather than by
// calling the ring directly.

// asyncTestFence is a Fence whose landing the test controls. `landed` starts true for a transfer
// that completed underneath earlier work, false for one the demand path must wait on.
type asyncTestFence struct {
	landed  bool
	waits   int
	release func() // applied on Wait, so a SKIPPED fence is observable rather than merely unsafe
}

func (f *asyncTestFence) Done() bool { return f.landed }
func (f *asyncTestFence) Wait() {
	f.waits++
	if !f.landed {
		f.landed = true
		if f.release != nil {
			f.release()
		}
	}
}

// expertAsyncBackend is the order-recording backend plus compute.AsyncUploader. It keys each
// in-flight fence by the uploaded tensor's Buffer identity, so MatMul can detect the one thing this
// extension could get catastrophically wrong: multiplying a weight whose transfer is not yet
// visible. That check is the reason a delayed fence makes the TEST fail instead of quietly
// producing wrong numbers.
type expertAsyncBackend struct {
	*expertOrderRecordingBackend
	landImmediately bool
	inflight        map[compute.Buffer]*asyncTestFence
	fences          []*asyncTestFence
	// unfencedGEMMs counts GEMMs issued against a weight whose own transfer had not landed.
	// It must be 0. Anything else is a data race the ring would hit on a real async backend.
	unfencedGEMMs int
}

func newExpertAsyncBackend(landImmediately bool) *expertAsyncBackend {
	return &expertAsyncBackend{
		expertOrderRecordingBackend: &expertOrderRecordingBackend{Backend: compute.Default()},
		landImmediately:             landImmediately,
		inflight:                    map[compute.Buffer]*asyncTestFence{},
	}
}

func (b *expertAsyncBackend) Name() string { return "cuda-test-async" }

func (b *expertAsyncBackend) UploadAsync(t compute.Tensor, as compute.Dtype) (compute.Tensor, compute.Fence) {
	out := b.expertOrderRecordingBackend.Upload(t, as)
	f := &asyncTestFence{landed: b.landImmediately}
	buf := out.Buf()
	if !b.landImmediately && buf != nil {
		b.inflight[buf] = f
		f.release = func() { delete(b.inflight, buf) }
	}
	b.fences = append(b.fences, f)
	return out, f
}

func (b *expertAsyncBackend) MatMul(w, x compute.Tensor) compute.Tensor {
	if buf := w.Buf(); buf != nil {
		if f, pending := b.inflight[buf]; pending && !f.landed {
			b.unfencedGEMMs++
		}
	}
	return b.expertOrderRecordingBackend.MatMul(w, x)
}

func expertAsyncSession(m *Model, ringBytes int64, landImmediately bool) (*Session, *expertAsyncBackend) {
	be := newExpertAsyncBackend(landImmediately)
	return &Session{
		M: m, Backend: be, Q4K: true, halW: map[string]compute.Tensor{},
		ExpertRingBytes: ringBytes,
	}, be
}

// The witness R3 could not produce: on a k>1 layer whose activated set is prefetched, the transfers
// that landed underneath earlier work are counted as overlapped, and the fraction is non-zero.
func TestExpertRingPrefetchRecordsOverlapOnAnAsyncBackend(t *testing.T) {
	const H, E, K = 256, 8, 4
	m := expertPrefetchModel(t, H, E, K)
	perWeight := expertRingWeightBytes(t, m)

	s, be := expertAsyncSession(m, perWeight*3*K, true) // every transfer lands before it is demanded
	defer s.Close()
	moeFFN{}.apply(m, 0, expertRingTestInput(H), sessionQ4KKernel{s: s})

	if len(be.fences) == 0 {
		t.Fatal("no transfer went through UploadAsync; the async path was never taken and this proves nothing")
	}
	st := s.ExpertRing()
	if st.AsyncOverlapped == 0 {
		t.Fatalf("AsyncOverlapped=0 with %d async transfers issued: a prefetch whose transfers all landed before demand must read as overlap", len(be.fences))
	}
	if got := st.AsyncOverlapFraction(); got <= 0 {
		t.Fatalf("AsyncOverlapFraction=%v, want > 0 (overlapped=%d waited=%d)", got, st.AsyncOverlapped, st.AsyncWaited)
	}
	if be.unfencedGEMMs != 0 {
		t.Fatalf("%d GEMM(s) ran against a weight whose transfer had not landed", be.unfencedGEMMs)
	}
}

// A synchronous backend must report 0 and stay byte-for-byte unchanged: no fences, no overlap, and
// a fraction of 0 that means "this backend cannot report overlap", not "no overlap was achieved".
func TestExpertRingReportsNoOverlapOnASynchronousBackend(t *testing.T) {
	const H, E, K = 256, 8, 4
	m := expertPrefetchModel(t, H, E, K)
	perWeight := expertRingWeightBytes(t, m)

	s, _ := expertPrefetchSession(m, perWeight*3*K) // no AsyncUploader
	defer s.Close()
	moeFFN{}.apply(m, 0, expertRingTestInput(H), sessionQ4KKernel{s: s})

	st := s.ExpertRing()
	if st.AsyncOverlapped != 0 || st.AsyncWaited != 0 {
		t.Fatalf("synchronous backend reported overlapped=%d waited=%d, want 0/0", st.AsyncOverlapped, st.AsyncWaited)
	}
	if got := st.AsyncOverlapFraction(); got != 0 {
		t.Fatalf("AsyncOverlapFraction=%v on a backend with no fences, want 0", got)
	}
	if st.Prefetched == 0 {
		t.Fatal("nothing was prefetched; the synchronous path did not run the same seam and the comparison is empty")
	}
}

// The safety half, and the one the acceptance gate names: a transfer that has NOT landed must be
// waited on before its weight is multiplied. The backend flags any GEMM against an in-flight
// weight, so a skipped fence fails the test rather than silently producing wrong numbers.
func TestExpertRingAsyncNeverMultipliesBeforeItsFenceLands(t *testing.T) {
	const H, E, K = 256, 8, 4
	m := expertPrefetchModel(t, H, E, K)
	perWeight := expertRingWeightBytes(t, m)

	s, be := expertAsyncSession(m, perWeight*3*K, false) // nothing lands until someone Waits
	defer s.Close()
	moeFFN{}.apply(m, 0, expertRingTestInput(H), sessionQ4KKernel{s: s})

	if len(be.fences) == 0 {
		t.Fatal("no transfer went through UploadAsync; the fence discipline was never exercised")
	}
	if be.unfencedGEMMs != 0 {
		t.Fatalf("%d GEMM(s) ran against a weight whose transfer had not landed — the ring skipped a fence", be.unfencedGEMMs)
	}
	// Every issued transfer must end up settled: an un-awaited fence is device storage the ring
	// still believes is arriving.
	for i, f := range be.fences {
		if !f.landed {
			t.Fatalf("fence %d of %d was never satisfied; its weight is resident but not visible", i, len(be.fences))
		}
	}
	st := s.ExpertRing()
	if st.AsyncWaited == 0 {
		t.Fatalf("AsyncWaited=0 although no transfer landed on its own; the blocking path was not exercised (overlapped=%d)", st.AsyncOverlapped)
	}
}

// The R0 bit-equality claim has to survive the async path: a ring-served GEMM equals the
// synchronous one. Same weights, same input, both orderings — identical logits, not merely close.
func TestExpertRingAsyncIsBitEqualToTheSynchronousPath(t *testing.T) {
	const H, E, K = 256, 8, 4
	for _, land := range []bool{true, false} {
		m := expertPrefetchModel(t, H, E, K)
		perWeight := expertRingWeightBytes(t, m)
		in := expertRingTestInput(H)

		sync, _ := expertPrefetchSession(m, perWeight*3*K)
		wantOut := moeFFN{}.apply(m, 0, in, sessionQ4KKernel{s: sync})
		sync.Close()

		async, be := expertAsyncSession(m, perWeight*3*K, land)
		gotOut := moeFFN{}.apply(m, 0, in, sessionQ4KKernel{s: async})
		if be.unfencedGEMMs != 0 {
			t.Fatalf("land=%v: %d unfenced GEMM(s)", land, be.unfencedGEMMs)
		}
		async.Close()

		if len(gotOut) != len(wantOut) {
			t.Fatalf("land=%v: async produced %d values, sync %d", land, len(gotOut), len(wantOut))
		}
		for i := range wantOut {
			if gotOut[i] != wantOut[i] {
				t.Fatalf("land=%v: async output differs from the synchronous path at %d: %v != %v",
					land, i, gotOut[i], wantOut[i])
			}
		}
	}
}

type expertRingCheckedCompletion struct {
	result     compute.OwnedTransferResult
	next       compute.OwnedTransferResult
	waits      int
	panicValue any
}

func (c *expertRingCheckedCompletion) Query() compute.OwnedTransferResult { return c.result }
func (c *expertRingCheckedCompletion) Wait(ctx context.Context) compute.OwnedTransferResult {
	c.waits++
	if c.panicValue != nil {
		panic(c.panicValue)
	}
	if err := ctx.Err(); err != nil {
		c.result = compute.OwnedTransferResult{State: compute.OwnedTransferUnknown, Err: err}
	} else {
		c.result = c.next
	}
	return c.result
}

type expertRingCheckedBackend struct {
	*expertAsyncBackend
	mu         sync.Mutex
	checked    map[compute.Buffer]*expertRingCheckedCompletion
	frees      int
	multiplies int
	unsafeUses int
}

func (b *expertRingCheckedBackend) ForwardExecutionOwner() *sync.Mutex { return &b.mu }
func (b *expertRingCheckedBackend) check(t compute.Tensor) {
	if c := b.checked[t.Buf()]; c != nil && (c.result.State != compute.OwnedTransferComplete || c.result.Err != nil) {
		b.unsafeUses++
	}
}
func (b *expertRingCheckedBackend) Free(t compute.Tensor) {
	b.check(t)
	b.frees++
	b.expertAsyncBackend.Free(t)
}
func (b *expertRingCheckedBackend) MatMul(w, x compute.Tensor) compute.Tensor {
	b.check(w)
	b.multiplies++
	return b.expertAsyncBackend.MatMul(w, x)
}

type expertRingClassedAsyncBackend struct {
	*expertAsyncBackend
	class compute.MemoryClass
}

func (b *expertRingClassedAsyncBackend) UploadClass(src compute.Tensor, dtype compute.Dtype, class compute.MemoryClass, _ string) compute.Tensor {
	b.class = class
	return b.Upload(src, dtype)
}

// This bounded fixture inserts the published transfer value at the consumer
// boundary. It is not a classed async provider or a pre-domain-release drain,
// and it does not claim that outer session Close can retry a refused cleanup.
// fak-test:runtime fast est=10ms lane=default
func TestExpertRingOwnedTransferRetirement(t *testing.T) {
	source := func() compute.Tensor {
		return compute.NewF32(compute.Default(), []int{1, 1}, []float32{3})
	}
	newRing := func() (*pagedRing, *expertRingCheckedBackend) {
		be := &expertRingCheckedBackend{expertAsyncBackend: newExpertAsyncBackend(false),
			checked: map[compute.Buffer]*expertRingCheckedCompletion{}}
		return newPagedRing(be, 8), be
	}
	attach := func(t *testing.T, r *pagedRing, be *expertRingCheckedBackend, id polymodel.ModelID, wt compute.Tensor, next compute.OwnedTransferResult) (*expertRingTransfer, *expertRingCheckedCompletion) {
		t.Helper()
		src := source()
		c := &expertRingCheckedCompletion{result: compute.OwnedTransferResult{State: compute.OwnedTransferPending}, next: next}
		owned, err := compute.NewOwnedTransfer(compute.OwnedTransferDescriptor{
			Class: compute.MemoryOffload, Domain: be, SourceOwner: &src, DestinationOwner: wt,
		}, c)
		if err != nil {
			t.Fatal(err)
		}
		p := &expertRingTransfer{id: id, class: owned.Descriptor().Class, source: &src, destination: wt, owned: owned}
		if r.pending == nil {
			r.pending = map[polymodel.ModelID]compute.Fence{}
		}
		r.pending[id] = p
		be.checked[wt.Buf()] = c
		return p, c
	}
	complete := compute.OwnedTransferResult{State: compute.OwnedTransferComplete}

	t.Run("demand-eviction-and-close", func(t *testing.T) {
		r, be := newRing()
		if _, ok := r.stageInFlight("pinned", source, compute.F32, 4, true); !ok {
			t.Fatal("pinned warmup refused")
		}
		wt, ok := r.stageInFlight("prefetch", source, compute.F32, 4, false)
		if !ok || len(be.fences) != 0 {
			t.Fatal("owned backend must retain synchronous upload guard")
		}
		p, c := attach(t, r, be, "prefetch", wt, complete)
		if p.owned.Domain() != &be.mu || p.owned.Descriptor().Class != compute.MemoryOffload {
			t.Fatal("class or wrapper domain identity lost")
		}
		x := compute.NewF32(compute.Default(), []int{1}, []float32{2})
		got := r.matMulStaged("prefetch", source, compute.F32, x, 4, false)
		if len(got) != 1 || got[0] != 6 || c.waits != 1 || be.multiplies != 1 || be.unsafeUses != 0 || len(r.pending) != 0 || r.asyncWaited != 1 {
			t.Fatal("demand did not settle before the cpu-ref multiply")
		}
		if p.owned.Descriptor().SourceOwner != nil {
			t.Fatal("successful demand did not retire its owners")
		}
		p.Wait()
		p.Wait()
		if err := r.awaitStaged("prefetch"); err != nil || c.waits != 1 || r.asyncWaited != 1 || !p.Done() {
			t.Fatal("repeated checked settlement was not idempotent")
		}
		// A new undemanded record is discarded by eviction, not credited as overlap.
		_, c = attach(t, r, be, "prefetch", wt, complete)
		if _, ok := r.stage("next", source, compute.F32, 4, false); !ok {
			t.Fatal("settled victim was not evicted")
		}
		if c.waits != 1 || be.frees != 1 || !r.isResident("pinned") || r.isResident("prefetch") || r.used() != 8 || r.peakUsed() > r.budget() || r.asyncWaited != 1 || r.asyncOverlapped != 0 {
			t.Fatal("eviction changed pin, budget or overlap accounting")
		}
		_, c = attach(t, r, be, "next", r.resident["next"], complete)
		c.result = complete
		if _, ok := r.stage("next", source, compute.F32, 4, false); !ok || c.waits != 0 || r.asyncOverlapped != 1 {
			t.Fatal("already-complete demand did not use checked overlap accounting")
		}
		attach(t, r, be, "next", r.resident["next"], complete)
		r.freeAll()
		if be.unsafeUses != 0 || be.frees != 3 || r.used() != 0 || len(r.pending) != 0 || r.asyncWaited != 1 || r.asyncOverlapped != 1 {
			t.Fatal("close did not settle before Free without overlap credit")
		}
	})

	providerErr := errors.New("checked copy failed")
	for _, tc := range []struct {
		name   string
		result compute.OwnedTransferResult
		cancel bool
	}{
		{name: "pending", result: compute.OwnedTransferResult{State: compute.OwnedTransferPending}},
		{name: "failed", result: compute.OwnedTransferResult{State: compute.OwnedTransferFailed, Err: providerErr}},
		{name: "unknown", result: compute.OwnedTransferResult{State: compute.OwnedTransferUnknown}},
		{name: "cancel", result: complete, cancel: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			r, be := newRing()
			wt, _ := r.stageInFlight("prefetch", source, compute.F32, 4, false)
			p, completion := attach(t, r, be, "prefetch", wt, tc.result)
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			if tc.cancel {
				cancel()
			}
			err := r.settleStaged(ctx, "prefetch", true)
			var refusal *expertRingTransferError
			if !errors.As(err, &refusal) || refusal.ID != "prefetch" || refusal.Class != compute.MemoryOffload {
				t.Fatalf("missing typed settlement refusal: %v", err)
			}
			if tc.result.Err != nil && !errors.Is(err, tc.result.Err) || tc.cancel && !errors.Is(err, context.Canceled) {
				t.Fatal("original completion error lost")
			}
			if got := r.matMulStaged("prefetch", source, compute.F32, wt, 4, false); got != nil || be.multiplies != 0 {
				t.Fatal("refused demand reached MatMul")
			}
			builds := 0
			if _, err := r.stageWithError("next", func() compute.Tensor { builds++; return source() }, compute.F32, 8, false); err == nil || builds != 0 {
				t.Fatal("uncertain eviction mutated admission")
			}
			r.policy = ExpertRingEvictValueAware
			for _, refuse := range []func(){r.freeAll, func() { r.evictForPolicy(8) }, func() { r.hold("prefetch") }, p.Wait} {
				func() {
					defer func() {
						if _, ok := recover().(*expertRingTransferError); !ok {
							t.Error("void lifecycle seam must refuse rather than Free")
						}
					}()
					refuse()
				}()
			}
			if r.pending["prefetch"] != p || p.owned.Descriptor().SourceOwner != p.source || p.owned.Descriptor().DestinationOwner == nil || p.owned.Domain() != &be.mu || be.frees != 0 || be.unsafeUses != 0 || r.used() != 4 || !r.isResident("prefetch") || len(r.holds) != 0 || r.asyncWaited != 0 || r.asyncOverlapped != 0 {
				t.Fatal("refusal lost roots, domain, residency or truthful accounting")
			}
			// Rollback uses the same checked barrier, even without a pool entry.
			r.pool.Evict("prefetch")
			delete(r.resident, "prefetch")
			if err := r.rollbackStaged("prefetch", wt); err == nil || r.pending["prefetch"] != p || be.frees != 0 {
				t.Fatal("rollback freed or forgot a live transfer")
			}
			if tc.name == "pending" {
				completion.next = complete
				r.freeAll()
				if be.frees != 1 || be.unsafeUses != 0 || len(r.pending) != 0 {
					t.Fatal("close lost the retained unadmitted allocation")
				}
			}
		})
	}

	t.Run("provider-panic-and-pin-refusal", func(t *testing.T) {
		r, be := newRing()
		wt, _ := r.stageInFlight("held", source, compute.F32, 4, false)
		r.hold("held")
		before, _ := r.pool.Get("held")
		p, c := attach(t, r, be, "held", wt, complete)
		marker := errors.New("completion panic")
		c.panicValue = marker
		func() {
			defer func() {
				if recover() != marker {
					t.Error("provider panic was not preserved")
				}
			}()
			_ = r.awaitStaged("held")
		}()
		unpinned := before
		unpinned.Pinned = false
		for _, refuse := range []func(){func() { r.release("held") }, func() { r.repin("held", unpinned) }} {
			func() {
				defer func() {
					if _, ok := recover().(*expertRingTransferError); !ok {
						t.Error("pin mutation did not refuse unknown settlement")
					}
				}()
				refuse()
			}()
		}
		after, _ := r.pool.Get("held")
		if after != before || !after.Pinned || r.holds["held"] != 1 || !r.heldPins["held"] || r.pending["held"] != p || p.owned.Descriptor().SourceOwner == nil || p.owned.Query().State != compute.OwnedTransferUnknown || c.waits != 1 || be.frees != 0 || r.asyncWaited != 0 {
			t.Fatal("panic or pin refusal lost pending roots or changed bookkeeping")
		}
	})

	t.Run("rollback-and-classed-guard", func(t *testing.T) {
		r, be := newRing()
		wt := be.Upload(source(), compute.F32)
		_, c := attach(t, r, be, "unadmitted", wt, complete)
		if err := r.rollbackStaged("unadmitted", wt); err != nil || c.waits != 1 || be.frees != 1 || be.unsafeUses != 0 || len(r.pending) != 0 || r.asyncWaited != 0 || r.asyncOverlapped != 0 {
			t.Fatal("successful rollback did not settle before Free")
		}
		classed := &expertRingClassedAsyncBackend{expertAsyncBackend: newExpertAsyncBackend(false)}
		guarded := newPagedRing(classed, 4)
		if _, ok := guarded.stageInFlight("classed", source, compute.F32, 4, false); !ok || len(classed.fences) != 0 || len(guarded.pending) != 0 || classed.class != compute.MemoryOffload {
			t.Fatal("classed upload guard or class forwarding changed")
		}
		guarded.freeAll()
	})
	t.Run("legacy-admit-rollback", func(t *testing.T) {
		rec := &pinnedPreflightBackend{Backend: compute.Default()}
		be := &pinnedPreflightAsyncBackend{pinnedPreflightBackend: rec}
		r := newPagedRing(be, 4)
		// Empty IDs pass the ring's size/pin checks but are refused by Admit.
		if _, err := r.stageWithError("", source, compute.F32, 4, false); !errors.Is(err, polymodel.ErrEmptyID) {
			t.Fatalf("expected failed Admit: %v", err)
		}
		if len(be.fences) != 1 || be.fences[0].waits != 1 || !be.fences[0].landed || rec.frees != 1 || len(r.pending) != 0 || r.used() != 0 || r.asyncWaited != 0 || r.asyncOverlapped != 0 {
			t.Fatal("legacy rollback did not preserve wait-before-Free without overlap credit")
		}
	})
}
