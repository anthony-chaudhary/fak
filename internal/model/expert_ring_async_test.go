package model

import (
	"context"
	"errors"
	"sync"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/polymodel"
)

// These completion fixtures retain their transfers independently of a ring,
// including after a failed scope drain releases its domain. A native provider
// must supply its own quarantine and native retention; Go references are not pins.
type retirementTestCompletion struct {
	legacy *asyncTestFence
	query  compute.OwnedTransferResult
	wait   compute.OwnedTransferResult
}

func (c *retirementTestCompletion) Query() compute.OwnedTransferResult {
	if c.query.State == compute.OwnedTransferComplete && c.query.Err == nil {
		c.legacy.Wait()
	}
	return c.query
}

func (c *retirementTestCompletion) Wait(ctx context.Context) compute.OwnedTransferResult {
	if err := ctx.Err(); err != nil {
		return compute.OwnedTransferResult{State: compute.OwnedTransferUnknown, Err: err}
	}
	if c.wait.State == compute.OwnedTransferComplete && c.wait.Err == nil {
		c.legacy.Wait()
	}
	return c.wait
}

type retirementTestFence struct {
	transfer     *compute.OwnedTransfer
	completion   *retirementTestCompletion
	legacyCalls  int
	handoffs     int
	returned     compute.OwnedTransferDescriptor
	handoffErr   error
	handoffPanic bool
}

func (f *retirementTestFence) OwnedTransfer() *compute.OwnedTransfer { return f.transfer }
func (f *retirementTestFence) Done() bool                            { f.legacyCalls++; return true }
func (f *retirementTestFence) Wait()                                 { f.legacyCalls++ }

type retirementTestBackend struct {
	*ownedClassedAsyncBackend
	result           compute.OwnedTransferResult
	transfers        []*compute.OwnedTransfer
	checked          []*retirementTestFence
	frees            int
	unsafeFrees      int
	mutateDescriptor func(*compute.OwnedTransferDescriptor)
	handoffErr       error
	handoffPanic     bool
	panicFree        compute.Buffer
	freed            []compute.Buffer
}

func (f *retirementTestFence) RetireOwnedTransfer(d compute.OwnedTransferDescriptor) error {
	f.handoffs++
	f.returned = d
	if f.handoffPanic {
		panic("retirement handoff outcome unknown")
	}
	return f.handoffErr
}

func newRetirementTestBackend() *retirementTestBackend {
	return &retirementTestBackend{
		ownedClassedAsyncBackend: newOwnedClassedAsyncBackend(new(sync.Mutex), false),
		result:                   compute.OwnedTransferResult{State: compute.OwnedTransferComplete},
	}
}

func (b *retirementTestBackend) UploadAsyncClass(src compute.Tensor, dtype compute.Dtype, class compute.MemoryClass, site string) (compute.Tensor, compute.Fence) {
	tensor, legacy := b.ownedClassedAsyncBackend.UploadAsyncClass(src, dtype, class, site)
	completion := &retirementTestCompletion{
		legacy: legacy.(*asyncTestFence),
		query:  compute.OwnedTransferResult{State: compute.OwnedTransferPending},
		wait:   b.result,
	}
	descriptor := compute.OwnedTransferDescriptor{
		Class: class, Domain: b, SourceOwner: src, DestinationOwner: tensor,
	}
	if b.mutateDescriptor != nil {
		b.mutateDescriptor(&descriptor)
	}
	transfer, err := compute.NewOwnedTransfer(descriptor, completion)
	if err != nil {
		panic(err)
	}
	fence := &retirementTestFence{transfer: transfer, completion: completion, handoffErr: b.handoffErr, handoffPanic: b.handoffPanic}
	b.transfers = append(b.transfers, transfer)
	b.checked = append(b.checked, fence)
	return tensor, fence
}

func (b *retirementTestBackend) Free(t compute.Tensor) {
	if b.panicFree != nil && b.panicFree == t.Buf() {
		panic("injected victim free failure")
	}
	if f := b.inflight[t.Buf()]; f != nil && !f.landed {
		b.unsafeFrees++
	}
	b.frees++
	b.Backend.Free(t)
	b.freed = append(b.freed, t.Buf())
}

type retirementPanicFence struct{ compute.Fence }

func (*retirementPanicFence) Done() bool { return false }
func (*retirementPanicFence) Wait()      { panic("native transfer outcome unknown") }

type retirementPanicBackend struct{ *expertAsyncBackend }

func (b *retirementPanicBackend) UploadAsync(src compute.Tensor, dtype compute.Dtype) (compute.Tensor, compute.Fence) {
	tensor, fence := b.expertAsyncBackend.UploadAsync(src, dtype)
	return tensor, &retirementPanicFence{Fence: fence}
}

func retirementRecover(f func()) (caught any) {
	defer func() { caught = recover() }()
	f()
	return nil
}

// fak-test:runtime fast est=300ms lane=default
func TestExpertRingOwnedTransferRetirement(t *testing.T) {
	mk := func() compute.Tensor {
		return compute.NewF32(compute.Default(), []int{2, 2}, []float32{1, 2, 3, 4})
	}
	t.Run("legacy panic preserves pending ownership", func(t *testing.T) {
		be := &retirementPanicBackend{expertAsyncBackend: newExpertAsyncBackend(false)}
		r := newPagedRing(be, 16)
		if _, ok := r.stageInFlight("weight", mk, compute.F32, 16, false); !ok {
			t.Fatal("stage refused")
		}
		if retirementRecover(func() { r.stage("weight", mk, compute.F32, 16, false) }) == nil {
			t.Fatal("native failure disappeared")
		}
		if len(r.pending) != 1 || !r.isResident("weight") || !r.pool.Has("weight") {
			t.Fatal("failed transfer lost its pending record or resident identity")
		}
		if r.asyncOverlapped != 0 || r.asyncWaited != 0 {
			t.Fatal("failed transfer earned overlap accounting")
		}
	})
	t.Run("checked failures cannot publish evict or close", func(t *testing.T) {
		for _, state := range []compute.OwnedTransferState{compute.OwnedTransferFailed, compute.OwnedTransferUnknown, compute.OwnedTransferPending} {
			t.Run(string(rune('0'+state)), func(t *testing.T) {
				be := newRetirementTestBackend()
				failure := errors.New("checked copy failed")
				be.result = compute.OwnedTransferResult{State: state}
				if state != compute.OwnedTransferPending {
					be.result.Err = failure
				}
				r := newPagedRing(be, 16)
				s := &Session{Backend: be}
				scope := s.beginBackendExecution()
				defer func() {
					if scope.live {
						retirementRecover(scope.finish)
					}
				}()
				s.ringEnter(r)()
				if _, ok := r.stageInFlight("weight", mk, compute.F32, 16, false); !ok {
					t.Fatal("stage refused")
				}
				if _, err := r.stageWithError("weight", mk, compute.F32, 16, false); err == nil {
					t.Error("checked unsuccessful destination was published")
				} else if state != compute.OwnedTransferPending && !errors.Is(err, failure) {
					t.Errorf("lost provider error: %v", err)
				}
				if _, err := r.stageWithError("replacement", mk, compute.F32, 16, false); err == nil {
					t.Error("unsettled victim was evicted")
				}
				if r.isResident("replacement") || !r.isResident("weight") || !r.pool.Has("weight") || r.used() != 16 {
					t.Error("refusal mutated bounded residency")
				}
				if retirementRecover(func() { r.freeAll() }) == nil {
					t.Error("close reported success with retained transfers")
				}
				if len(r.pending) == 0 || be.frees != 0 || be.unsafeFrees != 0 {
					t.Error("unsettled storage was freed or forgotten")
				}
				if r.asyncOverlapped != 0 || r.asyncWaited != 0 {
					t.Error("unsuccessful transfer earned overlap accounting")
				}
				for _, fence := range be.checked {
					if fence.legacyCalls != 0 {
						t.Error("checked transfer trusted its legacy Fence methods")
					}
				}
				if retirementRecover(scope.finish) == nil {
					t.Error("failed domain drain was hidden")
				}
				requireOwnerReleased(t, be.owner)
				// Scope release is not quarantine: this provider independently retains the roots.
				for _, transfer := range be.transfers {
					d := transfer.Descriptor()
					if d.SourceOwner == nil || d.DestinationOwner == nil || transfer.Domain() != be.owner {
						t.Error("failed transfer lost provider-owned roots/domain")
					}
				}
			})
		}
	})
	t.Run("binding rejects unrelated ownership", func(t *testing.T) {
		for _, field := range []string{"class", "domain", "source", "destination"} {
			t.Run(field, func(t *testing.T) {
				be := newRetirementTestBackend()
				be.mutateDescriptor = func(d *compute.OwnedTransferDescriptor) {
					switch field {
					case "class":
						d.Class = compute.MemoryWeights
					case "domain":
						d.Domain = newRetirementTestBackend()
					case "source":
						d.SourceOwner = mk()
					case "destination":
						d.DestinationOwner = mk()
					}
				}
				r := newPagedRing(be, 16)
				s := &Session{Backend: be}
				scope := s.beginBackendExecution()
				defer func() {
					if scope.live {
						retirementRecover(scope.finish)
					}
				}()
				s.ringEnter(r)()
				if _, err := r.stageWithError("weight", mk, compute.F32, 16, false); err == nil {
					t.Error("unrelated ownership was accepted")
				}
				if r.residentCount() != 0 || r.used() != 0 || len(r.pending) != 1 || be.frees != 0 {
					t.Error("invalid binding was admitted, freed or forgotten")
				}
				if be.checked[0].legacyCalls != 0 || be.checked[0].handoffs != 0 {
					t.Error("invalid binding was settled or retired")
				}
			})
		}
	})
	t.Run("retirement handoff is never retried", func(t *testing.T) {
		for _, panics := range []bool{false, true} {
			be := newRetirementTestBackend()
			be.handoffErr = errors.New("retirement failed")
			be.handoffPanic = panics
			r := newPagedRing(be, 16)
			s := &Session{Backend: be}
			scope := s.beginBackendExecution()
			s.ringEnter(r)()
			for i := 0; i < 2; i++ {
				caught := retirementRecover(func() {
					if _, err := r.stageWithError("weight", mk, compute.F32, 16, false); err == nil {
						t.Error("failed handoff published destination")
					}
				})
				if panics && i == 0 && caught == nil {
					t.Error("handoff panic disappeared")
				}
			}
			if len(be.checked) != 1 || be.checked[0].handoffs != 1 || len(r.pending) != 1 || be.frees != 0 {
				t.Error("uncertain handoff repeated, freed or lost")
			}
			if len(be.checked) > 0 {
				f := be.checked[0]
				if f.returned.SourceOwner == nil || f.returned.DestinationOwner == nil {
					t.Error("handoff lost retained descriptor")
				}
			}
			retirementRecover(scope.finish)
			requireOwnerReleased(t, be.owner)
		}
	})
	t.Run("settle only the selected policy victims", func(t *testing.T) {
		for _, tc := range []struct {
			name    string
			policy  ExpertRingEvictPolicy
			failed  int
			refused bool
			victim  string
		}{
			{"value-selected-failed", ExpertRingEvictValueAware, 2, true, ""},
			{"value-unselected-failed", ExpertRingEvictValueAware, 0, false, "newer-cold"},
			{"lru-unselected-failed", ExpertRingEvictLRU, 2, false, "older-hot"},
		} {
			t.Run(tc.name, func(t *testing.T) {
				be := newRetirementTestBackend()
				r := newPagedRing(be, 48)
				s := &Session{Backend: be}
				scope := s.beginBackendExecution()
				defer func() {
					if scope.live {
						retirementRecover(scope.finish)
					}
				}()
				s.ringEnter(r)()
				for _, name := range []string{"older-hot", "pin", "newer-cold"} {
					if _, ok := r.stageInFlight(name, mk, compute.F32, 16, name == "pin"); !ok {
						t.Fatal("seed refused")
					}
				}
				r.policy = tc.policy
				r.heat = map[polymodel.ModelID]int{"older-hot": 10, "pin": 0, "newer-cold": 0}
				r.lastUse = map[polymodel.ModelID]uint64{"older-hot": 1, "pin": 2, "newer-cold": 3}
				be.checked[tc.failed].completion.wait = compute.OwnedTransferResult{State: compute.OwnedTransferUnknown, Err: context.Canceled}
				_, err := r.stageWithError("replacement", mk, compute.F32, 16, false)
				if (err != nil) != tc.refused {
					t.Fatalf("stage error=%v, refusal=%v", err, tc.refused)
				}
				if tc.refused {
					if !errors.Is(err, context.Canceled) || r.isResident("replacement") || r.residentCount() != 3 {
						t.Fatal("failed victim changed residency")
					}
				} else if r.isResident(tc.victim) || !r.isResident("replacement") || r.evict != 1 {
					t.Fatal("wrong policy victim")
				}
				pin, ok := r.pool.Get("pin")
				if !ok || !pin.Pinned || !r.isResident("pin") || r.used() != 48 || be.unsafeFrees != 0 {
					t.Fatal("pin, budget or settlement invariant changed")
				}
				if be.checked[tc.failed].transfer.Descriptor().DestinationOwner == nil {
					t.Fatal("unsettled owner retired")
				}
			})
		}
	})
	t.Run("holds protect pending weights without claiming completion", func(t *testing.T) {
		be := newRetirementTestBackend()
		r := newPagedRing(be, 32)
		s := &Session{Backend: be}
		scope := s.beginBackendExecution()
		defer func() {
			if scope.live {
				retirementRecover(scope.finish)
			}
		}()
		s.ringEnter(r)()
		r.stageInFlight("held", mk, compute.F32, 16, false)
		r.stageInFlight("pin", mk, compute.F32, 16, true)
		r.hold("held")
		if _, err := r.stageWithError("next", mk, compute.F32, 16, false); !errors.Is(err, polymodel.ErrPinnedNoRoom) {
			t.Fatalf("held admission=%v", err)
		}
		if len(be.checked) != 2 || be.checked[0].handoffs != 0 || len(r.pending) != 2 {
			t.Fatal("hold issued or retired a transfer")
		}
		r.release("held")
		if _, err := r.stageWithError("next", mk, compute.F32, 16, false); err != nil {
			t.Fatal(err)
		}
		if r.isResident("held") || !r.isResident("pin") || !r.isResident("next") || r.used() != 32 || be.unsafeFrees != 0 {
			t.Fatal("release changed pin/budget safety")
		}
		if r.asyncWaited != 1 || r.asyncOverlapped != 0 {
			t.Fatal("discarded victim earned overlap credit")
		}
		scope.finish()
		r.freeAll()
	})
	t.Run("failed admission retains or safely retires its orphan", func(t *testing.T) {
		for _, fail := range []bool{false, true} {
			be := newRetirementTestBackend()
			if fail {
				be.result = compute.OwnedTransferResult{State: compute.OwnedTransferUnknown, Err: context.Canceled}
			}
			r := newPagedRing(be, 16)
			s := &Session{Backend: be}
			scope := s.beginBackendExecution()
			s.ringEnter(r)()
			_, err := r.stageWithError("incoming", func() compute.Tensor {
				// Inject a changed admission snapshot after the initial precheck.
				if _, e := r.pool.Admit(polymodel.Model{ID: "pin", WeightBytes: 16, Pinned: true}); e != nil {
					panic(e)
				}
				r.resident["pin"] = mk()
				return mk()
			}, compute.F32, 16, false)
			if err == nil || r.isResident("incoming") || r.pool.Has("incoming") || !r.isResident("pin") || r.used() != 16 {
				t.Fatal("failed admission changed residency")
			}
			if fail {
				if !errors.Is(err, context.Canceled) || len(r.pending) != 1 || be.frees != 0 {
					t.Fatal("uncertain rollback was freed or forgotten")
				}
			} else if !errors.Is(err, polymodel.ErrPinnedNoRoom) || len(r.pending) != 0 || be.frees != 1 {
				t.Fatal("completed rollback did not free exactly once")
			}
			if r.asyncOverlapped != 0 || r.asyncWaited != 0 || be.unsafeFrees != 0 {
				t.Fatal("rollback earned overlap or freed live storage")
			}
			retirementRecover(scope.finish)
			requireOwnerReleased(t, be.owner)
		}
	})
	t.Run("victim Free panic cannot roll back admitted transfer", func(t *testing.T) {
		be := newRetirementTestBackend()
		r := newPagedRing(be, 16)
		s := &Session{Backend: be}
		scope := s.beginBackendExecution()
		defer func() {
			if scope.live {
				retirementRecover(scope.finish)
			}
		}()
		s.ringEnter(r)()
		old, err := r.stageWithError("old", mk, compute.F32, 16, false)
		if err != nil {
			t.Fatal(err)
		}
		be.panicFree = old.Buf()
		if retirementRecover(func() { r.stageWithError("new", mk, compute.F32, 16, false) }) == nil {
			t.Fatal("victim Free was not exercised")
		}
		incoming, resident := r.resident["new"]
		if !resident || !r.pool.Has("new") || r.used() != 16 {
			t.Fatal("admitted incoming tensor lost its resident binding")
		}
		if retirementRecover(scope.finish) != nil {
			t.Fatal("completed incoming transfer could not drain")
		}
		if !r.isResident("new") || !r.pool.Has("new") {
			t.Fatal("scope drain rolled back admitted tensor")
		}
		for _, freed := range be.freed {
			if freed == incoming.Buf() {
				t.Fatal("scope drain freed an admitted tensor")
			}
		}
		// The incumbent victim-Free failure itself remains outside this transfer
		// proof; this fixture never retries that uncertain native Free.
	})
	t.Run("real routed consumer uses checked completion", func(t *testing.T) {
		const H, E, K = 256, 4, 2
		m := expertPrefetchModel(t, H, E, K)
		budget := expertRingWeightBytes(t, m) * 3 * K
		input := expertRingTestInput(H)
		control, _ := expertPrefetchSession(m, budget)
		want := moeFFN{}.apply(m, 0, input, sessionQ4KKernel{s: control})
		control.Close()
		be := newRetirementTestBackend()
		s := &Session{M: m, Backend: be, Q4K: true, halW: map[string]compute.Tensor{}, ExpertRingBytes: budget}
		scope := s.beginBackendExecution()
		var got []float32
		func() { defer scope.finish(); got = moeFFN{}.apply(m, 0, input, sessionQ4KKernel{s: s}) }()
		if len(be.checked) == 0 || be.unfencedGEMMs != 0 {
			t.Fatal("routed consumer did not await checked copies")
		}
		if len(got) != len(want) {
			t.Fatal("output size changed")
		}
		for i := range got {
			if got[i] != want[i] {
				t.Fatalf("output[%d] changed", i)
			}
		}
		for _, fence := range be.checked {
			if fence.legacyCalls != 0 {
				t.Fatal("legacy fence substituted for checked completion")
			}
			d := fence.transfer.Descriptor()
			if d.SourceOwner != nil || d.DestinationOwner != nil {
				t.Fatal("completed transfer did not retire its roots")
			}
			if fence.handoffs != 1 || fence.returned.SourceOwner == nil || fence.returned.DestinationOwner == nil {
				t.Fatal("retention responsibility was not handed back exactly once")
			}
		}
		s.Close()
		if be.unsafeFrees != 0 {
			t.Fatal("close freed an in-flight destination")
		}
	})
}

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
