package model

import (
	"sync"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// expert_ring_async_owner_test.go — the ring routes a backend that is BOTH placement-aware
// (UploadClass) and recording-domain owned (ForwardExecutionOwnerBackend) — the CUDA shape —
// through compute.ClassedAsyncUploader, but only while a drain is joined to the live execution
// scope, so no in-flight transfer outlives the lease that issued it.

// ownedClassedAsyncBackend is expertAsyncBackend plus the two capabilities that used to force the
// ring synchronous: a recording-domain owner and classed placement.
type ownedClassedAsyncBackend struct {
	*expertAsyncBackend
	owner      *sync.Mutex
	classes    []compute.MemoryClass
	plainAsync int // UploadAsync calls, which drop the class label; must stay 0
}

func newOwnedClassedAsyncBackend(owner *sync.Mutex, landImmediately bool) *ownedClassedAsyncBackend {
	return &ownedClassedAsyncBackend{expertAsyncBackend: newExpertAsyncBackend(landImmediately), owner: owner}
}

func (b *ownedClassedAsyncBackend) ForwardExecutionOwner() *sync.Mutex { return b.owner }

func (b *ownedClassedAsyncBackend) UploadClass(t compute.Tensor, as compute.Dtype, _ compute.MemoryClass, _ string) compute.Tensor {
	return b.expertAsyncBackend.Upload(t, as)
}

func (b *ownedClassedAsyncBackend) UploadAsync(t compute.Tensor, as compute.Dtype) (compute.Tensor, compute.Fence) {
	b.plainAsync++
	return b.expertAsyncBackend.UploadAsync(t, as)
}

func (b *ownedClassedAsyncBackend) UploadAsyncClass(t compute.Tensor, as compute.Dtype, class compute.MemoryClass, _ string) (compute.Tensor, compute.Fence) {
	b.classes = append(b.classes, class)
	return b.expertAsyncBackend.UploadAsync(t, as)
}

var (
	_ compute.ClassedAsyncUploader         = (*ownedClassedAsyncBackend)(nil)
	_ compute.ForwardExecutionOwnerBackend = (*ownedClassedAsyncBackend)(nil)
	_ classedUploadBackend                 = (*ownedClassedAsyncBackend)(nil)
)

func ownedClassedAsyncSession(m *Model, ringBytes int64, be *ownedClassedAsyncBackend) *Session {
	return &Session{
		M: m, Backend: be, Q4K: true, halW: map[string]compute.Tensor{},
		ExpertRingBytes: ringBytes,
	}
}

func requireOwnerReleased(t *testing.T, owner *sync.Mutex) {
	t.Helper()
	if !owner.TryLock() {
		t.Fatal("execution owner still held after the scope finished")
	}
	owner.Unlock()
}

// fak-test:runtime fast est=200ms lane=default
func TestExpertRingAsyncOnOwnedClassedBackendOverlapsInsideTheScope(t *testing.T) {
	const H, E, K = 256, 8, 4
	m := expertPrefetchModel(t, H, E, K)
	perWeight := expertRingWeightBytes(t, m)
	var owner sync.Mutex
	be := newOwnedClassedAsyncBackend(&owner, true)
	s := ownedClassedAsyncSession(m, perWeight*3*K, be)

	scope := s.beginBackendExecution()
	func() {
		defer scope.finish()
		moeFFN{}.apply(m, 0, expertRingTestInput(H), sessionQ4KKernel{s: s})
	}()
	requireOwnerReleased(t, &owner)

	if len(be.fences) == 0 {
		t.Fatal("no transfer went async on an owned classed backend inside a live scope")
	}
	if be.plainAsync != 0 {
		t.Fatalf("%d unlabelled UploadAsync call(s) on a placement-aware backend", be.plainAsync)
	}
	for i, c := range be.classes {
		if c != compute.MemoryOffload {
			t.Fatalf("async upload %d booked as %q, want %q", i, c, compute.MemoryOffload)
		}
	}
	if be.unfencedGEMMs != 0 {
		t.Fatalf("%d GEMM(s) ran against a weight whose transfer had not landed", be.unfencedGEMMs)
	}
	if st := s.ExpertRing(); st.AsyncOverlapped == 0 || st.AsyncOverlapFraction() <= 0 {
		t.Fatalf("overlapped=%d waited=%d, want overlap > 0", st.AsyncOverlapped, st.AsyncWaited)
	}
	s.Close()
	requireOwnerReleased(t, &owner)
}

// Without a live scope there is no drain to join, so the same backend stays synchronous.
// fak-test:runtime fast est=200ms lane=default
func TestExpertRingAsyncOnOwnedBackendStaysSynchronousOutsideAScope(t *testing.T) {
	const H, E, K = 256, 8, 4
	m := expertPrefetchModel(t, H, E, K)
	perWeight := expertRingWeightBytes(t, m)
	var owner sync.Mutex
	be := newOwnedClassedAsyncBackend(&owner, true)
	s := ownedClassedAsyncSession(m, perWeight*3*K, be)
	defer s.Close()

	moeFFN{}.apply(m, 0, expertRingTestInput(H), sessionQ4KKernel{s: s})
	if len(be.fences) != 0 || be.plainAsync != 0 {
		t.Fatalf("unjoined owned ring issued %d async transfer(s) (%d unlabelled)", len(be.fences), be.plainAsync)
	}
	if s.ExpertRing().Prefetched == 0 {
		t.Fatal("nothing was prefetched; the comparison is empty")
	}
}

// An undemanded prefetch still in flight when the forward ends is settled before the recording
// domain is released, without booking the overlap meter, and the stale arming is inert afterwards.
// fak-test:runtime fast est=100ms lane=default
func TestExpertRingScopeReleaseDrainsUndemandedTransfers(t *testing.T) {
	var owner sync.Mutex
	be := newOwnedClassedAsyncBackend(&owner, false) // nothing lands until someone Waits
	s := &Session{Backend: be}
	r := newPagedRing(be, 1<<20)
	mk := func() compute.Tensor { return compute.NewF32(compute.Default(), []int{4}, []float32{1, 2, 3, 4}) }

	scope := s.beginBackendExecution()
	func() {
		defer scope.finish()
		s.ringEnter(r)()
		if _, ok := r.stageInFlight("w0", mk, compute.F32, 16, false); !ok {
			t.Fatal("stageInFlight refused")
		}
		if len(r.pending) != 1 || len(be.fences) != 1 || be.fences[0].landed {
			t.Fatalf("pending=%d fences=%d: the prefetch did not stay in flight", len(r.pending), len(be.fences))
		}
	}()
	requireOwnerReleased(t, &owner)
	if f := be.fences[0]; !f.landed || f.waits != 1 {
		t.Fatalf("in-flight transfer not settled at release (landed=%v waits=%d)", f.landed, f.waits)
	}
	if len(r.pending) != 0 || r.drainScope != nil {
		t.Fatalf("pending=%d drainArmed=%v after release", len(r.pending), r.drainScope != nil)
	}
	if r.asyncOverlapped != 0 || r.asyncWaited != 0 {
		t.Fatalf("drain booked the overlap meter: overlapped=%d waited=%d", r.asyncOverlapped, r.asyncWaited)
	}
	if _, ok := r.stageInFlight("w1", mk, compute.F32, 16, false); !ok || len(be.fences) != 1 || len(r.pending) != 0 {
		t.Fatalf("post-release staging went async (fences=%d pending=%d)", len(be.fences), len(r.pending))
	}
	r.freeAll()
}
