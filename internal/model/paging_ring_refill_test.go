package model

import (
	"errors"
	"fmt"
	"math"
	"reflect"
	"runtime"
	"slices"
	"sync"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/polymodel"
)

// A synchronous host provider exercises the real ring/HAL, not CUDA. Its slot
// owns an independently copied payload; replacement changes the handle while
// retaining exactly the same payload allocation. Old handles are rejected.
type refillTestSlot struct {
	f   []float32
	raw []byte
}
type refillTestBackend struct {
	*expertHALRecordingBackend
	mu                                  sync.Mutex
	enabled, reject                     bool
	slots                               map[compute.Buffer]*refillTestSlot
	invalid                             map[compute.Buffer]bool
	names                               map[compute.Buffer]string
	retained                            []compute.Tensor
	allocations, refills, probes, frees int
	victims                             []string
	before                              func(compute.Tensor, compute.Tensor)
	mode                                string
	failure                             error
}

func newRefillTestBackend(enabled bool) *refillTestBackend {
	return &refillTestBackend{
		expertHALRecordingBackend: &expertHALRecordingBackend{Backend: compute.Default(), uploads: map[compute.Dtype]int{}},
		enabled:                   enabled, slots: map[compute.Buffer]*refillTestSlot{}, invalid: map[compute.Buffer]bool{}, names: map[compute.Buffer]string{},
	}
}
func (b *refillTestBackend) ForwardExecutionOwner() *sync.Mutex { return &b.mu }
func (b *refillTestBackend) UploadClass(src compute.Tensor, dt compute.Dtype, class compute.MemoryClass, site string) compute.Tensor {
	if class != compute.MemoryOffload {
		return b.Upload(src, dt)
	}
	hb := src.Buf().(compute.HostBuffer)
	slot := &refillTestSlot{}
	var out compute.Tensor
	switch dt {
	case compute.F32:
		slot.f = slices.Clone(hb.F32())
		out = compute.NewF32(b, slices.Clone(src.Shape), slot.f)
	case compute.Q4_K:
		slot.raw = make([]byte, len(hb.I8()))
		for i, v := range hb.I8() {
			slot.raw[i] = byte(v)
		}
		out = compute.NewQ4K(b, slices.Clone(src.Shape), slot.raw)
	default:
		panic("unsupported test weight")
	}
	b.allocations++
	b.slots[out.Buf()] = slot
	return out
}
func (b *refillTestBackend) OffloadRefillEnabled() bool { return b.enabled }
func (b *refillTestBackend) CanRefillOffload(dst, src compute.Tensor, bytes int64) bool {
	b.probes++
	return !b.reject && b.slots[dst.Buf()] != nil && expertRingRefillGeometry(dst, src)
}
func (b *refillTestBackend) RefillOffload(dst, src compute.Tensor, bytes int64) (compute.Tensor, compute.OwnedTransferResult) {
	slot := b.slots[dst.Buf()]
	if slot == nil {
		panic("unowned slot")
	}
	b.refills++
	b.victims = append(b.victims, b.names[dst.Buf()])
	b.retained = append(b.retained, dst, src) // Independent quarantine before any callback/effect.
	delete(b.slots, dst.Buf())
	b.invalid[dst.Buf()] = true // Retire old backend identity before the copy.
	if b.before != nil {
		b.before(dst, src)
	}
	hb := src.Buf().(compute.HostBuffer)
	var out compute.Tensor
	if dst.Dtype == compute.F32 {
		copy(slot.f, hb.F32())
		out = compute.NewF32(b, slices.Clone(dst.Shape), slot.f)
	} else {
		for i, v := range hb.I8() {
			slot.raw[i] = byte(v)
		}
		out = compute.NewQ4K(b, slices.Clone(dst.Shape), slot.raw)
	}
	b.retained = append(b.retained, out)
	oldHost, newHost := dst.Buf().(compute.HostBuffer), out.Buf().(compute.HostBuffer)
	if dst.Dtype == compute.F32 {
		if reflect.ValueOf(oldHost.F32()).Pointer() != reflect.ValueOf(newHost.F32()).Pointer() {
			panic("fresh destination allocation")
		}
	} else if reflect.ValueOf(oldHost.I8()).Pointer() != reflect.ValueOf(newHost.I8()).Pointer() {
		panic("fresh destination allocation")
	}
	switch b.mode {
	case "panic":
		panic("refill interrupted after copy")
	case "goexit":
		runtime.Goexit()
	case "failed":
		return out, compute.OwnedTransferResult{State: compute.OwnedTransferFailed, Err: b.failure}
	case "unknown":
		return out, compute.OwnedTransferResult{State: compute.OwnedTransferUnknown}
	case "pending":
		return out, compute.OwnedTransferResult{State: compute.OwnedTransferPending}
	case "same-handle":
		return dst, compute.OwnedTransferResult{State: compute.OwnedTransferComplete}
	case "old-buffer-new-quant":
		out = dst
		q := *dst.Quant
		out.Quant = &q
		return out, compute.OwnedTransferResult{State: compute.OwnedTransferComplete}
	}
	b.slots[out.Buf()] = slot
	b.retained = nil
	return out, compute.OwnedTransferResult{State: compute.OwnedTransferComplete}
}
func (b *refillTestBackend) Free(t compute.Tensor) {
	if b.invalid[t.Buf()] {
		panic("free of retired refill handle")
	}
	if b.slots[t.Buf()] != nil {
		b.frees++
		b.victims = append(b.victims, b.names[t.Buf()])
		delete(b.slots, t.Buf())
	}
	b.Backend.Free(t)
}
func (b *refillTestBackend) MatMul(w, x compute.Tensor) compute.Tensor {
	if b.invalid[w.Buf()] {
		panic("read of retired refill handle")
	}
	return b.expertHALRecordingBackend.MatMul(w, x)
}
func refillTestPanic(t *testing.T, f func()) any {
	t.Helper()
	var got any
	normal := false
	func() { defer func() { got = recover() }(); f(); normal = true }()
	if normal || got == nil {
		t.Fatal("expected a nonnil panic")
	}
	return got
}
func refillTestSession(m *Model, b *refillTestBackend, budget int64, policy ExpertRingEvictPolicy) *Session {
	return &Session{M: m, Backend: b, Q4K: true, halW: map[string]compute.Tensor{}, ExpertRingBytes: budget, ExpertRingEvict: policy}
}
func refillTestRun(s *Session, e int, x []float32) []float32 {
	scope := s.beginBackendExecution()
	defer scope.finish()
	out := expertSwiGLU(s.M, 0, e, x, sessionQ4KKernel{s: s})
	b := s.Backend.(*refillTestBackend)
	for id, wt := range s.expertRing.resident {
		b.names[wt.Buf()] = string(id)
	}
	return out
}
func refillTestDirect(b *refillTestBackend, budget int64) *pagedRing {
	r := newPagedRing(b, budget)
	// Single-threaded fixture emulates an already acquired domain, without native work.
	r.drainScope = &backendExecutionScope{owner: b.ForwardExecutionOwner(), live: true}
	return r
}
func refillTestF32(v float32) func() compute.Tensor {
	return func() compute.Tensor { return compute.NewF32(compute.Default(), []int{1, 2}, []float32{v, -v}) }
}

// fak-test:runtime medium est=2s lane=default
func TestExpertRingStaticRefill(t *testing.T) {
	t.Run("real-routed-consumer-and-policy", func(t *testing.T) {
		const H = 256
		m := expertRingTestModel(t, H, 4)
		x := expertRingTestInput(H)
		bytes := expertRingWeightBytes(t, m)
		for _, policy := range []ExpertRingEvictPolicy{ExpertRingEvictLRU, ExpertRingEvictValueAware} {
			t.Run(policy.String(), func(t *testing.T) {
				baseBE, refillBE := newRefillTestBackend(false), newRefillTestBackend(true)
				base, got := refillTestSession(m, baseBE, 6*bytes, policy), refillTestSession(m, refillBE, 6*bytes, policy)
				refillBE.before = func(old, src compute.Tensor) {
					r := got.expertRing
					oldName := refillBE.names[old.Buf()]
					if r.isResident(oldName) || r.isResident(string(r.refill.id)) {
						t.Fatal("old/new name visible during overwrite")
					}
					if _, live := r.pool.Get(polymodel.ModelID(oldName)); live {
						t.Fatal("old pool identity survived")
					}
					if r.used() != 6*bytes || r.refill == nil || !expertRingSameTensor(r.refill.destination, old) ||
						!expertRingSameTensor(r.refill.source, src) {
						t.Fatal("missing roots or detached-byte charge")
					}
				}
				for step, e := range []int{0, 1, 0, 2, 3, 2, 0, 1} {
					want := refillTestRun(base, e, x)
					out := refillTestRun(got, e, x)
					if !slices.Equal(want, out) {
						t.Fatalf("step %d: routed output differs", step)
					}
					for _, v := range out {
						if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
							t.Fatal("nonfinite output")
						}
					}
					if !slices.Equal(base.expertRing.pool.Resident(), got.expertRing.pool.Resident()) {
						t.Fatal("victim membership changed")
					}
					a, z := base.ExpertRing(), got.ExpertRing()
					if a.PageIns != z.PageIns || a.Hits != z.Hits || a.Evictions != z.Evictions || a.PageInBytes != z.PageInBytes ||
						a.ResidentBytes != z.ResidentBytes || z.ResidentBytes > z.BudgetBytes || z.Lookups != z.PageIns+z.Hits+z.Refusals {
						t.Fatalf("accounting changed: base=%+v refill=%+v", a, z)
					}
				}
				if refillBE.refills == 0 || refillBE.allocations != 6 || baseBE.allocations <= 6 ||
					!slices.Equal(baseBE.victims, refillBE.victims) {
					t.Fatalf("reuse/order: alloc=%d/%d refill=%d victims=%v/%v",
						baseBE.allocations, refillBE.allocations, refillBE.refills, baseBE.victims, refillBE.victims)
				}
				if refillBE.frees != 0 || baseBE.probes != 0 {
					t.Fatal("opt-out or live-slot free contract")
				}
				base.expertRing.freeAll()
				got.expertRing.freeAll()
				if len(refillBE.slots) != 0 || got.expertRing.used() != 0 {
					t.Fatal("successful teardown leaked slots")
				}
			})
		}
	})
	t.Run("portable-hal-does-not-fallback", func(t *testing.T) {
		m := expertRingTestModel(t, 256, 2)
		x := expertRingTestInput(256)
		b := newRefillTestBackend(true)
		s := refillTestSession(m, b, 3*expertRingWeightBytes(t, m), ExpertRingEvictLRU)
		refillTestRun(s, 0, x)
		allocs := b.allocations
		b.mode = "failed"
		b.failure = errors.New("selected copy failure")
		refillTestPanic(t, func() { refillTestRun(s, 1, x) })
		if b.allocations != allocs || b.refills != 1 || s.expertRing.refill == nil {
			t.Fatal("portable HAL fell back after selection")
		}
		for name := range s.halW {
			if isRoutedExpertWeight(name) {
				t.Fatal("failed refill became permanent residency")
			}
		}
	})
	t.Run("real-scope-goexit-retains-before-unlock", func(t *testing.T) {
		m := expertRingTestModel(t, 256, 2)
		x := expertRingTestInput(256)
		bytes := 3 * expertRingWeightBytes(t, m)
		b := newRefillTestBackend(true)
		s := refillTestSession(m, b, bytes, ExpertRingEvictLRU)
		refillTestRun(s, 0, x)
		b.mode = "goexit"
		done := make(chan struct{})
		var got any
		normal := false
		go func() {
			defer close(done)
			defer func() { got = recover() }()
			refillTestRun(s, 1, x)
			normal = true
		}()
		select {
		case <-done:
		case <-time.After(2 * time.Second):
			t.Fatal("owned Goexit hung")
		}
		// The existing legacy scope.finish converts its terminal drain failure into
		// a cleanup panic during Goexit. Pin that boundary; do not mislabel it as
		// end-to-end Goexit preservation. No successful output may escape either way.
		err, ok := got.(error)
		if normal || !ok || !expertRingIsTransferError(err) || s.backendExecutionScope != nil ||
			s.expertRing.refill == nil || s.expertRing.refill.failure == nil || s.expertRing.used() != bytes ||
			len(b.retained) != 3 || s.expertRing.drainScope != nil {
			t.Fatal("owned unwind lost failure/roots")
		}
		if !b.mu.TryLock() {
			t.Fatal("execution domain stranded")
		}
		b.mu.Unlock()
		allocs, copies, frees := b.allocations, b.refills, b.frees
		if allocs != 3 || copies != 1 {
			t.Fatal("owned failure fell back")
		}
		p := s.expertRing.refill
		refillTestPanic(t, func() { s.expertRing.stageWithError("retry", refillTestF32(1), compute.F32, 8, false) })
		if s.expertRing.refill != p || s.expertRing.used() != bytes || b.allocations != allocs || b.refills != copies || b.frees != frees {
			t.Fatal("owned terminal retry changed roots")
		}
	})
	t.Run("refusals-and-repin-order", func(t *testing.T) {
		b := newRefillTestBackend(true)
		r := refillTestDirect(b, 16)
		a, _ := r.stage("a", refillTestF32(1), compute.F32, 8, false)
		r.stage("b", refillTestF32(2), compute.F32, 8, false)
		r.hold("a")
		r.release("a") // Changes pool recency; separate ring clocks are not a victim oracle.
		var chosen compute.Buffer
		b.before = func(dst, src compute.Tensor) { chosen = dst.Buf() }
		r.stage("c", refillTestF32(3), compute.F32, 8, false)
		if chosen == nil || chosen == a.Buf() || !r.isResident("a") || r.isResident("b") {
			t.Fatal("repin victim changed")
		}
		// A live hold prevents any overwrite of a. A durable pin prevents c too.
		r.hold("a")
		desc, _ := r.pool.Get("c")
		desc.Pinned = true
		r.repin("c", desc)
		before := b.refills
		builds := 0
		if _, err := r.stageWithError("d", func() compute.Tensor { builds++; return refillTestF32(4)() }, compute.F32, 8, false); !errors.Is(err, polymodel.ErrPinnedNoRoom) || builds != 0 || b.refills != before {
			t.Fatal("held/pinned preflight")
		}
		r.release("a")
		b.reject = true
		r.stage("d", refillTestF32(4), compute.F32, 8, false)
		if b.refills != before || b.allocations != 3 {
			t.Fatal("negative capability changed incumbent upload")
		}
	})
	t.Run("geometry-and-domain-fallback", func(t *testing.T) {
		for _, mode := range []string{"shape", "unowned", "shared"} {
			b := newRefillTestBackend(true)
			r := refillTestDirect(b, 8)
			r.stage("a", refillTestF32(1), compute.F32, 8, false)
			mk := refillTestF32(2)
			switch mode {
			case "shape":
				mk = func() compute.Tensor { return compute.NewF32(compute.Default(), []int{2, 1}, []float32{2, -2}) }
			case "unowned":
				r.drainScope = nil
			case "shared":
				r.shared = &SharedExpertRing{}
			}
			// A synthetic empty shared owner has bookkeeping requirements unrelated to
			// this test; check the pure selection refusal without stage bookkeeping.
			if mode == "shared" {
				if _, used := r.tryRefill(mk(), polymodel.Model{ID: "b", WeightBytes: 8}, compute.F32); used || b.probes != 0 {
					t.Fatal("shared refill")
				}
				continue
			}
			r.stage("b", mk, compute.F32, 8, false)
			if b.refills != 0 || b.probes != 0 || b.allocations != 2 {
				t.Fatalf("%s did not retain incumbent", mode)
			}
		}
		a := compute.NewQ8(compute.Default(), []int{1, 32}, make([]int8, 32), []float32{1}, 32)
		if expertRingRefillGeometry(a, a) {
			t.Fatal("side-channel scales admitted")
		}
		// Absent provider remains the original ring path.
		r := newPagedRing(compute.Default(), 8)
		r.stage("a", refillTestF32(1), compute.F32, 8, false)
		r.stage("b", refillTestF32(2), compute.F32, 8, false)
		if !r.isResident("b") || r.isResident("a") || r.refill != nil {
			t.Fatal("absent capability")
		}
	})
	t.Run("pending-victim-settles-before-copy", func(t *testing.T) {
		for _, fail := range []bool{false, true} {
			b := newRefillTestBackend(true)
			r := refillTestDirect(b, 8)
			old, _ := r.stage("a", refillTestF32(1), compute.F32, 8, false)
			fence := &asyncTestFence{release: func() {
				if fail {
					panic("pending copy failed")
				}
			}}
			if err := r.retainStaged("a", refillTestF32(1)(), old, fence); err != nil {
				t.Fatal(err)
			}
			b.before = func(dst, src compute.Tensor) {
				if fence.waits != 1 || len(r.pending) != 0 {
					t.Fatal("copy before settlement")
				}
			}
			if fail {
				refillTestPanic(t, func() { r.stage("b", refillTestF32(2), compute.F32, 8, false) })
				if b.refills != 0 || r.refill == nil || r.used() != 8 || !r.isResident("a") {
					t.Fatal("failed pending victim was modified")
				}
			} else {
				r.stage("b", refillTestF32(2), compute.F32, 8, false)
				if b.refills != 1 || fence.waits != 1 {
					t.Fatal("missing settlement/refill")
				}
			}
		}
	})
	t.Run("old-buffer-new-quant", func(t *testing.T) {
		b := newRefillTestBackend(true)
		r := refillTestDirect(b, 144)
		mk := func() compute.Tensor { return compute.NewQ4K(compute.Default(), []int{1, 256}, make([]byte, 144)) }
		old, _ := r.stage("a", mk, compute.Q4_K, 144, false)
		alias := old
		q := *old.Quant
		alias.Quant = &q
		if _, used := r.tryRefill(alias, polymodel.Model{ID: "alias", WeightBytes: 144}, compute.Q4_K); used || b.probes != 0 || r.refill != nil || !r.isResident("a") {
			t.Fatal("metadata disguised source alias")
		}
		b.mode = "old-buffer-new-quant"
		refillTestPanic(t, func() { r.stageWithError("b", mk, compute.Q4_K, 144, false) })
		p := r.refill
		if p == nil || p.failure == nil || p.replacement.Buf() != old.Buf() || p.replacement.Quant == old.Quant ||
			!expertRingRefillGeometry(old, p.replacement) || expertRingSameTensor(old, p.replacement) ||
			r.isResident("a") || r.isResident("b") || r.used() != 144 {
			t.Fatal("metadata disguised old handle")
		}
	})
	t.Run("failure-and-abnormal-retention", func(t *testing.T) {
		for _, mode := range []string{"failed", "unknown", "pending", "same-handle", "panic", "goexit", "caught-reentry"} {
			t.Run(mode, func(t *testing.T) {
				b := newRefillTestBackend(true)
				r := refillTestDirect(b, 8)
				old, _ := r.stage("a", refillTestF32(1), compute.F32, 8, false)
				b.mode = mode
				b.failure = fmt.Errorf("checked refill failed")
				if mode == "caught-reentry" {
					b.before = func(dst, src compute.Tensor) {
						refillTestPanic(t, func() { r.stage("a", refillTestF32(1), compute.F32, 8, false) })
					}
				}
				normal := false
				if mode == "goexit" {
					done := make(chan struct{})
					var recovered any
					go func() {
						defer close(done)
						defer func() { recovered = recover() }()
						r.stageWithError("b", refillTestF32(2), compute.F32, 8, false)
						normal = true
					}()
					select {
					case <-done:
					case <-time.After(2 * time.Second):
						t.Fatal("Goexit hung")
					}
					if normal || recovered != nil {
						t.Fatal("Goexit control transfer changed")
					}
				} else {
					refillTestPanic(t, func() { r.stage("b", refillTestF32(2), compute.F32, 8, false); normal = true })
				}
				p := r.refill
				if normal || p == nil || p.failure == nil || !p.reserved || !expertRingSameTensor(p.destination, old) ||
					p.source.Buf() == nil || r.used() != 8 || r.isResident("a") || r.isResident("b") {
					t.Fatal("failure roots/charge/visibility")
				}
				if mode == "failed" && !errors.Is(p.failure, b.failure) {
					t.Fatal("first cause lost")
				}
				allocs, copies, frees := b.allocations, b.refills, b.frees
				refillTestPanic(t, func() { r.stage("a", refillTestF32(1), compute.F32, 8, false) })
				refillTestPanic(t, func() { r.freeAll() })
				refillTestPanic(t, func() { r.settlePendingTransfers() })
				if b.allocations != allocs || b.refills != copies || b.frees != frees || r.refill != p || r.used() != 8 {
					t.Fatal("terminal retry/fallback/free or lost roots")
				}
			})
		}
	})
}
