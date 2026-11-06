package compute

import (
	"errors"
	"math"
	"runtime"
	"testing"
	"time"
	"unsafe"
)

type kvFP8OwnerFake struct {
	allocs, releases                      int
	failAlloc, failRelease                int
	failAfterAllocation, retiredOnFailure bool
}

func (f *kvFP8OwnerFake) ops() kvFP8OwnerOps {
	return kvFP8OwnerOps{
		allocate: func(dst *kvFP8Allocation, bytes, layer, plane int) error {
			f.allocs++
			if f.allocs == f.failAlloc && !f.failAfterAllocation {
				return errKVFP8Budget
			}
			dst.ptr = unsafe.Pointer(new(byte)) // identity only; never dereferenced as device data
			if f.allocs == f.failAlloc {
				return errKVFP8Budget
			}
			return nil
		},
		release: func(a kvFP8Allocation) kvFP8Retirement {
			f.releases++
			if f.releases == f.failRelease {
				return kvFP8Retirement{retired: f.retiredOnFailure, status: 7}
			}
			return kvFP8Retirement{retired: true}
		},
	}
}

func kvFP8OwnerComplete(_, _ *kvFP8Image) kvFP8Completion {
	return kvFP8Completion{completed: true, numericalChecked: true}
}

func kvFP8OwnerFixture(t *testing.T) *kvFP8Owner {
	t.Helper()
	o, err := newKVFP8Owner(new(kvFP8BackendIdentity), 0, 2, 4, 256)
	if err != nil {
		t.Fatal(err)
	}
	return o
}

// fak-test:runtime fast est=25ms lane=default
func TestKVFP8OwnerTransaction(t *testing.T) {
	// No parallelism: fake quarantine intentionally exercises the package root.
	saved := kvFP8RetainedOwners
	defer func() { kvFP8RetainedOwners = saved }()
	t.Run("every allocation slot rolls back including scales", func(t *testing.T) {
		for fail := 1; fail <= 10; fail++ {
			for _, after := range []bool{false, true} {
				o := kvFP8OwnerFixture(t)
				view, _ := o.borrow(0, kvFP8Key)
				f := &kvFP8OwnerFake{failAlloc: fail, failAfterAllocation: after}
				err := o.replace(3, nil, 1<<30, f.ops(), kvFP8OwnerComplete, false)
				want := fail - 1
				if after {
					want++
				}
				if !errors.Is(err, errKVFP8Budget) || f.releases != want || o.allocatedBytes() != 0 ||
					o.image.positions != 0 || !view.Ready() || o.usable() != nil {
					t.Fatalf("slot=%d after=%v err=%v releases=%d want=%d", fail, after, err, f.releases, want)
				}
			}
		}
	})
	t.Run("partial rotary mapping and numerical eviction rollback", func(t *testing.T) {
		o := kvFP8OwnerFixture(t)
		linear := []bool{true, true, true, false, true, true, true, false}
		theta := []float64{1, 1, 1, 10000, 1, 1, 1, 1000000}
		if err := o.bindQwenRotary(linear, theta, 64); err != nil {
			t.Fatal(err)
		}
		if o.rotary[0].mainLayer != 3 || o.rotary[1].mainLayer != 7 ||
			o.rotary[0].thetaBits != math.Float64bits(10000) ||
			o.rotary[1].thetaBits != math.Float64bits(1000000) {
			t.Fatal("compact rotary mapping lost per-layer identity")
		}
		theta[7] = 20000
		before := o.rotary[1]
		if o.bindQwenRotary(linear, theta, 64) == nil || o.rotary[1] != before {
			t.Fatal("conflicting rebind")
		}
		f := new(kvFP8OwnerFake)
		if err := o.replace(3, nil, 1<<30, f.ops(), kvFP8OwnerComplete, false); err != nil {
			t.Fatal(err)
		}
		old := o.image
		view, _ := o.borrow(0, kvFP8Key)
		plan, err := o.planEvict(1, 1)
		if err != nil {
			t.Fatal(err)
		}
		for row, want := range []int{0, 2} {
			src, rotate, err := plan.sourceRow(row)
			if err != nil || src != want || rotate != (row == 1) {
				t.Fatal("bad moved-row mapping")
			}
		}
		err = plan.apply(nil, 1<<30, f.ops(), func(dst, src *kvFP8Image, p *kvFP8EvictionPlan) kvFP8Completion {
			if src != old || dst.positions != 2 || len(dst.layers) != 2 ||
				p.rotary[1].rotaryDim != 64 || p.rotary[1].thetaBits != before.thetaBits {
				t.Fatal("stage did not receive complete bound owner")
			}
			if view.Ready() {
				t.Fatal("pending producer became ready")
			}
			return kvFP8Completion{completed: true, numericalChecked: true, numerical: 1}
		})
		if !errors.Is(err, ErrKVFP8NonFinite) || o.image != old || !view.Ready() || o.stage != nil {
			t.Fatal("rejected stage changed published owner")
		}
		if err := plan.apply(nil, 1<<30, f.ops(), func(dst, src *kvFP8Image, p *kvFP8EvictionPlan) kvFP8Completion {
			return kvFP8OwnerComplete(dst, src)
		}); err != nil {
			t.Fatal(err)
		}
		if view.Ready() || o.image.positions != 2 {
			t.Fatal("successful eviction did not invalidate old view")
		}
		allocs, releases := f.allocs, f.releases
		image, stamp := o.image, o.stamp.Load()
		stageCalls := 0
		if err := plan.apply(nil, 1<<30, f.ops(), func(dst, src *kvFP8Image, p *kvFP8EvictionPlan) kvFP8Completion {
			stageCalls++
			return kvFP8OwnerComplete(dst, src)
		}); !errors.Is(err, errKVFP8Owner) {
			t.Fatal("stale plan reused", err)
		}
		if stageCalls != 0 || f.allocs != allocs || f.releases != releases || o.image != image || o.stamp.Load() != stamp {
			t.Fatal("stale plan refusal happened after effects")
		}
		if err := o.free(f.ops()); err != nil || o.allocatedBytes() != 0 {
			t.Fatal("five-plane free", err)
		}
	})
	t.Run("unknown rotary and hd65 cannot masquerade as Qwen", func(t *testing.T) {
		o := kvFP8OwnerFixture(t)
		f := new(kvFP8OwnerFake)
		if err := o.replace(3, nil, 1<<30, f.ops(), kvFP8OwnerComplete, false); err != nil {
			t.Fatal(err)
		}
		if _, err := o.planEvict(0, 1); !errors.Is(err, errKVFP8Rotary) {
			t.Fatal("unknown moved-row contract")
		}
		if _, err := o.planEvict(2, 1); err != nil {
			t.Fatal("tail removal needs no re-RoPE")
		}
		linear := []bool{true, true, true, false, true, true, true, false}
		theta := []float64{1, 1, 1, 10000, 1, 1, 1, 1000000}
		if o.bindQwenRotary(linear, theta, 64) == nil {
			t.Fatal("retroactive populated binding")
		}
		tail, _ := newKVFP8Owner(new(kvFP8BackendIdentity), 0, 2, 4, 65)
		if tail.bindQwenRotary(linear, theta, 64) == nil {
			t.Fatal("hd65 widened Qwen geometry")
		}
	})
	t.Run("clone view identity budget and local poison", func(t *testing.T) {
		o := kvFP8OwnerFixture(t)
		f := new(kvFP8OwnerFake)
		if err := o.replace(3, nil, 1<<30, f.ops(), kvFP8OwnerComplete, false); err != nil {
			t.Fatal(err)
		}
		view, _ := o.borrow(0, kvFP8Value)
		for _, wrong := range []struct {
			key       *kvFP8BackendIdentity
			dev, n, w int
		}{
			{new(kvFP8BackendIdentity), 0, 3, 1024}, {o.backend, 1, 3, 1024},
			{o.backend, 0, 2, 1024}, {o.backend, 0, 3, 1023},
		} {
			if _, _, err := view.resolve(wrong.key, wrong.dev, wrong.n, wrong.w); err == nil {
				t.Fatal("altered view accepted")
			}
		}
		allocs := f.allocs
		if _, err := o.clone(o.allocatedBytes(), f.ops(), kvFP8OwnerComplete); !errors.Is(err, errKVFP8Budget) || allocs != f.allocs {
			t.Fatal("clone budget was checked after effects")
		}
		clone, err := o.clone(1<<30, f.ops(), func(dst, src *kvFP8Image) kvFP8Completion {
			if src != o.image || dst == src || dst.layers[0][kvFP8Key].ptr == src.layers[0][kvFP8Key].ptr {
				t.Fatal("clone aliases")
			}
			return kvFP8OwnerComplete(dst, src)
		})
		if err != nil || clone == o || &clone.rotary[0] == &o.rotary[0] {
			t.Fatal("clone metadata aliases", err)
		}
		err = o.replace(4, nil, 1<<30, f.ops(), func(_, _ *kvFP8Image) kvFP8Completion {
			return kvFP8Completion{completed: true, numericalChecked: true, numerical: 1}
		}, true)
		if !errors.Is(err, ErrKVFP8NonFinite) || view.Ready() || o.usable() == nil || clone.usable() != nil ||
			o.uncertain || o.retained {
			t.Fatal("late numerical poison escaped owner or missed owner")
		}
		if _, _, err := view.resolve(o.backend, 0, 3, 1024); err == nil {
			t.Fatal("poisoned direct view")
		}
		if _, err := o.borrow(0, kvFP8Key); err == nil {
			t.Fatal("poisoned new view")
		}
		if _, err := o.planEvict(0, 1); err == nil {
			t.Fatal("poisoned eviction")
		}
		if _, err := o.clone(1<<30, f.ops(), kvFP8OwnerComplete); err == nil {
			t.Fatal("poisoned prefix reuse")
		}
		if o.allocatedBytes() == 0 {
			t.Fatal("poison lost residency accounting")
		}
		if err := o.free(f.ops()); err != nil || o.allocatedBytes() != 0 {
			t.Fatal("completed numerical owner cannot retire", err)
		}
	})
	t.Run("uncertain completion retains old and all staged planes", func(t *testing.T) {
		o := kvFP8OwnerFixture(t)
		f := new(kvFP8OwnerFake)
		if err := o.replace(3, nil, 1<<30, f.ops(), kvFP8OwnerComplete, false); err != nil {
			t.Fatal(err)
		}
		oldBytes := o.allocatedBytes()
		err := o.replace(3, nil, 1<<30, f.ops(), func(_, _ *kvFP8Image) kvFP8Completion { return kvFP8Completion{} }, false)
		if !errors.Is(err, errKVFP8Completion) || !o.retained || o.stage == nil || o.allocatedBytes() != 2*oldBytes {
			t.Fatal("uncertain ownership lost")
		}
		releases := f.releases
		if o.free(f.ops()) == nil || f.releases != releases {
			t.Fatal("uncertain owner was freed")
		}
	})
	t.Run("cleanup preserves initial cause and native status", func(t *testing.T) {
		for _, allocateFail := range []bool{false, true} {
			o := kvFP8OwnerFixture(t)
			f := &kvFP8OwnerFake{failRelease: 1}
			want := ErrKVFP8NonFinite
			if allocateFail {
				f.failAlloc = 2
				want = errKVFP8Budget
			}
			err := o.replace(3, nil, 1<<30, f.ops(), func(_, _ *kvFP8Image) kvFP8Completion {
				return kvFP8Completion{completed: true, numericalChecked: true, numerical: 1}
			}, false)
			var receipt *kvFP8ReceiptError
			if !errors.Is(err, want) || !errors.Is(err, errKVFP8Retirement) ||
				!errors.As(err, &receipt) || receipt.status != 7 || !o.retained {
				t.Fatal("cleanup replaced the original cause or native status", err)
			}
		}
		o := kvFP8OwnerFixture(t)
		f := new(kvFP8OwnerFake)
		err := o.replace(3, nil, 1<<30, f.ops(), func(_, _ *kvFP8Image) kvFP8Completion {
			return kvFP8Completion{status: 23}
		}, false)
		var receipt *kvFP8ReceiptError
		if !errors.Is(err, errKVFP8Completion) || !errors.As(err, &receipt) || receipt.status != 23 {
			t.Fatal("completion status lost", err)
		}
	})
	t.Run("cleanup unwind retains the preexisting primary cause", func(t *testing.T) {
		for _, mode := range []string{"panic", "Goexit"} {
			for _, allocateFail := range []bool{false, true} {
				o := kvFP8OwnerFixture(t)
				f := new(kvFP8OwnerFake)
				want := ErrKVFP8NonFinite
				if allocateFail {
					f.failAlloc = 2
					want = errKVFP8Budget
				}
				ops := f.ops()
				ops.release = func(kvFP8Allocation) kvFP8Retirement {
					if mode == "panic" {
						panic("cleanup interruption")
					}
					runtime.Goexit()
					return kvFP8Retirement{}
				}
				done := make(chan struct{})
				var recovered any
				normal := false
				go func() {
					defer close(done)
					defer func() { recovered = recover() }()
					_ = o.replace(3, nil, 1<<30, ops, func(_, _ *kvFP8Image) kvFP8Completion {
						return kvFP8Completion{completed: true, numericalChecked: true, numerical: 1}
					}, false)
					normal = true
				}()
				select {
				case <-done:
				case <-time.After(5 * time.Second):
					t.Fatal("cleanup unwind witness timed out", mode)
				}
				if !errors.Is(o.primaryFailure, want) || !o.cleanupInterrupted || !o.retained || normal {
					t.Fatal("primary cause or abnormal cleanup was lost", mode, o.primaryFailure)
				}
				if mode == "panic" && recovered != "cleanup interruption" {
					t.Fatal("original panic replaced", recovered)
				}
				if mode == "Goexit" && recovered != nil {
					t.Fatal("Goexit replaced by panic", recovered)
				}
			}
		}
	})
	t.Run("backend closure invalidates independent completed views", func(t *testing.T) {
		o := kvFP8OwnerFixture(t)
		peer, err := newKVFP8Owner(o.backend, 0, 2, 4, 256)
		if err != nil {
			t.Fatal(err)
		}
		view, _ := peer.borrow(0, kvFP8Key)
		o.retainUncertain()
		if view.Ready() || peer.usable() == nil {
			t.Fatal("closed backend appeared ready")
		}
		f := new(kvFP8OwnerFake)
		if peer.free(f.ops()) == nil || f.releases != 0 || !peer.retained {
			t.Fatal("closed-backend free")
		}
		if _, err := newKVFP8Owner(o.backend, 0, 2, 4, 256); err == nil {
			t.Fatal("closed backend reopened")
		}
	})
	t.Run("Goexit quarantines every interrupted owner", func(t *testing.T) {
		for _, operation := range []string{"replace", "clone", "free"} {
			o := kvFP8OwnerFixture(t)
			f := new(kvFP8OwnerFake)
			if err := o.replace(3, nil, 1<<30, f.ops(), kvFP8OwnerComplete, false); err != nil {
				t.Fatal(err)
			}
			done := make(chan struct{})
			var recovered any
			normal := false
			go func() {
				defer close(done)
				defer func() { recovered = recover() }()
				stop := func(_, _ *kvFP8Image) kvFP8Completion { runtime.Goexit(); return kvFP8Completion{} }
				switch operation {
				case "replace":
					_ = o.replace(3, nil, 1<<30, f.ops(), stop, false)
				case "clone":
					_, _ = o.clone(1<<30, f.ops(), stop)
				case "free":
					ops := f.ops()
					ops.release = func(kvFP8Allocation) kvFP8Retirement { runtime.Goexit(); return kvFP8Retirement{} }
					_ = o.free(ops)
				}
				normal = true
			}()
			select {
			case <-done: // no concurrent metadata inspection or shared root mutation
			case <-time.After(5 * time.Second):
				t.Fatal("Goexit lifecycle witness timed out", operation)
			}
			if normal || recovered != nil || !o.retained || !o.backend.stopped.Load() {
				t.Fatal("Goexit changed control transfer or lost owner", operation, normal, recovered)
			}
			if operation == "clone" {
				destination := false
				for retained := kvFP8RetainedOwners; retained != nil; retained = retained.retainNext {
					if retained != o && retained.stage != nil && retained.backend == o.backend {
						destination = true
					}
				}
				if !destination {
					t.Fatal("Goexit lost clone destination")
				}
			}
		}
	})
	t.Run("clone panic roots both owners", func(t *testing.T) {
		o := kvFP8OwnerFixture(t)
		f := new(kvFP8OwnerFake)
		if err := o.replace(3, nil, 1<<30, f.ops(), kvFP8OwnerComplete, false); err != nil {
			t.Fatal(err)
		}
		func() {
			defer func() {
				if recover() == nil {
					t.Fatal("missing original panic")
				}
			}()
			_, _ = o.clone(1<<30, f.ops(), func(_, _ *kvFP8Image) kvFP8Completion { panic("copy may have submitted") })
		}()
		destination := false
		for retained := kvFP8RetainedOwners; retained != nil; retained = retained.retainNext {
			if retained != o && retained.stage != nil && retained.backend == o.backend {
				destination = true
			}
		}
		if !o.retained || !destination || f.releases != 0 {
			t.Fatal("panic lost a copy owner")
		}
	})
	t.Run("workspace status survives uncertain completion", func(t *testing.T) {
		o := kvFP8OwnerFixture(t)
		f := new(kvFP8OwnerFake)
		err := o.replace(3, []int{4, 4096}, 1<<30, f.ops(), func(dst, src *kvFP8Image) kvFP8Completion {
			if len(dst.workspace) != 2 || dst.workspace[0].ptr == nil || dst.workspace[1].ptr == nil {
				t.Fatal("workspace not attached before submission")
			}
			return kvFP8Completion{}
		}, false)
		_, payload, _ := o.planeBytes(3)
		if !errors.Is(err, errKVFP8Completion) || o.allocatedBytes() != payload+4100 ||
			f.releases != 0 || !o.retained {
			t.Fatal("uncertain workspace lost")
		}
	})
	t.Run("retired-with-error pointer is not retried", func(t *testing.T) {
		o := kvFP8OwnerFixture(t)
		f := new(kvFP8OwnerFake)
		if err := o.replace(3, nil, 1<<30, f.ops(), kvFP8OwnerComplete, false); err != nil {
			t.Fatal(err)
		}
		f.failRelease, f.retiredOnFailure = 1, true
		err := o.replace(3, nil, 1<<30, f.ops(), kvFP8OwnerComplete, false)
		if !errors.Is(err, errKVFP8Retirement) || o.retiring.layers[0][0].ptr != nil || !o.retained {
			t.Fatal("retirement receipt not preserved")
		}
		_ = o.free(f.ops())
		if f.releases != 1 {
			t.Fatal("retirement retried")
		}
	})
}
