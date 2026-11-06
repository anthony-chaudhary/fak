package compute

import (
	"bytes"
	"errors"
	"fmt"
	"math"
	"testing"
	"unsafe"
)

// These callbacks operate on deliberately non-numerical byte patterns. They
// prove copy layout/order/publication/retention, never CUDA or rotary accuracy.
type kvFP8EagerFake struct {
	t                                                                 *testing.T
	data                                                              map[unsafe.Pointer][]byte
	events                                                            []string
	allocations, releases, copies, rotations, quantizations, finishes int
	failCopy, rejectFinish, statusFinish                              int
	rejectMask                                                        uint32
	lastLayer                                                         int
	rotaryPending, quantPending                                       bool
}

func (f *kvFP8EagerFake) ops() kvFP8EagerOps {
	return kvFP8EagerOps{
		owner: kvFP8OwnerOps{
			allocate: func(a *kvFP8Allocation, n, layer, plane int) error {
				b := make([]byte, n)
				a.ptr = unsafe.Pointer(&b[0])
				f.data[a.ptr] = b
				f.allocations++
				return nil
			},
			release: func(a kvFP8Allocation) kvFP8Retirement {
				f.releases++
				delete(f.data, a.ptr)
				return kvFP8Retirement{retired: true}
			},
		},
		begin: func(status kvFP8Allocation) int {
			f.events = append(f.events, "begin")
			if status.bytes != 4 {
				f.t.Fatal("status extent")
			}
			return 0
		},
		copy: func(dst kvFP8Allocation, doff int, src kvFP8Allocation, soff, n int) int {
			f.copies++
			f.events = append(f.events, "copy")
			if f.failCopy == f.copies {
				return 39
			}
			if doff < 0 || soff < 0 || n <= 0 || doff > dst.bytes-n || soff > src.bytes-n {
				f.t.Fatal("bad byte range")
			}
			copy(f.data[dst.ptr][doff:doff+n], f.data[src.ptr][soff:soff+n])
			return 0
		},
		rotary: func(dst, raw kvFP8Allocation, before, from, end, heads int, c kvFP8RotaryContract, status kvFP8Allocation) int {
			if f.rotaryPending || f.quantPending {
				f.t.Fatal("scratch reused before checked completion")
			}
			f.rotations++
			f.lastLayer = c.mainLayer / 4
			f.rotaryPending = true
			f.events = append(f.events, fmt.Sprintf("rotary%d", f.lastLayer))
			wantTheta := float64(10000)
			if f.lastLayer == 1 {
				wantTheta = 1000000
			}
			if c.producer != kvFP8QwenPartialPanelV1 || c.headDim != 256 || c.rotaryDim != 64 ||
				c.thetaBits != math.Float64bits(wantTheta) || before != 4 || from != 1 || end != 2 || heads != 4 ||
				dst.bytes != 2*4*256*4 || raw.bytes != 4*4*256*4 {
				f.t.Fatal("wrong bound suffix/partial rotary contract")
			}
			return 0
		},
		quantize: func(codes kvFP8Allocation, coff int, scales kvFP8Allocation, soff int,
			src kvFP8Allocation, rows, hd int, status kvFP8Allocation) int {
			if f.rotaryPending || f.quantPending {
				f.t.Fatal("quantizer read before checked re-RoPE")
			}
			f.quantizations++
			f.quantPending = true
			f.events = append(f.events, fmt.Sprintf("quant%d", f.lastLayer))
			if coff != 4*256 || soff != 4*4 || rows != 2*4 || hd != 256 {
				f.t.Fatal("wrong suffix quantize offsets")
			}
			for i := coff; i < coff+rows*hd; i++ {
				f.data[codes.ptr][i] = byte(0xa0 + f.lastLayer)
			}
			for i := soff; i < soff+rows*4; i++ {
				f.data[scales.ptr][i] = byte(0xb0 + f.lastLayer)
			}
			return 0
		},
		finish: func(status *kvFP8Allocation) kvFP8Completion {
			f.finishes++
			f.events = append(f.events, fmt.Sprintf("finish%d", f.finishes))
			r := kvFP8Completion{completed: true, numericalChecked: true}
			if f.rejectFinish == f.finishes {
				r.numerical = f.rejectMask
			}
			if f.statusFinish == f.finishes {
				r.status = 17
			}
			if r.status == 0 && r.numerical == 0 {
				f.rotaryPending = false
				f.quantPending = false
			}
			return r
		},
	}
}

func kvFP8EagerFixture(t *testing.T) (*kvFP8Owner, *kvFP8EagerFake) {
	t.Helper()
	o, err := newKVFP8Owner(new(kvFP8BackendIdentity), 0, 2, 4, 256)
	if err != nil {
		t.Fatal(err)
	}
	if err := o.bindQwenRotary([]bool{true, true, true, false, true, true, true, false},
		[]float64{1, 1, 1, 10000, 1, 1, 1, 1000000}, 64); err != nil {
		t.Fatal(err)
	}
	f := &kvFP8EagerFake{t: t, data: map[unsafe.Pointer][]byte{}}
	err = o.replace(4, nil, 1<<20, f.ops().owner, func(dst, src *kvFP8Image) kvFP8Completion {
		units, _, _ := o.planeBytes(1)
		for l := range dst.layers {
			for p, a := range dst.layers[l] {
				for row := 0; row < 4; row++ {
					for i := row * units[p]; i < (row+1)*units[p]; i++ {
						f.data[a.ptr][i] = byte(32*p + 8*l + row)
					}
				}
			}
		}
		return kvFP8Completion{completed: true, numericalChecked: true}
	}, false)
	if err != nil {
		t.Fatal(err)
	}
	return o, f
}

func kvFP8EagerImageBytes(o *kvFP8Owner, f *kvFP8EagerFake) []byte {
	var out []byte
	for _, layer := range o.image.layers {
		for _, a := range layer {
			out = append(out, f.data[a.ptr]...)
		}
	}
	return out
}

// fak-test:runtime fast est=50ms lane=default
func TestKVFP8EagerLifecycle(t *testing.T) {
	saved := kvFP8RetainedOwners
	defer func() { kvFP8RetainedOwners = saved }()
	t.Run("two-layer exact copies and staged suffix order", func(t *testing.T) {
		o, f := kvFP8EagerFixture(t)
		before := kvFP8EagerImageBytes(o, f)
		view, _ := o.borrow(0, kvFP8Key)
		plan, _ := o.planEvict(1, 1)
		removed, err := kvFP8ApplyEvictionEager(plan, 1<<20, f.ops())
		if err != nil || removed != 1 || o.image.positions != 3 || view.Ready() {
			t.Fatal("publication", removed, err)
		}
		if f.rotations != 2 || f.quantizations != 2 || f.finishes != 4 || f.copies != 16 || f.releases != 12 {
			t.Fatalf("wrong stage counts: %+v", f)
		}
		units, _, _ := o.planeBytes(1)
		oldOffset := 0
		for l := range o.image.layers {
			for p, a := range o.image.layers[l] {
				oldPlane := before[oldOffset : oldOffset+4*units[p]]
				oldOffset += 4 * units[p]
				got := f.data[a.ptr]
				if !bytes.Equal(got[:units[p]], oldPlane[:units[p]]) {
					t.Fatal("prefix changed", l, p)
				}
				if p != kvFP8Key && p != kvFP8KeyScale {
					if !bytes.Equal(got[units[p]:], oldPlane[2*units[p]:]) {
						t.Fatal("retained bits changed", l, p)
					}
				} else {
					marker := byte(0xa0 + l)
					if p == kvFP8KeyScale {
						marker = byte(0xb0 + l)
					}
					for _, b := range got[units[p]:] {
						if b != marker {
							t.Fatal("suffix was not staged", l, p)
						}
					}
				}
			}
		}
		allocs, calls := f.allocations, len(f.events)
		if _, err := kvFP8ApplyEvictionEager(plan, 1<<20, f.ops()); err == nil ||
			f.allocations != allocs || len(f.events) != calls {
			t.Fatal("stale plan executed callbacks")
		}
	})
	t.Run("numeric stage failure cannot reach a dependent read", func(t *testing.T) {
		for _, at := range []int{1, 2, 3, 4} {
			o, f := kvFP8EagerFixture(t)
			old := o.image
			before := kvFP8EagerImageBytes(o, f)
			view, _ := o.borrow(0, kvFP8Key)
			f.rejectFinish = at
			f.rejectMask = 8
			removed, err := kvFP8EvictEager(o, 1, 1, 1<<20, f.ops())
			wantQuant := at / 2
			if at%2 == 1 {
				wantQuant = (at - 1) / 2
			}
			if !errors.Is(err, ErrKVFP8NonFinite) || removed != 0 || o.image != old || !view.Ready() ||
				f.quantizations != wantQuant || !bytes.Equal(before, kvFP8EagerImageBytes(o, f)) ||
				o.numericalFailure != 8 || o.stage != nil {
				t.Fatal("rejected phase published or read downstream", at, err)
			}
		}
	})
	t.Run("status retirement failure retains exact numerical mask", func(t *testing.T) {
		o, f := kvFP8EagerFixture(t)
		f.rejectFinish = 1
		f.rejectMask = 4
		f.statusFinish = 1
		_, err := kvFP8EvictEager(o, 1, 1, 1<<20, f.ops())
		var receipt *kvFP8ReceiptError
		if !errors.Is(err, ErrKVFP8NonFinite) || !errors.Is(err, errKVFP8Completion) ||
			!errors.As(err, &receipt) || receipt.status != 17 || o.numericalFailure != 4 ||
			!o.retained || o.stage == nil || f.quantizations != 0 || f.releases != 0 {
			t.Fatal("lost combined receipt", err)
		}
	})
	t.Run("copy failure is uncertain and clone copies all planes", func(t *testing.T) {
		o, f := kvFP8EagerFixture(t)
		before := kvFP8EagerImageBytes(o, f)
		clone, err := kvFP8CloneEager(o, 1<<20, f.ops())
		if err != nil || f.copies != 10 || f.finishes != 1 ||
			!bytes.Equal(before, kvFP8EagerImageBytes(clone, f)) {
			t.Fatal("clone bit-copy", err)
		}
		f.data[o.image.layers[0][kvFP8Key].ptr][0] ^= 1
		if !bytes.Equal(before, kvFP8EagerImageBytes(clone, f)) {
			t.Fatal("clone aliases source")
		}
		o2, f2 := kvFP8EagerFixture(t)
		f2.failCopy = 2
		if _, err := kvFP8CloneEager(o2, 1<<20, f2.ops()); !errors.Is(err, errKVFP8Completion) ||
			!o2.retained || f2.releases != 0 {
			t.Fatal("uncertain clone source freed", err)
		}
	})
	t.Run("tail/full removal needs no rotary and budget is pre-effect", func(t *testing.T) {
		for _, from := range []int{0, 3} {
			o, f := kvFP8EagerFixture(t)
			o.rotary = make([]kvFP8RotaryContract, o.layers) // intentionally unknown, no moved suffix
			removed, err := kvFP8EvictEager(o, from, 100, 1<<20, f.ops())
			if err != nil || removed != 4-from || f.rotations != 0 || f.quantizations != 0 || f.finishes != 1 {
				t.Fatal("unneeded rotary or missed final fence", err)
			}
		}
		o, f := kvFP8EagerFixture(t)
		allocs := f.allocations
		if _, err := kvFP8EvictEager(o, 1, 1, o.allocatedBytes(), f.ops()); !errors.Is(err, errKVFP8Budget) ||
			f.allocations != allocs || len(f.events) != 0 {
			t.Fatal("late peak refusal", err)
		}
	})
}
