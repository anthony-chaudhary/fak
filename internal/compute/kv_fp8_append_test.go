package compute

import (
	"bytes"
	"encoding/binary"
	"errors"
	"math"
	"runtime"
	"testing"
	"time"
	"unsafe"
)

// The transport is host memory. Codec calls are the existing host oracle,
// not a CUDA implementation or evidence of source-producer rotary parity.
type kvFP8AppendFake struct {
	base                                       *kvFP8EagerFake
	o                                          *kvFP8Owner
	checks, validations, ingests, reads, stops int
	rejectCheck, statusCheck                   int
	mask                                       uint32
}

func (f *kvFP8AppendFake) input(layer, tokens int) kvFP8AppendInput {
	values := [3][]float32{}
	var a [3]kvFP8Allocation
	for p := range a {
		values[p] = make([]float32, tokens*4*256)
		b := make([]byte, len(values[p])*4)
		for i := range values[p] {
			values[p][i] = float32((i%31)-15) * float32(p+1) / 8
			binary.LittleEndian.PutUint32(b[i*4:], math.Float32bits(values[p][i]))
		}
		a[p] = kvFP8Allocation{ptr: unsafe.Pointer(&b[0]), bytes: len(b)}
		f.base.data[a[p].ptr] = b
	}
	return kvFP8AppendInput{backend: f.o.backend, device: f.o.device, first: f.o.image.positions,
		tokens: tokens, contract: f.o.rotary[layer], raw: a[0], key: a[1], value: a[2], root: values}
}

func (f *kvFP8AppendFake) floats(a kvFP8Allocation) []float32 {
	b := f.base.data[a.ptr]
	v := make([]float32, len(b)/4)
	for i := range v {
		v[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[4*i:]))
	}
	return v
}

func (f *kvFP8AppendFake) ops() kvFP8AppendOps {
	eager := f.base.ops()
	eager.finish = func(*kvFP8Allocation) kvFP8Completion {
		f.checks++
		if f.checks == f.rejectCheck {
			f.mask = 8
		}
		r := kvFP8Completion{completed: true, numericalChecked: true, numerical: f.mask}
		if f.checks == f.statusCheck {
			r.status = 23
			r.completed = false
			r.numericalChecked = false
		}
		return r
	}
	return kvFP8AppendOps{
		eager: eager,
		validate: func(in kvFP8AppendInput, status kvFP8Allocation) int {
			f.validations++
			for _, a := range [3]kvFP8Allocation{in.raw, in.key, in.value} {
				for _, v := range f.floats(a) {
					if !kvFP8Finite(v) {
						f.mask |= 1
					}
				}
			}
			return 0
		},
		ingest: func(dst [kvFP8Planes]kvFP8Allocation, prefix int, in kvFP8AppendInput, status kvFP8Allocation) int {
			f.ingests++
			if f.mask != 0 {
				f.base.t.Fatal("ingest read rejected/unwritten source")
			}
			units, _, _ := f.o.planeBytes(1)
			copy(f.base.data[dst[kvFP8Raw].ptr][prefix*units[kvFP8Raw]:], f.base.data[in.raw.ptr])
			for p, src := range [2]kvFP8Allocation{in.key, in.value} {
				q, err := QuantizeKVFP8Plane(f.floats(src), in.tokens, 4, 256)
				if err != nil {
					f.base.t.Fatal(err)
				}
				copy(f.base.data[dst[kvFP8Key+p].ptr][prefix*units[kvFP8Key+p]:], q.Codes)
				s := f.base.data[dst[kvFP8KeyScale+p].ptr][prefix*units[kvFP8KeyScale+p]:]
				for i, v := range q.Scales {
					binary.LittleEndian.PutUint32(s[i*4:], math.Float32bits(v))
				}
			}
			return 0
		},
		stop: func(int) { f.stops++ },
	}
}

func (f *kvFP8AppendFake) run(t *kvFP8AppendTransaction) error {
	for l := 0; l < f.o.layers; l++ {
		if err := t.append(l, f.input(l, t.tokens)); err != nil {
			return err
		}
		if err := t.consume(l, func(k kvFP8AppendLayer) int {
			f.reads++
			if f.mask != 0 || k.layer != l || k.prefix != 4 || k.tokens != t.tokens ||
				k.heads != 4 || k.headDim != 256 {
				f.base.t.Fatal("staged read contract")
			}
			return 0
		}); err != nil {
			return err
		}
	}
	return nil
}

func kvFP8AppendFixture(t *testing.T) (*kvFP8Owner, *kvFP8AppendFake) {
	o, b := kvFP8EagerFixture(t)
	return o, &kvFP8AppendFake{base: b, o: o}
}

// fak-test:runtime fast est=100ms lane=default
func TestKVFP8AppendTransaction(t *testing.T) {
	savedOwners, savedAppend := kvFP8RetainedOwners, kvFP8AppendRetained
	defer func() { kvFP8RetainedOwners = savedOwners; kvFP8AppendRetained = savedAppend }()
	t.Run("two layers exact prefix raw and oracle tails", func(t *testing.T) {
		o, f := kvFP8AppendFixture(t)
		before := kvFP8EagerImageBytes(o, f.base)
		view, _ := o.borrow(0, kvFP8Key)
		var escaped *kvFP8AppendTransaction
		var input [2]kvFP8AppendInput
		err := kvFP8AppendEager(o, 4, 2, 1<<20, f.ops(), func(tx *kvFP8AppendTransaction) error {
			escaped = tx
			if view.Ready() {
				t.Fatal("old view live during stage")
			}
			for l := 0; l < 2; l++ {
				input[l] = f.input(l, 2)
				if err := tx.append(l, input[l]); err != nil {
					return err
				}
				if err := tx.consume(l, func(k kvFP8AppendLayer) int {
					if k.planes[kvFP8Raw].bytes != 6*4*256*4 {
						t.Fatal("wrong complete image extent")
					}
					return 0
				}); err != nil {
					return err
				}
			}
			return nil
		})
		if err != nil || o.image.positions != 6 || o.stage != nil || view.Ready() || f.checks != 8 ||
			f.validations != 2 || f.ingests != 2 || f.stops != 0 {
			t.Fatal("commit", err)
		}
		units, _, _ := o.planeBytes(1)
		offset := 0
		for l := range o.image.layers {
			for p, a := range o.image.layers[l] {
				b := f.base.data[a.ptr]
				n := 4 * units[p]
				if !bytes.Equal(b[:n], before[offset:offset+n]) {
					t.Fatal("old bytes changed", l, p)
				}
				offset += n
			}
			if !bytes.Equal(f.base.data[o.image.layers[l][kvFP8Raw].ptr][4*units[kvFP8Raw]:],
				f.base.data[input[l].raw.ptr]) {
				t.Fatal("normalized raw not exact")
			}
			for p, a := range [2]kvFP8Allocation{input[l].key, input[l].value} {
				q, _ := QuantizeKVFP8Plane(f.floats(a), 2, 4, 256)
				if !bytes.Equal(f.base.data[o.image.layers[l][kvFP8Key+p].ptr][4*units[kvFP8Key+p]:], q.Codes) {
					t.Fatal("codec tail")
				}
				scales := f.base.data[o.image.layers[l][kvFP8KeyScale+p].ptr][4*units[kvFP8KeyScale+p]:]
				for i, v := range q.Scales {
					if binary.LittleEndian.Uint32(scales[4*i:]) != math.Float32bits(v) {
						t.Fatal("scale tail")
					}
				}
			}
		}
		calls := f.checks
		if escaped.append(0, input[0]) == nil || escaped.consume(0, func(kvFP8AppendLayer) int { return 0 }) == nil ||
			f.checks != calls || o.usable() != nil || escaped.inputs != nil {
			t.Fatal("escaped handle executed")
		}
	})
	t.Run("all checked boundaries reject before dependent access", func(t *testing.T) {
		for at := 1; at <= 8; at++ {
			o, f := kvFP8AppendFixture(t)
			old := o.image
			before := kvFP8EagerImageBytes(o, f.base)
			view, _ := o.borrow(0, kvFP8Key)
			f.rejectCheck = at
			err := kvFP8AppendEager(o, 4, 1, 1<<20, f.ops(), f.run)
			if !errors.Is(err, ErrKVFP8NonFinite) || o.image != old || view.Ready() || o.usable() == nil ||
				o.backend.stopped.Load() || o.stage != nil || o.numericalFailure != 8 ||
				!bytes.Equal(before, kvFP8EagerImageBytes(o, f.base)) {
				t.Fatal("numerical boundary", at, err)
			}
			if at == 2 && (f.ingests != 0 || f.reads != 0) {
				t.Fatal("validate rejection reached downstream")
			}
			if at == 3 && f.reads != 0 {
				t.Fatal("quantize rejection reached attention")
			}
			if _, err := o.borrow(0, kvFP8Key); err == nil {
				t.Fatal("poisoned view")
			}
			if _, err := kvFP8CloneEager(o, 1<<20, f.ops().eager); err == nil {
				t.Fatal("poisoned clone")
			}
			if _, err := kvFP8EvictEager(o, 0, 1, 1<<20, f.ops().eager); err == nil {
				t.Fatal("poisoned eviction")
			}
		}
	})
	t.Run("raw alone is validated and retained separately", func(t *testing.T) {
		o, f := kvFP8AppendFixture(t)
		err := kvFP8AppendEager(o, 4, 1, 1<<20, f.ops(), func(tx *kvFP8AppendTransaction) error {
			in := f.input(0, 1)
			binary.LittleEndian.PutUint32(f.base.data[in.raw.ptr], 0x7f800000)
			return tx.append(0, in)
		})
		if !errors.Is(err, ErrKVFP8NonFinite) || f.validations != 1 || f.ingests != 0 || o.numericalFailure != 1 {
			t.Fatal(err)
		}
	})
	t.Run("metadata geometry and budget before callbacks", func(t *testing.T) {
		for mode := 0; mode < 4; mode++ {
			o, f := kvFP8AppendFixture(t)
			allocs := f.base.allocations
			first, tokens, budget := 4, 1, int64(1<<20)
			if mode == 0 {
				first = 3
			}
			if mode == 1 {
				tokens = 0
			}
			if mode == 2 {
				tokens = int(^uint(0) >> 1)
			}
			if mode == 3 {
				budget = o.allocatedBytes()
			}
			if err := kvFP8AppendEager(o, first, tokens, budget, f.ops(), f.run); err == nil ||
				f.base.allocations != allocs || f.checks != 0 || o.usable() != nil {
				t.Fatal("late refusal", mode, err)
			}
		}
	})
	t.Run("uncertain status roots inputs and all planes", func(t *testing.T) {
		o, f := kvFP8AppendFixture(t)
		f.statusCheck = 3
		err := kvFP8AppendEager(o, 4, 1, 1<<20, f.ops(), f.run)
		if !errors.Is(err, errKVFP8Completion) || !o.retained || !o.backend.stopped.Load() ||
			o.stage == nil || f.base.releases != 0 || kvFP8AppendRetained == nil ||
			kvFP8AppendRetained.inputs[0].root == nil || f.reads != 0 || f.stops == 0 {
			t.Fatal("lost roots", err)
		}
	})
	t.Run("missing duplicate reordered and recursive layers cannot commit", func(t *testing.T) {
		for mode := 0; mode < 4; mode++ {
			o, f := kvFP8AppendFixture(t)
			err := kvFP8AppendEager(o, 4, 1, 1<<20, f.ops(), func(tx *kvFP8AppendTransaction) error {
				if mode == 0 {
					return nil
				}
				if mode == 1 {
					return tx.append(1, f.input(1, 1))
				}
				if err := tx.append(0, f.input(0, 1)); err != nil {
					return err
				}
				if mode == 2 {
					return tx.append(0, f.input(0, 1))
				}
				return tx.consume(0, func(kvFP8AppendLayer) int {
					_ = tx.append(1, f.input(1, 1))
					return 0
				})
			})
			if err == nil || !o.retained || !o.backend.stopped.Load() || f.base.releases != 0 {
				t.Fatal("bad order published", mode, err)
			}
		}
	})
	t.Run("panic and Goexit preserve control transfer and roots", func(t *testing.T) {
		for _, exit := range []bool{false, true} {
			o, f := kvFP8AppendFixture(t)
			done := make(chan struct{})
			normal := false
			var recovered any
			go func() {
				defer close(done)
				defer func() { recovered = recover() }()
				_ = kvFP8AppendEager(o, 4, 1, 1<<20, f.ops(), func(tx *kvFP8AppendTransaction) error {
					if err := tx.append(0, f.input(0, 1)); err != nil {
						return err
					}
					if exit {
						runtime.Goexit()
					}
					panic("append-unwind")
				})
				normal = true
			}()
			select {
			case <-done:
			case <-time.After(5 * time.Second):
				t.Fatal("append unwind timed out")
			}
			if normal || !o.retained || !o.backend.stopped.Load() || f.base.releases != 0 || f.stops == 0 ||
				kvFP8AppendRetained.inputs[0].root == nil {
				t.Fatal("unwind lost ownership")
			}
			if exit && recovered != nil {
				t.Fatal("Goexit converted to panic")
			}
			if !exit && recovered != "append-unwind" {
				t.Fatal("panic changed", recovered)
			}
		}
	})
}
