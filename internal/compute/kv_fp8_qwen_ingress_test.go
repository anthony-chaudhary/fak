package compute

import (
	"encoding/binary"
	"errors"
	"math"
	"runtime"
	"testing"
	"time"
	"unsafe"
)

func kvFP8QwenFixtureInput(f *kvFP8AppendFake) kvFP8QwenInput {
	in := kvFP8QwenInput{backend: f.o.backend, device: 0, heads: 24}
	owners := [5][]byte{}
	for i, n := range [5]int{24 * 256 * 4, 4 * 256 * 4, 4 * 256 * 4, 256 * 4, 256 * 4} {
		owners[i] = make([]byte, n)
		for j := 0; j < n; j += 4 {
			binary.LittleEndian.PutUint32(owners[i][j:], math.Float32bits(1))
		}
		in.operands[i] = kvFP8Allocation{unsafe.Pointer(&owners[i][0]), n}
		f.base.data[in.operands[i].ptr] = owners[i]
	}
	in.root = &owners
	return in
}

// Host producer uses sentinels, deliberately not an RMSNorm/RoPE oracle.
// Arithmetic/source parity is a separate required CUDA witness.
func kvFP8QwenFixtureOps(t *testing.T, f *kvFP8AppendFake, tx *kvFP8AppendTransaction,
	calls *[4]int, roots *[]*kvFP8QwenRoots) kvFP8QwenOps {
	check := func(s kvFP8QwenStage, phase int) {
		calls[phase]++
		r, ok := tx.inputs[tx.next].root.(*kvFP8QwenRoots)
		if !ok || r.input.root != s.input.root || r.work != s.work || s.first != 4 ||
			s.tokens != 1 || s.contract != f.o.rotary[tx.next] || !tx.busy {
			t.Fatal("producer stage lost exact roots or contract")
		}
		if phase == 0 {
			*roots = append(*roots, r)
		}
		if f.checks != 1+7*tx.next+phase {
			t.Fatal("producer read before checked prior phase", f.checks, phase)
		}
	}
	return kvFP8QwenOps{
		admit:    func() int { return 0 },
		validate: func(s kvFP8QwenStage) int { check(s, 0); return 0 },
		norm: func(s kvFP8QwenStage) int {
			check(s, 1)
			for i := 0; i < 2; i++ {
				b := f.base.data[s.work[i].ptr]
				for j := 0; j < len(b); j += 4 {
					binary.LittleEndian.PutUint32(b[j:], math.Float32bits(float32(i+2)))
				}
			}
			return 0
		},
		rope: func(s kvFP8QwenStage) int {
			check(s, 2)
			for i := 0; i < 2; i++ {
				copy(f.base.data[s.work[i+2].ptr], f.base.data[s.work[i].ptr])
			}
			return 0
		},
		attention: func(k kvFP8AppendLayer, q, out kvFP8Allocation, heads int, scale float32) int {
			calls[3]++
			if heads != 24 || q.bytes != 24*256*4 || out.bytes != q.bytes ||
				f.checks != 6+7*k.layer || f.floats(q)[0] != 2 || scale != 0.0625 {
				t.Fatal("wrong produced query/fence")
			}
			for i := 0; i < len(f.base.data[out.ptr]); i++ {
				f.base.data[out.ptr][i] = 0x3f
			}
			return 0
		},
	}
}

// fak-test:runtime fast est=100ms lane=default
func TestKVFP8QwenIngressOwnership(t *testing.T) {
	savedOwners, savedAppend := kvFP8RetainedOwners, kvFP8AppendRetained
	defer func() { kvFP8RetainedOwners = savedOwners; kvFP8AppendRetained = savedAppend }()
	t.Run("two layer exact producer contract and checked retirement", func(t *testing.T) {
		o, f := kvFP8AppendFixture(t)
		var calls [4]int
		var roots []*kvFP8QwenRoots
		uses := 0
		err := kvFP8AppendEager(o, 4, 1, 1<<20, f.ops(), func(tx *kvFP8AppendTransaction) error {
			ops := kvFP8QwenFixtureOps(t, f, tx, &calls, &roots)
			for l := 0; l < o.layers; l++ {
				err := kvFP8QwenIngressEager(tx, l, kvFP8QwenFixtureInput(f), 1e-6, 0.0625, 1<<20, ops,
					func(out kvFP8AttentionOutput) int {
						uses++
						if f.checks != 7+7*l || f.base.releases != 5*l || out.storage.ptr == nil || !tx.busy {
							t.Fatal("early use/free")
						}
						return 0
					})
				if err != nil {
					return err
				}
				for _, a := range roots[l].work {
					if a.ptr != nil {
						t.Fatal("temporary not retired")
					}
				}
				if f.base.releases != 5*(l+1) {
					t.Fatal("wrong per-layer retirement", f.base.releases)
				}
			}
			return nil
		})
		if err != nil || calls != [4]int{2, 2, 2, 2} || uses != 2 || f.checks != 16 || f.base.releases != 21 ||
			o.image.positions != 5 || o.usable() != nil {
			t.Fatal("ingress transaction", err, calls, uses, f.checks, f.base.releases)
		}
		for l := 0; l < o.layers; l++ {
			raw := f.floats(o.image.layers[l][kvFP8Raw])
			if raw[4*4*256] != 3 {
				t.Fatal("caller raw rather than producer normalized K published")
			}
		}
	})
	for phase := 0; phase < 3; phase++ {
		t.Run([]string{"input reject", "norm reject", "rotary reject"}[phase], func(t *testing.T) {
			o, f := kvFP8AppendFixture(t)
			f.rejectCheck = 2 + phase
			var calls [4]int
			var roots []*kvFP8QwenRoots
			err := kvFP8AppendEager(o, 4, 1, 1<<20, f.ops(), func(tx *kvFP8AppendTransaction) error {
				return kvFP8QwenIngressEager(tx, 0, kvFP8QwenFixtureInput(f), 1e-6, 0.0625, 1<<20,
					kvFP8QwenFixtureOps(t, f, tx, &calls, &roots), func(kvFP8AttentionOutput) int { t.Fatal("used rejected output"); return 0 })
			})
			if !errors.Is(err, ErrKVFP8NonFinite) || calls[3] != 0 || f.ingests != 0 ||
				f.base.releases != 15 || o.retained || o.backend.stopped.Load() || o.usable() == nil {
				t.Fatal("numerical cleanup", err, calls, f.base.releases)
			}
			for p := phase + 1; p < 3; p++ {
				if calls[p] != 0 {
					t.Fatal("read rejected producer workspace")
				}
			}
		})
	}
	t.Run("unknown producer completion retains current exact roots", func(t *testing.T) {
		o, f := kvFP8AppendFixture(t)
		f.statusCheck = 3
		var calls [4]int
		var roots []*kvFP8QwenRoots
		var tx *kvFP8AppendTransaction
		err := kvFP8AppendEager(o, 4, 1, 1<<20, f.ops(), func(x *kvFP8AppendTransaction) error {
			tx = x
			return kvFP8QwenIngressEager(tx, 0, kvFP8QwenFixtureInput(f), 1e-6, 0.0625, 1<<20,
				kvFP8QwenFixtureOps(t, f, tx, &calls, &roots), func(kvFP8AttentionOutput) int { return 0 })
		})
		if err == nil || kvFP8AppendRetained != tx || !o.retained || !o.backend.stopped.Load() ||
			f.base.releases != 0 || calls[2] != 0 || tx.inputs[0].root != roots[0] {
			t.Fatal("lost exact failed producer", err)
		}
		for _, a := range roots[0].work {
			if a.ptr == nil {
				t.Fatal("unrooted workspace")
			}
		}
	})
	for phase := 0; phase < 3; phase++ {
		t.Run([]string{"caught validate panic", "caught norm panic", "caught rope panic"}[phase], func(t *testing.T) {
			o, f := kvFP8AppendFixture(t)
			var calls [4]int
			var roots []*kvFP8QwenRoots
			var tx *kvFP8AppendTransaction
			var caught any
			after := false
			err := kvFP8AppendEager(o, 4, 1, 1<<20, f.ops(), func(x *kvFP8AppendTransaction) error {
				tx = x
				ops := kvFP8QwenFixtureOps(t, f, tx, &calls, &roots)
				in := kvFP8QwenFixtureInput(f)
				callbacks := []*func(kvFP8QwenStage) int{&ops.validate, &ops.norm, &ops.rope}
				saved := *callbacks[phase]
				once := true
				*callbacks[phase] = func(s kvFP8QwenStage) int {
					if once {
						once = false
						panic("producer panic")
					}
					return saved(s)
				}
				func() {
					defer func() { caught = recover() }()
					_ = kvFP8QwenIngressEager(tx, 0, in, 1e-6, 0.0625, 1<<20, ops, func(kvFP8AttentionOutput) int { return 0 })
					after = true
				}()
				before := f.base.allocations
				retry := kvFP8QwenIngressEager(tx, 0, in, 1e-6, 0.0625, 1<<20, ops, func(kvFP8AttentionOutput) int { return 0 })
				if retry == nil || f.base.allocations != before {
					t.Fatal("caught callback replay admitted")
				}
				return retry
			})
			if err == nil || caught != "producer panic" || after || kvFP8AppendRetained != tx ||
				!o.backend.stopped.Load() || f.base.releases != 0 {
				t.Fatal("caught panic escaped poison", err, caught)
			}
		})
	}
	t.Run("Goexit preserves control transfer and current roots", func(t *testing.T) {
		o, f := kvFP8AppendFixture(t)
		var tx *kvFP8AppendTransaction
		var calls [4]int
		var roots []*kvFP8QwenRoots
		done := make(chan struct{})
		returned := false
		var recovered any
		go func() {
			defer close(done)
			defer func() { recovered = recover() }()
			_ = kvFP8AppendEager(o, 4, 1, 1<<20, f.ops(), func(x *kvFP8AppendTransaction) error {
				tx = x
				ops := kvFP8QwenFixtureOps(t, f, tx, &calls, &roots)
				ops.norm = func(kvFP8QwenStage) int { runtime.Goexit(); return 0 }
				return kvFP8QwenIngressEager(tx, 0, kvFP8QwenFixtureInput(f), 1e-6, 0.0625, 1<<20, ops, func(kvFP8AttentionOutput) int { return 0 })
			})
			returned = true
		}()
		select {
		case <-done:
		case <-time.After(5 * time.Second):
			t.Fatal("Goexit did not unwind")
		}
		if returned || recovered != nil || kvFP8AppendRetained != tx || !o.retained ||
			f.base.releases != 0 || tx.active || tx.inputs[0].root != roots[0] {
			t.Fatal("Goexit lifetime/control transfer")
		}
	})
	t.Run("workspace retirement disposition is consumed before later error", func(t *testing.T) {
		o, f := kvFP8AppendFixture(t)
		var calls [4]int
		var roots []*kvFP8QwenRoots
		ops := f.ops()
		release := ops.eager.owner.release
		ops.eager.owner.release = func(a kvFP8Allocation) kvFP8Retirement {
			r := release(a)
			if f.base.releases == 2 {
				r.status = 91
			}
			return r
		}
		err := kvFP8AppendEager(o, 4, 1, 1<<20, ops, func(tx *kvFP8AppendTransaction) error {
			return kvFP8QwenIngressEager(tx, 0, kvFP8QwenFixtureInput(f), 1e-6, 0.0625, 1<<20,
				kvFP8QwenFixtureOps(t, f, tx, &calls, &roots), func(kvFP8AttentionOutput) int { return 0 })
		})
		if !errors.Is(err, errKVFP8Retirement) || f.base.releases != 2 || roots[0].work[3].ptr != nil ||
			roots[0].work[2].ptr == nil || !o.retained {
			t.Fatal("retired pointer retried/lost", err)
		}
	})
	t.Run("timing admission refuses before Qwen allocation or producer", func(t *testing.T) {
		o, f := kvFP8AppendFixture(t)
		var calls [4]int
		var roots []*kvFP8QwenRoots
		extra := 0
		err := kvFP8AppendEager(o, 4, 1, 1<<20, f.ops(), func(tx *kvFP8AppendTransaction) error {
			ops := kvFP8QwenFixtureOps(t, f, tx, &calls, &roots)
			ops.admit = func() int { return -76301 }
			before := f.base.allocations
			e := kvFP8QwenIngressEager(tx, 0, kvFP8QwenFixtureInput(f), 1e-6, 0.0625, 1<<20, ops, func(kvFP8AttentionOutput) int { t.Fatal("timed ingress used"); return 0 })
			extra = f.base.allocations - before
			return e
		})
		if err == nil || extra != 0 || len(roots) != 0 || calls != [4]int{} || f.ingests != 0 {
			t.Fatal("late timing admission", err)
		}
	})
	t.Run("shape and full peak budget refuse before workspace effects", func(t *testing.T) {
		for _, mode := range []string{"gain", "heads", "budget", "theta"} {
			o, f := kvFP8AppendFixture(t)
			var calls [4]int
			var roots []*kvFP8QwenRoots
			extra := 0
			err := kvFP8AppendEager(o, 4, 1, 1<<20, f.ops(), func(tx *kvFP8AppendTransaction) error {
				in := kvFP8QwenFixtureInput(f)
				budget := int64(1 << 20)
				switch mode {
				case "gain":
					in.operands[4].bytes--
				case "heads":
					in.heads = 8
				case "budget":
					budget = o.allocatedBytes()
				case "theta":
					o.rotary[0].thetaBits = math.Float64bits(math.Inf(1))
				}
				before := f.base.allocations
				e := kvFP8QwenIngressEager(tx, 0, in, 1e-6, 0.0625, budget, kvFP8QwenFixtureOps(t, f, tx, &calls, &roots), func(kvFP8AttentionOutput) int { return 0 })
				extra = f.base.allocations - before
				return e
			})
			if err == nil || extra != 0 || calls != [4]int{} {
				t.Fatal("late ingress refusal", mode, err)
			}
		}
	})
}
