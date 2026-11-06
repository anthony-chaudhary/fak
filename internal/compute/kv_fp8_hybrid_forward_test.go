package compute

import (
	"errors"
	"math"
	"runtime"
	"sync/atomic"
	"testing"
	"time"
	"unsafe"
)

func kvFP8HybridFixture(o *kvFP8Owner) (kvFP8HybridContract, []kvFP8HybridState) {
	c := kvFP8HybridContract{first: 4, tokens: 1, qkNorm: true, linear: make([]bool, 4*o.layers), theta: make([]uint64, 4*o.layers)}
	for i := range c.linear {
		c.linear[i] = (i+1)%4 != 0
		c.theta[i] = math.Float64bits(10000)
		if !c.linear[i] {
			c.theta[i] = o.rotary[i/4].thetaBits
		}
	}
	states := make([]kvFP8HybridState, 6*o.layers)
	for i := range states {
		owner := &struct {
			bytes   [16]byte
			invalid uint32
		}{}
		states[i] = kvFP8HybridState{backend: o.backend, device: 0, mainLayer: (i/6)*4 + (i%6)/2, kind: i % 2,
			storage: kvFP8Allocation{unsafe.Pointer(&owner.bytes[0]), 16}, invalid: &owner.invalid, root: owner}
	}
	return c, states
}
func kvFP8HybridFixtureSteps(f *kvFP8AppendFake) []func(*kvFP8AppendTransaction, int) error {
	steps := make([]func(*kvFP8AppendTransaction, int) error, 4*f.o.layers)
	for i := range steps {
		steps[i] = func(tx *kvFP8AppendTransaction, main int) error {
			if (main+1)%4 != 0 {
				return nil
			}
			layer := main / 4
			if err := tx.append(layer, f.input(layer, 1)); err != nil {
				return err
			}
			return tx.consume(layer, func(kvFP8AppendLayer) int { return 0 })
		}
	}
	return steps
}

// fak-test:runtime fast est=100ms lane=default
func TestKVFP8HybridForwardGuard(t *testing.T) {
	savedOwners, savedAppend, savedHybrid := kvFP8RetainedOwners, kvFP8AppendRetained, kvFP8HybridRetained
	defer func() {
		kvFP8RetainedOwners = savedOwners
		kvFP8AppendRetained = savedAppend
		kvFP8HybridRetained = savedHybrid
	}()
	t.Run("whole hybrid checked before publication", func(t *testing.T) {
		o, f := kvFP8AppendFixture(t)
		c, s := kvFP8HybridFixture(o)
		tail := false
		err := kvFP8HybridForwardEager(o, c, s, 1<<20, f.ops(), func() error { return nil }, kvFP8HybridFixtureSteps(f),
			func(tx *kvFP8AppendTransaction) error {
				tail = true
				if o.image.positions != 4 || o.usable() == nil || tx.next != 2 {
					t.Fatal("published before tail")
				}
				return nil
			})
		if err != nil || !tail || f.checks != 17 || o.image.positions != 5 || o.usable() != nil {
			t.Fatal("hybrid completion", err, f.checks)
		}
		for _, x := range s {
			if atomic.LoadUint32(x.invalid) != 0 {
				t.Fatal("successful GDN poisoned")
			}
		}
	})
	t.Run("query refusal precedes outer append allocation", func(t *testing.T) {
		o, f := kvFP8AppendFixture(t)
		c, s := kvFP8HybridFixture(o)
		before := f.base.allocations
		refusal := errors.New("timing unsupported")
		err := kvFP8HybridForwardEager(o, c, s, 1<<20, f.ops(), func() error { return refusal }, kvFP8HybridFixtureSteps(f), func(*kvFP8AppendTransaction) error { t.Fatal("tail ran"); return nil })
		if err != refusal || f.base.allocations != before || f.checks != 0 || o.usable() != nil {
			t.Fatal("effectful decline", err)
		}
		for _, x := range s {
			if atomic.LoadUint32(x.invalid) != 0 {
				t.Fatal("declined healthy state poisoned")
			}
		}
	})
	t.Run("known numerical GDN failure poisons both without device loss", func(t *testing.T) {
		o, f := kvFP8AppendFixture(t)
		c, s := kvFP8HybridFixture(o)
		f.rejectCheck = 3
		err := kvFP8HybridForwardEager(o, c, s, 1<<20, f.ops(), func() error { return nil }, kvFP8HybridFixtureSteps(f), func(*kvFP8AppendTransaction) error { t.Fatal("rejected tail"); return nil })
		if !errors.Is(err, ErrKVFP8NonFinite) || o.usable() == nil || o.backend.stopped.Load() || o.retained || f.ingests != 0 {
			t.Fatal("numerical failure scope", err)
		}
		for _, x := range s {
			if atomic.LoadUint32(x.invalid) != 1 {
				t.Fatal("GDN not poisoned")
			}
		}
	})
	for _, mode := range []string{"step panic", "tail panic", "step Goexit"} {
		t.Run(mode, func(t *testing.T) {
			o, f := kvFP8AppendFixture(t)
			c, s := kvFP8HybridFixture(o)
			steps := kvFP8HybridFixtureSteps(f)
			tail := func(*kvFP8AppendTransaction) error { return nil }
			if mode == "step panic" {
				steps[1] = func(*kvFP8AppendTransaction, int) error { panic("hybrid panic") }
			}
			if mode == "tail panic" {
				tail = func(*kvFP8AppendTransaction) error { panic("hybrid panic") }
			}
			if mode == "step Goexit" {
				steps[1] = func(*kvFP8AppendTransaction, int) error { runtime.Goexit(); return nil }
			}
			done := make(chan struct{})
			var recovered any
			returned := false
			go func() {
				defer close(done)
				defer func() { recovered = recover() }()
				_ = kvFP8HybridForwardEager(o, c, s, 1<<20, f.ops(), func() error { return nil }, steps, tail)
				returned = true
			}()
			select {
			case <-done:
			case <-time.After(5 * time.Second):
				t.Fatal("abnormal forward did not unwind")
			}
			if returned || (mode == "step Goexit" && recovered != nil) || (mode != "step Goexit" && recovered != "hybrid panic") ||
				o.usable() == nil || !o.retained || !o.backend.stopped.Load() ||
				kvFP8HybridRetained == nil || kvFP8HybridRetained.owner != o {
				t.Fatal("control transfer/quarantine", recovered)
			}
			for i, x := range s {
				if atomic.LoadUint32(x.invalid) != 1 || kvFP8HybridRetained.states[i].root != x.root {
					t.Fatal("lost exact GDN root")
				}
			}
			before := f.base.allocations
			err := kvFP8HybridForwardEager(o, c, s, 1<<20, f.ops(), func() error { t.Fatal("retry query"); return nil }, steps, tail)
			if err == nil || f.base.allocations != before {
				t.Fatal("retry admitted")
			}
		})
	}
	t.Run("silent tail state poison prevents KV publication", func(t *testing.T) {
		o, f := kvFP8AppendFixture(t)
		c, s := kvFP8HybridFixture(o)
		err := kvFP8HybridForwardEager(o, c, s, 1<<20, f.ops(), func() error { return nil }, kvFP8HybridFixtureSteps(f),
			func(*kvFP8AppendTransaction) error { atomic.StoreUint32(s[0].invalid, 1); return nil })
		if err == nil || o.image.positions != 4 || o.usable() == nil || !o.backend.stopped.Load() {
			t.Fatal("incoherent state published", err)
		}
	})
	t.Run("attention cannot advance at a GDN main layer", func(t *testing.T) {
		o, f := kvFP8AppendFixture(t)
		c, s := kvFP8HybridFixture(o)
		steps := kvFP8HybridFixtureSteps(f)
		steps[0] = func(tx *kvFP8AppendTransaction, _ int) error {
			if e := tx.append(0, f.input(0, 1)); e != nil {
				return e
			}
			return tx.consume(0, func(kvFP8AppendLayer) int { return 0 })
		}
		err := kvFP8HybridForwardEager(o, c, s, 1<<20, f.ops(), func() error { return nil }, steps, func(*kvFP8AppendTransaction) error { return nil })
		if err == nil || o.usable() == nil || !o.backend.stopped.Load() {
			t.Fatal("wrong main order accepted")
		}
		for _, x := range s {
			if atomic.LoadUint32(x.invalid) != 1 {
				t.Fatal("missed failed forward state")
			}
		}
	})
	for _, mode := range []string{"scaled", "long", "fused", "sparse", "graph", "norm", "theta", "mapping", "state alias", "invalid alias", "state poison"} {
		t.Run("preflight "+mode, func(t *testing.T) {
			o, f := kvFP8AppendFixture(t)
			c, s := kvFP8HybridFixture(o)
			before := f.base.allocations
			switch mode {
			case "scaled":
				c.scaledRoPE = true
			case "long":
				c.longRoPE = true
			case "fused":
				c.fusedQK = true
			case "sparse":
				c.sparseAttention = true
			case "graph":
				c.graph = true
			case "norm":
				c.qkNorm = false
			case "theta":
				c.theta[3] = math.Float64bits(9000)
			case "mapping":
				c.linear[0] = false
			case "state alias":
				s[1].storage = s[0].storage
			case "invalid alias":
				s[1].invalid = s[0].invalid
			case "state poison":
				atomic.StoreUint32(s[0].invalid, 1)
			}
			err := kvFP8HybridForwardEager(o, c, s, 1<<20, f.ops(), func() error { t.Fatal("preflight reached query"); return nil }, kvFP8HybridFixtureSteps(f), func(*kvFP8AppendTransaction) error { return nil })
			if err == nil || f.base.allocations != before || o.usable() != nil {
				t.Fatal("preflight effects", mode, err)
			}
		})
	}
}
