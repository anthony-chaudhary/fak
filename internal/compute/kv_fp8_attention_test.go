package compute

import (
	"errors"
	"runtime"
	"testing"
	"time"
	"unsafe"
)

func kvFP8AttentionFixtureQuery(f *kvFP8AppendFake, heads int) kvFP8AttentionQuery {
	b := make([]byte, heads*256*4)
	p := unsafe.Pointer(&b[0])
	f.base.data[p] = b
	return kvFP8AttentionQuery{backend: f.o.backend, device: 0, storage: kvFP8Allocation{p, len(b)}, root: b}
}

// These host controls establish ordering, metadata and roots only. The launch
// writes sentinel bytes, not an attention oracle or CUDA accuracy witness.
// fak-test:runtime fast est=100ms lane=default
func TestKVFP8StagedAttentionOwnership(t *testing.T) {
	savedOwners, savedAppend := kvFP8RetainedOwners, kvFP8AppendRetained
	defer func() { kvFP8RetainedOwners = savedOwners; kvFP8AppendRetained = savedAppend }()
	t.Run("two layers hold Q output then retire only after checked consumer", func(t *testing.T) {
		o, f := kvFP8AppendFixture(t)
		var roots []*kvFP8AttentionRoots
		launches, uses := 0, 0
		err := kvFP8AppendEager(o, 4, 1, 1<<20, f.ops(), func(tx *kvFP8AppendTransaction) error {
			for l := 0; l < o.layers; l++ {
				if err := tx.append(l, f.input(l, 1)); err != nil {
					return err
				}
				q := kvFP8AttentionFixtureQuery(f, 8)
				if err := kvFP8AttendEager(tx, l, q, 8, 0.0625, 1<<20,
					func(k kvFP8AppendLayer, q, out kvFP8Allocation, heads int, scale float32) int {
						launches++
						root, ok := tx.inputs[l].root.(*kvFP8AttentionRoots)
						if !ok || root.query == nil || root.prior == nil || root.output.ptr != out.ptr ||
							out.bytes != 8*256*4 || k.prefix != 4 || k.tokens != 1 || heads != 8 || scale != 0.0625 {
							t.Fatal("missing exact source/output ownership")
						}
						roots = append(roots, root)
						for i := range f.base.data[out.ptr] {
							f.base.data[out.ptr][i] = 0xa0
						}
						return 0
					}, func(out kvFP8AttentionOutput) int {
						uses++
						if f.checks != 4+4*l || out.tokens != 1 || out.heads != 8 || out.headDim != 256 ||
							out.status.ptr != tx.stage.workspace[0].ptr || len(f.base.data[out.storage.ptr]) != 8*256*4 {
							t.Fatal("consumer preceded checked attention")
						}
						return 0
					}); err != nil {
					return err
				}
				if roots[l].output.ptr != nil || f.base.releases != l+1 {
					t.Fatal("output retirement")
				}
			}
			return nil
		})
		if err != nil || launches != 2 || uses != 2 || f.checks != 10 || f.base.releases != 13 ||
			o.image.positions != 5 || o.retained || o.backend.stopped.Load() {
			t.Fatal("commit", err)
		}
	})
	t.Run("numerical attention and consumer rejection cannot expose output", func(t *testing.T) {
		for _, at := range []int{4, 5} {
			o, f := kvFP8AppendFixture(t)
			f.rejectCheck = at
			uses := 0
			var root *kvFP8AttentionRoots
			err := kvFP8AppendEager(o, 4, 1, 1<<20, f.ops(), func(tx *kvFP8AppendTransaction) error {
				if err := tx.append(0, f.input(0, 1)); err != nil {
					return err
				}
				err := kvFP8AttendEager(tx, 0, kvFP8AttentionFixtureQuery(f, 8), 8, 1, 1<<20,
					func(k kvFP8AppendLayer, q, out kvFP8Allocation, h int, s float32) int { return 0 },
					func(kvFP8AttentionOutput) int { uses++; return 0 })
				root = tx.inputs[0].root.(*kvFP8AttentionRoots)
				return err
			})
			wantUse := 0
			if at == 5 {
				wantUse = 1
			}
			if !errors.Is(err, ErrKVFP8NonFinite) || uses != wantUse || root == nil || root.output.ptr != nil ||
				f.base.releases != 12 || o.stage != nil || o.usable() == nil || o.retained || o.backend.stopped.Load() {
				t.Fatal("numeric output boundary", at, err)
			}
		}
	})
	t.Run("uncertain fence retains exact Q output and images", func(t *testing.T) {
		o, f := kvFP8AppendFixture(t)
		f.statusCheck = 4
		var tx *kvFP8AppendTransaction
		uses := 0
		err := kvFP8AppendEager(o, 4, 1, 1<<20, f.ops(), func(current *kvFP8AppendTransaction) error {
			tx = current
			if err := tx.append(0, f.input(0, 1)); err != nil {
				return err
			}
			return kvFP8AttendEager(tx, 0, kvFP8AttentionFixtureQuery(f, 8), 8, 1, 1<<20,
				func(k kvFP8AppendLayer, q, out kvFP8Allocation, h int, s float32) int { return 0 },
				func(kvFP8AttentionOutput) int { uses++; return 0 })
		})
		root, ok := tx.inputs[0].root.(*kvFP8AttentionRoots)
		if !errors.Is(err, errKVFP8Completion) || !ok || root.query == nil || root.prior == nil ||
			root.output.ptr == nil || kvFP8AppendRetained != tx || o.stage != tx.stage ||
			!o.retained || f.base.releases != 0 || uses != 0 {
			t.Fatal("uncertain output freed", err)
		}
	})
	t.Run("caught launch and consumer panic stop before retry", func(t *testing.T) {
		for _, phase := range []string{"launch", "use"} {
			o, f := kvFP8AppendFixture(t)
			var tx *kvFP8AppendTransaction
			var recovered any
			retryCalled := false
			err := kvFP8AppendEager(o, 4, 1, 1<<20, f.ops(), func(current *kvFP8AppendTransaction) error {
				tx = current
				if err := tx.append(0, f.input(0, 1)); err != nil {
					return err
				}
				q := kvFP8AttentionFixtureQuery(f, 8)
				func() {
					defer func() { recovered = recover() }()
					_ = kvFP8AttendEager(tx, 0, q, 8, 1, 1<<20,
						func(k kvFP8AppendLayer, q, out kvFP8Allocation, h int, s float32) int {
							if phase == "launch" {
								panic("attention-unwind")
							}
							return 0
						},
						func(kvFP8AttentionOutput) int { panic("attention-unwind") })
				}()
				if !o.uncertain || !tx.retained || !o.backend.stopped.Load() {
					t.Fatal("caller recovered before quarantine")
				}
				_ = kvFP8AttendEager(tx, 0, q, 8, 1, 1<<20,
					func(k kvFP8AppendLayer, q, out kvFP8Allocation, h int, s float32) int { retryCalled = true; return 0 },
					func(kvFP8AttentionOutput) int { retryCalled = true; return 0 })
				return nil
			})
			root := tx.inputs[0].root.(*kvFP8AttentionRoots)
			if err == nil || recovered != "attention-unwind" || retryCalled || root.output.ptr == nil ||
				f.base.releases != 0 || kvFP8AppendRetained != tx {
				t.Fatal("caught panic revived output", phase, err)
			}
		}
	})
	t.Run("consumer Goexit retains without normal return", func(t *testing.T) {
		o, f := kvFP8AppendFixture(t)
		done := make(chan struct{})
		normal := false
		var tx *kvFP8AppendTransaction
		var recovered any
		go func() {
			defer close(done)
			defer func() { recovered = recover() }()
			_ = kvFP8AppendEager(o, 4, 1, 1<<20, f.ops(), func(current *kvFP8AppendTransaction) error {
				tx = current
				if err := tx.append(0, f.input(0, 1)); err != nil {
					return err
				}
				return kvFP8AttendEager(tx, 0, kvFP8AttentionFixtureQuery(f, 8), 8, 1, 1<<20,
					func(k kvFP8AppendLayer, q, out kvFP8Allocation, h int, s float32) int { return 0 },
					func(kvFP8AttentionOutput) int { runtime.Goexit(); return 0 })
			})
			normal = true
		}()
		select {
		case <-done:
		case <-time.After(5 * time.Second):
			t.Fatal("attention Goexit timed out")
		}
		root := tx.inputs[0].root.(*kvFP8AttentionRoots)
		if normal || recovered != nil || !o.retained || !o.backend.stopped.Load() ||
			root.output.ptr == nil || f.base.releases != 0 || tx.active {
			t.Fatal("Goexit ownership")
		}
	})
	t.Run("retired disposition survives later bookkeeping error", func(t *testing.T) {
		o, f := kvFP8AppendFixture(t)
		ops := f.ops()
		release := ops.eager.owner.release
		ops.eager.owner.release = func(a kvFP8Allocation) kvFP8Retirement {
			r := release(a)
			r.status = 19
			return r
		}
		var tx *kvFP8AppendTransaction
		err := kvFP8AppendEager(o, 4, 1, 1<<20, ops, func(current *kvFP8AppendTransaction) error {
			tx = current
			if err := tx.append(0, f.input(0, 1)); err != nil {
				return err
			}
			return kvFP8AttendEager(tx, 0, kvFP8AttentionFixtureQuery(f, 8), 8, 1, 1<<20,
				func(k kvFP8AppendLayer, q, out kvFP8Allocation, h int, s float32) int { return 0 },
				func(kvFP8AttentionOutput) int { return 0 })
		})
		root := tx.inputs[0].root.(*kvFP8AttentionRoots)
		if !errors.Is(err, errKVFP8Retirement) || root.output.ptr != nil || f.base.releases != 1 ||
			!o.retained || o.stage != tx.stage {
			t.Fatal("double-retire risk", err)
		}
	})
	t.Run("geometry and incremental output budget precede allocation", func(t *testing.T) {
		for _, mode := range []string{"heads", "query", "budget"} {
			o, f := kvFP8AppendFixture(t)
			extra := 0
			launched := false
			err := kvFP8AppendEager(o, 4, 1, 1<<20, f.ops(), func(tx *kvFP8AppendTransaction) error {
				if err := tx.append(0, f.input(0, 1)); err != nil {
					return err
				}
				q := kvFP8AttentionFixtureQuery(f, 8)
				heads, budget := 8, int64(1<<20)
				if mode == "heads" {
					heads = 5
				}
				if mode == "query" {
					q.storage.bytes--
				}
				if mode == "budget" {
					budget = o.allocatedBytes()
				}
				before := f.base.allocations
				err := kvFP8AttendEager(tx, 0, q, heads, 1, budget,
					func(k kvFP8AppendLayer, q, out kvFP8Allocation, h int, s float32) int { launched = true; return 0 },
					func(kvFP8AttentionOutput) int { return 0 })
				extra = f.base.allocations - before
				return err
			})
			if err == nil || extra != 0 || launched {
				t.Fatal("late refusal", mode, err)
			}
		}
	})
}
