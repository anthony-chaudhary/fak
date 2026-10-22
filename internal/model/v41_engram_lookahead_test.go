package model

import (
	"bytes"
	"context"
	"errors"
	"io"
	"math"
	"runtime"
	"sync"
	"sync/atomic"
	"testing"
)

// Structural gating lets the exact regression compile on the parent. Absence
// is a missing-capability RED, not evidence of a historical concurrency bug.
type v41LookAheadTestFuture = interface {
	wait()
	consume(*v41StepTransaction, int, []uint32, func(*v41EngramGatheredRows) error) error
	cancel()
	close()
}
type v41LookAheadTestFactory interface {
	startV41EngramLookAhead(context.Context, *v41StepTransaction, bool, int, []uint32, float32) (v41LookAheadTestFuture, error)
}

type v41LookAheadTestReader struct {
	io.ReaderAt
	eligible, accept, active, block   atomic.Bool
	reading, consuming, releasedEarly atomic.Bool
	holds, releases, reads            atomic.Int64
	ioCalls                           atomic.Int64
	staging, width                    int
	closing, entered, resume          chan struct{}
	readReturned, released            chan struct{}
	enterOnce, resumeOnce, readOnce   sync.Once
	fault                             error
	panicValue                        any
	exit                              bool
}

func (r *v41LookAheadTestReader) V41EngramLookAheadReadPolicy(rowBytes int) (int, bool) {
	return r.staging, r.eligible.Load() && rowBytes == r.width
}
func (r *v41LookAheadTestReader) V41EngramHoldSourceReads() (func(), <-chan struct{}, bool) {
	r.holds.Add(1)
	if !r.accept.Load() {
		return nil, nil, false
	}
	var once sync.Once
	return func() {
		once.Do(func() {
			if r.reading.Load() || r.consuming.Load() {
				r.releasedEarly.Store(true)
			}
			r.releases.Add(1)
			r.released <- struct{}{}
		})
	}, r.closing, true
}
func (r *v41LookAheadTestReader) unblock() { r.resumeOnce.Do(func() { close(r.resume) }) }
func (r *v41LookAheadTestReader) ReadAt(dst []byte, off int64) (int, error) {
	r.ioCalls.Add(1)
	if !r.active.Load() {
		return r.ReaderAt.ReadAt(dst, off)
	}
	r.reads.Add(1)
	r.reading.Store(true)
	defer r.reading.Store(false)
	r.enterOnce.Do(func() { close(r.entered) })
	if r.block.Load() {
		<-r.resume
	}
	defer r.readOnce.Do(func() { close(r.readReturned) })
	if r.releases.Load() != 0 {
		return 0, errors.New("source owner released before read returned")
	}
	if r.panicValue != nil {
		panic(r.panicValue)
	}
	if r.exit {
		runtime.Goexit()
	}
	if r.fault != nil {
		return 0, r.fault
	}
	return r.ReaderAt.ReadAt(dst, off)
}

func v41LookAheadFixture(t *testing.T) (*Model, *v41LookAheadTestReader, []uint32) {
	t.Helper()
	m, layout := v41FullEngramModel(t)
	data := v41EngramTestPackedRows(int(layout.Rows[0]))
	r := &v41LookAheadTestReader{ReaderAt: bytes.NewReader(data), width: V41EngramPackedRowBytes, closing: make(chan struct{}),
		entered: make(chan struct{}), resume: make(chan struct{}), readReturned: make(chan struct{}), released: make(chan struct{}, 8)}
	r.eligible.Store(true)
	r.accept.Store(true)
	shard := V41EngramShard{ID: "look-ahead-owned", ExpectedID: "look-ahead-owned", Size: int64(len(data)), Rows: int(layout.Rows[0])}
	binding := V41EngramArtifactBinding{Shard: shard.ID, Size: shard.Size, Rows: shard.Rows, RowBytes: V41EngramPackedRowBytes, Digest: v41EngramTestDigest(data)}
	source, err := NewV41VerifiedEngramRowSource(r, shard, binding)
	if err != nil {
		t.Fatal(err)
	}
	if err := m.wireV41Engram(layout, []V41EngramRowSource{source}, 4*V41EngramPackedRowBytes); err != nil {
		t.Fatal(err)
	}
	ids := make([]uint32, m.v41EngramStageFor().columns)
	for i := range ids {
		ids[i] = uint32((len(ids) - i) % 3)
	}
	r.active.Store(true)
	t.Cleanup(func() { r.unblock(); m.v41DetachEngramStage() })
	return m, r, ids
}

// fak-test:runtime fast est=1s lane=default
func TestV41EngramLookAheadOwnership(t *testing.T) {
	if _, ok := any(&Model{}).(v41LookAheadTestFactory); !ok {
		t.Fatal("model is missing the bounded Engram look-ahead future")
	}
	for _, mode := range []string{"ready-credit", "caller-ids", "cancel-read", "closing-owner", "cancel-ready", "context-ready", "closing-ready", "read-error", "panic", "goexit", "wrong-transaction", "row-identity"} {
		t.Run(mode, func(t *testing.T) {
			m, r, ids := v41LookAheadFixture(t)
			capturedIDs := append([]uint32(nil), ids...)
			tx := &v41StepTransaction{}
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			cause := errors.New("look-ahead read fault")
			r.block.Store(mode == "cancel-read" || mode == "closing-owner")
			if mode == "read-error" {
				r.fault = cause
			}
			if mode == "panic" {
				r.panicValue = cause
			}
			if mode == "goexit" {
				r.exit = true
			}
			f, err := any(m).(v41LookAheadTestFactory).startV41EngramLookAhead(ctx, tx, true, 0, ids, float32(m.Cfg.RMSNormEps))
			if err != nil || f == nil {
				t.Fatalf("issue: future=%v err=%v", f, err)
			}
			t.Cleanup(func() { r.unblock(); f.close() })
			<-r.entered
			if mode == "caller-ids" {
				ids[0] = (ids[0] + 1) % 3
			}
			// Another model/session cannot hold a second job while this one runs
			// or owns a ready panel. This attempt must not acquire another hold.
			if mode == "cancel-read" || mode == "closing-owner" {
				if mode == "cancel-read" {
					cancel()
					f.cancel()
				} else {
					close(r.closing)
					r.accept.Store(false)
				}
				joined := make(chan struct{})
				go func() { f.close(); close(joined) }()
				r.unblock()
				<-joined
				<-r.readReturned
				if r.releases.Load() != 1 || r.releasedEarly.Load() {
					t.Fatal("joined abandonment did not release exactly once")
				}
				if err := f.consume(tx, 0, ids, func(*v41EngramGatheredRows) error { t.Error("abandoned future consumed"); return nil }); err == nil {
					t.Fatal("abandoned future accepted consumption")
				}
				f.close()
				return
			}
			f.wait()
			holds := r.holds.Load()
			busyModel, busyReader, busyIDs := m, r, ids
			if mode == "ready-credit" {
				busyModel, busyReader, busyIDs = v41LookAheadFixture(t)
			}
			busyHolds := busyReader.holds.Load()
			busy, busyErr := any(busyModel).(v41LookAheadTestFactory).startV41EngramLookAhead(ctx, &v41StepTransaction{}, true, 0, busyIDs, float32(busyModel.Cfg.RMSNormEps))
			if busy != nil || busyErr != nil || r.holds.Load() != holds || busyReader.holds.Load() != busyHolds || r.releases.Load() != 0 {
				t.Fatal("ready panel lost its process-wide credit or source hold")
			}
			useTx, useIDs := tx, append([]uint32(nil), capturedIDs...)
			if mode == "wrong-transaction" {
				useTx = &v41StepTransaction{}
			}
			if mode == "row-identity" {
				useIDs[0] = (useIDs[0] + 1) % 3
			}
			if mode == "cancel-ready" {
				f.cancel()
			}
			if mode == "context-ready" {
				cancel()
			}
			if mode == "closing-ready" {
				close(r.closing)
			}
			calls := 0
			var recovered any
			func() {
				defer func() { recovered = recover() }()
				err = f.consume(useTx, 0, useIDs, func(g *v41EngramGatheredRows) error {
					calls++
					if len(g.rows) != len(ids) {
						t.Fatal("owned panel lost rows")
					}
					for i, row := range capturedIDs {
						want := v41EngramTestPackedRows(int(row) + 1)[int(row)*V41EngramPackedRowBytes:]
						if !bytes.Equal(g.rows[i], want) {
							t.Fatalf("row %d changed", i)
						}
					}
					if r.releases.Load() != 0 {
						t.Fatal("owner released during consumption")
					}
					return nil
				})
			}()
			switch mode {
			case "ready-credit", "caller-ids":
				if err != nil || calls != 1 || recovered != nil {
					t.Fatalf("consume: calls=%d err=%v panic=%v", calls, err, recovered)
				}
			case "panic":
				if recovered != cause || calls != 0 {
					t.Fatal("gather panic was not deferred to consumption")
				}
			case "read-error":
				if !errors.Is(err, cause) || calls != 0 {
					t.Fatalf("gather error changed: %v", err)
				}
			case "cancel-ready", "context-ready":
				if !errors.Is(err, context.Canceled) || calls != 0 || recovered != nil {
					t.Fatalf("cancellation changed: %v", err)
				}
			default:
				if !errors.Is(err, ErrV41NativeUnsupported) || calls != 0 || recovered != nil {
					t.Fatalf("failed future consumed: calls=%d err=%v panic=%v", calls, err, recovered)
				}
			}
			f.close()
			f.close()
			if r.releases.Load() != 1 || r.releasedEarly.Load() {
				t.Fatal("future release was not exactly once")
			}
			if err := f.consume(tx, 0, ids, func(*v41EngramGatheredRows) error { calls++; return nil }); err == nil {
				t.Fatal("future replay was accepted")
			}
		})
	}
	t.Run("close-during-consumption", func(t *testing.T) {
		m, r, ids := v41LookAheadFixture(t)
		tx := &v41StepTransaction{}
		f, err := any(m).(v41LookAheadTestFactory).startV41EngramLookAhead(context.Background(), tx, true, 0, ids, float32(m.Cfg.RMSNormEps))
		if err != nil || f == nil {
			t.Fatalf("issue: %v", err)
		}
		entered, resume := make(chan struct{}), make(chan struct{})
		var once sync.Once
		unblock := func() { once.Do(func() { close(resume) }) }
		t.Cleanup(func() { unblock(); f.close() })
		result := make(chan error, 1)
		go func() {
			result <- f.consume(tx, 0, ids, func(*v41EngramGatheredRows) error {
				r.consuming.Store(true)
				defer r.consuming.Store(false)
				close(entered)
				<-resume
				return nil
			})
		}()
		<-entered
		f.cancel()
		joined := make(chan struct{})
		go func() { f.close(); close(joined) }()
		unblock()
		if err := <-result; !errors.Is(err, context.Canceled) {
			t.Fatalf("consumption cancellation changed: %v", err)
		}
		<-joined
		if r.releases.Load() != 1 || r.releasedEarly.Load() {
			t.Fatal("owner retired before joined consumption")
		}
	})
	for _, mode := range []string{"callback-panic", "callback-goexit"} {
		t.Run(mode, func(t *testing.T) {
			m, r, ids := v41LookAheadFixture(t)
			tx := &v41StepTransaction{}
			f, err := any(m).(v41LookAheadTestFactory).startV41EngramLookAhead(context.Background(), tx, true, 0, ids, float32(m.Cfg.RMSNormEps))
			if err != nil || f == nil {
				t.Fatalf("issue: %v", err)
			}
			t.Cleanup(f.close)
			cause := errors.New(mode)
			done := make(chan struct{})
			var recovered any
			returned := false
			go func() {
				defer close(done)
				defer func() { recovered = recover() }()
				_ = f.consume(tx, 0, ids, func(*v41EngramGatheredRows) error {
					if mode == "callback-panic" {
						panic(cause)
					}
					runtime.Goexit()
					return nil
				})
				returned = true
			}()
			<-done
			if returned || (mode == "callback-panic" && recovered != cause) || (mode == "callback-goexit" && recovered != nil) {
				t.Fatal("callback termination was swallowed or changed")
			}
			f.close()
			if r.releases.Load() != 1 || r.releasedEarly.Load() {
				t.Fatal("callback unwind stranded or released ownership early")
			}
			// A new model must be able to reserve the process credit after unwind.
			other, otherReader, otherIDs := v41LookAheadFixture(t)
			next, err := any(other).(v41LookAheadTestFactory).startV41EngramLookAhead(context.Background(), &v41StepTransaction{}, true, 0, otherIDs, float32(other.Cfg.RMSNormEps))
			if err != nil || next == nil {
				t.Fatalf("callback unwind stranded global credit: %v", err)
			}
			next.close()
			if otherReader.releases.Load() != 1 {
				t.Fatal("next job ownership not retired")
			}
		})
	}
	t.Run("preflight", func(t *testing.T) {
		m, r, ids := v41LookAheadFixture(t)
		factory := any(m).(v41LookAheadTestFactory)
		if f, err := factory.startV41EngramLookAhead(nil, nil, false, -1, nil, 0); f != nil || err != nil {
			t.Fatal("disabled default did work")
		}
		r.eligible.Store(false)
		if f, err := factory.startV41EngramLookAhead(context.Background(), &v41StepTransaction{}, true, 0, ids, float32(m.Cfg.RMSNormEps)); f != nil || err != nil {
			t.Fatal("ineligible source entered async")
		}
		r.eligible.Store(true)
		r.accept.Store(false)
		if f, err := factory.startV41EngramLookAhead(context.Background(), &v41StepTransaction{}, true, 0, ids, float32(m.Cfg.RMSNormEps)); f != nil || !errors.Is(err, ErrV41NativeUnsupported) {
			t.Fatal("eligible failed hold was converted to fallback")
		}
		r.accept.Store(true)
		holds := r.holds.Load()
		for _, bad := range []string{"bounds", "count", "gate", "prefetch", "staging", "cache-capacity", "resident-budget"} {
			badIDs := append([]uint32(nil), ids...)
			eps := float32(m.Cfg.RMSNormEps)
			cache := m.v41EngramStageFor().caches[0]
			oldBudget, oldOrder := cache.opts.BudgetBytes, cache.inner.order
			switch bad {
			case "bounds":
				badIDs[0] = math.MaxUint32
			case "count":
				badIDs = badIDs[:len(badIDs)-1]
			case "gate":
				eps = float32(math.NaN())
			case "prefetch":
				cache.opts.PrefetchRows = 1
			case "staging":
				r.staging = 1 << 20
			case "cache-capacity":
				cache.inner.order = make([]v41ExpertKey, 0, 1024)
			case "resident-budget":
				cache.opts.BudgetBytes = 257 * int64(cache.opts.RowBytes)
				cache.inner.budgetBytes = cache.opts.BudgetBytes
			}
			f, err := factory.startV41EngramLookAhead(context.Background(), &v41StepTransaction{}, true, 0, badIDs, eps)
			cache.opts.PrefetchRows = 0
			cache.opts.BudgetBytes, cache.inner.budgetBytes, cache.inner.order = oldBudget, oldBudget, oldOrder
			r.staging = 0
			if f != nil || err == nil || r.holds.Load() != holds || r.reads.Load() != 0 {
				t.Fatalf("%s escaped preflight: %v", bad, err)
			}
		}
		f, err := factory.startV41EngramLookAhead(context.Background(), &v41StepTransaction{}, true, 0, ids, float32(m.Cfg.RMSNormEps))
		if err != nil || f == nil {
			t.Fatalf("refused hold leaked global credit: %v", err)
		}
		f.close()
		if r.releases.Load() != 1 {
			t.Fatal("final admitted job did not release")
		}
	})
}

type v41LookAheadEncodedTestReader struct {
	*v41LookAheadTestReader
	encoded []byte
}

func (r *v41LookAheadEncodedTestReader) V41EngramEncodedRows(start, count int) (io.ReaderAt, int64, int64, string, error) {
	return bytes.NewReader(r.encoded), int64(start) * v41EngramQ2KEncodedRowBytes, int64(count) * v41EngramQ2KEncodedRowBytes, V41EngramEncodingQ2K, nil
}

// fak-test:runtime fast est=100ms lane=default
func TestV41EngramLookAheadPolicyWiring(t *testing.T) {
	t.Run("nil-and-unaligned", func(t *testing.T) {
		probe := &v41LookAheadTestReader{ReaderAt: bytes.NewReader(nil), width: V41EngramPackedRowBytes}
		probe.eligible.Store(true)
		for _, source := range []any{(*v41PackedEngramRowSource)(nil), (*v41VerifiedEngramRowSource)(nil), (*v41EngramQ2KRowSource)(nil),
			&v41PackedEngramRowSource{reader: probe, rowBytes: V41EngramPackedRowBytes, shard: V41EngramShard{Offset: 1}},
			&v41PackedEngramRowSource{reader: bytes.NewReader(nil), rowBytes: V41EngramPackedRowBytes}} {
			if policy, exists := source.(interface{ V41EngramLookAheadReadPolicy(int) (int, bool) }); exists {
				if _, eligible := policy.V41EngramLookAheadReadPolicy(V41EngramPackedRowBytes); eligible {
					t.Fatal("nil, unaligned or undeclared source admitted look-ahead")
				}
			}
		}
	})
	cfg := Config{VocabSize: 4, DeepSeekV41: &DeepSeekV41Config{EngramLayerIDs: []int{1}, EngramHeadDim: 4}}
	layout := v41EngramTestLayout(cfg)
	rows := int(layout.Rows[0])
	packed := v41EngramTestPackedRows(rows)
	f32 := v41EngramF32RowsForPacked(packed, rows, 256)
	encoded := make([]byte, rows*v41EngramQ2KEncodedRowBytes)
	for _, constructor := range []string{"verified", "acquired"} {
		for _, dialect := range []string{"packed", "f32", "q2-k"} {
			t.Run(constructor+"/"+dialect, func(t *testing.T) {
				data, width := packed, V41EngramPackedRowBytes
				if dialect != "packed" {
					data, width = f32, V41EngramF32RowBytes
				}
				probe := &v41LookAheadTestReader{ReaderAt: bytes.NewReader(data), staging: 117, width: width}
				probe.eligible.Store(true)
				var reader io.ReaderAt = probe
				if dialect == "q2-k" {
					reader = &v41LookAheadEncodedTestReader{v41LookAheadTestReader: probe, encoded: encoded}
				}
				binding := V41EngramArtifactBinding{Shard: "look-ahead-policy", Size: int64(len(data)), Rows: rows, RowBytes: width}
				shard := V41EngramShard{ID: binding.Shard, ExpectedID: binding.Shard, Size: binding.Size, Rows: rows}
				if constructor == "verified" {
					binding.Digest = v41EngramTestDigest(data)
					if dialect == "q2-k" {
						var err error
						binding.Digest, err = v41EngramTreeDigest(bytes.NewReader(encoded), 0, int64(len(encoded)), V41EngramEncodingQ2K, 1)
						if err != nil {
							t.Fatal(err)
						}
					}
				}
				var source V41EngramRowSource
				var err error
				if constructor == "verified" {
					source, err = NewV41VerifiedEngramRowSource(reader, shard, binding)
				} else {
					source, err = NewV41AcquiredEngramRowSource(reader, shard, binding)
				}
				if err != nil {
					t.Fatal(err)
				}
				m := &Model{Cfg: cfg}
				t.Cleanup(m.v41DetachEngramStage)
				if err := m.wireV41Engram(layout, []V41EngramRowSource{source}, int64(width)); err != nil {
					t.Fatal(err)
				}
				beforeIO := probe.ioCalls.Load()
				policy, ok := m.v41EngramStageFor().caches[0].src.(interface{ V41EngramLookAheadReadPolicy(int) (int, bool) })
				if !ok {
					t.Fatal("wired source is missing explicit look-ahead policy propagation")
				}
				if cost, eligible := policy.V41EngramLookAheadReadPolicy(m.v41EngramStageFor().caches[0].src.RowBytes()); cost != 117 || !eligible {
					t.Fatalf("policy changed: %d/%t", cost, eligible)
				}
				if _, eligible := policy.V41EngramLookAheadReadPolicy(m.v41EngramStageFor().caches[0].src.RowBytes() + 1); eligible {
					t.Fatal("wrapper promoted a different row geometry")
				}
				probe.eligible.Store(false)
				if _, eligible := policy.V41EngramLookAheadReadPolicy(m.v41EngramStageFor().caches[0].src.RowBytes()); eligible {
					t.Fatal("ineligible producer was promoted")
				}
				if probe.holds.Load() != 0 || probe.ioCalls.Load() != beforeIO {
					t.Fatal("eligibility query acquired a source hold or performed I/O")
				}
			})
		}
	}
}
