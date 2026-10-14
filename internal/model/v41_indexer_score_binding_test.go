package model

import (
	"errors"
	"reflect"
	"sync"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// The wrapper adds a real execution domain while preserving the tensors'
// outer backend identity. Its inner CPU recorder is also the no-owner control.
type v41BorrowScoreBackend struct {
	*v41ScoreTestBackend
	owner *sync.Mutex
}

func newV41BorrowScoreBackend() *v41BorrowScoreBackend {
	b := &v41BorrowScoreBackend{v41ScoreTestBackend: newV41ScoreTestBackend(), owner: new(sync.Mutex)}
	b.identity = b
	return b
}

func (b *v41BorrowScoreBackend) ForwardExecutionOwner() *sync.Mutex { return b.owner }

// These are CPU execution/ownership and host-copy witnesses, not GPU timing.
// The estimate remains unmeasured until the separately authorized test run.
// fak-test:runtime medium est=5s lane=default
func TestV41IndexerScoreBinding(t *testing.T) {
	fixture := func(t *testing.T) (*Model, *Session, *v41BorrowScoreBackend, *v41ProjScratch) {
		t.Helper()
		m := v41IndexerTestFixture(t)
		b := newV41BorrowScoreBackend()
		s := v41DenseTestSession(t, m, b)
		v41ScoreOnly(s)
		scope := s.beginBackendExecution()
		t.Cleanup(scope.finish)
		st := s.v41State()
		return m, s, b, &v41ProjScratch{indexerScoreBinding: st.indexerScoreBinding}
	}
	operands := func(m *Model) ([]float32, []float32, [][]float32) {
		q, hidden := make([]float32, m.Cfg.QLoraRank), make([]float32, m.Cfg.HiddenSize)
		for i := range q {
			q[i] = float32(i%7-3) / 8
		}
		for i := range hidden {
			hidden[i] = float32(i%5-2) / 8
		}
		keys := make([][]float32, 8)
		for r := range keys {
			keys[r] = make([]float32, m.Cfg.IndexHeadDim)
			for d := range keys[r] {
				keys[r][d] = float32((r*7+d)%19-9) / 16
			}
		}
		return q, hidden, keys
	}

	t.Run("canonical-prefixes-and-copy-allocation-bound", func(t *testing.T) {
		m, s, b, scratch := fixture(t)
		st, binding := s.v41State(), scratch.indexerScoreBinding
		q, hidden, keys := operands(m)
		const seq, ratio = 16, 2
		dim := m.Cfg.IndexHeadDim
		borrow := binding.prepareBorrow(m, st, scratch)
		if !borrow.valid() {
			t.Fatal("real owned canonical binding was refused")
		}
		var expected []float32
		b.onKeyUpload = func(flat []float32) {
			if expected != nil && (len(flat) != len(expected) || cap(flat) != len(expected) || &flat[0] != &expected[0]) {
				t.Fatal("canonical upload did not borrow the exact causal prefix")
			}
		}
		panel := v41ColdIndexerKeyPanel(keys, dim)
		copyElements := 0
		for pos := 0; pos < seq; pos++ {
			groups := (pos + 1) / ratio
			end := groups * dim
			expected = panel[:end:end]
			before := b.healthCalls
			got, err := m.v41IndexRowsWithPreparedBinding(0, pos, q, hidden, keys[:groups], nil, binding, st.indexScoreHealth, nil, expected, &borrow)
			expected = nil
			want, wantErr := m.v41IndexRowsWithDeviceOps(0, pos, q, hidden, keys[:groups], nil, binding.scoreFunc(), st.indexScoreHealth, nil)
			if err != nil || wantErr != nil || !reflect.DeepEqual(got, want) || b.healthCalls != before+4 {
				t.Fatalf("prefix %d changed selection/admission count: %v/%v", groups, err, wantErr)
			}
			copyElements += end
		}
		if copyElements != 64*dim || len(panel) != 8*dim || cap(panel) != len(panel) {
			t.Fatal("host copy or payload bound changed")
		}
		b.onKeyUpload = nil
		assertRetired := func() {
			if len(b.allocations) != 0 || len(b.keyOwned) != 0 || len(b.copyPending) != 0 || b.output != nil || b.keyLiveBytes != 0 {
				t.Fatal("allocation measurement retained transient score resources")
			}
			for _, held := range s.halW {
				if b.managed[held.Buf()] {
					t.Fatal("allocation measurement would discard a live Session handle")
				}
			}
		}
		run := func(prepared bool) {
			assertRetired()
			// Free retains managed history for double-free diagnostics. Clear
			// only fully retired history at this fixture's batch boundary so
			// sequential arms cannot cross different map-growth thresholds.
			// Ownership checks remain active throughout every scored query.
			clear(b.managed)
			b.calls = b.calls[:0]
			var panel []float32
			if prepared {
				panel = v41ColdIndexerKeyPanel(keys, dim)
			}
			for pos := 0; pos < seq; pos++ {
				groups := (pos + 1) / ratio
				var prefix []float32
				if panel != nil {
					end := groups * dim
					prefix = panel[:end:end]
				}
				if _, err := m.v41IndexRowsWithPreparedBinding(0, pos, q, hidden, keys[:groups], nil, binding, st.indexScoreHealth, nil, prefix, &borrow); err != nil {
					t.Fatal(err)
				}
			}
			assertRetired()
			if len(b.managed) != 4*(seq-ratio+1) {
				t.Fatalf("score recorder history=%d, want exactly four retired handles per nonempty query", len(b.managed))
			}
		}
		fresh := testing.AllocsPerRun(5, func() { run(false) })
		prepared := testing.AllocsPerRun(5, func() { run(true) })
		preparedReverse := testing.AllocsPerRun(5, func() { run(true) })
		freshReverse := testing.AllocsPerRun(5, func() { run(false) })
		if fresh-prepared != seq-ratio || freshReverse-preparedReverse != seq-ratio {
			t.Fatalf("native host allocations fresh/prepared=%g/%g reverse=%g/%g, want reduction %d in both orders", fresh, prepared, freshReverse, preparedReverse, seq-ratio)
		}
		if fresh != freshReverse || prepared != preparedReverse {
			t.Fatal("native allocation measurement depends on arm order")
		}
		t.Logf("native recorder allocations fresh=%g prepared=%g reverse_fresh=%g reverse_prepared=%g; retired history=%d; host copied elements fresh=%d prepared=%d; H2D unchanged", fresh, prepared, freshReverse, preparedReverse, len(b.managed), copyElements, len(panel))
	})

	t.Run("replacement-and-provenance-revoke-without-probing", func(t *testing.T) {
		m, s, b, scratch := fixture(t)
		st, binding := s.v41State(), scratch.indexerScoreBinding
		before := b.healthCalls
		refused := func(name string, mutate func() func()) {
			t.Helper()
			restore := mutate()
			if binding.nativeOwner(m, st, scratch) != nil {
				t.Fatalf("%s retained borrowing authority", name)
			}
			restore()
		}
		refused("same-function setter", func() func() {
			st.setIndexerScore(st.indexerScore())
			return func() { st.indexerScoreBinding = binding }
		})
		refused("ordinary wrapper", func() func() {
			original := binding.scoreFunc()
			st.setIndexerScore(func(l int, q, k, w []float32, h, d, n int) ([]float32, error) { return original(l, q, k, w, h, d, n) })
			return func() { st.indexerScoreBinding = binding }
		})
		refused("owner", func() func() { st.callbackOwner = &Session{}; return func() { st.callbackOwner = s } })
		refused("model", func() func() { s.M = &Model{}; return func() { s.M = m } })
		refused("backend", func() func() { s.Backend = newV41BorrowScoreBackend(); return func() { s.Backend = b } })
		refused("closed", func() func() { s.halClosed = true; return func() { s.halClosed = false } })
		refused("weights", func() func() { s.modelWeightsHeld = false; return func() { s.modelWeightsHeld = true } })
		refused("scope", func() func() {
			old := s.backendExecutionScope
			s.backendExecutionScope = nil
			return func() { s.backendExecutionScope = old }
		})
		refused("scope membership", func() func() {
			scope := s.backendExecutionScope
			old := scope.sessions
			scope.sessions = nil
			return func() { scope.sessions = old }
		})
		refused("stale scratch", func() func() {
			scratch.indexerScoreBinding = newV41IndexerScoreBinding(binding.scoreFunc())
			return func() { scratch.indexerScoreBinding = binding }
		})
		if b.healthCalls != before || binding.nativeOwner(m, st, scratch) != s {
			t.Fatal("provenance checks performed admission or changed the original binding")
		}
		plain := newV41ScoreTestBackend()
		other := v41DenseTestSession(t, m, plain)
		otherState := other.v41State()
		otherScratch := &v41ProjScratch{indexerScoreBinding: otherState.indexerScoreBinding}
		if otherState.indexerScoreBinding.nativeOwner(m, otherState, otherScratch) != nil {
			t.Fatal("no-owner backend gained borrowing authority")
		}
	})

	t.Run("admission-mutation-preserves-captured-provenance", func(t *testing.T) {
		m, s, b, scratch := fixture(t)
		other := newV41BorrowScoreBackend()
		before := b.healthCalls
		b.afterAdmission = func() { s.Backend = other }
		bound := s.v41BindIndexerScore()
		b.afterAdmission = nil
		s.v41State().bindCanonicalIndexerScore(bound)
		scratch.indexerScoreBinding = s.v41State().indexerScoreBinding
		if bound.backend != b || bound.model != m || bound.owner != s || b.healthCalls != before+1 || other.healthCalls != 0 ||
			scratch.indexerScoreBinding.nativeOwner(m, s.v41State(), scratch) != nil {
			t.Fatal("mutating admission forged provenance or caused a second probe")
		}
		s.Backend = b
	})

	for _, when := range []string{"projection", "health", "runtime-admission", "query-upload"} {
		for _, mutation := range []string{"callback", "model", "backend", "epoch"} {
			// Changing backend identity during the query upload already causes
			// the unchanged query-tensor check to fail before key use. Exercise
			// backend revocation at the other three boundaries instead.
			if when == "query-upload" && mutation == "backend" {
				continue
			}
			t.Run(when+"-"+mutation+"-revocation-keeps-frozen-input", func(t *testing.T) {
				m, s, b, scratch := fixture(t)
				st, binding := s.v41State(), scratch.indexerScoreBinding
				q, hidden, keys := operands(m)
				panel := v41ColdIndexerKeyPanel(keys, m.Cfg.IndexHeadDim)
				prefix := panel[: 2*m.Cfg.IndexHeadDim : 2*m.Cfg.IndexHeadDim]
				original := append([]float32(nil), prefix...)
				borrow := binding.prepareBorrow(m, st, scratch)
				other := newV41BorrowScoreBackend()
				other.owner = b.owner // Distinct backend identity, same recording domain.
				if compute.Backend(other) == compute.Backend(b) || other.ForwardExecutionOwner() != b.ForwardExecutionOwner() {
					t.Fatal("alternate backend must differ in identity but share the held owner")
				}
				epoch := s.cacheGeometryEpoch
				t.Cleanup(func() { s.M, s.Backend, s.cacheGeometryEpoch = m, b, epoch })
				newCalls, mutations, uploads := 0, 0, 0
				mutate := func() {
					if mutations != 0 {
						return
					}
					mutations++
					keys[0][0]++ // The old flat snapshot must survive this mutation.
					switch mutation {
					case "callback":
						st.setIndexerScore(func(int, []float32, []float32, []float32, int, int, int) ([]float32, error) {
							newCalls++
							return nil, errors.New("replacement was invoked")
						})
					case "model":
						s.M = &Model{}
					case "backend":
						s.Backend = other
					case "epoch":
						s.cacheGeometryEpoch++
					}
				}
				observe := func(flat []float32) {
					uploads++
					if len(flat) != len(prefix) || cap(flat) != len(prefix) || &flat[0] == &prefix[0] || !reflect.DeepEqual(flat, original) {
						t.Fatal("revocation did not privatize the exact original snapshot")
					}
				}
				b.onKeyUpload, other.onKeyUpload = observe, observe
				var project v41DenseProjectionFunc
				health := st.indexScoreHealth
				before, dispatch := b.healthCalls, b.dispatchCount
				switch when {
				case "projection":
					project = func(int, string, []float32, int, int, int) ([]float32, v41DenseProjectionOutcome, error) {
						mutate()
						return nil, v41ProjectionDeclined, nil
					}
				case "health":
					originalHealth := health
					health = func(layer int) error { err := originalHealth(layer); mutate(); return err }
				case "runtime-admission":
					b.afterAdmission = func() {
						if b.healthCalls == before+2 { // Existing core probe, after outer health.
							mutate()
						}
					}
				case "query-upload":
					b.afterQueryUpload = mutate
				}
				_, err := m.v41IndexRowsWithPreparedBinding(0, 3, q, hidden, keys[:2], project, binding, health, nil, prefix, &borrow)
				// A changed backend preserves the ordinary core's original captured
				// operation. Its normal tensor-identity check rejects the new uploads.
				if (mutation == "backend") != (err != nil) || !borrow.revoked || borrow.valid() || newCalls != 0 || mutations != 1 || uploads != 1 ||
					b.healthCalls != before+2 || b.dispatchCount != dispatch+1 || other.healthCalls != 0 || other.dispatchCount != 0 || !reflect.DeepEqual(prefix, original) {
					t.Fatalf("revocation changed call/refusal semantics: err=%v probes=%d dispatch=%d uploads=%d", err, b.healthCalls-before, b.dispatchCount-dispatch, uploads)
				}
				s.M, s.Backend, s.cacheGeometryEpoch = m, b, epoch
				b.afterAdmission, b.afterQueryUpload = nil, nil
				if mutation != "backend" { // Backend refusal retired this Session.
					st.indexerScoreBinding = binding
					if borrow.valid() {
						t.Fatal("restoring source state resurrected a revoked loop borrow")
					}
					b.onKeyUpload = func(flat []float32) {
						if &flat[0] == &prefix[0] || flat[0] != keys[0][0] {
							t.Fatal("next query reused the revoked panel instead of fresh source rows")
						}
					}
					if _, err := m.v41IndexRowsWithPreparedBinding(0, 4, q, hidden, keys[:2], nil, binding, st.indexScoreHealth, nil, prefix, &borrow); err != nil {
						t.Fatal(err)
					}
				}
			})
		}
	}

	t.Run("carrier-copy-allocation-and-storage-bound", func(t *testing.T) {
		m, s, _, scratch := fixture(t)
		st, binding := s.v41State(), scratch.indexerScoreBinding
		_, _, keys := operands(m)
		panel := v41ColdIndexerKeyPanel(keys, m.Cfg.IndexHeadDim)
		prefix := panel[: 2*m.Cfg.IndexHeadDim : 2*m.Cfg.IndexHeadDim]
		borrow := binding.prepareBorrow(m, st, scratch)
		input := v41IndexerScoreKeys{borrow: &borrow, frozen: prefix}
		var retained []float32
		stable := testing.AllocsPerRun(20, func() { retained = input.beforeUpload(prefix) })
		if stable != 0 || &retained[0] != &prefix[0] {
			t.Fatal("stable carrier allocated or copied its bounded prefix")
		}
		st.setIndexerScore(binding.scoreFunc())
		late := testing.AllocsPerRun(20, func() {
			borrow.revoked = false
			retained = input.beforeUpload(prefix)
		})
		if late != 1 || !borrow.revoked || len(retained) != len(prefix) || cap(retained) != len(prefix) ||
			&retained[0] == &prefix[0] || !reflect.DeepEqual(retained, prefix) {
			t.Fatalf("late revocation must allocate one exact frozen prefix, allocations=%g", late)
		}
		// The sole new owned allocation is exactly the visible prefix. Its
		// capacity plus the one original panel is the measured storage bound.
		if cap(retained)+cap(panel) > 2*cap(panel) {
			t.Fatal("observed prefix storage exceeded the exceptional two-panel bound")
		}
		early := testing.AllocsPerRun(20, func() {
			_, private, zero := borrow.resolve(prefix)
			retained = zero.beforeUpload(private)
			if &retained[0] != &private[0] {
				t.Fatal("already-private fallback was copied a second time")
			}
		})
		if early != 1 {
			t.Fatalf("early fallback must allocate exactly once, got %g", early)
		}
		t.Logf("key carrier allocations stable=%g late-revoked=%g already-private=%g; panel=%d prefix=%d float32 slots", stable, late, early, cap(panel), cap(retained))
	})

	t.Run("actual-cold-step-suffix-state-parity", func(t *testing.T) {
		m := v41IndexerTestFixture(t)
		leftBackend, rightBackend := newV41BorrowScoreBackend(), newV41BorrowScoreBackend()
		left, right := v41DenseTestSession(t, m, leftBackend), v41DenseTestSession(t, m, rightBackend)
		v41ScoreOnly(left)
		v41ScoreOnly(right)
		original := right.v41State().indexerScore()
		right.v41State().setIndexerScore(func(l int, q, k, w []float32, h, d, n int) ([]float32, error) { return original(l, q, k, w, h, d, n) })
		var leftInputs, rightInputs [][]float32
		leftBackend.onKeyUpload = func(keys []float32) { leftInputs = append(leftInputs, keys) }
		rightBackend.onKeyUpload = func(keys []float32) { rightInputs = append(rightInputs, keys) }
		prefix, next := v41IncrementalPrefix(m.Cfg, 16)
		check := func(got, want []float32) {
			v41ScoreTestBits(t, got, want)
			if !reflect.DeepEqual(captureV41ForwardSnapshot(left.v41Forward), captureV41ForwardSnapshot(right.v41Forward)) {
				t.Fatal("binding changed retained KV/index/top-k state")
			}
		}
		check(left.Prefill(prefix), right.Prefill(prefix))
		if len(leftInputs) < 2 || len(leftInputs) != len(rightInputs) || &leftInputs[0][0] != &leftInputs[1][0] || &rightInputs[0][0] == &rightInputs[1][0] ||
			!reflect.DeepEqual(leftBackend.calls, rightBackend.calls) || leftBackend.keyUploadBytes != rightBackend.keyUploadBytes {
			t.Fatal("cold composer did not reuse native host panel or changed H2D operands")
		}
		leftBackend.onKeyUpload, rightBackend.onKeyUpload = nil, nil
		check(left.Step(next), right.Step(next))
		check(left.Prefill([]int{2, 5}), right.Prefill([]int{2, 5}))
	})

	_ = compute.F32 // Keep the recorder's capability dependency explicit.
}
