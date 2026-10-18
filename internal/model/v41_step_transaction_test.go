package model

import (
	"errors"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"runtime"
	"testing"
	"time"
)

// Estimate is unmeasured; this is a small synthetic correctness witness, not a
// throughput benchmark or a receipt for cross-session shared projection work.
// fak-test:runtime fast est=500ms lane=default
func TestV41StepTransactionInterleavedSessions(t *testing.T) {
	m := v41CompressorTestFixtureIndex(t, 128)
	var sessions, serial [2]*Session
	var states, refs [2]*v41ForwardState
	var scratch [2]v41ProjScratch
	var calls [2]int
	active := -1
	for lane, prefix := range [][]int{{1, 2, 3}, {1, 2, 3, 4}} {
		source := m.NewSession()
		t.Cleanup(source.Close)
		if len(source.Prefill(prefix)) == 0 {
			t.Fatal("prefill returned no logits")
		}
		// A restored session must rebind its own callbacks, never keep those
		// of the prefix source. Make accidental source invocation fatal.
		source.v41State().setDenseProjection(func(int, string, []float32, int, int, int) ([]float32, v41DenseProjectionOutcome, error) {
			t.Fatal("restored transaction called prefix source callback")
			return nil, v41ProjectionDeclined, nil
		})
		snapshot := captureV41ForwardSnapshot(source.v41State())
		sessions[lane], serial[lane] = m.NewSession(), m.NewSession()
		t.Cleanup(sessions[lane].Close)
		t.Cleanup(serial[lane].Close)
		sessions[lane].v41Forward = snapshot.clone().restore()
		serial[lane].v41Forward = snapshot.clone().restore()
		states[lane], refs[lane] = sessions[lane].v41State(), serial[lane].v41State()
		if states[lane].callbackOwner != sessions[lane] || refs[lane].callbackOwner != serial[lane] {
			t.Fatal("restored callbacks do not belong to destination sessions")
		}
		states[lane].setDenseProjection(func(int, string, []float32, int, int, int) ([]float32, v41DenseProjectionOutcome, error) {
			if active != lane {
				t.Fatalf("lane %d callback ran while lane %d owned the step", lane, active)
			}
			calls[lane]++
			return nil, v41ProjectionDeclined, nil
		})
		// Match the callback's arithmetic path in the serial reference.
		refs[lane].setDenseProjection(nil)
	}
	runPair := func(fail bool) {
		t.Helper()
		var tx [2]*v41StepTransaction
		var want [2][]float32
		var before [2]*v41ForwardSnapshot
		var registry [2]*V41AttentionState
		var pointers [2][]*V41AttentionState
		sentinel := errors.New("lane one selected head failure")
		for lane := range states {
			// Populate the next overwritten slot so rollback must restore actual
			// bytes even with this bounded, short-prefix fixture.
			for _, st := range []*v41ForwardState{states[lane], refs[lane]} {
				for _, layer := range st.layers {
					slot := layer.nextWindowPos % layer.windowSize
					layer.window[slot] = append([]float32(nil), layer.window[0]...)
				}
			}
			if fail && lane == 1 {
				states[lane].setFinalNorm(func([]float32) ([]float32, error) { return nil, sentinel })
			}
			before[lane] = captureV41ForwardSnapshot(states[lane])
			registry[lane] = states[lane].attn
			pointers[lane] = append([]*V41AttentionState(nil), states[lane].layers...)
			if !fail || lane == 0 {
				var err error
				want[lane], _, err = m.forwardV41Step(6+lane, refs[lane], nil)
				if err != nil {
					t.Fatal(err)
				}
			}
			active = lane
			var err error
			tx[lane], err = m.beginV41Step(6+lane, states[lane], &scratch[lane])
			if err != nil {
				t.Fatal(err)
			}
			defer tx[lane].abort()
		}
		for layer := 0; layer < m.Cfg.NumLayers; layer++ {
			for lane := range tx {
				active = lane
				if err := tx[lane].advanceLayer(); err != nil {
					t.Fatal(err)
				}
			}
		}
		if fail && (registry[1].kvPublishedEnd[0] != before[1].attn.kvPublishedEnd[0]+1 || registry[1].indexPublishedEnd[0] != before[1].attn.indexPublishedEnd[0]+1) {
			t.Fatal("failure witness did not mutate both source publication ends")
		}
		// Neither history may commit before its own head succeeds.
		for lane := range states {
			if !reflect.DeepEqual(states[lane].history, before[lane].history) {
				t.Fatal("premature commit")
			}
		}
		active = 0
		got, err := tx[0].finalize()
		if err != nil || !reflect.DeepEqual(got, want[0]) || !tx[0].stats.Committed {
			t.Fatalf("lane zero logits/commit: %v", err)
		}
		committed := captureV41ForwardSnapshot(states[0])
		active = 1
		if fail {
			panicked := recoverError(func() { _, err = tx[1].finalize() })
			if !errors.Is(panicked, sentinel) {
				t.Fatalf("head failure cause lost: returned=%v panic=%v", err, panicked)
			}
			if tx[1].stats.Committed || tx[1].stats.LayersRolledBack != m.Cfg.NumLayers || !tx[1].closed {
				t.Fatalf("failed lane stats: %+v", tx[1].stats)
			}
			if states[1].attn != registry[1] || !reflect.DeepEqual(captureV41ForwardSnapshot(states[1]), before[1]) {
				t.Fatal("failed lane ring/publication/history rollback differs")
			}
			for layer, pointer := range pointers[1] {
				if states[1].layers[layer] != pointer {
					t.Fatal("rollback replaced retained layer identity")
				}
			}
			if !reflect.DeepEqual(captureV41ForwardSnapshot(states[0]), committed) {
				t.Fatal("lane one rollback changed committed lane zero")
			}
		} else {
			got, err = tx[1].finalize()
			if err != nil || !reflect.DeepEqual(got, want[1]) || !tx[1].stats.Committed {
				t.Fatalf("lane one logits/commit: %v", err)
			}
		}
		for lane := range states {
			if !fail || lane == 0 {
				if !reflect.DeepEqual(captureV41ForwardSnapshot(states[lane]), captureV41ForwardSnapshot(refs[lane])) {
					t.Fatalf("lane %d differs from serial continuation", lane)
				}
			}
			if scratch[lane].denseProjection() != nil || scratch[lane].mhcCarry != nil {
				t.Fatal("transaction retained session scratch callbacks")
			}
			tx[lane].abort() // no duplicate rollback and no committed-token undo
		}
	}
	runPair(false)
	runPair(true)
	if calls[0] == 0 || calls[1] == 0 {
		t.Fatal("session callback ownership witness was vacuous")
	}
}

// fak-test:runtime medium est=5s lane=default
func TestV41StepRoundRetainedPrefix(t *testing.T) {
	withV41ActivationTraceEnv(t, "")
	for _, fixture := range []struct {
		name   string
		prefix int
		ratio  int
		window int
		steps  int
	}{
		{"plain", 7, 0, -1, 4},
		{"compressed-2", 3, 2, -1, 4},
		{"compressed-2-eight-steps", 7, 2, -1, 8},
		{"wrapped-window", 127, 0, 8, 4},
	} {
		for _, keep := range []int{0, 1, fixture.steps / 2, fixture.steps} {
			t.Run(fixture.name+"/keep-"+itoa(keep), func(t *testing.T) {
				var m *Model
				if fixture.ratio != 0 {
					m = v41CompressorTestFixtureIndex(t, 128)
					m.Cfg.DeepSeekV41.CompressRatios = []int{fixture.ratio, fixture.ratio}
				} else {
					m = v41IncrementalPlainModel(t, 2)
				}
				t.Cleanup(func() { _ = m.CloseWeights() })
				m.Cfg.Window = make([]int, m.Cfg.NumLayers)
				for layer := range m.Cfg.Window {
					m.Cfg.Window[layer] = fixture.window
					if m.Cfg.windowForLayer(layer) != fixture.window {
						t.Fatal("fixture did not configure every layer window")
					}
				}
				prefix, next := v41IncrementalPrefix(m.Cfg, fixture.prefix)
				st := v41IncrementalState(t, m, prefix)
				ref := v41IncrementalState(t, m, prefix)
				if fixture.window > 0 {
					for _, branch := range []*v41ForwardState{st, ref} {
						for layer, state := range branch.layers {
							if state.windowSize != v41WindowSize || fixture.prefix >= state.windowSize {
								t.Fatal("wrap fixture did not seed a pre-wrap ring")
							}
							for pos := fixture.prefix; pos < fixture.prefix+fixture.steps; pos++ {
								for _, key := range v41PlainWindowKeys(pos, m.Cfg.windowForLayer(layer)) {
									if key == 0 {
										t.Fatal("marker slot is visible to configured attention")
									}
								}
							}
							// An invisible old slot must still be restored after a rejected wrap.
							state.window[0][0] = float32(10000 + layer)
						}
					}
				}
				st.history = append(make([]int, 0, len(prefix)+fixture.steps+1), prefix...)
				ref.history = append(make([]int, 0, len(prefix)+fixture.steps+1), prefix...)
				pointers := append([]*V41AttentionState(nil), st.layers...)
				history := &st.history[0]
				coverage := []*NativeForwardCoverage{st.forwardCoverage}
				if st.forwardCoverage == nil || st.forwardCoverage.Validate() != nil {
					t.Fatal("seed did not produce complete coverage")
				}
				coverageValues := []NativeForwardCoverage{*st.forwardCoverage}
				registry := []*V41AttentionState{st.attn}
				storage := [][]uintptr{v41RoundStorageIdentity(st)}
				before := captureV41ForwardSnapshot(st)
				wrappedRows := make([][]float32, len(st.layers))
				wrappedPointers := make([]uintptr, len(st.layers))
				for layer, state := range st.layers {
					wrappedRows[layer] = append([]float32(nil), state.window[0]...)
					wrappedPointers[layer] = reflect.ValueOf(state.window[0]).Pointer()
				}
				round, err := m.beginV41StepRound(st, fixture.steps)
				if err != nil {
					t.Fatal(err)
				}
				defer round.abort()
				if st.forwardCoverage != coverage[0] {
					t.Fatal("opening a round cleared prior coverage")
				}
				for i := 0; i < fixture.steps; i++ {
					id := (next + i*3) % m.Cfg.VocabSize
					tx, err := round.beginStep(id, &v41ProjScratch{})
					if err != nil {
						t.Fatal(err)
					}
					for layer := 0; layer < m.Cfg.NumLayers; layer++ {
						if err := tx.advanceLayer(); err != nil {
							t.Fatal(err)
						}
					}
					got, err := tx.finalize()
					if err != nil || !tx.stats.Committed {
						t.Fatalf("step %d did not commit: %v", i, err)
					}
					committed := captureV41ForwardSnapshot(st)
					tx.abort()
					if !reflect.DeepEqual(captureV41ForwardSnapshot(st), committed) {
						t.Fatal("ordinary transaction abort undid a committed round entry")
					}
					coverage = append(coverage, st.forwardCoverage)
					if st.forwardCoverage == nil || st.forwardCoverage.Validate() != nil {
						t.Fatal("committed step did not produce complete coverage")
					}
					coverageValues = append(coverageValues, *st.forwardCoverage)
					registry = append(registry, st.attn)
					storage = append(storage, v41RoundStorageIdentity(st))
					if i < keep {
						want, _, err := m.forwardV41Step(id, ref, nil)
						if err != nil {
							t.Fatal(err)
						}
						if !reflect.DeepEqual(got, want) {
							t.Fatal("round logits differ from ordinary serial step")
						}
					}
				}
				if fixture.window > 0 {
					for layer, state := range st.layers {
						if state.nextWindowPos != fixture.prefix+fixture.steps ||
							state.nextWindowPos <= v41WindowSize || reflect.DeepEqual(state.window[0], wrappedRows[layer]) ||
							reflect.ValueOf(state.window[0]).Pointer() != wrappedPointers[layer] {
							t.Fatal("wrap fixture did not overwrite a real retained ring slot")
						}
					}
				}
				if len(round.entries) != fixture.steps || cap(round.entries) > fixture.steps || round.pending != nil {
					t.Fatal("journal did not remain bounded by consumed steps")
				}
				entries := append([]*v41StepRoundEntry(nil), round.entries...)
				var undoOrder []int
				for i, entry := range entries {
					undo := entry.undo
					if undo == nil {
						t.Fatal("committed entry lost its retained undo before finish")
					}
					entry.undo = func() {
						undoOrder = append(undoOrder, i)
						undo()
					}
				}
				if err := round.finish(keep); err != nil {
					t.Fatal(err)
				}
				for i, ordinal := range undoOrder {
					if ordinal != fixture.steps-1-i {
						t.Fatal("rejected suffix was not undone in reverse order")
					}
				}
				if len(undoOrder) != fixture.steps-keep {
					t.Fatal("finish undid a kept token or skipped a rejected token")
				}
				for _, entry := range entries {
					if !reflect.DeepEqual(*entry, v41StepRoundEntry{}) {
						t.Fatal("finished journal retained kept or rejected entry references")
					}
				}
				if len(round.entries) != 0 || round.model != nil || round.state != nil || round.layers != nil || round.pending != nil {
					t.Fatal("finished round retained model/state/journal references")
				}
				if !reflect.DeepEqual(v41RoundStorageIdentity(st), storage[keep]) {
					t.Fatal("suffix rewind replaced retained row or partial-group storage")
				}
				for i, pointer := range coverage {
					if *pointer != coverageValues[i] {
						t.Fatal("round mutated previously published coverage contents")
					}
				}
				if st.forwardCoverage != coverage[keep] || st.attn != registry[keep] ||
					&st.history[0] != history || !reflect.DeepEqual(captureV41ForwardSnapshot(st), captureV41ForwardSnapshot(ref)) {
					t.Fatal("retained prefix differs from ordinary serial state or replaced prior coverage/history/registry")
				}
				if keep == 0 && !reflect.DeepEqual(captureV41ForwardSnapshot(st), before) {
					t.Fatal("keep-zero did not restore the complete original state")
				}
				for i, pointer := range pointers {
					if st.layers[i] != pointer {
						t.Fatal("suffix rewind replaced a retained layer")
					}
				}
				if err := round.abort(); err != nil {
					t.Fatal(err)
				}
				got, _, err := m.forwardV41Step((next+13)%m.Cfg.VocabSize, st, nil)
				if err != nil {
					t.Fatal(err)
				}
				want, _, err := m.forwardV41Step((next+13)%m.Cfg.VocabSize, ref, nil)
				if err != nil || !reflect.DeepEqual(got, want) ||
					!reflect.DeepEqual(captureV41ForwardSnapshot(st), captureV41ForwardSnapshot(ref)) {
					t.Fatal("continued target logits or state differ after retaining prefix")
				}
			})
		}
	}
}

func v41RoundStorageIdentity(st *v41ForwardState) []uintptr {
	var out []uintptr
	for _, layer := range st.layers {
		out = append(out, reflect.ValueOf(layer.window).Pointer(),
			reflect.ValueOf(layer.partialInputs).Pointer(), reflect.ValueOf(layer.partialPositions).Pointer(),
			uintptr(cap(layer.partialInputs)), uintptr(cap(layer.partialPositions)))
		for _, row := range layer.window {
			out = append(out, reflect.ValueOf(row).Pointer())
		}
		for _, row := range layer.partialInputs {
			out = append(out, reflect.ValueOf(row).Pointer())
		}
	}
	return out
}

// fak-test:runtime medium est=3s lane=default
func TestV41StepRoundLifecycle(t *testing.T) {
	withV41ActivationTraceEnv(t, "")
	newState := func(t *testing.T) (*Model, *v41ForwardState) {
		t.Helper()
		m := v41IncrementalPlainModel(t, 2)
		t.Cleanup(func() { _ = m.CloseWeights() })
		return m, v41IncrementalState(t, m, []int{1, 2, 3})
	}
	t.Run("admission-and-stale-handles", func(t *testing.T) {
		m, st := newState(t)
		before, coverage := captureV41ForwardSnapshot(st), st.forwardCoverage
		for _, capacity := range []int{-1, 0, 33, 1 << 30} {
			if r, err := m.beginV41StepRound(st, capacity); r != nil || !errors.Is(err, errV41StepRoundState) {
				t.Fatal("invalid capacity was admitted")
			}
		}
		for _, state := range []*v41ForwardState{nil, {}, {history: []int{1}}} {
			if r, err := m.beginV41StepRound(state, 1); r != nil || !errors.Is(err, errV41StepRoundState) {
				t.Fatal("unseeded state was admitted")
			}
		}
		old, err := m.beginV41StepRound(st, 32)
		if err != nil || cap(old.entries) > 32 {
			t.Fatal("maximum bounded capacity was not admitted")
		}
		if _, err := m.beginV41StepRound(st, 1); !errors.Is(err, errV41StepRoundState) {
			t.Fatal("nested round was admitted")
		}
		if err := old.abort(); err != nil {
			t.Fatal(err)
		}
		current, err := m.beginV41StepRound(st, 1)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := old.beginStep(4, nil); !errors.Is(err, errV41StepRoundState) {
			t.Fatal("stale round began a new step")
		}
		if err := old.finish(0); !errors.Is(err, errV41StepRoundState) {
			t.Fatal("finished round accepted a second finish")
		}
		if err := old.abort(); err != nil || st.activeStepRound != current {
			t.Fatal("idempotent stale abort changed a newer owner")
		}
		if err := current.abort(); err != nil || st.activeStepRound != nil ||
			st.forwardCoverage != coverage || !reflect.DeepEqual(captureV41ForwardSnapshot(st), before) {
			t.Fatal("empty round changed the seeded state")
		}
	})
	t.Run("pending-fences-and-capacity", func(t *testing.T) {
		m, st := newState(t)
		before, coverage := captureV41ForwardSnapshot(st), st.forwardCoverage
		r, err := m.beginV41StepRound(st, 1)
		if err != nil {
			t.Fatal(err)
		}
		tx, err := r.beginStep(4, nil)
		if err != nil {
			t.Fatal(err)
		}
		pending := captureV41ForwardSnapshot(st)
		pendingCoverage := st.forwardCoverage
		if _, err := r.beginStep(5, nil); !errors.Is(err, errV41StepRoundState) {
			t.Fatal("a second pending transaction was admitted")
		}
		if err := r.finish(0); !errors.Is(err, errV41StepRoundState) {
			t.Fatal("finish consumed a pending transaction")
		}
		if _, err := m.beginV41Step(5, st, nil); !errors.Is(err, errV41StepRoundState) {
			t.Fatal("ordinary transaction bypassed the active round")
		}
		if _, err := m.beginV41StepAdmitted(5, st, nil); !errors.Is(err, errV41StepRoundState) {
			t.Fatal("cohort transaction bypassed the active round")
		}
		if _, _, err := m.forwardV41Step(5, st, nil); !errors.Is(err, errV41StepRoundState) {
			t.Fatal("ordinary incremental forward bypassed the active round")
		}
		if _, err := m.forwardV41([]int{1, 2, 3, 5}, st); !errors.Is(err, errV41StepRoundState) {
			t.Fatal("stateful full forward bypassed the active round")
		}
		if st.forwardCoverage != pendingCoverage || !reflect.DeepEqual(captureV41ForwardSnapshot(st), pending) {
			t.Fatal("a refused caller mutated pending round state")
		}
		if err := tx.advanceLayer(); err != nil {
			t.Fatal(err)
		}
		if reflect.DeepEqual(captureV41ForwardSnapshot(st), before) {
			t.Fatal("pending abort witness did not execute a real layer")
		}
		if err := r.abort(); err != nil || !tx.closed || st.forwardCoverage != coverage ||
			!reflect.DeepEqual(captureV41ForwardSnapshot(st), before) {
			t.Fatal("pending transaction did not roll back completely")
		}
		r, err = m.beginV41StepRound(st, 1)
		if err != nil {
			t.Fatal(err)
		}
		tx = v41RoundPreparedTestStep(t, r, 4)
		if _, err := tx.finalize(); err != nil {
			t.Fatal(err)
		}
		committed := captureV41ForwardSnapshot(st)
		if _, err := r.beginStep(5, nil); !errors.Is(err, errV41StepRoundState) {
			t.Fatal("journal exceeded its step capacity")
		}
		for _, keep := range []int{-1, 2} {
			if err := r.finish(keep); !errors.Is(err, errV41StepRoundState) {
				t.Fatal("invalid retained prefix was accepted")
			}
		}
		if !reflect.DeepEqual(captureV41ForwardSnapshot(st), committed) || st.activeStepRound != r {
			t.Fatal("invalid finish changed a committed prefix")
		}
		if err := r.finish(1); err != nil {
			t.Fatal(err)
		}
	})
	t.Run("active-identity-drift-refuses-before-write", func(t *testing.T) {
		m, st := newState(t)
		r, err := m.beginV41StepRound(st, 1)
		if err != nil {
			t.Fatal(err)
		}
		coverage := st.forwardCoverage
		layer := st.layers[0]
		st.layers[0] = st.layers[1]
		before := captureV41ForwardSnapshot(st)
		if _, err := r.beginStep(4, nil); !errors.Is(err, errV41StepRoundState) ||
			st.forwardCoverage != coverage || !reflect.DeepEqual(captureV41ForwardSnapshot(st), before) {
			t.Fatal("stale layer binding was used or mutated")
		}
		st.layers[0] = layer
		layer.nextWindowPos++
		before = captureV41ForwardSnapshot(st)
		if _, err := r.beginStep(4, nil); !errors.Is(err, errV41StepRoundState) ||
			st.forwardCoverage != coverage || !reflect.DeepEqual(captureV41ForwardSnapshot(st), before) {
			t.Fatal("stale cursor was used or mutated")
		}
		layer.nextWindowPos--
		if err := r.abort(); err != nil {
			t.Fatal(err)
		}
	})
	t.Run("begin-failure-restores-prior-coverage", func(t *testing.T) {
		m, st := newState(t)
		before, coverage := captureV41ForwardSnapshot(st), st.forwardCoverage
		r, err := m.beginV41StepRound(st, 1)
		if err != nil {
			t.Fatal(err)
		}
		// Fault the real admission path after the round has been admitted.
		schedule := m.Cfg.DeepSeekV41.CompressRatios
		m.Cfg.DeepSeekV41.CompressRatios = nil
		tx, err := r.beginStep(4, nil)
		m.Cfg.DeepSeekV41.CompressRatios = schedule
		if !errors.Is(err, ErrV41ForwardStage) || tx == nil || !tx.closed || st.forwardCoverage != nil {
			t.Fatal("begin failure did not cross the real coverage-clearing admission path")
		}
		if len(r.entries) != 1 || r.entries[0].coverage != coverage || r.entries[0].historyLen != len(before.history) {
			t.Fatal("begin failure was not registered before mutation")
		}
		if err := r.abort(); err != nil || st.forwardCoverage != coverage ||
			!reflect.DeepEqual(captureV41ForwardSnapshot(st), before) || st.activeStepRound != nil {
			t.Fatal("begin failure lost the preceding completed coverage or state")
		}
	})
	t.Run("trace-refusal-publishes-nothing", func(t *testing.T) {
		m, st := newState(t)
		path := filepath.Join(t.TempDir(), "last-ordinary-step.json")
		withV41ActivationTraceEnv(t, path)
		if _, _, err := m.forwardV41Step(4, st, nil); err != nil {
			t.Fatal(err)
		}
		artifact, err := os.ReadFile(path)
		if err != nil || len(artifact) == 0 || !st.activationCapture.Published {
			t.Fatal("ordinary step did not create the trace witness")
		}
		receipt, before, coverage := st.activationCapture, captureV41ForwardSnapshot(st), st.forwardCoverage
		if _, err := m.beginV41StepRound(st, 1); !errors.Is(err, errV41StepRoundState) || st.activeStepRound != nil {
			t.Fatal("trace-enabled round was installed")
		}
		withV41ActivationTraceEnv(t, "")
		r, err := m.beginV41StepRound(st, 1)
		if err != nil {
			t.Fatal(err)
		}
		withV41ActivationTraceEnv(t, path)
		if _, err := r.beginStep(5, nil); !errors.Is(err, errV41StepRoundState) {
			t.Fatal("trace enabled after round begin was admitted")
		}
		withV41ActivationTraceEnv(t, "")
		trace := &v41ActivationTraceState{}
		scratch := &v41ProjScratch{activationTrace: trace}
		if _, err := r.beginStep(5, scratch); !errors.Is(err, errV41StepRoundState) ||
			scratch.activationTrace != trace || len(r.entries) != 0 {
			t.Fatal("caller-supplied trace was admitted or mutated")
		}
		after, err := os.ReadFile(path)
		if err != nil || !reflect.DeepEqual(after, artifact) || !reflect.DeepEqual(st.activationCapture, receipt) ||
			st.forwardCoverage != coverage || !reflect.DeepEqual(captureV41ForwardSnapshot(st), before) {
			t.Fatal("refused round changed an existing trace artifact, receipt or state")
		}
		if err := r.abort(); err != nil {
			t.Fatal(err)
		}
	})
	t.Run("committed-cleanup-panic-keeps-undo-and-poisons", func(t *testing.T) {
		m, st := newState(t)
		before, coverage := captureV41ForwardSnapshot(st), st.forwardCoverage
		r, err := m.beginV41StepRound(st, 2)
		if err != nil {
			t.Fatal(err)
		}
		tx := v41RoundPreparedTestStep(t, r, 4)
		if _, err := tx.finalize(); err != nil {
			t.Fatal(err)
		}
		tx = v41RoundPreparedTestStep(t, r, 5)
		primary := errors.New("cleanup failed after actual release")
		cleanup, calls := tx.cleanup, 0
		tx.cleanup = func() {
			calls++
			cleanup()
			panic(primary)
		}
		got := recoverError(func() { _, _ = tx.finalize() })
		if !errors.Is(got, primary) || !tx.stats.Committed || !tx.closed || calls != 1 ||
			len(r.entries) != 2 || !r.entries[1].committed || r.entries[1].undo == nil {
			t.Fatal("committed cleanup failure lost the primary or preregistered undo")
		}
		entries := append([]*v41StepRoundEntry(nil), r.entries...)
		if err := r.finish(1); !errors.Is(err, errV41StepRoundState) {
			t.Fatal("uncertain cleanup retained a positive speculative prefix")
		}
		if err := r.abort(); !errors.Is(err, errV41StepRoundState) || calls != 1 ||
			st.forwardCoverage != coverage || !reflect.DeepEqual(captureV41ForwardSnapshot(st), before) {
			t.Fatal("cleanup failure did not restore host state without repeating release")
		}
		v41RoundAssertPoisoned(t, m, st, r, entries)
	})
	for _, activePrimary := range []bool{false, true} {
		t.Run("undo-panic/primary-"+itoa(boolToIntV41Expert(activePrimary)), func(t *testing.T) {
			m, st := newState(t)
			before, coverage := captureV41ForwardSnapshot(st), st.forwardCoverage
			r, err := m.beginV41StepRound(st, 2)
			if err != nil {
				t.Fatal(err)
			}
			for _, id := range []int{4, 5} {
				tx := v41RoundPreparedTestStep(t, r, id)
				if _, err := tx.finalize(); err != nil {
					t.Fatal(err)
				}
			}
			entries := append([]*v41StepRoundEntry(nil), r.entries...)
			var order []int
			undoFault, primary := errors.New("retained undo failed"), errors.New("selected caller failure")
			for i, entry := range entries {
				undo := entry.undo
				entry.undo = func() {
					order = append(order, i)
					undo()
					if i == 1 {
						panic(undoFault)
					}
				}
			}
			got := recoverError(func() {
				if activePrimary {
					defer r.abort()
					panic(primary)
				}
				_ = r.abort()
			})
			want := undoFault
			if activePrimary {
				want = primary
			}
			if !errors.Is(got, want) || !reflect.DeepEqual(order, []int{1, 0}) ||
				st.forwardCoverage != coverage || !reflect.DeepEqual(captureV41ForwardSnapshot(st), before) {
				t.Fatal("undo failure replaced the primary, skipped remaining undo or lost host restoration")
			}
			v41RoundAssertPoisoned(t, m, st, r, entries)
		})
	}
}

func v41RoundPreparedTestStep(t *testing.T, r *v41StepRound, id int) *v41StepTransaction {
	t.Helper()
	tx, err := r.beginStep(id, &v41ProjScratch{})
	if err != nil {
		t.Fatal(err)
	}
	for layer := 0; layer < r.model.Cfg.NumLayers; layer++ {
		if err := tx.advanceLayer(); err != nil {
			t.Fatal(err)
		}
	}
	return tx
}

func v41RoundAssertPoisoned(t *testing.T, m *Model, st *v41ForwardState, r *v41StepRound, entries []*v41StepRoundEntry) {
	t.Helper()
	if !r.closed || !r.poisoned || st.activeStepRound != r || r.model != nil || r.state != nil ||
		r.layers != nil || r.entries != nil || r.pending != nil {
		t.Fatal("uncertain rollback lost its tombstone or retained journal resources")
	}
	for _, entry := range entries {
		if !reflect.DeepEqual(*entry, v41StepRoundEntry{}) {
			t.Fatal("failed journal retained an entry reference")
		}
	}
	if _, err := m.beginV41StepRound(st, 1); !errors.Is(err, errV41StepRoundState) {
		t.Fatal("uncertain state reopened a new round")
	}
	if _, _, err := m.forwardV41Step(6, st, nil); !errors.Is(err, errV41StepRoundState) {
		t.Fatal("uncertain state reopened ordinary incremental execution")
	}
}

func v41RoundModeStates(t *testing.T) (*Model, *v41ForwardState, *v41ForwardState) {
	t.Helper()
	m := v41CompressorTestFixtureIndex(t, 128)
	t.Cleanup(func() { _ = m.CloseWeights() })
	m.Cfg.DeepSeekV41.CompressRatios = []int{2, 2}
	m.Cfg.Window = make([]int, m.Cfg.NumLayers)
	for i := range m.Cfg.Window {
		m.Cfg.Window[i] = -1
	}
	prefix := []int{1, 2, 3}
	st, ref := v41IncrementalState(t, m, prefix), v41IncrementalState(t, m, prefix)
	for _, state := range []*v41ForwardState{st, ref} {
		state.history = append(make([]int, 0, len(prefix)+3), prefix...)
		if state.forwardCoverage == nil || state.forwardCoverage.Validate() != nil || state.attn == nil {
			t.Fatal("mode fixture must seed valid coverage and a retained role registry")
		}
	}
	m.v41SetExpertFaultPhase(V41PhaseDecode)
	t.Cleanup(func() { m.v41SetExpertFaultPhase(V41PhaseUnknown) })
	return m, st, ref
}

// fak-test:justify why=regression when=changed:internal/model/**
// fak-test:runtime medium est=5s lane=default
func TestV41StepRoundNoLogitsRetention(t *testing.T) {
	withV41ActivationTraceEnv(t, "")
	for _, seedNil := range []bool{false, true} {
		for _, plan := range []struct {
			name   string
			logits [2]bool
			keep   int
		}{
			{"state-only-keep-zero", [2]bool{false, false}, 0},
			{"state-only-keep-one", [2]bool{false, false}, 1},
			{"state-only-keep-full", [2]bool{false, false}, 2},
			{"keep-logits-reject-state-only", [2]bool{true, false}, 1},
			{"keep-state-only-reject-logits", [2]bool{false, true}, 1},
		} {
			t.Run(plan.name+"/initial-nil-"+itoa(boolToIntV41Expert(seedNil)), func(t *testing.T) {
				m, st, ref := v41RoundModeStates(t)
				if seedNil {
					st.forwardCoverage, ref.forwardCoverage = nil, nil
				}
				history := &st.history[0]
				layers := append([]*V41AttentionState(nil), st.layers...)
				coverage := []*NativeForwardCoverage{st.forwardCoverage}
				registry := []*V41AttentionState{st.attn}
				storage := [][]uintptr{v41RoundStorageIdentity(st)}
				values := map[*NativeForwardCoverage]NativeForwardCoverage{}
				if st.forwardCoverage != nil {
					values[st.forwardCoverage] = *st.forwardCoverage
				}
				r, err := m.beginV41StepRound(st, 2)
				if err != nil {
					t.Fatal(err)
				}
				defer r.abort()
				for i, withLogits := range plan.logits {
					headCalls := m.V41ExpertFaultAttribution().Decode.HeadProjectionHostCalls
					tx := v41RoundPreparedTestStep(t, r, 4+i)
					var got []float32
					if withLogits {
						got, err = tx.finalize()
					} else {
						err = tx.finalizeNoLogits()
					}
					if err != nil || !tx.stats.Committed || !tx.closed || tx.commit != nil || len(st.history) != 4+i {
						t.Fatalf("consumed token did not finish exactly once: %v", err)
					}
					if !withLogits && (st.forwardCoverage != nil || m.V41ExpertFaultAttribution().Decode.HeadProjectionHostCalls != headCalls) {
						t.Fatal("state-only round token projected a head or invented logits coverage")
					}
					if withLogits && (st.forwardCoverage == nil || st.forwardCoverage.Validate() != nil || m.V41ExpertFaultAttribution().Decode.HeadProjectionHostCalls != headCalls+1) {
						t.Fatal("logits round token lost its completed coverage")
					}
					coverage = append(coverage, st.forwardCoverage)
					registry = append(registry, st.attn)
					storage = append(storage, v41RoundStorageIdentity(st))
					if st.forwardCoverage != nil {
						values[st.forwardCoverage] = *st.forwardCoverage
					}
					if i < plan.keep {
						want, stats, err := m.forwardV41StepCore(4+i, ref, nil, false, withLogits)
						if err != nil || !stats.Committed || !reflect.DeepEqual(got, want) {
							t.Fatalf("retained token differs from ordinary serial execution: %v", err)
						}
					}
				}
				entries := append([]*v41StepRoundEntry(nil), r.entries...)
				if err := r.finish(plan.keep); err != nil {
					t.Fatal(err)
				}
				if st.activeStepRound != nil || r.entries != nil || r.pending != nil || r.state != nil || r.model != nil || r.layers != nil {
					t.Fatal("mode finish retained an active owner or journal references")
				}
				for _, entry := range entries {
					if !reflect.DeepEqual(*entry, v41StepRoundEntry{}) {
						t.Fatal("mode finish retained a kept or rejected entry")
					}
				}
				if st.forwardCoverage != coverage[plan.keep] || st.attn != registry[plan.keep] ||
					&st.history[0] != history || !reflect.DeepEqual(v41RoundStorageIdentity(st), storage[plan.keep]) ||
					!reflect.DeepEqual(captureV41ForwardSnapshot(st), captureV41ForwardSnapshot(ref)) {
					t.Fatal("mode disposition differs from the exact ordinary retained prefix")
				}
				if !reflect.DeepEqual(st.forwardCoverage, ref.forwardCoverage) {
					t.Fatal("mode disposition lost the distinction between nil and completed logits coverage")
				}
				for pointer, value := range values {
					if *pointer != value {
						t.Fatal("mode rollback mutated previously published coverage")
					}
				}
				for i, pointer := range layers {
					if st.layers[i] != pointer {
						t.Fatal("mode rollback replaced a retained layer")
					}
				}
				got, _, err := m.forwardV41Step(6, st, nil)
				want, _, refErr := m.forwardV41Step(6, ref, nil)
				if err != nil || refErr != nil || !reflect.DeepEqual(got, want) ||
					!reflect.DeepEqual(captureV41ForwardSnapshot(st), captureV41ForwardSnapshot(ref)) {
					t.Fatal("ordinary continuation differs after mixed-mode prefix retention")
				}
			})
		}
	}
}

// fak-test:justify why=guard twin=TestV41StepRoundNoLogitsRetention when=changed:internal/model/**
// fak-test:runtime medium est=3s lane=default
func TestV41StepRoundNoLogitsFailure(t *testing.T) {
	withV41ActivationTraceEnv(t, "")
	for _, firstLogits := range []bool{false, true} {
		for _, fault := range []string{"normalization-error", "nonfinite-normalization"} {
			t.Run(fault+"/prefix-logits-"+itoa(boolToIntV41Expert(firstLogits)), func(t *testing.T) {
				m, st, ref := v41RoundModeStates(t)
				before, coverage, registry := captureV41ForwardSnapshot(st), st.forwardCoverage, st.attn
				storage := v41RoundStorageIdentity(st)
				r, err := m.beginV41StepRound(st, 2)
				if err != nil {
					t.Fatal(err)
				}
				defer r.abort()
				first := v41RoundPreparedTestStep(t, r, 4)
				if firstLogits {
					_, err = first.finalize()
				} else {
					err = first.finalizeNoLogits()
				}
				if err != nil || !first.stats.Committed {
					t.Fatal("failure fixture did not commit its first token")
				}
				committed := captureV41ForwardSnapshot(st)
				tx := v41RoundPreparedTestStep(t, r, 5)
				if reflect.DeepEqual(captureV41ForwardSnapshot(st), committed) {
					t.Fatal("failure fixture never changed live layer or registry state")
				}
				primary := errors.New("independent final normalization failure")
				want := primary
				if fault == "nonfinite-normalization" {
					want = errV41ProjectionResult
				}
				normalize := st.finalNorm()
				calls := 0
				st.setFinalNorm(func(input []float32) ([]float32, error) {
					calls++
					if fault == "normalization-error" {
						return nil, primary
					}
					row := append([]float32(nil), input...)
					row[0] = float32(math.NaN())
					return row, nil
				})
				headCalls := m.V41ExpertFaultAttribution().Decode.HeadProjectionHostCalls
				got := recoverError(func() { _ = tx.finalizeNoLogits() })
				st.setFinalNorm(normalize)
				var selected *V41ProjectionOperationError
				if calls != 1 || !errors.Is(got, want) || !errors.As(got, &selected) ||
					tx.stats.Committed || !tx.closed || len(st.history) != len(before.history)+1 ||
					m.V41ExpertFaultAttribution().Decode.HeadProjectionHostCalls != headCalls ||
					!reflect.DeepEqual(captureV41ForwardSnapshot(st), committed) {
					t.Fatal("failed state-only finalization committed or failed to restore its pending token")
				}
				if err := r.finish(1); !errors.Is(err, errV41StepRoundState) {
					t.Fatal("a failed round retained a positive speculative prefix")
				}
				entries := append([]*v41StepRoundEntry(nil), r.entries...)
				if err := r.abort(); err != nil || st.activeStepRound != nil ||
					st.forwardCoverage != coverage || st.attn != registry ||
					!reflect.DeepEqual(v41RoundStorageIdentity(st), storage) ||
					!reflect.DeepEqual(captureV41ForwardSnapshot(st), before) {
					t.Fatal("failed state-only round did not atomically restore its original host prefix")
				}
				for _, entry := range entries {
					if !reflect.DeepEqual(*entry, v41StepRoundEntry{}) {
						t.Fatal("failed state-only round retained entry references")
					}
				}
				gotNext, _, err := m.forwardV41Step(6, st, nil)
				wantNext, _, refErr := m.forwardV41Step(6, ref, nil)
				if err != nil || refErr != nil || !reflect.DeepEqual(gotNext, wantNext) ||
					!reflect.DeepEqual(captureV41ForwardSnapshot(st), captureV41ForwardSnapshot(ref)) {
					t.Fatal("host state differs from untouched serial continuation after failure")
				}
			})
		}
	}
}

// fak-test:justify why=guard twin=TestV41StepRoundNoLogitsRetention when=changed:internal/model/**
// fak-test:runtime medium est=2s lane=default
func TestV41StepRoundNoLogitsFence(t *testing.T) {
	withV41ActivationTraceEnv(t, "")
	for _, boundary := range []string{"empty", "state-only-committed", "pending-layer"} {
		t.Run(boundary, func(t *testing.T) {
			m, st, _ := v41RoundModeStates(t)
			r, err := m.beginV41StepRound(st, 2)
			if err != nil {
				t.Fatal(err)
			}
			defer r.abort()
			if boundary == "state-only-committed" {
				tx := v41RoundPreparedTestStep(t, r, 4)
				if err := tx.finalizeNoLogits(); err != nil {
					t.Fatal(err)
				}
			} else if boundary == "pending-layer" {
				tx, err := r.beginStep(4, nil)
				if err != nil {
					t.Fatal(err)
				}
				if err := tx.advanceLayer(); err != nil {
					t.Fatal(err)
				}
			}
			before, coverage, registry := captureV41ForwardSnapshot(st), st.forwardCoverage, st.attn
			storage := v41RoundStorageIdentity(st)
			ids := append(append([]int(nil), st.history...), 6)
			for _, lastOnly := range []bool{false, true} {
				act, err := m.forwardV41WithHeadMode(ids, st, lastOnly, false)
				if !errors.Is(err, errV41StepRoundState) || act != nil || st.forwardCoverage != coverage ||
					st.attn != registry || !reflect.DeepEqual(v41RoundStorageIdentity(st), storage) ||
					!reflect.DeepEqual(captureV41ForwardSnapshot(st), before) {
					t.Fatal("direct discarded-head full forward bypassed or mutated the active round")
				}
			}
			if err := r.abort(); err != nil {
				t.Fatal(err)
			}
			ids = append(append([]int(nil), st.history...), 6)
			phase := m.v41ExpertFaultPhase()
			m.v41SetExpertFaultPhase(V41PhasePrefill)
			defer m.v41SetExpertFaultPhase(phase)
			headCalls := m.V41ExpertFaultAttribution().Prefill.HeadProjectionHostCalls
			if _, err := m.forwardV41WithHeadMode([]int{6}, st, true, false); err != nil ||
				st.forwardCoverage != nil || !reflect.DeepEqual(st.history, ids) ||
				m.V41ExpertFaultAttribution().Prefill.HeadProjectionHostCalls != headCalls {
				t.Fatal("ordinary discarded-head full forward remained fenced or fabricated logits")
			}
		})
	}
}

func v41RoundInjectIncompleteUnwind(kind string) {
	if kind == "goexit" {
		runtime.Goexit()
	}
	panic(nil)
}

func v41RoundCheckIncompleteUnwind(t *testing.T, kind string, fn func()) {
	t.Helper()
	mode := "1"
	if kind == "nil-modern" {
		mode = "0"
	}
	t.Setenv("GODEBUG", os.Getenv("GODEBUG")+",panicnil="+mode)
	var payload any
	returned, boundaryReturned := false, false
	done := make(chan struct{})
	go func() {
		defer close(done)
		func() {
			defer func() { payload = recover() }()
			fn()
			returned = true
		}()
		boundaryReturned = true
	}()
	select {
	case <-done:
	case <-time.After(5 * time.Second):
		t.Fatal("round unwind did not complete")
	}
	if returned || boundaryReturned == (kind == "goexit") {
		t.Fatalf("unwind kind=%s returned=%t boundaryReturned=%t", kind, returned, boundaryReturned)
	}
	if kind == "nil-modern" {
		if _, ok := payload.(*runtime.PanicNilError); !ok {
			t.Fatalf("nil panic changed type: %T", payload)
		}
	} else if payload != nil {
		t.Fatalf("abrupt completion replaced by %T", payload)
	}
}

// fak-test:justify why=regression when=changed:internal/model/**
// fak-test:runtime medium est=3s lane=default
func TestV41StepRoundPreservesIncompleteUnwind(t *testing.T) {
	for _, site := range []string{"constructor", "layer", "state-only-head"} {
		for _, kind := range []string{"nil-modern", "nil-legacy", "goexit"} {
			t.Run(site+"/"+kind, func(t *testing.T) {
				m, st, _ := v41RoundModeStates(t)
				before := captureV41ForwardSnapshot(st)
				coverage := st.forwardCoverage
				r, err := m.beginV41StepRound(st, 2)
				if err != nil {
					t.Fatal(err)
				}
				first := v41RoundPreparedTestStep(t, r, 4)
				if _, err := first.finalize(); err != nil {
					t.Fatal(err)
				}
				committed := captureV41ForwardSnapshot(st)
				var tx *v41StepTransaction
				var entry *v41StepRoundEntry
				var scratch v41ProjScratch
				faults, cleanups := 0, 0
				secondary := errors.New("round cleanup sentinel")
				bind := func(current *v41StepTransaction) {
					tx, entry = current, current.roundEntry
					cleanup := current.cleanup
					current.cleanup = func() {
						cleanups++
						if cleanup != nil {
							cleanup()
						}
						panic(secondary)
					}
				}
				inject := func() {
					faults++
					v41RoundInjectIncompleteUnwind(kind)
				}
				gather := st.gatherOps
				var call func()
				if site == "constructor" {
					ops := &v41DeviceGatherOps{}
					ops.embedGather = func(ids []int) ([]float32, error) {
						bind(r.pending.tx)
						inject()
						return m.v41EmbeddingPanel(ids)
					}
					st.gatherOps = ops
					call = func() { _, _ = r.beginStep(5, &scratch) }
				} else {
					current, err := r.beginStep(5, &scratch)
					if err != nil {
						t.Fatal(err)
					}
					bind(current)
					if site == "layer" {
						tx.afterLayer = func(int) error { inject(); return nil }
						call = func() { _ = tx.advanceLayer() }
					} else {
						for tx.nextLayer < tx.layers {
							if err := tx.advanceLayer(); err != nil {
								t.Fatal(err)
							}
						}
						head := tx.headInput
						tx.headInput = func() ([]float32, error) {
							row, err := head()
							if err == nil {
								inject()
							}
							return row, err
						}
						call = func() { _ = tx.finalizeNoLogits() }
					}
				}
				v41RoundCheckIncompleteUnwind(t, kind, call)
				st.gatherOps = gather
				if faults != 1 || cleanups != 1 || tx == nil || !tx.closed || tx.stats.Committed ||
					entry == nil || !entry.closed || entry.cleanupOK || entry.tx != nil ||
					r.pending != nil || !r.failed || !r.poisoned || tx.roundEntry != nil ||
					scratch.gatherOps != nil {
					t.Fatal("round-bound incomplete operation did not retire its pending transaction once")
				}
				if st.forwardCoverage != nil || !reflect.DeepEqual(captureV41ForwardSnapshot(st), committed) {
					t.Fatal("incomplete pending token changed the committed prefix")
				}
				if err := r.finish(1); !errors.Is(err, errV41StepRoundState) {
					t.Fatal("uncertain cleanup accepted a positive retained prefix")
				}
				entries := append([]*v41StepRoundEntry(nil), r.entries...)
				if err := r.abort(); !errors.Is(err, errV41StepRoundState) {
					t.Fatal("uncertain cleanup lost its closed failure")
				}
				if st.forwardCoverage != coverage || !reflect.DeepEqual(captureV41ForwardSnapshot(st), before) {
					t.Fatal("round abort failed to restore its original boundary")
				}
				v41RoundAssertPoisoned(t, m, st, r, entries)
				tx.abort()
				if cleanups != 1 {
					t.Fatal("closed transaction repeated resource cleanup")
				}
			})
		}
	}
}

// fak-test:justify why=regression when=changed:internal/model/**
// fak-test:runtime medium est=2s lane=default
func TestV41StepRoundDrainPreservesIncompleteUnwind(t *testing.T) {
	for _, site := range []string{"pending-cleanup", "retained-undo"} {
		for _, kind := range []string{"nil-modern", "nil-legacy", "goexit"} {
			t.Run(site+"/"+kind, func(t *testing.T) {
				m, st, _ := v41RoundModeStates(t)
				before := captureV41ForwardSnapshot(st)
				coverage := st.forwardCoverage
				r, err := m.beginV41StepRound(st, 2)
				if err != nil {
					t.Fatal(err)
				}
				first := v41RoundPreparedTestStep(t, r, 4)
				if _, err := first.finalize(); err != nil {
					t.Fatal(err)
				}
				firstUndo, firstUndos := r.entries[0].undo, 0
				r.entries[0].undo = func() { firstUndos++; firstUndo() }
				faults := 0
				inject := func() { faults++; v41RoundInjectIncompleteUnwind(kind) }
				var call func()
				if site == "pending-cleanup" {
					tx, err := r.beginStep(5, &v41ProjScratch{})
					if err != nil {
						t.Fatal(err)
					}
					if err := tx.advanceLayer(); err != nil {
						t.Fatal(err)
					}
					cleanup := tx.cleanup
					tx.cleanup = func() {
						if cleanup != nil {
							cleanup()
						}
						inject()
					}
					call = func() { _ = r.abort() }
				} else {
					tx := v41RoundPreparedTestStep(t, r, 5)
					if err := tx.finalizeNoLogits(); err != nil {
						t.Fatal(err)
					}
					entry := r.entries[1]
					undo := entry.undo
					entry.undo = func() { undo(); inject() }
					call = func() { _ = r.finish(0) }
				}
				entries := append([]*v41StepRoundEntry(nil), r.entries...)
				v41RoundCheckIncompleteUnwind(t, kind, call)
				if faults != 1 || firstUndos != 1 {
					t.Fatal("incomplete drain repeated its fault or skipped the earlier retained undo")
				}
				if st.forwardCoverage != coverage || !reflect.DeepEqual(captureV41ForwardSnapshot(st), before) {
					t.Fatal("incomplete drain failed to restore the original boundary")
				}
				v41RoundAssertPoisoned(t, m, st, r, entries)
			})
		}
	}
}
