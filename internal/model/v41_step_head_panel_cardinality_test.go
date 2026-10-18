package model

import (
	"context"
	"errors"
	"maps"
	"reflect"
	"sync"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// Reuse the actual cpu-ref recorder and the first two independent prefix seeds.
// Extra lanes have distinct positions, tokens, caches and deep-owned state.
func newV41SharedHeadCardinalityFixture(t *testing.T, m *Model) v41SharedHeadFixture {
	t.Helper()
	f := newV41SharedHeadFixture(t, m)
	for lane := len(f.lanes); lane < 5; lane++ {
		prefix := make([]int, lane+3)
		for i := range prefix {
			prefix[i] = (i + 1) % m.Cfg.VocabSize
		}
		source := m.NewSession()
		t.Cleanup(source.Close)
		if len(source.Prefill(prefix)) == 0 {
			t.Fatal("extra prefix produced no logits")
		}
		snapshot := captureV41ForwardSnapshot(source.v41State())
		source.Close()
		s, ref := v41DenseTestSession(t, m, f.b), v41DenseTestSession(t, m, f.b)
		s.v41Forward, ref.v41Forward = snapshot.clone().restore(), snapshot.clone().restore()
		for _, st := range []*v41ForwardState{s.v41State(), ref.v41State()} {
			for _, state := range st.layers {
				// Exercise restoration of the overwritten row, not only its cursor.
				slot := state.nextWindowPos % state.windowSize
				state.window[slot] = append([]float32(nil), state.window[0]...)
			}
		}
		f.lanes = append(f.lanes, v41StepCohortLane{session: s, id: (6 + lane) % m.Cfg.VocabSize})
		f.refs = append(f.refs, ref)
		f.before = append(f.before, captureV41ForwardSnapshot(s.v41Forward))
	}
	for lane, input := range f.lanes {
		if len(input.session.v41Forward.history) != lane+3 {
			t.Fatalf("lane %d lost its independent position", lane)
		}
		for other := 0; other <= lane; other++ {
			for _, peer := range []*Session{f.lanes[other].session, f.refs[other]} {
				if peer == input.session {
					continue
				}
				if peer.Cache == input.session.Cache || peer.v41Forward == input.session.v41Forward || (input.session.v41Forward.attn != nil && peer.v41Forward.attn == input.session.v41Forward.attn) {
					t.Fatalf("lane %d aliases cache, continuation or source registry", lane)
				}
				for layer, state := range input.session.v41Forward.layers {
					if state == peer.v41Forward.layers[layer] {
						t.Fatalf("lane %d aliases retained layer %d", lane, layer)
					}
				}
			}
		}
	}
	return f
}

// Every row uses cpu-ref's real MatMul/BatchedMatMul with identical F32/Q2_K/
// Q6_K reduction order. The fixture and four scalar oracles are reused within
// each dtype; each case restores independently owned full continuation state.
// This qualifies software semantics only, not physical Vulkan or throughput.
// fak-test:runtime medium est=8s lane=default
func TestV41StepCohortSharedHeadCardinality(t *testing.T) {
	t.Setenv(v41StepBatchEnv, "1")
	for _, dtype := range []compute.Dtype{compute.F32, compute.Q2_K, compute.Q6_K} {
		t.Run(dtype.String(), func(t *testing.T) {
			var m *Model
			if dtype == compute.F32 {
				m = v41CompressorTestFixtureIndex(t, 128)
				m.Cfg.LogitScale = .5
				t.Cleanup(func() {
					if err := m.CloseWeights(); err != nil {
						t.Error(err)
					}
				})
			} else {
				m = v41HeadProjFixture(t)
				if dtype == compute.Q6_K {
					m.kqw[m.headName()] = q6kTestTensor(m.Cfg.VocabSize, m.Cfg.HiddenSize, 0x61366)
				}
			}
			f := newV41SharedHeadCardinalityFixture(t, m)
			if f.b.Name() != "cpu-ref" {
				t.Fatalf("arithmetic backend=%q, want cpu-ref", f.b.Name())
			}
			weight := f.b.headWeight.(compute.HostBuffer)
			weightF32 := append([]float32(nil), weight.F32()...)
			weightI8 := append([]int8(nil), weight.I8()...)
			assertWeight := func(t *testing.T) {
				t.Helper()
				for lane, input := range f.lanes {
					gotDtype, ok := input.session.v41HeadProjectionDtype(m.headName(), m.Cfg.VocabSize, m.Cfg.HiddenSize)
					if !ok || gotDtype != dtype || input.session.v41HeadWeightHAL(m.headName(), gotDtype, m.Cfg.VocabSize, m.Cfg.HiddenSize).Buf() != f.b.headWeight {
						t.Fatalf("lane %d did not share the same %s weight handle", lane, dtype)
					}
				}
				if !reflect.DeepEqual(weight.F32(), weightF32) || !reflect.DeepEqual(weight.I8(), weightI8) {
					t.Fatal("shared head mutated immutable weight payload")
				}
			}
			assertWeight(t)
			f.b.headRows, f.b.headInputs, f.b.ops = nil, nil, nil
			want := make([][]float32, 4)
			wantState := make([]*v41ForwardSnapshot, 4)
			for lane := range want {
				var err error
				want[lane], _, err = m.forwardV41Step(f.lanes[lane].id, f.refs[lane].v41State(), nil)
				if err != nil {
					t.Fatal(err)
				}
				wantState[lane] = captureV41ForwardSnapshot(f.refs[lane].v41Forward)
			}
			if !reflect.DeepEqual(f.b.headRows, []int{1, 1, 1, 1}) || len(f.b.headInputs) != 4 {
				t.Fatalf("scalar oracle head rows=%v", f.b.headRows)
			}
			wantInputs := append([][]float32(nil), f.b.headInputs...)
			// Rank-one norm weights are persistent uploads in this recorder.
			// After scalar warmup, every cohort must retain exactly that set.
			resident := maps.Clone(f.b.live)
			for lane, input := range f.lanes {
				if !reflect.DeepEqual(captureV41ForwardSnapshot(input.session.v41Forward), f.before[lane]) {
					t.Fatalf("scalar oracle mutated independent lane %d", lane)
				}
			}
			for lane := 1; lane < len(wantInputs); lane++ {
				for peer := 0; peer < lane; peer++ {
					if reflect.DeepEqual(wantInputs[lane], wantInputs[peer]) {
						t.Fatalf("lanes %d and %d have indistinguishable head rows", lane, peer)
					}
				}
			}

			reset := func() {
				for lane := range f.lanes {
					f.lanes[lane].session.v41Forward = f.before[lane].clone().restore()
					f.lanes[lane].session.v41State()
					f.lanes[lane].ctx = nil
				}
				f.b.reset()
				f.b.headRows, f.b.headInputs, f.b.ops = nil, nil, nil
				f.b.uploads, f.b.reads = map[compute.Buffer]int{}, map[compute.Buffer]int{}
				f.b.beforeHead, f.b.afterHead = nil, nil
				f.b.freeAttempts = 0
				m.v41SetExpertFaultPhase(V41PhaseDecode)
			}
			for _, tc := range []struct {
				name      string
				lanes     int
				survivors uint
				afterHead bool
			}{
				{name: "three independent lanes", lanes: 3, survivors: 0b0111},
				{name: "four independent lanes", lanes: 4, survivors: 0b1111},
				{name: "preparation leaves one", lanes: 4, survivors: 0b1000},
				{name: "preparation leaves two with hole", lanes: 4, survivors: 0b1010},
				{name: "preparation leaves three", lanes: 4, survivors: 0b1110},
				{name: "projection leaves one", lanes: 4, survivors: 0b1000, afterHead: true},
				{name: "projection leaves two with hole", lanes: 4, survivors: 0b1010, afterHead: true},
				{name: "projection leaves three", lanes: 4, survivors: 0b1110, afterHead: true},
			} {
				t.Run(tc.name, func(t *testing.T) {
					reset()
					var cancels []context.CancelFunc
					var panel []float32
					for lane := 0; lane < tc.lanes; lane++ {
						survives := tc.survivors&(1<<lane) != 0
						if survives || tc.afterHead {
							panel = append(panel, wantInputs[lane]...)
						}
						if survives {
							continue
						}
						ctx, cancel := context.WithCancel(context.Background())
						t.Cleanup(cancel)
						f.lanes[lane].ctx = ctx
						cancels = append(cancels, cancel)
						if !tc.afterHead {
							st := f.lanes[lane].session.v41State()
							norm := st.finalNorm()
							st.setFinalNorm(func(x []float32) ([]float32, error) {
								cancel()
								if norm != nil {
									return norm(x)
								}
								return m.finalNorm(x), nil
							})
						}
					}
					if tc.afterHead {
						f.b.afterHead = func() {
							for _, cancel := range cancels {
								cancel()
							}
						}
					}
					var owner sync.Mutex
					f.b.beforeHead = func() {
						if owner.TryLock() {
							owner.Unlock()
							t.Fatal("shared head ran without device owner")
						}
						if f.b.active || f.b.ends != 1 {
							t.Fatal("head ran before final layer flush")
						}
						for lane, input := range f.lanes {
							if !reflect.DeepEqual(input.session.v41Forward.history, f.before[lane].history) {
								t.Fatalf("lane %d committed before shared head", lane)
							}
						}
					}
					before := m.V41ExpertFaultAttribution().Decode
					got, err := m.forwardV41StepCohortSharedHead(&owner, f.lanes[:tc.lanes])
					if err != nil || len(got) != tc.lanes {
						t.Fatalf("cohort results=%d err=%v", len(got), err)
					}
					rows := len(panel) / m.Cfg.HiddenSize
					if !reflect.DeepEqual(f.b.headRows, []int{rows}) || !reflect.DeepEqual(f.b.headInputs, [][]float32{panel}) || f.b.beginCalls != 1 || f.b.ends != 1 || f.b.active {
						t.Fatalf("head rows=%v want=%d begin=%d end=%d or panel changed", f.b.headRows, rows, f.b.beginCalls, f.b.ends)
					}
					operations := 0
					for _, op := range f.b.ops {
						if op.weight == f.b.headWeight {
							operations++
							if op.batch != (rows > 1) || op.rows != rows || op.in != m.Cfg.HiddenSize || op.out != m.Cfg.VocabSize || f.b.uploads[op.input] != 4*rows*m.Cfg.HiddenSize || f.b.reads[op.output] != 4*rows*m.Cfg.VocabSize {
								t.Fatalf("actual head operation=%+v want rows=%d", op, rows)
							}
						}
					}
					if operations != 1 || f.b.freeAttempts != 2 || len(f.b.headTensors) != 0 || len(f.b.headOutputs) != 0 || !reflect.DeepEqual(f.b.live, resident) {
						t.Fatalf("head operations=%d frees=%d live=%d resident=%d or transient head tensors leaked", operations, f.b.freeAttempts, len(f.b.live), len(resident))
					}
					// Attribution counts the actual projection, including rows whose
					// contexts were cancelled after it; it must not count commits.
					after := m.V41ExpertFaultAttribution().Decode
					if after.HeadProjectionDeviceCalls-before.HeadProjectionDeviceCalls != 1 || after.HeadProjectionDeviceRows-before.HeadProjectionDeviceRows != rows || after.HeadProjectionHostCalls != before.HeadProjectionHostCalls || after.HeadProjectionHostRows != before.HeadProjectionHostRows || after.HeadProjectionActivationUploadBytes-before.HeadProjectionActivationUploadBytes != int64(4*rows*m.Cfg.HiddenSize) || after.HeadProjectionReadbackBytes-before.HeadProjectionReadbackBytes != int64(4*rows*m.Cfg.VocabSize) {
						t.Fatalf("head attribution did not count one actual %d-row operation without host replay: before=%+v after=%+v", rows, before, after)
					}
					for lane, input := range f.lanes {
						state := captureV41ForwardSnapshot(input.session.v41Forward)
						if input.session.BackendSessionClosed() {
							t.Fatalf("lane %d backend retired on cancellation", lane)
						}
						if lane >= tc.lanes {
							if !reflect.DeepEqual(state, f.before[lane]) {
								t.Fatalf("unsubmitted lane %d mutated", lane)
							}
							continue
						}
						result := got[lane]
						if tc.survivors&(1<<lane) != 0 {
							if result.err != nil || !result.stats.Committed || !reflect.DeepEqual(result.logits, want[lane]) || !reflect.DeepEqual(state, wantState[lane]) {
								t.Fatalf("survivor %d differs from scalar logits or full state: %+v err=%v", lane, result.stats, result.err)
							}
						} else if !errors.Is(result.err, context.Canceled) || result.stats.Committed || len(result.logits) != 0 || !reflect.DeepEqual(state, f.before[lane]) {
							t.Fatalf("cancelled lane %d did not fully roll back: %+v err=%v", lane, result.stats, result.err)
						}
					}
					assertWeight(t)
					if !owner.TryLock() {
						t.Fatal("cohort leaked device owner")
					}
					owner.Unlock()
				})
			}
			t.Run("five lanes decline before effects", func(t *testing.T) {
				reset()
				if v41SharedHeadMaxLanes != 4 {
					t.Fatalf("shared-head lane cap=%d, want 4", v41SharedHeadMaxLanes)
				}
				if err := m.v41StepCohortAdmit(f.lanes); err != nil {
					t.Fatalf("five independent lanes failed ordinary cohort admission: %v", err)
				}
				before := m.V41ExpertFaultAttribution()
				uploads, reads, live := maps.Clone(f.b.uploads), maps.Clone(f.b.reads), maps.Clone(f.b.live)
				var owner sync.Mutex
				got, err := m.forwardV41StepCohortSharedHead(&owner, f.lanes)
				if !errors.Is(err, errV41StepCohortDeclined) || len(got) != 0 || f.b.beginCalls != 0 || f.b.ends != 0 || len(f.b.ops) != 0 || len(f.b.headRows) != 0 || f.b.freeAttempts != 0 {
					t.Fatalf("five-lane admission had effects: err=%v results=%d begin=%d end=%d ops=%d", err, len(got), f.b.beginCalls, f.b.ends, len(f.b.ops))
				}
				if !reflect.DeepEqual(before, m.V41ExpertFaultAttribution()) || !reflect.DeepEqual(uploads, f.b.uploads) || !reflect.DeepEqual(reads, f.b.reads) || !reflect.DeepEqual(live, f.b.live) {
					t.Fatal("five-lane decline changed backend resources or attribution")
				}
				for lane, input := range f.lanes {
					if input.session.BackendSessionClosed() || !reflect.DeepEqual(captureV41ForwardSnapshot(input.session.v41Forward), f.before[lane]) {
						t.Fatalf("five-lane decline changed lane %d", lane)
					}
				}
				assertWeight(t)
				if !owner.TryLock() {
					t.Fatal("decline leaked device owner")
				}
				owner.Unlock()
			})
		})
	}
}
