package model

import (
	"errors"
	"reflect"
	"slices"
	"testing"
)

// The CPU recorder exercises the selected projection path, not physical GPU
// execution. This is an ownership/rollback witness, not a throughput benchmark.
// The runtime estimate is unmeasured.
// fak-test:runtime medium est=10s lane=default
func TestV41StepPartialGroupUndoRetainsOwnedStorage(t *testing.T) {
	for _, device := range []bool{false, true} {
		for _, prefixLen := range []int{3, 4} {
			t.Run(itoa(boolToIntV41Expert(device))+"/prefix-"+itoa(prefixLen), func(t *testing.T) {
				m := v41CompressorTestFixtureIndex(t, 128)
				t.Cleanup(func() { _ = m.CloseWeights() })
				backend := newV41DenseTestBackend()
				newSession := func() *Session {
					if device {
						return v41DenseTestSession(t, m, backend)
					}
					s := m.NewSession()
					t.Cleanup(s.Close)
					return s
				}
				source := newSession()
				source.Prefill([]int{1, 2, 3, 4}[:prefixLen])
				cached, err := source.PrefixSnapshot()
				if err != nil {
					t.Fatal(err)
				}
				t.Cleanup(cached.Close)
				cached.PrepareForCache()
				pristine := cached.v41.clone()
				newBranch := func() *Session {
					t.Helper()
					fork, err := cached.CloneForRestore()
					if err != nil {
						t.Fatal(err)
					}
					defer fork.Close()
					s := newSession()
					if err := fork.Restore(s); err != nil {
						t.Fatal(err)
					}
					return s
				}
				branch, peer := newBranch(), newBranch()
				st := branch.v41State()
				state := st.layers[0]
				if st.attn.publicationPrefix == nil || st.attn.publicationPrefix != peer.v41State().attn.publicationPrefix {
					t.Fatal("fixture did not restore shared sealed publications")
				}
				if len(state.partialInputs) != prefixLen%2 {
					t.Fatal("fixture did not retain the expected partial group")
				}
				// Force append to reuse a backing array. In the supported ratio-two
				// schedule a partial group either has one row and completes, or is
				// empty and appends its first row. Neither may overwrite old values.
				inputs := make([][]float32, len(state.partialInputs), 2)
				copy(inputs, state.partialInputs)
				positions := make([]int, len(state.partialPositions), 2)
				copy(positions, state.partialPositions)
				state.partialInputs, state.partialPositions = inputs, positions
				inputValues := cloneV41Rows(inputs)
				positionValues := slices.Clone(positions)
				before := captureV41ForwardSnapshot(st)
				peerBefore := captureV41ForwardSnapshot(peer.v41State())
				oldEnd := state.kvPublishedEnd[0]
				failure := errors.New("head failure after partial-group transition")
				observed := 0
				st.finalNorm = func([]float32) ([]float32, error) {
					observed++
					if prefixLen%2 == 1 {
						if len(state.partialInputs) != 0 || state.kvPublishedEnd[0] != oldEnd+1 {
							t.Fatal("failure did not follow completed-group publication")
						}
					} else if len(state.partialInputs) != 1 || len(state.partialPositions) != 1 ||
						&state.partialInputs[0] != &inputs[:cap(inputs)][0] || &state.partialPositions[0] != &positions[:cap(positions)][0] {
						t.Fatal("incomplete append did not reuse the reserved slice capacity")
					}
					if !reflect.DeepEqual(inputs, inputValues) || !slices.Equal(positions, positionValues) {
						t.Fatal("compressor projection/pooling mutated retained input values")
					}
					return nil, failure
				}
				// Repeat the fault so an append retry also overwrites a stale slot
				// outside the restored length, without corrupting retained ownership.
				for attempt := 0; attempt < 2; attempt++ {
					var returned error
					panicked := recoverError(func() { _, _, returned = m.forwardV41Step(5, st, nil) })
					if returned != nil || !errors.Is(panicked, failure) || observed != attempt+1 {
						t.Fatalf("failure cause/transition lost: returned=%v panic=%v observations=%d", returned, panicked, observed)
					}
					if !reflect.DeepEqual(captureV41ForwardSnapshot(st), before) ||
						len(state.partialInputs) != len(inputs) || cap(state.partialInputs) != cap(inputs) ||
						len(state.partialPositions) != len(positions) || cap(state.partialPositions) != cap(positions) ||
						&state.partialInputs[:cap(state.partialInputs)][0] != &inputs[:cap(inputs)][0] ||
						&state.partialPositions[:cap(state.partialPositions)][0] != &positions[:cap(positions)][0] {
						t.Fatal("rollback copied or changed retained partial-group storage")
					}
					for _, row := range state.partialInputs[len(inputs):cap(state.partialInputs)] {
						if row != nil {
							t.Fatal("rollback retained an aborted input beyond the restored length")
						}
					}
					for row := range inputs {
						if &state.partialInputs[row][0] != &inputs[row][0] ||
							&inputs[row][0] == &cached.v41.layers[0].partialInputs[row][0] ||
							&inputs[row][0] == &peer.v41State().layers[0].partialInputs[row][0] {
							t.Fatal("rollback replaced a retained row or a snapshot shared a mutable row")
						}
					}
				}
				if !reflect.DeepEqual(captureV41ForwardSnapshot(peer.v41State()), peerBefore) || !reflect.DeepEqual(cached.v41.clone(), pristine) {
					t.Fatal("failed continuation changed a sibling or cached snapshot")
				}
				// Public inspection and snapshots still own deep copies, even though
				// transaction-local rollback now borrows immutable retained inputs.
				if len(inputs) > 0 {
					state.PartialGroupInputs()[0][0] += 1
					state.PartialGroupPositions()[0]++
					inspected := captureV41ForwardSnapshot(st)
					inspected.layers[0].partialInputs[0][0] += 1
					inspected.layers[0].partialPositions[0]++
					if !reflect.DeepEqual(captureV41ForwardSnapshot(st), before) {
						t.Fatal("public reader or snapshot aliases retained partial input")
					}
				}
				st.finalNorm = nil
				if got, want := branch.Step(5), peer.Step(5); !reflect.DeepEqual(got, want) {
					t.Fatal("retry logits differ from untouched sibling continuation")
				}
				if !reflect.DeepEqual(captureV41ForwardSnapshot(st), captureV41ForwardSnapshot(peer.v41State())) ||
					!reflect.DeepEqual(inputs, inputValues) || !slices.Equal(positions, positionValues) ||
					!reflect.DeepEqual(cached.v41.clone(), pristine) {
					t.Fatal("successful retry changed retained inputs, snapshot, or continuation values")
				}
				if device && len(backend.ops) == 0 {
					t.Fatal("CPU recording projection witness was vacuous")
				}
			})
		}
	}
}
