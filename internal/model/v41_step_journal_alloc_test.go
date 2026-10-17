package model

import "testing"

// fak-test:runtime fast est=10ms lane=default
func TestV41StepJournalAllocationAndRollback(t *testing.T) {
	previous, registry := &V41AttentionState{}, &V41AttentionState{}
	st := &v41ForwardState{attn: registry}
	for i := 0; i < 2; i++ {
		inputs := make([][]float32, 1, 2)
		inputs[0] = []float32{float32(i + 1)}
		positions := make([]int, 1, 2)
		positions[0] = i
		st.layers = append(st.layers, &V41AttentionState{
			windowSize: 1, window: [][]float32{{float32(i + 3)}},
			nextWindowPos: i + 1, nextCompressRow: i + 2,
			partialInputs: inputs, partialPositions: positions,
			retainedCopies: i + 4, retainedWindowRows: i + 5,
		})
	}
	var stats v41StepStats
	allocs := testing.AllocsPerRun(100, func() {
		st.attn = registry
		j := newV41StepJournal(st, &stats, previous, registry, len(st.layers))
		for i, state := range st.layers {
			j.stage(i, state)
			state.nextWindowPos, state.nextCompressRow = 99, 99
			state.retainedCopies, state.retainedWindowRows = 99, 99
			state.window[0][0] = 99
			state.partialInputs = append(state.partialInputs, state.partialInputs[0])
			state.partialPositions = append(state.partialPositions, 99)
		}
		j.rollback()
		j.rollback() // A second rollback must not restore or count twice.
		if st.attn != previous || stats.LayersRolledBack != len(st.layers) {
			panic("journal registry or rollback count changed")
		}
		for i, state := range st.layers {
			if state.nextWindowPos != i+1 || state.nextCompressRow != i+2 ||
				state.retainedCopies != i+4 || state.retainedWindowRows != i+5 ||
				state.window[0][0] != float32(i+3) || len(state.partialInputs) != 1 ||
				state.partialInputs[0][0] != float32(i+1) || state.partialInputs[:2][1] != nil ||
				len(state.partialPositions) != 1 || state.partialPositions[0] != i {
				panic("journal failed to restore layer state or release failed input")
			}
		}
	})
	t.Logf("two-layer journal stage/rollback: %g allocations", allocs)
	// One journal, one layer ledger, two registry ledgers and two owned ring
	// backups. Per-field backing arrays must not return to the decode path.
	if allocs > 6 {
		t.Fatalf("journal allocated %g objects, want at most 6", allocs)
	}
}
