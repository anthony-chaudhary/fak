package model

import (
	"reflect"
	"testing"
)

func TestHybridSpecStateRollbackRestoresBothFamilies(t *testing.T) {
	state, err := NewHybridSpecState(
		[][]byte{{1, 2}, {3, 4}},
		[][]byte{{11, 12}, {13, 14}},
	)
	if err != nil {
		t.Fatal(err)
	}
	checkpoint := state.Begin()
	if err := state.Append([]byte{5, 6}, []byte{15, 16}); err != nil {
		t.Fatal(err)
	}
	if err := state.Append([]byte{7, 8}, []byte{17, 18}); err != nil {
		t.Fatal(err)
	}

	if err := state.Rollback(checkpoint); err != nil {
		t.Fatal(err)
	}
	assertHybridSpecState(t, state, 2,
		[][]byte{{1, 2}, {3, 4}},
		[][]byte{{11, 12}, {13, 14}},
	)
}

func TestHybridSpecStateCommitKeepsExactAcceptedPrefix(t *testing.T) {
	state, err := NewHybridSpecState(
		[][]byte{{1}, {2}},
		[][]byte{{11}, {12}},
	)
	if err != nil {
		t.Fatal(err)
	}
	checkpoint := state.Begin()
	for i := byte(0); i < 3; i++ {
		if err := state.Append([]byte{3 + i}, []byte{13 + i}); err != nil {
			t.Fatal(err)
		}
	}

	if err := state.Commit(checkpoint, 2); err != nil {
		t.Fatal(err)
	}
	assertHybridSpecState(t, state, 4,
		[][]byte{{1}, {2}, {3}, {4}},
		[][]byte{{11}, {12}, {13}, {14}},
	)
}

func TestHybridSpecStateRestoresContentAndRejectsUnsafeCheckpoint(t *testing.T) {
	state, err := NewHybridSpecState([][]byte{{1, 2}}, [][]byte{{11, 12}})
	if err != nil {
		t.Fatal(err)
	}
	checkpoint := state.Begin()

	// Mutating caller-owned and returned buffers must not corrupt a checkpoint.
	kv := []byte{3, 4}
	recurrent := []byte{13, 14}
	if err := state.Append(kv, recurrent); err != nil {
		t.Fatal(err)
	}
	kv[0], recurrent[0] = 99, 99
	view := state.KVState()
	view[0][0] = 88

	if err := state.Rollback(checkpoint); err != nil {
		t.Fatal(err)
	}
	assertHybridSpecState(t, state, 1, [][]byte{{1, 2}}, [][]byte{{11, 12}})
	if err := state.Rollback(checkpoint); err == nil {
		t.Fatal("stale checkpoint rollback succeeded")
	}

	other, err := NewHybridSpecState(nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	if err := other.Rollback(otherCheckpointForTest(state)); err == nil {
		t.Fatal("foreign checkpoint rollback succeeded")
	}
}

func TestHybridSpecStateRejectsUnpairedFamilies(t *testing.T) {
	if _, err := NewHybridSpecState([][]byte{{1}}, nil); err == nil {
		t.Fatal("constructed state with mismatched families")
	}
	state, err := NewHybridSpecState(nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	checkpoint := state.Begin()
	if err := state.Commit(checkpoint, 1); err == nil {
		t.Fatal("accepted beyond speculative suffix")
	}
	assertHybridSpecState(t, state, 0, nil, nil)
}

func otherCheckpointForTest(state *HybridSpecState) HybridSpecCheckpoint {
	return state.Begin()
}

func assertHybridSpecState(t *testing.T, state *HybridSpecState, cursor int, kv, recurrent [][]byte) {
	t.Helper()
	if state.Cursor() != cursor {
		t.Fatalf("cursor = %d, want %d", state.Cursor(), cursor)
	}
	if got := state.KVState(); !reflect.DeepEqual(got, kv) {
		t.Fatalf("KV state = %v, want %v", got, kv)
	}
	if got := state.RecurrentState(); !reflect.DeepEqual(got, recurrent) {
		t.Fatalf("recurrent state = %v, want %v", got, recurrent)
	}
}
