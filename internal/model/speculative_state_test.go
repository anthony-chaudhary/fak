package model

import (
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"math/bits"
	"math/rand"
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

func TestHybridSpecStateLongHorizonEquivalence(t *testing.T) {
	const (
		seed          = int64(0x9980)
		rounds        = 4096
		cacheBoundary = 127
		contextWindow = 257
		maxDraft      = 7
	)

	rng := rand.New(rand.NewSource(seed))
	targetKV := make([][]byte, 0, contextWindow)
	targetRecurrent := make([][]byte, 0, contextWindow)
	targetTokens := make([]int, 0, rounds*2)
	speculativeTokens := make([]int, 0, rounds*2)
	state, err := NewHybridSpecState(nil, nil)
	if err != nil {
		t.Fatal(err)
	}

	acceptedTotal := 0
	for round := 0; round < rounds; round++ {
		if round > 0 && round%cacheBoundary == 0 {
			keep := contextWindow/2 + round%23
			targetKV = retainHybridTail(targetKV, keep)
			targetRecurrent = retainHybridTail(targetRecurrent, keep)
			if err := state.RetainLast(keep); err != nil {
				t.Fatalf("seed=%d round=%d retain: %v", seed, round, err)
			}
		}

		checkpoint := state.Begin()
		draftLen := 1 + rng.Intn(maxDraft)
		draftKV := make([][]byte, draftLen)
		draftRecurrent := make([][]byte, draftLen)
		draftTokens := make([]int, draftLen)
		for i := range draftKV {
			position := state.Cursor()
			draftTokens[i] = seededHybridTokenID(seed, round, i, position)
			draftKV[i], draftRecurrent[i] = seededHybridToken(seed, round, i, position)
			if err := state.Append(draftKV[i], draftRecurrent[i]); err != nil {
				t.Fatalf("seed=%d round=%d append=%d: %v", seed, round, i, err)
			}
		}

		accepted := rng.Intn(draftLen + 1)
		if round%17 == 0 {
			accepted = 0
		} else if round%19 == 0 {
			accepted = draftLen
		}
		if err := state.Commit(checkpoint, accepted); err != nil {
			t.Fatalf("seed=%d round=%d commit=%d/%d: %v", seed, round, accepted, draftLen, err)
		}
		targetKV = append(targetKV, cloneHybridState(draftKV[:accepted])...)
		targetRecurrent = append(targetRecurrent, cloneHybridState(draftRecurrent[:accepted])...)
		targetTokens = append(targetTokens, draftTokens[:accepted]...)
		speculativeTokens = append(speculativeTokens, draftTokens[:accepted]...)
		acceptedTotal += accepted

		if len(targetKV) > contextWindow {
			targetKV = retainHybridTail(targetKV, contextWindow)
			targetRecurrent = retainHybridTail(targetRecurrent, contextWindow)
			if err := state.RetainLast(contextWindow); err != nil {
				t.Fatalf("seed=%d round=%d context retain: %v", seed, round, err)
			}
		}
		assertHybridSpecState(t, state, acceptedTotal, targetKV, targetRecurrent)
		if !reflect.DeepEqual(speculativeTokens, targetTokens) {
			t.Fatalf("seed=%d round=%d speculative tokens diverged from target-only tokens", seed, round)
		}
		if got := len(state.KVState()); got > contextWindow {
			t.Fatalf("seed=%d round=%d resident state = %d, want <= %d", seed, round, got, contextWindow)
		}
		if cap(state.kv) > contextWindow || cap(state.recurrent) > contextWindow {
			t.Fatalf("seed=%d round=%d resident capacity KV=%d recurrent=%d, want <= %d", seed, round, cap(state.kv), cap(state.recurrent), contextWindow)
		}
	}
}

func TestHybridSpecStateLongHorizonCancellationErrorsAndSessionReuse(t *testing.T) {
	const (
		seed          = int64(0x9979)
		rounds        = 3072
		contextWindow = 193
	)

	rng := rand.New(rand.NewSource(seed))
	state, err := NewHybridSpecState(nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	targetKV := [][]byte(nil)
	targetRecurrent := [][]byte(nil)
	targetTokens := make([]int, 0, rounds*2)
	speculativeTokens := make([]int, 0, rounds*2)
	acceptedTotal := 0

	for round := 0; round < rounds; round++ {
		checkpoint := state.Begin()
		draftLen := 1 + rng.Intn(5)
		draftKV := make([][]byte, draftLen)
		draftRecurrent := make([][]byte, draftLen)
		draftTokens := make([]int, draftLen)
		for i := 0; i < draftLen; i++ {
			draftTokens[i] = seededHybridTokenID(seed, round, i, state.Cursor())
			draftKV[i], draftRecurrent[i] = seededHybridToken(seed, round, i, state.Cursor())
			if err := state.Append(draftKV[i], draftRecurrent[i]); err != nil {
				t.Fatalf("seed=%d round=%d append=%d: %v", seed, round, i, err)
			}
		}

		switch {
		case round%29 == 0:
			ctx, cancel := context.WithCancel(context.Background())
			cancel()
			if !errors.Is(ctx.Err(), context.Canceled) {
				t.Fatalf("seed=%d round=%d cancellation = %v", seed, round, ctx.Err())
			}
			if err := state.Rollback(checkpoint); err != nil {
				t.Fatalf("seed=%d round=%d cancellation rollback: %v", seed, round, err)
			}
		case round%31 == 0:
			injected := fmt.Errorf("seed=%d round=%d: %w", seed, round, errInjectedSpecFailure)
			if !errors.Is(injected, errInjectedSpecFailure) {
				t.Fatalf("seed=%d round=%d injected error lost identity", seed, round)
			}
			if err := state.Rollback(checkpoint); err != nil {
				t.Fatalf("seed=%d round=%d error rollback: %v", seed, round, err)
			}
		default:
			accepted := rng.Intn(draftLen + 1)
			if err := state.Commit(checkpoint, accepted); err != nil {
				t.Fatalf("seed=%d round=%d commit=%d/%d: %v", seed, round, accepted, draftLen, err)
			}
			targetKV = append(targetKV, cloneHybridState(draftKV[:accepted])...)
			targetRecurrent = append(targetRecurrent, cloneHybridState(draftRecurrent[:accepted])...)
			targetTokens = append(targetTokens, draftTokens[:accepted]...)
			speculativeTokens = append(speculativeTokens, draftTokens[:accepted]...)
			acceptedTotal += accepted
		}

		if len(targetKV) > contextWindow {
			targetKV = retainHybridTail(targetKV, contextWindow)
			targetRecurrent = retainHybridTail(targetRecurrent, contextWindow)
			if err := state.RetainLast(contextWindow); err != nil {
				t.Fatalf("seed=%d round=%d retain: %v", seed, round, err)
			}
		}
		assertHybridSpecState(t, state, acceptedTotal, targetKV, targetRecurrent)
		if !reflect.DeepEqual(speculativeTokens, targetTokens) {
			t.Fatalf("seed=%d round=%d speculative tokens diverged after rollback/reuse", seed, round)
		}
	}
}

func TestHybridSpecStateRetainLastBoundaries(t *testing.T) {
	state, err := NewHybridSpecState(
		[][]byte{{1}, {2}, {3}, {4}},
		[][]byte{{11}, {12}, {13}, {14}},
	)
	if err != nil {
		t.Fatal(err)
	}
	if err := state.RetainLast(2); err != nil {
		t.Fatal(err)
	}
	assertHybridSpecState(t, state, 4, [][]byte{{3}, {4}}, [][]byte{{13}, {14}})

	checkpoint := state.Begin()
	if err := state.RetainLast(1); err == nil {
		t.Fatal("retained state during active checkpoint")
	}
	if err := state.Append([]byte{5}, []byte{15}); err != nil {
		t.Fatal(err)
	}
	if err := state.Commit(checkpoint, 1); err != nil {
		t.Fatal(err)
	}
	assertHybridSpecState(t, state, 5, [][]byte{{3}, {4}, {5}}, [][]byte{{13}, {14}, {15}})

	if err := state.RetainLast(0); err != nil {
		t.Fatal(err)
	}
	assertHybridSpecState(t, state, 5, [][]byte{}, [][]byte{})
	if err := state.RetainLast(-1); err == nil {
		t.Fatal("negative retain succeeded")
	}
}

var errInjectedSpecFailure = errors.New("injected speculative failure")

func seededHybridTokenID(seed int64, round, draft, position int) int {
	return int((uint64(seed) ^ uint64(round+1)*131 ^ uint64(draft+1)*17 ^ uint64(position+1)*257) % 32000)
}

func seededHybridToken(seed int64, round, draft, position int) ([]byte, []byte) {
	value := uint64(seed) ^ uint64(round+1)*0x9e3779b97f4a7c15 ^ uint64(draft+1)*0xbf58476d1ce4e5b9 ^ uint64(position+1)*0x94d049bb133111eb
	kv := make([]byte, 16)
	recurrent := make([]byte, 12)
	binary.LittleEndian.PutUint64(kv, value)
	binary.LittleEndian.PutUint64(kv[8:], bits.RotateLeft64(value, 23))
	binary.LittleEndian.PutUint64(recurrent, ^value)
	binary.LittleEndian.PutUint32(recurrent[8:], uint32(value>>17))
	return kv, recurrent
}

func retainHybridTail(state [][]byte, keep int) [][]byte {
	if keep >= len(state) {
		return state
	}
	return cloneHybridState(state[len(state)-keep:])
}
