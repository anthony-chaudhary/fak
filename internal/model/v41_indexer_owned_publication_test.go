package model

import (
	"reflect"
	"testing"
)

type v41OwnedIndexFixture struct {
	q, keys, weights                []float32
	blocks, blockSize, topK, offset int
}

func newV41OwnedIndexFixture() v41OwnedIndexFixture {
	// Rows 0/1 and 4/5 tie. The newest block must win even though its
	// scores are below block 0; masked padding then uses the lower row.
	return v41OwnedIndexFixture{
		q: []float32{1}, keys: []float32{4, 4, 1, 1, 2, 2}, weights: []float32{1},
		blocks: 1, blockSize: 2, topK: 3, offset: 7,
	}
}

func (f v41OwnedIndexFixture) publication() (*V41IndexerPublication, error) {
	return NewV41IndexerPublication(0, f.q, f.keys, f.weights, 1, 1,
		len(f.keys), f.blocks, f.blockSize, f.topK, f.offset)
}

// This independently assembles only the storage required by the public
// publication contract, using the same score/select arithmetic. It does not
// call NewV41IndexerPublication or add artificial copies to its control.
func (f v41OwnedIndexFixture) control() (*V41IndexerPublication, error) {
	score, err := V41IndexerScore(f.q, f.keys, f.weights, 1, 1, len(f.keys))
	if err != nil {
		return nil, err
	}
	var mask []bool
	if f.blocks > 0 {
		mask, err = V41SelectCandidateBlocks(score, len(f.keys), f.blocks, f.blockSize)
		if err != nil {
			return nil, err
		}
	}
	rows, err := V41SelectIndexRows(score, len(f.keys), f.topK, f.offset, mask)
	if err != nil {
		return nil, err
	}
	return &V41IndexerPublication{layer: 0, compressLen: len(f.keys), candidates: mask, rows: rows}, nil
}

func v41OwnedIndexEqual(t *testing.T, p *V41IndexerPublication, mask []bool, rows []int32) {
	t.Helper()
	if p == nil || !reflect.DeepEqual(p.Candidates(), mask) || !reflect.DeepEqual(p.Rows(), rows) {
		t.Fatalf("publication=%+v, want mask=%v rows=%v", p, mask, rows)
	}
}

// Runtime is a provisional estimate. The real host index consumer is exercised;
// no selected device operation or full-model throughput is measured.
// fak-test:runtime fast est=20ms lane=default
func TestV41IndexerOwnedPublication(t *testing.T) {
	t.Run("ownership", func(t *testing.T) {
		f := newV41OwnedIndexFixture()
		wantMask := []bool{false, false, false, false, true, true}
		wantRows := []int32{-1, 11, 12}
		pub, err := f.publication()
		if err != nil {
			t.Fatal(err)
		}
		v41OwnedIndexEqual(t, pub, wantMask, wantRows)
		readerMask, readerRows := pub.Candidates(), pub.Rows()
		readerMask[0], readerRows[0] = true, 100
		v41OwnedIndexEqual(t, pub, wantMask, wantRows)
		f.q[0], f.keys[0], f.weights[0] = 9, 9, 9
		v41OwnedIndexEqual(t, pub, wantMask, wantRows)

		// With no block mask, equal row scores select the lower row 4 over 5.
		f = newV41OwnedIndexFixture()
		f.blocks = 0
		plain, err := f.publication()
		if err != nil {
			t.Fatal(err)
		}
		v41OwnedIndexEqual(t, plain, nil, []int32{7, 8, 11})
		// Keeping two blocks selects tied block 0 before block 1, alongside
		// the pinned newest block. Ranking and final position order stay exact.
		f.blocks = 2
		f.keys[2], f.keys[3] = 4, 4
		tied, err := f.publication()
		if err != nil {
			t.Fatal(err)
		}
		v41OwnedIndexEqual(t, tied, []bool{true, true, false, false, true, true}, []int32{7, 8, 11})

		// Preserve observable nil versus allocated-empty getter results.
		f = newV41OwnedIndexFixture()
		f.keys = nil
		empty, err := f.publication()
		if err != nil {
			t.Fatal(err)
		}
		v41OwnedIndexEqual(t, empty, []bool{}, []int32{})
		f.blocks = 0
		emptyPlain, err := f.publication()
		if err != nil {
			t.Fatal(err)
		}
		v41OwnedIndexEqual(t, emptyPlain, nil, []int32{})
		f = newV41OwnedIndexFixture()
		f.topK = 0
		zeroK, err := f.publication()
		if err != nil {
			t.Fatal(err)
		}
		v41OwnedIndexEqual(t, zeroK, wantMask, []int32{})
	})

	t.Run("allocation", func(t *testing.T) {
		f := newV41OwnedIndexFixture()
		inputQ, inputKeys, inputWeights := append([]float32(nil), f.q...), append([]float32(nil), f.keys...), append([]float32(nil), f.weights...)
		got, gotErr := f.publication()
		want, wantErr := f.control()
		if gotErr != nil || wantErr != nil {
			t.Fatalf("publication/control errors=%v/%v", gotErr, wantErr)
		}
		mask, rows := []bool{false, false, false, false, true, true}, []int32{-1, 11, 12}
		v41OwnedIndexEqual(t, got, mask, rows)
		v41OwnedIndexEqual(t, want, mask, rows)
		gotAllocs := testing.AllocsPerRun(100, func() { got, gotErr = f.publication() })
		wantAllocs := testing.AllocsPerRun(100, func() { want, wantErr = f.control() })
		if gotErr != nil || wantErr != nil {
			t.Fatalf("sampled publication/control errors=%v/%v", gotErr, wantErr)
		}
		v41OwnedIndexEqual(t, got, mask, rows)
		v41OwnedIndexEqual(t, want, mask, rows)
		if !reflect.DeepEqual(inputQ, f.q) || !reflect.DeepEqual(inputKeys, f.keys) ||
			!reflect.DeepEqual(inputWeights, f.weights) {
			t.Fatal("publication sampling changed borrowed inputs")
		}
		if gotAllocs > wantAllocs {
			t.Fatalf("redundant index publication copies: publication=%g fresh_control=%g rows=%v mask=%v inputs_unchanged=true",
				gotAllocs, wantAllocs, got.Rows(), got.Candidates())
		}
	})

	t.Run("host-consumer", func(t *testing.T) {
		f := newV41OwnedIndexFixture()
		inputQ, inputKeys, inputWeights := append([]float32(nil), f.q...), append([]float32(nil), f.keys...), append([]float32(nil), f.weights...)
		healthCalls := 0
		health := func(layer int) error {
			if layer != 0 {
				t.Fatal("host index consumer changed the layer")
			}
			healthCalls++
			return nil
		}
		read := func() ([]int32, error) {
			return v41IndexRowsSelected(0, f.q, f.keys, f.weights, 1, 1, len(f.keys),
				f.blocks, f.blockSize, f.topK, nil, health, nil)
		}
		want := []int32{-1, 4, 5}
		for invocation := 0; invocation < 2; invocation++ {
			got, err := read()
			if err != nil || !reflect.DeepEqual(got, want) {
				t.Fatalf("host index consumer got=%v want=%v err=%v", got, want, err)
			}
			if healthCalls != invocation+1 {
				t.Fatalf("health calls=%d, want one per invocation", healthCalls)
			}
			got[0] = 100 // The next actual consumer invocation must be independent.
		}
		if !reflect.DeepEqual(inputQ, f.q) || !reflect.DeepEqual(inputKeys, f.keys) ||
			!reflect.DeepEqual(inputWeights, f.weights) {
			t.Fatal("host index consumer changed borrowed inputs")
		}
	})
}
