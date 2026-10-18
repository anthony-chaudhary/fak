package model

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"math"
	"testing"
)

func dflash2EmbeddingReuseFixture(scale float64) *dflash2Reference {
	const hidden, vocab = 4, 5
	raw := make([]byte, hidden*vocab*4)
	for i := 0; i < hidden*vocab; i++ {
		value := float32(i-9) / 8
		binary.LittleEndian.PutUint32(raw[i*4:], math.Float32bits(value))
	}
	m := &Model{
		Cfg: Config{HiddenSize: hidden, VocabSize: vocab, EmbedScale: scale},
		manifest: map[string]tensorMeta{
			"model.embed_tokens.weight": {Dtype: "F32", Shape: []int{vocab, hidden}, Nbytes: len(raw)},
		},
		raw: raw,
	}
	return &dflash2Reference{
		target: &Session{M: m},
		g:      dflash2Geometry{hidden: hidden, vocab: vocab, mask: 0},
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestDFlash2InputEmbeddingRowsMatchScalar(t *testing.T) {
	for _, scale := range []float64{0, 1, 1.5} {
		for _, proposals := range []int{0, 1, 7} {
			for _, anchor := range []int{0, 3} {
				t.Run(fmt.Sprintf("scale%g/proposals%d/anchor%d", scale, proposals, anchor), func(t *testing.T) {
					f := dflash2EmbeddingReuseFixture(scale)
					before := append([]byte(nil), f.target.M.raw...)
					got, err := f.inputEmbeddings(anchor, proposals)
					if err != nil {
						t.Fatal(err)
					}
					h := f.g.hidden
					if len(got) != (proposals+1)*h {
						t.Fatalf("rows=%d, requested elements=%d", len(got), (proposals+1)*h)
					}
					// Independent raw-row oracle retains the original per-position
					// multiplication, including the identity-scale branch.
					for row := 0; row <= proposals; row++ {
						token := f.g.mask
						if row == 0 {
							token = anchor
						}
						for k := 0; k < h; k++ {
							offset := (token*h + k) * 4
							want := math.Float32frombits(binary.LittleEndian.Uint32(before[offset:]))
							if scale != 0 && scale != 1 {
								want *= float32(scale)
							}
							if math.Float32bits(got[row*h+k]) != math.Float32bits(want) {
								t.Fatalf("row%d col%d got=%g want=%g", row, k, got[row*h+k], want)
							}
						}
					}
					if !bytes.Equal(before, f.target.M.raw) {
						t.Fatal("lookup or scaling modified shared target weights")
					}
				})
			}
		}
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestDFlash2InputEmbeddingRowsAreIndependent(t *testing.T) {
	f := dflash2EmbeddingReuseFixture(1.5)
	const proposals = 7
	first, err := f.inputEmbeddings(3, proposals)
	if err != nil {
		t.Fatal(err)
	}
	original := append([]float32(nil), first...)
	h := f.g.hidden
	first[h] = 12345
	for row := 0; row <= proposals; row++ {
		if row != 1 && first[row*h] != original[row*h] {
			t.Fatalf("mutating one mask row changed row%d", row)
		}
	}
	next, err := f.inputEmbeddings(3, proposals)
	if err != nil {
		t.Fatal(err)
	}
	for i, want := range original {
		if math.Float32bits(next[i]) != math.Float32bits(want) {
			t.Fatalf("next invocation retained prior mutation at%d", i)
		}
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestDFlash2InputEmbeddingFailureOrder(t *testing.T) {
	f := dflash2EmbeddingReuseFixture(1)
	f.g.mask = f.g.vocab
	got, err := f.inputEmbeddings(-1, 7)
	if got != nil || err == nil || err.Error() != "model: token embedding id -1 is out of range" {
		t.Fatalf("anchor failure lost priority: got=%v err=%v", got, err)
	}
	got, err = f.inputEmbeddings(3, 7)
	want := fmt.Sprintf("model: token embedding id %d is out of range", f.g.mask)
	if got != nil || err == nil || err.Error() != want {
		t.Fatalf("mask failure published partial rows: got=%v err=%v", got, err)
	}
	// No draft positions means no mask access, as in the original loop.
	got, err = f.inputEmbeddings(3, 0)
	if err != nil || len(got) != f.g.hidden {
		t.Fatalf("anchor-only lookup touched invalid mask: rows=%d err=%v", len(got), err)
	}
}
