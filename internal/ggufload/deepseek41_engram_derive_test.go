package ggufload

import (
	"errors"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/model"
)

func v41RealHeaderEngram(t *testing.T) *File {
	t.Helper()
	f := loadV41HeaderFixture(t)
	if _, err := f.Config(); err != nil {
		t.Fatalf("Config: %v", err)
	}
	if f.DeepSeek41Engram == nil {
		t.Fatal("real header declares no Engram")
	}
	return f
}

// The staged vcruz Q2_K header's converter-supplied primes, pad id and
// multipliers equal the DeepInfer derivation, so the attach-time check admits
// the served artifact.
// fak-test:runtime medium est=1s lane=default
func TestDeepSeek41RealHeaderEngramPrimesMatchDerivation(t *testing.T) {
	f := v41RealHeaderEngram(t)
	layout, err := buildV41EngramLayout(f.DeepSeek41Engram)
	if err != nil {
		t.Fatal(err)
	}
	if err := model.ValidateV41PublishedEngramLayout(f.DeepSeek41Engram.LayerIDs, layout); err != nil {
		t.Fatalf("real header refused by the derivation: %v", err)
	}
	if len(layout.Rows) != 2 || layout.Rows[0] != 384006168 || layout.Rows[1] != 384016682 {
		t.Fatalf("rows = %v, want [384006168 384016682]", layout.Rows)
	}
}

// A sum-preserving head-major transposition of one layer's primes is refused at
// the serve attach seam before any table is opened; the unpermuted header gets
// past the check and stops only at the absent table bytes.
// fak-test:runtime medium est=1s lane=default
func TestDeepSeek41EngramServeRefusesPermutedPrimes(t *testing.T) {
	f := v41RealHeaderEngram(t)
	err := attachV41EngramServing(&model.Model{}, &WeightSource{File: f})
	if !errors.Is(err, model.ErrV41NativeUnsupported) || errors.Is(err, model.ErrV41EngramDerivation) {
		t.Fatalf("unpermuted header: err = %v, want table-open refusal without ErrV41EngramDerivation", err)
	}

	eng := f.DeepSeek41Engram
	heads, tables := eng.NHeads, eng.MaxNgramSize-1
	orig := append([]int(nil), eng.Primes[:heads*tables]...)
	for ng := 0; ng < tables; ng++ {
		for h := 0; h < heads; h++ {
			eng.Primes[h*tables+ng] = orig[ng*heads+h]
		}
	}
	err = attachV41EngramServing(&model.Model{}, &WeightSource{File: f})
	if !errors.Is(err, model.ErrV41EngramDerivation) || !errors.Is(err, model.ErrV41NativeUnsupported) {
		t.Fatalf("permuted primes: err = %v, want ErrV41EngramDerivation and ErrV41NativeUnsupported", err)
	}
}
