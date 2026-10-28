package ggufload

import (
	"errors"
	"math"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/model"
)

// fak-test:justify why=regression when=changed:internal/ggufload/**
// fak-test:runtime fast est=100ms lane=default
func TestDeepSeek41EngramGeometryOverflowRefuses(t *testing.T) {
	if _, err := (&File{Metadata: vcruzQ2KReceiptMeta()}).Config(); err != nil {
		t.Fatalf("valid metadata control: %v", err)
	}
	for _, tc := range []struct {
		name         string
		ngram, heads uint64
	}{
		// Four layers times (ngram-1) wraps to zero on either Go int width.
		{"inference denominator", uint64(math.MaxInt)/2 + 2, 8},
		// Four layers, two n-gram widths and huge heads wrap back to 48 primes.
		{"validation cardinality", 3, uint64(math.MaxInt)/4 + 7},
	} {
		t.Run(tc.name, func(t *testing.T) {
			defer func() {
				if p := recover(); p != nil {
					t.Errorf("Config panicked on untrusted Engram geometry: %v", p)
				}
			}()
			meta := vcruzQ2KReceiptMeta()
			meta["deepseek41.engram.layer_ids"] = intArrayValue(TypeUint32, []uint64{1, 2, 3, 14})
			meta["deepseek41.engram.max_ngram_size"] = Value{Type: TypeUint64, Value: tc.ngram}
			meta["deepseek41.engram.head_count"] = Value{Type: TypeUint64, Value: tc.heads}
			if _, err := (&File{Metadata: meta}).Config(); !errors.Is(err, model.ErrV41ConfigAdmission) {
				t.Fatalf("Config error = %v, want typed geometry refusal", err)
			}
		})
	}
}
