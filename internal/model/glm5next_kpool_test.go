package model

import (
	"math"
	"testing"
)

func TestDownsampleGLM5NextDSAKeys(t *testing.T) {
	const dim = 4
	const stride = 4

	t.Run("exact 4x reduction", func(t *testing.T) {
		const seqLen = 8
		keys := make([]float32, seqLen*dim)
		// Block 0 (tokens 0..3) has value 2.0
		for t := 0; t < 4; t++ {
			for d := 0; d < dim; d++ {
				keys[t*dim+d] = 2.0
			}
		}
		// Block 1 (tokens 4..7) has value 6.0
		for t := 4; t < 8; t++ {
			for d := 0; d < dim; d++ {
				keys[t*dim+d] = 6.0
			}
		}

		pooled := DownsampleGLM5NextDSAKeys(keys, seqLen, dim, stride)
		if len(pooled) != 2*dim {
			t.Fatalf("len(pooled) = %d, want %d", len(pooled), 2*dim)
		}

		// Block 0 mean = 2.0, Block 1 mean = 6.0
		for d := 0; d < dim; d++ {
			if math.Abs(float64(pooled[d]-2.0)) > 1e-6 {
				t.Fatalf("block 0 pooled[%d] = %g, want 2.0", d, pooled[d])
			}
			if math.Abs(float64(pooled[dim+d]-6.0)) > 1e-6 {
				t.Fatalf("block 1 pooled[%d] = %g, want 6.0", d, pooled[dim+d])
			}
		}
	})

	t.Run("partial trailing block", func(t *testing.T) {
		const seqLen = 7 // block 0: 4 tokens (all 4.0), block 1: 3 tokens (1.0, 2.0, 3.0 -> mean 2.0)
		keys := make([]float32, seqLen*dim)
		for t := 0; t < 4; t++ {
			for d := 0; d < dim; d++ {
				keys[t*dim+d] = 4.0
			}
		}
		// Tokens 4, 5, 6
		for d := 0; d < dim; d++ {
			keys[4*dim+d] = 1.0
			keys[5*dim+d] = 2.0
			keys[6*dim+d] = 3.0
		}

		pooled := DownsampleGLM5NextDSAKeys(keys, seqLen, dim, stride)
		if len(pooled) != 2*dim {
			t.Fatalf("len(pooled) = %d, want %d", len(pooled), 2*dim)
		}

		for d := 0; d < dim; d++ {
			if math.Abs(float64(pooled[d]-4.0)) > 1e-6 {
				t.Fatalf("block 0 mean = %g, want 4.0", pooled[d])
			}
			if math.Abs(float64(pooled[dim+d]-2.0)) > 1e-6 {
				t.Fatalf("block 1 mean = %g, want 2.0", pooled[dim+d])
			}
		}
	})

	t.Run("empty input", func(t *testing.T) {
		if res := DownsampleGLM5NextDSAKeys(nil, 0, dim, stride); res != nil {
			t.Fatalf("expected nil on empty input, got %v", res)
		}
	})
}
