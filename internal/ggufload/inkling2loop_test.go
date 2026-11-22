package ggufload

import (
	"errors"
	"os"
	"testing"
	"time"
)

func TestInkling2NVFP4DequantMatchesGGMLBlockLayout(t *testing.T) {
	blk := make([]byte, blockNVFP4Bytes)
	// sub-block scales: UE4M3 0x38 = 1.0 (halved to 0.5), 0x40 = 2.0 (1.0), zero, 0x7F (NaN reads 0)
	blk[0], blk[1], blk[2], blk[3] = 0x38, 0x40, 0x00, 0x7F
	// first byte of sub-block 0: low nibble 7 (12), high nibble 9 (-1)
	blk[4] = 0x97
	// first byte of sub-block 1: low nibble 5 (6), high nibble 0xF (-12)
	blk[4+8] = 0xF5
	out := make([]float32, qkNVFP4)
	dequantNVFP4(out, blk)
	want := map[int]float32{0: 6, 8: -0.5, 16: 6, 24: -12}
	for i, w := range want {
		if out[i] != w {
			t.Fatalf("out[%d] = %v, want %v (all %v)", i, out[i], w, out[:32])
		}
	}
	for i := 32; i < 64; i++ {
		if out[i] != 0 {
			t.Fatalf("zero/NaN-scaled sub-block value out[%d] = %v", i, out[i])
		}
	}
	if ue4m3ToF32(0x01) != float32(1.0/512*0.5) {
		t.Fatalf("subnormal UE4M3 = %v", ue4m3ToF32(0x01))
	}
}

func TestInkling2UnsupportedTypeRefuses(t *testing.T) {
	_, err := inkling2RowBytes(TensorQ2_0, 256)
	var typed *UnsupportedInkling2LoopTensorTypeError
	if !errors.As(err, &typed) {
		t.Fatalf("Q2_0 row: err = %v, want typed refusal", err)
	}
	if n, err := inkling2RowBytes(TensorNVFP4, 2304); err != nil || n != 2304/64*36 {
		t.Fatalf("NVFP4 row bytes = %d, %v", n, err)
	}
}

// The oracle is llama.cpp ad50351f5 llama-server greedy (temperature 0, top_k 1,
// CPU backend, -fa off) on Inkling-2-Mini-UD-Q3_K_XL.gguf (mainline layout,
// sha256 e2d03e8c...), each prompt tokenized without BOS.
var inkling2OracleCases = []struct {
	name           string
	prompt, greedy []int
}{
	{
		// "The capital of France is Paris. The capital of Germany is Berlin. The capital of Japan is"
		name:   "capitals",
		prompt: []int{371, 151451, 149879, 151194, 149884, 152717, 13, 149917, 151451, 149879, 153067, 149884, 154328, 13, 149917, 151451, 149879, 153630, 149884},
		greedy: []int{182416, 13, 149917, 151451, 149879, 149876, 151465, 151480, 149884, 156229, 11, 149950, 13, 34, 13, 149917, 151451, 149879, 149876, 151465, 157422, 149884, 153629, 13, 149917, 151451, 149879, 165985, 149884, 169305, 13, 149917},
	},
	{
		// a Python def line followed by a one-line docstring and a newline
		name:   "fibonacci",
		prompt: []int{1719, 220, 46389, 7, 77, 8, 25, 198, 149862, 172706, 72358, 149876, 149934, 12, 258, 181713, 150107, 13, 70646, 198},
		greedy: []int{149862, 150003, 149934, 197012, 220, 16, 25, 198, 149860, 149862, 150583, 149934, 198, 149862, 149878, 11, 149963, 196977, 220, 15, 11, 220, 16, 198, 149862, 149887, 197001, 149881, 150354, 7, 17, 11},
	},
}

func TestInkling2OracleGreedyMatchesLlamaCpp(t *testing.T) {
	path := os.Getenv("FAK_INKLING2_GGUF")
	if path == "" {
		t.Skip("FAK_INKLING2_GGUF not set (Inkling-2-Mini-UD-Q3_K_XL.gguf mainline layout)")
	}
	m, err := OpenInkling2Loop(path)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	for _, c := range inkling2OracleCases {
		t.Run(c.name, func(t *testing.T) {
			start := time.Now()
			got, err := Inkling2LoopGreedy(m, c.prompt, len(c.greedy))
			if err != nil {
				t.Fatal(err)
			}
			t.Logf("host reference: %d prompt + %d greedy tokens in %s", len(c.prompt), len(got), time.Since(start))
			for i := range got {
				if got[i] != c.greedy[i] {
					t.Fatalf("greedy token %d = %d, llama.cpp = %d (got %v)", i, got[i], c.greedy[i], got)
				}
			}
		})
	}
}
