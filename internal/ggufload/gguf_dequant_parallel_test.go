package ggufload

import (
	"encoding/binary"
	"math"
	"math/rand"
	"runtime"
	"testing"
)

func TestKQuantDequantParallelMatchesScalar(t *testing.T) {
	oldProcs := runtime.GOMAXPROCS(4)
	t.Cleanup(func() { runtime.GOMAXPROCS(oldProcs) })

	blocks := dequantParallelMinBlocks + dequantParallelBlocksPerWorker
	if workers := dequantParallelWorkers(blocks); workers < 2 {
		t.Fatalf("parallel dequant test did not force the parallel branch: workers=%d", workers)
	}

	for _, tc := range []struct {
		name       string
		blockBytes int
		dequant    func([]float32, []byte)
		scalar     func([]float32, []byte)
		fixRaw     func([]byte)
	}{
		{
			name:       "Q2_K",
			blockBytes: blockQ2KBytes,
			dequant:    dequantQ2K,
			scalar:     dequantQ2KScalar,
			fixRaw: func(raw []byte) {
				for b := 0; b < len(raw)/blockQ2KBytes; b++ {
					base := b * blockQ2KBytes
					putFiniteF16ParallelTest(raw, base+qkK/16+qkK/4, b)
					putFiniteF16ParallelTest(raw, base+qkK/16+qkK/4+2, b+1)
				}
			},
		},
		{
			name:       "Q3_K",
			blockBytes: blockQ3KBytes,
			dequant:    dequantQ3K,
			scalar:     dequantQ3KScalar,
			fixRaw: func(raw []byte) {
				for b := 0; b < len(raw)/blockQ3KBytes; b++ {
					putFiniteF16ParallelTest(raw, b*blockQ3KBytes+blockQ3KBytes-2, b)
				}
			},
		},
		{
			name:       "Q4_K",
			blockBytes: blockQ4KBytes,
			dequant:    dequantQ4K,
			scalar:     dequantQ4KScalar,
			fixRaw: func(raw []byte) {
				for b := 0; b < len(raw)/blockQ4KBytes; b++ {
					base := b * blockQ4KBytes
					putFiniteF16ParallelTest(raw, base, b)
					putFiniteF16ParallelTest(raw, base+2, b+1)
				}
			},
		},
		{
			name:       "Q5_K",
			blockBytes: blockQ5KBytes,
			dequant:    dequantQ5K,
			scalar:     dequantQ5KScalar,
			fixRaw: func(raw []byte) {
				for b := 0; b < len(raw)/blockQ5KBytes; b++ {
					base := b * blockQ5KBytes
					putFiniteF16ParallelTest(raw, base, b)
					putFiniteF16ParallelTest(raw, base+2, b+1)
				}
			},
		},
		{
			name:       "Q6_K",
			blockBytes: blockQ6KBytes,
			dequant:    dequantQ6K,
			scalar:     dequantQ6KScalar,
			fixRaw: func(raw []byte) {
				for b := 0; b < len(raw)/blockQ6KBytes; b++ {
					putFiniteF16ParallelTest(raw, b*blockQ6KBytes+blockQ6KBytes-2, b)
				}
			},
		},
		{
			name:       "IQ4_XS",
			blockBytes: blockIQ4XSBytes,
			dequant:    dequantIQ4XS,
			scalar:     dequantIQ4XSScalar,
			fixRaw: func(raw []byte) {
				for b := 0; b < len(raw)/blockIQ4XSBytes; b++ {
					putFiniteF16ParallelTest(raw, b*blockIQ4XSBytes, b)
				}
			},
		},
		{
			name:       "IQ3_XXS",
			blockBytes: blockIQ3XXSBytes,
			dequant:    dequantIQ3XXS,
			scalar:     dequantIQ3XXSScalar,
			fixRaw: func(raw []byte) {
				for b := 0; b < len(raw)/blockIQ3XXSBytes; b++ {
					putFiniteF16ParallelTest(raw, b*blockIQ3XXSBytes, b)
				}
			},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			raw := randomParallelDequantRaw(blocks, tc.blockBytes)
			tc.fixRaw(raw)

			want := make([]float32, blocks*qkK)
			got := make([]float32, blocks*qkK)
			tc.scalar(want, raw)
			tc.dequant(got, raw)
			assertF32BitsEqualParallelTest(t, tc.name, got, want)
		})
	}
}

func TestDequantParallelWorkersUseGOMAXPROCSBudget(t *testing.T) {
	oldProcs := runtime.GOMAXPROCS(8)
	t.Cleanup(func() { runtime.GOMAXPROCS(oldProcs) })

	blocks := dequantParallelMinBlocks + 7*dequantParallelBlocksPerWorker
	runtime.GOMAXPROCS(2)
	if got := dequantParallelWorkers(blocks); got != 2 {
		t.Fatalf("dequantParallelWorkers(%d) with GOMAXPROCS=2 = %d, want 2", blocks, got)
	}

	runtime.GOMAXPROCS(1)
	if got := dequantParallelWorkers(blocks); got != 1 {
		t.Fatalf("dequantParallelWorkers(%d) with GOMAXPROCS=1 = %d, want serial", blocks, got)
	}
}

func randomParallelDequantRaw(blocks, blockBytes int) []byte {
	raw := make([]byte, blocks*blockBytes)
	rng := rand.New(rand.NewSource(1102))
	if _, err := rng.Read(raw); err != nil {
		panic(err)
	}
	return raw
}

func putFiniteF16ParallelTest(raw []byte, off, i int) {
	vals := [...]uint16{
		0x0000, // 0
		0x3400, // 0.25
		0x3800, // 0.5
		0x3c00, // 1
		0x4000, // 2
		0xbc00, // -1
		0xb800, // -0.5
	}
	binary.LittleEndian.PutUint16(raw[off:], vals[i%len(vals)])
}

func assertF32BitsEqualParallelTest(t *testing.T, label string, got, want []float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: got %d values, want %d", label, len(got), len(want))
	}
	for i := range want {
		if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
			t.Fatalf("%s[%d]=%v bits=%#x, want %v bits=%#x",
				label, i, got[i], math.Float32bits(got[i]), want[i], math.Float32bits(want[i]))
		}
	}
}
