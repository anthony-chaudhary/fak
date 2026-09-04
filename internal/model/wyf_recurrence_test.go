package model

import (
	"math"
	"math/rand"
	"testing"
)

func generateDeterministicRecurrenceInputs(numTokens, headDim int, seed int64) (q, k, v, beta, gate []float32) {
	rng := rand.New(rand.NewSource(seed))
	totalElems := numTokens * headDim

	q = make([]float32, totalElems)
	k = make([]float32, totalElems)
	v = make([]float32, totalElems)
	beta = make([]float32, numTokens)
	gate = make([]float32, numTokens)

	for i := 0; i < totalElems; i++ {
		q[i] = (rng.Float32() - 0.5) * 0.1
		k[i] = (rng.Float32() - 0.5) * 0.1
		v[i] = (rng.Float32() - 0.5) * 0.5
	}
	for i := 0; i < numTokens; i++ {
		beta[i] = 0.2 + rng.Float32()*0.6   // Update gate in [0.2, 0.8]
		gate[i] = 0.92 + rng.Float32()*0.07 // Decay factor in [0.92, 0.99]
	}
	return q, k, v, beta, gate
}

func computeMaxDiff(a, b []float32) float32 {
	var maxDiff float32
	for i := range a {
		diff := float32(math.Abs(float64(a[i] - b[i])))
		if diff > maxDiff {
			maxDiff = diff
		}
	}
	return maxDiff
}

func computeCosineSimilarity(a, b []float32) float64 {
	var dot, normA, normB float64
	for i := range a {
		va := float64(a[i])
		vb := float64(b[i])
		dot += va * vb
		normA += va * va
		normB += vb * vb
	}
	if normA == 0 || normB == 0 {
		return 0.0
	}
	return dot / (math.Sqrt(normA) * math.Sqrt(normB))
}

func TestWyfParitySingleChunk(t *testing.T) {
	n := 32
	d := 128
	c := 32
	q, k, v, beta, gate := generateDeterministicRecurrenceInputs(n, d, 42)

	wyfOut, wyfState, err := WyfChunkwiseRecurrence(q, k, v, beta, gate, n, d, c, nil)
	if err != nil {
		t.Fatalf("WyfChunkwiseRecurrence failed: %v", err)
	}

	seqOut, seqState, err := SequentialGatedDeltaNet(q, k, v, beta, gate, n, d, nil)
	if err != nil {
		t.Fatalf("SequentialGatedDeltaNet failed: %v", err)
	}

	maxDiffOut := computeMaxDiff(wyfOut, seqOut)
	maxDiffState := computeMaxDiff(wyfState, seqState)
	cosSim := computeCosineSimilarity(wyfOut, seqOut)

	t.Logf("SingleChunk (N=%d, D=%d, C=%d): maxDiffOut=%.7e, maxDiffState=%.7e, cosSim=%.9f",
		n, d, c, maxDiffOut, maxDiffState, cosSim)

	const tol = 1e-4
	if maxDiffOut >= tol {
		t.Fatalf("Output parity violation: maxDiff=%.7e >= threshold %.1e", maxDiffOut, tol)
	}
	if maxDiffState >= tol {
		t.Fatalf("State parity violation: maxDiff=%.7e >= threshold %.1e", maxDiffState, tol)
	}
	if cosSim < 0.9999 {
		t.Fatalf("Cosine similarity %.9f fell below 0.9999", cosSim)
	}
}

func TestWyfParityMultiChunk(t *testing.T) {
	tokenLengths := []int{64, 128, 256, 512}
	d := 128
	c := 32

	for _, n := range tokenLengths {
		q, k, v, beta, gate := generateDeterministicRecurrenceInputs(n, d, int64(100+n))

		wyfOut, wyfState, err := WyfChunkwiseRecurrence(q, k, v, beta, gate, n, d, c, nil)
		if err != nil {
			t.Fatalf("WyfChunkwiseRecurrence failed for N=%d: %v", n, err)
		}

		seqOut, seqState, err := SequentialGatedDeltaNet(q, k, v, beta, gate, n, d, nil)
		if err != nil {
			t.Fatalf("SequentialGatedDeltaNet failed for N=%d: %v", n, err)
		}

		maxDiffOut := computeMaxDiff(wyfOut, seqOut)
		maxDiffState := computeMaxDiff(wyfState, seqState)
		cosSim := computeCosineSimilarity(wyfOut, seqOut)

		t.Logf("MultiChunk (N=%d, D=%d, C=%d): maxDiffOut=%.7e, maxDiffState=%.7e, cosSim=%.9f",
			n, d, c, maxDiffOut, maxDiffState, cosSim)

		const tol = 1e-4
		if maxDiffOut >= tol {
			t.Fatalf("N=%d: output parity violation: maxDiff=%.7e >= threshold %.1e", n, maxDiffOut, tol)
		}
		if maxDiffState >= tol {
			t.Fatalf("N=%d: state parity violation: maxDiff=%.7e >= threshold %.1e", n, maxDiffState, tol)
		}
		if cosSim < 0.9999 {
			t.Fatalf("N=%d: cosine similarity %.9f fell below 0.9999", n, cosSim)
		}
	}
}

func TestWyfParityNonMultipleChunk(t *testing.T) {
	// Tests sequences where N is not an exact multiple of C=32 (e.g. 77 tokens = 2 full chunks + 13 remainder)
	nonMultiples := []int{17, 45, 77, 101}
	d := 128
	c := 32

	for _, n := range nonMultiples {
		q, k, v, beta, gate := generateDeterministicRecurrenceInputs(n, d, int64(200+n))

		wyfOut, wyfState, err := WyfChunkwiseRecurrence(q, k, v, beta, gate, n, d, c, nil)
		if err != nil {
			t.Fatalf("WyfChunkwiseRecurrence failed for N=%d: %v", n, err)
		}

		seqOut, seqState, err := SequentialGatedDeltaNet(q, k, v, beta, gate, n, d, nil)
		if err != nil {
			t.Fatalf("SequentialGatedDeltaNet failed for N=%d: %v", n, err)
		}

		maxDiffOut := computeMaxDiff(wyfOut, seqOut)
		maxDiffState := computeMaxDiff(wyfState, seqState)

		t.Logf("NonMultipleChunk (N=%d, D=%d, C=%d): maxDiffOut=%.7e, maxDiffState=%.7e",
			n, d, c, maxDiffOut, maxDiffState)

		const tol = 1e-4
		if maxDiffOut >= tol {
			t.Fatalf("N=%d: output parity violation: maxDiff=%.7e >= threshold %.1e", n, maxDiffOut, tol)
		}
		if maxDiffState >= tol {
			t.Fatalf("N=%d: state parity violation: maxDiff=%.7e >= threshold %.1e", n, maxDiffState, tol)
		}
	}
}

func TestWyfParityNonZeroInitialState(t *testing.T) {
	n := 64
	d := 128
	c := 32
	q, k, v, beta, gate := generateDeterministicRecurrenceInputs(n, d, 999)

	// Seed non-zero initial state
	initState := make([]float32, d*d)
	rng := rand.New(rand.NewSource(12345))
	for i := range initState {
		initState[i] = (rng.Float32() - 0.5) * 0.05
	}

	wyfOut, wyfState, err := WyfChunkwiseRecurrence(q, k, v, beta, gate, n, d, c, initState)
	if err != nil {
		t.Fatalf("WyfChunkwiseRecurrence failed: %v", err)
	}

	seqOut, seqState, err := SequentialGatedDeltaNet(q, k, v, beta, gate, n, d, initState)
	if err != nil {
		t.Fatalf("SequentialGatedDeltaNet failed: %v", err)
	}

	maxDiffOut := computeMaxDiff(wyfOut, seqOut)
	maxDiffState := computeMaxDiff(wyfState, seqState)
	cosSim := computeCosineSimilarity(wyfOut, seqOut)

	t.Logf("NonZeroInitialState (N=%d, D=%d): maxDiffOut=%.7e, maxDiffState=%.7e, cosSim=%.9f",
		n, d, maxDiffOut, maxDiffState, cosSim)

	const tol = 1e-4
	if maxDiffOut >= tol {
		t.Fatalf("Output parity violation: maxDiff=%.7e >= threshold %.1e", maxDiffOut, tol)
	}
	if maxDiffState >= tol {
		t.Fatalf("State parity violation: maxDiff=%.7e >= threshold %.1e", maxDiffState, tol)
	}
	if cosSim < 0.9999 {
		t.Fatalf("Cosine similarity %.9f fell below 0.9999", cosSim)
	}
}

func TestWyfParityBatchedMultiHead(t *testing.T) {
	cfg := WyfRecurrenceConfig{
		BatchSize: 2,
		NumHeads:  4,
		SeqLen:    64,
		HeadDim:   128,
		ChunkSize: 32,
	}
	totalTokens := cfg.BatchSize * cfg.NumHeads * cfg.SeqLen
	q, k, v, beta, gate := generateDeterministicRecurrenceInputs(totalTokens, cfg.HeadDim, 777)

	wyfOut, wyfState, err := WyfChunkwiseRecurrenceConfig(q, k, v, beta, gate, cfg, nil)
	if err != nil {
		t.Fatalf("WyfChunkwiseRecurrenceConfig failed: %v", err)
	}

	seqOut, seqState, err := SequentialGatedDeltaNetConfig(q, k, v, beta, gate, cfg, nil)
	if err != nil {
		t.Fatalf("SequentialGatedDeltaNetConfig failed: %v", err)
	}

	maxDiffOut := computeMaxDiff(wyfOut, seqOut)
	maxDiffState := computeMaxDiff(wyfState, seqState)

	t.Logf("BatchedMultiHead (B=%d, H=%d, N=%d): maxDiffOut=%.7e, maxDiffState=%.7e",
		cfg.BatchSize, cfg.NumHeads, cfg.SeqLen, maxDiffOut, maxDiffState)

	const tol = 1e-4
	if maxDiffOut >= tol {
		t.Fatalf("Output parity violation: maxDiff=%.7e >= threshold %.1e", maxDiffOut, tol)
	}
	if maxDiffState >= tol {
		t.Fatalf("State parity violation: maxDiff=%.7e >= threshold %.1e", maxDiffState, tol)
	}
}

func TestWyfInputValidation(t *testing.T) {
	d := 128
	n := 32
	c := 32
	q, k, v, beta, gate := generateDeterministicRecurrenceInputs(n, d, 1)

	// Test empty beta/gate
	if _, _, err := WyfChunkwiseRecurrence(q, k, v, nil, gate, n, d, c, nil); err == nil {
		t.Fatal("Expected error on nil beta")
	}

	// Test truncated q
	if _, _, err := WyfChunkwiseRecurrence(q[:len(q)-1], k, v, beta, gate, n, d, c, nil); err == nil {
		t.Fatal("Expected error on truncated Q")
	}

	// Test negative sequence length
	if _, _, err := WyfChunkwiseRecurrence(q, k, v, beta, gate, -1, d, c, nil); err == nil {
		t.Fatal("Expected error on negative sequence length")
	}
}

// Benchmarks demonstrating chunkwise recurrence efficiency over sequential loop.

func BenchmarkWyfChunkwiseRecurrence_Seq512(b *testing.B) {
	n := 512
	d := 128
	c := 32
	q, k, v, beta, gate := generateDeterministicRecurrenceInputs(n, d, 512)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _, _ = WyfChunkwiseRecurrence(q, k, v, beta, gate, n, d, c, nil)
	}
}

func BenchmarkSequentialGatedDeltaNet_Seq512(b *testing.B) {
	n := 512
	d := 128
	q, k, v, beta, gate := generateDeterministicRecurrenceInputs(n, d, 512)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _, _ = SequentialGatedDeltaNet(q, k, v, beta, gate, n, d, nil)
	}
}

func BenchmarkWyfChunkwiseRecurrence_Seq1024(b *testing.B) {
	n := 1024
	d := 128
	c := 32
	q, k, v, beta, gate := generateDeterministicRecurrenceInputs(n, d, 1024)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _, _ = WyfChunkwiseRecurrence(q, k, v, beta, gate, n, d, c, nil)
	}
}

func BenchmarkSequentialGatedDeltaNet_Seq1024(b *testing.B) {
	n := 1024
	d := 128
	q, k, v, beta, gate := generateDeterministicRecurrenceInputs(n, d, 1024)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _, _ = SequentialGatedDeltaNet(q, k, v, beta, gate, n, d, nil)
	}
}
