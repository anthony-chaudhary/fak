package compute

import (
	"math"
	"math/rand"
	"testing"
)

func referenceUnpaddedMatMul(tokens []float32, weights []float32, inDim, outDim int, scale float32) []float32 {
	numTokens := len(tokens) / inDim
	out := make([]float32, numTokens*outDim)
	for t := 0; t < numTokens; t++ {
		tIn := t * inDim
		tOut := t * outDim
		for o := 0; o < outDim; o++ {
			wOff := o * inDim
			var dot float64
			for i := 0; i < inDim; i++ {
				dot += float64(weights[wOff+i]) * float64(tokens[tIn+i])
			}
			out[tOut+o] = float32(dot * float64(scale))
		}
	}
	return out
}

func TestRaggedContinuousQKVMatMulWitness(t *testing.T) {
	// First witness requirements (#9934):
	// 1. Mixed-length batches (e.g. lengths [1, 16, 5]).
	// 2. Exact per-request reference equality against unpadded scalar oracle.
	// 3. Cancellation canary: cancelled requests skip work without corrupting active requests.
	// 4. Padding canaries: proves zero accesses beyond ragged boundaries.
	// 5. Significant padding elimination (> 50% waste reduction).

	inDim := 32
	outDim := 64

	rng := rand.New(rand.NewSource(9934))

	weightsQKV := make([]float32, outDim*inDim)
	for i := range weightsQKV {
		weightsQKV[i] = rng.Float32()*2 - 1
	}

	requests := []RaggedQKVBatchRequest{
		{RequestID: "req-short", NumTokens: 1, Scale: 1.0, Cancelled: false},
		{RequestID: "req-long", NumTokens: 16, Scale: 0.5, Cancelled: false},
		{RequestID: "req-cancelled", NumTokens: 4, Scale: 1.0, Cancelled: true},
		{RequestID: "req-medium", NumTokens: 5, Scale: 2.0, Cancelled: false},
	}

	totalRaggedTokens := 1 + 16 + 4 + 5 // 26 tokens (vs 16 * 4 = 64 tokens if padded!)
	packedTokens := make([]float32, totalRaggedTokens*inDim)
	for i := range packedTokens {
		packedTokens[i] = rng.Float32()*2 - 1
	}

	// Setup memory with padding canary buffer at the end
	canaryVal := float32(0xDEADBEEF)
	outBuffer := make([]float32, totalRaggedTokens*outDim+4)
	for i := totalRaggedTokens * outDim; i < len(outBuffer); i++ {
		outBuffer[i] = canaryVal
	}
	outQKV := outBuffer[:totalRaggedTokens*outDim]

	canaryCheckedCount := 0
	checkHook := func(tokenIdx int) {
		canaryCheckedCount++
	}

	receipt, err := ExecuteRaggedContinuousQKVMatMul(
		requests, packedTokens, weightsQKV, inDim, outDim, outQKV, checkHook,
	)
	if err != nil {
		t.Fatalf("ExecuteRaggedContinuousQKVMatMul failed: %v", err)
	}

	// 4. Verify padding canaries were NEVER touched
	for i := totalRaggedTokens * outDim; i < len(outBuffer); i++ {
		if outBuffer[i] != canaryVal {
			t.Fatalf("padding canary at index %d was corrupted: %v", i, outBuffer[i])
		}
	}

	// 5. Verify padding elimination
	if receipt.TotalRaggedTokens != 26 {
		t.Fatalf("expected 26 ragged tokens, got %d", receipt.TotalRaggedTokens)
	}
	if receipt.PaddedTokens != 64 { // max 16 * 4 = 64
		t.Fatalf("expected 64 padded tokens, got %d", receipt.PaddedTokens)
	}
	if receipt.PaddingEliminated != 38 {
		t.Fatalf("expected 38 eliminated padding tokens, got %d", receipt.PaddingEliminated)
	}
	if receipt.WasteReductionPct < 55.0 {
		t.Fatalf("waste reduction %v < 55%%", receipt.WasteReductionPct)
	}
	if receipt.ProcessedTokens != 22 { // 26 - 4 cancelled = 22
		t.Fatalf("expected 22 processed tokens, got %d", receipt.ProcessedTokens)
	}

	// 2 & 3. Verify per-request equality against reference oracle
	tokenCursor := 0
	for _, req := range requests {
		gotSlice := outQKV[tokenCursor*outDim : (tokenCursor+req.NumTokens)*outDim]

		if req.Cancelled {
			// Cancelled tokens must be zeroed
			for _, v := range gotSlice {
				if v != 0 {
					t.Fatalf("cancelled request %s output non-zero: %v", req.RequestID, v)
				}
			}
		} else {
			inSlice := packedTokens[tokenCursor*inDim : (tokenCursor+req.NumTokens)*inDim]
			wantSlice := referenceUnpaddedMatMul(inSlice, weightsQKV, inDim, outDim, req.Scale)

			for i := range wantSlice {
				if math.Abs(float64(gotSlice[i]-wantSlice[i])) > 1e-5 {
					t.Fatalf("request %s token mismatch at %d: got %v, want %v", req.RequestID, i, gotSlice[i], wantSlice[i])
				}
			}
		}
		tokenCursor += req.NumTokens
	}
}

func TestRaggedContinuousQKVMatMulFailClosed(t *testing.T) {
	// Empty requests
	if _, err := ExecuteRaggedContinuousQKVMatMul(nil, nil, nil, 32, 64, nil, nil); err == nil {
		t.Fatal("expected error on empty requests")
	}

	// Mismatched token lengths
	reqs := []RaggedQKVBatchRequest{{RequestID: "r1", NumTokens: 5}}
	if _, err := ExecuteRaggedContinuousQKVMatMul(reqs, make([]float32, 10), make([]float32, 64), 32, 64, make([]float32, 64), nil); err == nil {
		t.Fatal("expected error on mismatched token length")
	}
}
