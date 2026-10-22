package model

import (
	"math"
	"testing"
)

// fak-test:runtime fast est=1s lane=default
// fak-test:justify why=contract when=changed:internal/model/wyf_recurrence*.go
func TestWYFChunkwiseRecurrenceVectorGatesMatchSequential(t *testing.T) {
	const (
		tokens    = 7
		dim       = 4
		chunkSize = 3
		tolerance = 1e-4
	)

	k := make([][]float32, tokens)
	v := make([][]float32, tokens)
	alphaVector := make([][]float32, tokens)
	betaVector := make([][]float32, tokens)
	alphaScalar := make([][]float32, tokens)
	betaScalar := make([][]float32, tokens)
	alphaMixedChunks := make([][]float32, tokens)
	for token := 0; token < tokens; token++ {
		k[token] = make([]float32, dim)
		v[token] = make([]float32, dim)
		alphaVector[token] = make([]float32, dim)
		betaVector[token] = make([]float32, dim)
		for d := 0; d < dim; d++ {
			k[token][d] = float32(((token+1)*(d+2))%9-4) * 0.06
			v[token][d] = float32(((token+3)*(d+1))%11-5) * 0.07
			alphaVector[token][d] = 0.82 + float32((token+2*d)%7)*0.02
			betaVector[token][d] = 0.24 + float32((2*token+d)%8)*0.06
		}
		alphaScalar[token] = []float32{0.84 + float32(token%5)*0.025}
		betaScalar[token] = []float32{0.30 + float32(token%4)*0.09}
		alphaMixedChunks[token] = alphaScalar[token]
		if token >= chunkSize && token < 2*chunkSize {
			alphaMixedChunks[token] = alphaVector[token]
		}
	}

	s0 := make([][]float32, dim)
	for i := 0; i < dim; i++ {
		s0[i] = make([]float32, dim)
		for j := 0; j < dim; j++ {
			s0[i][j] = float32((i+1)*(j+2)-5) * 0.015
		}
	}

	testCases := []struct {
		name  string
		alpha [][]float32
		beta  [][]float32
	}{
		{name: "vector_alpha_scalar_beta", alpha: alphaVector, beta: betaScalar},
		{name: "scalar_alpha_vector_beta", alpha: alphaScalar, beta: betaVector},
		{name: "vector_alpha_vector_beta", alpha: alphaVector, beta: betaVector},
		{name: "scalar_vector_scalar_chunks", alpha: alphaMixedChunks, beta: betaScalar},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			initialState := cloneWYFTestMatrix(s0)
			wantOutputs, wantStates := SequentialGDN(k, v, tc.beta, tc.alpha, s0)
			gotOutputs, gotStates := WYFChunkwiseRecurrence(k, v, tc.beta, tc.alpha, s0, chunkSize)

			assertWYFResultsClose(t, "output", gotOutputs, wantOutputs, tolerance)
			if len(gotStates) != len(wantStates) {
				t.Fatalf("state count = %d, want %d", len(gotStates), len(wantStates))
			}
			for token := range wantStates {
				assertWYFResultsClose(t, "state", gotStates[token], wantStates[token], tolerance)
			}
			assertWYFResultsClose(t, "initial state", s0, initialState, 0)

			laterValue := gotStates[1][0][0]
			gotStates[0][0][0]++
			if gotStates[1][0][0] != laterValue {
				t.Fatalf("state snapshots alias: mutating token 0 changed token 1 from %v to %v", laterValue, gotStates[1][0][0])
			}
			assertWYFResultsClose(t, "initial state after snapshot mutation", s0, initialState, 0)
		})
	}
}

func cloneWYFTestMatrix(matrix [][]float32) [][]float32 {
	clone := make([][]float32, len(matrix))
	for i := range matrix {
		clone[i] = append([]float32(nil), matrix[i]...)
	}
	return clone
}

func assertWYFResultsClose(t *testing.T, label string, got, want [][]float32, tolerance float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s row count = %d, want %d", label, len(got), len(want))
	}
	for row := range want {
		if len(got[row]) != len(want[row]) {
			t.Fatalf("%s row %d length = %d, want %d", label, row, len(got[row]), len(want[row]))
		}
		for col := range want[row] {
			gotValue := float64(got[row][col])
			wantValue := float64(want[row][col])
			if math.IsNaN(gotValue) || math.IsInf(gotValue, 0) || math.IsNaN(wantValue) || math.IsInf(wantValue, 0) {
				t.Fatalf("%s[%d][%d] is nonfinite: got %v, want %v", label, row, col, gotValue, wantValue)
			}
			if diff := math.Abs(gotValue - wantValue); diff > tolerance {
				t.Fatalf("%s[%d][%d] diff %e > tolerance %e: got %v, want %v", label, row, col, diff, tolerance, gotValue, wantValue)
			}
		}
	}
}
