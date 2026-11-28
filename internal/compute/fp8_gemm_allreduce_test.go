package compute

import (
	"errors"
	"math"
	"testing"
)

// Closed in-memory oracle for FlashInfer cake SM90 FP8 rowwise GEMM+all-reduce (ADAPT).
// [SW-VERIFIED] structure and oracle only: two-peer fixture, no P2P/NVLink, no speed claim.
var errFP8ARNotReady = errors.New("fp8 gemm allreduce: peer flag word not ready")

type fp8ARPeerSlot struct {
	flag    int
	partial [][]float32
}

func fp8RowwiseGEMM(a, b [][]float32, rowScaleA, colScaleB []float32) [][]float32 {
	out := make([][]float32, len(a))
	for i := range a {
		out[i] = make([]float32, len(b[0]))
		for j := range b[0] {
			var acc float32
			for k := range b {
				acc += a[i][k] * b[k][j]
			}
			out[i][j] = acc * rowScaleA[i] * colScaleB[j]
		}
	}
	return out
}

func fp8ARConsume(peers []fp8ARPeerSlot, flagReady int) ([][]float32, error) {
	for _, p := range peers {
		if p.flag != flagReady {
			return nil, errFP8ARNotReady
		}
	}
	rows, cols := len(peers[0].partial), len(peers[0].partial[0])
	out := make([][]float32, rows)
	for i := 0; i < rows; i++ {
		out[i] = make([]float32, cols)
		for j := 0; j < cols; j++ {
			for _, p := range peers {
				out[i][j] += p.partial[i][j]
			}
		}
	}
	return out, nil
}

func TestFP8GemmAllreduce(t *testing.T) {
	t.Parallel()
	if cuhDefine(t, "fp8_gemm_allreduce.cuh", "FAK_FI_FP8_AR_CTA_THREADS") != 512 ||
		cuhDefine(t, "fp8_gemm_allreduce.cuh", "FAK_FI_FP8_AR_PIPELINE_STAGES") != 5 ||
		cuhDefine(t, "fp8_gemm_allreduce.cuh", "FAK_FI_FP8_AR_MMA_M") != 64 ||
		cuhDefine(t, "fp8_gemm_allreduce.cuh", "FAK_FI_FP8_AR_MMA_N") != 128 ||
		cuhDefine(t, "fp8_gemm_allreduce.cuh", "FAK_FI_FP8_AR_MMA_K") != 32 {
		t.Fatal("fp8_gemm_allreduce.cuh geometry constants drifted")
	}
	flagEmpty := cuhDefine(t, "fp8_gemm_allreduce.cuh", "FAK_FI_FP8_AR_FLAG_EMPTY")
	flagReady := cuhDefine(t, "fp8_gemm_allreduce.cuh", "FAK_FI_FP8_AR_FLAG_READY")

	a0 := [][]float32{{1, 2}, {3, 4}}
	a1 := [][]float32{{2, 1}, {0, 3}}
	b0 := [][]float32{{1, -1}, {2, 0}}
	b1 := [][]float32{{0, 2}, {1, 1}}
	rs := []float32{0.5, 0.25}
	cs := []float32{1.0, 2.0}

	p0 := fp8RowwiseGEMM(a0, b0, rs, cs)
	p1 := fp8RowwiseGEMM(a1, b1, rs, cs)

	peers := []fp8ARPeerSlot{
		{flag: flagReady, partial: p0},
		{flag: flagEmpty, partial: p1},
	}
	if _, err := fp8ARConsume(peers, flagReady); !errors.Is(err, errFP8ARNotReady) {
		t.Fatalf("unread peer flag err = %v, want errFP8ARNotReady", err)
	}

	peers[1].flag = flagReady
	got, err := fp8ARConsume(peers, flagReady)
	if err != nil {
		t.Fatal(err)
	}
	for i := range got {
		for j := range got[i] {
			want := p0[i][j] + p1[i][j]
			if math.Abs(float64(got[i][j]-want)) > 1e-6 {
				t.Fatalf("out[%d][%d] = %v, want %v", i, j, got[i][j], want)
			}
		}
	}
}
