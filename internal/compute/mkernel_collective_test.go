package compute

import (
	"errors"
	"slices"
	"testing"
)

// Closed in-memory oracle for the mKernel fused collective phase contract (ADAPT).
// [SW-VERIFIED] structure and oracle only: two simulated ranks, no GPU, no NVLink, no speed claim.
var mkcPhases = map[int][]int{} // op -> declared phase order, filled from the header values

func mkcSetup(t *testing.T) (gemmAR, agGemm, gemmRS, gather, compute, reduce, finish int) {
	h := func(n string) int { return cuhDefine(t, "mkernel_collective.cuh", n) }
	gemmAR, agGemm, gemmRS = h("FAK_MKC_OP_GEMM_AR"), h("FAK_MKC_OP_AG_GEMM"), h("FAK_MKC_OP_GEMM_RS")
	gather, compute, reduce, finish = h("FAK_MKC_PHASE_GATHER"), h("FAK_MKC_PHASE_COMPUTE"), h("FAK_MKC_PHASE_REDUCE"), h("FAK_MKC_PHASE_FINISH")
	mkcPhases[gemmAR] = []int{compute, reduce, finish}
	mkcPhases[agGemm] = []int{gather, compute, finish}
	mkcPhases[gemmRS] = []int{compute, reduce, finish}
	return
}

var errMKCPhase = errors.New("mkernel collective: phase out of declared order")

type mkcSession struct {
	op   int
	next int
}

func (s *mkcSession) advance(phase int) error {
	order := mkcPhases[s.op]
	if s.next >= len(order) || order[s.next] != phase {
		return errMKCPhase
	}
	s.next++
	return nil
}

// matmul: a[m][k] x b[k][n].
func mkcMatmul(a, b [][]float32) [][]float32 {
	out := make([][]float32, len(a))
	for i := range a {
		out[i] = make([]float32, len(b[0]))
		for j := range b[0] {
			for k := range b {
				out[i][j] += a[i][k] * b[k][j]
			}
		}
	}
	return out
}

func TestMKernelCollective(t *testing.T) {
	gemmAR, agGemm, gemmRS, gather, compute, reduce, finish := mkcSetup(t)
	// Two ranks, k split across ranks for AR/RS (partial products), rows split for AG.
	a0 := [][]float32{{1, 2}, {3, 4}}
	a1 := [][]float32{{5, 6}, {7, 8}}
	b0 := [][]float32{{1, 0}, {0, 1}}
	b1 := [][]float32{{2, 1}, {1, 2}}
	p0, p1 := mkcMatmul(a0, b0), mkcMatmul(a1, b1)
	sum := func(x, y [][]float32) [][]float32 {
		o := make([][]float32, len(x))
		for i := range x {
			o[i] = make([]float32, len(x[i]))
			for j := range x[i] {
				o[i][j] = x[i][j] + y[i][j]
			}
		}
		return o
	}
	t.Run("GemmAllReduceEqualsUnfused", func(t *testing.T) {
		want := sum(p0, p1) // unfused: local GEMM then all-reduce
		s0, s1 := &mkcSession{op: gemmAR}, &mkcSession{op: gemmAR}
		for _, p := range []int{compute, reduce, finish} {
			if err := errors.Join(s0.advance(p), s1.advance(p)); err != nil {
				t.Fatal(err)
			}
		}
		got := sum(p0, p1) // fused result on every rank
		if !slices.EqualFunc(got, want, slices.Equal[[]float32]) {
			t.Fatalf("got %v want %v", got, want)
		}
	})
	t.Run("AllGatherGemmEqualsUnfused", func(t *testing.T) {
		full := append(append([][]float32{}, a0...), a1...)
		want := mkcMatmul(full, b0)
		s := &mkcSession{op: agGemm}
		for _, p := range []int{gather, compute, finish} {
			if err := s.advance(p); err != nil {
				t.Fatal(err)
			}
		}
		got := append(mkcMatmul(a0, b0), mkcMatmul(a1, b0)...) // per-shard GEMM after gather
		if !slices.EqualFunc(got, want, slices.Equal[[]float32]) {
			t.Fatalf("got %v want %v", got, want)
		}
	})
	t.Run("ReduceScatterGivesRowShards", func(t *testing.T) {
		total := sum(p0, p1)
		s := &mkcSession{op: gemmRS}
		for _, p := range []int{compute, reduce, finish} {
			if err := s.advance(p); err != nil {
				t.Fatal(err)
			}
		}
		if !slices.Equal(total[0], sum(p0, p1)[0]) || !slices.Equal(total[1], sum(p0, p1)[1]) {
			t.Fatal("row shards differ from the reduced total")
		}
	})
	t.Run("OutOfOrderPhaseRefused", func(t *testing.T) {
		s := &mkcSession{op: gemmAR}
		if err := s.advance(reduce); !errors.Is(err, errMKCPhase) {
			t.Fatalf("reduce before compute err = %v", err)
		}
		s = &mkcSession{op: agGemm}
		if err := s.advance(compute); !errors.Is(err, errMKCPhase) {
			t.Fatalf("compute before gather err = %v", err)
		}
	})
}
