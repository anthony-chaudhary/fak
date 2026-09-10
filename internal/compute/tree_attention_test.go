package compute

import (
	"math"
	"strings"
	"testing"
)

func treeSpecFixture(be Backend, qLen, prefix, nH, nHkv, d int, seed lcg) (Tensor, Tensor, Tensor) {
	kvLen := prefix + qLen
	q := randVec(&seed, qLen*nH*d)
	k := randVec(&seed, kvLen*nHkv*d)
	v := randVec(&seed, kvLen*nHkv*d)
	return be.Upload(NewF32(Default(), []int{qLen, nH, d}, q), F32),
		be.Upload(NewF32(Default(), []int{kvLen, nHkv, d}, k), F32),
		be.Upload(NewF32(Default(), []int{kvLen, nHkv, d}, v), F32)
}

func linearTreeMask(qLen int) []uint32 {
	rows := make([]uint32, qLen)
	for q := range rows {
		rows[q] = (uint32(1) << uint(q+1)) - 1
	}
	return rows
}

func branchingTreeMask() []uint32 { return []uint32{0b0001, 0b0010, 0b0101, 0b1010} }

func requireTreeApprox(t *testing.T, want, got []float32, minCos, maxDelta float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("output length=%d, want %d", len(got), len(want))
	}
	if i := firstNonFiniteTreeOutput(want, got); i >= 0 {
		t.Fatalf("non-finite output at %d: want=%v got=%v", i, want[i], got[i])
	}
	if c := cosine(want, got); c < minCos {
		t.Fatalf("cosine=%.8f, want >= %.8f (max delta %.3g)", c, minCos, maxAbsDelta(want, got))
	}
	if d := maxAbsDelta(want, got); d > maxDelta {
		t.Fatalf("max delta=%.3g, want <= %.3g", d, maxDelta)
	}
}

func firstNonFiniteTreeOutput(want, got []float32) int {
	for i := range want {
		if math.IsNaN(float64(want[i])) || math.IsInf(float64(want[i]), 0) || math.IsNaN(float64(got[i])) || math.IsInf(float64(got[i]), 0) {
			return i
		}
	}
	return -1
}

func TestTreeSpecVerifyAttentionFiniteResultGuard(t *testing.T) {
	if got := firstNonFiniteTreeOutput([]float32{1, 2}, []float32{1, 2}); got != -1 {
		t.Fatalf("finite vectors flagged at %d", got)
	}
	if got := firstNonFiniteTreeOutput([]float32{1, 2}, []float32{1, float32(math.NaN())}); got != 1 {
		t.Fatalf("NaN index=%d, want 1", got)
	}
	if got := firstNonFiniteTreeOutput([]float32{float32(math.Inf(1))}, []float32{1}); got != 0 {
		t.Fatalf("Inf index=%d, want 0", got)
	}
}

func TestTreeSpecVerifyAttentionBranchMaskAndPrefix(t *testing.T) {
	ref := Default()
	const qLen, prefix, nH, nHkv, d = 4, 1, 1, 1, 2
	q := NewF32(ref, []int{qLen, nH, d}, make([]float32, qLen*nH*d))
	k := NewF32(ref, []int{prefix + qLen, nHkv, d}, make([]float32, (prefix+qLen)*nHkv*d))
	v := NewF32(ref, []int{prefix + qLen, nHkv, d}, []float32{10, 20, 1, 2, 3, 4, 5, 6, 7, 8})
	var out Tensor
	if err := TreeVerifyAttention(&q, &k, &v, &out, branchingTreeMask(), qLen, prefix+qLen, nH, nHkv, d); err != nil {
		t.Fatal(err)
	}
	want := []float32{5.5, 11, 6.5, 12, 16.0 / 3, 28.0 / 3, 20.0 / 3, 32.0 / 3}
	if d := maxAbsDelta(want, ref.Read(out)); d > 1e-6 {
		t.Fatalf("branch/prefix result max delta %.3g; got %v want %v", d, ref.Read(out), want)
	}
}

func TestTreeSpecVerifyAttentionLinearRegression(t *testing.T) {
	ref := Default()
	const qLen, prefix, nH, nHkv, d = 4, 3, 4, 2, 16
	q, k, v := treeSpecFixture(ref, qLen, prefix, nH, nHkv, d, 12724)
	var linear, masked Tensor
	if err := SpecVerifyAttention(&q, &k, &v, &linear, qLen, prefix+qLen, nH, nHkv, d); err != nil {
		t.Fatal(err)
	}
	if err := TreeVerifyAttention(&q, &k, &v, &masked, linearTreeMask(qLen), qLen, prefix+qLen, nH, nHkv, d); err != nil {
		t.Fatal(err)
	}
	if d := maxAbsDelta(ref.Read(linear), ref.Read(masked)); d > 1e-6 {
		t.Fatalf("dense causal tree mask differs from linear verify by %.3g (>1e-6)", d)
	}
}

func TestTreeSpecVerifyAttentionSiblingIsolation(t *testing.T) {
	ref := Default()
	const qLen, prefix, nH, nHkv, d = 4, 2, 2, 1, 8
	q, k, v := treeSpecFixture(ref, qLen, prefix, nH, nHkv, d, 12725)
	var before Tensor
	if err := TreeVerifyAttention(&q, &k, &v, &before, branchingTreeMask(), qLen, prefix+qLen, nH, nHkv, d); err != nil {
		t.Fatal(err)
	}
	k2 := append([]float32(nil), ref.Read(k)...)
	v2 := append([]float32(nil), ref.Read(v)...)
	siblingKVIndex := prefix + 1 // candidate 1 is invisible to candidate 2.
	for i := siblingKVIndex * nHkv * d; i < (siblingKVIndex+1)*nHkv*d; i++ {
		k2[i], v2[i] = k2[i]+100, v2[i]+100
	}
	kCorrupt := NewF32(ref, k.Shape, k2)
	vCorrupt := NewF32(ref, v.Shape, v2)
	var after Tensor
	if err := TreeVerifyAttention(&q, &kCorrupt, &vCorrupt, &after, branchingTreeMask(), qLen, prefix+qLen, nH, nHkv, d); err != nil {
		t.Fatal(err)
	}
	start, end := 2*nH*d, 3*nH*d
	if d := maxAbsDelta(ref.Read(before)[start:end], ref.Read(after)[start:end]); d != 0 {
		t.Fatalf("candidate 2 changed after corrupting sibling candidate 1: max delta %.3g", d)
	}
}

func TestTreeSpecVerifyAttentionBit31AndValidation(t *testing.T) {
	ref := Default()
	const qLen, nH, nHkv, d = 32, 1, 1, 1
	q := NewF32(ref, []int{qLen, nH, d}, make([]float32, qLen))
	k := NewF32(ref, []int{qLen, nHkv, d}, make([]float32, qLen))
	v := NewF32(ref, []int{qLen, nHkv, d}, make([]float32, qLen))
	rows := make([]uint32, qLen)
	for i := range rows {
		rows[i] = uint32(1) << uint(i)
	}
	var out Tensor
	if err := TreeVerifyAttention(&q, &k, &v, &out, rows, qLen, qLen, nH, nHkv, d); err != nil {
		t.Fatalf("bit 31 rejected: %v", err)
	}

	valid4 := NewF32(ref, []int{4, 1, 1}, make([]float32, 4))
	checks := []struct {
		name, contains           string
		rows                     []uint32
		qLen, kvLen, nH, nHkv, d int
	}{
		{"row count", "mask rows", []uint32{1}, 4, 4, 1, 1, 1},
		{"missing self", "omits self", []uint32{1, 1, 4, 8}, 4, 4, 1, 1, 1},
		{"future", "future key", []uint32{3, 2, 4, 8}, 4, 4, 1, 1, 1},
		{"beyond qLen", "beyond qLen", []uint32{1 | 16, 2, 4, 8}, 4, 4, 1, 1, 1},
		{"qLen zero", "invalid lengths", nil, 0, 4, 1, 1, 1},
		{"qLen 33", "invalid lengths", make([]uint32, 33), 33, 33, 1, 1, 1},
		{"kv shorter", "invalid lengths", linearTreeMask(4), 4, 3, 1, 1, 1},
		{"heads", "invalid heads", linearTreeMask(4), 4, 4, 3, 2, 1},
		{"dim zero", "head dim", linearTreeMask(4), 4, 4, 1, 1, 0},
		{"dim too large", "head dim", linearTreeMask(4), 4, 4, 1, 1, 1025},
	}
	for _, tc := range checks {
		t.Run(tc.name, func(t *testing.T) {
			err := TreeVerifyAttention(&valid4, &valid4, &valid4, &out, tc.rows, tc.qLen, tc.kvLen, tc.nH, tc.nHkv, tc.d)
			if err == nil || !strings.Contains(err.Error(), tc.contains) {
				t.Fatalf("error=%v, want containing %q", err, tc.contains)
			}
		})
	}
	if math.IsNaN(float64(ref.Read(out)[0])) {
		t.Fatal("valid bit-31 output is NaN")
	}
}
