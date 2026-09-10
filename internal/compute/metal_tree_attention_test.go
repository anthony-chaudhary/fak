//go:build darwin && arm64 && cgo

package compute

import "testing"

func TestTreeSpecVerifyAttentionMetalParity(t *testing.T) {
	be, ref := metalOrSkip(t), Default()
	if !metalTreeAttentionAvailable() {
		t.Fatal("Metal backend registered without the native tree-attention pipeline")
	}
	const qLen, prefix, nH, nHkv, d = 4, 3, 4, 2, 64
	qRef, kRef, vRef := treeSpecFixture(ref, qLen, prefix, nH, nHkv, d, 12729)
	var want Tensor
	if err := ref.(TreeVerifyAttentionBackend).TreeVerifyAttention(&qRef, &kRef, &vRef, &want, branchingTreeMask(), qLen, prefix+qLen, nH, nHkv, d); err != nil {
		t.Fatal(err)
	}
	q, k, value := be.Upload(qRef, F32), be.Upload(kRef, F32), be.Upload(vRef, F32)
	var got Tensor
	t.Cleanup(func() { be.Free(q); be.Free(k); be.Free(value); be.Free(got) })
	if err := be.TreeVerifyAttention(&q, &k, &value, &got, branchingTreeMask(), qLen, prefix+qLen, nH, nHkv, d); err != nil {
		t.Fatal(err)
	}
	requireTreeApprox(t, ref.Read(want), be.Read(got), 0.9999, 1e-4)
}

func TestTreeSpecVerifyAttentionMetalLinearRegression(t *testing.T) {
	be, ref := metalOrSkip(t), Default()
	if !metalTreeAttentionAvailable() {
		t.Fatal("Metal backend registered without the native tree-attention pipeline")
	}
	const qLen, prefix, nH, nHkv, d = 4, 2, 4, 2, 16
	qRef, kRef, vRef := treeSpecFixture(ref, qLen, prefix, nH, nHkv, d, 12730)
	var want Tensor
	if err := SpecVerifyAttention(&qRef, &kRef, &vRef, &want, qLen, prefix+qLen, nH, nHkv, d); err != nil {
		t.Fatal(err)
	}
	q, k, value := be.Upload(qRef, F32), be.Upload(kRef, F32), be.Upload(vRef, F32)
	var got, gotLinear Tensor
	t.Cleanup(func() { be.Free(q); be.Free(k); be.Free(value); be.Free(got); be.Free(gotLinear) })
	if err := be.TreeVerifyAttention(&q, &k, &value, &got, linearTreeMask(qLen), qLen, prefix+qLen, nH, nHkv, d); err != nil {
		t.Fatal(err)
	}
	requireTreeApprox(t, ref.Read(want), be.Read(got), 0.9999, 1e-4)
	if err := be.SpecVerifyAttention(&q, &k, &value, &gotLinear, qLen, prefix+qLen, nH, nHkv, d); err != nil {
		t.Fatal(err)
	}
	requireTreeApprox(t, ref.Read(want), be.Read(gotLinear), 0.9999, 1e-4)
}

func TestTreeSpecVerifyAttentionMetalLinearBeyondMaskLimit(t *testing.T) {
	be, ref := metalOrSkip(t), Default()
	const qLen, prefix, nH, nHkv, d = 33, 1, 2, 1, 16
	qRef, kRef, vRef := treeSpecFixture(ref, qLen, prefix, nH, nHkv, d, 12731)
	var want Tensor
	if err := SpecVerifyAttention(&qRef, &kRef, &vRef, &want, qLen, prefix+qLen, nH, nHkv, d); err != nil {
		t.Fatal(err)
	}
	q, k, value := be.Upload(qRef, F32), be.Upload(kRef, F32), be.Upload(vRef, F32)
	var got Tensor
	t.Cleanup(func() { be.Free(q); be.Free(k); be.Free(value); be.Free(got) })
	if err := be.SpecVerifyAttention(&q, &k, &value, &got, qLen, prefix+qLen, nH, nHkv, d); err != nil {
		t.Fatal(err)
	}
	requireTreeApprox(t, ref.Read(want), be.Read(got), 0.9999, 1e-4)
}

func TestTreeSpecVerifyAttentionMetalBoundaryShapes(t *testing.T) {
	be, ref := metalOrSkip(t), Default()
	if !metalTreeAttentionAvailable() {
		t.Fatal("Metal backend registered without the native tree-attention pipeline")
	}
	for _, tc := range []struct {
		name    string
		qLen, d int
	}{{"bit31_odd_d37", 32, 37}, {"wide_d256", 4, 256}} {
		t.Run(tc.name, func(t *testing.T) {
			qRef, kRef, vRef := treeSpecFixture(ref, tc.qLen, 1, 2, 1, tc.d, lcg(13000+tc.qLen+tc.d))
			q, k, value := be.Upload(qRef, F32), be.Upload(kRef, F32), be.Upload(vRef, F32)
			var out Tensor
			defer func() { be.Free(q); be.Free(k); be.Free(value); be.Free(out) }()
			if err := be.TreeVerifyAttention(&q, &k, &value, &out, linearTreeMask(tc.qLen), tc.qLen, tc.qLen+1, 2, 1, tc.d); err != nil {
				t.Fatal(err)
			}
			if out.Dtype != F32 || len(out.Shape) != 3 || out.Shape[0] != tc.qLen || out.Shape[1] != 2 || out.Shape[2] != tc.d {
				t.Fatalf("output contract dtype=%v shape=%v", out.Dtype, out.Shape)
			}
		})
	}
	qRef, kRef, vRef := treeSpecFixture(ref, 4, 1, 2, 1, 16, 13100)
	q, k, value := be.Upload(qRef, F32), be.Upload(kRef, F32), be.Upload(vRef, F32)
	defer func() { be.Free(q); be.Free(k); be.Free(value) }()
	badQ := q
	badQ.Dtype = F16
	var out Tensor
	if err := be.TreeVerifyAttention(&badQ, &k, &value, &out, linearTreeMask(4), 4, 5, 2, 1, 16); err == nil {
		t.Fatal("accepted non-F32 q")
	}
}
