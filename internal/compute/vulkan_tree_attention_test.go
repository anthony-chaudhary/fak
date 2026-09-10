//go:build vulkan && (windows || linux) && cgo

package compute

import (
	"encoding/json"
	"os"
	"sort"
	"testing"
	"time"
)

func TestTreeSpecVerifyAttentionVulkanParity(t *testing.T) {
	be, ref := vk(t), Default()
	const qLen, prefix, nH, nHkv, d = 4, 3, 4, 2, 64
	qRef, kRef, vRef := treeSpecFixture(ref, qLen, prefix, nH, nHkv, d, 12726)
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

func TestTreeSpecVerifyAttentionVulkanLinearRegression(t *testing.T) {
	be, ref := vk(t), Default()
	const qLen, prefix, nH, nHkv, d = 4, 2, 4, 2, 16
	qRef, kRef, vRef := treeSpecFixture(ref, qLen, prefix, nH, nHkv, d, 12727)
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

func TestTreeSpecVerifyAttentionVulkanLinearBeyondMaskLimit(t *testing.T) {
	be, ref := vk(t), Default()
	const qLen, prefix, nH, nHkv, d = 33, 1, 2, 1, 16
	qRef, kRef, vRef := treeSpecFixture(ref, qLen, prefix, nH, nHkv, d, 12732)
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

func TestTreeSpecVerifyAttentionVulkanBoundaryShapesAndBuffers(t *testing.T) {
	be, ref := vk(t), Default()
	for _, tc := range []struct {
		name    string
		qLen, d int
	}{{"bit31_odd_d37", 32, 37}, {"wide_d256", 4, 256}, {"max_d1024", 2, 1024}} {
		t.Run(tc.name, func(t *testing.T) {
			qRef, kRef, vRef := treeSpecFixture(ref, tc.qLen, 1, 2, 1, tc.d, lcg(12800+tc.qLen+tc.d))
			var want Tensor
			if err := ref.(TreeVerifyAttentionBackend).TreeVerifyAttention(&qRef, &kRef, &vRef, &want, linearTreeMask(tc.qLen), tc.qLen, tc.qLen+1, 2, 1, tc.d); err != nil {
				t.Fatal(err)
			}
			q, k, value := be.Upload(qRef, F32), be.Upload(kRef, F32), be.Upload(vRef, F32)
			var out Tensor
			defer func() { be.Free(q); be.Free(k); be.Free(value); be.Free(out) }()
			if err := be.TreeVerifyAttention(&q, &k, &value, &out, linearTreeMask(tc.qLen), tc.qLen, tc.qLen+1, 2, 1, tc.d); err != nil {
				t.Fatal(err)
			}
			if out.Dtype != F32 || len(out.Shape) != 3 || out.Shape[0] != tc.qLen || out.Shape[1] != 2 || out.Shape[2] != tc.d {
				t.Fatalf("output contract dtype=%v shape=%v", out.Dtype, out.Shape)
			}
			requireTreeApprox(t, ref.Read(want), be.Read(out), 0.9999, 1e-4)
		})
	}
	qRef, kRef, vRef := treeSpecFixture(ref, 4, 1, 2, 1, 16, 12900)
	q, k, value := be.Upload(qRef, F32), be.Upload(kRef, F32), be.Upload(vRef, F32)
	defer func() { be.Free(q); be.Free(k); be.Free(value) }()
	badQ := q
	badQ.Dtype = F16
	var out Tensor
	if err := be.TreeVerifyAttention(&badQ, &k, &value, &out, linearTreeMask(4), 4, 5, 2, 1, 16); err == nil {
		t.Fatal("accepted non-F32 q")
	}
	wrong := be.Upload(NewF32(ref, []int{1}, []float32{0}), F32)
	defer be.Free(wrong)
	if err := be.TreeVerifyAttention(&q, &k, &value, &wrong, linearTreeMask(4), 4, 5, 2, 1, 16); err == nil {
		t.Fatal("accepted wrong preallocated output shape")
	}
}

func TestTreeSpecVerifyAttentionVulkanBatchMaskLifetime(t *testing.T) {
	be, ref := vk(t), Default()
	const qLen, prefix, nH, nHkv, d = 4, 3, 4, 2, 16
	qRef, kRef, vRef := treeSpecFixture(ref, qLen, prefix, nH, nHkv, d, 13200)
	branchRows, linearRows := branchingTreeMask(), linearTreeMask(qLen)
	var wantBranch, wantLinear Tensor
	for _, tc := range []struct {
		rows []uint32
		out  *Tensor
	}{{branchRows, &wantBranch}, {linearRows, &wantLinear}} {
		if err := ref.(TreeVerifyAttentionBackend).TreeVerifyAttention(&qRef, &kRef, &vRef, tc.out, tc.rows, qLen, prefix+qLen, nH, nHkv, d); err != nil {
			t.Fatal(err)
		}
	}
	q, k, value := be.Upload(qRef, F32), be.Upload(kRef, F32), be.Upload(vRef, F32)
	zeroOut := NewF32(ref, []int{qLen, nH, d}, make([]float32, qLen*nH*d))
	outBranch, outLinear := be.Upload(zeroOut, F32), be.Upload(zeroOut, F32)
	be.BeginBatch()
	t.Cleanup(func() {
		be.FlushBatch()
		be.Free(q)
		be.Free(k)
		be.Free(value)
		be.Free(outBranch)
		be.Free(outLinear)
	})
	if err := be.TreeVerifyAttention(&q, &k, &value, &outBranch, branchRows, qLen, prefix+qLen, nH, nHkv, d); err != nil {
		t.Fatal(err)
	}
	if !be.VulkanDebugBatchActive() {
		t.Fatal("batch inactive after first tree-attention dispatch")
	}
	if err := be.TreeVerifyAttention(&q, &k, &value, &outLinear, linearRows, qLen, prefix+qLen, nH, nHkv, d); err != nil {
		t.Fatal(err)
	}
	if !be.VulkanDebugBatchActive() {
		t.Fatal("batch inactive after second tree-attention dispatch")
	}
	be.FlushBatch()
	requireTreeApprox(t, ref.Read(wantBranch), be.Read(outBranch), 0.9999, 1e-4)
	requireTreeApprox(t, ref.Read(wantLinear), be.Read(outLinear), 0.9999, 1e-4)
}

func TestTreeSpecVerifyAttentionVulkanHardwareWitness(t *testing.T) {
	be := vk(t)
	const qLen, prefix, nH, nHkv, d, samples = 16, 64, 8, 2, 64, 30
	q, k, value := treeSpecFixture(be, qLen, prefix, nH, nHkv, d, 12728)
	rows := make([]uint32, qLen)
	for i := range rows {
		rows[i] = uint32(1) << uint(i)
		for p := (i - 1) / 2; i > 0 && p >= 0; p = (p - 1) / 2 {
			rows[i] |= uint32(1) << uint(p)
			if p == 0 {
				break
			}
		}
	}
	zeroOut := NewF32(Default(), []int{qLen, nH, d}, make([]float32, qLen*nH*d))
	out := be.Upload(zeroOut, F32)
	t.Cleanup(func() { be.Free(q); be.Free(k); be.Free(value); be.Free(out) })
	if err := be.TreeVerifyAttention(&q, &k, &value, &out, rows, qLen, prefix+qLen, nH, nHkv, d); err != nil {
		t.Fatal(err)
	}
	before := be.VulkanDebugDispatchProfileSnapshot()
	times := make([]time.Duration, samples)
	for i := range times {
		start := time.Now()
		if err := be.TreeVerifyAttention(&q, &k, &value, &out, rows, qLen, prefix+qLen, nH, nHkv, d); err != nil {
			t.Fatal(err)
		}
		times[i] = time.Since(start)
	}
	after := be.VulkanDebugDispatchProfileSnapshot()
	if d2h := after.OneShotD2HSubmits - before.OneShotD2HSubmits; d2h != 0 {
		t.Fatalf("TreeVerifyAttention issued %d D2H submissions during calls", d2h)
	}
	if dispatches := after.ComputeDispatches - before.ComputeDispatches; dispatches == 0 {
		if os.Getenv("FAK_VULKAN_DISPATCH_PROFILE") == "1" {
			t.Fatal("TreeVerifyAttention recorded no Vulkan compute dispatches")
		}
		t.Log("Vulkan dispatch profiling disabled; set FAK_VULKAN_DISPATCH_PROFILE=1 for native-route attribution")
	}
	sort.Slice(times, func(i, j int) bool { return times[i] < times[j] })
	p50, p90 := times[len(times)/2], times[(len(times)*9-1)/10]
	t.Logf("[HW-WITNESSED] timing_domain=host_wall_dispatch_to_completion samples=%d p50=%s p90=%s; synchronous upper bound, excludes setup and output D2H read", samples, p50, p90)
	rawNS := make([]int64, len(times))
	for i := range times {
		rawNS[i] = times[i].Nanoseconds()
	}
	receipt, err := json.Marshal(struct {
		TimingDomain string  `json:"timing_domain"`
		SamplesNS    []int64 `json:"samples_ns"`
		P50NS        int64   `json:"p50_ns"`
		P90NS        int64   `json:"p90_ns"`
	}{"host_wall_dispatch_to_completion", rawNS, p50.Nanoseconds(), p90.Nanoseconds()})
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("TREE_ATTENTION_TIMING_JSON=%s", receipt)
	if os.Getenv("FAK_VULKAN_TREE_ATTENTION_ENFORCE_SUB_MS") == "1" && p90 >= time.Millisecond {
		t.Fatalf("dispatch-to-completion host-wall p90=%s, want <1ms", p90)
	}
	if got := be.Read(out); len(got) != qLen*nH*d {
		t.Fatalf("output length=%d", len(got))
	}
}
