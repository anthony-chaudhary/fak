package compute

import (
	"math"
	"os"
	"strings"
	"testing"
)

func decodeCutOperand(id int64, allocation uint64, offset uint64, shape ...int) vulkanF32DecodeOperand {
	n, _ := vulkanF32DecodeBytes(shape)
	return vulkanF32DecodeOperand{BufferID: id, Bytes: n, Shape: shape,
		AllocationID: allocation, BindingOffset: offset, BufferBytes: uint64(n), ReservationBytes: 4096}
}

// TestVulkanBatchDecodeCutAccesses is a host contract, not a GPU or whole-model
// witness. Caller source selection and native capture have separately named cases.
func TestVulkanBatchDecodeCutAccesses(t *testing.T) {
	w := decodeCutOperand(1, 1, 0, 8, 4)
	x := decodeCutOperand(2, 2, 0, 4)
	q := decodeCutOperand(3, 3, 0, 8)
	wk := decodeCutOperand(4, 1, 128, 4, 4) // same arena, disjoint binding
	k := decodeCutOperand(5, 4, 0, 4)
	qa, ok := vulkanF32DecodeMatMulAccesses(w, x, q, 8, 4)
	if !ok || qa[0].Size != 128 || qa[1].Size != 16 || qa[2].Size != 32 ||
		qa[0].Mode != VulkanAccessRead || qa[1].Mode != VulkanAccessRead || qa[2].Mode != VulkanAccessWrite {
		t.Fatalf("incomplete Q declaration: %+v, %v", qa, ok)
	}
	ka, ok := vulkanF32DecodeMatMulAccesses(wk, x, k, 4, 4)
	if !ok {
		t.Fatal("valid K declaration declined")
	}
	decl := func(ordinal int, accesses []VulkanBufferAccess) VulkanDispatchAccess {
		return VulkanDispatchAccess{NodeID: ordinal, ShimOrdinal: ordinal, Declared: true, Accesses: accesses}
	}
	t.Run("ordinary_f32_independent_qk", func(t *testing.T) {
		var l vulkanBatchHazardLedger
		l.begin()
		if !l.recordVerdict(decl(2, qa[:])) || l.recordVerdict(decl(3, ka[:])) {
			t.Fatal("first projection must sync; consecutive independent K must elide")
		}
		consume := []VulkanBufferAccess{qa[2]}
		consume[0].Mode = VulkanAccessRead
		if !l.recordVerdict(decl(4, consume)) {
			t.Fatal("older Q producer was forgotten")
		}
	})
	t.Run("physical_binding_overlap", func(t *testing.T) {
		a := qa[2]
		b := a
		b.BufferID = 99
		b.Mode = VulkanAccessRead
		b.BindingOffset = 16
		if kind, sync := accessModesConflict(a, b); !sync || kind != VulkanHazardReadAfterWrite {
			t.Fatal("distinct overlapping bindings lost RAW")
		}
		b.BindingOffset = 32
		if _, sync := accessModesConflict(a, b); sync {
			t.Fatal("disjoint backing ranges conflicted")
		}
		b.AllocationID = 0
		b.BindingOffset = 0
		if _, sync := accessModesConflict(a, b); !sync {
			t.Fatal("mixed provenance licensed elision")
		}
		b = a
		b.AllocationID++
		if _, sync := accessModesConflict(a, b); !sync {
			t.Fatal("reused binding identity licensed elision")
		}
	})
	t.Run("unknown_hidden_access_and_gaps", func(t *testing.T) {
		for _, unknown := range []bool{false, true} {
			var l vulkanBatchHazardLedger
			l.begin()
			l.recordVerdict(decl(1, qa[:]))
			if unknown {
				l.recordVerdict(decl(2, nil))
			}
			if !l.recordVerdict(decl(3, ka[:])) {
				t.Fatal("unknown dispatch or ordinal gap licensed elision")
			}
		}
	})
	t.Run("cold_upload_epoch_rollover_and_retired_reuse", func(t *testing.T) {
		var l vulkanBatchHazardLedger
		l.begin()
		l.recordVerdict(decl(2, qa[:]))
		// Production recordBatchDispatch observes a native epoch change and calls
		// begin. A cold upload/fenced reuse cannot carry a prior authorization.
		l.begin()
		if !l.recordVerdict(decl(0, ka[:])) {
			t.Fatal("new epoch inherited a verdict")
		}
		if !l.recordVerdict(decl(0, qa[:])) {
			t.Fatal("skipped/repeated ordinal licensed reuse")
		}
	})
	t.Run("metadata_exhaustion", func(t *testing.T) {
		var l vulkanBatchHazardLedger
		l.begin()
		for i := 0; i < 258; i++ {
			sync := l.recordVerdict(decl(i, qa[:]))
			if i >= 256 && !sync {
				t.Fatal("exhausted metadata licensed elision")
			}
		}
		if len(l.dispatches) != 257 {
			t.Fatal("unbounded exhausted ledger")
		}
	})
	t.Run("invalid_owners_ranges_and_in_dispatch_aliases", func(t *testing.T) {
		cases := []struct {
			name   string
			change func(*vulkanF32DecodeOperand)
		}{
			{"nil", func(a *vulkanF32DecodeOperand) { a.BufferID = 0 }},
			{"unknown_backing", func(a *vulkanF32DecodeOperand) { a.AllocationID = 0 }},
			{"short", func(a *vulkanF32DecodeOperand) { a.Bytes = 4 }},
			{"binding_overflow", func(a *vulkanF32DecodeOperand) { a.BindingOffset = math.MaxUint64 }},
			{"reservation_short", func(a *vulkanF32DecodeOperand) { a.ReservationBytes = 4 }},
			{"bad_shape", func(a *vulkanF32DecodeOperand) { a.Shape = []int{-1, 4} }},
			{"oversized_shape", func(a *vulkanF32DecodeOperand) { a.Shape = []int{int(^uint(0) >> 1), 4} }},
		}
		for _, tc := range cases {
			t.Run(tc.name, func(t *testing.T) {
				a := w
				tc.change(&a)
				if _, ok := vulkanF32DecodeMatMulAccesses(a, x, q, 8, 4); ok {
					t.Fatal("unsafe metadata admitted")
				}
			})
		}
		alias := q
		alias.AllocationID = w.AllocationID
		alias.BindingOffset = 0
		if _, ok := vulkanF32DecodeMatMulAccesses(w, x, alias, 8, 4); ok {
			t.Fatal("in-dispatch alias admitted")
		}
	})
	t.Run("source_admission_and_nondefault_caller", testVulkanDecodeCutSourceContract)
	t.Run("native_entrypoint_recorder_capture", testVulkanDecodeCutNativeCapture)
}

// These are source-contract assertions, not a claim to execute Session.Step.
// They deliberately check the AttentionBias guards which bypass Vulkan's fused
// interfaces: an ordinary bias-free decode does NOT select three plain MatMuls.
func testVulkanDecodeCutSourceContract(t *testing.T) {
	read := func(path string) string {
		b, e := os.ReadFile(path)
		if e != nil {
			t.Fatal(e)
		}
		return string(b)
	}
	section := func(s, from, to string) string {
		i := strings.Index(s, from)
		if i < 0 {
			t.Fatalf("missing %s", from)
		}
		s = s[i:]
		j := strings.Index(s, to)
		if j < 0 {
			t.Fatalf("missing %s", to)
		}
		return s[:j]
	}
	s := section(read("vulkan.go"), "func (v *vulkanBackend) MatMul(w, x Tensor) Tensor {", "// strixWave32DecodeGEMVRequested")
	f32 := section(s, "case F32:", "case Q8_0:")
	if strings.Count(s, "v.recordF32DecodeMatMul(") != 1 ||
		!strings.Contains(f32, "if VulkanHazardLoweringArmed(os.Getenv)") ||
		strings.Index(f32, "v.recordF32DecodeMatMul(") > strings.Index(f32, "C.fvk_matmul_f32(") {
		t.Fatal("declaration escaped gated F32 native seam")
	}
	adapter := read("vulkan_matmul_fused.go")
	adapter = adapter[strings.Index(adapter, "func (v *vulkanBackend) recordF32DecodeMatMul("):]
	for _, guard := range []string{"t.Dtype != F32", "t.Layout != RowMajor", "t.Quant != nil", "b.scalePtr != nil", "len(b.q8Chunks) != 0", "t.be != v", "v.recordBatchDispatch()", "C.fvk_buffer_reservation"} {
		if !strings.Contains(adapter, guard) {
			t.Fatalf("missing admission guard %s", guard)
		}
	}
	hal := read("../model/hal.go")
	attention := section(hal, "var q, kRaw, v compute.Tensor", "if cfg.AttentionBias {\n\t\t\t\tbe.AddBias")
	for _, proof := range []string{"be.(rmsNormMatMul3Backend); ok && !cfg.AttentionBias", "be.(matMul3Backend); ok && !cfg.AttentionBias", `q = be.MatMul(s.matWeightHAL(p("self_attn.q_proj.weight")), xn)`, `kRaw = be.MatMul(s.matWeightHAL(p("self_attn.k_proj.weight")), xn)`, `v = be.MatMul(s.matWeightHAL(p("self_attn.v_proj.weight")), xn)`} {
		if !strings.Contains(attention, proof) {
			t.Fatalf("ordinary caller source drift: %s", proof)
		}
	}
	record := section(read("vulkan.go"), "func (v *vulkanBackend) recordBatchDispatch", "// bufferIdentity")
	if !strings.Contains(record, "epoch != v.batchHazardEpoch") || !strings.Contains(record, "vulkanBatchLedger().begin()") {
		t.Fatal("native reopen no longer resets ledger")
	}
}
