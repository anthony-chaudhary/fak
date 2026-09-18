package ggufload

import (
	"bytes"
	"os"
	"path/filepath"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/model"
)

// deepseek41_ffn_experts_seam_test.go — the #13271 regression guard.
//
// The pinned vcruz Q2_K DeepSeek-V4.1 artifact stores batched routed experts as
// blk.<L>.ffn_{gate,up,down}_exps.weight (model shape [E,out,in]). The loader 1->E
// splitter used to emit the deepseek2/GLM per-expert spelling
// model.layers.<L>.mlp.experts.<e>.{gate,up,down}_proj.weight for EVERY arch —
// including deepseek41 — while the native non-MLA V4.1 forward
// (internal/model/v41_forward.go:700-710 admit, :1388-1391 read) admits and reads
// model.layers.<L>.ffn.experts.<e>.{w1,w3,w2}.weight (gate_proj->w1, up_proj->w3,
// down_proj->w2). The mismatch meant:
//
//   - the forward's v41AdmitShape loop refused the model by name, and
//   - model.ResidentKQuantEligible was asked about a name the V4 forward never
//     reads, so the whole Q2_K routed-expert bulk fell back to eager f32 dequant
//     (the ~366 GiB OOM the streaming/resident spine exists to avoid).
//
// The splitter is now arch-aware (batchedExpertCanonicalName). These tests pin the
// emitted name, the residency gate, and the runtime offload placement, with a
// negative arm proving the GLM spelling is untouched for every other arch.

// TestDeepSeek41BatchedExpertCanonicalName pins the arch-aware name composer
// directly: deepseek41 maps every GGUF projection onto the V4 w1/w3/w2 leaves,
// and every other arch keeps the deepseek2/GLM spelling byte-for-byte.
func TestDeepSeek41BatchedExpertCanonicalName(t *testing.T) {
	cases := []struct {
		arch   string
		layer  int
		expert int
		proj   string
		want   string
	}{
		{"deepseek41", 0, 0, "gate_proj", "model.layers.0.ffn.experts.0.w1.weight"},
		{"deepseek41", 0, 0, "up_proj", "model.layers.0.ffn.experts.0.w3.weight"},
		{"deepseek41", 0, 0, "down_proj", "model.layers.0.ffn.experts.0.w2.weight"},
		{"deepseek41", 39, 255, "gate_proj", "model.layers.39.ffn.experts.255.w1.weight"},
		{"deepseek41", 7, 3, "down_proj", "model.layers.7.ffn.experts.3.w2.weight"},
		// negative arm: the GLM/deepseek2/qwen3moe spelling must be unchanged.
		{"glm_moe_dsa", 0, 0, "gate_proj", "model.layers.0.mlp.experts.0.gate_proj.weight"},
		{"glm_moe_dsa", 5, 2, "up_proj", "model.layers.5.mlp.experts.2.up_proj.weight"},
		{"deepseek2", 1, 1, "down_proj", "model.layers.1.mlp.experts.1.down_proj.weight"},
		{"qwen3moe", 2, 4, "gate_proj", "model.layers.2.mlp.experts.4.gate_proj.weight"},
	}
	for _, c := range cases {
		if got := batchedExpertCanonicalName(c.arch, c.layer, c.expert, c.proj); got != c.want {
			t.Errorf("batchedExpertCanonicalName(%q,%d,%d,%q) = %q, want %q",
				c.arch, c.layer, c.expert, c.proj, got, c.want)
		}
	}
}

// TestDeepSeek41SplitGLMMoeDsaExpertsEmitsFFNNames drives the F32 splitter itself:
// a batched [E,out,in] deepseek41 blob must split 1->E into the native
// ffn.experts.<e>.{w1,w3,w2} names the V4.1 forward reads, with the per-expert
// payload still the correct contiguous slice.
func TestDeepSeek41SplitGLMMoeDsaExpertsEmitsFFNNames(t *testing.T) {
	const e, out, in = 2, 3, 2
	per := out * in
	data := sequenceF32ForTest(0, e*per)

	tests := []struct {
		proj string
		leaf string
	}{
		{"gate_proj", "w1"},
		{"up_proj", "w3"},
		{"down_proj", "w2"},
	}
	for _, tc := range tests {
		got, err := splitGLMMoeDsaExperts("deepseek41", 4, tc.proj, []int{e, out, in}, data)
		if err != nil {
			t.Fatalf("splitGLMMoeDsaExperts(deepseek41, %s): %v", tc.proj, err)
		}
		if len(got) != e {
			t.Fatalf("splitGLMMoeDsaExperts(deepseek41, %s) returned %d experts, want %d", tc.proj, len(got), e)
		}
		for x := 0; x < e; x++ {
			wantName := "model.layers.4.ffn.experts." + itoaForTest(x) + "." + tc.leaf + ".weight"
			if got[x].Name != wantName {
				t.Errorf("expert %d %s name = %q, want the native V4 leaf %q", x, tc.proj, got[x].Name, wantName)
			}
			want := data[x*per : (x+1)*per]
			for i := range want {
				if got[x].Data[i] != want[i] {
					t.Fatalf("expert %d %s payload diverged at %d: got %v want %v", x, tc.proj, i, got[x].Data[i], want[i])
				}
			}
		}
	}

	// The same split for glm_moe_dsa must keep the mlp.experts spelling.
	glm, err := splitGLMMoeDsaExperts("glm_moe_dsa", 4, "gate_proj", []int{e, out, in}, data)
	if err != nil {
		t.Fatalf("splitGLMMoeDsaExperts(glm_moe_dsa): %v", err)
	}
	if glm[0].Name != "model.layers.4.mlp.experts.0.gate_proj.weight" {
		t.Errorf("glm split name = %q, want the unchanged mlp.experts spelling", glm[0].Name)
	}
}

// TestDeepSeek41SplitRawQuantEmitsFFNNames is the raw-resident twin: the name the
// raw Q2_K/Q4_K splitter emits for deepseek41 MUST already be the V4 name BEFORE
// model.ResidentKQuantEligible is consulted (quant_q4k_loader.go:1113), or the
// residency gate answers about a name the forward never reads and the expert bulk
// eager-dequantizes.
func TestDeepSeek41SplitRawQuantEmitsFFNNames(t *testing.T) {
	const e, out, in = 2, 256, 256
	per := out * in
	raw := make([]byte, 0, e*(per/qkK*blockQ4KBytes))
	for i := 0; i < e*(per/qkK)*blockQ4KBytes; i++ {
		raw = append(raw, byte(i))
	}
	got, aligned, err := splitGLMMoeDsaExpertsRawQuant("deepseek41", 3, "gate_proj", []int{e, out, in}, raw, qkK, blockQ4KBytes)
	if err != nil {
		t.Fatalf("splitGLMMoeDsaExpertsRawQuant(deepseek41): %v", err)
	}
	if !aligned {
		t.Fatalf("splitGLMMoeDsaExpertsRawQuant(deepseek41) aligned=false for a 256-aligned reduction dim")
	}
	for x := 0; x < e; x++ {
		wantName := "model.layers.3.ffn.experts." + itoaForTest(x) + ".w1.weight"
		if got[x].Name != wantName {
			t.Errorf("raw expert %d name = %q, want the native V4 leaf %q", x, got[x].Name, wantName)
		}
	}
}

// TestDeepSeek41ResidentKQuantEligibleFFNExperts pins the residency gate on the
// names the V4.1 forward actually reads: with the new isQuantWeight arms a
// deepseek41 routed expert is eligible for the resident raw-quant store, and a
// NON-deepseek41 config must not change (the mlp spelling stays eligible, the
// ffn spelling is not magically admitted for other arches).
func TestDeepSeek41ResidentKQuantEligibleFFNExperts(t *testing.T) {
	cfg := model.Config{ModelType: "deepseek41"}
	names := []string{
		"model.layers.0.ffn.experts.0.w1.weight",
		"model.layers.0.ffn.experts.0.w3.weight",
		"model.layers.0.ffn.experts.0.w2.weight",
		"model.layers.39.ffn.experts.255.w1.weight",
	}
	for _, n := range names {
		if !model.ResidentKQuantEligible(cfg, n) {
			t.Errorf("ResidentKQuantEligible(deepseek41, %q)=false, want true (the Q2_K routed-expert bulk must stay resident)", n)
		}
	}

	// Negative arm: the GLM spelling stays eligible (unchanged). Note
	// ResidentKQuantEligible is a NAME gate, not an arch gate — the arch-specific
	// behavior lives in batchedExpertCanonicalName, which emits the V4 leaf only for
	// deepseek41 (pinned above). The loader cannot emit the ffn.experts spelling for
	// another arch, so admitting the name here is harmless and keeps the two
	// checks (emit + hold) on the same canonical name.
	glm := model.Config{ModelType: "glm_moe_dsa"}
	if !model.ResidentKQuantEligible(glm, "model.layers.0.mlp.experts.0.gate_proj.weight") {
		t.Error("ResidentKQuantEligible(glm_moe_dsa, mlp.experts.0.gate_proj.weight)=false, want true (unchanged)")
	}
}

// TestDeepSeek41GGUFExpertSplitE2E loads a minimal deepseek41 GGUF carrying a
// batched routed-expert blob through the real F32Tensors path and asserts the
// emitted per-expert names are the native ffn.experts.<e>.w{1,3,2} leaves, not
// the mlp.experts spelling. This is the loader half of the #13271 fix.
func TestDeepSeek41GGUFExpertSplitE2E(t *testing.T) {
	const E, I, H = 2, 3, 2
	perGate := I * H
	perDown := H * I
	gate := sequenceF32ForTest(10, E*perGate)
	up := sequenceF32ForTest(100, E*perGate)
	down := sequenceF32ForTest(200, E*perDown)

	path := filepath.Join(t.TempDir(), "deepseek41_experts.gguf")
	if err := os.WriteFile(path, deepseek41ExpertGGUF(E, I, H, gate, up, down), 0o644); err != nil {
		t.Fatalf("write fixture: %v", err)
	}
	ws, err := OpenWeights(path)
	if err != nil {
		t.Fatalf("OpenWeights: %v", err)
	}
	defer ws.Close()

	cfg, tensors, err := ws.F32Tensors()
	if err != nil {
		t.Fatalf("F32Tensors: %v", err)
	}
	if cfg.ModelType != "deepseek41" {
		t.Fatalf("ModelType=%q, want deepseek41", cfg.ModelType)
	}
	byName := map[string]modelTensorForTest{}
	for _, tt := range tensors {
		byName[tt.Name] = modelTensorForTest{shape: tt.Shape, data: tt.Data}
	}
	for x := 0; x < E; x++ {
		assertModelTensorForTest(t, byName,
			"model.layers.0.ffn.experts."+itoaForTest(x)+".w1.weight", []int{I, H}, gate[x*perGate:(x+1)*perGate])
		assertModelTensorForTest(t, byName,
			"model.layers.0.ffn.experts."+itoaForTest(x)+".w3.weight", []int{I, H}, up[x*perGate:(x+1)*perGate])
		assertModelTensorForTest(t, byName,
			"model.layers.0.ffn.experts."+itoaForTest(x)+".w2.weight", []int{H, I}, down[x*perDown:(x+1)*perDown])
	}
	if _, ok := byName["model.layers.0.mlp.experts.0.gate_proj.weight"]; ok {
		t.Error("the splitter still emitted the mlp.experts spelling for deepseek41 — the V4.1 forward would refuse it by name")
	}
}

// TestDeepSeek41LoadTimePlannerExpertClassification pins the LOAD-TIME planner
// predicates on the V4.1 native names: isSharedExpertCanonicalName must recognize
// ffn.shared_experts.* (mirroring model.isSharedExpertWeight) and
// tensorCPUOffloadExpert must classify a batched ffn.experts slab as expert-scoped
// so the offload byte partition and device-fit plan are correct for the Q2_K
// artifact (fak#13271).
func TestDeepSeek41LoadTimePlannerExpertClassification(t *testing.T) {
	if !isSharedExpertCanonicalName("model.layers.0.ffn.shared_experts.w1.weight") {
		t.Error("isSharedExpertCanonicalName(ffn.shared_experts.w1.weight)=false, want true (mirror of isSharedExpertWeight)")
	}
	if isSharedExpertCanonicalName("model.layers.0.ffn.experts.0.w1.weight") {
		t.Error("isSharedExpertCanonicalName(ffn.experts.0.w1.weight)=true — routed and shared must stay disjoint")
	}
	// tensorCPUOffloadExpert takes the RAW GGUF name and maps it through the arch.
	offload, err := tensorCPUOffloadExpert("blk.0.ffn_gate_exps.weight", "deepseek41")
	if err != nil {
		t.Fatalf("tensorCPUOffloadExpert(deepseek41): %v", err)
	}
	if !offload {
		t.Error("tensorCPUOffloadExpert(blk.0.ffn_gate_exps.weight, deepseek41)=false, want true (the routed-expert bulk must be host-scoped)")
	}
}

// deepseek41ExpertGGUF assembles a minimal single-layer deepseek41 GGUF carrying
// only the router gate and the three batched routed-expert blobs (F32). Enough for
// the loader's batched-expert branch; no attention/shared tensors are needed to
// observe the 1->E name emission.
func deepseek41ExpertGGUF(E, I, H int, gate, up, down []float32) []byte {
	type tens struct {
		name string
		dims []uint64
		data []float32
	}
	tensors := []tens{
		{name: "blk.0.ffn_gate_inp.weight", dims: []uint64{uint64(H), uint64(E)}, data: sequenceF32ForTest(300, E*H)},
		{name: "blk.0.ffn_gate_exps.weight", dims: []uint64{uint64(H), uint64(I), uint64(E)}, data: gate},
		{name: "blk.0.ffn_up_exps.weight", dims: []uint64{uint64(H), uint64(I), uint64(E)}, data: up},
		{name: "blk.0.ffn_down_exps.weight", dims: []uint64{uint64(I), uint64(H), uint64(E)}, data: down},
	}
	align := func(x int) int { return (x + 31) / 32 * 32 }
	nvals := func(dims []uint64) int {
		n := 1
		for _, d := range dims {
			n *= int(d)
		}
		return n
	}
	offsets := make([]uint64, len(tensors))
	off := 0
	for i, tt := range tensors {
		offsets[i] = uint64(off)
		off = align(off + nvals(tt.dims)*4)
	}

	var b bytes.Buffer
	writeMinimalHeader(&b, uint64(len(tensors)), 10)
	writeKVString(&b, "general.architecture", "deepseek41")
	writeKVUint32(&b, "general.alignment", 32)
	writeKVUint32(&b, "deepseek41.embedding_length", uint32(H))
	writeKVUint32(&b, "deepseek41.block_count", 1)
	writeKVUint32(&b, "deepseek41.attention.head_count", 1)
	writeKVUint32(&b, "deepseek41.feed_forward_length", uint32(I))
	writeKVFloat32(&b, "deepseek41.attention.layer_norm_rms_epsilon", 1e-6)
	writeKVUint32(&b, "deepseek41.expert_count", uint32(E))
	writeKVUint32(&b, "deepseek41.expert_used_count", 1)
	writeKVUint32(&b, "deepseek41.expert_feed_forward_length", uint32(I))
	for i, tt := range tensors {
		writeTensorInfoForTest(&b, tt.name, tt.dims, TensorF32, offsets[i])
	}
	padToAlignment(&b, 32)
	for _, tt := range tensors {
		if len(tt.data) != nvals(tt.dims) {
			panic("deepseek41ExpertGGUF: bad tensor data length")
		}
		for _, v := range tt.data {
			writeF32ForTest(&b, v)
		}
		padToAlignment(&b, 32)
	}
	return b.Bytes()
}
