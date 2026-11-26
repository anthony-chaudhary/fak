package ggufload

import (
	"strings"
	"testing"
)

// Admission table for the tied Q6_K token table across families, on real tiny GGUFs: the
// default selects it only for a Qwen3.5 hybrid, an explicit opt-in also admits dense Qwen2, and
// every other family or an untied checkpoint is a typed refusal. ResolvesTiedQ6KEmbedding and
// the admission estimate must agree with what each load actually retains.

func TestTiedQ6KEmbeddingAdmissionTable(t *testing.T) {
	t.Setenv("FAK_W3_MLP", "")
	t.Setenv("FAK_GGUF_MMAP", "0")
	const dim, vocab = 256, 4
	optIn := []Q4KLoadOption{WithTiedQ6KEmbeddingResident(true)}
	for _, tc := range []struct {
		label     string
		arch      string
		untied    bool
		opts      []Q4KLoadOption
		shared    bool
		refusal   string
		wantModel string
	}{
		{label: "qwen35 default", arch: "qwen35", shared: true, wantModel: "qwen35"},
		{label: "qwen2 default", arch: "qwen2", shared: false, wantModel: "qwen2"},
		{label: "qwen2 opt-in", arch: "qwen2", opts: optIn, shared: true, wantModel: "qwen2"},
		{label: "qwen2 opt-out", arch: "qwen2", opts: []Q4KLoadOption{WithTiedQ6KEmbeddingResident(false)}, wantModel: "qwen2"},
		{label: "qwen2 opt-in no dense Q6_K", arch: "qwen2", opts: []Q4KLoadOption{WithTiedQ6KEmbeddingResident(true), WithDenseKQuantResident(false)}, shared: true, wantModel: "qwen2"},
		{label: "qwen2 untied opt-in", arch: "qwen2", untied: true, opts: optIn, refusal: "tied", wantModel: "qwen2"},
		{label: "llama opt-in", arch: "llama", opts: optIn, refusal: "dense Qwen2", wantModel: "llama"},
		{label: "llama default", arch: "llama", wantModel: "llama"},
	} {
		path := buildQwen35GGUFFixture(t, tc.arch, dim, vocab, TensorQ6_K, TensorQ6_K, false, !tc.untied)
		ws, err := OpenWeights(path)
		if err != nil {
			t.Fatalf("%s: open: %v", tc.label, err)
		}
		cfg, err := ws.File.Config()
		if err != nil {
			ws.Close()
			t.Fatalf("%s: config: %v", tc.label, err)
		}
		if cfg.ModelType != tc.wantModel || cfg.IsMoE() || cfg.TieWordEmbeddings == tc.untied {
			ws.Close()
			t.Fatalf("%s: fixture model=%q moe=%v tied=%v", tc.label, cfg.ModelType, cfg.IsMoE(), cfg.TieWordEmbeddings)
		}

		m, err := LoadModelQ4KProfileOptions(path, nil, tc.opts...)
		if tc.refusal != "" {
			ws.Close()
			if err == nil {
				m.CloseWeights()
				t.Fatalf("%s: load admitted, want refusal containing %q", tc.label, tc.refusal)
			}
			if !strings.Contains(err.Error(), tc.refusal) {
				t.Fatalf("%s: err=%v, want refusal containing %q", tc.label, err, tc.refusal)
			}
			continue
		}
		if err != nil {
			ws.Close()
			t.Fatalf("%s: load: %v", tc.label, err)
		}
		shared := m.TiedQ6KEmbeddingShared()
		const name = "model.embed_tokens.weight"
		legacy := m.HasF32(name) && m.HasQ8(name) && m.Q2KEmbedding == nil
		r := m.ResidentReport()
		m.CloseWeights()
		if shared != tc.shared || (!tc.shared && !legacy) {
			ws.Close()
			t.Fatalf("%s: shared=%v legacy=%v, want shared=%v", tc.label, shared, legacy, tc.shared)
		}
		if got := ws.ResolvesTiedQ6KEmbedding(cfg, tc.opts...); got != shared {
			ws.Close()
			t.Fatalf("%s: ResolvesTiedQ6KEmbedding=%v, loader shared=%v", tc.label, got, shared)
		}
		plan, err := ws.EstimateQ4KLoadMemoryPlan(tc.opts...)
		ws.Close()
		if err != nil {
			t.Fatalf("%s: estimate: %v", tc.label, err)
		}
		if plan.Total() != r.TotalResidentBytes {
			t.Fatalf("%s: estimate=%d, loaded resident=%d", tc.label, plan.Total(), r.TotalResidentBytes)
		}
	}
}

// A Qwen2 opt-in load keeps the GGUF payload once, and its rows equal ggufload's own Q6_K
// dequantizer over that payload, independent of the model's gather.
func TestTiedQ6KEmbeddingQwen2OptInGatherMatchesIndependentDequant(t *testing.T) {
	t.Setenv("FAK_W3_MLP", "")
	t.Setenv("FAK_GGUF_MMAP", "0")
	const dim, vocab = 256, 4
	path := buildQwen35GGUFFixture(t, "qwen2", dim, vocab, TensorQ6_K, TensorQ6_K, false, true)
	ws, err := OpenWeights(path)
	if err != nil {
		t.Fatal(err)
	}
	defer ws.Close()
	m, err := LoadModelQ4KProfileOptions(path, nil, WithTiedQ6KEmbeddingResident(true))
	if err != nil {
		t.Fatalf("qwen2 opt-in load: %v", err)
	}
	defer m.CloseWeights()
	r := m.ResidentReport()
	if q6k := int64(vocab * dim / 256 * blockQ6KBytes); r.Q6KEmbedBytes != q6k || r.TiedEmbedF32Bytes != 0 || r.TiedHeadQ8Bytes != 0 {
		t.Fatalf("resident report %+v, want q6k_embed=%d and no f32/q8 tied copies", r, q6k)
	}
	if r.LMHead != "cpu-q6k" && r.LMHead != "metal-q6k" {
		t.Fatalf("lm_head route = %q, want cpu-q6k or metal-q6k", r.LMHead)
	}
	info, ok := ws.Tensor("token_embd.weight")
	if !ok {
		t.Fatal("token_embd.weight missing")
	}
	_, raw, err := ws.shapeAndBytes(info)
	if err != nil {
		t.Fatal(err)
	}
	table, err := dequantF32(info, raw)
	if err != nil {
		t.Fatal(err)
	}
	row := make([]float32, dim)
	for id := 0; id < vocab; id++ {
		if err := m.Q2KEmbedding.GatherRow(id, row, 1); err != nil {
			t.Fatal(err)
		}
		for i, v := range row {
			if want := table[id*dim+i]; v != want {
				t.Fatalf("gather row %d[%d] = %v, independent dequant %v", id, i, v, want)
			}
		}
	}
}
