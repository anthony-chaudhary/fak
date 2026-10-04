package ggufload

import (
	"strings"
	"testing"
)

// tied_q6k_embedding_test.go — fak#13567 load witness on a real tiny GGUF: a dense qwen35
// hybrid with a tied Q6_K token_embd.weight and NO output.weight loads the table ONCE (packed,
// shared by gather and head), with no f32 gather table and no duplicate Q8 head, and the
// admission estimate prices exactly what the loader retains. Gathered rows are checked against
// ggufload's own Q6_K dequantizer, an implementation independent of the model's row gather.

func tiedQ6KFixture(t *testing.T) (path string, dim, vocab int) {
	t.Helper()
	t.Setenv("FAK_W3_MLP", "")
	t.Setenv("FAK_GGUF_MMAP", "0")
	dim, vocab = 256, 4
	return buildQwen35GGUFFixture(t, "qwen35", dim, vocab, TensorQ6_K, TensorQ4_K, false, true), dim, vocab
}

func TestTiedQ6KEmbeddingDefaultLoadKeepsOnePackedCopy(t *testing.T) {
	path, dim, vocab := tiedQ6KFixture(t)
	ws, err := OpenWeights(path)
	if err != nil {
		t.Fatal(err)
	}
	defer ws.Close()
	cfg, err := ws.File.Config()
	if err != nil {
		t.Fatal(err)
	}
	if !cfg.TieWordEmbeddings || !cfg.IsQwen35Hybrid() || cfg.IsMoE() {
		t.Fatalf("fixture must be a tied dense Qwen hybrid: %+v", cfg)
	}

	m, err := LoadModelQ4KProfileOptions(path, nil)
	if err != nil {
		t.Fatalf("default tied Q6_K load: %v", err)
	}
	defer m.CloseWeights()
	const name = "model.embed_tokens.weight"
	if m.HasF32(name) || m.HasQ8(name) {
		t.Fatal("tied Q6_K load retained an f32 gather table or a duplicate Q8 head")
	}
	if !m.TiedQ6KEmbeddingShared() || m.Q2KEmbedding == nil || m.Q2KEmbedding.Format() != "Q6_K" {
		t.Fatalf("tied Q6_K table must be one shared packed copy, got shared=%v embed=%v", m.TiedQ6KEmbeddingShared(), m.Q2KEmbedding)
	}

	q6kBytes := int64(vocab * dim / 256 * blockQ6KBytes)
	const targetQ4 int64 = 3 * 256 * 256 / 256 * 144
	r := m.ResidentReport()
	if r.Q6KEmbedBytes != q6kBytes || r.TiedEmbedF32Bytes != 0 || r.TiedHeadQ8Bytes != 0 || r.F32Bytes != 0 || r.Q8Bytes != 0 {
		t.Fatalf("resident report %+v, want q6k_embed=%d and no f32/q8 copies", r, q6kBytes)
	}
	if r.TotalResidentBytes != targetQ4+q6kBytes {
		t.Fatalf("total resident = %d, want %d", r.TotalResidentBytes, targetQ4+q6kBytes)
	}
	if r.LMHead != "cpu-q6k" && r.LMHead != "metal-q6k" {
		t.Fatalf("lm_head route = %q, want cpu-q6k or metal-q6k", r.LMHead)
	}

	// Independent row reference: ggufload's dequantizer over the raw GGUF payload.
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

	plan, err := ws.EstimateQ4KLoadMemoryPlan()
	if err != nil {
		t.Fatalf("estimate: %v", err)
	}
	if plan.Total() != r.TotalResidentBytes {
		t.Fatalf("estimate=%d, loaded resident=%d", plan.Total(), r.TotalResidentBytes)
	}
}

func TestTiedQ6KEmbeddingOptOutAndDeclines(t *testing.T) {
	path, dim, vocab := tiedQ6KFixture(t)
	ws, err := OpenWeights(path)
	if err != nil {
		t.Fatal(err)
	}
	defer ws.Close()
	const name = "model.embed_tokens.weight"
	gatherF32 := int64(vocab * dim * 4)
	headQ8 := int64(vocab * dim / 32 * 36)

	// Explicit opt-out and a backend that cannot run resident Q6_K both keep the legacy layout,
	// and the estimate tracks it.
	for _, tc := range []struct {
		label string
		opts  []Q4KLoadOption
	}{
		{"opt-out", []Q4KLoadOption{WithTiedQ6KEmbeddingResident(false)}},
		{"no dense Q6_K residency", []Q4KLoadOption{WithDenseKQuantResident(false)}},
	} {
		m, err := LoadModelQ4KProfileOptions(path, nil, tc.opts...)
		if err != nil {
			t.Fatalf("%s load: %v", tc.label, err)
		}
		if !m.HasF32(name) || !m.HasQ8(name) || m.Q2KEmbedding != nil || m.TiedQ6KEmbeddingShared() {
			m.CloseWeights()
			t.Fatalf("%s must keep the legacy f32 gather + Q8 head", tc.label)
		}
		r := m.ResidentReport()
		m.CloseWeights()
		if r.TiedEmbedF32Bytes != gatherF32 || r.TiedHeadQ8Bytes != headQ8 || r.LMHead != "cpu-q8" {
			t.Fatalf("%s report f32=%d q8=%d head=%q, want %d/%d/cpu-q8", tc.label, r.TiedEmbedF32Bytes, r.TiedHeadQ8Bytes, r.LMHead, gatherF32, headQ8)
		}
		plan, err := ws.EstimateQ4KLoadMemoryPlan(tc.opts...)
		if err != nil {
			t.Fatalf("%s estimate: %v", tc.label, err)
		}
		if plan.Total() != r.TotalResidentBytes {
			t.Fatalf("%s estimate=%d, loaded resident=%d", tc.label, plan.Total(), r.TotalResidentBytes)
		}
	}

	// Explicit opt-in on an untied checkpoint is a typed refusal, never a silent fallback.
	untied := buildQwen35GGUFFixture(t, "qwen35", dim, vocab, TensorQ6_K, TensorQ6_K, false, false)
	if _, err := LoadModelQ4KProfileOptions(untied, nil, WithTiedQ6KEmbeddingResident(true)); err == nil || !strings.Contains(err.Error(), "tied") {
		t.Fatalf("opt-in on untied checkpoint: err=%v, want a tied-embedding refusal", err)
	}
	// And it conflicts with another packed embedding format.
	if _, err := LoadModelQ4KProfileOptions(path, nil, WithTiedQ6KEmbeddingResident(true), WithQ2KEmbeddingResident(true)); err == nil {
		t.Fatal("tied Q6_K + Q2_K packed embedding must conflict")
	}
}
