package ggufload

import (
	"bytes"
	"math/rand"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/model"
)

func deviceQKExactCfg() model.Config {
	cfg := nativeRowsTestCfg(true)
	cfg.NumLayers = 64
	cfg.LayerTypes = make([]string, 64)
	for l := range cfg.LayerTypes {
		cfg.LayerTypes[l] = "linear_attention"
		if (l+1)%4 == 0 {
			cfg.LayerTypes[l] = "full_attention"
		}
	}
	return cfg
}

func TestQwen35DeviceQKNativeRowsAdmission(t *testing.T) {
	cfg := deviceQKExactCfg()
	if !model.Qwen38ExactMetalQ8Runtime(cfg) {
		t.Fatal("test geometry is not the exact Qwen3.8 runtime")
	}
	iq := q4kLoadOptions{residentDenseIQ: map[TensorType]bool{TensorIQ2_S: true}, residentDenseQ2K: true}
	q := "model.layers.3.self_attn.q_proj.weight"
	k := "model.layers.3.self_attn.k_proj.weight"
	qShape, kShape := []int{2 * 2 * 128, 256}, []int{128, 256}
	cases := []struct {
		name  string
		canon string
		typ   TensorType
		shape []int
		opts  q4kLoadOptions
		want  bool
	}{
		{"retained i-quant q_proj", q, TensorIQ2_S, qShape, iq, true},
		{"Q2_K k_proj", k, TensorQ2_K, kShape, iq, true},
		{"Q4_K q_proj", q, TensorQ4_K, qShape, iq, true},
		{"i-quant the backend does not retain", q, TensorIQ3_S, qShape, iq, false},
		{"no raw i-quant backend", q, TensorQ4_K, qShape, q4kLoadOptions{residentDenseQ2K: true}, false},
		{"GDN in_proj_qkv", "model.layers.0.linear_attn.in_proj_qkv.weight", TensorIQ2_S, []int{512, 256}, iq, true},
		{"GDN in_proj_z", "model.layers.0.linear_attn.in_proj_z.weight", TensorIQ2_S, []int{256, 256}, iq, true},
		{"GDN in_proj_b stays Q8", "model.layers.0.linear_attn.in_proj_b.weight", TensorIQ2_S, []int{4, 256}, iq, false},
		{"native rows forced off", q, TensorIQ2_S, qShape, q4kLoadOptions{residentDenseIQ: iq.residentDenseIQ, nativeRowsOff: true}, false},
	}
	for _, tc := range cases {
		if got := qwen35NativeRowResident(cfg, tc.canon, tc.typ, tc.shape, tc.opts); got != tc.want {
			t.Errorf("%s: qwen35NativeRowResident = %v, want %v", tc.name, got, tc.want)
		}
	}
	qkv := "model.layers.0.linear_attn.in_proj_qkv.weight"
	t.Setenv("FAK_QWEN35_NATIVE_QK_ROWS", "0")
	if qwen35NativeRowResident(cfg, q, TensorIQ2_S, qShape, iq) {
		t.Error("FAK_QWEN35_NATIVE_QK_ROWS=0 did not restore the Q8 route")
	}
	if !qwen35NativeRowResident(cfg, qkv, TensorIQ2_S, []int{512, 256}, iq) {
		t.Error("q/k kill switch also disabled the GDN in-proj class")
	}
	t.Setenv("FAK_QWEN35_NATIVE_GDN_INPROJ_ROWS", "0")
	if qwen35NativeRowResident(cfg, qkv, TensorIQ2_S, []int{512, 256}, iq) {
		t.Error("FAK_QWEN35_NATIVE_GDN_INPROJ_ROWS=0 did not restore the Q8 route")
	}
}

func TestQwen35DeviceQKNativeRowsStoreCanonicalRawRows(t *testing.T) {
	cfg := deviceQKExactCfg()
	q := "model.layers.3.self_attn.q_proj.weight"
	shape := []int{2 * 2 * 128, 256}
	raw := make([]byte, shape[0]*blockIQ2SBytes)
	rand.New(rand.NewSource(7)).Read(raw)
	want, err := normalizeQwen35NativeRows(q, shape, raw, cfg)
	if err != nil {
		t.Fatal(err)
	}
	if bytes.Equal(want, raw) {
		t.Fatal("gated rotary unpermute is identity; the witness would not see a missing reorder")
	}
	tw := tensorWork{pending: []pendingTensor{{resident: true, canonicalRows: true, deviceQKRows: true,
		residentType: TensorIQ2_S, name: q, shape: shape, raw: want}}}
	builder := model.NewQuantBuilder(cfg, false)
	if err := applyQ4KTensorWork(tw, nil, cfg, builder, map[int]glmKVBHalf{}, false); err != nil {
		t.Fatalf("applyQ4KTensorWork: %v", err)
	}
	f32Work := tensorWork{pending: []pendingTensor{{name: "model.layers.3.self_attn.o_proj.weight", shape: []int{256, 256}, f32: make([]float32, 256*256)}}}
	if err := applyQ4KTensorWork(f32Work, nil, cfg, builder, map[int]glmKVBHalf{}, false); err != nil {
		t.Fatalf("applyQ4KTensorWork(f32): %v", err)
	}
	m, err := builder.Build()
	if err != nil {
		t.Fatalf("Build: %v", err)
	}
	got, ok := m.KQuantRaw(q)
	if !ok {
		t.Fatal("device q_proj not stored in the raw k-quant store")
	}
	if !bytes.Equal(got, want) {
		t.Fatal("stored rows differ from the canonical raw row reorder")
	}
}
