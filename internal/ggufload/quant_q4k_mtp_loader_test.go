package ggufload

import (
	"bytes"
	"encoding/binary"
	"math"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/model"
)

type qwen38MTPFixtureTensor struct {
	name string
	dims []uint64
	typ  TensorType
}

func qwen38MTPFixture(t *testing.T, omit string, override map[string]TensorType) *WeightSource {
	t.Helper()
	const h = 256
	tensors := []qwen38MTPFixtureTensor{
		{"blk.1.nextn.eh_proj.weight", []uint64{2 * h, h}, TensorQ8_0},
		{"blk.1.nextn.enorm.weight", []uint64{h}, TensorF32},
		{"blk.1.nextn.hnorm.weight", []uint64{h}, TensorF32},
		{"blk.1.nextn.shared_head_norm.weight", []uint64{h}, TensorF32},
		{"blk.1.attn_norm.weight", []uint64{h}, TensorF32},
		{"blk.1.ffn_norm.weight", []uint64{h}, TensorF32},
		{"blk.1.attn_q_norm.weight", []uint64{64}, TensorF32},
		{"blk.1.attn_k_norm.weight", []uint64{64}, TensorF32},
		{"blk.1.attn_q.weight", []uint64{h, 2 * h}, TensorQ4_K},
		{"blk.1.attn_k.weight", []uint64{h, 128}, TensorQ4_K},
		{"blk.1.attn_v.weight", []uint64{h, 128}, TensorQ6_K},
		{"blk.1.attn_output.weight", []uint64{h, h}, TensorQ4_K},
		{"blk.1.ffn_gate.weight", []uint64{h, h}, TensorQ4_K},
		{"blk.1.ffn_up.weight", []uint64{h, h}, TensorQ4_K},
		{"blk.1.ffn_down.weight", []uint64{h, h}, TensorQ6_K},
		{"output.weight", []uint64{h, h}, TensorQ6_K},
	}
	meta := synthQwen35Meta(2, 1)
	meta["qwen35.embedding_length"] = Value{Type: TypeUint64, Value: uint64(h)}
	meta["qwen35.feed_forward_length"] = Value{Type: TypeUint64, Value: uint64(h)}
	tokens := make([]Value, h)
	for i := range tokens {
		tokens[i] = Value{Type: TypeString, Value: "t"}
	}
	meta["tokenizer.ggml.tokens"] = Value{Type: TypeArray, Value: tokens}

	var payload bytes.Buffer
	infos := make([]TensorInfo, 0, len(tensors))
	for _, tensor := range tensors {
		if tensor.name == omit {
			continue
		}
		if typ, ok := override[tensor.name]; ok {
			tensor.typ = typ
		}
		info := TensorInfo{Name: tensor.name, Dims: tensor.dims, Type: tensor.typ, FileOffset: int64(payload.Len())}
		n, err := tensorPayloadBytes(info)
		if err != nil {
			t.Fatalf("tensorPayloadBytes(%s): %v", tensor.name, err)
		}
		raw := make([]byte, int(n))
		if tensor.typ == TensorF32 {
			for off := 0; off < len(raw); off += 4 {
				binary.LittleEndian.PutUint32(raw[off:], math.Float32bits(1))
			}
		}
		// Give q/k every independently quantized row a distinct marker. The
		// loader must preserve each block while moving the rows into model order.
		if tensor.name == "blk.1.attn_q.weight" || tensor.name == "blk.1.attn_k.weight" {
			shape, err := modelShapeFromGGUFDims(tensor.name, tensor.dims)
			if err != nil {
				t.Fatal(err)
			}
			rowBytes := len(raw) / shape[0]
			for row := 0; row < shape[0]; row++ {
				for i := row * rowBytes; i < (row+1)*rowBytes; i++ {
					raw[i] = byte(row)
				}
			}
		}
		infos = append(infos, info)
		payload.Write(raw)
	}
	ws, err := NewWeightSource(&File{Metadata: meta, Tensors: infos}, bytes.NewReader(payload.Bytes()), int64(payload.Len()))
	if err != nil {
		t.Fatalf("NewWeightSource: %v", err)
	}
	return ws
}

func TestQ4KLoaderRetainsExactQwen38MTPMixedLayout(t *testing.T) {
	orig := model.RetainMTP
	defer func() { model.RetainMTP = orig }()
	model.RetainMTP = true

	m, err := qwen38MTPFixture(t, "", nil).QuantModelQ4K()
	if err != nil {
		t.Fatalf("QuantModelQ4K: %v", err)
	}
	if m.Cfg.NumLayers != 1 || m.Cfg.NumMTPLayers() != 1 {
		t.Fatalf("target/MTP layers=(%d,%d), want (1,1)", m.Cfg.NumLayers, m.Cfg.NumMTPLayers())
	}
	layout, err := m.Qwen38MTPTensorLayout()
	if err != nil {
		t.Fatalf("Qwen38MTPTensorLayout: %v", err)
	}
	if layout.Format != model.Qwen38MTPFormatQ4K {
		t.Fatalf("layout format=%q, want Q4_K", layout.Format)
	}
	for name, want := range map[string]string{
		"mtp.fc.weight":                        "Q8_0",
		"mtp.layers.0.self_attn.q_proj.weight": "Q4_K",
		"mtp.layers.0.self_attn.k_proj.weight": "Q4_K",
		"mtp.layers.0.self_attn.v_proj.weight": "Q6_K",
		"mtp.layers.0.self_attn.o_proj.weight": "Q4_K",
		"mtp.layers.0.mlp.gate_proj.weight":    "Q4_K",
		"mtp.layers.0.mlp.up_proj.weight":      "Q4_K",
		"mtp.layers.0.mlp.down_proj.weight":    "Q6_K",
		"lm_head.weight":                       "Q6_K",
	} {
		if got := layout.TensorTypes[name]; got != want {
			t.Errorf("%s format=%q, want %q", name, got, want)
		}
	}
	for _, name := range []string{
		"mtp.pre_fc_norm_embedding.weight", "mtp.pre_fc_norm_hidden.weight", "mtp.norm.weight",
		"mtp.layers.0.input_layernorm.weight", "mtp.layers.0.post_attention_layernorm.weight",
		"mtp.layers.0.self_attn.q_norm.weight", "mtp.layers.0.self_attn.k_norm.weight",
	} {
		if got := layout.TensorTypes[name]; got != "F32" {
			t.Errorf("%s format=%q, want F32", name, got)
		}
	}
	qRaw, ok := m.Q4KRaw("mtp.layers.0.self_attn.q_proj.weight")
	if !ok {
		t.Fatal("reordered MTP q projection is not resident Q4_K")
	}
	qRowBytes := len(qRaw) / 512
	for row, wantMarker := range map[int]byte{0: 0, 1: 2, 32: 1, 64: 64} {
		if got := qRaw[row*qRowBytes]; got != wantMarker {
			t.Errorf("reordered q row %d marker=%d, want source-row marker %d", row, got, wantMarker)
		}
	}
	kRaw, ok := m.Q4KRaw("mtp.layers.0.self_attn.k_proj.weight")
	if !ok {
		t.Fatal("reordered MTP k projection is not resident Q4_K")
	}
	kRowBytes := len(kRaw) / 128
	for row, wantMarker := range map[int]byte{0: 0, 1: 2, 32: 1} {
		if got := kRaw[row*kRowBytes]; got != wantMarker {
			t.Errorf("reordered k row %d marker=%d, want source-row marker %d", row, got, wantMarker)
		}
	}
}

func TestQ4KLoaderQwen38MTPFailsClosed(t *testing.T) {
	orig := model.RetainMTP
	defer func() { model.RetainMTP = orig }()
	model.RetainMTP = true

	t.Run("incomplete", func(t *testing.T) {
		_, err := qwen38MTPFixture(t, "blk.1.ffn_down.weight", nil).QuantModelQ4K()
		if err == nil || !strings.Contains(err.Error(), "incomplete retained Qwen MTP head") {
			t.Fatalf("error=%v, want incomplete retained head", err)
		}
	})
	t.Run("wrong type", func(t *testing.T) {
		_, err := qwen38MTPFixture(t, "", map[string]TensorType{"blk.1.nextn.eh_proj.weight": TensorQ4_K}).QuantModelQ4K()
		if err == nil || !strings.Contains(err.Error(), "mtp.fc.weight has type Q4_K, want Q8_0") {
			t.Fatalf("error=%v, want exact-type refusal", err)
		}
	})
}

func TestQ4KLoaderQwen38MTPDefaultStillDropsSidecar(t *testing.T) {
	orig := model.RetainMTP
	defer func() { model.RetainMTP = orig }()
	model.RetainMTP = false

	m, err := qwen38MTPFixture(t, "", nil).QuantModelQ4K()
	if err != nil {
		t.Fatalf("QuantModelQ4K target-only: %v", err)
	}
	if m.Cfg.NumLayers != 1 {
		t.Fatalf("target layers=%d, want 1", m.Cfg.NumLayers)
	}
	if m.HasQ4K("mtp.layers.0.self_attn.q_proj.weight") || m.HasKQuant("mtp.fc.weight") {
		t.Fatal("target-only load retained MTP sidecar storage")
	}
}
