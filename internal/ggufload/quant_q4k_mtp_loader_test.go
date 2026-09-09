package ggufload

import (
	"bytes"
	"encoding/binary"
	"os"
	"path/filepath"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/model"
)

func TestQwen38MTPQ4KProductionLoaderRetainsAdmittedLayout(t *testing.T) {
	original := model.RetainMTP
	model.SetRetainMTP(true)
	t.Cleanup(func() { model.SetRetainMTP(original) })

	m, err := LoadModelQ4K(writeQwen38MTPQ4KFixture(t))
	if err != nil {
		t.Fatalf("load Qwen3.8 Q4_K_M fixture: %v", err)
	}
	layout, err := m.Qwen38MTPTensorLayout()
	if err != nil {
		t.Fatalf("admit retained MTP layout: %v", err)
	}
	if layout.Format != model.Qwen38MTPFormatQ4K {
		t.Fatalf("MTP layout=%q, want %q; tensors=%v", layout.Format, model.Qwen38MTPFormatQ4K, layout.TensorTypes)
	}
	if len(layout.TensorTypes) != 15 {
		t.Fatalf("retained MTP tensors=%d, want 15: %v", len(layout.TensorTypes), layout.TensorTypes)
	}
	var q4, f32 int
	for name, typ := range layout.TensorTypes {
		switch typ {
		case string(model.Qwen38MTPFormatQ4K):
			q4++
		case string(model.Qwen38MTPFormatF32):
			f32++
		default:
			t.Fatalf("retained MTP tensor %s has unsupported type %q", name, typ)
		}
	}
	if q4 != 8 || f32 != 7 {
		t.Fatalf("retained MTP layout has Q4_K=%d F32=%d, want 8/7", q4, f32)
	}
}

func writeQwen38MTPQ4KFixture(t *testing.T) string {
	t.Helper()
	const (
		align = 32
		h     = 256
		hd    = 64
	)
	tensors := []struct {
		name string
		dims []uint64
		typ  TensorType
	}{
		{"blk.2.nextn.eh_proj.weight", []uint64{2 * h, h}, TensorQ4_K},
		{"blk.2.nextn.enorm.weight", []uint64{h}, TensorF32},
		{"blk.2.nextn.hnorm.weight", []uint64{h}, TensorF32},
		{"blk.2.nextn.shared_head_norm.weight", []uint64{h}, TensorF32},
		{"blk.2.attn_norm.weight", []uint64{h}, TensorF32},
		{"blk.2.ffn_norm.weight", []uint64{h}, TensorF32},
		{"blk.2.attn_q_norm.weight", []uint64{hd}, TensorF32},
		{"blk.2.attn_k_norm.weight", []uint64{hd}, TensorF32},
		{"blk.2.attn_q.weight", []uint64{h, 2 * h}, TensorQ4_K},
		{"blk.2.attn_k.weight", []uint64{h, 2 * hd}, TensorQ4_K},
		{"blk.2.attn_v.weight", []uint64{h, 2 * hd}, TensorQ4_K},
		{"blk.2.attn_output.weight", []uint64{h, h}, TensorQ4_K},
		{"blk.2.ffn_gate.weight", []uint64{h, h}, TensorQ4_K},
		{"blk.2.ffn_up.weight", []uint64{h, h}, TensorQ4_K},
		{"blk.2.ffn_down.weight", []uint64{h, h}, TensorQ4_K},
	}

	offsets := make([]uint64, len(tensors))
	var payload bytes.Buffer
	for i, tensor := range tensors {
		padToAlignment(&payload, align)
		offsets[i] = uint64(payload.Len())
		elems := 1
		for _, dim := range tensor.dims {
			elems *= int(dim)
		}
		if tensor.typ == TensorQ4_K {
			payload.Write(make([]byte, elems/qkK*blockQ4KBytes))
			continue
		}
		for range elems {
			_ = binary.Write(&payload, binary.LittleEndian, float32(1))
		}
	}

	var fixture bytes.Buffer
	writeMinimalHeader(&fixture, uint64(len(tensors)), 11)
	writeKVUint32(&fixture, "general.alignment", align)
	writeKVString(&fixture, "general.architecture", "qwen35")
	writeKVUint64(&fixture, "qwen35.context_length", 16)
	writeKVUint64(&fixture, "qwen35.embedding_length", h)
	writeKVUint64(&fixture, "qwen35.block_count", 3)
	writeKVUint64(&fixture, "qwen35.feed_forward_length", h)
	writeKVUint64(&fixture, "qwen35.attention.head_count", 4)
	writeKVUint64(&fixture, "qwen35.attention.head_count_kv", 2)
	writeKVFloat32(&fixture, "qwen35.attention.layer_norm_rms_epsilon", 1e-5)
	writeKVUint64(&fixture, "qwen35.full_attention_interval", 2)
	writeKVUint64(&fixture, "qwen35.nextn_predict_layers", 1)
	for i, tensor := range tensors {
		writeTensorInfoForTest(&fixture, tensor.name, tensor.dims, tensor.typ, offsets[i])
	}
	padToAlignment(&fixture, align)
	fixture.Write(payload.Bytes())

	path := filepath.Join(t.TempDir(), "qwen38-nextn-q4_k_m.gguf")
	if err := os.WriteFile(path, fixture.Bytes(), 0o644); err != nil {
		t.Fatalf("write Qwen3.8 Q4_K_M fixture: %v", err)
	}
	return path
}
