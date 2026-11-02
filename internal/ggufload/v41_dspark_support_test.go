package ggufload

import (
	"errors"
	"fmt"
	"os"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/model"
)

// supportGGUFFixture independently spells the pinned ds4 converter's 81
// destination roles and published dimensions. It contains no weight payload.
func supportGGUFFixture(t *testing.T) *File {
	t.Helper()
	raw, err := os.ReadFile("../model/testdata/deepseek_v41_flash_config.json")
	if err != nil {
		t.Fatal(err)
	}
	f := &File{Metadata: map[string]Value{}}
	for key, value := range map[string]string{
		"general.architecture": "deepseek41-dspark", "dspark.target_architecture": "deepseek41",
		"dspark.source_config": string(raw), "dspark.weight_encoding": "native MXFP4; lossless F16/F32 expansion",
	} {
		f.Metadata[key] = Value{TypeString, value}
	}
	for key, value := range map[string]uint32{"block_size": 5, "markov_rank": 256, "noise_token_id": 128799, "n_routed_experts": 128, "num_experts_per_tok": 3} {
		f.Metadata["dspark."+key] = Value{TypeUint32, value}
	}
	f.Metadata["dspark.target_layer_ids"] = Value{TypeArray, []Value{{TypeUint32, uint32(37)}, {TypeUint32, uint32(38)}, {TypeUint32, uint32(39)}}}
	add := func(name string, typ TensorType, dims ...uint64) {
		f.Tensors = append(f.Tensors, TensorInfo{Name: name, Type: typ, Dims: dims})
	}
	for stage := 0; stage < 3; stage++ {
		p := fmt.Sprintf("mtp.%d.", stage)
		for _, site := range []string{"attn", "ffn"} {
			add(p+"hc_"+site+"_fn.weight", TensorF32, 20480, 24)
			add(p+"hc_"+site+"_base.weight", TensorF32, 24)
			add(p+"hc_"+site+"_scale.weight", TensorF32, 3)
			add(p+site+"_norm.weight", TensorF32, 5120)
		}
		for _, x := range []struct {
			name string
			dims []uint64
		}{
			{"attn_sinks.weight", []uint64{64}}, {"attn_q_a.weight", []uint64{5120, 1280}},
			{"attn_q_b.weight", []uint64{1280, 32768}}, {"attn_q_a_norm.weight", []uint64{1280}},
			{"attn_kv.weight", []uint64{5120, 512}}, {"attn_kv_a_norm.weight", []uint64{512}},
			{"attn_output_a.weight", []uint64{4096, 8192}}, {"attn_output_b.weight", []uint64{8192, 5120}},
			{"ffn_gate_inp.weight", []uint64{5120, 128}}, {"exp_probs_b.bias", []uint64{128}}, {"exp_probs_b_vl.bias", []uint64{128}},
		} {
			add(p+x.name, TensorF32, x.dims...)
		}
		for _, part := range []string{"gate", "up", "down"} {
			in, out := uint64(5120), uint64(2304)
			if part == "down" {
				in, out = out, in
			}
			add(p+"ffn_"+part+"_shexp.weight", TensorF32, in, out)
			add(p+"ffn_"+part+"_exps.weight", TensorMXFP4, in, out, 128)
		}
	}
	add("mtp.0.main_proj.weight", TensorF32, 15360, 5120)
	add("mtp.0.main_norm.weight", TensorF32, 5120)
	add("mtp.2.norm.weight", TensorF32, 5120)
	add("mtp.2.markov_head.markov_w1.weight", TensorF32, 256, 129280)
	add("mtp.2.markov_head.markov_w2.weight", TensorF32, 256, 129280)
	add("mtp.2.confidence_head.proj.weight", TensorF32, 5376, 1)
	if len(f.Tensors) != 81 {
		t.Fatal("fixture role count drift")
	}
	return f
}

// fak-test:justify why=regression when=changed:internal/ggufload/gguf_config.go
// fak-test:runtime fast est=50ms lane=default
func TestDeepSeekV41DSparkSupportConfigAdmission(t *testing.T) {
	f := supportGGUFFixture(t)
	_, err := f.Config()
	if !errors.Is(err, model.ErrV41DSparkUnsupported) {
		t.Fatalf("valid support header must reach typed execution-disabled gate, got %v", err)
	}
	for _, mutate := range []struct {
		name  string
		apply func(*File)
	}{
		{"missing metadata", func(f *File) { delete(f.Metadata, "dspark.weight_encoding") }},
		{"wrong targets", func(f *File) {
			f.Metadata["dspark.target_layer_ids"] = Value{TypeArray, []Value{{TypeUint32, uint32(37)}}}
		}},
		{"wrong scalar type", func(f *File) { f.Metadata["dspark.block_size"] = Value{TypeString, "5"} }},
		{"unclaimed tensor", func(f *File) {
			f.Tensors = append(f.Tensors, TensorInfo{Name: "mtp.0.unknown.weight", Type: TensorF32, Dims: []uint64{1}})
		}},
		{"missing required bias", func(f *File) {
			for i, x := range f.Tensors {
				if x.Name == "mtp.0.exp_probs_b_vl.bias" {
					f.Tensors = append(f.Tensors[:i], f.Tensors[i+1:]...)
					return
				}
			}
		}},
		{"wrong expert shape", func(f *File) {
			for i, x := range f.Tensors {
				if x.Name == "mtp.0.ffn_gate_exps.weight" {
					f.Tensors[i].Dims = []uint64{2560, 2304, 128}
					return
				}
			}
		}},
		{"duplicate tensor", func(f *File) { f.Tensors = append(f.Tensors, f.Tensors[0]) }},
		{"half norm", func(f *File) { f.Tensors[0].Type = TensorF16 }},
	} {
		t.Run(mutate.name, func(t *testing.T) {
			f := supportGGUFFixture(t)
			mutate.apply(f)
			if _, err := f.Config(); !errors.Is(err, model.ErrV41DSparkArtifact) {
				t.Fatalf("malformed support header must refuse as artifact, got %v", err)
			}
		})
	}
}
