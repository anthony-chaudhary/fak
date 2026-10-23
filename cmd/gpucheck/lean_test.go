package main

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// leanQ8UploadBackend is the reference floor advertising quantized uploads, the CUDA
// capability that makes a Quant session stage resident Q8 weights instead of f32.
type leanQ8UploadBackend struct {
	compute.Backend
	q8Uploads int
}

func (b *leanQ8UploadBackend) Name() string { return "gpucheck-lean-q8-upload" }

func (b *leanQ8UploadBackend) Caps() compute.Caps {
	c := b.Backend.Caps()
	c.UploadDtype = true
	return c
}

func (b *leanQ8UploadBackend) Upload(t compute.Tensor, as compute.Dtype) compute.Tensor {
	if as == compute.Q8_0 {
		b.q8Uploads++
	}
	return b.Backend.Upload(t, as)
}

// fak-test:runtime fast est=500ms lane=default
func TestRunLeanStagesQ8WeightsOnUploadDtypeBackend(t *testing.T) {
	dir := writeLeanQ8Checkpoint(t)
	be := &leanQ8UploadBackend{Backend: compute.Default()}
	compute.Register(be)
	var stdout bytes.Buffer
	code, panicked := runRecovering([]string{"-lean", "-json", "-n", "4", "-backend", be.Name(), "-hf", dir}, &stdout, io.Discard)
	if panicked != nil {
		t.Fatalf("lean device session panicked: %v", panicked)
	}
	var got parityReceipt
	if err := json.Unmarshal(stdout.Bytes(), &got); err != nil {
		t.Fatalf("decode receipt: %v", err)
	}
	if code != 0 || !got.Pass || got.ReasonCode != "" || got.ComparedTokens == 0 {
		t.Fatalf("exit=%d pass=%t reason_code=%q compared=%d", code, got.Pass, got.ReasonCode, got.ComparedTokens)
	}
	if got.ObservedBackend != be.Name() || got.ReferenceLabel != "cpu-Q8" {
		t.Fatalf("observed=%q reference=%q", got.ObservedBackend, got.ReferenceLabel)
	}
	if be.q8Uploads == 0 {
		t.Fatal("device session staged no Q8 weight")
	}
}

// leanNoQuantUploadBackend is a device backend that cannot stage quantized uploads, like
// cpu-ref or Vulkan without its Q8 kernel.
type leanNoQuantUploadBackend struct{ compute.Backend }

func (b leanNoQuantUploadBackend) Name() string { return "gpucheck-lean-no-quant-upload" }

func (b leanNoQuantUploadBackend) Caps() compute.Caps {
	c := b.Backend.Caps()
	c.UploadDtype = false
	return c
}

// fak-test:runtime fast est=300ms lane=default
func TestRunLeanRefusesBackendWithoutQuantUploads(t *testing.T) {
	dir := writeLeanQ8Checkpoint(t)
	be := leanNoQuantUploadBackend{Backend: compute.Default()}
	compute.Register(be)
	var stdout bytes.Buffer
	code, panicked := runRecovering([]string{"-lean", "-json", "-n", "4", "-backend", be.Name(), "-hf", dir}, &stdout, io.Discard)
	if panicked != nil {
		t.Fatalf("lean run panicked instead of refusing: %v", panicked)
	}
	var got parityReceipt
	if err := json.Unmarshal(stdout.Bytes(), &got); err != nil {
		t.Fatalf("decode receipt: %v", err)
	}
	if code != 2 || got.Pass || got.ReasonCode != reasonUnsupportedOptions {
		t.Fatalf("exit=%d pass=%t reason_code=%q", code, got.Pass, got.ReasonCode)
	}
}

func runRecovering(args []string, stdout, stderr io.Writer) (code int, panicked any) {
	defer func() { panicked = recover() }()
	return run(args, stdout, stderr), nil
}

// writeLeanQ8Checkpoint writes a tiny sharded Qwen2-shaped HF snapshot (config.json,
// index, one f32 shard) whose matmul weights the lean loader keeps only as Q8.
func writeLeanQ8Checkpoint(t *testing.T) string {
	t.Helper()
	const hidden, layers, heads, kvHeads, headDim, inter, vocab = 64, 2, 4, 2, 16, 128, 128
	dir := t.TempDir()
	cfg := fmt.Sprintf(`{"model_type":"qwen2","hidden_size":%d,"num_hidden_layers":%d,"num_attention_heads":%d,"num_key_value_heads":%d,"head_dim":%d,"intermediate_size":%d,"vocab_size":%d,"rms_norm_eps":1e-6,"rope_theta":10000,"tie_word_embeddings":true,"eos_token_id":%d}`,
		hidden, layers, heads, kvHeads, headDim, inter, vocab, vocab-1)
	if err := os.WriteFile(filepath.Join(dir, "config.json"), []byte(cfg), 0o600); err != nil {
		t.Fatal(err)
	}
	type tensor struct {
		name  string
		shape []int
		norm  bool
	}
	tensors := []tensor{{"model.embed_tokens.weight", []int{vocab, hidden}, false}, {"model.norm.weight", []int{hidden}, true}}
	for l := 0; l < layers; l++ {
		p := fmt.Sprintf("model.layers.%d.", l)
		tensors = append(tensors,
			tensor{p + "input_layernorm.weight", []int{hidden}, true},
			tensor{p + "self_attn.q_proj.weight", []int{heads * headDim, hidden}, false},
			tensor{p + "self_attn.q_proj.bias", []int{heads * headDim}, false},
			tensor{p + "self_attn.k_proj.weight", []int{kvHeads * headDim, hidden}, false},
			tensor{p + "self_attn.k_proj.bias", []int{kvHeads * headDim}, false},
			tensor{p + "self_attn.v_proj.weight", []int{kvHeads * headDim, hidden}, false},
			tensor{p + "self_attn.v_proj.bias", []int{kvHeads * headDim}, false},
			tensor{p + "self_attn.o_proj.weight", []int{hidden, heads * headDim}, false},
			tensor{p + "post_attention_layernorm.weight", []int{hidden}, true},
			tensor{p + "mlp.gate_proj.weight", []int{inter, hidden}, false},
			tensor{p + "mlp.up_proj.weight", []int{inter, hidden}, false},
			tensor{p + "mlp.down_proj.weight", []int{hidden, inter}, false},
		)
	}
	type entry struct {
		Dtype       string `json:"dtype"`
		Shape       []int  `json:"shape"`
		DataOffsets []int  `json:"data_offsets"`
	}
	const shard = "model-00001-of-00001.safetensors"
	hdr := map[string]entry{}
	weightMap := map[string]string{}
	var data []byte
	seed := uint64(0x9e3779b97f4a7c15)
	for _, tn := range tensors {
		n := 1
		for _, d := range tn.shape {
			n *= d
		}
		start := len(data)
		for i := 0; i < n; i++ {
			seed = seed*6364136223846793005 + 1442695040888963407
			v := (float32(seed>>40)/float32(1<<24)*2 - 1) * 0.5
			if tn.norm {
				v = 1 + v*0.1
			}
			data = binary.LittleEndian.AppendUint32(data, math.Float32bits(v))
		}
		hdr[tn.name] = entry{Dtype: "F32", Shape: tn.shape, DataOffsets: []int{start, len(data)}}
		weightMap[tn.name] = shard
	}
	hdrBytes, err := json.Marshal(hdr)
	if err != nil {
		t.Fatal(err)
	}
	blob := binary.LittleEndian.AppendUint64(nil, uint64(len(hdrBytes)))
	blob = append(append(blob, hdrBytes...), data...)
	if err := os.WriteFile(filepath.Join(dir, shard), blob, 0o600); err != nil {
		t.Fatal(err)
	}
	index, err := json.Marshal(map[string]any{"weight_map": weightMap})
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "model.safetensors.index.json"), index, 0o600); err != nil {
		t.Fatal(err)
	}
	return dir
}
