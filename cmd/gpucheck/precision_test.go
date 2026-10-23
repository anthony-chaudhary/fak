package main

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// precisionRecordingBackend witnesses the upload requests made by the real model
// forward path. The CPU implementation deliberately does not certify device math.
type precisionRecordingBackend struct {
	compute.Backend
	uploadDtype bool
	uploads     map[compute.Dtype]int
}

func (b *precisionRecordingBackend) Name() string { return "gpucheck-precision-test" }

func (b *precisionRecordingBackend) Caps() compute.Caps {
	c := b.Backend.Caps()
	c.UploadDtype = b.uploadDtype
	return c
}

func (b *precisionRecordingBackend) Upload(t compute.Tensor, as compute.Dtype) compute.Tensor {
	if len(t.Shape) == 2 {
		b.uploads[as]++
	}
	return b.Backend.Upload(t, as)
}

// fak-test:runtime fast est=1ms lane=default
func TestRunHelpPreservesSuccessStatus(t *testing.T) {
	var stdout, stderr bytes.Buffer
	if code := run([]string{"-h"}, &stdout, &stderr); code != 0 {
		t.Fatalf("help exit=%d want=0", code)
	}
	if stdout.Len() != 0 || !bytes.Contains(stderr.Bytes(), []byte("Usage of gpucheck:")) {
		t.Fatalf("unexpected help output: stdout=%q stderr=%q", stdout.String(), stderr.String())
	}
}

// fak-test:runtime fast est=200ms lane=default
func TestRunPrecisionModesSelectActualWeightUploads(t *testing.T) {
	dir := writePrecisionCheckpoint(t)
	for _, tc := range []struct {
		name, f16, q8 string
		lean          bool
		noDtype       bool
		want          compute.Dtype
		wantCode      int
	}{
		{name: "default f32", want: compute.F32},
		{name: "f16", f16: "1", want: compute.F16},
		{name: "q8 from f32 checkpoint", q8: "1", want: compute.Q8_0},
		{name: "lean implies device q8", lean: true, want: compute.Q8_0},
		{name: "only exact one enables modes", f16: "true", q8: "01", want: compute.F32},
		{name: "conflicting env modes", f16: "1", q8: "1", wantCode: 2},
		{name: "f16 conflicts with lean", f16: "1", lean: true, wantCode: 2},
		{name: "backend cannot honor mode", f16: "1", noDtype: true, wantCode: 2},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Setenv("FAK_CUDA_F16", tc.f16)
			t.Setenv("FAK_CUDA_Q8", tc.q8)
			be := &precisionRecordingBackend{
				Backend: compute.Default(), uploadDtype: !tc.noDtype,
				uploads: make(map[compute.Dtype]int),
			}
			compute.Register(be)
			args := []string{"-json", "-hf", dir, "-backend", be.Name(), "-n", "2"}
			if tc.lean {
				args = append(args, "-lean")
			}
			var stdout, stderr bytes.Buffer
			code := run(args, &stdout, &stderr)
			if code != tc.wantCode {
				t.Fatalf("exit=%d want=%d stdout=%q stderr=%q", code, tc.wantCode, stdout.String(), stderr.String())
			}
			var receipt parityReceipt
			if err := json.Unmarshal(stdout.Bytes(), &receipt); err != nil {
				t.Fatalf("decode receipt: %v; stdout=%q", err, stdout.String())
			}
			if tc.wantCode != 0 {
				if receipt.Pass || receipt.ReasonCode != reasonUnsupportedOptions || len(be.uploads) != 0 {
					t.Fatalf("invalid mode reached execution: receipt=%+v uploads=%v", receipt, be.uploads)
				}
				return
			}
			if !receipt.Pass || receipt.ComparedTokens != 2 {
				t.Fatalf("tiny checkpoint did not complete parity: %+v", receipt)
			}
			if be.uploads[tc.want] == 0 {
				t.Fatalf("mode requested no %s matrix upload: %v", tc.want, be.uploads)
			}
			for _, narrow := range []compute.Dtype{compute.F16, compute.Q8_0} {
				if narrow != tc.want && be.uploads[narrow] != 0 {
					t.Fatalf("mode unexpectedly requested %s matrix uploads: %v", narrow, be.uploads)
				}
			}
			wantReference := "f32 legacy"
			if tc.lean {
				wantReference = "cpu-Q8"
			}
			if receipt.ReferenceLabel != wantReference {
				t.Fatalf("reference_label=%q want=%q", receipt.ReferenceLabel, wantReference)
			}
		})
	}
}

// A small, valid one-layer safetensors checkpoint keeps the regression on run's
// real load/session/generate path. Uniform embeddings and zero projection weights
// produce stable argmax ties, so the witness isolates dtype routing, not quality.
func writePrecisionCheckpoint(t *testing.T) string {
	t.Helper()
	const h, v = 32, 8
	dir := t.TempDir()
	cfg := []byte(`{"hidden_size":32,"num_hidden_layers":1,"num_attention_heads":2,"num_key_value_heads":1,"head_dim":16,"intermediate_size":32,"vocab_size":8,"rms_norm_eps":0.00001,"rope_theta":10000,"tie_word_embeddings":true,"eos_token_id":-1}`)
	if err := os.WriteFile(filepath.Join(dir, "config.json"), cfg, 0o600); err != nil {
		t.Fatal(err)
	}
	type tensorHeader struct {
		Dtype   string `json:"dtype"`
		Shape   []int  `json:"shape"`
		Offsets [2]int `json:"data_offsets"`
	}
	header := make(map[string]tensorHeader)
	var payload []byte
	for _, tensor := range []struct {
		name  string
		shape []int
		value float32
	}{
		{"model.embed_tokens.weight", []int{v, h}, 1},
		{"model.norm.weight", []int{h}, 1},
		{"model.layers.0.input_layernorm.weight", []int{h}, 1},
		{"model.layers.0.post_attention_layernorm.weight", []int{h}, 1},
		{"model.layers.0.self_attn.q_proj.weight", []int{h, h}, 0},
		{"model.layers.0.self_attn.k_proj.weight", []int{16, h}, 0},
		{"model.layers.0.self_attn.v_proj.weight", []int{16, h}, 0},
		{"model.layers.0.self_attn.o_proj.weight", []int{h, h}, 0},
		{"model.layers.0.mlp.gate_proj.weight", []int{h, h}, 0},
		{"model.layers.0.mlp.up_proj.weight", []int{h, h}, 0},
		{"model.layers.0.mlp.down_proj.weight", []int{h, h}, 0},
	} {
		count := 1
		for _, dim := range tensor.shape {
			count *= dim
		}
		start := len(payload)
		for range count {
			payload = binary.LittleEndian.AppendUint32(payload, math.Float32bits(tensor.value))
		}
		header[tensor.name] = tensorHeader{"F32", tensor.shape, [2]int{start, len(payload)}}
	}
	rawHeader, err := json.Marshal(header)
	if err != nil {
		t.Fatal(err)
	}
	for len(rawHeader)%8 != 0 {
		rawHeader = append(rawHeader, ' ')
	}
	data := binary.LittleEndian.AppendUint64(nil, uint64(len(rawHeader)))
	data = append(data, rawHeader...)
	data = append(data, payload...)
	if err := os.WriteFile(filepath.Join(dir, "model.safetensors"), data, 0o600); err != nil {
		t.Fatal(err)
	}
	return dir
}
