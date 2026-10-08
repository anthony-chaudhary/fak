package model_test

import (
	"bytes"
	"context"
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/ggufload"
	"github.com/anthony-chaudhary/fak/internal/model"
)

type engProjGGUFTensor struct {
	name  string
	dims  []uint64
	dtype ggufload.TensorType
	data  []byte
}

func engProjPut(b *bytes.Buffer, value any) {
	if err := binary.Write(b, binary.LittleEndian, value); err != nil {
		panic(err)
	}
}
func engProjString(b *bytes.Buffer, value string) {
	engProjPut(b, uint64(len(value)))
	b.WriteString(value)
}
func engProjQ2(out, in int, seed uint64) []byte {
	raw := make([]byte, out*in/256*84)
	for i := range raw {
		seed = seed*6364136223846793005 + 1442695040888963407
		raw[i] = byte(seed >> 40)
	}
	for offset := 0; offset < len(raw); offset += 84 {
		// Modest finite scales and varied codes keep the Engram contribution visible without saturation.
		binary.LittleEndian.PutUint16(raw[offset+80:], 0x1400)
		binary.LittleEndian.PutUint16(raw[offset+82:], 0)
	}
	return raw
}
func engProjGGUF(t *testing.T) string {
	t.Helper()
	const H, I, E, cols, dim, hc, vocab = 64, 32, model.V41RouterExperts, 24, 256, 4, 8
	var tensors []engProjGGUFTensor
	seed := uint64(1366808)
	f32 := func(name string, dims ...uint64) {
		n := 1
		for _, d := range dims {
			n *= int(d)
		}
		data := make([]byte, 4*n)
		for i := 0; i < n; i++ {
			seed = seed*6364136223846793005 + 1442695040888963407
			value := (float32(seed>>40)/float32(1<<24)*2 - 1) * 0.05
			if strings.Contains(name, "norm") || strings.HasSuffix(name, "mhc_scale.weight") || strings.HasSuffix(name, "engram_q.weight") || strings.HasSuffix(name, "engram_k.weight") {
				value = 1
			}
			if strings.HasSuffix(name, "mhc_base.weight") {
				value = 0
			}
			binary.LittleEndian.PutUint32(data[i*4:], math.Float32bits(value))
		}
		tensors = append(tensors, engProjGGUFTensor{name, dims, ggufload.TensorF32, data})
	}
	f32("token_embd.weight", H, vocab)
	f32("output.weight", H, vocab)
	f32("output_norm.weight", H)
	p := "blk.0."
	f32(p+"attn_norm.weight", H)
	f32(p+"ffn_norm.weight", H)
	f32(p+"mhc_mixes.weight", hc*H, 24)
	f32(p+"mhc_base.weight", 24)
	f32(p+"mhc_scale.weight", 3)
	f32(p+"attn_q_a.weight", H, 32)
	f32(p+"attn_q_b.weight", 32, 64)
	f32(p+"attn_kv.weight", H, 32)
	f32(p+"attn_kv_a_norm.weight", 32)
	f32(p+"attn_output_a.weight", 64, 16)
	f32(p+"attn_output_b.weight", 32, H)
	f32(p+"attn_sinks.weight", 2)
	f32(p+"ffn_gate_inp.weight", H, E)
	f32(p+"exp_probs_b.bias", E)
	f32(p+"ffn_gate_shexp.weight", H, I)
	f32(p+"ffn_up_shexp.weight", H, I)
	f32(p+"ffn_down_shexp.weight", I, H)
	f32(p+"ffn_gate_exps.weight", H, I, E)
	f32(p+"ffn_up_exps.weight", H, I, E)
	f32(p+"ffn_down_exps.weight", I, H, E)
	f32(p+"engram_q.weight", hc*H)
	f32(p+"engram_k.weight", hc*H)
	tensors = append(tensors,
		engProjGGUFTensor{p + "engram_wkv.weight", []uint64{cols * dim, (hc + 1) * H}, ggufload.TensorQ2_K, engProjQ2((hc+1)*H, cols*dim, 136681)},
		engProjGGUFTensor{p + "engram_embd.weight", []uint64{256, 72}, ggufload.TensorQ2_K, engProjQ2(72, 256, 136682)},
	)
	var metadata []func(*bytes.Buffer)
	u64 := func(key string, value uint64) {
		metadata = append(metadata, func(b *bytes.Buffer) {
			engProjString(b, key)
			engProjPut(b, uint32(ggufload.TypeUint64))
			engProjPut(b, value)
		})
	}
	u32 := func(key string, value uint32) {
		metadata = append(metadata, func(b *bytes.Buffer) {
			engProjString(b, key)
			engProjPut(b, uint32(ggufload.TypeUint32))
			engProjPut(b, value)
		})
	}
	floating := func(key string, value float32) {
		metadata = append(metadata, func(b *bytes.Buffer) {
			engProjString(b, key)
			engProjPut(b, uint32(ggufload.TypeFloat32))
			engProjPut(b, value)
		})
	}
	str := func(key, value string) {
		metadata = append(metadata, func(b *bytes.Buffer) {
			engProjString(b, key)
			engProjPut(b, uint32(ggufload.TypeString))
			engProjString(b, value)
		})
	}
	ints := func(key string, values []int32) {
		metadata = append(metadata, func(b *bytes.Buffer) {
			engProjString(b, key)
			engProjPut(b, uint32(ggufload.TypeArray))
			engProjPut(b, uint32(ggufload.TypeInt32))
			engProjPut(b, uint64(len(values)))
			for _, v := range values {
				engProjPut(b, v)
			}
		})
	}
	uints := func(key string, values []uint64) {
		metadata = append(metadata, func(b *bytes.Buffer) {
			engProjString(b, key)
			engProjPut(b, uint32(ggufload.TypeArray))
			engProjPut(b, uint32(ggufload.TypeUint64))
			engProjPut(b, uint64(len(values)))
			for _, v := range values {
				engProjPut(b, v)
			}
		})
	}
	str("general.architecture", "deepseek41")
	u32("general.alignment", 32)
	p = "deepseek41."
	for key, value := range map[string]uint64{
		"embedding_length": H, "block_count": 1, "attention.head_count": 2, "expert_count": E, "expert_used_count": model.V41RouterTopK,
		"expert_feed_forward_length": I, "expert_shared_count": 1, "expert_shared_feed_forward_length": I,
		"attention.q_lora_rank": 32, "attention.kv_lora_rank": 32, "attention.qk_nope_head_dim": 16, "attention.qk_rope_head_dim": 16,
		"attention.value_length_mla": 16, "attention.output_group_count": 2, "attention.output_lora_rank": 16, "hyper_connection.count": hc, "hyper_connection.sinkhorn_iterations": 11,
	} {
		u64(p+key, value)
	}
	u32(p+"engram.head_count", 8)
	u32(p+"engram.key_length", dim)
	u32(p+"engram.max_ngram_size", 4)
	u32(p+"engram.pad_id", 2)
	floating(p+"attention.layer_norm_rms_epsilon", 1e-6)
	floating(p+"hyper_connection.epsilon", 1e-6)
	floating(p+"expert_weights_scale", 1.5)
	ints(p+"attention.compress_ratios", []int32{0})
	ints(p+"engram.layer_ids", []int32{0})
	ints(p+"engram.token_map", []int32{0, 1, 2, 3, 4, 5, 6, 7})
	primes, offsets := make([]uint64, cols), make([]uint64, cols)
	for i := range primes {
		primes[i], offsets[i] = 3, uint64(3*i)
	}
	uints(p+"engram.primes", primes)
	uints(p+"engram.offsets", offsets)
	uints(p+"engram.multipliers", []uint64{1, 3, 5, 7})
	metadata = append(metadata, func(b *bytes.Buffer) {
		engProjString(b, "tokenizer.ggml.tokens")
		engProjPut(b, uint32(ggufload.TypeArray))
		engProjPut(b, uint32(ggufload.TypeString))
		engProjPut(b, uint64(vocab))
		for _, token := range []string{"a", "b", "c", "d", "e", "f", "g", "h"} {
			engProjString(b, token)
		}
	})
	var out bytes.Buffer
	out.WriteString("GGUF")
	engProjPut(&out, uint32(3))
	engProjPut(&out, uint64(len(tensors)))
	engProjPut(&out, uint64(len(metadata)))
	for _, write := range metadata {
		write(&out)
	}
	offset := uint64(0)
	for _, tensor := range tensors {
		engProjString(&out, tensor.name)
		engProjPut(&out, uint32(len(tensor.dims)))
		for _, d := range tensor.dims {
			engProjPut(&out, d)
		}
		engProjPut(&out, uint32(tensor.dtype))
		engProjPut(&out, offset)
		offset = (offset + uint64(len(tensor.data)) + 31) / 32 * 32
	}
	for out.Len()%32 != 0 {
		out.WriteByte(0)
	}
	for _, tensor := range tensors {
		out.Write(tensor.data)
		for out.Len()%32 != 0 {
			out.WriteByte(0)
		}
	}
	path := filepath.Join(t.TempDir(), "v41-engram-projection.gguf")
	if err := os.WriteFile(path, out.Bytes(), 0o600); err != nil {
		t.Fatal(err)
	}
	return path
}

type engProjLoaderBackend struct {
	compute.Backend
	declineKV                           bool
	weights                             map[compute.Buffer]bool
	outputs                             map[compute.Buffer]bool
	calls, rows, uploads, reads, stages int
}

func (b *engProjLoaderBackend) Caps() compute.Caps {
	c := b.Backend.Caps()
	c.DeviceMemory, c.UploadDtype = true, true
	return c
}
func (b *engProjLoaderBackend) SupportsDeviceWeightDtype(dt compute.Dtype) bool {
	return !(b.declineKV && dt == compute.Q2_K) && compute.BackendSupportsDeviceWeightDtype(b.Backend, dt)
}
func (b *engProjLoaderBackend) Upload(x compute.Tensor, dt compute.Dtype) compute.Tensor {
	out := b.Backend.Upload(x, dt)
	if len(x.Shape) == 2 && x.Shape[0] == 320 && x.Shape[1] == 6144 {
		if dt != compute.Q2_K {
			panic("Engram projection did not preserve native Q2_K staging")
		}
		b.weights[out.Buf()] = true
		b.stages++
	}
	if dt == compute.F32 && len(x.Shape) == 1 && x.Shape[0] == 6144 {
		b.uploads += 4 * x.Shape[0]
	}
	return out
}
func (b *engProjLoaderBackend) multiply(w, x compute.Tensor, rows int, batch bool) compute.Tensor {
	var out compute.Tensor
	if batch {
		out = b.Backend.BatchedMatMul(w, x, rows)
	} else {
		out = b.Backend.MatMul(w, x)
	}
	if b.weights[w.Buf()] {
		b.calls++
		b.rows += rows
		b.outputs[out.Buf()] = true
	}
	return out
}
func (b *engProjLoaderBackend) MatMul(w, x compute.Tensor) compute.Tensor {
	return b.multiply(w, x, 1, false)
}
func (b *engProjLoaderBackend) BatchedMatMul(w, x compute.Tensor, rows int) compute.Tensor {
	return b.multiply(w, x, rows, true)
}
func (b *engProjLoaderBackend) Read(x compute.Tensor) []float32 {
	values := b.Backend.Read(x)
	if b.outputs[x.Buf()] {
		b.reads += 4 * len(values)
	}
	return values
}

func engProjLoaderPhase(t *testing.T, m *model.Model, phase string) map[string]float64 {
	t.Helper()
	raw, err := json.Marshal(m.V41ExpertFaultAttribution())
	if err != nil {
		t.Fatal(err)
	}
	var objects map[string]map[string]json.RawMessage
	if err := json.Unmarshal(raw, &objects); err != nil {
		t.Fatal(err)
	}
	values := map[string]float64{}
	for _, key := range []string{"engram_projection_device_calls", "engram_projection_host_calls", "engram_projection_device_rows", "engram_projection_host_rows", "engram_projection_matmul_calls", "engram_projection_activation_upload_bytes", "engram_projection_readback_bytes", "engram_projection_nanos", "engram_projection_host_weight_f32_bytes", "dense_projection_device_rows", "dense_projection_host_rows", "grouped_output_device_rows", "grouped_output_host_rows"} {
		if err := json.Unmarshal(objects[phase][key], new(float64)); err != nil {
			t.Errorf("default %s.%s unavailable: %v", phase, key, err)
			continue
		}
		var n float64
		if err := json.Unmarshal(objects[phase][key], &n); err != nil {
			t.Fatal(err)
		}
		values[key] = n
	}
	return values
}

// The complete GGUF is about 10 MiB; the largest ordinary F32 tensor is a 3 MiB expert slab.
// Its Engram KV is 645120 packed bytes versus 7864320 materialized F32 bytes.
// fak-test:justify why=integration when=changed:internal/**
// fak-test:runtime medium est=5s lane=default
func TestV41EngramProjectionGGUFLoaderToDefaultSession(t *testing.T) {
	path := engProjGGUF(t)
	m, err := func() (*model.Model, error) {
		budget := ggufload.MaxEagerF32Bytes
		ggufload.MaxEagerF32Bytes = 4 << 20
		defer func() { ggufload.MaxEagerF32Bytes = budget }()
		return ggufload.LoadModelQ4KStreamedDenseContext(context.Background(), path, nil)
	}()
	if err != nil {
		t.Fatalf("ordinary complete GGUF load must retain packed Engram KV below eager-F32 budget: %v", err)
	}
	t.Parallel()
	if m == nil || !m.V41EngramAttached() {
		t.Fatal("ordinary GGUF load did not attach Engram")
	}
	d41 := m.Cfg.DeepSeekV41
	if len(m.Cfg.CompressRatios) != 1 || m.Cfg.CompressRatios[0] != 0 || d41 == nil || len(d41.CompressRatios) != 1 || d41.CompressRatios[0] != 0 {
		t.Fatal("ordinary GGUF attachment lost the declared compression schedule")
	}
	if m.Cfg.HCMult != 4 || m.Cfg.HCSinkhornIters != 11 || math.Abs(m.Cfg.HCEps-1e-6) > 1e-12 || d41.HCMult != m.Cfg.HCMult || d41.HCSinkhornIters != m.Cfg.HCSinkhornIters || d41.HCEps != m.Cfg.HCEps {
		t.Fatal("ordinary GGUF attachment lost the declared HC4 execution axes")
	}
	if d41.CandidateSourceLayerID != -1 || len(d41.IndexSourceLayerIDs) != 0 || len(d41.KVSourceLayerIDs) != 0 {
		t.Fatal("ordinary GGUF attachment fabricated an undeclared source layer")
	}
	m.Cfg.CompressRatios[0] = 2
	if d41.CompressRatios[0] != 0 {
		t.Fatal("attached compression schedule aliases mutable flat metadata")
	}
	m.Cfg.CompressRatios[0] = 0
	bindings, ok := m.V41EngramStageBindings()
	if !ok || len(bindings) != 1 || bindings[0].Rows != 72 || bindings[0].RowBytes != model.V41EngramF32RowBytes {
		t.Fatal("ordinary GGUF load lacks verified executable table binding")
	}
	b := &engProjLoaderBackend{Backend: compute.Default(), weights: map[compute.Buffer]bool{}, outputs: map[compute.Buffer]bool{}}
	s, err := m.NewBackendSessionChecked(b)
	if err != nil {
		t.Fatal(err)
	}
	s.Quant = true
	hostBackend := &engProjLoaderBackend{Backend: compute.Default(), declineKV: true, weights: map[compute.Buffer]bool{}, outputs: map[compute.Buffer]bool{}}
	control, err := m.NewBackendSessionChecked(hostBackend)
	if err != nil {
		t.Fatal(err)
	}
	control.Quant = true
	defer func() {
		s.Close()
		control.Close()
		if err := m.CloseWeights(); err != nil {
			t.Error(err)
		}
	}()
	for index, ids := range [][]int{{1, 2, 3}, {4}, {5, 6}} {
		phase := "prefill"
		if index == 1 {
			phase = "decode"
		}
		before := engProjLoaderPhase(t, m, phase)
		calls, rows, uploads, reads := b.calls, b.rows, b.uploads, b.reads
		var got, want []float32
		if index == 1 {
			got = s.Step(ids[0])
		} else {
			got = s.Prefill(ids)
		}
		after := engProjLoaderPhase(t, m, phase)
		for key, value := range before {
			after[key] -= value
		}
		if b.rows-rows != len(ids) || b.calls-calls == 0 {
			t.Errorf("loaded default session projected device rows=%d calls=%d want rows=%d", b.rows-rows, b.calls-calls, len(ids))
		}
		if after["engram_projection_device_rows"] != float64(len(ids)) || after["engram_projection_host_calls"] != 0 || after["engram_projection_host_rows"] != 0 || after["engram_projection_host_weight_f32_bytes"] != 0 || after["engram_projection_matmul_calls"] != float64(b.calls-calls) || after["engram_projection_activation_upload_bytes"] != float64(b.uploads-uploads) || after["engram_projection_readback_bytes"] != float64(b.reads-reads) || after["engram_projection_nanos"] <= 0 {
			t.Errorf("loaded actual projection accounting=%v", after)
		}
		before = engProjLoaderPhase(t, m, phase)
		if index == 1 {
			want = control.Step(ids[0])
		} else {
			want = control.Prefill(ids)
		}
		host := engProjLoaderPhase(t, m, phase)
		for key, value := range before {
			host[key] -= value
		}
		materializations := 1
		if index == 2 {
			materializations = len(ids)
		}
		if host["engram_projection_host_rows"] != float64(len(ids)) || host["engram_projection_host_weight_f32_bytes"] != float64(7864320*materializations) {
			t.Errorf("loaded actual host materialization=%v", host)
		}
		for _, ordinary := range []map[string]float64{after, host} {
			if ordinary["dense_projection_device_rows"] != float64(7*len(ids)) || ordinary["dense_projection_host_rows"] != 0 || ordinary["grouped_output_device_rows"] != float64(len(ids)) || ordinary["grouped_output_host_rows"] != 0 {
				t.Errorf("loaded KV-only comparison changed ordinary default HAL arithmetic=%v", ordinary)
			}
		}
		if hostBackend.stages != 0 || hostBackend.calls != 0 || hostBackend.rows != 0 {
			t.Error("host-KV control invoked declined Q2 projection")
		}
		if len(got) != len(want) || len(got) == 0 {
			t.Fatal("loaded projection parity widths invalid")
		}
		var signal, diff float64
		for i, value := range got {
			if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
				t.Fatal("loaded projection logits nonfinite")
			}
			signal = max(signal, math.Abs(float64(want[i])))
			diff = max(diff, math.Abs(float64(value-want[i])))
		}
		if signal == 0 || diff > 1e-4*max(1, signal) {
			t.Errorf("loaded Engram projection parity signal=%g maximum_diff=%g", signal, diff)
		}
	}
	if b.stages != 1 {
		t.Errorf("loaded immutable packed KV staged %d times want once", b.stages)
	}
}
