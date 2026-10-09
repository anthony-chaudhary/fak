package model_test

import (
	"bytes"
	"context"
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/ggufload"
	"github.com/anthony-chaudhary/fak/internal/model"
)

func headLoaderFixture(t *testing.T, tied bool) (string, []byte) {
	t.Helper()
	original, err := os.ReadFile(engProjGGUF(t))
	if err != nil {
		t.Fatal(err)
	}
	header, err := ggufload.Read(bytes.NewReader(original))
	if err != nil {
		t.Fatal(err)
	}
	var encoded bytes.Buffer
	engProjString(&encoded, header.Tensors[0].Name)
	first := bytes.Index(original[:header.TensorDataOffset], encoded.Bytes())
	if first < 24 {
		t.Fatal("complete fixture directory missing")
	}
	prefix := append([]byte(nil), original[:first]...)
	encoded.Reset()
	engProjString(&encoded, "deepseek41.embedding_length")
	if bytes.Count(prefix, encoded.Bytes()) != 1 {
		t.Fatal("fixture embedding metadata not unique")
	}
	at := bytes.Index(prefix, encoded.Bytes()) + encoded.Len()
	if binary.LittleEndian.Uint32(prefix[at:]) != uint32(ggufload.TypeUint64) || binary.LittleEndian.Uint64(prefix[at+4:]) != 64 {
		t.Fatal("fixture embedding metadata changed")
	}
	binary.LittleEndian.PutUint64(prefix[at+4:], 256)
	var tensors []engProjGGUFTensor
	packed := engProjQ2(8, 256, 1366813002)
	for _, source := range header.Tensors {
		if tied && source.Name == "output.weight" {
			continue
		}
		dims := append([]uint64(nil), source.Dims...)
		switch source.Name {
		case "token_embd.weight", "output.weight", "blk.0.attn_q_a.weight", "blk.0.attn_kv.weight", "blk.0.ffn_gate_inp.weight", "blk.0.ffn_gate_shexp.weight", "blk.0.ffn_up_shexp.weight", "blk.0.ffn_gate_exps.weight", "blk.0.ffn_up_exps.weight":
			dims[0] = 256
		case "output_norm.weight", "blk.0.attn_norm.weight", "blk.0.ffn_norm.weight":
			dims[0] = 256
		case "blk.0.mhc_mixes.weight", "blk.0.engram_q.weight", "blk.0.engram_k.weight":
			dims[0] = 1024
		case "blk.0.attn_output_b.weight", "blk.0.ffn_down_shexp.weight", "blk.0.ffn_down_exps.weight":
			dims[1] = 256
		case "blk.0.engram_wkv.weight":
			dims[1] = 1280
		}
		kind := source.Type
		var data []byte
		switch {
		case source.Name == "output.weight" || (tied && source.Name == "token_embd.weight"):
			kind = ggufload.TensorQ2_K
			data = packed
		case source.Name == "blk.0.engram_wkv.weight":
			kind = ggufload.TensorF32
			data = make([]byte, 1280*6144*4)
			for cell := 0; cell < 1280*6144; cell++ {
				value := float32((cell+cell/6144)%23-11) / 1024
				binary.LittleEndian.PutUint32(data[cell*4:], math.Float32bits(value))
			}
		case kind == ggufload.TensorF32:
			n, oldn := 1, 1
			for _, d := range dims {
				n *= int(d)
			}
			for _, d := range source.Dims {
				oldn *= int(d)
			}
			data = make([]byte, n*4)
			for cell := 0; cell < n; cell++ {
				copy(data[cell*4:cell*4+4], original[int(source.FileOffset)+4*(cell%oldn):int(source.FileOffset)+4*(cell%oldn)+4])
			}
		case kind == ggufload.TensorQ2_K:
			n := 1
			for _, d := range source.Dims {
				n *= int(d)
			}
			data = append([]byte(nil), original[source.FileOffset:source.FileOffset+int64(n/256*84)]...)
		default:
			t.Fatalf("unhandled source fixture dtype %d", kind)
		}
		tensors = append(tensors, engProjGGUFTensor{source.Name, dims, kind, data})
	}
	binary.LittleEndian.PutUint64(prefix[8:], uint64(len(tensors)))
	var output bytes.Buffer
	output.Write(prefix)
	offset := uint64(0)
	for _, tensor := range tensors {
		engProjString(&output, tensor.name)
		engProjPut(&output, uint32(len(tensor.dims)))
		for _, d := range tensor.dims {
			engProjPut(&output, d)
		}
		engProjPut(&output, uint32(tensor.dtype))
		engProjPut(&output, offset)
		offset = (offset + uint64(len(tensor.data)) + 31) / 32 * 32
	}
	for output.Len()%32 != 0 {
		output.WriteByte(0)
	}
	for _, tensor := range tensors {
		output.Write(tensor.data)
		for output.Len()%32 != 0 {
			output.WriteByte(0)
		}
	}
	path := filepath.Join(t.TempDir(), "head-packed.gguf")
	if err := os.WriteFile(path, output.Bytes(), 0600); err != nil {
		t.Fatal(err)
	}
	parsed, err := ggufload.Read(bytes.NewReader(output.Bytes()))
	if err != nil {
		t.Fatal(err)
	}
	wantTensors := len(header.Tensors)
	if tied {
		wantTensors--
	}
	if len(parsed.Tensors) != wantTensors {
		t.Fatal("complete loader fixture lost source tensors")
	}
	return path, packed
}

type headLoaderBackend struct {
	*mhcLoaderBackend
	head       []byte
	logical    []float32
	declineF32 bool
}

func (b *headLoaderBackend) SupportsDeviceWeightDtype(dt compute.Dtype) bool {
	return !(b.declineF32 && dt == compute.F32) && b.mhcLoaderBackend.SupportsDeviceWeightDtype(dt)
}
func headLoaderValuesEqual(got, want []float32) bool {
	if len(got) != len(want) || len(want) == 0 {
		return false
	}
	for i, value := range got {
		if value != want[i] {
			return false
		}
	}
	return true
}
func (b *headLoaderBackend) Upload(x compute.Tensor, dt compute.Dtype) compute.Tensor {
	y := b.mhcLoaderBackend.Upload(x, dt)
	if host, ok := x.Buf().(compute.HostBuffer); ok && ((dt == compute.Q2_K && mhcLoaderCodesEqual(host.I8(), b.head)) || (dt == compute.F32 && len(x.Shape) == 2 && x.Shape[0] == 8 && x.Shape[1] == 256 && headLoaderValuesEqual(host.F32(), b.logical))) {
		b.weights[y.Buf()] = "head"
		b.stages["head"]++
	}
	return y
}
func headLoaderPhase(t *testing.T, m *model.Model, phase string) map[string]float64 {
	t.Helper()
	out := mhcLoaderPhase(t, m, phase)
	raw, err := json.Marshal(m.V41ExpertFaultAttribution())
	if err != nil {
		t.Fatal(err)
	}
	var objects map[string]map[string]json.RawMessage
	if err := json.Unmarshal(raw, &objects); err != nil {
		t.Fatal(err)
	}
	for _, suffix := range []string{"device_calls", "host_calls", "device_rows", "host_rows", "activation_upload_bytes", "readback_bytes", "nanos"} {
		key := "head_projection_" + suffix
		var n float64
		if err := json.Unmarshal(objects[phase][key], &n); err != nil {
			t.Errorf("default %s.%s missing", phase, key)
		}
		out[key] = n
	}
	return out
}

// The complete H256/HC4 GGUF retains 384 experts, the retained F32 router gate, and active F32-source Engram loaded as Q8; about 70 MiB is loaded twice without changing global loader budgets.
// fak-test:justify why=integration when=changed:internal/**
// fak-test:runtime medium est=12s lane=default
func TestV41HeadProjectionGGUFLoaderToDefaultSession(t *testing.T) {
	t.Parallel()
	for _, tied := range []bool{false, true} {
		t.Run(map[bool]string{false: "untied-head-only-q2-capability", true: "tied-global-f32-capability"}[tied], func(t *testing.T) {
			path, packed := headLoaderFixture(t, tied)
			logical := mhcLoaderQ2Oracle(t, packed)
			load := func() *model.Model {
				m, err := ggufload.LoadModelQ4KStreamedDenseContext(context.Background(), path, nil)
				if err != nil {
					t.Fatal(err)
				}
				t.Cleanup(func() {
					if err := m.CloseWeights(); err != nil {
						t.Error(err)
					}
				})
				return m
			}
			m, control := load(), load()
			for _, loaded := range []*model.Model{m, control} {
				if !loaded.HasF32("model.layers.0.ffn.gate.weight") {
					t.Fatal("default GGUF loader did not retain the actual F32 router gate used by global dtype capability control")
				}
				if loaded.Cfg.HiddenSize != 256 || loaded.Cfg.DeepSeekV41 == nil || loaded.Cfg.DeepSeekV41.HCMult != 4 || len(loaded.Cfg.DeepSeekV41.EngramLayerIDs) != 1 || loaded.HasF32("lm_head.weight") {
					t.Fatal("actual packed loader lost head/full-HC4/Engram representation")
				}
				if loaded.Cfg.VocabSize <= 0 || len(logical) != loaded.Cfg.VocabSize*loaded.Cfg.HiddenSize {
					t.Fatal("loaded vocabulary does not match the independent packed head matrix")
				}
			}
			b, hb := &headLoaderBackend{mhcLoaderBackend: newMHCLoaderBackend(nil, nil), head: packed}, &headLoaderBackend{mhcLoaderBackend: newMHCLoaderBackend(nil, nil), head: packed}
			if tied {
				if !m.HasF32("model.embed_tokens.weight") || !control.HasF32("model.embed_tokens.weight") {
					t.Fatal("default GGUF tied Q2 source did not retain its actual F32 embedding representation")
				}
				b.logical, hb.logical = logical, logical
				hb.declineF32 = true
			} else {
				hb.declineQ2 = true
			}
			s, err := m.NewBackendSessionChecked(b)
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(s.Close)
			host, err := control.NewBackendSessionChecked(hb)
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(host.Close)
			s.Quant, host.Quant = true, true
			for index, ids := range [][]int{{1, 2, 3}, {4}, {5, 6}} {
				phase := "prefill"
				if index == 1 {
					phase = "decode"
				}
				before, hbefore := headLoaderPhase(t, m, phase), headLoaderPhase(t, control, phase)
				from := len(b.operations)
				var got, want []float32
				if index == 1 {
					got, want = s.Step(ids[0]), host.Step(ids[0])
				} else {
					got, want = s.Prefill(ids), host.Prefill(ids)
				}
				data, hd := mhcLoaderDelta(headLoaderPhase(t, m, phase), before), mhcLoaderDelta(headLoaderPhase(t, control, phase), hbefore)
				calls, rows, upload, read := 0, 0, 0, 0
				for _, op := range b.operations[from:] {
					if op.component != "head" {
						continue
					}
					calls++
					rows += op.rows
					upload += op.upload
					read += op.read
					if len(op.activation) != 256*op.rows || len(op.result) != 8*op.rows {
						t.Fatal("actual packed head operand/result geometry invalid")
					}
					var signal float64
					for row := 0; row < op.rows; row++ {
						for output := 0; output < 8; output++ {
							var dot float64
							for cell := 0; cell < 256; cell++ {
								dot += float64(logical[output*256+cell]) * float64(op.activation[row*256+cell])
							}
							signal = math.Max(signal, math.Abs(dot))
							if math.Abs(float64(op.result[row*8+output])-dot) > 1e-4*math.Max(1, math.Abs(dot)) {
								t.Fatal("packed head independent scalar mismatch")
							}
						}
					}
					if signal == 0 {
						t.Fatal("head scalar witness has zero signal")
					}
				}
				if rows != len(ids) || data["head_projection_device_calls"] != float64(calls) || data["head_projection_device_rows"] != float64(rows) || data["head_projection_host_calls"] != 0 || data["head_projection_host_rows"] != 0 || data["head_projection_activation_upload_bytes"] != float64(upload) || data["head_projection_readback_bytes"] != float64(read) || upload != 1024*len(ids) || read != 32*len(ids) || data["head_projection_nanos"] <= 0 {
					t.Errorf("loaded actual head route calls=%d rows=%d upload=%d read=%d ledger=%v", calls, rows, upload, read, data)
				}
				if hd["head_projection_device_calls"] != 0 || hd["head_projection_device_rows"] != 0 || hd["head_projection_host_rows"] != float64(len(ids)) || hd["head_projection_host_calls"] <= 0 || hd["head_projection_activation_upload_bytes"] != 0 || hd["head_projection_readback_bytes"] != 0 || hd["head_projection_nanos"] <= 0 {
					t.Errorf("natural packed head capability decline=%v", hd)
				}
				for arm, ledger := range []map[string]float64{data, hd} {
					denseDevice, denseHost := 7, 0
					// A global F32 capability decline also routes the actual F32 router gate to host.
					if tied && arm == 1 {
						denseDevice, denseHost = 6, 1
					}
					if ledger["dense_projection_device_rows"] != float64(denseDevice*len(ids)) || ledger["dense_projection_host_rows"] != float64(denseHost*len(ids)) || ledger["grouped_output_device_rows"] != float64(len(ids)) || ledger["grouped_output_host_rows"] != 0 || ledger["mhc_projection_device_rows"] != float64(len(ids)) || ledger["mhc_projection_host_rows"] != 0 || ledger["engram_projection_device_rows"] != float64(len(ids)) || ledger["engram_projection_host_rows"] != 0 {
						t.Errorf("actual default capability composition arm=%d dense_device=%d dense_host=%d ledger=%v", arm, denseDevice, denseHost, ledger)
					}
				}
				if len(got) != m.Cfg.VocabSize || len(want) != control.Cfg.VocabSize {
					t.Fatal("loaded head logit width invalid")
				}
				var diff, signal float64
				for i, x := range got {
					if math.IsNaN(float64(x)) || math.IsInf(float64(x), 0) || math.IsNaN(float64(want[i])) || math.IsInf(float64(want[i]), 0) {
						t.Fatal("loaded logits nonfinite")
					}
					diff = math.Max(diff, math.Abs(float64(x-want[i])))
					signal = math.Max(signal, math.Abs(float64(want[i])))
				}
				if signal == 0 || diff > 1e-4*math.Max(1, signal) {
					t.Errorf("loaded capability-control parity maxdiff=%g signal=%g", diff, signal)
				}
				t.Logf("phase=%s tokens=%d head_calls=%d rows=%d upload=%d read=%d control_host_rows=%g maxdiff=%g", phase, len(ids), calls, rows, upload, read, hd["head_projection_host_rows"], diff)
			}
			if b.stages["head"] != 1 || hb.stages["head"] != 0 || len(hb.operations) != 0 {
				t.Errorf("packed head cache staging candidate=%v control=%v", b.stages, hb.stages)
			}
		})
	}

}
