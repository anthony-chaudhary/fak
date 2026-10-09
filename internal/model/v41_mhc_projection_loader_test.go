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

func mhcLoaderQ2Oracle(t *testing.T, raw []byte) []float32 {
	t.Helper()
	if len(raw)%84 != 0 {
		t.Fatal("Q2 fixture block length invalid")
	}
	values := make([]float32, len(raw)/84*256)
	for block := 0; block < len(raw)/84; block++ {
		b := raw[84*block : 84*(block+1)]
		if binary.LittleEndian.Uint16(b[80:]) != 0x1400 || binary.LittleEndian.Uint16(b[82:]) != 0 {
			t.Fatal("fixed-scale independent Q2 oracle fixture changed")
		}
		for cell := 0; cell < 256; cell++ {
			h, group, lane := cell/128, (cell%128)/32, cell%32
			scale := b[h*8+2*group+lane/16] & 15
			code := (b[16+h*32+lane] >> uint(2*group)) & 3
			values[block*256+cell] = float32(scale) * float32(code) / 1024
		}
	}
	return values
}

func mhcLoaderFixtures(t *testing.T) (string, []byte, []byte, []float32) {
	t.Helper()
	original, err := os.ReadFile(engProjGGUF(t))
	if err != nil {
		t.Fatal(err)
	}
	header, err := ggufload.Read(bytes.NewReader(original))
	if err != nil {
		t.Fatal(err)
	}
	const name = "blk.0.mhc_mixes.weight"
	var targetOffset int64 = -1
	var engram []byte
	for _, tensor := range header.Tensors {
		if tensor.Name == name {
			if tensor.Type != ggufload.TensorF32 || len(tensor.Dims) != 2 || tensor.Dims[0] != 256 || tensor.Dims[1] != 24 {
				t.Fatal("complete parent fixture lacks full logical HC4 mHC source")
			}
			targetOffset = tensor.FileOffset
		}
		if tensor.Name == "blk.0.engram_wkv.weight" {
			if tensor.Type != ggufload.TensorQ2_K || len(tensor.Dims) != 2 || tensor.Dims[0] != 6144 || tensor.Dims[1] != 320 {
				t.Fatal("complete fixture Engram Q2 source changed")
			}
			engram = append([]byte(nil), original[tensor.FileOffset:tensor.FileOffset+645120]...)
		}
	}
	if targetOffset < 0 || len(engram) == 0 {
		t.Fatal("complete GGUF projection sources missing")
	}
	var encoded bytes.Buffer
	engProjString(&encoded, name)
	directory := original[:header.TensorDataOffset]
	if bytes.Count(directory, encoded.Bytes()) != 1 {
		t.Fatal("fixture mHC tensor descriptor not unique")
	}
	start := bytes.Index(directory, encoded.Bytes()) + encoded.Len()
	if binary.LittleEndian.Uint32(original[start:]) != 2 {
		t.Fatal("fixture descriptor rank changed")
	}
	dtypeAt := start + 4 + 16
	packed := engProjQ2(24, 256, 136681111)
	logical := mhcLoaderQ2Oracle(t, packed)
	deviceBytes := append([]byte(nil), original...)
	binary.LittleEndian.PutUint32(deviceBytes[dtypeAt:], uint32(ggufload.TensorQ2_K))
	copy(deviceBytes[targetOffset:], packed)
	// Retain the following GGUF offsets: the replaced F32 tail is legal unused tensor-data padding, not a model metadata transplant.
	write := func(filename string, data []byte) string {
		path := filepath.Join(t.TempDir(), filename)
		if err := os.WriteFile(path, data, 0o600); err != nil {
			t.Fatal(err)
		}
		parsed, err := ggufload.Read(bytes.NewReader(data))
		if err != nil {
			t.Fatal(err)
		}
		if len(parsed.Tensors) != len(header.Tensors) {
			t.Fatal("complete GGUF tensor inventory changed")
		}
		return path
	}
	return write("mhc-packed.gguf", deviceBytes), packed, engram, logical
}

type mhcLoaderOperation struct {
	component          string
	rows, upload, read int
	activation, result []float32
}
type mhcLoaderBackend struct {
	compute.Backend
	declineQ2   bool
	mhc, engram []byte
	weights     map[compute.Buffer]string
	stages      map[string]int
	outputs     map[compute.Buffer]int
	activations map[compute.Buffer][]float32
	operations  []mhcLoaderOperation
}

func newMHCLoaderBackend(mhc, engram []byte) *mhcLoaderBackend {
	return &mhcLoaderBackend{Backend: compute.Default(), mhc: mhc, engram: engram, weights: map[compute.Buffer]string{}, stages: map[string]int{}, outputs: map[compute.Buffer]int{}, activations: map[compute.Buffer][]float32{}}
}
func (b *mhcLoaderBackend) Caps() compute.Caps {
	c := b.Backend.Caps()
	c.DeviceMemory, c.UploadDtype = true, true
	return c
}
func (b *mhcLoaderBackend) SupportsDeviceWeightDtype(dt compute.Dtype) bool {
	return !(b.declineQ2 && dt == compute.Q2_K) && compute.BackendSupportsDeviceWeightDtype(b.Backend, dt)
}
func mhcLoaderCodesEqual(codes []int8, raw []byte) bool {
	if len(codes) != len(raw) {
		return false
	}
	for i, code := range codes {
		if byte(code) != raw[i] {
			return false
		}
	}
	return true
}
func (b *mhcLoaderBackend) Upload(x compute.Tensor, dt compute.Dtype) compute.Tensor {
	component := ""
	host, hostOK := x.Buf().(compute.HostBuffer)
	if hostOK && dt == compute.Q2_K {
		if mhcLoaderCodesEqual(host.I8(), b.mhc) {
			component = "mhc"
		}
		if mhcLoaderCodesEqual(host.I8(), b.engram) {
			component = "engram"
		}
	}
	y := b.Backend.Upload(x, dt)
	if component != "" {
		b.weights[y.Buf()] = component
		b.stages[component]++
	}
	if hostOK && dt == compute.F32 && len(x.Shape) == 1 {
		b.activations[y.Buf()] = append([]float32(nil), host.F32()...)
	}
	return y
}
func (b *mhcLoaderBackend) multiply(w, x compute.Tensor, rows int, batch bool) compute.Tensor {
	var y compute.Tensor
	if batch {
		y = b.Backend.BatchedMatMul(w, x, rows)
	} else {
		y = b.Backend.MatMul(w, x)
	}
	if component := b.weights[w.Buf()]; component != "" {
		input := append([]float32(nil), b.activations[x.Buf()]...)
		b.operations = append(b.operations, mhcLoaderOperation{component: component, rows: rows, upload: 4 * len(input), activation: input})
		b.outputs[y.Buf()] = len(b.operations) - 1
	}
	return y
}
func (b *mhcLoaderBackend) MatMul(w, x compute.Tensor) compute.Tensor {
	return b.multiply(w, x, 1, false)
}
func (b *mhcLoaderBackend) BatchedMatMul(w, x compute.Tensor, rows int) compute.Tensor {
	return b.multiply(w, x, rows, true)
}
func (b *mhcLoaderBackend) Read(x compute.Tensor) []float32 {
	y := b.Backend.Read(x)
	if index, ok := b.outputs[x.Buf()]; ok {
		b.operations[index].read += 4 * len(y)
		b.operations[index].result = append([]float32(nil), y...)
	}
	return y
}
func (b *mhcLoaderBackend) Free(x compute.Tensor) {
	delete(b.outputs, x.Buf())
	delete(b.activations, x.Buf())
	b.Backend.Free(x)
}

func mhcLoaderPhase(t *testing.T, m *model.Model, phase string) map[string]float64 {
	t.Helper()
	raw, err := json.Marshal(m.V41ExpertFaultAttribution())
	if err != nil {
		t.Fatal(err)
	}
	var phases map[string]map[string]json.RawMessage
	if err := json.Unmarshal(raw, &phases); err != nil {
		t.Fatal(err)
	}
	out := map[string]float64{}
	for _, prefix := range []string{"mhc_projection_", "engram_projection_"} {
		for _, suffix := range []string{"device_calls", "host_calls", "device_rows", "host_rows", "matmul_calls", "activation_upload_bytes", "readback_bytes", "nanos", "host_weight_f32_bytes"} {
			key := prefix + suffix
			var value float64
			if err := json.Unmarshal(phases[phase][key], &value); err != nil {
				t.Fatalf("default %s.%s unavailable", phase, key)
			}
			out[key] = value
		}
	}
	for _, key := range []string{"dense_projection_device_rows", "dense_projection_host_rows", "grouped_output_device_rows", "grouped_output_host_rows"} {
		var value float64
		if err := json.Unmarshal(phases[phase][key], &value); err != nil {
			t.Fatal(err)
		}
		out[key] = value
	}
	return out
}
func mhcLoaderDelta(after, before map[string]float64) map[string]float64 {
	out := map[string]float64{}
	for key, value := range after {
		out[key] = value - before[key]
	}
	return out
}

// The complete GGUF is about 10 MiB and loaded twice, retaining the existing Engram table plus 384 tiny experts; the full HC4 mHC leaf is 2016 packed bytes versus 24576 F32 bytes.
// fak-test:justify why=integration when=changed:internal/**
// fak-test:runtime medium est=5s lane=default
func TestV41MHCProjectionGGUFLoaderToDefaultSession(t *testing.T) {
	t.Parallel()
	devicePath, packed, engram, logical := mhcLoaderFixtures(t)
	load := func(path string) *model.Model {
		m, err := ggufload.LoadModelQ4KStreamedDenseContext(context.Background(), path, nil)
		if err != nil {
			t.Fatalf("ordinary complete GGUF load failed: %v", err)
		}
		if m != nil {
			t.Cleanup(func() {
				if err := m.CloseWeights(); err != nil {
					t.Error(err)
				}
			})
		}
		if m == nil || !m.V41EngramAttached() {
			t.Fatal("ordinary load lost executable Engram binding")
		}
		return m
	}
	m, control := load(devicePath), load(devicePath)
	const nativeName = "model.layers.0.mhc.mixes.weight"
	if m.HasF32(nativeName) || control.HasF32(nativeName) {
		t.Fatal("ordinary loader did not retain packed mHC representation")
	}
	for _, loaded := range []*model.Model{m, control} {
		if loaded.Cfg.HiddenSize != 64 || loaded.Cfg.DeepSeekV41 == nil || loaded.Cfg.DeepSeekV41.HCMult != 4 || len(loaded.Cfg.DeepSeekV41.EngramLayerIDs) != 1 || loaded.Cfg.DeepSeekV41.EngramLayerIDs[0] != 0 {
			t.Fatal("ordinary GGUF load lost full HC4/active Engram geometry")
		}
	}
	hc := m.Cfg.DeepSeekV41.HCMult
	inputWidth := hc * m.Cfg.HiddenSize
	// Each stream has a pre/post coefficient and a residual coefficient for every stream pair.
	coefficientWidth := 2*hc + hc*hc
	b, hostBackend := newMHCLoaderBackend(packed, engram), newMHCLoaderBackend(packed, engram)
	hostBackend.declineQ2 = true
	s, err := m.NewBackendSessionChecked(b)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(s.Close)
	host, err := control.NewBackendSessionChecked(hostBackend)
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
		before, hostBefore := mhcLoaderPhase(t, m, phase), mhcLoaderPhase(t, control, phase)
		from := len(b.operations)
		var got, want []float32
		if index == 1 {
			got, want = s.Step(ids[0]), host.Step(ids[0])
		} else {
			got, want = s.Prefill(ids), host.Prefill(ids)
		}
		data, hostData := mhcLoaderDelta(mhcLoaderPhase(t, m, phase), before), mhcLoaderDelta(mhcLoaderPhase(t, control, phase), hostBefore)
		for component, prefix := range map[string]string{"mhc": "mhc_projection_", "engram": "engram_projection_"} {
			calls, rows, upload, read := 0, 0, 0, 0
			for _, op := range b.operations[from:] {
				if op.component == component {
					calls++
					rows += op.rows
					upload += op.upload
					read += op.read
				}
			}
			if rows != len(ids) || data[prefix+"device_calls"] != float64(calls) || data[prefix+"device_rows"] != float64(rows) || data[prefix+"matmul_calls"] != float64(calls) || data[prefix+"activation_upload_bytes"] != float64(upload) || data[prefix+"readback_bytes"] != float64(read) || data[prefix+"host_calls"] != 0 || data[prefix+"host_rows"] != 0 || data[prefix+"host_weight_f32_bytes"] != 0 || data[prefix+"nanos"] <= 0 {
				t.Errorf("loaded %s actual payload route/ledger=%v observed=%d/%d/%d/%d", component, data, calls, rows, upload, read)
			}
		}
		for _, op := range b.operations[from:] {
			if op.component == "mhc" {
				if len(op.activation) != inputWidth || len(op.result) != coefficientWidth || op.upload != 1024 || op.read != 96 {
					t.Fatal("loaded full HC4 mHC payload geometry incorrect")
				}
				var ss, signal float64
				for _, value := range op.activation {
					ss += float64(value) * float64(value)
				}
				for output := 0; output < 24; output++ {
					var dot float64
					for cell, value := range op.activation {
						dot += float64(logical[output*256+cell]) * float64(value)
					}
					signal = math.Max(signal, math.Abs(dot))
					if math.Abs(float64(op.result[output])-dot) > 1e-4*math.Max(1, math.Abs(dot)) {
						t.Fatalf("loaded raw-before-RMS scalar mismatch output=%d", output)
					}
				}
				if signal == 0 || math.Abs(ss/256-1) < .01 {
					t.Fatal("loaded raw-vs-prenormalized negative control vacuous")
				}
			}
		}
		materializations := 1
		if index == 2 {
			materializations = len(ids)
		}
		if hostData["mhc_projection_host_calls"] != float64(len(ids)) || hostData["mhc_projection_host_rows"] != float64(len(ids)) || hostData["mhc_projection_host_weight_f32_bytes"] != float64(24576*materializations) || hostData["mhc_projection_device_calls"] != 0 || hostData["mhc_projection_device_rows"] != 0 || hostData["mhc_projection_matmul_calls"] != 0 || hostData["mhc_projection_activation_upload_bytes"] != 0 || hostData["mhc_projection_readback_bytes"] != 0 || hostData["mhc_projection_nanos"] <= 0 {
			t.Errorf("loaded Q2 capability whole-mHC host decline=%v", hostData)
		}
		if hostData["engram_projection_device_calls"] != 0 || hostData["engram_projection_device_rows"] != 0 || hostData["engram_projection_matmul_calls"] != 0 || hostData["engram_projection_activation_upload_bytes"] != 0 || hostData["engram_projection_readback_bytes"] != 0 || hostData["engram_projection_host_calls"] != float64(len(ids)) || hostData["engram_projection_host_rows"] != float64(len(ids)) || hostData["engram_projection_host_weight_f32_bytes"] != float64(7864320*materializations) || hostData["engram_projection_nanos"] <= 0 {
			t.Errorf("Q2 capability control lost executable Engram host path=%v", hostData)
		}
		if data["dense_projection_device_rows"] != float64(7*len(ids)) || data["dense_projection_host_rows"] != 0 || data["grouped_output_device_rows"] != float64(len(ids)) || data["grouped_output_host_rows"] != 0 {
			t.Errorf("loaded candidate default ordinary composition lost=%v", data)
		}
		if hostData["dense_projection_device_rows"] != float64(7*len(ids)) || hostData["dense_projection_host_rows"] != 0 || hostData["grouped_output_device_rows"] != float64(len(ids)) || hostData["grouped_output_host_rows"] != 0 {
			t.Errorf("Q2 capability control default ordinary HAL composition lost=%v", hostData)
		}
		if len(got) == 0 || len(got) != len(want) {
			t.Fatal("loaded numeric parity widths invalid")
		}
		var signal, diff float64
		for cell, value := range got {
			if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) || math.IsNaN(float64(want[cell])) || math.IsInf(float64(want[cell]), 0) {
				t.Fatal("loaded logits nonfinite")
			}
			signal = math.Max(signal, math.Abs(float64(want[cell])))
			diff = math.Max(diff, math.Abs(float64(value-want[cell])))
		}
		if signal == 0 || diff > 1e-4*math.Max(1, signal) {
			t.Errorf("loaded independent Q2 host parity signal=%g maxdiff=%g", signal, diff)
		}
		t.Logf("phase=%s cold=%t tokens=%d mhc_device_calls=%g mhc_device_rows=%g mhc_api_matmuls=%g mhc_activation_upload_bytes=%g mhc_readback_bytes=%g mhc_host_f32_bytes=%g control_host_calls=%g control_host_rows=%g control_host_f32_bytes=%g max_diff=%g", phase, index == 0, len(ids), data["mhc_projection_device_calls"], data["mhc_projection_device_rows"], data["mhc_projection_matmul_calls"], data["mhc_projection_activation_upload_bytes"], data["mhc_projection_readback_bytes"], data["mhc_projection_host_weight_f32_bytes"], hostData["mhc_projection_host_calls"], hostData["mhc_projection_host_rows"], hostData["mhc_projection_host_weight_f32_bytes"], diff)
	}
	if b.stages["mhc"] != 1 || b.stages["engram"] != 1 {
		t.Errorf("loaded immutable payload cache staging target=%v", b.stages)
	}
	if hostBackend.stages["mhc"] != 0 || hostBackend.stages["engram"] != 0 || len(hostBackend.operations) != 0 {
		t.Errorf("declined Q2 payload reached control staging/API stages=%v operations=%d", hostBackend.stages, len(hostBackend.operations))
	}
}
