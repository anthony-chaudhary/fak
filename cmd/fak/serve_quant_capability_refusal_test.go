package main

import (
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/ggufload"
	"github.com/anthony-chaudhary/fak/internal/gpulease"
	fakmodel "github.com/anthony-chaudhary/fak/internal/model"
)

// fak-test:runtime fast est=10ms
func TestServeQuantCapabilityRefusalPrecedesModelLoad(t *testing.T) {
	unsupported := []ggufload.TensorType{
		ggufload.TensorIQ3_XXS, ggufload.TensorIQ2_S, ggufload.TensorIQ3_S,
		ggufload.TensorIQ2_XXS, ggufload.TensorIQ2_XS, ggufload.TensorIQ1_S,
		ggufload.TensorIQ4_XS, ggufload.TensorQ3_K, ggufload.TensorIQ1_M,
		ggufload.TensorIQ4_NL,
	}
	path := writeServeQuantCapabilityFixture(t, unsupported)
	ws, err := ggufload.OpenWeights(path)
	if err != nil {
		t.Fatal(err)
	}
	routes, err := metalQuantCapabilityCensus(ws)
	ws.Close()
	if err == nil {
		t.Fatal("unsupported Metal inventory produced no capability refusal")
	}
	var refusal *fakmodel.QuantCapabilityRefusal
	if !errors.As(err, &refusal) {
		t.Fatalf("Metal inventory error type = %T, want *model.QuantCapabilityRefusal", err)
	}
	wantUnsupported := make(map[string]bool, len(unsupported))
	for _, typ := range unsupported {
		wantUnsupported[typ.String()] = true
	}
	if len(refusal.Unsupported) != len(wantUnsupported) {
		t.Fatalf("refusal names %d quant types, want %d", len(refusal.Unsupported), len(wantUnsupported))
	}
	for _, name := range refusal.Unsupported {
		if !wantUnsupported[name] {
			t.Errorf("refusal names unexpected quant type %s", name)
		}
	}
	if len(routes) != len(unsupported)+1 {
		t.Fatalf("selection receipt has %d bands, want %d", len(routes), len(unsupported)+1)
	}
	for _, route := range routes {
		if route.QuantType == ggufload.TensorQ4_K.String() {
			if route.Backend != "metal" || route.Refused {
				t.Errorf("Q4_K selection = %+v, want admitted metal", route)
			}
			continue
		}
		if route.Backend != "unselected" || !route.Refused {
			t.Errorf("unsupported selection = %+v, want refused unselected", route)
		}
	}
	loads := 0
	kernelCalls := 0
	loadAndExecuteCPU := func() bool {
		loads++
		ws, err := ggufload.OpenWeights(path)
		if err != nil {
			t.Error(err)
			return false
		}
		defer ws.Close()
		be := compute.Default()
		if !compute.RequireReference(be) {
			t.Error("CPU reference backend required")
			return false
		}
		src, err := ggufload.NewComputeSource(be, ws)
		if err != nil {
			t.Error(err)
			return false
		}
		x := make([]float32, 256)
		x[0] = 1 // impulse pins a decoded matrix entry independently of dot order
		input := compute.NewF32(be, []int{256}, x)
		for _, tensor := range ws.File.Tensors {
			canon, ok := ggufload.CanonicalTensorNameArch(tensor.Name, "llama")
			if !ok {
				t.Errorf("unmapped fixture %s", tensor.Name)
				return false
			}
			w, err := src.Weight(canon, compute.F32)
			if err != nil {
				t.Error(err)
				return false
			}
			result := be.MatMul(w, input)
			kernelCalls++
			got, host := be.Host(result)
			var want float32
			switch tensor.Type {
			case ggufload.TensorQ4_K, ggufload.TensorIQ3_XXS:
				want = 1
			case ggufload.TensorQ3_K:
				want = 128
			}
			if !host || len(got) != 1 || got[0] != want {
				t.Errorf("CPU executed %s (%s) result=%v host=%t, want [%v]", tensor.Name, tensor.Type, got, host, want)
				return false
			}
			// The IQ/Q3 source was actually decoded and multiplied on the CPU.
			// Other zero-scale bands are execution coverage, not numeric-quality
			// qualification; selected Metal targets were never executed here.
			t.Logf("stage=execution backend=%s source_quant=%s compute_dtype=f32 tensor=%s", be.Name(), tensor.Type, tensor.Name)
		}
		return true
	}
	release, err := admitLocalMetalModel(true, path, gpulease.Options{Path: filepath.Join(t.TempDir(), "gpu.lease")}, metalAdmissionSpec{
		Load: loadAndExecuteCPU, ReceiptOut: io.Discard,
	})
	if err == nil {
		t.Fatal("unsupported Metal inventory reached admission without refusal")
	}
	if !errors.As(err, &refusal) {
		t.Fatalf("admission error type = %T, want *model.QuantCapabilityRefusal", err)
	}
	if loads != 0 || kernelCalls != 0 {
		t.Fatalf("unsupported Metal admission invoked loaders=%d kernels=%d, want 0/0", loads, kernelCalls)
	}
	if release != nil {
		release()
	}
	release, err = admitLocalMetalModel(false, path, gpulease.Options{}, metalAdmissionSpec{Load: loadAndExecuteCPU})
	if err != nil || loads != 1 || kernelCalls != len(unsupported)+1 {
		t.Fatalf("explicit CPU admission: loads=%d kernels=%d error=%v, want 1/%d", loads, kernelCalls, err, len(unsupported)+1)
	}
	release()
}

func writeServeQuantCapabilityFixture(t *testing.T, types []ggufload.TensorType) string {
	t.Helper()
	var b bytes.Buffer
	writeMinimalHeaderForTest(&b, uint64(len(types)+1), 9)
	writeKVStringForTest(&b, "general.architecture", "llama")
	writeKVUint32ForTest(&b, "general.alignment", 32)
	writeKVUint32ForTest(&b, "llama.embedding_length", 256)
	writeKVUint32ForTest(&b, "llama.block_count", uint32(len(types)))
	writeKVUint32ForTest(&b, "llama.attention.head_count", 1)
	writeKVUint32ForTest(&b, "llama.attention.key_length", 256)
	writeKVUint32ForTest(&b, "llama.feed_forward_length", 256)
	writeKVUint32ForTest(&b, "llama.context_length", 16)
	writeKVFloat32ForTest(&b, "llama.attention.layer_norm_rms_epsilon", 1e-6)
	writeTensorInfoForTest(&b, "blk.0.ffn_up.weight", []uint64{256, 1}, uint32(ggufload.TensorQ4_K), 0)
	for i, typ := range types {
		writeTensorInfoForTest(&b, fmt.Sprintf("blk.%d.ffn_down.weight", i), []uint64{256, 1}, uint32(typ), uint64((i+1)*256))
	}
	padToAlignmentForTest(&b, 32)
	payload := make([]byte, 256*(len(types)+1))
	// Q4_K control: d=1, dmin=0, scale=1, code=1 => each weight is 1.
	binary.LittleEndian.PutUint16(payload, 0x3c00)
	for i := 4; i < 16; i++ {
		payload[i] = 1
	}
	for i := 16; i < 144; i++ {
		payload[i] = 0x11
	}
	for i, typ := range types {
		block := payload[(i+1)*256:]
		switch typ {
		case ggufload.TensorIQ3_XXS:
			binary.LittleEndian.PutUint16(block, 0x3c00) // grid0=4, quarter scale => 1
		case ggufload.TensorQ3_K:
			binary.LittleEndian.PutUint16(block[108:], 0x3c00) // (-4)*(-32) => 128
		}
	}
	b.Write(payload)
	path := filepath.Join(t.TempDir(), "mixed-unsupported.gguf")
	if err := os.WriteFile(path, b.Bytes(), 0o600); err != nil {
		t.Fatal(err)
	}
	return path
}
