package model

import (
	"math"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

const modelOptNVFP4TestBase = "model.layers.0.mlp.down_proj"

func modelOptNVFP4TestTensors(scaleSuffix, globalSuffix string) map[string]tinySTTensor {
	packed := make([]byte, 16)
	packed[0], packed[1] = 0x21, 0x08
	packed[8] = 0x47
	return map[string]tinySTTensor{
		modelOptNVFP4TestBase + ".weight": {
			dtype: "U8",
			shape: []int{2, 8},
			data:  packed,
		},
		modelOptNVFP4TestBase + scaleSuffix: {
			dtype: "F8_E4M3",
			shape: []int{2, 1},
			data:  []byte{0x38, 0x40},
		},
		modelOptNVFP4TestBase + globalSuffix: {
			dtype: "F32",
			shape: []int{},
			data:  f32TestBytes([]float32{0.5}),
		},
	}
}

// fak-test:runtime fast est=20ms lane=default
func TestModelOptNVFP4SourceWeightOwnershipAndCPUParity(t *testing.T) {
	// The initial source contract is deliberately single-file. Sharded ModelOpt
	// assembly remains outside this acceptance test until the loader exposes it.
	families := []struct {
		name, scale, global string
	}{
		{"hf", ".weight_scale", ".weight_scale_2"},
		{"trtllm", ".weights_scaling_factor", ".weights_scaling_factor_2"},
	}
	for _, family := range families {
		t.Run(family.name, func(t *testing.T) {
			be := compute.Default()
			path := writeTinySafetensors(t, modelOptNVFP4TestTensors(family.scale, family.global))
			src, err := OpenModelOptNVFP4Source(path, be)
			if err != nil {
				t.Fatal(err)
			}
			weight, err := src.Weight(modelOptNVFP4TestBase, compute.NVFP4)
			if err != nil {
				t.Fatal(err)
			}
			byWeightName, err := src.Weight(modelOptNVFP4TestBase+".weight", compute.NVFP4)
			if err != nil {
				t.Fatalf("exact weight name: %v", err)
			}
			if weight.Dtype != compute.NVFP4 || byWeightName.Dtype != compute.NVFP4 ||
				len(weight.Shape) != 2 || weight.Shape[0] != 2 || weight.Shape[1] != 16 ||
				weight.Quant == nil || len(weight.Quant.Scale) != 1 || weight.Quant.Scale[0] != 0.5 {
				t.Fatalf("decoded metadata = dtype %s shape %v quant %#v", weight.Dtype, weight.Shape, weight.Quant)
			}

			if err := src.Close(); err != nil {
				t.Fatal(err)
			}
			payload, ok := weight.Buf().(compute.HostBuffer)
			if !ok {
				t.Fatalf("owned NVFP4 buffer = %T, want compute.HostBuffer", weight.Buf())
			}
			wantPayload := append(make([]byte, 16), 0x38, 0x40)
			wantPayload[0], wantPayload[1], wantPayload[8] = 0x21, 0x08, 0x47
			gotPayload := payload.I8()
			if len(gotPayload) != len(wantPayload) {
				t.Fatalf("owned payload length = %d, want %d", len(gotPayload), len(wantPayload))
			}
			for i, want := range wantPayload {
				if byte(gotPayload[i]) != want {
					t.Fatalf("owned payload[%d] = %#x, want %#x after Close", i, byte(gotPayload[i]), want)
				}
			}

			x := make([]float32, 16)
			x[0], x[1], x[2] = 2, 3, 99
			got := be.Read(be.MatMul(weight, compute.NewF32(be, []int{16}, x)))
			if len(got) != 2 || got[0] != 2 || got[1] != 18 {
				t.Fatalf("CPU MatMul = %v, want [2 18] (even-low nibble, signed zero, scale_2)", got)
			}
			_, first := src.Weight(modelOptNVFP4TestBase, compute.NVFP4)
			_, second := src.Weight(modelOptNVFP4TestBase, compute.NVFP4)
			if first == nil || second == nil || first.Error() != second.Error() ||
				!strings.Contains(strings.ToLower(first.Error()), "closed") {
				t.Fatalf("post-Close errors = (%v, %v), want stable closed error", first, second)
			}
		})
	}
}

// fak-test:runtime fast est=20ms lane=default
func TestModelOptNVFP4SourceRejectsMalformedTriplets(t *testing.T) {
	type rejectCase struct {
		name   string
		want   compute.Dtype
		mutate func(map[string]tinySTTensor)
	}
	hfScale := modelOptNVFP4TestBase + ".weight_scale"
	hfGlobal := modelOptNVFP4TestBase + ".weight_scale_2"
	weightName := modelOptNVFP4TestBase + ".weight"
	cases := []rejectCase{
		{"requested dtype", compute.F32, nil},
		{"weight dtype", compute.NVFP4, func(ts map[string]tinySTTensor) {
			x := ts[weightName]
			x.dtype = "I8"
			ts[weightName] = x
		}},
		{"weight rank", compute.NVFP4, func(ts map[string]tinySTTensor) {
			x := ts[weightName]
			x.shape = []int{16}
			ts[weightName] = x
		}},
		{"weight K alignment", compute.NVFP4, func(ts map[string]tinySTTensor) {
			x := ts[weightName]
			x.shape, x.data = []int{2, 7}, make([]byte, 14)
			ts[weightName] = x
		}},
		{"scale dtype", compute.NVFP4, func(ts map[string]tinySTTensor) {
			x := ts[hfScale]
			x.dtype = "U8"
			ts[hfScale] = x
		}},
		{"scale shape", compute.NVFP4, func(ts map[string]tinySTTensor) {
			x := ts[hfScale]
			x.shape, x.data = []int{1, 1}, []byte{0x38}
			ts[hfScale] = x
		}},
		{"scale2 dtype", compute.NVFP4, func(ts map[string]tinySTTensor) {
			x := ts[hfGlobal]
			x.dtype, x.data = "F16", []byte{0x00, 0x38}
			ts[hfGlobal] = x
		}},
		{"scale2 shape", compute.NVFP4, func(ts map[string]tinySTTensor) {
			x := ts[hfGlobal]
			x.shape, x.data = []int{2}, f32TestBytes([]float32{0.5, 0.5})
			ts[hfGlobal] = x
		}},
		{"scale2 NaN", compute.NVFP4, func(ts map[string]tinySTTensor) {
			x := ts[hfGlobal]
			x.data = f32TestBytes([]float32{float32(math.NaN())})
			ts[hfGlobal] = x
		}},
		{"missing weight", compute.NVFP4, func(ts map[string]tinySTTensor) { delete(ts, weightName) }},
		{"missing scale", compute.NVFP4, func(ts map[string]tinySTTensor) { delete(ts, hfScale) }},
		{"missing scale2", compute.NVFP4, func(ts map[string]tinySTTensor) { delete(ts, hfGlobal) }},
		{"ambiguous scale family", compute.NVFP4, func(ts map[string]tinySTTensor) {
			ts[modelOptNVFP4TestBase+".weights_scaling_factor"] = tinySTTensor{
				dtype: "F8_E4M3", shape: []int{2, 1}, data: []byte{0x38, 0x40},
			}
			ts[modelOptNVFP4TestBase+".weights_scaling_factor_2"] = tinySTTensor{
				dtype: "F32", shape: []int{}, data: f32TestBytes([]float32{0.5}),
			}
		}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			tensors := modelOptNVFP4TestTensors(".weight_scale", ".weight_scale_2")
			if tc.mutate != nil {
				tc.mutate(tensors)
			}
			src, err := OpenModelOptNVFP4Source(writeTinySafetensors(t, tensors), compute.Default())
			if err == nil {
				_, err = src.Weight(modelOptNVFP4TestBase, tc.want)
				if closeErr := src.Close(); closeErr != nil {
					t.Fatalf("Close after rejection: %v", closeErr)
				}
			}
			if err == nil {
				t.Fatal("malformed ModelOpt NVFP4 triplet was accepted")
			}
		})
	}
}
