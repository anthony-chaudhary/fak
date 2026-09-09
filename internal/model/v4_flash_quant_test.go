package model

import (
	"encoding/binary"
	"errors"
	"math"
	"reflect"
	"runtime"
	"strings"
	"testing"
)

// DeepSeek-V4-Flash-0731 routed-expert FP4 dimensions are pinned from the
// official checkpoint at revision 7872f01b1d1fe23eabc4c98b48bffcef5a386062:
//
//   - config: https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/blob/7872f01b1d1fe23eabc4c98b48bffcef5a386062/config.json
//   - inference: https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/blob/7872f01b1d1fe23eabc4c98b48bffcef5a386062/inference/model.py
//
// These tests use the real packed tensor dimensions but deterministic in-memory
// bytes. They neither contain nor read the published checkpoint weights.
func TestDeepSeekV4FlashExpertQuantDecodesOfficialPackedShapes(t *testing.T) {
	proDefaults := cloneV4ExpertQuantSpecs()
	defer func() { v4ExpertQuantSpecs = proDefaults }()

	tests := []struct {
		projection  string
		weightShape []int
		scaleShape  []int
		wantShape   []int
	}{
		{projection: "w1", weightShape: []int{2048, 2048}, scaleShape: []int{2048, 128}, wantShape: []int{2048, 4096}},
		{projection: "w3", weightShape: []int{2048, 2048}, scaleShape: []int{2048, 128}, wantShape: []int{2048, 4096}},
		{projection: "w2", weightShape: []int{4096, 1024}, scaleShape: []int{4096, 64}, wantShape: []int{4096, 2048}},
	}

	for _, tt := range tests {
		t.Run(tt.projection, func(t *testing.T) {
			weightBytes, ok := checkedShapeProduct(tt.weightShape...)
			if !ok {
				t.Fatalf("invalid test weight shape %v", tt.weightShape)
			}
			scaleBytes, ok := checkedShapeProduct(tt.scaleShape...)
			if !ok {
				t.Fatalf("invalid test scale shape %v", tt.scaleShape)
			}
			weights := make([]byte, weightBytes)
			scales := make([]byte, scaleBytes)

			// One E8M0 byte scales 16 packed bytes / 32 unpacked values.
			// Pin positive zero, negative zero, negative values, nibble order,
			// and the exact boundary between the first two scale groups.
			weights[0] = 0x90  // low=+0, high=-0.5
			weights[1] = 0x08  // low=-0, high=+0
			weights[15] = 0xa4 // final pair under scale group zero: +2, -1
			weights[16] = 0xb3 // first pair under scale group one: +1.5, -1.5
			scales[0] = 127    // 2^0
			scales[1] = 128    // 2^1

			name := "layers.3.ffn.experts.17." + tt.projection
			got, shape, err := decodeV4ExpertQuant(
				name+".weight", name+".scale",
				stEntry{Dtype: "I8", Shape: tt.weightShape},
				stEntry{Dtype: "F8_E8M0", Shape: tt.scaleShape},
				weights, scales,
			)
			if err != nil {
				t.Fatal(err)
			}
			if !sameShape(shape, tt.wantShape) {
				t.Fatalf("shape = %v, want %v", shape, tt.wantShape)
			}
			want := map[int]float32{
				0:  0,
				1:  -0.5,
				2:  float32(math.Copysign(0, -1)),
				3:  0,
				30: 2,
				31: -1,
				32: 3,
				33: -3,
			}
			for index, wantValue := range want {
				gotBits := binary.LittleEndian.Uint32(got[index*4:])
				if wantBits := math.Float32bits(wantValue); gotBits != wantBits {
					t.Fatalf("decoded[%d] bits = %08x, want %08x (%v)", index, gotBits, wantBits, wantValue)
				}
			}
			if !reflect.DeepEqual(v4ExpertQuantSpecs, proDefaults) {
				t.Fatalf("Flash decode mutated Pro quant defaults: got %#v, want %#v", v4ExpertQuantSpecs, proDefaults)
			}
		})

		// Each real-shape decode expands to about 32 MiB. Keep the cases
		// sequential and make the completed case collectible before the next.
		runtime.GC()
	}
}

func TestDeepSeekV4FlashExpertQuantRejectsMixedFlashProShapesBeforeOutput(t *testing.T) {
	tests := []struct {
		name        string
		projection  string
		weightShape []int
		scaleShape  []int
	}{
		{name: "Flash w1 with Pro scale", projection: "w1", weightShape: []int{2048, 2048}, scaleShape: []int{3072, 224}},
		{name: "Pro w1 with Flash scale", projection: "w1", weightShape: []int{3072, 3584}, scaleShape: []int{2048, 128}},
		{name: "Flash w3 with Pro scale", projection: "w3", weightShape: []int{2048, 2048}, scaleShape: []int{3072, 224}},
		{name: "Pro w3 with Flash scale", projection: "w3", weightShape: []int{3072, 3584}, scaleShape: []int{2048, 128}},
		{name: "Flash w2 with Pro scale", projection: "w2", weightShape: []int{4096, 1024}, scaleShape: []int{7168, 96}},
		{name: "Pro w2 with Flash scale", projection: "w2", weightShape: []int{7168, 1536}, scaleShape: []int{4096, 64}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			name := "layers.3.ffn.experts.17." + tt.projection
			got, shape, err := decodeV4ExpertQuant(
				name+".weight", name+".scale",
				stEntry{Dtype: "I8", Shape: tt.weightShape},
				stEntry{Dtype: "F8_E8M0", Shape: tt.scaleShape},
				nil, nil,
			)
			if got != nil || shape != nil {
				t.Fatalf("mixed-shape rejection produced output: bytes=%d shape=%v", len(got), shape)
			}
			if !errors.Is(err, ErrV4ExpertQuantMetadata) {
				t.Fatalf("error = %v, want ErrV4ExpertQuantMetadata", err)
			}
			if !strings.Contains(err.Error(), "shape") {
				t.Fatalf("error = %v, want shape validation before payload-length/allocation work", err)
			}
		})
	}
}

func cloneV4ExpertQuantSpecs() map[string]v4ExpertQuantSpec {
	clone := make(map[string]v4ExpertQuantSpec, len(v4ExpertQuantSpecs))
	for name, spec := range v4ExpertQuantSpecs {
		clone[name] = spec
	}
	return clone
}
