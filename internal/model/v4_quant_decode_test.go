package model

import (
	"encoding/binary"
	"errors"
	"math"
	"strings"
	"testing"
)

func TestV4ExpertQuantPinnedContractDecodesOCPMXFP4(t *testing.T) {
	// Each repeated byte puts the same E2M1 code in both nibble positions.
	// The first 16 packed bytes therefore cover all 16 positive/negative codes;
	// the second 16 prove the next scale group is selected exactly at K=32.
	packedCodes := []byte{0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, 0xaa, 0xbb, 0xcc, 0xdd, 0xee, 0xff}
	weights := make([]byte, 3072*3584)
	for row := 0; row < 3072; row++ {
		base := row * 3584
		copy(weights[base:base+16], packedCodes)
		copy(weights[base+16:base+32], packedCodes)
		weights[base+32] = 0x11 // first code in the E8M0-byte-zero group
	}
	scales := make([]byte, 3072*224)
	for row := 0; row < 3072; row++ {
		scales[row*224] = 127   // 2^0
		scales[row*224+1] = 128 // 2^1
	}

	got, shape, err := decodeV4ExpertQuant(
		"layers.0.ffn.experts.0.w1.weight", "layers.0.ffn.experts.0.w1.scale",
		stEntry{Dtype: "I8", Shape: []int{3072, 3584}},
		stEntry{Dtype: "F8_E8M0", Shape: []int{3072, 224}}, weights, scales,
	)
	if err != nil {
		t.Fatal(err)
	}
	if !sameShape(shape, []int{3072, 7168}) {
		t.Fatalf("shape = %v, want [3072 7168]", shape)
	}
	wantCodes := []float32{0, .5, 1, 1.5, 2, 3, 4, 6, float32(math.Copysign(0, -1)), -.5, -1, -1.5, -2, -3, -4, -6}
	for group, scale := range []float32{1, 2} {
		for code, wantCode := range wantCodes {
			for nibble := 0; nibble < 2; nibble++ {
				index := group*32 + code*2 + nibble
				gotBits := binary.LittleEndian.Uint32(got[index*4:])
				wantBits := math.Float32bits(wantCode * scale)
				if gotBits != wantBits {
					t.Fatalf("group=%d code=%d nibble=%d bits=%08x, want %08x", group, code, nibble, gotBits, wantBits)
				}
			}
		}
	}
	// E8M0 byte zero is the OCP minimum 2^-127, not zero.
	index := 64
	if gotBits, wantBits := binary.LittleEndian.Uint32(got[index*4:]), math.Float32bits(float32(math.Ldexp(.5, -127))); gotBits != wantBits {
		t.Fatalf("E8M0 byte zero decode bits=%08x, want %08x", gotBits, wantBits)
	}
}

func TestV4ExpertQuantRejectsMalformedMetadataBeforeOutput(t *testing.T) {
	baseName := "layers.0.ffn.experts.0.w1.weight"
	baseScaleName := "layers.0.ffn.experts.0.w1.scale"
	baseWeight := stEntry{Dtype: "I8", Shape: []int{3072, 3584}}
	baseScale := stEntry{Dtype: "F8_E8M0", Shape: []int{3072, 224}}
	weights := make([]byte, 3072*3584)
	scales := make([]byte, 3072*224)

	tests := []struct {
		name        string
		weightName  string
		scaleName   string
		weightEntry stEntry
		scaleEntry  stEntry
		weights     []byte
		scales      []byte
	}{
		{name: "shared expert", weightName: "layers.0.ffn.shared_experts.0.w1.weight", scaleName: baseScaleName, weightEntry: baseWeight, scaleEntry: baseScale, weights: weights, scales: scales},
		{name: "bad identity", weightName: "layers.x.ffn.experts.0.w1.weight", scaleName: baseScaleName, weightEntry: baseWeight, scaleEntry: baseScale, weights: weights, scales: scales},
		{name: "unsupported suffix", weightName: "layers.0.ffn.experts.0.w4.weight", scaleName: baseScaleName, weightEntry: baseWeight, scaleEntry: baseScale, weights: weights, scales: scales},
		{name: "wrong scale pair", weightName: baseName, scaleName: "layers.0.ffn.experts.1.w1.scale", weightEntry: baseWeight, scaleEntry: baseScale, weights: weights, scales: scales},
		{name: "weight dtype", weightName: baseName, scaleName: baseScaleName, weightEntry: stEntry{Dtype: "U8", Shape: baseWeight.Shape}, scaleEntry: baseScale, weights: weights, scales: scales},
		{name: "scale dtype", weightName: baseName, scaleName: baseScaleName, weightEntry: baseWeight, scaleEntry: stEntry{Dtype: "U8", Shape: baseScale.Shape}, weights: weights, scales: scales},
		{name: "weight rank", weightName: baseName, scaleName: baseScaleName, weightEntry: stEntry{Dtype: "I8", Shape: []int{3072, 224, 16}}, scaleEntry: baseScale, weights: weights, scales: scales},
		{name: "weight dimensions", weightName: baseName, scaleName: baseScaleName, weightEntry: stEntry{Dtype: "I8", Shape: []int{3072, 3583}}, scaleEntry: baseScale, weights: weights, scales: scales},
		{name: "scale rank", weightName: baseName, scaleName: baseScaleName, weightEntry: baseWeight, scaleEntry: stEntry{Dtype: "F8_E8M0", Shape: []int{3072, 224, 1}}, weights: weights, scales: scales},
		{name: "scale ratio", weightName: baseName, scaleName: baseScaleName, weightEntry: baseWeight, scaleEntry: stEntry{Dtype: "F8_E8M0", Shape: []int{3072, 223}}, weights: weights, scales: scales},
		{name: "weight length", weightName: baseName, scaleName: baseScaleName, weightEntry: baseWeight, scaleEntry: baseScale, weights: weights[:len(weights)-1], scales: scales},
		{name: "scale length", weightName: baseName, scaleName: baseScaleName, weightEntry: baseWeight, scaleEntry: baseScale, weights: weights, scales: scales[:len(scales)-1]},
		{name: "scale NaN", weightName: baseName, scaleName: baseScaleName, weightEntry: baseWeight, scaleEntry: baseScale, weights: weights, scales: func() []byte { bad := append([]byte(nil), scales...); bad[17] = 0xff; return bad }()},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, shape, err := decodeV4ExpertQuant(tt.weightName, tt.scaleName, tt.weightEntry, tt.scaleEntry, tt.weights, tt.scales)
			if got != nil || shape != nil {
				t.Fatalf("rejection mutated output: bytes=%v shape=%v", got, shape)
			}
			if !errors.Is(err, ErrV4ExpertQuantMetadata) {
				t.Fatalf("error = %v, want ErrV4ExpertQuantMetadata", err)
			}
		})
	}
}

func TestV4ExpertQuantDoesNotSelectGenericMXFP4Decoder(t *testing.T) {
	weightName := "layers.0.ffn.experts.0.w1.weight"
	scaleName := "layers.0.ffn.experts.0.w1.scale"
	weightEntry := stEntry{Dtype: "I8", Shape: []int{3072, 3584}}
	scaleEntry := stEntry{Dtype: "F8_E8M0", Shape: []int{3072, 224}}
	weights := make([]byte, 3072*3584)
	scales := make([]byte, 3072*224)

	decoded, shape, v4Err := decodeV4ExpertQuant(weightName, scaleName, weightEntry, scaleEntry, weights, scales)
	if v4Err != nil || len(decoded) == 0 || !sameShape(shape, []int{3072, 7168}) {
		t.Fatalf("V4 decoder = bytes:%d shape:%v err:%v, want admitted V4 decode", len(decoded), shape, v4Err)
	}
	_, _, genericErr := decodeMXFP4Blocks(weightName, weightEntry, scaleEntry, weights, scales)
	if genericErr == nil || !strings.Contains(genericErr.Error(), "dtype \"I8\", want U8") {
		t.Fatalf("generic MXFP4 decoder error = %v, want incompatible-layout rejection", genericErr)
	}
}
