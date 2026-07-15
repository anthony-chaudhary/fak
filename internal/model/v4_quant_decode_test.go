package model

import (
	"errors"
	"strings"
	"testing"
)

func TestV4ExpertQuantPinnedContractRefusesUnknownArithmetic(t *testing.T) {
	for _, projection := range []string{"w1", "w2", "w3"} {
		spec := v4ExpertQuantSpecs[projection]
		weightName := "layers.60.ffn.experts.383." + projection + ".weight"
		scaleName := "layers.60.ffn.experts.383." + projection + ".scale"
		weightEntry := stEntry{Dtype: "I8", Shape: []int{spec.weightRows, spec.weightCols}}
		scaleEntry := stEntry{Dtype: "F8_E8M0", Shape: []int{spec.scaleRows, spec.scaleCols}}
		weights := make([]byte, spec.weightRows*spec.weightCols)
		scales := make([]byte, spec.scaleRows*spec.scaleCols)

		got, shape, err := decodeV4ExpertQuant(weightName, scaleName, weightEntry, scaleEntry, weights, scales)
		if got != nil || shape != nil {
			t.Fatalf("%s refusal mutated output: bytes=%v shape=%v", projection, got, shape)
		}
		if !errors.Is(err, ErrV4ExpertQuantUnsupported) {
			t.Fatalf("%s error = %v, want ErrV4ExpertQuantUnsupported", projection, err)
		}
		var unsupported *V4ExpertQuantUnsupportedError
		if !errors.As(err, &unsupported) {
			t.Fatalf("%s error type = %T, want *V4ExpertQuantUnsupportedError", projection, err)
		}
		wantMissing := []string{"nibble order", "FP4 value mapping", "F8_E8M0 special-value behavior", "scale application rule"}
		if strings.Join(unsupported.MissingSemantics, "|") != strings.Join(wantMissing, "|") {
			t.Fatalf("%s missing semantics = %v, want %v", projection, unsupported.MissingSemantics, wantMissing)
		}
		if errors.Is(err, ErrV4ExpertQuantMetadata) {
			t.Fatalf("%s valid pinned metadata classified malformed: %v", projection, err)
		}
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
			if errors.Is(err, ErrV4ExpertQuantUnsupported) {
				t.Fatalf("malformed metadata reached unsupported arithmetic boundary: %v", err)
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

	_, _, v4Err := decodeV4ExpertQuant(weightName, scaleName, weightEntry, scaleEntry, weights, scales)
	if !errors.Is(v4Err, ErrV4ExpertQuantUnsupported) {
		t.Fatalf("V4 decoder error = %v, want typed unsupported refusal", v4Err)
	}
	_, _, genericErr := decodeMXFP4Blocks(weightName, weightEntry, scaleEntry, weights, scales)
	if genericErr == nil || !strings.Contains(genericErr.Error(), "dtype \"I8\", want U8") {
		t.Fatalf("generic MXFP4 decoder error = %v, want incompatible-layout rejection", genericErr)
	}
}
