package model

import (
	"errors"
	"fmt"
	"strings"
)

var (
	// ErrV4ExpertQuantMetadata identifies a routed-expert weight/scale pair that
	// does not match the immutable DeepSeek-V4-Pro safetensors contract.
	ErrV4ExpertQuantMetadata = errors.New("model: invalid V4 expert quant metadata")
	// ErrV4ExpertQuantUnsupported identifies a valid pinned artifact pair whose
	// arithmetic semantics have not yet been grounded by an authoritative oracle.
	ErrV4ExpertQuantUnsupported = errors.New("model: unsupported V4 expert quant format")
)

// V4ExpertQuantUnsupportedError records the semantic facts that immutable
// safetensors metadata cannot establish. Keeping these fields explicit prevents
// an I8/F8_E8M0 payload from silently taking the unrelated GPT-OSS MXFP4 path.
type V4ExpertQuantUnsupportedError struct {
	WeightName       string
	MissingSemantics []string
}

func (e *V4ExpertQuantUnsupportedError) Error() string {
	return fmt.Sprintf("%v for %s: missing authoritative %s", ErrV4ExpertQuantUnsupported, e.WeightName, strings.Join(e.MissingSemantics, ", "))
}

func (e *V4ExpertQuantUnsupportedError) Unwrap() error { return ErrV4ExpertQuantUnsupported }

var v4ExpertQuantMissingSemantics = []string{
	"nibble order",
	"FP4 value mapping",
	"F8_E8M0 special-value behavior",
	"scale application rule",
}

type v4ExpertQuantSpec struct {
	weightRows int
	weightCols int
	scaleRows  int
	scaleCols  int
}

var v4ExpertQuantSpecs = map[string]v4ExpertQuantSpec{
	"w1": {weightRows: 3072, weightCols: 3584, scaleRows: 3072, scaleCols: 224},
	"w2": {weightRows: 7168, weightCols: 1536, scaleRows: 7168, scaleCols: 96},
	"w3": {weightRows: 3072, weightCols: 3584, scaleRows: 3072, scaleCols: 224},
}

// decodeV4ExpertQuant admits only the exact routed-expert format observed in
// deepseek-ai/DeepSeek-V4-Pro@b5968e9190ef611bbf34a7229255be88a0e937c1.
//
// The pinned headers establish identity, layout, and byte counts, but not the
// four arithmetic semantics named by V4ExpertQuantUnsupportedError. Therefore
// a metadata-valid pair deliberately returns a typed refusal and no output.
// An authoritative independent oracle can replace that final refusal without
// weakening this admission boundary.
func decodeV4ExpertQuant(weightName, scaleName string, weightEntry, scaleEntry stEntry, weights, scales []byte) ([]byte, []int, error) {
	stem, projection, err := parseV4ExpertQuantWeightName(weightName)
	if err != nil {
		return nil, nil, err
	}
	spec := v4ExpertQuantSpecs[projection]
	wantScaleName := stem + projection + ".scale"
	if scaleName != wantScaleName {
		return nil, nil, v4QuantMetadataf("scale name %q, want %q", scaleName, wantScaleName)
	}

	if weightEntry.Dtype != "I8" {
		return nil, nil, v4QuantMetadataf("%s dtype %q, want I8", weightName, weightEntry.Dtype)
	}
	if scaleEntry.Dtype != "F8_E8M0" {
		return nil, nil, v4QuantMetadataf("%s dtype %q, want F8_E8M0", scaleName, scaleEntry.Dtype)
	}
	if !sameShape(weightEntry.Shape, []int{spec.weightRows, spec.weightCols}) {
		return nil, nil, v4QuantMetadataf("%s shape %v, want [%d %d]", weightName, weightEntry.Shape, spec.weightRows, spec.weightCols)
	}
	if !sameShape(scaleEntry.Shape, []int{spec.scaleRows, spec.scaleCols}) {
		return nil, nil, v4QuantMetadataf("%s shape %v, want [%d %d]", scaleName, scaleEntry.Shape, spec.scaleRows, spec.scaleCols)
	}
	if spec.weightCols != spec.scaleCols*16 {
		return nil, nil, v4QuantMetadataf("%s scale ratio %d:%d, want 16 packed weight bytes per scale", weightName, spec.weightCols, spec.scaleCols)
	}
	weightBytes, ok := checkedShapeProduct(spec.weightRows, spec.weightCols)
	if !ok {
		return nil, nil, v4QuantMetadataf("%s shape overflows byte count", weightName)
	}
	scaleBytes, ok := checkedShapeProduct(spec.scaleRows, spec.scaleCols)
	if !ok {
		return nil, nil, v4QuantMetadataf("%s scale shape overflows byte count", scaleName)
	}
	if len(weights) != weightBytes {
		return nil, nil, v4QuantMetadataf("%s has %d bytes, want %d", weightName, len(weights), weightBytes)
	}
	if len(scales) != scaleBytes {
		return nil, nil, v4QuantMetadataf("%s has %d bytes, want %d", scaleName, len(scales), scaleBytes)
	}

	missing := append([]string(nil), v4ExpertQuantMissingSemantics...)
	return nil, nil, &V4ExpertQuantUnsupportedError{WeightName: weightName, MissingSemantics: missing}
}

func parseV4ExpertQuantWeightName(name string) (stem, projection string, err error) {
	const marker = ".ffn.experts."
	if !strings.HasPrefix(name, "layers.") || !strings.Contains(name, marker) {
		return "", "", v4QuantMetadataf("weight name %q is not a routed expert", name)
	}
	for projection := range v4ExpertQuantSpecs {
		suffix := projection + ".weight"
		if strings.HasSuffix(name, suffix) {
			stem := strings.TrimSuffix(name, suffix)
			identity := strings.TrimSuffix(stem, ".")
			parts := strings.Split(identity, ".")
			if len(parts) == 5 && parts[0] == "layers" && decimalOnly(parts[1]) && parts[2] == "ffn" && parts[3] == "experts" && decimalOnly(parts[4]) {
				return stem, projection, nil
			}
		}
	}
	return "", "", v4QuantMetadataf("weight name %q has unsupported suffix or identity", name)
}

func decimalOnly(s string) bool {
	if s == "" {
		return false
	}
	for _, r := range s {
		if r < '0' || r > '9' {
			return false
		}
	}
	return true
}

func v4QuantMetadataf(format string, args ...any) error {
	return fmt.Errorf("%w: %s", ErrV4ExpertQuantMetadata, fmt.Sprintf(format, args...))
}
