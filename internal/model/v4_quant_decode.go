package model

import (
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"strings"
)

// ErrV4ExpertQuantMetadata identifies a routed-expert weight/scale pair that
// does not match the immutable DeepSeek-V4-Pro safetensors contract.
var ErrV4ExpertQuantMetadata = errors.New("model: invalid V4 expert quant metadata")

// v4ExpertE2M1Values is the OCP MX E2M1 finite value table indexed by
// one unpacked nibble. PyTorch float4_e2m1fn_x2 stores val0 in the low
// nibble and val1 in the high nibble.
var v4ExpertE2M1Values = [16]float32{
	0, 0.5, 1, 1.5, 2, 3, 4, 6,
	float32(math.Copysign(0, -1)), -0.5, -1, -1.5, -2, -3, -4, -6,
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

// decodeV4ExpertQuant decodes the exact routed-expert format observed in
// deepseek-ai/DeepSeek-V4-Pro@b5968e9190ef611bbf34a7229255be88a0e937c1.
// The pinned inference code views each byte as torch.float4_e2m1fn_x2 and
// applies one F8_E8M0 scale per 32 unpacked K values. PyTorch's OCP type
// definition establishes low-nibble val0/high-nibble val1 ordering. The
// destination is allocated only after all metadata and scale bytes validate.
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

	for i, scale := range scales {
		if scale == 0xff {
			return nil, nil, v4QuantMetadataf("%s scale byte %d is F8_E8M0 NaN", scaleName, i)
		}
	}

	unpackedCols, ok := checkedShapeProduct(spec.weightCols, 2)
	if !ok {
		return nil, nil, v4QuantMetadataf("%s unpacked shape overflows", weightName)
	}
	outElems, ok := checkedShapeProduct(spec.weightRows, unpackedCols)
	if !ok || outElems > int(^uint(0)>>1)/4 {
		return nil, nil, v4QuantMetadataf("%s decoded byte count overflows", weightName)
	}
	out := make([]byte, outElems*4)
	for row := 0; row < spec.weightRows; row++ {
		packedRow := weights[row*spec.weightCols : (row+1)*spec.weightCols]
		scaleRow := scales[row*spec.scaleCols : (row+1)*spec.scaleCols]
		for packedCol, packed := range packedRow {
			exp := int(scaleRow[packedCol/16]) - 127
			base := (row*unpackedCols + packedCol*2) * 4
			lo := float32(math.Ldexp(float64(v4ExpertE2M1Values[packed&0x0f]), exp))
			hi := float32(math.Ldexp(float64(v4ExpertE2M1Values[packed>>4]), exp))
			binary.LittleEndian.PutUint32(out[base:], math.Float32bits(lo))
			binary.LittleEndian.PutUint32(out[base+4:], math.Float32bits(hi))
		}
	}
	return out, []int{spec.weightRows, unpackedCols}, nil
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
