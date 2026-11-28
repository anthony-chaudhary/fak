package compute

import (
	"errors"
	"math"
	"testing"
)

// Closed in-memory oracle for Luce-Org/lucebox-hub Blackwell NVFP4 persistent decode (ADAPT).
// [SW-VERIFIED] structure and oracle only: no device run, no loader change, no speed claim.
var errNVFP4DecodeShape = errors.New("megakernel nvfp4 decode: invalid packed weight/scale geometry")

// E2M1 4-bit FP4 lookup table (sign bit 3, 2 exp bits, 1 mantissa bit).
var nvfp4E2M1LUT = [16]float32{
	0, 0.5, 1, 1.5, 2, 3, 4, 6,
	-0, -0.5, -1, -1.5, -2, -3, -4, -6,
}

func nvfp4FusedGEMV(packed []byte, scales []float32, x []float32, groupSize int) (float32, error) {
	if len(x) == 0 || len(x)%groupSize != 0 || len(packed)*2 != len(x) || len(scales) != len(x)/groupSize {
		return 0, errNVFP4DecodeShape
	}
	var acc float32
	for g := 0; g < len(scales); g++ {
		sc := scales[g]
		base := g * groupSize
		for i := 0; i < groupSize; i++ {
			b := packed[(base+i)/2]
			nib := b & 0x0f
			if (base+i)%2 == 1 {
				nib = (b >> 4) & 0x0f
			}
			acc += nvfp4E2M1LUT[nib] * sc * x[base+i]
		}
	}
	return acc, nil
}

func TestMegakernelNVFP4Decode(t *testing.T) {
	t.Parallel()
	group := cuhDefine(t, "megakernel_nvfp4_decode.cuh", "FAK_NVFP4_DEC_GROUP_SIZE")
	if cuhDefine(t, "megakernel_nvfp4_decode.cuh", "FAK_NVFP4_DEC_PACKED_PER_BYTE") != 2 ||
		cuhDefine(t, "megakernel_nvfp4_decode.cuh", "FAK_NVFP4_DEC_SM_MIN") != 120 ||
		cuhDefine(t, "megakernel_nvfp4_decode.cuh", "FAK_NVFP4_DEC_SINGLE_LAUNCH") != 1 {
		t.Fatal("megakernel_nvfp4_decode.cuh constants drifted")
	}

	// 1 group of 16 elements -> 8 packed bytes, 1 block scale.
	packed := []byte{0x21, 0x43, 0xA9, 0x05, 0x11, 0x22, 0x33, 0x80}
	scales := []float32{0.5}
	x := make([]float32, group)
	for i := range x {
		x[i] = float32(i + 1)
	}

	// Reference: unpack+dequantize first, then dot product.
	dequant := make([]float32, group)
	for i := 0; i < group; i++ {
		b := packed[i/2]
		nib := b & 0x0f
		if i%2 == 1 {
			nib = (b >> 4) & 0x0f
		}
		dequant[i] = nvfp4E2M1LUT[nib] * scales[0]
	}
	var want float32
	for i := 0; i < group; i++ {
		want += dequant[i] * x[i]
	}

	got, err := nvfp4FusedGEMV(packed, scales, x, group)
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(float64(got-want)) > 1e-5 {
		t.Fatalf("fused=%v want=%v", got, want)
	}

	if _, err := nvfp4FusedGEMV(packed[:7], scales, x, group); !errors.Is(err, errNVFP4DecodeShape) {
		t.Fatalf("bad packed len err = %v, want errNVFP4DecodeShape", err)
	}
}
