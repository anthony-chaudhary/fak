package compute

import (
	"math"
	"testing"
)

func flashDecodeFixture() flashMLADecodeInput {
	x := flashMLADecodeInput{Query: make([]uint16, 32768), KV: make([]byte, 528), Indices: make([]int32, 64), Length: 64, Blocks: 1, PageSize: 1, Scale: 1}
	for i := 512; i < 528; i++ {
		x.KV[i] = 127
	}
	return x
}
func TestFlashMLADecodeQualificationContracts(t *testing.T) {
	if err := validateFlashMLADecode(flashDecodeFixture()); err != nil {
		t.Fatal(err)
	}
	for name, mutate := range map[string]func(*flashMLADecodeInput){
		"stride":   func(x *flashMLADecodeInput) { x.KV = x.KV[:527] },
		"bf16":     func(x *flashMLADecodeInput) { x.Query[0] = 0x7f80 },
		"fp8":      func(x *flashMLADecodeInput) { x.KV[0] = 0x7f },
		"e8m0":     func(x *flashMLADecodeInput) { x.KV[512] = 255 },
		"index":    func(x *flashMLADecodeInput) { x.Indices[63] = 1 },
		"length":   func(x *flashMLADecodeInput) { x.Length = 65 },
		"scale":    func(x *flashMLADecodeInput) { x.Scale = float32(math.Inf(1)) },
		"extra":    func(x *flashMLADecodeInput) { x.ExtraLength = 1 },
		"overflow": func(x *flashMLADecodeInput) { x.Blocks = math.MaxInt32 },
	} {
		t.Run(name, func(t *testing.T) {
			x := flashDecodeFixture()
			mutate(&x)
			if validateFlashMLADecode(x) == nil {
				t.Fatal("admitted malformed input")
			}
		})
	}
}
func TestFlashMLADecodeDonorEmptyLSE(t *testing.T) {
	x := flashDecodeFixture()
	for i := range x.Indices {
		x.Indices[i] = -1
	}
	y := flashMLADecodeOutput{ValuesBF16: make([]uint16, 32768), LSE: make([]float32, 64)}
	for i := range y.LSE {
		y.LSE[i] = float32(math.Inf(1))
	}
	if err := validateFlashMLADecodeOutput(x, y); err != nil {
		t.Fatal(err)
	}
	y.LSE[0] = float32(math.Inf(-1))
	if validateFlashMLADecodeOutput(x, y) == nil {
		t.Fatal("accepted F32 oracle empty LSE")
	}
	y.LSE[0] = float32(math.Inf(1))
	x.Indices[0] = 0
	if validateFlashMLADecodeOutput(x, y) == nil {
		t.Fatal("accepted nonempty infinite LSE")
	}
}

func TestFlashMLADecodeScaleOverflow(t *testing.T) {
	x := flashDecodeFixture()
	x.KV[512] = 254
	x.KV[0] = 0x3f
	if err := validateFlashMLADecode(x); err != nil {
		t.Fatal(err)
	}
	x.KV[0] = 0x40
	if validateFlashMLADecode(x) == nil {
		t.Fatal("accepted scaled BF16 overflow")
	}
	x.KV[0] = 0
	if err := validateFlashMLADecode(x); err != nil {
		t.Fatal("zero at large finite scale", err)
	}
}
