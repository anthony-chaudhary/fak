package compute

import (
	"fmt"
	"math"
)

// Qualification-only input, never the gathered-F32 shared-attention contract.
// KV records are 512 E4M3 bytes followed by 16 E8M0 scale bytes (528 bytes).
type flashMLADecodeInput struct {
	Query                                        []uint16
	KV, ExtraKV                                  []byte
	Indices, ExtraIndices                        []int32
	Length, ExtraLength                          int32
	Sink                                         []float32
	Blocks, PageSize, ExtraBlocks, ExtraPageSize int
	Scale                                        float32
	BatchInvariant                               bool
}
type flashMLADecodeOutput struct {
	ValuesBF16 []uint16
	LSE        []float32
}

const flashMLADecodeHostLimit = 64 << 20

func validateFlashMLADecode(x flashMLADecodeInput) error {
	bad := func(s string) error { return fmt.Errorf("FlashMLA decode qualification: %s", s) }
	if len(x.Query) != 32768 || (len(x.Sink) != 0 && len(x.Sink) != 64) || !(x.Scale > 0) || math.IsInf(float64(x.Scale*1.4426950408889634), 0) || math.IsNaN(float64(x.Scale)) {
		return bad("query/sink/scale")
	}
	total := 65536 + len(x.Sink)*4
	validate := func(kv []byte, ids []int32, n int32, blocks, page int, extra bool) error {
		if extra && len(ids) == 0 {
			if len(kv) != 0 || n != 0 || blocks != 0 || page != 0 {
				return bad("unused extra cache")
			}
			return nil
		}
		if blocks <= 0 || page <= 0 || page > math.MaxInt32/528 || blocks > (math.MaxInt32/528)/page || len(kv) != blocks*page*528 {
			return bad("cache geometry")
		}
		if len(ids) == 0 || len(ids) > 1<<20 || len(ids)%64 != 0 || n < 0 || int(n) > len(ids) {
			return bad("topk/length")
		}
		if len(kv) > flashMLADecodeHostLimit-total || len(ids)*4 > flashMLADecodeHostLimit-total-len(kv) {
			return bad("qualification staging limit")
		}
		total += len(kv) + len(ids)*4
		for _, id := range ids {
			if id < -1 || int64(id) >= int64(blocks)*int64(page) {
				return bad("index out of range")
			}
		}
		for row := 0; row < len(kv); row += 528 {
			for d := 0; d < 512; d++ {
				if kv[row+d]&127 == 127 {
					return bad("nonfinite E4M3")
				}
			}
			for d := 512; d < 528; d++ {
				if kv[row+d] == 255 {
					return bad("nonfinite E8M0")
				}
			}
			// Actual donor converts E4M3 and E8M0 to BF16, then multiplies BF16.
			// Refuse a mathematically finite encoding whose decoded operand overflows.
			for d := 0; d < 512; d++ {
				v := kv[row+d] & 127
				exponent := int(v >> 3)
				mantissa := int(v & 7)
				power := -9
				if exponent != 0 {
					mantissa += 8
					power = exponent - 10
				}
				decoded := math.Ldexp(float64(mantissa), power+int(kv[row+512+d/32])-127)
				if decoded > float64(math.Float32frombits(0x7f7f0000)) {
					return bad("decoded BF16 overflow")
				}
			}
		}
		return nil
	}
	if err := validate(x.KV, x.Indices, x.Length, x.Blocks, x.PageSize, false); err != nil {
		return err
	}
	if err := validate(x.ExtraKV, x.ExtraIndices, x.ExtraLength, x.ExtraBlocks, x.ExtraPageSize, true); err != nil {
		return err
	}
	for _, q := range x.Query {
		if q&0x7f80 == 0x7f80 {
			return bad("nonfinite BF16 query")
		}
	}
	for _, s := range x.Sink {
		if math.IsNaN(float64(s)) || math.IsInf(float64(s), 0) {
			return bad("nonfinite sink")
		}
	}
	return nil
}

// Donor LSE excludes the sink and returns +Inf (not the F32 oracle's -Inf)
// for all-invalid selections. Numerical failure never publishes partial output.
func validateFlashMLADecodeOutput(x flashMLADecodeInput, y flashMLADecodeOutput) error {
	if len(y.ValuesBF16) != 32768 || len(y.LSE) != 64 {
		return fmt.Errorf("FlashMLA decode output shape")
	}
	valid := false
	for _, id := range x.Indices[:int(x.Length)] {
		valid = valid || id >= 0
	}
	for _, id := range x.ExtraIndices[:int(x.ExtraLength)] {
		valid = valid || id >= 0
	}
	for _, v := range y.ValuesBF16 {
		if v&0x7f80 == 0x7f80 || (!valid && v&0x7fff != 0) {
			return fmt.Errorf("FlashMLA decode invalid output")
		}
	}
	for _, v := range y.LSE {
		if math.IsNaN(float64(v)) || (valid && math.IsInf(float64(v), 0)) || (!valid && !math.IsInf(float64(v), 1)) {
			return fmt.Errorf("FlashMLA decode invalid LSE")
		}
	}
	return nil
}
