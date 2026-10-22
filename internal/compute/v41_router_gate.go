package compute

import (
	"fmt"
	"slices"
)

// Bounds of the fused router gate module: the routing workgroup keeps one
// selection value per expert in shared memory, and each 1024-wide input chunk
// of a logit runs on its own lane of a 64-lane workgroup.
const (
	V41RouterGateTopKMaxExperts = 1024
	V41RouterGateTopKMaxHidden  = 64 * 1024
)

// V41RouterGateTopKBackend is the optional DeepSeek-V4.1 router with the gate
// projection fused ahead of the top-k, for one token row. weight is the live
// row-major F32 [experts,hidden] gate, x the live F32 [hidden] activation, and
// bias follows V41RouterTopKBackend. The logits never leave the device; the
// result, faults and failure classes are V41RouterTopKBackend's.
//
// The logits are the one-token MatMul(weight, x) F32 path bit for bit, so the
// picks and weights equal MatMul followed by V41RouterTopK. Available reports
// true only after that equality was witnessed on the executing device.
type V41RouterGateTopKBackend interface {
	Backend
	V41RouterGateTopKAvailable() bool
	V41RouterGateTopK(weight, x, bias Tensor, experts, hidden, topK int, routeScale float32, full, hasBias bool) (V41RouterPicks, error)
}

// v41RouterGateGroups is the fused module's workgroup count: one lane per
// 1024-wide input chunk, so each 64-lane group owns 64/chunks logits.
func v41RouterGateGroups(experts, hidden int) int {
	chunks := (hidden + 1023) / 1024
	perGroup := 64 / chunks
	return (experts + perGroup - 1) / perGroup
}

// validateV41RouterGateTopK rejects malformed geometry before narrowing it
// for the 32-bit shader ABI. It reads no tensor data and allocates nothing.
func validateV41RouterGateTopK(weight, x, bias Tensor, experts, hidden, topK int, routeScale float32, hasBias bool) (weightBytes, xBytes, biasBytes, logitBytes, pickBytes int, err error) {
	const maxABIInt = 1<<31 - 1
	if experts <= 0 || experts > V41RouterGateTopKMaxExperts || hidden <= 0 || hidden > V41RouterGateTopKMaxHidden ||
		topK <= 0 || topK > experts || topK > V41RouterTopKMaxK || uint64(experts)*uint64(hidden) > maxABIInt {
		return 0, 0, 0, 0, 0, fmt.Errorf("compute: V4.1 router gate invalid geometry experts=%d hidden=%d topK=%d", experts, hidden, topK)
	}
	if !(routeScale > 0) || routeScale > 3.4028234663852886e38 {
		return 0, 0, 0, 0, 0, fmt.Errorf("compute: V4.1 router gate scale must be finite and positive, got %g", routeScale)
	}
	biasShape := []int{1}
	if hasBias {
		biasShape = []int{experts}
	}
	for _, operand := range []struct {
		name string
		t    Tensor
		want []int
	}{
		{"weight", weight, []int{experts, hidden}},
		{"x", x, []int{hidden}},
		{"bias", bias, biasShape},
	} {
		if operand.t.Dtype != F32 || operand.t.Layout != RowMajor || operand.t.Quant != nil || !slices.Equal(operand.t.Shape, operand.want) {
			return 0, 0, 0, 0, 0, fmt.Errorf("compute: V4.1 router gate %s must be unquantized row-major F32 with shape %v", operand.name, operand.want)
		}
	}
	return experts * hidden * 4, hidden * 4, biasShape[0] * 4, experts * 4, topK * 4, nil
}

// v41RouterGateStatusSeed is the host seed of the four status words: an
// invalid stage in word 0 until the routing group publishes, and the arrival
// counter in word 3, which wraps to zero on the first arrival.
var v41RouterGateStatusSeed = [4]uint32{^uint32(0), 0, 0, ^uint32(0)}

// v41RouterGateProbeFixture is a deterministic row with mixed-sign mantissa
// noise. Experts 5, 37 and 69 share one weight row aligned with x and one bias,
// so they tie exactly at the top and exercise the lower-expert-first order.
func v41RouterGateProbeFixture(experts, hidden int) (w, x, bias []float32) {
	value := func(i, salt int) float32 {
		h := uint32(i)*2654435761 + uint32(salt)*40503
		h ^= h >> 15
		h *= 2246822519
		h ^= h >> 13
		return float32(int32(h>>8)-(1<<23)) / float32(1<<23)
	}
	x = make([]float32, hidden)
	for i := range x {
		x[i] = value(i, 104729)
	}
	w = make([]float32, experts*hidden)
	for o := 0; o < experts; o++ {
		for i := 0; i < hidden; i++ {
			if o%32 == 5 {
				w[o*hidden+i] = 0.006 * x[i]
			} else {
				w[o*hidden+i] = 0.05 * value(o*hidden+i, 7919)
			}
		}
	}
	bias = make([]float32, experts)
	for e := range bias {
		if e%32 != 5 {
			bias[e] = 0.01 * float32(e%7)
		}
	}
	return w, x, bias
}
