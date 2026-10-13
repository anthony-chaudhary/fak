package compute

import "fmt"

// V41RoutedFusedBackend records one DeepSeek-V4.1 decode layer's routed FFN as
// two dispatches over every pick (fak-private#3341), instead of the routed id
// path's five: gate and up GEMV with the BF16 clamp-SwiGLU route-weight
// publication folded into the GEMV epilogue, then down GEMV with the checked
// BF16 publication and the routed sum folded into its epilogue. It ports
// antirez/ds4's moe_gate_up_mid_decode_lut_qwarp32 and moe_down_sum_qwarp32
// kernels onto the routed id path's ids/slot-map binding.
//
// Every per-pick term and status record is the id path's: the gate and up rows
// are the Q2_K id kernel term for term, the down row is the Q3_K id kernel term
// for term, and each pick's 16-byte status record carries exactly the words the
// per-stage id record carries. A fused layer's mid, sum and verdicts are
// therefore bitwise the id path's, which the parity witness checks.
type V41RoutedFusedBackend interface {
	V41RoutedFusedAvailable() bool
	// SupportsV41RoutedFusedGateUp reports a (gate, up) dtype pair the fused
	// gate/up kernel serves on this device.
	SupportsV41RoutedFusedGateUp(gate, up Dtype) bool
	// SupportsV41RoutedFusedDown reports a down dtype the fused down/sum kernel
	// serves on this device.
	SupportsV41RoutedFusedDown(down Dtype) bool
	// V41RoutedGateUpSwiGLUDeferred is one dispatch over every pick: for pick p,
	// gates[p] and ups[p] are its [inter,hidden] slot weights, x is the GEMV
	// activation (len hidden, or picks*hidden for a per-pick row), and
	// mid[p*inter+i] is the clamped, weighted, BF16-published down operand. ids
	// is the [3*picks] image V41RoutedIDs uploads: each pick's slot, its route
	// weight's binary32 bits and the routed-sum order. The status owns picks
	// records, each exactly the record V41RoutedSwiGLUIDDeferred writes for that
	// pick.
	V41RoutedGateUpSwiGLUDeferred(gates, ups []Tensor, ids Tensor, x Tensor, picks int, limit float32) (Tensor, V41DeferredStatus, error)
	// V41RoutedDownSumDeferred is one dispatch over every pick: downs[p] is pick
	// p's [hidden,inter] slot weight, mid is V41RoutedGateUpSwiGLUDeferred's
	// output, and the result is the checked BF16 publication of every pick's down
	// row summed in ids order. The status owns picks records, each exactly the
	// record V41RoutedPublishSumDeferred writes for that pick.
	V41RoutedDownSumDeferred(downs []Tensor, mid, ids Tensor, picks int, site uint32) (Tensor, V41DeferredStatus, error)
}

// validateV41RoutedFusedWeights checks one projection's pick weights share the
// single [out,in] shape and dtype the fused GEMV serves, and returns that shape.
// It precedes any native call and reads no tensor data.
func validateV41RoutedFusedWeights(name string, weights []Tensor, picks int, supported func(Dtype) bool) (out, in int, err error) {
	if picks <= 0 || picks > V41RoutedIDMaxPicks || len(weights) != picks {
		return 0, 0, fmt.Errorf("compute: V4.1 routed fused %s needs 1..%d picks and one weight each", name, V41RoutedIDMaxPicks)
	}
	first := weights[0]
	if len(first.Shape) != 2 {
		return 0, 0, fmt.Errorf("compute: V4.1 routed fused %s slots must be [out,in] matrices", name)
	}
	if !supported(first.Dtype) {
		return 0, 0, fmt.Errorf("compute: V4.1 routed fused %s dtype %v is not served by this device", name, first.Dtype)
	}
	out, in = first.Shape[0], first.Shape[1]
	if out <= 0 || in <= 0 || in%256 != 0 || out > maxInt32/picks {
		return 0, 0, fmt.Errorf("compute: V4.1 routed fused %s geometry [%d,%d] is not a k-quant row set", name, out, in)
	}
	for _, w := range weights[1:] {
		if w.Dtype != first.Dtype || len(w.Shape) != 2 || w.Shape[0] != out || w.Shape[1] != in {
			return 0, 0, fmt.Errorf("compute: V4.1 routed fused %s slots must share one [out,in] shape and dtype", name)
		}
	}
	return out, in, nil
}

// validateV41RoutedFusedActivation checks the shared or per-pick GEMV activation
// and returns its per-pick stride (0 = one shared row).
func validateV41RoutedFusedActivation(x Tensor, picks, in int) (int, error) {
	if x.Dtype != F32 || x.Quant != nil || len(x.Shape) != 1 {
		return 0, fmt.Errorf("compute: V4.1 routed fused activation must be an unquantized F32 vector")
	}
	switch x.Shape[0] {
	case in:
		return 0, nil
	case picks * in:
		return in, nil
	default:
		return 0, fmt.Errorf("compute: V4.1 routed fused activation width %d is neither %d nor %d", x.Shape[0], in, picks*in)
	}
}

// maxInt32 is the device count domain the fused GEMV checks against, matching
// the id path's math.MaxInt32 guards.
const maxInt32 = 1<<31 - 1
