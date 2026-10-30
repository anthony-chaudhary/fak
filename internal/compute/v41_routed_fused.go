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
// therefore bitwise the same backend's id path's, which the parity witness
// checks (see V41RoutedIDBackend for how the id path relates to the per-pick
// path on each backend).
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
	if out <= 0 || in <= 0 || in > maxInt32 || in%256 != 0 || out > maxInt32/picks {
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
	if picks <= 0 || picks > V41RoutedIDMaxPicks || in <= 0 || in > maxInt32 {
		return 0, fmt.Errorf("compute: V4.1 routed fused activation geometry exceeds the device count domain")
	}
	// A shared row remains valid even when repeating it would exceed the
	// device domain. Do not multiply the per-pick width in a host int.
	perPick := uint64(picks) * uint64(in)
	switch {
	case x.Shape[0] == in:
		return 0, nil
	case uint64(x.Shape[0]) == perPick:
		if perPick > maxInt32 {
			return 0, fmt.Errorf("compute: V4.1 routed fused activation exceeds the device count domain")
		}
		return in, nil
	default:
		return 0, fmt.Errorf("compute: V4.1 routed fused activation width %d is neither %d nor %d", x.Shape[0], in, perPick)
	}
}

// v41RoutedFusedGeometry carries checked byte extents, not allocation sizes
// inferred from tensor shapes after narrowing to C.int. Weight bindings may
// be larger ring pages; these are the minimum bytes each dispatch reads.
type v41RoutedFusedGeometry struct {
	weightBytes, inputBytes, idsBytes int
	outputBytes, statusBytes          int
}

// checkedV41RoutedFusedGeometry mirrors the native count/address domains before
// any native call or output allocation. Q2_K is gate/up; Q3_K is down/sum.
func checkedV41RoutedFusedGeometry(rows, in, picks, xStride int, dtype Dtype) (v41RoutedFusedGeometry, error) {
	bad := func() (v41RoutedFusedGeometry, error) {
		return v41RoutedFusedGeometry{}, fmt.Errorf("compute: V4.1 routed fused geometry exceeds the native count or byte domain")
	}
	if picks <= 0 || picks > V41RoutedIDMaxPicks || rows <= 0 || rows > maxInt32/picks ||
		in <= 0 || in > maxInt32 || in%256 != 0 || (xStride != 0 && xStride != in) {
		return bad()
	}
	blockBytes, outputCount, statusBytes := 84, rows*picks, v41RoutedSwiGLUStatusBytes
	inputCount := in
	switch dtype {
	case Q2_K:
		if xStride != 0 {
			if in > maxInt32/picks {
				return bad()
			}
			inputCount = picks * in
		}
	case Q3_K:
		if xStride != in || in > maxInt32/picks {
			return bad()
		}
		blockBytes, outputCount, statusBytes = 110, rows, v41BF16CheckedStatusBytes
		inputCount = picks * in
	default:
		return bad()
	}
	// All byte counts must also fit the host allocation/size_t domain. The
	// shader addresses quantized weights in uint32 bytes, even on 64-bit hosts.
	const quantizedByteLimit = uint64(1<<32 - 1)
	rowBytes, ok := checkedV41RoutedFusedBytes(in/256, blockBytes, quantizedByteLimit)
	if !ok {
		return bad()
	}
	var g v41RoutedFusedGeometry
	if g.weightBytes, ok = checkedV41RoutedFusedWeightBytes(rows, rowBytes); !ok {
		return bad()
	}
	hostLimit := uint64(^uint(0) >> 1)
	if g.inputBytes, ok = checkedV41RoutedFusedBytes(inputCount, 4, hostLimit); !ok {
		return bad()
	}
	if g.outputBytes, ok = checkedV41RoutedFusedBytes(outputCount, 4, hostLimit); !ok {
		return bad()
	}
	if g.idsBytes, ok = checkedV41RoutedFusedBytes(picks, 12, hostLimit); !ok {
		return bad()
	}
	if g.statusBytes, ok = checkedV41RoutedFusedBytes(picks, statusBytes, hostLimit); !ok {
		return bad()
	}
	return g, nil
}

// The shaders read quantized bytes through uint32 words. An odd number of
// 110-byte Q3_K blocks therefore needs two resident tail bytes, matching upload.
func checkedV41RoutedFusedWeightBytes(rows, rowBytes int) (int, bool) {
	const limit = uint64(1<<32 - 1)
	payload, ok := checkedV41RoutedFusedBytes(rows, rowBytes, limit)
	if !ok {
		return 0, false
	}
	resident := (uint64(payload) + 3) &^ uint64(3)
	if resident > limit || resident > uint64(^uint(0)>>1) {
		return 0, false
	}
	return int(resident), true
}

func v41RoutedFusedExtentOK(available uint64, required int) bool {
	return required > 0 && available >= uint64(required)
}

func checkedV41RoutedFusedBytes(count, width int, limit uint64) (int, bool) {
	if hostLimit := uint64(^uint(0) >> 1); limit > hostLimit {
		limit = hostLimit
	}
	if count <= 0 || width <= 0 || uint64(count) > limit/uint64(width) {
		return 0, false
	}
	return count * width, true
}

// maxInt32 is the device count domain the fused GEMV checks against, matching
// the id path's math.MaxInt32 guards.
const maxInt32 = 1<<31 - 1
