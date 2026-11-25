package model

// v41ExpertGateUpOutcome is the closed result vocabulary of one v41ExpertGateUpFunc
// call. handled-success returns the I-wide fused intermediate for the existing host
// down contraction; declined leaves the pick to the historical host gate/up/down
// triple; handled-error is a SELECTED execution failure and must surface, never be
// swallowed as a decline.
type v41ExpertGateUpOutcome uint8

const (
	// v41GateUpDeclined: the shared device operation did not admit this expert, so
	// the caller falls through to the host triple byte-for-byte.
	v41GateUpDeclined v41ExpertGateUpOutcome = iota
	// v41GateUpHandled: gate/up MatMuls ran on the backend, and the returned
	// slice is the I-wide intermediate after the configured SwiGLU activation.
	v41GateUpHandled
	// v41GateUpError: a selected device execution failed; it must remain visible.
	v41GateUpError
)

// v41ExpertGateUpFunc is the optional per-pick device gate/up operation. It
// receives the routed expert's layer stem and the normalized input, and reports
// one of the closed v41ExpertGateUpOutcome values. On v41GateUpHandled it returns
// the I-wide fused intermediate from device gate/up projections and the
// configured SwiGLU activation, sized to the expert intermediate width. Full
// V4.1 bindings publish BF16 gate/up operands before F32 host SwiGLU; the
// returned intermediate remains F32 until its caller applies the route weight.
type v41ExpertGateUpFunc func(layer int, stem string, xn []float32) ([]float32, v41ExpertGateUpOutcome, error)

// v41DeviceOnlyGateUp is the device-only gate/up wrapper (see
// v41BindDeviceOnlyDeclineRefusals): an unbound or declining seam refuses typed.
func v41DeviceOnlyGateUp(s *Session, f v41ExpertGateUpFunc) v41ExpertGateUpFunc {
	return func(layer int, stem string, xn []float32) ([]float32, v41ExpertGateUpOutcome, error) {
		if f != nil {
			y, outcome, err := f(layer, stem, xn)
			if outcome != v41GateUpDeclined {
				return y, outcome, err
			}
		}
		return nil, v41GateUpError, s.v41DeviceOnlyDeclined(layer, "routed-expert-gate-up")
	}
}

// v41ExpertDownOutcome is the closed result vocabulary of one v41ExpertDownFunc
// call. handled-success returns the H-wide expert output with the down projection
// computed on the backend; declined leaves the caller on the historical host
// f32 down contraction (hostExpertDown + matRows); handled-error is a SELECTED
// execution failure and must surface, never be swallowed as a decline.
type v41ExpertDownOutcome uint8

const (
	// v41DownDeclined: no device kernel resolved for this expert's down encoding,
	// so the caller falls through to the host down contraction byte-for-byte.
	v41DownDeclined v41ExpertDownOutcome = iota
	// v41DownHandled: the down MatMul ran on the backend and the returned slice is
	// the H-wide expert output.
	v41DownHandled
	// v41DownError: a selected device execution failed; it must remain visible.
	v41DownError
)

// v41ExpertDownFunc is the optional device down-projection operation, the
// symmetric partner of v41ExpertGateUpFunc (#13358/#13511 moved gate/up to the
// device; without this the down projection — the last expert GEMM — still ran on
// the host, so a streamed serve that fits only via host expert placement could
// never satisfy the GPU-only guard, #13668/#13128). It receives the routed
// expert's layer stem and the I-wide fused intermediate the gate/up seam
// produced, and reports one of the closed v41ExpertDownOutcome values. On
// v41DownHandled it returns the H-wide expert output computed on the backend.
// Full V4.1 supplies the already-weighted BF16 operand and owns a BF16 result
// before the device tensor is freed; reduced bindings retain their F32 contract.
type v41ExpertDownFunc func(layer int, stem string, fused []float32) ([]float32, v41ExpertDownOutcome, error)
