package model

import "fmt"

// v41DeviceGatherOps carries the session's slice-3 device bindings (Ref #13668):
// the inverse output rotation, the attention index selection and the
// embedding gather. One pointer keeps the forward plumbing to a single field per
// carrier. It is a per-session binding, never continuation data, so snapshots
// omit it and a restored session rebinds through Session.v41State.
//
// A nil operation keeps its historical host arm, except under
// ExecutionPolicyDeviceOnly: there the accessor returns a refusal that closes
// the session with a typed error naming the op, so no host compute runs.
type v41DeviceGatherOps struct {
	owner         *Session
	inverseOutput v41InverseOutputRoPEFunc
	indexSelect   v41IndexSelectFunc
	embedGather   v41EmbeddingGatherFunc
}

// v41DeviceGatherOpsFunc binds every available slice-3 operation. The policy is
// read at call time, because a session may enter device-only after binding.
func (s *Session) v41DeviceGatherOpsFunc() *v41DeviceGatherOps {
	if s == nil {
		return nil
	}
	return &v41DeviceGatherOps{owner: s, inverseOutput: s.v41InverseOutputRoPEFunc(), indexSelect: s.v41IndexSelectFunc(), embedGather: s.v41EmbeddingGatherFunc()}
}

func (o *v41DeviceGatherOps) inverseOutputFunc() v41InverseOutputRoPEFunc {
	if o == nil {
		return nil
	}
	if o.inverseOutput == nil && o.owner.v41DeviceOnlyPolicy() {
		return func(layer int, _, _, _ []float32, _, _, _ int) ([]float32, error) {
			return nil, o.owner.v41DeviceOnlyMissingOp(layer, "attention-output-rotation")
		}
	}
	return o.inverseOutput
}

func (o *v41DeviceGatherOps) indexSelectFunc() v41IndexSelectFunc {
	if o == nil {
		return nil
	}
	if o.indexSelect == nil && o.owner.v41DeviceOnlyPolicy() {
		return func(layer int, _ []float32, _, _, _ int) ([]int32, error) {
			return nil, o.owner.v41DeviceOnlyMissingOp(layer, "attention-index-list")
		}
	}
	return o.indexSelect
}

func (o *v41DeviceGatherOps) embeddingGatherFunc() v41EmbeddingGatherFunc {
	if o == nil {
		return nil
	}
	if o.embedGather == nil && o.owner.v41DeviceOnlyPolicy() {
		return func([]int) ([]float32, error) {
			return nil, o.owner.v41DeviceOnlyMissingOp(-1, "embedding-panel")
		}
	}
	return o.embedGather
}

func (s *Session) v41DeviceOnlyPolicy() bool {
	return s != nil && s.executionPolicy == ExecutionPolicyDeviceOnly
}

// v41DeviceOnlyMissingOp closes a device-only session whose backend lacks the
// device operation for op. The cause wraps errV41DeviceOnlyHostOp.
func (s *Session) v41DeviceOnlyMissingOp(layer int, op string) error {
	backend := ""
	if s.Backend != nil {
		backend = s.Backend.Name()
	}
	closed := &BackendForwardOperationError{
		Backend: backend, Forward: ForwardPathKind("deepseek41"), Path: "device-only", Layer: layer,
		Stage: op + ": backend lacks the device operation",
		Cause: fmt.Errorf("%w: %s", errV41DeviceOnlyHostOp, op),
	}
	s.halFailure = closed
	s.Close()
	return closed
}
