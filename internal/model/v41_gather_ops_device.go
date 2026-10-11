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
	owner                     *Session
	inverseOutput             v41InverseOutputRoPEFunc
	indexSelect               v41IndexSelectFunc
	indexRoPE                 v41IndexRoPEFunc
	embedGather               v41EmbeddingGatherFunc
	compressedPublicationRoPE v41CompressedPublicationRoPEFunc
	engramGate                v41EngramGateFunc
	compressorPool            v41CompressorPoolFunc
	bf16Publish               v41BF16BoundaryFunc
}

// v41DeviceGatherOpsFunc binds every available slice-3 operation. The policy is
// read at call time, because a session may enter device-only after binding.
func (s *Session) v41DeviceGatherOpsFunc() *v41DeviceGatherOps {
	if s == nil {
		return nil
	}
	ops := &v41DeviceGatherOps{owner: s, inverseOutput: s.v41InverseOutputRoPEFunc(), indexSelect: s.v41IndexSelectFunc(), embedGather: s.v41EmbeddingGatherFunc(), indexRoPE: s.v41IndexRoPEFunc(), compressedPublicationRoPE: s.v41CompressedPublicationRoPEFunc(), engramGate: s.v41EngramGateFunc(), compressorPool: s.v41CompressorPoolFunc()}
	ops.bf16Publish = s.v41BF16PublishFunc()
	return ops
}

// bf16PublishFunc returns the ordinary BF16 publication; under device-only an
// unbound seam refuses typed instead of rounding on the host.
func (o *v41DeviceGatherOps) bf16PublishFunc() v41BF16BoundaryFunc {
	if o == nil {
		return nil
	}
	if o.bf16Publish == nil && o.owner.v41DeviceOnlyPolicy() {
		return func(layer int, _ string, _ []float32) ([]float32, error) {
			return nil, o.owner.v41DeviceOnlyMissingOp(layer, "bf16-boundary")
		}
	}
	return o.bf16Publish
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

func (o *v41DeviceGatherOps) indexRoPEFunc() v41IndexRoPEFunc {
	if o == nil {
		return nil
	}
	if o.indexRoPE == nil && o.owner.v41DeviceOnlyPolicy() {
		return func(layer int, _, _, _ []float32, _, _, _ int) ([]float32, error) {
			return nil, o.owner.v41DeviceOnlyMissingOp(layer, "indexer-query-rope")
		}
	}
	return o.indexRoPE
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

// compressedPublicationRoPEFunc keeps the ordinary host arm while the strict
// arithmetic profile is unavailable. Under device-only an unbound seam refuses
// typed instead of rotating on the host; ordinary Vulkan binds it only when the
// strict tail-RoPE twin is admitted for the selected device.
func (o *v41DeviceGatherOps) compressedPublicationRoPEFunc() v41CompressedPublicationRoPEFunc {
	if o == nil {
		return nil
	}
	if o.compressedPublicationRoPE == nil && o.owner.v41DeviceOnlyPolicy() {
		return func(layer int, _, _, _ []float32, _, _ int) ([]float32, error) {
			return nil, o.owner.v41DeviceOnlyMissingOp(layer, "compressed-publication-rope")
		}
	}
	return o.compressedPublicationRoPE
}

// engramGateFunc returns the resident Engram gate/add seam. Under device-only an
// unbound or declining seam refuses typed instead of handing the panel back to
// the host injection.
func (o *v41DeviceGatherOps) engramGateFunc() v41EngramGateFunc {
	if o == nil {
		return nil
	}
	if !o.owner.v41DeviceOnlyPolicy() {
		return o.engramGate
	}
	gate := o.engramGate
	return func(layer int, streams *v41DeviceStreams, rows []float32, positions int, geometry v41EngramGeometry, eps float32) (bool, error) {
		if gate == nil {
			return false, o.owner.v41DeviceOnlyMissingOp(layer, "engram-stream-injection")
		}
		handled, err := gate(layer, streams, rows, positions, geometry, eps)
		if err == nil && !handled {
			return false, o.owner.v41DeviceOnlyDeclined(layer, "engram-stream-injection")
		}
		return handled, err
	}
}

// compressorPoolFunc returns the gated-pool seam; under device-only an unbound
// seam refuses typed instead of pooling on the host.
func (o *v41DeviceGatherOps) compressorPoolFunc() v41CompressorPoolFunc {
	if o == nil {
		return nil
	}
	if o.compressorPool == nil && o.owner.v41DeviceOnlyPolicy() {
		return func(layer int, _, _ []float32, _, _ int) ([]float32, error) {
			return nil, o.owner.v41DeviceOnlyMissingOp(layer, "compressor-gated-pool")
		}
	}
	return o.compressorPool
}
