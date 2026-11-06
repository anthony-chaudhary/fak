package compute

import (
	"errors"
	"fmt"
	"reflect"
)

// Qwen35GDNPackedInputContractV1 describes canonical row-major block-256
// Q4_K/Q6_K QKV/Z weights consumed by BOTH existing CUDA whole-GDN decode and
// sequence paths, with F32 mutable state. It promises neither capture nor a
// measured numerical/performance result. Other projection roles are excluded.
const Qwen35GDNPackedInputContractV1 = "cuda/qwen35-gdn-canonical-qkv-z-v1"

var ErrQwen35GDNPackedPolicy = errors.New("compute: canonical packed GDN input policy refused")

// Qwen35GDNPackedInputCapability is a source capability declaration, not
// activation permission. Each dtype still needs the actual backend upload and
// resident-device dtype probes. Unknown contracts fail closed.
type Qwen35GDNPackedInputCapability struct {
	Contract string
	Q4K, Q6K bool
}

type Qwen35GDNPackedInputBackend interface {
	Qwen35GDNPackedInputCapability() Qwen35GDNPackedInputCapability
}

// This cycle-free structural boundary deliberately requires actual operation
// methods, not path-name markers or generic MatMul alone. Its signatures match
// the model's Qwen35GDNBackend and Qwen35SequencePrefillBackend contracts.
type qwen35GDNPackedImplementation interface {
	Qwen35GDNPath() string
	Qwen35GDNDecode(
		normalizedInput, inProjQKV, inProjZ, inProjB, inProjA,
		conv1D, aLog, dtBias, norm, outProj, convState, recurrentState Tensor,
		numKeyHeads, numValueHeads, keyHeadDim, valueHeadDim, convKernel int,
		rmsNormEpsilon float32,
	) (output, nextConvState, nextRecurrentState Tensor, err error)
	Qwen35SequencePrefillPath() string
	Qwen35SequencePrefill(Qwen35SequencePrefillRequest) (Qwen35SequencePrefillResult, error)
}

// Qwen35GDNPackedInputRequest is resolved once per load. Frontends must parse
// their existing controls explicitly before calling this function. No environment
// is read here. Both force-off controls dominate Enable regardless of option order.
type Qwen35GDNPackedInputRequest struct {
	Enable        bool
	NativeRowsOff bool
	GDNRowsOff    bool
}

// Qwen35GDNPackedInputPolicy is a copyable immutable decision. Its zero value
// selects no packed GDN inputs; callers cannot construct an enabled policy by
// setting exported fields. It is not a device-health or tensor-residency receipt.
type Qwen35GDNPackedInputPolicy struct {
	q4k, q6k bool
}

func (p Qwen35GDNPackedInputPolicy) Enabled() bool { return p.q4k || p.q6k }
func (p Qwen35GDNPackedInputPolicy) Allows(dt Dtype) bool {
	return (dt == Q4_K && p.q4k) || (dt == Q6_K && p.q6k)
}

// CaptureLegal stays false even on CUDA backends whose F32/Q8 GDN capture is legal.
func (p Qwen35GDNPackedInputPolicy) CaptureLegal() bool { return false }

// ResolveQwen35GDNPackedInputPolicy makes no upload, allocation or execution call.
// An explicit unsupported request returns a typed refusal instead of manufacturing
// permission from a backend name, IQ MatMul or tiled-output capability.
func ResolveQwen35GDNPackedInputPolicy(be Backend, request Qwen35GDNPackedInputRequest) (Qwen35GDNPackedInputPolicy, error) {
	empty := Qwen35GDNPackedInputPolicy{}
	if !request.Enable || request.NativeRowsOff || request.GDNRowsOff {
		return empty, nil
	}
	if be == nil {
		return empty, fmt.Errorf("%w: missing backend", ErrQwen35GDNPackedPolicy)
	}
	value := reflect.ValueOf(be)
	switch value.Kind() {
	case reflect.Chan, reflect.Func, reflect.Interface, reflect.Map, reflect.Pointer, reflect.Slice:
		if value.IsNil() {
			return empty, fmt.Errorf("%w: nil backend value", ErrQwen35GDNPackedPolicy)
		}
	}
	implementation, ok := be.(qwen35GDNPackedImplementation)
	if !ok {
		return empty, fmt.Errorf("%w: whole decode and sequence operations required", ErrQwen35GDNPackedPolicy)
	}
	if implementation.Qwen35GDNPath() != Qwen35GDNCUDAPath ||
		implementation.Qwen35SequencePrefillPath() != Qwen35SequencePrefillPath {
		return empty, fmt.Errorf("%w: incompatible whole-operation identity", ErrQwen35GDNPackedPolicy)
	}
	capability, ok := be.(Qwen35GDNPackedInputBackend)
	if !ok {
		return empty, fmt.Errorf("%w: missing canonical-input capability", ErrQwen35GDNPackedPolicy)
	}
	declared := capability.Qwen35GDNPackedInputCapability()
	if declared.Contract != Qwen35GDNPackedInputContractV1 {
		return empty, fmt.Errorf("%w: unknown canonical-input contract", ErrQwen35GDNPackedPolicy)
	}
	caps := be.Caps()
	if !caps.DeviceMemory || !caps.UploadDtype {
		return empty, fmt.Errorf("%w: device residency and quantized upload required", ErrQwen35GDNPackedPolicy)
	}
	policy := Qwen35GDNPackedInputPolicy{
		q4k: declared.Q4K && BackendSupportsDeviceWeightDtype(be, Q4_K),
		q6k: declared.Q6K && BackendSupportsDeviceWeightDtype(be, Q6_K),
	}
	// Preserve a more restrictive backend-specific Q6 implementation probe.
	if native, ok := be.(interface{ SupportsQ6KMatMul() bool }); ok && policy.q6k {
		policy.q6k = native.SupportsQ6KMatMul()
	}
	if !policy.Enabled() {
		return empty, fmt.Errorf("%w: no admitted packed dtype", ErrQwen35GDNPackedPolicy)
	}
	return policy, nil
}

type Qwen35GDNPackedInputRole uint8

const (
	Qwen35GDNPackedQKV Qwen35GDNPackedInputRole = iota + 1
	Qwen35GDNPackedZ
)

// Qwen35GDNPackedInputGeometry is model metadata, never inferred from a filename.
// The loader must first establish a target-layer QKV/Z role through its existing
// model predicate, excluding MTP, A/B, attention Q/K and output projection.
type Qwen35GDNPackedInputGeometry struct {
	Hidden, KeyHeads, ValueHeads, KeyHeadDim, ValueHeadDim int
}

type Qwen35GDNPackedInputPlan struct {
	Role                        Qwen35GDNPackedInputRole
	Dtype                       Dtype
	Rows, Columns, Block, Bytes int
}

// Plan validates representation metadata shared by loading and estimation. It
// neither normalizes raw rows nor validates a whole model/operation. Callers must
// keep the existing row permutation and device-canonical builder. At execution,
// larger token panels, allocator/device identity, state, selectors and every
// operand still require the existing whole-operation preflight before effects.
func (p Qwen35GDNPackedInputPolicy) Plan(role Qwen35GDNPackedInputRole, dt Dtype, shape []int, geometry Qwen35GDNPackedInputGeometry) (Qwen35GDNPackedInputPlan, error) {
	empty := Qwen35GDNPackedInputPlan{}
	refuse := func(reason string) (Qwen35GDNPackedInputPlan, error) {
		return empty, fmt.Errorf("%w: %s", ErrQwen35GDNPackedPolicy, reason)
	}
	if !p.Allows(dt) {
		return refuse("dtype is not admitted")
	}
	if role != Qwen35GDNPackedQKV && role != Qwen35GDNPackedZ {
		return refuse("only canonical QKV/Z roles are admitted")
	}
	g := geometry
	if g.Hidden <= 0 || g.Hidden%256 != 0 || g.KeyHeads <= 0 || g.ValueHeads <= 0 ||
		g.ValueHeads%g.KeyHeads != 0 || g.KeyHeadDim <= 0 || g.KeyHeadDim > 1024 ||
		g.ValueHeadDim <= 0 || g.ValueHeadDim > 1024 {
		return refuse("invalid hidden or GDN head geometry")
	}
	key, ok := qwen35GDNCheckedMul(qwen35GDNMaxCInt, int64(g.KeyHeads), int64(g.KeyHeadDim))
	if !ok {
		return refuse("key width exceeds CUDA indexing")
	}
	value, ok := qwen35GDNCheckedMul(qwen35GDNMaxCInt, int64(g.ValueHeads), int64(g.ValueHeadDim))
	if !ok {
		return refuse("value width exceeds CUDA indexing")
	}
	twiceKey, ok := qwen35GDNCheckedMul(qwen35GDNMaxCInt, 2, key)
	if !ok {
		return refuse("QKV width exceeds CUDA indexing")
	}
	qkv, ok := qwen35GDNCheckedAdd(qwen35GDNMaxCInt, twiceKey, value)
	if !ok {
		return refuse("QKV width exceeds CUDA indexing")
	}
	rows := value
	if role == Qwen35GDNPackedQKV {
		rows = qkv
	}
	if len(shape) != 2 || int64(shape[0]) != rows || shape[1] != g.Hidden {
		return refuse("shape disagrees with resolved model geometry")
	}
	elements, ok := qwen35GDNCheckedMul(qwen35GDNMaxCInt, rows, int64(g.Hidden))
	if !ok {
		return refuse("weight elements exceed CUDA indexing")
	}
	// The packed input admission also supports independent decode panels of
	// eight lanes. Matching that minimum bound avoids a load-only false positive.
	for _, width := range []int64{rows, int64(g.Hidden)} {
		if _, ok := qwen35GDNCheckedMul(qwen35GDNMaxCInt, 8, width); !ok {
			return refuse("minimum eight-lane panel exceeds CUDA indexing")
		}
	}
	blockBytes := int64(144)
	if dt == Q6_K {
		blockBytes = 210
	}
	rawBytes, ok := qwen35GDNCheckedMul(int64(^uint(0)>>1), elements/256, blockBytes)
	if !ok {
		return refuse("raw bytes exceed host capacity")
	}
	return Qwen35GDNPackedInputPlan{Role: role, Dtype: dt, Rows: int(rows), Columns: g.Hidden, Block: 256, Bytes: int(rawBytes)}, nil
}

// ValidatePayload checks exact raw extent without reading or allocating payload.
// Estimation uses Bytes directly; materialization must additionally call this.
func (p Qwen35GDNPackedInputPlan) ValidatePayload(bytes int) error {
	if p.Bytes <= 0 || bytes != p.Bytes {
		return fmt.Errorf("%w: raw payload length %d, want %d", ErrQwen35GDNPackedPolicy, bytes, p.Bytes)
	}
	return nil
}
