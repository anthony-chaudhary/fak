package compute

import (
	"fmt"
	"math"
)

// Qwen35GDNCUDAPath is the stable whole-operation identity shared with the
// model.Qwen35GDNBackend contract. It is duplicated at the cycle-free compute
// boundary deliberately: compute cannot import model, while a structural
// implementation must still return the exact production path.
const Qwen35GDNCUDAPath = "cuda/qwen35-gdn-ssm-decode-v1"

// Qwen35GDNParityCosineMin is the device/reference acceptance floor for the
// deterministic whole-operation fixture. The value records the gate; only a
// non-skipped real-device test run witnesses that the implementation clears it.
const Qwen35GDNParityCosineMin = 0.999

// Qwen35GDNGeometryError is a fail-closed refusal raised before any GDN kernel
// is launched. Operand is either a tensor name or "geometry" for a relation
// between scalar dimensions; Want describes the accepted shape/relation.
type Qwen35GDNGeometryError struct {
	Operand string
	Got     []int
	Want    string
	Reason  string
}

func (e *Qwen35GDNGeometryError) Error() string {
	if e == nil {
		return "compute: nil Qwen3.5 GDN geometry error"
	}
	if e.Reason != "" {
		return fmt.Sprintf("compute: cuda Qwen3.5 GDN geometry refused for %s: %s", e.Operand, e.Reason)
	}
	return fmt.Sprintf("compute: cuda Qwen3.5 GDN geometry refused for %s: shape %v, want %s", e.Operand, e.Got, e.Want)
}

// Qwen35GDNResidencyError refuses a tensor that is not an F32 row-major buffer
// resident on the CUDA backend executing the operation. In particular, the
// implementation never obtains a HostBuffer and never falls back to CPU math.
type Qwen35GDNResidencyError struct {
	Operand string
	Reason  string
}

func (e *Qwen35GDNResidencyError) Error() string {
	if e == nil {
		return "compute: nil Qwen3.5 GDN residency error"
	}
	return fmt.Sprintf("compute: cuda Qwen3.5 GDN residency refused for %s: %s", e.Operand, e.Reason)
}

// Qwen35GDNAllocationError is the typed refusal returned when the strict GDN
// path cannot obtain device-only scratch/output storage. Unlike the general
// CUDA allocator, this path never falls back to managed memory: UVM migration
// would make the whole-operation residency and transfer witness untrue.
type Qwen35GDNAllocationError struct {
	Operand string
	Bytes   int
}

func (e *Qwen35GDNAllocationError) Error() string {
	if e == nil {
		return "compute: nil Qwen3.5 GDN allocation error"
	}
	return fmt.Sprintf("compute: cuda Qwen3.5 GDN device-only allocation failed for %s (%d bytes); managed-memory fallback refused", e.Operand, e.Bytes)
}

// Qwen35GDNInvalidStateError marks an in-place state buffer whose contents can
// no longer be trusted after a CUDA launch or asynchronous execution failure.
// The caller may Free the tensor, but it cannot Read it or submit it to another
// operation as though the partial update were a valid next state.
type Qwen35GDNInvalidStateError struct {
	Operand string
}

func (e *Qwen35GDNInvalidStateError) Error() string {
	if e == nil {
		return "compute: nil Qwen3.5 GDN invalid-state error"
	}
	return fmt.Sprintf("compute: cuda Qwen3.5 GDN %s is invalid after a failed in-place operation; free it and reinitialize state", e.Operand)
}

// Qwen35GDNKernelError is the typed fail-closed return for an ABI or CUDA
// launch failure inside the whole operation. Stage names the fused stage; Code
// preserves the flat C ABI status for device-side diagnosis.
type Qwen35GDNKernelError struct {
	Stage string
	Code  int
}

func (e *Qwen35GDNKernelError) Error() string {
	if e == nil {
		return "compute: nil Qwen3.5 GDN kernel error"
	}
	return fmt.Sprintf("compute: cuda Qwen3.5 GDN kernel failed closed at %s (code %d); no CPU fallback", e.Stage, e.Code)
}

const qwen35GDNMaxCInt = int64(1<<31 - 1)

func qwen35GDNCheckedAdd(limit int64, values ...int64) (int64, bool) {
	var sum int64
	for _, value := range values {
		if value < 0 || sum > limit-value {
			return 0, false
		}
		sum += value
	}
	return sum, true
}

func qwen35GDNCheckedMul(limit int64, values ...int64) (int64, bool) {
	product := int64(1)
	for _, value := range values {
		if value < 0 || (value != 0 && product > limit/value) {
			return 0, false
		}
		product *= value
	}
	return product, true
}

func qwen35GDNShapeBytes(shape []int, bytesPerElement int) (int, bool) {
	maxInt := int64(^uint(0) >> 1)
	product := int64(1)
	for _, dimension := range shape {
		if dimension < 0 {
			return 0, false
		}
		var ok bool
		product, ok = qwen35GDNCheckedMul(maxInt, product, int64(dimension))
		if !ok {
			return 0, false
		}
	}
	bytes, ok := qwen35GDNCheckedMul(maxInt, product, int64(bytesPerElement))
	if !ok {
		return 0, false
	}
	return int(bytes), true
}

type qwen35GDNOperand struct {
	name string
	t    Tensor
}

func qwen35GDNOperands(
	normalizedInput,
	inProjQKV, inProjZ, inProjB, inProjA,
	conv1D, aLog, dtBias, norm, outProj,
	convState, recurrentState Tensor,
) []qwen35GDNOperand {
	return []qwen35GDNOperand{
		{"normalized_input", normalizedInput},
		{"in_proj_qkv", inProjQKV}, {"in_proj_z", inProjZ},
		{"in_proj_b", inProjB}, {"in_proj_a", inProjA},
		{"conv1d", conv1D}, {"A_log", aLog}, {"dt_bias", dtBias},
		{"norm", norm}, {"out_proj", outProj},
		{"conv_state", convState}, {"recurrent_state", recurrentState},
	}
}

type qwen35GDNAllocation struct {
	name  string
	shape []int
}

// qwen35GDNAllocations describes the nine scratch/output buffers shared by the
// decode and sequence kernels. A zero row count preserves the single-row decode
// layout; sequence callers independently select whether scratch and output are
// panel-shaped because both native entry points intentionally support that mix.
func qwen35GDNAllocations(namePrefix, nameSeparator string, scratchRows, outputRows, hidden, keyDim, valueDim, numValueHeads, convDim int) []qwen35GDNAllocation {
	shape := func(rows, width int) []int {
		if rows > 0 {
			return []int{rows, width}
		}
		return []int{width}
	}
	return []qwen35GDNAllocation{
		{namePrefix + "mixed", shape(scratchRows, convDim)},
		{namePrefix + "z", shape(scratchRows, valueDim)},
		{namePrefix + "b", shape(scratchRows, numValueHeads)},
		{namePrefix + "a", shape(scratchRows, numValueHeads)},
		{namePrefix + "conv" + nameSeparator + "out", shape(scratchRows, convDim)},
		{namePrefix + "q" + nameSeparator + "norm", shape(scratchRows, keyDim)},
		{namePrefix + "k" + nameSeparator + "norm", shape(scratchRows, keyDim)},
		{namePrefix + "core", shape(scratchRows, valueDim)},
		{namePrefix + "output", shape(outputRows, hidden)},
	}
}

// qwen35GDNInputs bundles the operand tensors every Qwen3.5 GDN entry point
// threads through geometry validation and operand residency checks. Each
// backend binds its method parameters once and reuses the derived helpers
// instead of re-spelling the operand pack at every seam.
type qwen35GDNInputs struct {
	normalizedInput, inProjQKV, inProjZ, inProjB, inProjA,
	conv1D, aLog, dtBias, norm, outProj,
	convState, recurrentState Tensor
}

// entry validates the shared entry-point geometry and builds the operand list
// the backend residency checks walk. input is the tensor whose leading shape is
// hidden: the decode paths pass normalizedInput itself, while the sequence path
// validates a shape-flattened view of the same tensor (its operands keep the
// panel tensor). On failure every entry point propagates the zero output
// triple.
func (in qwen35GDNInputs) entry(
	input Tensor,
	numKeyHeads, numValueHeads, keyHeadDim, valueHeadDim, convKernel int,
	rmsNormEpsilon float32,
) (hidden, keyDim, valueDim, convDim int, operands []qwen35GDNOperand, err error) {
	hidden, keyDim, valueDim, convDim, err = validateQwen35GDNGeometry(
		input,
		in.inProjQKV, in.inProjZ, in.inProjB, in.inProjA,
		in.conv1D, in.aLog, in.dtBias, in.norm, in.outProj,
		in.convState, in.recurrentState,
		numKeyHeads, numValueHeads, keyHeadDim, valueHeadDim, convKernel,
		rmsNormEpsilon,
	)
	if err != nil {
		return 0, 0, 0, 0, nil, err
	}
	operands = qwen35GDNOperands(
		in.normalizedInput,
		in.inProjQKV, in.inProjZ, in.inProjB, in.inProjA,
		in.conv1D, in.aLog, in.dtBias, in.norm, in.outProj,
		in.convState, in.recurrentState,
	)
	return hidden, keyDim, valueDim, convDim, operands, nil
}

func validateQwen35GDNGeometry(
	normalizedInput,
	inProjQKV, inProjZ, inProjB, inProjA,
	conv1D, aLog, dtBias, norm, outProj,
	convState, recurrentState Tensor,
	numKeyHeads, numValueHeads, keyHeadDim, valueHeadDim, convKernel int,
	rmsNormEpsilon float32,
) (hidden, keyDim, valueDim, convDim int, err error) {
	if len(normalizedInput.Shape) != 1 || normalizedInput.Shape[0] <= 0 {
		return 0, 0, 0, 0, qwen35GDNShapeError("normalized_input", normalizedInput.Shape, "[hidden], hidden > 0")
	}
	hidden = normalizedInput.Shape[0]
	if numKeyHeads <= 0 || numValueHeads <= 0 || keyHeadDim <= 0 || valueHeadDim <= 0 || convKernel < 1 {
		return 0, 0, 0, 0, &Qwen35GDNGeometryError{Operand: "geometry", Reason: "head counts/dimensions must be positive and conv_kernel must be >= 1"}
	}
	if numValueHeads%numKeyHeads != 0 {
		return 0, 0, 0, 0, &Qwen35GDNGeometryError{Operand: "geometry", Reason: "num_value_heads must be divisible by num_key_heads"}
	}
	if keyHeadDim > 1024 || valueHeadDim > 1024 {
		return 0, 0, 0, 0, &Qwen35GDNGeometryError{Operand: "geometry", Reason: "key/value head dimensions must fit one CUDA block (<= 1024)"}
	}
	if !(rmsNormEpsilon > 0) || math.IsNaN(float64(rmsNormEpsilon)) || math.IsInf(float64(rmsNormEpsilon), 0) {
		return 0, 0, 0, 0, &Qwen35GDNGeometryError{Operand: "rms_norm_epsilon", Reason: "epsilon must be finite and > 0"}
	}
	for _, dimension := range []int{hidden, numKeyHeads, numValueHeads, keyHeadDim, valueHeadDim, convKernel} {
		if int64(dimension) > qwen35GDNMaxCInt {
			return 0, 0, 0, 0, &Qwen35GDNGeometryError{Operand: "geometry", Reason: "a scalar dimension overflows the CUDA int ABI"}
		}
	}

	key64, ok := qwen35GDNCheckedMul(qwen35GDNMaxCInt, int64(numKeyHeads), int64(keyHeadDim))
	if !ok {
		return 0, 0, 0, 0, &Qwen35GDNGeometryError{Operand: "geometry", Reason: "key dimension overflows the CUDA int ABI"}
	}
	value64, ok := qwen35GDNCheckedMul(qwen35GDNMaxCInt, int64(numValueHeads), int64(valueHeadDim))
	if !ok {
		return 0, 0, 0, 0, &Qwen35GDNGeometryError{Operand: "geometry", Reason: "value dimension overflows the CUDA int ABI"}
	}
	twiceKey, ok := qwen35GDNCheckedMul(qwen35GDNMaxCInt, 2, key64)
	if !ok {
		return 0, 0, 0, 0, &Qwen35GDNGeometryError{Operand: "geometry", Reason: "convolution dimension overflows the CUDA int ABI"}
	}
	conv64, ok := qwen35GDNCheckedAdd(qwen35GDNMaxCInt, twiceKey, value64)
	if !ok {
		return 0, 0, 0, 0, &Qwen35GDNGeometryError{Operand: "geometry", Reason: "convolution dimension overflows the CUDA int ABI"}
	}
	twiceValueHeads, ok := qwen35GDNCheckedMul(qwen35GDNMaxCInt, 2, int64(numValueHeads))
	if !ok {
		return 0, 0, 0, 0, &Qwen35GDNGeometryError{Operand: "geometry", Reason: "fused projection grid overflows the CUDA int ABI"}
	}
	if _, ok := qwen35GDNCheckedAdd(qwen35GDNMaxCInt, conv64, value64, twiceValueHeads); !ok {
		return 0, 0, 0, 0, &Qwen35GDNGeometryError{Operand: "geometry", Reason: "fused projection grid overflows the CUDA int ABI"}
	}
	// The convolution launch uses ceil(convDim/256). Keeping the addition in
	// checked 64-bit arithmetic prevents the former signed-int convDim+255 wrap.
	convGridNumerator, ok := qwen35GDNCheckedAdd(qwen35GDNMaxCInt+255, conv64, 255)
	if !ok || convGridNumerator/256 > qwen35GDNMaxCInt {
		return 0, 0, 0, 0, &Qwen35GDNGeometryError{Operand: "geometry", Reason: "convolution grid overflows CUDA grid capacity"}
	}

	elementProducts := []struct {
		name   string
		values []int64
	}{
		{"in_proj_qkv", []int64{conv64, int64(hidden)}},
		{"in_proj_z", []int64{value64, int64(hidden)}},
		{"in_proj_b", []int64{int64(numValueHeads), int64(hidden)}},
		{"in_proj_a", []int64{int64(numValueHeads), int64(hidden)}},
		{"conv1d", []int64{conv64, int64(convKernel)}},
		{"conv_state", []int64{int64(convKernel - 1), conv64}},
		{"recurrent_state", []int64{int64(numValueHeads), int64(keyHeadDim), int64(valueHeadDim)}},
		{"out_proj", []int64{int64(hidden), value64}},
	}
	for _, product := range elementProducts {
		if _, ok := qwen35GDNCheckedMul(qwen35GDNMaxCInt, product.values...); !ok {
			return 0, 0, 0, 0, &Qwen35GDNGeometryError{Operand: product.name, Reason: "element count overflows CUDA indexing capacity"}
		}
	}

	keyDim, valueDim, convDim = int(key64), int(value64), int(conv64)
	require := func(name string, tensor Tensor, shapes ...[]int) error {
		for _, shape := range shapes {
			if qwen35GDNSameShape(tensor.Shape, shape) {
				if _, ok := qwen35GDNShapeBytes(tensor.Shape, F32.Bytes()); !ok {
					return &Qwen35GDNGeometryError{Operand: name, Reason: "shape byte size overflows host allocation capacity"}
				}
				return nil
			}
		}
		want := ""
		for i, shape := range shapes {
			if i > 0 {
				want += " or "
			}
			want += fmt.Sprint(shape)
		}
		return qwen35GDNShapeError(name, tensor.Shape, want)
	}
	checks := []error{
		require("normalized_input", normalizedInput, []int{hidden}),
		require("in_proj_qkv", inProjQKV, []int{convDim, hidden}),
		require("in_proj_z", inProjZ, []int{valueDim, hidden}),
		require("in_proj_b", inProjB, []int{numValueHeads, hidden}),
		require("in_proj_a", inProjA, []int{numValueHeads, hidden}),
		require("conv1d", conv1D, []int{convDim, convKernel}, []int{convDim, 1, convKernel}, []int{convDim * convKernel}),
		require("A_log", aLog, []int{numValueHeads}),
		require("dt_bias", dtBias, []int{numValueHeads}),
		require("norm", norm, []int{valueHeadDim}),
		require("out_proj", outProj, []int{hidden, valueDim}),
		require("conv_state", convState, []int{convKernel - 1, convDim}),
		require("recurrent_state", recurrentState, []int{numValueHeads, keyHeadDim, valueHeadDim}),
	}
	for _, check := range checks {
		if check != nil {
			return 0, 0, 0, 0, check
		}
	}
	return hidden, keyDim, valueDim, convDim, nil
}

func qwen35GDNSameShape(got, want []int) bool {
	if len(got) != len(want) {
		return false
	}
	for i := range got {
		if got[i] != want[i] {
			return false
		}
	}
	return true
}

func qwen35GDNShapeError(name string, got []int, want string) error {
	return &Qwen35GDNGeometryError{Operand: name, Got: append([]int(nil), got...), Want: want}
}
