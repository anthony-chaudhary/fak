//go:build cuda && cgo

package compute

import "fmt"

// qwen35GDNStrictAllocFailAfter is a one-shot package-local test seam. A
// non-negative value permits that many strict allocations, then makes the next
// one return the same typed allocation refusal as a real cudaMalloc miss.
// Access is serialized by cudaMu; production leaves it disarmed at -1.
var qwen35GDNStrictAllocFailAfter = -1

func qwen35GDNInjectAllocationFailureForTest(successfulAllocations int) {
	cudaMu.Lock()
	defer cudaMu.Unlock()
	qwen35GDNStrictAllocFailAfter = successfulAllocations
}

// qwen35GDNInjectedAllocationFailure is called only from the strict allocator
// while cudaMu is held. The hook is consumed on refusal so one failed attempt
// cannot contaminate a later operation.
func qwen35GDNInjectedAllocationFailure(site string, nbytes int) error {
	if qwen35GDNStrictAllocFailAfter < 0 {
		return nil
	}
	if qwen35GDNStrictAllocFailAfter == 0 {
		qwen35GDNStrictAllocFailAfter = -1
		return &Qwen35GDNAllocationError{Operand: site, Bytes: nbytes}
	}
	qwen35GDNStrictAllocFailAfter--
	return nil
}

// Packed input kinds reuse the existing native matmul families. Only QKV/Z
// are canonical-row inputs; this does not advertise tiled-output capability.
func qwen35GDNPackedInput(t Tensor) bool { return t.Dtype == Q4_K || t.Dtype == Q6_K }

func qwen35GDNPackedInputBytes(t Tensor) (int, bool) {
	if !qwen35GDNPackedInput(t) || len(t.Shape) != 2 || t.Shape[0] <= 0 || t.Shape[1] <= 0 ||
		t.Shape[1]%256 != 0 || t.Quant == nil || t.Quant.Block != 256 {
		return 0, false
	}
	n, ok := qwen35GDNCheckedMul(qwen35GDNMaxCInt, int64(t.Shape[0]), int64(t.Shape[1]))
	if !ok {
		return 0, false
	}
	blockBytes := 144
	if t.Dtype == Q6_K {
		blockBytes = 210
	}
	return int(n/256) * blockBytes, true
}

func (c *cudaBackend) validateQwen35GDNPackedInput(name string, t Tensor, tokens int) error {
	if !qwen35GDNPackedInput(t) {
		return nil
	}
	if name != "in_proj_qkv" && name != "in_proj_z" && name != "gdn_in_qkv" && name != "gdn_in_z" {
		return &Qwen35GDNResidencyError{Operand: name, Reason: "packed GDN weights are supported only for canonical-row QKV/Z inputs"}
	}
	if c.capturing {
		return &Qwen35GDNResidencyError{Operand: name, Reason: "packed GDN inputs are unsupported during capture; existing packed matmul scratch may grow"}
	}
	if _, ok := qwen35GDNPackedInputBytes(t); !ok {
		return &Qwen35GDNGeometryError{Operand: name, Reason: "packed GDN inputs require positive block-256 rows and a CUDA-int-safe element count"}
	}
	// The same admission is used by independent decode batches of up to eight
	// lanes. Bound that panel too, even when each lane is preflighted separately.
	rows := max(tokens, 8)
	if tokens <= 0 || tokens > 65535 {
		return &Qwen35GDNGeometryError{Operand: name, Reason: "packed GDN token panel must fit CUDA grid.y (1..65535)"}
	}
	for _, width := range t.Shape {
		if _, ok := qwen35GDNCheckedMul(qwen35GDNMaxCInt, int64(rows), int64(width)); !ok {
			return &Qwen35GDNGeometryError{Operand: name, Reason: "packed GDN panel exceeds CUDA indexing capacity"}
		}
	}
	if t.Dtype == Q4_K && tokens >= 128 {
		if _, err := CUDAQ4KPrefillArmInUse(); err != nil {
			return &Qwen35GDNResidencyError{Operand: name, Reason: "configured Q4_K prefill arm is unavailable", Err: err}
		}
	}
	return nil
}

func (c *cudaBackend) validateQwen35GDNTensor(name string, t Tensor) error {
	if t.Backend() != c {
		return &Qwen35GDNResidencyError{Operand: name, Reason: "tensor is not owned by this CUDA backend"}
	}
	matrix := name == "in_proj_qkv" || name == "in_proj_z" || name == "in_proj_b" || name == "in_proj_a" || name == "out_proj"
	isState := name == "conv_state" || name == "recurrent_state"
	if isState {
		// Both standalone GDN ABIs consume state through float pointers.
		if t.Dtype != F32 {
			return &Qwen35GDNResidencyError{Operand: name, Reason: "dtype " + t.Dtype.String() + " is unsupported; state buffers require resident f32"}
		}
	} else if t.Dtype != F32 && !(matrix && t.Dtype == Q8_0) && !qwen35GDNPackedInput(t) {
		return &Qwen35GDNResidencyError{Operand: name, Reason: "dtype " + t.Dtype.String() + " is unsupported; whole-operation kernel requires resident f32 or q8_0 projection weights"}
	}
	if err := c.validateQwen35GDNPackedInput(name, t, 1); err != nil {
		return err
	}
	if t.Layout != RowMajor {
		return &Qwen35GDNResidencyError{Operand: name, Reason: "layout is not row-major"}
	}
	buf, ok := t.buf.(*cudaBuf)
	if !ok || buf == nil || buf.ptr == nil {
		return &Qwen35GDNResidencyError{Operand: name, Reason: "tensor has no live CUDA allocation (Upload is required before the operation)"}
	}
	if err := buf.invalidStateError(name); err != nil {
		return err
	}
	if buf.device != 0 {
		return &Qwen35GDNResidencyError{Operand: name, Reason: "tensor is resident on a non-default CUDA device"}
	}
	if buf.managed {
		return &Qwen35GDNResidencyError{Operand: name, Reason: "managed memory is forbidden; strict whole-operation operands must be device-only"}
	}
	required, ok := qwen35GDNShapeBytes(t.Shape, t.Dtype.Bytes())
	if qwen35GDNPackedInput(t) {
		required, ok = qwen35GDNPackedInputBytes(t)
	}
	if !ok {
		return &Qwen35GDNGeometryError{Operand: name, Reason: "shape byte size overflows host allocation capacity"}
	}
	if buf.n < required {
		return &Qwen35GDNResidencyError{Operand: name, Reason: fmt.Sprintf("CUDA allocation capacity is %d bytes, shape requires %d", buf.n, required)}
	}
	if t.Dtype == Q8_0 && (t.Quant == nil || t.Quant.Block != 32 || buf.scales == nil) {
		return &Qwen35GDNResidencyError{Operand: name, Reason: "q8_0 projection is missing resident block-32 scales"}
	}
	return nil
}

func (c *cudaBackend) validateQwen35GDNOperands(operands []qwen35GDNOperand, convState, recurrentState Tensor) error {
	tokens := 1
	if len(operands) > 0 && len(operands[0].t.Shape) == 2 {
		tokens = operands[0].t.Shape[0]
	}
	for _, operand := range operands {
		if err := c.validateQwen35GDNTensor(operand.name, operand.t); err != nil {
			return err
		}
		if err := c.validateQwen35GDNPackedInput(operand.name, operand.t, tokens); err != nil {
			return err
		}
		if err := c.validateQ56KMMVQWeight(operand.t, tokens); err != nil {
			return &Qwen35GDNResidencyError{Operand: operand.name, Reason: "Q56 qualification admission", Err: err}
		}
	}
	return c.validateQwen35GDNStateOperands(operands, convState, recurrentState)
}

func (c *cudaBackend) validateQwen35GDNStateOperands(operands []qwen35GDNOperand, convState, recurrentState Tensor) error {
	convBuf := convState.buf.(*cudaBuf)
	recurrentBuf := recurrentState.buf.(*cudaBuf)
	states := []struct {
		name   string
		buffer *cudaBuf
	}{
		{"conv_state", convBuf},
		{"recurrent_state", recurrentBuf},
	}
	for _, state := range states {
		if state.buffer.class != MemoryKVCache {
			return &Qwen35GDNResidencyError{
				Operand: state.name,
				Reason:  fmt.Sprintf("mutable state allocation class is %q, want %q for durable in-place state", state.buffer.class, MemoryKVCache),
			}
		}
	}
	if convBuf.ptr == recurrentBuf.ptr {
		return &Qwen35GDNResidencyError{Operand: "state", Reason: "conv_state and recurrent_state alias the same CUDA allocation"}
	}
	for _, operand := range operands[:len(operands)-2] {
		readOnly := operand.t.buf.(*cudaBuf)
		if readOnly.ptr == convBuf.ptr || readOnly.ptr == recurrentBuf.ptr {
			return &Qwen35GDNResidencyError{Operand: operand.name, Reason: "read-only operand aliases mutable GDN state"}
		}
	}
	return nil
}

func qwen35GDNKernelStage(status int) string {
	if status < 0 {
		return "abi-validation"
	}
	switch status / 10000 {
	case 1:
		return "preflight"
	case 2:
		return "fused-input-projections"
	case 3:
		return "causal-conv-state"
	case 4:
		return "qk-normalization"
	case 5:
		return "recurrent-gated-norm"
	case 6:
		return "output-projection"
	case 7:
		return "stream-synchronize"
	case 8:
		return "scratch-allocation" // 80001: a P>=128 dequant + fp16 projection could not allocate scratch
	default:
		return "unknown"
	}
}
