package compute

// The sequence route composes the existing llama.cpp IQ2 port (iq_gemv.cuh,
// f7c01c5226cac1de1052c17920ff38cdefea0ae6, MIT; THIRD_PARTY_NOTICES.md).
// Weights stay GGUF-packed and activations stay F32. This is not a new kernel
// or an expansion of the loader's transformed GDN/QK retention policy.
const qwen35SequenceIQ2MaxTokens = 65535 // existing GEMV grid.y, not an admission heuristic
const cudaQwen35IQ2SequenceOp = "Qwen35IQ2Sequence"

// CUDAQwen35IQ2SequenceQualificationEnabled reports compiled build selection,
// never device availability, correctness, performance, or serving qualification.
func CUDAQwen35IQ2SequenceQualificationEnabled() bool { return qwen35SequenceIQ2Qualification }

func qwen35SequenceIQ2MatrixBytes(t Tensor) (int, bool) {
	block := 0
	switch t.Dtype {
	case IQ2_XXS:
		block = iq2xxsSuperBlock
	case IQ2_XS:
		block = iq2xsSuperBlock
	default:
		return 0, false
	}
	if len(t.Shape) != 2 || t.Shape[0] <= 0 || t.Shape[1] <= 0 || t.Shape[1]%rawIQSuper != 0 ||
		int64(t.Shape[0]) > qwen35GDNMaxCInt-7 || int64(t.Shape[1]) > qwen35GDNMaxCInt {
		// The launcher computes (out+7)/8 in signed C int. Alignment is per
		// row, not merely Numel()%256, and no unchecked Numel is used here.
		return 0, false
	}
	n, ok := qwen35GDNCheckedMul(int64(^uint(0)>>1), int64(t.Shape[0]), int64(t.Shape[1]/rawIQSuper), int64(block))
	return int(n), ok
}

// validateQwen35SequenceIQ2 runs for every matrix at request preflight, before
// KV reservation, allocations, transfers or recurrent mutation. It also guards
// the private checked-launch seam before allocating output.
// Non-IQ2 policy is unchanged; GDN still has its separate dtype admission.
func validateQwen35SequenceIQ2(t Tensor, tokens int, enabled bool) error {
	if t.Dtype != IQ2_XXS && t.Dtype != IQ2_XS {
		return nil
	}
	fail := func(reason string) error {
		return &Qwen35SequenceError{Stage: "iq2-preflight", Layer: -1, Reason: reason}
	}
	if !enabled {
		return fail("retained IQ2 sequence projection requires typed --qwen35-iq2-sequence-qualification; default off")
	}
	if _, ok := qwen35SequenceIQ2MatrixBytes(t); !ok {
		return fail("IQ2 matrix requires positive [out,in], in divisible by 256, bounded CUDA launch dimensions and packed extent")
	}
	if tokens < 1 || tokens > qwen35SequenceIQ2MaxTokens {
		return fail("IQ2 sequence projection requires 1 <= tokens <= 65535 (CUDA grid.y)")
	}
	for _, width := range t.Shape {
		if _, ok := qwen35GDNShapeBytes([]int{tokens, width}, F32.Bytes()); !ok {
			return fail("IQ2 input/output F32 panel byte extent overflows host int")
		}
	}
	return nil
}
