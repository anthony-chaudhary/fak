package compute

import (
	"errors"
	"fmt"
)

// ErrQwen35SequenceUnsupportedMatrixDtype is the closed verdict for a resident
// sequence matrix operand whose dtype has no Vulkan sequence projection kernel.
// Callers match it with errors.Is; the wrapped text names the operand and dtype.
var ErrQwen35SequenceUnsupportedMatrixDtype = errors.New("compute: Qwen3.5 resident sequence has no projection kernel for matrix dtype")

// qwen35SequenceKQuantPipelines records which optional packed K-quant pipelines
// the loaded Vulkan bundle carries (fvk_have_q{6,5,3}k_matmul).
type qwen35SequenceKQuantPipelines struct {
	q6k, q5k, q3k bool
}

// qwen35VulkanSequenceMatrixDtypeErr is the admission half of the dtype contract
// whose execution half is qwen35VulkanSequenceMatMulLocked: every dtype admitted
// here must have a case there. Raw i-quants are admitted separately, by the
// native-pipeline probe in the validator. Q6_K/Q5_K/Q3_K are admitted only when
// the bundle carries the matching optional pipeline, the same probe serve uses
// to retain dense packed K-quants on Vulkan.
func qwen35VulkanSequenceMatrixDtypeErr(dt Dtype, have qwen35SequenceKQuantPipelines) error {
	var present bool
	switch dt {
	case F32, Q8_0, Q4_K, Q2_K:
		return nil
	case Q6_K:
		present = have.q6k
	case Q5_K:
		present = have.q5k
	case Q3_K:
		present = have.q3k
	default:
		return fmt.Errorf("%w %s", ErrQwen35SequenceUnsupportedMatrixDtype, dt)
	}
	if present {
		return nil
	}
	return fmt.Errorf("%w %s (bundle lacks the optional matmul pipeline)", ErrQwen35SequenceUnsupportedMatrixDtype, dt)
}

// qwen35SequenceKQuantBlockBytes returns the packed bytes per 256-weight
// super-block for the K-quants the sequence route dispatches.
func qwen35SequenceKQuantBlockBytes(dt Dtype) (int64, bool) {
	switch dt {
	case Q4_K:
		return 144, true
	case Q2_K:
		return 84, true
	case Q6_K:
		return 210, true
	case Q5_K:
		return 176, true
	case Q3_K:
		return 110, true
	default:
		return 0, false
	}
}
