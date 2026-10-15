//go:build vulkan && (windows || linux) && cgo

package compute

/*
#include "vulkan_backend.h"
*/
import "C"

var _ V41StepBatchBackend = (*vulkanBackend)(nil)
var _ V41StepBatchRecordingBackend = (*vulkanBackend)(nil)

// V41StepBatchRecording observes the process-global V4.1 recording state. The
// window-active flag alone includes suspended recording; both native predicates
// and a clear sticky submission status are required for this momentary snapshot.
func (v *vulkanBackend) V41StepBatchRecording() bool {
	if v == nil {
		return false
	}
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	return C.fvk_v41_step_batch_active() != 0 && bool(C.fvk_batch_active()) && C.fvk_submission_status() == 0
}

// BeginV41StepBatch opens the V4.1 decode-step recording window (#13820). It
// returns false when another batch is open or the context holds a sticky
// fault; the step then runs on the per-operation submit path.
func (v *vulkanBackend) BeginV41StepBatch() bool {
	if v == nil {
		return false
	}
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	return C.fvk_v41_step_batch_begin() == 0
}

// EndV41StepBatch submits any recorded work left in the step window and closes
// it. A nonzero native status is the sticky submission fault, never a replay.
func (v *vulkanBackend) EndV41StepBatch() error {
	if v == nil {
		return nil
	}
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	if status := int(C.fvk_v41_step_batch_end()); status != 0 {
		err := vulkanReadError(status)
		err.Site = "V41StepBatch"
		return err
	}
	return nil
}
