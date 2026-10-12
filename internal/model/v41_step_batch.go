package model

import (
	"fmt"
	"os"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// v41StepBatchEnv disables the step recording window when set to "0", so the
// per-operation submit path stays one switch away for A/B and bisection.
const v41StepBatchEnv = "FAK_V41_STEP_BATCH"

// v41BeginStepBatch opens the backend's V4.1 step recording window when the
// backend offers one and returns the closer. The closer reports a submission
// fault from the final flush; it never replays work on the host.
func (s *Session) v41BeginStepBatch() func() error {
	b, ok := s.Backend.(compute.V41StepBatchBackend)
	if !ok || os.Getenv(v41StepBatchEnv) == "0" || !b.BeginV41StepBatch() {
		return func() error { return nil }
	}
	return func() error {
		if err := b.EndV41StepBatch(); err != nil {
			return fmt.Errorf("v41 decode step batch: %w", err)
		}
		return nil
	}
}
