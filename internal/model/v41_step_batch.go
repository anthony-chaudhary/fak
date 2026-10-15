package model

import (
	"errors"
	"fmt"
	"os"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// v41StepBatchEnv disables the step recording window when set to "0", so the
// per-operation submit path stays one switch away for A/B and bisection.
const v41StepBatchEnv = "FAK_V41_STEP_BATCH"

type v41StepBatchError struct {
	err error
}

func (e *v41StepBatchError) Error() string { return fmt.Sprintf("v41 step batch: %v", e.err) }
func (e *v41StepBatchError) Unwrap() error { return e.err }

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
			return &v41StepBatchError{err: err}
		}
		return nil
	}
}

func v41FinishStepBatch(endBatch func() error, forwardErr *error, bodyComplete bool) {
	if !bodyComplete {
		// Recover only a secondary End panic while the forward unwind continues.
		defer func() { _ = recover() }()
		_ = endBatch()
		return
	}
	if endErr := endBatch(); endErr != nil {
		if *forwardErr == nil {
			*forwardErr = endErr
		} else {
			*forwardErr = errors.Join(*forwardErr, endErr)
		}
	}
}
