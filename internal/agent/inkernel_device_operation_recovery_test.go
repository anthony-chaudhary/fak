package agent

import (
	"errors"
	"fmt"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/polymodel"
)

// SW-VERIFIED only: request-boundary panic values require no device.
// fak-test:runtime fast est=1ms
func TestDeviceOnlyExpertRingOperationRecoveryPreservesIdentity(t *testing.T) {
	for _, cause := range []error{polymodel.ErrTooLarge, polymodel.ErrPinnedNoRoom, &compute.DeviceAllocError{Bytes: 64, Class: compute.MemoryWeights}} {
		operation := &model.BackendForwardOperationError{Backend: "cpu-recording", Cause: cause}
		for _, input := range []error{operation, fmt.Errorf("request: %w", operation)} {
			got, handled := recoverDevicePanic(input)
			if !handled {
				t.Errorf("operation with cause %T was not handled", cause)
				continue
			}
			if got != input {
				t.Errorf("recovery replaced input %T with %T", input, got)
			}
			var recovered *model.BackendForwardOperationError
			if !errors.As(got, &recovered) || recovered != operation || !errors.Is(got, cause) {
				t.Error("recovery lost operation identity or cause")
			}
			var oom *InKernelOOMError
			if errors.As(got, &oom) {
				t.Error("operation was relabeled as OOM")
			}
		}
	}
	for _, input := range []any{errors.New("unrelated"), "unrelated", 42} {
		got, handled := recoverDevicePanic(input)
		if handled || got != nil {
			t.Errorf("unrelated panic %T was handled", input)
		}
	}
}

// Only the two existing Flash architecture-admission sentinels are recoverable;
// textual lookalikes remain ordinary panics for the caller to re-raise.
// fak-test:runtime fast est=1ms
func TestV4FlashAdmissionRecoveryPreservesIdentity(t *testing.T) {
	for _, cause := range []error{model.ErrV4FlashAttentionUnimplemented, model.ErrV4FlashAttentionRatioInvalid} {
		for _, input := range []error{cause, fmt.Errorf("direct session: %w", cause)} {
			got, handled := recoverDevicePanic(input)
			if !handled || got != input || !errors.Is(got, cause) {
				t.Errorf("recovery = (%v, %v), want unchanged %v", got, handled, input)
			}
			var oom *InKernelOOMError
			var operation *model.BackendForwardOperationError
			if errors.As(got, &oom) || errors.As(got, &operation) {
				t.Error("architecture refusal relabeled as device failure")
			}
		}
		for _, input := range []any{cause.Error(), errors.New(cause.Error())} {
			if got, handled := recoverDevicePanic(input); handled || got != nil {
				t.Errorf("textual lookalike %T was handled as architecture refusal", input)
			}
		}
	}
}
