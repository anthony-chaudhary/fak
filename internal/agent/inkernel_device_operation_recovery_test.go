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
