package agent

import (
	"errors"
	"fmt"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/model"
)

// The V4.1 stage and prefill-latency refusals are recovered unchanged; textual
// lookalikes stay ordinary panics for the caller to re-raise.
// fak-test:runtime fast est=1ms
func TestV41AdmissionRecoveryPreservesIdentity(t *testing.T) {
	stage := &model.V41ForwardError{Layer: 2, Err: model.ErrV41ForwardStage}
	for _, c := range []struct {
		input error
		cause error
	}{
		{stage, model.ErrV41ForwardStage},
		{fmt.Errorf("direct session: %w", stage), model.ErrV41ForwardStage},
		{&model.V41PrefillLatencyError{}, model.ErrV41PrefillLatency},
	} {
		got, handled := recoverDevicePanic(c.input)
		if !handled || got != c.input || !errors.Is(got, c.cause) {
			t.Errorf("recovery = (%v, %v), want unchanged %v", got, handled, c.input)
		}
		var oom *InKernelOOMError
		if errors.As(got, &oom) {
			t.Error("V4.1 admission refusal relabeled as OOM")
		}
	}
	for _, input := range []any{stage.Error(), errors.New(stage.Error()), model.ErrV41PrefillLatency.Error()} {
		if got, handled := recoverDevicePanic(input); handled || got != nil {
			t.Errorf("textual lookalike %T was handled as a V4.1 refusal", input)
		}
	}
}

// Every selected V4.1 operation error is a request failure even when its cause is
// not a stage refusal (the ratio-one and device seams return these unwrapped).
// fak-test:runtime fast est=1ms
func TestV41OperationErrorsRecovered(t *testing.T) {
	cause := errors.New("selected operation failed")
	for _, op := range []error{
		&model.V41DecoderError{Err: cause},
		&model.V41EngramStreamError{},
		&model.V41LogitScaleOperationError{Cause: cause},
		&model.V41TailRoPEOperationError{Cause: cause},
		&model.V41IndexWeightScaleOperationError{Cause: cause},
		&model.V41IndexerScoreOperationError{Cause: cause},
		&model.V41QueryPrefixOperationError{Cause: cause},
		&model.V41MHCOperationError{Cause: cause},
		&model.V41SharedAttentionOperationError{Cause: cause},
		&model.V41BF16BoundaryOperationError{Cause: cause},
	} {
		for _, input := range []error{op, fmt.Errorf("direct session: %w", op)} {
			if got, handled := recoverDevicePanic(input); !handled || got != input {
				t.Errorf("%T: recovery = (%v, %v), want unchanged", op, got, handled)
			}
		}
	}
}
