package gateway

import (
	"errors"
	"fmt"
	"net/http"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/model"
)

// fak-test:runtime fast est=1ms lane=default
func TestNativeStructuredOutputErrorClassification(t *testing.T) {
	for _, tc := range []struct {
		name   string
		err    error
		status int
		code   string
	}{
		{"unsupported", agent.ErrNativeStructuredUnsupported, http.StatusBadRequest, "native_structured_output_unsupported"},
		{"decode_failure", agent.ErrNativeStructuredNoToken, http.StatusInternalServerError, "native_structured_decode_failed"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			status, code, message := upstreamErrorStatus(fmt.Errorf("private request detail: %w", tc.err))
			if status != tc.status || code != tc.code {
				t.Fatalf("error classification %d/%s, want %d/%s", status, code, tc.status, tc.code)
			}
			if message == "" || strings.Contains(message, "private request detail") {
				t.Fatalf("unsafe error detail: %q", message)
			}
		})
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestNativeStructuredOutputErrorsRegression(t *testing.T) {
	// The structured-output arms must not disturb the pre-existing upstream
	// error mappings: these rows pin status+code for errors that predate
	// native structured output.
	for _, tc := range []struct {
		name   string
		err    error
		status int
		code   string
	}{
		{"seed_unsupported", agent.ErrSeedUnsupported, http.StatusBadRequest, "sampling_seed_unsupported"},
		{"sampling_mode_unsupported", model.ErrSamplingModeUnsupported, http.StatusBadRequest, "sampling_mode_unsupported"},
		{"nonfinite_logits", agent.ErrNonFiniteLogits, http.StatusInternalServerError, "native_sampler_nonfinite_logits"},
		{"upstream_unreachable", &agent.UpstreamUnreachableError{Err: errors.New("dial tcp: connection refused")}, http.StatusBadGateway, "upstream_unreachable"},
		{"generic_planner_error", errors.New("planner transport failure"), http.StatusBadGateway, ""},
	} {
		t.Run(tc.name, func(t *testing.T) {
			status, code, message := upstreamErrorStatus(fmt.Errorf("private request detail: %w", tc.err))
			if status != tc.status || code != tc.code {
				t.Fatalf("regression mapping %d/%q, want %d/%q", status, code, tc.status, tc.code)
			}
			if message == "" || strings.Contains(message, "private request detail") {
				t.Fatalf("unsafe or missing error detail: %q", message)
			}
		})
	}
}
