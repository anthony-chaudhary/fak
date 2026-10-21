package gateway

import (
	"fmt"
	"net/http"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
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
