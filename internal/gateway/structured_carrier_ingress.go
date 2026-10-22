package gateway

import (
	"errors"
	"net/http"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

// ValidateStructuredOutputCarriers applies the gateway's existing carrier envelope
// and resource checks to another served entry point. Provider schema semantics and
// unsupported native kinds remain the planner's responsibility.
func ValidateStructuredOutputCarriers(w http.ResponseWriter, req ChatRequest) bool {
	return !rejectGuidedRegexLimits(w, req) &&
		!rejectInvalidStructuredOutputs(w, req.StructuredOutputs) &&
		!rejectInvalidResponseFormatCarrier(w, req.ResponseFormat)
}

// WriteStructuredOutputError shares the gateway's typed native structured-output
// refusals without changing a caller's handling of unrelated planner errors.
func WriteStructuredOutputError(w http.ResponseWriter, err error) bool {
	if !errors.Is(err, agent.ErrNativeStructuredUnsupported) && !errors.Is(err, agent.ErrNativeStructuredNoToken) {
		return false
	}
	status, code, message := upstreamErrorStatus(err)
	writeErrCode(w, status, code, message)
	return true
}
