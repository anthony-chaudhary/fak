package gateway

import (
	"bytes"
	"encoding/json"
	"net/http"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

func validateChatRequestIngress(w http.ResponseWriter, req ChatRequest) bool {
	if !validateChatBeamSearch(w, req) {
		return false
	}
	if rejectStopLimits(w, normalizeStop(req.Stop)) {
		return false
	}
	if rejectGuidedRegexLimits(w, req) {
		return false
	}
	if rejectInvalidStructuredOutputs(w, req.StructuredOutputs) {
		return false
	}
	if len(req.Messages) == 0 {
		writeErr(w, http.StatusBadRequest, "messages: field required")
		return false
	}
	if rejectInvalidSampling(w, validateSampling(req)) {
		return false
	}
	if msg := validateSamplingPenalties(req.FrequencyPenalty, req.PresencePenalty); msg != "" {
		writeErrCode(w, http.StatusBadRequest, codeSamplingParamOutOfRange, msg)
		return false
	}
	if rejectInvalidResponseFormatCarrier(w, req.ResponseFormat) {
		return false
	}
	if rejectInvalidServiceTier(w, req.ServiceTier) {
		return false
	}
	receiptRequested := req.Fak != nil && req.Fak.NativeInferenceReceipt
	decodeTraceRequested := req.FakDecodeTrace
	decodeTokenIDsRequested := req.Fak != nil && req.Fak.NativeDecodeTokenIDs
	if decodeTokenIDsRequested && !decodeTraceRequested {
		writeErr(w, http.StatusBadRequest, "native decode token IDs require fak_decode_trace")
		return false
	}
	if req.Fak != nil && req.Fak.NativeDecodeTop5 {
		writeErr(w, http.StatusBadRequest, "fak.native_decode_top5 is served only on /v1/completions")
		return false
	}
	if decodeTraceRequested && req.Stream {
		writeErr(w, http.StatusBadRequest, "fak_decode_trace requires a buffered fak-native request")
		return false
	}
	if receiptRequested && req.Stream {
		writeErr(w, http.StatusBadRequest, "native inference receipts require a buffered request")
		return false
	}
	if receiptRequested && ((req.Temperature != nil && *req.Temperature != 0) || (req.TopP != nil && *req.TopP != 0) || len(req.LogitBias) > 0 || (req.FrequencyPenalty != nil && *req.FrequencyPenalty != 0) || (req.PresencePenalty != nil && *req.PresencePenalty != 0)) {
		writeErr(w, http.StatusBadRequest, "native inference receipts require greedy sampling over unmodified logits")
		return false
	}
	return true
}

// rejectInvalidStructuredOutputs refuses a non-object structured_outputs before
// dispatch; null stays absent, as it does for the other guided carriers.
func rejectInvalidStructuredOutputs(w http.ResponseWriter, raw json.RawMessage) bool {
	trimmed := bytes.TrimSpace(raw)
	if len(trimmed) == 0 || bytes.Equal(trimmed, []byte("null")) || trimmed[0] == '{' {
		return false
	}
	writeErrCode(w, http.StatusBadRequest, "invalid_structured_outputs", "structured_outputs must be a JSON object")
	return true
}

func determineChatRestoreTool(tools []agent.ToolDef) (restoreToolName string, present bool, autoAdvertise bool) {
	for _, t := range tools {
		if t.Function.Name == "mcp__fak_guard__fak_context_restore" {
			return "mcp__fak_guard__fak_context_restore", true, false
		}
	}
	for _, t := range tools {
		if t.Function.Name == "mcp__fak__fak_context_restore" {
			return "mcp__fak__fak_context_restore", true, false
		}
	}
	for _, t := range tools {
		if t.Function.Name == "fak_context_restore" {
			return "fak_context_restore", true, false
		}
	}
	for _, t := range tools {
		if isRestoreTool(t.Function.Name) {
			return t.Function.Name, true, false
		}
	}
	for _, t := range tools {
		if strings.HasPrefix(t.Function.Name, "mcp__fak_guard__") {
			return "mcp__fak_guard__fak_context_restore", false, true
		}
		if strings.HasPrefix(t.Function.Name, "mcp__fak__") {
			return "mcp__fak__fak_context_restore", false, true
		}
	}
	return "fak_context_restore", false, false
}

func hasElidedToolResult(messages []agent.Message) bool {
	for _, m := range messages {
		if strings.HasPrefix(m.Content, "...[fak: tool output elided") {
			return true
		}
	}
	return false
}
