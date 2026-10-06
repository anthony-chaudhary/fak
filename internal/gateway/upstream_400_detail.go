package gateway

import (
	"encoding/json"
	"errors"
	"net/http"
	"os"
	"strconv"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

// upstream400Detail is what the gateway can learn from an upstream 400 body without
// forwarding the body itself. ContextOverflow and the two token counts are typed values
// (a bool and integers), so they can ride to any caller; Message is the upstream's raw
// prose and is only ever surfaced scrubbed, bounded, and opt-in.
type upstream400Detail struct {
	ContextOverflow bool
	// PromptTokens is the upstream-reported prompt size (llama-server n_prompt_tokens).
	PromptTokens int
	// ContextWindow is the upstream-reported per-request window (llama-server n_ctx,
	// which is the per-slot context, not the whole --ctx-size across --parallel slots).
	ContextWindow int
	Message       string
}

// upstreamContextOverflowCodes are the typed error code/type values engines put on a
// context-window overflow. Matched before any message text.
var upstreamContextOverflowCodes = map[string]bool{
	"context_length_exceeded":   true, // OpenAI-compatible code
	"exceed_context_size_error": true, // llama-server type
}

// upstreamContextOverflowPhrases are lowercase message substrings naming an overflow,
// used only when no typed code/type decided it (vLLM and some OpenAI-compatible servers
// put the reason in the message alone).
var upstreamContextOverflowPhrases = []string{
	"exceeds the available context size", // llama-server
	"maximum context length",             // OpenAI / vLLM
	"context length exceeded",
	"exceeds the context window",
	"context window exceeded",
	"prompt is too long", // Anthropic-compatible
}

type upstreamErrorObject struct {
	Message       string          `json:"message"`
	Type          string          `json:"type"`
	Code          json.RawMessage `json:"code"`
	NPromptTokens json.RawMessage `json:"n_prompt_tokens"`
	SlotWindow    json.RawMessage `json:"n_ctx"`
}

// parseUpstream400 classifies an upstream 400 body. Typed fields win: the JSON error
// code/type decide an overflow first, the message text only as a fallback, and a body
// that is not JSON (or was truncated at the agent seam) falls back to the raw-body
// signature match.
func parseUpstream400(body string) upstream400Detail {
	var d upstream400Detail
	trimmed := strings.TrimSpace(body)
	if trimmed == "" {
		return d
	}
	var env struct {
		Error json.RawMessage `json:"error"`
		upstreamErrorObject
	}
	if err := json.Unmarshal([]byte(trimmed), &env); err != nil {
		d.ContextOverflow = isUpstreamContextOverflow(trimmed)
		d.Message = trimmed
		return d
	}
	objs := []upstreamErrorObject{env.upstreamErrorObject}
	if len(env.Error) > 0 {
		var inner upstreamErrorObject
		var text string
		switch {
		case json.Unmarshal(env.Error, &inner) == nil:
			objs = append([]upstreamErrorObject{inner}, objs...)
		case json.Unmarshal(env.Error, &text) == nil:
			objs = append([]upstreamErrorObject{{Message: text}}, objs...)
		}
	}
	for _, o := range objs {
		if d.Message == "" {
			d.Message = strings.TrimSpace(o.Message)
		}
		if d.PromptTokens == 0 {
			d.PromptTokens = rawPositiveInt(o.NPromptTokens)
		}
		if d.ContextWindow == 0 {
			d.ContextWindow = rawPositiveInt(o.SlotWindow)
		}
		if upstreamContextOverflowCodes[strings.ToLower(strings.TrimSpace(o.Type))] ||
			upstreamContextOverflowCodes[strings.ToLower(rawString(o.Code))] {
			d.ContextOverflow = true
		}
	}
	if !d.ContextOverflow {
		lower := strings.ToLower(d.Message)
		for _, phrase := range upstreamContextOverflowPhrases {
			if lower != "" && strings.Contains(lower, phrase) {
				d.ContextOverflow = true
				break
			}
		}
	}
	return d
}

func rawString(raw json.RawMessage) string {
	var s string
	if len(raw) == 0 || json.Unmarshal(raw, &s) != nil {
		return ""
	}
	return strings.TrimSpace(s)
}

func rawPositiveInt(raw json.RawMessage) int {
	if len(raw) == 0 {
		return 0
	}
	var n json.Number
	if json.Unmarshal(raw, &n) != nil {
		return 0
	}
	v, err := strconv.ParseInt(n.String(), 10, 64)
	if err != nil || v <= 0 || v > 1<<31-1 {
		return 0
	}
	return int(v)
}

// upstreamErrorFields returns the extra typed fields the OpenAI error envelope carries
// beside code/message for a context overflow or an upstream 400: prompt_tokens and
// context_window (integers, safe on every path) and, only where upstream detail exposure
// is enabled, a scrubbed and bounded upstream_message. nil when there is nothing to add.
func (s *Server) upstreamErrorFields(err error, code string) map[string]any {
	fields := map[string]any{}
	var contextErr *agent.InKernelContextLengthError
	var se *agent.UpstreamStatusError
	switch {
	case errors.As(err, &contextErr):
		if code == "context_length_exceeded" {
			putPositive(fields, "prompt_tokens", contextErr.PromptTokens)
			putPositive(fields, "context_window", contextErr.MaxContext)
		}
	case errors.As(err, &se) && se.Status == http.StatusBadRequest:
		if code != "context_length_exceeded" && code != "upstream_invalid_request" {
			break
		}
		d := parseUpstream400(se.Body)
		if code == "context_length_exceeded" {
			putPositive(fields, "prompt_tokens", d.PromptTokens)
			putPositive(fields, "context_window", d.ContextWindow)
		}
		if s != nil && s.exposeUpstreamErrorDetail {
			if msg := scrubForbiddenDetail(d.Message); msg != "" {
				fields["upstream_message"] = msg
			}
		}
	}
	if len(fields) == 0 {
		return nil
	}
	return fields
}

func putPositive(fields map[string]any, key string, v int) {
	if v > 0 {
		fields[key] = v
	}
}

// exposeUpstreamErrorDetailEnv lets a host that owns its callers (an appliance whose
// clients are its own router/agents) opt the externally-exposed serve path into the
// scrubbed upstream 400 detail. Off unless FAK_EXPOSE_UPSTREAM_ERROR_DETAIL is truthy.
func exposeUpstreamErrorDetailEnv() bool {
	switch strings.ToLower(strings.TrimSpace(os.Getenv("FAK_EXPOSE_UPSTREAM_ERROR_DETAIL"))) {
	case "1", "true", "on", "yes":
		return true
	default:
		return false
	}
}
