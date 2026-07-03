package resume

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/accountobs"
)

// LimitClassification is the shared closed-vocabulary verdict for a provider
// rate/usage refusal. It is intentionally small enough to be used both by
// post-mortem transcript scan and by the live upstream 429 path.
type LimitClassification struct {
	Reason    string `json:"reason,omitempty"`
	ResetHint string `json:"reset_hint,omitempty"`
}

// ClassifyLimitText maps a provider refusal string onto exactly one closed
// rate-limit reason, or reports that it is not a rate limit at all. Specific
// account caps win before generic throttles; usage/quotas are checked last so a
// phrase like "not your usage limit; rate limited" is still a throttle.
func ClassifyLimitText(text string) (string, bool) {
	s := strings.ToLower(text)
	switch {
	case strings.Contains(s, "session limit"):
		return LimitSession, true
	case strings.Contains(s, "weekly limit"):
		return LimitWeekly, true
	case strings.Contains(s, "rate_limit"), strings.Contains(s, "rate limit"),
		strings.Contains(s, "rate-limit"), strings.Contains(s, "too many requests"),
		strings.Contains(s, "429"):
		return LimitRate, true
	case strings.Contains(s, "usage limit"), strings.Contains(s, "usage cap"),
		strings.Contains(s, "fable 5 limit"), strings.Contains(s, "fable limit"),
		strings.Contains(s, "/usage-credits"),
		strings.Contains(s, "quota exceeded"), strings.Contains(s, "insufficient_quota"):
		return LimitUsage, true
	default:
		return "", false
	}
}

// ClassifyLimitResponse classifies the live upstream 429 shape using the same
// vocabulary as ClassifyLimitText. The body may be OpenAI-style
// {error:{type,message}}, Anthropic-style {type:"error",error:{...}}, Gemini's
// {error:{status,message}}, or a plain text refusal. It sees NO headers — a
// usage/overage cap that surfaces as a 403 (its body text indistinguishable from an
// org-permission wall) is invisible to this function; use ClassifyLimitResponseWithHeader
// for the header-aware path. Kept for callers that only have a body (post-mortem text scan).
func ClassifyLimitResponse(status int, body []byte) (LimitClassification, bool) {
	if status != 429 {
		return LimitClassification{}, false
	}
	signals := extractLimitSignals(body)
	for _, msg := range signals.messages {
		if reason, ok := ClassifyLimitText(msg); ok {
			return LimitClassification{Reason: reason, ResetHint: firstResetHint(signals, msg)}, true
		}
	}
	for _, typ := range signals.types {
		if reason, ok := ClassifyLimitText(typ); ok {
			return LimitClassification{Reason: reason, ResetHint: firstResetHint(signals, "")}, true
		}
	}
	// HTTP 429 with no richer body signal is still the server-side throttle
	// member of the vocabulary, not a session/weekly/usage cap.
	return LimitClassification{Reason: LimitRate, ResetHint: firstResetHint(signals, "")}, true
}

// ClassifyLimitResponseWithHeader is ClassifyLimitResponse plus the HEADER signal that
// disambiguates a self-recovering usage/overage cap from a permanent wall. A Claude
// subscription cap can arrive as a 403 (not only a 429) — its org-flavored body text is
// identical to a permanent org/permission wall, so the body alone (ClassifyLimitResponse)
// cannot see it. The anthropic-ratelimit-unified-* / -overage-status headers can:
// accountobs.UsageOverageRejection folds them into "is this a recovering cap, and which
// window." So on a 403/402 (or any status) whose headers show an overage/window rejection,
// this returns a cap classification — session_limit for the 5h window, weekly_limit for the
// 7d, else the generic usage_limit — and the caller waits toward the reset instead of failing
// over or giving up. For a 429 with no such header it defers to ClassifyLimitResponse, so that
// path is byte-for-byte unchanged. A response with neither a 429 nor an overage header is not a
// limit (ok=false).
func ClassifyLimitResponseWithHeader(status int, body []byte, h http.Header) (LimitClassification, bool) {
	// The header signal is authoritative for the cap-vs-wall question and works for a 403/402
	// (the population ClassifyLimitResponse's 429 gate misses). Check it first.
	if rej := accountobs.UsageOverageRejection(h); rej.Rejected {
		reason := LimitUsage
		switch {
		case strings.HasPrefix(rej.Window, "5h"):
			reason = LimitSession
		case strings.HasPrefix(rej.Window, "7d"):
			reason = LimitWeekly
		}
		return LimitClassification{Reason: reason}, true
	}
	// No overage header: fall back to the body-only 429 vocabulary (unchanged).
	return ClassifyLimitResponse(status, body)
}

// ResetHint extracts the human "resets ..." tail from a provider refusal
// message, e.g. "You've hit your session limit · resets 8pm" -> "8pm".
func ResetHint(msg string) string {
	i := strings.Index(strings.ToLower(msg), "resets ")
	if i < 0 {
		return ""
	}
	return strings.TrimSpace(msg[i+len("resets "):])
}

type limitSignals struct {
	messages []string
	types    []string
	resets   []string
}

func firstResetHint(sig limitSignals, preferred string) string {
	if h := ResetHint(preferred); h != "" {
		return h
	}
	for _, msg := range sig.messages {
		if h := ResetHint(msg); h != "" {
			return h
		}
	}
	for _, h := range sig.resets {
		if h = strings.TrimSpace(h); h != "" {
			return h
		}
	}
	return ""
}

func extractLimitSignals(body []byte) limitSignals {
	var sig limitSignals
	raw := strings.TrimSpace(string(body))
	if raw == "" {
		return sig
	}
	dec := json.NewDecoder(bytes.NewReader(body))
	dec.UseNumber()
	var v any
	if err := dec.Decode(&v); err == nil {
		collectLimitSignals(v, "", &sig)
		return sig
	}
	sig.messages = append(sig.messages, raw)
	return sig
}

func collectLimitSignals(v any, key string, sig *limitSignals) {
	switch x := v.(type) {
	case map[string]any:
		for k, val := range x {
			collectLimitSignals(val, strings.ToLower(k), sig)
		}
	case []any:
		for _, val := range x {
			collectLimitSignals(val, key, sig)
		}
	case string:
		addLimitStringSignal(key, x, sig)
	case json.Number:
		if key == "code" || key == "status" {
			sig.types = append(sig.types, x.String())
		}
	case float64:
		if key == "code" || key == "status" {
			sig.types = append(sig.types, fmt.Sprintf("%.0f", x))
		}
	}
}

func addLimitStringSignal(key, value string, sig *limitSignals) {
	value = strings.TrimSpace(value)
	if value == "" {
		return
	}
	switch key {
	case "message", "detail", "error_description", "description":
		sig.messages = append(sig.messages, value)
	case "type", "status", "code", "reason":
		sig.types = append(sig.types, value)
	}
	if strings.Contains(key, "reset") {
		sig.resets = append(sig.resets, value)
	}
}
