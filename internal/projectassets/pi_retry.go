package projectassets

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
)

// PiRetryPolicy is Pi's agent-turn auto-retry block in settings.json
// (`retry.enabled`, `retry.maxRetries`, `retry.baseDelayMs`,
// `retry.maxAgentDelayMs`). Pi restarts a turn that failed with a retryable
// provider error (429, 5xx, rate limit, overloaded, connection refused,
// timeout) after baseDelayMs*2^(attempt-1), capped at maxAgentDelayMs. That
// loop ignores Retry-After. Pi's defaults (3 retries from 2s, ~14s in all)
// end a turn while a saturated router is still queueing, so a session dies
// on a transient 429 or a router restart.
type PiRetryPolicy struct {
	MaxRetries      int
	BaseDelayMs     int
	MaxAgentDelayMs int
}

// DefaultPiRouterRetryPolicy outlasts one saturated fak router window: two
// Halos at their max_in_flight bound kept routed upstream waits at a p95 of
// 135-185s, and a router restart answers 503 for under a minute. The backoff
// 4+8+16+32+60+60+60+60 seconds gives about five minutes before the turn fails.
var DefaultPiRouterRetryPolicy = PiRetryPolicy{MaxRetries: 8, BaseDelayMs: 4000, MaxAgentDelayMs: 60000}

// TotalDelayMs is the summed backoff before Pi gives up on a turn.
func (p PiRetryPolicy) TotalDelayMs() int {
	total := 0
	for attempt := 1; attempt <= p.MaxRetries; attempt++ {
		d := p.BaseDelayMs << (attempt - 1)
		if d > p.MaxAgentDelayMs || d <= 0 {
			d = p.MaxAgentDelayMs
		}
		total += d
	}
	return total
}

// ReadPiRetryPolicy reports the agent-turn retry keys a settings object
// carries, and whether retry is enabled (Pi's default when unset is true).
func ReadPiRetryPolicy(raw map[string]interface{}) (PiRetryPolicy, bool) {
	block, _ := raw["retry"].(map[string]interface{})
	enabled := true
	if v, ok := block["enabled"].(bool); ok {
		enabled = v
	}
	var p PiRetryPolicy
	p.MaxRetries, _ = numericField(block["maxRetries"])
	p.BaseDelayMs, _ = numericField(block["baseDelayMs"])
	p.MaxAgentDelayMs, _ = numericField(block["maxAgentDelayMs"])
	return p, enabled
}

// PiRetryPolicyMatches reports whether settings already carry policy with
// retry explicitly enabled.
func PiRetryPolicyMatches(raw map[string]interface{}, policy PiRetryPolicy) bool {
	block, _ := raw["retry"].(map[string]interface{})
	enabled, ok := block["enabled"].(bool)
	cur, _ := ReadPiRetryPolicy(raw)
	return ok && enabled && cur == policy
}

// EnsurePiRetryPolicy sets settings.json retry.enabled/maxRetries/baseDelayMs/
// maxAgentDelayMs to policy (fak owns those keys). retry.provider and every
// unrelated key are preserved. It returns (resolvedPath, modified, error).
func EnsurePiRetryPolicy(target string, policy PiRetryPolicy) (string, bool, error) {
	path := ResolvePiSettingsPath(target)
	raw := map[string]interface{}{}
	data, err := os.ReadFile(path)
	switch {
	case err == nil:
		if unmarshalErr := json.Unmarshal(stripUTF8BOM(data), &raw); unmarshalErr != nil {
			return path, false, fmt.Errorf("parse existing %s: %w", path, unmarshalErr)
		}
		if raw == nil {
			raw = map[string]interface{}{}
		}
	case os.IsNotExist(err):
		if mkErr := os.MkdirAll(filepath.Dir(path), 0755); mkErr != nil {
			return path, false, fmt.Errorf("mkdir %s: %w", filepath.Dir(path), mkErr)
		}
	default:
		return path, false, fmt.Errorf("read %s: %w", path, err)
	}
	if PiRetryPolicyMatches(raw, policy) {
		return path, false, nil
	}
	block, _ := raw["retry"].(map[string]interface{})
	if block == nil {
		block = map[string]interface{}{}
	}
	block["enabled"] = true
	block["maxRetries"] = policy.MaxRetries
	block["baseDelayMs"] = policy.BaseDelayMs
	block["maxAgentDelayMs"] = policy.MaxAgentDelayMs
	raw["retry"] = block
	out, marshalErr := json.MarshalIndent(raw, "", "  ")
	if marshalErr != nil {
		return path, false, fmt.Errorf("serialize %s: %w", path, marshalErr)
	}
	if writeErr := os.WriteFile(path, append(out, '\n'), 0644); writeErr != nil {
		return path, false, fmt.Errorf("write %s: %w", path, writeErr)
	}
	return path, true, nil
}
