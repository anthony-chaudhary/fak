package stopgate

import (
	"crypto/md5"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"strings"
	"sync"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// DefaultCircuitBreakerThreshold is the default number of consecutive identical
// failing tool calls required to trip the circuit breaker (#11771).
const DefaultCircuitBreakerThreshold = 3

// CanonicalJSONArgs normalizes rawArgs into a canonical JSON representation:
// JSON object keys are sorted lexicographically at all nesting levels, and
// insignificant whitespace is stripped.
// If rawArgs is empty or whitespace-only, it returns "".
// If rawArgs cannot be parsed as valid JSON, it returns strings.TrimSpace(rawArgs).
func CanonicalJSONArgs(rawArgs string) string {
	trimmed := strings.TrimSpace(rawArgs)
	if trimmed == "" {
		return ""
	}
	var v any
	if err := json.Unmarshal([]byte(trimmed), &v); err != nil {
		return trimmed
	}
	canon, err := json.Marshal(v)
	if err != nil {
		return trimmed
	}
	return string(canon)
}

// ToolInvocationSignature computes MD5(tool_name || canonical_json_args).
// The signature is returned as a 32-character lowercase hex string.
// Key ordering and whitespace variations in JSON args produce identical signatures (#11771).
func ToolInvocationSignature(toolName string, rawArgs string) string {
	canon := CanonicalJSONArgs(rawArgs)
	payload := []byte(toolName + canon)
	h := md5.Sum(payload)
	return hex.EncodeToString(h[:])
}

// ComputeToolSignature is an alias for ToolInvocationSignature.
func ComputeToolSignature(toolName string, rawArgs string) string {
	return ToolInvocationSignature(toolName, rawArgs)
}

// CircuitBreaker tracks tool invocation signatures across turns and trips
// only when identical failing calls repeat >= threshold (default 3) consecutive
// times without progress, or when a valid signed refusal token is presented (#11771).
type CircuitBreaker struct {
	mu               sync.Mutex
	threshold        int
	consecutiveCount int
	lastSignature    string
	lastTool         string
	lastArgs         string
	tripped          bool
	tripReason       string
}

// NewCircuitBreaker creates a new CircuitBreaker with the default threshold (3).
func NewCircuitBreaker() *CircuitBreaker {
	return NewCircuitBreakerWithThreshold(DefaultCircuitBreakerThreshold)
}

// NewCircuitBreakerWithThreshold creates a CircuitBreaker with a custom consecutive failure threshold.
// If threshold < 1, it falls back to DefaultCircuitBreakerThreshold (3).
func NewCircuitBreakerWithThreshold(threshold int) *CircuitBreaker {
	if threshold < 1 {
		threshold = DefaultCircuitBreakerThreshold
	}
	return &CircuitBreaker{
		threshold: threshold,
	}
}

// RecordInvocation records a tool call's invocation outcome.
//
// If hasProgress is true or failed is false (success):
//   - Consecutive failure tracking is reset to 0.
//   - If the breaker is not yet tripped, it remains closed (false).
//
// If failed is true and hasProgress is false:
//   - The call's MD5 invocation signature MD5(tool_name || canonical_json_args) is computed.
//   - If the signature matches the previous failing signature, the consecutive counter increments.
//   - If the signature differs (e.g. exploratory retry with different tool or args),
//     the consecutive counter resets to 1 for the new signature.
//   - If the consecutive counter reaches or exceeds the threshold (default 3),
//     the circuit breaker trips and returns true.
func (cb *CircuitBreaker) RecordInvocation(toolName string, rawArgs string, failed bool, hasProgress bool) bool {
	cb.mu.Lock()
	defer cb.mu.Unlock()

	if cb.tripped {
		return true
	}

	if !failed || hasProgress {
		cb.consecutiveCount = 0
		cb.lastSignature = ""
		cb.lastTool = ""
		cb.lastArgs = ""
		return false
	}

	sig := ToolInvocationSignature(toolName, rawArgs)
	if sig == cb.lastSignature && cb.lastSignature != "" {
		cb.consecutiveCount++
	} else {
		cb.consecutiveCount = 1
		cb.lastSignature = sig
		cb.lastTool = toolName
		cb.lastArgs = rawArgs
	}

	if cb.consecutiveCount >= cb.threshold {
		cb.tripped = true
		cb.tripReason = fmt.Sprintf("tool %q repeatedly failed with identical invocation signature %s (%d consecutive occurrences without progress)", toolName, sig, cb.consecutiveCount)
		return true
	}

	return false
}

// RecordFailure records an invocation failure without forward progress.
func (cb *CircuitBreaker) RecordFailure(toolName string, rawArgs string) bool {
	return cb.RecordInvocation(toolName, rawArgs, true, false)
}

// RecordSuccess records an invocation success with forward progress.
func (cb *CircuitBreaker) RecordSuccess(toolName string, rawArgs string) {
	cb.RecordInvocation(toolName, rawArgs, false, true)
}

// RecordProgress explicitly signals forward progress, clearing consecutive failure counters.
func (cb *CircuitBreaker) RecordProgress() {
	cb.mu.Lock()
	defer cb.mu.Unlock()
	cb.consecutiveCount = 0
	cb.lastSignature = ""
	cb.lastTool = ""
	cb.lastArgs = ""
}

// RecordSignedRefusal trips the circuit breaker if the given refusal receipt
// carries a valid signed refusal token or represents a verified terminal boundary (#11771).
func (cb *CircuitBreaker) RecordSignedRefusal(receipt *BoundaryRefusalReceipt) bool {
	cb.mu.Lock()
	defer cb.mu.Unlock()

	if receipt == nil {
		return false
	}

	isSigned := IsSignedRefusalToken(receipt)
	isTerminal := IsTerminalBoundary(receipt)
	if isSigned || (receipt.Verified && isTerminal) {
		cb.tripped = true
		reason := receipt.Reason
		if reason == "" && receipt.ReasonCode != abi.ReasonNone {
			reason = abi.ReasonName(receipt.ReasonCode)
		}
		if reason == "" {
			reason = "SIGNED_REFUSAL"
		}
		cb.tripReason = fmt.Sprintf("valid signed refusal token received: %s", reason)
		return true
	}
	return false
}

// Trip manually trips the circuit breaker with a given explanation.
func (cb *CircuitBreaker) Trip(reason string) {
	cb.mu.Lock()
	defer cb.mu.Unlock()
	cb.tripped = true
	if reason == "" {
		reason = "manual circuit breaker trip"
	}
	cb.tripReason = reason
}

// IsTripped returns true if the circuit breaker has tripped.
func (cb *CircuitBreaker) IsTripped() bool {
	cb.mu.Lock()
	defer cb.mu.Unlock()
	return cb.tripped
}

// TripReason returns the reason why the circuit breaker tripped.
func (cb *CircuitBreaker) TripReason() string {
	cb.mu.Lock()
	defer cb.mu.Unlock()
	return cb.tripReason
}

// ConsecutiveFailures returns the count of consecutive identical failing calls.
func (cb *CircuitBreaker) ConsecutiveFailures() int {
	cb.mu.Lock()
	defer cb.mu.Unlock()
	return cb.consecutiveCount
}

// LastSignature returns the MD5 invocation signature of the last recorded failing call.
func (cb *CircuitBreaker) LastSignature() string {
	cb.mu.Lock()
	defer cb.mu.Unlock()
	return cb.lastSignature
}

// Threshold returns the threshold count for tripping.
func (cb *CircuitBreaker) Threshold() int {
	cb.mu.Lock()
	defer cb.mu.Unlock()
	return cb.threshold
}

// Reset clears failure tracking and closes the circuit breaker.
func (cb *CircuitBreaker) Reset() {
	cb.mu.Lock()
	defer cb.mu.Unlock()
	cb.consecutiveCount = 0
	cb.lastSignature = ""
	cb.lastTool = ""
	cb.lastArgs = ""
	cb.tripped = false
	cb.tripReason = ""
}
