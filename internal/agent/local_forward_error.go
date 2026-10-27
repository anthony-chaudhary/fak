package agent

import (
	"errors"
	"fmt"
)

// LocalForwardError wraps an error from the local inference engine's forward
// function (e.g. the CUDA forward or batcher tokenGenFn). It is a LOCAL error
// — the request failed inside fak's own engine, not at an upstream provider —
// so the gateway must NOT surface it as the opaque 502 "upstream model error"
// fallthrough. Instead it is classified as a 500 (non-retryable) or 503
// (retryable with a smaller prompt) with a distinct OpenAI error code.
//
// Retryable indicates whether the request can be retried, typically with a
// smaller prompt/context size. A non-retryable LocalForwardError means the
// engine cannot serve this request at all on the current hardware/model.
type LocalForwardError struct {
	// Op is the operation that failed, e.g. "prefill" or "decode".
	Op string
	// Err is the underlying error from the forward function.
	Err error
	// Retryable indicates whether retrying with a smaller prompt may succeed.
	Retryable bool
}

func (e *LocalForwardError) Error() string {
	if e == nil {
		return "local forward error"
	}
	return fmt.Sprintf("local forward error during %s: %v", e.Op, e.Err)
}

// Unwrap returns the underlying error so errors.Is and errors.As can traverse it.
func (e *LocalForwardError) Unwrap() error {
	return e.Err
}

// NewLocalForwardError constructs a LocalForwardError for the given operation.
func NewLocalForwardError(op string, err error, retryable bool) *LocalForwardError {
	return &LocalForwardError{Op: op, Err: err, Retryable: retryable}
}

// IsLocalForwardError reports whether err wraps a *LocalForwardError.
func IsLocalForwardError(err error) bool {
	var lfe *LocalForwardError
	return errors.As(err, &lfe)
}

// LocalForwardErrorRetryable returns the Retryable flag if err is a
// *LocalForwardError, false otherwise.
func LocalForwardErrorRetryable(err error) bool {
	var lfe *LocalForwardError
	if errors.As(err, &lfe) {
		return lfe.Retryable
	}
	return false
}
