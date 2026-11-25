package agent

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"net/http"
)

// ErrIdenticalRetry matches an *IdenticalRetryError through errors.Is.
var ErrIdenticalRetry = errors.New("planner: identical request retry limit exceeded")

// IdenticalRetryError reports that the planner stopped re-sending a byte-identical
// request body after HTTPPlanner.IdenticalRetryLimit generation-class failures.
// Status is the last upstream HTTP status (0 for a transport/read failure), Sends is
// the number of times the identical body was sent, Cause is the last failure, and
// BodySHA256 is the hex digest that identifies the repeated body in logs.
type IdenticalRetryError struct {
	Status     int
	Sends      int
	Cause      error
	BodySHA256 string
}

func (e *IdenticalRetryError) Error() string {
	if e.Status != 0 {
		return fmt.Sprintf("planner: identical request failed %d times (last status %d); not re-sending: %v", e.Sends, e.Status, e.Cause)
	}
	return fmt.Sprintf("planner: identical request failed %d times; not re-sending: %v", e.Sends, e.Cause)
}

func (e *IdenticalRetryError) Unwrap() error { return e.Cause }

func (e *IdenticalRetryError) Is(target error) bool { return target == ErrIdenticalRetry }

// ErrContextOverflow matches a *ContextOverflowError through errors.Is.
var ErrContextOverflow = errors.New("planner: request exceeds the model context window")

// ContextOverflowError is the terminal form of an upstream refusal that says the
// prompt does not fit the model's context. The same request can never fit on a
// resend, so it is never retried.
type ContextOverflowError struct {
	Status int
	Cause  error
}

func (e *ContextOverflowError) Error() string {
	if e.Status != 0 {
		return fmt.Sprintf("planner: context overflow (status %d): %v", e.Status, e.Cause)
	}
	return fmt.Sprintf("planner: context overflow: %v", e.Cause)
}

func (e *ContextOverflowError) Unwrap() error { return e.Cause }

func (e *ContextOverflowError) Is(target error) bool { return target == ErrContextOverflow }

// generationClassStatus is the subset of retryable statuses that mean the upstream
// tried to generate and failed, as opposed to backpressure that clears with time.
func generationClassStatus(status int) bool {
	switch status {
	case http.StatusInternalServerError, http.StatusBadGateway, http.StatusGatewayTimeout:
		return true
	}
	return false
}

// backpressureStatus is a refusal to start generating that clears with time, so its body
// is never read as a verdict on the request itself.
func backpressureStatus(status int) bool {
	switch status {
	case http.StatusRequestTimeout, http.StatusTooManyRequests, http.StatusServiceUnavailable, 529:
		return true
	}
	return false
}

// noteIdenticalFailure counts one generation-class failure of body and returns an
// *IdenticalRetryError once the planner's IdenticalRetryLimit is exceeded. status is
// 0 for a transport or body-read failure. A done caller context is not counted: that
// failure is the caller's cancellation, not the request's.
func (s *retryState) noteIdenticalFailure(ctx context.Context, p *HTTPPlanner, body []byte, status int, cause error) error {
	if p == nil || p.IdenticalRetryLimit <= 0 {
		return nil
	}
	if ctx != nil && ctx.Err() != nil {
		return nil
	}
	if !s.identicalDigestSet {
		s.identicalDigest = sha256.Sum256(body)
		s.identicalDigestSet = true
	}
	s.identicalSends++
	if s.identicalSends <= p.IdenticalRetryLimit {
		return nil
	}
	return &IdenticalRetryError{Status: status, Sends: s.identicalSends, Cause: cause, BodySHA256: hex.EncodeToString(s.identicalDigest[:])}
}
