package microagent

import (
	"context"
	"fmt"
)

// VerificationInput is the evidence boundary presented before a microagent may
// declare a completed action successful. Verifier implementations should read
// external artifacts (tests, commits, receipts), not the agent self-report.
type VerificationInput struct {
	Agent string
	Steps int
}

// Verifier independently checks the effect claimed by a completed action.
// Returning an error refuses completion and supplies evidence to the configured
// retry feedback hook. A nil Verifier disables this gate without extra work.
type Verifier interface {
	Verify(context.Context, VerificationInput) error
}

// VerifierFunc adapts a function to Verifier.
type VerifierFunc func(context.Context, VerificationInput) error

func (f VerifierFunc) Verify(ctx context.Context, in VerificationInput) error {
	return f(ctx, in)
}

// VerificationError preserves independently gathered evidence while allowing
// callers and retry hooks to distinguish a self-check refusal from execution
// failure.
type VerificationError struct {
	Evidence error
}

func (e *VerificationError) Error() string {
	return fmt.Sprintf("microagent: verification failed: %v", e.Evidence)
}

func (e *VerificationError) Unwrap() error { return e.Evidence }
