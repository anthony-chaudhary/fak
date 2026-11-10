package security

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"time"
)

var (
	// ErrInvalidCredentialRequest is returned when a credential request is malformed,
	// such as when the requested session binding is empty.
	ErrInvalidCredentialRequest = errors.New("security: invalid credential request")

	// ErrNoActiveSession is returned when a credential request names a session that is
	// unknown or no longer active.
	ErrNoActiveSession = errors.New("security: no active session bound to credential request")

	// ErrCredentialMintFailed is returned when the underlying minter fails to produce a
	// credential for an otherwise valid request.
	ErrCredentialMintFailed = errors.New("security: credential mint failed")

	// ErrNilCredentialMinter is returned when a broker is constructed without a minter.
	ErrNilCredentialMinter = errors.New("security: credential minter is required")
)

// ScmCredential is a short-lived source-control credential minted on demand for a
// sandboxed worker. The secret value is never rendered by formatting verbs.
type ScmCredential struct {
	Scheme    string
	Value     string
	ExpiresAt time.Time
}

// String renders the credential with its secret value redacted.
func (c ScmCredential) String() string {
	return fmt.Sprintf("ScmCredential{scheme=%s, expiry=%s}", c.Scheme, c.ExpiresAt.UTC().Format(time.RFC3339))
}

// GoString renders the credential with its secret value redacted for %#v.
func (c ScmCredential) GoString() string {
	return c.String()
}

// Expired reports whether the credential is no longer valid at now. A zero expiry is
// treated as expired.
func (c ScmCredential) Expired(now time.Time) bool {
	if c.ExpiresAt.IsZero() {
		return true
	}
	return !now.Before(c.ExpiresAt)
}

// CredentialMinter mints a short-lived SCM credential for an authenticated session.
type CredentialMinter interface {
	MintScmCredential(ctx context.Context, sessionID string) (ScmCredential, error)
}

// CredentialBroker mints short-lived SCM credentials for sandboxed workers after
// validating that the caller is bound to an active session.
type CredentialBroker struct {
	minter CredentialMinter
}

// NewCredentialBroker constructs a CredentialBroker around the supplied minter.
func NewCredentialBroker(minter CredentialMinter) *CredentialBroker {
	return &CredentialBroker{minter: minter}
}

// MintForSession validates the caller's session binding and mints a short-lived SCM
// credential. It rejects empty or unknown session bindings and never includes the
// minted secret in any returned error.
func (b *CredentialBroker) MintForSession(ctx context.Context, sessionID string) (ScmCredential, error) {
	if strings.TrimSpace(sessionID) == "" {
		return ScmCredential{}, fmt.Errorf("%w: session binding is empty", ErrInvalidCredentialRequest)
	}
	if b == nil || b.minter == nil {
		return ScmCredential{}, fmt.Errorf("%w: %w", ErrCredentialMintFailed, ErrNilCredentialMinter)
	}

	cred, err := b.minter.MintScmCredential(ctx, sessionID)
	if err != nil {
		if errors.Is(err, ErrNoActiveSession) {
			return ScmCredential{}, fmt.Errorf("%w: session %q is not active", ErrNoActiveSession, sessionID)
		}
		return ScmCredential{}, fmt.Errorf("%w: session %q", ErrCredentialMintFailed, sessionID)
	}
	if cred.Value == "" {
		return ScmCredential{}, fmt.Errorf("%w: minter returned an empty credential for session %q", ErrCredentialMintFailed, sessionID)
	}
	if cred.ExpiresAt.IsZero() {
		return ScmCredential{}, fmt.Errorf("%w: minter returned a credential without expiry for session %q", ErrCredentialMintFailed, sessionID)
	}
	return cred, nil
}
