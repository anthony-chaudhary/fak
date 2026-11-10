package security

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"sync"
	"testing"
	"time"
)

// sentinelSecret is a distinctive value used to prove the broker never leaks a
// minted credential value through formatting or error strings.
const sentinelSecret = "SUPER-SECRET-TOKEN-abc123"

// stubMinter is a local CredentialMinter test double. Its behaviour is driven by
// the function field so each subtest can express a precise minter contract.
type stubMinter struct {
	mint func(ctx context.Context, sessionID string) (ScmCredential, error)
}

func (m stubMinter) MintScmCredential(ctx context.Context, sessionID string) (ScmCredential, error) {
	return m.mint(ctx, sessionID)
}

// validMinter returns a minter that yields a live credential for any session.
func validMinter(now time.Time) stubMinter {
	return stubMinter{mint: func(_ context.Context, sessionID string) (ScmCredential, error) {
		return ScmCredential{
			Scheme:    "https",
			Value:     sentinelSecret,
			ExpiresAt: now.Add(time.Hour),
		}, nil
	}}
}

// TestBrokerScmCredentials covers the credential-broker seam: happy-path minting,
// secrecy of the credential value, session-binding rejection, minter-failure
// propagation, and race-free concurrent use.
func TestBrokerScmCredentials(t *testing.T) {
	now := time.Date(2026, 9, 12, 12, 0, 0, 0, time.UTC)
	ctx := context.Background()

	t.Run("HappyPathShortLivedCredential", func(t *testing.T) {
		broker := NewCredentialBroker(validMinter(now))
		cred, err := broker.MintForSession(ctx, "session-happy")
		if err != nil {
			t.Fatalf("MintForSession returned error: %v", err)
		}
		if cred.Value == "" {
			t.Fatalf("minted credential has empty value")
		}
		if cred.ExpiresAt.IsZero() {
			t.Fatalf("minted credential has zero expiry")
		}
		if !cred.ExpiresAt.After(now) {
			t.Fatalf("expiry %v is not in the future relative to %v", cred.ExpiresAt, now)
		}
		if cred.Expired(now) {
			t.Fatalf("Expired(now) = true for a freshly minted credential")
		}
		if !cred.Expired(cred.ExpiresAt.Add(time.Second)) {
			t.Fatalf("Expired(after expiry) = false, want true")
		}
		if !cred.Expired(cred.ExpiresAt) {
			t.Fatalf("Expired(exact expiry) = false, want true (expiry boundary is inclusive)")
		}
		if !(ScmCredential{}).Expired(now) {
			t.Fatalf("zero-value credential Expired(now) = false, want true")
		}
	})

	t.Run("SecrecyNeverAppearsInFormattingOrErrors", func(t *testing.T) {
		broker := NewCredentialBroker(validMinter(now))

		cred, err := broker.MintForSession(ctx, "session-secret")
		if err != nil {
			t.Fatalf("MintForSession returned error: %v", err)
		}

		rendered := map[string]string{
			"%v cred":    fmt.Sprintf("%v", cred),
			"%+v cred":   fmt.Sprintf("%+v", cred),
			"%#v cred":   fmt.Sprintf("%#v", cred),
			"%s cred":    fmt.Sprintf("%s", cred),
			"String()":   cred.String(),
			"GoString()": cred.GoString(),
		}
		for name, out := range rendered {
			if strings.Contains(out, sentinelSecret) {
				t.Errorf("%s leaked the raw secret: %q", name, out)
			}
		}
		// The redacted rendering must still identify the credential's scheme.
		if !strings.Contains(cred.String(), "ScmCredential") {
			t.Errorf("String() %q does not identify the credential", cred.String())
		}
	})

	t.Run("SecrecyErrorStringsDoNotLeakSecret", func(t *testing.T) {
		// A minter whose failure message embeds the secret must not have that
		// secret propagated into the broker's returned error.
		leakyMinter := stubMinter{mint: func(_ context.Context, _ string) (ScmCredential, error) {
			return ScmCredential{}, fmt.Errorf("provider rejected token %s", sentinelSecret)
		}}
		broker := NewCredentialBroker(leakyMinter)

		_, err := broker.MintForSession(ctx, "session-leak")
		if err == nil {
			t.Fatalf("MintForSession returned nil error for failing minter")
		}
		if !errors.Is(err, ErrCredentialMintFailed) {
			t.Errorf("error = %v, want ErrCredentialMintFailed", err)
		}
		if strings.Contains(err.Error(), sentinelSecret) {
			t.Errorf("error string leaked the raw secret: %q", err.Error())
		}
		if s := fmt.Sprintf("%s", err); strings.Contains(s, sentinelSecret) {
			t.Errorf("fmt %%s error leaked the raw secret: %q", s)
		}
		if s := fmt.Sprintf("%+v", err); strings.Contains(s, sentinelSecret) {
			t.Errorf("fmt %%+v error leaked the raw secret: %q", s)
		}
	})

	t.Run("InvalidAndAbsentSessionBindingRejected", func(t *testing.T) {
		broker := NewCredentialBroker(validMinter(now))

		for _, sessionID := range []string{"", "   ", "\t\n"} {
			_, err := broker.MintForSession(ctx, sessionID)
			if !errors.Is(err, ErrInvalidCredentialRequest) {
				t.Errorf("MintForSession(%q) error = %v, want ErrInvalidCredentialRequest", sessionID, err)
			}
		}
	})

	t.Run("UnknownSessionRejected", func(t *testing.T) {
		unknownMinter := stubMinter{mint: func(_ context.Context, _ string) (ScmCredential, error) {
			return ScmCredential{}, ErrNoActiveSession
		}}
		broker := NewCredentialBroker(unknownMinter)

		_, err := broker.MintForSession(ctx, "session-unknown")
		if !errors.Is(err, ErrNoActiveSession) {
			t.Errorf("MintForSession unknown session error = %v, want ErrNoActiveSession", err)
		}
	})

	t.Run("MinterFailurePropagated", func(t *testing.T) {
		cases := []struct {
			name    string
			minter  stubMinter
			wantErr error
		}{
			{
				name: "generic minter error",
				minter: stubMinter{mint: func(_ context.Context, _ string) (ScmCredential, error) {
					return ScmCredential{}, errors.New("upstream exploded")
				}},
				wantErr: ErrCredentialMintFailed,
			},
			{
				name: "empty credential value",
				minter: stubMinter{mint: func(_ context.Context, _ string) (ScmCredential, error) {
					return ScmCredential{Scheme: "https", Value: "", ExpiresAt: now.Add(time.Hour)}, nil
				}},
				wantErr: ErrCredentialMintFailed,
			},
			{
				name: "missing expiry",
				minter: stubMinter{mint: func(_ context.Context, _ string) (ScmCredential, error) {
					return ScmCredential{Scheme: "https", Value: sentinelSecret}, nil
				}},
				wantErr: ErrCredentialMintFailed,
			},
		}

		for _, tc := range cases {
			t.Run(tc.name, func(t *testing.T) {
				broker := NewCredentialBroker(tc.minter)
				_, err := broker.MintForSession(ctx, "session-fail")
				if !errors.Is(err, tc.wantErr) {
					t.Errorf("error = %v, want %v", err, tc.wantErr)
				}
				if err != nil && strings.Contains(err.Error(), sentinelSecret) {
					t.Errorf("error leaked the raw secret: %q", err.Error())
				}
			})
		}
	})

	t.Run("NilMinterRejected", func(t *testing.T) {
		broker := NewCredentialBroker(nil)
		_, err := broker.MintForSession(ctx, "session-nil")
		if !errors.Is(err, ErrCredentialMintFailed) {
			t.Errorf("nil minter error = %v, want ErrCredentialMintFailed", err)
		}
		if !errors.Is(err, ErrNilCredentialMinter) {
			t.Errorf("nil minter error = %v, want wrapped ErrNilCredentialMinter", err)
		}
	})

	t.Run("ConcurrentMintZeroRace", func(t *testing.T) {
		const goroutines = 64
		broker := NewCredentialBroker(validMinter(now))

		var wg sync.WaitGroup
		wg.Add(goroutines)
		for i := 0; i < goroutines; i++ {
			go func(id int) {
				defer wg.Done()

				cred, err := broker.MintForSession(ctx, fmt.Sprintf("session-%d", id))
				if err != nil {
					t.Errorf("goroutine %d: MintForSession failed: %v", id, err)
					return
				}
				if cred.Expired(now) {
					t.Errorf("goroutine %d: credential already expired", id)
				}

				if _, err := broker.MintForSession(ctx, ""); !errors.Is(err, ErrInvalidCredentialRequest) {
					t.Errorf("goroutine %d: empty session error = %v", id, err)
				}
			}(i)
		}
		wg.Wait()
	})
}
