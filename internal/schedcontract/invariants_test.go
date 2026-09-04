package schedcontract

import (
	"testing"
	"time"
)

func TestPriorityRankingAndValidity(t *testing.T) {
	priorities := []Priority{
		PriorityBackground,
		PriorityLow,
		PriorityNormal,
		PriorityHigh,
		PriorityCritical,
	}

	for _, p := range priorities {
		if !p.Valid() {
			t.Errorf("expected priority %q to be valid", p)
		}
	}

	// Verify strictly monotonic ordering
	for i := 0; i < len(priorities)-1; i++ {
		curr := priorities[i]
		next := priorities[i+1]
		if curr.Rank() >= next.Rank() {
			t.Errorf("priority ranking violation: %s (rank %d) should be less than %s (rank %d)", curr, curr.Rank(), next, next.Rank())
		}
	}

	invalid := Priority("unrecognized-tier")
	if invalid.Valid() {
		t.Errorf("expected unrecognized priority to be invalid")
	}
	if invalid.Rank() != -1 {
		t.Errorf("expected invalid priority rank to be -1, got: %d", invalid.Rank())
	}
}

func TestExecutionTokenStructuralValidation(t *testing.T) {
	now := time.Now()

	validToken := ExecutionToken{
		TokenID:      "tok-100",
		Issuer:       "fak-kernel",
		Subject:      "worker-subagent",
		Lane:         "schedcontract",
		IssuedAt:     now.Add(-5 * time.Minute),
		ExpiresAt:    now.Add(15 * time.Minute),
		Capabilities: []string{"read", "write"},
		Signature:    "expected-sig-12345",
		Nonce:        "nonce-abc-789",
	}

	t.Run("valid token passes", func(t *testing.T) {
		if err := validToken.Validate(now); err != nil {
			t.Fatalf("expected valid token to pass, got: %v", err)
		}
	})

	t.Run("empty token ID fails", func(t *testing.T) {
		tok := validToken
		tok.TokenID = ""
		if err := tok.Validate(now); err == nil {
			t.Fatalf("expected error on empty token ID, got nil")
		}
	})

	t.Run("empty issuer fails", func(t *testing.T) {
		tok := validToken
		tok.Issuer = "  "
		if err := tok.Validate(now); err == nil {
			t.Fatalf("expected error on empty issuer, got nil")
		}
	})

	t.Run("empty subject fails", func(t *testing.T) {
		tok := validToken
		tok.Subject = ""
		if err := tok.Validate(now); err == nil {
			t.Fatalf("expected error on empty subject, got nil")
		}
	})

	t.Run("empty signature fails", func(t *testing.T) {
		tok := validToken
		tok.Signature = ""
		if err := tok.Validate(now); err == nil {
			t.Fatalf("expected error on empty signature, got nil")
		}
	})

	t.Run("empty nonce fails", func(t *testing.T) {
		tok := validToken
		tok.Nonce = "  "
		if err := tok.Validate(now); err == nil {
			t.Fatalf("expected error on empty nonce, got nil")
		}
	})

	t.Run("zero timestamps fail", func(t *testing.T) {
		tok := validToken
		tok.IssuedAt = time.Time{}
		if err := tok.Validate(now); err == nil {
			t.Fatalf("expected error on zero issued_at, got nil")
		}

		tok = validToken
		tok.ExpiresAt = time.Time{}
		if err := tok.Validate(now); err == nil {
			t.Fatalf("expected error on zero expires_at, got nil")
		}
	})

	t.Run("expiration before issuance fails", func(t *testing.T) {
		tok := validToken
		tok.ExpiresAt = tok.IssuedAt.Add(-1 * time.Second)
		if err := tok.Validate(now); err == nil {
			t.Fatalf("expected error when expires_at is before issued_at, got nil")
		}
	})

	t.Run("token presented before issuance fails", func(t *testing.T) {
		tok := validToken
		tok.IssuedAt = now.Add(5 * time.Minute)
		tok.ExpiresAt = now.Add(25 * time.Minute)
		if err := tok.Validate(now); err == nil {
			t.Fatalf("expected error when token is presented before issuance time, got nil")
		}
	})

	t.Run("token presented after expiration fails", func(t *testing.T) {
		tok := validToken
		tok.IssuedAt = now.Add(-20 * time.Minute)
		tok.ExpiresAt = now.Add(-5 * time.Minute)
		if err := tok.Validate(now); err == nil {
			t.Fatalf("expected error when token is expired, got nil")
		}
	})

	t.Run("has capability checks case insensitively", func(t *testing.T) {
		if !validToken.HasPermit("READ") {
			t.Errorf("expected case-insensitive match for READ")
		}
		if !validToken.HasPermit("write") {
			t.Errorf("expected match for write")
		}
		if validToken.HasPermit("delete") {
			t.Errorf("expected no match for delete")
		}
	})
}

func TestExecutionTokenCryptographicSignature(t *testing.T) {
	tok := ExecutionToken{
		Signature: "hmac-sha256-digest-987123",
	}

	if !tok.VerifySignature("hmac-sha256-digest-987123") {
		t.Errorf("expected matching signature to verify successfully")
	}

	if tok.VerifySignature("tampered-signature-digest") {
		t.Errorf("expected mismatched signature to fail verification")
	}

	if tok.VerifySignature("") {
		t.Errorf("expected empty expected signature to fail verification")
	}

	emptyTok := ExecutionToken{Signature: ""}
	if emptyTok.VerifySignature("some-sig") {
		t.Errorf("expected token with empty signature to fail verification")
	}
}
