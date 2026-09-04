package dosadapter

import (
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"strings"
	"testing"
	"time"
)

func TestValidateLease(t *testing.T) {
	tests := []struct {
		name    string
		req     LeaseRequest
		wantErr bool
		errIs   error
	}{
		{
			name: "valid exclusive lease",
			req: LeaseRequest{
				ID:       "req-1",
				Lane:     "gateway",
				LaneKind: "cluster",
				LockMode: LockModeExclusive,
				Tree:     []string{"internal/gateway/**"},
				WorkerID: "worker-1",
			},
			wantErr: false,
		},
		{
			name: "valid shared lease",
			req: LeaseRequest{
				ID:       "req-2",
				Lane:     "shared-docs",
				LaneKind: "docs",
				LockMode: LockModeShared,
				Tree:     []string{"docs/**"},
				WorkerID: "worker-2",
			},
			wantErr: false,
		},
		{
			name: "empty lock mode defaults to valid",
			req: LeaseRequest{
				ID:   "req-3",
				Lane: "core",
				Tree: []string{"internal/core/**"},
			},
			wantErr: false,
		},
		{
			name: "missing lease ID",
			req: LeaseRequest{
				ID:   "",
				Lane: "gateway",
				Tree: []string{"internal/gateway/**"},
			},
			wantErr: true,
			errIs:   ErrInvalidLease,
		},
		{
			name: "missing lane name",
			req: LeaseRequest{
				ID:   "req-4",
				Lane: "  ",
				Tree: []string{"internal/gateway/**"},
			},
			wantErr: true,
			errIs:   ErrInvalidLease,
		},
		{
			name: "empty tree",
			req: LeaseRequest{
				ID:   "req-5",
				Lane: "gateway",
				Tree: []string{},
			},
			wantErr: true,
			errIs:   ErrInvalidLease,
		},
		{
			name: "tree containing empty pattern",
			req: LeaseRequest{
				ID:   "req-6",
				Lane: "gateway",
				Tree: []string{"internal/gateway/**", " "},
			},
			wantErr: true,
			errIs:   ErrInvalidLease,
		},
		{
			name: "unsupported lock mode",
			req: LeaseRequest{
				ID:       "req-7",
				Lane:     "gateway",
				LockMode: "super-exclusive",
				Tree:     []string{"internal/gateway/**"},
			},
			wantErr: true,
			errIs:   ErrInvalidLease,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			err := ValidateLease(tc.req)
			if (err != nil) != tc.wantErr {
				t.Fatalf("ValidateLease() error = %v, wantErr = %v", err, tc.wantErr)
			}
			if tc.errIs != nil && !errors.Is(err, tc.errIs) {
				t.Errorf("ValidateLease() error = %v, want errors.Is %v", err, tc.errIs)
			}
		})
	}
}

func TestParseRefusal(t *testing.T) {
	tests := []struct {
		token        string
		wantToken    string
		wantCategory string
		wantRefusal  bool
	}{
		{
			token:        ReasonCollisionRisk,
			wantToken:    ReasonCollisionRisk,
			wantCategory: CategoryMisroute,
			wantRefusal:  true,
		},
		{
			token:        ReasonLaneDrained,
			wantToken:    ReasonLaneDrained,
			wantCategory: CategoryTrueDrain,
			wantRefusal:  true,
		},
		{
			token:        ReasonOperatorGate,
			wantToken:    ReasonOperatorGate,
			wantCategory: CategoryOperatorGate,
			wantRefusal:  true,
		},
		{
			token:        ReasonStaleClaim,
			wantToken:    ReasonStaleClaim,
			wantCategory: CategoryStaleClaim,
			wantRefusal:  true,
		},
		{
			token:        ReasonOffTrunk,
			wantToken:    ReasonOffTrunk,
			wantCategory: CategoryMisroute,
			wantRefusal:  true,
		},
		{
			token:        "collision_risk", // lowercase normalization
			wantToken:    ReasonCollisionRisk,
			wantCategory: CategoryMisroute,
			wantRefusal:  true,
		},
		{
			token:        "", // empty token fallback
			wantToken:    ReasonUnclassified,
			wantCategory: CategoryUnclassified,
			wantRefusal:  true,
		},
		{
			token:        "UNKNOWN_MYSTERY_TOKEN",
			wantToken:    "UNKNOWN_MYSTERY_TOKEN",
			wantCategory: CategoryUnclassified,
			wantRefusal:  true,
		},
	}

	for _, tc := range tests {
		t.Run("token_"+tc.token, func(t *testing.T) {
			r := ParseRefusal(tc.token)
			if r.Token != tc.wantToken {
				t.Errorf("ParseRefusal(%q).Token = %q, want %q", tc.token, r.Token, tc.wantToken)
			}
			if r.Category != tc.wantCategory {
				t.Errorf("ParseRefusal(%q).Category = %q, want %q", tc.token, r.Category, tc.wantCategory)
			}
			if r.Refusal != tc.wantRefusal {
				t.Errorf("ParseRefusal(%q).Refusal = %v, want %v", tc.token, r.Refusal, tc.wantRefusal)
			}
			if r.Description == "" {
				t.Errorf("ParseRefusal(%q).Description is unexpectedly empty", tc.token)
			}
			if r.Remedy == "" {
				t.Errorf("ParseRefusal(%q).Remedy is unexpectedly empty", tc.token)
			}
		})
	}
}

func TestDecisionWitness(t *testing.T) {
	now := time.Date(2026, 9, 3, 12, 0, 0, 0, time.UTC)
	h := sha256.Sum256([]byte("test-arbitration-decision-content"))
	validSHA := hex.EncodeToString(h[:])

	t.Run("valid witness passes verification", func(t *testing.T) {
		w := MintWitness("wit-1", validSHA, "fak/test-issuer", now, 15*time.Minute)
		if err := VerifyWitness(w); err != nil {
			t.Fatalf("VerifyWitness() error = %v, want nil", err)
		}
	})

	t.Run("empty witness ID rejected", func(t *testing.T) {
		w := MintWitness("", validSHA, "fak/test-issuer", now, 15*time.Minute)
		if err := VerifyWitness(w); !errors.Is(err, ErrCorruptedEvidence) {
			t.Errorf("VerifyWitness() error = %v, want ErrCorruptedEvidence", err)
		}
	})

	t.Run("invalid decision sha length", func(t *testing.T) {
		w := MintWitness("wit-2", "tooshort", "fak/test-issuer", now, 15*time.Minute)
		if err := VerifyWitness(w); !errors.Is(err, ErrCorruptedEvidence) {
			t.Errorf("VerifyWitness() error = %v, want ErrCorruptedEvidence", err)
		}
	})

	t.Run("non-hex decision sha", func(t *testing.T) {
		badHex := strings.Repeat("z", 64)
		w := MintWitness("wit-3", badHex, "fak/test-issuer", now, 15*time.Minute)
		if err := VerifyWitness(w); !errors.Is(err, ErrCorruptedEvidence) {
			t.Errorf("VerifyWitness() error = %v, want ErrCorruptedEvidence", err)
		}
	})

	t.Run("empty issuer rejected", func(t *testing.T) {
		w := MintWitness("wit-4", validSHA, "", now, 15*time.Minute)
		if err := VerifyWitness(w); !errors.Is(err, ErrCorruptedEvidence) {
			t.Errorf("VerifyWitness() error = %v, want ErrCorruptedEvidence", err)
		}
	})

	t.Run("zero timestamp rejected", func(t *testing.T) {
		w := MintWitness("wit-5", validSHA, "fak/test-issuer", time.Time{}, 15*time.Minute)
		if err := VerifyWitness(w); !errors.Is(err, ErrCorruptedEvidence) {
			t.Errorf("VerifyWitness() error = %v, want ErrCorruptedEvidence", err)
		}
	})

	t.Run("valid until before issued at", func(t *testing.T) {
		w := MintWitness("wit-6", validSHA, "fak/test-issuer", now, 15*time.Minute)
		w.ValidUntil = now.Add(-1 * time.Hour)
		// Re-signature to test the timestamp check specifically
		w.Signature = ComputeWitnessSignature(w.WitnessID, w.DecisionSHA, w.Issuer, w.IssuedAt.Unix())
		if err := VerifyWitness(w); !errors.Is(err, ErrCorruptedEvidence) {
			t.Errorf("VerifyWitness() error = %v, want ErrCorruptedEvidence", err)
		}
	})

	t.Run("signature tampering detected", func(t *testing.T) {
		w := MintWitness("wit-7", validSHA, "fak/test-issuer", now, 15*time.Minute)
		w.Signature = "corrupted-tampered-signature"
		if err := VerifyWitness(w); !errors.Is(err, ErrCorruptedEvidence) {
			t.Errorf("VerifyWitness() error = %v, want ErrCorruptedEvidence", err)
		}
	})

	t.Run("payload tampering detected", func(t *testing.T) {
		w := MintWitness("wit-8", validSHA, "fak/test-issuer", now, 15*time.Minute)
		// Tamper with decision sha while keeping old signature
		hOther := sha256.Sum256([]byte("different-content"))
		w.DecisionSHA = hex.EncodeToString(hOther[:])
		if err := VerifyWitness(w); !errors.Is(err, ErrCorruptedEvidence) {
			t.Errorf("VerifyWitness() error = %v, want ErrCorruptedEvidence", err)
		}
	})
}

func TestHandleFallback(t *testing.T) {
	req := LeaseRequest{
		ID:       "req-fb-1",
		Lane:     "gateway",
		LaneKind: "cluster",
		LockMode: LockModeExclusive,
		Tree:     []string{"internal/gateway/**"},
	}
	cause := errors.New("upstream connection reset")

	outcome := HandleFallback(req, cause)

	if outcome.Outcome != OutcomeRefuse {
		t.Errorf("HandleFallback outcome = %q, want %q", outcome.Outcome, OutcomeRefuse)
	}
	if outcome.Lane != "gateway" {
		t.Errorf("HandleFallback lane = %q, want gateway", outcome.Lane)
	}
	if outcome.Reason.Category != CategoryOperatorGate {
		t.Errorf("HandleFallback category = %q, want %q", outcome.Reason.Category, CategoryOperatorGate)
	}
	if !outcome.Reason.Refusal {
		t.Errorf("HandleFallback reason refusal = false, want true")
	}
	if err := VerifyWitness(outcome.Witness); err != nil {
		t.Errorf("HandleFallback witness failed verification: %v", err)
	}
	if !strings.Contains(outcome.Interpretation, "STOP") {
		t.Errorf("HandleFallback interpretation %q does not contain STOP", outcome.Interpretation)
	}
}

func TestAdapterClientArbitrate(t *testing.T) {
	fixedTime := time.Date(2026, 9, 3, 14, 0, 0, 0, time.UTC)
	client := NewAdapterClient(
		WithClock(func() time.Time { return fixedTime }),
		WithIssuer("fak/test-runner"),
	)

	// Register an existing active lease
	existing := LeaseRequest{
		ID:       "held-1",
		Lane:     "gateway",
		LaneKind: "cluster",
		LockMode: LockModeExclusive,
		Tree:     []string{"internal/gateway/**"},
		TTL:      30 * time.Minute,
	}
	if err := client.RegisterLease(existing); err != nil {
		t.Fatalf("RegisterLease() failed: %v", err)
	}

	t.Run("clean acquire on disjoint tree", func(t *testing.T) {
		req := LeaseRequest{
			ID:       "new-1",
			Lane:     "dosadapter",
			LaneKind: "leaf",
			LockMode: LockModeExclusive,
			Tree:     []string{"internal/dosadapter/**"},
			TTL:      15 * time.Minute,
		}
		outcome, err := client.Arbitrate(req)
		if err != nil {
			t.Fatalf("Arbitrate() unexpected error: %v", err)
		}
		if outcome.Outcome != OutcomeAcquire {
			t.Errorf("Arbitrate() outcome = %q, want %q", outcome.Outcome, OutcomeAcquire)
		}
		if outcome.Lane != "dosadapter" {
			t.Errorf("Arbitrate() lane = %q, want dosadapter", outcome.Lane)
		}
		if err := client.Verify(outcome.Witness); err != nil {
			t.Errorf("Arbitrate() witness invalid: %v", err)
		}
		if !strings.Contains(outcome.Interpretation, "GO") {
			t.Errorf("Arbitrate() interpretation %q does not advise GO", outcome.Interpretation)
		}
	})

	t.Run("refuse on collision with active exclusive lease", func(t *testing.T) {
		collidingReq := LeaseRequest{
			ID:       "new-2",
			Lane:     "gateway-edit",
			LaneKind: "cluster",
			LockMode: LockModeExclusive,
			Tree:     []string{"internal/gateway/mcp.go"},
			TTL:      15 * time.Minute,
		}
		outcome, err := client.Arbitrate(collidingReq)
		if err == nil {
			t.Fatalf("Arbitrate() expected refusal error, got nil")
		}
		if !errors.Is(err, ErrArbitrationRefused) {
			t.Errorf("Arbitrate() error = %v, want ErrArbitrationRefused", err)
		}
		if outcome.Outcome != OutcomeRefuse {
			t.Errorf("Arbitrate() outcome = %q, want %q", outcome.Outcome, OutcomeRefuse)
		}
		if outcome.Reason.Token != ReasonCollisionRisk {
			t.Errorf("Arbitrate() reason token = %q, want %q", outcome.Reason.Token, ReasonCollisionRisk)
		}
		if err := client.Verify(outcome.Witness); err != nil {
			t.Errorf("Arbitrate() witness invalid: %v", err)
		}
		if !strings.Contains(outcome.Interpretation, "STOP") {
			t.Errorf("Arbitrate() interpretation %q does not advise STOP", outcome.Interpretation)
		}
	})

	t.Run("invalid request returns ErrInvalidLease and fallback refuse", func(t *testing.T) {
		invalidReq := LeaseRequest{
			ID:   "",
			Lane: "broken",
			Tree: []string{"internal/broken/**"},
		}
		outcome, err := client.Arbitrate(invalidReq)
		if err == nil {
			t.Fatalf("Arbitrate() expected validation error, got nil")
		}
		if !errors.Is(err, ErrInvalidLease) {
			t.Errorf("Arbitrate() error = %v, want ErrInvalidLease", err)
		}
		if outcome.Outcome != OutcomeRefuse {
			t.Errorf("Arbitrate() outcome = %q, want %q", outcome.Outcome, OutcomeRefuse)
		}
	})
}

func TestAdapterClientLifecycle(t *testing.T) {
	client := NewAdapterClient()

	req := LeaseRequest{
		ID:       "lease-alpha",
		Lane:     "alpha",
		LockMode: LockModeExclusive,
		Tree:     []string{"internal/alpha/**"},
	}

	if err := client.RegisterLease(req); err != nil {
		t.Fatalf("RegisterLease() unexpected error: %v", err)
	}

	active := client.ActiveLeases()
	if len(active) != 1 || active[0].ID != "lease-alpha" {
		t.Fatalf("ActiveLeases() = %+v, want 1 lease with ID lease-alpha", active)
	}

	// Colliding registration rejected
	colliding := LeaseRequest{
		ID:       "lease-alpha-2",
		Lane:     "alpha-sub",
		LockMode: LockModeExclusive,
		Tree:     []string{"internal/alpha/sub/**"},
	}
	if err := client.RegisterLease(colliding); !errors.Is(err, ErrDisjointnessViolation) {
		t.Errorf("RegisterLease() error = %v, want ErrDisjointnessViolation", err)
	}

	// Release lease
	released := client.ReleaseLease("lease-alpha")
	if !released {
		t.Errorf("ReleaseLease(lease-alpha) = false, want true")
	}
	if len(client.ActiveLeases()) != 0 {
		t.Errorf("ActiveLeases() after release = %+v, want empty", client.ActiveLeases())
	}

	// Release non-existent returns false
	if client.ReleaseLease("non-existent") {
		t.Errorf("ReleaseLease(non-existent) = true, want false")
	}

	// Client helper methods
	if err := client.Validate(req); err != nil {
		t.Errorf("client.Validate() unexpected error: %v", err)
	}
	reason := client.Translate(ReasonCollisionRisk)
	if reason.Token != ReasonCollisionRisk {
		t.Errorf("client.Translate() token = %q, want %q", reason.Token, ReasonCollisionRisk)
	}
}

func BenchmarkArbitrationValidation(b *testing.B) {
	client := NewAdapterClient(
		WithInitialLeases([]LeaseRequest{
			{
				ID:       "held-gateway",
				Lane:     "gateway",
				LockMode: LockModeExclusive,
				Tree:     []string{"internal/gateway/**"},
			},
			{
				ID:       "held-model",
				Lane:     "model",
				LockMode: LockModeExclusive,
				Tree:     []string{"internal/model/**"},
			},
		}),
	)

	disjointReq := LeaseRequest{
		ID:       "bench-req",
		Lane:     "dosadapter",
		LaneKind: "leaf",
		LockMode: LockModeExclusive,
		Tree:     []string{"internal/dosadapter/**"},
		TTL:      10 * time.Minute,
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := client.Arbitrate(disjointReq)
		if err != nil {
			b.Fatalf("Arbitrate failed in benchmark: %v", err)
		}
	}
}
