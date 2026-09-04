package verattest_test

import (
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/verattest"
)

func makePassingFixture() (verattest.Attestation, verattest.Expected) {
	now := time.Date(2026, 9, 3, 12, 0, 0, 0, time.UTC)
	generatedAt := now.Add(-10 * time.Minute)
	expiresAt := now.Add(24 * time.Hour)

	commitSHA := "0123456789abcdef0123456789abcdef01234567"
	configDigest := verattest.ComputeConfigDigest([]byte("build: true\ntests: affected\n"))

	att := verattest.New(
		commitSHA,
		configDigest,
		verattest.Producer{Tool: "fak validate", Version: "v0.45.0"},
		[]verattest.Step{
			{Name: "build", Status: verattest.StatusPass, ElapsedMS: 1250},
			{Name: "affected-tests", Status: verattest.StatusPass, ElapsedMS: 4320},
		},
		generatedAt,
		&expiresAt,
	)

	exp := verattest.Expected{
		CommitSHA:                commitSHA,
		VerificationConfigDigest: configDigest,
		RequiredSteps:            []string{"build", "affected-tests"},
		MaxAge:                   1 * time.Hour,
		Now:                      now,
	}

	return att, exp
}

func TestPassingAttestation(t *testing.T) {
	att, exp := makePassingFixture()

	res := verattest.Verify(att, exp)
	if res.Verdict != verattest.VerdictVerified {
		t.Fatalf("Verify() = %s (reason: %s), want %s", res.Verdict, res.Reason, verattest.VerdictVerified)
	}
	if !res.Verified() {
		t.Errorf("res.Verified() = false, want true")
	}

	verdict := verattest.Check(att, exp)
	if verdict != verattest.VerdictVerified {
		t.Errorf("Check() = %s, want %s", verdict, verattest.VerdictVerified)
	}
}

func TestCommitMismatch(t *testing.T) {
	att, exp := makePassingFixture()
	exp.CommitSHA = "deadbeefdeadbeefdeadbeefdeadbeefdeadbeef"

	res := verattest.Verify(att, exp)
	if res.Verdict != verattest.VerdictCommitMismatch {
		t.Errorf("Verify() = %s, want %s", res.Verdict, verattest.VerdictCommitMismatch)
	}
	if res.Verified() {
		t.Errorf("res.Verified() = true, want false")
	}
}

func TestCommitCaseInsensitive(t *testing.T) {
	att, exp := makePassingFixture()
	exp.CommitSHA = strings.ToUpper(att.CommitSHA)

	res := verattest.Verify(att, exp)
	if res.Verdict != verattest.VerdictVerified {
		t.Errorf("Verify() with uppercase expected SHA = %s, want %s", res.Verdict, verattest.VerdictVerified)
	}
}

func TestConfigMismatch(t *testing.T) {
	att, exp := makePassingFixture()
	exp.VerificationConfigDigest = "sha256:0000000000000000000000000000000000000000000000000000000000000000"

	res := verattest.Verify(att, exp)
	if res.Verdict != verattest.VerdictConfigMismatch {
		t.Errorf("Verify() = %s, want %s", res.Verdict, verattest.VerdictConfigMismatch)
	}
}

func TestIncompleteMissingStep(t *testing.T) {
	att, exp := makePassingFixture()
	exp.RequiredSteps = append(exp.RequiredSteps, "lint-check")

	res := verattest.Verify(att, exp)
	if res.Verdict != verattest.VerdictIncomplete {
		t.Errorf("Verify() = %s, want %s", res.Verdict, verattest.VerdictIncomplete)
	}
	if len(res.Violations) != 1 || res.Violations[0] != "lint-check" {
		t.Errorf("Violations = %v, want ['lint-check']", res.Violations)
	}
}

func TestIncompleteSkippedRequiredStep(t *testing.T) {
	att, exp := makePassingFixture()
	att.Steps[1].Status = verattest.StatusSkip

	res := verattest.Verify(att, exp)
	if res.Verdict != verattest.VerdictIncomplete {
		t.Errorf("Verify() with skipped required step = %s, want %s", res.Verdict, verattest.VerdictIncomplete)
	}
}

func TestStaleExpired(t *testing.T) {
	att, exp := makePassingFixture()
	expired := exp.Now.Add(-1 * time.Minute)
	att.ExpiresAt = &expired

	res := verattest.Verify(att, exp)
	if res.Verdict != verattest.VerdictStale {
		t.Errorf("Verify() with expired timestamp = %s, want %s", res.Verdict, verattest.VerdictStale)
	}
}

func TestStaleMaxAgeExceeded(t *testing.T) {
	att, exp := makePassingFixture()
	exp.MaxAge = 5 * time.Minute // attestation is 10 minutes old

	res := verattest.Verify(att, exp)
	if res.Verdict != verattest.VerdictStale {
		t.Errorf("Verify() with exceeded max_age = %s, want %s", res.Verdict, verattest.VerdictStale)
	}
}

func TestResultFailed(t *testing.T) {
	att, exp := makePassingFixture()
	att.Steps[1].Status = verattest.StatusFail
	att.Steps[1].Detail = "assertion failed: expected 42 got 0"

	res := verattest.Verify(att, exp)
	if res.Verdict != verattest.VerdictResultFailed {
		t.Errorf("Verify() with failed step = %s, want %s", res.Verdict, verattest.VerdictResultFailed)
	}
	if len(res.Violations) != 1 || res.Violations[0] != "affected-tests" {
		t.Errorf("Violations = %v, want ['affected-tests']", res.Violations)
	}
}

func TestMalformedVariations(t *testing.T) {
	tests := []struct {
		name   string
		mutate func(a *verattest.Attestation)
	}{
		{
			name: "wrong schema",
			mutate: func(a *verattest.Attestation) {
				a.Schema = "fak-verification-attestation/v2"
			},
		},
		{
			name: "empty schema",
			mutate: func(a *verattest.Attestation) {
				a.Schema = ""
			},
		},
		{
			name: "empty commit_sha",
			mutate: func(a *verattest.Attestation) {
				a.CommitSHA = "   "
			},
		},
		{
			name: "empty verification_config_digest",
			mutate: func(a *verattest.Attestation) {
				a.VerificationConfigDigest = ""
			},
		},
		{
			name: "empty producer tool",
			mutate: func(a *verattest.Attestation) {
				a.Producer.Tool = ""
			},
		},
		{
			name: "zero generated_at",
			mutate: func(a *verattest.Attestation) {
				a.GeneratedAt = time.Time{}
			},
		},
		{
			name: "empty steps",
			mutate: func(a *verattest.Attestation) {
				a.Steps = nil
			},
		},
		{
			name: "step with empty name",
			mutate: func(a *verattest.Attestation) {
				a.Steps = []verattest.Step{{Name: "", Status: verattest.StatusPass}}
			},
		},
		{
			name: "step with invalid status",
			mutate: func(a *verattest.Attestation) {
				a.Steps = []verattest.Step{{Name: "build", Status: "maybe"}}
			},
		},
		{
			name: "duplicate step name",
			mutate: func(a *verattest.Attestation) {
				a.Steps = []verattest.Step{
					{Name: "build", Status: verattest.StatusPass},
					{Name: "build", Status: verattest.StatusPass},
				}
			},
		},
		{
			name: "expiry precedes generation",
			mutate: func(a *verattest.Attestation) {
				badExpiry := a.GeneratedAt.Add(-1 * time.Hour)
				a.ExpiresAt = &badExpiry
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			att, exp := makePassingFixture()
			tc.mutate(&att)
			res := verattest.Verify(att, exp)
			if res.Verdict != verattest.VerdictMalformed {
				t.Errorf("Verify() = %s (reason: %s), want %s", res.Verdict, res.Reason, verattest.VerdictMalformed)
			}
		})
	}
}

func TestVerifyBytesValidAndInvalid(t *testing.T) {
	att, exp := makePassingFixture()
	raw, err := att.JSON()
	if err != nil {
		t.Fatalf("att.JSON() error: %v", err)
	}

	res := verattest.VerifyBytes(raw, exp)
	if res.Verdict != verattest.VerdictVerified {
		t.Errorf("VerifyBytes() = %s, want %s", res.Verdict, verattest.VerdictVerified)
	}

	verdict := verattest.CheckBytes(raw, exp)
	if verdict != verattest.VerdictVerified {
		t.Errorf("CheckBytes() = %s, want %s", verdict, verattest.VerdictVerified)
	}

	// Empty bytes
	resEmpty := verattest.VerifyBytes([]byte("   "), exp)
	if resEmpty.Verdict != verattest.VerdictMalformed {
		t.Errorf("VerifyBytes(empty) = %s, want %s", resEmpty.Verdict, verattest.VerdictMalformed)
	}

	// Invalid JSON
	resCorrupted := verattest.VerifyBytes([]byte("{not-valid-json"), exp)
	if resCorrupted.Verdict != verattest.VerdictMalformed {
		t.Errorf("VerifyBytes(corrupted) = %s, want %s", resCorrupted.Verdict, verattest.VerdictMalformed)
	}
}

func TestComputeConfigDigest(t *testing.T) {
	d1 := verattest.ComputeConfigDigest([]byte("content-a"))
	d2 := verattest.ComputeConfigDigest([]byte("content-a"))
	d3 := verattest.ComputeConfigDigest([]byte("content-b"))

	if !strings.HasPrefix(d1, "sha256:") {
		t.Errorf("digest %q missing sha256: prefix", d1)
	}
	if d1 != d2 {
		t.Errorf("identical content produced different digests: %q vs %q", d1, d2)
	}
	if d1 == d3 {
		t.Errorf("different content produced identical digest: %q", d1)
	}
}
