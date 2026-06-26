package policy

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/adjudicator"
)

// TestSecretPostureParses: a manifest's secret_posture resolves to the policy
// posture; an unknown token is refused at load (fail-loud).
func TestSecretPostureParses(t *testing.T) {
	p, err := Parse([]byte(`{"allow":["read_file"],"secret_posture":"fail_closed"}`))
	if err != nil {
		t.Fatalf("valid secret_posture failed to parse: %v", err)
	}
	if p.SecretPosture != adjudicator.SecretFailClosed {
		t.Errorf("SecretPosture = %v, want fail_closed", p.SecretPosture)
	}

	// Omitted -> the default (quarantine), today's behavior.
	d, err := Parse([]byte(`{"allow":["read_file"]}`))
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	if d.SecretPosture != adjudicator.SecretQuarantine {
		t.Errorf("omitted secret_posture = %v, want quarantine (default)", d.SecretPosture)
	}

	// An unknown posture is refused, not coerced.
	if _, err := Parse([]byte(`{"secret_posture":"open"}`)); err == nil {
		t.Error("unknown secret_posture must fail at load")
	}
}

// TestDeclaredSecretPatternsCompile: a declared pattern catches a shape the floor
// misses; an invalid regex fails LOUD at load, not at runtime.
func TestDeclaredSecretPatternsCompile(t *testing.T) {
	p, err := Parse([]byte(`{"secret_patterns":["ACME-[A-Z0-9]{8}"]}`))
	if err != nil {
		t.Fatalf("valid secret_patterns failed to parse: %v", err)
	}
	if !p.MatchesDeclaredSecret([]byte("creds ACME-AB12CD34 end")) {
		t.Error("declared pattern should catch the ACME-shaped token the floor misses")
	}

	// An invalid RE2 string must fail at LOAD.
	if _, err := Parse([]byte(`{"secret_patterns":["[unclosed"]}`)); err == nil {
		t.Error("an invalid declared secret pattern must fail at load")
	}
}

// TestSecretPostureRoundTrips: FromPolicy(p).ToPolicy() preserves the secret
// posture + declared patterns (the --dump round-trip), and the default omits both.
func TestSecretPostureRoundTrips(t *testing.T) {
	src, err := Parse([]byte(`{"secret_posture":"admit_and_log","secret_patterns":["ACME-[A-Z0-9]{8}"]}`))
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	got, err := FromPolicy(src).ToPolicy()
	if err != nil {
		t.Fatalf("round-trip: %v", err)
	}
	if got.SecretPosture != adjudicator.SecretAdmitAndLog {
		t.Errorf("round-trip posture = %v, want admit_and_log", got.SecretPosture)
	}
	if !got.MatchesDeclaredSecret([]byte("ACME-AB12CD34")) {
		t.Error("round-trip lost the declared pattern")
	}

	// The default policy dumps neither field.
	m := FromPolicy(adjudicator.DefaultPolicy())
	if m.SecretPosture != "" || len(m.SecretPatterns) != 0 {
		t.Errorf("default policy must omit secret fields, got posture=%q patterns=%v", m.SecretPosture, m.SecretPatterns)
	}
}
