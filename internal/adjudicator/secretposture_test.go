package adjudicator

import (
	"regexp"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// TestParseSecretPosture: the three tokens resolve, empty is the default
// (quarantine), and an unknown token is rejected (fail-loud, not coerced).
func TestParseSecretPosture(t *testing.T) {
	cases := map[string]struct {
		want SecretPosture
		ok   bool
	}{
		"":              {SecretQuarantine, true},
		"quarantine":    {SecretQuarantine, true},
		"fail_closed":   {SecretFailClosed, true},
		"admit_and_log": {SecretAdmitAndLog, true},
		"open":          {SecretQuarantine, false},
		"YOLO":          {SecretQuarantine, false},
	}
	for tok, exp := range cases {
		got, ok := ParseSecretPosture(tok)
		if got != exp.want || ok != exp.ok {
			t.Errorf("ParseSecretPosture(%q) = (%v,%v), want (%v,%v)", tok, got, ok, exp.want, exp.ok)
		}
	}
	// Round-trips through String for the known set.
	for _, p := range []SecretPosture{SecretQuarantine, SecretFailClosed, SecretAdmitAndLog} {
		if got, ok := ParseSecretPosture(p.String()); !ok || got != p {
			t.Errorf("round-trip %v -> %q -> (%v,%v)", p, p.String(), got, ok)
		}
	}
}

// TestSecretVerdict: each posture maps to the right verdict on a discovered secret.
func TestSecretVerdict(t *testing.T) {
	// Default / quarantine: hold the result out of context.
	q := Policy{SecretPosture: SecretQuarantine}.SecretVerdict("read_file")
	if q.Kind != abi.VerdictQuarantine || q.Reason != abi.ReasonSecretDiscovered {
		t.Errorf("quarantine posture verdict = %v/%s, want Quarantine/RESULT_SECRET_DISCOVERED", q.Kind, abi.ReasonName(q.Reason))
	}

	// fail_closed: a provable Deny on discovery.
	d := Policy{SecretPosture: SecretFailClosed}.SecretVerdict("read_file")
	if d.Kind != abi.VerdictDeny || d.Reason != abi.ReasonSecretDiscovered {
		t.Errorf("fail_closed verdict = %v/%s, want Deny/RESULT_SECRET_DISCOVERED", d.Kind, abi.ReasonName(d.Reason))
	}

	// admit_and_log on a READ-shaped tool: admit + record the would-deny.
	a := Policy{SecretPosture: SecretAdmitAndLog}.SecretVerdict("read_file")
	if a.Kind != abi.VerdictAllow {
		t.Fatalf("admit_and_log (read-shaped) verdict = %v, want Allow", a.Kind)
	}
	if a.Meta["posture"] != "admit_and_log" || a.Meta["would_deny"] != "RESULT_SECRET_DISCOVERED" {
		t.Errorf("admit_and_log must record the would-deny in Meta, got %v", a.Meta)
	}

	// admit_and_log on a NON-read-shaped tool: fail-safe back to quarantine — never
	// admit a secret from a write/exec-shaped result just because the posture is lenient.
	w := Policy{SecretPosture: SecretAdmitAndLog}.SecretVerdict("write_file")
	if w.Kind != abi.VerdictQuarantine {
		t.Errorf("admit_and_log (non-read-shaped) verdict = %v, want Quarantine (fail-safe)", w.Kind)
	}
}

// TestMatchesDeclaredSecret: a declared extra pattern catches a shape the floor
// would miss; an empty declared set matches nothing.
func TestMatchesDeclaredSecret(t *testing.T) {
	// An internal credential shape the canon floor does not enumerate.
	re := regexp.MustCompile(`ACME-[A-Z0-9]{8}`)
	p := Policy{SecretPatterns: []*regexp.Regexp{re}}
	if !p.MatchesDeclaredSecret([]byte("creds: ACME-AB12CD34 here")) {
		t.Error("declared pattern should catch the ACME-shaped token")
	}
	if p.MatchesDeclaredSecret([]byte("nothing secret here")) {
		t.Error("declared pattern matched a clean body")
	}
	if (Policy{}).MatchesDeclaredSecret([]byte("ACME-AB12CD34")) {
		t.Error("empty declared set must match nothing")
	}
}

// TestSecretPolicyAccessor: the accessor reflects a SetPolicy under the lock.
func TestSecretPolicyAccessor(t *testing.T) {
	a := New(DefaultPolicy())
	if p, pats := a.SecretPolicy(); p != SecretQuarantine || len(pats) != 0 {
		t.Fatalf("default secret policy = (%v, %d patterns), want (quarantine, 0)", p, len(pats))
	}
	a.SetPolicy(Policy{SecretPosture: SecretFailClosed, SecretPatterns: []*regexp.Regexp{regexp.MustCompile(`x`)}})
	if p, pats := a.SecretPolicy(); p != SecretFailClosed || len(pats) != 1 {
		t.Errorf("after SetPolicy: (%v, %d patterns), want (fail_closed, 1)", p, len(pats))
	}
}
