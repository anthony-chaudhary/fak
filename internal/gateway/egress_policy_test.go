package gateway

import (
	"errors"
	"strings"
	"testing"
)

// fak-test:runtime fast est=1s
func TestEgressPolicyDefaultDeny(t *testing.T) {
	pol, err := NewEgressPolicy(EgressPolicyConfig{})
	if err != nil {
		t.Fatalf("NewEgressPolicy(default): %v", err)
	}
	for _, target := range []string{"api.openai.com:443", "8.8.8.8:53", "[2001:db8::1]:443"} {
		dec, err := pol.Evaluate(target)
		if err != nil {
			t.Fatalf("Evaluate(%q) unexpected error: %v", target, err)
		}
		if dec.Allowed || dec.Action != EgressDeny || dec.MatchedRule != -1 {
			t.Fatalf("Evaluate(%q) = %+v, want default deny", target, dec)
		}
	}

	var nilPol *EgressPolicy
	if dec, err := nilPol.Evaluate("api.openai.com:443"); err != nil || dec.Allowed || dec.Action != EgressDeny {
		t.Fatalf("nil policy Evaluate = %+v, %v; want default deny", dec, err)
	}

	if _, err := NewEgressPolicy(EgressPolicyConfig{DefaultAction: EgressAllow}); err == nil {
		t.Fatal("non-deny DefaultAction must fail closed")
	}

	// Port-scoped allow rule must still deny unmatched ports and unspecified port 0.
	portPol, err := NewEgressPolicy(EgressPolicyConfig{
		Rules: []EgressRule{{Action: EgressAllow, Target: "api.openai.com", Ports: []uint16{443}}},
	})
	if err != nil {
		t.Fatalf("NewEgressPolicy(port): %v", err)
	}
	if dec, err := portPol.Evaluate("api.openai.com:443"); err != nil || !dec.Allowed {
		t.Fatalf("expected 443 allowed, got %+v, %v", dec, err)
	}
	if dec, err := portPol.Evaluate("api.openai.com:80"); err != nil || dec.Allowed {
		t.Fatalf("expected port 80 denied by default deny, got %+v, %v", dec, err)
	}
	if dec, err := portPol.Evaluate("api.openai.com"); err != nil || dec.Allowed {
		t.Fatalf("expected unspecified port denied for port-scoped rule, got %+v, %v", dec, err)
	}
}

// fak-test:runtime fast est=1s
func TestEgressPolicyCIDRAllowDenyPrecedence(t *testing.T) {
	pol, err := NewEgressPolicy(EgressPolicyConfig{
		Rules: []EgressRule{
			// Allow a single exception inside the denied /16, deny the rest of /16,
			// allow the remainder of /8, and allow an IPv6 prefix.
			{Action: EgressAllow, Target: "10.1.2.3/32"},
			{Action: EgressDeny, Target: "10.1.0.0/16"},
			{Action: EgressAllow, Target: "10.0.0.0/8"},
			{Action: EgressDeny, Target: "2001:db8:dead::/48"},
			{Action: EgressAllow, Target: "2001:db8::/32"},
		},
	})
	if err != nil {
		t.Fatalf("NewEgressPolicy: %v", err)
	}

	cases := []struct {
		target   string
		wantOK   bool
		wantRule int
	}{
		{"10.1.2.3:443", true, 0},
		{"10.1.5.9:443", false, 1},
		{"[::ffff:10.1.5.9]:443", false, 1}, // IPv4-mapped IPv6 must not bypass IPv4 deny CIDR
		{"10.2.3.4:443", true, 2},
		{"[2001:db8:dead::1]:443", false, 3},
		{"[2001:db8:cafe::1]:443", true, 4},
		{"192.0.2.1:443", false, -1},
	}
	for _, tc := range cases {
		dec, err := pol.Evaluate(tc.target)
		if err != nil {
			t.Fatalf("Evaluate(%q) err=%v", tc.target, err)
		}
		if dec.Allowed != tc.wantOK || dec.MatchedRule != tc.wantRule {
			t.Fatalf("Evaluate(%q) = allowed:%v rule:%d, want allowed:%v rule:%d (reason=%s)",
				tc.target, dec.Allowed, dec.MatchedRule, tc.wantOK, tc.wantRule, dec.Reason)
		}
	}
}

// fak-test:runtime fast est=1s
func TestEgressPolicyWildcardDomainMatch(t *testing.T) {
	pol, err := NewEgressPolicy(EgressPolicyConfig{
		Rules: []EgressRule{
			{Action: EgressDeny, Target: "blocked.openai.com"},
			{Action: EgressAllow, Target: "*.openai.com"},
		},
	})
	if err != nil {
		t.Fatalf("NewEgressPolicy: %v", err)
	}

	cases := []struct {
		target   string
		wantOK   bool
		wantHost string
	}{
		{"API.OpenAI.COM.:443", true, "api.openai.com"},
		{"v1.api.openai.com:443", true, "v1.api.openai.com"},
		{"blocked.openai.com:443", false, "blocked.openai.com"},
		{"openai.com:443", false, "openai.com"},         // wildcard *.openai.com must not match apex
		{"evilopenai.com:443", false, "evilopenai.com"}, // must respect dot boundary
	}
	for _, tc := range cases {
		dec, err := pol.Evaluate(tc.target)
		if err != nil {
			t.Fatalf("Evaluate(%q): %v", tc.target, err)
		}
		if dec.Allowed != tc.wantOK || dec.Destination.Host != tc.wantHost {
			t.Fatalf("Evaluate(%q) = allowed:%v host:%q, want allowed:%v host:%q",
				tc.target, dec.Allowed, dec.Destination.Host, tc.wantOK, tc.wantHost)
		}
	}

	for _, bad := range []string{"*", "*.", "*.*.openai.com", "a.*.openai.com", "*foo.openai.com", "api.openai.com:443"} {
		if _, err := NewEgressPolicy(EgressPolicyConfig{
			Rules: []EgressRule{{Action: EgressAllow, Target: bad}},
		}); err == nil {
			t.Fatalf("expected invalid rule target %q to be rejected", bad)
		}
	}
}

// fak-test:runtime fast est=1s
func TestEgressPolicySafeIPv6AndPortNormalization(t *testing.T) {
	d, err := ParseEgressDestination("[2001:0db8:0000:0000::1]:443")
	if err != nil {
		t.Fatalf("ParseEgressDestination IPv6+port: %v", err)
	}
	if !d.IsIP || d.Host != "2001:db8::1" || d.Port != 443 {
		t.Fatalf("unexpected IPv6 normalization: %+v", d)
	}

	d2, err := ParseEgressDestination("2001:0db8::2")
	if err != nil {
		t.Fatalf("ParseEgressDestination bare IPv6: %v", err)
	}
	if !d2.IsIP || d2.Host != "2001:db8::2" || d2.Port != 0 {
		t.Fatalf("unexpected bare IPv6 normalization: %+v", d2)
	}

	d3, err := ParseEgressDestination("[::ffff:127.0.0.1]:8443")
	if err != nil {
		t.Fatalf("ParseEgressDestination mapped IPv6: %v", err)
	}
	if !d3.IsIP || d3.Host != "127.0.0.1" || d3.Port != 8443 {
		t.Fatalf("expected IPv4-mapped IPv6 unmapped to 127.0.0.1:8443, got %+v", d3)
	}

	for _, invalid := range []string{
		"",
		"   ",
		"api.openai.com:0",
		"api.openai.com:65536",
		"api.openai.com:-1",
		"api.openai.com:https",
		"api.openai.com:",
		"[2001:db8::1]:",
		"[2001:db8::1",
		"[127.0.0.1]:443",
		"[example.com]:443",
		"fe80::1%eth0",
		"[fe80::1%eth0]:443",
		"api.openai.com@evil.com:443",
		"https://api.openai.com:443",
		"api.openai.com/v1/chat",
		"api.openai.com?x=1",
	} {
		if _, err := ParseEgressDestination(invalid); !errors.Is(err, ErrEgressInvalidTarget) {
			t.Fatalf("ParseEgressDestination(%q) err=%v, want ErrEgressInvalidTarget", invalid, err)
		}
	}
}

// fak-test:runtime fast est=1s
func TestEgressPolicyVaultSubstitutionAndRefusals(t *testing.T) {
	const rawSecret = "sk-live-super-secret-{{not_reexpanded}}"
	vault, err := NewEgressCredentialVault(
		VaultCredential{
			Name:         "openai_key",
			Placeholder:  "{{OPENAI_KEY}}",
			Secret:       rawSecret,
			AllowedHosts: []string{"api.openai.com", "*.trusted-proxy.internal", "10.20.0.0/16"},
		},
	)
	if err != nil {
		t.Fatalf("NewEgressCredentialVault: %v", err)
	}

	// String representation must never leak Secret.
	credStr := (VaultCredential{Name: "openai_key", Secret: rawSecret, AllowedHosts: []string{"api.openai.com"}}).String()
	if strings.Contains(credStr, rawSecret) {
		t.Fatalf("VaultCredential.String leaked secret: %q", credStr)
	}

	pol, err := NewEgressPolicy(EgressPolicyConfig{
		Rules: []EgressRule{
			{Action: EgressAllow, Target: "api.openai.com", Ports: []uint16{443}},
			{Action: EgressAllow, Target: "evil.example.com", Ports: []uint16{443}},
			{Action: EgressAllow, Target: "10.20.1.5", Ports: []uint16{443}},
		},
	})
	if err != nil {
		t.Fatalf("NewEgressPolicy: %v", err)
	}
	boundary := &EgressBoundary{Policy: pol, Vault: vault}

	// Authorized host substitutes placeholder cleanly without re-expanding secret contents.
	dec, hdrs, err := boundary.AuthorizeAndSubstituteHeaders("api.openai.com:443", map[string]string{
		"Authorization": "Bearer {{OPENAI_KEY}}",
		"Content-Type":  "application/json",
	})
	if err != nil || !dec.Allowed {
		t.Fatalf("authorized substitution failed: dec=%+v err=%v", dec, err)
	}
	if hdrs["Authorization"] != "Bearer "+rawSecret {
		t.Fatalf("Authorization = %q, want substituted secret", hdrs["Authorization"])
	}

	// IPv4-mapped IPv6 within allowed CIDR succeeds.
	_, hdrsIP, err := boundary.AuthorizeAndSubstituteHeaders("[::ffff:10.20.1.5]:443", map[string]string{
		"X-Api-Key": "{{OPENAI_KEY}}",
	})
	if err != nil || hdrsIP["X-Api-Key"] != rawSecret {
		t.Fatalf("CIDR-authorized IPv6-mapped substitution failed: %+v, %v", hdrsIP, err)
	}

	// Missing placeholder fails closed even on authorized host.
	if _, _, err := boundary.AuthorizeAndSubstituteHeaders("api.openai.com:443", map[string]string{
		"Authorization": "Bearer {{OPENAI_KEY}}",
		"X-Backup":      "Bearer {{MISSING_KEY}}",
	}); !errors.Is(err, ErrVaultPlaceholderMissing) {
		t.Fatalf("expected ErrVaultPlaceholderMissing, got %v", err)
	}

	// Disallowed host fails closed even when egress policy allows that host, without leaking secret.
	if _, _, err := boundary.AuthorizeAndSubstituteHeaders("evil.example.com:443", map[string]string{
		"Authorization": "Bearer {{OPENAI_KEY}}",
	}); !errors.Is(err, ErrVaultHostDisallowed) || strings.Contains(err.Error(), rawSecret) {
		t.Fatalf("expected ErrVaultHostDisallowed without secret leak, got %v", err)
	}

	// Disallowed egress destination fails at policy stage before vault lookup.
	if _, _, err := boundary.AuthorizeAndSubstituteHeaders("blocked.example.com:443", map[string]string{
		"Authorization": "Bearer {{OPENAI_KEY}}",
	}); !errors.Is(err, ErrEgressDenied) {
		t.Fatalf("expected ErrEgressDenied, got %v", err)
	}

	// Invalid host/port fails closed without touching secrets.
	if _, _, err := boundary.AuthorizeAndSubstituteHeaders("api.openai.com:99999", map[string]string{
		"Authorization": "Bearer {{OPENAI_KEY}}",
	}); !errors.Is(err, ErrEgressInvalidTarget) {
		t.Fatalf("expected ErrEgressInvalidTarget, got %v", err)
	}

	// Vault constructor rejects empty secret or missing AllowedHosts.
	if _, err := NewEgressCredentialVault(VaultCredential{Name: "empty", Secret: "", AllowedHosts: []string{"api.openai.com"}}); err == nil {
		t.Fatal("expected error for empty Secret")
	}
	if _, err := NewEgressCredentialVault(VaultCredential{Name: "unscoped", Secret: "s3cr3t"}); err == nil {
		t.Fatal("expected error for empty AllowedHosts")
	}
}
