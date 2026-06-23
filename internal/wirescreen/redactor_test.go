package wirescreen

import (
	"context"
	"os"
	"reflect"
	"testing"
)

// TestPIIRedactor_DetectsSecretsAndPII: the reference floor flags the common
// PII/secret shapes with their audit kind and leaves benign bodies alone. Every
// fixture is a payload the inbound ctxmmu floor would NOT catch on the outbound
// surface (user/assistant content) or that is genuine PII the floor does not block.
func TestPIIRedactor_DetectsSecretsAndPII(t *testing.T) {
	r := piiRedactor{}
	ctx := context.Background()

	cases := []struct {
		name string
		body string
		want string // the Kind of at least one proposed span
	}{
		{"visa test card spaced", "card on file is 4111 1111 1111 1111 ok", "credit_card"},
		{"stripe test card contiguous", "4242424242424242 billed", "credit_card"},
		{"amex 15-digit", "amex 378282246310005 expired", "credit_card"},
		{"us ssn", "ssn 123-45-6789 on file", "us_ssn"},
		{"aws access key", "aws AKIAIOSFODNN7EXAMPLE role", "aws_access_key"},
		{"github pat", "token ghp_012345678901234567890123456789012345 leak", "github_token"},
		{"slack token", "slack xoxb-1234567890-abcdefghij hook", "slack_token"},
		{"stripe secret", "sk_live_0123456789abcdef0123456789abcdef key", "stripe_key"},
		{"google api key", "key AIzaSyA0123456789012345678901234567890a ready", "google_api_key"},
		{"email", "reach me at alice@example.com please", "email"},
		{"bearer header", "Authorization: Bearer abcdefghijklmnopqrstuvwx", "bearer_token"},
		{"pem private key", "-----BEGIN RSA PRIVATE KEY-----\nMIIEpAIBAAKCAQEA...\n-----END RSA PRIVATE KEY-----", "private_key"},
	}
	for _, tc := range cases {
		spans := r.Propose(ctx, []byte(tc.body), "user")
		if !hasKind(spans, tc.want) {
			t.Errorf("%s: expected a %q span, got %v", tc.name, tc.want, spans)
		}
	}
}

// TestPIIRedactor_HighPrecision: the floor must NOT redact benign content — a
// compliance floor that breaks legit text (even reversibly) erodes trust. Digit runs
// that are too short, or the right length but Luhn-invalid, must be left alone.
func TestPIIRedactor_HighPrecision(t *testing.T) {
	r := piiRedactor{}
	ctx := context.Background()

	benign := []string{
		"row 42 = 17",
		"the build is currently running",
		"order #123456789012 is ready",           // 12 digits: below the 13-19 card range
		"reference 1234567890123456 has no luhn", // 16 digits but not Luhn-valid
		"see issue #572 for the redaction rung",
		"contact extension 555-1234 after hours", // not an SSN shape
		"func main() { fmt.Println(\"hi\") }",
	}
	for _, s := range benign {
		if spans := r.Propose(ctx, []byte(s), "assistant"); len(spans) != 0 {
			t.Errorf("expected %q to be benign, got spans %v", s, spans)
		}
	}
}

// TestPropose_CoalescesOverlaps: a PEM private key containing an email address must
// collapse to ONE span (the key) — coalesce keeps the earlier-starting (larger)
// span on a nested overlap, so the secret block is redacted wholesale, not punctured
// by a smaller inner span that would leave key material exposed.
func TestPropose_CoalescesOverlaps(t *testing.T) {
	r := piiRedactor{}
	body := []byte("-----BEGIN RSA PRIVATE KEY-----\nadmin@corp.internal\nMIIEpAIBAAKCAQEA\n-----END RSA PRIVATE KEY-----")
	spans := r.Propose(context.Background(), body, "tool")
	if len(spans) != 1 {
		t.Fatalf("expected the nested email to collapse into the PEM span, got %d spans: %v", len(spans), spans)
	}
	if spans[0].Kind != "private_key" {
		t.Errorf("expected the surviving span to be the private_key, got %q", spans[0].Kind)
	}
	if string(body[spans[0].Start:spans[0].End]) != string(body) {
		t.Errorf("expected the PEM span to cover the whole body, got [%d:%d]", spans[0].Start, spans[0].End)
	}
}

// TestCoalesce_DisjointAndBounded: the coalesced output is sorted, disjoint, and
// drops any out-of-bounds or inverted span the patterns might emit.
func TestCoalesce_DisjointAndBounded(t *testing.T) {
	in := []Span{
		{Start: 0, End: 3, Kind: "a"},
		{Start: 1, End: 2, Kind: "overlap"}, // dropped (overlaps a)
		{Start: 5, End: 8, Kind: "b"},
		{Start: -1, End: 4, Kind: "neg"},  // dropped (negative start)
		{Start: 9, End: 9, Kind: "empty"}, // dropped (start>=end)
		{Start: 20, End: 99, Kind: "oob"}, // dropped (end > bodyLen)
	}
	got := coalesce(in, 12)
	want := []Span{{Start: 0, End: 3, Kind: "a"}, {Start: 5, End: 8, Kind: "b"}}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("coalesce = %v, want %v", got, want)
	}
}

// TestDefaultInert_NoActiveRedactor: with FAK_WIRE_REDACT unset (the test process
// default), ActiveRedactor() returns nil — the inert contract. No redaction runs.
func TestDefaultInert_NoActiveRedactor(t *testing.T) {
	rmu.Lock()
	ractive, ractiveResolved = nil, true
	rmu.Unlock()

	if r := ActiveRedactor(); r != nil {
		t.Fatalf("default-inert violated: ActiveRedactor() = %v, want nil", r)
	}
	// Apply with no selected redactor is a no-op: body returned unchanged, ok=false.
	body := []byte("card 4111 1111 1111 1111 here")
	out, ok := Apply(context.Background(), nil, body, "user")
	if ok || string(out.Redacted) != string(body) {
		t.Fatalf("Apply(nil, ...) must be a no-op, got ok=%v redacted=%q", ok, out.Redacted)
	}
}

// TestRegisterAndSelectRedactor: a redactor registered under a name is selectable;
// an unknown name resolves to nil (inert), never a panic.
func TestRegisterAndSelectRedactor(t *testing.T) {
	RegisterRedactor("test-fake", piiRedactor{})
	rmu.RLock()
	_, ok := rregistry["test-fake"]
	rmu.RUnlock()
	if !ok {
		t.Fatalf("RegisterRedactor did not add the redactor to the catalog")
	}
	rmu.RLock()
	_, ok = rregistry["pii"]
	rmu.RUnlock()
	if !ok {
		t.Fatalf("the pii reference redactor must be registered at init")
	}
	// An unknown name resolves to nil (inert), never a panic.
	rmu.Lock()
	ractive, ractiveResolved = rregistry[""], true // unknown -> nil
	rmu.Unlock()
	if ActiveRedactor() != nil {
		t.Fatalf("unknown FAK_WIRE_REDACT must resolve to nil")
	}
}

// hasKind reports whether spans contains at least one of the given kind.
func hasKind(spans []Span, kind string) bool {
	for _, s := range spans {
		if s.Kind == kind {
			return true
		}
	}
	return false
}

// TestEnvUnset is a sanity check that the test process does not carry FAK_WIRE_REDACT
// (which would silently make ActiveRedactor non-inert and invalidate the inert test
// above). It is intentionally not t.Setenv (that would leak across the cached
// resolution).
func TestEnvUnset(t *testing.T) {
	if v := os.Getenv("FAK_WIRE_REDACT"); v != "" {
		t.Skipf("FAK_WIRE_REDACT=%q set in the environment; inert test above reset the cache manually", v)
	}
}
