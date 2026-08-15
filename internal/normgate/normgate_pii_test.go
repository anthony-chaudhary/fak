package normgate_test

import (
	"context"
	"encoding/base64"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	_ "github.com/anthony-chaudhary/fak/internal/blob" // registers the blob PageOut/Resolver backend
	"github.com/anthony-chaudhary/fak/internal/normgate"
)

// piiBodies are the deterministic general-PII families (#5378). Each carries one needle in
// otherwise-legitimate output, so the warn-first mask must keep the surrounding text.
var piiBodies = map[string]struct {
	body   string
	needle string
}{
	"email": {`{"owner":"alice.customer@acme.example","plan":"pro"}`, "alice.customer@acme.example"},
	"ssn":   {`customer SSN 123-45-6789 verified`, "123-45-6789"},
	"phone": {`callback number +1 415-555-0132 on file`, "415-555-0132"},
	"pan":   {`payment card 4111 1111 1111 1111 charged`, "4111 1111 1111 1111"},
	"iban":  {`wire to GB82WEST12345698765432 today`, "GB82WEST12345698765432"},
}

// WARN-FIRST DEFAULT, covering BOTH served directions: a PII-bearing tool result is MASKED
// IN PLACE (PII_REDACTED Transform) — the needle is removed but the surrounding output stays
// in context. normgate's PII policy does not branch on provenance, so the SAME mask applies
// to an untrusted-egress result (outbound, provider->client) and a trusted-local read
// (inbound, client->provider) — this test asserts both tool sources to pin direction
// independence, the #5378 "inbound and outbound" DoD.
func TestPIIRedactedByDefaultBothDirections(t *testing.T) {
	ctx := context.Background()
	// "read_webpage" is untrusted egress (outbound content); "Read" is trusted-local
	// (inbound, the agent's own read). Both must mask PII in place.
	for _, tool := range []string{"read_webpage", "Read"} {
		for name, tc := range piiBodies {
			g := normgate.New()
			r := result(tc.body)
			v := g.Admit(ctx, untrusted(tool), r)
			if v.Kind != abi.VerdictTransform || v.Reason != abi.ReasonPIIRedacted {
				t.Errorf("%s/%s: want Transform/PII_REDACTED (warn-first default), got %v/%s",
					tool, name, v.Kind, abi.ReasonName(v.Reason))
				continue
			}
			after := resolve(t, ctx, r.Payload)
			if strings.Contains(after, tc.needle) {
				t.Errorf("%s/%s: PII needle leaked into context after mask: %q", tool, name, after)
			}
		}
	}
}

// OPT-IN fail_closed posture: the SAME PII bodies SEAL the whole result (PII_EXFIL) — the
// strict path a residency/privacy-conscious operator turns on, shared with the secret
// posture. Covers both directions for parity with the default case.
func TestPIISealsUnderFailClosedBothDirections(t *testing.T) {
	sealPolicy(t)
	ctx := context.Background()
	for _, tool := range []string{"read_webpage", "Read"} {
		for name, tc := range piiBodies {
			g := normgate.New()
			r := result(tc.body)
			v := g.Admit(ctx, untrusted(tool), r)
			if v.Kind != abi.VerdictQuarantine || v.Reason != abi.ReasonPIIExfil {
				t.Errorf("%s/%s: want Quarantine/PII_EXFIL under fail_closed, got %v/%s",
					tool, name, v.Kind, abi.ReasonName(v.Reason))
				continue
			}
			after := resolve(t, ctx, r.Payload)
			if strings.Contains(after, tc.needle) {
				t.Errorf("%s/%s: sealed PII needle leaked into the stub: %q", tool, name, after)
			}
		}
	}
}

// An OBFUSCATED PII needle (caught only on a de-obfuscated view — here base64) SEALS even
// under the permissive default: the in-place redactor cannot reach a needle with no raw
// span, exactly like the obfuscated-secret case.
func TestObfuscatedPIISealsEvenByDefault(t *testing.T) {
	ctx := context.Background()
	g := normgate.New()
	r := result("decode: " + base64.StdEncoding.EncodeToString([]byte("carol.customer@acme.example")))
	v := g.Admit(ctx, untrusted("read_webpage"), r)
	if v.Kind != abi.VerdictQuarantine || v.Reason != abi.ReasonPIIExfil {
		t.Errorf("obfuscated PII: want Quarantine/PII_EXFIL, got %v/%s", v.Kind, abi.ReasonName(v.Reason))
	}
}

// A clean result with no PII (and no secret/injection) is DEFERRED untouched — the PII rung
// never fires on benign output, so it composes in front of ctxmmu without stealing clean
// results.
func TestNoPIIDefers(t *testing.T) {
	ctx := context.Background()
	g := normgate.New()
	r := result(`{"weather":"sunny","temp":72,"city":"Portland"}`)
	v := g.Admit(ctx, untrusted("read_webpage"), r)
	if v.Kind != abi.VerdictDefer {
		t.Errorf("clean result: want Defer, got %v/%s", v.Kind, abi.ReasonName(v.Reason))
	}
}

func TestPublicEmailContextAdmitsEmailUnchanged(t *testing.T) {
	t.Setenv("FAK_SECRET_POSTURE", "warn")
	body := "contacts: hiring@example.com and jobs@example.org"
	r := result(body)
	call := &abi.ToolCall{Meta: map[string]string{"fak.pii.public_classes": "email"}}
	v := normgate.New().Admit(context.Background(), call, r)
	if v.Kind != abi.VerdictAllow {
		t.Fatalf("public email context: want Allow, got %v/%v", v.Kind, v.Reason)
	}
	if got := string(resolve(t, context.Background(), r.Payload)); got != body {
		t.Fatalf("public email was rewritten into garbage: got %q want %q", got, body)
	}
}

func TestPublicEmailContextStillMasksProtectedPII(t *testing.T) {
	t.Setenv("FAK_SECRET_POSTURE", "warn")
	body := "contact hiring@example.com; SSN 123-45-6789"
	r := result(body)
	call := &abi.ToolCall{Meta: map[string]string{"fak.pii.public_classes": "email"}}
	v := normgate.New().Admit(context.Background(), call, r)
	if v.Kind != abi.VerdictTransform || v.Reason != abi.ReasonPIIRedacted {
		t.Fatalf("mixed classes: want Transform/PII_REDACTED, got %v/%v", v.Kind, v.Reason)
	}
	got := string(resolve(t, context.Background(), r.Payload))
	if !strings.Contains(got, "hiring@example.com") || strings.Contains(got, "123-45-6789") {
		t.Fatalf("class-scoped result wrong: %q", got)
	}
}

func TestPublicEmailContextCannotHideObfuscatedProtectedPII(t *testing.T) {
	t.Setenv("FAK_SECRET_POSTURE", "warn")
	// A raw email is public, but a second email exists only in a decoded view. Because
	// it has no exact raw span, the exemption cannot prove a safe edit and must seal.
	r := result("contact hiring@example.com; encoded dXNlckBleGFtcGxlLmNvbQ==")
	call := &abi.ToolCall{Meta: map[string]string{"fak.pii.public_classes": "email"}}
	v := normgate.New().Admit(context.Background(), call, r)
	if v.Kind != abi.VerdictQuarantine || v.Reason != abi.ReasonPIIExfil {
		t.Fatalf("mixed obfuscated class: want Quarantine/PII_EXFIL, got %v/%v", v.Kind, v.Reason)
	}
}

func TestResultCannotSelfDeclarePublicPII(t *testing.T) {
	t.Setenv("FAK_SECRET_POSTURE", "warn")
	r := result("contact hiring@example.com")
	r.Meta = map[string]string{"fak.pii.public_classes": "email"}
	v := normgate.New().Admit(context.Background(), &abi.ToolCall{}, r)
	if v.Kind != abi.VerdictTransform || v.Reason != abi.ReasonPIIRedacted {
		t.Fatalf("result-authored exemption must be ignored, got %v/%v", v.Kind, v.Reason)
	}
}

func TestUnknownPublicPIIClassFailsClosed(t *testing.T) {
	t.Setenv("FAK_SECRET_POSTURE", "warn")
	for _, declaration := range []string{"all", "emails", "email,all", ",email"} {
		r := result("contact hiring@example.com")
		call := &abi.ToolCall{Meta: map[string]string{"fak.pii.public_classes": declaration}}
		v := normgate.New().Admit(context.Background(), call, r)
		if v.Kind != abi.VerdictTransform || v.Reason != abi.ReasonPIIRedacted {
			t.Errorf("declaration %q: want default redaction, got %v/%v", declaration, v.Kind, v.Reason)
		}
	}
}
