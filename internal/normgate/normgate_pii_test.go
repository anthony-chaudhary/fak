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
