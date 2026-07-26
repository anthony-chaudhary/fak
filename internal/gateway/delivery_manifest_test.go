package gateway

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/egressfloor"
)

// TestManifestDeliverySenderEndToEnd is #4392's live-path witness: it drives one outbound
// platform message end-to-end through a sender BUILT FROM A POLICY MANIFEST and a file-backed
// witness ledger — the two pieces #2850 deferred. It proves the promotion residual is closed:
// the allowlist is operator-configured (loaded from egress.delivery_allow, not code-baked),
// every decision persists a witness line to disk BEFORE send, a deny resolves to DELIVERY_BLOCK
// through the abi vocabulary, and only the redacted body of an ALLOWED message reaches the wire.
func TestManifestDeliverySenderEndToEnd(t *testing.T) {
	// Re-register the delivery reason vocabulary (a sibling test may have called
	// abi.ResetForTest, which wipes dynamically registered reasons). This is the same
	// contract gateway's init() wires in production.
	for _, pr := range egressfloor.DeliveryReasonPairs {
		abi.RegisterReason(pr.Code, pr.Name)
	}

	const manifest = `{
		"egress": {
			"delivery_allow": {
				"slack": ["C0OPS"],
				"telegram": ["*"]
			}
		}
	}`
	const slackTok = "xoxb-2411234567-abcdEFGH1234ijklMNOP"

	ledgerPath := filepath.Join(t.TempDir(), "delivery-witness.log")
	ledger := OpenDeliveryLedger(ledgerPath)

	type sent struct {
		platform egressfloor.Platform
		dest     string
		body     string
	}
	var sends []sent
	sender, err := NewManifestDeliverySender([]byte(manifest), ledger,
		func(p egressfloor.Platform, dest, body string) error {
			sends = append(sends, sent{p, dest, body})
			return nil
		})
	if err != nil {
		t.Fatalf("manifest sender construction failed: %v", err)
	}

	// 1) Allowed destination (loaded from the manifest) carrying a secret: it must reach the
	// transport exactly once, in redacted form.
	recOK, err := sender.Send(egressfloor.Delivery{
		Platform:    egressfloor.PlatformSlack,
		Destination: "C0OPS",
		Message:     "ship it " + slackTok,
	})
	if err != nil {
		t.Fatalf("allowed send errored: %v", err)
	}
	if !recOK.Allowed {
		t.Fatalf("manifest-allowlisted destination must be allowed, got %+v", recOK)
	}
	if len(sends) != 1 {
		t.Fatalf("allowed delivery must reach the transport once, got %d", len(sends))
	}
	if strings.Contains(sends[0].body, slackTok) {
		t.Fatalf("raw secret hit the wire: %q", sends[0].body)
	}
	if !strings.Contains(sends[0].body, "[REDACTED:slack-token]") {
		t.Fatalf("transport must receive the redacted body, got %q", sends[0].body)
	}

	// 2) Off-allowlist destination: denied, DELIVERY_BLOCK, never transported.
	recDeny, err := sender.Send(egressfloor.Delivery{
		Platform:    egressfloor.PlatformDiscord, // absent from the manifest
		Destination: "haxor",
		Message:     "leak " + slackTok,
	})
	if err != nil {
		t.Fatalf("a deny is a verdict, not a transport error: %v", err)
	}
	if recDeny.Allowed {
		t.Fatalf("off-manifest destination must be denied, got %+v", recDeny)
	}
	if len(sends) != 1 {
		t.Fatalf("denied delivery must NOT reach the transport, got %d total send(s)", len(sends))
	}
	if got := abi.ReasonName(recDeny.Reason); got != egressfloor.ReasonDeliveryBlockName {
		t.Fatalf("deny must resolve to %s in audit output, got %q", egressfloor.ReasonDeliveryBlockName, got)
	}

	// 3) The ledger persisted BOTH decisions, before send, with no raw secret.
	if err := ledger.Err(); err != nil {
		t.Fatalf("ledger persistence error: %v", err)
	}
	data, err := os.ReadFile(ledgerPath)
	if err != nil {
		t.Fatalf("witness ledger not written: %v", err)
	}
	lines := strings.Split(strings.TrimSpace(string(data)), "\n")
	if len(lines) != 2 {
		t.Fatalf("expected 2 persisted witness lines (allow+deny), got %d: %q", len(lines), string(data))
	}
	if !strings.Contains(lines[0], "verdict=allow") || !strings.Contains(lines[0], "platform=slack") {
		t.Fatalf("first witness line must record the allowed slack send, got %q", lines[0])
	}
	if !strings.Contains(lines[1], "verdict=deny") || !strings.Contains(lines[1], "reason=DELIVERY_BLOCK") {
		t.Fatalf("second witness line must record the DELIVERY_BLOCK deny, got %q", lines[1])
	}
	if strings.Contains(string(data), slackTok) {
		t.Fatalf("witness ledger leaked the raw secret: %q", string(data))
	}
}

// TestDeliveryPolicyFromManifest covers the manifest loader's edges: the deny-by-default
// posture for an absent/empty allowlist, a hard error on malformed JSON (an operator typo must
// surface, not silently deny everything), and a faithful platform->destination parse.
func TestDeliveryPolicyFromManifest(t *testing.T) {
	t.Run("absent or empty key is deny-by-default (permits nothing)", func(t *testing.T) {
		for _, raw := range []string{"", "{}", `{"egress":{}}`, `{"egress":{"delivery_allow":{}}}`} {
			p, err := DeliveryPolicyFromManifest([]byte(raw))
			if err != nil {
				t.Fatalf("%q: unexpected error: %v", raw, err)
			}
			rec := p.Adjudicate(egressfloor.Delivery{Platform: egressfloor.PlatformSlack, Destination: "C0OPS", Message: "hi"})
			if rec.Allowed {
				t.Fatalf("%q: empty allowlist must permit nothing, got allow", raw)
			}
		}
	})

	t.Run("malformed manifest is a hard error", func(t *testing.T) {
		if _, err := DeliveryPolicyFromManifest([]byte(`{"egress": {`)); err == nil {
			t.Fatal("malformed manifest must error, not silently deny everything")
		}
	})

	t.Run("parses platform -> destinations faithfully", func(t *testing.T) {
		p, err := DeliveryPolicyFromManifest([]byte(`{"egress":{"delivery_allow":{"telegram":["chat-7","*"]}}}`))
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		rec := p.Adjudicate(egressfloor.Delivery{Platform: egressfloor.PlatformTelegram, Destination: "anything", Message: "hi"})
		if !rec.Allowed {
			t.Fatalf("wildcard destination must be permitted, got %+v", rec)
		}
	})
}
