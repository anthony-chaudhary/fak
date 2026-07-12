package gateway

import (
	"errors"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/egressfloor"
)

// TestDeliverySenderGatesOnFloor is #2850's send-path contract: an outbound platform
// message is transmitted ONLY when the egress capability floor allows it, ONLY in redacted
// form, and ALWAYS produces a witness line BEFORE the send — the live-seam half of the leaf
// whose pure floor lives in internal/egressfloor. It proves the gate closes the residual the
// pure classifier could not close alone: nothing gated the actual transmit on the verdict.
func TestDeliverySenderGatesOnFloor(t *testing.T) {
	policy := egressfloor.DeliveryPolicy{Allow: map[egressfloor.Platform][]string{
		egressfloor.PlatformSlack:    {"C0OPS"},
		egressfloor.PlatformTelegram: {egressfloor.AnyDestination},
	}}
	const slackTok = "xoxb-2411234567-abcdEFGH1234ijklMNOP"

	type sent struct {
		platform egressfloor.Platform
		dest     string
		body     string
	}
	// newSender builds a DeliverySender with a capturing transport (its send error is
	// injectable) and a capturing witness sink, returning handles to both capture slices.
	newSender := func(transportErr error) (DeliverySender, *[]sent, *[]string) {
		sends := &[]sent{}
		witnessed := &[]string{}
		s := DeliverySender{
			Policy: policy,
			Transport: func(p egressfloor.Platform, dest, body string) error {
				*sends = append(*sends, sent{p, dest, body})
				return transportErr
			},
			Witness: func(line string) { *witnessed = append(*witnessed, line) },
		}
		return s, sends, witnessed
	}

	t.Run("denied destination is never handed to the transport", func(t *testing.T) {
		s, sends, witnessed := newSender(nil)
		rec, err := s.Send(egressfloor.Delivery{
			Platform:    egressfloor.PlatformDiscord, // not on the allowlist
			Destination: "123",
			Message:     "here is my key " + slackTok,
		})
		if err != nil {
			t.Fatalf("a deny is a verdict, not a transport error: %v", err)
		}
		if rec.Allowed {
			t.Fatalf("off-allowlist platform must be denied, got allow: %+v", rec)
		}
		if len(*sends) != 0 {
			t.Fatalf("denied delivery must NOT reach the transport, got %d send(s)", len(*sends))
		}
		if len(*witnessed) != 1 || !strings.Contains((*witnessed)[0], "verdict=deny") {
			t.Fatalf("a deny must still be witnessed, got %v", *witnessed)
		}
		if strings.Contains((*witnessed)[0], slackTok) {
			t.Fatalf("witness leaked the raw secret: %q", (*witnessed)[0])
		}
	})

	t.Run("allowed delivery transmits only the redacted body, witnessed first", func(t *testing.T) {
		s, sends, witnessed := newSender(nil)
		rec, err := s.Send(egressfloor.Delivery{
			Platform:    egressfloor.PlatformSlack,
			Destination: "C0OPS",
			Message:     "ship it, token " + slackTok + " fyi",
		})
		if err != nil {
			t.Fatalf("unexpected transport error: %v", err)
		}
		if !rec.Allowed {
			t.Fatalf("allowlisted destination must be allowed, got deny: %+v", rec)
		}
		if len(*sends) != 1 {
			t.Fatalf("allowed delivery must reach the transport exactly once, got %d", len(*sends))
		}
		got := (*sends)[0]
		if strings.Contains(got.body, slackTok) {
			t.Fatalf("raw secret hit the wire: %q", got.body)
		}
		if !strings.Contains(got.body, "[REDACTED:slack-token]") {
			t.Fatalf("transport must receive the redacted body, got %q", got.body)
		}
		if got.platform != egressfloor.PlatformSlack || got.dest != "C0OPS" {
			t.Fatalf("transport received the wrong destination: %+v", got)
		}
		if len(*witnessed) != 1 || !strings.Contains((*witnessed)[0], "verdict=allow") {
			t.Fatalf("an allowed send must be witnessed, got %v", *witnessed)
		}
	})

	t.Run("transport error surfaces for an allowed send", func(t *testing.T) {
		wantErr := errors.New("wire down")
		s, sends, _ := newSender(wantErr)
		_, err := s.Send(egressfloor.Delivery{
			Platform:    egressfloor.PlatformTelegram,
			Destination: "any-chat-7",
			Message:     "hello",
		})
		if !errors.Is(err, wantErr) {
			t.Fatalf("an allowed send must surface the transport error, got %v", err)
		}
		if len(*sends) != 1 {
			t.Fatalf("expected exactly one send attempt, got %d", len(*sends))
		}
	})

	t.Run("no transport wired is a safe no-op that still adjudicates", func(t *testing.T) {
		s := DeliverySender{Policy: policy} // no Transport, no Witness
		rec, err := s.Send(egressfloor.Delivery{
			Platform:    egressfloor.PlatformSlack,
			Destination: "C0OPS",
			Message:     "hi",
		})
		if err != nil {
			t.Fatalf("a nil-transport sender must not error, got %v", err)
		}
		if !rec.Allowed {
			t.Fatalf("a nil-transport sender must still adjudicate, got %+v", rec)
		}
	})
}

// TestDeliveryReasonRegistered proves egressfloor's out-of-tree ReasonDeliveryBlock (a code
// above the closed core range) resolves to DELIVERY_BLOCK through the abi vocabulary — the
// registration half of the residual the pure floor (init-free by design) could not close on
// its own, so a delivery deny renders as DELIVERY_BLOCK instead of REASON_1025 in audit output
// that resolves codes via abi.ReasonName (e.g. a2achan correction reporting).
//
// It re-registers the pairs (the SAME loop gateway's init() runs) before asserting, so it is
// order-independent: a sibling gateway test may have called abi.ResetForTest(), which wipes the
// dynamically registered reason map (it restores only page-out codecs, never reasons). This is
// the sanctioned "re-register after reset" pattern — it verifies the DeliveryReasonPairs
// vocabulary wiring, the exact contract gateway's init() depends on in production.
func TestDeliveryReasonRegistered(t *testing.T) {
	for _, pr := range egressfloor.DeliveryReasonPairs {
		abi.RegisterReason(pr.Code, pr.Name)
	}
	if got := abi.ReasonName(egressfloor.ReasonDeliveryBlock); got != egressfloor.ReasonDeliveryBlockName {
		t.Fatalf("ReasonDeliveryBlock must resolve to %q, got %q — the DeliveryReasonPairs vocabulary is miswired",
			egressfloor.ReasonDeliveryBlockName, got)
	}
}
