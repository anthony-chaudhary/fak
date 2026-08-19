// This test file is deliberately an EXTERNAL test package (egressfloor_test, not
// egressfloor). That is the whole point of #2883's guarantee: the claim is that code OUTSIDE
// this package cannot mint a sendable DeliveryRecord, and only an external package can
// honestly test it. An internal test could set the unexported `adjudicated` field and would
// therefore prove nothing about the boundary a real caller sits behind.
package egressfloor_test

import (
	"errors"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/egressfloor"
)

// testPolicy is the allowlist the adapter tests adjudicate against: one named Slack
// destination and a wildcard Telegram platform, mirroring the shapes delivery_test.go
// already exercises.
//enumlint:exempt this fixture deliberately allows only Slack and Telegram to prove platform allowlisting.
func testPolicy() egressfloor.DeliveryPolicy {
	return egressfloor.DeliveryPolicy{Allow: map[egressfloor.Platform][]string{
		egressfloor.PlatformSlack:    {"C0OPS"},
		egressfloor.PlatformTelegram: {egressfloor.AnyDestination},
	}}
}

// syntheticSecret builds a redactable token at RUNTIME rather than embedding one as a source
// literal, so this fixture carries no credential-shaped span in the tree. The shape matches
// the github-token class in delivery.go's redaction allowlist.
func syntheticSecret() string {
	return "ghp_" + strings.Repeat("A", 36)
}

// TestForgedRecordIsRefusedByEveryAdapterPath is #2883's REPRODUCTION: the defect the
// pre-#2883 seam permitted, expressed as a test the pre-#2883 tree had no vocabulary for.
//
// Before this change the only send seam was gateway.DeliveryTransport, a
// `func(Platform, destination, redactedBody string) error`. Nothing in that signature
// distinguishes a body the floor redacted from a cleartext one, so a call site that forgot
// (or never knew) to call Adjudicate could hand a raw secret straight to the wire and the
// compiler, the floor, and the witness ledger would all stay silent. The failure mode is not
// exotic — it is the DEFAULT behaviour of any new caller.
//
// This proves the seam now fails closed on exactly that mistake: a record a caller builds
// itself, with Allowed set true and a raw secret as the body, is refused with
// ErrNotAdjudicated and NOTHING is written — via the routing table AND via a direct call on
// the adapter that skips the table entirely.
func TestForgedRecordIsRefusedByEveryAdapterPath(t *testing.T) {
	secret := syntheticSecret()
	// The forgery: a caller that skipped the floor and asserted its own verdict. This
	// compiles — DeliveryRecord's fields are exported because the witness path needs to
	// read them — which is precisely why the refusal must be a runtime guarantee.
	forged := egressfloor.DeliveryRecord{
		Platform:        egressfloor.PlatformSlack,
		Destination:     "C0OPS", // an ALLOWED destination: the allowlist is not what saves us here
		Allowed:         true,
		RedactedMessage: "here is my key " + secret,
	}

	t.Run("routing a forged record transmits nothing", func(t *testing.T) {
		var sink strings.Builder
		set := egressfloor.NewAdapterSet(egressfloor.WriterAdapter{
			To: egressfloor.PlatformSlack, Out: &sink,
		})
		err := set.Route(forged)
		if !errors.Is(err, egressfloor.ErrNotAdjudicated) {
			t.Fatalf("Route must refuse a record the floor never produced with ErrNotAdjudicated, got %v", err)
		}
		if sink.Len() != 0 {
			t.Fatalf("a forged record must transmit nothing, wrote %q", sink.String())
		}
	})

	t.Run("calling the adapter directly transmits nothing", func(t *testing.T) {
		var sink strings.Builder
		// Skipping AdapterSet entirely — the back door a per-adapter guard has to close.
		adapter := egressfloor.WriterAdapter{To: egressfloor.PlatformSlack, Out: &sink}
		err := adapter.Deliver(forged)
		if !errors.Is(err, egressfloor.ErrNotAdjudicated) {
			t.Fatalf("a direct Deliver must refuse a forged record with ErrNotAdjudicated, got %v", err)
		}
		if sink.Len() != 0 {
			t.Fatalf("a direct Deliver of a forged record must transmit nothing, wrote %q", sink.String())
		}
	})

	t.Run("EnsureSendable refuses the zero record", func(t *testing.T) {
		if err := egressfloor.EnsureSendable(egressfloor.DeliveryRecord{}); !errors.Is(err, egressfloor.ErrNotAdjudicated) {
			t.Fatalf("the zero record must be refused with ErrNotAdjudicated, got %v", err)
		}
	})
}

// TestAdjudicatedDeliveryReachesTheAdapter is #2883's done-condition witness: a delivery
// payload is PROVEN to pass the egress floor before being sent, and the adapter receives
// only the redacted body.
func TestAdjudicatedDeliveryReachesTheAdapter(t *testing.T) {
	secret := syntheticSecret()
	var sink strings.Builder
	set := egressfloor.NewAdapterSet(egressfloor.WriterAdapter{
		To: egressfloor.PlatformSlack, Out: &sink,
	})

	rec := testPolicy().Adjudicate(egressfloor.Delivery{
		Platform:    egressfloor.PlatformSlack,
		Destination: "C0OPS",
		Message:     "ship it, key " + secret + " fyi",
	})
	if !rec.Allowed {
		t.Fatalf("allowlisted destination must be allowed, got %+v", rec)
	}
	if err := set.Route(rec); err != nil {
		t.Fatalf("routing an adjudicated allowed record must succeed, got %v", err)
	}

	got := sink.String()
	if strings.Contains(got, secret) {
		t.Fatalf("the raw secret reached the adapter: %q", got)
	}
	if !strings.Contains(got, "[REDACTED:github-token]") {
		t.Fatalf("the adapter must receive the REDACTED body, got %q", got)
	}
	if !strings.Contains(got, "platform=slack") || !strings.Contains(got, "dest=C0OPS") {
		t.Fatalf("the adapter must receive the adjudicated destination, got %q", got)
	}
}

// TestRouteRefusalsFailClosed covers the remaining refusal rungs: a floor DENY, an allowed
// record with no adapter for its platform, an adapter with no sink, and the zero AdapterSet.
// Every one of them must transmit nothing — the deny-by-default posture delivery.go
// establishes has to survive the trip through the adapter layer.
func TestRouteRefusalsFailClosed(t *testing.T) {
	policy := testPolicy()

	t.Run("a floor deny is refused with ErrDeliveryDenied", func(t *testing.T) {
		var sink strings.Builder
		set := egressfloor.NewAdapterSet(egressfloor.WriterAdapter{
			To: egressfloor.PlatformDiscord, Out: &sink,
		})
		// Discord is not on the allowlist at all, so the floor denies it — yet an adapter
		// for discord IS registered, which is exactly the case a platform-only check would
		// wave through.
		rec := policy.Adjudicate(egressfloor.Delivery{
			Platform:    egressfloor.PlatformDiscord,
			Destination: "123",
			Message:     "hello",
		})
		if rec.Allowed {
			t.Fatalf("off-allowlist platform must be denied, got %+v", rec)
		}
		if err := set.Route(rec); !errors.Is(err, egressfloor.ErrDeliveryDenied) {
			t.Fatalf("a denied record must be refused with ErrDeliveryDenied, got %v", err)
		}
		if sink.Len() != 0 {
			t.Fatalf("a denied record must transmit nothing, wrote %q", sink.String())
		}
	})

	t.Run("an allowed record with no adapter is refused with ErrNoAdapter", func(t *testing.T) {
		var sink strings.Builder
		// An adapter for slack only; the allowed delivery targets telegram.
		set := egressfloor.NewAdapterSet(egressfloor.WriterAdapter{
			To: egressfloor.PlatformSlack, Out: &sink,
		})
		rec := policy.Adjudicate(egressfloor.Delivery{
			Platform:    egressfloor.PlatformTelegram,
			Destination: "any-chat-7",
			Message:     "hello",
		})
		if !rec.Allowed {
			t.Fatalf("wildcard destination must be allowed, got %+v", rec)
		}
		if err := set.Route(rec); !errors.Is(err, egressfloor.ErrNoAdapter) {
			t.Fatalf("an unserved platform must be refused with ErrNoAdapter, got %v", err)
		}
		if sink.Len() != 0 {
			t.Fatalf("an unserved platform must not leak into another platform's adapter, wrote %q", sink.String())
		}
	})

	t.Run("the zero AdapterSet refuses every record", func(t *testing.T) {
		rec := policy.Adjudicate(egressfloor.Delivery{
			Platform:    egressfloor.PlatformSlack,
			Destination: "C0OPS",
			Message:     "hello",
		})
		var zero egressfloor.AdapterSet
		if err := zero.Route(rec); !errors.Is(err, egressfloor.ErrNoAdapter) {
			t.Fatalf("the zero AdapterSet must refuse with ErrNoAdapter, got %v", err)
		}
	})

	t.Run("an adapter with no sink is refused with ErrNoSink", func(t *testing.T) {
		set := egressfloor.NewAdapterSet(egressfloor.WriterAdapter{To: egressfloor.PlatformSlack})
		rec := policy.Adjudicate(egressfloor.Delivery{
			Platform:    egressfloor.PlatformSlack,
			Destination: "C0OPS",
			Message:     "hello",
		})
		if err := set.Route(rec); !errors.Is(err, egressfloor.ErrNoSink) {
			t.Fatalf("an unwired adapter must be refused with ErrNoSink, got %v", err)
		}
	})

	t.Run("a nil adapter is skipped, not stored", func(t *testing.T) {
		set := egressfloor.NewAdapterSet(nil)
		rec := policy.Adjudicate(egressfloor.Delivery{
			Platform:    egressfloor.PlatformSlack,
			Destination: "C0OPS",
			Message:     "hello",
		})
		if err := set.Route(rec); !errors.Is(err, egressfloor.ErrNoAdapter) {
			t.Fatalf("a nil adapter must surface as ErrNoAdapter, not a panic, got %v", err)
		}
	})
}

// countingAdapter is a minimal OUT-OF-TREE adapter: it proves the Adapter interface is
// implementable from another package using only the exported surface, and that
// EnsureSendable is enough for such an implementation to inherit the whole guarantee.
type countingAdapter struct {
	to     egressfloor.Platform
	bodies []string
}

func (c *countingAdapter) Platform() egressfloor.Platform { return c.to }

func (c *countingAdapter) Deliver(rec egressfloor.DeliveryRecord) error {
	if err := egressfloor.EnsureSendable(rec); err != nil {
		return err
	}
	c.bodies = append(c.bodies, rec.RedactedMessage)
	return nil
}

// TestOutOfTreeAdapterInheritsTheGuarantee proves the plug point is real: a third-party
// adapter that does nothing but call EnsureSendable gets the same fail-closed behaviour as
// the shipped one, including the last-adapter-wins override in NewAdapterSet.
func TestOutOfTreeAdapterInheritsTheGuarantee(t *testing.T) {
	policy := testPolicy()
	custom := &countingAdapter{to: egressfloor.PlatformSlack}

	var shipped strings.Builder
	// The custom adapter is registered AFTER the shipped one for the same platform, so it
	// must win — the documented override path.
	set := egressfloor.NewAdapterSet(
		egressfloor.WriterAdapter{To: egressfloor.PlatformSlack, Out: &shipped},
		custom,
	)
	if !set.Serves(egressfloor.PlatformSlack) {
		t.Fatal("Serves must report a registered platform")
	}
	if set.Serves(egressfloor.PlatformSignal) {
		t.Fatal("Serves must not report an unregistered platform")
	}

	rec := policy.Adjudicate(egressfloor.Delivery{
		Platform:    egressfloor.PlatformSlack,
		Destination: "C0OPS",
		Message:     "hello",
	})
	if err := set.Route(rec); err != nil {
		t.Fatalf("routing to the custom adapter must succeed, got %v", err)
	}
	if len(custom.bodies) != 1 || custom.bodies[0] != "hello" {
		t.Fatalf("the later adapter must win and receive the body, got %v", custom.bodies)
	}
	if shipped.Len() != 0 {
		t.Fatalf("the overridden adapter must not also receive the delivery, wrote %q", shipped.String())
	}

	// And the same custom adapter refuses a forgery, with no extra code of its own.
	if err := custom.Deliver(egressfloor.DeliveryRecord{Allowed: true}); !errors.Is(err, egressfloor.ErrNotAdjudicated) {
		t.Fatalf("an out-of-tree adapter must inherit the forgery refusal, got %v", err)
	}
	if len(custom.bodies) != 1 {
		t.Fatalf("the refused forgery must not have been recorded, got %v", custom.bodies)
	}
}
