// Delivery send path: the live gateway seam that puts an OUTBOUND PLATFORM MESSAGE
// (Telegram/Discord/Slack/WhatsApp/Signal) through the egress capability floor
// (internal/egressfloor) before it leaves the box. Issue #2850 (Track D, #2834).
//
// WHY THIS LIVES HERE. egressfloor.DeliveryPolicy is a PURE, init-free classifier: given a
// Delivery it returns a witnessed DeliveryRecord (allowlist verdict + pre-send redaction),
// but it deliberately does no I/O and registers nothing. That leaf's own doc names "a
// gateway send path" as the CONSUMER that (1) registers the DELIVERY_BLOCK reason name so a
// deny resolves in audit output instead of REASON_1025/UNCLASSIFIED, (2) writes the witness
// to a ledger, and (3) gates the actual transmit on record.Allowed, sending only the
// redacted body. This file is that consumer — the same consumer-registers-and-gates seam
// the adjudicator already uses for egressfloor's metadata-SSRF rung (ReasonEgressBlock) and
// toolprocgate uses for internal/toolproc's reasons.
//
// GENERATION: gen/next. The transport is INJECTED (DeliveryTransport) rather than a live
// per-platform network client, so the DECISION seam lands and is contract-tested offline
// while the live Telegram/Discord/Slack client stays a follow-on increment. Nothing in the
// running server calls Send yet — it is a dormant, dogfoodable handoff contract that a real
// platform adapter plugs a transport into. Promotion evidence toward `now`: a real adapter
// wiring a live transport plus one end-to-end witnessed send. Demotion/retirement evidence:
// the floor being absorbed wholesale into the adapter layer (this seam deleted). Invalidating
// assumption: that pre-send redaction of the raw message body suffices without
// platform-specific structural parsing (attachments, Slack blocks, Discord embeds) — a
// structured payload could carry a secret in a field this body-level redactor never sees.
package gateway

import (
	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/egressfloor"
)

// init registers the delivery refusal vocabulary. gateway is the in-kernel CONSUMER of
// egressfloor's out-of-tree ReasonDeliveryBlock (a code above abi.ReasonCoreMax), so — like
// toolprocgate for internal/toolproc and the adjudicator for ReasonEgressBlock — it owns the
// name registration: without it a DELIVERY_BLOCK deny renders as REASON_1025 in audit output
// (e.g. a2achan correction reporting via abi.ReasonName). RegisterReason is an idempotent map
// write. gateway is on architest's regOffList (wired into the running process by cmd/fak, not
// blank-imported by the defconfig), so this init() is a sanctioned self-registration.
func init() {
	for _, pr := range egressfloor.DeliveryReasonPairs {
		abi.RegisterReason(pr.Code, pr.Name)
	}
}

// DeliveryTransport is the injected network send — the Hermes-style per-platform client that
// actually puts bytes on the wire. It is handed ONLY the redacted body and ONLY for an
// allowed destination; it never sees a denied delivery or a raw secret. A returned error
// reports a wire failure; the floor's verdict is decided before the transport is ever called.
type DeliveryTransport func(platform egressfloor.Platform, destination, redactedBody string) error

// DeliveryWitness records one delivery-witness line (egressfloor.DeliveryRecord.Witness()) to
// a ledger or log. It is called for EVERY adjudicated delivery — allow or deny — before the
// transport runs, so the witness exists whether or not the message ever leaves the box.
type DeliveryWitness func(line string)

// DeliverySender is the gateway send path: it composes the egress capability floor (Policy)
// with an injected Transport and Witness sink. Send is the single choke point through which
// an outbound platform message passes — adjudicated, witnessed, and (only when allowed)
// transmitted in redacted form. The zero value with only a Policy set is a safe no-op sender
// that adjudicates and witnesses but transmits nothing (useful for a dormant/dry-run path).
type DeliverySender struct {
	Policy    egressfloor.DeliveryPolicy
	Transport DeliveryTransport
	Witness   DeliveryWitness
}

// Send routes one outbound delivery through the floor and returns its witnessed record. The
// order is load-bearing: it (1) adjudicates to get the record (allowlist verdict + redaction),
// (2) writes the witness BEFORE any send, then (3) transmits ONLY when record.Allowed and ONLY
// record.RedactedMessage. A denied delivery is never handed to the transport, and a raw secret
// never reaches either the wire or the witness. The returned error is the transport's: nil for
// a deny, and nil when no transport is wired (the message is adjudicated and witnessed but not
// sent — the caller reads record.Allowed to know the verdict).
func (s DeliverySender) Send(d egressfloor.Delivery) (egressfloor.DeliveryRecord, error) {
	rec := s.Policy.Adjudicate(d)
	if s.Witness != nil {
		s.Witness(rec.Witness())
	}
	if !rec.Allowed || s.Transport == nil {
		return rec, nil
	}
	return rec, s.Transport(rec.Platform, rec.Destination, rec.RedactedMessage)
}
