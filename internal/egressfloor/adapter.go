// Delivery ADAPTERS: the typed plug point an outbound platform surface implements, issue
// #2883 (Track D, #2834) — the second half of the delivery floor whose adjudicator lands in
// delivery.go.
//
// WHY THIS LIVES HERE, AND WHAT IT CLOSES. #2850 gave the floor a verdict
// (DeliveryPolicy.Adjudicate) and #4392 gave the gateway a send path that consults it. Both
// stop at the same place: the actual transmit is an INJECTED func of raw strings
// (gateway.DeliveryTransport, `func(Platform, destination, redactedBody string) error`). A
// bare string seam cannot tell an adjudicated, redacted body from a cleartext one — the type
// system is indifferent, so "the floor ran" is a convention the call site must remember
// rather than a property the compiler keeps. Every discipline the floor established
// (deny-by-default allowlisting, always-redact, witness-before-send) is one forgetful caller
// away from being bypassed, and nothing in the tree would notice.
//
// This half replaces that convention with a structural guarantee. An Adapter is handed a
// DeliveryRecord — a value ONLY DeliveryPolicy.Adjudicate can mint, because Adjudicate stamps
// an unexported field no code outside this package can set. A record forged elsewhere
// (`DeliveryRecord{Allowed: true, RedactedMessage: rawSecret}`) is therefore refused at the
// adapter boundary, not merely discouraged in a doc comment. The unsafe state is
// unrepresentable rather than unrecommended.
//
// The guard is applied in BOTH directions, so there is no back door: AdapterSet.Route checks
// before it dispatches, and each shipped adapter re-checks in its own Deliver, so calling an
// adapter directly — skipping the set entirely — fails closed the same way. An adapter is
// thus safe to hand to code that never learned the delivery discipline.
//
// It stays PURE and init-free like the rest of this tier-1 leaf: the Adapter interface and
// AdapterSet are values, and the one shipped adapter writes to an injected io.Writer, so the
// leaf still performs no I/O of its own, opens nothing, and reads no clock. A live
// network-transport adapter is a consumer's concern, exactly as the witness ledger is.
package egressfloor

import (
	"errors"
	"io"
)

// ErrNotAdjudicated is the refusal for a DeliveryRecord that did not come from
// DeliveryPolicy.Adjudicate — a zero value, or a struct literal built by a caller that
// skipped the floor. This is the failure the raw-string transport seam could not express: it
// is what makes "adjudicated" a checkable property of the value rather than a claim about
// the call site's history.
var ErrNotAdjudicated = errors.New("egressfloor: delivery record was not produced by DeliveryPolicy.Adjudicate")

// ErrDeliveryDenied is the refusal for a record the floor adjudicated and DENIED. It is
// distinct from ErrNotAdjudicated on purpose: a deny means the floor ran and said no (the
// record's Reason is ReasonDeliveryBlock), while a not-adjudicated record means the floor
// never ran at all — an operator debugging a silent non-delivery needs to tell those apart.
var ErrDeliveryDenied = errors.New("egressfloor: delivery denied by the egress capability floor")

// ErrNoAdapter is the refusal for an allowed record whose platform has no registered
// adapter. It fails closed rather than falling back to some default surface: a message must
// never reach a platform nobody deliberately wired.
var ErrNoAdapter = errors.New("egressfloor: no delivery adapter registered for platform")

// ErrNoSink is the refusal for an adapter with no destination wired (a nil io.Writer). Like
// ErrNoAdapter it fails closed — a misconfigured adapter reports that it delivered nothing
// instead of silently discarding the message and reporting success.
var ErrNoSink = errors.New("egressfloor: delivery adapter has no sink wired")

// Adapter is ONE outbound delivery surface — the plug point a per-platform client (a CLI
// writer, a webhook poster, a platform SDK) implements so the floor can hand it a message
// that has already cleared adjudication.
//
// The contract is deliberately narrow, and every part of it is load-bearing:
//
//   - Deliver receives a DeliveryRecord, never raw strings. The record carries its own proof
//     that the floor ran, which is what an implementation checks via EnsureSendable.
//   - An implementation MUST transmit rec.RedactedMessage. The cleartext body is not on the
//     record at all, so this is enforced by absence rather than by instruction — an adapter
//     cannot leak what it was never handed.
//   - An implementation MUST call EnsureSendable first and return its error unchanged, so a
//     direct call that bypasses AdapterSet.Route is refused identically.
type Adapter interface {
	// Platform names the platform this adapter serves; AdapterSet routes by it.
	Platform() Platform
	// Deliver transmits an adjudicated, allowed delivery, or returns why it did not.
	Deliver(rec DeliveryRecord) error
}

// EnsureSendable is the guard every adapter runs before it transmits a byte, and the single
// definition of "may this record leave the box". It refuses a record the floor never
// adjudicated (ErrNotAdjudicated) and a record the floor denied (ErrDeliveryDenied); on nil
// it is safe to transmit rec.RedactedMessage to rec.Destination.
//
// It is exported precisely so an out-of-tree adapter can honour the same contract without
// reimplementing — and therefore without misimplementing — it.
func EnsureSendable(rec DeliveryRecord) error {
	if !rec.adjudicated {
		return ErrNotAdjudicated
	}
	if !rec.Allowed {
		return ErrDeliveryDenied
	}
	return nil
}

// AdapterSet is the per-platform adapter table: it routes an adjudicated record to the one
// adapter that serves its platform. It is the seam a send path composes over the floor —
// adjudicate once, route once — replacing the single untyped transport func with a table
// that can hold a different surface per platform.
//
// The zero AdapterSet holds no adapters and therefore refuses every record with
// ErrNoAdapter, which is the correct deny-by-default posture for this leaf.
type AdapterSet struct {
	byPlatform map[Platform]Adapter
}

// NewAdapterSet builds the routing table from the given adapters, keyed by each adapter's
// declared Platform. A later adapter for a platform replaces an earlier one, so a caller can
// layer a specific override over a default without a separate remove step. A nil adapter in
// the list is skipped rather than stored, so it surfaces as ErrNoAdapter at route time
// instead of a nil-pointer panic mid-send.
func NewAdapterSet(adapters ...Adapter) AdapterSet {
	set := AdapterSet{byPlatform: make(map[Platform]Adapter, len(adapters))}
	for _, a := range adapters {
		if a == nil {
			continue
		}
		set.byPlatform[a.Platform()] = a
	}
	return set
}

// Serves reports whether an adapter is registered for the platform — the lookup a send path
// uses to fail fast (or to fall back to a dry run) before it bothers adjudicating.
func (s AdapterSet) Serves(p Platform) bool {
	_, ok := s.byPlatform[p]
	return ok
}

// Route hands an adjudicated record to the adapter for its platform and returns that
// adapter's error unchanged. It refuses — transmitting nothing — when the record did not
// come from the floor (ErrNotAdjudicated), when the floor denied it (ErrDeliveryDenied), or
// when no adapter serves the platform (ErrNoAdapter).
//
// Route is where the send path's whole discipline collapses to one call: because the record
// is the only way in, there is no argument shape by which a caller could route a body the
// floor has not seen.
func (s AdapterSet) Route(rec DeliveryRecord) error {
	if err := EnsureSendable(rec); err != nil {
		return err
	}
	a, ok := s.byPlatform[rec.Platform]
	if !ok {
		return ErrNoAdapter
	}
	return a.Deliver(rec)
}

// WriterAdapter is the shipped concrete adapter: the CLI/stream delivery surface. It renders
// one adjudicated delivery as a single stable line to an injected io.Writer — the surface a
// `fak deliver` style command, a dry-run preview, or a test harness wires when it wants the
// full floor-to-send path exercised without a network.
//
// It is the honest minimum a real adapter must do and no more: guard, then write only the
// redacted body. Keeping it io.Writer-shaped is what lets this tier-1 leaf ship a WORKING
// adapter while still performing no I/O of its own — the caller owns the file, socket, or
// buffer, so the leaf stays pure and testable.
type WriterAdapter struct {
	// To is the platform this adapter claims; AdapterSet keys the routing table by it.
	To Platform
	// Out is the sink the rendered delivery is written to. A nil Out is refused with
	// ErrNoSink rather than silently dropping the message.
	Out io.Writer
}

// Platform reports the platform this adapter serves, satisfying Adapter.
func (w WriterAdapter) Platform() Platform { return w.To }

// Deliver guards the record and, when it is sendable, writes one line naming the platform,
// the destination, and the REDACTED body. It re-runs EnsureSendable itself so a direct call
// that skipped AdapterSet.Route is refused identically — the guarantee holds per-adapter,
// not merely per-route.
func (w WriterAdapter) Deliver(rec DeliveryRecord) error {
	if err := EnsureSendable(rec); err != nil {
		return err
	}
	if w.Out == nil {
		return ErrNoSink
	}
	_, err := io.WriteString(w.Out,
		"deliver platform="+string(rec.Platform)+
			" dest="+rec.Destination+
			" body="+rec.RedactedMessage+"\n")
	return err
}
