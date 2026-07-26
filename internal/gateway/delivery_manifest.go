// Delivery manifest + witness ledger: the OPERATOR-CONFIGURED half of the delivery send
// path (internal/gateway/delivery_send.go). Issue #4392 (promotion of #2850, Track D, #2834).
//
// WHY THIS LIVES HERE. delivery_send.go's DeliverySender is a pure composition seam: it gates
// a send on a hand-constructed egressfloor.DeliveryPolicy and writes each witness line to an
// INJECTED sink. #2850 deliberately deferred the two pieces that make the seam live for an
// operator: (1) loading the destination allowlist from the policy manifest so it is
// deployment-configured rather than code-baked, and (2) a durable witness sink that persists
// every allow/deny decision to a ledger file before the message can leave the box. This file
// adds exactly those two, plus a constructor that composes them into a live sender.
//
// GENERATION: gen/next. The manifest key (egress.delivery_allow) is parsed gateway-locally
// rather than threaded through internal/policy's EgressRule, so the delivery allowlist lands
// operator-configurable without a cross-lane change to the adjudicator's Policy struct.
// Promotion evidence toward `now`: a real platform adapter constructing this sender from the
// running server's manifest bytes + ledger path and driving a live send. Demotion/retirement
// evidence: egress.delivery_allow absorbed as a first-class field on internal/policy's
// EgressRule (this gateway-local loader deleted in favor of the unified manifest parse).
// Invalidating assumption: witness persistence is FAIL-OPEN — a ledger write error is recorded
// on the ledger (Err) but does NOT currently abort an allowed send, because the DeliveryWitness
// sink signature cannot signal a persistence failure back to the gate in delivery_send.go.

package gateway

import (
	"encoding/json"
	"fmt"
	"os"
	"sync"

	"github.com/anthony-chaudhary/fak/internal/egressfloor"
)

// deliveryManifest is the minimal view of the policy manifest this seam reads: only the
// egress.delivery_allow map (platform -> permitted destinations). It is intentionally a
// narrow, gateway-local shape — not internal/policy's full manifest — so the delivery
// allowlist is operator-configurable from the same manifest file without a cross-lane
// dependency. Unknown manifest keys are ignored (encoding/json is lenient), so this parses a
// full policy manifest and reads only its delivery block.
type deliveryManifest struct {
	Egress struct {
		DeliveryAllow map[string][]string `json:"delivery_allow"`
	} `json:"egress"`
}

// DeliveryPolicyFromManifest parses the egress.delivery_allow block of a policy manifest
// (JSON) into an egressfloor.DeliveryPolicy. The posture is DENY BY DEFAULT, inherited from
// the floor: an absent key, an empty map, or nil bytes yield an empty policy that permits
// nothing, so a mis-scoped or missing manifest fails closed rather than leaking a message.
// Platform keys are the lower-case delivery tokens (egressfloor.Platform values); each
// destination is a chat/channel/user id, or egressfloor.AnyDestination ("*") for a
// platform-wide grant. A malformed manifest is a HARD error (not a silent empty policy) so an
// operator typo surfaces at load time instead of silently denying everything.
func DeliveryPolicyFromManifest(raw []byte) (egressfloor.DeliveryPolicy, error) {
	var m deliveryManifest
	if len(raw) > 0 {
		if err := json.Unmarshal(raw, &m); err != nil {
			return egressfloor.DeliveryPolicy{}, fmt.Errorf("delivery allowlist: parse manifest: %w", err)
		}
	}
	if len(m.Egress.DeliveryAllow) == 0 {
		return egressfloor.DeliveryPolicy{}, nil
	}
	allow := make(map[egressfloor.Platform][]string, len(m.Egress.DeliveryAllow))
	for platform, dests := range m.Egress.DeliveryAllow {
		allow[egressfloor.Platform(platform)] = append([]string(nil), dests...)
	}
	return egressfloor.DeliveryPolicy{Allow: allow}, nil
}

// DeliveryLedger is a file-backed delivery-witness sink: it appends one
// egressfloor.DeliveryRecord.Witness() line per adjudicated delivery — allow OR deny — to a
// ledger file, so the what/where/why of every send decision is durably recorded. It is the
// persistence #2850's floor deferred: the pure record is a value, and this is where it lands
// on disk. Safe for concurrent Send calls (a mutex serializes appends). Because DeliverySender
// writes the witness BEFORE the transport, a line reaches this ledger before the message can
// leave the box.
type DeliveryLedger struct {
	mu   sync.Mutex
	path string
	err  error
}

// OpenDeliveryLedger returns a ledger that appends witness lines to path (created on first
// write). It holds no open handle — each Witness call opens, appends, and closes — so a ledger
// is cheap to keep and tolerates an external truncate/rotate of the file between sends.
func OpenDeliveryLedger(path string) *DeliveryLedger { return &DeliveryLedger{path: path} }

// Witness appends one witness line to the ledger file. Its signature matches DeliveryWitness,
// so ledger.Witness is a drop-in sink for DeliverySender.Witness. A write error is recorded on
// the ledger (surfaced via Err) rather than returned, because DeliveryWitness is a
// fire-and-forget line sink; a caller that must fail closed on a persistence failure checks Err
// after Send. The line is exactly DeliveryRecord.Witness() output — never a cleartext body or a
// raw secret, by the floor's construction.
func (l *DeliveryLedger) Witness(line string) {
	l.mu.Lock()
	defer l.mu.Unlock()
	f, err := os.OpenFile(l.path, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644)
	if err != nil {
		l.err = err
		return
	}
	defer f.Close()
	if _, err := f.WriteString(line + "\n"); err != nil {
		l.err = err
	}
}

// Err returns the first witness-persistence error the ledger encountered, or nil. A live send
// path that must not transmit an unaudited message checks this after Send and treats a non-nil
// result as a hard failure — the witness for that send did not durably land.
func (l *DeliveryLedger) Err() error {
	l.mu.Lock()
	defer l.mu.Unlock()
	return l.err
}

// Path is the ledger file path (for tests and operator diagnostics).
func (l *DeliveryLedger) Path() string { return l.path }

// NewManifestDeliverySender composes a live delivery send path from operator inputs: it loads
// the destination allowlist from the policy manifest (raw JSON, egress.delivery_allow), wires
// the witness sink to a file-backed ledger, and attaches the injected transport. The result is
// the single choke point through which an outbound platform message passes — adjudicated
// against the manifest allowlist, witnessed to the ledger before send, and (only when allowed)
// transmitted in redacted form. A nil ledger leaves the witness unwired; a nil transport yields
// a safe adjudicate-and-witness-only sender (dry-run). The manifest-parse error is returned
// unchanged so an operator typo surfaces at construction rather than as a silent deny-all.
func NewManifestDeliverySender(manifest []byte, ledger *DeliveryLedger, transport DeliveryTransport) (DeliverySender, error) {
	policy, err := DeliveryPolicyFromManifest(manifest)
	if err != nil {
		return DeliverySender{}, err
	}
	s := DeliverySender{Policy: policy, Transport: transport}
	if ledger != nil {
		s.Witness = ledger.Witness
	}
	return s, nil
}
