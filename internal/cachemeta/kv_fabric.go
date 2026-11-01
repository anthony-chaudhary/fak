package cachemeta

import (
	"encoding/json"
	"errors"
	"fmt"
	"strconv"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/fabricmap"
)

// KVFabricIntegrity identifies the canonical logical dense-f32 witness used by
// the native receiver. It is not the padded wire-frame SpanDigest.
const KVFabricIntegrity = "sha256-fak-dense-kv-f32-v1"

var ErrKVFabricNotReady = errors.New("KV fabric receiver is not ready")

// KVFabricReceipt joins a consumer-published plan reference to its actual
// fabricmap route and execution. It records evidence, not routing authority:
// the receiver must obtain the provider and authorizer from trusted configuration.
// LogicalBytes counts live K/Kraw/V bytes; KVTransfer.BytesMoved separately
// accounts for transport padding. A pending receipt cannot normalize to a HIT.
type KVFabricReceipt struct {
	PlanID        string                     `json:"plan_id"`
	Provider      string                     `json:"provider"`
	Generation    uint64                     `json:"generation"`
	ValidUntil    time.Time                  `json:"valid_until"`
	Landing       string                     `json:"landing"`
	ReadyAt       time.Time                  `json:"ready_at,omitempty"`
	SpanDigest    string                     `json:"span_digest,omitempty"`
	ContentDigest string                     `json:"content_digest,omitempty"`
	LogicalBytes  uint64                     `json:"logical_bytes"`
	Route         fabricmap.Route            `json:"route"`
	Execution     fabricmap.ExecutionReceipt `json:"execution"`
}

// Validate checks the complete, ordered public-to-landing-to-ready witness at
// an admission boundary. Callers must also revalidate the provider generation;
// this value alone is neither a capability nor proof of current authorization.
func (f *KVFabricReceipt) Validate(t KVTransfer, now time.Time) error {
	bad := func(reason string) error { return fmt.Errorf("%w: %s", ErrKVFabricNotReady, reason) }
	if f == nil || strings.TrimSpace(f.PlanID) == "" || strings.TrimSpace(f.Provider) == "" || f.Generation == 0 {
		return bad("missing consumer plan")
	}
	if now.IsZero() || f.ReadyAt.IsZero() || f.ReadyAt.After(now) || f.ValidUntil.IsZero() || !now.Before(f.ValidUntil) {
		return bad("missing or expired readiness")
	}
	if t.Outcome != KVTransferOK || f.SpanDigest == "" || f.SpanDigest != t.SpanDigest || f.ContentDigest == "" || f.LogicalBytes == 0 {
		return bad("unbound KV content")
	}
	r, x := f.Route, f.Execution
	if r.From == "" || r.To == "" || f.Landing == "" || r.From == f.Landing || r.To == f.Landing || r.From == r.To || len(r.Links) != 2 || r.Bytes != f.LogicalBytes {
		return bad("unsupported receiver route")
	}
	if r.Links[0].From != r.From || r.Links[0].To != f.Landing || r.Links[1].From != f.Landing || r.Links[1].To != r.To {
		return bad("route does not end at receiver readiness")
	}
	if x.State != fabricmap.ExecutionComplete || x.Recovery != fabricmap.RecoveryNone || x.Operation != fabricmap.OperationCopy || x.From != r.From || x.To != r.To || len(x.Hops) != len(r.Links) {
		return bad("incomplete fabric execution")
	}
	var previous time.Time
	for i, hop := range x.Hops {
		link := r.Links[i]
		if link.ID == "" || link.Transport == "" || hop.HopIndex != i || hop.LinkID != link.ID || hop.From != link.From || hop.To != link.To || hop.Transport != link.Transport || hop.Bytes != f.LogicalBytes {
			return bad("hop identity mismatch")
		}
		if hop.State != fabricmap.HopComplete || hop.Error != "" || hop.Integrity.Algorithm != KVFabricIntegrity || hop.Integrity.Digest != f.ContentDigest {
			return bad("hop content is not witnessed")
		}
		if hop.StartedAt.IsZero() || hop.CompletedAt.IsZero() || hop.CompletedAt.Before(hop.StartedAt) || hop.StartedAt.Before(previous) || hop.CompletedAt.After(f.ReadyAt) {
			return bad("invalid hop ordering")
		}
		previous = hop.CompletedAt
	}
	return nil
}

func putKVFabricLabels(labels map[string]string, f *KVFabricReceipt) {
	if f == nil {
		return
	}
	labels["fabric_plan_id"] = f.PlanID
	labels["fabric_provider"] = f.Provider
	labels["fabric_generation"] = strconv.FormatUint(f.Generation, 10)
	labels["fabric_ready_endpoint"] = f.Route.To
	labels["fabric_state"] = string(f.Execution.State)
	labels["fabric_recovery"] = string(f.Execution.Recovery)
	if encoded, err := json.Marshal(f); err == nil {
		labels["fabric_receipt"] = string(encoded)
	}
}
