// Package harnessprotocol implements the first-party producer and projections for the public harnesskit protocol.
// Provider wire objects terminate before Append and cannot enter the public stream.
package harnessprotocol

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"sync"

	"github.com/anthony-chaudhary/fak/pkg/harnesskit"
)

var (
	// ErrRunMismatch indicates an input or cursor specifies a different run ID than the producer.
	ErrRunMismatch = errors.New("cursor belongs to another run")
	// ErrBadCheckpoint indicates an invalid, forged, or out-of-bounds checkpoint token.
	ErrBadCheckpoint = errors.New("invalid checkpoint")
	// ErrCanceled indicates operations are rejected because the run has been canceled.
	ErrCanceled = errors.New("run canceled")
	// ErrApprovalBoundary indicates an approval input does not match any pending approval request.
	ErrApprovalBoundary = errors.New("approval does not match a pending request")
)

// Producer sequences, validates, and seals protocol event envelopes for a single execution run.
type Producer struct {
	mu               sync.Mutex
	runID            string
	events           []harnesskit.Envelope
	inputs           map[string]struct{}
	pendingApprovals map[string]struct{}
	canceled         bool
	secret           []byte
}

// NewProducer initializes a new Producer for the given run ID with an HMAC secret for checkpoint integrity.
func NewProducer(runID string, secret []byte) *Producer {
	return &Producer{runID: runID, inputs: map[string]struct{}{}, pendingApprovals: map[string]struct{}{}, secret: append([]byte(nil), secret...)}
}

// Append accepts only product-neutral semantic payloads and assigns total run order and stable IDs.
func (p *Producer) Append(eventType harnesskit.EventType, correlationID, causationID string, sensitivity harnesskit.Sensitivity, payload any) (harnesskit.Envelope, error) {
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.canceled && eventType != harnesskit.EventRunCanceled && eventType != harnesskit.EventCheckpoint {
		return harnesskit.Envelope{}, ErrCanceled
	}
	raw, err := json.Marshal(payload)
	if err != nil {
		return harnesskit.Envelope{}, err
	}
	seq := uint64(len(p.events) + 1)
	e := harnesskit.Envelope{Version: harnesskit.ProtocolVersion, RunID: p.runID, Sequence: seq, EventID: fmt.Sprintf("%s:%d", p.runID, seq), CorrelationID: correlationID, CausationID: causationID, Type: eventType, Sensitivity: sensitivity, Payload: raw}
	if eventType == harnesskit.EventApprovalRequested {
		var v harnesskit.ApprovalPayload
		if e.DecodePayload(&v) != nil || v.ApprovalID == "" {
			return harnesskit.Envelope{}, harnesskit.ErrInvalidProtocol
		}
		p.pendingApprovals[v.ApprovalID] = struct{}{}
	}
	p.events = append(p.events, e)
	return e, nil
}

// Apply handles controls exactly once. Repeated InputID values are acknowledged without repeating effects.
func (p *Producer) Apply(in harnesskit.Input) (bool, error) {
	if err := in.Validate(); err != nil {
		return false, err
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	if in.RunID != p.runID {
		return false, ErrRunMismatch
	}
	if _, ok := p.inputs[in.InputID]; ok {
		return true, nil
	}
	if in.Type == harnesskit.InputApproval {
		if _, ok := p.pendingApprovals[in.Approval.ApprovalID]; !ok {
			return false, ErrApprovalBoundary
		}
		delete(p.pendingApprovals, in.Approval.ApprovalID)
	}
	if in.Type == harnesskit.InputCancel {
		p.canceled = true
	}
	p.inputs[in.InputID] = struct{}{}
	return false, nil
}

func (p *Producer) cursorLocked(sequence uint64) harnesskit.Cursor {
	sum := sha256.Sum256([]byte(fmt.Sprintf("%s\x00%d\x00%s", p.runID, sequence, p.secret)))
	return harnesskit.Cursor{Version: harnesskit.ProtocolVersion, RunID: p.runID, Sequence: sequence, Checkpoint: hex.EncodeToString(sum[:])}
}

// Cursor returns the HMAC-authenticated checkpoint representing the producer's current sequence position.
func (p *Producer) Cursor() harnesskit.Cursor {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.cursorLocked(uint64(len(p.events)))
}

// Resume replays envelopes starting after cursor.Sequence up to the given credit limit.
func (p *Producer) Resume(ctx context.Context, cursor harnesskit.Cursor, credit uint32) ([]harnesskit.Envelope, harnesskit.Cursor, error) {
	p.mu.Lock()
	defer p.mu.Unlock()
	if err := ctx.Err(); err != nil {
		return nil, harnesskit.Cursor{}, err
	}
	if cursor.RunID != "" && cursor.RunID != p.runID {
		return nil, harnesskit.Cursor{}, ErrRunMismatch
	}
	if cursor.Sequence > uint64(len(p.events)) {
		return nil, harnesskit.Cursor{}, ErrBadCheckpoint
	}
	if cursor.Sequence > 0 && cursor.Checkpoint != p.cursorLocked(cursor.Sequence).Checkpoint {
		return nil, harnesskit.Cursor{}, ErrBadCheckpoint
	}
	start := cursor.Sequence
	end := uint64(len(p.events))
	if credit > 0 && end-start > uint64(credit) {
		end = start + uint64(credit)
	}
	out := append([]harnesskit.Envelope(nil), p.events[start:end]...)
	return out, p.cursorLocked(end), nil
}

// Redact enforces the sensitive-data boundary before an untrusted consumer receives events.
func Redact(events []harnesskit.Envelope, allowPrivate bool) []harnesskit.Envelope {
	out := make([]harnesskit.Envelope, 0, len(events))
	for _, e := range events {
		if e.Sensitivity == harnesskit.SensitivitySecret || (e.Sensitivity == harnesskit.SensitivityPrivate && !allowPrivate) {
			e.Payload = json.RawMessage(`{"redacted":true}`)
		}
		out = append(out, e)
	}
	return out
}
