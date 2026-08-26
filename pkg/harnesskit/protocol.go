package harnesskit

import (
	"encoding/json"
	"errors"
	"fmt"
)

// ProtocolVersion is the stable major version of the product-neutral run protocol.
const ProtocolVersion = "fak.harness.run/v1"

// ProtocolContract is the machine-readable schema and invariant index for run adapters.
type ProtocolContract struct {
	Version               string      `json:"version"`
	Events                []EventType `json:"events"`
	Inputs                []InputType `json:"inputs"`
	Ordering              string      `json:"ordering"`
	Resume                string      `json:"resume"`
	UnknownEvents         string      `json:"unknown_events"`
	ApprovalBoundary      string      `json:"approval_boundary"`
	SensitiveDataBoundary string      `json:"sensitive_data_boundary"`
}

// PublicProtocolContract returns the normative run protocol inventory.
func PublicProtocolContract() ProtocolContract {
	return ProtocolContract{
		Version:               ProtocolVersion,
		Events:                []EventType{EventRunStarted, EventRunCompleted, EventMessageStarted, EventMessageDelta, EventMessageCompleted, EventUIBlock, EventToolStarted, EventToolCompleted, EventPlanUpdated, EventApprovalRequested, EventApprovalResolved, EventArtifactPublished, EventError, EventUsage, EventRunCanceled, EventCheckpoint},
		Inputs:                []InputType{InputMessage, InputApproval, InputCancel, InputCredit},
		Ordering:              "sequence starts at 1 and increases exactly once per immutable run event",
		Resume:                "cursor is exclusive; checkpoint authenticates run_id and sequence; credit bounds delivery",
		UnknownEvents:         "validate envelope, advance cursor, ignore payload without rendering or authority",
		ApprovalBoundary:      "approval.resolve must name one pending approval and grants only its declared scope",
		SensitiveDataBoundary: "secret always redacted; private requires an authenticated private consumer",
	}
}

// EventType identifies semantic output. New values are additive within a major version.
type EventType string

const (
	EventRunStarted        EventType = "run.started"
	EventRunCompleted      EventType = "run.completed"
	EventMessageStarted    EventType = "message.started"
	EventMessageDelta      EventType = "message.delta"
	EventMessageCompleted  EventType = "message.completed"
	EventUIBlock           EventType = "ui.block"
	EventToolStarted       EventType = "tool.started"
	EventToolCompleted     EventType = "tool.completed"
	EventPlanUpdated       EventType = "plan.updated"
	EventApprovalRequested EventType = "approval.requested"
	EventApprovalResolved  EventType = "approval.resolved"
	EventArtifactPublished EventType = "artifact.published"
	EventError             EventType = "error"
	EventUsage             EventType = "usage"
	EventRunCanceled       EventType = "run.canceled"
	EventCheckpoint        EventType = "checkpoint"
)

// Sensitivity labels data before it crosses a product boundary.
type Sensitivity string

const (
	SensitivityPublic  Sensitivity = "public"
	SensitivityPrivate Sensitivity = "private"
	SensitivitySecret  Sensitivity = "secret"
)

// Envelope is the ordered, replayable protocol record. Payload is semantic JSON and never a provider wire type.
type Envelope struct {
	Version       string          `json:"version"`
	RunID         string          `json:"run_id"`
	Sequence      uint64          `json:"sequence"`
	EventID       string          `json:"event_id"`
	CorrelationID string          `json:"correlation_id,omitempty"`
	CausationID   string          `json:"causation_id,omitempty"`
	Type          EventType       `json:"type"`
	Sensitivity   Sensitivity     `json:"sensitivity,omitempty"`
	Payload       json.RawMessage `json:"payload,omitempty"`
}

// DecodePayload decodes a known event payload without coupling an adapter to provider types.
func (e Envelope) DecodePayload(dst any) error {
	if len(e.Payload) == 0 {
		return nil
	}
	if err := json.Unmarshal(e.Payload, dst); err != nil {
		return fmt.Errorf("%s payload: %w", e.Type, err)
	}
	return nil
}

// Known reports whether this library understands the event. Consumers must ignore unknown events while advancing cursors.
func (e Envelope) Known() bool {
	switch e.Type {
	case EventRunStarted, EventRunCompleted, EventMessageStarted, EventMessageDelta, EventMessageCompleted,
		EventUIBlock, EventToolStarted, EventToolCompleted, EventPlanUpdated, EventApprovalRequested,
		EventApprovalResolved, EventArtifactPublished, EventError, EventUsage, EventRunCanceled, EventCheckpoint:
		return true
	default:
		return false
	}
}

type RunPayload struct {
	Status string `json:"status"`
	Reason string `json:"reason,omitempty"`
}
type MessagePayload struct {
	MessageID string `json:"message_id"`
	Role      string `json:"role,omitempty"`
	Text      string `json:"text,omitempty"`
}
type UIBlockPayload struct {
	BlockID string          `json:"block_id"`
	Kind    string          `json:"kind"`
	Data    json.RawMessage `json:"data,omitempty"`
}
type ToolPayload struct {
	CallID  string `json:"call_id"`
	Name    string `json:"name"`
	Status  string `json:"status,omitempty"`
	Summary string `json:"summary,omitempty"`
}
type PlanStep struct {
	ID     string `json:"id"`
	Text   string `json:"text"`
	Status string `json:"status"`
}
type PlanPayload struct {
	Steps []PlanStep `json:"steps"`
}
type ApprovalPayload struct {
	ApprovalID         string `json:"approval_id"`
	Prompt             string `json:"prompt,omitempty"`
	Status             string `json:"status"`
	Scope              string `json:"scope,omitempty"`
	Kind               string `json:"kind,omitempty"`
	Summary            string `json:"summary,omitempty"`
	Workspace          string `json:"workspace,omitempty"`
	Risk               string `json:"risk,omitempty"`
	Consequence        string `json:"consequence,omitempty"`
	PolicyReason       string `json:"policy_reason,omitempty"`
	FakCapabilityFloor string `json:"fak_capability_floor,omitempty"`
	CodexSandboxPolicy string `json:"codex_sandbox_policy,omitempty"`
}
type ArtifactPayload struct {
	ArtifactID string `json:"artifact_id"`
	MediaType  string `json:"media_type"`
	URI        string `json:"uri"`
	Name       string `json:"name,omitempty"`
}
type ErrorPayload struct {
	Code      string `json:"code"`
	Message   string `json:"message"`
	Retryable bool   `json:"retryable"`
}
type UsagePayload struct {
	InputTokens  int64 `json:"input_tokens"`
	OutputTokens int64 `json:"output_tokens"`
	CostMicros   int64 `json:"cost_micros,omitempty"`
}
type CheckpointPayload struct {
	Cursor Cursor `json:"cursor"`
}

// Cursor is the exclusive resume position. Sequence zero means from the beginning.
type Cursor struct {
	Version    string `json:"version"`
	RunID      string `json:"run_id"`
	Sequence   uint64 `json:"sequence"`
	Checkpoint string `json:"checkpoint"`
}

// InputType identifies idempotent controls sent to a run.
type InputType string

const (
	InputMessage  InputType = "message.submit"
	InputApproval InputType = "approval.resolve"
	InputCancel   InputType = "run.cancel"
	InputCredit   InputType = "flow.credit"
)

// Input is a client command. InputID is an idempotency key scoped to RunID.
type Input struct {
	Version       string         `json:"version"`
	RunID         string         `json:"run_id"`
	InputID       string         `json:"input_id"`
	CorrelationID string         `json:"correlation_id,omitempty"`
	Type          InputType      `json:"type"`
	Message       *MessageInput  `json:"message,omitempty"`
	Approval      *ApprovalInput `json:"approval,omitempty"`
	Credit        uint32         `json:"credit,omitempty"`
	Reason        string         `json:"reason,omitempty"`
}
type MessageInput struct {
	Text        string      `json:"text"`
	Sensitivity Sensitivity `json:"sensitivity,omitempty"`
}
type ApprovalInput struct {
	ApprovalID string `json:"approval_id"`
	Decision   string `json:"decision"`
	Scope      string `json:"scope,omitempty"`
}

var ErrInvalidProtocol = errors.New("invalid harness protocol record")

func (e Envelope) Validate() error {
	if e.Version != ProtocolVersion || e.RunID == "" || e.Sequence == 0 || e.EventID == "" || e.Type == "" {
		return ErrInvalidProtocol
	}
	return nil
}
func (i Input) Validate() error {
	if i.Version != ProtocolVersion || i.RunID == "" || i.InputID == "" {
		return ErrInvalidProtocol
	}
	switch i.Type {
	case InputMessage:
		if i.Message == nil || i.Message.Text == "" {
			return ErrInvalidProtocol
		}
	case InputApproval:
		if i.Approval == nil || i.Approval.ApprovalID == "" || (i.Approval.Decision != "approve" && i.Approval.Decision != "deny") {
			return ErrInvalidProtocol
		}
	case InputCancel:
		if i.Reason == "" {
			return ErrInvalidProtocol
		}
	case InputCredit:
		if i.Credit == 0 {
			return ErrInvalidProtocol
		}
	default:
		return ErrInvalidProtocol
	}
	return nil
}
