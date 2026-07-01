package a2achan

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"strings"
	"unicode/utf8"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

const (
	WorkerStatusSchema    = "fak.a2a.worker-status.v1"
	CorrectionSchema      = "fak.a2a.correction.v1"
	CorrectionAckSchema   = "fak.a2a.correction-ack.v1"
	CorrectionAuditSchema = "fak.a2a.correction-audit.v1"

	// DefaultCorrectionMaxBytes is the hard per-correction text ceiling. The
	// caller may choose a smaller MaxBytes on the request, but cannot widen it.
	DefaultCorrectionMaxBytes = 1024
)

// WorkerStatus is the witnessed live row a worker exposes before accepting an
// orchestrator correction. The correction must cite Digest(), so a stale or
// fabricated resume message is refused before anything reaches the worker inbox.
type WorkerStatus struct {
	Schema     string `json:"schema"`
	WorkerID   string `json:"worker_id"`
	Issue      int    `json:"issue"`
	TaskID     string `json:"task_id,omitempty"`
	Lane       string `json:"lane"`
	Live       bool   `json:"live"`
	Seq        uint64 `json:"seq"`
	LastAction string `json:"last_action,omitempty"`
}

// Digest returns the stable status witness a correction must cite.
func (s WorkerStatus) Digest() string {
	return jsonDigest(s.normalized())
}

func (s WorkerStatus) normalized() WorkerStatus {
	s.Schema = WorkerStatusSchema
	return s
}

// CorrectionRequest is the orchestrator's bounded, scoped control message. The
// request is not trusted by itself: it is admitted only if StatusDigest matches
// the live WorkerStatus supplied to SendCorrection.
type CorrectionRequest struct {
	OrchestratorID string `json:"orchestrator_id"`
	WorkerID       string `json:"worker_id"`
	Issue          int    `json:"issue"`
	TaskID         string `json:"task_id,omitempty"`
	Lane           string `json:"lane"`
	Message        string `json:"message"`
	StatusDigest   string `json:"status_digest"`
	MaxBytes       int    `json:"max_bytes,omitempty"`
}

// NewCorrectionRequest builds the common in-scope request shape from a witnessed
// worker status row.
func NewCorrectionRequest(orchestratorID string, status WorkerStatus, message string) CorrectionRequest {
	status = status.normalized()
	return CorrectionRequest{
		OrchestratorID: orchestratorID,
		WorkerID:       status.WorkerID,
		Issue:          status.Issue,
		TaskID:         status.TaskID,
		Lane:           status.Lane,
		Message:        message,
		StatusDigest:   status.Digest(),
	}
}

// Correction is the typed payload delivered to the worker over the normal a2a
// bus. It is JSON inside a ScopeFleet Ref so the existing taint/scope floor still
// decides the actual transfer.
type Correction struct {
	Schema         string `json:"schema"`
	CorrectionID   string `json:"correction_id"`
	OrchestratorID string `json:"orchestrator_id"`
	WorkerID       string `json:"worker_id"`
	Issue          int    `json:"issue"`
	TaskID         string `json:"task_id,omitempty"`
	Lane           string `json:"lane"`
	Message        string `json:"message"`
	StatusDigest   string `json:"status_digest"`
	MaxBytes       int    `json:"max_bytes"`
}

// CorrectionAudit is an appendable audit row for an attempted control message.
// It records the bounded byte count, the cited status witness, and the closed
// verdict/reason pair returned by the gate.
type CorrectionAudit struct {
	Schema         string `json:"schema"`
	CorrectionID   string `json:"correction_id,omitempty"`
	OrchestratorID string `json:"orchestrator_id"`
	WorkerID       string `json:"worker_id"`
	Issue          int    `json:"issue"`
	TaskID         string `json:"task_id,omitempty"`
	Lane           string `json:"lane"`
	StatusDigest   string `json:"status_digest,omitempty"`
	Bytes          int    `json:"bytes"`
	MaxBytes       int    `json:"max_bytes"`
	Verdict        string `json:"verdict"`
	Reason         string `json:"reason,omitempty"`
	By             string `json:"by"`
}

// CorrectionReceipt is returned to the orchestrator after an attempted send.
type CorrectionReceipt struct {
	Correction Correction      `json:"correction"`
	Audit      CorrectionAudit `json:"audit"`
	Verdict    abi.Verdict     `json:"-"`
}

// WorkerAction is the worker's next planned action after applying a correction.
// The CorrectionID field is the structural reflection witness.
type WorkerAction struct {
	WorkerID     string `json:"worker_id"`
	Issue        int    `json:"issue"`
	TaskID       string `json:"task_id,omitempty"`
	Lane         string `json:"lane"`
	CorrectionID string `json:"correction_id"`
	StatusDigest string `json:"status_digest"`
	Summary      string `json:"summary"`
}

// CorrectionAck is the worker's acknowledgement. It deliberately carries the
// next action so the orchestrator can require both acked and reflected.
type CorrectionAck struct {
	Schema       string       `json:"schema"`
	CorrectionID string       `json:"correction_id"`
	WorkerID     string       `json:"worker_id"`
	StatusDigest string       `json:"status_digest"`
	Acked        bool         `json:"acked"`
	NextAction   WorkerAction `json:"next_action"`
}

// CorrectionObservation is the orchestrator-side witness fold over the ack.
type CorrectionObservation struct {
	CorrectionID string        `json:"correction_id"`
	Acked        bool          `json:"acked"`
	Reflected    bool          `json:"reflected"`
	Ack          CorrectionAck `json:"ack"`
}

// CorrectionInbox is the worker control mailbox. The payload is still admitted
// by Send/Recv; this helper only standardizes the address.
func CorrectionInbox(workerID string) ChannelKey {
	return ChannelKey{Locale: InKernel, ID: "worker:" + workerID + ":corrections"}
}

// CorrectionAckInbox is the orchestrator's acknowledgement mailbox for one
// worker.
func CorrectionAckInbox(orchestratorID, workerID string) ChannelKey {
	return ChannelKey{Locale: InKernel, ID: "orchestrator:" + orchestratorID + ":worker:" + workerID + ":correction-acks"}
}

// SendCorrection validates the status handshake, sends the correction on Allow,
// and returns a witness-grade audit row for both allow and deny paths.
func (b *Bus) SendCorrection(ctx context.Context, req CorrectionRequest, status WorkerStatus, caps ...abi.Capability) CorrectionReceipt {
	correction, audit, v := buildCorrection(req, status)
	if v.Kind != abi.VerdictAllow {
		audit = auditVerdict(audit, v)
		return CorrectionReceipt{Correction: correction, Audit: audit, Verdict: v}
	}
	payload, err := json.Marshal(correction)
	if err != nil {
		v = verdict(abi.VerdictDeny, abi.ReasonMalformed, "a2achan/correction")
		audit = auditVerdict(audit, v)
		return CorrectionReceipt{Correction: correction, Audit: audit, Verdict: v}
	}
	v = b.Send(ctx, req.OrchestratorID, CorrectionInbox(req.WorkerID), Shared(payload), caps...)
	audit = auditVerdict(audit, v)
	return CorrectionReceipt{Correction: correction, Audit: audit, Verdict: v}
}

// SendCorrection sends on the process-global Default bus.
func SendCorrection(ctx context.Context, req CorrectionRequest, status WorkerStatus, caps ...abi.Capability) CorrectionReceipt {
	return Default.SendCorrection(ctx, req, status, caps...)
}

// ReceiveCorrection receives and re-validates the next correction for a worker.
func (b *Bus) ReceiveCorrection(ctx context.Context, status WorkerStatus, caps ...abi.Capability) (Correction, abi.Verdict, error) {
	status = status.normalized()
	if v := liveStatusVerdict(status); v.Kind != abi.VerdictAllow {
		return Correction{}, v, nil
	}
	msg, v, err := b.Recv(ctx, CorrectionInbox(status.WorkerID), caps...)
	if err != nil || v.Kind != abi.VerdictAllow {
		return Correction{}, v, err
	}
	var correction Correction
	if err := json.Unmarshal(msg.Body.Inline, &correction); err != nil {
		return Correction{}, verdict(abi.VerdictDeny, abi.ReasonMalformed, "a2achan/correction"), nil
	}
	if v := validateCorrection(correction, status); v.Kind != abi.VerdictAllow {
		return Correction{}, v, nil
	}
	return correction, verdict(abi.VerdictAllow, abi.ReasonNone, "a2achan/correction"), nil
}

// AckCorrection sends the worker's acknowledgement and next-action reflection.
func (b *Bus) AckCorrection(ctx context.Context, status WorkerStatus, correction Correction, actionSummary string, caps ...abi.Capability) (CorrectionAck, abi.Verdict) {
	if v := validateCorrection(correction, status); v.Kind != abi.VerdictAllow {
		return CorrectionAck{}, v
	}
	if strings.TrimSpace(actionSummary) == "" || !utf8.ValidString(actionSummary) {
		return CorrectionAck{}, verdict(abi.VerdictDeny, abi.ReasonMalformed, "a2achan/correction")
	}
	action := WorkerAction{
		WorkerID:     status.WorkerID,
		Issue:        status.Issue,
		TaskID:       status.TaskID,
		Lane:         status.Lane,
		CorrectionID: correction.CorrectionID,
		StatusDigest: correction.StatusDigest,
		Summary:      actionSummary,
	}
	ack := CorrectionAck{
		Schema:       CorrectionAckSchema,
		CorrectionID: correction.CorrectionID,
		WorkerID:     status.WorkerID,
		StatusDigest: correction.StatusDigest,
		Acked:        true,
		NextAction:   action,
	}
	payload, err := json.Marshal(ack)
	if err != nil {
		return CorrectionAck{}, verdict(abi.VerdictDeny, abi.ReasonMalformed, "a2achan/correction")
	}
	v := b.Send(ctx, status.WorkerID, CorrectionAckInbox(correction.OrchestratorID, status.WorkerID), Shared(payload), caps...)
	return ack, v
}

// ObserveCorrection waits for the worker's ack and verifies that the worker's
// next action structurally references the correction id.
func (b *Bus) ObserveCorrection(ctx context.Context, receipt CorrectionReceipt, caps ...abi.Capability) (CorrectionObservation, abi.Verdict, error) {
	if receipt.Verdict.Kind != abi.VerdictAllow {
		return CorrectionObservation{}, receipt.Verdict, nil
	}
	correction := receipt.Correction
	msg, v, err := b.Recv(ctx, CorrectionAckInbox(correction.OrchestratorID, correction.WorkerID), caps...)
	if err != nil || v.Kind != abi.VerdictAllow {
		return CorrectionObservation{}, v, err
	}
	var ack CorrectionAck
	if err := json.Unmarshal(msg.Body.Inline, &ack); err != nil {
		return CorrectionObservation{}, verdict(abi.VerdictDeny, abi.ReasonMalformed, "a2achan/correction"), nil
	}
	if ack.Schema != CorrectionAckSchema {
		return CorrectionObservation{}, verdict(abi.VerdictDeny, abi.ReasonMalformed, "a2achan/correction"), nil
	}
	if ack.CorrectionID != correction.CorrectionID ||
		ack.WorkerID != correction.WorkerID ||
		ack.StatusDigest != correction.StatusDigest {
		return CorrectionObservation{}, verdict(abi.VerdictDeny, abi.ReasonTrustViolation, "a2achan/correction"), nil
	}
	reflected := ack.NextAction.CorrectionID == correction.CorrectionID &&
		ack.NextAction.StatusDigest == correction.StatusDigest &&
		ack.NextAction.WorkerID == correction.WorkerID &&
		ack.NextAction.Issue == correction.Issue &&
		ack.NextAction.TaskID == correction.TaskID &&
		ack.NextAction.Lane == correction.Lane &&
		strings.TrimSpace(ack.NextAction.Summary) != ""
	if !ack.Acked || !reflected {
		return CorrectionObservation{}, verdict(abi.VerdictDeny, abi.ReasonUnwitnessed, "a2achan/correction"), nil
	}
	return CorrectionObservation{
		CorrectionID: correction.CorrectionID,
		Acked:        ack.Acked,
		Reflected:    reflected,
		Ack:          ack,
	}, verdict(abi.VerdictAllow, abi.ReasonNone, "a2achan/correction"), nil
}

func buildCorrection(req CorrectionRequest, status WorkerStatus) (Correction, CorrectionAudit, abi.Verdict) {
	status = status.normalized()
	limit := correctionLimit(req.MaxBytes)
	audit := CorrectionAudit{
		Schema:         CorrectionAuditSchema,
		OrchestratorID: req.OrchestratorID,
		WorkerID:       req.WorkerID,
		Issue:          req.Issue,
		TaskID:         req.TaskID,
		Lane:           req.Lane,
		StatusDigest:   req.StatusDigest,
		Bytes:          len([]byte(req.Message)),
		MaxBytes:       limit,
		By:             "a2achan/correction",
	}
	if strings.TrimSpace(req.OrchestratorID) == "" ||
		strings.TrimSpace(req.WorkerID) == "" ||
		req.Issue <= 0 ||
		strings.TrimSpace(req.Lane) == "" ||
		strings.TrimSpace(req.Message) == "" ||
		!utf8.ValidString(req.Message) {
		return Correction{}, audit, verdict(abi.VerdictDeny, abi.ReasonMalformed, "a2achan/correction")
	}
	if audit.Bytes > limit {
		return Correction{}, audit, verdict(abi.VerdictDeny, abi.ReasonOversize, "a2achan/correction")
	}
	if v := liveStatusVerdict(status); v.Kind != abi.VerdictAllow {
		return Correction{}, audit, v
	}
	if req.StatusDigest == "" || req.StatusDigest != status.Digest() {
		return Correction{}, audit, verdict(abi.VerdictDeny, abi.ReasonUnwitnessed, "a2achan/correction")
	}
	if !requestMatchesStatus(req, status) {
		return Correction{}, audit, verdict(abi.VerdictDeny, abi.ReasonTrustViolation, "a2achan/correction")
	}
	correction := Correction{
		Schema:         CorrectionSchema,
		OrchestratorID: req.OrchestratorID,
		WorkerID:       req.WorkerID,
		Issue:          req.Issue,
		TaskID:         req.TaskID,
		Lane:           req.Lane,
		Message:        req.Message,
		StatusDigest:   req.StatusDigest,
		MaxBytes:       limit,
	}
	correction.CorrectionID = "corr_" + shortDigest(correction)
	audit.CorrectionID = correction.CorrectionID
	return correction, audit, verdict(abi.VerdictAllow, abi.ReasonNone, "a2achan/correction")
}

func validateCorrection(c Correction, status WorkerStatus) abi.Verdict {
	status = status.normalized()
	if c.Schema != CorrectionSchema ||
		strings.TrimSpace(c.CorrectionID) == "" ||
		strings.TrimSpace(c.OrchestratorID) == "" ||
		strings.TrimSpace(c.WorkerID) == "" ||
		c.Issue <= 0 ||
		strings.TrimSpace(c.Lane) == "" ||
		strings.TrimSpace(c.Message) == "" ||
		!utf8.ValidString(c.Message) {
		return verdict(abi.VerdictDeny, abi.ReasonMalformed, "a2achan/correction")
	}
	if len([]byte(c.Message)) > correctionLimit(c.MaxBytes) {
		return verdict(abi.VerdictDeny, abi.ReasonOversize, "a2achan/correction")
	}
	if v := liveStatusVerdict(status); v.Kind != abi.VerdictAllow {
		return v
	}
	if c.StatusDigest == "" || c.StatusDigest != status.Digest() {
		return verdict(abi.VerdictDeny, abi.ReasonUnwitnessed, "a2achan/correction")
	}
	if c.WorkerID != status.WorkerID || c.Issue != status.Issue || c.TaskID != status.TaskID || c.Lane != status.Lane {
		return verdict(abi.VerdictDeny, abi.ReasonTrustViolation, "a2achan/correction")
	}
	return verdict(abi.VerdictAllow, abi.ReasonNone, "a2achan/correction")
}

func requestMatchesStatus(req CorrectionRequest, status WorkerStatus) bool {
	return req.WorkerID == status.WorkerID &&
		req.Issue == status.Issue &&
		req.TaskID == status.TaskID &&
		req.Lane == status.Lane
}

func liveStatusVerdict(status WorkerStatus) abi.Verdict {
	if status.WorkerID == "" || status.Issue <= 0 || status.Lane == "" || status.Seq == 0 {
		return verdict(abi.VerdictDeny, abi.ReasonUnwitnessed, "a2achan/correction")
	}
	if !status.Live {
		return verdict(abi.VerdictDeny, abi.ReasonUnwitnessed, "a2achan/correction")
	}
	return verdict(abi.VerdictAllow, abi.ReasonNone, "a2achan/correction")
}

func correctionLimit(requested int) int {
	if requested > 0 && requested < DefaultCorrectionMaxBytes {
		return requested
	}
	return DefaultCorrectionMaxBytes
}

func auditVerdict(a CorrectionAudit, v abi.Verdict) CorrectionAudit {
	a.Verdict = verdictKindName(v.Kind)
	a.By = v.By
	if v.Reason != abi.ReasonNone {
		a.Reason = abi.ReasonName(v.Reason)
	}
	return a
}

func verdict(kind abi.VerdictKind, reason abi.ReasonCode, by string) abi.Verdict {
	return abi.Verdict{Kind: kind, Reason: reason, By: by}
}

func jsonDigest(v any) string {
	b, _ := json.Marshal(v)
	sum := sha256.Sum256(b)
	return "sha256:" + hex.EncodeToString(sum[:])
}

func shortDigest(v any) string {
	d := jsonDigest(v)
	return d[len("sha256:") : len("sha256:")+16]
}

func verdictKindName(k abi.VerdictKind) string {
	switch k {
	case abi.VerdictAllow:
		return "ALLOW"
	case abi.VerdictDeny:
		return "DENY"
	case abi.VerdictTransform:
		return "TRANSFORM"
	case abi.VerdictQuarantine:
		return "QUARANTINE"
	case abi.VerdictRequireWitness:
		return "REQUIRE_WITNESS"
	case abi.VerdictDefer:
		return "DEFER"
	case abi.VerdictIndeterminate:
		return "INDETERMINATE"
	default:
		return "VERDICT"
	}
}
