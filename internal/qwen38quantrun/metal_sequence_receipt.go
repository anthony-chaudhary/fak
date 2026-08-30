package qwen38quantrun

import (
	"github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/qwen38quant"
)

const (
	QwenMetalSequencePASS = "PASS"
	QwenMetalSequenceHOLD = "HOLD"

	QwenMetalSequenceControl   = "control"
	QwenMetalSequenceCandidate = "candidate"
	QwenMetalSequencePair      = "pair"

	qwenMetalBackend            = "metal"
	qwenMetalNativeEngine       = qwen38quant.EngineFakNative
	qwenMetalControlForwardPath = "metal/qwen35-hybrid-session-v1"
)

// QwenMetalSequenceHoldReason is the stable machine identity for one fail-closed
// campaign finding. These values describe evidence integrity, not performance.
type QwenMetalSequenceHoldReason string

const (
	HoldQwenMetalReceiptMissing      QwenMetalSequenceHoldReason = "receipt_missing"
	HoldQwenMetalEngine              QwenMetalSequenceHoldReason = "engine_not_fak_native"
	HoldQwenMetalBackend             QwenMetalSequenceHoldReason = "backend_not_metal"
	HoldQwenMetalSelectorOn          QwenMetalSequenceHoldReason = "control_selector_on"
	HoldQwenMetalSelectorOff         QwenMetalSequenceHoldReason = "candidate_selector_off"
	HoldQwenMetalForwardPath         QwenMetalSequenceHoldReason = "forward_path_mismatch"
	HoldQwenMetalQuantization        QwenMetalSequenceHoldReason = "q4k_not_executed"
	HoldQwenMetalFallback            QwenMetalSequenceHoldReason = "fallback_active"
	HoldQwenMetalArtifactInvalid     QwenMetalSequenceHoldReason = "artifact_identity_invalid"
	HoldQwenMetalArtifactModel       QwenMetalSequenceHoldReason = "artifact_model_mismatch"
	HoldQwenMetalSequenceUnexpected  QwenMetalSequenceHoldReason = "control_sequence_receipt_present"
	HoldQwenMetalSequenceMissing     QwenMetalSequenceHoldReason = "candidate_sequence_receipt_missing"
	HoldQwenMetalSelectorReceipt     QwenMetalSequenceHoldReason = "selector_receipt_mismatch"
	HoldQwenMetalEvidenceState       QwenMetalSequenceHoldReason = "sequence_evidence_state_mismatch"
	HoldQwenMetalSequenceUnavailable QwenMetalSequenceHoldReason = "sequence_unavailable"
	HoldQwenMetalSequencePath        QwenMetalSequenceHoldReason = "sequence_path_mismatch"
	HoldQwenMetalSequenceTokens      QwenMetalSequenceHoldReason = "sequence_tokens_not_p32"
	HoldQwenMetalCommandBuffers      QwenMetalSequenceHoldReason = "command_buffers_not_one"
	HoldQwenMetalEncoders            QwenMetalSequenceHoldReason = "encoders_not_positive"
	HoldQwenMetalTerminalWaits       QwenMetalSequenceHoldReason = "terminal_waits_not_one"
	HoldQwenMetalTerminalReadbacks   QwenMetalSequenceHoldReason = "terminal_readbacks_not_one"
	HoldQwenMetalUncommitted         QwenMetalSequenceHoldReason = "sequence_uncommitted"
	HoldQwenMetalIncomplete          QwenMetalSequenceHoldReason = "sequence_incomplete"
	HoldQwenMetalUploadBytes         QwenMetalSequenceHoldReason = "host_upload_bytes_mismatch"
	HoldQwenMetalReadbackBytes       QwenMetalSequenceHoldReason = "host_readback_bytes_mismatch"
	HoldQwenMetalIntermediateWaits   QwenMetalSequenceHoldReason = "intermediate_waits_nonzero"
	HoldQwenMetalIntermediateReads   QwenMetalSequenceHoldReason = "intermediate_readbacks_nonzero"
	HoldQwenMetalArtifactMismatch    QwenMetalSequenceHoldReason = "control_candidate_artifact_mismatch"
)

// QwenMetalSequenceArm binds the public native receipt to the artifact and
// transfer-byte expectations owned by one campaign arm.
type QwenMetalSequenceArm struct {
	SelectorEnabled           bool
	Artifact                  qwen38quant.Identity
	ExpectedHostUploadBytes   uint64
	ExpectedHostReadbackBytes uint64
	Receipt                   *model.NativeInferenceReceipt
}

// QwenMetalSequenceFinding identifies one arm and its stable HOLD reason.
type QwenMetalSequenceFinding struct {
	Arm    string                      `json:"arm"`
	Reason QwenMetalSequenceHoldReason `json:"reason"`
}

// QwenMetalSequenceValidation is the fail-closed verdict for one control and
// candidate pair.
type QwenMetalSequenceValidation struct {
	Verdict  string                     `json:"verdict"`
	Findings []QwenMetalSequenceFinding `json:"findings,omitempty"`
}

// ValidateQwenMetalSequencePair validates both arms independently before it
// compares their artifacts. Its check order is fixed, so callers get the same
// typed finding order regardless of mutation or JSON field order.
func ValidateQwenMetalSequencePair(control, candidate QwenMetalSequenceArm) QwenMetalSequenceValidation {
	findings := validateQwenMetalSequenceArm(QwenMetalSequenceControl, control, false)
	if len(findings) != 0 {
		return QwenMetalSequenceValidation{Verdict: QwenMetalSequenceHOLD, Findings: findings}
	}
	findings = validateQwenMetalSequenceArm(QwenMetalSequenceCandidate, candidate, true)
	if len(findings) == 0 && control.Artifact != candidate.Artifact {
		findings = append(findings, QwenMetalSequenceFinding{Arm: QwenMetalSequencePair, Reason: HoldQwenMetalArtifactMismatch})
	}
	if len(findings) != 0 {
		return QwenMetalSequenceValidation{Verdict: QwenMetalSequenceHOLD, Findings: findings}
	}
	return QwenMetalSequenceValidation{Verdict: QwenMetalSequencePASS}
}

func validateQwenMetalSequenceArm(name string, arm QwenMetalSequenceArm, candidate bool) []QwenMetalSequenceFinding {
	add := func(findings []QwenMetalSequenceFinding, reason QwenMetalSequenceHoldReason) []QwenMetalSequenceFinding {
		return append(findings, QwenMetalSequenceFinding{Arm: name, Reason: reason})
	}
	var findings []QwenMetalSequenceFinding
	if arm.Receipt == nil {
		return add(findings, HoldQwenMetalReceiptMissing)
	}
	receipt := arm.Receipt
	// NativeInferenceReceipt names the in-process product engine as "inkernel".
	// qwen38quant.EngineFakNative is the campaign-level classification and must
	// not be substituted into this public request receipt.
	if receipt.Engine != qwenMetalNativeEngine {
		findings = add(findings, HoldQwenMetalEngine)
	}
	if receipt.Backend != qwenMetalBackend {
		findings = add(findings, HoldQwenMetalBackend)
	}
	if candidate {
		if !arm.SelectorEnabled {
			findings = add(findings, HoldQwenMetalSelectorOff)
		}
		if receipt.ForwardPath != model.Qwen35MetalGDNSequenceForwardPath {
			findings = add(findings, HoldQwenMetalForwardPath)
		}
	} else {
		if arm.SelectorEnabled {
			findings = add(findings, HoldQwenMetalSelectorOn)
		}
		if receipt.ForwardPath != qwenMetalControlForwardPath {
			findings = add(findings, HoldQwenMetalForwardPath)
		}
	}
	if !receipt.Q4K {
		findings = add(findings, HoldQwenMetalQuantization)
	}
	if receipt.FallbackActive {
		findings = add(findings, HoldQwenMetalFallback)
	}
	if !validQwenMetalArtifact(arm.Artifact) {
		findings = add(findings, HoldQwenMetalArtifactInvalid)
	}
	if receipt.Model != arm.Artifact.Model {
		findings = add(findings, HoldQwenMetalArtifactModel)
	}

	sequence := receipt.Qwen35MetalForwardSequence
	if sequence == nil {
		return add(findings, HoldQwenMetalSequenceMissing)
	}
	wantSelector := model.Qwen35MetalSequenceSelectorOff
	wantEvidence := model.Qwen35MetalSequenceEvidenceNotSelected
	if candidate {
		wantSelector = model.Qwen35MetalSequenceSelectorOn
		wantEvidence = model.Qwen35MetalSequenceEvidenceExecuted
	}
	if sequence.SelectorState != wantSelector {
		findings = add(findings, HoldQwenMetalSelectorReceipt)
	}
	if sequence.EvidenceState != wantEvidence {
		findings = add(findings, HoldQwenMetalEvidenceState)
	}
	if sequence.IntermediateWaits != 0 {
		findings = add(findings, HoldQwenMetalIntermediateWaits)
	}
	if sequence.IntermediateReadbacks != 0 {
		findings = add(findings, HoldQwenMetalIntermediateReads)
	}
	if !candidate {
		if sequence.Available || sequence.Tokens != 0 || sequence.CommandBuffers != 0 || sequence.Encoders != 0 ||
			sequence.TerminalWaits != 0 || sequence.TerminalReadbacks != 0 || sequence.HostUploadBytes != 0 ||
			sequence.HostReadbackBytes != 0 || sequence.Committed || sequence.CompletedWait || sequence.TimingAvailable ||
			sequence.GPUMilliseconds != 0 || sequence.WaitMilliseconds != 0 || sequence.StateIdentity != nil {
			findings = add(findings, HoldQwenMetalSequenceUnexpected)
		}
		return findings
	}
	if !sequence.Available {
		findings = add(findings, HoldQwenMetalSequenceUnavailable)
	}
	if sequence.Path != model.Qwen35MetalGDNSequenceForwardPath {
		findings = add(findings, HoldQwenMetalSequencePath)
	}
	if sequence.Tokens != 32 {
		findings = add(findings, HoldQwenMetalSequenceTokens)
	}
	if sequence.CommandBuffers != 1 {
		findings = add(findings, HoldQwenMetalCommandBuffers)
	}
	if sequence.Encoders <= 0 {
		findings = add(findings, HoldQwenMetalEncoders)
	}
	if sequence.TerminalWaits != 1 {
		findings = add(findings, HoldQwenMetalTerminalWaits)
	}
	if sequence.TerminalReadbacks != 1 {
		findings = add(findings, HoldQwenMetalTerminalReadbacks)
	}
	if !sequence.Committed {
		findings = add(findings, HoldQwenMetalUncommitted)
	}
	if !sequence.CompletedWait {
		findings = add(findings, HoldQwenMetalIncomplete)
	}
	if arm.ExpectedHostUploadBytes == 0 || sequence.HostUploadBytes != arm.ExpectedHostUploadBytes {
		findings = add(findings, HoldQwenMetalUploadBytes)
	}
	if arm.ExpectedHostReadbackBytes == 0 || sequence.HostReadbackBytes != arm.ExpectedHostReadbackBytes {
		findings = add(findings, HoldQwenMetalReadbackBytes)
	}
	return findings
}

func validQwenMetalArtifact(id qwen38quant.Identity) bool {
	if len(missingAdapterIdentity(id)) != 0 {
		return false
	}
	return validHex(id.CheckpointSHA256, 32) && validHex(id.ArtifactSHA256, 32) &&
		validHex(id.TokenizerSHA256, 32) && validHex(id.TemplateSHA256, 32)
}
