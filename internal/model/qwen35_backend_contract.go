package model

import (
	"fmt"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// Qwen35GDNPreprojectedSequencePath is the backend-neutral capability identity
// for one complete, preprojected GDN sequence. It deliberately does not name a
// device: Metal and other native runtimes can implement the same ownership and
// no-fallback contract without becoming a compute.Backend.
const Qwen35GDNPreprojectedSequencePath = "qwen35/gdn-preprojected-sequence-v1"

// Qwen35GDNAuxHandle is an opaque backend-owned auxiliary-state identity. Zero
// is never live; the model compares non-zero values only to enforce stable,
// session-local ownership.
type Qwen35GDNAuxHandle uint64

// Qwen35GDNAuxState identifies the convolution window and recurrent matrix for
// one session/layer pair.
type Qwen35GDNAuxState struct {
	Convolution Qwen35GDNAuxHandle
	Recurrent   Qwen35GDNAuxHandle
}

func (s Qwen35GDNAuxState) valid() bool {
	return s.Convolution != 0 && s.Recurrent != 0 && s.Convolution != s.Recurrent
}

func (s Qwen35GDNAuxState) present() bool {
	return s.Convolution != 0 || s.Recurrent != 0
}

// Qwen35GDNSequenceGeometry is the fixed per-layer shape used both to allocate
// auxiliary state and to validate a sequence operation.
type Qwen35GDNSequenceGeometry struct {
	NumKeyHeads, NumValueHeads int
	KeyHeadDim, ValueHeadDim   int
	ConvKernel                 int
}

// Qwen35GDNPreprojectedSequenceRequest contains the projections consumed by the
// GDN convolution/recurrent scan. State and Geometry are filled by Session at
// dispatch; callers supply the preprojected panels and scalar parameters.
type Qwen35GDNPreprojectedSequenceRequest struct {
	Layer, Tokens              int
	Mixed, Z, B, A             []float32
	Conv1D, ALog, DTBias, Norm []float32
	RMSNormEpsilon             float32
	State                      Qwen35GDNAuxState
	Geometry                   Qwen35GDNSequenceGeometry
}

// Qwen35GDNPreprojectedSequenceResult returns the pre-out-projection core and
// the same in-place state identities supplied in the request.
type Qwen35GDNPreprojectedSequenceResult struct {
	Core  []float32
	State Qwen35GDNAuxState
}

// Qwen35GDNPreprojectedSequenceBackend is an optional native whole-operation
// capability. New/Free own auxiliary state; Sequence must mutate that state in
// place and must not return replacement handles.
type Qwen35GDNPreprojectedSequenceBackend interface {
	Qwen35GDNPreprojectedSequencePath() string
	NewQwen35GDNAuxState(layer int, geometry Qwen35GDNSequenceGeometry) (Qwen35GDNAuxState, error)
	Qwen35GDNPreprojectedSequence(Qwen35GDNPreprojectedSequenceRequest) (Qwen35GDNPreprojectedSequenceResult, error)
	FreeQwen35GDNAuxState(Qwen35GDNAuxState) error
}

type qwen35GDNPreprojectedSequencePathMarker interface {
	Qwen35GDNPreprojectedSequencePath() string
}

// UnsupportedGDNPreprojectedSequenceError is a pre-allocation refusal. An
// advertised-but-incomplete or wrong-path capability never mutates session or
// backend state.
type UnsupportedGDNPreprojectedSequenceError struct {
	Path   string
	Reason string
}

func (e *UnsupportedGDNPreprojectedSequenceError) Error() string {
	return fmt.Sprintf("model: cannot admit Qwen GDN preprojected sequence via %q: %s; refusing host recurrence fallback", e.Path, e.Reason)
}

func qwen35GDNPreprojectedSequenceBackend(candidate any) (Qwen35GDNPreprojectedSequenceBackend, bool, error) {
	marker, advertised := candidate.(qwen35GDNPreprojectedSequencePathMarker)
	if !advertised {
		return nil, false, nil
	}
	path := marker.Qwen35GDNPreprojectedSequencePath()
	backend, ok := candidate.(Qwen35GDNPreprojectedSequenceBackend)
	if !ok {
		return nil, true, &UnsupportedGDNPreprojectedSequenceError{Path: path, Reason: "path marker has no lifecycle/operation implementation"}
	}
	if path != Qwen35GDNPreprojectedSequencePath {
		return nil, true, &UnsupportedGDNPreprojectedSequenceError{Path: path, Reason: "wrong capability identity"}
	}
	return backend, true, nil
}

// Qwen35GDNCapabilityIdentity is the canonical versioned capability identity for
// the Qwen3.5/3.6 Gated-DeltaNet/SSM token mixer.
const Qwen35GDNCapabilityIdentity = "qwen35/gdn-token-mixer-v1"

// Qwen35GDNCUDAPath is the production path identity reserved for a Qwen3.5/3.6
// Gated-DeltaNet/SSM token mixer implemented by the CUDA compute backend.
const (
	Qwen35GDNCUDAPath   = "cuda/qwen35-gdn-ssm-decode-v1"
	Qwen35GDNVulkanPath = "vulkan/qwen35-gdn-ssm-decode-v1"
)

// IsSupportedQwen35GDNPath returns true if the given path identity represents a supported
// Qwen3.5/3.6 GDN capability or legacy device path.
func IsSupportedQwen35GDNPath(p string) bool {
	return p == Qwen35GDNCapabilityIdentity || p == Qwen35GDNCUDAPath || p == Qwen35GDNVulkanPath
}

// Qwen35GDNParityCosineMin is the deterministic device/reference acceptance floor.
const Qwen35GDNParityCosineMin = 0.999

// Qwen35GDNBackend is the whole-operation CUDA/HAL seam for the hybrid token mixer.
// The signature uses only compute-owned and built-in types, so a compute backend
// can implement it structurally without importing model and creating a cycle.
type Qwen35GDNBackend interface {
	Qwen35GDNPath() string
	Qwen35GDNDecode(
		normalizedInput,
		inProjQKV, inProjZ, inProjB, inProjA,
		conv1D, aLog, dtBias, norm, outProj,
		convState, recurrentState compute.Tensor,
		numKeyHeads, numValueHeads, keyHeadDim, valueHeadDim, convKernel int,
		rmsNormEpsilon float32,
	) (output, nextConvState, nextRecurrentState compute.Tensor, err error)
}

type qwen35GDNPathMarker interface {
	Qwen35GDNPath() string
}

// UnsupportedBackendForwardError is the fail-closed verdict returned when a
// recognized model forward cannot execute wholly on the selected compute backend.
// For #4714 it prevents the Qwen3.6 GDN/SSM hybrid from entering tokenHALOutput's
// standard Q/K/V loop or silently using the legacy CPU recurrent implementation.
type UnsupportedBackendForwardError struct {
	Backend         string
	Forward         ForwardPathKind
	IntendedPath    string
	ParityCosineMin float64
	Reason          string
}

func (e *UnsupportedBackendForwardError) Error() string {
	return fmt.Sprintf(
		"model: backend %q cannot execute forward %q via %q: %s; refusing generic QKV/CPU fallback (required deterministic CPU-reference parity cosine >= %.3f; issue #4714)",
		e.Backend, e.Forward, e.IntendedPath, e.Reason, e.ParityCosineMin,
	)
}

// ValidateBackendForwardConfig checks an architecture/backend pair without requiring a
// constructed or weight-loaded Model. Serve header preflight can therefore refuse a
// missing, marker-only, or wrong-path hybrid backend before it allocates model storage.
// A nil backend still means the caller selected the legacy CPU/reference path; that path
// remains admitted and never enters the compute HAL.
func ValidateBackendForwardConfig(cfg Config, be compute.Backend) error {
	if be == nil || !cfg.IsQwen35Hybrid() {
		return nil
	}
	gdn, ok := be.(Qwen35GDNBackend)
	if ok && IsSupportedQwen35GDNPath(gdn.Qwen35GDNPath()) {
		return nil
	}
	reason := "backend does not structurally implement model.Qwen35GDNBackend"
	if marker, marked := be.(qwen35GDNPathMarker); marked {
		if !ok {
			reason = fmt.Sprintf("backend advertises marker path %q but does not structurally implement model.Qwen35GDNBackend", marker.Qwen35GDNPath())
		} else {
			reason = fmt.Sprintf("backend implements model.Qwen35GDNBackend with wrong path %q", marker.Qwen35GDNPath())
		}
	}
	return &UnsupportedBackendForwardError{
		Backend:         be.Name(),
		Forward:         ForwardQwen35GDN,
		IntendedPath:    Qwen35GDNCUDAPath + " or " + Qwen35GDNVulkanPath,
		ParityCosineMin: Qwen35GDNParityCosineMin,
		Reason:          reason,
	}
}

// ValidateBackendForwardPath is the Model-bound twin retained for callers that already
// constructed a Model. All admission logic lives in ValidateBackendForwardConfig so the
// pre-load and session-construction decisions cannot drift.
func (m *Model) ValidateBackendForwardPath(be compute.Backend) error {
	if m == nil {
		return nil
	}
	return ValidateBackendForwardConfig(m.Cfg, be)
}

// WholeSequencePath is the backend-neutral capability identity for one complete
// native whole-sequence execution (prefill panels + resident decode handoff). It
// deliberately does not name a device or architecture: the concrete Qwen35 Metal
// runtime adapts its own receipt into the neutral seam below, and a later HIP /
// CUDA / other runtime can implement the same interface without editing the
// whole-token witness.
const WholeSequencePath = "native/whole-sequence-v1"

// WholeSequenceSelectorState is backend-authored selection provenance. It
// reflects whether the session admitted a native whole-sequence owner; callers
// never supply or override it.
type WholeSequenceSelectorState string

const (
	WholeSequenceSelectorOff WholeSequenceSelectorState = "off"
	WholeSequenceSelectorOn  WholeSequenceSelectorState = "on"
)

// WholeSequenceEvidenceState distinguishes a truthful zero from a route that did
// not run or cannot run in the current execution envelope.
type WholeSequenceEvidenceState string

const (
	WholeSequenceEvidenceNotSelected WholeSequenceEvidenceState = "not_selected"
	WholeSequenceEvidenceUnsupported WholeSequenceEvidenceState = "unsupported"
	WholeSequenceEvidenceUnavailable WholeSequenceEvidenceState = "unavailable"
	WholeSequenceEvidenceExecuted    WholeSequenceEvidenceState = "executed"
)

// WholeSequenceReceipt is the backend-neutral immutable value snapshot of a
// native whole-sequence owner. The concrete Qwen35 Metal receipt is adapted into
// this shape; Metal-only observation fields (for example the state-identity
// binding) are intentionally not part of the neutral contract.
type WholeSequenceReceipt struct {
	Path                  string                     `json:"path"`
	Available             bool                       `json:"available"`
	SelectorState         WholeSequenceSelectorState `json:"selector_state"`
	EvidenceState         WholeSequenceEvidenceState `json:"evidence_state"`
	Tokens                int                        `json:"tokens"`
	CommandBuffers        int                        `json:"command_buffers"`
	Encoders              int                        `json:"encoders"`
	IntermediateWaits     int                        `json:"intermediate_waits"`
	IntermediateReadbacks int                        `json:"intermediate_readbacks"`
	TerminalWaits         int                        `json:"terminal_waits"`
	TerminalReadbacks     int                        `json:"terminal_readbacks"`
	HostUploadBytes       uint64                     `json:"host_upload_bytes"`
	HostReadbackBytes     uint64                     `json:"host_readback_bytes"`
	Committed             bool                       `json:"committed"`
	CompletedWait         bool                       `json:"completed_wait"`
	TimingAvailable       bool                       `json:"timing_available"`
	GPUMilliseconds       float64                    `json:"gpu_milliseconds"`
	WaitMilliseconds      float64                    `json:"wait_milliseconds"`
	SelectedPanels        int                        `json:"selected_panels,omitempty"`
	ExecutedPanels        int                        `json:"executed_panels,omitempty"`
	FallbackCount         int                        `json:"fallback_count,omitempty"`
	Device                string                     `json:"device,omitempty"`
	SourceRevision        string                     `json:"source_revision,omitempty"`
	ArtifactSHA256        string                     `json:"artifact_sha256,omitempty"`
}

// WholeSequenceHandoff is the backend-neutral decode-route accounting snapshot.
// Counts advance only after the corresponding operation accepts ownership. The
// wire keys preserve the historical receipt field names so the serialized
// whole-token witness stays byte-compatible.
type WholeSequenceHandoff struct {
	Mode                  string `json:"mode"`
	BlockAcceptedCalls    uint64 `json:"block_accepted_calls"`
	MixerAcceptedCalls    uint64 `json:"mixer_accepted_calls"`
	ResidentAcceptedCalls uint64 `json:"resident_gdn_accepted_calls"`
}

// WholeSequenceSession is the backend-neutral whole-sequence capability seam.
// It expresses enable/finalize of a whole sequence, before/after receipt reads,
// handoff-counter reads, and the path + evidence-state tokens the witness
// validates against — none of which name a device or architecture. The concrete
// Qwen35 Metal path implements it through wholeSequenceAdapter; a non-Qwen test
// double can implement it directly.
type WholeSequenceSession interface {
	EnableWholeSequence() error
	FinalizeWholeSequence() (bool, error)
	WholeSequenceReceipt() WholeSequenceReceipt
	WholeSequenceHandoffReceipt() WholeSequenceHandoff
	WholeSequencePath() string
	WholeSequenceExecutedEvidence() WholeSequenceEvidenceState
	WholeSequenceSelectorOnState() WholeSequenceSelectorState
}

// wholeSequenceAdapter adapts a *Session onto the backend-neutral
// WholeSequenceSession seam. It owns no state of its own: every method delegates
// to the concrete Session and converts the Qwen35 receipt/handoff value into the
// neutral shapes. The Metal execution path is unchanged.
type wholeSequenceAdapter struct{ s *Session }

// WholeSequence returns the backend-neutral whole-sequence capability owner for
// this session. It is the seam cmd/modelbench consumes so the witness never names
// the concrete Qwen35 Metal receipt or backend types.
func (s *Session) WholeSequence() WholeSequenceSession {
	return wholeSequenceAdapter{s: s}
}

func (a wholeSequenceAdapter) EnableWholeSequence() error {
	if a.s == nil {
		return &UnsupportedGDNPreprojectedSequenceError{Path: Qwen35MetalGDNSequenceForwardPath, Reason: "session is nil"}
	}
	return a.s.EnableQwen35MetalGDNPreprojectedSequence()
}

func (a wholeSequenceAdapter) FinalizeWholeSequence() (bool, error) {
	if a.s == nil {
		return false, nil
	}
	return a.s.FinalizeQwen35MetalGDNPreprojectedSequence()
}

func (a wholeSequenceAdapter) WholeSequenceReceipt() WholeSequenceReceipt {
	if a.s == nil {
		return WholeSequenceReceipt{}
	}
	return WholeSequenceReceiptFromQwen35(a.s.Qwen35MetalForwardSequenceReceipt())
}

func (a wholeSequenceAdapter) WholeSequenceHandoffReceipt() WholeSequenceHandoff {
	if a.s == nil {
		return WholeSequenceHandoff{Mode: "AUTO"}
	}
	return WholeSequenceHandoffFromQwen35(a.s.Qwen35DecodeHandoffReceipt())
}

// WholeSequencePath returns the runtime's concrete capability token. It is the
// value the adapted receipts carry in their Path field; the witness validates
// the receipt against this token rather than a hard-coded neutral constant.
func (a wholeSequenceAdapter) WholeSequencePath() string { return Qwen35MetalGDNSequenceForwardPath }

func (a wholeSequenceAdapter) WholeSequenceExecutedEvidence() WholeSequenceEvidenceState {
	return WholeSequenceEvidenceExecuted
}

func (a wholeSequenceAdapter) WholeSequenceSelectorOnState() WholeSequenceSelectorState {
	return WholeSequenceSelectorOn
}

// WholeSequenceReceiptFromQwen35 adapts the concrete Qwen35 Metal receipt into
// the backend-neutral shape. It preserves every neutral field verbatim; the
// Metal-only state-identity binding is intentionally dropped.
func WholeSequenceReceiptFromQwen35(r Qwen35MetalForwardSequenceReceipt) WholeSequenceReceipt {
	return WholeSequenceReceipt{
		Path: r.Path, Available: r.Available,
		SelectorState: WholeSequenceSelectorState(r.SelectorState),
		EvidenceState: WholeSequenceEvidenceState(r.EvidenceState),
		Tokens:        r.Tokens, CommandBuffers: r.CommandBuffers, Encoders: r.Encoders,
		IntermediateWaits: r.IntermediateWaits, IntermediateReadbacks: r.IntermediateReadbacks,
		TerminalWaits: r.TerminalWaits, TerminalReadbacks: r.TerminalReadbacks,
		HostUploadBytes: r.HostUploadBytes, HostReadbackBytes: r.HostReadbackBytes,
		Committed: r.Committed, CompletedWait: r.CompletedWait, TimingAvailable: r.TimingAvailable,
		GPUMilliseconds: r.GPUMilliseconds, WaitMilliseconds: r.WaitMilliseconds,
		SelectedPanels: r.SelectedPanels, ExecutedPanels: r.ExecutedPanels, FallbackCount: r.FallbackCount,
		Device: r.Device, SourceRevision: r.SourceRevision, ArtifactSHA256: r.ArtifactSHA256,
	}
}

// WholeSequenceHandoffFromQwen35 adapts the concrete decode-handoff receipt into
// the backend-neutral counter snapshot.
func WholeSequenceHandoffFromQwen35(r Qwen35DecodeHandoffReceipt) WholeSequenceHandoff {
	return WholeSequenceHandoff{
		Mode: string(r.Mode), BlockAcceptedCalls: r.BlockAcceptedCalls,
		MixerAcceptedCalls: r.MixerAcceptedCalls, ResidentAcceptedCalls: r.ResidentGDNAcceptedCalls,
	}
}

// WholeSequenceOperation pairs the before/after neutral receipts of one decode
// step with the handoff-counter snapshots either side of it. It is retained in
// the serialized report so readback can reject stale receipts without relying on
// pointer identity after JSON decoding.
type WholeSequenceOperation struct {
	Before       WholeSequenceReceipt `json:"before"`
	After        WholeSequenceReceipt `json:"after"`
	CountsBefore WholeSequenceHandoff `json:"counts_before"`
	CountsAfter  WholeSequenceHandoff `json:"counts_after"`
	CacheBefore  int                  `json:"cache_before"`
	CacheAfter   int                  `json:"cache_after"`
}

// ValidateWholeSequenceOperation is the backend-neutral whole-token lockstep
// validator. route is the expected executed route ("whole-token" or "per-layer");
// path is the session's declared capability path and executed is the session's
// executed-evidence token, both read from the WholeSequenceSession seam.
//
// path is the required capability token. Readback of a serialized artifact must
// supply the token the report recorded at build time; the validator never trusts
// a path the receipt declares for itself, so stripping that token cannot re-open
// the check. Calling with an empty path on the whole-token route is refused.
// It is the exact semantics of the former cmd/modelbench validation, lifted into
// the model so a non-Qwen test double can drive it.
func ValidateWholeSequenceOperation(route, path string, executed WholeSequenceEvidenceState, op WholeSequenceOperation) error {
	if op.CacheAfter != op.CacheBefore+1 {
		return fmt.Errorf("model: whole-sequence Step did not advance exactly one cache position")
	}
	a, c, n := op.After, op.CountsBefore, op.CountsAfter
	switch route {
	// The promoted trunk whole-token route records its acceptance on
	// BlockAcceptedCalls and rewrites the HAL forward receipt to a fresh
	// Tokens=1 executed receipt. Requiring ResidentAcceptedCalls to advance would
	// wrongly reject that real route; requiring it to stay unchanged still
	// rejects a per-layer fallback.
	case "whole-token":
		if path == "" {
			return fmt.Errorf("model: whole-sequence capability path is required")
		}
		if op.Before == op.After || a.Tokens != 1 || !a.Committed || !a.CompletedWait || a.CommandBuffers != 1 || a.TerminalWaits != 1 || a.TerminalReadbacks != 1 || a.IntermediateReadbacks != 0 || a.IntermediateWaits != 0 || a.Path != path || a.EvidenceState != executed || n.BlockAcceptedCalls != c.BlockAcceptedCalls+1 || n.ResidentAcceptedCalls != c.ResidentAcceptedCalls || n.MixerAcceptedCalls != c.MixerAcceptedCalls {
			return fmt.Errorf("model: whole-sequence Step lacks a fresh successful whole-token receipt; fallback is non-qualifying")
		}
	case "per-layer":
		if op.Before != op.After || n.BlockAcceptedCalls != c.BlockAcceptedCalls+1 || n.MixerAcceptedCalls != c.MixerAcceptedCalls || n.ResidentAcceptedCalls != c.ResidentAcceptedCalls {
			return fmt.Errorf("model: whole-sequence source-control Step did not execute exactly one prior AUTO per-layer block route")
		}
	default:
		return fmt.Errorf("model: unknown whole-sequence route %q", route)
	}
	return nil
}

// Qwen35SequencePrefillBackend is the optional whole-prompt seam for native
// Qwen3.5/3.8 hybrid execution. The compute-owned request avoids a model/compute
// package cycle and does not widen compute.Backend.
type Qwen35SequencePrefillBackend interface {
	Qwen35SequencePrefillPath() string
	Qwen35SequencePrefill(compute.Qwen35SequencePrefillRequest) (compute.Qwen35SequencePrefillResult, error)
}

type qwen35SequencePrefillPathMarker interface {
	Qwen35SequencePrefillPath() string
}

// UnsupportedSequencePrefillError reports a backend that advertises the native
// sequence path but cannot execute its complete contract. Callers must not retry
// through scalar token replay after this error.
type UnsupportedSequencePrefillError struct {
	Backend string
	Path    string
	Reason  string
}

func (e *UnsupportedSequencePrefillError) Error() string {
	return fmt.Sprintf("model: backend %q cannot execute Qwen hybrid sequence prefill via %q: %s; refusing scalar/CPU fallback", e.Backend, e.Path, e.Reason)
}

func qwen35SequencePrefillBackend(be compute.Backend) (Qwen35SequencePrefillBackend, bool, error) {
	marker, advertised := be.(qwen35SequencePrefillPathMarker)
	if !advertised {
		return nil, false, nil
	}
	path := marker.Qwen35SequencePrefillPath()
	seq, ok := be.(Qwen35SequencePrefillBackend)
	if !ok {
		return nil, true, &UnsupportedSequencePrefillError{Backend: be.Name(), Path: path, Reason: "path marker has no operation implementation"}
	}
	if path != compute.Qwen35SequencePrefillPath {
		return nil, true, &UnsupportedSequencePrefillError{Backend: be.Name(), Path: path, Reason: "wrong capability identity"}
	}
	return seq, true, nil
}
