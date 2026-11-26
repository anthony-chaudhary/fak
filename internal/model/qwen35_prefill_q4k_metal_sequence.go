package model

import (
	"fmt"

	"github.com/anthony-chaudhary/fak/internal/metalgemm"
)

// Qwen35MetalGDNSequenceForwardPath names the resident-state candidate. It is
// reported only after the native operation ran and its final state was handed
// to the historical decode path.
const Qwen35MetalGDNSequenceForwardPath = "metal/qwen35-gdn-preprojected-sequence-v1"

// PrefillPanelRoundTripBudget is the L1 ceiling on the aggregate command-buffer
// round-trips a resident-Q4_K Qwen3.8 prefill may spend serving the fixed
// 256-token L1 prompt. It mirrors the private
// perfpattern DecodeRoundTripBudgetPerToken recipe (issue #13042) without
// importing the private package: the ceiling is a small per-prompt constant, NOT
// a function of len/32, so a regression back to one terminal GPU wait per
// 32-token panel (256/32 == 8 round-trips) trips the assertion instead of merely
// showing up in a timing table. At the widest witnessed panel
// (metalgemm.PromptPanelMaxTokens == 256) 256 tokens is exactly one panel, so
// the budget is 1. PrefillPanelRoundTripBudgetPromptTokens names the prompt the
// ceiling is fixed at; a longer prompt legitimately needs more panels
// (O(len/256)) and is not what this L1 guard measures.
const PrefillPanelRoundTripBudget = 1

// PrefillPanelRoundTripBudgetPromptTokens is the prompt length the
// PrefillPanelRoundTripBudget ceiling is asserted against.
const PrefillPanelRoundTripBudgetPromptTokens = 256

// The Darwin implementation installs this factory at package initialization.
// The model-only type keeps pure-Go builds free of Darwin/cgo GDNState symbols.
var newQwen35MetalGDNSequenceBackend func() Qwen35GDNPreprojectedSequenceBackend

// Native projection admission runs before allocating sequence auxiliary state.
var qwen35MetalForwardProjectionAdmission func(*Session) error

// Qwen35MetalGDNPreprojectedSequenceAvailable reports whether this build owns
// the native Metal sequence backend. Callers use this capability readback to
// distinguish an unselected supported route from a platform-unavailable route;
// it does not admit a fallback implementation.
func Qwen35MetalGDNPreprojectedSequenceAvailable() bool {
	return newQwen35MetalGDNSequenceBackend != nil
}

// Qwen35MetalSequenceSelectorState is model-authored selection provenance. It
// reflects whether this Session admitted the native sequence owner; callers do
// not supply or override it when the public receipt is assembled.
type Qwen35MetalSequenceSelectorState string

const (
	Qwen35MetalSequenceSelectorOff Qwen35MetalSequenceSelectorState = "off"
	Qwen35MetalSequenceSelectorOn  Qwen35MetalSequenceSelectorState = "on"
)

// Qwen35MetalSequenceEvidenceState distinguishes a truthful zero from a route
// that did not run or cannot run in the current execution envelope.
type Qwen35MetalSequenceEvidenceState string

const (
	Qwen35MetalSequenceEvidenceNotSelected Qwen35MetalSequenceEvidenceState = "not_selected"
	Qwen35MetalSequenceEvidenceUnsupported Qwen35MetalSequenceEvidenceState = "unsupported"
	Qwen35MetalSequenceEvidenceUnavailable Qwen35MetalSequenceEvidenceState = "unavailable"
	Qwen35MetalSequenceEvidenceExecuted    Qwen35MetalSequenceEvidenceState = "executed"
)

type qwen35GDNSequenceSnapshotter interface {
	SnapshotQwen35GDNAuxState(Qwen35GDNAuxState) (conv, recurrent []float32, err error)
}

// Qwen35MetalForwardSequenceReceipt is an immutable value snapshot of the
// model-owned whole-sequence Metal graph. Available is false when that route
// has not produced a terminal receipt; a committed post-submit failure remains
// available so callers can distinguish it from the default path.
type Qwen35MetalForwardSequenceReceipt struct {
	Path                        string                           `json:"path"`
	Available                   bool                             `json:"available"`
	SelectorState               Qwen35MetalSequenceSelectorState `json:"selector_state"`
	EvidenceState               Qwen35MetalSequenceEvidenceState `json:"evidence_state"`
	Tokens                      int                              `json:"tokens"`
	CommandBuffers              int                              `json:"command_buffers"`
	Encoders                    int                              `json:"encoders"`
	IntermediateWaits           int                              `json:"intermediate_waits"`
	IntermediateReadbacks       int                              `json:"intermediate_readbacks"`
	TerminalWaits               int                              `json:"terminal_waits"`
	TerminalReadbacks           int                              `json:"terminal_readbacks"`
	HostUploadBytes             uint64                           `json:"host_upload_bytes"`
	HostReadbackBytes           uint64                           `json:"host_readback_bytes"`
	DeviceKVAttentionLayers     int                              `json:"device_kv_attention_layers,omitempty"`
	DeviceKVPrefixUploadBytes   uint64                           `json:"device_kv_prefix_upload_bytes,omitempty"`
	DeviceKVSuffixReadbackBytes uint64                           `json:"device_kv_suffix_readback_bytes,omitempty"`
	Committed                   bool                             `json:"committed"`
	CompletedWait               bool                             `json:"completed_wait"`
	TimingAvailable             bool                             `json:"timing_available"`
	GPUMilliseconds             float64                          `json:"gpu_milliseconds"`
	WaitMilliseconds            float64                          `json:"wait_milliseconds"`
	StateIdentity               *Qwen35MetalStateIdentityReceipt `json:"state_identity,omitempty"`
	SelectedPanels              int                              `json:"selected_panels,omitempty"`
	ExecutedPanels              int                              `json:"executed_panels,omitempty"`
	FallbackCount               int                              `json:"fallback_count,omitempty"`
	Device                      string                           `json:"device,omitempty"`
	SourceRevision              string                           `json:"source_revision,omitempty"`
	ArtifactSHA256              string                           `json:"artifact_sha256,omitempty"`
	// Graph-tracked input, intermediate and terminal Metal buffers retained until
	// graph teardown; sequential panels report the maximum over their graphs.
	// Excludes untracked constants and attention temporaries, GDN state, persistent
	// KV and weights; this is not a process or device working-set measure.
	AllocatedBuffers    int    `json:"allocated_buffers"`
	RetainedBufferBytes uint64 `json:"retained_buffer_bytes"`
}

// qwen35MetalReceiptSourceRevision and qwen35MetalReceiptArtifactSHA256 are the
// build-provenance stamps carried onto every Qwen35MetalForwardSequenceReceipt. They
// replace the retired ambient reads FAK_SOURCE_REVISION and FAK_ARTIFACT_SHA256
// (#13605 leaf A-cfg-int4): release and installer builds stamp them at link time with
//
//	-ldflags "-X github.com/anthony-chaudhary/fak/internal/model.qwen35MetalReceiptSourceRevision=<commit> \
//	            -X github.com/anthony-chaudhary/fak/internal/model.qwen35MetalReceiptArtifactSHA256=<sha256>"
//
// (the appversion.BuildVersion idiom; the vulkan warm-disk stamps use the same seam).
// An unstamped build leaves both empty, and the receipt fields stay empty too —
// unknown provenance, exactly the historical unset-environment behavior. Only a
// build-time fact lands here: no operator text is labelled verified and nothing
// hashes per turn.
var (
	qwen35MetalReceiptSourceRevision string
	qwen35MetalReceiptArtifactSHA256 string
)

type qwen35MetalForwardSequenceRunner interface {
	Qwen35MetalForwardSequence(*Session, []int) ([]float32, Qwen35MetalForwardSequenceReceipt, bool, error)
	Qwen35MetalForwardSequenceReceipt() (Qwen35MetalForwardSequenceReceipt, bool)
}

// qwen35MetalDeviceKVAdmitter is implemented by the native sequence backend that can
// hold the full-attention KV prefix on the device across a panel walk (#13087). The
// walk owner allocates one DeviceKV pair for the whole prompt, threads it onto the
// backend for the walk, and downloads it once at the end. A backend that does not
// implement this keeps the historical host-append walk; the walk owner treats a
// decline (nil pair, or the backend not implementing the interface) as fail-open.
type qwen35MetalDeviceKVAdmitter interface {
	// AdmitDeviceKV sizes and attaches the walk's device KV pair. It returns nil when
	// the geometry is unsupported or the device allocation fails, in which case the
	// caller keeps the host walk with KV/state unmutated.
	AdmitDeviceKV(s *Session, tokens int) *metalgemm.DeviceKV
	// DetachDeviceKV clears the walk's pair and returns it so the caller can download
	// and free it. It never mutates host KV/state.
	DetachDeviceKV() *metalgemm.DeviceKV
	// ReconcileDeviceKV appends the triple's device rows for every full-attention
	// layer to the host cache exactly once, after the last panel. `base` is the host
	// prefix length the walk started from and `rows` the rows the panels appended. It
	// returns an error without mutating any host cache row if the pair's geometry
	// disagrees with the cache.
	ReconcileDeviceKV(s *Session, base, rows int) error
}

// qwen35MetalDeviceKVKeeper is the optional persistent-decode extension. Keeping
// it separate preserves panel-only admitters and their historical detach contract.
type qwen35MetalDeviceKVKeeper interface{ KeepDeviceKV(*Session) bool }

type qwen35MetalStateIdentityBinder interface {
	bindQwen35MetalStateIdentity(Qwen35MetalStateIdentityReceipt)
}

// EnableQwen35MetalGDNPreprojectedSequence admits the opt-in owner before any
// prompt state is mutated. A restored (non-fresh) session is admitted only when
// its host linear-attention state is a complete prefix state; the freshly
// allocated owners are then seeded from it so the sequence continues at
// base=Cache.Len(). Unsupported builds and incomplete restored state refuse
// explicitly (owners freed, host state untouched) instead of changing the
// requested path to host recurrence.
func (s *Session) EnableQwen35MetalGDNPreprojectedSequence() error {
	path := Qwen35MetalGDNSequenceForwardPath
	if s == nil || s.M == nil || !s.M.Cfg.IsQwen35Hybrid() {
		return &UnsupportedGDNPreprojectedSequenceError{Path: path, Reason: "session is not a Qwen hybrid"}
	}
	if s.M.prism != nil {
		return &UnsupportedGDNPreprojectedSequenceError{Path: path, Reason: "Prism Hadamard activation transform is not in the fused Metal graph"}
	}
	if s.Backend != nil || !s.Q4K || !s.MetalQ4K {
		return &UnsupportedGDNPreprojectedSequenceError{Path: path, Reason: "requires backend-nil resident-Q4_K Metal session"}
	}
	if s.Cache == nil {
		return &UnsupportedGDNPreprojectedSequenceError{Path: path, Reason: "requires a session cache before prompt-state mutation"}
	}
	if q := s.qwen35HAL; q != nil {
		// A non-fresh session is admissible now, so the prior-owner states the old
		// fresh-prompt check excluded must be refused explicitly. A recorded failure
		// stays the audit authority (re-admission would clear it), and a live owner
		// set (admitted sequence or promoted decode) holds state the host cache does
		// not; neither may be replaced by a fresh seed.
		switch {
		case q.sequenceFailure != nil:
			return &UnsupportedGDNPreprojectedSequenceError{Path: path, Reason: "session carries a recorded resident GDN failure: " + q.sequenceFailure.Error()}
		case q.decodeAccepted:
			return &UnsupportedGDNPreprojectedSequenceError{Path: path, Reason: "session already holds promoted resident GDN decode owners"}
		case q.sequenceAccepted || q.sequenceBackend != nil:
			return &UnsupportedGDNPreprojectedSequenceError{Path: path, Reason: "session already owns sequence auxiliary state"}
		}
	}
	var restored []qwen35GDNLayerSnapshot
	if s.Cache.Len() != 0 {
		var reason string
		if restored, reason = s.qwen35HostGDNSeedSnapshots(); reason != "" {
			return &UnsupportedGDNPreprojectedSequenceError{Path: path, Reason: "restored prefix cannot seed resident owners: " + reason}
		}
	}
	if newQwen35MetalGDNSequenceBackend == nil {
		return &UnsupportedGDNPreprojectedSequenceError{Path: path, Reason: "native Metal GDN sequence is unavailable in this build"}
	}
	if qwen35MetalForwardProjectionAdmission != nil {
		if err := qwen35MetalForwardProjectionAdmission(s); err != nil {
			return err
		}
	}
	accepted, err := s.initQwen35GDNPreprojectedSequence(newQwen35MetalGDNSequenceBackend())
	if !accepted && err == nil {
		err = &UnsupportedGDNPreprojectedSequenceError{Path: path, Reason: "native capability did not advertise the canonical sequence contract"}
	}
	if err != nil || s.Cache.Len() == 0 {
		return err
	}
	// Restored prefix: seed every freshly allocated (zero) owner from the host
	// recurrent/convolution state before any prompt token runs. A seed failure
	// releases the owners and leaves the host cache as the sole authority.
	if seedErr := s.seedQwen35GDNOwnersFromHost(restored); seedErr != nil {
		s.qwen35HAL.freeSequence()
		s.qwen35HAL.sequenceAccepted = false
		return &UnsupportedGDNPreprojectedSequenceError{Path: path, Reason: "restored prefix GDN state could not seed resident owners: " + seedErr.Error()}
	}
	return nil
}

// FinalizeQwen35MetalGDNPreprojectedSequence performs the one explicit state
// transfer required by the still-host-owned decode path. All layers are
// snapshotted and validated before any historical cache row is changed.
func (s *Session) FinalizeQwen35MetalGDNPreprojectedSequence() (bool, error) {
	if s == nil || s.qwen35HAL == nil || !s.qwen35HAL.sequenceAccepted {
		return false, nil
	}
	q := s.qwen35HAL
	if q.sequenceFailure != nil {
		return true, q.sequenceFailure
	}
	cfg := s.M.Cfg
	snapshots, failLayer, failStage, err := s.snapshotQwen35GDNOwners()
	if err != nil {
		return true, s.failQwen35GDNSequence(failLayer, failStage, err)
	}
	// Invariant: once the owners are read and validated, the host cache receives
	// the finalized prompt state BEFORE anything below can fail. Every later
	// failure therefore either records sequenceFailure (state identity) or leaves
	// a correct host cache (promotion/seed failure frees the owners), so no exit
	// -- including the auto route's discarded deferred Finalize -- leaves a stale
	// host Cache.linear that a snapshot or bare clone could publish. A bare host KV
	// clone at this boundary (prefix admission) carries the real recurrent state.
	writeQwen35GDNSnapshotsToHost(cfg, s.Cache, snapshots)
	var stateIdentity *Qwen35MetalStateIdentityReceipt
	if s.qwen35MetalStateIdentity != nil {
		if _, ok := q.sequenceBackend.(qwen35GDNStateSeeder); !ok {
			return len(snapshots) > 0, s.failQwen35GDNSequence(-1, "state identity accounting", fmt.Errorf("receipt-opted sequence backend cannot seed the finalized GDN owners"))
		}
		stateBytes := uint64(0)
		for _, snapshot := range snapshots {
			stateBytes += uint64(len(snapshot.conv)+len(snapshot.recurrent)) * 4
		}
		receipt, err := s.buildQwen35MetalStateIdentityReceipt(Qwen35MetalStateAuthoritySequence, snapshots, qwen35MetalStateIdentityAccounting{
			GDNSnapshotOps: len(snapshots), GDNSeedOps: len(snapshots),
			GDNStateD2HBytes: stateBytes, GDNStateH2DBytes: stateBytes,
		})
		if err != nil {
			return len(snapshots) > 0, s.failQwen35GDNSequence(-1, "state identity", err)
		}
		stateIdentity = &receipt
	}
	// Promoted owners become the decode authority; a promotion failure releases
	// them and the host cache written above is the sole, correct authority.
	selected, err := s.promoteQwen35MetalGDNDecode(snapshots)
	if err != nil {
		return len(snapshots) > 0, err
	}
	if selected && stateIdentity != nil {
		s.installQwen35MetalStateIdentityReceipt(*stateIdentity)
	}
	return len(snapshots) > 0, nil
}

func (s *Session) failQwen35MetalForwardSequence(cause error) error {
	err := &Qwen35GDNSequenceOperationError{Layer: -1, Stage: "whole-sequence forward", Cause: cause}
	if s == nil || s.qwen35HAL == nil {
		return err
	}
	q := s.qwen35HAL
	backend := q.sequenceBackend
	states := q.sequenceLayers
	q.sequenceLayers = nil
	q.metalForwardReceipt = nil
	for _, state := range states {
		if state.valid() && backend != nil {
			_ = backend.FreeQwen35GDNAuxState(state)
		}
	}
	// Retain only the receipt-bearing backend object so the accepted failure can
	// still be audited. It owns no live state after the loop above.
	q.sequenceAccepted = true
	q.decodeAccepted = false
	q.sequenceFailure = err
	return err
}

// Qwen35MetalForwardSequenceStatus returns model-authored selection and support
// state even when no terminal execution receipt exists. The status is separate
// from Qwen35MetalForwardSequenceReceipt so historical execution-only callers
// retain the zero-value absence contract.
func (s *Session) Qwen35MetalForwardSequenceStatus() Qwen35MetalForwardSequenceReceipt {
	base := Qwen35MetalForwardSequenceReceipt{
		Path:          Qwen35MetalGDNSequenceForwardPath,
		SelectorState: Qwen35MetalSequenceSelectorOff,
		EvidenceState: Qwen35MetalSequenceEvidenceUnsupported,
	}
	if s == nil || s.M == nil || !s.M.Cfg.IsQwen35Hybrid() || s.Backend != nil || !s.Q4K || !s.MetalQ4K {
		return base
	}
	// A stored whole-sequence receipt is execution evidence regardless of whether
	// the native backend factory is linked into THIS build: the Go-only panel-walk
	// double deliberately reports through this seam on every host
	// (TestPrefillPanelRoundTripBudget). Only fall to the build-availability status
	// when the session holds no such receipt.
	if s.qwen35HAL != nil && (s.qwen35HAL.sequenceAccepted || s.qwen35HAL.decodeAccepted) {
		if agg, ok := s.qwen35HAL.getMetalForwardReceipt(); ok && agg.Available {
			agg.SelectorState = Qwen35MetalSequenceSelectorOn
			if runner, ok := s.qwen35HAL.sequenceBackend.(qwen35MetalForwardSequenceRunner); ok {
				if r, ok := runner.Qwen35MetalForwardSequenceReceipt(); ok && r.StateIdentity != nil {
					if agg.StateIdentity == nil || agg.StateIdentity.BindingSHA256 != r.StateIdentity.BindingSHA256 {
						agg.StateIdentity = r.StateIdentity
						agg.HostUploadBytes += r.StateIdentity.GDNStateH2DBytes
						agg.HostReadbackBytes += r.StateIdentity.GDNStateD2HBytes
					}
				}
			}
			return agg
		}
	}
	if newQwen35MetalGDNSequenceBackend == nil {
		base.EvidenceState = Qwen35MetalSequenceEvidenceUnavailable
		return base
	}
	base.EvidenceState = Qwen35MetalSequenceEvidenceNotSelected
	if s.qwen35HAL == nil || !s.qwen35HAL.sequenceAccepted && !s.qwen35HAL.decodeAccepted {
		return base
	}
	base.SelectorState = Qwen35MetalSequenceSelectorOn
	base.EvidenceState = Qwen35MetalSequenceEvidenceUnavailable
	runner, ok := s.qwen35HAL.sequenceBackend.(qwen35MetalForwardSequenceRunner)
	if !ok {
		return base
	}
	receipt, ok := runner.Qwen35MetalForwardSequenceReceipt()
	if !ok {
		return base
	}
	receipt.SelectorState = Qwen35MetalSequenceSelectorOn
	receipt.EvidenceState = Qwen35MetalSequenceEvidenceExecuted
	return receipt
}

// Qwen35MetalForwardSequenceReceipt returns the last terminal whole-sequence
// receipt as a scalar-only value snapshot. The zero value has Available=false
// and means this session has no such execution evidence.
func (s *Session) Qwen35MetalForwardSequenceReceipt() Qwen35MetalForwardSequenceReceipt {
	receipt := s.Qwen35MetalForwardSequenceStatus()
	if receipt.EvidenceState != Qwen35MetalSequenceEvidenceExecuted {
		return Qwen35MetalForwardSequenceReceipt{}
	}
	return receipt
}
