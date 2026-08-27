package qwen4exp

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"time"
)

const StateSnapshotSchema = "fak-qwen4exp-state/1"

var (
	ErrStateCorrupt  = errors.New("qwen4exp state: corrupt snapshot")
	ErrStateStale    = errors.New("qwen4exp state: stale position")
	ErrStateIdentity = errors.New("qwen4exp state: identity mismatch")
)

// StateIdentity binds managed recurrent state to one exact model execution path.
type StateIdentity struct {
	Engine             string `json:"engine"`
	Artifact           string `json:"artifact"`
	CheckpointRevision string `json:"checkpoint_revision"`
	ManifestSHA256     string `json:"manifest_sha256"`
}

// ManagedState captures all state that affects QSA/GDN continuation semantics.
type ManagedState struct {
	Position      uint64            `json:"position"`
	GDN           [][]float32       `json:"gdn_fp32"`
	QSAIndices    []uint32          `json:"qsa_indices"`
	QSASelection  []float32         `json:"qsa_selection"`
	CacheMetadata map[string]uint64 `json:"cache_metadata"`
}

// StateSnapshot is the portable, checksummed split-run boundary.
type StateSnapshot struct {
	Schema        string        `json:"schema"`
	Identity      StateIdentity `json:"identity"`
	State         ManagedState  `json:"state"`
	CreatedAt     time.Time     `json:"created_at"`
	PayloadSHA256 string        `json:"payload_sha256"`
}

type statePayload struct {
	Schema    string        `json:"schema"`
	Identity  StateIdentity `json:"identity"`
	State     ManagedState  `json:"state"`
	CreatedAt time.Time     `json:"created_at"`
}

func (s StateSnapshot) payload() statePayload {
	return statePayload{s.Schema, s.Identity, s.State, s.CreatedAt}
}

func NewStateSnapshot(id StateIdentity, state ManagedState, now time.Time) (StateSnapshot, error) {
	s := StateSnapshot{Schema: StateSnapshotSchema, Identity: id, State: cloneManagedState(state), CreatedAt: now.UTC()}
	if err := validateStateIdentity(id); err != nil {
		return StateSnapshot{}, err
	}
	sum, err := snapshotDigest(s.payload())
	if err != nil {
		return StateSnapshot{}, err
	}
	s.PayloadSHA256 = sum
	return s, nil
}

func (s StateSnapshot) Marshal() ([]byte, error) { return json.Marshal(s) }

func ParseStateSnapshot(raw []byte) (StateSnapshot, error) {
	var s StateSnapshot
	if err := json.Unmarshal(raw, &s); err != nil {
		return StateSnapshot{}, fmt.Errorf("%w: %v", ErrStateCorrupt, err)
	}
	if s.Schema != StateSnapshotSchema {
		return StateSnapshot{}, fmt.Errorf("%w: schema %q", ErrStateCorrupt, s.Schema)
	}
	if err := validateStateIdentity(s.Identity); err != nil {
		return StateSnapshot{}, err
	}
	sum, err := snapshotDigest(s.payload())
	if err != nil {
		return StateSnapshot{}, err
	}
	if s.PayloadSHA256 == "" || sum != s.PayloadSHA256 {
		return StateSnapshot{}, fmt.Errorf("%w: payload digest", ErrStateCorrupt)
	}
	s.State = cloneManagedState(s.State)
	return s, nil
}

// RestoreState checks identity and monotonic position before returning an owned copy.
func (s StateSnapshot) RestoreState(want StateIdentity, minimumPosition uint64) (ManagedState, error) {
	if s.Identity != want {
		return ManagedState{}, fmt.Errorf("%w: got %+v", ErrStateIdentity, s.Identity)
	}
	if s.State.Position < minimumPosition {
		return ManagedState{}, fmt.Errorf("%w: position %d before %d", ErrStateStale, s.State.Position, minimumPosition)
	}
	return cloneManagedState(s.State), nil
}

// StateReceipt accounts the immutable continuation boundary.
type StateReceipt struct {
	Schema             string `json:"schema"`
	Engine             string `json:"engine"`
	Artifact           string `json:"artifact"`
	CheckpointRevision string `json:"checkpoint_revision"`
	ContextPosition    uint64 `json:"context_position"`
	StateBytes         int    `json:"state_bytes"`
	RestoreNanoseconds int64  `json:"restore_nanoseconds"`
}

func (s StateSnapshot) Receipt(encodedBytes int, restoreLatency time.Duration) StateReceipt {
	return StateReceipt{"fak-qwen4exp-state-receipt/1", s.Identity.Engine, s.Identity.Artifact, s.Identity.CheckpointRevision, s.State.Position, encodedBytes, restoreLatency.Nanoseconds()}
}

func snapshotDigest(p statePayload) (string, error) {
	raw, err := json.Marshal(p)
	if err != nil {
		return "", fmt.Errorf("%w: %v", ErrStateCorrupt, err)
	}
	sum := sha256.Sum256(raw)
	return "sha256:" + hex.EncodeToString(sum[:]), nil
}
func validateStateIdentity(id StateIdentity) error {
	if id.Engine == "" || id.Artifact == "" || id.CheckpointRevision == "" || id.ManifestSHA256 == "" {
		return fmt.Errorf("%w: incomplete identity", ErrStateIdentity)
	}
	return nil
}
func cloneManagedState(s ManagedState) ManagedState {
	out := s
	out.GDN = make([][]float32, len(s.GDN))
	for i := range s.GDN {
		out.GDN[i] = append([]float32(nil), s.GDN[i]...)
	}
	out.QSAIndices = append([]uint32(nil), s.QSAIndices...)
	out.QSASelection = append([]float32(nil), s.QSASelection...)
	if s.CacheMetadata != nil {
		out.CacheMetadata = make(map[string]uint64, len(s.CacheMetadata))
		for k, v := range s.CacheMetadata {
			out.CacheMetadata[k] = v
		}
	}
	return out
}
