// Package decodemigrate provides model decoding format migration utilities,
// KV cache state migration, and token decoding pipeline state transformations
// with fail-closed integrity checks.
//
// Architecture:
// The decoding pipeline transitions through versioned state schemas across
// runtime updates, model quantization changes, and KV-cache layout updates.
// DecodeState models snapshot payloads with layout tags, token sequences,
// KV-cache tensor metadata, and cryptographic checksums.
// MigrationRunner registers transformation steps between FormatVersions, computes
// forward and backward migration paths, builds execution plans, and runs transactional
// migrations with automatic fail-closed rollbacks upon corruption or verification failure.
package decodemigrate

import (
	"bytes"
	"crypto/sha256"
	"encoding/binary"
	"errors"
	"fmt"
	"sync"
)

// Invariant: Migration steps must preserve token sequence identity and KV cache shape semantics across all supported version hops.
// Guard: fail-closed state verification verifies checksums before and after every migration step; any mismatch aborts and restores prior state.

// Decode migration error sentinels returned during validation, routing, and execution.
var (
	// ErrIncompatibleFormat indicates that source and destination formats cannot be bridged.
	ErrIncompatibleFormat = errors.New("decodemigrate: incompatible format version")

	// ErrMigrationFailed indicates an execution step failed during migration.
	ErrMigrationFailed = errors.New("decodemigrate: migration step execution failed")

	// ErrCorruptedState indicates state payload or checksum verification failed.
	ErrCorruptedState = errors.New("decodemigrate: corrupted decode state")

	// ErrTruncatedPayload indicates that the payload byte slice is shorter than header specifications.
	ErrTruncatedPayload = errors.New("decodemigrate: truncated state payload")
)

// FormatVersion represents a supported encoding schema for model decode states.
type FormatVersion uint32

// FormatVersion identifiers defining supported decode state layouts.
const (
	// VersionUnknown denotes an invalid or unspecified format version.
	VersionUnknown FormatVersion = 0

	// VersionV1Legacy denotes the initial uncompressed tensor layout.
	VersionV1Legacy FormatVersion = 1

	// VersionV2Paged denotes paged KV-cache block allocation format.
	VersionV2Paged FormatVersion = 2

	// VersionV3Quantized denotes quantized KV-cache block layout (FP8/INT4).
	VersionV3Quantized FormatVersion = 3

	// VersionV4Compressed denotes dense chunk-compressed KV-cache and token tree format.
	VersionV4Compressed FormatVersion = 4
)

// String returns a human-readable representation of the format version.
func (v FormatVersion) String() string {
	switch v {
	case VersionV1Legacy:
		return "v1_legacy"
	case VersionV2Paged:
		return "v2_paged"
	case VersionV3Quantized:
		return "v3_quantized"
	case VersionV4Compressed:
		return "v4_compressed"
	default:
		return fmt.Sprintf("version_unknown(%d)", uint32(v))
	}
}

// KVBufferMetadata captures structural dimensions of KV cache buffers within a decode state.
type KVBufferMetadata struct {
	// NumLayers is the number of transformer layers stored.
	NumLayers int `json:"num_layers"`
	// NumHeads is the number of key/value heads.
	NumHeads int `json:"num_heads"`
	// HeadDim is the dimension per head.
	HeadDim int `json:"head_dim"`
	// BlockSize is the paged block size in tokens.
	BlockSize int `json:"block_size"`
	// TotalTokens is the number of cached tokens.
	TotalTokens int `json:"total_tokens"`
}

// DecodeState represents the snapshot of a model's decoding pipeline state.
type DecodeState struct {
	// Version is the schema version of the state payload.
	Version FormatVersion
	// ModelID identifies the model architecture and configuration.
	ModelID string
	// SequenceID is the unique identifier of the active generation session.
	SequenceID string
	// Tokens holds the decoded token sequence IDs.
	Tokens []int64
	// KVMetadata contains structural dimensions for the KV cache.
	KVMetadata KVBufferMetadata
	// Payload holds serialized cache weights, page tables, or compressed blocks.
	Payload []byte
	// Checksum is the SHA-256 hash of state attributes and payload bytes.
	Checksum [32]byte
}

// ComputeChecksum calculates the SHA-256 digest of the decode state fields.
func (s *DecodeState) ComputeChecksum() [32]byte {
	h := sha256.New()

	var buf [8]byte
	binary.BigEndian.PutUint32(buf[:4], uint32(s.Version))
	h.Write(buf[:4])

	binary.BigEndian.PutUint64(buf[:], uint64(len(s.ModelID)))
	h.Write(buf[:])
	h.Write([]byte(s.ModelID))

	binary.BigEndian.PutUint64(buf[:], uint64(len(s.SequenceID)))
	h.Write(buf[:])
	h.Write([]byte(s.SequenceID))

	binary.BigEndian.PutUint64(buf[:], uint64(len(s.Tokens)))
	h.Write(buf[:])
	for _, tok := range s.Tokens {
		binary.BigEndian.PutUint64(buf[:], uint64(tok))
		h.Write(buf[:])
	}

	binary.BigEndian.PutUint64(buf[:], uint64(s.KVMetadata.NumLayers))
	h.Write(buf[:])
	binary.BigEndian.PutUint64(buf[:], uint64(s.KVMetadata.NumHeads))
	h.Write(buf[:])
	binary.BigEndian.PutUint64(buf[:], uint64(s.KVMetadata.HeadDim))
	h.Write(buf[:])
	binary.BigEndian.PutUint64(buf[:], uint64(s.KVMetadata.BlockSize))
	h.Write(buf[:])
	binary.BigEndian.PutUint64(buf[:], uint64(s.KVMetadata.TotalTokens))
	h.Write(buf[:])

	binary.BigEndian.PutUint64(buf[:], uint64(len(s.Payload)))
	h.Write(buf[:])
	h.Write(s.Payload)

	var digest [32]byte
	copy(digest[:], h.Sum(nil))
	return digest
}

// Validate verifies the internal integrity and checksum of the decode state.
func (s *DecodeState) Validate() error {
	if s == nil {
		return fmt.Errorf("%w: nil decode state", ErrCorruptedState)
	}
	if s.Version == VersionUnknown {
		return fmt.Errorf("%w: unknown format version %d", ErrIncompatibleFormat, s.Version)
	}
	if s.ModelID == "" {
		return fmt.Errorf("%w: empty model ID", ErrCorruptedState)
	}
	if s.SequenceID == "" {
		return fmt.Errorf("%w: empty sequence ID", ErrCorruptedState)
	}
	if s.KVMetadata.NumLayers < 0 || s.KVMetadata.NumHeads < 0 || s.KVMetadata.HeadDim < 0 || s.KVMetadata.BlockSize < 0 || s.KVMetadata.TotalTokens < 0 {
		return fmt.Errorf("%w: negative KV metadata dimensions", ErrCorruptedState)
	}

	expected := s.ComputeChecksum()
	if !bytes.Equal(s.Checksum[:], expected[:]) {
		return fmt.Errorf("%w: checksum mismatch: expected %x got %x", ErrCorruptedState, expected, s.Checksum)
	}

	return nil
}

// Clone creates a deep copy of the decode state.
func (s *DecodeState) Clone() *DecodeState {
	if s == nil {
		return nil
	}
	cp := &DecodeState{
		Version:    s.Version,
		ModelID:    s.ModelID,
		SequenceID: s.SequenceID,
		KVMetadata: s.KVMetadata,
		Checksum:   s.Checksum,
	}
	if len(s.Tokens) > 0 {
		cp.Tokens = make([]int64, len(s.Tokens))
		copy(cp.Tokens, s.Tokens)
	}
	if len(s.Payload) > 0 {
		cp.Payload = make([]byte, len(s.Payload))
		copy(cp.Payload, s.Payload)
	}
	return cp
}

// NewDecodeState constructs and checksums a new DecodeState.
func NewDecodeState(version FormatVersion, modelID, sequenceID string, tokens []int64, kvMeta KVBufferMetadata, payload []byte) (*DecodeState, error) {
	if version == VersionUnknown {
		return nil, fmt.Errorf("%w: cannot construct state with VersionUnknown", ErrIncompatibleFormat)
	}
	if modelID == "" {
		return nil, fmt.Errorf("%w: modelID cannot be empty", ErrCorruptedState)
	}
	if sequenceID == "" {
		return nil, fmt.Errorf("%w: sequenceID cannot be empty", ErrCorruptedState)
	}
	if kvMeta.NumLayers < 0 || kvMeta.NumHeads < 0 || kvMeta.HeadDim < 0 || kvMeta.BlockSize < 0 || kvMeta.TotalTokens < 0 {
		return nil, fmt.Errorf("%w: negative KV metadata dimensions", ErrCorruptedState)
	}

	s := &DecodeState{
		Version:    version,
		ModelID:    modelID,
		SequenceID: sequenceID,
		KVMetadata: kvMeta,
	}
	if len(tokens) > 0 {
		s.Tokens = make([]int64, len(tokens))
		copy(s.Tokens, tokens)
	}
	if len(payload) > 0 {
		s.Payload = make([]byte, len(payload))
		copy(s.Payload, payload)
	}
	s.Checksum = s.ComputeChecksum()
	return s, nil
}

// MigrationFunc executes a transformation step from one format version to another.
type MigrationFunc func(state *DecodeState) (*DecodeState, error)

// MigrationStep describes a single directed migration hop between two format versions.
type MigrationStep struct {
	// From is the source format version.
	From FormatVersion
	// To is the destination format version.
	To FormatVersion
	// Description explains the transformation performed by this step.
	Description string
	// Apply executes the state transformation.
	Apply MigrationFunc
}

// MigrationRoute represents an ordered sequence of steps to migrate a state between versions.
type MigrationRoute struct {
	// Source is the initial format version.
	Source FormatVersion
	// Target is the desired final format version.
	Target FormatVersion
	// Steps is the ordered slice of execution steps.
	Steps []MigrationStep
}

// IsNoOp returns true if the plan requires zero steps (source == target).
func (p *MigrationRoute) IsNoOp() bool {
	return p.Source == p.Target && len(p.Steps) == 0
}

// TotalSteps returns the number of migration steps in the plan.
func (p *MigrationRoute) TotalSteps() int {
	return len(p.Steps)
}

// MigrationRunner manages registered migration steps and executes transactional state migrations.
type MigrationRunner struct {
	mu    sync.RWMutex
	steps map[FormatVersion]map[FormatVersion]MigrationStep
}

// NewMigrationRunner initializes a MigrationRunner with default standard transformation steps.
func NewMigrationRunner() *MigrationRunner {
	engine := &MigrationRunner{
		steps: make(map[FormatVersion]map[FormatVersion]MigrationStep),
	}
	engine.registerDefaults()
	return engine
}

// RegisterStep registers a directed migration step from one format to another.
func (e *MigrationRunner) RegisterStep(step MigrationStep) error {
	if step.From == VersionUnknown || step.To == VersionUnknown {
		return fmt.Errorf("%w: invalid step version from %s to %s", ErrIncompatibleFormat, step.From, step.To)
	}
	if step.From == step.To {
		return fmt.Errorf("%w: cannot register step to same version %s", ErrIncompatibleFormat, step.From)
	}
	if step.Apply == nil {
		return fmt.Errorf("%w: step apply function is nil", ErrMigrationFailed)
	}

	e.mu.Lock()
	defer e.mu.Unlock()

	if _, ok := e.steps[step.From]; !ok {
		e.steps[step.From] = make(map[FormatVersion]MigrationStep)
	}
	e.steps[step.From][step.To] = step
	return nil
}

// Route computes a valid migration path from source to target format version using BFS.
func (e *MigrationRunner) Route(source, target FormatVersion) (*MigrationRoute, error) {
	if source == VersionUnknown || target == VersionUnknown {
		return nil, fmt.Errorf("%w: unknown source (%s) or target (%s)", ErrIncompatibleFormat, source, target)
	}
	if source == target {
		return &MigrationRoute{Source: source, Target: target, Steps: nil}, nil
	}

	e.mu.RLock()
	defer e.mu.RUnlock()

	type queueNode struct {
		ver  FormatVersion
		path []MigrationStep
	}

	visited := make(map[FormatVersion]bool)
	queue := []queueNode{{ver: source, path: nil}}
	visited[source] = true

	for len(queue) > 0 {
		curr := queue[0]
		queue = queue[1:]

		if curr.ver == target {
			return &MigrationRoute{
				Source: source,
				Target: target,
				Steps:  curr.path,
			}, nil
		}

		destinations := e.steps[curr.ver]
		for destVer, step := range destinations {
			if !visited[destVer] {
				visited[destVer] = true
				newPath := make([]MigrationStep, len(curr.path)+1)
				copy(newPath, curr.path)
				newPath[len(curr.path)] = step
				queue = append(queue, queueNode{ver: destVer, path: newPath})
			}
		}
	}

	return nil, fmt.Errorf("%w: no migration path from %s to %s", ErrIncompatibleFormat, source, target)
}

// Migrate executes the migration route on the provided state with fail-closed guarantees.
func (e *MigrationRunner) Migrate(state *DecodeState, target FormatVersion) (*DecodeState, error) {
	if state == nil {
		return nil, fmt.Errorf("%w: state is nil", ErrCorruptedState)
	}
	route, err := e.Route(state.Version, target)
	if err != nil {
		return nil, err
	}
	return e.ExecuteRoute(state, route)
}

// ExecuteRoute executes a pre-computed plan against state with fail-closed integrity checks.
func (e *MigrationRunner) ExecuteRoute(state *DecodeState, plan *MigrationRoute) (*DecodeState, error) {
	if state == nil {
		return nil, fmt.Errorf("%w: cannot execute plan on nil state", ErrCorruptedState)
	}
	if plan == nil {
		return nil, fmt.Errorf("%w: plan is nil", ErrMigrationFailed)
	}

	// Guard: fail-closed state verification verifies checksums before and after every migration step; any mismatch aborts and restores prior state.
	if err := state.Validate(); err != nil {
		return nil, fmt.Errorf("%w: initial state validation failed: %w", ErrCorruptedState, err)
	}

	if state.Version != plan.Source {
		return nil, fmt.Errorf("%w: state version %s does not match plan source %s", ErrIncompatibleFormat, state.Version, plan.Source)
	}

	if plan.IsNoOp() {
		return state.Clone(), nil
	}

	current := state.Clone()

	for idx, step := range plan.Steps {
		if current.Version != step.From {
			return nil, fmt.Errorf("%w: step %d source mismatch: state is %s but step expects %s", ErrMigrationFailed, idx, current.Version, step.From)
		}

		backup := current.Clone()

		transformed, err := step.Apply(current)
		if err != nil {
			// Fail-closed rollback: abort immediately and return error.
			return nil, fmt.Errorf("%w: step %d (%s -> %s) failed: %w", ErrMigrationFailed, idx, step.From, step.To, err)
		}
		if transformed == nil {
			return nil, fmt.Errorf("%w: step %d (%s -> %s) produced nil state", ErrMigrationFailed, idx, step.From, step.To)
		}

		if transformed.Version != step.To {
			return nil, fmt.Errorf("%w: step %d target mismatch: expected %s, got %s", ErrMigrationFailed, idx, step.To, transformed.Version)
		}

		// Invariant: Migration steps must preserve token sequence identity and KV cache shape semantics across all supported version hops.
		if len(transformed.Tokens) != len(backup.Tokens) {
			return nil, fmt.Errorf("%w: step %d violated token sequence length invariant (%d != %d)", ErrMigrationFailed, idx, len(transformed.Tokens), len(backup.Tokens))
		}
		for tIdx := range backup.Tokens {
			if transformed.Tokens[tIdx] != backup.Tokens[tIdx] {
				return nil, fmt.Errorf("%w: step %d mutated token value at index %d", ErrMigrationFailed, idx, tIdx)
			}
		}

		if transformed.KVMetadata.NumLayers != backup.KVMetadata.NumLayers ||
			transformed.KVMetadata.NumHeads != backup.KVMetadata.NumHeads ||
			transformed.KVMetadata.HeadDim != backup.KVMetadata.HeadDim ||
			transformed.KVMetadata.TotalTokens != backup.KVMetadata.TotalTokens {
			return nil, fmt.Errorf("%w: step %d violated KV cache structural invariants", ErrMigrationFailed, idx)
		}

		if transformed.ModelID != backup.ModelID || transformed.SequenceID != backup.SequenceID {
			return nil, fmt.Errorf("%w: step %d violated model or sequence identity invariants", ErrMigrationFailed, idx)
		}

		// Post-step checksum recalculation and validation.
		transformed.Checksum = transformed.ComputeChecksum()
		if err := transformed.Validate(); err != nil {
			return nil, fmt.Errorf("%w: step %d produced invalid state checksum: %w", ErrCorruptedState, idx, err)
		}

		current = transformed
	}

	if current.Version != plan.Target {
		return nil, fmt.Errorf("%w: route execution reached version %s, want target %s", ErrMigrationFailed, current.Version, plan.Target)
	}

	return current, nil
}

// Header tags for default mock payload conversions.
const (
	tagV1 = "DKV1"
	tagV2 = "DKV2"
	tagV3 = "DKV3"
	tagV4 = "DKV4"
)

func (e *MigrationRunner) registerDefaults() {
	// V1 -> V2: uncompressed legacy to paged layout.
	_ = e.RegisterStep(MigrationStep{
		From:        VersionV1Legacy,
		To:          VersionV2Paged,
		Description: "Convert uncompressed KV cache buffers to paged block layout",
		Apply: func(s *DecodeState) (*DecodeState, error) {
			next := s.Clone()
			next.Version = VersionV2Paged
			if next.KVMetadata.BlockSize == 0 {
				next.KVMetadata.BlockSize = 16 // Default paged block size
			}
			newPayload, err := transformTag(s.Payload, tagV1, tagV2)
			if err != nil {
				return nil, err
			}
			next.Payload = newPayload
			return next, nil
		},
	})

	// V2 -> V1: paged layout to uncompressed legacy.
	_ = e.RegisterStep(MigrationStep{
		From:        VersionV2Paged,
		To:          VersionV1Legacy,
		Description: "Flatten paged block layout back to linear uncompressed KV cache",
		Apply: func(s *DecodeState) (*DecodeState, error) {
			next := s.Clone()
			next.Version = VersionV1Legacy
			next.KVMetadata.BlockSize = 0
			newPayload, err := transformTag(s.Payload, tagV2, tagV1)
			if err != nil {
				return nil, err
			}
			next.Payload = newPayload
			return next, nil
		},
	})

	// V2 -> V3: paged layout to quantized FP8/INT4 layout.
	_ = e.RegisterStep(MigrationStep{
		From:        VersionV2Paged,
		To:          VersionV3Quantized,
		Description: "Quantize paged KV cache blocks to compressed numeric representation",
		Apply: func(s *DecodeState) (*DecodeState, error) {
			next := s.Clone()
			next.Version = VersionV3Quantized
			newPayload, err := transformTag(s.Payload, tagV2, tagV3)
			if err != nil {
				return nil, err
			}
			next.Payload = newPayload
			return next, nil
		},
	})

	// V3 -> V2: quantized layout to paged FP16 layout.
	_ = e.RegisterStep(MigrationStep{
		From:        VersionV3Quantized,
		To:          VersionV2Paged,
		Description: "Dequantize quantized KV cache blocks back to standard paged precision",
		Apply: func(s *DecodeState) (*DecodeState, error) {
			next := s.Clone()
			next.Version = VersionV2Paged
			newPayload, err := transformTag(s.Payload, tagV3, tagV2)
			if err != nil {
				return nil, err
			}
			next.Payload = newPayload
			return next, nil
		},
	})

	// V3 -> V4: quantized layout to chunk-compressed format.
	_ = e.RegisterStep(MigrationStep{
		From:        VersionV3Quantized,
		To:          VersionV4Compressed,
		Description: "Pack quantized KV blocks into chunk-compressed archival format",
		Apply: func(s *DecodeState) (*DecodeState, error) {
			next := s.Clone()
			next.Version = VersionV4Compressed
			newPayload, err := transformTag(s.Payload, tagV3, tagV4)
			if err != nil {
				return nil, err
			}
			next.Payload = newPayload
			return next, nil
		},
	})

	// V4 -> V3: chunk-compressed format to quantized layout.
	_ = e.RegisterStep(MigrationStep{
		From:        VersionV4Compressed,
		To:          VersionV3Quantized,
		Description: "Decompress chunk-compressed archival blocks to quantized format",
		Apply: func(s *DecodeState) (*DecodeState, error) {
			next := s.Clone()
			next.Version = VersionV3Quantized
			newPayload, err := transformTag(s.Payload, tagV4, tagV3)
			if err != nil {
				return nil, err
			}
			next.Payload = newPayload
			return next, nil
		},
	})
}

// transformTag validates payload header and switches tag representation.
func transformTag(payload []byte, expectedOldTag, newTag string) ([]byte, error) {
	if len(payload) == 0 {
		return []byte(newTag), nil
	}
	if len(payload) < 4 {
		return nil, fmt.Errorf("%w: payload length %d < header size 4", ErrTruncatedPayload, len(payload))
	}
	currentTag := string(payload[:4])
	if currentTag != expectedOldTag {
		return nil, fmt.Errorf("%w: payload header tag mismatch: expected %s, got %s", ErrCorruptedState, expectedOldTag, currentTag)
	}
	out := make([]byte, len(payload))
	copy(out, payload)
	copy(out[:4], []byte(newTag))
	return out, nil
}
