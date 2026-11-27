package agent

import (
	"context"
	"errors"
	"fmt"
	"slices"
	"sync"
	"time"
)

// StartupArtifactKind identifies one immutable model startup artifact published
// to a shared Cairn-style pool (arXiv:2611.16180, KAIST + SK hynix).
type StartupArtifactKind string

const (
	ArtifactWeights         StartupArtifactKind = "weights"
	ArtifactMetadata        StartupArtifactKind = "metadata"
	ArtifactCompileCache    StartupArtifactKind = "compile_cache"
	ArtifactCompilerKey     StartupArtifactKind = "compiler_key"
	ArtifactCUDAGraphMemEst StartupArtifactKind = "cuda_graph_mem_estimate"
	ArtifactPrewarmedEnv    StartupArtifactKind = "prewarmed_env"
)

// ErrArtifactImmutable is returned when a caller attempts to overwrite an
// already-published immutable startup artifact key.
var ErrArtifactImmutable = errors.New("agent: startup artifact key is immutable and already published")

// ErrArtifactKeyInvalid is returned when a startup artifact key is incomplete.
var ErrArtifactKeyInvalid = errors.New("agent: invalid startup artifact key")

// StartupArtifactKey uniquely addresses an immutable startup artifact.
type StartupArtifactKey struct {
	ModelID           string              `json:"model_id"`
	RuntimeGeneration string              `json:"runtime_generation"`
	Kind              StartupArtifactKind `json:"kind"`
	Variant           string              `json:"variant,omitempty"`
}

func (k StartupArtifactKey) validate() error {
	if k.ModelID == "" || k.RuntimeGeneration == "" || k.Kind == "" {
		return ErrArtifactKeyInvalid
	}
	return nil
}

func (k StartupArtifactKey) stringKey() string {
	return fmt.Sprintf("%s|%s|%s|%s", k.ModelID, k.RuntimeGeneration, k.Kind, k.Variant)
}

// StartupArtifactEntry is an immutable entry stored in WarmArtifactPool.
type StartupArtifactEntry struct {
	Key         StartupArtifactKey
	payload     []byte
	PublishedBy string
	PublishedAt time.Time
}

// PayloadCopy returns a defensive copy of the immutable artifact bytes.
func (e StartupArtifactEntry) PayloadCopy() []byte {
	return slices.Clone(e.payload)
}

// PayloadLen returns the byte length of the artifact without copying.
func (e StartupArtifactEntry) PayloadLen() int {
	return len(e.payload)
}

// WarmArtifactPool is an in-memory shared pool of immutable model startup
// artifacts. The first server to start a model publishes weights, metadata,
// compile cache, compiler keys, and CUDA-graph memory estimates; peer servers
// read them in place without repeating local compile/profile/setup.
type WarmArtifactPool struct {
	mu      sync.RWMutex
	entries map[string]StartupArtifactEntry
}

// NewWarmArtifactPool creates an empty shared startup artifact pool.
func NewWarmArtifactPool() *WarmArtifactPool {
	return &WarmArtifactPool{
		entries: make(map[string]StartupArtifactEntry),
	}
}

// Publish stores payload under key as an immutable entry. A duplicate publish of
// the same key is refused with ErrArtifactImmutable without mutating the stored
// entry.
func (p *WarmArtifactPool) Publish(key StartupArtifactKey, payload []byte, nodeID string) error {
	if err := key.validate(); err != nil {
		return err
	}
	if nodeID == "" {
		return errors.New("agent: publisher nodeID is required")
	}
	sk := key.stringKey()
	p.mu.Lock()
	defer p.mu.Unlock()
	if _, exists := p.entries[sk]; exists {
		return ErrArtifactImmutable
	}
	p.entries[sk] = StartupArtifactEntry{
		Key:         key,
		payload:     slices.Clone(payload),
		PublishedBy: nodeID,
		PublishedAt: time.Now().UTC(),
	}
	return nil
}

// ReadInPlace looks up key in the pool. Callers receive an immutable entry
// handle without triggering local setup.
func (p *WarmArtifactPool) ReadInPlace(key StartupArtifactKey) (StartupArtifactEntry, bool) {
	if key.validate() != nil {
		return StartupArtifactEntry{}, false
	}
	p.mu.RLock()
	defer p.mu.RUnlock()
	e, ok := p.entries[key.stringKey()]
	return e, ok
}

// StartupArtifactResult summarizes whether a node's startup consumed pooled
// immutable entries in place or fell back to local setup.
type StartupArtifactResult struct {
	Hits        int
	Misses      int
	LocalSetups int
	Entries     []StartupArtifactEntry
}

// LocalArtifactSetupFn builds an artifact locally on a pool miss.
type LocalArtifactSetupFn func(ctx context.Context, key StartupArtifactKey) ([]byte, error)

// StartupOrReuse resolves each requested key from the shared pool in place.
// On a pool hit, localSetup is skipped. On a pool miss, localSetup runs and
// publishes the result to the pool (if a peer published concurrently, the peer's
// immutable entry is read in place).
func (p *WarmArtifactPool) StartupOrReuse(ctx context.Context, nodeID string, keys []StartupArtifactKey, localSetup LocalArtifactSetupFn) (StartupArtifactResult, error) {
	var res StartupArtifactResult
	for _, key := range keys {
		if err := key.validate(); err != nil {
			return res, err
		}
		if entry, ok := p.ReadInPlace(key); ok {
			res.Hits++
			res.Entries = append(res.Entries, entry)
			continue
		}
		res.Misses++
		if localSetup == nil {
			return res, fmt.Errorf("agent: pool miss for %s and no local setup provided", key.stringKey())
		}
		payload, err := localSetup(ctx, key)
		if err != nil {
			return res, err
		}
		res.LocalSetups++
		pubErr := p.Publish(key, payload, nodeID)
		if pubErr != nil && !errors.Is(pubErr, ErrArtifactImmutable) {
			return res, pubErr
		}
		entry, ok := p.ReadInPlace(key)
		if !ok {
			return res, fmt.Errorf("agent: published entry %s missing on readback", key.stringKey())
		}
		res.Entries = append(res.Entries, entry)
	}
	return res, nil
}
