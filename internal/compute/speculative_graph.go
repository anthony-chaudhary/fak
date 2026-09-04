package compute

import (
	"errors"
	"fmt"
	"sync"
	"time"
)

// GraphCaptureKind classifies the operational role of a device execution graph.
type GraphCaptureKind string

const (
	// GraphCapturePrimary identifies a graph captured for primary prompt prefill / chunked execution.
	GraphCapturePrimary GraphCaptureKind = "primary_prompt"

	// GraphCaptureSpeculative identifies a graph captured for speculative draft verification.
	GraphCaptureSpeculative GraphCaptureKind = "speculative_draft"
)

// GraphCaptureKey represents a stable lookup key for captured device execution graphs.
// It decouples the fixed micro-batch capture dimension for speculative draft verification
// from dynamic prompt chunk sizes.
type GraphCaptureKey struct {
	Kind      GraphCaptureKind `json:"kind"`
	BatchSize int              `json:"batch_size"` // fixed capture dimension
	Tag       string           `json:"tag,omitempty"`
}

// String returns a human-readable, deterministic identifier for the capture key.
func (k GraphCaptureKey) String() string {
	if k.Tag != "" {
		return fmt.Sprintf("%s:b%d:%s", k.Kind, k.BatchSize, k.Tag)
	}
	return fmt.Sprintf("%s:b%d", k.Kind, k.BatchSize)
}

// IsSpeculative reports whether the capture key is for speculative draft verification.
func (k GraphCaptureKey) IsSpeculative() bool {
	return k.Kind == GraphCaptureSpeculative
}

// IsPrimary reports whether the capture key is for primary prompt evaluation.
func (k GraphCaptureKey) IsPrimary() bool {
	return k.Kind == GraphCapturePrimary
}

// SpeculativeGraphConfig configures graph capture dimensions for primary chunking
// and speculative draft verification.
type SpeculativeGraphConfig struct {
	// PrimaryUBatchSize is the primary chunk size for prompt evaluation (e.g. 1024).
	PrimaryUBatchSize int `json:"primary_ubatch_size"`

	// SpecDraftUBatchSize is the decoupled speculative draft micro-batch size (e.g. 256 or 512).
	SpecDraftUBatchSize int `json:"spec_draft_ubatch_size"`

	// DeviceTag optionally tags the device architecture or stream for graph differentiation.
	DeviceTag string `json:"device_tag,omitempty"`
}

// Validate checks the configuration parameters.
func (c SpeculativeGraphConfig) Validate() error {
	if c.PrimaryUBatchSize <= 0 {
		return fmt.Errorf("speculative graph: PrimaryUBatchSize must be positive, got %d", c.PrimaryUBatchSize)
	}
	if c.SpecDraftUBatchSize <= 0 {
		return fmt.Errorf("speculative graph: SpecDraftUBatchSize must be positive, got %d", c.SpecDraftUBatchSize)
	}
	return nil
}

// SpeculativeGraphPlanner manages graph capture dimensions and produces stable capture keys,
// ensuring speculative draft verification does not perturb primary prompt graphs or trigger graph misses.
type SpeculativeGraphPlanner struct {
	cfg SpeculativeGraphConfig
}

// NewSpeculativeGraphPlanner creates a new planner with the provided configuration.
func NewSpeculativeGraphPlanner(cfg SpeculativeGraphConfig) (*SpeculativeGraphPlanner, error) {
	if err := cfg.Validate(); err != nil {
		return nil, err
	}
	return &SpeculativeGraphPlanner{cfg: cfg}, nil
}

// Config returns the planner's configuration.
func (p *SpeculativeGraphPlanner) Config() SpeculativeGraphConfig {
	return p.cfg
}

// PrimaryUBatchSize returns the configured primary chunk size.
func (p *SpeculativeGraphPlanner) PrimaryUBatchSize() int {
	return p.cfg.PrimaryUBatchSize
}

// SpecDraftUBatchSize returns the dedicated fixed speculative micro-batch size.
func (p *SpeculativeGraphPlanner) SpecDraftUBatchSize() int {
	return p.cfg.SpecDraftUBatchSize
}

// SpeculativeCaptureKey returns the dedicated, fixed capture key for speculative draft verification.
// It is guaranteed to remain invariant under varying prompt chunk sizes and dynamic draft token counts.
func (p *SpeculativeGraphPlanner) SpeculativeCaptureKey() GraphCaptureKey {
	return GraphCaptureKey{
		Kind:      GraphCaptureSpeculative,
		BatchSize: p.cfg.SpecDraftUBatchSize,
		Tag:       p.cfg.DeviceTag,
	}
}

// PrimaryCaptureKey returns the capture key for a given primary prompt chunk size.
func (p *SpeculativeGraphPlanner) PrimaryCaptureKey(chunkSize int) GraphCaptureKey {
	return GraphCaptureKey{
		Kind:      GraphCapturePrimary,
		BatchSize: chunkSize,
		Tag:       p.cfg.DeviceTag,
	}
}

// CapturedGraph represents an instantiated device execution graph ready for replay.
type CapturedGraph struct {
	Key         GraphCaptureKey `json:"key"`
	CreatedAt   time.Time       `json:"created_at"`
	ReplayCount int             `json:"replay_count"`
}

// SpeculativeGraphStats records runtime metrics for graph execution.
type SpeculativeGraphStats struct {
	CapturesTotal     int               `json:"captures_total"`
	ReplaysTotal      int               `json:"replays_total"`
	SpeculativeHits   int               `json:"speculative_hits"`
	SpeculativeMisses int               `json:"speculative_misses"`
	PrimaryHits       int               `json:"primary_hits"`
	PrimaryMisses     int               `json:"primary_misses"`
	ActiveCaptures    []GraphCaptureKey `json:"active_captures"`
}

// SpeculativeGraphRunner manages graph capture, caching, and execution for both primary
// prompt chunking and decoupled speculative verification.
type SpeculativeGraphRunner struct {
	mu       sync.RWMutex
	planner  *SpeculativeGraphPlanner
	graphs   map[string]*CapturedGraph
	capturer PrefillGraphCapturer
	stats    SpeculativeGraphStats
}

// NewSpeculativeGraphRunner creates a graph runner using the given planner.
func NewSpeculativeGraphRunner(planner *SpeculativeGraphPlanner) *SpeculativeGraphRunner {
	return &SpeculativeGraphRunner{
		planner: planner,
		graphs:  make(map[string]*CapturedGraph),
	}
}

// SetCapturer attaches an optional device capturer interface.
func (r *SpeculativeGraphRunner) SetCapturer(capturer PrefillGraphCapturer) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.capturer = capturer
}

// Planner returns the runner's graph planner.
func (r *SpeculativeGraphRunner) Planner() *SpeculativeGraphPlanner {
	return r.planner
}

// LookupGraph returns the captured graph for key if already captured.
func (r *SpeculativeGraphRunner) LookupGraph(key GraphCaptureKey) (*CapturedGraph, bool) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	g, ok := r.graphs[key.String()]
	return g, ok
}

// ExecuteSpeculativeDraft executes speculative verification using the fixed micro-batch dimension.
// If numTokens < SpecDraftUBatchSize, the dispatch is mapped to the fixed capture dimension,
// preserving graph hits across dynamic draft lengths and preventing driver timeouts.
func (r *SpeculativeGraphRunner) ExecuteSpeculativeDraft(numTokens int, body func(effectiveBatch int) error) error {
	if numTokens <= 0 {
		return errors.New("speculative draft token count must be positive")
	}
	specBatchSize := r.planner.SpecDraftUBatchSize()
	if numTokens > specBatchSize {
		return fmt.Errorf("numTokens (%d) exceeds decoupled SpecDraftUBatchSize (%d)", numTokens, specBatchSize)
	}

	key := r.planner.SpeculativeCaptureKey()
	keyStr := key.String()

	r.mu.Lock()
	g, exists := r.graphs[keyStr]
	if !exists {
		// Graph miss: capture for the dedicated fixed dimension.
		r.stats.SpeculativeMisses++
		r.stats.CapturesTotal++
		g = &CapturedGraph{
			Key:       key,
			CreatedAt: time.Now(),
		}
		r.graphs[keyStr] = g

		capturer := r.capturer
		if capturer != nil && capturer.GraphBegin() {
			r.mu.Unlock()
			err := body(specBatchSize)
			capturer.GraphEndLaunch()
			return err
		}
		r.mu.Unlock()
		return body(specBatchSize)
	}

	// Graph hit: replay the cached graph.
	r.stats.SpeculativeHits++
	r.stats.ReplaysTotal++
	g.ReplayCount++
	r.mu.Unlock()

	return body(specBatchSize)
}

// ExecutePrimaryChunk executes a primary prompt chunk.
func (r *SpeculativeGraphRunner) ExecutePrimaryChunk(chunkSize int, body func(chunkSize int) error) error {
	if chunkSize <= 0 {
		return errors.New("primary chunk size must be positive")
	}

	key := r.planner.PrimaryCaptureKey(chunkSize)
	keyStr := key.String()

	r.mu.Lock()
	g, exists := r.graphs[keyStr]
	if !exists {
		r.stats.PrimaryMisses++
		r.stats.CapturesTotal++
		g = &CapturedGraph{
			Key:       key,
			CreatedAt: time.Now(),
		}
		r.graphs[keyStr] = g

		capturer := r.capturer
		if capturer != nil && capturer.GraphBegin() {
			r.mu.Unlock()
			err := body(chunkSize)
			capturer.GraphEndLaunch()
			return err
		}
		r.mu.Unlock()
		return body(chunkSize)
	}

	r.stats.PrimaryHits++
	r.stats.ReplaysTotal++
	g.ReplayCount++
	r.mu.Unlock()

	return body(chunkSize)
}

// Stats returns a copy of current execution statistics.
func (r *SpeculativeGraphRunner) Stats() SpeculativeGraphStats {
	r.mu.RLock()
	defer r.mu.RUnlock()

	st := r.stats
	st.ActiveCaptures = make([]GraphCaptureKey, 0, len(r.graphs))
	for _, g := range r.graphs {
		st.ActiveCaptures = append(st.ActiveCaptures, g.Key)
	}
	return st
}

// Reset clears all captured graphs and resets statistics.
func (r *SpeculativeGraphRunner) Reset() {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.capturer != nil {
		r.capturer.GraphReset()
	}
	r.graphs = make(map[string]*CapturedGraph)
	r.stats = SpeculativeGraphStats{}
}
