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
	Kind               GraphCaptureKind `json:"kind"`
	BatchSize          int              `json:"batch_size"` // fixed capture dimension
	Tag                string           `json:"tag,omitempty"`
	NumSequences       int              `json:"num_sequences,omitempty"`
	TokensPerReq       int              `json:"tokens_per_req,omitempty"`
	SchedulerReachable bool             `json:"scheduler_reachable"`
}

// String returns a human-readable, deterministic identifier for the capture key.
func (k GraphCaptureKey) String() string {
	var suffix string
	if k.NumSequences > 0 && k.TokensPerReq > 0 {
		suffix = fmt.Sprintf(":s%d:k%d", k.NumSequences, k.TokensPerReq)
	}
	if k.Tag != "" {
		return fmt.Sprintf("%s:b%d%s:%s", k.Kind, k.BatchSize, suffix, k.Tag)
	}
	return fmt.Sprintf("%s:b%d%s", k.Kind, k.BatchSize, suffix)
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

	// MaxActiveSeqs optionally limits the active sequence capacity for reachability validation.
	MaxActiveSeqs int `json:"max_active_seqs,omitempty"`

	// MaxTokensPerReq optionally limits the decode tokens (K+1) per sequence for reachability validation.
	MaxTokensPerReq int `json:"max_tokens_per_req,omitempty"`
}

// Validate checks the configuration parameters.
func (c SpeculativeGraphConfig) Validate() error {
	if c.PrimaryUBatchSize <= 0 {
		return fmt.Errorf("speculative graph: PrimaryUBatchSize must be positive, got %d", c.PrimaryUBatchSize)
	}
	if c.SpecDraftUBatchSize <= 0 {
		return fmt.Errorf("speculative graph: SpecDraftUBatchSize must be positive, got %d", c.SpecDraftUBatchSize)
	}
	if c.MaxActiveSeqs < 0 {
		return fmt.Errorf("speculative graph: MaxActiveSeqs must be non-negative, got %d", c.MaxActiveSeqs)
	}
	if c.MaxTokensPerReq < 0 {
		return fmt.Errorf("speculative graph: MaxTokensPerReq must be non-negative, got %d", c.MaxTokensPerReq)
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

// ValidateKey validates whether a capture key is scheduler-reachable under the planner configuration.
func (p *SpeculativeGraphPlanner) ValidateKey(key GraphCaptureKey) error {
	if key.Kind != GraphCapturePrimary && key.Kind != GraphCaptureSpeculative {
		return fmt.Errorf("speculative graph: unknown capture key kind %q", key.Kind)
	}
	if key.BatchSize <= 0 {
		return fmt.Errorf("speculative graph: capture key batch size must be positive, got %d", key.BatchSize)
	}

	maxSeqs := p.cfg.MaxActiveSeqs
	if maxSeqs <= 0 {
		maxSeqs = 64
	}
	maxTokensReq := p.cfg.MaxTokensPerReq
	if maxTokensReq <= 0 {
		maxTokensReq = 16
	}

	if key.IsSpeculative() {
		if key.BatchSize > p.cfg.SpecDraftUBatchSize {
			return fmt.Errorf("speculative graph: batch size %d exceeds SpecDraftUBatchSize %d",
				key.BatchSize, p.cfg.SpecDraftUBatchSize)
		}
		if key.NumSequences > maxSeqs {
			return fmt.Errorf("speculative graph: sequence count %d exceeds MaxActiveSeqs %d",
				key.NumSequences, maxSeqs)
		}
		if key.TokensPerReq > maxTokensReq {
			return fmt.Errorf("speculative graph: tokens per request %d exceeds MaxTokensPerReq %d",
				key.TokensPerReq, maxTokensReq)
		}
		if key.NumSequences > 0 && key.TokensPerReq > 0 {
			total := key.NumSequences * key.TokensPerReq
			if total > p.cfg.SpecDraftUBatchSize {
				return fmt.Errorf("speculative graph: required tokens (%d * %d = %d) exceeds SpecDraftUBatchSize %d",
					key.NumSequences, key.TokensPerReq, total, p.cfg.SpecDraftUBatchSize)
			}
		}
	} else if key.IsPrimary() {
		if key.BatchSize > p.cfg.PrimaryUBatchSize {
			return fmt.Errorf("speculative graph: primary chunk size %d exceeds PrimaryUBatchSize %d",
				key.BatchSize, p.cfg.PrimaryUBatchSize)
		}
	}

	return nil
}

// SpeculativeCaptureKey returns the dedicated, fixed capture key for speculative draft verification.
// It is guaranteed to remain invariant under varying prompt chunk sizes and dynamic draft token counts.
func (p *SpeculativeGraphPlanner) SpeculativeCaptureKey() GraphCaptureKey {
	return GraphCaptureKey{
		Kind:               GraphCaptureSpeculative,
		BatchSize:          p.cfg.SpecDraftUBatchSize,
		Tag:                p.cfg.DeviceTag,
		SchedulerReachable: true,
	}
}

// PrimaryCaptureKey returns the capture key for a given primary prompt chunk size.
func (p *SpeculativeGraphPlanner) PrimaryCaptureKey(chunkSize int) GraphCaptureKey {
	reachable := chunkSize > 0 && chunkSize <= p.cfg.PrimaryUBatchSize
	return GraphCaptureKey{
		Kind:               GraphCapturePrimary,
		BatchSize:          chunkSize,
		Tag:                p.cfg.DeviceTag,
		SchedulerReachable: reachable,
	}
}

// ShapeCaptureKey produces a graph capture key for a specific sequence count and tokens per request,
// verifying reachability before returning the key.
func (p *SpeculativeGraphPlanner) ShapeCaptureKey(numSeqs, tokensPerReq int) (GraphCaptureKey, error) {
	key := GraphCaptureKey{
		Kind:               GraphCaptureSpeculative,
		BatchSize:          p.cfg.SpecDraftUBatchSize,
		Tag:                p.cfg.DeviceTag,
		NumSequences:       numSeqs,
		TokensPerReq:       tokensPerReq,
		SchedulerReachable: true,
	}
	if err := p.ValidateKey(key); err != nil {
		key.SchedulerReachable = false
		return key, err
	}
	return key, nil
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

// AllocateGraph validates reachability and allocates/caches a device graph for the given key.
// Unreachable shapes fail validation before graph allocation occurs.
func (r *SpeculativeGraphRunner) AllocateGraph(key GraphCaptureKey) (*CapturedGraph, error) {
	if err := r.planner.ValidateKey(key); err != nil {
		return nil, fmt.Errorf("speculative graph: cannot allocate graph for unreachable shape %s: %w", key, err)
	}

	r.mu.Lock()
	defer r.mu.Unlock()

	keyStr := key.String()
	if g, exists := r.graphs[keyStr]; exists {
		return g, nil
	}

	g := &CapturedGraph{
		Key:       key,
		CreatedAt: time.Now(),
	}
	r.graphs[keyStr] = g
	r.stats.CapturesTotal++
	return g, nil
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
	if err := r.planner.ValidateKey(key); err != nil {
		return fmt.Errorf("speculative graph: unreachable capture key: %w", err)
	}
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
	if err := r.planner.ValidateKey(key); err != nil {
		return fmt.Errorf("speculative graph: unreachable capture key: %w", err)
	}
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

// ExecuteCaptureKey executes a graph dispatch for an arbitrary capture key, validating reachability
// before allocating or executing the graph.
func (r *SpeculativeGraphRunner) ExecuteCaptureKey(key GraphCaptureKey, body func(batchSize int) error) error {
	if err := r.planner.ValidateKey(key); err != nil {
		return fmt.Errorf("speculative graph: cannot execute unreachable graph key %s: %w", key, err)
	}
	keyStr := key.String()

	r.mu.Lock()
	g, exists := r.graphs[keyStr]
	if !exists {
		r.stats.CapturesTotal++
		if key.IsSpeculative() {
			r.stats.SpeculativeMisses++
		} else {
			r.stats.PrimaryMisses++
		}
		g = &CapturedGraph{
			Key:       key,
			CreatedAt: time.Now(),
		}
		r.graphs[keyStr] = g

		capturer := r.capturer
		if capturer != nil && capturer.GraphBegin() {
			r.mu.Unlock()
			err := body(key.BatchSize)
			capturer.GraphEndLaunch()
			return err
		}
		r.mu.Unlock()
		return body(key.BatchSize)
	}

	if key.IsSpeculative() {
		r.stats.SpeculativeHits++
	} else {
		r.stats.PrimaryHits++
	}
	r.stats.ReplaysTotal++
	g.ReplayCount++
	r.mu.Unlock()

	return body(key.BatchSize)
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
