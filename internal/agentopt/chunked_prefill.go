package agentopt

import (
	"errors"
	"fmt"
	"sync"
	"time"
)

// ItemKind identifies whether a scheduled item is a prefill chunk or a decode step.
type ItemKind string

const (
	// KindPrefill denotes a bounded chunk of prompt prefill computation.
	KindPrefill ItemKind = "prefill"
	// KindDecode denotes an autoregressive decode generation step.
	KindDecode ItemKind = "decode"
)

// ScheduledItem represents a single sequence's work scheduled in one batch step.
type ScheduledItem struct {
	RequestID       string   `json:"request_id"`
	Kind            ItemKind `json:"kind"`
	Tokens          int      `json:"tokens"`           // tokens scheduled in this step
	PromptOffset    int      `json:"prompt_offset"`    // token offset within the prompt for prefill
	TotalPrompt     int      `json:"total_prompt"`     // total prompt tokens for prefill
	IsLastChunk     bool     `json:"is_last_chunk"`    // true if this chunk completes prompt prefill
	RemainingTokens int      `json:"remaining_tokens"` // tokens remaining after this step
	StepCount       int      `json:"step_count"`       // number of steps this request has participated in
}

// ScheduledBatch contains all prefill chunks and decode steps scheduled in one forward pass.
type ScheduledBatch struct {
	StepIndex        int             `json:"step_index"`
	Items            []ScheduledItem `json:"items"`
	TotalTokens      int             `json:"total_tokens"`
	PrefillTokens    int             `json:"prefill_tokens"`
	DecodeTokens     int             `json:"decode_tokens"`
	CompletedPrefill []string        `json:"completed_prefills,omitempty"`
	CompletedDecode  []string        `json:"completed_decodes,omitempty"`
}

// IsEmpty reports whether the scheduled batch contains any items.
func (b ScheduledBatch) IsEmpty() bool {
	return len(b.Items) == 0
}

// HasPrefill reports whether a prefill item with requestID is present in the batch.
func (b ScheduledBatch) HasPrefill(requestID string) bool {
	for _, item := range b.Items {
		if item.Kind == KindPrefill && item.RequestID == requestID {
			return true
		}
	}
	return false
}

// HasDecode reports whether a decode item with requestID is present in the batch.
func (b ScheduledBatch) HasDecode(requestID string) bool {
	for _, item := range b.Items {
		if item.Kind == KindDecode && item.RequestID == requestID {
			return true
		}
	}
	return false
}

// DecodeIDs returns the IDs of all decode requests scheduled in this batch.
func (b ScheduledBatch) DecodeIDs() []string {
	var ids []string
	for _, item := range b.Items {
		if item.Kind == KindDecode {
			ids = append(ids, item.RequestID)
		}
	}
	return ids
}

// PrefillIDs returns the IDs of all prefill requests scheduled in this batch.
func (b ScheduledBatch) PrefillIDs() []string {
	var ids []string
	for _, item := range b.Items {
		if item.Kind == KindPrefill {
			ids = append(ids, item.RequestID)
		}
	}
	return ids
}

// ChunkedInterleaverConfig configures continuous batching and token chunk limits.
type ChunkedInterleaverConfig struct {
	MaxBatchTokens         int  // Maximum total tokens allowed in a single scheduled batch (default: 512)
	ChunkSize              int  // Maximum tokens per prefill chunk in a single batch (default: 256)
	MaxBatchSeqs           int  // Maximum number of sequences (requests) per batch (default: 32)
	DefaultDecodeTokens    int  // Default number of tokens to generate if unspecified in AddDecode (default: 1)
	AutoTransitionToDecode bool // Automatically transition completed prefills into decode requests
}

// InterleaverStats tracks cumulative scheduling metrics and decode latency pacing.
type InterleaverStats struct {
	TotalSteps          int              `json:"total_steps"`
	ScheduledPrefills   int              `json:"scheduled_prefills"`
	ScheduledDecodes    int              `json:"scheduled_decodes"`
	TotalPrefillTokens  int              `json:"total_prefill_tokens"`
	TotalDecodeTokens   int              `json:"total_decode_tokens"`
	MaxDecodeInterval   int              `json:"max_decode_interval"`
	ActivePrefillCount  int              `json:"active_prefill_count"`
	ActiveDecodeCount   int              `json:"active_decode_count"`
	DecodePacingHistory map[string][]int `json:"decode_pacing_history,omitempty"`
}

type prefillRequest struct {
	id              string
	promptTokens    int
	processedTokens int
	stepCount       int
	createdAt       time.Time
}

type decodeRequest struct {
	id              string
	targetTokens    int // <= 0 means unbounded
	processedTokens int
	stepCount       int
	scheduledSteps  []int
	createdAt       time.Time
}

// ChunkedInterleaver interleaves bounded prefill chunks with decode steps to bound decode latency.
type ChunkedInterleaver struct {
	mu     sync.Mutex
	config ChunkedInterleaverConfig

	stepIndex int
	prefills  []*prefillRequest
	decodes   []*decodeRequest
	requests  map[string]bool

	stats InterleaverStats
}

// NewChunkedInterleaver initializes a ChunkedInterleaver with sanitized configuration defaults.
func NewChunkedInterleaver(config ChunkedInterleaverConfig) *ChunkedInterleaver {
	if config.MaxBatchTokens <= 0 {
		config.MaxBatchTokens = 512
	}
	if config.ChunkSize <= 0 {
		config.ChunkSize = 256
	}
	if config.ChunkSize > config.MaxBatchTokens {
		config.ChunkSize = config.MaxBatchTokens
	}
	if config.MaxBatchSeqs <= 0 {
		config.MaxBatchSeqs = 32
	}
	if config.DefaultDecodeTokens <= 0 {
		config.DefaultDecodeTokens = 1
	}

	return &ChunkedInterleaver{
		config:   config,
		requests: make(map[string]bool),
		stats: InterleaverStats{
			DecodePacingHistory: make(map[string][]int),
		},
	}
}

// AddPrefill enqueues a prompt prefill request to be scheduled in bounded token chunks.
func (s *ChunkedInterleaver) AddPrefill(id string, promptTokens int) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if id == "" {
		return errors.New("request id cannot be empty")
	}
	if promptTokens <= 0 {
		return fmt.Errorf("prompt tokens must be positive, got %d", promptTokens)
	}
	if s.requests[id] {
		return fmt.Errorf("request %q already exists", id)
	}

	s.requests[id] = true
	s.prefills = append(s.prefills, &prefillRequest{
		id:           id,
		promptTokens: promptTokens,
		createdAt:    time.Now(),
	})
	return nil
}

// AddDecode enqueues a decode request to generate the specified number of tokens (or DefaultDecodeTokens if omitted).
func (s *ChunkedInterleaver) AddDecode(id string, decodeTokens ...int) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if id == "" {
		return errors.New("request id cannot be empty")
	}
	if s.requests[id] {
		return fmt.Errorf("request %q already exists", id)
	}

	target := s.config.DefaultDecodeTokens
	if len(decodeTokens) > 0 {
		target = decodeTokens[0]
	}

	s.requests[id] = true
	s.decodes = append(s.decodes, &decodeRequest{
		id:           id,
		targetTokens: target,
		createdAt:    time.Now(),
	})
	return nil
}

// TransitionToDecode converts a completed prefill request into a decode request.
func (s *ChunkedInterleaver) TransitionToDecode(id string, decodeTokens ...int) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if id == "" {
		return errors.New("request id cannot be empty")
	}
	for _, p := range s.prefills {
		if p.id == id {
			return fmt.Errorf("request %q is still prefilling", id)
		}
	}
	target := s.config.DefaultDecodeTokens
	if len(decodeTokens) > 0 {
		target = decodeTokens[0]
	}

	s.requests[id] = true
	s.decodes = append(s.decodes, &decodeRequest{
		id:           id,
		targetTokens: target,
		createdAt:    time.Now(),
	})
	return nil
}

// Step evaluates pending decodes and prefills, emitting the next scheduled continuous batch.
// Decode steps are given priority to bound worst-case decode latency, and remaining token
// capacity is packed with bounded prefill chunks.
func (s *ChunkedInterleaver) Step() ScheduledBatch {
	s.mu.Lock()
	defer s.mu.Unlock()

	s.stepIndex++
	batch := ScheduledBatch{
		StepIndex: s.stepIndex,
	}

	if len(s.decodes) == 0 && len(s.prefills) == 0 {
		return batch
	}

	availableTokens := s.config.MaxBatchTokens
	availableSeqs := s.config.MaxBatchSeqs

	// 1. Schedule Decodes: 1 token per decode sequence up to available seqs/tokens
	var nextDecodes []*decodeRequest
	for _, d := range s.decodes {
		if availableTokens <= 0 || availableSeqs <= 0 {
			nextDecodes = append(nextDecodes, d)
			continue
		}

		tokensToSchedule := 1
		d.processedTokens += tokensToSchedule
		d.stepCount++

		// Track decode step pacing
		if len(d.scheduledSteps) > 0 {
			prevStep := d.scheduledSteps[len(d.scheduledSteps)-1]
			interval := s.stepIndex - prevStep
			if interval > s.stats.MaxDecodeInterval {
				s.stats.MaxDecodeInterval = interval
			}
		} else {
			if s.stats.MaxDecodeInterval == 0 {
				s.stats.MaxDecodeInterval = 1
			}
		}
		d.scheduledSteps = append(d.scheduledSteps, s.stepIndex)
		s.stats.DecodePacingHistory[d.id] = append(s.stats.DecodePacingHistory[d.id], s.stepIndex)

		availableTokens -= tokensToSchedule
		availableSeqs--

		rem := 0
		if d.targetTokens > 0 {
			rem = d.targetTokens - d.processedTokens
			if rem < 0 {
				rem = 0
			}
		}

		item := ScheduledItem{
			RequestID:       d.id,
			Kind:            KindDecode,
			Tokens:          tokensToSchedule,
			RemainingTokens: rem,
			StepCount:       d.stepCount,
		}
		batch.Items = append(batch.Items, item)
		batch.DecodeTokens += tokensToSchedule
		s.stats.ScheduledDecodes++
		s.stats.TotalDecodeTokens += tokensToSchedule

		if d.targetTokens > 0 && d.processedTokens >= d.targetTokens {
			batch.CompletedDecode = append(batch.CompletedDecode, d.id)
			delete(s.requests, d.id)
		} else {
			nextDecodes = append(nextDecodes, d)
		}
	}
	s.decodes = nextDecodes

	// 2. Schedule Prefill Chunks with leftover token & sequence capacity.
	// In one forward pass, each prefill sequence can execute at most one chunk.
	numPrefills := len(s.prefills)
	var completedPrefillIDs []string

	for i := 0; i < numPrefills && availableTokens > 0 && availableSeqs > 0; i++ {
		p := s.prefills[0]
		s.prefills = s.prefills[1:]

		remPrompt := p.promptTokens - p.processedTokens
		chunk := remPrompt
		if chunk > s.config.ChunkSize {
			chunk = s.config.ChunkSize
		}
		if chunk > availableTokens {
			chunk = availableTokens
		}

		if chunk <= 0 {
			s.prefills = append([]*prefillRequest{p}, s.prefills...)
			break
		}

		offset := p.processedTokens
		p.processedTokens += chunk
		p.stepCount++
		availableTokens -= chunk
		availableSeqs--

		isLast := p.processedTokens >= p.promptTokens
		remTokens := p.promptTokens - p.processedTokens

		item := ScheduledItem{
			RequestID:       p.id,
			Kind:            KindPrefill,
			Tokens:          chunk,
			PromptOffset:    offset,
			TotalPrompt:     p.promptTokens,
			IsLastChunk:     isLast,
			RemainingTokens: remTokens,
			StepCount:       p.stepCount,
		}
		batch.Items = append(batch.Items, item)
		batch.PrefillTokens += chunk
		s.stats.ScheduledPrefills++
		s.stats.TotalPrefillTokens += chunk

		if isLast {
			batch.CompletedPrefill = append(batch.CompletedPrefill, p.id)
			completedPrefillIDs = append(completedPrefillIDs, p.id)
			if !s.config.AutoTransitionToDecode {
				delete(s.requests, p.id)
			}
		} else {
			// Round-robin: append to end of queue for subsequent steps
			s.prefills = append(s.prefills, p)
		}
	}

	// Auto-transition completed prefills to decodes if configured
	if s.config.AutoTransitionToDecode {
		for _, id := range completedPrefillIDs {
			s.decodes = append(s.decodes, &decodeRequest{
				id:           id,
				targetTokens: s.config.DefaultDecodeTokens,
				createdAt:    time.Now(),
			})
		}
	}

	batch.TotalTokens = batch.DecodeTokens + batch.PrefillTokens
	s.stats.TotalSteps++
	s.stats.ActivePrefillCount = len(s.prefills)
	s.stats.ActiveDecodeCount = len(s.decodes)

	return batch
}

// DecodeIntervals returns the step intervals between consecutive decodes for a given request.
func (s *ChunkedInterleaver) DecodeIntervals(requestID string) []int {
	s.mu.Lock()
	defer s.mu.Unlock()

	steps, ok := s.stats.DecodePacingHistory[requestID]
	if !ok || len(steps) < 2 {
		return nil
	}
	intervals := make([]int, len(steps)-1)
	for i := 1; i < len(steps); i++ {
		intervals[i-1] = steps[i] - steps[i-1]
	}
	return intervals
}

// RemoveRequest cancels and removes an active prefill or decode request.
func (s *ChunkedInterleaver) RemoveRequest(id string) bool {
	s.mu.Lock()
	defer s.mu.Unlock()

	if !s.requests[id] {
		return false
	}
	delete(s.requests, id)

	for i, p := range s.prefills {
		if p.id == id {
			s.prefills = append(s.prefills[:i], s.prefills[i+1:]...)
			return true
		}
	}
	for i, d := range s.decodes {
		if d.id == id {
			s.decodes = append(s.decodes[:i], s.decodes[i+1:]...)
			return true
		}
	}
	return true
}

// PendingPrefillCount returns the number of active prefill requests remaining.
func (s *ChunkedInterleaver) PendingPrefillCount() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return len(s.prefills)
}

// PendingDecodeCount returns the number of active decode requests remaining.
func (s *ChunkedInterleaver) PendingDecodeCount() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return len(s.decodes)
}

// HasPending reports whether any prefill or decode requests remain to be processed.
func (s *ChunkedInterleaver) HasPending() bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	return len(s.prefills) > 0 || len(s.decodes) > 0
}

// ActivePrefillTokens returns the total prompt tokens remaining across all pending prefills.
func (s *ChunkedInterleaver) ActivePrefillTokens() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	total := 0
	for _, p := range s.prefills {
		total += (p.promptTokens - p.processedTokens)
	}
	return total
}

// ActiveDecodeTokens returns the total decode tokens remaining across all pending decodes.
func (s *ChunkedInterleaver) ActiveDecodeTokens() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	total := 0
	for _, d := range s.decodes {
		if d.targetTokens > 0 {
			rem := d.targetTokens - d.processedTokens
			if rem > 0 {
				total += rem
			}
		}
	}
	return total
}

// Stats returns a snapshot copy of scheduling statistics.
func (s *ChunkedInterleaver) Stats() InterleaverStats {
	s.mu.Lock()
	defer s.mu.Unlock()

	historyCopy := make(map[string][]int, len(s.stats.DecodePacingHistory))
	for k, v := range s.stats.DecodePacingHistory {
		cp := make([]int, len(v))
		copy(cp, v)
		historyCopy[k] = cp
	}

	res := s.stats
	res.ActivePrefillCount = len(s.prefills)
	res.ActiveDecodeCount = len(s.decodes)
	res.DecodePacingHistory = historyCopy
	return res
}

// Reset clears all active requests and resets statistics.
func (s *ChunkedInterleaver) Reset() {
	s.mu.Lock()
	defer s.mu.Unlock()

	s.stepIndex = 0
	s.prefills = nil
	s.decodes = nil
	s.requests = make(map[string]bool)
	s.stats = InterleaverStats{
		DecodePacingHistory: make(map[string][]int),
	}
}
