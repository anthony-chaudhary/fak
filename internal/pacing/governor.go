package pacing

import (
	"context"
	"errors"
	"sync"
	"time"

	"github.com/anthony-chaudhary/fak/internal/leaseref"
)

var (
	ErrContextCancelled = errors.New("pacing: context cancelled while waiting for inference slot")
	ErrSlotNotHeld      = errors.New("pacing: ticket does not hold an active inference slot")
	ErrAlreadyHolding   = errors.New("pacing: ticket already holds an active inference slot")
)

// FeedbackSignal represents runtime feedback from model execution or the gateway.
type FeedbackSignal string

const (
	SignalRateLimit   FeedbackSignal = "RATE_LIMIT_429"
	SignalHighLatency FeedbackSignal = "HIGH_LATENCY"
	SignalSuccess     FeedbackSignal = "SUCCESS"
)

// ContractStore is the minimal contract lease interface needed by Governor.
// It matches *leaseref.Store.
type ContractStore interface {
	LiveContracts(ctx context.Context, now ...time.Time) ([]leaseref.ContractRecord, error)
	UpdateContractState(ctx context.Context, ticketID, holder string, newState leaseref.ContractState, tokensUsed int64, now ...time.Time) (leaseref.ContractRecord, error)
}

// GovernorConfig configures the pacing governor.
type GovernorConfig struct {
	// MaxInferenceSlots is the maximum number of concurrent model generation slots.
	MaxInferenceSlots int `json:"max_inference_slots"`
	// MinInferenceSlots is the floor for adaptive slot reduction (default: 1).
	MinInferenceSlots int `json:"min_inference_slots"`
	// TokenBucketConfig sets rate limit ceilings (TPM / RPM).
	TokenBucket TokenBucketConfig `json:"token_bucket"`
	// InterTurnDelay is the baseline pause between consecutive inference turns.
	InterTurnDelay time.Duration `json:"inter_turn_delay"`
	// ContractStore is an optional leaseref contract store to synchronize contract states.
	ContractStore ContractStore `json:"-"`
	// HolderID is the worker/governor identity passed when updating contract state.
	HolderID string `json:"holder_id"`
}

// InferenceLease represents an active grant to consume model inference capacity.
type InferenceLease struct {
	TicketID        string    `json:"ticket_id"`
	AcquiredAt      time.Time `json:"acquired_at"`
	EstimatedTokens int64     `json:"estimated_tokens"`
	SlotIndex       int       `json:"slot_index"`
}

type slotWaiter struct {
	ticketID        string
	estimatedTokens int64
	ready           chan struct{}
	cancelled       bool
}

// Governor coordinates phase-aware inference slots and token bucket rate limits
// across concurrent agent workers.
type Governor struct {
	mu sync.Mutex

	maxSlots        int
	minSlots        int
	currentSlots    int
	interTurnDelay  time.Duration
	baseDelay       time.Duration
	consecSuccesses int

	bucket *TokenBucket
	store  ContractStore
	holder string

	activeHolders map[string]*InferenceLease
	priorityWait  []*slotWaiter // Resuming workers (higher priority)
	standardWait  []*slotWaiter // New acquisitions
}

// NewGovernor initializes a pacing governor.
func NewGovernor(cfg GovernorConfig) *Governor {
	maxSlots := cfg.MaxInferenceSlots
	if maxSlots <= 0 {
		maxSlots = 2
	}
	minSlots := cfg.MinInferenceSlots
	if minSlots <= 0 {
		minSlots = 1
	}
	if minSlots > maxSlots {
		minSlots = maxSlots
	}

	holder := cfg.HolderID
	if holder == "" {
		holder = "pacing-governor"
	}

	return &Governor{
		maxSlots:       maxSlots,
		minSlots:       minSlots,
		currentSlots:   maxSlots,
		interTurnDelay: cfg.InterTurnDelay,
		baseDelay:      cfg.InterTurnDelay,
		bucket:         NewTokenBucket(cfg.TokenBucket),
		store:          cfg.ContractStore,
		holder:         holder,
		activeHolders:  make(map[string]*InferenceLease),
	}
}

// AcquireInference blocks until token bucket headroom and an inference slot
// are available, then marks the ticket as actively executing.
func (g *Governor) AcquireInference(ctx context.Context, ticketID string, estimatedTokens int64) (*InferenceLease, error) {
	return g.acquireSlot(ctx, ticketID, estimatedTokens, false)
}

// ResumeInference re-enters the inference queue with priority over new acquisitions.
func (g *Governor) ResumeInference(ctx context.Context, ticketID string, estimatedTokens int64) (*InferenceLease, error) {
	return g.acquireSlot(ctx, ticketID, estimatedTokens, true)
}

func (g *Governor) acquireSlot(ctx context.Context, ticketID string, estimatedTokens int64, priority bool) (*InferenceLease, error) {
	// First: reserve token bucket headroom (waits for rate limit if needed)
	if err := g.bucket.Reserve(ctx, estimatedTokens); err != nil {
		return nil, err
	}

	// Apply inter-turn delay if configured
	g.mu.Lock()
	delay := g.interTurnDelay
	g.mu.Unlock()
	if delay > 0 {
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		case <-time.After(delay):
		}
	}

	// Next: acquire one of the dynamic inference slots
	g.mu.Lock()

	// If already holding a slot, return the active lease
	if lease, ok := g.activeHolders[ticketID]; ok {
		lease.EstimatedTokens = estimatedTokens
		g.mu.Unlock()
		return lease, nil
	}

	// If a slot is available right now and no higher priority waiters exist
	if len(g.activeHolders) < g.currentSlots && (priority || len(g.priorityWait) == 0) {
		lease := &InferenceLease{
			TicketID:        ticketID,
			AcquiredAt:      time.Now(),
			EstimatedTokens: estimatedTokens,
			SlotIndex:       len(g.activeHolders),
		}
		g.activeHolders[ticketID] = lease
		g.mu.Unlock()

		g.syncContractState(ctx, ticketID, leaseref.ContractStateExecuting)
		return lease, nil
	}

	// Enqueue waiter
	waiter := &slotWaiter{
		ticketID:        ticketID,
		estimatedTokens: estimatedTokens,
		ready:           make(chan struct{}),
	}
	if priority {
		g.priorityWait = append(g.priorityWait, waiter)
	} else {
		g.standardWait = append(g.standardWait, waiter)
	}
	g.mu.Unlock()

	// Wait for slot or cancellation
	select {
	case <-ctx.Done():
		g.mu.Lock()
		waiter.cancelled = true
		g.removeWaiter(waiter, priority)
		g.mu.Unlock()
		return nil, ctx.Err()

	case <-waiter.ready:
		g.mu.Lock()
		lease := &InferenceLease{
			TicketID:        ticketID,
			AcquiredAt:      time.Now(),
			EstimatedTokens: estimatedTokens,
			SlotIndex:       len(g.activeHolders),
		}
		g.activeHolders[ticketID] = lease
		g.mu.Unlock()

		g.syncContractState(ctx, ticketID, leaseref.ContractStateExecuting)
		return lease, nil
	}
}

// YieldInference is called when an agent enters tool or test execution,
// relinquishing its inference slot for other waiting agents.
func (g *Governor) YieldInference(ticketID string) error {
	g.mu.Lock()
	defer g.mu.Unlock()

	lease, ok := g.activeHolders[ticketID]
	if !ok {
		return ErrSlotNotHeld
	}
	_ = lease
	delete(g.activeHolders, ticketID)

	// Wake the highest-priority waiting worker
	g.wakeNextWaiter()

	// Synchronize state to YIELDED_IO
	go g.syncContractState(context.Background(), ticketID, leaseref.ContractStateYieldedIO)
	return nil
}

// Release completely releases any held inference slot and clears registration
// for a ticket (e.g. when completed or failed).
func (g *Governor) Release(ticketID string) {
	g.mu.Lock()
	defer g.mu.Unlock()

	if _, ok := g.activeHolders[ticketID]; ok {
		delete(g.activeHolders, ticketID)
		g.wakeNextWaiter()
	}
}

// wakeNextWaiter signals the next eligible waiter. Must be called with g.mu held.
func (g *Governor) wakeNextWaiter() {
	if len(g.activeHolders) >= g.currentSlots {
		return
	}

	// 1. Check priority queue first
	for len(g.priorityWait) > 0 {
		w := g.priorityWait[0]
		g.priorityWait = g.priorityWait[1:]
		if !w.cancelled {
			close(w.ready)
			return
		}
	}

	// 2. Check standard queue
	for len(g.standardWait) > 0 {
		w := g.standardWait[0]
		g.standardWait = g.standardWait[1:]
		if !w.cancelled {
			close(w.ready)
			return
		}
	}
}

func (g *Governor) removeWaiter(target *slotWaiter, priority bool) {
	if priority {
		for i, w := range g.priorityWait {
			if w == target {
				g.priorityWait = append(g.priorityWait[:i], g.priorityWait[i+1:]...)
				break
			}
		}
	} else {
		for i, w := range g.standardWait {
			if w == target {
				g.standardWait = append(g.standardWait[:i], g.standardWait[i+1:]...)
				break
			}
		}
	}
}

func (g *Governor) syncContractState(ctx context.Context, ticketID string, state leaseref.ContractState) {
	if g.store == nil {
		return
	}
	// Best-effort update of contract state
	_, _ = g.store.UpdateContractState(ctx, ticketID, g.holder, state, 0)
}

// ReportFeedback adjusts capacity adaptively based on runtime execution signals.
func (g *Governor) ReportFeedback(signal FeedbackSignal) {
	g.mu.Lock()
	defer g.mu.Unlock()

	switch signal {
	case SignalRateLimit:
		// Multiplicative decrease of slots
		g.currentSlots = g.currentSlots / 2
		if g.currentSlots < g.minSlots {
			g.currentSlots = g.minSlots
		}
		// Apply token bucket backoff window
		g.bucket.ApplyBackoff(2 * time.Second)
		// Increase inter-turn delay
		if g.interTurnDelay == 0 {
			g.interTurnDelay = 500 * time.Millisecond
		} else {
			g.interTurnDelay = g.interTurnDelay * 2
			if g.interTurnDelay > 5*time.Second {
				g.interTurnDelay = 5 * time.Second
			}
		}
		g.consecSuccesses = 0

	case SignalHighLatency:
		// Slight slot throttle
		if g.currentSlots > g.minSlots {
			g.currentSlots--
		}
		g.bucket.ApplyBackoff(500 * time.Millisecond)
		g.consecSuccesses = 0

	case SignalSuccess:
		g.consecSuccesses++
		g.bucket.ResetBackoff()
		// Additive increase after sustained success
		if g.consecSuccesses >= 5 {
			if g.currentSlots < g.maxSlots {
				g.currentSlots++
			}
			if g.interTurnDelay > g.baseDelay {
				g.interTurnDelay = (g.interTurnDelay + g.baseDelay) / 2
			}
			g.consecSuccesses = 0
			// Wake waiters if we expanded capacity
			g.wakeNextWaiter()
		}
	}
}

// GovernorStats reports a point-in-time snapshot of the governor.
type GovernorStats struct {
	MaxSlots       int              `json:"max_slots"`
	CurrentSlots   int              `json:"current_slots"`
	ActiveSlots    int              `json:"active_slots"`
	PriorityQueue  int              `json:"priority_queue_len"`
	StandardQueue  int              `json:"standard_queue_len"`
	InterTurnDelay time.Duration    `json:"inter_turn_delay"`
	TokenBucket    TokenBucketStats `json:"token_bucket"`
}

// Stats returns a snapshot of the governor's state.
func (g *Governor) Stats() GovernorStats {
	g.mu.Lock()
	defer g.mu.Unlock()

	return GovernorStats{
		MaxSlots:       g.maxSlots,
		CurrentSlots:   g.currentSlots,
		ActiveSlots:    len(g.activeHolders),
		PriorityQueue:  len(g.priorityWait),
		StandardQueue:  len(g.standardWait),
		InterTurnDelay: g.interTurnDelay,
		TokenBucket:    g.bucket.Stats(),
	}
}
