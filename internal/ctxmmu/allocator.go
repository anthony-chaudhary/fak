package ctxmmu

import (
	"errors"
	"os"
	"strconv"
	"sync"
)

// Default constants for GTT attention page allocations on Strix Halo APU architectures.
// By default, 524,288 tokens (512K context) in FP8 GQA require 64 KiB per token (32 GiB KV cache),
// mapped into a 35.0 GiB contiguous slab of GTT memory with headroom for page tables and guard pages.
const (
	DefaultMaxTokens         = 524288
	DefaultPoolBytesPerToken = int64(65536)                    // 64 KiB per token for FP8 GQA
	DefaultTotalGTTBytes     = int64(35) * 1024 * 1024 * 1024 // 35.0 GiB contiguous slab
	EnvHalogenKVPoolFit      = "HALOGEN_KV_POOL_FIT"           // Env var for dynamic downscale ratio
)

// Standard typed errors for token pool operations.
var (
	// ErrPoolExhausted is returned when a reservation cannot be satisfied because either
	// the effective token capacity or the total GTT byte budget would be exceeded.
	ErrPoolExhausted = errors.New("ctxmmu: token pool exhausted")

	// ErrInvalidTokens is returned when a requested token count is zero or negative.
	ErrInvalidTokens = errors.New("ctxmmu: token count must be greater than zero")

	// ErrEmptyStreamID is returned when an empty stream or session ID is provided.
	ErrEmptyStreamID = errors.New("ctxmmu: stream ID cannot be empty")

	// ErrExceedsReservation is returned when attempting to commit more tokens than currently reserved.
	ErrExceedsReservation = errors.New("ctxmmu: committed tokens exceed reserved headroom")

	// ErrStreamNotFound is returned when attempting to commit or release headroom for an unknown stream.
	ErrStreamNotFound = errors.New("ctxmmu: stream reservation not found")
)

// SharedTokenPool represents a contiguous slab of GTT attention pages for public core runtime
// context-MMU token management. It arbitrates multi-session KV-cache allocations with dynamic
// capacity admission, two-phase reservation/commit cycles, and immediate headroom reclamation.
type SharedTokenPool struct {
	mu              sync.RWMutex
	maxTokens       int            // Default 524,288 tokens
	committedTokens int            // Sum of tokens committed across all active streams
	reservedTokens  int            // Sum of uncommitted output headroom reserved across active streams
	bytesPerToken   int64          // Default 65,536 bytes/token (FP8 GQA)
	totalGTTBytes   int64          // Default 35 GiB aperture
	downscaleFactor float64        // Default 1.0, or derived from HALOGEN_KV_POOL_FIT
	reservations    map[string]int // session/stream ID -> reserved uncommitted output headroom
	committed       map[string]int // session/stream ID -> committed tokens
}

// NewSharedTokenPool initializes a SharedTokenPool with default Strix Halo GTT parameters:
// 524,288 tokens, 35.0 GiB total GTT aperture, and 65,536 bytes/tok.
// If the HALOGEN_KV_POOL_FIT environment variable is set to a valid positive float,
// downscaleFactor is initialized from it (defaulting to 1.0 otherwise).
func NewSharedTokenPool() *SharedTokenPool {
	factor := 1.0
	if val := os.Getenv(EnvHalogenKVPoolFit); val != "" {
		if f, err := strconv.ParseFloat(val, 64); err == nil && f > 0 {
			factor = f
		}
	}
	return &SharedTokenPool{
		maxTokens:       DefaultMaxTokens,
		bytesPerToken:   DefaultPoolBytesPerToken,
		totalGTTBytes:   DefaultTotalGTTBytes,
		downscaleFactor: factor,
		reservations:    make(map[string]int),
		committed:       make(map[string]int),
	}
}

// NewSharedTokenPoolWithConfig creates a SharedTokenPool with explicit capacity parameters.
// If downscaleFactor <= 0, it falls back to 1.0 (or HALOGEN_KV_POOL_FIT if set).
func NewSharedTokenPoolWithConfig(maxTokens int, totalGTTBytes int64, bytesPerToken int64, downscaleFactor float64) *SharedTokenPool {
	if maxTokens <= 0 {
		maxTokens = DefaultMaxTokens
	}
	if totalGTTBytes <= 0 {
		totalGTTBytes = DefaultTotalGTTBytes
	}
	if bytesPerToken <= 0 {
		bytesPerToken = DefaultPoolBytesPerToken
	}
	if downscaleFactor <= 0 {
		downscaleFactor = 1.0
		if val := os.Getenv(EnvHalogenKVPoolFit); val != "" {
			if f, err := strconv.ParseFloat(val, 64); err == nil && f > 0 {
				downscaleFactor = f
			}
		}
	}
	return &SharedTokenPool{
		maxTokens:       maxTokens,
		bytesPerToken:   bytesPerToken,
		totalGTTBytes:   totalGTTBytes,
		downscaleFactor: downscaleFactor,
		reservations:    make(map[string]int),
		committed:       make(map[string]int),
	}
}

// effectiveMaxLocked calculates the dynamic effective token capacity of the pool,
// scaling maxTokens by downscaleFactor. Must be called while holding mu.
func (p *SharedTokenPool) effectiveMaxLocked() int {
	eff := int(float64(p.maxTokens) * p.downscaleFactor)
	if eff < 0 {
		return 0
	}
	return eff
}

// Reserve performs dynamic capacity admission and reserves requested uncommitted output headroom
// for streamID. Capacity admission enforces two invariants:
// 1. committed_tokens + reserved_tokens + requested_tokens <= effectiveMax
// 2. (committed + reserved + requested) * bytesPerToken <= totalGTTBytes
// Returns ErrPoolExhausted if capacity is exceeded.
func (p *SharedTokenPool) Reserve(streamID string, requested int) error {
	if streamID == "" {
		return ErrEmptyStreamID
	}
	if requested <= 0 {
		return ErrInvalidTokens
	}

	p.mu.Lock()
	defer p.mu.Unlock()

	effectiveMax := p.effectiveMaxLocked()
	projectedTokens := p.committedTokens + p.reservedTokens + requested

	// Check 1: Token limit against effectiveMax
	if projectedTokens > effectiveMax {
		return ErrPoolExhausted
	}

	// Check 2: Byte limit against total GTT capacity
	if p.bytesPerToken > 0 && int64(projectedTokens)*p.bytesPerToken > p.totalGTTBytes {
		return ErrPoolExhausted
	}

	p.reservedTokens += requested
	p.reservations[streamID] += requested
	return nil
}

// Commit shifts tokens from reserved output headroom to committed tokens for streamID.
// This is called as tokens are generated or upon turn completion.
// Returns ErrStreamNotFound if streamID has no reservation, or ErrExceedsReservation
// if tokens exceeds the reserved headroom.
func (p *SharedTokenPool) Commit(streamID string, tokens int) error {
	if streamID == "" {
		return ErrEmptyStreamID
	}
	if tokens <= 0 {
		return ErrInvalidTokens
	}

	p.mu.Lock()
	defer p.mu.Unlock()

	reserved, ok := p.reservations[streamID]
	if !ok || reserved == 0 {
		return ErrStreamNotFound
	}
	if tokens > reserved {
		return ErrExceedsReservation
	}

	// Shift from reserved to committed
	p.reservations[streamID] -= tokens
	if p.reservations[streamID] == 0 {
		delete(p.reservations, streamID)
	}
	p.reservedTokens -= tokens

	p.committed[streamID] += tokens
	p.committedTokens += tokens
	return nil
}

// Release releases both committed and reserved tokens for streamID, returning all
// associated GTT pages back to the shared pool.
func (p *SharedTokenPool) Release(streamID string) {
	if streamID == "" {
		return
	}

	p.mu.Lock()
	defer p.mu.Unlock()

	if res, ok := p.reservations[streamID]; ok {
		p.reservedTokens -= res
		delete(p.reservations, streamID)
	}
	if com, ok := p.committed[streamID]; ok {
		p.committedTokens -= com
		delete(p.committed, streamID)
	}
}

// ReleaseHeadroom immediately returns unused reserved output headroom for streamID
// upon terminal stop or finish reason, freeing the headroom for other streams while
// preserving the stream's committed tokens.
func (p *SharedTokenPool) ReleaseHeadroom(streamID string) {
	if streamID == "" {
		return
	}

	p.mu.Lock()
	defer p.mu.Unlock()

	if res, ok := p.reservations[streamID]; ok {
		p.reservedTokens -= res
		delete(p.reservations, streamID)
	}
}

// FitPoolToAvailableMemory dynamically adjusts the pool to available GTT memory,
// clamping maxTokens to what availableGTTBytes can support per the HALOGEN_KV_POOL_FIT pattern.
// If HALOGEN_KV_POOL_FIT is set in the environment, downscaleFactor is refreshed.
// It returns the new effective maximum token capacity.
func (p *SharedTokenPool) FitPoolToAvailableMemory(availableGTTBytes int64) int {
	p.mu.Lock()
	defer p.mu.Unlock()

	if val := os.Getenv(EnvHalogenKVPoolFit); val != "" {
		if f, err := strconv.ParseFloat(val, 64); err == nil && f > 0 {
			p.downscaleFactor = f
		}
	}

	p.totalGTTBytes = availableGTTBytes
	if p.bytesPerToken > 0 {
		memTokens := int(availableGTTBytes / p.bytesPerToken)
		if memTokens < 0 {
			memTokens = 0
		}
		if memTokens < p.maxTokens {
			p.maxTokens = memTokens
		}
	}
	return p.effectiveMaxLocked()
}

// MaxTokens returns the configured maximum token capacity of the pool.
func (p *SharedTokenPool) MaxTokens() int {
	p.mu.RLock()
	defer p.mu.RUnlock()
	return p.maxTokens
}

// CommittedTokens returns the total number of tokens currently committed across all streams.
func (p *SharedTokenPool) CommittedTokens() int {
	p.mu.RLock()
	defer p.mu.RUnlock()
	return p.committedTokens
}

// ReservedTokens returns the total number of tokens currently reserved as headroom across all streams.
func (p *SharedTokenPool) ReservedTokens() int {
	p.mu.RLock()
	defer p.mu.RUnlock()
	return p.reservedTokens
}

// FreeTokens returns the remaining unallocated token capacity that can be admitted
// before either the effectiveMax or totalGTTBytes limit is reached.
func (p *SharedTokenPool) FreeTokens() int {
	p.mu.RLock()
	defer p.mu.RUnlock()

	effectiveMax := p.effectiveMaxLocked()
	used := p.committedTokens + p.reservedTokens
	freeTokens := effectiveMax - used

	if p.bytesPerToken > 0 {
		byteFree := int(p.totalGTTBytes/p.bytesPerToken) - used
		if byteFree < freeTokens {
			freeTokens = byteFree
		}
	}

	if freeTokens < 0 {
		return 0
	}
	return freeTokens
}

// AllocatedBytes returns the total GTT bytes currently tied up by committed and reserved tokens.
func (p *SharedTokenPool) AllocatedBytes() int64 {
	p.mu.RLock()
	defer p.mu.RUnlock()
	return int64(p.committedTokens+p.reservedTokens) * p.bytesPerToken
}

// ActiveStreams returns the number of unique streams that have active reservations or committed tokens.
func (p *SharedTokenPool) ActiveStreams() int {
	p.mu.RLock()
	defer p.mu.RUnlock()

	active := make(map[string]struct{})
	for k, v := range p.reservations {
		if v > 0 {
			active[k] = struct{}{}
		}
	}
	for k, v := range p.committed {
		if v > 0 {
			active[k] = struct{}{}
		}
	}
	return len(active)
}

// EffectiveMaxTokens returns the dynamic admission ceiling after applying downscaleFactor.
func (p *SharedTokenPool) EffectiveMaxTokens() int {
	p.mu.RLock()
	defer p.mu.RUnlock()
	return p.effectiveMaxLocked()
}

// DownscaleFactor returns the current downscale factor.
func (p *SharedTokenPool) DownscaleFactor() float64 {
	p.mu.RLock()
	defer p.mu.RUnlock()
	return p.downscaleFactor
}

// TotalGTTBytes returns the configured GTT memory slab size in bytes.
func (p *SharedTokenPool) TotalGTTBytes() int64 {
	p.mu.RLock()
	defer p.mu.RUnlock()
	return p.totalGTTBytes
}

// BytesPerToken returns the memory cost per token in bytes.
func (p *SharedTokenPool) BytesPerToken() int64 {
	p.mu.RLock()
	defer p.mu.RUnlock()
	return p.bytesPerToken
}

// StreamUsage returns the committed and reserved token counts for a specific stream.
func (p *SharedTokenPool) StreamUsage(streamID string) (committed int, reserved int) {
	p.mu.RLock()
	defer p.mu.RUnlock()
	return p.committed[streamID], p.reservations[streamID]
}
