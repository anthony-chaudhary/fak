package model

import (
	"errors"
	"fmt"
	"sync"
)

// Standard errors for MTP speculative decoding and prompt cache coexistence.
var (
	ErrMTPCacheEmptyPrompt       = errors.New("model: empty prompt provided to MTP cache coexist")
	ErrMTPCacheSessionClosed     = errors.New("model: MTP cache coexist session is closed")
	ErrMTPCacheRollbackExceeded  = errors.New("model: prompt divergence exceeds recurrent rollback depth")
	ErrMTPCacheInvalidCheckpoint = errors.New("model: invalid MTP cache checkpoint")
	ErrMTPCacheWipedNoTolerance  = errors.New("model: empty data_spec rollback wiped prompt cache (tolerance disabled)")
)

// MTPPromptCacheEntry stores a cached prompt prefix and its session snapshot.
type MTPPromptCacheEntry struct {
	Prompt   []int
	Snapshot *PrefixSnapshot
	HitCount int
}

// MTPPromptCache is a thread-safe prompt cache store.
type MTPPromptCache struct {
	mu      sync.RWMutex
	entries map[string]*MTPPromptCacheEntry
}

// NewMTPPromptCache initializes an empty prompt cache.
func NewMTPPromptCache() *MTPPromptCache {
	return &MTPPromptCache{
		entries: make(map[string]*MTPPromptCacheEntry),
	}
}

func promptKey(tokens []int) string {
	return fmt.Sprint(tokens)
}

// Put stores a snapshot for the given prompt tokens.
func (c *MTPPromptCache) Put(prompt []int, snapshot *PrefixSnapshot) error {
	if c == nil {
		return nil
	}
	c.mu.Lock()
	defer c.mu.Unlock()

	key := promptKey(prompt)
	if existing, ok := c.entries[key]; ok && existing.Snapshot != nil {
		existing.Snapshot.Close()
	}

	var snapClone *PrefixSnapshot
	if snapshot != nil {
		var err error
		snapClone, err = snapshot.Clone()
		if err != nil {
			return fmt.Errorf("model: prompt cache clone failed: %w", err)
		}
	}

	c.entries[key] = &MTPPromptCacheEntry{
		Prompt:   append([]int(nil), prompt...),
		Snapshot: snapClone,
		HitCount: 0,
	}
	return nil
}

// MatchPrefix finds the cached entry with the longest common prefix for the given prompt.
func (c *MTPPromptCache) MatchPrefix(prompt []int) (*MTPPromptCacheEntry, int) {
	if c == nil {
		return nil, 0
	}
	c.mu.Lock()
	defer c.mu.Unlock()

	var bestEntry *MTPPromptCacheEntry
	bestLen := 0

	for _, entry := range c.entries {
		matchLen := commonPrefixTokens(entry.Prompt, prompt)
		if matchLen > bestLen || (matchLen == bestLen && matchLen > 0 && entry != nil && bestEntry != nil && entry.HitCount > bestEntry.HitCount) {
			bestLen = matchLen
			bestEntry = entry
		}
	}

	if bestEntry != nil {
		bestEntry.HitCount++
	}
	return bestEntry, bestLen
}

// Get finds an exact matching cached entry for the prompt.
func (c *MTPPromptCache) Get(prompt []int) *MTPPromptCacheEntry {
	if c == nil {
		return nil
	}
	c.mu.Lock()
	defer c.mu.Unlock()

	key := promptKey(prompt)
	entry, ok := c.entries[key]
	if ok && entry != nil {
		entry.HitCount++
		return entry
	}
	return nil
}

// Len returns the number of cached prompt entries.
func (c *MTPPromptCache) Len() int {
	if c == nil {
		return 0
	}
	c.mu.RLock()
	defer c.mu.RUnlock()
	return len(c.entries)
}

// Clear releases all snapshots and clears the prompt cache.
func (c *MTPPromptCache) Clear() {
	if c == nil {
		return
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	for _, entry := range c.entries {
		if entry.Snapshot != nil {
			entry.Snapshot.Close()
		}
	}
	c.entries = make(map[string]*MTPPromptCacheEntry)
}

// MTPCacheCheckpoint captures the target and draft state for rollback.
type MTPCacheCheckpoint struct {
	owner          *MTPCacheCoexistSession
	TargetSnapshot *PrefixSnapshot
	DataSpec       []int // Speculative draft tokens (empty at prefill time!)
	PromptLen      int
	CommittedLen   int
	IsPrefill      bool
	TargetCacheLen int
	RecurrentPos   int
}

// Close releases resources associated with this checkpoint.
func (cp *MTPCacheCheckpoint) Close() {
	if cp == nil {
		return
	}
	if cp.TargetSnapshot != nil {
		cp.TargetSnapshot.Close()
		cp.TargetSnapshot = nil
	}
	cp.owner = nil
}

// MTPCacheCoexistConfig sets parameters for MTP speculative decoding with prompt cache coexistence.
type MTPCacheCoexistConfig struct {
	MaxRecurrentRollbackDepth int  // maximum tokens recurrent state can safely roll back (default 4)
	DraftDepth                int  // speculative draft depth K (1..4, default 4)
	EmptyDataSpecTolerance    bool // when true, empty DataSpec in prefill checkpoint does not wipe prompt cache
}

// DefaultMTPCacheCoexistConfig returns the canonical production configuration for Qwen 3.8 MTP coexistence.
func DefaultMTPCacheCoexistConfig() MTPCacheCoexistConfig {
	return MTPCacheCoexistConfig{
		MaxRecurrentRollbackDepth: Qwen35MTPMaxDraftDepth,
		DraftDepth:                Qwen35MTPMaxDraftDepth,
		EmptyDataSpecTolerance:    true,
	}
}

// MTPPrefillResult reports the outcome of prefilling with prompt cache coexistence.
type MTPPrefillResult struct {
	CacheHit            bool
	CachedTokens        int
	PrefilledTokens     int
	Divergence          int
	RollbackApplied     bool
	ColdPrefillFallback bool
	MTPRetained         bool
	DraftReady          bool
}

// MTPCacheCoexistSession manages MTP speculative decoding alongside prompt cache coexistence.
type MTPCacheCoexistSession struct {
	Target           *Session
	DraftSession     *Qwen35MTPDraftSession
	Config           MTPCacheCoexistConfig
	PromptCache      *MTPPromptCache
	CachedPrompt     []int
	ActiveCheckpoint *MTPCacheCheckpoint
	MTPActive        bool
	Closed           bool

	// Telemetry & metrics
	RecurrentRollbackCount   int
	ColdPrefillFallbackCount int
	EmptySpecRollbackCount   int
	CacheRetentionCount      int
	PrefillCount             int
	DraftCount               int
}

// NewMTPCacheCoexistSession creates an MTP speculative decoding session with prompt cache coexistence.
func NewMTPCacheCoexistSession(target *Session, promptCache *MTPPromptCache, cfg MTPCacheCoexistConfig) (*MTPCacheCoexistSession, error) {
	if target == nil || target.M == nil {
		return nil, errors.New("model: valid target session is required for MTP cache coexistence")
	}
	if cfg.MaxRecurrentRollbackDepth <= 0 {
		cfg.MaxRecurrentRollbackDepth = Qwen35MTPMaxDraftDepth
	}
	if cfg.DraftDepth <= 0 {
		cfg.DraftDepth = Qwen35MTPMaxDraftDepth
	}

	target.captureTargetHidden = true

	s := &MTPCacheCoexistSession{
		Target:      target,
		Config:      cfg,
		PromptCache: promptCache,
		MTPActive:   true,
	}
	return s, nil
}

// Checkpoint captures the current target state and draft speculative tokens.
func (s *MTPCacheCoexistSession) Checkpoint(isPrefill bool, dataSpec []int) (*MTPCacheCheckpoint, error) {
	if s == nil || s.Closed {
		return nil, ErrMTPCacheSessionClosed
	}

	snap, err := s.Target.PrefixSnapshot()
	if err != nil {
		return nil, fmt.Errorf("model: failed to snapshot target session for checkpoint: %w", err)
	}

	if s.ActiveCheckpoint != nil {
		s.ActiveCheckpoint.Close()
	}

	cp := &MTPCacheCheckpoint{
		owner:          s,
		TargetSnapshot: snap,
		DataSpec:       append([]int(nil), dataSpec...),
		PromptLen:      len(s.CachedPrompt),
		CommittedLen:   s.Target.Cache.Len(),
		IsPrefill:      isPrefill,
		TargetCacheLen: s.Target.Cache.Len(),
		RecurrentPos:   len(s.Target.targetHidden),
	}
	s.ActiveCheckpoint = cp
	return cp, nil
}

// Rollback restores state to the checkpoint.
// Crucially, when DataSpec is empty (prefill-time checkpoint or zero speculative tokens):
// - Under EmptyDataSpecTolerance = true: the prompt cache and KV cache are preserved without wiping.
// - Under EmptyDataSpecTolerance = false: emulates the unpatched behavior that resets/wipes cache.
func (s *MTPCacheCoexistSession) Rollback(cp *MTPCacheCheckpoint) error {
	if s == nil || s.Closed {
		return ErrMTPCacheSessionClosed
	}
	if cp == nil || cp.owner != s {
		return ErrMTPCacheInvalidCheckpoint
	}

	// Empty data_spec handling:
	if len(cp.DataSpec) == 0 {
		if !s.Config.EmptyDataSpecTolerance {
			// Buggy/unpatched behavior: wipe prompt cache and target cache
			s.Target.Cache = NewKVCache(s.Target.M.Cfg)
			s.CachedPrompt = nil
			if s.PromptCache != nil {
				s.PromptCache.Clear()
			}
			s.ActiveCheckpoint = nil
			return ErrMTPCacheWipedNoTolerance
		}

		// Empty-data_spec tolerance enabled:
		// Restore the target session state cleanly without wiping the prompt cache or KV prefix.
		if cp.TargetSnapshot != nil {
			clone, err := cp.TargetSnapshot.Clone()
			if err != nil {
				return fmt.Errorf("model: failed to clone checkpoint target snapshot: %w", err)
			}
			if err := clone.Restore(s.Target); err != nil {
				clone.Close()
				return fmt.Errorf("model: failed to restore target snapshot: %w", err)
			}
		}
		if len(s.CachedPrompt) > cp.PromptLen {
			s.CachedPrompt = s.CachedPrompt[:cp.PromptLen]
		}
		s.EmptySpecRollbackCount++
		s.CacheRetentionCount++
		s.ActiveCheckpoint = nil
		return nil
	}

	// Normal speculative rollback with non-empty draft state:
	if cp.TargetSnapshot != nil {
		clone, err := cp.TargetSnapshot.Clone()
		if err != nil {
			return fmt.Errorf("model: failed to clone checkpoint target snapshot: %w", err)
		}
		if err := clone.Restore(s.Target); err != nil {
			clone.Close()
			return fmt.Errorf("model: failed to restore target snapshot: %w", err)
		}
	}
	if len(s.CachedPrompt) > cp.PromptLen {
		s.CachedPrompt = s.CachedPrompt[:cp.PromptLen]
	}
	s.ActiveCheckpoint = nil
	return nil
}

// PrefillWithCache prefills a prompt while leveraging prompt caching and MTP coexistence.
// When prompt divergence exceeds the recurrent rollback depth, it performs a graceful fallback
// to cold prefill without wiping the prompt cache store and without disabling MTP for subsequent turns.
func (s *MTPCacheCoexistSession) PrefillWithCache(prompt []int) (*MTPPrefillResult, error) {
	if s == nil || s.Closed {
		return nil, ErrMTPCacheSessionClosed
	}
	if len(prompt) == 0 {
		return nil, ErrMTPCacheEmptyPrompt
	}

	s.PrefillCount++

	// Determine common prefix length
	sharedLen := 0
	if len(s.CachedPrompt) > 0 {
		sharedLen = commonPrefixTokens(s.CachedPrompt, prompt)
	} else if s.PromptCache != nil && s.PromptCache.Len() > 0 {
		entry, matchLen := s.PromptCache.MatchPrefix(prompt)
		if entry != nil && matchLen > 0 && entry.Snapshot != nil {
			clone, err := entry.Snapshot.Clone()
			if err == nil {
				if restoreErr := clone.Restore(s.Target); restoreErr == nil {
					s.CachedPrompt = append([]int(nil), entry.Prompt[:matchLen]...)
					sharedLen = matchLen
				} else {
					clone.Close()
				}
			}
		}
	}

	divergence := len(s.CachedPrompt) - sharedLen

	// Case 1: Fresh start (no cached prompt in current session)
	if len(s.CachedPrompt) == 0 {
		s.Target.Cache = NewKVCache(s.Target.M.Cfg)
		s.Target.captureTargetHidden = true
		s.Target.Prefill(prompt)
		s.CachedPrompt = append([]int(nil), prompt...)

		// Save snapshot into prompt cache
		if s.PromptCache != nil {
			if snap, err := s.Target.PrefixSnapshot(); err == nil {
				_ = s.PromptCache.Put(prompt, snap)
				snap.Close()
			}
		}

		err := s.ensureDraftSession()
		s.CacheRetentionCount++
		return &MTPPrefillResult{
			CacheHit:            false,
			CachedTokens:        0,
			PrefilledTokens:     len(prompt),
			Divergence:          0,
			RollbackApplied:     false,
			ColdPrefillFallback: false,
			MTPRetained:         err == nil,
			DraftReady:          err == nil,
		}, nil
	}

	// Case 2: Zero divergence (cached prefix is a prefix of or identical to prompt)
	if divergence == 0 {
		prefilled := 0
		if sharedLen < len(prompt) {
			remainder := prompt[sharedLen:]
			for _, token := range remainder {
				s.Target.Step(token)
			}
			prefilled = len(remainder)
			s.CachedPrompt = append([]int(nil), prompt...)

			// Update prompt cache with extended prompt
			if s.PromptCache != nil {
				if snap, err := s.Target.PrefixSnapshot(); err == nil {
					_ = s.PromptCache.Put(prompt, snap)
					snap.Close()
				}
			}
		}

		err := s.ensureDraftSession()
		s.CacheRetentionCount++
		return &MTPPrefillResult{
			CacheHit:            true,
			CachedTokens:        sharedLen,
			PrefilledTokens:     prefilled,
			Divergence:          0,
			RollbackApplied:     false,
			ColdPrefillFallback: false,
			MTPRetained:         err == nil,
			DraftReady:          err == nil,
		}, nil
	}

	// Case 3: Divergence within recurrent rollback depth -> Recurrent rollback
	if divergence <= s.Config.MaxRecurrentRollbackDepth {
		// Roll back target cache to sharedLen
		if s.Target.Cache.Len() > sharedLen {
			_, _ = s.Target.Cache.TryEvict(sharedLen, divergence)
		}

		// Roll back target hidden history
		s.Target.targetHiddenMu.Lock()
		if len(s.Target.targetHidden) > sharedLen {
			s.Target.targetHidden = s.Target.targetHidden[:sharedLen]
			s.Target.targetHiddenTokens = s.Target.targetHiddenTokens[:sharedLen]
		}
		s.Target.targetHiddenMu.Unlock()

		// Prefill divergent tail
		remainder := prompt[sharedLen:]
		for _, token := range remainder {
			s.Target.Step(token)
		}
		s.CachedPrompt = append([]int(nil), prompt...)
		s.RecurrentRollbackCount++
		s.CacheRetentionCount++

		if s.PromptCache != nil {
			if snap, err := s.Target.PrefixSnapshot(); err == nil {
				_ = s.PromptCache.Put(prompt, snap)
				snap.Close()
			}
		}

		err := s.ensureDraftSession()
		return &MTPPrefillResult{
			CacheHit:            true,
			CachedTokens:        sharedLen,
			PrefilledTokens:     len(remainder),
			Divergence:          divergence,
			RollbackApplied:     true,
			ColdPrefillFallback: false,
			MTPRetained:         err == nil,
			DraftReady:          err == nil,
		}, nil
	}

	// Case 4: Divergence EXCEEDS recurrent rollback depth -> Graceful fallback!
	// We fall back to cold prefill for the prompt without wiping the prompt cache store,
	// and reinitialize MTP so it remains fully available on subsequent turns.
	s.ColdPrefillFallbackCount++

	// Reset active target session working state for the new prompt
	s.Target.Cache = NewKVCache(s.Target.M.Cfg)
	s.Target.captureTargetHidden = true
	s.Target.Prefill(prompt)
	s.CachedPrompt = append([]int(nil), prompt...)

	if s.PromptCache != nil {
		if snap, err := s.Target.PrefixSnapshot(); err == nil {
			_ = s.PromptCache.Put(prompt, snap)
			snap.Close()
		}
	}

	// Cleanly recreate/recover the draft session
	if s.DraftSession != nil {
		s.DraftSession.Close()
		s.DraftSession = nil
	}
	err := s.ensureDraftSession()

	return &MTPPrefillResult{
		CacheHit:            false,
		CachedTokens:        0,
		PrefilledTokens:     len(prompt),
		Divergence:          divergence,
		RollbackApplied:     false,
		ColdPrefillFallback: true,
		MTPRetained:         err == nil,
		DraftReady:          err == nil,
	}, nil
}

// Draft generates speculative draft tokens for the committed prefix.
// If the draft session was cleared or needs recovery, it re-establishes the draft session cleanly.
func (s *MTPCacheCoexistSession) Draft(committed []int) ([]int, error) {
	if s == nil || s.Closed {
		return nil, ErrMTPCacheSessionClosed
	}

	if s.DraftSession == nil {
		if err := s.ensureDraftSession(); err != nil {
			return nil, fmt.Errorf("model: failed to initialize MTP draft session: %w", err)
		}
	}

	draft := s.DraftSession.Propose(committed)
	if err := s.DraftSession.Err(); err != nil {
		// Attempt draft recovery
		s.DraftSession.Close()
		s.DraftSession = nil
		if recErr := s.ensureDraftSession(); recErr != nil {
			return nil, fmt.Errorf("model: draft proposal failed: %v; recovery failed: %w", err, recErr)
		}
		draft = s.DraftSession.Propose(committed)
		if err2 := s.DraftSession.Err(); err2 != nil {
			return nil, fmt.Errorf("model: draft proposal failed after recovery: %w", err2)
		}
	}

	s.DraftCount++
	return draft, nil
}

// ensureDraftSession establishes an active MTP draft session connected to the target.
func (s *MTPCacheCoexistSession) ensureDraftSession() error {
	if s.DraftSession != nil && s.DraftSession.Err() == nil {
		return nil
	}
	if s.DraftSession != nil {
		s.DraftSession.Close()
		s.DraftSession = nil
	}
	d, err := NewQwen35MTPDraftSession(s.Target, s.Config.DraftDepth)
	if err != nil {
		s.MTPActive = false
		return err
	}
	s.DraftSession = d
	s.MTPActive = true
	return nil
}

// Close releases all checkpoints, draft sessions, and resources.
func (s *MTPCacheCoexistSession) Close() {
	if s == nil || s.Closed {
		return
	}
	s.Closed = true
	if s.ActiveCheckpoint != nil {
		s.ActiveCheckpoint.Close()
		s.ActiveCheckpoint = nil
	}
	if s.DraftSession != nil {
		s.DraftSession.Close()
		s.DraftSession = nil
	}
	s.CachedPrompt = nil
}

func commonPrefixTokens(a, b []int) int {
	n := len(a)
	if len(b) < n {
		n = len(b)
	}
	for i := 0; i < n; i++ {
		if a[i] != b[i] {
			return i
		}
	}
	return n
}
