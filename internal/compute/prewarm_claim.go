package compute

import (
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"sync"
	"time"
)

// PrewarmEntryState represents the lifecycle phase of a prewarm entry.
type PrewarmEntryState string

const (
	PrewarmStateUnknown   PrewarmEntryState = "unknown"
	PrewarmStatePreparing PrewarmEntryState = "preparing"
	PrewarmStateReady     PrewarmEntryState = "ready"
	PrewarmStateClaimed   PrewarmEntryState = "claimed"
	PrewarmStateFailed    PrewarmEntryState = "failed"
	PrewarmStateExpired   PrewarmEntryState = "expired"
)

// String returns the string representation of the prewarm entry state.
func (s PrewarmEntryState) String() string {
	if s == "" {
		return string(PrewarmStateUnknown)
	}
	return string(s)
}

// PrewarmClaimStatus represents the outcome of a prewarm claim or prepare request.
type PrewarmClaimStatus string

const (
	ClaimStatusOK                 PrewarmClaimStatus = "OK"
	ClaimStatusNotFound           PrewarmClaimStatus = "NOT_FOUND"
	ClaimStatusNotReady           PrewarmClaimStatus = "NOT_READY"
	ClaimStatusAlreadyClaimed     PrewarmClaimStatus = "ALREADY_CLAIMED"
	ClaimStatusExpired            PrewarmClaimStatus = "EXPIRED"
	ClaimStatusConfigMismatch     PrewarmClaimStatus = "CONFIG_MISMATCH"
	ClaimStatusFailed             PrewarmClaimStatus = "FAILED"
	ClaimStatusSecretClassRefused PrewarmClaimStatus = "SECRET_CLASS_REFUSED"
)

// String returns the string representation of the prewarm claim status.
func (s PrewarmClaimStatus) String() string {
	return string(s)
}

// Sentinel errors for prewarm claim registry operations.
var (
	ErrSecretClassRefused  = errors.New("compute: prewarm refused for secret-classified configuration")
	ErrWarmKeyRequired     = errors.New("compute: warmKey is required")
	ErrEntryNotFound       = errors.New("compute: prewarm entry not found")
	ErrPrewarmExpired      = errors.New("compute: prewarm entry has expired")
	ErrPrewarmStateInvalid = errors.New("compute: invalid prewarm state transition")
)

const (
	// DefaultPrewarmMaxEntries is the default capacity limit for a claim registry.
	DefaultPrewarmMaxEntries = 1024
	// DefaultPrewarmTTL is the default time-to-live for a prepared entry.
	DefaultPrewarmTTL = 5 * time.Minute
)

// PrewarmEffectiveConfig defines the configuration fingerprint required to validate
// whether a prewarmed resident state matches a claimant's expectations.
type PrewarmEffectiveConfig struct {
	// Model configuration
	ModelID        string `json:"model_id"`
	LayerCount     int    `json:"layer_count"`
	HeadCount      int    `json:"head_count"`
	KVHeadCount    int    `json:"kv_head_count"`
	HeadDimension  int    `json:"head_dimension"`
	PageSizeTokens int    `json:"page_size_tokens"`
	Quantization   string `json:"quantization"`

	// Transport and topology configuration
	BackendType      string `json:"backend_type"`
	DevicePCI        string `json:"device_pci"`
	NUMANode         int    `json:"numa_node"`
	Transport        string `json:"transport"`
	NICStripingCount int    `json:"nic_striping_count"`

	// Security and identity configuration (presence flags only, zero raw secrets)
	PasswordPresent  bool   `json:"password_present"`
	SecretClassified bool   `json:"secret_classified"`
	IdentityClass    string `json:"identity_class"`
}

// Digest computes a canonical SHA-256 hex digest across all behavior-affecting fields.
func (c PrewarmEffectiveConfig) Digest() string {
	h := sha256.New()
	fmt.Fprintf(h, "model_id=%s\n", c.ModelID)
	fmt.Fprintf(h, "layer_count=%d\n", c.LayerCount)
	fmt.Fprintf(h, "head_count=%d\n", c.HeadCount)
	fmt.Fprintf(h, "kv_head_count=%d\n", c.KVHeadCount)
	fmt.Fprintf(h, "head_dimension=%d\n", c.HeadDimension)
	fmt.Fprintf(h, "page_size_tokens=%d\n", c.PageSizeTokens)
	fmt.Fprintf(h, "quantization=%s\n", c.Quantization)
	fmt.Fprintf(h, "backend_type=%s\n", c.BackendType)
	fmt.Fprintf(h, "device_pci=%s\n", c.DevicePCI)
	fmt.Fprintf(h, "numa_node=%d\n", c.NUMANode)
	fmt.Fprintf(h, "transport=%s\n", c.Transport)
	fmt.Fprintf(h, "nic_striping_count=%d\n", c.NICStripingCount)
	fmt.Fprintf(h, "password_present=%t\n", c.PasswordPresent)
	fmt.Fprintf(h, "secret_classified=%t\n", c.SecretClassified)
	fmt.Fprintf(h, "identity_class=%s\n", c.IdentityClass)
	return hex.EncodeToString(h.Sum(nil))
}

// PrewarmClaimReceipt represents the immutable receipt of a successfully claimed prewarm entry.
type PrewarmClaimReceipt struct {
	WarmKey      string    `json:"warm_key"`
	ConfigDigest string    `json:"config_digest"`
	TokensWarmed int       `json:"tokens_warmed"`
	ClaimantID   string    `json:"claimant_id"`
	CreatedAt    time.Time `json:"created_at"`
	ClaimedAt    time.Time `json:"claimed_at"`
}

type prewarmEntry struct {
	warmKey      string
	cfg          PrewarmEffectiveConfig
	configDigest string
	state        PrewarmEntryState
	tokensWarmed int
	failReason   string
	claimantID   string
	createdAt    time.Time
	claimedAt    time.Time
	expiresAt    time.Time
}

// PrewarmClaimRegistry provides a thread-safe, bounded-capacity registry
// for preparing, completing, and single-claiming prewarmed artifacts.
type PrewarmClaimRegistry struct {
	mu         sync.Mutex
	maxEntries int
	defaultTTL time.Duration
	clock      func() time.Time
	entries    map[string]*prewarmEntry
}

// NewPrewarmClaimRegistry creates a bounded-capacity registry with the specified limits.
func NewPrewarmClaimRegistry(maxEntries int, defaultTTL time.Duration) *PrewarmClaimRegistry {
	if maxEntries <= 0 {
		maxEntries = DefaultPrewarmMaxEntries
	}
	if defaultTTL <= 0 {
		defaultTTL = DefaultPrewarmTTL
	}
	return &PrewarmClaimRegistry{
		maxEntries: maxEntries,
		defaultTTL: defaultTTL,
		entries:    make(map[string]*prewarmEntry),
	}
}

// SetClock configures a custom time source for deterministic testing.
func (r *PrewarmClaimRegistry) SetClock(fn func() time.Time) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.clock = fn
}

func (r *PrewarmClaimRegistry) now() time.Time {
	if r.clock != nil {
		return r.clock()
	}
	return time.Now().UTC()
}

// Prepare initiates a prewarm registration. It enforces the secret-class default-deny fence,
// assigns the initial PrewarmStatePreparing state (making it invisible to claims), and sets TTL.
func (r *PrewarmClaimRegistry) Prepare(warmKey string, cfg PrewarmEffectiveConfig, ttl time.Duration) (PrewarmClaimStatus, error) {
	if warmKey == "" {
		return ClaimStatusNotFound, ErrWarmKeyRequired
	}
	if cfg.SecretClassified {
		return ClaimStatusSecretClassRefused, ErrSecretClassRefused
	}

	r.mu.Lock()
	defer r.mu.Unlock()

	if ttl <= 0 {
		ttl = r.defaultTTL
	}
	now := r.now()
	expiresAt := now.Add(ttl)

	if _, exists := r.entries[warmKey]; !exists && len(r.entries) >= r.maxEntries {
		r.reapLocked(now)
		if len(r.entries) >= r.maxEntries {
			r.purgeOldestLocked()
		}
	}

	digest := cfg.Digest()
	r.entries[warmKey] = &prewarmEntry{
		warmKey:      warmKey,
		cfg:          cfg,
		configDigest: digest,
		state:        PrewarmStatePreparing,
		createdAt:    now,
		expiresAt:    expiresAt,
	}
	return ClaimStatusOK, nil
}

// CommitSuccess transitions an entry from PrewarmStatePreparing to PrewarmStateReady.
func (r *PrewarmClaimRegistry) CommitSuccess(warmKey string, tokensWarmed int) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	entry, ok := r.entries[warmKey]
	if !ok {
		return ErrEntryNotFound
	}

	now := r.now()
	if !entry.expiresAt.IsZero() && now.After(entry.expiresAt) {
		entry.state = PrewarmStateExpired
		return ErrPrewarmExpired
	}

	if entry.state != PrewarmStatePreparing {
		return fmt.Errorf("%w: cannot commit success from state %s", ErrPrewarmStateInvalid, entry.state)
	}

	entry.state = PrewarmStateReady
	entry.tokensWarmed = tokensWarmed
	return nil
}

// CommitFailure transitions an entry to PrewarmStateFailed with an explanatory reason.
func (r *PrewarmClaimRegistry) CommitFailure(warmKey string, reason string) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	entry, ok := r.entries[warmKey]
	if !ok {
		return ErrEntryNotFound
	}

	entry.state = PrewarmStateFailed
	entry.failReason = reason
	return nil
}

// Claim attempts to acquire single-use ownership of a prewarmed entry.
func (r *PrewarmClaimRegistry) Claim(warmKey string, expectedConfigDigest string, claimantID string) (PrewarmClaimStatus, *PrewarmClaimReceipt) {
	r.mu.Lock()
	defer r.mu.Unlock()

	entry, ok := r.entries[warmKey]
	if !ok {
		return ClaimStatusNotFound, nil
	}

	if entry.state == PrewarmStateClaimed {
		return ClaimStatusAlreadyClaimed, nil
	}

	now := r.now()
	if entry.state == PrewarmStatePreparing {
		if !entry.expiresAt.IsZero() && now.After(entry.expiresAt) {
			entry.state = PrewarmStateExpired
			return ClaimStatusExpired, nil
		}
		return ClaimStatusNotReady, nil
	}

	if entry.state == PrewarmStateFailed {
		return ClaimStatusFailed, nil
	}

	if entry.state == PrewarmStateExpired || (!entry.expiresAt.IsZero() && now.After(entry.expiresAt)) {
		entry.state = PrewarmStateExpired
		return ClaimStatusExpired, nil
	}

	if expectedConfigDigest != entry.configDigest {
		return ClaimStatusConfigMismatch, nil
	}

	if entry.state != PrewarmStateReady {
		return ClaimStatusNotReady, nil
	}

	entry.state = PrewarmStateClaimed
	entry.claimedAt = now
	entry.claimantID = claimantID

	receipt := &PrewarmClaimReceipt{
		WarmKey:      entry.warmKey,
		ConfigDigest: entry.configDigest,
		TokensWarmed: entry.tokensWarmed,
		ClaimantID:   claimantID,
		CreatedAt:    entry.createdAt,
		ClaimedAt:    now,
	}
	return ClaimStatusOK, receipt
}

// Reap purges expired and claimed entries, returning the number of entries removed.
func (r *PrewarmClaimRegistry) Reap(now time.Time) int {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.reapLocked(now)
}

func (r *PrewarmClaimRegistry) reapLocked(now time.Time) int {
	if now.IsZero() {
		now = r.now()
	}
	purged := 0
	for k, entry := range r.entries {
		isExpired := !entry.expiresAt.IsZero() && now.After(entry.expiresAt)
		if entry.state == PrewarmStateClaimed || entry.state == PrewarmStateExpired || isExpired {
			delete(r.entries, k)
			purged++
		}
	}
	return purged
}

func (r *PrewarmClaimRegistry) purgeOldestLocked() {
	var oldestKey string
	var oldestTime time.Time
	first := true
	for k, entry := range r.entries {
		if first || entry.createdAt.Before(oldestTime) {
			oldestKey = k
			oldestTime = entry.createdAt
			first = false
		}
	}
	if oldestKey != "" {
		delete(r.entries, oldestKey)
	}
}

// Len returns the current count of stored entries in the registry.
func (r *PrewarmClaimRegistry) Len() int {
	r.mu.Lock()
	defer r.mu.Unlock()
	return len(r.entries)
}

// GetState returns the current lifecycle state of an entry by warmKey.
func (r *PrewarmClaimRegistry) GetState(warmKey string) (PrewarmEntryState, bool) {
	r.mu.Lock()
	defer r.mu.Unlock()
	entry, ok := r.entries[warmKey]
	if !ok {
		return PrewarmStateUnknown, false
	}
	now := r.now()
	if !entry.expiresAt.IsZero() && now.After(entry.expiresAt) {
		entry.state = PrewarmStateExpired
	}
	return entry.state, true
}
