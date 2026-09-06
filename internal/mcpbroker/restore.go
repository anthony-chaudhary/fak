package mcpbroker

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"strings"
	"sync"
)

// Error definitions for exact-original preservation and restoration.
var (
	// ErrRestoreUnauthorized is returned when a caller attempts to restore a handle
	// not owned by or bound to the requesting session or trace ID (fail-closed security).
	ErrRestoreUnauthorized = errors.New("mcpbroker: unauthorized restore handle access")

	// ErrRestoreNotFound is returned when the requested restore handle does not exist.
	ErrRestoreNotFound = errors.New("mcpbroker: restore handle not found")

	// ErrRestoreSizeExceeded is returned when exact-original preservation exceeds
	// the configured per-entry or cumulative storage size cap.
	ErrRestoreSizeExceeded = errors.New("mcpbroker: exact-original size cap exceeded")

	// ErrRestorePreserveFailed is returned when exact-original retention storage fails.
	ErrRestorePreserveFailed = errors.New("mcpbroker: exact-original preservation failed")
)

// RestoreStore manages content-addressed storage (CAS) of exact original payload bytes
// produced prior to structured JSON compression, bound strictly to producing session/trace IDs.
type RestoreStore struct {
	mu            sync.RWMutex
	maxBytes      int                        // maximum cumulative bytes allowed in store
	maxEntryBytes int                        // maximum bytes allowed for a single original entry
	currBytes     int                        // current cumulative stored byte count
	entries       map[string][]byte          // normalized sha256 hex -> original bytes
	owners        map[string]map[string]bool // normalized sha256 hex -> set of authorized session IDs
	injectedErr   error                      // simulated failure for testing
}

// RestoreStoreOption configures a RestoreStore instance.
type RestoreStoreOption func(*RestoreStore)

// WithStoreMaxBytes sets the cumulative storage capacity bound for the RestoreStore.
func WithStoreMaxBytes(maxBytes int) RestoreStoreOption {
	return func(s *RestoreStore) {
		s.maxBytes = maxBytes
	}
}

// WithStoreMaxEntryBytes sets the per-entry byte limit for the RestoreStore.
func WithStoreMaxEntryBytes(maxEntry int) RestoreStoreOption {
	return func(s *RestoreStore) {
		s.maxEntryBytes = maxEntry
	}
}

// NewRestoreStore creates a new in-memory CAS RestoreStore with default boundaries.
func NewRestoreStore(opts ...RestoreStoreOption) *RestoreStore {
	s := &RestoreStore{
		maxBytes:      64 << 20,                      // 64 MiB cumulative default
		maxEntryBytes: maxStructuredCompressionBytes, // 16 MiB per-entry default
		entries:       make(map[string][]byte),
		owners:        make(map[string]map[string]bool),
	}
	for _, opt := range opts {
		opt(s)
	}
	return s
}

// SetInjectedError configures a simulated failure error for testing store failures.
func (s *RestoreStore) SetInjectedError(err error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.injectedErr = err
}

// Store records the exact original bytes under a content-addressed SHA-256 handle
// and binds ownership strictly to sessionID. Returns the handle or an error.
func (s *RestoreStore) Store(sessionID string, data []byte) (string, error) {
	sessionID = strings.TrimSpace(sessionID)
	if sessionID == "" {
		return "", ErrRestoreUnauthorized
	}
	if len(data) == 0 {
		return "", ErrRestorePreserveFailed
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	if s.injectedErr != nil {
		return "", s.injectedErr
	}

	if s.maxEntryBytes > 0 && len(data) > s.maxEntryBytes {
		return "", ErrRestoreSizeExceeded
	}

	sum := sha256.Sum256(data)
	hash := hex.EncodeToString(sum[:])
	handle := "sha256:" + hash

	// If entry is not yet stored, check total capacity and allocate entry
	if _, exists := s.entries[hash]; !exists {
		if s.maxBytes > 0 && s.currBytes+len(data) > s.maxBytes {
			return "", ErrRestoreSizeExceeded
		}
		copied := make([]byte, len(data))
		copy(copied, data)
		s.entries[hash] = copied
		s.currBytes += len(data)
	}

	// Register session ownership
	if s.owners[hash] == nil {
		s.owners[hash] = make(map[string]bool)
	}
	s.owners[hash][sessionID] = true

	return handle, nil
}

// RestoreOriginal retrieves the exact original uncompressed bytes for handle, strictly
// enforcing that sessionID is an authorized owner. Returns ErrRestoreUnauthorized on
// cross-session retrieval or empty session ID, and ErrRestoreNotFound if the handle does not exist.
func (s *RestoreStore) RestoreOriginal(sessionID string, handle string) ([]byte, error) {
	sessionID = strings.TrimSpace(sessionID)
	if sessionID == "" {
		return nil, ErrRestoreUnauthorized
	}

	handle = strings.TrimSpace(handle)
	if handle == "" {
		return nil, ErrRestoreNotFound
	}

	hash := strings.TrimPrefix(handle, "sha256:")
	hash = strings.ToLower(strings.TrimSpace(hash))
	if hash == "" {
		return nil, ErrRestoreNotFound
	}

	s.mu.RLock()
	defer s.mu.RUnlock()

	data, exists := s.entries[hash]
	if !exists {
		return nil, ErrRestoreNotFound
	}

	sessionOwners, ok := s.owners[hash]
	if !ok || !sessionOwners[sessionID] {
		return nil, ErrRestoreUnauthorized
	}

	out := make([]byte, len(data))
	copy(out, data)
	return out, nil
}

// PurgeSession removes session ownership for all stored handles and frees entries
// that have no remaining owners.
func (s *RestoreStore) PurgeSession(sessionID string) {
	sessionID = strings.TrimSpace(sessionID)
	if sessionID == "" {
		return
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	for hash, owners := range s.owners {
		if owners[sessionID] {
			delete(owners, sessionID)
			if len(owners) == 0 {
				delete(s.owners, hash)
				if data, ok := s.entries[hash]; ok {
					s.currBytes -= len(data)
					delete(s.entries, hash)
				}
			}
		}
	}
}

var (
	defaultStoreMu sync.Mutex
	defaultStore   *RestoreStore
)

// DefaultRestoreStore returns the package-level singleton RestoreStore.
func DefaultRestoreStore() *RestoreStore {
	defaultStoreMu.Lock()
	defer defaultStoreMu.Unlock()
	if defaultStore == nil {
		defaultStore = NewRestoreStore()
	}
	return defaultStore
}

// ResetDefaultRestoreStore resets the package-level singleton RestoreStore.
func ResetDefaultRestoreStore() {
	defaultStoreMu.Lock()
	defer defaultStoreMu.Unlock()
	defaultStore = NewRestoreStore()
}

// RestoreOriginal retrieves the exact original uncompressed bytes for handle and sessionID
// using the package default RestoreStore.
func RestoreOriginal(sessionID string, handle string) ([]byte, error) {
	return DefaultRestoreStore().RestoreOriginal(sessionID, handle)
}

// Context keys for propagating exact-original options through call pipelines.
type exactOriginalContextKey struct{}
type sessionIDContextKey struct{}
type restoreStoreContextKey struct{}

// WithExactOriginalRetention returns a functional CompressionOption configuring exact-original retention.
func WithExactOriginalRetention(enabled bool) CompressionOption {
	return func(o *CompressionOptions) {
		o.ExactOriginal = enabled
	}
}

// WithSessionID returns a functional CompressionOption setting the owning session/trace ID.
func WithSessionID(sessionID string) CompressionOption {
	return func(o *CompressionOptions) {
		o.SessionID = sessionID
	}
}

// WithRestoreStore returns a functional CompressionOption setting a specific RestoreStore.
func WithRestoreStore(store *RestoreStore) CompressionOption {
	return func(o *CompressionOptions) {
		o.Store = store
	}
}

// WithExactOriginalContext returns a derived context carrying the exact-original retention flag.
func WithExactOriginalContext(ctx context.Context, enabled bool) context.Context {
	if ctx == nil {
		ctx = context.Background()
	}
	return context.WithValue(ctx, exactOriginalContextKey{}, enabled)
}

// ExactOriginalFromContext extracts the exact-original retention flag from ctx, if present.
func ExactOriginalFromContext(ctx context.Context) (bool, bool) {
	if ctx == nil {
		return false, false
	}
	v, ok := ctx.Value(exactOriginalContextKey{}).(bool)
	return v, ok
}

// WithSessionContext returns a derived context carrying the session or trace ID.
func WithSessionContext(ctx context.Context, sessionID string) context.Context {
	if ctx == nil {
		ctx = context.Background()
	}
	return context.WithValue(ctx, sessionIDContextKey{}, sessionID)
}

// SessionIDFromContext extracts the session or trace ID from ctx, if present.
func SessionIDFromContext(ctx context.Context) (string, bool) {
	if ctx == nil {
		return "", false
	}
	v, ok := ctx.Value(sessionIDContextKey{}).(string)
	return v, ok
}

// WithRestoreStoreContext returns a derived context carrying a RestoreStore instance.
func WithRestoreStoreContext(ctx context.Context, store *RestoreStore) context.Context {
	if ctx == nil {
		ctx = context.Background()
	}
	return context.WithValue(ctx, restoreStoreContextKey{}, store)
}

// RestoreStoreFromContext extracts a RestoreStore from ctx, if present.
func RestoreStoreFromContext(ctx context.Context) (*RestoreStore, bool) {
	if ctx == nil {
		return nil, false
	}
	v, ok := ctx.Value(restoreStoreContextKey{}).(*RestoreStore)
	return v, ok
}

// IsExactOriginalEnabled returns true if val represents an affirmative exact-original retention setting.
func IsExactOriginalEnabled(val string) bool {
	switch strings.ToLower(strings.TrimSpace(val)) {
	case "true", "1", "yes", "on", "enable", "enabled", "exact":
		return true
	default:
		return false
	}
}

// IsExactOriginalMetadataEnabled inspects a metadata map for exact-original retention request keys.
func IsExactOriginalMetadataEnabled(md map[string]string) bool {
	if len(md) == 0 {
		return false
	}
	for k, v := range md {
		switch strings.ToLower(strings.TrimSpace(k)) {
		case "exact_original", "exact-original", "exactoriginal",
			"exact_original_retention", "exact-original-retention",
			"restore_original", "restore-original":
			if IsExactOriginalEnabled(v) {
				return true
			}
		}
	}
	return false
}

// ExtractSessionOrTraceID retrieves session or trace identification from metadata tags.
func ExtractSessionOrTraceID(md map[string]string) string {
	if len(md) == 0 {
		return ""
	}
	for k, v := range md {
		switch strings.ToLower(strings.TrimSpace(k)) {
		case "session_id", "sessionid", "session-id",
			"trace_id", "traceid", "trace-id",
			"caller_token", "callertoken", "caller-token":
			s := strings.TrimSpace(v)
			if s != "" {
				return s
			}
		}
	}
	return ""
}

var (
	brokerStoresMu      sync.RWMutex
	brokerStores        = make(map[*Broker]*RestoreStore)
	sessionExactOrigMu  sync.RWMutex
	sessionExactOrig    = make(map[string]bool)
	brokerDefaultsMu    sync.RWMutex
	brokerExactDefaults = make(map[*Broker]bool)
)

// RestoreOriginal retrieves exact original uncompressed bytes for handle and sessionID
// via the broker's configured or default restore store.
func (b *Broker) RestoreOriginal(sessionID string, handle string) ([]byte, error) {
	return b.RestoreStore().RestoreOriginal(sessionID, handle)
}

// SetSessionExactOriginal sets the default exact-original retention preference for sessionID.
func (b *Broker) SetSessionExactOriginal(sessionID string, enabled bool) {
	if sessionID == "" {
		return
	}
	sessionExactOrigMu.Lock()
	defer sessionExactOrigMu.Unlock()
	sessionExactOrig[sessionID] = enabled
}

// GetSessionExactOriginal returns the default exact-original retention preference for sessionID.
func (b *Broker) GetSessionExactOriginal(sessionID string) bool {
	if sessionID == "" {
		return false
	}
	sessionExactOrigMu.RLock()
	defer sessionExactOrigMu.RUnlock()
	return sessionExactOrig[sessionID]
}

// RestoreStore returns the RestoreStore associated with b, or the package DefaultRestoreStore().
func (b *Broker) RestoreStore() *RestoreStore {
	if b == nil {
		return DefaultRestoreStore()
	}
	brokerStoresMu.RLock()
	s, ok := brokerStores[b]
	brokerStoresMu.RUnlock()
	if ok && s != nil {
		return s
	}
	return DefaultRestoreStore()
}

// SetRestoreStore binds a custom RestoreStore to b.
func (b *Broker) SetRestoreStore(store *RestoreStore) {
	if b == nil {
		return
	}
	brokerStoresMu.Lock()
	defer brokerStoresMu.Unlock()
	brokerStores[b] = store
}

// WithBrokerRestoreStore configures a Broker with a custom RestoreStore.
func WithBrokerRestoreStore(store *RestoreStore) BrokerOption {
	return func(b *Broker) {
		b.SetRestoreStore(store)
	}
}

// WithBrokerExactOriginalRetention sets the broker-level default exact-original retention preference.
func WithBrokerExactOriginalRetention(enabled bool) BrokerOption {
	return func(b *Broker) {
		brokerDefaultsMu.Lock()
		defer brokerDefaultsMu.Unlock()
		brokerExactDefaults[b] = enabled
	}
}

// cleanupBrokerRestore cleans up broker registrations when a broker is closed.
func cleanupBrokerRestore(b *Broker) {
	if b == nil {
		return
	}
	brokerStoresMu.Lock()
	delete(brokerStores, b)
	brokerStoresMu.Unlock()

	brokerDefaultsMu.Lock()
	delete(brokerExactDefaults, b)
	brokerDefaultsMu.Unlock()
}
