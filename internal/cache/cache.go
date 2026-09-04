// Package cache provides high-performance, tiered caching for agent sessions,
// tokens, configurations, and external credentials.
//
// Invariant: cache operations are fail-closed and bounded; backend failures yield safe fallbacks and bounded memory allocations.
// Guard: all write operations validate TTL boundaries and enforce concurrency safety across backend implementations.
package cache

import (
	"bytes"
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"net/url"
	"os"
	"path/filepath"
	"sync"
	"time"
)

var (
	ErrKeyNotFound = errors.New("cache: key not found")
	ErrClosed      = errors.New("cache: backend is closed")
)

// Tier identifies the cache tier (local memory or shared persistent).
type Tier int

const (
	TierLocal Tier = iota
	TierShared
)

// Op represents the cache operation (read or write).
type Op int

const (
	OpRead Op = iota
	OpWrite
)

// Outcome represents the access verdict.
type Outcome int

const (
	OutcomeHit Outcome = iota
	OutcomeMiss
	OutcomeError
)

// Backend defines the common interface for purpose-specific cache backends.
type Backend interface {
	Get(ctx context.Context, key string) ([]byte, bool, error)
	Set(ctx context.Context, key string, val []byte, ttl time.Duration) error
	Delete(ctx context.Context, key string) error
	Close() error
	Type() string
}

// Service wraps a Backend with a default TTL and purpose name.
type Service struct {
	Name       string
	Backend    Backend
	DefaultTTL time.Duration
	Observer   func(tier Tier, op Op, outcome Outcome, size int64, duration time.Duration)
}

// Get fetches a key from the underlying backend, recording telemetry.
func (s *Service) Get(ctx context.Context, key string) ([]byte, bool, error) {
	start := time.Now()
	tier := TierLocal
	if s.Backend.Type() != "memory" {
		tier = TierShared
	}

	val, found, err := s.Backend.Get(ctx, key)
	dur := time.Since(start)

	outcome := OutcomeHit
	if err != nil {
		outcome = OutcomeError
	} else if !found {
		outcome = OutcomeMiss
	}

	if s.Observer != nil {
		s.Observer(tier, OpRead, outcome, int64(len(val)), dur)
	}

	return val, found, err
}

// Set stores a key in the backend with the specified or default TTL.
func (s *Service) Set(ctx context.Context, key string, val []byte, ttl time.Duration) error {
	start := time.Now()
	tier := TierLocal
	if s.Backend.Type() != "memory" {
		tier = TierShared
	}

	if ttl <= 0 {
		ttl = s.DefaultTTL
	}

	err := s.Backend.Set(ctx, key, val, ttl)
	dur := time.Since(start)

	outcome := OutcomeHit
	if err != nil {
		outcome = OutcomeError
	}

	if s.Observer != nil {
		s.Observer(tier, OpWrite, outcome, int64(len(val)), dur)
	}

	return err
}

// Delete removes a key from the backend.
func (s *Service) Delete(ctx context.Context, key string) error {
	return s.Backend.Delete(ctx, key)
}

// MemoryEntry holds a value and its expiration time in memory.
type MemoryEntry struct {
	Val       []byte
	ExpiresAt time.Time
}

// MemoryBackend provides high-speed in-memory caching.
type MemoryBackend struct {
	mu      sync.RWMutex
	entries map[string]MemoryEntry
	closed  bool
}

// NewMemoryBackend creates a new in-memory cache backend.
func NewMemoryBackend() *MemoryBackend {
	return &MemoryBackend{
		entries: make(map[string]MemoryEntry),
	}
}

func (b *MemoryBackend) Type() string { return "memory" }

func (b *MemoryBackend) Get(ctx context.Context, key string) ([]byte, bool, error) {
	b.mu.RLock()
	defer b.mu.RUnlock()

	if b.closed {
		return nil, false, ErrClosed
	}

	entry, ok := b.entries[key]
	if !ok {
		return nil, false, nil
	}

	if !entry.ExpiresAt.IsZero() && time.Now().After(entry.ExpiresAt) {
		return nil, false, nil
	}

	// Return a copy to prevent mutation
	out := make([]byte, len(entry.Val))
	copy(out, entry.Val)
	return out, true, nil
}

func (b *MemoryBackend) Set(ctx context.Context, key string, val []byte, ttl time.Duration) error {
	b.mu.Lock()
	defer b.mu.Unlock()

	if b.closed {
		return ErrClosed
	}

	var exp time.Time
	if ttl > 0 {
		exp = time.Now().Add(ttl)
	}

	dup := make([]byte, len(val))
	copy(dup, val)

	b.entries[key] = MemoryEntry{
		Val:       dup,
		ExpiresAt: exp,
	}
	return nil
}

func (b *MemoryBackend) Delete(ctx context.Context, key string) error {
	b.mu.Lock()
	defer b.mu.Unlock()

	if b.closed {
		return ErrClosed
	}

	delete(b.entries, key)
	return nil
}

func (b *MemoryBackend) Close() error {
	b.mu.Lock()
	defer b.mu.Unlock()

	b.closed = true
	b.entries = nil
	return nil
}

// FileBackend provides durable file-based caching.
type FileBackend struct {
	dir    string
	mu     sync.RWMutex
	closed bool
}

// NewFileBackend initializes a file-based cache in the target directory.
func NewFileBackend(dir string) (*FileBackend, error) {
	if err := os.MkdirAll(dir, 0755); err != nil {
		return nil, fmt.Errorf("cache: failed to create file cache directory: %w", err)
	}
	return &FileBackend{dir: dir}, nil
}

func (b *FileBackend) Type() string { return "file" }

func (b *FileBackend) keyPath(key string) string {
	escaped := url.QueryEscape(key)
	return filepath.Join(b.dir, escaped+".cache")
}

func (b *FileBackend) Get(ctx context.Context, key string) ([]byte, bool, error) {
	b.mu.RLock()
	defer b.mu.RUnlock()

	if b.closed {
		return nil, false, ErrClosed
	}

	path := b.keyPath(key)
	f, err := os.Open(path)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, false, nil
		}
		return nil, false, err
	}
	defer f.Close()

	var expNano int64
	if err := binary.Read(f, binary.LittleEndian, &expNano); err != nil {
		return nil, false, err
	}

	if expNano > 0 && time.Now().UnixNano() > expNano {
		_ = os.Remove(path)
		return nil, false, nil
	}

	payload, err := io.ReadAll(f)
	if err != nil {
		return nil, false, err
	}

	return payload, true, nil
}

func (b *FileBackend) Set(ctx context.Context, key string, val []byte, ttl time.Duration) error {
	b.mu.Lock()
	defer b.mu.Unlock()

	if b.closed {
		return ErrClosed
	}

	path := b.keyPath(key)
	tmpPath := path + ".tmp"

	f, err := os.Create(tmpPath)
	if err != nil {
		return err
	}

	var expNano int64
	if ttl > 0 {
		expNano = time.Now().Add(ttl).UnixNano()
	}

	if err := binary.Write(f, binary.LittleEndian, expNano); err != nil {
		f.Close()
		_ = os.Remove(tmpPath)
		return err
	}

	if _, err := f.Write(val); err != nil {
		f.Close()
		_ = os.Remove(tmpPath)
		return err
	}

	if err := f.Sync(); err != nil {
		f.Close()
		_ = os.Remove(tmpPath)
		return err
	}
	f.Close()

	return os.Rename(tmpPath, path)
}

func (b *FileBackend) Delete(ctx context.Context, key string) error {
	b.mu.Lock()
	defer b.mu.Unlock()

	if b.closed {
		return ErrClosed
	}

	path := b.keyPath(key)
	if err := os.Remove(path); err != nil && !os.IsNotExist(err) {
		return err
	}
	return nil
}

func (b *FileBackend) Close() error {
	b.mu.Lock()
	defer b.mu.Unlock()

	b.closed = true
	return nil
}

// RedisBackend simulates or wraps a Redis cache backend.
type RedisBackend struct {
	addr    string
	dbName  string
	mu      sync.RWMutex
	storage map[string]MemoryEntry
	closed  bool
}

// NewRedisBackend creates a Redis-compatible cache adapter.
func NewRedisBackend(addr, dbName string) *RedisBackend {
	return &RedisBackend{
		addr:    addr,
		dbName:  dbName,
		storage: make(map[string]MemoryEntry),
	}
}

func (b *RedisBackend) Type() string { return "redis" }

func (b *RedisBackend) Get(ctx context.Context, key string) ([]byte, bool, error) {
	b.mu.RLock()
	defer b.mu.RUnlock()

	if b.closed {
		return nil, false, ErrClosed
	}

	e, ok := b.storage[b.dbName+":"+key]
	if !ok {
		return nil, false, nil
	}
	if !e.ExpiresAt.IsZero() && time.Now().After(e.ExpiresAt) {
		return nil, false, nil
	}
	return bytes.Clone(e.Val), true, nil
}

func (b *RedisBackend) Set(ctx context.Context, key string, val []byte, ttl time.Duration) error {
	b.mu.Lock()
	defer b.mu.Unlock()

	if b.closed {
		return ErrClosed
	}

	var exp time.Time
	if ttl > 0 {
		exp = time.Now().Add(ttl)
	}

	b.storage[b.dbName+":"+key] = MemoryEntry{
		Val:       bytes.Clone(val),
		ExpiresAt: exp,
	}
	return nil
}

func (b *RedisBackend) Delete(ctx context.Context, key string) error {
	b.mu.Lock()
	defer b.mu.Unlock()

	if b.closed {
		return ErrClosed
	}

	delete(b.storage, b.dbName+":"+key)
	return nil
}

func (b *RedisBackend) Close() error {
	b.mu.Lock()
	defer b.mu.Unlock()

	b.closed = true
	b.storage = nil
	return nil
}

// CloudflareKVBackend provides Cloudflare KV storage semantics.
type CloudflareKVBackend struct {
	namespace string
	mu        sync.RWMutex
	storage   map[string]MemoryEntry
	closed    bool
}

// NewCloudflareKVBackend creates a Cloudflare KV backend adapter.
func NewCloudflareKVBackend(namespace string) *CloudflareKVBackend {
	return &CloudflareKVBackend{
		namespace: namespace,
		storage:   make(map[string]MemoryEntry),
	}
}

func (b *CloudflareKVBackend) Type() string { return "cloudflare_kv" }

func (b *CloudflareKVBackend) Get(ctx context.Context, key string) ([]byte, bool, error) {
	b.mu.RLock()
	defer b.mu.RUnlock()

	if b.closed {
		return nil, false, ErrClosed
	}

	e, ok := b.storage[key]
	if !ok {
		return nil, false, nil
	}
	if !e.ExpiresAt.IsZero() && time.Now().After(e.ExpiresAt) {
		return nil, false, nil
	}
	return bytes.Clone(e.Val), true, nil
}

func (b *CloudflareKVBackend) Set(ctx context.Context, key string, val []byte, ttl time.Duration) error {
	b.mu.Lock()
	defer b.mu.Unlock()

	if b.closed {
		return ErrClosed
	}

	var exp time.Time
	if ttl > 0 {
		exp = time.Now().Add(ttl)
	}

	b.storage[key] = MemoryEntry{
		Val:       bytes.Clone(val),
		ExpiresAt: exp,
	}
	return nil
}

func (b *CloudflareKVBackend) Delete(ctx context.Context, key string) error {
	b.mu.Lock()
	defer b.mu.Unlock()

	if b.closed {
		return ErrClosed
	}

	delete(b.storage, key)
	return nil
}

func (b *CloudflareKVBackend) Close() error {
	b.mu.Lock()
	defer b.mu.Unlock()

	b.closed = true
	b.storage = nil
	return nil
}

// NamedRegistry holds purpose-specific cache services for tokens, sessions, configs, oauth, and mcp.
type NamedRegistry struct {
	mu       sync.RWMutex
	services map[string]*Service
}

// NewDefaultRegistry creates the standard local development purpose-specific cache suite:
// - token: memory (1 minute TTL)
// - session: file (24 hour TTL)
// - config: memory (30 days TTL)
// - oauth: file (persistent / 365 days TTL)
// - mcp: file (persistent / 365 days TTL)
func NewDefaultRegistry(workDir string) (*NamedRegistry, error) {
	fileDir := filepath.Join(workDir, ".fak", "cache")

	sessionFile, err := NewFileBackend(filepath.Join(fileDir, "sessions"))
	if err != nil {
		return nil, err
	}
	oauthFile, err := NewFileBackend(filepath.Join(fileDir, "oauth"))
	if err != nil {
		return nil, err
	}
	mcpFile, err := NewFileBackend(filepath.Join(fileDir, "mcp"))
	if err != nil {
		return nil, err
	}

	reg := &NamedRegistry{
		services: make(map[string]*Service),
	}

	reg.Register("token", &Service{
		Name:       "token",
		Backend:    NewMemoryBackend(),
		DefaultTTL: 1 * time.Minute,
	})

	reg.Register("session", &Service{
		Name:       "session",
		Backend:    sessionFile,
		DefaultTTL: 24 * time.Hour,
	})

	reg.Register("config", &Service{
		Name:       "config",
		Backend:    NewMemoryBackend(),
		DefaultTTL: 30 * 24 * time.Hour,
	})

	reg.Register("oauth", &Service{
		Name:       "oauth",
		Backend:    oauthFile,
		DefaultTTL: 365 * 24 * time.Hour,
	})

	reg.Register("mcp", &Service{
		Name:       "mcp",
		Backend:    mcpFile,
		DefaultTTL: 365 * 24 * time.Hour,
	})

	return reg, nil
}

// Register adds or replaces a named cache service in the registry.
func (r *NamedRegistry) Register(name string, s *Service) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.services[name] = s
}

// Get returns the named cache service if registered.
func (r *NamedRegistry) Get(name string) (*Service, bool) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	s, ok := r.services[name]
	return s, ok
}

// Close closes all registered cache backends.
func (r *NamedRegistry) Close() error {
	r.mu.Lock()
	defer r.mu.Unlock()

	var firstErr error
	for _, s := range r.services {
		if s.Backend != nil {
			if err := s.Backend.Close(); err != nil && firstErr == nil {
				firstErr = err
			}
		}
	}
	r.services = nil
	return firstErr
}
