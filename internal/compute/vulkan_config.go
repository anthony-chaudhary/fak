package compute

import (
	"container/list"
	"errors"
	"fmt"
	"sync"
)

type vulkanQ4KConfigurer interface {
	configureVulkanQ4K(profile, stage bool)
}

// ConfigureVulkanQ4K applies explicit diagnostic/staging settings to a selected Vulkan
// backend. It returns false for nil or non-Vulkan backends so operator-facing callers can
// fail loudly instead of claiming an inert configuration.
func ConfigureVulkanQ4K(backend Backend, profile, stage bool) bool {
	cfg, ok := backend.(vulkanQ4KConfigurer)
	if !ok {
		return false
	}
	cfg.configureVulkanQ4K(profile, stage)
	return true
}

// Default constants for Strix Halo (gfx1151) speculative draft execution.
const (
	// DefaultSpecDraftUbatchSize is the default decoupled micro-batch size for speculative draft generation
	// on AMD Strix Halo APUs (gfx1151), isolating draft batching from dynamic primary prompt chunks.
	DefaultSpecDraftUbatchSize = 512

	// DefaultGraphCacheCapacity is the default maximum number of cached power-of-two execution graphs.
	DefaultGraphCacheCapacity = 64

	// StrixHaloTargetArch identifies the AMD Strix Halo RDNA 3.5 APU architecture.
	StrixHaloTargetArch = "gfx1151"
)

// QuantizeDraftTokenLength quantizes variable draft token lengths (1..64) into
// power-of-two buckets (1, 2, 4, 8, 16, 32, 64) with upper clamp at 64 and lower
// clamp at 1 for non-positive inputs.
//
// Dynamic graph recording for variable-length draft acceptance causes Vulkan command
// buffer recording timeouts and GPU ring resets on Strix Halo APUs. Quantizing into
// discrete power-of-two buckets guarantees graph reuse across iterations.
func QuantizeDraftTokenLength(n int) int {
	if n <= 1 {
		return 1
	}
	if n > 64 {
		return 64
	}
	p := 1
	for p < n {
		p <<= 1
	}
	return p
}

// GraphCacheKey represents a thread-safe lookup key for cached Vulkan compute command graphs/pipelines,
// identifying quantized batch and sequence dimensions.
type GraphCacheKey struct {
	BatchSize int    `json:"batch_size"`
	SeqLen    int    `json:"seq_len"`
	Tag       string `json:"tag,omitempty"`
}

// String returns a deterministic string representation of the cache key.
func (k GraphCacheKey) String() string {
	if k.Tag != "" {
		return fmt.Sprintf("b%d:s%d:%s", k.BatchSize, k.SeqLen, k.Tag)
	}
	return fmt.Sprintf("b%d:s%d", k.BatchSize, k.SeqLen)
}

// IsPowerOfTwo reports whether the key's sequence length is a power of two in [1, 64].
func (k GraphCacheKey) IsPowerOfTwo() bool {
	return k.SeqLen >= 1 && k.SeqLen <= 64 && (k.SeqLen&(k.SeqLen-1)) == 0
}

// NewGraphCacheKey constructs a GraphCacheKey with the given batch size and sequence length.
func NewGraphCacheKey(batchSize, seqLen int) GraphCacheKey {
	return GraphCacheKey{
		BatchSize: batchSize,
		SeqLen:    seqLen,
	}
}

// NewQuantizedGraphCacheKey constructs a GraphCacheKey where draft token length is quantized
// to the nearest power-of-two bucket in [1, 64].
func NewQuantizedGraphCacheKey(batchSize, draftTokens int) GraphCacheKey {
	return GraphCacheKey{
		BatchSize: batchSize,
		SeqLen:    QuantizeDraftTokenLength(draftTokens),
	}
}

// PowerOfTwoGraphCacheStats records hit/miss and capacity metrics for PowerOfTwoGraphCache.
type PowerOfTwoGraphCacheStats struct {
	Hits      int64 `json:"hits"`
	Misses    int64 `json:"misses"`
	Evictions int64 `json:"evictions"`
	Entries   int   `json:"entries"`
	Capacity  int   `json:"capacity"`
}

type graphCacheEntry struct {
	key GraphCacheKey
	val any
}

// PowerOfTwoGraphCache is a thread-safe LRU cache mapping quantized batch and sequence dimensions
// to reusable Vulkan compute command graphs/pipelines.
//
// By reusing pre-recorded graphs mapped to power-of-two token lengths (1, 2, 4, 8, 16, 32, 64),
// this avoids dynamic graph recording overhead, driver command buffer stalls, and GPU ring
// resets on AMD Strix Halo unified-memory APUs during variable speculative draft acceptance.
type PowerOfTwoGraphCache struct {
	mu       sync.RWMutex
	capacity int
	items    map[GraphCacheKey]*list.Element
	lru      *list.List
	stats    PowerOfTwoGraphCacheStats
	onEvict  func(key GraphCacheKey, val any)
	closed   bool
}

// NewPowerOfTwoGraphCache instantiates an empty power-of-two graph cache with the specified capacity limit.
func NewPowerOfTwoGraphCache(capacity int) *PowerOfTwoGraphCache {
	if capacity <= 0 {
		capacity = DefaultGraphCacheCapacity
	}
	return &PowerOfTwoGraphCache{
		capacity: capacity,
		items:    make(map[GraphCacheKey]*list.Element),
		lru:      list.New(),
		stats: PowerOfTwoGraphCacheStats{
			Capacity: capacity,
		},
	}
}

// normalizeKey ensures the sequence dimension of the key is bucketed into a power-of-two length.
func (c *PowerOfTwoGraphCache) normalizeKey(key GraphCacheKey) GraphCacheKey {
	key.SeqLen = QuantizeDraftTokenLength(key.SeqLen)
	return key
}

// Get retrieves a cached graph or pipeline by key. If the key's sequence length is not already
// quantized, it is automatically normalized to the corresponding power-of-two bucket.
func (c *PowerOfTwoGraphCache) Get(key GraphCacheKey) (any, bool) {
	c.mu.Lock()
	defer c.mu.Unlock()

	if c.closed {
		return nil, false
	}

	normKey := c.normalizeKey(key)
	elem, ok := c.items[normKey]
	if !ok {
		c.stats.Misses++
		return nil, false
	}

	c.lru.MoveToFront(elem)
	c.stats.Hits++
	return elem.Value.(*graphCacheEntry).val, true
}

// GetQuantized retrieves a cached graph using explicit batch size and unquantized draft token count.
func (c *PowerOfTwoGraphCache) GetQuantized(batchSize, draftTokens int) (any, bool) {
	return c.Get(NewQuantizedGraphCacheKey(batchSize, draftTokens))
}

// Put stores a graph or pipeline in the cache. The key's sequence length is normalized to
// its power-of-two bucket. If capacity is exceeded, the least recently used entry is evicted.
func (c *PowerOfTwoGraphCache) Put(key GraphCacheKey, val any) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	if c.closed {
		return errors.New("power-of-two graph cache is closed")
	}
	if val == nil {
		return errors.New("cannot cache nil graph")
	}

	normKey := c.normalizeKey(key)
	if elem, ok := c.items[normKey]; ok {
		c.lru.MoveToFront(elem)
		entry := elem.Value.(*graphCacheEntry)
		entry.val = val
		return nil
	}

	if c.capacity > 0 && c.lru.Len() >= c.capacity {
		c.evictOldestLocked()
	}

	entry := &graphCacheEntry{key: normKey, val: val}
	elem := c.lru.PushFront(entry)
	c.items[normKey] = elem
	c.stats.Entries = c.lru.Len()
	return nil
}

// PutQuantized stores a graph using explicit batch size and draft token count.
func (c *PowerOfTwoGraphCache) PutQuantized(batchSize, draftTokens int, val any) error {
	return c.Put(NewQuantizedGraphCacheKey(batchSize, draftTokens), val)
}

// GetOrCreate returns the cached graph for key or creates and caches it under lock using createFn.
func (c *PowerOfTwoGraphCache) GetOrCreate(key GraphCacheKey, createFn func() (any, error)) (any, bool, error) {
	c.mu.Lock()
	defer c.mu.Unlock()

	if c.closed {
		return nil, false, errors.New("power-of-two graph cache is closed")
	}

	normKey := c.normalizeKey(key)
	if elem, ok := c.items[normKey]; ok {
		c.lru.MoveToFront(elem)
		c.stats.Hits++
		return elem.Value.(*graphCacheEntry).val, true, nil
	}

	c.stats.Misses++
	val, err := createFn()
	if err != nil {
		return nil, false, err
	}
	if val == nil {
		return nil, false, errors.New("createFn returned nil graph")
	}

	if c.capacity > 0 && c.lru.Len() >= c.capacity {
		c.evictOldestLocked()
	}

	entry := &graphCacheEntry{key: normKey, val: val}
	elem := c.lru.PushFront(entry)
	c.items[normKey] = elem
	c.stats.Entries = c.lru.Len()
	return val, false, nil
}

func (c *PowerOfTwoGraphCache) evictOldestLocked() {
	elem := c.lru.Back()
	if elem == nil {
		return
	}
	c.lru.Remove(elem)
	entry := elem.Value.(*graphCacheEntry)
	delete(c.items, entry.key)
	c.stats.Evictions++
	c.stats.Entries = c.lru.Len()
	if c.onEvict != nil {
		c.onEvict(entry.key, entry.val)
	}
}

// Delete removes an entry matching key from the cache.
func (c *PowerOfTwoGraphCache) Delete(key GraphCacheKey) bool {
	c.mu.Lock()
	defer c.mu.Unlock()

	if c.closed {
		return false
	}
	normKey := c.normalizeKey(key)
	elem, ok := c.items[normKey]
	if !ok {
		return false
	}
	c.lru.Remove(elem)
	entry := elem.Value.(*graphCacheEntry)
	delete(c.items, normKey)
	c.stats.Entries = c.lru.Len()
	if c.onEvict != nil {
		c.onEvict(entry.key, entry.val)
	}
	return true
}

// SetOnEvict registers an eviction callback invoked when an entry is evicted.
func (c *PowerOfTwoGraphCache) SetOnEvict(fn func(key GraphCacheKey, val any)) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.onEvict = fn
}

// Clear removes all entries from the cache.
func (c *PowerOfTwoGraphCache) Clear() {
	c.mu.Lock()
	defer c.mu.Unlock()

	if c.onEvict != nil {
		for _, elem := range c.items {
			entry := elem.Value.(*graphCacheEntry)
			c.onEvict(entry.key, entry.val)
		}
	}
	c.items = make(map[GraphCacheKey]*list.Element)
	c.lru.Init()
	c.stats.Entries = 0
}

// Close closes the cache and clears all entries.
func (c *PowerOfTwoGraphCache) Close() {
	c.mu.Lock()
	defer c.mu.Unlock()

	if c.closed {
		return
	}
	if c.onEvict != nil {
		for _, elem := range c.items {
			entry := elem.Value.(*graphCacheEntry)
			c.onEvict(entry.key, entry.val)
		}
	}
	c.items = make(map[GraphCacheKey]*list.Element)
	c.lru.Init()
	c.stats.Entries = 0
	c.closed = true
}

// Len returns the current number of cached items.
func (c *PowerOfTwoGraphCache) Len() int {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return len(c.items)
}

// Capacity returns the maximum capacity of the cache.
func (c *PowerOfTwoGraphCache) Capacity() int {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.capacity
}

// Stats returns a snapshot copy of cache hit/miss/eviction metrics.
func (c *PowerOfTwoGraphCache) Stats() PowerOfTwoGraphCacheStats {
	c.mu.RLock()
	defer c.mu.RUnlock()
	st := c.stats
	st.Entries = len(c.items)
	st.Capacity = c.capacity
	return st
}

// GetCommandGraph retrieves a cached VulkanCommandGraph if present and type-asserts it.
func (c *PowerOfTwoGraphCache) GetCommandGraph(key GraphCacheKey) (*VulkanCommandGraph, bool) {
	val, ok := c.Get(key)
	if !ok {
		return nil, false
	}
	g, ok := val.(*VulkanCommandGraph)
	return g, ok
}

// PutCommandGraph stores a VulkanCommandGraph into the cache.
func (c *PowerOfTwoGraphCache) PutCommandGraph(key GraphCacheKey, g *VulkanCommandGraph) error {
	return c.Put(key, g)
}

// ---- Strix Halo MTP Configuration & Runtime Interface ------------------------

// StrixHaloMTPConfig holds configuration for decoupled speculative draft micro-batching
// and power-of-two graph caching on AMD Strix Halo (gfx1151).
type StrixHaloMTPConfig struct {
	// SpecDraftUbatchSize is the configurable micro-batch size for speculative draft generation (default: 512).
	SpecDraftUbatchSize int `json:"spec_draft_ubatch_size"`

	// GraphCacheCapacity specifies the maximum entries in the power-of-two graph cache (default: 64).
	GraphCacheCapacity int `json:"graph_cache_capacity"`

	// TargetArch optionally identifies the target GPU architecture (default: "gfx1151").
	TargetArch string `json:"target_arch,omitempty"`

	// EnablePowerOfTwoBucketing controls whether draft token lengths are quantized into power-of-two buckets.
	EnablePowerOfTwoBucketing bool `json:"enable_power_of_two_bucketing"`
}

// DefaultStrixHaloMTPConfig returns standard production defaults for Strix Halo MTP.
func DefaultStrixHaloMTPConfig() StrixHaloMTPConfig {
	return StrixHaloMTPConfig{
		SpecDraftUbatchSize:       DefaultSpecDraftUbatchSize,
		GraphCacheCapacity:        DefaultGraphCacheCapacity,
		TargetArch:                StrixHaloTargetArch,
		EnablePowerOfTwoBucketing: true,
	}
}

// Validate verifies that configuration parameters are valid.
func (c StrixHaloMTPConfig) Validate() error {
	if c.SpecDraftUbatchSize <= 0 {
		return fmt.Errorf("strix halo mtp: SpecDraftUbatchSize must be positive, got %d", c.SpecDraftUbatchSize)
	}
	if c.GraphCacheCapacity <= 0 {
		return fmt.Errorf("strix halo mtp: GraphCacheCapacity must be positive, got %d", c.GraphCacheCapacity)
	}
	return nil
}

// StrixHaloMTPOption defines a functional option for configuring StrixHaloMTPConfig.
type StrixHaloMTPOption func(*StrixHaloMTPConfig)

// WithSpecDraftUbatchSize configures the speculative draft micro-batch size.
func WithSpecDraftUbatchSize(size int) StrixHaloMTPOption {
	return func(c *StrixHaloMTPConfig) {
		c.SpecDraftUbatchSize = size
	}
}

// WithGraphCacheCapacity configures the maximum capacity of the power-of-two graph cache.
func WithGraphCacheCapacity(capacity int) StrixHaloMTPOption {
	return func(c *StrixHaloMTPConfig) {
		c.GraphCacheCapacity = capacity
	}
}

// WithTargetArch configures the target GPU architecture string.
func WithTargetArch(arch string) StrixHaloMTPOption {
	return func(c *StrixHaloMTPConfig) {
		c.TargetArch = arch
	}
}

// WithPowerOfTwoBucketing enables or disables power-of-two draft token length quantization.
func WithPowerOfTwoBucketing(enable bool) StrixHaloMTPOption {
	return func(c *StrixHaloMTPConfig) {
		c.EnablePowerOfTwoBucketing = enable
	}
}

// NewStrixHaloMTPConfig creates a StrixHaloMTPConfig starting from defaults and applying options.
func NewStrixHaloMTPConfig(opts ...StrixHaloMTPOption) StrixHaloMTPConfig {
	cfg := DefaultStrixHaloMTPConfig()
	for _, opt := range opts {
		if opt != nil {
			opt(&cfg)
		}
	}
	if cfg.SpecDraftUbatchSize <= 0 {
		cfg.SpecDraftUbatchSize = DefaultSpecDraftUbatchSize
	}
	if cfg.GraphCacheCapacity <= 0 {
		cfg.GraphCacheCapacity = DefaultGraphCacheCapacity
	}
	return cfg
}

// StrixHaloMTPConfigBuilder provides a fluent builder for StrixHaloMTPConfig.
type StrixHaloMTPConfigBuilder struct {
	cfg StrixHaloMTPConfig
}

// NewStrixHaloMTPConfigBuilder constructs a builder initialized with default configuration.
func NewStrixHaloMTPConfigBuilder() *StrixHaloMTPConfigBuilder {
	return &StrixHaloMTPConfigBuilder{
		cfg: DefaultStrixHaloMTPConfig(),
	}
}

// WithSpecDraftUbatchSize sets the speculative draft micro-batch size.
func (b *StrixHaloMTPConfigBuilder) WithSpecDraftUbatchSize(size int) *StrixHaloMTPConfigBuilder {
	b.cfg.SpecDraftUbatchSize = size
	return b
}

// WithGraphCacheCapacity sets the graph cache capacity.
func (b *StrixHaloMTPConfigBuilder) WithGraphCacheCapacity(capacity int) *StrixHaloMTPConfigBuilder {
	b.cfg.GraphCacheCapacity = capacity
	return b
}

// WithTargetArch sets the target architecture.
func (b *StrixHaloMTPConfigBuilder) WithTargetArch(arch string) *StrixHaloMTPConfigBuilder {
	b.cfg.TargetArch = arch
	return b
}

// WithPowerOfTwoBucketing sets power-of-two bucketing flag.
func (b *StrixHaloMTPConfigBuilder) WithPowerOfTwoBucketing(enable bool) *StrixHaloMTPConfigBuilder {
	b.cfg.EnablePowerOfTwoBucketing = enable
	return b
}

// Build finalizes and validates the StrixHaloMTPConfig.
func (b *StrixHaloMTPConfigBuilder) Build() StrixHaloMTPConfig {
	cfg := b.cfg
	if cfg.SpecDraftUbatchSize <= 0 {
		cfg.SpecDraftUbatchSize = DefaultSpecDraftUbatchSize
	}
	if cfg.GraphCacheCapacity <= 0 {
		cfg.GraphCacheCapacity = DefaultGraphCacheCapacity
	}
	return cfg
}

// BuildRuntime constructs a StrixHaloMTPRuntime using the built configuration.
func (b *StrixHaloMTPConfigBuilder) BuildRuntime() *StrixHaloMTPRuntime {
	return NewStrixHaloMTPRuntime(b.Build())
}

// StrixHaloMTPConfigurer is the configuration interface for compute backends or drivers
// that support Strix Halo MTP decoupled speculative draft micro-batching and power-of-two graph caching.
type StrixHaloMTPConfigurer interface {
	ConfigureStrixHaloMTP(cfg StrixHaloMTPConfig) error
}

type unexportedStrixHaloMTPConfigurer interface {
	configureStrixHaloMTP(cfg StrixHaloMTPConfig) error
}

// ConfigureStrixHaloMTP applies Strix Halo MTP settings to a selected backend if supported.
// Returns false if the backend is nil or does not implement StrixHaloMTPConfigurer.
func ConfigureStrixHaloMTP(backend Backend, cfg StrixHaloMTPConfig) bool {
	if backend == nil {
		return false
	}
	if c, ok := backend.(StrixHaloMTPConfigurer); ok {
		return c.ConfigureStrixHaloMTP(cfg) == nil
	}
	if c, ok := backend.(unexportedStrixHaloMTPConfigurer); ok {
		return c.configureStrixHaloMTP(cfg) == nil
	}
	return false
}

// StrixHaloMTPRuntime coordinates decoupled speculative draft micro-batching and
// power-of-two graph caching for Strix Halo MTP execution.
type StrixHaloMTPRuntime struct {
	cfg   StrixHaloMTPConfig
	cache *PowerOfTwoGraphCache
}

// NewStrixHaloMTPRuntime creates a new runtime using the provided configuration.
func NewStrixHaloMTPRuntime(cfg StrixHaloMTPConfig) *StrixHaloMTPRuntime {
	if cfg.SpecDraftUbatchSize <= 0 {
		cfg.SpecDraftUbatchSize = DefaultSpecDraftUbatchSize
	}
	if cfg.GraphCacheCapacity <= 0 {
		cfg.GraphCacheCapacity = DefaultGraphCacheCapacity
	}
	return &StrixHaloMTPRuntime{
		cfg:   cfg,
		cache: NewPowerOfTwoGraphCache(cfg.GraphCacheCapacity),
	}
}

// Config returns the active StrixHaloMTPConfig.
func (r *StrixHaloMTPRuntime) Config() StrixHaloMTPConfig {
	return r.cfg
}

// SpecDraftUbatchSize returns the configured speculative draft micro-batch size.
func (r *StrixHaloMTPRuntime) SpecDraftUbatchSize() int {
	return r.cfg.SpecDraftUbatchSize
}

// GraphCache returns the power-of-two graph cache.
func (r *StrixHaloMTPRuntime) GraphCache() *PowerOfTwoGraphCache {
	return r.cache
}

// QuantizeDraftLength quantizes draft token length using the runtime's configuration.
func (r *StrixHaloMTPRuntime) QuantizeDraftLength(n int) int {
	if r.cfg.EnablePowerOfTwoBucketing {
		return QuantizeDraftTokenLength(n)
	}
	return n
}

// GetOrRecordGraph retrieves a cached graph matching the quantized draft token length,
// or invokes recordFn to record and cache a new command graph/pipeline.
func (r *StrixHaloMTPRuntime) GetOrRecordGraph(draftTokens int, recordFn func(quantizedTokens int) (any, error)) (any, bool, error) {
	quantized := r.QuantizeDraftLength(draftTokens)
	key := GraphCacheKey{
		BatchSize: r.cfg.SpecDraftUbatchSize,
		SeqLen:    quantized,
		Tag:       r.cfg.TargetArch,
	}
	return r.cache.GetOrCreate(key, func() (any, error) {
		return recordFn(quantized)
	})
}
