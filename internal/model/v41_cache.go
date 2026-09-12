package model

// v41_cache.go — single-node bounded caching + streaming telemetry for native DeepSeek
// V4.1 Flash execution (issue #12904, leaf of parent #12640). When the main weights exceed
// physical RAM, routed experts and Engram embedding rows are demand-streamed from slower
// storage. This file makes that streaming cost MEASURABLE rather than assumed: a bounded
// byte-budget LRU expert cache with deterministic eviction, and an Engram row cache with
// eager bounded prefetch that reports page amplification (bytes read / useful bytes).
// There is no distributed placement, no quantization change, and no performance claim here;
// only the cache and its traffic counters.

import "fmt"

// V41ExpertCacheOptions configures a bounded, in-memory expert block cache.
type V41ExpertCacheOptions struct {
	BudgetBytes int64 // max resident payload bytes; <=0 means unlimited (still tracked)
	MaxEntries  int   // max entries; <=0 means unbounded entry count
}

// V41ExpertCache is a bounded LRU byte cache for routed-expert payloads, keyed by a
// caller-supplied (layer, expert) identity. Eviction is by least-recently-used with
// deterministic tie-break: an explicit monotonic clock makes recency unique by
// construction, and victims are taken from an ordered residency slice (never map
// iteration) so a replay is reproducible.
type V41ExpertCache struct {
	budgetBytes int64
	maxEntries  int

	clock    uint64
	order    []v41ExpertKey
	entries  map[v41ExpertKey][]byte
	cached   map[v41ExpertKey]uint64
	resident int64
	stats    V41ExpertCacheStats
}

type v41ExpertKey struct {
	layer  int
	expert int
}

// V41ExpertCacheStats is a deterministic snapshot of cache traffic.
type V41ExpertCacheStats struct {
	Hits            int64 `json:"hits"`
	Misses          int64 `json:"misses"`
	Evictions       int64 `json:"evictions"`
	ResidentBytes   int64 `json:"resident_bytes"`
	ResidentEntries int   `json:"resident_entries"`
	BytesRead       int64 `json:"bytes_read"`
}

// NewV41ExpertCache returns an empty bounded expert cache.
func NewV41ExpertCache(opts V41ExpertCacheOptions) *V41ExpertCache {
	return &V41ExpertCache{
		budgetBytes: opts.BudgetBytes,
		maxEntries:  opts.MaxEntries,
		entries:     make(map[v41ExpertKey][]byte),
		cached:      make(map[v41ExpertKey]uint64),
	}
}

// Get returns the cached payload for (layer, expert) and whether it was resident.
func (c *V41ExpertCache) Get(layer, expert int) ([]byte, bool) {
	key := v41ExpertKey{layer: layer, expert: expert}
	payload, ok := c.entries[key]
	if !ok {
		c.stats.Misses++
		return nil, false
	}
	c.stats.Hits++
	c.clock++
	c.cached[key] = c.clock
	return payload, true
}

// Put stores payload for (layer, expert), evicting other entries to honor the budget.
// The payload is copied (owned by the cache). Returns an error if the single payload
// exceeds the whole byte budget or entry budget — in that case it is NOT stored.
func (c *V41ExpertCache) Put(layer, expert int, payload []byte) error {
	key := v41ExpertKey{layer: layer, expert: expert}
	if c.budgetBytes > 0 && int64(len(payload)) > c.budgetBytes {
		return fmt.Errorf("model: V41 expert payload %d bytes exceeds cache budget %d", len(payload), c.budgetBytes)
	}
	if c.maxEntries > 0 && c.maxEntries < 1 {
		return fmt.Errorf("model: V41 expert cache max entries %d cannot hold one payload", c.maxEntries)
	}

	if old, ok := c.entries[key]; ok {
		c.resident -= int64(len(old))
		c.removeFromOrder(key)
	}

	for c.overBudget(int64(len(payload)), 1) {
		if !c.evictOne() {
			break
		}
	}

	stored := make([]byte, len(payload))
	copy(stored, payload)
	c.entries[key] = stored
	c.clock++
	c.cached[key] = c.clock
	c.order = append(c.order, key)
	c.resident += int64(len(stored))
	c.stats.BytesRead += int64(len(stored))
	return nil
}

// Stats returns a deterministic snapshot of cache traffic.
func (c *V41ExpertCache) Stats() V41ExpertCacheStats {
	out := c.stats
	out.ResidentBytes = c.resident
	out.ResidentEntries = len(c.entries)
	return out
}

func (c *V41ExpertCache) overBudget(addBytes int64, addEntries int) bool {
	if c.budgetBytes > 0 && c.resident+addBytes > c.budgetBytes {
		return true
	}
	if c.maxEntries > 0 && len(c.entries)+addEntries > c.maxEntries {
		return true
	}
	return false
}

func (c *V41ExpertCache) evictOne() bool {
	if len(c.order) == 0 {
		return false
	}
	victimIdx := 0
	victimClock := c.cached[c.order[0]]
	for i := 1; i < len(c.order); i++ {
		key := c.order[i]
		if c.cached[key] < victimClock {
			victimIdx = i
			victimClock = c.cached[key]
		}
	}
	key := c.order[victimIdx]
	c.resident -= int64(len(c.entries[key]))
	delete(c.entries, key)
	delete(c.cached, key)
	c.order = append(c.order[:victimIdx], c.order[victimIdx+1:]...)
	c.stats.Evictions++
	return true
}

func (c *V41ExpertCache) removeFromOrder(key v41ExpertKey) {
	for i := range c.order {
		if c.order[i] == key {
			c.order = append(c.order[:i], c.order[i+1:]...)
			return
		}
	}
}

// V41EngramRowSource is the backing read seam for Engram rows, so tests can inject
// an in-memory or synthetic NVMe source without a real file.
type V41EngramRowSource interface {
	// RowBytes returns the byte width of one row (constant per table).
	RowBytes() int
	// ReadRows reads count contiguous rows starting at row index `start` into dst,
	// returning the number of bytes written. Implementations count the physical IO.
	ReadRows(start, count int, dst []byte) (int, error)
}

// V41EngramRowCacheOptions configures the bounded Engram row cache + prefetch.
type V41EngramRowCacheOptions struct {
	TableRows      int   // total rows in the table
	RowBytes       int   // bytes per row
	BudgetBytes    int64 // resident cache budget
	PrefetchRows   int   // rows to prefetch ahead on a miss; <=0 disables prefetch
	PrefetchWindow int   // max readahead window in rows (clamped >= PrefetchRows)
}

// V41EngramRowCache is a bounded row cache over a V41EngramRowSource with eager,
// bounded prefetch and page-amplification telemetry.
type V41EngramRowCache struct {
	src   V41EngramRowSource
	opts  V41EngramRowCacheOptions
	inner *V41ExpertCache
	stats V41EngramRowStats
}

// V41EngramRowStats reports hit/miss, prefetch and streaming telemetry.
type V41EngramRowStats struct {
	Hits              int64   `json:"hits"`
	Misses            int64   `json:"misses"`
	Prefetches        int64   `json:"prefetches"`
	RowsServed        int64   `json:"rows_served"`
	BytesRead         int64   `json:"bytes_read"`
	BytesServed       int64   `json:"bytes_served"`
	UsefulBytes       int64   `json:"useful_bytes"`
	PageAmplification float64 `json:"page_amplification"`
}

// NewV41EngramRowCache validates geometry and budget and returns a bounded row cache.
func NewV41EngramRowCache(src V41EngramRowSource, opts V41EngramRowCacheOptions) (*V41EngramRowCache, error) {
	if src == nil {
		return nil, fmt.Errorf("model: V41 Engram row cache requires a source")
	}
	if opts.TableRows <= 0 {
		return nil, fmt.Errorf("model: V41 Engram table rows %d must be positive", opts.TableRows)
	}
	if opts.RowBytes <= 0 {
		return nil, fmt.Errorf("model: V41 Engram row bytes %d must be positive", opts.RowBytes)
	}
	if opts.BudgetBytes < int64(opts.RowBytes) {
		return nil, fmt.Errorf("model: V41 Engram budget %d smaller than one row %d", opts.BudgetBytes, opts.RowBytes)
	}
	if opts.PrefetchWindow < opts.PrefetchRows {
		opts.PrefetchWindow = opts.PrefetchRows
	}
	c := &V41EngramRowCache{src: src, opts: opts}
	c.inner = NewV41ExpertCache(V41ExpertCacheOptions{BudgetBytes: opts.BudgetBytes})
	return c, nil
}

// Row returns row `idx`, serving from cache on hit and reading via the source on miss,
// additionally issuing a best-effort eager prefetch of the next <=PrefetchRows rows.
func (c *V41EngramRowCache) Row(idx int) ([]byte, error) {
	if idx < 0 || idx >= c.opts.TableRows {
		return nil, fmt.Errorf("model: V41 Engram row %d out of range [0,%d)", idx, c.opts.TableRows)
	}
	if payload, ok := c.inner.Get(0, idx); ok {
		c.stats.Hits++
		c.stats.RowsServed++
		c.stats.BytesServed += int64(len(payload))
		c.stats.UsefulBytes += int64(len(payload))
		return payload, nil
	}

	c.stats.Misses++
	row, n, err := c.read(idx, 1)
	if err != nil {
		return nil, err
	}
	c.stats.BytesRead += int64(n)
	c.stats.RowsServed++
	c.stats.BytesServed += int64(n)
	c.stats.UsefulBytes += int64(n)

	if c.opts.PrefetchRows > 0 {
		c.prefetch(idx)
	}
	return row, nil
}

// Stats returns hit/miss, bytes read from source, bytes served, useful bytes, and the
// page-amplification ratio (bytes read / useful bytes), 0 when no useful bytes.
func (c *V41EngramRowCache) Stats() V41EngramRowStats {
	out := c.stats
	if out.UsefulBytes > 0 {
		out.PageAmplification = float64(out.BytesRead) / float64(out.UsefulBytes)
	}
	return out
}

func (c *V41EngramRowCache) read(start, count int) ([]byte, int, error) {
	buf := make([]byte, count*c.opts.RowBytes)
	n, err := c.src.ReadRows(start, count, buf)
	if err != nil {
		return nil, n, fmt.Errorf("model: V41 Engram row read: %w", err)
	}
	return buf[:n], n, nil
}

func (c *V41EngramRowCache) prefetch(idx int) {
	start := idx + 1
	remaining := c.opts.TableRows - start
	if remaining <= 0 {
		return
	}
	count := c.opts.PrefetchRows
	if count > remaining {
		count = remaining
	}
	buf := make([]byte, count*c.opts.RowBytes)
	n, err := c.src.ReadRows(start, count, buf)
	if err != nil {
		return
	}
	c.stats.BytesRead += int64(n)
	c.stats.Prefetches += int64(count)
	for r := 0; r < count; r++ {
		lo := r * c.opts.RowBytes
		hi := lo + c.opts.RowBytes
		if hi > n {
			hi = n
		}
		if lo >= hi {
			break
		}
		_ = c.inner.Put(0, start+r, buf[lo:hi])
	}
}

// V41EngramGeometryFromConfig returns (rowBytes, totalRows) for the Engram table at
// engLayerIndex within c.DeepSeekV41, using EngramNumEmbeddings and EngramHeadDim. A row
// is one hidden vector, so rowBytes = EngramHeadDim. It errors if DeepSeekV41 is nil or
// the index is out of range.
func V41EngramGeometryFromConfig(c Config, engLayerIndex int) (rowBytes, totalRows int, err error) {
	m := c.DeepSeekV41
	if m == nil {
		return 0, 0, fmt.Errorf("model: Engram geometry requires DeepSeekV41 metadata")
	}
	if engLayerIndex < 0 || engLayerIndex >= len(m.EngramNumEmbeddings) || engLayerIndex >= len(m.EngramLayerIDs) {
		return 0, 0, fmt.Errorf("model: Engram layer index %d out of range", engLayerIndex)
	}
	if m.EngramHeadDim <= 0 {
		return 0, 0, fmt.Errorf("model: Engram head dim %d must be positive", m.EngramHeadDim)
	}
	if m.EngramNumEmbeddings[engLayerIndex] <= 0 {
		return 0, 0, fmt.Errorf("model: Engram rows %d must be positive", m.EngramNumEmbeddings[engLayerIndex])
	}
	return m.EngramHeadDim, m.EngramNumEmbeddings[engLayerIndex], nil
}
