package vdso

import (
	"container/list"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// SearchQuery encapsulates the parameters of a search operation for caching.
type SearchQuery struct {
	Tool      string            `json:"tool"`
	Pattern   string            `json:"pattern,omitempty"`
	Path      string            `json:"path,omitempty"`
	Include   string            `json:"include,omitempty"`
	Options   map[string]string `json:"options,omitempty"`
	Principal string            `json:"principal,omitempty"`
}

// SearchEntry holds a cached search query and its materialized result.
type SearchEntry struct {
	Key       string
	Tool      string
	Directory string // normalized directory scope
	CanonDir  string // canonical absolute directory scope
	Result    *abi.Result
	Witness   string
	FilledAt  time.Time
}

// SearchCache is a thread-safe Tier-2 search result memoization cache with
// hierarchical path-scope invalidation (#11492). It memoizes search operations
// (fak_grep, fak_glob, Grep, Glob, etc.) across unchanging directory trees and
// invalidates them hierarchically when files or directories mutate.
type SearchCache struct {
	mu           sync.Mutex
	capacity     int
	entries      map[string]*list.Element       // key -> list.Element containing *SearchEntry
	lru          *list.List                     // front = most recent
	dirIndex     map[string]map[string]struct{} // directory -> set of keys
	witnessIndex map[string]map[string]struct{} // witness -> set of keys

	lookups       int64
	hits          int64
	fills         int64
	evictions     int64
	invalidations int64
}

// NewSearchCache creates a new SearchCache with the specified entry capacity.
func NewSearchCache(capacity int) *SearchCache {
	if capacity <= 0 {
		capacity = DefaultCacheSize
	}
	return &SearchCache{
		capacity:     capacity,
		entries:      make(map[string]*list.Element),
		lru:          list.New(),
		dirIndex:     make(map[string]map[string]struct{}),
		witnessIndex: make(map[string]map[string]struct{}),
	}
}

// SearchKey derives the canonical compound cache key for a search tool call and its arguments.
func SearchKey(tool string, args []byte, principal string) string {
	tool = strings.ToLower(strings.TrimSpace(tool))
	pattern := ExtractToolPattern(args)
	dir := cleanSearchDir(ExtractToolDirectory(args))
	include := ExtractToolInclude(args)

	var opts string
	if len(args) > 0 {
		var m map[string]any
		if json.Unmarshal(args, &m) == nil {
			var keys []string
			for k := range m {
				switch k {
				case "pattern", "query", "regex", "search", "patterns", "queries",
					"path", "dir", "directory", "filePath", "file_path", "root", "workspace",
					"include", "glob", "filePattern", "trace_id", "traceId":
					// primary or volatile fields excluded
				default:
					keys = append(keys, k)
				}
			}
			if len(keys) > 0 {
				sort.Strings(keys)
				var sb strings.Builder
				for _, k := range keys {
					sb.WriteString(fmt.Sprintf("%s=%v;", k, m[k]))
				}
				opts = sb.String()
			}
		}
	}

	h := sha256.New()
	h.Write([]byte(tool))
	h.Write([]byte{0})
	h.Write([]byte(pattern))
	h.Write([]byte{0})
	h.Write([]byte(dir))
	h.Write([]byte{0})
	h.Write([]byte(include))
	h.Write([]byte{0})
	h.Write([]byte(opts))
	h.Write([]byte{0})
	h.Write([]byte(principal))

	sum := hex.EncodeToString(h.Sum(nil))[:24]
	return fmt.Sprintf("search:%s:%s:%s", tool, dir, sum)
}

// Get retrieves a cached result for the search tool call, if present.
func (sc *SearchCache) Get(c *abi.ToolCall, args []byte) (*abi.Result, bool) {
	if sc == nil || c == nil {
		return nil, false
	}
	atomic.AddInt64(&sc.lookups, 1)
	key := SearchKey(c.Tool, args, principalOf(c))

	sc.mu.Lock()
	el, ok := sc.entries[key]
	if !ok {
		sc.mu.Unlock()
		return nil, false
	}
	sc.lru.MoveToFront(el)
	e := el.Value.(*SearchEntry)
	atomic.AddInt64(&sc.hits, 1)

	r := e.Result
	filledAt := e.FilledAt
	sc.mu.Unlock()

	return cloneResultForServe(c, r, filledAt), true
}

// Put inserts or updates a search result in the cache. Returns true if a new entry was added.
func (sc *SearchCache) Put(c *abi.ToolCall, args []byte, r *abi.Result, witness string) bool {
	if sc == nil || c == nil || r == nil || r.Status != abi.StatusOK || destructive(c) {
		return false
	}
	key := SearchKey(c.Tool, args, principalOf(c))
	dir := cleanSearchDir(ExtractToolDirectory(args))
	canonDir := canonicalDir(dir)

	sc.mu.Lock()
	defer sc.mu.Unlock()

	if el, ok := sc.entries[key]; ok {
		e := el.Value.(*SearchEntry)
		e.Result = r
		e.Witness = witness
		e.FilledAt = time.Now()
		sc.lru.MoveToFront(el)
		return false
	}

	for sc.lru.Len() >= sc.capacity {
		back := sc.lru.Back()
		if back == nil {
			break
		}
		be := back.Value.(*SearchEntry)
		sc.lru.Remove(back)
		delete(sc.entries, be.Key)
		sc.unindexEntryLocked(be)
		if be.Result != nil {
			abi.UnpinResolved(be.Result.Payload)
		}
		sc.evictions++
	}

	entry := &SearchEntry{
		Key:       key,
		Tool:      strings.ToLower(c.Tool),
		Directory: dir,
		CanonDir:  canonDir,
		Result:    r,
		Witness:   witness,
		FilledAt:  time.Now(),
	}
	el := sc.lru.PushFront(entry)
	sc.entries[key] = el
	sc.indexEntryLocked(entry)
	abi.PinResolved(r.Payload)
	sc.fills++
	return true
}

// Resize changes the LRU capacity of the search cache and evicts overflow entries.
func (sc *SearchCache) Resize(capacity int) int {
	if sc == nil || capacity <= 0 {
		return 0
	}
	sc.mu.Lock()
	defer sc.mu.Unlock()

	sc.capacity = capacity
	evicted := 0
	for sc.lru.Len() > sc.capacity {
		back := sc.lru.Back()
		if back == nil {
			break
		}
		be := back.Value.(*SearchEntry)
		sc.lru.Remove(back)
		delete(sc.entries, be.Key)
		sc.unindexEntryLocked(be)
		if be.Result != nil {
			abi.UnpinResolved(be.Result.Payload)
		}
		sc.evictions++
		evicted++
	}
	return evicted
}

// InvalidatePath hierarchically invalidates all search cache entries whose directory scope
// overlaps with path (path is inside dir, dir is inside path, or dir == path).
func (sc *SearchCache) InvalidatePath(path string) int {
	if sc == nil || path == "" {
		return 0
	}
	sc.mu.Lock()
	defer sc.mu.Unlock()

	evicted := 0
	for el := sc.lru.Front(); el != nil; {
		next := el.Next()
		e := el.Value.(*SearchEntry)
		if pathOverlap(e.Directory, path) || (e.CanonDir != "" && pathOverlap(e.CanonDir, path)) {
			sc.lru.Remove(el)
			delete(sc.entries, e.Key)
			sc.unindexEntryLocked(e)
			if e.Result != nil {
				abi.UnpinResolved(e.Result.Payload)
			}
			evicted++
			sc.invalidations++
		}
		el = next
	}
	return evicted
}

// Revoke invalidates all search cache entries admitted under the specified witness.
func (sc *SearchCache) Revoke(witness string) int {
	if sc == nil || witness == "" {
		return 0
	}
	sc.mu.Lock()
	defer sc.mu.Unlock()

	keys, ok := sc.witnessIndex[witness]
	if !ok {
		return 0
	}
	keysList := make([]string, 0, len(keys))
	for key := range keys {
		keysList = append(keysList, key)
	}
	evicted := 0
	for _, key := range keysList {
		if el, found := sc.entries[key]; found {
			e := el.Value.(*SearchEntry)
			sc.lru.Remove(el)
			delete(sc.entries, key)
			sc.unindexEntryLocked(e)
			if e.Result != nil {
				abi.UnpinResolved(e.Result.Payload)
			}
			evicted++
			sc.invalidations++
		}
	}
	delete(sc.witnessIndex, witness)
	return evicted
}

// Clear flushes all entries in the search cache.
func (sc *SearchCache) Clear() {
	if sc == nil {
		return
	}
	sc.mu.Lock()
	defer sc.mu.Unlock()

	for el := sc.lru.Front(); el != nil; el = el.Next() {
		e := el.Value.(*SearchEntry)
		if e.Result != nil {
			abi.UnpinResolved(e.Result.Payload)
		}
	}
	sc.entries = make(map[string]*list.Element)
	sc.lru.Init()
	sc.dirIndex = make(map[string]map[string]struct{})
	sc.witnessIndex = make(map[string]map[string]struct{})
}

// Len returns the current number of cached search entries.
func (sc *SearchCache) Len() int {
	if sc == nil {
		return 0
	}
	sc.mu.Lock()
	defer sc.mu.Unlock()
	return sc.lru.Len()
}

// Capacity returns the maximum number of entries before LRU eviction.
func (sc *SearchCache) Capacity() int {
	if sc == nil {
		return 0
	}
	return sc.capacity
}

// Stats returns cumulative metrics for search cache operations.
func (sc *SearchCache) Stats() (lookups, hits, fills, evictions, invalidations int64) {
	if sc == nil {
		return 0, 0, 0, 0, 0
	}
	sc.mu.Lock()
	defer sc.mu.Unlock()
	return atomic.LoadInt64(&sc.lookups),
		atomic.LoadInt64(&sc.hits),
		atomic.LoadInt64(&sc.fills),
		sc.evictions,
		sc.invalidations
}

func (sc *SearchCache) indexEntryLocked(e *SearchEntry) {
	if e.Directory != "" {
		if sc.dirIndex[e.Directory] == nil {
			sc.dirIndex[e.Directory] = make(map[string]struct{})
		}
		sc.dirIndex[e.Directory][e.Key] = struct{}{}
	}
	if e.CanonDir != "" && e.CanonDir != e.Directory {
		if sc.dirIndex[e.CanonDir] == nil {
			sc.dirIndex[e.CanonDir] = make(map[string]struct{})
		}
		sc.dirIndex[e.CanonDir][e.Key] = struct{}{}
	}
	if e.Witness != "" {
		if sc.witnessIndex[e.Witness] == nil {
			sc.witnessIndex[e.Witness] = make(map[string]struct{})
		}
		sc.witnessIndex[e.Witness][e.Key] = struct{}{}
	}
}

func (sc *SearchCache) unindexEntryLocked(e *SearchEntry) {
	if e.Directory != "" {
		sc.unindexDirLocked(e.Directory, e.Key)
	}
	if e.CanonDir != "" && e.CanonDir != e.Directory {
		sc.unindexDirLocked(e.CanonDir, e.Key)
	}
	if e.Witness != "" {
		if m := sc.witnessIndex[e.Witness]; m != nil {
			delete(m, e.Key)
			if len(m) == 0 {
				delete(sc.witnessIndex, e.Witness)
			}
		}
	}
}

func (sc *SearchCache) unindexDirLocked(dir, key string) {
	if m := sc.dirIndex[dir]; m != nil {
		delete(m, key)
		if len(m) == 0 {
			delete(sc.dirIndex, dir)
		}
	}
}

func cloneResultForServe(c *abi.ToolCall, orig *abi.Result, filledAt time.Time) *abi.Result {
	if orig == nil {
		return nil
	}
	meta := make(map[string]string)
	if orig.Meta != nil {
		for k, v := range orig.Meta {
			meta[k] = v
		}
	}
	meta["served_by"] = "vdso"
	meta["tier"] = "2"
	ageMs := time.Since(filledAt).Milliseconds()
	if ageMs < 0 {
		ageMs = 0
	}
	meta["age_ms"] = strconv.FormatInt(ageMs, 10)
	return &abi.Result{
		Call:    c,
		Payload: orig.Payload,
		Status:  orig.Status,
		Meta:    meta,
	}
}

func cleanSearchDir(dir string) string {
	dir = strings.TrimSpace(dir)
	if dir == "" || dir == "." {
		return "."
	}
	cleaned := filepath.ToSlash(filepath.Clean(dir))
	cleaned = strings.TrimPrefix(cleaned, "./")
	if cleaned == "" || cleaned == "." {
		return "."
	}
	if isCaseInsensitiveOS {
		cleaned = strings.ToLower(cleaned)
	}
	return cleaned
}

func canonicalDir(p string) string {
	p = strings.TrimSpace(p)
	if p == "" || p == "." {
		if abs, err := filepath.Abs("."); err == nil {
			p = abs
		} else {
			p = "."
		}
	} else if !isAbsPath(p) {
		if abs, err := filepath.Abs(p); err == nil {
			p = abs
		}
	}
	p = filepath.ToSlash(filepath.Clean(p))
	if isCaseInsensitiveOS {
		p = strings.ToLower(p)
	}
	return p
}

// pathOverlap reports whether directory scope dir and mutation/invalidation target path
// overlap hierarchically (target is inside dir, dir is inside target, or dir == target).
func pathOverlap(dir, target string) bool {
	dir = cleanSearchDir(dir)
	target = cleanSearchDir(target)

	if dir == target {
		return true
	}
	if dir == "." || dir == "" {
		if !isAbsPath(target) {
			return true
		}
		c := canonicalDir(".")
		return c == target || isSubpath(c, target)
	}
	if target == "." || target == "" {
		if !isAbsPath(dir) {
			return true
		}
		c := canonicalDir(".")
		return c == dir || isSubpath(c, dir)
	}
	if isSubpath(dir, target) || isSubpath(target, dir) {
		return true
	}

	cDir := canonicalDir(dir)
	cTarget := canonicalDir(target)
	if cDir != dir || cTarget != target {
		if cDir == cTarget {
			return true
		}
		if isSubpath(cDir, cTarget) || isSubpath(cTarget, cDir) {
			return true
		}
	}

	return false
}

func isAbsPath(p string) bool {
	return filepath.IsAbs(p) || isWindowsDrivePath(p) || strings.HasPrefix(p, "/") || strings.HasPrefix(p, "\\")
}

func isSubpath(parent, child string) bool {
	if parent == "." || parent == "" {
		return true
	}
	if !strings.HasSuffix(parent, "/") {
		parent += "/"
	}
	return strings.HasPrefix(child, parent)
}
