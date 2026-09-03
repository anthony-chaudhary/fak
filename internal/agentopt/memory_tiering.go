package agentopt

import (
	"errors"
	"fmt"
	"sort"
	"strings"
	"sync"
	"time"
)

// Family 6: Agent memory architectures.
//
// Hierarchical memory tiering partitions agent context into two distinct tiers:
// 1. Hot Working Memory: maintains active task variables and immediate constraints
//    under a bounded token capacity (< 1000 tokens).
// 2. Cold Episodic Memory: stores completed milestones, past turn facts, and historical
//    observations, demand-paging relevant items via keyword and semantic search.

const (
	// DefaultWorkingMemoryCapacity defines the standard token ceiling (< 1000 tokens)
	// for active task variables and immediate constraints.
	DefaultWorkingMemoryCapacity = 1000

	// DefaultEpisodicTopK defines the default number of items returned for episodic queries.
	DefaultEpisodicTopK = 5
)

// EstimateTokens approximates token count for text (~4 characters per token, minimum 1 for non-empty string).
func EstimateTokens(text string) int {
	if len(text) == 0 {
		return 0
	}
	return (len(text) + 3) / 4
}

// WorkingItem represents an active task variable or immediate constraint in hot working memory.
type WorkingItem struct {
	Key         string    `json:"key"`
	Value       string    `json:"value"`
	Tokens      int       `json:"tokens"`
	Timestamp   time.Time `json:"timestamp"`
	AccessCount int       `json:"access_count"`
}

// WorkingMemory maintains active task variables and immediate constraints in hot memory
// subject to a strict token capacity ceiling.
type WorkingMemory struct {
	mu          sync.RWMutex
	items       map[string]*WorkingItem
	order       []string // keys in chronological order (oldest to newest)
	maxTokens   int
	totalTokens int
	onExpire    func(item *WorkingItem)
}

// NewWorkingMemory creates a bounded working hot memory store.
func NewWorkingMemory(capacityTokens int) *WorkingMemory {
	if capacityTokens <= 0 {
		capacityTokens = DefaultWorkingMemoryCapacity
	}
	return &WorkingMemory{
		items:     make(map[string]*WorkingItem),
		order:     make([]string, 0),
		maxTokens: capacityTokens,
	}
}

// Set adds or updates an active task variable in working memory.
// If adding the item would breach the bounded token capacity, oldest items are expired
// until total tokens remain within maxTokens.
func (w *WorkingMemory) Set(key, val string) error {
	w.mu.Lock()
	defer w.mu.Unlock()

	key = strings.TrimSpace(key)
	if key == "" {
		return errors.New("working memory key cannot be empty")
	}

	tok := EstimateTokens(key) + EstimateTokens(val)
	if tok > w.maxTokens {
		return fmt.Errorf("item tokens (%d) exceed working memory capacity (%d)", tok, w.maxTokens)
	}

	existing, exists := w.items[key]
	existingTok := 0
	if exists {
		existingTok = existing.Tokens
	}

	// Expire oldest items if adding this item would breach capacity
	for len(w.items) > 0 && (w.totalTokens-existingTok+tok) > w.maxTokens {
		if len(w.items) == 1 && exists {
			break
		}
		w.expireOldestLocked(key)
	}

	if (w.totalTokens - existingTok + tok) > w.maxTokens {
		return fmt.Errorf("insufficient capacity in working memory for item tokens (%d)", tok)
	}

	if exists {
		w.totalTokens -= existingTok
		existing.Value = val
		existing.Tokens = tok
		existing.Timestamp = time.Now()
		existing.AccessCount++
		w.totalTokens += tok
		w.promoteOrderLocked(key)
		return nil
	}

	item := &WorkingItem{
		Key:         key,
		Value:       val,
		Tokens:      tok,
		Timestamp:   time.Now(),
		AccessCount: 1,
	}
	w.items[key] = item
	w.order = append(w.order, key)
	w.totalTokens += tok
	return nil
}

// Get retrieves the value of an active task variable and increments its access count.
func (w *WorkingMemory) Get(key string) (string, bool) {
	w.mu.Lock()
	defer w.mu.Unlock()

	item, exists := w.items[key]
	if !exists {
		return "", false
	}
	item.AccessCount++
	return item.Value, true
}

// Delete removes an item from working memory.
func (w *WorkingMemory) Delete(key string) bool {
	w.mu.Lock()
	defer w.mu.Unlock()

	item, exists := w.items[key]
	if !exists {
		return false
	}
	delete(w.items, key)
	w.totalTokens -= item.Tokens
	for i, k := range w.order {
		if k == key {
			w.order = append(w.order[:i], w.order[i+1:]...)
			break
		}
	}
	return true
}

// ExpireOldest removes the oldest item from working memory to free token capacity.
func (w *WorkingMemory) ExpireOldest() (*WorkingItem, bool) {
	w.mu.Lock()
	defer w.mu.Unlock()
	return w.expireOldestLocked("")
}

func (w *WorkingMemory) expireOldestLocked(excludeKey string) (*WorkingItem, bool) {
	idx := -1
	for i, k := range w.order {
		if k != excludeKey {
			idx = i
			break
		}
	}
	if idx == -1 {
		return nil, false
	}

	oldestKey := w.order[idx]
	w.order = append(w.order[:idx], w.order[idx+1:]...)
	item := w.items[oldestKey]
	delete(w.items, oldestKey)
	if item != nil {
		w.totalTokens -= item.Tokens
		if w.onExpire != nil {
			w.onExpire(item)
		}
	}
	return item, true
}

func (w *WorkingMemory) promoteOrderLocked(key string) {
	for i, k := range w.order {
		if k == key {
			w.order = append(w.order[:i], w.order[i+1:]...)
			w.order = append(w.order, key)
			break
		}
	}
}

// TotalTokens returns the sum of tokens resident in working memory.
func (w *WorkingMemory) TotalTokens() int {
	w.mu.RLock()
	defer w.mu.RUnlock()
	return w.totalTokens
}

// CapacityTokens returns the maximum token limit of working memory.
func (w *WorkingMemory) CapacityTokens() int {
	w.mu.RLock()
	defer w.mu.RUnlock()
	return w.maxTokens
}

// Count returns the number of active items in working memory.
func (w *WorkingMemory) Count() int {
	w.mu.RLock()
	defer w.mu.RUnlock()
	return len(w.items)
}

// Keys returns all active keys in working memory in chronological order.
func (w *WorkingMemory) Keys() []string {
	w.mu.RLock()
	defer w.mu.RUnlock()
	out := make([]string, len(w.order))
	copy(out, w.order)
	return out
}

// Items returns snapshot copies of all active items in working memory.
func (w *WorkingMemory) Items() []*WorkingItem {
	w.mu.RLock()
	defer w.mu.RUnlock()
	out := make([]*WorkingItem, 0, len(w.order))
	for _, k := range w.order {
		if it, ok := w.items[k]; ok {
			cp := *it
			out = append(out, &cp)
		}
	}
	return out
}

// Variables returns a copy of active key-value pairs in working memory.
func (w *WorkingMemory) Variables() map[string]string {
	w.mu.RLock()
	defer w.mu.RUnlock()
	out := make(map[string]string, len(w.items))
	for k, it := range w.items {
		out[k] = it.Value
	}
	return out
}

// EpisodicRecord models a cold memory item (completed milestone, past turn fact, observation).
type EpisodicRecord struct {
	ID          string            `json:"id"`
	Category    string            `json:"category,omitempty"` // "milestone", "turn_fact", "observation", "constraint"
	Content     string            `json:"content"`
	Keywords    []string          `json:"keywords,omitempty"`
	Metadata    map[string]string `json:"metadata,omitempty"`
	Timestamp   time.Time         `json:"timestamp"`
	Tokens      int               `json:"tokens"`
	DemandPaged bool              `json:"demand_paged,omitempty"`
	AccessCount int               `json:"access_count,omitempty"`
}

// NewEpisodicRecord creates a general episodic record.
func NewEpisodicRecord(id, category, content string, keywords ...string) EpisodicRecord {
	return EpisodicRecord{
		ID:        id,
		Category:  category,
		Content:   content,
		Keywords:  keywords,
		Timestamp: time.Now(),
		Tokens:    EstimateTokens(content),
	}
}

// NewMilestoneRecord creates an episodic record for a completed milestone.
func NewMilestoneRecord(id, content string, keywords ...string) EpisodicRecord {
	return NewEpisodicRecord(id, "milestone", content, keywords...)
}

// NewTurnFactRecord creates an episodic record for a past turn fact or observation.
func NewTurnFactRecord(id, content string, keywords ...string) EpisodicRecord {
	return NewEpisodicRecord(id, "turn_fact", content, keywords...)
}

// HierarchicalMemoryTier defines the two-tier hierarchical agent memory operations.
type HierarchicalMemoryTier interface {
	SetWorking(key, val string) error
	GetWorking(key string) (string, bool)
	ArchiveToEpisodic(item EpisodicRecord) error
	QueryEpisodic(query string, topK int) []EpisodicRecord
}

// HierarchicalMemoryManager coordinates hot working memory (< 1000 tokens) and
// cold episodic memory with demand-paged semantic and keyword query capabilities.
type HierarchicalMemoryManager struct {
	mu                  sync.RWMutex
	working             *WorkingMemory
	episodic            []EpisodicRecord
	nextEpisodicID      uint64
	autoArchiveOnExpire bool
}

// Compile-time check that HierarchicalMemoryManager implements HierarchicalMemoryTier.
var _ HierarchicalMemoryTier = (*HierarchicalMemoryManager)(nil)

// NewHierarchicalMemoryManager creates a HierarchicalMemoryManager with bounded working memory.
func NewHierarchicalMemoryManager(maxWorkingTokens int) *HierarchicalMemoryManager {
	if maxWorkingTokens <= 0 {
		maxWorkingTokens = DefaultWorkingMemoryCapacity
	}
	return &HierarchicalMemoryManager{
		working:  NewWorkingMemory(maxWorkingTokens),
		episodic: make([]EpisodicRecord, 0),
	}
}

// NewHierarchicalMemoryTier returns a HierarchicalMemoryTier interface backed by HierarchicalMemoryManager.
func NewHierarchicalMemoryTier(maxWorkingTokens int) *HierarchicalMemoryManager {
	return NewHierarchicalMemoryManager(maxWorkingTokens)
}

// SetWorking stores an active task variable or immediate constraint in working hot memory.
func (m *HierarchicalMemoryManager) SetWorking(key, val string) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.working.Set(key, val)
}

// GetWorking retrieves an active task variable from working hot memory.
func (m *HierarchicalMemoryManager) GetWorking(key string) (string, bool) {
	return m.working.Get(key)
}

// ArchiveToEpisodic pushes completed milestones, past turn facts, or observations to cold episodic memory.
func (m *HierarchicalMemoryManager) ArchiveToEpisodic(item EpisodicRecord) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	if strings.TrimSpace(item.Content) == "" {
		return errors.New("episodic record content cannot be empty")
	}

	m.archiveInternalLocked(item)
	return nil
}

func (m *HierarchicalMemoryManager) archiveInternalLocked(item EpisodicRecord) {
	if item.ID == "" {
		m.nextEpisodicID++
		item.ID = fmt.Sprintf("ep-%05d", m.nextEpisodicID)
	}
	if item.Timestamp.IsZero() {
		item.Timestamp = time.Now()
	}
	if item.Tokens <= 0 {
		item.Tokens = EstimateTokens(item.Content)
	}
	m.episodic = append(m.episodic, item)
}

// QueryEpisodic searches cold episodic memory using keyword and semantic matching,
// demand-paging the top-K matching records into the returned active result set.
func (m *HierarchicalMemoryManager) QueryEpisodic(query string, topK int) []EpisodicRecord {
	m.mu.Lock()
	defer m.mu.Unlock()

	if topK <= 0 {
		topK = DefaultEpisodicTopK
	}

	if len(m.episodic) == 0 {
		return nil
	}

	type scoredRecord struct {
		index int
		score float64
	}

	queryLower := strings.ToLower(strings.TrimSpace(query))
	queryTerms := extractQueryTerms(queryLower)

	var scored []scoredRecord
	for i := range m.episodic {
		rec := &m.episodic[i]
		score := computeRelevanceScore(rec, queryLower, queryTerms)
		if score > 0 {
			scored = append(scored, scoredRecord{index: i, score: score})
		}
	}

	sort.Slice(scored, func(i, j int) bool {
		if scored[i].score != scored[j].score {
			return scored[i].score > scored[j].score
		}
		return m.episodic[scored[i].index].Timestamp.After(m.episodic[scored[j].index].Timestamp)
	})

	limit := topK
	if limit > len(scored) {
		limit = len(scored)
	}

	results := make([]EpisodicRecord, limit)
	for i := 0; i < limit; i++ {
		origIdx := scored[i].index
		m.episodic[origIdx].DemandPaged = true
		m.episodic[origIdx].AccessCount++
		results[i] = m.episodic[origIdx]
	}

	return results
}

// DemandPage is an alias for QueryEpisodic to emphasize demand-paged access to cold episodic memory.
func (m *HierarchicalMemoryManager) DemandPage(query string, topK int) []EpisodicRecord {
	return m.QueryEpisodic(query, topK)
}

// DemandPageToWorking retrieves an episodic record by ID and stages its content into hot working memory.
func (m *HierarchicalMemoryManager) DemandPageToWorking(recordID, workingKey string) error {
	m.mu.Lock()
	var found *EpisodicRecord
	for i := range m.episodic {
		if m.episodic[i].ID == recordID {
			m.episodic[i].DemandPaged = true
			m.episodic[i].AccessCount++
			found = &m.episodic[i]
			break
		}
	}
	m.mu.Unlock()

	if found == nil {
		return fmt.Errorf("episodic record with id %q not found", recordID)
	}

	return m.SetWorking(workingKey, found.Content)
}

// ExpireOldestWorking removes the oldest item from working memory.
func (m *HierarchicalMemoryManager) ExpireOldestWorking() (*WorkingItem, bool) {
	return m.working.ExpireOldest()
}

// WorkingTokens returns current resident token count in working hot memory.
func (m *HierarchicalMemoryManager) WorkingTokens() int {
	return m.working.TotalTokens()
}

// WorkingCapacityTokens returns maximum token capacity of working hot memory.
func (m *HierarchicalMemoryManager) WorkingCapacityTokens() int {
	return m.working.CapacityTokens()
}

// WorkingCount returns the number of active variables in working hot memory.
func (m *HierarchicalMemoryManager) WorkingCount() int {
	return m.working.Count()
}

// WorkingMemory returns the underlying WorkingMemory instance.
func (m *HierarchicalMemoryManager) WorkingMemory() *WorkingMemory {
	return m.working
}

// EpisodicCount returns the total number of records archived in cold episodic memory.
func (m *HierarchicalMemoryManager) EpisodicCount() int {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return len(m.episodic)
}

// EpisodicRecords returns a snapshot copy of all records archived in cold episodic memory.
func (m *HierarchicalMemoryManager) EpisodicRecords() []EpisodicRecord {
	m.mu.RLock()
	defer m.mu.RUnlock()
	out := make([]EpisodicRecord, len(m.episodic))
	copy(out, m.episodic)
	return out
}

// DemandPagedCount returns the count of episodic records that have been demand-paged.
func (m *HierarchicalMemoryManager) DemandPagedCount() int {
	m.mu.RLock()
	defer m.mu.RUnlock()
	count := 0
	for _, rec := range m.episodic {
		if rec.DemandPaged {
			count++
		}
	}
	return count
}

// DemandPagedRecords returns all episodic records that have been demand-paged.
func (m *HierarchicalMemoryManager) DemandPagedRecords() []EpisodicRecord {
	m.mu.RLock()
	defer m.mu.RUnlock()
	var out []EpisodicRecord
	for _, rec := range m.episodic {
		if rec.DemandPaged {
			out = append(out, rec)
		}
	}
	return out
}

// SetAutoArchiveOnExpiration enables or disables automatic archival of expired working memory items into episodic memory.
func (m *HierarchicalMemoryManager) SetAutoArchiveOnExpiration(enable bool) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.autoArchiveOnExpire = enable
	if enable {
		m.working.onExpire = func(item *WorkingItem) {
			m.archiveInternalLocked(EpisodicRecord{
				ID:        fmt.Sprintf("archived-%s-%d", item.Key, time.Now().UnixNano()),
				Category:  "expired_working",
				Content:   fmt.Sprintf("%s: %s", item.Key, item.Value),
				Keywords:  []string{item.Key, "working_memory"},
				Timestamp: time.Now(),
				Tokens:    item.Tokens,
			})
		}
	} else {
		m.working.onExpire = nil
	}
}

func extractQueryTerms(query string) []string {
	fields := strings.FieldsFunc(query, func(r rune) bool {
		return (r < 'a' || r > 'z') && (r < '0' || r > '9') && r != '_' && r != '-'
	})
	var terms []string
	for _, f := range fields {
		f = strings.TrimSpace(f)
		if len(f) > 0 {
			terms = append(terms, f)
		}
	}
	return terms
}

func computeRelevanceScore(rec *EpisodicRecord, queryLower string, terms []string) float64 {
	if queryLower == "" {
		return 1.0
	}

	var score float64
	contentLower := strings.ToLower(rec.Content)
	categoryLower := strings.ToLower(rec.Category)

	// 1. Exact phrase match in content
	if strings.Contains(contentLower, queryLower) {
		score += 15.0
	}

	// 2. Category matching
	if categoryLower != "" {
		if categoryLower == queryLower {
			score += 10.0
		} else if strings.Contains(queryLower, categoryLower) {
			score += 5.0
		}
	}

	// 3. Keyword matching
	for _, kw := range rec.Keywords {
		kwLower := strings.ToLower(kw)
		if kwLower == queryLower {
			score += 12.0
		}
		for _, term := range terms {
			if kwLower == term {
				score += 8.0
			} else if strings.Contains(kwLower, term) {
				score += 4.0
			}
		}
	}

	// 4. Term matches in content
	for _, term := range terms {
		count := strings.Count(contentLower, term)
		if count > 0 {
			score += float64(count) * 2.0
		}
	}

	// 5. Metadata matching
	for mk, mv := range rec.Metadata {
		mkLower := strings.ToLower(mk)
		mvLower := strings.ToLower(mv)
		for _, term := range terms {
			if strings.Contains(mkLower, term) || strings.Contains(mvLower, term) {
				score += 2.0
			}
		}
	}

	return score
}
