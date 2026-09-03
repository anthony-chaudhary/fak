package session

import (
	"sort"
	"sync"
)

// TokenSource identifies the origin of a token record, ordered by priority.
type TokenSource int

const (
	TokenSourceFallback TokenSource = iota
	TokenSourceNativeTranscript
	TokenSourceStdout
	TokenSourceNetwork
)

// String returns the canonical name for the token source.
func (s TokenSource) String() string {
	switch s {
	case TokenSourceFallback:
		return "fallback"
	case TokenSourceNativeTranscript:
		return "transcript"
	case TokenSourceStdout:
		return "stdout"
	case TokenSourceNetwork:
		return "network"
	default:
		return "unknown"
	}
}

// TokenRecord captures token consumption reported by a source for a turn.
type TokenRecord struct {
	Source        TokenSource `json:"source"`
	SessionID     string      `json:"session_id"`
	TurnIndex     int         `json:"turn_index"`
	CallID        string      `json:"call_id,omitempty"`
	PromptTokens  int         `json:"prompt_tokens"`
	OutputTokens  int         `json:"output_tokens"`
	TotalTokens   int         `json:"total_tokens"`
	CachedTokens  int         `json:"cached_tokens,omitempty"`
	CreatedTokens int         `json:"created_tokens,omitempty"`
}

// TokenMergeKey uniquely identifies a turn call within a session.
type TokenMergeKey struct {
	SessionID string
	TurnIndex int
	CallID    string
}

// Key returns the merge key for the token record.
func (r TokenRecord) Key() TokenMergeKey {
	return TokenMergeKey{
		SessionID: r.SessionID,
		TurnIndex: r.TurnIndex,
		CallID:    r.CallID,
	}
}

// MultiSourceTokenMerger stores deduplicated token records merged across multiple sources.
type MultiSourceTokenMerger struct {
	mu      sync.RWMutex
	records map[TokenMergeKey]TokenRecord
}

// NewMultiSourceTokenMerger creates an empty token merger.
func NewMultiSourceTokenMerger() *MultiSourceTokenMerger {
	return &MultiSourceTokenMerger{
		records: make(map[TokenMergeKey]TokenRecord),
	}
}

// Ingest merges a token record into the store according to source priority.
func (m *MultiSourceTokenMerger) Ingest(r TokenRecord) {
	key := r.Key()
	if r.TotalTokens == 0 && (r.PromptTokens > 0 || r.OutputTokens > 0) {
		r.TotalTokens = r.PromptTokens + r.OutputTokens
	}

	m.mu.Lock()
	defer m.mu.Unlock()

	existing, exists := m.records[key]
	if !exists {
		m.records[key] = r
		return
	}

	if r.Source > existing.Source {
		merged := r
		if merged.CachedTokens == 0 && existing.CachedTokens > 0 {
			merged.CachedTokens = existing.CachedTokens
		}
		if merged.CreatedTokens == 0 && existing.CreatedTokens > 0 {
			merged.CreatedTokens = existing.CreatedTokens
		}
		m.records[key] = merged
	} else if r.Source < existing.Source {
		merged := existing
		if merged.CachedTokens == 0 && r.CachedTokens > 0 {
			merged.CachedTokens = r.CachedTokens
		}
		if merged.CreatedTokens == 0 && r.CreatedTokens > 0 {
			merged.CreatedTokens = r.CreatedTokens
		}
		m.records[key] = merged
	} else {
		merged := existing
		if r.PromptTokens > 0 {
			merged.PromptTokens = r.PromptTokens
		}
		if r.OutputTokens > 0 {
			merged.OutputTokens = r.OutputTokens
		}
		if r.TotalTokens > 0 {
			merged.TotalTokens = r.TotalTokens
		}
		if merged.TotalTokens == 0 && (merged.PromptTokens > 0 || merged.OutputTokens > 0) {
			merged.TotalTokens = merged.PromptTokens + merged.OutputTokens
		}
		if merged.CachedTokens == 0 && r.CachedTokens > 0 {
			merged.CachedTokens = r.CachedTokens
		}
		if merged.CreatedTokens == 0 && r.CreatedTokens > 0 {
			merged.CreatedTokens = r.CreatedTokens
		}
		m.records[key] = merged
	}
}

// Records returns deduplicated records in deterministic order.
func (m *MultiSourceTokenMerger) Records() []TokenRecord {
	m.mu.RLock()
	defer m.mu.RUnlock()

	records := make([]TokenRecord, 0, len(m.records))
	for _, r := range m.records {
		records = append(records, r)
	}

	sort.Slice(records, func(i, j int) bool {
		if records[i].SessionID != records[j].SessionID {
			return records[i].SessionID < records[j].SessionID
		}
		if records[i].TurnIndex != records[j].TurnIndex {
			return records[i].TurnIndex < records[j].TurnIndex
		}
		return records[i].CallID < records[j].CallID
	})

	return records
}

// Totals returns sums across all deduplicated turns.
func (m *MultiSourceTokenMerger) Totals() (prompt, output, total, cached, created int) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	for _, r := range m.records {
		prompt += r.PromptTokens
		output += r.OutputTokens
		total += r.TotalTokens
		cached += r.CachedTokens
		created += r.CreatedTokens
	}
	return
}

// TotalUsage returns the combined session usage.
func (m *MultiSourceTokenMerger) TotalUsage() Usage {
	prompt, output, _, _, _ := m.Totals()
	return Usage{
		OutputTokens:  output,
		ContextTokens: prompt,
	}
}

// MergeTokenRecords ingests a slice of records and returns deduplicated records in deterministic order.
func MergeTokenRecords(records []TokenRecord) []TokenRecord {
	merger := NewMultiSourceTokenMerger()
	for _, r := range records {
		merger.Ingest(r)
	}
	return merger.Records()
}
