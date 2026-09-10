package gateway

import (
	"encoding/json"
	"sync"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

const (
	defaultResponsesContinuationMaxEntries = 1024
	defaultResponsesContinuationMaxBytes   = int64(64 << 20)
	defaultResponsesContinuationTTL        = 30 * time.Minute
)

type responsesContinuationStoreConfig struct {
	MaxEntries int
	MaxBytes   int64
	TTL        time.Duration
}

type responsesContinuationEntry struct {
	id       string
	owner    string
	messages []agent.Message
	bytes    int64
	storedAt time.Time
}

// responsesContinuationStore is the bounded, process-local Responses application-state
// store. Entries are immutable after insertion; callers always receive deep copies.
type responsesContinuationStore struct {
	mu      sync.Mutex
	cfg     responsesContinuationStoreConfig
	entries map[string]responsesContinuationEntry
	order   []string
	bytes   int64
}

func newResponsesContinuationStore(cfg responsesContinuationStoreConfig) *responsesContinuationStore {
	if cfg.MaxEntries <= 0 {
		cfg.MaxEntries = defaultResponsesContinuationMaxEntries
	}
	if cfg.MaxBytes <= 0 {
		cfg.MaxBytes = defaultResponsesContinuationMaxBytes
	}
	if cfg.TTL <= 0 {
		cfg.TTL = defaultResponsesContinuationTTL
	}
	return &responsesContinuationStore{cfg: cfg, entries: make(map[string]responsesContinuationEntry)}
}

func (s *Server) responsesContinuationState() *responsesContinuationStore {
	if s == nil {
		return nil
	}
	s.responsesContinuationOnce.Do(func() {
		if s.responsesContinuations == nil {
			s.responsesContinuations = newResponsesContinuationStore(responsesContinuationStoreConfig{})
		}
	})
	return s.responsesContinuations
}

func cloneResponsesMessages(in []agent.Message) []agent.Message {
	if len(in) == 0 {
		return nil
	}
	out := make([]agent.Message, len(in))
	copy(out, in)
	for i := range out {
		out[i].ToolCalls = append([]agent.ToolCall(nil), in[i].ToolCalls...)
		out[i].RedactedThinking = append([]string(nil), in[i].RedactedThinking...)
		if in[i].FunctionCall != nil {
			fn := *in[i].FunctionCall
			out[i].FunctionCall = &fn
		}
		if in[i].CacheControl != nil {
			cc := *in[i].CacheControl
			out[i].CacheControl = &cc
		}
	}
	return out
}

func responsesMessagesBytes(messages []agent.Message) int64 {
	raw, err := json.Marshal(messages)
	if err != nil {
		return 0
	}
	return int64(len(raw))
}

func (s *responsesContinuationStore) Get(id, owner string, now time.Time) ([]agent.Message, bool) {
	if s == nil || id == "" {
		return nil, false
	}
	s.mu.Lock()
	s.expireLocked(now)
	e, ok := s.entries[id]
	if !ok || e.owner != owner || now.Sub(e.storedAt) >= s.cfg.TTL {
		s.mu.Unlock()
		return nil, false
	}
	s.mu.Unlock()
	return cloneResponsesMessages(e.messages), true
}

func (s *responsesContinuationStore) Put(id, owner string, messages []agent.Message, now time.Time) bool {
	if s == nil || id == "" {
		return false
	}
	cloned := cloneResponsesMessages(messages)
	bytes := responsesMessagesBytes(cloned)
	if bytes <= 0 || bytes > s.cfg.MaxBytes {
		return false
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	s.expireLocked(now)
	if _, exists := s.entries[id]; exists {
		return false
	}
	e := responsesContinuationEntry{id: id, owner: owner, messages: cloned, bytes: bytes, storedAt: now}
	s.entries[id] = e
	s.order = append(s.order, id)
	s.bytes += bytes
	for len(s.entries) > s.cfg.MaxEntries || s.bytes > s.cfg.MaxBytes {
		s.evictOldestLocked()
	}
	return true
}

func (s *responsesContinuationStore) Len() int {
	if s == nil {
		return 0
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	s.expireLocked(time.Now())
	return len(s.entries)
}

func (s *responsesContinuationStore) expireLocked(now time.Time) {
	for len(s.order) > 0 {
		e, ok := s.entries[s.order[0]]
		if !ok {
			s.order = s.order[1:]
			continue
		}
		if now.Sub(e.storedAt) < s.cfg.TTL {
			return
		}
		s.evictOldestLocked()
	}
}

func (s *responsesContinuationStore) evictOldestLocked() {
	if len(s.order) == 0 {
		return
	}
	id := s.order[0]
	s.order = s.order[1:]
	if e, ok := s.entries[id]; ok {
		delete(s.entries, id)
		s.bytes -= e.bytes
	}
}
