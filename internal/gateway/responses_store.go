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
	// response is the rendered response body GET /v1/responses/{id} returns; nil
	// when the entry was stored without one.
	response []byte
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
	return s.PutResponse(id, owner, messages, nil, now)
}

// PutResponse stores the continuation messages together with the rendered
// response snapshot; both count against the store's byte cap.
func (s *responsesContinuationStore) PutResponse(id, owner string, messages []agent.Message, response []byte, now time.Time) bool {
	if s == nil || id == "" {
		return false
	}
	cloned := cloneResponsesMessages(messages)
	bytes := responsesMessagesBytes(cloned)
	if bytes <= 0 || bytes > s.cfg.MaxBytes {
		return false
	}
	// An oversized snapshot is dropped rather than the entry, so a response that
	// was continuable before snapshots existed stays continuable.
	var snapshot []byte
	if bytes+int64(len(response)) <= s.cfg.MaxBytes {
		snapshot = append([]byte(nil), response...)
		bytes += int64(len(snapshot))
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	s.expireLocked(now)
	if _, exists := s.entries[id]; exists {
		return false
	}
	e := responsesContinuationEntry{id: id, owner: owner, messages: cloned, response: snapshot, bytes: bytes, storedAt: now}
	s.entries[id] = e
	s.order = append(s.order, id)
	s.bytes += bytes
	for len(s.entries) > s.cfg.MaxEntries || s.bytes > s.cfg.MaxBytes {
		s.evictOldestLocked()
	}
	return true
}

// Snapshot returns a copy of the stored response body. An entry stored without
// one is reported as absent, the same as an unknown, expired or foreign id.
func (s *responsesContinuationStore) Snapshot(id, owner string, now time.Time) ([]byte, bool) {
	if s == nil || id == "" {
		return nil, false
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	s.expireLocked(now)
	e, ok := s.entries[id]
	if !ok || e.owner != owner || now.Sub(e.storedAt) >= s.cfg.TTL || len(e.response) == 0 {
		return nil, false
	}
	return append([]byte(nil), e.response...), true
}

// Delete removes an owned, unexpired entry and reports whether it did.
func (s *responsesContinuationStore) Delete(id, owner string, now time.Time) bool {
	if s == nil || id == "" {
		return false
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	s.expireLocked(now)
	e, ok := s.entries[id]
	if !ok || e.owner != owner || now.Sub(e.storedAt) >= s.cfg.TTL {
		return false
	}
	delete(s.entries, id)
	s.bytes -= e.bytes
	for i, queued := range s.order {
		if queued == id {
			s.order = append(s.order[:i], s.order[i+1:]...)
			break
		}
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
