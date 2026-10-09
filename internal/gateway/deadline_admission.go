package gateway

import (
	"context"
	"crypto/sha256"
	"net/http"
	"sync"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/pkg/deadlineadmit"
)

// Deadline-aware admission for served chat requests.
//
// A client that declares a time budget (X-Fak-Deadline-Ms, X-Request-Timeout,
// or the X-Stainless-Timeout the OpenAI/Anthropic SDKs send) gets two
// guarantees from this node:
//
//  1. The request's context carries that deadline, so prefill, decode, and
//     any proxied upstream call (managed llama-server included) are canceled
//     when it passes, even if the client keeps its socket open.
//  2. A request whose completion estimate, from the measured per-request
//     prefill and decode throughput rescaled to the live in-flight count,
//     exceeds the remaining budget is refused up front with a typed 503 +
//     Retry-After (pkg/deadlineadmit wire contract), so a router fails over
//     instead of the node computing tokens nobody will read.
//
// A request with no declared budget is admitted unchanged, and a node that
// has not measured any throughput yet admits everything.
//
// The estimate charges prefill only for the prompt this node is expected to
// re-ingest. A multi-turn agent resends its whole history each turn, and the
// prefix this node already served is still resident in its KV cache, so a
// 50k-token turn with a 48k resident prefix prefills ~2k tokens, not 50k.
// Both sides use that currency: observation measures the prefill rate over
// the uncached tokens only, and admission predicts the resident prefix from
// deadlinePrefixMemory. A prompt with no resident-prefix evidence is
// estimated cold, which keeps the protection for genuinely huge cold prompts.

// deadlineEstimator returns the node-wide estimator, creating it on first use
// so a directly constructed gatewayMetrics works too.
func (m *gatewayMetrics) deadlineEstimator() *deadlineadmit.Estimator {
	if m == nil {
		return nil
	}
	m.deadlineOnce.Do(func() {
		if m.deadlineEst == nil {
			m.deadlineEst = deadlineadmit.NewEstimator(deadlineadmit.Config{}, nil)
		}
	})
	return m.deadlineEst
}

// observeDeadlineTiming feeds one finished turn into the estimator. promptTok
// is the full resident prompt and cachedTok the part of it served from the KV
// cache. ttft <= 0 (a buffered reply) is recorded as decode over the whole
// duration.
func (m *gatewayMetrics) observeDeadlineTiming(promptTok, cachedTok, complTok int, dur, ttft time.Duration) {
	est := m.deadlineEstimator()
	if est == nil || dur <= 0 {
		return
	}
	o := deadlineadmit.Observation{PromptTokens: promptTok, CachedTokens: cachedTok, CompletionTokens: complTok, Decode: dur}
	if ttft > 0 && ttft < dur {
		o.Prefill = ttft
		o.Decode = dur - ttft
	}
	est.Observe(o)
}

// admitClientDeadline runs deadline-aware admission for one served request.
// On refusal it has already written the typed 503 and returns ok=false. On
// admission it returns the request (carrying the client deadline when one was
// declared) and a release func the caller must defer.
func (s *Server) admitClientDeadline(w http.ResponseWriter, r *http.Request, arrived time.Time, promptTok, maxTokens int) (*http.Request, func(), bool) {
	return s.admitClientDeadlineCached(w, r, arrived, promptTok, 0, maxTokens)
}

// admitClientDeadlineMessages is admitClientDeadline for a chat request: it
// predicts the resident KV prefix from the conversations this node recently
// finished, and on a clean finish records this request's messages as resident
// for the next turn.
func (s *Server) admitClientDeadlineMessages(w http.ResponseWriter, r *http.Request, arrived time.Time, messages []agent.Message, maxTokens int) (*http.Request, func(), bool) {
	mem := s.metrics.deadlinePrefixes()
	chain := deadlinePrefixChain(messages)
	promptTok := estimateMessageContentTokens(messages)
	cached := mem.residentTokens(chain, arrived)
	r, release, ok := s.admitClientDeadlineCached(w, r, arrived, promptTok, cached, maxTokens)
	if !ok {
		return r, release, false
	}
	ctx := r.Context()
	return r, func() {
		if ctx.Err() == nil && len(chain) > 0 {
			mem.remember(chain[len(chain)-1], promptTok, time.Now())
		}
		release()
	}, true
}

func (s *Server) admitClientDeadlineCached(w http.ResponseWriter, r *http.Request, arrived time.Time, promptTok, cachedTok, maxTokens int) (*http.Request, func(), bool) {
	est := s.metrics.deadlineEstimator()
	if est == nil {
		return r, func() {}, true
	}
	budget, has := deadlineadmit.Budget(r.Header)
	remaining := budget - time.Since(arrived)
	v := est.AdmitCached(promptTok, cachedTok, maxTokens, remaining, has)
	if !v.Admit {
		s.logf("gateway: deadline admission refused: reason=%s estimate=%s remaining=%s in_flight=%d prompt_tokens=%d predicted_cached=%d",
			v.Reason, v.Estimate, v.Remaining, est.InFlight(), promptTok, cachedTok)
		deadlineadmit.WriteRefusal(w, v)
		return r, nil, false
	}
	end := est.Begin()
	if !has {
		return r, end, true
	}
	ctx, cancel := context.WithDeadline(r.Context(), arrived.Add(budget))
	return r.WithContext(ctx), func() { end(); cancel() }, true
}

// deadlinePrefixMemory remembers, per node, the message prefixes of chat
// requests this node finished recently, so admission can predict how much of a
// new turn's prompt is already resident in the KV cache. It is keyed by a
// chained digest of the leading messages (role + content), so a hit means the
// new request starts with exactly a conversation this node served. The memory
// is bounded and entries expire, because an engine slot evicted by other
// traffic stops being resident; a miss falls back to the cold estimate.
type deadlinePrefixMemory struct {
	mu      sync.Mutex
	entries map[[sha256.Size]byte]deadlinePrefixEntry
}

type deadlinePrefixEntry struct {
	tokens int
	at     time.Time
}

const (
	deadlinePrefixTTL        = 10 * time.Minute
	deadlinePrefixMaxEntries = 1024
)

func (m *gatewayMetrics) deadlinePrefixes() *deadlinePrefixMemory {
	if m == nil {
		return nil
	}
	m.deadlinePrefixOnce.Do(func() {
		m.deadlinePrefix = &deadlinePrefixMemory{entries: map[[sha256.Size]byte]deadlinePrefixEntry{}}
	})
	return m.deadlinePrefix
}

// deadlinePrefixChain returns one digest per leading-message prefix:
// chain[i] identifies messages[0..i].
func deadlinePrefixChain(messages []agent.Message) [][sha256.Size]byte {
	chain := make([][sha256.Size]byte, len(messages))
	var prev [sha256.Size]byte
	for i, msg := range messages {
		h := sha256.New()
		h.Write(prev[:])
		h.Write([]byte(msg.Role))
		h.Write([]byte{0})
		h.Write([]byte(msg.Content))
		h.Sum(prev[:0])
		chain[i] = prev
	}
	return chain
}

// residentTokens returns the token size of the longest remembered, unexpired
// prefix of chain, or 0 when none is remembered.
func (p *deadlinePrefixMemory) residentTokens(chain [][sha256.Size]byte, now time.Time) int {
	if p == nil || len(chain) == 0 {
		return 0
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	for i := len(chain) - 1; i >= 0; i-- {
		e, ok := p.entries[chain[i]]
		if !ok {
			continue
		}
		if now.Sub(e.at) > deadlinePrefixTTL {
			delete(p.entries, chain[i])
			continue
		}
		return e.tokens
	}
	return 0
}

func (p *deadlinePrefixMemory) remember(key [sha256.Size]byte, tokens int, now time.Time) {
	if p == nil || tokens <= 0 {
		return
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	if _, ok := p.entries[key]; !ok && len(p.entries) >= deadlinePrefixMaxEntries {
		var oldestKey [sha256.Size]byte
		var oldest time.Time
		first := true
		for k, e := range p.entries {
			if first || e.at.Before(oldest) {
				oldestKey, oldest, first = k, e.at, false
			}
		}
		delete(p.entries, oldestKey)
	}
	p.entries[key] = deadlinePrefixEntry{tokens: tokens, at: now}
}
