package gateway

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"hash/fnv"
	"net/http"
	"sync"
	"sync/atomic"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

// Deadline cache history is observational, not current residency evidence.
// An accepted buffered completion records provider-reported cache usage for its message
// chain before the final socket write; delivery is not established. The ledger
// has no selected-route lease or eviction notification, so
// its lookup must never discount a served request's deadline admission.
// Native admission uses its execution-owned prefix; other routes stay cold.
// Retain the historical estimator and counters for explicit observation and
// provenance. A served cache miss replaces the record, but cannot retroactively
// establish whether a previous record was resident when a request arrived.

const (
	// deadlineCacheCreditMaxColdOverrun bounds measured residency credit
	// (deadline_residency.go); historical observations here never credit.
	deadlineCacheCreditMaxColdOverrun = 0.5
	deadlineCacheCreditTTL            = 10 * time.Minute
	deadlineCacheCreditMaxConvs       = 4096
)

// Historical lookup outcomes retain the previous vocabulary for compatibility.
const (
	cacheCreditNone      = "none"       // no record for this conversation
	cacheCreditStale     = "stale"      // record older than the TTL
	cacheCreditUncached  = "uncached"   // recorded turn reported no cache hit
	cacheCreditMismatch  = "mismatch"   // no leading message matches the record
	cacheCreditCredited  = "credited"   // historical match, not current residency
	cacheCreditOverrun   = "overrun"    // legacy bounded-credit outcome
	cacheCreditNoMessage = "no_message" // no conversation key (no user turn)
)

type warmPrefixRecord struct {
	chain       []uint64 // chain[i] fingerprints messages[0..i]
	cumTok      []int    // cumTok[i] estimated tokens of messages[0..i]
	residentTok int      // upstream-reported prompt + completion tokens
	cachedTok   int      // upstream-reported cached prompt tokens
	at          time.Time
}

type warmPrefixLedger struct {
	mu      sync.Mutex
	entries map[uint64]warmPrefixRecord

	creditedAdmits atomic.Uint64
	creditedTokens atomic.Uint64
	overrunDenied  atomic.Uint64
	mispredicts    atomic.Uint64
}

// deadlineCacheTicket travels from the original request to an accepted buffered
// completion. Live streaming returns before this observational recording hook.
type deadlineCacheTicket struct {
	key      uint64
	keyed    bool
	chain    []uint64
	cumTok   []int
	credited int
}

type deadlineCacheTicketKey struct{}

// withDeadlineCacheObservation snapshots the untouched request for a later
// served-usage observation. It neither looks up credit nor reserves admission.
func withDeadlineCacheObservation(r *http.Request, model string, tools []agent.ToolDef, messages []agent.Message) *http.Request {
	ticket := newDeadlineCacheTicket(model, tools, messages)
	return r.WithContext(context.WithValue(r.Context(), deadlineCacheTicketKey{}, ticket))
}

func (m *gatewayMetrics) warmPrefixLedger() *warmPrefixLedger {
	if m == nil {
		return nil
	}
	m.warmPrefixOnce.Do(func() {
		if m.warmPrefix == nil {
			m.warmPrefix = &warmPrefixLedger{entries: map[uint64]warmPrefixRecord{}}
		}
	})
	return m.warmPrefix
}

// newDeadlineCacheTicket fingerprints a request's messages. The conversation
// key binds model, tools, system/developer text and the first user turn.
func newDeadlineCacheTicket(model string, tools []agent.ToolDef, messages []agent.Message) *deadlineCacheTicket {
	t := &deadlineCacheTicket{chain: make([]uint64, len(messages)), cumTok: make([]int, len(messages))}
	kh := fnv.New64a()
	_, _ = kh.Write([]byte(model))
	if b, err := json.Marshal(tools); err == nil {
		_, _ = kh.Write(b)
	}
	var prev uint64
	tok := 0
	for i, m := range messages {
		b, err := json.Marshal(m)
		if err != nil {
			return &deadlineCacheTicket{}
		}
		h := fnv.New64a()
		var buf [8]byte
		binary.LittleEndian.PutUint64(buf[:], prev)
		_, _ = h.Write(buf[:])
		_, _ = h.Write(b)
		prev = h.Sum64()
		t.chain[i] = prev
		if m.Content != "" {
			tok += (len(m.Content) + 3) / 4
		}
		t.cumTok[i] = tok
		if !t.keyed {
			_, _ = kh.Write(b)
			if m.Role == agent.RoleUser {
				t.keyed = true
			}
		}
	}
	t.key = kh.Sum64()
	return t
}

// lookup returns a historical prefix estimate, not an admission credit.
func (l *warmPrefixLedger) lookup(t *deadlineCacheTicket, now time.Time) (int, string) {
	if l == nil || t == nil || !t.keyed {
		return 0, cacheCreditNoMessage
	}
	l.mu.Lock()
	rec, ok := l.entries[t.key]
	l.mu.Unlock()
	switch {
	case !ok:
		return 0, cacheCreditNone
	case now.Sub(rec.at) > deadlineCacheCreditTTL:
		return 0, cacheCreditStale
	case rec.cachedTok <= 0:
		return 0, cacheCreditUncached
	}
	n := 0
	for n < len(rec.chain) && n < len(t.chain) && rec.chain[n] == t.chain[n] {
		n++
	}
	if n == 0 {
		return 0, cacheCreditMismatch
	}
	credit := t.cumTok[n-1]
	if credit > rec.residentTok {
		credit = rec.residentTok
	}
	if credit <= 0 {
		return 0, cacheCreditMismatch
	}
	return credit, cacheCreditCredited
}

// record stores a served turn's chain with its upstream cache evidence.
func (l *warmPrefixLedger) record(t *deadlineCacheTicket, usage agent.Usage, now time.Time) {
	if l == nil || t == nil || !t.keyed || len(t.chain) == 0 {
		return
	}
	cached := usage.CachedPromptTokens()
	if t.credited > 0 && cached < t.credited/2 {
		l.mispredicts.Add(1)
	}
	resident := cached + usage.UncachedPromptTokens() + usage.CompletionTokens
	l.mu.Lock()
	defer l.mu.Unlock()
	l.entries[t.key] = warmPrefixRecord{chain: t.chain, cumTok: t.cumTok, residentTok: resident, cachedTok: cached, at: now}
	if len(l.entries) > deadlineCacheCreditMaxConvs {
		var oldest uint64
		var oldestAt time.Time
		for k, e := range l.entries {
			if oldestAt.IsZero() || e.at.Before(oldestAt) {
				oldest, oldestAt = k, e.at
			}
		}
		delete(l.entries, oldest)
	}
}

// recordDeadlineWarmPrefix records reported cache usage after accepted buffered
// generation, before the final socket write. It does not prove delivery.
// The observation never establishes residency for the next admission.
func (s *Server) recordDeadlineWarmPrefix(ctx context.Context, usage agent.Usage) {
	if s == nil || ctx == nil {
		return
	}
	t, _ := ctx.Value(deadlineCacheTicketKey{}).(*deadlineCacheTicket)
	if t == nil {
		return
	}
	s.metrics.warmPrefixLedger().record(t, usage, time.Now())
}

// DeadlineCacheCreditStats retains the historical cache-credit metric schema.
// Observational-only served routes do not increment credited admissions/tokens.
type DeadlineCacheCreditStats struct {
	CreditedAdmits uint64 `json:"credited_admits"`
	CreditedTokens uint64 `json:"credited_tokens"`
	OverrunDenied  uint64 `json:"overrun_denied"`
	Mispredicts    uint64 `json:"mispredicts"`
}

func (l *warmPrefixLedger) stats() DeadlineCacheCreditStats {
	if l == nil {
		return DeadlineCacheCreditStats{}
	}
	return DeadlineCacheCreditStats{
		CreditedAdmits: l.creditedAdmits.Load(),
		CreditedTokens: l.creditedTokens.Load(),
		OverrunDenied:  l.overrunDenied.Load(),
		Mispredicts:    l.mispredicts.Load(),
	}
}
