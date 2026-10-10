package gateway

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"hash/fnv"
	"sync"
	"sync/atomic"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

// Deadline cache credit: a chat turn may be priced with part of its prompt
// cached only when this node holds served evidence for that exact prefix.
//
// Evidence is a completed turn of the same conversation (same model, tools,
// system text and first user turn) whose upstream usage reported a nonzero
// prompt-cache hit. Its message chain is recorded after the response, never at
// admission or release, so a refused, canceled or failed turn proves nothing.
// The next turn earns credit only for the leading messages whose fingerprints
// equal the recorded chain, capped at the resident tokens the upstream
// reported. Every served turn overwrites the record, so a turn that reports a
// cache miss (KV evicted, route changed) removes the credit for the one after.
//
// A wrong credit is bounded: credit is withdrawn when the cold estimate exceeds
// the admission ceiling by more than deadlineCacheCreditMaxColdOverrun, and the
// admitted request still carries the client deadline in its context, so a
// mispredicted turn is canceled at the deadline instead of running past it.

const (
	// deadlineCacheCreditMaxColdOverrun bounds the cold estimate of a credited
	// request at (1+overrun) times the admission ceiling (Headroom*remaining).
	deadlineCacheCreditMaxColdOverrun = 0.5
	deadlineCacheCreditTTL            = 10 * time.Minute
	deadlineCacheCreditMaxConvs       = 4096
)

// Credit outcomes, the closed vocabulary of the admission log field.
const (
	cacheCreditNone      = "none"       // no record for this conversation
	cacheCreditStale     = "stale"      // record older than the TTL
	cacheCreditUncached  = "uncached"   // recorded turn reported no cache hit
	cacheCreditMismatch  = "mismatch"   // no leading message matches the record
	cacheCreditCredited  = "credited"   // verified prefix credited
	cacheCreditOverrun   = "overrun"    // credit withdrawn by the cold-overrun bound
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

// deadlineCacheTicket travels in the request context from admission to the
// served response.
type deadlineCacheTicket struct {
	key      uint64
	keyed    bool
	chain    []uint64
	cumTok   []int
	credited int
}

type deadlineCacheTicketKey struct{}

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

// lookup returns the credited tokens for the ticket's verified common prefix.
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

// recordDeadlineWarmPrefix records a successfully served chat turn as cache
// evidence for the conversation's next admission.
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

// DeadlineCacheCreditStats is the observable deadline cache-credit roll-up.
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
