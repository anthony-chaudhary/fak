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

// Deadline cache credit for proxied chat from engine-owned residency evidence.
//
// After a chat turn is served by an OpenAI-compatible llama-server upstream with
// exactly one slot, the gateway reads that upstream's GET /slots. When the slot
// is idle and its last task's n_prompt_tokens equals the prompt the turn
// reported, that task id is remembered with the turn's message prefix. A later
// turn is credited with that prefix only when all of these hold at admission:
//
//   - it is admitted onto the same planner and model and extends exactly the
//     same message prefix (chained role+content digest);
//   - the upstream's slot is idle and still reports the remembered task id and
//     prompt length, so llama-server itself proves no request from any client
//     has used the slot since, and the prefix is still resident;
//   - no chat request is in flight here, and the evidence is younger than
//     deadlineResidencyTTL;
//   - the cold estimate stays within deadlineCacheCreditMaxColdOverrun of the
//     remaining budget, bounding the cost if a request reaches the slot first.
//
// A credited turn whose engine reported less than half the credit as cached
// clears every entry. Upstreams that do not expose a single-slot /slots, and
// non-HTTP planners, get no credit.

const (
	deadlineResidencyTTL        = 10 * time.Minute
	deadlineResidencyMaxEntries = 1024
	deadlineSlotReadTimeout     = 250 * time.Millisecond
)

type deadlineResidencyMemory struct {
	mu      sync.Mutex
	entries map[[sha256.Size]byte]deadlineResidencyEntry
}

type deadlineResidencyEntry struct {
	tokens     int
	model      string
	planner    *agent.HTTPPlanner
	engineTask int
	slotPrompt int
	at         time.Time
}

// deadlineResidencyTicket follows one admitted chat request from admission to
// release.
type deadlineResidencyTicket struct {
	key      [sha256.Size]byte
	estTok   int
	model    string
	planner  *agent.HTTPPlanner
	credited int

	mu       sync.Mutex
	measured int
}

type deadlineResidencyCtxKey struct{}

func (m *gatewayMetrics) deadlineResidencyMem() *deadlineResidencyMemory {
	if m == nil {
		return nil
	}
	m.deadlineResidencyOnce.Do(func() {
		m.deadlineResidency = &deadlineResidencyMemory{entries: map[[sha256.Size]byte]deadlineResidencyEntry{}}
	})
	return m.deadlineResidency
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

// lookup returns the longest unexpired remembered prefix of chain for model on
// planner.
func (p *deadlineResidencyMemory) lookup(chain [][sha256.Size]byte, model string, planner *agent.HTTPPlanner, now time.Time) (deadlineResidencyEntry, bool) {
	if p == nil || planner == nil {
		return deadlineResidencyEntry{}, false
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	for i := len(chain) - 1; i >= 0; i-- {
		e, ok := p.entries[chain[i]]
		if !ok {
			continue
		}
		if now.Sub(e.at) > deadlineResidencyTTL {
			delete(p.entries, chain[i])
			continue
		}
		if e.model == model && e.planner == planner {
			return e, true
		}
	}
	return deadlineResidencyEntry{}, false
}

func (p *deadlineResidencyMemory) remember(key [sha256.Size]byte, e deadlineResidencyEntry) {
	if p == nil || e.tokens <= 0 {
		return
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	if _, ok := p.entries[key]; !ok && len(p.entries) >= deadlineResidencyMaxEntries {
		var oldestKey [sha256.Size]byte
		var oldest time.Time
		first := true
		for k, old := range p.entries {
			if first || old.at.Before(oldest) {
				oldestKey, oldest, first = k, old.at, false
			}
		}
		delete(p.entries, oldestKey)
	}
	p.entries[key] = e
}

func (p *deadlineResidencyMemory) forgetAll() {
	if p == nil {
		return
	}
	p.mu.Lock()
	clear(p.entries)
	p.mu.Unlock()
}

// deadlineResidencyCredit returns the engine-proven resident prefix for a chat
// request, or 0.
func (s *Server) deadlineResidencyCredit(r *http.Request, t *deadlineResidencyTicket, chain [][sha256.Size]byte, arrived time.Time, maxTokens int) int {
	est := s.metrics.deadlineEstimator()
	budget, has := deadlineadmit.Budget(r.Header)
	if est == nil || !has || t.planner == nil || est.InFlight() != 0 {
		return 0
	}
	e, ok := s.metrics.deadlineResidencyMem().lookup(chain, t.model, t.planner, arrived)
	if !ok {
		return 0
	}
	slot, ok := t.planner.LlamaSingleSlot(r.Context(), deadlineSlotReadTimeout)
	if !ok || slot.Processing || slot.IDTask != e.engineTask || slot.PromptTokens != e.slotPrompt {
		return 0
	}
	remaining := budget - time.Since(arrived)
	if cold := est.AdmitCached(t.estTok, 0, maxTokens, remaining, true); cold.Estimate.Seconds() > remaining.Seconds()*(1+deadlineCacheCreditMaxColdOverrun) {
		return 0
	}
	return min(e.tokens, slot.PromptTokens)
}

// noteDeadlineResidency binds a served chat Completion to the admitted request
// in ctx. Call it after the metrics observation of the same Completion.
func (s *Server) noteDeadlineResidency(ctx context.Context, comp *agent.Completion) {
	t, _ := ctx.Value(deadlineResidencyCtxKey{}).(*deadlineResidencyTicket)
	if t == nil || comp == nil {
		return
	}
	hit := max(comp.Usage.CachedPromptTokens(), perfDetailFromCompletion(comp).kvCacheN)
	if t.credited > 0 && hit*2 < t.credited {
		s.metrics.deadlineResidencyMem().forgetAll()
		s.logf("gateway: deadline residency mispredicted: credited=%d engine_cached=%d; cleared residency evidence", t.credited, hit)
	}
	t.mu.Lock()
	t.measured = comp.Usage.PromptTokens
	t.mu.Unlock()
}

// releaseDeadlineResidency records the request's prefix when its turn was
// served, finished with a healthy context, and the upstream's only slot holds
// exactly that prompt. The /slots read runs off the request path.
func (s *Server) releaseDeadlineResidency(ctx context.Context, t *deadlineResidencyTicket) {
	if t == nil || t.planner == nil || ctx.Err() != nil {
		return
	}
	t.mu.Lock()
	measured := t.measured
	t.mu.Unlock()
	if measured <= 0 {
		return
	}
	mem := s.metrics.deadlineResidencyMem()
	go func() {
		slot, ok := t.planner.LlamaSingleSlot(context.Background(), deadlineSlotReadTimeout)
		if !ok || slot.Processing || slot.PromptTokens != measured {
			return
		}
		mem.remember(t.key, deadlineResidencyEntry{
			tokens:     min(t.estTok, measured),
			model:      t.model,
			planner:    t.planner,
			engineTask: slot.IDTask,
			slotPrompt: slot.PromptTokens,
			at:         time.Now(),
		})
	}()
}
