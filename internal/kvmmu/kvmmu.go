// Package kvmmu bridges logical quarantine and elision decisions into mechanical
// eviction of token K/V spans from kernel-owned attention caches.
package kvmmu

import (
	"context"
	"sync"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/cachemeta"
	"github.com/anthony-chaudhary/fak/internal/ctxmmu"
	"github.com/anthony-chaudhary/fak/internal/ctxplan"
	"github.com/anthony-chaudhary/fak/internal/model"
)

// EvictDisposition records eviction fidelity relative to subsequent prefilled segments.
type EvictDisposition uint8

const (
	// DispositionNone indicates the span is still KV-resident.
	DispositionNone EvictDisposition = iota
	// DispositionNeverSaw indicates the span was evicted before subsequent segments prefilled over it.
	DispositionNeverSaw
	// DispositionCompacted indicates eviction occurred after subsequent segments absorbed attention.
	DispositionCompacted
)

// String renders the disposition for audit/log surfaces.
func (d EvictDisposition) String() string {
	switch d {
	case DispositionNeverSaw:
		return "never-saw"
	case DispositionCompacted:
		return "compacted"
	default:
		return "resident"
	}
}

// Segment tracks contiguous K/V positions of one appended prompt chunk or tool result.
type Segment struct {
	ID            string            // logical id: tool-call id or quarantine id
	Tool          string            // producing tool name for ledger reporting
	From          int               // first absolute cache position
	Len           int               // number of cached positions (0 when evicted)
	Held          bool              // true once segment is evicted from cache
	KV            cachemeta.EntryID // cachemeta identity for derived coherence entries
	Class         string            // write-time durability class (turn/session/bounded/durable)
	ExpiresAtUnix int64             // absolute expiry epoch seconds (0 if disarmed)
	Disposition   EvictDisposition  // eviction fidelity (never-saw vs compacted)
	Attended      float64           // in-flight attended mass in current turn
	Cumulative    float64           // lifetime undecayed attention mass
	EMA           float64           // recency-decayed rolling attention mass
	Pinned        bool              // protected from attention-driven eviction
	traj          []float64         // bounded ring of recent turn masses
	trajHead      int               // next write slot in traj ring
	trajLen       int               // valid entry count in traj ring
	expertHist    *ExpertHist       // MoE expert routing histogram
}

// Gate specifies admission policy decisions enforced against KV cache spans.
type Gate interface {
	Admit(ctx context.Context, c *abi.ToolCall, r *abi.Result) abi.Verdict
}

// FoldedGate enforces the kernel's registered ResultAdmitter chain with most-restrictive fold.
type FoldedGate struct{}

// Admit runs the kernel's registered ResultAdmitter chain over the result and returns
// the most-restrictive-wins folded verdict (the production KV-MMU decision).
func (FoldedGate) Admit(ctx context.Context, c *abi.ToolCall, r *abi.Result) abi.Verdict {
	best := abi.Verdict{Kind: abi.VerdictAllow, By: "default-admit"}
	bestRank := abi.FoldRank(abi.VerdictAllow)
	for _, ra := range abi.ResultAdmittersFor(c) {
		if v := ra.Admit(ctx, c, r); abi.FoldRank(v.Kind) > bestRank {
			bestRank, best = abi.FoldRank(v.Kind), v
		}
	}
	return best
}

// Context tracks contiguous K/V spans in a registered KV backend cache.
// Precondition: caller provides an initialized KV backend and result gate to establish the bridge.
// Postcondition: context is initialized with empty segment ledger and tracks live cache residency.
type Context struct {
	mu          sync.Mutex
	kv          abi.KVBackend
	gate        Gate
	segs        []*Segment
	meta        []cachemeta.Entry
	invalidated []cachemeta.Entry
	external    []cachemeta.ExternalInvalidationDirective

	// lastRetained records the hit-quality gauge of the latest eviction decision.
	lastRetained RetainedMassGauge
}

// New wires the kernel's full registered detector chain over a session's
// kernel-owned KV cache (the production gate). Convenience wrapper: it adapts the
// session into the in-process abi.KVBackend; enforcement runs through that interface.
func New(s *model.Session) *Context { return NewWithGate(s, FoldedGate{}) }

// NewWithGate enforces an explicit gate over the session's cache. Convenience
// wrapper that adapts the *model.Session into the in-process abi.KVBackend so the
// enforcement path is the abi seam, not the concrete session.
func NewWithGate(s *model.Session, gate Gate) *Context {
	kv, _ := model.KVBackend(s) // in-process default; ok=false only for a nil session
	return NewBackendWithGate(kv, gate)
}

// NewBackend wires the full registered detector chain over an already-built
// abi.KVBackend — the pure-abi constructor with no concrete model dependency, so a
// remote/zero-copy KV backend (the disaggregated-agent-memory direction) can be
// enforced the SAME way the in-process session is.
func NewBackend(kv abi.KVBackend) *Context { return NewBackendWithGate(kv, FoldedGate{}) }

// NewBackendWithGate enforces an explicit gate over an already-built abi.KVBackend.
func NewBackendWithGate(kv abi.KVBackend, gate Gate) *Context {
	return &Context{kv: kv, gate: gate}
}

// Append prefills a labeled prompt segment into the KV cache and returns logits and the recorded segment.
// Precondition: token ids sequence must be non-empty and represent a discrete contiguous prompt segment.
// Postcondition: returns non-nil segment record positioned at the prior cache boundary with matching length.
func (c *Context) Append(id, tool string, ids []int) ([]float32, *Segment) {
	from := c.kv.Len()
	seg := &Segment{ID: id, Tool: tool, From: from, Len: len(ids), KV: c.kvEntryID(ids), Pinned: defaultPinned(tool)}
	c.segs = append(c.segs, seg)
	logits := c.kv.Prefill(ids)
	return logits, seg
}

func defaultPinned(tool string) bool {
	return tool == "system"
}

func (c *Context) kvEntryID(ids []int) cachemeta.EntryID {
	modelID := ""
	if c.kv != nil {
		modelID = c.kv.ModelID()
	}
	return cachemeta.FromKVPrefix(cachemeta.KVPrefix{
		Tokens:  ids,
		ModelID: modelID,
		Owner:   "kvmmu",
	}).ID
}

// TrackEntry records a live cache metadata entry whose coherence parents may
// reference a segment K/V span. When a segment is evicted, dependent
// attention_index entries are invalidated with it.
func (c *Context) TrackEntry(e cachemeta.Entry) {
	c.meta = append(c.meta, e)
}

// AdmitResult executes admission evaluation for tool outputs and immediately evicts quarantined spans.
// Precondition: context and tool name must be initialized; body carries raw tool execution bytes.
// Postcondition: quarantined results are immediately evicted and stripped of next-token logits.
func (c *Context) AdmitResult(ctx context.Context, id, tool string, ids []int, body []byte) (v abi.Verdict, evicted bool, logits []float32) {
	logits, seg := c.Append(id, tool, ids)
	call := &abi.ToolCall{Tool: tool}
	res := &abi.Result{
		Call:    call,
		Payload: abi.Ref{Kind: abi.RefInline, Inline: body, Len: int64(len(body))},
	}
	v = c.gate.Admit(ctx, call, res)
	seg.Class = ctxmmu.NormalizeDurabilityClass(v.Meta[ctxmmu.DurabilityKey])
	if v.Kind == abi.VerdictQuarantine {
		c.evict(seg)
		return v, true, nil
	}
	return v, false, logits
}

// Quarantine evicts a recorded segment's span from the cache by segment identifier.
// Precondition: target segment identifier must specify a recorded, non-empty span in the ledger.
// Postcondition: returns evicted position count and true if target was found and successfully removed.
func (c *Context) Quarantine(id string) (evicted int, ok bool) {
	for _, s := range c.segs {
		if s.ID == id && !s.Held {
			return c.evict(s), true
		}
	}
	return 0, false
}

// SetTTL stamps an absolute expiration Unix timestamp on a recorded segment.
// Precondition: id must name an existing segment in the ledger that has not already been evicted.
// Postcondition: returns true if the segment was found and live, updating its expiration timestamp.
func (c *Context) SetTTL(id string, expiresAtUnix int64) bool {
	c.mu.Lock()
	defer c.mu.Unlock()
	for _, s := range c.segs {
		if s.ID == id && !s.Held {
			s.ExpiresAtUnix = expiresAtUnix
			return true
		}
	}
	return false
}

// Expire evicts all recorded segments whose expiration timestamp is at or before nowUnix.
// Precondition: nowUnix is an injected epoch timestamp compared against recorded segment expiration times.
// Postcondition: returns total count of expired segments evicted in reverse position order.
func (c *Context) Expire(nowUnix int64) (evicted int) {
	c.mu.Lock()
	defer c.mu.Unlock()
	for i := len(c.segs) - 1; i >= 0; i-- {
		s := c.segs[i]
		if s.Held {
			continue
		}
		if s.ExpiresAtUnix <= 0 || s.ExpiresAtUnix > nowUnix {
			continue
		}
		c.evict(s)
		if s.Held {
			evicted++
		}
	}
	return evicted
}

// Compact swaps candidate segments for a single verified summary segment if admission permits.
// Precondition: summary tokens and payload bytes must represent a coherent candidate replacement span.
// Postcondition: quarantined summaries leave existing segments untouched; allowed summaries swap all IDs.
func (c *Context) Compact(ctx context.Context, ids []string, summaryID, tool string, summaryIDs []int, summaryBody []byte) (v abi.Verdict, swapped int) {
	call := &abi.ToolCall{Tool: tool}
	res := &abi.Result{
		Call:    call,
		Payload: abi.Ref{Kind: abi.RefInline, Inline: summaryBody, Len: int64(len(summaryBody))},
	}
	v = c.gate.Admit(ctx, call, res)
	if v.Kind == abi.VerdictQuarantine {
		return v, 0 // poisoned summary -> refuse the swap; keep the spans, drop nothing
	}
	want := make(map[string]struct{}, len(ids))
	for _, id := range ids {
		want[id] = struct{}{}
	}
	for _, s := range c.segs {
		if s.Held {
			continue
		}
		if _, ok := want[s.ID]; !ok {
			continue
		}
		c.evict(s)
		swapped++
	}
	c.Append(summaryID, tool, summaryIDs)
	return v, swapped
}

// ApplyPlan executes elisions specified in plan against the resident KV cache.
// Precondition: plan specifies candidate elision and selection sets with stable segment identifiers.
// Postcondition: returns number of elided segments evicted to compress KV cache to the resident view.
func (c *Context) ApplyPlan(plan ctxplan.Plan) (evicted int) {
	before, cost := c.liveMassLedger()
	defer c.gradeRetention(before, cost)

	elide := make(map[string]bool, len(plan.Elided))
	for _, e := range plan.Elided {
		if e.ID != "" {
			elide[e.ID] = true
		}
	}
	// A selected span stays resident even if a malformed plan also lists it as
	// elided (faithful.go's disjointness check already forbids the overlap).
	for _, s := range plan.Selected {
		delete(elide, s.ID)
	}
	for _, seg := range c.segs {
		if seg.Held {
			continue
		}
		if elide[seg.ID] {
			c.evict(seg)
			evicted++
		}
	}
	return evicted
}

// evict drops a segment's span from the kernel-owned cache and renumbers the
// ledger so every segment after it shifts down by the evicted length — keeping
// the ledger consistent with model.KVCache.Evict's own compaction.
func (c *Context) evict(seg *Segment) int {
	if ce, ok := c.kv.(interface{ CanEvict() error }); ok && ce.CanEvict() != nil {
		return 0
	}
	n := c.kv.Evict(seg.From, seg.Len)
	c.external = append(c.external, cachemeta.PlanExternalInvalidations(seg.KV, c.meta)...)
	c.invalidateReferences(seg.KV)

	neverSaw := true
	for _, s := range c.segs {
		if s != seg && !s.Held && s.From > seg.From {
			neverSaw = false
			break
		}
	}
	for _, s := range c.segs {
		if s != seg && s.From > seg.From {
			s.From -= seg.Len
		}
	}
	seg.Held = true
	seg.Len = 0
	if neverSaw {
		seg.Disposition = DispositionNeverSaw
	} else {
		seg.Disposition = DispositionCompacted
	}
	seg.Attended = 0
	seg.Cumulative = 0
	seg.EMA = 0
	seg.traj = seg.traj[:0]
	seg.trajHead = 0
	seg.trajLen = 0
	seg.expertHist = nil
	return n
}

func (c *Context) invalidateReferences(kv cachemeta.EntryID) {
	if !kv.Valid() || len(c.meta) == 0 {
		return
	}
	live := c.meta[:0]
	for _, e := range c.meta {
		if cachemeta.AttentionIndexReferences(e, kv) || cachemeta.ExternalSelfReference(e, kv) {
			c.invalidated = append(c.invalidated, e)
			continue
		}
		live = append(live, e)
	}
	c.meta = live
}

// SegmentByID returns the recorded segment with the given id, or (nil, false)
// if not found.
func (c *Context) SegmentByID(id string) (*Segment, bool) {
	for _, s := range c.segs {
		if s.ID == id {
			return s, true
		}
	}
	return nil, false
}

// LivePositions returns the total number of live (un-evicted) positions across
// all recorded segments in the ledger.
func (c *Context) LivePositions() int {
	total := 0
	for _, s := range c.segs {
		if !s.Held {
			total += s.Len
		}
	}
	return total
}

// Segments returns the current ledger (a copy of the slice header is fine; the
// caller should treat the entries as read-only).
func (c *Context) Segments() []*Segment { return c.segs }

// Pin marks a live segment as protected from budget-driven attention eviction.
func (c *Context) Pin(id string, pinned bool) bool {
	for _, s := range c.segs {
		if s.ID == id && !s.Held {
			s.Pinned = pinned
			return true
		}
	}
	return false
}

// Entries returns currently live cache metadata entries tracked by this bridge.
func (c *Context) Entries() []cachemeta.Entry { return append([]cachemeta.Entry(nil), c.meta...) }

// InvalidatedEntries returns entries invalidated by K/V segment eviction.
func (c *Context) InvalidatedEntries() []cachemeta.Entry {
	return append([]cachemeta.Entry(nil), c.invalidated...)
}

// ExternalInvalidations returns remote-engine invalidation directives derived
// from evicted K/V spans and tracked cache metadata.
func (c *Context) ExternalInvalidations() []cachemeta.ExternalInvalidationDirective {
	return append([]cachemeta.ExternalInvalidationDirective(nil), c.external...)
}

// CacheLen is the number of live K/V positions in the enforced KV backend's cache.
func (c *Context) CacheLen() int { return c.kv.Len() }

// Evicted counts the segments whose K/V has been evicted from the cache — the
// KV-level analogue of ctxmmu's quarantine counter, derived from the ledger so it
// is independent of which Gate made the decision.
func (c *Context) Evicted() int {
	n := 0
	for _, s := range c.segs {
		if s.Held {
			n++
		}
	}
	return n
}
