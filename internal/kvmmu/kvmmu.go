// Package kvmmu is the bridge the in-kernel model makes possible: it turns
// ctxmmu's LOGICAL quarantine verdict — "these bytes may not enter the context
// window" — into a MECHANICAL one — eviction of that result's K/V span from the
// kernel-owned attention cache, so the model physically cannot attend to it.
//
// Why this is the deepest expression of the kernel thesis. The shipped ctxmmu
// gate (internal/ctxmmu) is a *content* defense over a serving boundary: when a
// tool result is poisoned it rewrites the result BYTES to a stub, so the poison
// never reaches the next prompt the model is shown. That is real and structural
// at the text layer — but it presumes the model lives behind an HTTP boundary and
// is re-prompted from text each turn. Once the model runs INSIDE the kernel
// (internal/model: the KV cache is a kernel-owned Go data structure), the same
// quarantine *decision* can be enforced one layer deeper: evict the offending
// span's K/V from every layer of the attention cache. The model is no longer
// "not shown" the poison; it is mechanically incapable of attending to it, and —
// because model.KVCache.Evict re-RoPEs and renumbers the survivors — the cache is
// byte-identical to a run that never saw the span (proven token-for-token vs
// HuggingFace by internal/model's rung-3 oracle test).
//
// kvmmu is the seam that was named-but-unbuilt in IN-KERNEL-MODEL-RESULTS.md
// ("the bridge is to have that same verdict call Session.Cache.Evict"). It is a
// pure CONSUMER of two shipped primitives — ctxmmu.MMU.Admit (the decision) and
// model.KVCache.Evict (the enforcement) — joined by a small span LEDGER that
// records which cache positions each segment occupies, so the verdict on a
// segment evicts exactly its span and the survivors' recorded spans are
// renumbered to match the compacted cache. Derived cache metadata that parents
// a span, such as GLM DSA attention_index entries, is invalidated with the span.
// It adds NOTHING to the frozen ABI (no abi.Register*; it imports abi only to
// read the verdict Kind).
//
// One decision, two enforcement media: ctxmmu bars the bytes from the text
// context; kvmmu bars the K/V from the attention state. The detector is shared
// and therefore so is its known ceiling — kvmmu makes the *enforcement* deeper,
// not the *detection* better (see RECALL-RESULTS.md on the evadable content
// gate). The contribution is that the trust floor now reaches the model's own
// working memory, model-independently.
package kvmmu

import (
	"context"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/cachemeta"
	"github.com/anthony-chaudhary/fak/internal/model"
)

// Segment is the contiguous K/V span one appended unit (a prompt chunk or a tool
// result) occupies in the session's attention cache.
type Segment struct {
	ID   string            // logical id: the tool-call id, or the quarantine id
	Tool string            // the tool that produced it (for the ledger / reporting)
	From int               // first absolute cache position (renumbered after an eviction)
	Len  int               // number of cached positions; 0 once evicted
	Held bool              // true once this segment's K/V has been evicted from the cache
	KV   cachemeta.EntryID // cachemeta identity for derived entries that parent this span
}

// Gate is the decision the bridge ENFORCES on the KV cache. It answers, for a
// produced result, whether the bytes may enter context (VerdictAllow), must be
// held out (VerdictQuarantine), or paged (VerdictTransform). A single
// ctxmmu.MMU satisfies it; so does the kernel's full ResultAdmitter fold
// (FoldedGate). Decoupling the decision from the enforcement is the whole point:
// kvmmu enforces WHATEVER the harness's detector chain decides, deeper — on the
// attention state — so a better detector (e.g. internal/normgate) makes the
// KV-level quarantine better with no edit here.
type Gate interface {
	Admit(ctx context.Context, c *abi.ToolCall, r *abi.Result) abi.Verdict
}

// FoldedGate enforces the kernel's REGISTERED ResultAdmitter chain, folded
// most-restrictive-wins exactly as kernel.admitResult does (kernel.go). Using it
// means the KV-MMU inherits every detection driver the fleet ships — normgate
// (rank 5) in front of ctxmmu (rank 10), and anything added later — with zero
// change here. This is the production gate.
type FoldedGate struct{}

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

// Context tracks the K/V span each segment occupies in a model.Session's cache,
// so a quarantine verdict on a segment evicts exactly that span and later
// segments are renumbered to match the compacted cache. Construct with New (the
// full registered detector chain) or NewWithGate (an explicit gate — e.g. a bare
// ctxmmu.MMU for a deterministic test, or the live session gate the recall leaf
// also persists, so byte-level and KV-level quarantine share one decision).
type Context struct {
	S           *model.Session
	gate        Gate
	segs        []*Segment
	meta        []cachemeta.Entry
	invalidated []cachemeta.Entry
	external    []cachemeta.ExternalInvalidationDirective
}

// New wires the kernel's full registered detector chain over a session's
// kernel-owned KV cache (the production gate).
func New(s *model.Session) *Context { return &Context{S: s, gate: FoldedGate{}} }

// NewWithGate enforces an explicit gate over the session's cache.
func NewWithGate(s *model.Session, gate Gate) *Context {
	return &Context{S: s, gate: gate}
}

// Append prefills a labeled segment of token ids into the session's KV cache,
// records the span it occupies, and returns the next-token logits (the
// distribution the model would decode from here) plus the segment. Use this for
// TRUSTED prompt chunks (system, user, prior trusted turns); use AdmitResult for
// untrusted tool output. The caller decodes from the returned logits via
// model.Session.Step (only the final chunk before a model turn needs them).
func (c *Context) Append(id, tool string, ids []int) ([]float32, *Segment) {
	from := c.S.Cache.Len()
	logits := c.S.Prefill(ids)
	seg := &Segment{ID: id, Tool: tool, From: from, Len: len(ids), KV: c.kvEntryID(ids)}
	c.segs = append(c.segs, seg)
	return logits, seg
}

func (c *Context) kvEntryID(ids []int) cachemeta.EntryID {
	modelID := ""
	if c.S != nil && c.S.M != nil {
		modelID = c.S.M.Cfg.ModelType
		if modelID == "" && len(c.S.M.Cfg.Architectures) > 0 {
			modelID = c.S.M.Cfg.Architectures[0]
		}
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

// AdmitResult is the write-time seam the agent loop calls the moment an untrusted
// tool result is produced. It (1) prefills the result's tokens into the cache,
// then (2) runs the result BYTES through the SHIPPED ctxmmu gate. If the verdict
// is Quarantine, it immediately EVICTS the just-appended span, so the model never
// attends to the poison on its next turn.
//
// Write-time is structural here, not a convention: the eviction happens before
// any later segment is prefilled, so the evicted span is causally upstream of
// nothing — which is precisely why the eviction equals never-having-seen it (the
// boundary internal/model's TestKVQuarantineEqualsNeverSaw proves a *late* evict
// cannot cross). It returns the gate's verdict and whether the span was evicted.
// It returns the gate's verdict, whether the span was evicted, and the next-token
// logits — which are nil when evicted (you cannot decode "from" a span that was
// just removed; the loop appends the next trusted query and decodes from there)
// and otherwise the distribution the model would continue from.
func (c *Context) AdmitResult(ctx context.Context, id, tool string, ids []int, body []byte) (v abi.Verdict, evicted bool, logits []float32) {
	logits, seg := c.Append(id, tool, ids)
	call := &abi.ToolCall{Tool: tool}
	res := &abi.Result{
		Call:    call,
		Payload: abi.Ref{Kind: abi.RefInline, Inline: body, Len: int64(len(body))},
	}
	v = c.gate.Admit(ctx, call, res)
	if v.Kind == abi.VerdictQuarantine {
		c.evict(seg)
		return v, true, nil
	}
	return v, false, logits
}

// Quarantine evicts a recorded segment's span from the cache by id, for the case
// the decision arrives AFTER the write — e.g. a recall re-screen, or a tightened
// pattern set, flags a page that was benign at write time (the lever
// RECALL-RESULTS.md keeps open). Returns positions evicted and whether the id was
// found and not already held. NOTE: this can only un-see the span if no later
// segment has attended to it yet (the same write-time boundary); callers that
// evict a span downstream tokens already absorbed get a compacted cache but not a
// never-saw guarantee — use AdmitResult for the guarantee.
func (c *Context) Quarantine(id string) (evicted int, ok bool) {
	for _, s := range c.segs {
		if s.ID == id && !s.Held {
			return c.evict(s), true
		}
	}
	return 0, false
}

// evict drops a segment's span from the kernel-owned cache and renumbers the
// ledger so every segment after it shifts down by the evicted length — keeping
// the ledger consistent with model.KVCache.Evict's own compaction.
func (c *Context) evict(seg *Segment) int {
	n := c.S.Cache.Evict(seg.From, seg.Len)
	c.external = append(c.external, cachemeta.PlanExternalInvalidations(seg.KV, c.meta)...)
	c.invalidateReferences(seg.KV)
	for _, s := range c.segs {
		if s != seg && s.From > seg.From {
			s.From -= seg.Len
		}
	}
	seg.Held = true
	seg.Len = 0
	return n
}

func (c *Context) invalidateReferences(kv cachemeta.EntryID) {
	if !kv.Valid() || len(c.meta) == 0 {
		return
	}
	live := c.meta[:0]
	for _, e := range c.meta {
		if cachemeta.AttentionIndexReferences(e, kv) || externalEntryReferencesKV(e, kv) {
			c.invalidated = append(c.invalidated, e)
			continue
		}
		live = append(live, e)
	}
	c.meta = live
}

func externalEntryReferencesKV(e cachemeta.Entry, kv cachemeta.EntryID) bool {
	if e.ID != kv {
		return false
	}
	return e.Residency.Tier == cachemeta.TierProvider || e.Residency.Tier == cachemeta.TierRemote
}

// Segments returns the current ledger (a copy of the slice header is fine; the
// caller should treat the entries as read-only).
func (c *Context) Segments() []*Segment { return c.segs }

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

// CacheLen is the number of live K/V positions in the session cache.
func (c *Context) CacheLen() int { return c.S.Cache.Len() }

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
