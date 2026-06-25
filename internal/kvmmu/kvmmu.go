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
	"github.com/anthony-chaudhary/fak/internal/ctxplan"
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

	// Attended is the witnessed post-softmax attention mass this span has received in the
	// CURRENT turn, accumulated by AttributeRow from the rung-1 attention observer
	// (#852/#853). It is the in-flight per-turn term a_s(t): AttributeRow adds to it across
	// the turn's rows, and CloseTurn folds it into the rolling accumulators (#855) then
	// resets it to 0 for the next turn. It is mass, not a position, so the From renumbering
	// on eviction does not touch it; a survivor keeps exactly the mass it accumulated.
	// evict() zeroes it on the evicted span (a held span can no longer be attended to).
	// Zero means not attended this turn.
	Attended float64

	// The rolling per-span accumulators (#855, rung 4): the temporal dual of Attended. A
	// span's value spans many turns — it can idle then become load-bearing, or run hot
	// then die — so CloseTurn(lambda) folds each turn's a_s(t) (the just-closed Attended)
	// into two reductions of the SAME per-turn stream, the only difference being lambda:
	//
	//   Cumulative = Σ_t a_s(t)                    (undecayed; "mattered overall" — audit)
	//   EMA        = lambda·EMA + a_s(t)           (recency-decayed; "hot now" — eviction)
	//
	// With lambda == 1 the EMA recurrence reduces exactly to the cumulative sum (the
	// identity #855 requires); with lambda < 1 it is the H2O / heavy-hitter signal — but as
	// a WITNESSED kernel quantity, not an inferred one (the #851 novelty boundary: we did
	// not invent heavy-hitters; we attribute them from real emitted attention mass).
	Cumulative float64 // Σ_t a_s(t): undecayed lifetime mass (post-hoc analyst)
	EMA        float64 // lambda·EMA + a_s(t): recency-decayed rolling mass (real-time controller)

	// traj is a bounded ring of the most recent per-turn masses {a_s(t)} — the trajectory
	// that reconstructs WHEN a span was hot. Bounded to trajCap turns so memory is O(cap)
	// per span regardless of session length (O(1) amortized per turn). trajLen is how many
	// entries are valid (< trajCap until the ring fills); trajHead is the next write slot.
	traj     []float64
	trajHead int
	trajLen  int
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

// Context tracks the K/V span each segment occupies in a registered KV backend's
// cache (the in-process model.Session by default), so a quarantine verdict on a
// segment evicts exactly that span and later segments are renumbered to match the
// compacted cache. Construct with New (the full registered detector chain) or
// NewWithGate (an explicit gate — e.g. a bare ctxmmu.MMU for a deterministic test,
// or the live session gate the recall leaf also persists, so byte-level and KV-level
// quarantine share one decision).
//
// Enforcement goes through the abi.KVBackend interface (Len / Prefill / Evict /
// ModelID), NOT a concrete model type: the seam that issue #385 inverts. The
// session-typed constructors (New / NewWithGate) are a convenience that wraps a
// *model.Session into the in-process backend; NewBackend / NewBackendWithGate take
// an already-built abi.KVBackend (a remote/zero-copy KV backend, the disaggregated
// direction) so enforcement can run against an engine fak does not itself host.
type Context struct {
	kv          abi.KVBackend
	gate        Gate
	segs        []*Segment
	meta        []cachemeta.Entry
	invalidated []cachemeta.Entry
	external    []cachemeta.ExternalInvalidationDirective
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

// Append prefills a labeled segment of token ids into the session's KV cache,
// records the span it occupies, and returns the next-token logits (the
// distribution the model would decode from here) plus the segment. Use this for
// TRUSTED prompt chunks (system, user, prior trusted turns); use AdmitResult for
// untrusted tool output. The caller decodes from the returned logits via
// model.Session.Step (only the final chunk before a model turn needs them).
func (c *Context) Append(id, tool string, ids []int) ([]float32, *Segment) {
	from := c.kv.Len()
	logits := c.kv.Prefill(ids)
	seg := &Segment{ID: id, Tool: tool, From: from, Len: len(ids), KV: c.kvEntryID(ids)}
	c.segs = append(c.segs, seg)
	return logits, seg
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

// Compact is the kernel-mediated external-compaction seam (issue #522): it swaps a
// set of recorded segments for a single summary segment, but ONLY after the summary
// passes the SAME result-admit gate that screens tool output (ctxmmu / the folded
// detector chain). A poisoned summary is REFUSED — the old spans stay in the cache
// and the summary never enters — so compaction is a coherence-checked span swap,
// not an opaque rewrite that could launder a poisoned summary back into context.
// This is the kvmmu analogue of ctxmmu's write-time gate, applied to compaction.
//
// On a clean summary (anything but Quarantine — a benign oversize summary that the
// gate pages to a pointer is still safe to swap in): each named id still present
// and not already held is evicted (via the proven KVCache.Evict, which renumbers
// the survivors), then the summary is appended as a new trusted segment. The
// eviction order is the ledger order, so a multi-segment compaction renumbers
// correctly. Returns the gate verdict and the count of segments actually evicted.
// An unknown id, or one already held, is silently skipped.
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

// ApplyPlan bridges a ctxplan.Plan to bit-exact KV eviction (issue #550): it
// evicts every recorded segment whose id is in the plan's ELIDED set (and not
// also in its SELECTED set), so the session's KV residency shrinks to the plan's
// O(1) resident view. The eviction uses the proven model.KVCache.Evict (re-RoPE
// + renumber), so the compacted cache is byte-identical to a run that only ever
// prefilled the resident spans — an O(1) VIEW becomes an O(1) KV RESIDENCY.
//
// Why this bridge is load-bearing. ctxplan decides WHICH spans are resident (the
// plan); without ApplyPlan that decision is an O(1) view over an O(N) KV cache —
// the model's attention state still physically holds every elided span, so the
// "O(1)" is a text-layer claim the KV layer does not honor. ApplyPlan makes the
// kernel-owned cache AGREE with the plan: the elided spans are removed, and
// because KVCache.Evict re-RoPEs and renumbers the survivors, the result equals a
// run that only ever saw the resident set. The plan's faithfulness witness
// (ctxplan.Audit) already guarantees every elided span carries a page-back-in
// handle, so evicting it loses nothing — it pages back in on demand from the
// lossless store, exactly as a forecast miss does.
//
// id correspondence is the adapter contract: a segment is evicted iff its
// Segment.ID matches an elided ctxplan.Span.ID. The layer that lowers history
// into both ctxplan.Spans and kvmmu segments MUST use the same ids (the same way
// recall/cdb share page ids). A segment whose id is neither selected nor elided
// is LEFT IN PLACE — the bridge evicts only what the plan decided to elide, never
// what it did not consider; when the plan's candidates partition the whole ledger
// (the faithful case), residency collapses to exactly the Selected set. A
// selected span is never evicted even if a malformed plan also elides it (defense
// in depth over ctxplan.Audit's disjointness check). Eviction runs in ledger
// order so a multi-span elision renumbers correctly, exactly as Compact does.
//
// Honest fence: ApplyPlan only SHRINKS residency to match the view; it does not
// page resident spans IN (that is the demand-fault path, ctxplan.Materialize).
// Like the KV-quarantine bridge it is bit-exact on a synthetic model (the witness
// test) and is not yet wired into the live agent HTTP loop.
func (c *Context) ApplyPlan(plan ctxplan.Plan) (evicted int) {
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
	n := c.kv.Evict(seg.From, seg.Len)
	c.external = append(c.external, cachemeta.PlanExternalInvalidations(seg.KV, c.meta)...)
	c.invalidateReferences(seg.KV)
	for _, s := range c.segs {
		if s != seg && s.From > seg.From {
			s.From -= seg.Len
		}
	}
	seg.Held = true
	seg.Len = 0
	// A held span can no longer be attended to: its in-flight mass AND its rolling
	// accumulators (cumulative/EMA/trajectory) all clear — the span is gone from the
	// cache, so its attention history is moot. Survivors keep their accumulators
	// untouched (mass is not renumbered with From). This is the reset-on-evict the
	// rolling accumulator (#855) inherits from the rung-2 Attended semantics.
	seg.Attended = 0
	seg.Cumulative = 0
	seg.EMA = 0
	seg.traj = seg.traj[:0]
	seg.trajHead = 0
	seg.trajLen = 0
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
