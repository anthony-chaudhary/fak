package toolproc

// repeatarm.go — the ARMED half of the live reuse seam (#5122, DoD of #4764):
// repeatreuse.go's ReuseStore is the reuse DECISION (identity + receipt, no body);
// ArmedCache is the caller-side content-addressed BYTE store that decision blesses —
// the piece that lets a live seam (the gateway supervisor) actually SERVE a repeat
// locally instead of only knowing that it could have.
//
// THE DIVISION OF TRUST, unchanged:
//   - The DECISION stays in ReuseStore and keeps its no-body contract: only digests
//     and sizes are ever journaled or reported.
//   - The BYTES live here, in process memory only, under a hard byte budget, keyed
//     on the same Identity the decision blesses ("read:<path>@<digest>" /
//     "query:<canon>"). Nothing here is persisted, exported, or logged — eviction
//     or process exit is total erasure.
//
// THE SAFETY CONTRACT, enforced as admission rules on Offer/Admit:
//   - IMMUTABLE reads are retained and served keyed on (resolved path + content
//     digest). A mutation (ReasonDigestChanged) evicts the stale body EAGERLY, so a
//     replaced read can never be served even before budget pressure reaps it.
//   - MUTABLE queries (the registered git/dispatch status family) are byte-served
//     ONLY when the operator OPTS IN (CoalesceQueries), and only inside the
//     decision's freshness window — every hit carries its stale-age + source on the
//     receipt.
//   - WRITES and UNKNOWNS are never retained and never served: Offer refuses them.
//   - A decision-layer hit whose bytes are absent (evicted, oversize, never
//     offered) reports BodyServed=false: the caller MUST fetch fresh. Served on the
//     receipt is the decision; BodyServed is the arming — only BodyServed=true may
//     skip the real spawn.
//
// The tallies (BodyHits/BodyMisses/ServedBytes) are the net-true numbers a dogfood
// witness reads: served repeats are avoided spawns, ServedBytes the tool-result
// bytes not re-fetched. The gateway wiring that feeds live calls through Admit and
// deposits fresh fetches via Offer has LANDED at the gateway leaf (#5119,
// internal/gateway/toolproc_reuse.go): the serve half runs in
// adjudicateProposedServed before dispatch and the deposit half in
// admitInboundResults on every ADMITTED tool_result, so this cache is a live seam
// and no longer a labeled next step. What that seam adds on top of the admission
// rules here is a STRICTER read gate — it refuses an immutable read whose current
// content digest it cannot witness, because Normalize's path-only fallback is right
// for offline analytics of a finished rollout and wrong on a wire where the file may
// have changed since. Arming stays the operator's (Server.SetToolprocReuse, default
// unarmed ⇒ inert).

// Armed-cache defaults: an 8 MiB total body budget and a 1 MiB per-entry cap. A
// payload over the entry cap is simply never retained (fail-open to a fresh fetch);
// budget pressure evicts oldest-first.
const (
	DefaultArmedMaxBytes      = 8 << 20
	DefaultArmedMaxEntryBytes = 1 << 20
)

// ArmedConfig tunes the armed cache. Zero values are safe: the defaults above, and
// query coalescing OFF — byte-serving mutable status output is strictly opt-in.
type ArmedConfig struct {
	// Repeat tunes the underlying decision store (freshness window etc.).
	Repeat RepeatConfig `json:"repeat,omitempty"`
	// MaxBytes bounds the total retained payload bytes. 0 => DefaultArmedMaxBytes.
	MaxBytes int64 `json:"max_bytes,omitempty"`
	// MaxEntryBytes caps a single retained payload. 0 => DefaultArmedMaxEntryBytes.
	MaxEntryBytes int64 `json:"max_entry_bytes,omitempty"`
	// CoalesceQueries opts IN to byte-serving registered mutable status commands
	// (git status, dispatch_status.py) inside their freshness window. Off, only
	// immutable reads are byte-served; the decision receipts still flow either way.
	CoalesceQueries bool `json:"coalesce_queries,omitempty"`
}

// ArmedResult is one armed admission: the decision Receipt plus the bytes, when the
// cache actually holds them. Receipt.Served is the decision layer's verdict;
// BodyServed is the armed verdict — only BodyServed=true entitles the caller to
// skip the real fetch and inject Body locally.
type ArmedResult struct {
	Receipt    Receipt
	Body       []byte
	BodyServed bool
}

// ArmedCache pairs a ReuseStore decision with a bounded, in-memory,
// content-addressed body store. Like the ReuseStore it wraps, it is NOT safe for
// concurrent use; the live seam serializes tool-call admission. Build with
// NewArmedCache.
type ArmedCache struct {
	store *ReuseStore
	cfg   ArmedConfig

	bodies   map[string][]byte // Identity -> retained payload
	order    []string          // deposit order, oldest first (eviction order)
	curBytes int64

	bodyHits    int
	bodyMisses  int
	servedBytes int64
}

// NewArmedCache builds an empty armed cache over a fresh decision store.
func NewArmedCache(cfg ArmedConfig) *ArmedCache {
	if cfg.MaxBytes == 0 {
		cfg.MaxBytes = DefaultArmedMaxBytes
	}
	if cfg.MaxEntryBytes == 0 {
		cfg.MaxEntryBytes = DefaultArmedMaxEntryBytes
	}
	return &ArmedCache{
		store:  NewReuseStore(cfg.Repeat),
		cfg:    cfg,
		bodies: map[string][]byte{},
	}
}

// Admit runs the reuse decision for one live tool call and, on a hit whose bytes
// are on hand, returns them. On a decision hit without bytes (not opted in,
// evicted, oversize, never offered) BodyServed is false and the caller must fetch
// fresh — then deposit the fresh payload via Offer so the NEXT repeat serves.
func (a *ArmedCache) Admit(rec CallRecord) ArmedResult {
	nc := Normalize(rec, a.store.cfg)

	// Capture the prior read identity BEFORE the decision mutates its state, so a
	// digest change can evict the replaced body eagerly.
	var priorID string
	if nc.Class == CmdImmutableRead {
		if prev, ok := a.store.reads[nc.Path]; ok {
			priorID = readIdentity(nc.Path, prev.digest)
		}
	}

	r := a.store.Admit(rec)
	if r.Reason == ReasonDigestChanged && priorID != "" {
		a.evict(priorID)
	}

	res := ArmedResult{Receipt: r}
	if !r.Served {
		return res
	}
	if nc.Class == CmdMutableQuery && !a.cfg.CoalesceQueries {
		// Freshness coalescing is opt-in: without it a status hit is advisory only.
		a.bodyMisses++
		return res
	}
	if b, ok := a.bodies[r.Identity]; ok {
		a.bodyHits++
		a.servedBytes += int64(len(b))
		res.Body = b
		res.BodyServed = true
		return res
	}
	a.bodyMisses++
	return res
}

// Offer deposits the payload of a REAL fetch under the identity the decision
// blesses, so subsequent repeats can be byte-served. It is fail-closed: writes and
// unknowns are refused outright, mutable-query bodies are refused unless
// coalescing is opted in, and an over-cap payload is not retained. Returns whether
// the body was retained. The payload is copied; the caller's buffer is not held.
func (a *ArmedCache) Offer(rec CallRecord, body []byte) bool {
	nc := Normalize(rec, a.store.cfg)
	switch nc.Class {
	case CmdImmutableRead:
		// always retainable
	case CmdMutableQuery:
		if !a.cfg.CoalesceQueries {
			return false
		}
	default: // idempotent write, unknown — never retained
		return false
	}
	n := int64(len(body))
	if n > a.cfg.MaxEntryBytes {
		return false
	}
	a.evict(nc.Identity) // replace any prior copy under this identity
	for a.curBytes+n > a.cfg.MaxBytes && len(a.order) > 0 {
		a.evict(a.order[0])
	}
	cp := make([]byte, n)
	copy(cp, body)
	a.bodies[nc.Identity] = cp
	a.order = append(a.order, nc.Identity)
	a.curBytes += n
	return true
}

// evict removes one identity's body, if present, releasing its budget.
func (a *ArmedCache) evict(id string) {
	b, ok := a.bodies[id]
	if !ok {
		return
	}
	delete(a.bodies, id)
	a.curBytes -= int64(len(b))
	for i, v := range a.order {
		if v == id {
			a.order = append(a.order[:i], a.order[i+1:]...)
			break
		}
	}
}

// readIdentity rebuilds an immutable-read Identity from its parts — kept in one
// place so the eager-eviction key can never drift from Normalize's fold.
func readIdentity(path, digest string) string {
	id := "read:" + path
	if digest != "" {
		id += "@" + digest
	}
	return id
}

// BodyHits, BodyMisses, ServedBytes, and RetainedBytes expose the armed tallies a
// dogfood witness reads: BodyHits are the avoided spawns actually served,
// ServedBytes the tool-result bytes not re-fetched, RetainedBytes the live budget
// use. Decision-layer tallies stay on the wrapped store (Decisions).
func (a *ArmedCache) BodyHits() int          { return a.bodyHits }
func (a *ArmedCache) BodyMisses() int        { return a.bodyMisses }
func (a *ArmedCache) ServedBytes() int64     { return a.servedBytes }
func (a *ArmedCache) RetainedBytes() int64   { return a.curBytes }
func (a *ArmedCache) Decisions() *ReuseStore { return a.store }
