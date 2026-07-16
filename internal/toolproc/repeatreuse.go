package toolproc

// repeatreuse.go — the LIVE reuse STORE for #4764: the stateful counterpart to
// repeatclass.go's offline ClassifyRepeats fold. Where the classifier folds a whole
// rollout into a typed inventory + net-true saving, ReuseStore is what a live seam
// (the gateway supervisor) calls per tool call to DECIDE, right now, whether this
// repeat may be served locally — and to emit a per-hit Receipt as the evidence.
//
// THE SAFETY CONTRACT, unchanged from the classifier and enforced here as STATE:
//   - IMMUTABLE reads are keyed on (resolved path + content digest). A hit serves;
//     a mutation (a NEW digest for the same path) INVALIDATES the stale entry and
//     forces a fresh fetch — the #2917 invalidation model, realized as a key, not a
//     competing store.
//   - MUTABLE queries (git/dispatch status) are served ONLY inside their freshness
//     window of the last real fetch, and every reuse exposes its stale-age + source.
//   - WRITES and UNKNOWNS are fail-closed: never served.
//
// NO BODY RETENTION. Like the analytics surface, the cache keeps only the content
// DIGEST and the output SIZE — never a payload. A caller holds the actual bytes in a
// content-addressed store keyed on the digest this store blesses; the store's job
// is the reuse DECISION and its receipt, so this leaf stays a tier-2 mechanism with
// no body and no secret. The live wiring (feeding it spawn/pulse events off the wire
// and holding the bytes) arms at the gateway — the labeled next step (see doc.go).

// ReuseSource is the CLOSED provenance of a served repeat — WHERE the value came
// from. Exposed on every reuse so a consumer never mistakes a local hit for a fresh
// fetch, nor a stale freshness-window value for a live one.
type ReuseSource string

const (
	SourceMiss      ReuseSource = "miss"             // no reuse; a real fetch is required
	SourceImmutable ReuseSource = "immutable_reuse"  // served from the digest-keyed immutable-read store
	SourceFreshness ReuseSource = "freshness_window" // served within a mutable query's freshness window
)

// ReuseReason is the CLOSED explanation for one reuse decision — a finding never
// carries free text.
type ReuseReason string

const (
	ReasonKeyedHit      ReuseReason = "keyed_hit"      // identity (path+digest) matched a live entry
	ReasonFreshnessHit  ReuseReason = "freshness_hit"  // within FreshMS of the last real fetch
	ReasonFirstFetch    ReuseReason = "first_fetch"    // first observation of this identity
	ReasonDigestChanged ReuseReason = "digest_changed" // a mutation invalidated the immutable entry
	ReasonWindowExpired ReuseReason = "window_expired" // past the freshness window → fresh fetch
	ReasonNeverReused   ReuseReason = "never_reused"   // a write / unknown → fail-closed
)

// Receipt is the per-hit evidence one Admit decision emits — the durable record
// #4764 requires on every reuse. It carries NO output body: only the identity, the
// decision + provenance, the content digest served, and (for a freshness hit) the
// stale-age. SavedBytes is the output size the hit did NOT re-fetch/re-inject.
type Receipt struct {
	Identity   string       `json:"identity"`
	Class      CommandClass `json:"class"`
	Reuse      ReuseMode    `json:"reuse"`
	Served     bool         `json:"served"`
	Source     ReuseSource  `json:"source"`
	Reason     ReuseReason  `json:"reason"`
	Digest     string       `json:"digest,omitempty"`
	StaleAgeMS int64        `json:"stale_age_ms,omitempty"`
	SavedBytes int64        `json:"saved_bytes,omitempty"`
}

// ReuseStore is the stateful reuse store. It is NOT safe for concurrent use; the
// live seam serializes tool-call admission, exactly like the process table. Zero
// value is not usable — build one with NewReuseStore so the maps and defaults exist.
type ReuseStore struct {
	cfg     RepeatConfig
	reads   map[string]readState  // resolved path -> last observed {digest,size}
	queries map[string]queryState // mutable-query identity -> last REAL fetch {atMS,size}
	hits    int
	misses  int
}

type readState struct {
	digest string
	size   int64
}

type queryState struct {
	atMS int64
	size int64
}

// NewReuseStore builds an empty store. A zero DefaultFreshnessMS falls back to the
// classifier's DefaultFreshnessWindowMS so the two halves agree on the window.
func NewReuseStore(cfg RepeatConfig) *ReuseStore {
	if cfg.DefaultFreshnessMS == 0 {
		cfg.DefaultFreshnessMS = DefaultFreshnessWindowMS
	}
	return &ReuseStore{
		cfg:     cfg,
		reads:   map[string]readState{},
		queries: map[string]queryState{},
	}
}

// Admit decides, for one normalized tool call, whether the repeat may be served
// locally and returns the Receipt. It mutates cache state: on a miss it records the
// real fetch (so the NEXT repeat can be served or freshness-bounded). Secrets and
// bodies never enter — Normalize redacts at the boundary and only sizes/digests are
// retained.
func (c *ReuseStore) Admit(rec CallRecord) Receipt {
	nc := Normalize(rec, c.cfg)
	switch nc.Class {
	case CmdImmutableRead:
		return c.admitRead(nc, rec)
	case CmdMutableQuery:
		return c.admitQuery(nc, rec)
	default: // idempotent write, unknown — fail-closed
		c.misses++
		return Receipt{
			Identity: nc.Identity, Class: nc.Class, Reuse: ReuseNever,
			Served: false, Source: SourceMiss, Reason: ReasonNeverReused,
		}
	}
}

// admitRead serves an immutable read from the digest-keyed cache, invalidating the
// stale entry when a mutation changes the digest.
func (c *ReuseStore) admitRead(nc NormalCall, rec CallRecord) Receipt {
	r := Receipt{Identity: nc.Identity, Class: CmdImmutableRead, Reuse: ReuseKeyed, Digest: nc.Digest}
	prev, ok := c.reads[nc.Path]
	switch {
	case ok && prev.digest == nc.Digest:
		// Identity matched (same path+digest, or path-only when no digest was
		// observed on either) — serve from the cache.
		c.hits++
		r.Served = true
		r.Source = SourceImmutable
		r.Reason = ReasonKeyedHit
		r.SavedBytes = prev.size
		return r
	case ok && nc.Digest != "" && prev.digest != nc.Digest:
		// A mutation changed the content digest — invalidate the stale entry and
		// fetch fresh. This is the invalidation-after-mutation contract.
		c.misses++
		c.reads[nc.Path] = readState{digest: nc.Digest, size: rec.OutputBytes}
		r.Served = false
		r.Source = SourceMiss
		r.Reason = ReasonDigestChanged
		return r
	default:
		// First observation of this path (or a digest<->no-digest transition, which
		// we treat conservatively as a fresh fetch).
		c.misses++
		c.reads[nc.Path] = readState{digest: nc.Digest, size: rec.OutputBytes}
		r.Served = false
		r.Source = SourceMiss
		r.Reason = ReasonFirstFetch
		return r
	}
}

// admitQuery serves a mutable query only inside its freshness window of the last
// REAL fetch, exposing the stale-age (and the freshness_window source) on every hit.
func (c *ReuseStore) admitQuery(nc NormalCall, rec CallRecord) Receipt {
	r := Receipt{Identity: nc.Identity, Class: CmdMutableQuery, Reuse: ReuseFreshnessBounded}
	fresh := nc.FreshMS
	if fresh == 0 {
		fresh = c.cfg.DefaultFreshnessMS
	}
	prev, ok := c.queries[nc.Identity]
	if ok && rec.AtMS >= prev.atMS && rec.AtMS-prev.atMS <= fresh {
		c.hits++
		r.Served = true
		r.Source = SourceFreshness
		r.Reason = ReasonFreshnessHit
		r.StaleAgeMS = rec.AtMS - prev.atMS
		r.SavedBytes = prev.size
		return r
	}
	// Miss: a real fetch. Reset the window to this observation.
	c.misses++
	reason := ReasonFirstFetch
	if ok {
		reason = ReasonWindowExpired
	}
	c.queries[nc.Identity] = queryState{atMS: rec.AtMS, size: rec.OutputBytes}
	r.Served = false
	r.Source = SourceMiss
	r.Reason = reason
	return r
}

// Replay admits a whole stream in order and returns the receipts — the live-store
// counterpart to ClassifyRepeats over the same records. Given records in ascending
// time order it is deterministic: same stream ⇒ same receipts.
func (c *ReuseStore) Replay(recs []CallRecord) []Receipt {
	out := make([]Receipt, 0, len(recs))
	for _, rec := range recs {
		out = append(out, c.Admit(rec))
	}
	return out
}

// Hits and Misses expose the served/fetched tallies for a net-true saving read
// (served repeats are the avoided spawns the classifier scores offline).
func (c *ReuseStore) Hits() int   { return c.hits }
func (c *ReuseStore) Misses() int { return c.misses }
