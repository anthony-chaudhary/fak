package vdso

// scope.go — the FINER ERASER: scoped, hierarchical tier-2 invalidation plus the
// write-notification COHERENCE BUS.
//
// The problem this fixes (measured in internal/turnbench/fleet.go). The tier-2
// cache key embeds a single scalar world-version; ANY write-shaped completion does
// worldVer++, which strands EVERY cached entry at once — a full flush. For a fleet
// sharing one cache that is fine when nobody writes, but one agent's write erases
// every OTHER agent's warmed reads, so cross-agent sharing flips from a clean win
// to a NET LOSS once ~1% of fleet actions are writes. The fix is a finer eraser:
// invalidate only the reads the write could actually have changed.
//
// The mechanism — a HIERARCHICAL EPOCH VECTOR (the TLB-shootdown / surrogate-key
// analogue). Every resource has a tag path rooted at "*":
//
//	"*"  ->  "<namespace>"  ->  "<namespace>:<entity>"
//	 root     e.g. "flights"     e.g. "flights:SFO-JFK"
//
//   - A READ binds, in its cache key, the epoch of EVERY node on its root->leaf
//     chain (e.g. ["*","flights","flights:SFO-JFK"]). The key changes iff ANY node
//     on the chain is bumped.
//   - A WRITE bumps the epoch of the single finest node it can conservatively name
//     (a booking on a known route bumps "flights:SFO-JFK"; a write that cannot name
//     the row bumps "flights"; the legacy BumpWorld panic-button bumps "*"). Bumping
//     node N strands exactly the SUBTREE of reads whose chain contains N — its own
//     entity and everything finer — and leaves sibling subtrees (other routes, other
//     namespaces) warm.
//
// Soundness ("a hit equals a fresh call", preserved). The tag functions are a
// conservative MAY-ALIAS over-approximation: a write that can affect a read must
// share a tag with it. Two guarantees keep that true:
//
//  1. Coarse writes catch fine reads. A write that can only name "flights" bumps the
//     namespace node, which every flights read binds — so it invalidates the whole
//     subtree. An unknown tool degrades to bumping "*", which every read binds — i.e.
//     it falls back to today's full flush. Soundness is never traded for precision.
//  2. Fine writes need fine reads (the documented invariant, enforced for the tested
//     workload and asserted in scope_test.go). A write finer than a read it affects
//     is the one unsound shape: bumping "flights:SFO-JFK" does NOT invalidate a read
//     bound only at "flights". We forbid it by deriving read-tags and write-tags from
//     the SAME (namespace, entity) extraction: in the airline workload every
//     search_direct_flight read AND every book_flight write names a route, so no read
//     is ever coarser than a write to its namespace. A namespace whose writes are
//     entity-fine MUST have entity-fine reads; this is the one general limitation a
//     host integrating a new tool must honor (see open-holes in the design notes).
//
// The write-notification point as a FEATURE, not a bug. Emit is the single place a
// write is observed by the kernel. Rather than have it silently bump a counter, it
// publishes a typed Mutation{tool, tags, worldVer, seq} on a COHERENCE BUS. The
// cache's scoped invalidation is the first subscriber (it is implicit in the epoch
// bump), but the same precise "what changed" feed is now available to any other
// observer: a fleet change-feed an agent watches to re-plan, an audit log, a
// cross-agent private-cache snoop (MESI-style), or a metrics tap. One write becomes
// a structured coherence signal instead of a blunt "something, somewhere, changed."

import (
	"container/list"
	"encoding/json"
	"strings"
	"sync/atomic"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// Granularity selects how broadly a write-shaped completion invalidates tier-2.
type Granularity int32

const (
	// Global is the v0.1 behavior and the default: every read binds only the root
	// epoch and every write bumps it, so one write strands the whole cache (a full
	// flush). It is the trivially-sound partition (one tag aliases everything).
	Global Granularity = iota
	// Namespace scopes invalidation to a resource class: a write to "flights" leaves
	// "fx"/"users"/"docs" reads warm. The win for cross-namespace fleets.
	Namespace
	// Resource scopes invalidation to a single entity: a booking on "flights:SFO-JFK"
	// leaves the OTHER routes' shared reads warm. The win for within-namespace fleets,
	// and where the per-write cache damage drops by ~1/pool (the "100x" lever).
	Resource
)

func (g Granularity) String() string {
	switch g {
	case Namespace:
		return "namespace"
	case Resource:
		return "resource"
	default:
		return "global"
	}
}

// ParseGranularity maps a CLI/string name to a Granularity (default Global).
func ParseGranularity(s string) (Granularity, bool) {
	switch strings.ToLower(strings.TrimSpace(s)) {
	case "", "global", "world", "flush":
		return Global, true
	case "namespace", "ns", "class":
		return Namespace, true
	case "resource", "entity", "row", "fine":
		return Resource, true
	}
	return Global, false
}

const rootTag = "*"

// SetGranularity selects the invalidation granularity (Global by default). It does
// not clear the cache: finer modes simply stamp more epochs into new keys, so prior
// entries remain reachable only by callers computing the same chain.
func (v *VDSO) SetGranularity(g Granularity) { atomic.StoreInt32((*int32)(&v.gran), int32(g)) }

// GranularityOf reports the current invalidation granularity.
func (v *VDSO) GranularityOf() Granularity { return Granularity(atomic.LoadInt32((*int32)(&v.gran))) }

// SetNodeEpochLimit bounds the number of non-root scoped epoch nodes retained.
// Evicting a node bumps the root epoch so old keys never become reachable again.
func (v *VDSO) SetNodeEpochLimit(limit int) {
	if limit <= 0 {
		limit = DefaultNodeEpochLimit
	}
	v.mu.Lock()
	v.nodeCap = limit
	if v.nodeLRU == nil {
		v.nodeLRU = list.New()
	}
	if v.nodeIndex == nil {
		v.nodeIndex = map[string]*list.Element{}
	}
	if v.trimNodesLocked() {
		atomic.AddUint64(&v.worldVer, 1)
	}
	v.mu.Unlock()
}

// NodeEpochLimit reports the configured non-root epoch-node cap.
func (v *VDSO) NodeEpochLimit() int {
	v.mu.Lock()
	defer v.mu.Unlock()
	v.ensureNodeStateLocked()
	return v.nodeCap
}

// NodeEpochs reports the number of retained non-root epoch nodes.
func (v *VDSO) NodeEpochs() int {
	v.mu.Lock()
	defer v.mu.Unlock()
	return len(v.nodes)
}

// readChain returns the root->leaf tag chain a read's result depends on, truncated
// to the configured granularity. The key binds the epoch of every node returned, so
// a bump of any of them strands the entry. Global => ["*"]; Namespace => ["*","ns"];
// Resource => ["*","ns","ns:entity"] (or ["*","ns"] when the entity is not nameable).
func (v *VDSO) readChain(c *abi.ToolCall, args []byte) []string {
	g := v.GranularityOf()
	if g == Global {
		return []string{rootTag}
	}
	ns := namespaceOf(c.Tool)
	if ns == "" {
		return []string{rootTag} // unknown class: root-only (any write invalidates it)
	}
	if g == Namespace {
		return []string{rootTag, ns}
	}
	if ent := entityOf(ns, args); ent != "" {
		return []string{rootTag, ns, ns + ":" + ent}
	}
	return []string{rootTag, ns} // resource mode but row not nameable: bind the class
}

// writeTags returns the node(s) a write-shaped completion must bump. It names the
// SINGLE finest node the write can conservatively claim — the same (namespace,
// entity) extraction reads use — falling back UP the chain (entity -> namespace ->
// root) whenever the finer name is unavailable, which only ever over-invalidates.
func (v *VDSO) writeTags(c *abi.ToolCall, args []byte) []string {
	g := v.GranularityOf()
	if g == Global {
		return []string{rootTag}
	}
	ns := namespaceOf(c.Tool)
	if ns == "" {
		return []string{rootTag} // unknown write: fall all the way back to a full flush
	}
	if g == Namespace {
		return []string{ns}
	}
	if ent := entityOf(ns, args); ent != "" {
		return []string{ns + ":" + ent}
	}
	return []string{ns} // resource mode but row not nameable: bump the whole class
}

// nsKeywords maps each resource class to the tool-name substrings that denote it.
// Reads and writes to the same resource MUST land here on the SAME class (book_flight
// and search_direct_flight both contain "flight" => "flights"), or a write would not
// invalidate the read it changed.
var nsKeywords = []struct {
	ns       string
	keywords []string
}{
	{"flights", []string{"flight"}},
	{"fx", []string{"currency", "convert", "fx", "exchange", "forex"}},
	{"hotels", []string{"hotel", "room"}},
	{"users", []string{"user", "account", "profile"}},
	{"airports", []string{"airport"}},
	{"docs", []string{"policy", "refund", "doc"}},
}

// namespaceOf maps a tool name to its resource class (the depth-1 tag), or "" if
// unknown. An unmatched tool returns "" and is scoped to the root (a full flush) —
// sound, never silently under-invalidated. A tool that matches TWO DIFFERENT classes
// (a join / cross-cutting read like "price_flight_in_currency") also returns "": a
// single root->leaf chain cannot soundly express a multi-resource dependency, so we
// conservatively degrade it to the root rather than bind only one of its namespaces
// and miss writes to the other (the fundamental tree limitation — see the package
// doc). This keeps the default tagger a sound over-approximation by construction.
func namespaceOf(tool string) string {
	ns, _ := classifyNamespace(tool)
	return ns
}

// ClassifyNamespace maps a tool name to its resource class (the depth-1 tag) and
// reports whether the name collided with MORE THAN ONE class. Both an unknown name
// and a multi-class name return ns=="" and bind the root (a full flush), but they
// are different situations: unknown is benign (the tool simply isn't tagged), while
// a multi-class collision (a join like "price_flight_in_currency") SILENTLY throws
// away all the finer-eraser precision the cache exists to provide. The static tool
// linter (internal/toollint) uses multi to flag the latter at definition time —
// from this one shared classifier, so the lint and the runtime tagging never drift.
func ClassifyNamespace(tool string) (ns string, multi bool) { return classifyNamespace(tool) }

func classifyNamespace(tool string) (string, bool) {
	t := strings.ToLower(tool)
	found := ""
	for _, g := range nsKeywords {
		for _, k := range g.keywords {
			if strings.Contains(t, k) {
				if found != "" && found != g.ns {
					return "", true // multi-namespace tool: degrade to "*" (sound)
				}
				found = g.ns
				break
			}
		}
	}
	return found, false
}

// entityOf extracts the specific entity a call names within its namespace, for the
// Resource granularity. It returns "" when the row is not nameable from args, so the
// caller falls back to the namespace tag (sound). Reads and writes share this
// extraction, so a booking and a search of the same route hash to the same entity.
func entityOf(ns string, args []byte) string {
	var m map[string]json.RawMessage
	if json.Unmarshal(args, &m) != nil {
		return ""
	}
	get := func(keys ...string) string {
		for _, k := range keys {
			if raw, ok := m[k]; ok {
				var s string
				if json.Unmarshal(raw, &s) == nil && s != "" {
					return strings.ToUpper(s)
				}
			}
		}
		return ""
	}
	switch ns {
	case "flights", "hotels":
		o := get("origin", "from", "src", "source")
		d := get("destination", "dest", "to", "target")
		if o != "" && d != "" {
			return o + "-" + d
		}
		// A single-endpoint call (e.g. "all flights from an origin") is NOT given an
		// entity tag. A prefix leaf like "flights:SFO" would be a SIBLING of the route
		// leaf "flights:SFO-JFK" (neither is the other's ancestor), so a booking on
		// SFO-JFK would NOT invalidate the origin-only read — a stale serve. Returning
		// "" makes such a read bind only [*, ns]; the resource-mode cacheability gate
		// (resourceMisnamed) then refuses to tier-2 it, so it can never be served stale.
		return ""
	case "fx":
		o := get("from", "source", "src")
		d := get("to", "target", "dest")
		if o != "" && d != "" {
			return o + "-" + d
		}
		return ""
	case "users":
		return get("user_id", "userid", "id", "user")
	}
	return ""
}

// epochLocked returns the current epoch of a tag. The root epoch is the atomic
// worldVer (kept lock-free for the Global hot path); deeper nodes live in the nodes
// map. Callers other than the root path must hold v.mu.
func (v *VDSO) epochLocked(tag string) uint64 {
	if tag == rootTag {
		return atomic.LoadUint64(&v.worldVer)
	}
	return v.nodes[tag]
}

// resourceMisnamed reports whether, at Resource granularity, a call maps to a KNOWN
// namespace but cannot name its entity (entityOf == ""). Such a read would bind only
// [*, ns], yet an entity-fine write bumps a leaf BELOW ns that the read does not bind
// — so the read would never be invalidated: a stale serve. We therefore refuse to
// tier-2 cache it (it always reaches the engine, which is sound). An unknown-namespace
// call binds [*] and is caught by the root writes its own namespace would issue, so it
// stays cacheable; at coarser granularities every read reaches its target depth, so
// this never fires. This is the gate that scopes the soundness claim to FULLY-NAMED
// reads without giving up route-level precision for them.
func (v *VDSO) resourceMisnamed(c *abi.ToolCall, args []byte) bool {
	if v.GranularityOf() != Resource {
		return false
	}
	ns := namespaceOf(c.Tool)
	return ns != "" && entityOf(ns, args) == ""
}

// bumpAndPublish advances the epoch of each named tag, snapshots the post-write root
// epoch + mutation sequence, and notifies the coherence bus — all as ONE consistent
// event. The epoch increments, the mutSeq increment, and the (WorldVer, Seq) capture
// happen under a SINGLE v.mu critical section, so two concurrent writers can never
// hand a subscriber a torn (Seq, WorldVer) tuple from different bumps; subscribers run
// AFTER Unlock so a subscriber may re-enter the vDSO. The root stays an atomic so the
// Global-mode chain-length-1 key read can still load it lock-free; reading it back
// under the lock here keeps the published WorldVer consistent with this write's bump.
func (v *VDSO) bumpAndPublish(c *abi.ToolCall, tags []string) {
	v.mu.Lock()
	v.ensureNodeStateLocked()
	bumpedRoot := false
	for _, t := range tags {
		if t == rootTag {
			atomic.AddUint64(&v.worldVer, 1)
			bumpedRoot = true
		} else {
			v.nodes[t]++
			v.touchNodeLocked(t)
		}
	}
	if v.trimNodesLocked() && !bumpedRoot {
		atomic.AddUint64(&v.worldVer, 1)
		tags = append(append([]string(nil), tags...), rootTag)
	}
	v.mutSeq++
	m := Mutation{
		Tool:     c.Tool,
		Tags:     append([]string(nil), tags...),
		WorldVer: atomic.LoadUint64(&v.worldVer),
		Seq:      v.mutSeq,
	}
	subs := append([]*subscription(nil), v.subs...)
	v.mu.Unlock()

	atomic.AddInt64(&v.mutations, 1)
	for _, s := range subs {
		s.fn(m)
	}
}

func (v *VDSO) ensureNodeStateLocked() {
	if v.nodeCap <= 0 {
		v.nodeCap = DefaultNodeEpochLimit
	}
	if v.nodeLRU == nil {
		v.nodeLRU = list.New()
	}
	if v.nodeIndex == nil {
		v.nodeIndex = map[string]*list.Element{}
	}
	if v.nodes == nil {
		v.nodes = map[string]uint64{}
	}
}

func (v *VDSO) touchNodeLocked(tag string) {
	if el := v.nodeIndex[tag]; el != nil {
		v.nodeLRU.MoveToFront(el)
		return
	}
	v.nodeIndex[tag] = v.nodeLRU.PushFront(tag)
}

func (v *VDSO) trimNodesLocked() bool {
	trimmed := false
	for len(v.nodes) > v.nodeCap {
		el := v.nodeLRU.Back()
		if el == nil {
			return trimmed
		}
		tag := el.Value.(string)
		v.nodeLRU.Remove(el)
		delete(v.nodeIndex, tag)
		delete(v.nodes, tag)
		trimmed = true
	}
	return trimmed
}

// ----------------------------------------------------------------------------
// Coherence bus — the single write-notification point, exposed as a feature.
// ----------------------------------------------------------------------------

// Mutation is one write-shaped completion observed at the kernel's single
// write-notification point. Tags names the resource tags the write may have changed
// — exactly the tags that scoped the cache invalidation — so a subscriber gets a
// precise "what changed" signal rather than a blunt "something changed" bump.
type Mutation struct {
	Tool     string   // the write-shaped tool that completed
	Tags     []string // resource tags bumped (the invalidation scope)
	WorldVer uint64   // root ("*") epoch after the write — a monotone global clock
	Seq      uint64   // per-VDSO monotone mutation sequence (ordering without a wall clock)
}

type subscription struct {
	id uint64
	fn func(Mutation)
}

// Subscribe registers an observer of write mutations and returns a cancel func.
// Subscribers are invoked synchronously AFTER the epoch bump, off the read hot path
// (only write completions fire), and outside v.mu so a subscriber may re-enter the
// vDSO. The cache's own invalidation is the epoch bump itself — subscribers are
// ADDITIONAL observers (change-feed, audit, snoop, metrics).
func (v *VDSO) Subscribe(fn func(Mutation)) (cancel func()) {
	if fn == nil {
		return func() {}
	}
	v.mu.Lock()
	v.subSeq++
	id := v.subSeq
	v.subs = append(v.subs, &subscription{id: id, fn: fn})
	v.mu.Unlock()
	return func() {
		v.mu.Lock()
		for i, s := range v.subs {
			if s.id == id {
				v.subs = append(v.subs[:i], v.subs[i+1:]...)
				break
			}
		}
		v.mu.Unlock()
	}
}

// Mutations reports how many write-shaped completions the vDSO has observed (the
// coherence-bus event count) — write-side observability the scalar worldVer hid.
func (v *VDSO) Mutations() int64 { return atomic.LoadInt64(&v.mutations) }
