// registry.go — the extension mechanism of the frozen ABI.
//
// Additive-only driver-registration surface: every subsystem attaches by calling a
// Register* function from its own package's init(); the kernel walks these
// registries and never imports a driver. internal/registrations blank-imports the
// built-in drivers, so enabling/disabling an idea is one import line. Numbers come
// from reserved per-subsystem ranges and Register* panics on a clash, so
// independently-built leaves cannot collide even when added in parallel; the
// dependency graph is a star (abi at the center, drivers depend only on abi).
//
// Written once at init(), read on every syscall: writers rebuild one immutable
// snapshot under the lock, published atomically; readers do a single atomic load
// and index pre-built slices/maps — no mutex, no allocation, O(1) regardless of
// how many ideas are registered.
package abi

import (
	"context"
	"fmt"
	"sort"
	"sync"
	"sync/atomic"
)

// Range bounds an OPEN [Lo,Hi) id range a subsystem draws its numbers from.
// Reserved per-subsystem ranges: a new subsystem claims a [lo, hi) block here
// (additive-only) and draws its numbers from it; Register* enforces disjointness at init().
type Range struct{ Lo, Hi uint32 }

// Reserved Range partitions for the open extension tables: ops, ExtKey sidecars, vendor verdicts, and event kinds draw ids from these blocks.
var (
	OpsCore   = Range{0, 64}            // sync call, submit, reap (built in)
	OpsSpec   = Range{64, 96}           // speculative commit/squash
	OpsAsync  = Range{96, 128}          // async submit/reap variants
	OpsVendor = Range{1 << 16, 1 << 17} // out-of-tree / experimental

	// ExtKey blocks (typed sidecar payloads on ToolCall.Ext / Result.Ext).
	ExtSpec   = Range{0, 16}  // speculation id / epoch sidecar
	ExtAsync  = Range{16, 32} // completion cursor sidecar
	ExtLabel  = Range{32, 48} // self-labeling rung rows
	ExtTrust  = Range{48, 64} // federated-trust score sidecar
	ExtVendor = Range{1 << 16, 1 << 17}

	// Above VerdictReservedMax (1023): open/registered kinds; below is the closed trainable core.
	VerdictsVendor = Range{1024, 1 << 16}

	EventsCore   = Range{0, 64}    // submit/decide/dispatch/complete
	EventsKPI    = Range{64, 128}  // metric taps
	EventsLabel  = Range{128, 192} // label-bearing (typed LabelRow)
	EventsVendor = Range{1 << 16, 1 << 17}
)

func inRange(n uint32, r Range) bool { return n >= r.Lo && n < r.Hi }

// ResetForTest clears every registry to empty. It exists only for tests that need
// a controlled driver set in isolation; it is never called in production.
func ResetForTest() {
	reg.mu.Lock()
	defer reg.mu.Unlock()
	reg.adjudicators = nil
	reg.resultAdmits = nil
	reg.fastpaths = nil
	reg.ops = map[OpCode]Op{}
	reg.emitters = nil
	reg.verdictKinds = map[VerdictKind]vkInfo{}
	reg.reasons = map[ReasonCode]string{}
	reg.caps = map[Capability]bool{}
	reg.engines = map[string]EngineDriver{}
	reg.regionBackend = nil
	reg.kvBackend = nil
	// Page-out codecs are the one registry this reset RESTORES rather than empties:
	// a page-out codec (notably internal/blob's "blob" default, wired at blob.init())
	// is a passive, stateless, content-addressed backend a test never means to strip.
	// Emptying it left a sibling paging/evict test's abi.PageOut("blob") backend-less — a silent, order-dependent flake.
	reg.pageOut = restoredBuiltinPageOut()
	reg.witnesses = map[string]WitnessResolver{}
	reg.stewards = nil
	reg.sinks = nil
	reg.screens = nil
	rebuildSnapshot()
}

// builtinPageOut snapshots the page-out codecs registered at init() the FIRST time
// ResetForTest runs — before it clears them; every package init() has completed by
// the first reset, so this baseline is exactly the process's built-in codec set.
// Capturing it here (rather than re-registering "blob") keeps abi a leaf: importing
// internal/blob would be an import cycle (blob imports abi, never the reverse); touched only inside ResetForTest, under reg.mu.
var builtinPageOut map[string]PageOutBackend

// restoredBuiltinPageOut returns a fresh copy of the built-in page-out codec baseline (captured from the live registry on first call); callers hold reg.mu.
func restoredBuiltinPageOut() map[string]PageOutBackend {
	if builtinPageOut == nil {
		builtinPageOut = make(map[string]PageOutBackend, len(reg.pageOut))
		for id, b := range reg.pageOut {
			builtinPageOut[id] = b
		}
	}
	out := make(map[string]PageOutBackend, len(builtinPageOut))
	for id, b := range builtinPageOut {
		out[id] = b
	}
	return out
}

// Builder state: guarded by one mutex, touched only by the write side (Register* /
// ResetForTest) at init() before the kernel boots; the read side never touches it.

var reg = struct {
	mu sync.Mutex

	adjudicators  []rankedAdj // LSM chain, sorted by rank ascending
	resultAdmits  []rankedRA  // write-time result-admission chain (the MMU seam)
	fastpaths     []rankedFP  // vDSO tiers, sorted by tier ascending
	ops           map[OpCode]Op
	emitters      []Emitter // KPI / steward / label sinks
	verdictKinds  map[VerdictKind]vkInfo
	reasons       map[ReasonCode]string
	caps          map[Capability]bool
	engines       map[string]EngineDriver
	regionBackend RegionBackend    // the Resolver provider (last registration wins)
	kvBackend     KVBackendFactory // the KV-MMU enforcement backend factory (last registration wins)
	pageOut       map[string]PageOutBackend
	witnesses     map[string]WitnessResolver
	stewards      []Steward
	sinks         []ProvisionalSink
	screens       []SemanticScreen // local-model-on-the-wire advisory chain (semscreen.go)
}{
	ops:          map[OpCode]Op{},
	verdictKinds: map[VerdictKind]vkInfo{},
	reasons:      map[ReasonCode]string{},
	caps:         map[Capability]bool{},
	engines:      map[string]EngineDriver{},
	pageOut:      map[string]PageOutBackend{},
	witnesses:    map[string]WitnessResolver{},
}

type rankedAdj struct {
	rank int
	a    Adjudicator
}
type rankedFP struct {
	tier int
	f    FastPath
}
type rankedRA struct {
	rank int
	ra   ResultAdmitter
}

// vkInfo is what a registered verdict kind declares; it carries a FoldRank so the core fold can order a new kind in the restrictiveness lattice without a core edit.
type vkInfo struct {
	name     string
	foldRank int           // position in the restrictiveness lattice (higher = wins)
	fallback FallbackClass // how an UNAWARE worker treats this kind
}

// Published read side: an immutable snapshot. rebuildSnapshot() builds a fresh one
// from the builder state and publishes it atomically; readers index pre-built
// slices/maps with no lock and no allocation. Never mutated after publish, so a
// reader holding an old snapshot across a concurrent registration is race-free (it sees the pre-registration view).

type snapshot struct {
	adjudicators []Adjudicator // rank-sorted, pre-extracted from rankedAdj (ALL rungs)

	// Adjudicator fan-out index (Fix C): adjByTool[t] is the rank-ordered chain a
	// call for tool t needs (unconditional rungs plus rungs scoped to t); uncondAdj
	// is the fallback for a tool no scoped rung claims. Scoping semantics in
	// CallScope; with no scoped rung adjByTool is nil and AdjudicatorsFor returns
	// the full chain (v0.1 exactly).
	adjByTool map[string][]Adjudicator
	uncondAdj []Adjudicator

	fastpaths    []FastPath       // tier-sorted, pre-extracted from rankedFP
	resultAdmits []ResultAdmitter // rank-sorted, pre-extracted from rankedRA (ALL gates)

	// Result-admit fan-out index (Fix D), the dual of adjByTool/uncondAdj:
	// raByTool[t] is the rank-ordered gate chain a result for tool t needs; uncondRA
	// is the fallback. Not implementing CallScope means always-run — the fail-closed
	// default (a result admitter is the exfil/quarantine floor) — so with no scoped
	// gate raByTool is nil and ResultAdmittersFor returns the full chain (v0.1 exactly).
	raByTool map[string][]ResultAdmitter
	uncondRA []ResultAdmitter

	emitters []Emitter // registration order (ALL observers; Emitters())

	// Event fan-out index (Fix B): emittersByKind[k] is the registration-ordered
	// list of observers receiving kind k; allEmitters is the universal-observer
	// fallback bucket. With no selective observer, emittersByKind is nil and
	// allEmitters == emitters, so EmittersFor reproduces v0.1 exactly.
	emittersByKind map[EventKind][]Emitter
	allEmitters    []Emitter

	stewards  []Steward
	sinks     []ProvisionalSink
	screens   []SemanticScreen  // local-model-on-the-wire advisory chain (registration order)
	witnesses []WitnessResolver // id-sorted (deterministic gate order)

	foldRanks map[VerdictKind]int // registered kinds only (core kinds use the switch)
	fallbacks map[VerdictKind]FallbackClass
	caps      map[Capability]bool
	engines   map[string]EngineDriver
	engineIDs []string     // id-sorted
	anyEngine EngineDriver // deterministic pick for Engine("") (engines[engineIDs[0]])
	region    RegionBackend
	kvBackend KVBackendFactory
	pageOut   map[string]PageOutBackend
	ops       map[OpCode]Op
}

// emptySnapshot backs reads before the first registration: all fields are zero, and ranging a nil slice / reading a nil map is well-defined in Go ("nothing registered").
var emptySnapshot = &snapshot{}

var published atomic.Pointer[snapshot]

// loadSnapshot returns the current immutable read view (never nil).
func loadSnapshot() *snapshot {
	if s := published.Load(); s != nil {
		return s
	}
	return emptySnapshot
}

// rebuildSnapshot constructs a fresh immutable snapshot from the builder state and
// publishes it atomically. Must be called with reg.mu held — every Register* and
// ResetForTest calls it as its last act under the lock; it runs only at registration
// time, so its O(features) build cost never touches a syscall.
func rebuildSnapshot() {
	s := &snapshot{
		adjudicators: make([]Adjudicator, len(reg.adjudicators)),
		fastpaths:    make([]FastPath, len(reg.fastpaths)),
		resultAdmits: make([]ResultAdmitter, len(reg.resultAdmits)),
		emitters:     append([]Emitter(nil), reg.emitters...),
		stewards:     append([]Steward(nil), reg.stewards...),
		sinks:        append([]ProvisionalSink(nil), reg.sinks...),
		screens:      append([]SemanticScreen(nil), reg.screens...),
		foldRanks:    make(map[VerdictKind]int, len(reg.verdictKinds)),
		fallbacks:    make(map[VerdictKind]FallbackClass, len(reg.verdictKinds)),
		caps:         make(map[Capability]bool, len(reg.caps)),
		engines:      make(map[string]EngineDriver, len(reg.engines)),
		pageOut:      make(map[string]PageOutBackend, len(reg.pageOut)),
		ops:          make(map[OpCode]Op, len(reg.ops)),
		region:       reg.regionBackend,
		kvBackend:    reg.kvBackend,
	}
	for i, r := range reg.adjudicators {
		s.adjudicators[i] = r.a
	}
	for i, r := range reg.fastpaths {
		s.fastpaths[i] = r.f
	}
	for i, r := range reg.resultAdmits {
		s.resultAdmits[i] = r.ra
	}
	// Witnesses: id-sorted so the require-witness gate consults resolvers deterministically (map-range order was random).
	wids := make([]string, 0, len(reg.witnesses))
	for id := range reg.witnesses {
		wids = append(wids, id)
	}
	sort.Strings(wids)
	s.witnesses = make([]WitnessResolver, len(wids))
	for i, id := range wids {
		s.witnesses[i] = reg.witnesses[id]
	}
	for k, vk := range reg.verdictKinds {
		s.foldRanks[k] = vk.foldRank
		s.fallbacks[k] = vk.fallback
	}
	for c := range reg.caps {
		s.caps[c] = true
	}
	eids := make([]string, 0, len(reg.engines))
	for id, d := range reg.engines {
		s.engines[id] = d
		eids = append(eids, id)
	}
	sort.Strings(eids)
	s.engineIDs = eids
	if len(eids) > 0 {
		s.anyEngine = reg.engines[eids[0]] // deterministic pick for Engine("")
	}
	for id, b := range reg.pageOut {
		s.pageOut[id] = b
	}
	for c, o := range reg.ops {
		s.ops[c] = o
	}
	s.adjByTool, s.uncondAdj = byToolScopeIndex(s.adjudicators) // Fix C (pre-call fold)
	s.raByTool, s.uncondRA = byToolScopeIndex(s.resultAdmits)   // Fix D (result-side fold)
	buildEmitterIndex(s)
	published.Store(s)
}

// byToolScopeIndex precomputes a per-tool fold chain for a rank-ordered driver list
// (Fix C for adjudicators, Fix D for result-admitters). A driver implementing
// CallScope with a non-empty Tools() is folded only into those tools; one that does
// not is unconditional (every call) — the fail-closed default. Returns (byTool,
// uncond): byTool[t] is the rank-ordered chain a call for tool t needs; uncond the
// fallback for a tool no scoped driver claims; byTool is nil when no driver is
// scoped. Registration-time only, so its O(tools x drivers) cost never touches a syscall.
func byToolScopeIndex[T any](items []T) (map[string][]T, []T) {
	scoped := make([][]string, len(items)) // nil entry => unconditional driver
	tools := map[string]struct{}{}
	nUncond := 0
	for i, it := range items {
		if cs, ok := any(it).(CallScope); ok {
			if ts := cs.Tools(); len(ts) > 0 {
				scoped[i] = ts
				for _, t := range ts {
					tools[t] = struct{}{}
				}
				continue
			}
		}
		nUncond++
	}
	uncond := make([]T, 0, nUncond)
	for i, it := range items {
		if scoped[i] == nil {
			uncond = append(uncond, it)
		}
	}
	if len(tools) == 0 {
		return nil, uncond // no scoped driver: every call folds the full chain
	}
	byTool := make(map[string][]T, len(tools))
	for t := range tools {
		var lst []T
		for i, it := range items {
			if scoped[i] == nil || containsStr(scoped[i], t) {
				lst = append(lst, it)
			}
		}
		byTool[t] = lst
	}
	return byTool, uncond
}

func containsStr(ss []string, s string) bool {
	for _, x := range ss {
		if x == s {
			return true
		}
	}
	return false
}

// buildEmitterIndex precomputes the per-EventKind fan-out (Fix B): an observer that
// implements EventSubscriber receives only the kinds it lists; one that does not
// (the v0.1 default) is "universal" and receives every kind. Lists keep global
// registration order so emit() preserves the existing ordering; registration-time only.
func buildEmitterIndex(s *snapshot) {
	subs := make([][]EventKind, len(s.emitters)) // nil entry => universal observer
	wanted := map[EventKind]struct{}{}
	universal := 0
	for i, e := range s.emitters {
		if sub, ok := e.(EventSubscriber); ok {
			if ks := sub.Subscriptions(); len(ks) > 0 {
				subs[i] = ks
				for _, k := range ks {
					wanted[k] = struct{}{}
				}
				continue
			}
		}
		universal++
	}
	// Fallback bucket: universal observers, registration order.
	s.allEmitters = make([]Emitter, 0, universal)
	for i, e := range s.emitters {
		if subs[i] == nil {
			s.allEmitters = append(s.allEmitters, e)
		}
	}
	if len(wanted) == 0 {
		return // no selective observers: every kind uses allEmitters (== all)
	}
	s.emittersByKind = make(map[EventKind][]Emitter, len(wanted))
	for k := range wanted {
		var lst []Emitter
		for i, e := range s.emitters {
			if subs[i] == nil || containsKind(subs[i], k) {
				lst = append(lst, e)
			}
		}
		s.emittersByKind[k] = lst
	}
}

func containsKind(ks []EventKind, k EventKind) bool {
	for _, x := range ks {
		if x == k {
			return true
		}
	}
	return false
}

// Register* — the only way to extend the kernel: each is called from a driver's init(), mutates the locked builder state, then rebuilds the snapshot.

// RegisterAdjudicator inserts a PDP/PEP link at a rank (lower rank runs first).
func RegisterAdjudicator(rank int, a Adjudicator) {
	reg.mu.Lock()
	defer reg.mu.Unlock()
	reg.adjudicators = append(reg.adjudicators, rankedAdj{rank, a})
	sort.SliceStable(reg.adjudicators, func(i, j int) bool {
		return reg.adjudicators[i].rank < reg.adjudicators[j].rank
	})
	rebuildSnapshot()
}

// RegisterResultAdmitter inserts a write-time result-admission link at a rank
// (lower rank runs first). This is the context-MMU seam: after the engine produces
// a Result, the kernel folds these to decide whether the result enters context
// as-is (VerdictAllow), is held out (VerdictQuarantine), or is rewritten to a
// pointer (VerdictTransform).
func RegisterResultAdmitter(rank int, ra ResultAdmitter) {
	reg.mu.Lock()
	defer reg.mu.Unlock()
	reg.resultAdmits = append(reg.resultAdmits, rankedRA{rank, ra})
	sort.SliceStable(reg.resultAdmits, func(i, j int) bool {
		return reg.resultAdmits[i].rank < reg.resultAdmits[j].rank
	})
	rebuildSnapshot()
}

// RegisterFastPath adds a vDSO tier (lower tier consulted first).
func RegisterFastPath(tier int, f FastPath) {
	reg.mu.Lock()
	defer reg.mu.Unlock()
	reg.fastpaths = append(reg.fastpaths, rankedFP{tier, f})
	sort.SliceStable(reg.fastpaths, func(i, j int) bool {
		return reg.fastpaths[i].tier < reg.fastpaths[j].tier
	})
	rebuildSnapshot()
}

// RegisterOp claims an OpCode. Panics on a clash (link-time disjointness).
func RegisterOp(o Op) {
	reg.mu.Lock()
	defer reg.mu.Unlock()
	if _, dup := reg.ops[o.Code()]; dup {
		panic(fmt.Sprintf("abi: duplicate OpCode %d", o.Code()))
	}
	reg.ops[o.Code()] = o
	rebuildSnapshot()
}

// RegisterVerdictKind registers an open-range kind with its fold rank + fallback;
// panics if k <= VerdictReservedMax (the closed core set) or on a clash.
func RegisterVerdictKind(k VerdictKind, name string, foldRank int, fb FallbackClass) {
	if k <= VerdictReservedMax {
		panic(fmt.Sprintf("abi: VerdictKind %d is in the closed core range", k))
	}
	reg.mu.Lock()
	defer reg.mu.Unlock()
	if _, dup := reg.verdictKinds[k]; dup {
		panic(fmt.Sprintf("abi: duplicate VerdictKind %d", k))
	}
	reg.verdictKinds[k] = vkInfo{name, foldRank, fb}
	rebuildSnapshot()
}

// RegisterReason adds to the closed refusal vocabulary (the model label space).
// Panics if c <= ReasonCoreMax, on a duplicate ReasonCode, or on a duplicate ReasonName under a different code; exact (code, name) re-registration is a no-op.
func RegisterReason(c ReasonCode, name string) {
	if c <= ReasonCoreMax {
		panic(fmt.Sprintf("abi: ReasonCode %d is in the closed core range", c))
	}
	reg.mu.Lock()
	defer reg.mu.Unlock()
	if existing, dup := reg.reasons[c]; dup {
		if existing != name {
			panic(fmt.Sprintf("abi: duplicate ReasonCode %d", c))
		}
		return
	}
	if _, dup := reasonCodesByName[name]; dup {
		panic(fmt.Sprintf("abi: duplicate ReasonName %q", name))
	}
	for _, existing := range reg.reasons {
		if existing == name {
			panic(fmt.Sprintf("abi: duplicate ReasonName %q", name))
		}
	}
	reg.reasons[c] = name
	rebuildSnapshot()
}

// RegisterCapability adds a negotiable feature token.
func RegisterCapability(c Capability) {
	reg.mu.Lock()
	defer reg.mu.Unlock()
	reg.caps[c] = true
	rebuildSnapshot()
}

// RegisterEmitter adds an observer (KPI / steward / label sink).
func RegisterEmitter(e Emitter) {
	reg.mu.Lock()
	defer reg.mu.Unlock()
	reg.emitters = append(reg.emitters, e)
	rebuildSnapshot()
}

// RegisterEngine adds an inference engine driver keyed by id (local/remote/multi).
func RegisterEngine(id string, d EngineDriver) {
	reg.mu.Lock()
	defer reg.mu.Unlock()
	reg.engines[id] = d
	rebuildSnapshot()
}

// RegisterRegionBackend sets the Ref/Resolver backend (CAS copy now, shared arena later); last registration wins, so a "zerocopy" driver overrides by import order.
func RegisterRegionBackend(b RegionBackend) {
	reg.mu.Lock()
	defer reg.mu.Unlock()
	reg.regionBackend = b
	rebuildSnapshot()
}

// RegisterKVBackend sets the KV-MMU enforcement backend factory (the in-process
// model.Session adapter now, a remote/zero-copy KV backend later). Last registration
// wins, but the value is a per-session FACTORY: a KV cache is per-session state, not
// a stateless global resolver. kvmmu.Context enforces through the factory-built
// KVBackend, so swapping the enforcement medium is a registration, not a kvmmu edit.
func RegisterKVBackend(f KVBackendFactory) {
	reg.mu.Lock()
	defer reg.mu.Unlock()
	reg.kvBackend = f
	rebuildSnapshot()
}

// RegisterPageOutBackend adds an MMU page-out codec (Go blob store default; headroom sidecar later).
func RegisterPageOutBackend(id string, b PageOutBackend) {
	reg.mu.Lock()
	defer reg.mu.Unlock()
	reg.pageOut[id] = b
	rebuildSnapshot()
}

// RegisterWitnessResolver backs the require-witness verdict (pass-through stub in v0.1; DOS read-back later).
func RegisterWitnessResolver(id string, w WitnessResolver) {
	reg.mu.Lock()
	defer reg.mu.Unlock()
	reg.witnesses[id] = w
	rebuildSnapshot()
}

// RegisterSteward adds a single-invariant validator to the population.
func RegisterSteward(s Steward) {
	reg.mu.Lock()
	defer reg.mu.Unlock()
	reg.stewards = append(reg.stewards, s)
	rebuildSnapshot()
}

// RegisterProvisionalSink registers a retractable-effect sink (the MMU); the kernel calls Promote/Rollback across all sinks when an epoch/txn resolves.
func RegisterProvisionalSink(s ProvisionalSink) {
	reg.mu.Lock()
	defer reg.mu.Unlock()
	reg.sinks = append(reg.sinks, s)
	rebuildSnapshot()
}

// Driver interfaces that attach via the registries above (frozen; impls live in driver packages).

// EngineDriver runs a tool call against an inference engine (LiteLLM-remote in v0.1; local vLLM / multi-route are additive registrations).
type EngineDriver interface {
	Complete(ctx context.Context, c *ToolCall) (*Result, error)
	Caps() []Capability
}

// ResultAdmitter is the context-MMU seam (write-time, post-tool): after a tool
// produces a Result, the kernel folds the admitter chain to decide whether the
// result enters context unchanged (VerdictAllow), is held out / paged out
// (VerdictQuarantine, payload QuarantinePayload{PageOut}), or is rewritten to a
// <2KB pointer (VerdictTransform, payload TransformPayload{NewArgs=pointer Ref}).
// It gates the RESULT entering context (Adjudicator gates the CALL); default-absent
// means every result is admitted as-is.
type ResultAdmitter interface {
	Admit(ctx context.Context, c *ToolCall, r *Result) Verdict
	Caps() []Capability
}

// RegionBackend provides the Resolver behind every Ref (the zero-copy seam).
type RegionBackend interface {
	Resolver() Resolver
	Caps() []Capability
}

// PageOutBackend is the context-MMU's swappable codec: a quarantined/cold result
// pages out to a handle and back (Go content-addressed store in v0.1; headroom Rust
// sidecar later). The shared Ref type is the currency, so the swap touches no MMU core.
type PageOutBackend interface {
	PageOut(ctx context.Context, r Ref) (Ref, error) // returns a handle Ref
	PageIn(ctx context.Context, handle Ref) (Ref, error)
}

// KVBackend is the kernel-owned attention-cache the KV-MMU (internal/kvmmu) enforces
// a quarantine verdict on: Len, Prefill (returns next-token logits), Evict [from,n)
// (the re-RoPE / renumber quarantine primitive, returning positions removed), and
// ModelID (keys the cachemeta entry). The in-process model.Session is the v0.1
// default; a remote/zero-copy backend (disaggregated agent memory) attaches the same
// way the region/page-out backends do, so the KV-MMU enforces against engines fak
// does not itself run, with no kvmmu edit. StageSpan / RestoreSpan add a residency
// transfer pair for spans fak does not host locally (a remote / disaggregated L3 KV
// tier), returning typed outcomes instead of dense logits.
type KVBackend interface {
	Len() int                    // live cached positions
	Prefill(ids []int) []float32 // prefill a span; return next-token logits
	Evict(from, n int) int       // evict [from,from+n); return positions removed
	ModelID() string             // model id for the cachemeta cache key

	// StageSpan offloads the [from,from+n) span to a remote / disaggregated residency
	// tier, addressed by digest, returning a typed outcome (ok | MISS | FAULT) — never
	// dense logits, never a silent recompute. The in-process backend keeps the span
	// resident and returns OK (no-op, BytesMoved=0); a remote backend serializes the
	// fak-owned pre-RoPE Kraw rows off-box, so the eviction moat survives
	// disaggregated. ctx bounds a remote stall: a hang surfaces as a FAULT.
	StageSpan(ctx context.Context, digest string, from, n int) (KVResidency, error)

	// RestoreSpan re-materializes a previously staged span by digest: OK on a hit,
	// MISS when the tier no longer holds the span (the caller recomputes — but is
	// told), FAULT on a transport / store error or a ctx deadline. The in-process
	// backend hosts no off-box tier, so it returns a typed MISS rather than
	// silently recomputing.
	RestoreSpan(ctx context.Context, digest string) (KVResidency, error)
}

// KVBackendFactory adapts a session-like value (the in-process *model.Session in
// v0.1) into a KVBackend — the per-session constructor seam: kvmmu.Context enforces
// through a KVBackend instance (one per session), so the registry holds a factory,
// not a process-global backend. The factory reports ok=false for an unrecognized
// value, so a mismatched session type fails closed instead of silently mis-enforcing.
type KVBackendFactory func(session any) (KVBackend, bool)

// WitnessResolver backs the require-witness verdict: confirm a claimed effect from
// evidence the agent did not author (v0.1 registers a pass-through stub; DOS dos_verify / dos-witness-claim is an additive registration).
type WitnessResolver interface {
	Resolve(ctx context.Context, c *ToolCall, claim string) WitnessOutcome
}

// WitnessOutcome is a WitnessResolver's verdict on a claimed effect: abstain (no evidence either way), confirmed (independently corroborated), refuted (contradicted).
type WitnessOutcome uint8

// Witness outcome classes: abstain is the no-either-way fail-to-abstain verdict; confirmed and refuted are the evidence-backed states.
const (
	WitnessAbstain   WitnessOutcome = iota // no evidence either way (fail-to-abstain)
	WitnessConfirmed                       // independently corroborated
	WitnessRefuted                         // contradicted by evidence
)

// Steward is one cheap single-invariant validator (the "10x stewards" population); it returns a violation only with an independently-authored witness, else abstains.
type Steward interface {
	Name() string
	Check(ctx context.Context) (violated bool, witness string)
}

// EventSubscriber is an OPTIONAL interface an Emitter may also implement to scope
// itself to specific EventKinds; the kernel then delivers only those kinds to it
// (see EmittersFor). An Emitter that does not implement it — or returns an empty
// list — receives every event, the v0.1 default, so this is purely additive.
// Subscriptions() is read once at registration (snapshot rebuild), so it must be stable.
type EventSubscriber interface {
	Subscriptions() []EventKind
}

// CallScope is an OPTIONAL interface a fold driver — an Adjudicator (pre-call) or a
// ResultAdmitter (result-side) — may also implement to declare the exact tool names
// it acts on; the kernel then folds it only into calls for those tools (see
// AdjudicatorsFor / ResultAdmittersFor). Not implementing CallScope — or returning
// an empty list — is
// UNCONDITIONAL (consulted on every call): the fail-closed default, because a wrong
// scope is a fail-open bug. Declaring Tools() MUST mean "I return my fold IDENTITY
// verdict for every call whose Tool is not listed" — Defer (no opinion) for an
// Adjudicator, Allow (admit-as-is) for a ResultAdmitter — and the fold treats that
// identity identically to absence, so skipping the driver is verdict-equivalent to
// running it. Tools() is read once at registration (snapshot rebuild), so it must
// be stable.
type CallScope interface {
	Tools() []string
}

// Read-side accessors the kernel uses to walk the registries (frozen): each is a
// single atomic load + index of a pre-built immutable view — no mutex, no allocation,
// O(1) regardless of feature count. Returned slices are registry-owned and immutable — walk them, never mutate them.

// Adjudicators returns the rank-sorted snapshot of the registered adjudicator chain the kernel folds on every call.
func Adjudicators() []Adjudicator { return loadSnapshot().adjudicators }

// AdjudicatorsFor returns the rung chain a call must fold, in rank order: the
// unconditional rungs plus any rung that scoped itself (via CallScope) to this call's
// Tool; when no rung is scoped it returns the full chain, identical to Adjudicators().
// A call for tool T thus folds only the rungs that can possibly refuse T. Immutable
// registry-owned slice; O(1), allocation-free.
func AdjudicatorsFor(c *ToolCall) []Adjudicator {
	s := loadSnapshot()
	if s.adjByTool != nil && c != nil {
		if lst, ok := s.adjByTool[c.Tool]; ok {
			return lst
		}
		return s.uncondAdj
	}
	return s.adjudicators
}

// ScopedFor filters a CALLER-SUPPLIED adjudicator chain down to the rungs a given
// call must fold, applying the same CallScope semantics AdjudicatorsFor applies to
// the global registry — against an explicit chain the caller owns. A rung that
// implements CallScope with a non-empty Tools() is kept only for a call whose Tool
// it lists; a rung that does not (the fail-closed default) is unconditional and
// always kept; with no scoped rung the result is the chain unchanged, allocation-free.
// This is the per-kernel injection seam: a kernel built with an explicit chain
// (kernel.WithAdjudicators) folds ScopedFor(chain, c) instead of AdjudicatorsFor(c),
// so K replay arms can each carry their own monitor and fold concurrently without
// colliding on the process-global registry or its policy.
func ScopedFor(chain []Adjudicator, c *ToolCall) []Adjudicator {
	if c == nil {
		return chain
	}
	scoped := false
	for _, a := range chain {
		if cs, ok := a.(CallScope); ok && len(cs.Tools()) > 0 {
			scoped = true
			break
		}
	}
	if !scoped {
		return chain // no tool-scoped rung: every call folds the whole chain
	}
	out := make([]Adjudicator, 0, len(chain))
	for _, a := range chain {
		if cs, ok := a.(CallScope); ok {
			if ts := cs.Tools(); len(ts) > 0 && !containsStr(ts, c.Tool) {
				continue // scoped to other tools — it would Defer on this call
			}
		}
		out = append(out, a)
	}
	return out
}

// FastPaths returns the registered vDSO fast-path tiers in tier order (the registry's own immutable slice; do not mutate).
func FastPaths() []FastPath { return loadSnapshot().fastpaths }

// ResultAdmitters returns the write-time result-admission chain in rank order.
func ResultAdmitters() []ResultAdmitter { return loadSnapshot().resultAdmits }

// ResultAdmittersFor returns the gate chain a produced result must fold, in rank
// order: the unconditional gates plus any gate that scoped itself (via CallScope) to
// this call's Tool; when no gate is scoped it returns the full chain, identical to
// ResultAdmitters(). Result-side dual of AdjudicatorsFor: a result for tool T folds
// only the gates that can act on T. Immutable registry-owned slice; O(1), allocation-free.
func ResultAdmittersFor(c *ToolCall) []ResultAdmitter {
	s := loadSnapshot()
	if s.raByTool != nil && c != nil {
		if lst, ok := s.raByTool[c.Tool]; ok {
			return lst
		}
		return s.uncondRA
	}
	return s.resultAdmits
}

// LookupOp returns the Op registered for an OpCode, and whether one is registered.
func LookupOp(code OpCode) (Op, bool) {
	o, ok := loadSnapshot().ops[code]
	return o, ok
}

// Emitters returns every registered observer in registration order (the registry's own immutable slice; do not mutate). Use EmittersFor to fan out a single event kind.
func Emitters() []Emitter { return loadSnapshot().emitters }

// EmittersFor returns the observers that should receive an event of the given kind,
// in registration order. This is the fan-out the kernel walks in emit(): an observer
// that declared a subscription via EventSubscriber is included only for its kinds; a
// universal observer for every kind, so an event reaches only its interested
// observers. Immutable registry-owned slice; O(1), allocation-free.
func EmittersFor(kind EventKind) []Emitter {
	s := loadSnapshot()
	if s.emittersByKind != nil {
		if lst, ok := s.emittersByKind[kind]; ok {
			return lst
		}
	}
	return s.allEmitters
}

// Stewards returns the registered single-invariant validators in registration order (the registry's own immutable slice; do not mutate).
func Stewards() []Steward { return loadSnapshot().stewards }

// ProvisionalSinks returns the registered retractable-effect sinks the kernel promotes or rolls back when an epoch resolves.
func ProvisionalSinks() []ProvisionalSink { return loadSnapshot().sinks }

// Witnesses returns the registered WitnessResolvers (the require-witness gate's
// evidence backends) in deterministic id-sorted order, as the registry's own
// immutable copy (do not mutate).
func Witnesses() []WitnessResolver { return loadSnapshot().witnesses }

// FoldRank returns a verdict kind's restrictiveness-lattice position. Core kinds
// have built-in ranks (a constant switch, no lookup); registered kinds use their
// declared foldRank from the snapshot — letting the FROZEN fold order a NEW kind
// without a core edit.
func FoldRank(k VerdictKind) int {
	switch k {
	case VerdictAllow:
		return 0
	case VerdictDefer:
		return 10
	case VerdictIndeterminate:
		return 15
	case VerdictTransform:
		return 20
	case VerdictQuarantine:
		return 30
	case VerdictRequireWitness:
		return 40
	case VerdictDeny:
		return 100 // most restrictive of the core set
	}
	if r, ok := loadSnapshot().foldRanks[k]; ok {
		return r
	}
	return 100 // unknown registered kind: treat as max-restrictive (fail-closed)
}

// Fallback returns how an unaware worker treats an unknown verdict kind.
func Fallback(k VerdictKind) FallbackClass {
	if k <= VerdictReservedMax {
		return FallbackDeny
	}
	if fb, ok := loadSnapshot().fallbacks[k]; ok {
		return fb
	}
	return FallbackDeny
}

// Supported reports whether a capability is registered (used by Negotiate).
func Supported(c Capability) bool { return loadSnapshot().caps[c] }

// ActiveResolver returns the Resolver from the registered RegionBackend (the blob store in v0.1); nil if no backend is registered.
func ActiveResolver() Resolver {
	if b := loadSnapshot().region; b != nil {
		return b.Resolver()
	}
	return nil
}

// KVBackendFor builds the KV-MMU enforcement backend for a session via the
// registered KVBackendFactory (the in-process model.Session adapter by default).
// It returns ok=false when no factory is registered, or when the factory does not
// recognize the session value — both fail-closed, so the KV-MMU never enforces
// against a backend it could not construct. Per-session dual of ActiveResolver.
func KVBackendFor(session any) (KVBackend, bool) {
	if f := loadSnapshot().kvBackend; f != nil {
		return f(session)
	}
	return nil, false
}

// CASPinner is an OPTIONAL capability a Resolver may implement when its backing
// store is BOUNDED (evicts to cap memory). A holder that keeps a Ref and resolves
// it in a LATER call — the vDSO tier-2 cache, the context-MMU's held quarantine
// set (a gated page-in) — pins the digest while the reference is live and unpins on
// its own eviction, so a bounded backend can never drop bytes a live holder will
// still resolve, keeping the vDSO soundness invariant ("a cache hit equals a fresh
// call") intact under eviction. Unbounded backends (and test stubs) omit it;
// PinResolved/UnpinResolved then no-op. Pins are refcounted by digest, so a digest
// shared by several holders survives until the LAST unpin.
type CASPinner interface {
	Pin(digest string)
	Unpin(digest string)
}

// casDigest returns a Ref's CAS digest IFF its bytes live in the backend store (RefBlob / RefRegion); inline Refs carry their own bytes and are never pinned.
func casDigest(r Ref) (string, bool) {
	if (r.Kind == RefBlob || r.Kind == RefRegion) && r.Digest != "" {
		return r.Digest, true
	}
	return "", false
}

// PinResolved pins the CAS bytes a Ref points at, IF the Ref is backend-resident and
// the active resolver is a CASPinner; a no-op otherwise. Call it when a holder takes
// a long-lived reference, BEFORE that reference can be resolved by anyone else, so a
// concurrent eviction cannot win the race.
func PinResolved(r Ref) {
	if d, ok := casDigest(r); ok {
		if p, ok := ActiveResolver().(CASPinner); ok {
			p.Pin(d)
		}
	}
}

// UnpinResolved releases a PinResolved when the holder drops its reference (its own cache/quarantine eviction); a no-op if the Ref is inline or the resolver does not bound its store.
func UnpinResolved(r Ref) {
	if d, ok := casDigest(r); ok {
		if p, ok := ActiveResolver().(CASPinner); ok {
			p.Unpin(d)
		}
	}
}

// PageOut returns a registered page-out backend by id (the MMU codec).
func PageOut(id string) (PageOutBackend, bool) {
	b, ok := loadSnapshot().pageOut[id]
	return b, ok
}

// Engine returns the engine driver bound to id; if id=="" it returns any single registered engine (deterministic: the lowest-id engine), else nil.
func Engine(id string) EngineDriver {
	s := loadSnapshot()
	if id != "" {
		return s.engines[id]
	}
	return s.anyEngine
}

// EngineIDs lists the registered engine ids (forensics / selection), id-sorted.
func EngineIDs() []string { return loadSnapshot().engineIDs }
