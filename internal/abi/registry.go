// registry.go — the extension mechanism of the frozen ABI.
//
// This file is part of the wave-0 freeze and is additive-only. It is the LSM
// driver-registration surface: every v0.1 subsystem and every future idea
// attaches by calling a Register* function from its OWN package's init(). The
// kernel WALKS these registries; it never imports a driver. A tiny
// internal/registrations package blank-imports the drivers that are built in (the
// Linux "built-in driver list"), so enabling/disabling an idea is one import line.
//
// DISJOINTNESS GUARANTEE: opcode / extkey / verdict-kind / event-kind numbers are
// claimed from RESERVED PER-SUBSYSTEM RANGES. Register* panics on a clash (a
// duplicate opcode is a link-time error, exactly like a duplicate syscall
// number), so two independently-built leaf packages can never collide even when
// different fleet workers add them in parallel. The dependency graph is a STAR:
// abi at the center, every driver a leaf depending only on abi, never on each
// other — which is what keeps the fleet trees disjoint as ideas are added.
//
// SCALING CONTRACT (the whole reason the read side looks the way it does): the
// registries are written ONCE at init() (one Register* per built-in/enabled idea)
// and then read on EVERY syscall. So the design rule is "writes may be expensive,
// reads must be O(1) and wait-free regardless of how many ideas are registered."
// Writers mutate the locked builder state (`reg`) and then rebuild ONE immutable
// `snapshot`, published via an atomic pointer. Readers do a single atomic load and
// index a pre-built slice/map — NO mutex, NO per-call allocation. Adding the 1000th
// adjudicator therefore costs the 1st syscall nothing in framework overhead; the
// only per-call cost that grows with feature count is the unavoidable "actually run
// each registered driver," and even the event fan-out is indexed by kind
// (EmittersFor) so an observer only runs for the events it subscribed to.
package abi

import (
	"context"
	"fmt"
	"sort"
	"sync"
	"sync/atomic"
)

// ---------------------------------------------------------------------------
// RESERVED RANGES — the static disjointness contract.
// A new subsystem claims a [lo, hi) block here (additive-only) and draws its
// OpCode/ExtKey/VerdictKind/EventKind numbers from it. Two leaves with disjoint
// blocks cannot collide; Register* enforces it at init() time.
// ---------------------------------------------------------------------------

type Range struct{ Lo, Hi uint32 }

var (
	// OpCode blocks.
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

	// VerdictKind blocks ABOVE VerdictReservedMax (1023). Below is the closed
	// trainable core set; above is open/registered.
	VerdictsVendor = Range{1024, 1 << 16}

	// EventKind blocks.
	EventsCore   = Range{0, 64}    // submit/decide/dispatch/complete
	EventsKPI    = Range{64, 128}  // metric taps
	EventsLabel  = Range{128, 192} // label-bearing (typed LabelRow)
	EventsVendor = Range{1 << 16, 1 << 17}
)

func inRange(n uint32, r Range) bool { return n >= r.Lo && n < r.Hi }

// ResetForTest clears every registry to empty. It exists ONLY for tests that need
// to assemble a controlled driver set in isolation; it is never called in
// production (the registries are populated once from init() and then read-only).
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
	reg.pageOut = map[string]PageOutBackend{}
	reg.witnesses = map[string]WitnessResolver{}
	reg.stewards = nil
	reg.sinks = nil
	reg.screens = nil
	rebuildSnapshot()
}

// ---------------------------------------------------------------------------
// The builder state. Guarded by one mutex and touched ONLY by the write side
// (Register* / ResetForTest), which runs at init() before the kernel boots. The
// read side never touches this — it reads the published immutable snapshot below.
// ---------------------------------------------------------------------------

var reg = struct {
	mu sync.Mutex

	adjudicators  []rankedAdj   // LSM chain, sorted by rank ascending
	resultAdmits  []rankedRA    // write-time result-admission chain (the MMU seam)
	fastpaths     []rankedFP    // vDSO tiers, sorted by tier ascending
	ops           map[OpCode]Op // operation table
	emitters      []Emitter     // KPI / steward / label sinks
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

// vkInfo is what a registered verdict kind declares. CRITICAL: it includes a
// FoldRank so the core fold can ORDER a new kind in the restrictiveness lattice
// without a core edit. (Without this, a new combinator-verdict would force a
// change to the fold function — the sharpest extensibility trap.)
type vkInfo struct {
	name     string
	foldRank int           // position in the restrictiveness lattice (higher = wins)
	fallback FallbackClass // how an UNAWARE worker treats this kind
}

// ---------------------------------------------------------------------------
// The published READ SIDE: an immutable snapshot. rebuildSnapshot() (write side,
// init-time) constructs a fresh one from `reg` and atomically publishes it; every
// reader does published.Load() and indexes pre-built slices/maps with no lock and
// no allocation. The snapshot is never mutated after publish, so a reader holding
// an old snapshot across a concurrent registration is race-free — it simply sees
// the pre-registration view (registration in production happens before any read).
// ---------------------------------------------------------------------------

type snapshot struct {
	adjudicators []Adjudicator // rank-sorted, pre-extracted from rankedAdj (ALL rungs)

	// Adjudicator fan-out index (Fix C). The fold must not be O(all rungs) when
	// most rungs are tool-scoped: adjByTool[t] is the rank-ordered chain a call for
	// tool t actually needs (unconditional rungs merged with the rungs scoped to t),
	// and uncondAdj is the fallback for a tool no scoped rung claims. A rung that
	// does NOT implement CallScope is unconditional (always run) — the fail-CLOSED
	// default — so if no rung opts in, adjByTool is nil and AdjudicatorsFor returns
	// the full chain, reproducing v0.1 exactly. Skipping a scoped rung for a
	// non-matching tool is verdict-equivalent to running it: it declared (via
	// Tools()) that it would Defer on that tool, and the fold treats Defer and
	// absence identically (both resolve to default-deny when nothing else opines).
	adjByTool map[string][]Adjudicator
	uncondAdj []Adjudicator

	fastpaths    []FastPath       // tier-sorted, pre-extracted from rankedFP
	resultAdmits []ResultAdmitter // rank-sorted, pre-extracted from rankedRA (ALL gates)

	// Result-admit fan-out index (Fix D) — the exact dual of adjByTool/uncondAdj.
	// admitResult folds the result-side gate chain on EVERY produced result, so it
	// must not be O(all gates): raByTool[t] is the rank-ordered gate chain a result
	// for tool t needs (unconditional gates merged with the gates scoped to t), and
	// uncondRA is the fallback for a tool no scoped gate claims. A gate that does not
	// implement CallScope is unconditional/always-run — the fail-CLOSED default (a
	// result admitter is the exfil/quarantine floor), so if no gate opts in raByTool
	// is nil and ResultAdmittersFor returns the full chain, reproducing v0.1 exactly.
	raByTool map[string][]ResultAdmitter
	uncondRA []ResultAdmitter

	emitters []Emitter // registration order (ALL observers; Emitters())

	// Event fan-out index (Fix B). emit() is called several times per syscall and
	// must not be O(all observers): emittersByKind[k] is the registration-ordered
	// list of observers that receive kind k (universal observers merged with the
	// selective ones that subscribed to k). allEmitters is the fallback bucket
	// (universal observers only) for any kind no selective observer asked for. If
	// no observer is selective, emittersByKind is nil and allEmitters == emitters,
	// so EmittersFor reproduces the v0.1 "everyone gets everything" behavior exactly.
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

// emptySnapshot backs reads that happen before the first registration. All its
// fields are zero, and ranging a nil slice / reading a nil map is well-defined in
// Go, so every accessor behaves exactly as "nothing registered."
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
// publishes it atomically. MUST be called with reg.mu held — every Register* and
// ResetForTest calls it as its last act while still holding the lock. It runs only
// at registration time (rare), so its O(features) build cost never touches a
// syscall.
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
	// Witnesses: map -> id-sorted slice, so the require-witness gate consults
	// resolvers in a deterministic order (the old map-range order was random).
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

// byToolScopeIndex precomputes a per-tool fold chain for any rank-ordered driver
// list (Fix C for adjudicators, Fix D for result-admitters). A driver that
// implements CallScope with a non-empty Tools() is folded ONLY into those tools;
// a driver that does not is unconditional (folded into every call) — the
// fail-CLOSED default. Returns (byTool, uncond): byTool[t] is the rank-ordered
// chain a call for tool t needs (unconditional ++ scoped-to-t, registration order
// preserved); uncond is the fallback for a tool no scoped driver claims. byTool is
// nil when no driver is scoped, so the caller returns the full chain — reproducing
// v0.1 exactly. Runs at registration time only, so its O(tools x drivers) build
// cost never touches a syscall. One primitive backs every tool-scoped fold, so the
// next one is a single call, not another hand-rolled bucketer.
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

// buildEmitterIndex precomputes the per-EventKind fan-out (Fix B). An observer
// that implements EventSubscriber receives ONLY the kinds it lists; one that does
// not (the v0.1 default) is "universal" and receives every kind. The lists are
// built in global registration order so emit() preserves the existing ordering.
// Runs at registration time only.
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

// ---------------------------------------------------------------------------
// Register* — the ONLY way to extend the kernel. Called from a driver's init().
// Each writer mutates the locked builder state then rebuilds the snapshot.
// ---------------------------------------------------------------------------

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

// RegisterResultAdmitter inserts a WRITE-TIME result-admission link at a rank
// (lower rank runs first). This is the context-MMU seam: after the engine
// produces a Result, the kernel folds these to decide whether the result may
// enter context as-is (VerdictAllow), must be held out (VerdictQuarantine), or
// rewritten to a pointer (VerdictTransform). It is the post-tool dual of the
// pre-call Adjudicator chain. Additive to the frozen registry surface.
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

// RegisterVerdictKind registers an open-range kind with its fold rank + fallback.
// Panics if k <= VerdictReservedMax (the closed core set) or on a clash.
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
func RegisterReason(c ReasonCode, name string) {
	reg.mu.Lock()
	defer reg.mu.Unlock()
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

// RegisterRegionBackend sets the Ref/Resolver backend (CAS copy now, shared arena
// later). The last registration wins, so a "zerocopy" driver overrides the
// default by blank-import order.
func RegisterRegionBackend(b RegionBackend) {
	reg.mu.Lock()
	defer reg.mu.Unlock()
	reg.regionBackend = b
	rebuildSnapshot()
}

// RegisterKVBackend sets the KV-MMU enforcement backend factory (the in-process
// model.Session adapter now, a remote/zero-copy KV backend later). The last
// registration wins, so a disaggregated KV driver overrides the default by
// blank-import order — exactly like RegisterRegionBackend, but the registered value
// is a per-session FACTORY (a KV cache is per-session state, not a stateless global
// resolver). kvmmu.Context enforces its quarantine through the KVBackend the factory
// builds, so inverting the enforcement medium is a registration, not a kvmmu edit.
func RegisterKVBackend(f KVBackendFactory) {
	reg.mu.Lock()
	defer reg.mu.Unlock()
	reg.kvBackend = f
	rebuildSnapshot()
}

// RegisterPageOutBackend adds an MMU page-out codec (Go blob store default;
// headroom sidecar later).
func RegisterPageOutBackend(id string, b PageOutBackend) {
	reg.mu.Lock()
	defer reg.mu.Unlock()
	reg.pageOut[id] = b
	rebuildSnapshot()
}

// RegisterWitnessResolver backs the require-witness verdict (no-op pass-through
// stub in v0.1; DOS read-back later).
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

// RegisterProvisionalSink registers a retractable-effect sink (the MMU). The
// kernel calls Promote/Rollback across all sinks when an epoch/txn resolves.
func RegisterProvisionalSink(s ProvisionalSink) {
	reg.mu.Lock()
	defer reg.mu.Unlock()
	reg.sinks = append(reg.sinks, s)
	rebuildSnapshot()
}

// ---------------------------------------------------------------------------
// Driver interfaces that attach via the registries above. (Frozen; impls live in
// driver packages.)
// ---------------------------------------------------------------------------

// EngineDriver runs a tool call against an inference engine (LiteLLM-remote in
// v0.1; local vLLM / multi-route are additive registrations).
type EngineDriver interface {
	Complete(ctx context.Context, c *ToolCall) (*Result, error)
	Caps() []Capability
}

// ResultAdmitter is the context-MMU seam (write-time, post-tool). After a tool
// produces a Result, the kernel folds the admitter chain to decide whether the
// result enters context unchanged (VerdictAllow), is held out / paged out
// (VerdictQuarantine, payload QuarantinePayload{PageOut}), or is rewritten to a
// <2KB pointer (VerdictTransform, payload TransformPayload{NewArgs=pointer Ref}).
// It is the dual of Adjudicator: Adjudicator gates the CALL, ResultAdmitter gates
// the RESULT entering context. Default-absent means every result is admitted
// as-is.
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
// pages out to a handle and back. Go content-addressed store is v0.1 default;
// headroom (Rust sidecar) is a later registration. The shared Ref type is the
// currency, so the swap touches no MMU core.
type PageOutBackend interface {
	PageOut(ctx context.Context, r Ref) (Ref, error) // returns a handle Ref
	PageIn(ctx context.Context, handle Ref) (Ref, error)
}

// KVBackend is the kernel-owned attention-cache the KV-MMU (internal/kvmmu)
// ENFORCES a quarantine verdict on. It captures EXACTLY the operations the bridge
// performs on a session's cache: read the live length, prefill a token span (and
// read the next-token logits), evict a span by [from,len) (the re-RoPE / renumber
// quarantine primitive, returning the count of positions removed), and report the
// model id used to key the cachemeta entry. The in-process model.Session is the
// v0.1 default (model registers a wrapping adapter); a remote/zero-copy backend —
// the disaggregated-agent-memory direction — is an additive registration that
// attaches the SAME way the region/page-out backends do, so the KV-MMU enforces
// against an engine fak does not itself run with no edit to the kvmmu composer.
//
// The residency-transfer pair (StageSpan / RestoreSpan) widens the seam for a span
// fak does NOT host locally: a remote / disaggregated L3 KV tier. Unlike Prefill —
// which returns a dense logits vector a network backend cannot serve at line rate —
// they return a TYPED outcome (ok | MISS | FAULT) addressed by digest, with a ctx so
// a remote stall is a typed FAULT rather than a hang. The in-process backend
// implements them as the local synchronous path (the span is already resident), so
// adding them does not change Prefill / Evict / Len / ModelID behavior for the
// inkernel engine; they unblock a remote L3 KV backend without shipping its transport.
type KVBackend interface {
	Len() int                    // live cached positions
	Prefill(ids []int) []float32 // prefill a span; return next-token logits
	Evict(from, n int) int       // evict [from,from+n); return positions removed
	ModelID() string             // model id for the cachemeta cache key

	// StageSpan offloads the [from,from+n) span to a remote / disaggregated residency
	// tier, addressing it by digest, and returns a TYPED outcome (ok | MISS | FAULT) —
	// never dense logits, never a silent recompute. The in-process backend keeps the
	// span resident locally and returns OK (a no-op, BytesMoved=0); a remote backend
	// serializes the fak-owned pre-RoPE Kraw rows off-box, so the eviction moat
	// survives disaggregated. ctx bounds a remote stall: a hang surfaces as a FAULT.
	StageSpan(ctx context.Context, digest string, from, n int) (KVResidency, error)

	// RestoreSpan re-materializes a previously staged span by digest. OK on a hit,
	// MISS when the tier no longer holds the span (the caller recomputes — but is
	// TOLD), FAULT on a transport / store error or a ctx deadline. The in-process
	// backend hosts no off-box tier, so it returns a typed MISS rather than silently
	// recomputing.
	RestoreSpan(ctx context.Context, digest string) (KVResidency, error)
}

// KVBackendFactory adapts a session-like value (the in-process *model.Session in
// v0.1) into a KVBackend. It is the per-session constructor seam: kvmmu.Context
// enforces through a KVBackend INSTANCE (one per session), so the registry holds a
// FACTORY that wraps a concrete session rather than a single process-global backend
// (the region/page-out backends are stateless and global; a KV cache is per-session
// state). The factory reports ok=false for a value it does not recognize, so a
// mismatched session type fails closed instead of silently mis-enforcing.
type KVBackendFactory func(session any) (KVBackend, bool)

// WitnessResolver backs the require-witness verdict: confirm a claimed effect
// from evidence the agent did not author. v0.1 registers a pass-through stub;
// DOS dos_verify / dos-witness-claim is an additive registration.
type WitnessResolver interface {
	Resolve(ctx context.Context, c *ToolCall, claim string) WitnessOutcome
}

// WitnessOutcome is a WitnessResolver's verdict on a claimed effect: abstain when there
// is no evidence either way, confirmed when independently corroborated, refuted when
// contradicted.
type WitnessOutcome uint8

const (
	WitnessAbstain   WitnessOutcome = iota // no evidence either way (fail-to-abstain)
	WitnessConfirmed                       // independently corroborated
	WitnessRefuted                         // contradicted by evidence
)

// Steward is one cheap single-invariant validator (the "10x stewards"
// population). It never blocks on its own opinion: it returns a violation only
// with an independently-authored witness, else abstains.
type Steward interface {
	Name() string
	Check(ctx context.Context) (violated bool, witness string)
}

// EventSubscriber is an OPTIONAL interface an Emitter may also implement to scope
// itself to specific EventKinds. The kernel then delivers ONLY those kinds to it
// (see EmittersFor). An Emitter that does not implement EventSubscriber — or
// returns an empty list — receives EVERY event, which is the v0.1 default, so this
// is purely additive: existing observers are unaffected. It exists so the event
// fan-out stays cheap as observers accumulate: a KPI tap that only cares about
// EvDeny should not be invoked on every EvSubmit. Subscriptions() is read once at
// registration time (snapshot rebuild), so it must be stable.
type EventSubscriber interface {
	Subscriptions() []EventKind
}

// CallScope is an OPTIONAL interface a fold driver — an Adjudicator (pre-call) OR a
// ResultAdmitter (result-side) — may also implement to declare the exact tool names
// it acts on. The kernel then folds it ONLY into calls for those tools (see
// AdjudicatorsFor / ResultAdmittersFor), so a per-tool driver (a rate limiter for
// one tool, a result gate for one tool) does not run on every unrelated call. A
// driver that does NOT implement CallScope — or returns an empty list — is
// UNCONDITIONAL and consulted on every call. That is the fail-CLOSED default:
// scoping is a deliberate opt-in, and the contract is strict — declaring Tools()
// MUST mean "I return my fold IDENTITY verdict for every call whose Tool is not
// listed": Defer (no opinion) for an Adjudicator, Allow (admit-as-is) for a
// ResultAdmitter. The fold treats that identity identically to absence, so skipping
// the driver is verdict-equivalent to running it. A wrong scope is a fail-OPEN bug,
// which is why the default is always-run. Tools() is read once at registration
// (snapshot rebuild), so it must be stable.
type CallScope interface {
	Tools() []string
}

// ---------------------------------------------------------------------------
// Read-side accessors the kernel uses to WALK the registries (frozen). Each is a
// single atomic load + index of a pre-built immutable view: no mutex, no
// allocation, O(1) framework overhead independent of how many ideas are
// registered. The returned slices are OWNED BY THE REGISTRY and immutable — walk
// them, never mutate them (every caller only ranges, which is safe and lock-free).
// ---------------------------------------------------------------------------

func Adjudicators() []Adjudicator { return loadSnapshot().adjudicators }

// AdjudicatorsFor returns the rung chain a call must fold, in rank order, as the
// registry's own immutable slice (do not mutate): the unconditional rungs plus any
// rung that scoped itself (via CallScope) to this call's Tool. When no rung is
// scoped it returns the full chain, identical to Adjudicators(). The lookup is a
// single atomic load + map index — O(1) and allocation-free regardless of how many
// tool-scoped rungs are registered, so a call for tool T folds only the rungs that
// can possibly refuse T, not the 100 rungs scoped to other tools.
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
// call must fold, applying the SAME CallScope semantics AdjudicatorsFor applies to
// the global registry — but against an explicit chain the caller owns, touching
// NOTHING in the process-global registry. A rung that implements CallScope with a
// non-empty Tools() is kept ONLY for a call whose Tool it lists; a rung that does
// not (the fail-CLOSED default) is unconditional and always kept. With no scoped
// rung in the chain the result is the chain unchanged, so the common case (a single
// unconditional monitor) is returned verbatim and allocation-free.
//
// This is the per-kernel injection seam: a kernel constructed with an EXPLICIT
// adjudicator chain (see kernel.WithAdjudicators) folds ScopedFor(chain, c) instead
// of AdjudicatorsFor(c), so K replay arms can each carry their own monitor and fold
// CONCURRENTLY without colliding on the process-global registry or its policy.
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

// FastPaths returns the registered vDSO fast-path tiers in tier order (the registry's own
// immutable slice; do not mutate).
func FastPaths() []FastPath { return loadSnapshot().fastpaths }

// ResultAdmitters returns the write-time result-admission chain in rank order.
func ResultAdmitters() []ResultAdmitter { return loadSnapshot().resultAdmits }

// ResultAdmittersFor returns the gate chain a produced result must fold, in rank
// order, as the registry's own immutable slice (do not mutate): the unconditional
// gates plus any gate that scoped itself (via CallScope) to this call's Tool. When
// no gate is scoped it returns the full chain, identical to ResultAdmitters(). The
// lookup is a single atomic load + map index — O(1) and allocation-free regardless
// of how many tool-scoped result gates are registered. It is the result-side dual
// of AdjudicatorsFor: a result for tool T folds only the gates that can act on T.
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

// Emitters returns every registered observer in registration order (the registry's own
// immutable slice; do not mutate). Use EmittersFor to fan out a single event kind.
func Emitters() []Emitter { return loadSnapshot().emitters }

// EmittersFor returns the observers that should receive an event of the given
// kind, in registration order, as the registry's own immutable slice (do not
// mutate). This is the fan-out the kernel walks in emit(): an observer that
// declared a subscription via EventSubscriber is included only for its kinds; a
// universal observer is included for every kind. The lookup is a single atomic
// load + map index — O(1) and allocation-free regardless of observer count, so an
// event reaches only its interested observers no matter how many ideas register.
func EmittersFor(kind EventKind) []Emitter {
	s := loadSnapshot()
	if s.emittersByKind != nil {
		if lst, ok := s.emittersByKind[kind]; ok {
			return lst
		}
	}
	return s.allEmitters
}

// Stewards returns the registered single-invariant validators in registration order (the
// registry's own immutable slice; do not mutate).
func Stewards() []Steward { return loadSnapshot().stewards }

func ProvisionalSinks() []ProvisionalSink { return loadSnapshot().sinks }

// Witnesses returns the registered WitnessResolvers (the require-witness gate's
// evidence backends) in deterministic id-sorted order. The kernel consults these
// to corroborate a RequireWitness verdict's claimed effect from evidence the agent
// did not author. The slice is the registry's own immutable copy (do not mutate);
// additive (mirrors Stewards/Emitters/ProvisionalSinks).
func Witnesses() []WitnessResolver { return loadSnapshot().witnesses }

// FoldRank returns a verdict kind's restrictiveness-lattice position. Core kinds
// have built-in ranks (a constant switch, no lookup); registered kinds use their
// declared foldRank from the snapshot. This is what lets the FROZEN fold order a
// NEW kind without a core edit.
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

// ActiveResolver returns the Resolver from the registered RegionBackend (the blob
// store in v0.1). Returns nil if no backend is registered.
func ActiveResolver() Resolver {
	if b := loadSnapshot().region; b != nil {
		return b.Resolver()
	}
	return nil
}

// KVBackendFor builds the KV-MMU enforcement backend for a session via the
// registered KVBackendFactory (the in-process model.Session adapter by default).
// It returns ok=false when no factory is registered, or when the registered factory
// does not recognize the session value — both fail-closed, so the KV-MMU never
// enforces against a backend it could not construct. This is the read side of the
// RegisterKVBackend seam, the per-session dual of ActiveResolver.
func KVBackendFor(session any) (KVBackend, bool) {
	if f := loadSnapshot().kvBackend; f != nil {
		return f(session)
	}
	return nil, false
}

// CASPinner is an OPTIONAL capability a Resolver may implement when its backing
// store is BOUNDED (evicts to cap memory). A holder that keeps a Ref and resolves
// it in a LATER call — the vDSO tier-2 cache (a hit returns a stored Ref the
// consumer resolves), the context-MMU's held quarantine set (a gated page-in) —
// pins the digest while the reference is live and unpins on its own eviction, so a
// bounded backend can never drop bytes a live holder will still resolve. This keeps
// the vDSO soundness invariant ("a cache hit equals a fresh call") and the gated
// page-in intact under eviction. An unbounded backend (or a test stub) simply does
// not implement it, and PinResolved/UnpinResolved degrade to no-ops. Pins are
// refcounted by digest: content-addressed dedup means several holders can share one
// digest, so it survives until the LAST unpin.
type CASPinner interface {
	Pin(digest string)
	Unpin(digest string)
}

// casDigest returns a Ref's CAS digest IFF its bytes live in the backend store
// (RefBlob / RefRegion). Inline Refs carry their own bytes and are never pinned.
func casDigest(r Ref) (string, bool) {
	if (r.Kind == RefBlob || r.Kind == RefRegion) && r.Digest != "" {
		return r.Digest, true
	}
	return "", false
}

// PinResolved pins the CAS bytes a Ref points at, IF the Ref is backend-resident and
// the active resolver is a CASPinner. A no-op otherwise. Call it at the moment a
// holder takes a long-lived reference, BEFORE that reference can be resolved by
// anyone else, so a concurrent eviction cannot win the race.
func PinResolved(r Ref) {
	if d, ok := casDigest(r); ok {
		if p, ok := ActiveResolver().(CASPinner); ok {
			p.Pin(d)
		}
	}
}

// UnpinResolved releases a PinResolved, at the point the holder drops its reference
// (its own cache/quarantine eviction). A no-op if the Ref is inline or the resolver
// does not bound its store.
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

// Engine returns the engine driver bound to id. If id=="" it returns any single
// registered engine (deterministic: the lowest-id engine), else nil.
func Engine(id string) EngineDriver {
	s := loadSnapshot()
	if id != "" {
		return s.engines[id]
	}
	return s.anyEngine
}

// EngineIDs lists the registered engine ids (forensics / selection), id-sorted.
func EngineIDs() []string { return loadSnapshot().engineIDs }
