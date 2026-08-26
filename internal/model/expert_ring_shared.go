package model

import (
	"errors"
	"fmt"
	"sync"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/polymodel"
)

// expert_ring_shared.go — R7 of the activated-expert offload ladder (#5618, epic #5606,
// docs/MOE-ACTIVATED-OFFLOAD-PLAN.md), and the missing mechanism under epic #5243's L2 lever: lift
// the bounded routed-expert ring from CONVERSATION scope to (model, device) scope, so B concurrent
// agents' activated sets land on ONE residency object and a page-in charged once is served many
// times.
//
// The gap this closes. #5243's thesis is that with B agents the union U(B) of experts a batch
// touches is far smaller than B*K, so aggregate throughput scales with B. That is a claim about a
// SHARED object, and fak had none: before R0 each session carried its own never-evicting halW
// memoizer, and R0's ring — while bounded, accounted and evicting — is still built lazily per
// Session (expert_ring_hal.go). Two agents on the same model paged the same expert twice, and no
// counter could have told you, because "which agent demanded this weight" was not a thing the ring
// knew. So L2 could not be measured, let alone optimized.
//
// What is shared and what is NOT. Routed-expert weights are model-constant bytes: expert 41's
// down_proj is the same bytes for every agent, so residency for it is a property of the (model,
// device) pair. NOTHING else moves. KV cache, conversation state, the sampler, halW, the per-session
// dense weights — all stay exactly where they were, per Session. Attach enforces the boundary at the
// only place it can be enforced rather than asserted: a session may attach only if it serves the
// SAME *Model and runs on the SAME Backend, because "the same bytes" is what makes sharing correct
// and a pointer comparison is the one identity that cannot be spoofed by a name collision. A ring
// shared between two different models would serve model A's expert bytes to model B's GEMM — the
// most damaging failure this rung could have, and the reason the check is a refusal and not a log.
//
// Concurrency discipline. A Session is single-goroutine today (halW is a plain map, no forward runs
// twice at once) and stays that way; what R7 adds is safety for DISTINCT sessions touching one ring
// concurrently. Every session entry point that reaches ring state serializes on this object's mutex
// through Session.ringEnter, which is REENTRANT per session — the demand path enters once around the
// whole three-projection expert acquisition, and weightHALStagedBounded enters again underneath it.
// pagedRing's own methods take no locks at all: the discipline is "the session-level entry points
// lock, the ring methods never do", so there is exactly one place to audit and no lock ordering to
// get wrong.
//
// The stage->hold window. Under a private ring, stage() returning a handle and hold() pinning it are
// adjacent statements with nothing in between. Under a SHARED ring another session's stage could
// land between them, evict the just-staged handle and Free it — a use-after-free the per-session
// ring could not produce. expertSwiGLUHAL therefore takes the span around BOTH (hal.go), which is
// why the lock has to be reentrant.
//
// The meter. The ledger below answers #5243's question directly rather than by simulation: every
// residency span records which agent PAID for the page-in and which agents were then SERVED off it.
// AgentsPerPageIn — distinct agents served per page-in — is exactly B*K/|U(B)| measured on real
// routing: it is 1.0 by construction for a single agent, and rises toward B as the batch's activated
// sets overlap. CrossAgentHits is the raw count of demands satisfied by bytes another agent paid
// for, which is the part plain temporal reuse cannot explain. Wiring these onto an operator surface
// is R6/#5617; the plan sanctions a private meter here so the bench schedule is not blocked on it.

// Errors Attach and Close return. They are sentinels because the model/backend mismatch is a safety
// refusal a caller must be able to branch on, not a diagnostic string.
var (
	// ErrSharedRingClosed means the ring has been torn down; its device handles are gone.
	ErrSharedRingClosed = errors.New("model: shared expert ring is closed")
	// ErrSharedRingModel means the session serves a different *Model. Sharing residency across models
	// would serve one model's expert bytes to another's GEMM.
	ErrSharedRingModel = errors.New("model: session serves a different model than the shared expert ring")
	// ErrSharedRingBackend means the session runs on a different Backend. A device handle is only
	// valid on the device that produced it.
	ErrSharedRingBackend = errors.New("model: session runs on a different backend than the shared expert ring")
	// ErrSharedRingAgent means the agent name was empty or is already attached. The name is the
	// ledger's identity for "a distinct agent", so a duplicate would silently understate coalescing.
	ErrSharedRingAgent = errors.New("model: agent name is empty or already attached to the shared expert ring")
	// ErrSharedRingResidency means the session already has routed-expert residency of its own — either
	// a private ring it already built or another shared ring. Re-pointing it would strand handles.
	ErrSharedRingResidency = errors.New("model: session already has routed-expert residency")
	// ErrSharedRingBusy means Close was called while agents are still attached. Freeing device
	// handles under a live session is a use-after-free, so the ring refuses rather than races.
	ErrSharedRingBusy = errors.New("model: shared expert ring still has attached agents")
)

// SharedExpertRingConfig declares the (model, device) pair a shared ring belongs to and the byte
// budget it holds routed experts under. Budget semantics are R0's exactly: it bounds the RESIDENT
// device footprint of routed-expert weights, in the quantized sizes they are actually staged at.
type SharedExpertRingConfig struct {
	// Model and Backend are the identity every attaching session must match.
	Model   *Model
	Backend compute.Backend
	// BudgetBytes is the resident ceiling. A budget <= 0 is refused rather than clamped: a shared
	// ring that admits nothing would report perfect coalescing over zero residency, which is the one
	// reading of this meter that is both plausible and meaningless.
	//
	// Sizing gains a constraint the per-session ring did not have. Each agent HOLDS its current
	// expert's three projections pinned for the span of that expert's GEMMs (hal.go), so with B
	// agents decoding at once the budget must exceed B x 3 projections or an agent's staging will
	// find every resident pinned by its peers (ErrPinnedNoRoom), be refused, and fall back to
	// PERMANENT halW residency — safe, and correct, but unbounded, which is what the ring exists to
	// stop. SharedExpertRingStats.Refusals counts exactly this, so an undersized budget is visible
	// rather than silently converging on the pre-R0 memoizer.
	BudgetBytes int64
	// Evict is R4/#5615's victim ranking, fixed for the ring's life for the reason R4 gives — scoring
	// one window under two policies makes the next measurement uninterpretable. The zero value is LRU.
	Evict ExpertRingEvictPolicy
}

// SharedExpertRing is the bounded routed-expert residency of ONE (model, device) pair, shared by
// every agent serving it. See the file header for what is shared, what is not, and why the identity
// check is a refusal.
//
// It is safe for concurrent use by distinct sessions. It is NOT a license to run one Session from
// two goroutines: that was never supported and is not what coalescing needs.
type SharedExpertRing struct {
	mu   sync.Mutex
	m    *Model
	be   compute.Backend
	ring *pagedRing

	// attached is the live agent name -> session map; peakAttached is the high-water mark of its
	// size, which is the B a coalescing ratio must be read against (a ratio of 3.0 means nothing
	// without knowing whether 3 or 30 agents produced it).
	attached     map[string]*Session
	peakAttached int
	closed       bool

	// inSpan is the agent name of the session currently inside a ring span, written under mu at the
	// span boundary and read only by the ledger hooks that run inside it. It is how a pagedRing
	// method — which knows nothing about sessions — attributes a demand to an agent.
	inSpan string

	ledger sharedRingLedger
}

// sharedRingLedger is the cross-agent coalescing account. Every field is guarded by
// SharedExpertRing.mu and is written only from inside a ring span, so the hooks below take no locks
// of their own.
//
// The unit of account is a RESIDENCY SPAN: the window between a weight being paged in and being
// evicted. Each span records the agent that paid for the page-in and the set of agents served off
// it. That framing is what separates cross-agent coalescing from ordinary temporal reuse — a span
// re-demanded ten times by its own payer is reuse, a span demanded once each by ten agents is
// coalescing, and Hits alone cannot tell them apart.
type sharedRingLedger struct {
	// demands counts DEMAND stagings routed through the ring (prefetch hints excluded — a hint is not
	// an agent asking for bytes). This is the B*K numerator.
	demands int64
	// payer names the agent whose staging paid for each live residency span; served is the set of
	// agents that have been served off that span.
	payer  map[polymodel.ModelID]string
	served map[polymodel.ModelID]map[string]bool

	// distinctServes sums, over all spans, the number of distinct agents each served. Divided by
	// page-ins it is the coalescing factor: 1.0 for a solo agent, rising toward B as activated sets
	// overlap.
	distinctServes int64
	// crossAgentHits counts EVERY demand satisfied by a residency some OTHER agent paid for — the
	// page-ins that did not happen because the ring was shared.
	crossAgentHits int64
	// sharedResidencies counts spans that served at least two distinct agents, so a ratio driven by
	// one hot expert can be told apart from one spread across the working set.
	sharedResidencies int64

	// pageInBytes is the bytes actually uploaded, and agentTokens the tokens agents produced over
	// them (NoteAgentToken). Their quotient is the "effective bytes per agent-token" #5243 asks to
	// fall as B grows: the same union, amortized over more output.
	pageInBytes int64
	agentTokens int64

	// refusals counts stagings the ring could not admit — on a shared ring, almost always because
	// the attached agents' concurrently HELD experts already fill the budget. Each one falls back to
	// permanent halW residency, so a rising count is the ring quietly turning back into the memoizer
	// it replaced. It is a sizing signal, not an error: see SharedExpertRingConfig.BudgetBytes.
	refusals int64
}

// NewSharedExpertRing builds a shared routed-expert ring over cfg's (model, device) pair. The
// caller OWNS it and must Close it after every agent has detached; it deliberately does not free
// itself when the last agent leaves, because the point of a shared ring is to survive the
// conversations that use it.
func NewSharedExpertRing(cfg SharedExpertRingConfig) (*SharedExpertRing, error) {
	if cfg.Model == nil {
		return nil, ErrSharedRingModel
	}
	if cfg.Backend == nil {
		return nil, ErrSharedRingBackend
	}
	if cfg.BudgetBytes <= 0 {
		return nil, errors.New("model: shared expert ring needs a positive byte budget")
	}
	sh := &SharedExpertRing{
		m:        cfg.Model,
		be:       cfg.Backend,
		attached: map[string]*Session{},
	}
	sh.ledger.payer = map[polymodel.ModelID]string{}
	sh.ledger.served = map[polymodel.ModelID]map[string]bool{}
	sh.ring = newPagedRing(cfg.Backend, cfg.BudgetBytes)
	sh.ring.policy = cfg.Evict
	sh.ring.shared = sh
	return sh, nil
}

// Attach points a session's routed-expert residency at this shared ring under the given agent name,
// which is the ledger's identity for "a distinct agent" and must be unique among live attachments.
//
// It refuses — rather than degrading to a private ring — when the session serves a different model,
// runs on a different backend, already has routed-expert residency, or reuses a live agent name.
// Each refusal is a case where attaching would be silently wrong: the first two would serve the
// wrong bytes, the third would strand device handles, the fourth would understate coalescing by
// merging two agents into one.
//
// Attaching sets the session's ExpertRingBytes to the shared budget so every existing routed-expert
// gate (the demand path, the R3 prefetch, the telemetry) reads the ring by exactly the rule it read
// a private one by. The shared ring owns the eviction policy and the pin-set: an attached session's
// own ExpertRingEvict / pin warm-start do not apply, because B agents' priors are one pin-set and
// merging them is a policy question this rung does not settle.
func (sh *SharedExpertRing) Attach(s *Session, agent string) error {
	if sh == nil {
		return ErrSharedRingClosed
	}
	if s == nil || s.M == nil {
		return ErrSharedRingModel
	}
	sh.mu.Lock()
	defer sh.mu.Unlock()
	if sh.closed {
		return ErrSharedRingClosed
	}
	if s.M != sh.m {
		return ErrSharedRingModel
	}
	if s.Backend == nil || s.Backend != sh.be {
		return ErrSharedRingBackend
	}
	if s.expertRing != nil || s.sharedRing != nil {
		return ErrSharedRingResidency
	}
	if agent == "" {
		return ErrSharedRingAgent
	}
	if _, live := sh.attached[agent]; live {
		return ErrSharedRingAgent
	}
	sh.attached[agent] = s
	if n := len(sh.attached); n > sh.peakAttached {
		sh.peakAttached = n
	}
	s.sharedRing = sh
	s.ringAgent = agent
	s.expertRing = sh.ring
	s.ExpertRingBytes = sh.ring.budget()
	return nil
}

// Detach removes a session from the ring WITHOUT freeing anything: the residency it paid for stays
// for the agents still attached, which is the whole point. It is idempotent and is what
// Session.Close calls, so a conversation ending never tears down bytes its peers are using.
func (sh *SharedExpertRing) Detach(s *Session) {
	if sh == nil || s == nil {
		return
	}
	sh.mu.Lock()
	defer sh.mu.Unlock()
	if s.ringAgent != "" && sh.attached[s.ringAgent] == s {
		delete(sh.attached, s.ringAgent)
	}
	s.sharedRing = nil
	s.ringAgent = ""
	s.expertRing = nil
}

// NoteAgentToken records one token produced by one agent over this ring's residency — the
// denominator of BytesPerAgentToken. It is the caller's to drive because the ring sees layers and
// weights, never tokens; the decode loop is the only place that knows a token finished. Wiring it
// to the serve path is R6/#5617's operator surface.
func (sh *SharedExpertRing) NoteAgentToken() {
	if sh == nil {
		return
	}
	sh.mu.Lock()
	sh.ledger.agentTokens++
	sh.mu.Unlock()
}

// Close frees every device handle the ring holds. It REFUSES while agents are still attached,
// because freeing a handle a live session is about to multiply against is a use-after-free the
// per-session ring could not produce. Closing twice is a no-op.
func (sh *SharedExpertRing) Close() error {
	if sh == nil {
		return nil
	}
	sh.mu.Lock()
	defer sh.mu.Unlock()
	if sh.closed {
		return nil
	}
	if len(sh.attached) > 0 {
		return ErrSharedRingBusy
	}
	sh.ring.freeAll()
	sh.closed = true
	return nil
}

// SharedExpertRingStats is the cross-agent residency account: R0's per-ring numbers plus the
// coalescing ledger that only a shared ring can produce.
type SharedExpertRingStats struct {
	Enabled bool `json:"enabled"`
	// Agents is how many are attached now; PeakAgents the high-water mark. A coalescing ratio is
	// uninterpretable without the B it was measured at, so the two travel together.
	Agents     int `json:"agents"`
	PeakAgents int `json:"peak_agents"`
	// Ring is the underlying bounded residency — budget, footprint, page-ins, hits, evictions — now
	// aggregated across every attached agent rather than per conversation.
	Ring ExpertRingStats `json:"ring"`
	// Demands is the routed-expert stagings agents asked for (prefetch hints excluded): the B*K
	// numerator. DistinctServes sums distinct agents served per residency span.
	Demands        int64 `json:"demands"`
	DistinctServes int64 `json:"distinct_serves"`
	// CrossAgentHits is demands satisfied by bytes another agent paid for — the page-ins sharing
	// avoided. SharedResidencies is how many spans served two or more agents.
	CrossAgentHits    int64 `json:"cross_agent_hits"`
	SharedResidencies int64 `json:"shared_residencies"`
	// PageInBytes is what was actually uploaded; AgentTokens what the agents produced over it.
	PageInBytes int64 `json:"page_in_bytes"`
	AgentTokens int64 `json:"agent_tokens"`
	// Refusals is stagings the budget could not admit, each falling back to PERMANENT per-session
	// residency. On a shared ring a non-zero count almost always means the budget is too small for
	// the attached agents' concurrently held experts (SharedExpertRingConfig.BudgetBytes), and it is
	// the number to watch before trusting any of the ratios above: a ring that refused half its
	// demands is reporting the coalescing of the half it kept.
	Refusals int64 `json:"refusals"`
}

// SwapExpertRingEvictPolicy serializes an operation-atomic policy transition with the existing
// shared-ring mutex while attached sessions remain live. The mutex does not establish a global
// turn barrier; callers requiring turn-level trace attribution must quiesce attached sessions.
func (sh *SharedExpertRing) SwapExpertRingEvictPolicy(policy ExpertRingEvictPolicy) (ExpertRingPolicySwapReceipt, error) {
	if sh == nil || sh.ring == nil {
		return ExpertRingPolicySwapReceipt{}, fmt.Errorf("shared expert ring: not enabled")
	}
	sh.mu.Lock()
	defer sh.mu.Unlock()
	return sh.ring.swapPolicy(policy)
}

// CoalescingRatio is demands per page-in — #5243's B*K/|U(B)| in its raw form, over the whole
// measurement window. It includes ordinary temporal reuse (one agent re-activating an expert), so
// read it next to AgentsPerPageIn, which isolates the cross-agent part. 0 when nothing paged in.
func (s SharedExpertRingStats) CoalescingRatio() float64 {
	if s.Ring.PageIns <= 0 {
		return 0
	}
	return float64(s.Demands) / float64(s.Ring.PageIns)
}

// AgentsPerPageIn is the distinct agents served per page-in: exactly 1.0 for a solo agent by
// construction, rising toward B as the batch's activated sets overlap. This is the number the L2
// lever is worth, because it is the factor by which one streamed expert's cost is divided.
//
// It can fall BELOW 1.0 when the R3 prefetch stages experts the router then did not pick — a span
// paid for and never served. That is not a defect of the meter; it is the prefetch's waste made
// visible, and hiding it would flatter both.
func (s SharedExpertRingStats) AgentsPerPageIn() float64 {
	if s.Ring.PageIns <= 0 {
		return 0
	}
	return float64(s.DistinctServes) / float64(s.Ring.PageIns)
}

// BytesPerAgentToken is the effective expert-weight traffic each agent-token cost. It is the number
// #5243 predicts falls as B grows at a fixed budget: the same union of experts, amortized over more
// output. 0 when no agent-token was noted (see NoteAgentToken), which reads as "not measured", not
// as "free".
func (s SharedExpertRingStats) BytesPerAgentToken() float64 {
	if s.AgentTokens <= 0 {
		return 0
	}
	return float64(s.PageInBytes) / float64(s.AgentTokens)
}

// Stats reports the shared residency and its coalescing ledger. Call it from outside a forward: it
// takes the ring lock, so a session that called it from inside its own ring span would deadlock —
// which is why the per-session view (Session.ExpertRing) goes through the reentrant span instead.
func (sh *SharedExpertRing) Stats() SharedExpertRingStats {
	if sh == nil {
		return SharedExpertRingStats{}
	}
	sh.mu.Lock()
	defer sh.mu.Unlock()
	return sh.statsLocked()
}

// statsLocked is Stats without the locking, for a caller already inside a ring span — the same
// "session-level entry points lock, the rest never does" discipline the rest of this file follows.
// The operator report (R6/#5617) reads it that way so its ring, coalescing and placement numbers are
// ONE snapshot rather than several taken while peers moved the ring underneath.
func (sh *SharedExpertRing) statsLocked() SharedExpertRingStats {
	if sh == nil {
		return SharedExpertRingStats{}
	}
	return SharedExpertRingStats{
		Enabled:           !sh.closed,
		Agents:            len(sh.attached),
		PeakAgents:        sh.peakAttached,
		Ring:              sh.ring.stats(),
		Demands:           sh.ledger.demands,
		DistinctServes:    sh.ledger.distinctServes,
		CrossAgentHits:    sh.ledger.crossAgentHits,
		SharedResidencies: sh.ledger.sharedResidencies,
		PageInBytes:       sh.ledger.pageInBytes,
		AgentTokens:       sh.ledger.agentTokens,
		Refusals:          sh.ledger.refusals,
	}
}

// ringEnter takes the shared ring's lock for the span of one session-level ring operation, and
// returns the function that ends it. It is a no-op for a private (per-session) ring, which is the
// default and which keeps every existing session on the exact path it had.
//
// It is REENTRANT per session, because the demand path enters once around a whole expert's
// three-projection acquisition (hal.go) and weightHALStagedBounded enters again underneath it. The
// counter lives on the Session rather than in the mutex because a Session is single-goroutine by
// construction: the depth can only ever be touched by the one goroutine running that session's
// forward, so it needs no synchronization of its own.
//
// The discipline this establishes, and the reason it is stated here rather than at each call site:
// SESSION-level entry points lock, pagedRing methods NEVER do. Any new ring caller belongs on this
// list, and any lock inside pagedRing would deadlock against it.
func (s *Session) ringEnter(r *pagedRing) func() {
	if s == nil || r == nil || r.shared == nil {
		return func() {}
	}
	if s.ringDepth > 0 {
		s.ringDepth++
		return func() { s.ringDepth-- }
	}
	sh := r.shared
	sh.mu.Lock()
	sh.inSpan = s.ringAgent
	s.ringDepth = 1
	return func() {
		s.ringDepth--
		if s.ringDepth == 0 {
			sh.inSpan = ""
			sh.mu.Unlock()
		}
	}
}

// noteRefusal books a staging the budget could not admit. It is deliberately NOT subtracted from
// demands: the agent did ask for those bytes, and hiding the ask would make an undersized ring read
// as a well-coalescing one.
func (sh *SharedExpertRing) noteRefusal() { sh.ledger.refusals++ }

// noteDemand books one agent-visible routed-expert staging. Prefetch hints are excluded by the
// caller: a hint is the ring guessing, not an agent asking, and counting it would inflate the B*K
// numerator with demand that never existed.
func (sh *SharedExpertRing) noteDemand() { sh.ledger.demands++ }

// notePageIn opens a residency span: the current agent paid for these bytes, and no one has been
// served off them yet.
func (sh *SharedExpertRing) notePageIn(id polymodel.ModelID, weightBytes int64) {
	sh.ledger.pageInBytes += weightBytes
	sh.ledger.payer[id] = sh.inSpan
	delete(sh.ledger.served, id)
}

// noteServe books one agent being served off a live residency span. A serve by an agent other than
// the span's payer is a cross-agent hit — a page-in that did not happen because the ring is shared —
// and the first such agent also promotes the span to a shared one.
func (sh *SharedExpertRing) noteServe(id polymodel.ModelID) {
	agent := sh.inSpan
	if agent == "" {
		return // no attached agent owns this span; nothing to attribute
	}
	if payer, ok := sh.ledger.payer[id]; ok && payer != agent {
		sh.ledger.crossAgentHits++
	}
	set := sh.ledger.served[id]
	if set == nil {
		set = map[string]bool{}
		sh.ledger.served[id] = set
	}
	if set[agent] {
		return
	}
	if len(set) == 1 {
		sh.ledger.sharedResidencies++
	}
	set[agent] = true
	sh.ledger.distinctServes++
}

// endResidency closes a span when its weight is evicted or freed. The next page-in of the same
// weight starts a fresh span with a fresh payer, so an expert that thrashes in and out is accounted
// once per residency rather than once per lifetime — which is what makes AgentsPerPageIn a
// statement about the budget in force rather than about the whole run.
func (sh *SharedExpertRing) endResidency(id polymodel.ModelID) {
	delete(sh.ledger.payer, id)
	delete(sh.ledger.served, id)
}
