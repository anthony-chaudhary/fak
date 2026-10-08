package agent

// inkernel_planner.go — the in-kernel chat Planner. When fak serve boots with a
// preloaded GGUF model (modelengine.Preload / PreloadQ4K) and a tokenizer, and no
// upstream --base-url, this planner drives BOTH /v1/chat/completions and
// /v1/messages (they share s.planner.Complete) from the model fused into the
// kernel — real ChatML chat through internal/tokenizer, not the byte-tokenized
// dispatch demo in modelengine.Complete.
//
// The decode recipe is the proven cmd/fakchat hybrid path: render ChatML → Encode
// → Session.Prefill → argmax/temperature sample → Session.Step → Decode, stopping
// on <|im_end|>/<|endoftext|>. fakchat's end-to-end coherent chat (Qwen2.5-1.5B/7B,
// FAK-NATIVE-CHAT-RESULTS.md) is the witness that this recipe produces real text;
// this file factors it into a Planner so the gateway can serve it on both wires.

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"log"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"time"
	"unicode"

	"github.com/anthony-chaudhary/fak/internal/cachemeta"
	"github.com/anthony-chaudhary/fak/internal/cacheobs"
	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/enginestep"
	"github.com/anthony-chaudhary/fak/internal/metalgemm"
	"github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/radixkv"
	"github.com/anthony-chaudhary/fak/internal/tokenizer"
)

// InKernelPlanner is an agent.Planner backed by the in-kernel model. One Complete
// call renders the transcript as ChatML, runs a real Prefill + decode over the
// kernel-owned session cache, and returns the assistant's text. It does not itself
// emit structured tool calls — the gateway's adjudication layer still runs on
// whatever the caller proposed.
type InKernelPlanner struct {
	m                      *model.Model
	tok                    *tokenizer.Tokenizer
	modelID                string
	q4k                    bool            // resident-Q4_K load: decode runs Session.Q4K (SDOT int8 GEMV)
	quant                  bool            // Q8_0 decode/prefill path (the served default); tests flip it to exercise the proven f32 reuse path
	backend                compute.Backend // non-nil → decode runs through the device HAL (e.g. CUDA) instead of the CPU session
	metal                  bool            // Apple-Silicon metalgemm GPU forward on the CPU session (s.Metal); engaged ONLY when backend==nil (the CPU-session seam). No-op on non-Metal builds.
	cpuOffloadExperts      bool            // with a backend, keep MoE experts host-resident while dense/attention use the device
	requireDeviceExecution bool
	// kvPrecision is the realized storage tier installed on every session this planner
	// builds (model.KVPrecisionFP32 default; Q8_0 for the dense mixed layout). Fixed at
	// construction from InKernelPlannerConfig.KVPrecision.
	kvPrecision model.KVPrecision
	// expertSpill is the resolved graded expert placement (`--n-cpu-moe`, #5612) this planner
	// installs on every session it builds: how many MoE layers spill to host and how many device
	// bytes the routed-expert ring may hold. nil — the default and every planner that was never
	// given a grade — leaves placement exactly as cpuOffloadExperts alone decided it. Resolved once
	// by SetExpertSpill (inkernel_expert_spill.go), never per request.
	expertSpill *model.ExpertSpillPlacement
	// contextTokens is the runtime ceiling; zero delegates to MaxPositionEmbeddings.
	contextTokens int
	maxNew        int
	temp          float64
	seed          int64
	// decodeTraceNow is an injectable monotonic clock used only by explicitly
	// traced requests. nil selects time.Now; the default path never reads it.
	decodeTraceNow func() time.Time
	// qwenQ4KPrefillChunkTokens and its typed parse error are resolved once at
	// construction. The error is request-gated to the exact resident hybrid path,
	// so an unrelated model remains byte-for-byte on its historical forward.
	qwenQ4KPrefillChunkTokens int
	// qwenQ4KPrefillChunkExplicit records that the operator enabled the existing
	// chunk-size experiment. Only that setting is capped to the device's widest
	// single-allocation token panel; the unset default remains historical.
	qwenQ4KPrefillChunkExplicit  bool
	qwenQ4KPrefillChunkConfigErr *model.InKernelQwenQ4KPrefillChunkConfigError
	qwen35MetalGDNSequence       bool
	q4kGateUpOutputSlab          bool
	denseGPULayers               int
	qwen35MetalGDNExecuted       atomic.Bool
	// weightResidency memoizes the resident-weight byte split for /healthz (fak#13567).
	weightResidency      inKernelWeightResidencyCache
	compactHistoryBudget int
	elideStaleReads      bool
	deferColdTools       bool
	restoreStash         func(trace, id, excerpt string, body []byte)

	// prefillQ4KGEMM is the last request's observed Q4_K prefill GEMM route
	// (model.Session.Q4KPrefillGEMMObservation), reported as gemm= on the
	// inkernel_chat summary line (#13694). It holds a string; empty means unobserved.
	prefillQ4KGEMM atomic.Value

	// tree is the process-scoped RadixAttention prefix cache (internal/radixkv): the
	// multi-thousand-token static system+tool-schema prefix is prefilled once and the
	// next turn REUSES its KV, prefilling only the divergent suffix — the candidate-#13
	// win, bit-identical to a full recompute (proven in internal/model's KV-prefix-reuse
	// rung). nil disables reuse (every turn full-prefills, the pre-#13 behavior); the
	// device-HAL path (backend != nil) never reuses (the reuse clone is a CPU session).
	// mu guards every tree access — the gateway can drive Complete concurrently, and the
	// tree is shared mutable state (radixkv itself is deliberately lock-free).
	//
	// MEMORY NOTE: radixkv stores the FULL-prefix KV per node, so a long single growing
	// conversation accumulates nested KV clones (see radixkv's Tokens-vs-PrefixTokens
	// note). FAK_INKERNEL_RADIX_BUDGET sets the edge-token budget (0 = unbounded, the
	// default — the maximal-reuse regime the witnesses measure), and
	// FAK_NATIVE_KV_VICTIM_RULE=cost-aware selects the KVBM victim rule for budget pressure.
	// Operators serving long sessions should set a budget; bounding the deep-chain
	// footprint is tracked.
	mu         sync.Mutex
	tree       *radixkv.Tree
	scopedTree *radixkv.ScopedTree
	// prefixFlights bridges the cold lookup-to-admission gap for concurrent
	// plain-CPU requests. Its zero value is ready for constructor and bare-test
	// planners alike; persistent ownership remains in tree/scopedTree.
	prefixFlights radixkv.PrefixFlightGroup

	// devMu serializes the WHOLE device forward pass (Prefill + the decode loop) when a
	// backend or native Metal is wired. These accelerators have one shared command stream and
	// model-level buffers. The CUDA backend's Go-side cudaMu makes
	// each INDIVIDUAL op atomic — but NOT a whole multi-op forward. Two Complete calls driven
	// concurrently by the gateway would interleave their per-token op sequences on that shared
	// device state and stomp each other's activation/KV buffers, faulting the kernel with an
	// illegal memory access that then poisons the CUDA context for every later request until a
	// process restart (observed live on an L4: a 2-way concurrent burst took the GPU serve down
	// with thousands of sticky cuda_kernels.cu illegal-access errors). The plain CPU path is
	// already session-local per turn and guards only its shared tree with p.mu, so devMu leaves
	// it untouched.
	// This serializes concurrent device requests into safe queuing — correct for a single-stream
	// device — instead of crashing; batched multi-user device decode is the separate throughput
	// follow-up (internal/model/batch.go), not a correctness fix.
	devMu sync.Mutex

	// concurrencyProfile is an OPT-IN phase-timeline recorder for the device
	// fan-out cell (issue #1589). Its zero value is nil, so a planner that never
	// opts in is byte-for-byte unaffected: the admit / forwardEnter / forwardExit
	// calls are all nil-receiver-safe. When set, it localizes the serialization
	// layer (devMu vs. concurrent-unfused) with a witnessing timeline instead of
	// a guessed mechanism.
	concurrencyProfile *concurrencyProfiler

	coalesceMu         sync.Mutex
	coalesceReady      []*inKernelCoalesceRequest
	coalesceRunning    bool
	coalesceReadyHook  func()
	coalesceBatchHook  func(int)
	coalesceSharedHook func(panels int, macs int64)
	coalesceCohortID   atomic.Uint64

	reqMemMu      sync.Mutex
	lastReqMemory RequestMemoryStats

	// nativePhaseMu guards nativePhaseLog, the bounded most-recent-observation-per-trace-id
	// ledger behind NativePhaseReporter (#13120). It is deliberately its OWN mutex rather than
	// sharing reqMemMu: the request-memory plan is written once per request admission, while a
	// phase observation is written at each phase boundary, so folding them would couple two
	// unrelated lock orderings on the serve hot path. The map is capped and evicts oldest-first
	// so a long-lived serve cannot grow it without bound.
	nativePhaseMu  sync.Mutex
	nativePhaseLog map[string]NativePhaseObservation
	nativePhaseSeq []string
	// moeResidencyState is the serve-scoped fold of every request's activated-expert residency
	// (R6/#5617, inkernel_moe_residency.go). It is embedded because the ring lives on a session
	// this planner builds and closes PER REQUEST, so without a planner-scoped ledger the whole
	// ladder's accounting is destroyed at each teardown and no serve surface can see it.
	moeResidencyState

	// inKernelTurnTaxState is the per-turn cache-decision ledger (#1538, inkernel_turntax.go),
	// embedded for the same reason as moeResidencyState directly above: the decision is taken
	// inside a session this planner builds and closes PER REQUEST, so planner-scoped storage is
	// the only place it survives that teardown. Its zero value is a usable empty ledger, so
	// every constructor — including a bare &InKernelPlanner{…} — records from its first turn.
	inKernelTurnTaxState

	oomRetryMu sync.Mutex
	oomRetry   map[string]*inKernelOOMRetryClassStats

	pressureTrimMu sync.Mutex
	pressureTrim   map[requestPressureTrimKey]*requestPressureTrimStats

	// hostBudget is the armed host-session memory ceiling (#13267, inkernel_host_budget.go):
	// nil keeps the historical unbounded host path; SetHostMemoryBudget arms it.
	hostBudgetMu sync.Mutex
	hostBudget   *hostMemoryBudget

	// kvSpanEvict gates the model-side KV-quarantine eviction BRIDGE (internal/kvmmu)
	// on the live serve path (issue #579). When on, a tool-result QUARANTINE drives a
	// real model.KVCache.Evict of the result's K/V span over a fresh model.Session built
	// from the loaded model — the bit-exact re-RoPE + renumber the kvmmu witnesses prove,
	// now fired by a live request instead of only a synthetic-model unit test. DEFAULT OFF
	// (FAK_INKERNEL_KVMMU=on opts in); off it is an inert no-op, so the served path is
	// byte-for-byte the pre-bridge behavior. It is independent of and additive to the
	// radixkv prefix-cache eviction above — that drops a reusable PREFIX node; this evicts
	// the per-session SPAN and is the model-independent KV-MMU floor.
	kvSpanEvict bool

	// batchDecode routes a request's decode through the continuous-batch
	// BatchSession.StepBatchActive machinery (as a batch of one on the resident chat serve)
	// instead of the serial Session.Step loop. It is the OPT-IN wiring seam
	// (FAK_INKERNEL_BATCH=on) that lets a future cross-request coalescer co-batch concurrent
	// chat turns onto the shared weight stream. DEFAULT OFF: unset, the decode path is the
	// byte-identical pre-seam serial loop, and even ON at B=1 StepBatchActive is exactly
	// Seqs[0].Step, so the served tokens are unchanged — the batched glm_moe_dsa GEMM that
	// yields throughput is the separate, box-gated lever, not this flag.
	batchDecode bool

	// kvPrefixEverAdmitted is the #3391 first-prefill latch: false until a reuse-enabled
	// turn has admitted its prompt into the radix tree (generateReused step 3). While
	// false, a turn's prompt tokens are booked as INELIGIBLE in the cacheobs hit-rate
	// denominator — nothing has been admitted, so nothing could match: the always-cold
	// first prefill the raw ratio unfairly counts against the cache. Deliberately one-way
	// and planner-scoped (matching the tree's scope): a tree later evicted back to empty
	// keeps counting prompts as eligible, and on the multi-tenant scoped tree a tenant's
	// own cold first prefill after another tenant warmed the latch counts as eligible too
	// — both over-count the denominator, which can only UNDER-state the filtered ratio,
	// never inflate it (the same honest-conservative direction as cacheobs's clamps).
	kvPrefixEverAdmitted atomic.Bool

	speculativeEngine *model.SpeculativeEngine
	specDraftDepth    int
	vulkanMTP         bool
	// specSessionCloseHook, when non-nil, observes the live reused-speculative
	// target session immediately before it is closed at the end of
	// generateReusedSpeculative. It exists so package tests can witness the
	// resident KV state after a round terminates early (the boundary-state
	// parity regression for #12422 defect 3) without exporting the per-request
	// session. nil on the served path is a literal no-op.
	specSessionCloseHook func(*model.Session)
	// vulkanMTPDraftFactory is request-scoped: each invocation binds one draft
	// cache to the live target session used by that request. Tests in this
	// package replace it to witness lifecycle behavior without Vulkan hardware.
	vulkanMTPDraftFactory func(*model.Session, int) (model.ProposalGenerator, func(), error)

	metalMTPMu          sync.Mutex
	metalMTPCoordinator *model.MetalMTPCoordinator
	// mtpExplicitOverride records that a coordinator was installed through the
	// pre-existing explicit API (SetMetalMTPCoordinator), which remains an
	// operator escape hatch. A coordinator installed by EnableMetalMTP without a
	// canary manager is NOT an override and stays fail-closed.
	mtpExplicitOverride bool
	mtpCanaryMu         sync.RWMutex
	mtpCanaryManager    *model.Qwen38MTPCanaryManager
	mtpCanaryRequest    model.Qwen38CanaryRequest
	mtpKillSwitch       *model.Qwen38MTPKillSwitch
	mtpCanaryResult     model.Qwen38CanaryDecision

	// warmState is the planner-owned startup KV-cache warm lifecycle (CW-27, the
	// InKernelWarmState type below). It owns the bounded single-profile coalescing slot,
	// the in-flight warm cancellation, and the idempotent release of the abstract
	// residency claim handed over by the later warm API. Its zero value is a usable
	// idle state, so every constructor — including a bare &InKernelPlanner{} — can
	// begin a startup warm with no initialization step to forget.
	warmState InKernelWarmState

	// cachePopulate marks the CW-03 (#13351) cache-POPULATE execution purpose for the
	// duration of one prime: generateReusedContextWithBias reads it once and returns at
	// the step-3b seam, after full-state admission and before any decode. It is set and
	// cleared by primeCacheStateOnce, so an ordinary demand turn never observes it true.
	cachePopulate bool

	// Cache-populate purpose observation taps (nil on the served path: a literal
	// no-op). They are bound onto the decodeLane a prime would have to construct, and
	// fired at the exact sample and token-emit seams, so a regression that let a prime
	// reach decode would trip them. They observe only; they never change output.
	cachePrimeSamplerHook func()
	cachePrimeEmitHook    func()

	// warmInputs carries the stable inputs a WarmPrefixSpec was derived from (CW-04,
	// #13344), so WarmPrefix can re-encode the descriptor's exact stable token boundary
	// without re-reading prompt text. It is set at the startup seam via
	// SetWarmPrefixInputs, where descriptor + inputs are both in hand; a planner with no
	// carrier refuses a warm closed rather than guessing the boundary. Guarded by mu
	// alongside every other planner-owned warm field.
	warmInputs *warmInputsCarrier

	// auxWarm / auxWarmSet carry the OPTIONAL auxiliary-cache warming config (CW-12,
	// #13340): the recorded expert profile, the V4.1 Engram prefix and the startup
	// reserve. They are inert until SetAuxWarmConfig is called, so a planner that never
	// opts in warms only KV and leaves WarmReceipt.Aux nil. Guarded by mu.
	auxWarm    AuxWarmConfig
	auxWarmSet bool
	// auxAdapter is the narrow override seam the warm orchestrator calls for optional
	// auxiliary warming; nil selects the planner's own model (the real adapter retained
	// for real requests). It exists so a witness can inject a counting fake without
	// building a checkpoint tier or an Engram stage; the served path never sets it.
	auxAdapter auxWarmAdapter

	// warmClaim carries the OPTIONAL bounded-residency startup claim config (CW-06,
	// #13341): the spare byte/token quota and the TTL. It is inert until
	// SetWarmClaimConfig is called, so a planner that never opts in warms exactly as
	// CW-04 did and leaves WarmReceipt.Claim nil. Guarded by mu.
	warmClaim    radixkv.StartupClaimConfig
	warmClaimSet bool
	// warmClaimCache is the lazily built bounded-residency boundary over p.tree. It
	// SHARES p.mu as its locker (the same ScopedTree discipline), so claim
	// acquisition/validation/release serialize with every other tree access and can
	// never race a demand path. Nil until the first claim-bearing warm. Guarded by mu.
	warmClaimCache *radixkv.StartupCache
	// warmClaimHandle is the currently bound startup claim (CW-06, #13341). It is the
	// finite residency handle the last successful warm acquired; the lifecycle's
	// BindRelease callback owns releasing it on Complete/Release. Guarded by mu.
	warmClaimHandle radixkv.StartupClaim
	// warmClaimScope / warmClaimScopeSet record the authenticated cache scope the bound
	// startup claim was acquired under (CW-16, #13336), so the first-demand handoff can
	// refuse to hand a different tenant's (or agent's) prepared prefix to a request that
	// did not ask for it. Inert until the first claim-bearing warm. Guarded by mu.
	warmClaimScope    radixkv.CacheIdentity
	warmClaimScopeSet bool

	// incrementalContext is the OPTIONAL composed set of independently refreshable
	// system-context sources (inkernel_incremental.go). nil — the default and every
	// planner that never calls SetIncrementalContext — leaves the historical planner
	// byte-for-byte inert. ctxSnapshot is the durable comparison state advanced on each
	// applied update; ctxMu guards both fields so a concurrent reconcile cannot race a
	// snapshot advance.
	incrementalContext *IncrementalContext
	ctxSnapshot        ContextSnapshot
	ctxMu              sync.Mutex
}

type inKernelOOMRetryClassStats struct {
	attempts        uint64
	successes       uint64
	failures        uint64
	lastFailedBytes uint64
	lastSite        string
}

type requestPressureTrimKey struct {
	scope  string
	class  string
	reason string
}

type requestPressureTrimStats struct {
	attempts        uint64
	trimmed         uint64
	noHooks         uint64
	resolved        uint64
	lastWantBytes   uint64
	lastBudgetBytes uint64
	lastMarginBytes int64
}

// Model reports the model id (for /v1/models provenance + the planner seam).
func (p *InKernelPlanner) Model() string { return p.modelID }

// NativeDecodeTraceSupported declares that this planner owns the token-commit
// seam used by NativeDecodeTrace. It performs no model work.
func (p *InKernelPlanner) NativeDecodeTraceSupported() bool { return true }

// nativePhaseTraceID returns the request trace id the planner binds a native-phase
// observation to. It reads the gateway's request trace id from the typed context key the
// served path stamps (WithRequestTraceID), falling back to the legacy plain-string
// "trace_id" key the restore stash historically used. It is empty when the request carried
// none — the observation then keys the empty bucket rather than fabricating an id (#13120).
func nativePhaseTraceID(ctx context.Context) string {
	if ctx == nil {
		return ""
	}
	if id := RequestTraceID(ctx); id != "" {
		return id
	}
	if v := ctx.Value("trace_id"); v != nil {
		if s, ok := v.(string); ok {
			return s
		}
	}
	return ""
}

// StreamingSupported enables the gateway's semantic SSE path for in-kernel runs.
// The backend projects each completed turn as one content delta; tool lifecycle
// progress still arrives independently from the owned loop.
func (p *InKernelPlanner) StreamingSupported() bool { return true }

// CompleteStream emits assistant PROSE to sink incrementally while the turn decodes,
// holding every tool-call span through post-decode lifting and never exposing tool-token
// text before adjudication can see it (the StreamingPlanner contract, stream.go).
// Mechanically it binds the sink into the decode path's existing per-token emit seam
// (generateReusedWithOOMRetry → the per-token emit closure in Complete), so each decoded
// piece of assistant prose is forwarded as it is produced — the same live token flow
// cmd/fakchat's streamDecode gives the direct process path, not one post-hoc delta.
// A toolSpanGuard sits between that raw seam and the caller's sink: it holds text that
// could open an explicit tool-call span, drops the span once confirmed, and forwards
// only prose, so raw <tool_call>/<function_call>/<|python_tag|>/[TOOL_CALLS] markup is
// never delivered as content before adjudication (toolSpanGuard, below).
//
// This per-token forwarding is OPT-IN via FAK_STREAM_INKERNEL_PER_TOKEN (default off) or
// a per-call WithPerTokenStream(true), which the turnkey gateway passes so `fak up`
// streams by default; flag OFF is the buffered projection — the whole turn decodes with
// the plain Complete (no observer, no forced DecodeTrace) and reaches the sink as AT MOST
// one post-hoc content delta, byte-identical to the pre-seam trunk. When no sink is
// provided (or the decode runs one of the speculative paths, whose draft-verify rounds
// batch tokens), the stream likewise degrades to the buffered projection and the returned
// Completion is unchanged.
//
// The sink observes the RAW incrementally decoded model text; the final Completion
// carries the POST-PROCESSED turn (reasoning split, tool-call lift) exactly as the
// buffered Complete produces it for the same prompt + seed. runArmStream wraps this raw
// sink with the post-decode projection (loop_project.go), so its emitted deltas land in
// the client's view of Content.
func (p *InKernelPlanner) CompleteStream(ctx context.Context, sink StreamSink, messages []Message, tools []ToolDef, opts ...SampleOpt) (*Completion, error) {
	if sink == nil {
		return p.Complete(ctx, messages, tools, opts...)
	}
	// A per-call override (WithPerTokenStream) wins over the package-level env
	// gate so the turnkey server streams by default while every other CompleteStream
	// caller keeps the historical env-gated default exactly.
	perToken := inKernelPerTokenStreamEnabled()
	if sp := applySampleOpts(opts...); sp.PerTokenStream != nil {
		perToken = *sp.PerTokenStream
	}
	if !perToken {
		// Default: the buffered projection. Decode the whole turn with the plain
		// Complete (no observer, no forced DecodeTrace), then emit the finished
		// content as one delta — byte-identical to the pre-seam trunk.
		comp, err := p.Complete(ctx, messages, tools, opts...)
		if err != nil {
			return comp, err
		}
		if comp != nil && comp.Message.Content != "" {
			return comp, sink(comp.Message.Content)
		}
		return comp, nil
	}
	streamCtx, cancel := context.WithCancel(ctx)
	defer cancel()
	var sinkErr error
	sp := applySampleOpts(opts...)
	rendered := renderInKernelChatMLRequest(messages, tools, p.m.Cfg, sp.ResponseFormat, sp.ToolChoice, sp)
	startsInReasoning := strings.HasSuffix(rendered, qwenThinkAssistantSeed)
	projector := newInKernelStreamProjector(sink, sp.Stop, startsInReasoning)
	// Prompt shrinking preserves the complete trailing tool run: stale-read elision
	// protects the recent tail, and compaction moves keepStart behind a trailing tool
	// batch. AssessTranscriptTurn therefore yields the same budget context here as it
	// does over Complete's prepared messages.
	var turnContext []TurnAssessment
	if ta, ok := AssessTranscriptTurn(messages); ok {
		turnContext = append(turnContext, ta)
	}
	effectiveBudget := ResolveEffortBudget(sp.ReasoningEffort, sp.ThinkingBudget, turnContext...)
	var streamThinkBudget *ThinkBudget
	if effectiveBudget > 0 {
		streamThinkBudget = NewThinkBudget(effectiveBudget, startsInReasoning)
	}
	comp, err := p.Complete(streamCtx, messages, tools, append(opts, WithDecodeTokenObserver(func(tokenPiece, rawText string) {
		if sinkErr != nil || tokenPiece == "" {
			return
		}
		if sinkErr = projector.feed(tokenPiece); sinkErr != nil {
			cancel()
			return
		}
		// Complete may inject a synthetic reasoning close when the budget is
		// exhausted. DecodeTokenObserver intentionally reports one decoded token,
		// so mirror that deterministic projection-only append here.
		if streamThinkBudget != nil && streamThinkBudget.Observe(tokenPiece) {
			if sinkErr = projector.feed("\n</think>\n\n"); sinkErr != nil {
				cancel()
			}
		}
	}))...)
	// Do not flush withheld stop prefixes or ambiguous control delimiters after a
	// decode error: those bytes were never established as safe visible content.
	if sinkErr == nil && err == nil {
		sinkErr = projector.flush()
	}
	if sinkErr != nil {
		return comp, sinkErr
	}
	return comp, err
}

// inKernelStreamProjector keeps raw decode control text off the content stream.
// The span guard suppresses reasoning and tool-call regions while stopSuffixGuard
// retains a possible stop suffix until it is known to be ordinary prose. Both run
// before the caller's sink, so a sink failure cancels decode without another delivery.
type inKernelStreamProjector struct {
	spans *toolSpanGuard
	stops *stopSuffixGuard
}

func newInKernelStreamProjector(emit StreamSink, stops []string, startsInReasoning ...bool) *inKernelStreamProjector {
	seededReasoning := len(startsInReasoning) > 0 && startsInReasoning[0]
	spanGuard := newToolSpanGuardState(emit, seededReasoning)
	stopGuard := newStopSuffixGuard(spanGuard.feed, stops)
	return &inKernelStreamProjector{spans: spanGuard, stops: stopGuard}
}

func (p *inKernelStreamProjector) feed(piece string) error { return p.stops.feed(piece) }

func (p *inKernelStreamProjector) flush() error {
	if err := p.stops.flush(); err != nil {
		return err
	}
	return p.spans.flush()
}

type stopSuffixGuard struct {
	emit    StreamSink
	stops   []string
	held    string
	stopped bool
}

func newStopSuffixGuard(emit StreamSink, stops []string) *stopSuffixGuard {
	filtered := make([]string, 0, len(stops))
	for _, stop := range stops {
		if stop != "" {
			filtered = append(filtered, stop)
		}
	}
	return &stopSuffixGuard{emit: emit, stops: filtered}
}

func (g *stopSuffixGuard) feed(piece string) error {
	if piece == "" || g.stopped {
		return nil
	}
	buf := g.held + piece
	g.held = ""
	if trimmed, hit := checkStop(buf, g.stops); hit {
		g.stopped = true
		return g.emitText(trimmed)
	}
	keep := 0
	for _, stop := range g.stops {
		max := len(stop) - 1
		if max > len(buf) {
			max = len(buf)
		}
		for n := max; n > keep; n-- {
			if strings.HasSuffix(buf, stop[:n]) {
				keep = n
				break
			}
		}
	}
	if keep > 0 {
		g.held = buf[len(buf)-keep:]
		buf = buf[:len(buf)-keep]
	}
	return g.emitText(buf)
}

func (g *stopSuffixGuard) flush() error {
	if g.stopped {
		return nil
	}
	held := g.held
	g.held = ""
	return g.emitText(held)
}

func (g *stopSuffixGuard) emitText(text string) error {
	if text == "" {
		return nil
	}
	return g.emit(text)
}

// toolSpanGuard sits between the in-kernel per-token decode seam and the client sink.
// The decode seam forwards RAW text, before the post-decode tool-call lift strips
// explicit tool-call markup; forwarding it verbatim would leak tool-call syntax into the
// client's prose stream before adjudication ever sees a ToolCall. The guard therefore
// holds text that could be the start of an explicit tool-call span, and once a span is
// confirmed it DROPS it (the span is recovered post-decode by the tool-call lift and
// returned in the final Completion, never as streamed content). Prose before and after a
// span is forwarded normally, so the concatenated content remains a prefix of the final
// post-lift Content the caller reconciles against.
//
// Only UNAMBIGUOUS tag/delimiter openers are guarded; a generic ``` fence or a bare `{`
// is legitimate prose and is never suppressed. Partial openers split across token pieces
// are held until they resolve (open) or definitively cannot be an opener (flushed).
type toolSpanGuard struct {
	emit                StreamSink
	held                strings.Builder
	span                strings.Builder
	inSpan              bool
	spanReasoning       bool
	trimReasoningOutput bool
	contentStarted      bool
	trailingSpace       strings.Builder
	closers             []string
	cap                 int
}

// toolSpanOpeners lists the explicit tool-call span openers the guard suppresses, each
// paired with the closers that end its span. An opener with no closer drops to the end
// of the turn (there is no reliable terminator in that dialect).
var toolSpanOpeners = []struct {
	open      string
	closers   []string
	reasoning bool
}{
	{open: thinkOpen, closers: []string{thinkClose}, reasoning: true},
	{open: "<tool_call>", closers: []string{"</tool_call>"}},
	{open: "<function_call>", closers: []string{"</function_call>"}},
	{open: "<|python_tag|>", closers: []string{"<|eom_id|>", "<|eot_id|>"}},
	{open: "[TOOL_CALLS]", closers: nil},
}

const toolSpanGuardMaxBytes = 4 << 20

func newToolSpanGuard(emit StreamSink) *toolSpanGuard {
	return newToolSpanGuardState(emit, false)
}

func newToolSpanGuardState(emit StreamSink, startsInReasoning bool) *toolSpanGuard {
	g := &toolSpanGuard{emit: emit, cap: toolSpanGuardMaxBytes}
	if startsInReasoning {
		g.inSpan = true
		g.spanReasoning = true
		g.closers = []string{thinkClose}
	}
	return g
}

// feed consumes one raw decoded piece and forwards only prose.
func (g *toolSpanGuard) feed(piece string) error {
	if g.inSpan {
		g.span.WriteString(piece)
		return g.drainSpan()
	}
	g.held.WriteString(piece)
	return g.drainHeld()
}

// flush emits any held non-span text at end of turn. A confirmed span is never flushed.
func (g *toolSpanGuard) flush() error {
	if g.inSpan {
		return nil
	}
	rest := g.held.String()
	g.held.Reset()
	if err := g.emitText(rest); err != nil {
		return err
	}
	// splitReasoning trims final content only after a reasoning span. Retained
	// whitespace is therefore a trailing trim candidate and must not be emitted.
	g.trailingSpace.Reset()
	return nil
}

// drainSpan drops buffered span text through the first closer, then hands any trailing
// text back to the prose scanner. A span that never closes (or an over-cap buffer) is
// dropped to end of turn without ever reaching the sink.
func (g *toolSpanGuard) drainSpan() error {
	buf := g.span.String()
	for _, closer := range g.closers {
		if idx := strings.Index(buf, closer); idx >= 0 {
			tail := buf[idx+len(closer):]
			wasReasoning := g.spanReasoning
			g.span.Reset()
			g.inSpan = false
			g.spanReasoning = false
			g.closers = nil
			if wasReasoning {
				g.trimReasoningOutput = true
			}
			g.held.WriteString(tail)
			return g.drainHeld()
		}
	}
	if g.closers == nil || len(buf) > g.cap {
		// No terminator will arrive: drop the rest of the turn.
		g.span.Reset()
	}
	return nil
}

// drainHeld scans buffered prose for the earliest confirmed opener. Text before it is
// emitted; the opener switches the guard into span mode. If the buffer ends in a partial
// opener prefix, that suffix is held until the next piece resolves it.
func (g *toolSpanGuard) drainHeld() error {
	buf := g.held.String()
	earliest := -1
	var matched []string
	matchedReasoning := false
	for _, o := range toolSpanOpeners {
		if idx := strings.Index(buf, o.open); idx >= 0 && (earliest < 0 || idx < earliest) {
			earliest, matched, matchedReasoning = idx, o.closers, o.reasoning
		}
	}
	if earliest >= 0 {
		if err := g.emitText(buf[:earliest]); err != nil {
			return err
		}
		tail := buf[earliest+lenMatchedOpener(buf[earliest:]):]
		g.held.Reset()
		g.span.Reset()
		g.span.WriteString(tail)
		g.inSpan = true
		g.spanReasoning = matchedReasoning
		g.closers = matched
		return g.drainSpan()
	}
	if keep := partialOpenerSuffix(buf); keep > 0 {
		if err := g.emitText(buf[:len(buf)-keep]); err != nil {
			return err
		}
		rest := buf[len(buf)-keep:]
		g.held.Reset()
		g.held.WriteString(rest)
		return nil
	}
	g.held.Reset()
	return g.emitText(buf)
}

func (g *toolSpanGuard) emitText(text string) error {
	if text == "" {
		return nil
	}
	if !g.trimReasoningOutput {
		return g.emit(text)
	}
	if !g.contentStarted {
		text = strings.TrimLeftFunc(text, unicode.IsSpace)
		if text == "" {
			return nil
		}
		g.contentStarted = true
	}
	text = g.trailingSpace.String() + text
	g.trailingSpace.Reset()
	visible := strings.TrimRightFunc(text, unicode.IsSpace)
	if len(visible) < len(text) {
		g.trailingSpace.WriteString(text[len(visible):])
	}
	if visible == "" {
		return nil
	}
	return g.emit(visible)
}

// lenMatchedOpener returns the byte length of the opener that prefixes s, so the opener
// itself is consumed into the span buffer (and later dropped) rather than re-scanned.
func lenMatchedOpener(s string) int {
	for _, o := range toolSpanOpeners {
		if strings.HasPrefix(s, o.open) {
			return len(o.open)
		}
	}
	return 0
}

// partialOpenerSuffix returns the length of the longest suffix of s that is a proper
// (non-empty, shorter-than-full) prefix of any guarded opener — the bytes that must be
// held because the next piece may complete an opener.
func partialOpenerSuffix(s string) int {
	best := 0
	for _, o := range toolSpanOpeners {
		max := len(o.open) - 1
		if max > len(s) {
			max = len(s)
		}
		for n := max; n > best; n-- {
			if strings.HasPrefix(o.open, s[len(s)-n:]) {
				best = n
				break
			}
		}
	}
	return best
}

// SetPromptShrinkLevers configures the prompt-shrink levers for this planner.
func (p *InKernelPlanner) SetPromptShrinkLevers(compactBudget int, elideStaleReads, deferColdTools bool) {
	if p == nil {
		return
	}
	p.compactHistoryBudget = compactBudget
	p.elideStaleReads = elideStaleReads
	p.deferColdTools = deferColdTools
}

// SetRestoreStash attaches a content-addressed storage callback for elided / compacted turns.
func (p *InKernelPlanner) SetRestoreStash(stash func(trace, id, excerpt string, body []byte)) {
	if p == nil {
		return
	}
	p.restoreStash = stash
}

// SetIncrementalContext installs the composed system-context registry for this planner
// (inkernel_incremental.go). The durable comparison snapshot is deliberately left in
// place: it is what lets a reconcile detect that a source has LEFT the registry (and
// require a replacement) rather than treating the swap as a fresh, empty comparison. A
// nil ctx is the inert identity, preserving the historical planner byte-for-byte.
func (p *InKernelPlanner) SetIncrementalContext(ctx *IncrementalContext) {
	if p == nil {
		return
	}
	p.ctxMu.Lock()
	defer p.ctxMu.Unlock()
	p.incrementalContext = ctx
}

// IncrementalContextGeneration reports a fresh baseline generation for the installed
// registry. The bool is false (and the generation zero) when no registry is set; the
// error is *InitializationBlockedError when any source is temporarily unavailable.
func (p *InKernelPlanner) IncrementalContextGeneration() (ContextGeneration, bool, error) {
	if p == nil {
		return ContextGeneration{}, false, nil
	}
	p.ctxMu.Lock()
	ctx := p.incrementalContext
	p.ctxMu.Unlock()
	if ctx == nil {
		return ContextGeneration{}, false, nil
	}
	generation, err := InitializeIncrementalContext(ctx)
	return generation, true, err
}

// ReconcileIncrementalContextNow reconciles the installed registry against the
// planner's stored snapshot. With no registry the result is Unchanged. On Updated or
// ReplacementReady the stored snapshot ADVANCES (correct context invalidation); on
// ReplacementBlocked it is left untouched so a later retry still sees the admitted
// state.
//
// The caller must NOT mutate the returned ReconcileResult.Snapshot: on Unchanged it
// aliases the planner's stored snapshot, and on Updated/ReplacementReady the planner
// retains the same map it returns. Treat it as read-only comparison state.
//
// ctxMu is held across the source load/baseline/update/removed callbacks, so those
// callbacks must NOT re-enter the planner (SetIncrementalContext,
// IncrementalContextGeneration, or this method) on pain of deadlock.
func (p *InKernelPlanner) ReconcileIncrementalContextNow() (ReconcileResult, error) {
	if p == nil {
		return ReconcileResult{Kind: ReconcileUnchanged}, nil
	}
	p.ctxMu.Lock()
	defer p.ctxMu.Unlock()
	if p.incrementalContext == nil {
		return ReconcileResult{Kind: ReconcileUnchanged}, nil
	}
	res, err := ReconcileIncrementalContext(p.incrementalContext, p.ctxSnapshot)
	if err != nil {
		return ReconcileResult{}, err
	}
	switch res.Kind {
	case ReconcileUpdated, ReconcileReplacementReady:
		p.ctxSnapshot = res.Snapshot
	}
	return res, nil
}

// ApplyIncrementalContextUpdate reconciles the installed registry and returns the
// model-visible incremental delta. invalidated is true exactly when a
// replacement_ready occurred, signalling the caller must re-baseline resident context.
func (p *InKernelPlanner) ApplyIncrementalContextUpdate() (string, bool, error) {
	res, err := p.ReconcileIncrementalContextNow()
	if err != nil {
		return "", false, err
	}
	return res.Text, res.Kind == ReconcileReplacementReady, nil
}

// ApplyPromptShrink evaluates the configured or request-level prompt-shrink levers
// over typed messages and tools before tokenization.
func (p *InKernelPlanner) ApplyPromptShrink(ctx context.Context, messages []Message, tools []ToolDef, opts ...SampleOpt) ([]Message, []ToolDef, TypedPromptShrinkOutcome) {
	sp := applySampleOpts(opts...)
	compactBudget := p.compactHistoryBudget
	if sp.CompactHistoryBudget != nil {
		compactBudget = *sp.CompactHistoryBudget
	}
	elideStale := p.elideStaleReads
	if sp.ElideStaleReads != nil {
		elideStale = *sp.ElideStaleReads
	}
	deferTools := p.deferColdTools
	if sp.DeferColdTools != nil {
		deferTools = *sp.DeferColdTools
	}

	if compactBudget <= 0 && !elideStale && !deferTools {
		return messages, tools, TypedPromptShrinkOutcome{}
	}

	var stash func(id, excerpt string, body []byte)
	if p.restoreStash != nil {
		traceID := nativePhaseTraceID(ctx)
		stash = func(id, excerpt string, body []byte) {
			p.restoreStash(traceID, id, excerpt, body)
		}
	}

	return ApplyTypedPromptShrinkLevers(messages, tools, TypedPromptShrinkConfig{
		CompactHistoryBudget:    compactBudget,
		ElideStaleReads:         elideStale,
		DeferColdTools:          deferTools,
		RestoreStash:            stash,
		observeNativeCompaction: nativeCompactionObserver(ctx) != nil,
	})
}

// KVMemoryStats reports the in-process KV prefix cache's physical resident shape.
// Native backend snapshots are split into hot device bytes, hot host metadata, and
// the independently owned host-DRAM L2. Proxy/provider counters never enter here.
// effectiveKVConfig is the compute.KVConfig this planner's admission math must use:
// the model's geometry plus the realized KV storage tier (p.kvPrecision, mapped to the
// compute tier). The F32 default maps to compute.KVPrecisionF32, so a planner that
// never set a tier estimates byte-identically to before.
func (p *InKernelPlanner) effectiveKVConfig() compute.KVConfig {
	cfg := p.m.Cfg.ContextSizeConfigWithPrecision(computeKVPrecisionFor(p.kvPrecision)).KV
	return cfg
}

// computeKVPrecisionFor maps a model.KVPrecision tier to the compute.KVPrecision the
// byte estimate understands. Only f32 and q8_0 are realized; any other declared tier
// falls back to f32 so the estimate never claims a density the engine does not have.
func computeKVPrecisionFor(prec model.KVPrecision) compute.KVPrecision {
	switch prec {
	case model.KVPrecisionQ8_0:
		return compute.KVPrecisionQ8
	default:
		return compute.KVPrecisionF32
	}
}

func (p *InKernelPlanner) KVMemoryStats() KVMemoryStats {
	if p == nil || p.m == nil {
		return KVMemoryStats{
			MemoryClass: string(compute.MemoryKVCache),
			Scope:       string(compute.MemoryScopeHost),
			DType:       compute.F32.String(),
		}
	}
	kvCfg := p.effectiveKVConfig()
	bytesPerToken := compute.EstimateKVStoreBytes(kvCfg, 1)
	stats := KVMemoryStats{
		Enabled:       p.tree != nil,
		Backend:       "radixkv",
		MemoryClass:   string(compute.MemoryKVCache),
		Scope:         string(compute.MemoryScopeHost),
		DType:         kvCfg.Precision.StorageLabel(),
		BytesPerToken: bytesPerToken,
		HeadroomRatio: inKernelKVMemoryHeadroom,
	}
	if p.backend != nil && p.tree == nil {
		stats.Enabled = false
		stats.Backend = p.backend.Name()
		stats.Scope = string(compute.MemoryScopeDevice)
		total, free, known := compute.DeviceMemoryInfo(p.backend)
		applyKVMemoryCapacity(&stats, total, free, known)
		return stats
	}
	hostTotal, hostFree, hostKnown := compute.HostSystemMemoryInfo()
	if p.tree == nil {
		applyKVMemoryCapacity(&stats, hostTotal, hostFree, hostKnown)
		return stats
	}
	p.mu.Lock()
	st := p.tree.Stats()
	p.mu.Unlock()
	if p.backend != nil && p.backend.Caps().DeviceMemory {
		stats.Backend = p.backend.Name()
		stats.Scope = string(compute.MemoryScopeDevice)
		stats.ResidentTokens = st.DeviceSnapshotTokens
		stats.ResidentBytes = st.DeviceSnapshotBytes
		total, free, known := compute.DeviceMemoryInfo(p.backend)
		applyKVMemoryCapacity(&stats, total, free, known)
	} else {
		stats.ResidentTokens = st.PrefixTokens
		stats.ResidentBytes = compute.EstimateKVStoreBytes(kvCfg, st.PrefixTokens)
		applyKVMemoryCapacity(&stats, hostTotal, hostFree, hostKnown)
	}
	stats.BudgetTokens = st.MaxTokens
	stats.LRUTokens = st.Tokens
	stats.MaxDepthTokens = st.MaxDepthTokens
	stats.Nodes = st.Nodes
	stats.Leaves = st.Leaves
	stats.Evictions = st.Evictions
	stats.PolicyEvictions = st.PolicyEvictions
	stats.Splits = st.Splits
	stats.L1DeviceResidentBytes = st.DeviceSnapshotBytes
	stats.L1HostResidentBytes = st.DeviceSnapshotHostBytes
	stats.L2HostResidentBytes = st.HostSnapshotBytes
	stats.L2HostCapacityBytes = st.MaxHostSnapshotBytes
	stats.L1Hits = st.L1Hits
	stats.L1Misses = st.L1Misses
	stats.L1Faults = st.L1Faults
	stats.L1HitTokens = st.L1HitTokens
	stats.L2Hits = st.L2Hits
	stats.L2Misses = st.L2Misses
	stats.L2Faults = st.L2Faults
	stats.L2HitTokens = st.L2HitTokens
	stats.L2StageBytes = st.L2StageBytes
	stats.L2RestoreBytes = st.L2RestoreBytes
	stats.L2Evictions = st.L2Evictions
	stats.L3Enabled = st.L3Enabled
	stats.L3ReferencedBytes = st.L3ReferencedBytes
	stats.L3Hits = st.L3Hits
	stats.L3Misses = st.L3Misses
	stats.L3Faults = st.L3Faults
	stats.L3HitTokens = st.L3HitTokens
	stats.L3StageBytes = st.L3StageBytes
	stats.L3RestoreBytes = st.L3RestoreBytes
	stats.L3StageNanos = st.L3StageNanos
	stats.L3RestoreNanos = st.L3RestoreNanos
	stats.L3StageFaults = st.L3StageFaults
	stats.L3RestoreFaults = st.L3RestoreFaults
	return stats
}

const inKernelKVMemoryHeadroom = 0.15

func applyKVMemoryCapacity(stats *KVMemoryStats, total, free int64, known bool) {
	if stats == nil || !known || total <= 0 {
		return
	}
	stats.CapacityKnown = true
	stats.CapacityTotalBytes = total
	if free != compute.FreeUnknown && free >= 0 {
		stats.CapacityFreeKnown = true
		stats.CapacityFreeBytes = free
	}
	budgetBase := total
	if stats.CapacityFreeKnown {
		budgetBase = SaturatingAddBytes(free, stats.ResidentBytes)
		if budgetBase > total {
			budgetBase = total
		}
	}
	stats.FitBudgetBytes = ApplyByteHeadroom(budgetBase, stats.HeadroomRatio)
	stats.FitMarginBytes = stats.FitBudgetBytes - stats.ResidentBytes
}

// SaturatingAddBytes adds two byte counts without ever wrapping: a non-positive b is a
// no-op, and a sum that would overflow int64 pins at maxInt64 instead. Device capacity
// figures arrive from several backends and a wrapped negative total would read as "no
// memory" and refuse a request that fits. Exported because the gateway's request-memory
// fit view (internal/gateway/memory_fit.go) totals the SAME quantities this planner
// reports and must saturate them identically, or the two views disagree at the ceiling.
func SaturatingAddBytes(a, b int64) int64 {
	const maxInt64 = int64(^uint64(0) >> 1)
	if b <= 0 {
		return a
	}
	if a > maxInt64-b {
		return maxInt64
	}
	return a + b
}

// ApplyByteHeadroom reserves a fraction of a byte budget: it returns bytes scaled down by
// headroom, treating a non-positive budget as zero and an out-of-range headroom (<=0 or
// >=1) as "reserve nothing" rather than as a clamp — a 0 budget must stay 0, and a bogus
// ratio must never silently zero a real budget. Exported for the same reason as
// SaturatingAddBytes: the gateway renders the headroom-adjusted view of these budgets.
func ApplyByteHeadroom(bytes int64, headroom float64) int64 {
	if bytes <= 0 {
		return 0
	}
	if headroom <= 0 || headroom >= 1 {
		return bytes
	}
	return int64(float64(bytes) * (1 - headroom))
}

func (p *InKernelPlanner) RequestMemoryStats() RequestMemoryStats {
	if p == nil {
		return RequestMemoryStats{}
	}
	p.reqMemMu.Lock()
	defer p.reqMemMu.Unlock()
	out := p.lastReqMemory
	out.MemoryPlan = append([]RequestMemoryDemand(nil), p.lastReqMemory.MemoryPlan...)
	out.Capacities = append([]RequestMemoryCapacity(nil), p.lastReqMemory.Capacities...)
	return out
}

// nativePhaseLogCap bounds the per-planner native-phase ledger. It is small because the
// ledger holds only the MOST RECENT observation per trace id, and one serve usually has a
// handful of in-flight requests; the cap exists so an unbounded stream of distinct trace ids
// (or the empty-trace-id bucket) cannot grow the map without limit.
const nativePhaseLogCap = 64

// recordNativePhase stores the most recent observation for traceID. It is a no-op for an
// unknown phase token, so the closed vocabulary cannot be widened by a caller's typo. The
// empty trace id is a valid key: a request that carried no trace id is still attributed to
// its phase rather than dropped (NativePhaseObservation.TraceID stays empty, never fabricated).
func (p *InKernelPlanner) recordNativePhase(traceID string, phase NativePhase, at time.Time, elapsed time.Duration, completed bool) {
	if p == nil || !nativePhaseKnown(phase) {
		return
	}
	obs := NativePhaseObservation{TraceID: traceID, Phase: phase, At: at, Elapsed: elapsed, Completed: completed}
	p.storeNativePhaseObservation(obs, false)
}

func (p *InKernelPlanner) storeNativePhaseObservation(obs NativePhaseObservation, reset bool) {
	traceID := obs.TraceID
	p.nativePhaseMu.Lock()
	defer p.nativePhaseMu.Unlock()
	if p.nativePhaseLog == nil {
		p.nativePhaseLog = make(map[string]NativePhaseObservation, nativePhaseLogCap)
	}
	if _, seen := p.nativePhaseLog[traceID]; !seen {
		for len(p.nativePhaseSeq) >= nativePhaseLogCap {
			oldest := p.nativePhaseSeq[0]
			p.nativePhaseSeq = p.nativePhaseSeq[1:]
			if _, still := p.nativePhaseLog[oldest]; still {
				delete(p.nativePhaseLog, oldest)
				break
			}
		}
	}
	p.touchNativePhaseLocked(traceID)
	if !reset {
		obs.FirstDraw = p.nativePhaseLog[traceID].FirstDraw
	}
	obs.UpdatedAt = time.Now()
	p.nativePhaseLog[traceID] = obs
}

func (p *InKernelPlanner) touchNativePhaseLocked(traceID string) {
	for i, key := range p.nativePhaseSeq {
		if key == traceID {
			p.nativePhaseSeq = append(p.nativePhaseSeq[:i], p.nativePhaseSeq[i+1:]...)
			break
		}
	}
	p.nativePhaseSeq = append(p.nativePhaseSeq, traceID)
}

func cloneNativePhaseObservation(obs NativePhaseObservation) NativePhaseObservation {
	if obs.FirstDraw != nil {
		draw := *obs.FirstDraw
		if draw.TokenID != nil {
			token := *draw.TokenID
			draw.TokenID = &token
		}
		obs.FirstDraw = &draw
	}
	return obs
}

func (p *InKernelPlanner) beginNativeFirstDraw(traceID string, ceiling int) {
	obs := NativePhaseObservation{TraceID: traceID, Phase: NativePhasePrefill, At: time.Now(),
		FirstDraw: &NativeFirstDrawObservation{ResolvedOutputCeiling: ceiling, Classification: "not_drawn"}}
	p.storeNativePhaseObservation(obs, true)
}

func (p *InKernelPlanner) recordNativeFirstDraw(traceID string, token int, stop bool) {
	p.nativePhaseMu.Lock()
	defer p.nativePhaseMu.Unlock()
	obs, ok := p.nativePhaseLog[traceID]
	if !ok || obs.FirstDraw == nil || obs.FirstDraw.Observed {
		return
	}
	draw := *obs.FirstDraw
	draw.Observed, draw.TokenID, draw.Classification = true, &token, "non_stop_token"
	if token < 0 {
		draw.Classification = "negative_token"
	} else if stop {
		draw.Classification = "stop_token"
	}
	obs.FirstDraw = &draw
	obs.UpdatedAt = time.Now()
	p.nativePhaseLog[traceID] = obs
	p.touchNativePhaseLocked(traceID)
}

// LatestNativePhaseObservation reads the latest actual phase from the same
// bounded ledger. Empty trace IDs remain valid exact keys in the original API.
func (p *InKernelPlanner) LatestNativePhaseObservation() (NativePhaseObservation, bool) {
	if p == nil {
		return NativePhaseObservation{}, false
	}
	p.nativePhaseMu.Lock()
	defer p.nativePhaseMu.Unlock()
	for i := len(p.nativePhaseSeq) - 1; i >= 0; i-- {
		if obs, ok := p.nativePhaseLog[p.nativePhaseSeq[i]]; ok {
			return cloneNativePhaseObservation(obs), true
		}
	}
	return NativePhaseObservation{}, false
}

// NativePhaseObservation reports the most recent native-phase observation recorded for traceID.
// It implements NativePhaseReporter, so the gateway can type-assert a planner for this optional
// seam and emit nothing for a proxy planner that does not implement it. A nil planner reports
// not-observed.
func (p *InKernelPlanner) NativePhaseObservation(traceID string) (NativePhaseObservation, bool) {
	if p == nil {
		return NativePhaseObservation{}, false
	}
	p.nativePhaseMu.Lock()
	defer p.nativePhaseMu.Unlock()
	obs, ok := p.nativePhaseLog[traceID]
	return cloneNativePhaseObservation(obs), ok
}

func (p *InKernelPlanner) InKernelOOMRetryStats() InKernelOOMRetryStats {
	if p == nil {
		return InKernelOOMRetryStats{}
	}
	backend := "unknown"
	if p.backend != nil {
		backend = p.backend.Name()
	}
	p.oomRetryMu.Lock()
	defer p.oomRetryMu.Unlock()
	out := InKernelOOMRetryStats{Backend: backend, Rows: make([]InKernelOOMRetryClassStats, 0, len(p.oomRetry))}
	for class, st := range p.oomRetry {
		if st == nil {
			continue
		}
		out.Rows = append(out.Rows, InKernelOOMRetryClassStats{
			Class:           class,
			Attempts:        st.attempts,
			Successes:       st.successes,
			Failures:        st.failures,
			LastFailedBytes: st.lastFailedBytes,
			LastSite:        st.lastSite,
		})
	}
	sort.SliceStable(out.Rows, func(i, j int) bool { return out.Rows[i].Class < out.Rows[j].Class })
	return out
}

func (p *InKernelPlanner) InKernelMemoryPressureTrimStats() InKernelMemoryPressureTrimStats {
	if p == nil {
		return InKernelMemoryPressureTrimStats{}
	}
	backend := "unknown"
	if p.backend != nil {
		backend = p.backend.Name()
	}
	p.pressureTrimMu.Lock()
	defer p.pressureTrimMu.Unlock()
	out := InKernelMemoryPressureTrimStats{Backend: backend, Rows: make([]InKernelMemoryPressureTrimClassStats, 0, len(p.pressureTrim))}
	for key, st := range p.pressureTrim {
		if st == nil {
			continue
		}
		out.Rows = append(out.Rows, InKernelMemoryPressureTrimClassStats{
			Scope:           key.scope,
			Class:           key.class,
			Reason:          key.reason,
			Attempts:        st.attempts,
			Trimmed:         st.trimmed,
			NoHooks:         st.noHooks,
			Resolved:        st.resolved,
			LastWantBytes:   st.lastWantBytes,
			LastBudgetBytes: st.lastBudgetBytes,
			LastMarginBytes: st.lastMarginBytes,
		})
	}
	sort.SliceStable(out.Rows, func(i, j int) bool {
		a, b := out.Rows[i], out.Rows[j]
		if a.Scope != b.Scope {
			return a.Scope < b.Scope
		}
		if a.Class != b.Class {
			return a.Class < b.Class
		}
		return a.Reason < b.Reason
	})
	return out
}

// Complete renders the transcript as ChatML and runs one in-kernel decode turn,
// returning the generated assistant text. Mirrors cmd/fakchat's hybrid path. The
// per-request SampleOpts override this planner's configured decode length,
// temperature, TopP (nucleus cutoff), and TopK (top-k cutoff) for THIS turn, and a
// per-request Stop sequence ends the turn early (string-suffix stop, orthogonal to
// the token-ID <|im_end|>/EOS stops). All five per-request sampling controls the
// HTTP wires forward are now honored on the in-kernel path too.
// InKernelOOMError is the agent-level, recovered form of an in-kernel device allocation
// failure (a *compute.DeviceAllocError that unwound out of a device decode path). It is
// in-kernel BY CONSTRUCTION — only the in-kernel planner / compute backend can produce it,
// never a real upstream — so the gateway can safely render a specific, actionable client
// message for it (an over-large prompt on a small GPU) without any risk of leaking upstream
// content. Bytes is the device allocation that failed; Class and Site preserve the allocator
// category for operator visibility without exposing model/provider content.
type InKernelOOMError struct {
	Bytes int
	Class compute.MemoryClass
	Site  string
}

func (e *InKernelOOMError) Error() string {
	class := e.Class
	if class == "" {
		class = compute.MemoryUnknown
	}
	if class == compute.MemoryUnknown {
		return fmt.Sprintf("in-kernel GPU out of memory (device allocation of %d bytes failed)", e.Bytes)
	}
	return fmt.Sprintf("in-kernel GPU out of memory (%s allocation of %d bytes failed)", class, e.Bytes)
}

// InKernelCapacityError is the request-time companion to InKernelOOMError: a backend
// with known capacity can refuse the planned in-kernel request memory before the device
// allocator is touched. It is still a local OOM-class resource exhaustion, but it is
// earlier and more actionable than a recovered DeviceAllocError.
type InKernelCapacityError struct {
	Want  int64
	Avail int64
	Class compute.MemoryClass
	Scope compute.MemoryScope
	Site  string
	// Detail carries the typed cause when no byte figure exists (e.g. the runtime-extras
	// estimator's missing bound), so the message never reports a misleading "needs 0 bytes".
	Detail string
	// Cause is the typed upstream reason when the refusal is not a byte-budget verdict
	// (the *compute.RuntimeExtraCapacityUnknownError behind runtime-extras-unknown), so
	// callers can errors.As the missing bound instead of parsing Detail.
	Cause error
	// AvailSigned is the unclamped budget (ceiling - used - reserved) on the host-memory
	// arm; negative means the resident footprint already exceeds the ceiling. Avail stays
	// clamped at zero for existing callers.
	AvailSigned int64
	// Structural marks a host-memory refusal with no turn in flight and the resident
	// footprint at or above the ceiling: waiting cannot free room, so it is not retryable.
	Structural bool
}

// inKernelRuntimeExtrasUnknownSite marks a refusal where the Qwen3.8 runtime-extras
// estimator could not bound the request plan (no Want/Avail figures exist).
const inKernelRuntimeExtrasUnknownSite = InKernelCapacitySiteRuntimeExtrasUnknown

// InKernelCapacitySiteRuntimeExtrasUnknown is the exported Site of a refusal raised because
// a required runtime-extras bound could not be priced. Want/Avail carry no meaning there.
const InKernelCapacitySiteRuntimeExtrasUnknown = "runtime-extras-unknown"

// ErrInKernelRuntimeExtrasUnknown matches (errors.Is) any InKernelCapacityError whose Site is
// InKernelCapacitySiteRuntimeExtrasUnknown: the closed contract for "the engine could not bound
// its runtime extras", distinct from a known-too-small byte budget.
var ErrInKernelRuntimeExtrasUnknown = errors.New("in-kernel runtime-extras capacity unknown")

// Is reports whether target is ErrInKernelRuntimeExtrasUnknown and this refusal carries that site.
func (e *InKernelCapacityError) Is(target error) bool {
	return target == ErrInKernelRuntimeExtrasUnknown && e != nil && e.Site == InKernelCapacitySiteRuntimeExtrasUnknown
}

// Unwrap exposes the typed cause (if any) to errors.As.
func (e *InKernelCapacityError) Unwrap() error { return e.Cause }

func (e *InKernelCapacityError) Error() string {
	class := e.Class
	if class == "" {
		class = compute.MemoryUnknown
	}
	scope := e.Scope
	if scope == "" {
		scope = compute.MemoryScopeDevice
	}
	subject := "GPU"
	if scope == compute.MemoryScopeHost {
		subject = "host-memory"
	}
	if e.Site == inKernelRuntimeExtrasUnknownSite {
		// No byte figure exists here: the runtime-extras estimator could not bound the plan.
		// Printing "plan needs 0 bytes, available budget is 0 bytes" hid the real cause.
		reason := strings.TrimSpace(e.Detail)
		if reason == "" {
			reason = "runtime extras capacity unknown"
		}
		return fmt.Sprintf("in-kernel %s capacity precheck refused request (%s %s plan could not be bounded: %s; site=%s)", subject, scope, class, reason, e.Site)
	}
	if e.AvailSigned < 0 {
		return fmt.Sprintf("in-kernel %s capacity precheck refused request (%s %s plan needs %d bytes, available budget is %d bytes: resident usage exceeds the ceiling by %d bytes)", subject, scope, class, e.Want, e.AvailSigned, -e.AvailSigned)
	}
	return fmt.Sprintf("in-kernel %s capacity precheck refused request (%s %s plan needs %d bytes, available budget is %d bytes)", subject, scope, class, e.Want, e.Avail)
}

// recoverDevicePanic is the body of Complete's deferred recover, factored out so it is
// unit-testable without a GPU (the panic payload is an ordinary Go value). It converts a
// recovered in-kernel device failure (allocation OOM, typed CUDA backend error, or device fault)
// into a typed, actionable error and reports handled=true; for ANY other recovered value it
// reports handled=false so the caller re-panics — the recover stays surgical and never swallows
// a genuine bug (a nil deref, a non-device validation panic).
func recoverDevicePanic(r any) (err error, handled bool) {
	if e, ok := r.(error); ok {
		var operation *model.BackendForwardOperationError
		if errors.As(e, &operation) {
			return e, true
		}
		var projectionOperation *model.V41ProjectionOperationError
		if errors.As(e, &projectionOperation) {
			return e, true
		}
		var expertOperation *model.V41ExpertOperationError
		if errors.As(e, &expertOperation) {
			return e, true
		}
		var stall metalgemm.MetalCommandBufferStallError
		if errors.As(e, &stall) {
			return e, true
		}
		var stallPtr *metalgemm.MetalCommandBufferStallError
		if errors.As(e, &stallPtr) {
			return e, true
		}
	}
	var dae *compute.DeviceAllocError
	if e, ok := r.(error); ok && errors.As(e, &dae) {
		return &InKernelOOMError{Bytes: dae.Bytes, Class: dae.DemandClass(), Site: dae.Site}, true
	}
	var cudaLaunchErr *compute.CUDALaunchError
	if e, ok := r.(error); ok && errors.As(e, &cudaLaunchErr) {
		return e, true
	}
	var cudaOpErr *compute.CUDAOpError
	if e, ok := r.(error); ok && errors.As(e, &cudaOpErr) {
		return e, true
	}
	var faultErr *compute.DeviceFaultError
	if e, ok := r.(error); ok && errors.As(e, &faultErr) {
		return e, true
	}
	return nil, false
}

type inKernelGenerateResult struct {
	gen, promptTok, matched int
	// cacheable is the lookup-side prefix match (#3390): tokens the radix index matched
	// BEFORE servability (nil KV, exact-hit refeed, unsupported truncate) could trim the
	// realized `matched` below it. cacheable >= matched always; 0 when reuse is off.
	cacheable         int
	sourceTier        radixkv.SnapshotTier
	prefillS, decodeS float64
	stopped           bool
	batchReceipt      InKernelBatchReceipt
	vulkanMTP         *VulkanMTPExecution
	// spec is this request's own verified draft rounds (zero off the speculative
	// paths); enginestep.Default only holds the process-wide sum.
	spec SpeculativeDecodeTally
}

// VulkanMTPExecution is the request-local execution receipt for the resident
// Vulkan MTP route. It is attached to Completion so the gateway reports what
// the completed request actually ran instead of reading mutable planner state.
type VulkanMTPExecution struct {
	Engine         string
	Backend        string
	RequestedDepth int
	EffectiveDepth int
	ProposalRounds int
	ProposedTokens int
	AcceptedTokens int
	RejectedTokens int
	RollbackTokens int
	// The target counters below cover observed resident-device verification
	// receipts. They do not include ordinary target work after a downgrade.
	TargetOperations      int
	TargetDecodeSteps     int
	FullTargetReplaySteps int
	RecurrentRepairTokens int
	Used                  bool
	DowngradeReason       string
	Cancelled             bool
	Elapsed               time.Duration
}

const (
	vulkanMTPExecutionEngine             = "mtp-vulkan"
	vulkanMTPDowngradeConstructorRefused = "constructor-refused"
	vulkanMTPDowngradeEmptyProposal      = "empty-proposal"
	vulkanMTPDowngradeProposalError      = "proposal-error"
	vulkanMTPDowngradeTargetVerifier     = "target-verifier-downgrade"
	vulkanMTPDowngradeAcceptanceFloor    = "acceptance-floor-fallback"
	vulkanMTPDowngradeCancelled          = "cancelled"
)

func (p *InKernelPlanner) generateReusedRecovering(ctx context.Context, ids []int, maxNew int, temp, topP float64, topK int, logitBias model.LogitBias, freqPenalty, presPenalty float64, stops map[int]bool, emit func(int) bool, measurementOpt ...*nativeInferenceMeasurement) (res inKernelGenerateResult, err error) {
	defer func() {
		if r := recover(); r != nil {
			if e, ok := recoverDevicePanic(r); ok {
				err = e
				return
			}
			panic(r)
		}
	}()
	targetOnly := false
	if coord := p.MetalMTPCoordinator(); coord != nil && p.qwen38MTPCanaryAllowsExecution() {
		return p.generateReusedMetalMTP(ctx, ids, maxNew, temp, topP, topK, logitBias, freqPenalty, presPenalty, stops, emit, measurementOpt...)
	} else if coord != nil {
		targetOnly = true
	}
	if !targetOnly && p.greedySpeculativeRequestEligible(temp, logitBias, freqPenalty, presPenalty) {
		return p.generateReusedSpeculative(ctx, ids, maxNew, temp, topP, topK, logitBias, freqPenalty, presPenalty, stops, emit, measurementOpt...)
	}
	gen, promptTok, cacheable, matched, sourceTier, prefillS, decodeS, stopped, err := p.generateReusedContextWithBias(ctx, ids, maxNew, temp, topP, topK, logitBias, freqPenalty, presPenalty, stops, emit, measurementOpt...)
	if err != nil {
		return inKernelGenerateResult{}, err
	}
	return inKernelGenerateResult{
		gen:        gen,
		promptTok:  promptTok,
		cacheable:  cacheable,
		matched:    matched,
		sourceTier: sourceTier,
		prefillS:   prefillS,
		decodeS:    decodeS,
		stopped:    stopped,
	}, nil
}

func (p *InKernelPlanner) generateReusedWithOOMRetry(ctx context.Context, ids []int, maxNew int, temp, topP float64, topK int, logitBias model.LogitBias, freqPenalty, presPenalty float64, stops map[int]bool, emit func(int) bool, onRetry func(), measurementOpt ...*nativeInferenceMeasurement) (inKernelGenerateResult, error) {
	res, err := p.generateReusedRecovering(ctx, ids, maxNew, temp, topP, topK, logitBias, freqPenalty, presPenalty, stops, emit, measurementOpt...)
	if err == nil {
		return res, nil
	}
	if retry, retryGate := p.admitDeviceOOMRetry(ctx, err, len(ids), maxNew); !retry {
		return res, retryGate
	}
	if onRetry != nil {
		onRetry()
	}
	retryRes, retryErr := p.generateReusedRecovering(ctx, ids, maxNew, temp, topP, topK, logitBias, freqPenalty, presPenalty, stops, emit, measurementOpt...)
	p.recordInKernelOOMRetry(err, retryErr == nil)
	return retryRes, retryErr
}

// admitDeviceOOMRetry decides whether a failed first attempt may run once more, and
// returns the error to surface when it may not. It is bounded (#13267): a cancelled
// request, a coalesced lane that already crossed the shared decode pass (a replay would
// fail closed with the fresh lane's nil error and book an empty completion as a
// successful retry), and a non-OOM failure never retry; a retry is admitted only after
// the idle pools were trimmed AND the request re-prices under the recovered capacity,
// otherwise it refuses typed instead of re-growing into the same wall.
func (p *InKernelPlanner) admitDeviceOOMRetry(ctx context.Context, err error, promptTokens, maxNew int) (bool, error) {
	if ctxErr := ctx.Err(); ctxErr != nil {
		return false, ctxErr
	}
	if req, ok := ctx.Value(inKernelCoalesceContextKey{}).(*inKernelCoalesceRequest); ok && req.decodePass.Load() > 0 {
		return false, err
	}
	if !p.prepareDeviceOOMRetry(err) {
		return false, err
	}
	if capErr := p.refuseOversizeRequest(promptTokens, maxNew); capErr != nil {
		// No retry ran, so none is booked: the retry counter counts attempted retries.
		return false, capErr
	}
	return true, nil
}

func (p *InKernelPlanner) prepareDeviceOOMRetry(err error) bool {
	if p == nil || p.backend == nil {
		return false
	}
	var oom *InKernelOOMError
	if !errors.As(err, &oom) {
		return false
	}
	released := p.trimBackendIdlePools()
	if released {
		log.Printf("inkernel_chat oom-retry model=%s backend=%s class=%s site=%s bytes=%d action=trim-idle-pools",
			p.modelID, p.backend.Name(), oom.Class, oom.Site, oom.Bytes)
	}
	return released
}

func (p *InKernelPlanner) trimBackendIdlePools() bool {
	if p == nil || p.backend == nil {
		return false
	}
	released := false
	if r, ok := p.backend.(interface{ Recycle() }); ok {
		r.Recycle()
		released = true
	}
	if t, ok := p.backend.(interface{ Trim() }); ok {
		t.Trim()
		released = true
	}
	if t, ok := p.backend.(interface{ TrimLarge(int) }); ok {
		t.TrimLarge(0)
		released = true
	}
	return released
}

func (p *InKernelPlanner) recordInKernelOOMRetry(trigger error, success bool) {
	if p == nil {
		return
	}
	class, bytes, site := inKernelOOMRetryTrigger(trigger)
	p.oomRetryMu.Lock()
	if p.oomRetry == nil {
		p.oomRetry = map[string]*inKernelOOMRetryClassStats{}
	}
	st := p.oomRetry[class]
	if st == nil {
		st = &inKernelOOMRetryClassStats{}
		p.oomRetry[class] = st
	}
	st.attempts++
	if success {
		st.successes++
	} else {
		st.failures++
	}
	st.lastFailedBytes = bytes
	st.lastSite = site
	p.oomRetryMu.Unlock()
}

func inKernelOOMRetryTrigger(err error) (class string, bytes uint64, site string) {
	var oom *InKernelOOMError
	if errors.As(err, &oom) {
		if oom.Bytes > 0 {
			bytes = uint64(oom.Bytes)
		}
		class = strings.TrimSpace(string(oom.Class))
		site = strings.TrimSpace(oom.Site)
	}
	if class == "" {
		class = string(compute.MemoryUnknown)
	}
	return class, bytes, site
}

// incrementalStopScanner is the decode loop's per-turn delta accrual state for the
// string-suffix Stop probe. Instead of materializing the whole accumulated text every
// token (the old per-token sb.String() was O(N) per token => O(N^2) per completion;
// issue #922 witnessed ~19x ns/token growth from 1024 to 8192 tokens), it accrues the
// exact bytes appended to the accumulator into a bounded tail window of the longest
// stop length and fingerprints only that window per token — O(maxStop) per token,
// O(N) total string work per completion. Fires identically to checkStop: non-empty
// stops only, suffix match, longest match wins, and only once len(stop) bytes have
// actually been appended (no false fire on a short window). maxStop <= 0 (no stop
// strings at all) degenerates to a no-op so the default no-stop decode path pays zero
// probe cost; the caller-level checkStop (hoisted empty-set early-out) stays the
// authoritative trim, re-derived once per completion on the rare fire path.
type incrementalStopScanner struct {
	// tail is the last min(total, maxStop) bytes of the accumulator — the window the
	// per-token probe fingerprints (len(tail) <= maxStop, the invariant tests pin).
	tail []byte
	// hist holds up to maxStop bytes immediately PRECEDING tail. It exists for the fire
	// path: a trim removes bytes from the END of the accumulator, so the window must
	// move BACK by len(matched) to remain the true suffix. Trimming "\n\n" from "\n\n\n"
	// leaves "\n" whose window is one byte OLDER than the pre-trim window; without hist
	// that byte is unrecoverable and the next "\n" would miss the "\n\n" fire. hist is
	// never part of the visible window, so the bounded-tail invariant still holds.
	hist  []byte
	stops []string
	total int
}

// newIncrementalStopScanner builds a scanner over the non-empty stop strings.
// maxStop is the length of the longest non-empty stop string (the caller derives it
// from the same request); it bounds the retained tail window and the per-token scan.
func newIncrementalStopScanner(maxStop int, stop []string) *incrementalStopScanner {
	s := &incrementalStopScanner{}
	if maxStop <= 0 {
		return s
	}
	s.tail = make([]byte, 0, maxStop)
	s.hist = make([]byte, 0, maxStop)
	for _, str := range stop {
		if str != "" {
			s.stops = append(s.stops, str)
		}
	}
	return s
}

// pushTail appends b to the logical sequence (hist+tail) while keeping tail at exactly
// the last min(logicalLen, cap(tail)) bytes and hist at up to cap(tail) bytes before it.
// Only the last 2*cap bytes can matter (a trim removes at most cap bytes from the end),
// so the work is O(maxStop) per call regardless of piece length.
func (s *incrementalStopScanner) pushTail(b []byte) {
	need := cap(s.tail)
	histLen, tailLen := len(s.hist), len(s.tail)
	total := histLen + tailLen + len(b)
	start := total - 2*need
	if start < 0 {
		start = 0
	}
	n := total - start
	buf := make([]byte, n)
	for i := 0; i < n; i++ {
		j := start + i
		switch {
		case j < histLen:
			buf[i] = s.hist[j]
		case j < histLen+tailLen:
			buf[i] = s.tail[j-histLen]
		default:
			buf[i] = b[j-histLen-tailLen]
		}
	}
	if n <= need {
		s.hist = s.hist[:0]
		s.tail = append(s.tail[:0], buf...)
		return
	}
	s.hist = append(s.hist[:0], buf[:n-need]...)
	s.tail = append(s.tail[:0], buf[n-need:]...)
}

// appendPiece accrues the exact bytes appended to the decode accumulator this token
// into the bounded tail window and re-fingerprints it (bytes.HasSuffix over at most
// maxStop bytes). Callers pass everything that was written to the accumulator — the
// token piece plus any budget-forced reasoning close — so the window sees precisely
// what checkStop would see over the whole buffer. The retained tail never exceeds
// maxStop, yet every fire the whole-buffer probe would report is still reported:
// a stop match lives entirely inside the last maxStop bytes once enough of them exist.
//
// On a fire the scanner trims the LONGEST matching stop from its own logical tail,
// mirroring checkStop's maximal trim of the accumulator. The caller never calls reset()
// on the ordinary fire path, so a scanner that kept the matched bytes would fire again
// on the next piece although checkStop (over the now-trimmed accumulator) would not.
// Refilling the window from hist keeps it an exact suffix of the caller's trimmed
// accumulator, so every future fire agrees and no retained byte is lost.
func (s *incrementalStopScanner) appendPiece(piece string) bool {
	if len(s.stops) == 0 {
		return false
	}
	s.total += len(piece)
	s.pushTail([]byte(piece))
	best := ""
	for _, str := range s.stops {
		if len(str) <= len(s.tail) && len(str) > len(best) && bytes.HasSuffix(s.tail, []byte(str)) {
			best = str
		}
	}
	if best == "" {
		return false
	}
	// Trim the matched stop from the tail, then refill from hist so the window is again
	// the last maxStop bytes of the trimmed accumulator.
	drop := len(best)
	if drop >= len(s.tail) {
		drop -= len(s.tail)
		s.tail = s.tail[:0]
		if drop >= len(s.hist) {
			s.hist = s.hist[:0]
		} else {
			s.hist = s.hist[:len(s.hist)-drop]
		}
	} else {
		s.tail = s.tail[:len(s.tail)-drop]
	}
	need := cap(s.tail) - len(s.tail)
	if need > len(s.hist) {
		need = len(s.hist)
	}
	if need > 0 {
		refill := s.hist[len(s.hist)-need:]
		s.tail = append(s.tail, refill...)
		s.hist = s.hist[:len(s.hist)-need]
	}
	return true
}

// reset re-initializes the scanner for a start-over retry turn, mirroring the
// accumulator's own reset (same fire decisions as a freshly constructed scanner).
func (s *incrementalStopScanner) reset() {
	s.tail = s.tail[:0]
	s.hist = s.hist[:0]
	s.total = 0
}

func (p *InKernelPlanner) Complete(ctx context.Context, messages []Message, tools []ToolDef, opts ...SampleOpt) (comp *Completion, err error) {
	// An in-kernel device-allocation failure (e.g. OOM on a small GPU under a large Claude
	// Code system prompt) panics deep below a CGO boundary with no error channel. Recover it
	// HERE — the narrowest Go frame that wraps the whole device decode (generateReused's
	// Prefill/Step + NewBackendSession's NewKV) AND returns the error the gateway already maps
	// to a client response — converting it into a typed error instead of crashing the serving
	// goroutine. Everything else re-panics, preserving today's crash/stack behavior for bugs.
	defer func() {
		if r := recover(); r != nil {
			if e, ok := recoverDevicePanic(r); ok {
				comp, err = nil, e
				return
			}
			panic(r)
		}
	}()
	defer enginestep.Default.RequestStart()()
	if p.qwenQ4KPrefillChunkTarget() && p.qwenQ4KPrefillChunkConfigErr != nil {
		return nil, p.qwenQ4KPrefillChunkConfigErr
	}
	if p.qwenQ4KPrefillChunkExplicit && p.qwenQ4KPrefillChunkTarget() {
		if _, capacityErr := p.deviceBoundedQwenQ4KPrefillChunkTokens(); capacityErr != nil {
			return nil, capacityErr
		}
	}
	sp := applySampleOpts(opts...)
	if sp.NativeDecodeTokenIDs && !sp.DecodeTrace {
		return nil, fmt.Errorf("native decode token IDs require a decode trace")
	}
	var requestStarted time.Time
	if sp.NativeInferenceReceipt {
		if _, _, err := p.nativeSelectionIdentity(); err != nil {
			return nil, &model.NativeInferenceReceiptUnsupportedError{Reason: err.Error()}
		}
		requestStarted = time.Now()
	}
	temp := p.temp
	if sp.Temperature != nil {
		temp = *sp.Temperature
	}
	// Per-request nucleus cutoff; 0 (the zero value) disables truncation so an omitted
	// top_p keeps the full softmax draw, identical to the pre-seam path.
	topP := 0.0
	if sp.TopP != nil {
		topP = *sp.TopP
	}
	// Per-request top-k; 0 (the zero value, and any value <=0) disables truncation so
	// an omitted top_k keeps the full distribution, identical to the pre-seam path.
	topK := 0
	if sp.TopK != nil {
		topK = *sp.TopK
	}
	var logitBias model.LogitBias
	if len(sp.LogitBias) > 0 {
		logitBias = model.LogitBias(sp.LogitBias)
	}
	// Per-request OpenAI repetition penalties; nil/omitted stays 0, which
	// sampleLogitsWithPenalty treats as a byte-for-byte no-op (the pre-#1705 path).
	var freqPenalty, presPenalty float64
	if sp.FrequencyPenalty != nil {
		freqPenalty = *sp.FrequencyPenalty
	}
	if sp.PresencePenalty != nil {
		presPenalty = *sp.PresencePenalty
	}
	if sp.NativeInferenceReceipt && (temp != 0 || topP != 0 || topK > 0 || len(logitBias) > 0 || freqPenalty != 0 || presPenalty != 0) {
		return nil, &model.NativeInferenceReceiptUnsupportedError{Reason: "requires greedy sampling over unmodified logits"}
	}
	prepared, err := p.preparePrompt(ctx, messages, tools, sp, opts...)
	if err != nil {
		return nil, err
	}
	messages, tools = prepared.messages, prepared.tools
	chat, ids, maxNew := prepared.rendered, prepared.ids, prepared.maxNew
	ctx = withInKernelSharedPrefixBoundary(ctx, prepared.sharedBoundary)
	if err := p.refuseContextLength(len(ids), maxNew); err != nil {
		return nil, err
	}
	stops := StopIDs(p.tok, p.m.Cfg)
	var turnContext []TurnAssessment
	if ta, ok := AssessTranscriptTurn(messages); ok {
		turnContext = append(turnContext, ta)
	}
	effectiveBudget := ResolveEffortBudget(sp.ReasoningEffort, sp.ThinkingBudget, turnContext...)
	startInSpan := strings.HasSuffix(chat, qwenThinkAssistantSeed)
	var tb *ThinkBudget
	if effectiveBudget > 0 {
		tb = NewThinkBudget(effectiveBudget, startInSpan)
	}
	// emit runs per generated token: decode the piece, accumulate the text, and apply the
	// per-request string-suffix Stop (orthogonal to the token-ID stops). Returning true
	// ends the turn with the token counted and its text trimmed (the stop string is not
	// echoed back, matching the HTTP wires). Factoring decode into this closure keeps the
	// token-level reuse/decode core (generateReused) tokenizer-free, so the candidate-#13
	// reuse and #14 eviction are witnessable on a synthetic model with no tokenizer fixture.
	//
	// Delta accrual: tokens append into the accumulator once; the stop probe scans only a
	// bounded tail window (the longest stop length) per token instead of re-materializing
	// the whole buffer per token — O(N) total string work per completion instead of
	// O(N^2) (issue #922). The fire path's whole-buffer checkStop stays the authoritative
	// longest-match trim and runs once per completion; with no stop strings the scanner
	// is a no-op and checkStop's hoisted empty-set early-out ends the probe at O(1).
	var sb strings.Builder
	maxStop := 0
	for _, s := range sp.Stop {
		if s != "" && len(s) > maxStop {
			maxStop = len(s)
		}
	}
	scanner := newIncrementalStopScanner(maxStop, sp.Stop)
	emit := func(next int) bool {
		if piece, derr := p.tok.Decode([]int{next}); derr == nil {
			sb.WriteString(piece)
			appended := piece
			if tb != nil && tb.Observe(piece) {
				sb.WriteString("\n</think>\n\n")
				appended += "\n</think>\n\n"
			}
			if sp.DecodeTokenObserver != nil {
				sp.DecodeTokenObserver(piece, "")
			}
			if scanner.appendPiece(appended) {
				if trimmed, hit := checkStop(sb.String(), sp.Stop); hit {
					sb.Reset()
					sb.WriteString(trimmed)
					return true
				}
			}
		}
		return false
	}

	// Serialize the entire device forward pass: a single-stream accelerator cannot run two
	// forwards at once without the concurrent op-streams corrupting shared device buffers
	// (see devMu). The plain CPU path owns a per-turn session and guards the shared radix tree
	// with p.mu itself, so it remains concurrent. Held across Prefill + decode.
	if p.requiresDeviceSerialization() && !p.coalescesQwenDecode() {
		// #1589 phase-timeline probe: record the fan-out boundary and the
		// serialized critical section. Both calls are nil-safe no-ops unless a
		// caller opted into a concurrency profile.
		phase := p.concurrencyProfile.admit()
		deviceWait := time.Now()
		p.devMu.Lock()
		enginestep.Default.ObservePhase(enginestep.PhaseDeviceWait, time.Since(deviceWait))
		p.concurrencyProfile.forwardEnter(phase, true)
		defer func() {
			p.concurrencyProfile.forwardExit(phase)
			p.devMu.Unlock()
		}()
		if err := p.refuseOversizeRequest(len(ids), maxNew); err != nil {
			return nil, err
		}
	} else if p.concurrencyProfile != nil {
		// Concurrent-unfused path (coalescing or CPU): still record the boundary
		// so the timeline can distinguish "no mutex serialized us" from "no
		// overlap happened at all".
		phase := p.concurrencyProfile.admit()
		p.concurrencyProfile.forwardEnter(phase, false)
		defer p.concurrencyProfile.forwardExit(phase)
	}
	if p.coalescesQwenDecode() {
		if err := p.refuseOversizeRequest(len(ids), maxNew); err != nil {
			return nil, err
		}
	}
	// #13267: the host-session (Metal) seam has no compute.Backend for refuseOversizeRequest
	// to ask, so an armed host ceiling prices it here, before any session or clone exists.
	ctx, releaseHostMemory, hostErr := p.admitHostMemory(ctx, len(ids), maxNew)
	if hostErr != nil {
		return nil, hostErr
	}
	defer releaseHostMemory()
	var measurement *nativeInferenceMeasurement
	if sp.NativeInferenceReceipt || (sp.DecodeTrace && sp.DecodeTokenObserver == nil) || sp.NativeDecodeTokenIDs {
		measurement = &nativeInferenceMeasurement{
			startedAt:             requestStarted,
			inferenceDisabled:     !sp.NativeInferenceReceipt,
			decodeTokenIDsEnabled: sp.NativeDecodeTokenIDs,
		}
		if sp.NativeDecodeTokenIDs {
			measurement.decodeTokenIDs = make([]int, 0, maxNew)
		}
		if sp.NativeInferenceReceipt {
			measurement.cudaImmutableWeightUploadsBefore, measurement.cudaImmutableWeightUploadsAvailable = cudaImmutableWeightUploadSnapshot(p.backend)
		}
		if sp.DecodeTrace {
			measurement.traceNow = p.decodeTraceNow
			if measurement.traceNow == nil {
				measurement.traceNow = time.Now
			}
		}
	}
	generate := func(runCtx context.Context) (inKernelGenerateResult, error) {
		return p.generateReusedWithOOMRetry(runCtx, ids, maxNew, temp, topP, topK, logitBias, freqPenalty, presPenalty, stops, emit, func() {
			sb.Reset()
			scanner.reset()
			if effectiveBudget > 0 {
				tb = NewThinkBudget(effectiveBudget, startInSpan)
			}
			if measurement != nil {
				measurement.reset()
			}
		}, measurement)
	}
	var genRes inKernelGenerateResult
	if p.coalescesQwenDecode() {
		genRes, err = p.runCoalescedGenerate(ctx, generate)
	} else {
		genRes, err = generate(ctx)
	}
	if err != nil {
		// A generate path that bailed before its decode boundary (prefix-snapshot, admission,
		// context cancel) never reached the in-loop terminal emit, so the request still gets a
		// terminal observation here — bound to the same trace id (#13120).
		p.recordNativePhase(nativePhaseTraceID(ctx), NativePhaseTerminal, time.Now(), 0, false)
		if genRes.vulkanMTP != nil {
			return &Completion{VulkanMTP: genRes.vulkanMTP}, err
		}
		return nil, err
	}
	gen, promptTok, matched, prefillS, decodeS, stopped := genRes.gen, genRes.promptTok, genRes.matched, genRes.prefillS, genRes.decodeS, genRes.stopped
	observeNativeCompaction(ctx, prepared.nativeCompaction)
	// finishReason is honest about WHY decode ended: "stop" when a token-ID stop or a
	// per-request Stop sequence fired, "length" when maxNew was the only limit hit.
	finishReason := "length"
	if stopped {
		finishReason = "stop"
	}

	// Witness line (mirrors cmd/fakchat): real per-turn prefill/decode tok/s through the
	// in-kernel model, now also reporting the RadixAttention prefix reuse (reused vs
	// prompt) so a served chat turn self-reports the candidate-#13 win. prefill tok/s is
	// over the COMPUTED suffix (prompt minus the reused prefix) — the work actually done.
	computed := promptTok - matched
	prefTPS, decTPS := 0.0, 0.0
	if prefillS > 0 {
		prefTPS = float64(computed) / prefillS
	}
	// Decode throughput is reported over COMPLETED post-prefill intervals only.
	// The first generated token is produced by the prefill block, so a one-token
	// turn has no decode interval to report; the historical gen/decodeS form
	// serialized that case as a near-arbitrary finite "success" rate (#13298).
	decTPS, _ = nativeDecodeRate(gen, decodeS)
	// #3176 Q1/Q2 decode witness: report the resolved Q8 SIMD kernel tier (+ whether the fused
	// fast decode GEMV engaged) and the effective decode-worker count, so an operator can SEE —
	// without wall-clock guessing — that the AVX2/AVX-512 lane fired (not the reference path) and
	// that decode parallelizes across a modest, capped stream count rather than either 1 core or
	// an oversubscribed all-core dispatch (the pathology 40e0afd fixed on many-core amd64).
	q8kern, q8fused := model.Q8DecodeKernel()
	q8fusedMark := ""
	if q8fused {
		q8fusedMark = "+fused"
	}
	p.logExecutionSummary(q8kern, q8fusedMark, promptTok, genRes.cacheable, matched, computed, prefillS, prefTPS, gen, decodeS, decTPS)
	// Feed the process-global KV-prefix reuse tap so this turn's split hit-rate reaches
	// /metrics, not just this log line — the live measurement of the frozen-trajectory
	// cache cliff (docs/explainers/frozen-trajectory-cache-cliff.md). #3390: BOTH halves —
	// the lookup-side index match (cacheability) and the realized serve (matched/prompt) —
	// so the gap lost to eviction/admission is observable, not folded into "miss".
	// #3391 adds (a) the eligibility-filtered denominator — sampled BEFORE this turn is
	// latched as admitted, so the always-cold first prefill books zero eligible tokens
	// instead of depressing the fair hit-rate — and (b) the (model, tenant) attribution,
	// with the tenant drawn from the SAME authenticated prefix-cache identity the scoped
	// tree isolates on (unscoped traffic normalizes to "unknown" inside the tap).
	eligibleTok := p.kvPrefixEligiblePromptTokens(promptTok)
	p.noteKVPrefixAdmitted()
	tenant := ""
	if owner, scoped := prefixCacheIdentityFromContext(ctx); scoped {
		tenant = owner.Tenant
	}
	cacheobs.Default.ObserveLabeled(cacheobs.Labels{Model: p.modelID, Tenant: tenant},
		promptTok, genRes.cacheable, matched, eligibleTok)
	// #3896 provenance axis: remote L3 matches are external transfers; L1/L2
	// matches stay local, and the unmatched suffix is local compute.
	localHit, externalHit := matched, 0
	if genRes.sourceTier == radixkv.SnapshotTierRemoteL3 {
		localHit, externalHit = 0, matched
	}
	cacheobs.Default.ObserveBySource(cacheobs.SourceLocalHit, localHit)
	cacheobs.Default.ObserveBySource(cacheobs.SourceExternalTransfer, externalHit)
	cacheobs.Default.ObserveBySource(cacheobs.SourceLocalCompute, promptTok-matched)
	compReuseEntry := cachemeta.FromProviderCache(cachemeta.ProviderCache{Provider: "fak-inkernel", ModelID: p.modelID, PromptTokens: int64(promptTok), CachedTokens: int64(matched)})

	// Split a Qwen3.5 reasoning block off the decoded text BEFORE it becomes Content
	// (and before the tool-call lift below reads it). A reasoning model (Ornith) opens
	// the turn with the open reasoning tag, then its final answer; renderChatMLTools does
	// NOT pre-seed the open tag, so the model emits both. splitReasoning is the in-kernel
	// equivalent of vLLM's --reasoning-parser qwen3: the reasoning lands in
	// ReasoningContent and only the post-reasoning answer flows into Content (and thus
	// into Claude Code's context). It is gated — a non-reasoning turn (no reasoning tags)
	// returns the decoded text untouched, so this is byte-identical to today for any
	// model that does not open a reasoning span.
	//
	// Raw-observer junction: the raw-text callback (rawText) carries the byte-identical
	// buffer the post-decode pipeline is about to consume, issued BEFORE splitReasoning
	// reads it. It is not called at all when no observer was requested.
	if sp.DecodeTokenObserver != nil {
		sp.DecodeTokenObserver("", sb.String())
	}
	reasoning, content := splitReasoning(sb.String())
	for strings.Contains(content, thinkClose) {
		idx := strings.Index(content, thinkClose)
		residual := strings.TrimSpace(content[:idx])
		if strings.HasPrefix(strings.ToLower(residual), thinkOpen) {
			residual = residual[len(thinkOpen):]
		}
		residual = strings.TrimSpace(residual)
		if residual != "" {
			if reasoning != "" {
				reasoning = strings.TrimSpace(reasoning + "\n" + residual)
			} else {
				reasoning = residual
			}
		}
		content = strings.TrimSpace(content[idx+len(thinkClose):])
	}
	if tb != nil && tb.Forced() {
		content = StripReasoning(content)
	}
	// Model reports the artifact this planner actually decoded, so the gateway
	// echoes the served model rather than whatever name the client sent.
	comp = &Completion{
		Model:         p.modelID,
		Message:       Message{Role: "assistant", Content: content, ReasoningContent: reasoning},
		FinishReason:  finishReason,
		ProviderCache: &compReuseEntry,
		Usage:         Usage{PromptTokens: promptTok, CompletionTokens: gen, TotalTokens: promptTok + gen, PromptTokensDetails: &UsageTokenDetails{CachedTokens: matched}},
		VulkanMTP:     genRes.vulkanMTP,
		Timings:       NewTimings(promptTok, matched, gen, prefillS, decodeS),
		NativeDecode:  newNativeDecodeSummary(genRes),
	}
	if sp.NativeInferenceReceipt {
		accounting := p.nativeCacheAccountingFor(nativeCacheAccountingFacts{
			promptTokens:    promptTok,
			cacheableTokens: genRes.cacheable,
			matchedTokens:   matched,
			sourceTier:      genRes.sourceTier,
			prefillSecs:     prefillS,
			generated:       gen,
		})
		receipt := p.buildNativeInferenceReceipt(measurement, prefillS, decodeS, accounting)
		renderedSum := sha256.Sum256([]byte(prepared.rendered))
		receipt.PromptTokenIDs = append([]int(nil), prepared.ids...)
		receipt.TokenizerID = p.tok.Identity()
		receipt.RendererID = inKernelPromptRendererID(p.m.Cfg, sp)
		receipt.RenderedSHA256 = hex.EncodeToString(renderedSum[:])
		comp.NativeInference = receipt
	}
	if genRes.batchReceipt.CohortID != 0 {
		receipt := genRes.batchReceipt
		comp.InKernelBatch = &receipt
	}
	if sp.DecodeTrace && measurement != nil {
		events := make([]NativeDecodeTraceEvent, len(measurement.traceEvents))
		copy(events, measurement.traceEvents)
		comp.DecodeTrace = &NativeDecodeTrace{
			Schema: NativeDecodeTraceSchema,
			Engine: NativeDecodeTraceEngine,
			Events: events,
		}
	}
	if sp.NativeDecodeTokenIDs && measurement != nil {
		comp.NativeDecodeTokenIDs = &NativeDecodeTokenIDs{
			Schema:   NativeDecodeTokenIDsSchema,
			Engine:   NativeDecodeTokenIDsEngine,
			TokenIDs: append([]int(nil), measurement.decodeTokenIDs...),
		}
	}
	// Lift the model's text-form <tool_call> emissions into structured Message.ToolCalls
	// (Hermes dialect == Qwen2.5 native), set FinishReason="tool_calls", and flag a
	// claimed-but-unparseable call — the SAME normalization every proxy adapter runs, so
	// the in-kernel forward becomes a first-class tool-calling planner. Without this the
	// gateway adjudicates nothing (it reads Message.ToolCalls) and the Anthropic wire never
	// emits a tool_use block, so Claude Code's agent loop has nothing to execute.
	comp = normalizeCompletionToolCalls(comp)
	// A length finish is a conformance failure only when the caller actually
	// forced a named tool. Calling enforceForcedToolChoice for an omitted/auto
	// choice marks every max_tokens completion as dropped before it even resolves
	// the effective tool name, which would make an exact T64 receipt unreachable.
	if inKernelEffectiveToolName(sp.ToolChoice, tools) != "" {
		comp = enforceForcedToolChoice(comp, sp.ToolChoice, tools, messages)
	}
	markInKernelDroppedToolCalls(comp)
	return comp, nil
}

func (p *InKernelPlanner) buildNativeInferenceReceipt(measurement *nativeInferenceMeasurement, prefillS, decodeS float64, accounting ...*model.NativeCacheAccounting) *model.NativeInferenceReceipt {
	backend, forwardPath := p.executionIdentity()
	nativeSelection, nativeSelectionDigest, _ := p.nativeSelectionIdentity()
	var qwen35MetalForwardSequence *model.Qwen35MetalForwardSequenceReceipt
	if measurement.qwen35MetalForwardSequence.EvidenceState != "" || measurement.qwen35MetalForwardSequence.Available {
		snapshot := measurement.qwen35MetalForwardSequence
		qwen35MetalForwardSequence = &snapshot
	}
	var qwen35MetalStateIdentity *model.Qwen35MetalStateIdentityReceipt
	if measurement.qwen35MetalStateIdentity != nil && measurement.qwen35MetalStateIdentity.Available {
		qwen35MetalStateIdentity = cloneQwen35MetalStateIdentityReceipt(*measurement.qwen35MetalStateIdentity)
	}
	var cudaImmutableWeightUploads *model.NativeCUDAImmutableWeightUploadDelta
	if after, ok := cudaImmutableWeightUploadSnapshot(p.backend); ok && measurement.cudaImmutableWeightUploadsAvailable {
		before := measurement.cudaImmutableWeightUploadsBefore
		if after.Calls >= before.Calls && after.TransferBytes >= before.TransferBytes && after.ResidentBytes >= before.ResidentBytes {
			cudaImmutableWeightUploads = &model.NativeCUDAImmutableWeightUploadDelta{
				Before: before,
				After:  after,
				Delta: model.NativeCUDAImmutableWeightUploadCounters{
					Calls:         after.Calls - before.Calls,
					TransferBytes: after.TransferBytes - before.TransferBytes,
					ResidentBytes: after.ResidentBytes - before.ResidentBytes,
				},
			}
		}
	}
	return &model.NativeInferenceReceipt{
		TokenIDs:                   append([]int(nil), measurement.tokenIDs...),
		TokenLogprobs:              append([]float64(nil), measurement.logprobs...),
		PrefillSeconds:             prefillS,
		TTFTSeconds:                measurement.ttftS,
		DecodeSeconds:              decodeS,
		Model:                      p.modelID,
		Engine:                     "inkernel",
		Planner:                    "inkernel",
		Owner:                      "fak",
		Backend:                    backend,
		ForwardPath:                forwardPath,
		Q4K:                        p.q4k,
		FallbackActive:             measurement.hostFallbackObserved,
		PrefillChunkTokens:         p.nativeInferencePrefillChunkTokens(),
		NativeSelection:            nativeSelection,
		NativeSelectionDigest:      nativeSelectionDigest,
		Qwen35MetalForwardSequence: qwen35MetalForwardSequence,
		Qwen35MetalStateIdentity:   qwen35MetalStateIdentity,
		Qwen35SequencePrefillRoute: cloneNativeSequencePrefillRouteReceipt(measurement.qwen35SequencePrefillRoute),
		CUDAImmutableWeightUploads: cudaImmutableWeightUploads,
		NativeCacheAccounting:      firstNativeCacheAccounting(accounting),
	}
}

// firstNativeCacheAccounting returns the first non-nil accounting block from the
// optional variadic argument, or nil. It lets existing 3-argument call sites
// (tests and internal builders that predate #13339) compile unchanged while the
// served path attaches the accounting block.
func firstNativeCacheAccounting(accounting []*model.NativeCacheAccounting) *model.NativeCacheAccounting {
	for _, a := range accounting {
		if a != nil {
			return a
		}
	}
	return nil
}

// nativeCacheAccountingFacts carries the served-request facts the cache
// accounting block is derived from. They are the SAME values the served path
// already computes for the witness line and the provider-cache projection
// (cacheable from #3390's lookup-side match, matched from the realized serve),
// so the accounting block is a re-projection of observed numbers, never a
// second, independently-guessed source of truth.
type nativeCacheAccountingFacts struct {
	promptTokens    int
	cacheableTokens int
	matchedTokens   int
	sourceTier      radixkv.SnapshotTier
	prefillSecs     float64
	generated       int
}

// nativeCacheAccountingFor re-projects the served request's already-observed
// facts into the optional model.NativeCacheAccounting block (#13339). It
// separates the request's own restore from the startup preparation that made
// that restore possible: the block reports restored vs computed tokens and the
// restore duration of THIS request, and startup priming (a populate that
// generates nothing and never enters demand) is accounted by the warming path
// that performs it, not folded into a served turn.
//
// Conservation is structural: computed is prompt - matched and restored is the
// matched serve, so restored + computed == prompt by construction and the
// block passes model.NativeCacheAccounting.Validate() for every branch. A cold
// turn (no reuse) is the always-usable fallback and needs no cache state.
//
// The block is attached only when a native receipt was requested, so an
// ordinary receipt keeps its existing shape. It is NOT attached when nothing
// was prompt-presented or generated: an absent block reads as "no cache state
// to account", which is more honest than a fabricated zero-token record.
func (p *InKernelPlanner) nativeCacheAccountingFor(f nativeCacheAccountingFacts) *model.NativeCacheAccounting {
	prompt := f.promptTokens
	if prompt < 0 {
		prompt = 0
	}
	matched := f.matchedTokens
	if matched < 0 {
		matched = 0
	}
	if matched > prompt {
		matched = prompt
	}
	cacheable := f.cacheableTokens
	if cacheable < 0 {
		cacheable = 0
	}
	if cacheable > prompt {
		cacheable = prompt
	}
	if cacheable < matched {
		cacheable = matched
	}
	if prompt == 0 && f.generated == 0 {
		// Nothing was prompt-presented or generated: leave the block absent
		// rather than emit a zero-token record that implies a decision.
		return nil
	}
	computed := prompt - matched
	src := nativeCacheRestoreSourceFor(f.sourceTier, matched)

	acct := &model.NativeCacheAccounting{
		TotalPromptTokens: prompt,
		StableTokens:      cacheable,
		CachedTokens:      cacheable,
		RestoredTokens:    matched,
		ComputedTokens:    computed,
		RestoreSource:     src,
		GeneratedTokens:   f.generated,
	}
	if matched > 0 {
		acct.RestoreDurationSecs = f.prefillSecs
	}
	switch {
	case matched == 0:
		acct.Operation = model.NativeCacheCold
		acct.RestoreSource = model.NativeCacheRestoreNone
		acct.RestoreDurationSecs = 0
	case matched == prompt:
		acct.Operation = model.NativeCacheExactHit
	default:
		acct.Operation = model.NativeCachePartialHit
	}
	return acct
}

// nativeCacheRestoreSourceFor maps a radix snapshot tier onto the closed
// restore-source vocabulary. A miss (or any non-positive match) is "none";
// device-resident state is "device"; every host-side and remote transfer plane
// is "host" (it crossed the host boundary, not the device-resident L1 cache).
func nativeCacheRestoreSourceFor(tier radixkv.SnapshotTier, matched int) model.NativeCacheRestoreSource {
	if matched <= 0 {
		return model.NativeCacheRestoreNone
	}
	switch tier {
	case radixkv.SnapshotTierDeviceL1:
		return model.NativeCacheRestoreDevice
	case radixkv.SnapshotTierHostL2, radixkv.SnapshotTierRemoteL3:
		return model.NativeCacheRestoreHost
	default:
		// A live match under an unknown tier cannot be claimed as a device hit;
		// leave it source-less so the block reads as an un-attributed restore
		// rather than an optimistic device claim.
		return model.NativeCacheRestoreNone
	}
}

func (p *InKernelPlanner) nativeSelectionIdentity() (model.NativeSelectionIdentity, string, error) {
	backend, forwardPath := p.executionIdentity()
	quantization := p.nativeSelectionQuantization()
	identity := model.NativeSelectionIdentity{
		Schema:              model.NativeSelectionIdentitySchemaV1,
		ModelRef:            p.modelID,
		Backend:             backend,
		ForwardPath:         forwardPath,
		Quantization:        quantization,
		PrefillChunkTokens:  p.nativeInferencePrefillChunkTokens(),
		CPUOffloadExperts:   p.nativeSelectionCPUOffloadExperts(),
		Q4KGateUpOutputSlab: p.q4kGateUpOutputSlab && quantization == model.NativeSelectionQuantizationQ4K,
	}
	digest, err := identity.Digest()
	if err != nil {
		return model.NativeSelectionIdentity{}, "", err
	}
	return identity, digest, nil
}

func (p *InKernelPlanner) nativeSelectionQuantization() string {
	if p != nil && p.q4k {
		// Resident Q2_0 and Prism PQ2_0 reuse Session.Q4K's mixed-quant
		// execution path. The attached Prism contract distinguishes the two
		// on-disk formats after both reach the same packed Q2 weight store.
		if p.m != nil && p.m.Q2Count() > 0 {
			if p.m.HasPrismHadamard() {
				return model.NativeSelectionQuantizationPQ2_0
			}
			return model.NativeSelectionQuantizationQ2_0
		}
		return model.NativeSelectionQuantizationQ4K
	}
	if p != nil && p.quant {
		return model.NativeSelectionQuantizationQ8_0
	}
	return model.NativeSelectionQuantizationF32
}

func (p *InKernelPlanner) nativeSelectionCPUOffloadExperts() int {
	if p == nil {
		return 0
	}
	if p.expertSpill != nil {
		return p.expertSpill.Fit.SpillLayers
	}
	if p.cpuOffloadExperts && p.m != nil {
		return len(p.m.MoEExpertLayers())
	}
	return 0
}

func cloneQwen35MetalStateIdentityReceipt(src model.Qwen35MetalStateIdentityReceipt) *model.Qwen35MetalStateIdentityReceipt {
	src.States = append([]model.Qwen35MetalStateDigest(nil), src.States...)
	return &src
}

type cudaImmutableWeightUploadSnapshotter interface {
	CUDAImmutableWeightUploadSnapshot() (calls, transferBytes, residentBytes uint64)
}

func cudaImmutableWeightUploadSnapshot(be compute.Backend) (model.NativeCUDAImmutableWeightUploadCounters, bool) {
	provider, ok := be.(cudaImmutableWeightUploadSnapshotter)
	if !ok {
		return model.NativeCUDAImmutableWeightUploadCounters{}, false
	}
	calls, transferBytes, residentBytes := provider.CUDAImmutableWeightUploadSnapshot()
	return model.NativeCUDAImmutableWeightUploadCounters{Calls: calls, TransferBytes: transferBytes, ResidentBytes: residentBytes}, true
}

func (p *InKernelPlanner) requiresDeviceSerialization() bool {
	return p != nil && (p.backend != nil || p.metal)
}

// EnableConcurrencyProfile turns on the opt-in device-fan-out phase-timeline
// recorder (issue #1589). It must be called before concurrent Complete calls.
// Enabling it changes no served tokens and no batching: it only records the
// admit / forward-enter / forward-exit timestamps at the existing devMu seam.
func (p *InKernelPlanner) EnableConcurrencyProfile() {
	if p == nil {
		return
	}
	if p.concurrencyProfile == nil {
		p.concurrencyProfile = newConcurrencyProfiler()
	}
}

// ConcurrencyProfile folds the recorded phase timeline into a named-mechanism
// receipt. It returns nil when no profile was enabled. Callers must invoke it
// after all profiled requests have returned.
func (p *InKernelPlanner) ConcurrencyProfile() *ConcurrencyProfile {
	if p == nil {
		return nil
	}
	_, forwardPath := p.executionIdentity()
	return p.concurrencyProfile.build(forwardPath)
}

const inKernelRequestDeviceHeadroom = 0.15
const inKernelRequestPressureTrimMarginRatio = 0.10
const inKernelRequestPressureTrimMinMarginBytes = 64 << 20

// nativeDecodeRate derives a sustained decode throughput from a generate call's
// raw counters, or reports it unavailable. The first generated token is emitted
// by the prefill block, so only the gen-1 completed post-prefill intervals are
// decode work; a rate needs at least one such interval AND positive elapsed
// decode time. When either is absent the rate is unavailable (ok=false) and the
// caller must not serialize a finite success value — the defect #13298 where a
// single warmup token with a near-zero window reported ~6957 tok/s. Raw counters
// (gen, decodeS) are preserved untouched; only the derived rate is gated.
func nativeDecodeRate(gen int, decodeS float64) (float64, bool) {
	if gen < 2 || decodeS <= 0 {
		return 0, false
	}
	return float64(gen-1) / decodeS, true
}

func (p *InKernelPlanner) logExecutionSummary(q8kern, q8fusedMark string, promptTok, cacheable, matched, computed int, prefillS, prefTPS float64, generated int, decodeS, decTPS float64) {
	backend, forwardPath := p.executionIdentity()
	log.Printf("inkernel_chat model=%s backend=%s forward_path=%s q4k=%v q8dec=%s%s/%dw prompt=%dtok cacheable=%dtok reused=%dtok prefill=%dtok/%.2fs/%.1ftok/s decode=%dtok/%.2fs/%.1ftok/s%s gemm=%s",
		p.modelID, backend, forwardPath, p.q4k, q8kern, q8fusedMark, model.Q8DecodeWorkers(), promptTok, cacheable, matched, computed, prefillS, prefTPS, generated, decodeS, decTPS, p.v41FaultAttributionClause(), p.prefillQ4KGEMMLabel())
}

// prefillQ4KGEMMLabel is the gemm= value of the execution summary: the Q4_K prefill GEMM
// route(s) the request's prefill observed (scalar, a newer small-P/MM32/M5 identity, cpu,
// gemv, token_loop, or "+"-joined when mixed), or model.Q4KPrefillGEMMNone when no prefill
// ran or nothing was observed. Appended last so every existing key keeps its position.
func (p *InKernelPlanner) prefillQ4KGEMMLabel() string {
	if p != nil {
		if label, _ := p.prefillQ4KGEMM.Load().(string); label != "" {
			return label
		}
	}
	return model.Q4KPrefillGEMMNone
}

// v41ExpertFaultAttributioner is the optional read seam a V4.1 model exposes
// so the execution summary can attribute the routed-expert fault cost per
// forward phase (#13294 DoD item 1). Every model WITHOUT the method — the
// whole non-V4.1 fleet — fails this assertion and skips the clause, so their
// summary lines stay byte-for-byte unchanged.
type v41ExpertFaultAttributioner interface {
	V41ExpertFaultAttribution() model.V41ExpertFaultAttribution
}

// v41FaultAttributionClause renders the appended clause of the execution
// summary: the phase-split routed-expert fault attribution for the model the
// turn ran on. Empty when the model does not expose it OR when both phase
// ledgers are untouched — a non-V4.1 turn reports nothing rather than a zero
// clause, keeping the existing line the physical receipts parse. The two
// phase ledgers are read through their exported FIELDS (field access through
// selectors never names the unexported ledger type), so the formatting stays
// entirely in the agent layer.
func (p *InKernelPlanner) v41FaultAttributionClause() string {
	if p == nil || p.m == nil {
		return ""
	}
	at := v41ExpertFaultAttributioner(p.m)
	if at == nil {
		return ""
	}
	return formatV41FaultClause(at.V41ExpertFaultAttribution())
}

// formatV41FaultClause is the pure renderer behind v41FaultAttributionClause,
// split out so a unit test can assert the exact clause without a live model.
// The byte clause is unchanged from the pre-elapsed form (every physical
// receipt already parses it); the #13299 elapsed split is APPENDED so an
// operator reading one physical serve log can rank disk wait against f32
// dequantization against the routed contraction — the choice PLAN
// halo-ds41-100-30 §101 makes between the residency (#12952), device-contraction
// (#13128) and dequant levers. Seconds and per-token milliseconds come straight
// from the ledger's nanosecond accumulators; the contraction backend is the
// observed engine identity, never a device receipt.
func formatV41FaultClause(fa model.V41ExpertFaultAttribution) string {
	if fa == (model.V41ExpertFaultAttribution{}) {
		return ""
	}
	const mib = 1.0 / (1 << 20)
	pre, dec := fa.Prefill, fa.Decode
	hits, reads := pre.ResidentHits+dec.ResidentHits, pre.ResidentHits+dec.ResidentHits+pre.Faults+dec.Faults
	hitFraction := 0.0
	if reads > 0 {
		hitFraction = float64(hits) / float64(reads)
	}
	// The backend identity is only stamped once a contraction ran; when neither
	// phase contracted, say so with a dash rather than an ambiguous empty value.
	backend := pre.ContractionBackend
	if backend == "" {
		backend = "-"
	}
	compressorClause := ""
	if v41CompressorPhaseSnapshot(pre) != (V41CompressorPhaseSnapshot{}) || v41CompressorPhaseSnapshot(dec) != (V41CompressorPhaseSnapshot{}) {
		compressorClause = fmt.Sprintf(
			" compressor_projection prefill=[device=%dcalls/%drows host=%dcalls/%drows upload=%dB readback=%dB elapsed=%.3fs] decode=[device=%dcalls/%drows host=%dcalls/%drows upload=%dB readback=%dB elapsed=%.3fs]",
			pre.CompressorProjectionDeviceCalls, pre.CompressorProjectionDeviceRows, pre.CompressorProjectionHostCalls, pre.CompressorProjectionHostRows, pre.CompressorProjectionActivationUploadBytes, pre.CompressorProjectionReadbackBytes, float64(pre.CompressorProjectionNanos)/1e9,
			dec.CompressorProjectionDeviceCalls, dec.CompressorProjectionDeviceRows, dec.CompressorProjectionHostCalls, dec.CompressorProjectionHostRows, dec.CompressorProjectionActivationUploadBytes, dec.CompressorProjectionReadbackBytes, float64(dec.CompressorProjectionNanos)/1e9)
	}
	return fmt.Sprintf(
		" v41_faults prefill=[faults=%df/%.1ffpt faulted_bytes=%.1fMiB dequant=%.2fGiB] decode=[faults=%df/%.1ffpt faulted_bytes=%.1fMiB dequant=%.2fGiB] hit_fraction=%.2f elapsed prefill=[fault=%.3fs/%.3fmspt dequant=%.3fs/%.3fmspt contraction=%.3fs/%.3fmspt] decode=[fault=%.3fs/%.3fmspt dequant=%.3fs/%.3fmspt contraction=%.3fs/%.3fmspt] contraction_backend=%s incremental_device prefill=[gate_up=%d down=%d dispatch=%.3fs] decode=[gate_up=%d down=%d dispatch=%.3fs] incremental_engram prefill=[injections=%d rows=%d hash_tokens=%d elapsed=%.3fs] decode=[injections=%d rows=%d hash_tokens=%d elapsed=%.3fs] expert_activation prefill=[device=%d host=%d readback=%dB elapsed=%.3fs] decode=[device=%d host=%d readback=%dB elapsed=%.3fs] dense_projection prefill=[device=%dcalls/%drows host=%dcalls/%drows upload=%dB readback=%dB elapsed=%.3fs] decode=[device=%dcalls/%drows host=%dcalls/%drows upload=%dB readback=%dB elapsed=%.3fs]%s head_projection prefill=[device=%dcalls/%drows host=%dcalls/%drows upload=%dB readback=%dB elapsed=%.3fs] decode=[device=%dcalls/%drows host=%dcalls/%drows upload=%dB readback=%dB elapsed=%.3fs] grouped_output prefill=[device=%dcalls/%drows host=%dcalls/%drows matmul=%d upload=%dB readback=%dB host_f32=%dB elapsed=%.3fs] decode=[device=%dcalls/%drows host=%dcalls/%drows matmul=%d upload=%dB readback=%dB host_f32=%dB elapsed=%.3fs] engram_projection prefill=[device=%dcalls/%drows host=%dcalls/%drows matmul=%d upload=%dB readback=%dB host_f32=%dB elapsed=%.3fs] decode=[device=%dcalls/%drows host=%dcalls/%drows matmul=%d upload=%dB readback=%dB host_f32=%dB elapsed=%.3fs] mhc_projection prefill=[device=%dcalls/%drows host=%dcalls/%drows matmul=%d upload=%dB readback=%dB host_f32=%dB elapsed=%.3fs] decode=[device=%dcalls/%drows host=%dcalls/%drows matmul=%d upload=%dB readback=%dB host_f32=%dB elapsed=%.3fs]",
		pre.Faults, pre.FaultsPerToken, mib*float64(pre.FaultedBytes), float64(pre.DequantBytes)/(1<<30),
		dec.Faults, dec.FaultsPerToken, mib*float64(dec.FaultedBytes), float64(dec.DequantBytes)/(1<<30),
		hitFraction,
		float64(pre.FaultDoorNanos)/1e9, pre.FaultNanosPerToken/1e6,
		float64(pre.DequantNanos)/1e9, pre.DequantNanosPerToken/1e6,
		float64(pre.ContractionNanos)/1e9, pre.ContractionNanosPerToken/1e6,
		float64(dec.FaultDoorNanos)/1e9, dec.FaultNanosPerToken/1e6,
		float64(dec.DequantNanos)/1e9, dec.DequantNanosPerToken/1e6,
		float64(dec.ContractionNanos)/1e9, dec.ContractionNanosPerToken/1e6,
		backend,
		pre.IncrementalDeviceGateUpCalls, pre.IncrementalDeviceDownCalls, float64(pre.IncrementalDeviceDispatchNanos)/1e9,
		dec.IncrementalDeviceGateUpCalls, dec.IncrementalDeviceDownCalls, float64(dec.IncrementalDeviceDispatchNanos)/1e9,
		pre.IncrementalEngramInjections, pre.IncrementalEngramRows, pre.IncrementalEngramHashTokens, float64(pre.IncrementalEngramNanos)/1e9,
		dec.IncrementalEngramInjections, dec.IncrementalEngramRows, dec.IncrementalEngramHashTokens, float64(dec.IncrementalEngramNanos)/1e9,
		pre.ExpertActivationDeviceCalls, pre.ExpertActivationHostCalls, pre.ExpertActivationReadbackBytes, float64(pre.ExpertActivationNanos)/1e9,
		dec.ExpertActivationDeviceCalls, dec.ExpertActivationHostCalls, dec.ExpertActivationReadbackBytes, float64(dec.ExpertActivationNanos)/1e9,
		pre.DenseProjectionDeviceCalls, pre.DenseProjectionDeviceRows, pre.DenseProjectionHostCalls, pre.DenseProjectionHostRows, pre.DenseProjectionActivationUploadBytes, pre.DenseProjectionReadbackBytes, float64(pre.DenseProjectionNanos)/1e9,
		dec.DenseProjectionDeviceCalls, dec.DenseProjectionDeviceRows, dec.DenseProjectionHostCalls, dec.DenseProjectionHostRows, dec.DenseProjectionActivationUploadBytes, dec.DenseProjectionReadbackBytes, float64(dec.DenseProjectionNanos)/1e9,
		compressorClause,
		pre.HeadProjectionDeviceCalls, pre.HeadProjectionDeviceRows, pre.HeadProjectionHostCalls, pre.HeadProjectionHostRows, pre.HeadProjectionActivationUploadBytes, pre.HeadProjectionReadbackBytes, float64(pre.HeadProjectionNanos)/1e9,
		dec.HeadProjectionDeviceCalls, dec.HeadProjectionDeviceRows, dec.HeadProjectionHostCalls, dec.HeadProjectionHostRows, dec.HeadProjectionActivationUploadBytes, dec.HeadProjectionReadbackBytes, float64(dec.HeadProjectionNanos)/1e9,
		pre.GroupedOutputDeviceCalls, pre.GroupedOutputDeviceRows, pre.GroupedOutputHostCalls, pre.GroupedOutputHostRows, pre.GroupedOutputMatMulCalls, pre.GroupedOutputActivationUploadBytes, pre.GroupedOutputReadbackBytes, pre.GroupedOutputHostWeightF32Bytes, float64(pre.GroupedOutputNanos)/1e9,
		dec.GroupedOutputDeviceCalls, dec.GroupedOutputDeviceRows, dec.GroupedOutputHostCalls, dec.GroupedOutputHostRows, dec.GroupedOutputMatMulCalls, dec.GroupedOutputActivationUploadBytes, dec.GroupedOutputReadbackBytes, dec.GroupedOutputHostWeightF32Bytes, float64(dec.GroupedOutputNanos)/1e9,
		pre.EngramProjectionDeviceCalls, pre.EngramProjectionDeviceRows, pre.EngramProjectionHostCalls, pre.EngramProjectionHostRows, pre.EngramProjectionMatMulCalls, pre.EngramProjectionActivationUploadBytes, pre.EngramProjectionReadbackBytes, pre.EngramProjectionHostWeightF32Bytes, float64(pre.EngramProjectionNanos)/1e9,
		dec.EngramProjectionDeviceCalls, dec.EngramProjectionDeviceRows, dec.EngramProjectionHostCalls, dec.EngramProjectionHostRows, dec.EngramProjectionMatMulCalls, dec.EngramProjectionActivationUploadBytes, dec.EngramProjectionReadbackBytes, dec.EngramProjectionHostWeightF32Bytes, float64(dec.EngramProjectionNanos)/1e9,
		pre.MHCProjectionDeviceCalls, pre.MHCProjectionDeviceRows, pre.MHCProjectionHostCalls, pre.MHCProjectionHostRows, pre.MHCProjectionMatMulCalls, pre.MHCProjectionActivationUploadBytes, pre.MHCProjectionReadbackBytes, pre.MHCProjectionHostWeightF32Bytes, float64(pre.MHCProjectionNanos)/1e9,
		dec.MHCProjectionDeviceCalls, dec.MHCProjectionDeviceRows, dec.MHCProjectionHostCalls, dec.MHCProjectionHostRows, dec.MHCProjectionMatMulCalls, dec.MHCProjectionActivationUploadBytes, dec.MHCProjectionReadbackBytes, dec.MHCProjectionHostWeightF32Bytes, float64(dec.MHCProjectionNanos)/1e9)
}

// executionIdentity makes the request log say which compute path actually produced
// the token. q8dec remains useful CPU implementation metadata, but it must not be
// mistaken for a fallback when a device-backed Qwen3.6 session is selected.
// ExecutionIdentity reports the compute backend and forward path this planner
// serves on, the same pair the inkernel_chat summary line logs.
func (p *InKernelPlanner) ExecutionIdentity() (backend, forwardPath string) {
	return p.executionIdentity()
}

func (p *InKernelPlanner) executionIdentity() (backend, forwardPath string) {
	backend, forwardPath = "cpu-ref", "cpu/reference"
	if p == nil {
		return backend, forwardPath
	}
	if p.backend != nil {
		backend, forwardPath = p.backend.Name(), "device/generic"
	} else if p.metal {
		// Metal is deliberately a CPU-session seam rather than a compute.Backend, but it still
		// owns the projection/MLP dispatch selected for this request. Report that resolved
		// accelerator instead of making a real Metal turn look like a CPU fallback (#8295).
		backend, forwardPath = "metal", "metal/session-forward"
	}
	if p.m != nil && p.m.Cfg.IsQwen35Hybrid() {
		if p.backend != nil {
			// Model.NewBackendSession has already validated the structural GDN
			// contract before this request can complete. Preserve the selected
			// backend's path instead of labeling every device as CUDA.
			if gdn, ok := p.backend.(interface{ Qwen35GDNPath() string }); ok {
				if path := gdn.Qwen35GDNPath(); model.IsSupportedQwen35GDNPath(path) {
					forwardPath = path
				}
			}
		} else if p.metal && p.qwen35MetalGDNExecuted.Load() {
			forwardPath = model.Qwen35MetalGDNSequenceForwardPath
		} else if p.metal {
			// Qwen3.5-family Metal uses the native Session forward with Metal projection/MLP
			// dispatch; q4k= in the same summary records the selected weight format.
			forwardPath = "metal/qwen35-hybrid-session-v1"
		} else {
			forwardPath = "cpu/qwen35-gdn-reference"
		}
	}
	return backend, forwardPath
}

// generateReused runs prefill + decode for an already-encoded prompt, REUSING the longest
// cached KV prefix (the radix tree) when enabled and FAILING OPEN to a full prefill on a
// miss — the candidate-#13 core, factored out of Complete so the reuse/decode path is
// exercisable on a synthetic model with no tokenizer.
//
// emit is invoked with each generated token id AFTER sampling and BEFORE the next Step;
// returning true stops decode with that token counted (Complete's string-suffix stop
// closes over the tokenizer there). A token-id stop (stops[next]) or next<0 ends decode
// WITHOUT emitting — the served contract that a stop token is not echoed.
//
// SNAPSHOT/LEASE discipline: the full-prompt KV is snapshotted (Cloned) right after
// Prefill — BEFORE the decode loop mutates s.Cache by appending generated positions — and
// inserted under a FRESH Lookup so radixkv's lease handoff (Lookup→Insert→Done) is honored
// entirely inside the lock, with no unexported *node escaping this scope. The reuse clone
// (SessionFromPrefix) is also taken under the lock, so a concurrent eviction of the tree
// node can never race our read of its KV. Returns the generated-token count, the prompt
// length, the reused-prefix length, prefill/decode seconds, and whether a stop (not maxNew)
// ended the turn.

// InKernelWarmState is the planner-owned startup KV-cache warm lifecycle (CW-27 of the
// agent-startup cache-warm program). It exists because a helper in another Go file cannot
// add struct fields to InKernelPlanner, and a package-global planner-pointer map would leak:
// the bounded, synchronized warm lifecycle must live on the planner itself.
//
// It deliberately makes NO cache and runs NO warming. It owns exactly four things:
//
//  1. A BOUNDED SINGLE-PROFILE coalescing slot: at most one startup warm profile is active
//     at a time; a second concurrent Begin for the SAME profile joins the in-flight work
//     (the coalescing win) while a Begin for a DIFFERENT profile is refused rather than
//     silently queued or duplicated.
//  2. Explicit synchronization over that slot (one mutex; no lock-free "just a bool").
//  3. A cancellation context derived from the caller's, cancelled by Release, so a
//     shutdown or an admitted demand can stop optional startup work.
//  4. An abstract RELEASE callback — the residency claim handover owned by the later warm
//     API (fak#13341). This leaf never dereferences a radix claim; it only guarantees the
//     callback fires at most once, idempotently, on Complete or Release.
//
// The zero value is a usable idle state, so every constructor — including a bare
// &InKernelPlanner{} in a test — can Begin a startup warm with no initialization step to
// forget (the same idiom as inKernelTurnTaxState and moeResidencyState).
type InKernelWarmState struct {
	mu sync.Mutex
	// phase is the closed lifecycle vocabulary. It starts at warmStateIdle (the zero
	// value) and only Release moves it to warmStateReleased, which is terminal: a released
	// planner never starts another startup warm until an explicit Reset.
	phase warmPhase
	// profileID names the single active/coalesced warm profile. Empty while idle.
	profileID string
	// slotComplete records that the active profile finished (Complete) and the release
	// callback has been invoked; the lock is still held by the lifecycle until Release/Reset.
	slotComplete bool
	// ctx is the cancellation context handed to every Begin caller joined onto this slot.
	ctx context.Context
	// cancel cancels ctx on Release. Nil while idle.
	cancel context.CancelFunc
	// release is the abstract residency-release callback (fak#13341), invoked at most once.
	release func()
	// coalesced counts Begin callers that joined an already-active profile (the winning
	// half of the bounded single-profile slot); refused counts Begin callers refused because
	// a DIFFERENT profile was active. Both are readbacks, never gates.
	coalesced int
	refused   int
}

type warmPhase uint8

const (
	// warmStateIdle is the zero value: no startup warm active, none released.
	warmStateIdle warmPhase = iota
	// warmStateActive: one profile owns the slot until Complete or Release.
	warmStateActive
	// warmStateReleased: the planner's startup warm owner shut down; Begin is refused and
	// the release callback has fired. Terminal until Reset.
	warmStateReleased
)

// ErrInKernelWarmReleased is returned by Begin after the planner's startup warm owner has
// released: a shut-down planner must not start new optional startup work.
var ErrInKernelWarmReleased = errors.New("agent: in-kernel startup warm already released")

// ErrInKernelWarmProfileBusy is returned by Begin when a DIFFERENT profile already owns the
// bounded single-profile slot. The caller may retry once the active profile completes or the
// planner releases; this is a refusal, never a silent queue (the "bounded" in the contract).
var ErrInKernelWarmProfileBusy = errors.New("agent: in-kernel startup warm slot held by another profile")

// Begin opens (or joins) the single startup warm slot for profileID, deriving a cancellable
// context from parent. It returns the context the caller must use for the optional startup
// work, and coalesced=true when this call JOINED an already-active profile with the SAME id
// (the winner of the coalescing contract) rather than starting new work.
//
// A different active profile is refused with ErrInKernelWarmProfileBusy; a released planner
// is refused with ErrInKernelWarmReleased. A nil parent uses context.Background. The returned
// context is cancelled by Release.
func (s *InKernelWarmState) Begin(parent context.Context, profileID string) (ctx context.Context, coalesced bool, err error) {
	if parent == nil {
		parent = context.Background()
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	switch s.phase {
	case warmStateReleased:
		return nil, false, ErrInKernelWarmReleased
	case warmStateActive:
		if !s.slotComplete && s.profileID == profileID {
			s.coalesced++
			if s.ctx == nil {
				s.ctx, s.cancel = context.WithCancel(parent)
			}
			return s.ctx, true, nil
		}
		if s.profileID != profileID || s.slotComplete {
			s.refused++
			return nil, false, ErrInKernelWarmProfileBusy
		}
	}
	// Idle, or the same profile after completion: (re)open the slot.
	s.phase = warmStateActive
	s.profileID = profileID
	s.slotComplete = false
	s.ctx, s.cancel = context.WithCancel(parent)
	return s.ctx, false, nil
}

// BindRelease attaches the abstract residency-release callback for the active slot. It is
// invoked at most once, by Complete or Release, whichever happens first; nil clears it. It
// is the seam the later warm API (fak#13341) uses to bind the radix claim without this leaf
// ever dereferencing one.
func (s *InKernelWarmState) BindRelease(release func()) {
	s.mu.Lock()
	s.release = release
	s.mu.Unlock()
}

// Complete marks the active profile finished and fires the bound release callback exactly
// once (idempotently). It is safe to call more than once and safe when no warm is active: a
// completed slot refuses new Begin calls for the same profile id until Reset, so a stale
// completion cannot resurrect finished startup work.
func (s *InKernelWarmState) Complete() {
	s.mu.Lock()
	if s.phase != warmStateActive || s.slotComplete {
		s.mu.Unlock()
		return
	}
	s.slotComplete = true
	release := s.release
	s.release = nil
	s.mu.Unlock()
	if release != nil {
		release()
	}
}

// Release is the startup-context shutdown owner: it cancels any in-flight warm, fires the
// bound release callback at most once, and moves the state to terminal released so no further
// startup warm can begin. It is idempotent — the second and later calls are no-ops, and the
// release callback never runs twice.
func (s *InKernelWarmState) Release() {
	s.mu.Lock()
	if s.phase == warmStateReleased {
		s.mu.Unlock()
		return
	}
	cancel := s.cancel
	release := s.release
	s.release = nil
	s.cancel = nil
	s.phase = warmStateReleased
	s.slotComplete = false
	s.profileID = ""
	s.mu.Unlock()
	if cancel != nil {
		cancel()
	}
	if release != nil {
		release()
	}
}

// Reset returns a released or completed state to a usable idle slot for a new startup warm
// (the explicit reset the contract requires). It is idempotent and never fires the release
// callback: a caller that wants release semantics must call Release, not Reset.
func (s *InKernelWarmState) Reset() {
	s.mu.Lock()
	s.phase = warmStateIdle
	s.profileID = ""
	s.slotComplete = false
	s.ctx = nil
	s.cancel = nil
	s.release = nil
	s.coalesced = 0
	s.refused = 0
	s.mu.Unlock()
}

// Active reports whether a startup warm profile currently owns the slot. It is a readback
// seam for startup callers and receipts; it never mutates the state.
func (s *InKernelWarmState) Active() bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.phase == warmStateActive && !s.slotComplete
}

// ProfileID returns the profile currently owning the active slot, or "" when idle/released.
func (s *InKernelWarmState) ProfileID() string {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.phase != warmStateActive || s.slotComplete {
		return ""
	}
	return s.profileID
}

// CoalescedCount returns how many Begin calls joined an already-active same-profile slot, and
// RefusedCount how many were refused because a different profile held the bounded slot. Both
// are cumulative for the slot's lifetime and reset by Reset.
func (s *InKernelWarmState) CoalescedCount() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.coalesced
}

// RefusedCount returns how many Begin calls were refused by the bounded single-profile slot.
func (s *InKernelWarmState) RefusedCount() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.refused
}

// StartupWarmState exposes the planner-owned startup warm lifecycle so startup callers can
// Begin a bounded, cancellable warm and the shutdown owner can Release it. The returned
// pointer aliases planner state; callers must not retain it past the planner's lifetime.
func (p *InKernelPlanner) StartupWarmState() *InKernelWarmState {
	if p == nil {
		return nil
	}
	return &p.warmState
}

// ReleaseStartupWarm is the planner's shutdown-owner hook: it delegates to the owned
// InKernelWarmState, cancelling any in-flight startup warm and firing the bound release
// callback at most once. A nil planner is a no-op. Startup callers that own the planner's
// lifetime call this in their shutdown path (there is no InKernelPlanner.Close yet).
func (p *InKernelPlanner) ReleaseStartupWarm() {
	if p == nil {
		return
	}
	p.warmState.Release()
}
