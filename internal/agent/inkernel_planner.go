package agent

// inkernel_planner.go ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â the in-kernel chat Planner. When fak serve boots with a
// preloaded GGUF model (modelengine.Preload / PreloadQ4K) and a tokenizer, and no
// upstream --base-url, this planner drives BOTH /v1/chat/completions and
// /v1/messages (they share s.planner.Complete) from the model fused into the
// kernel ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â real ChatML chat through internal/tokenizer, not the byte-tokenized
// dispatch demo in modelengine.Complete.
//
// The decode recipe is the proven cmd/fakchat hybrid path: render ChatML ÃƒÂ¢Ã¢â‚¬Â Ã¢â‚¬â„¢ Encode
// ÃƒÂ¢Ã¢â‚¬Â Ã¢â‚¬â„¢ Session.Prefill ÃƒÂ¢Ã¢â‚¬Â Ã¢â‚¬â„¢ argmax/temperature sample ÃƒÂ¢Ã¢â‚¬Â Ã¢â‚¬â„¢ Session.Step ÃƒÂ¢Ã¢â‚¬Â Ã¢â‚¬â„¢ Decode, stopping
// on <|im_end|>/<|endoftext|>. fakchat's end-to-end coherent chat (Qwen2.5-1.5B/7B,
// FAK-NATIVE-CHAT-RESULTS.md) is the witness that this recipe produces real text;
// this file factors it into a Planner so the gateway can serve it on both wires.

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	randv2 "math/rand/v2"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/anthony-chaudhary/fak/internal/cachemeta"
	"github.com/anthony-chaudhary/fak/internal/cacheobs"
	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/enginestep"
	"github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/radixkv"
	"github.com/anthony-chaudhary/fak/internal/tokenizer"
)

// InKernelPlanner is an agent.Planner backed by the in-kernel model. One Complete
// call renders the transcript as ChatML, runs a real Prefill + decode over the
// kernel-owned session cache, and returns the assistant's text. It does not itself
// emit structured tool calls ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â the gateway's adjudication layer still runs on
// whatever the caller proposed.
type InKernelPlanner struct {
	dflash2                *DFlash2Target
	m                      *model.Model
	tok                    *tokenizer.Tokenizer
	promptEncodeMemo       tokenizer.EncodeMemo
	modelID                string
	q4k                    bool            // resident-Q4_K load: decode runs Session.Q4K (SDOT int8 GEMV)
	quant                  bool            // Q8_0 decode/prefill path (the served default); tests flip it to exercise the proven f32 reuse path
	backend                compute.Backend // non-nil ÃƒÂ¢Ã¢â‚¬Â Ã¢â‚¬â„¢ decode runs through the device HAL (e.g. CUDA) instead of the CPU session
	metal                  bool            // Apple-Silicon metalgemm GPU forward on the CPU session (s.Metal); engaged ONLY when backend==nil (the CPU-session seam). No-op on non-Metal builds.
	cpuOffloadExperts      bool            // with a backend, keep MoE experts host-resident while dense/attention use the device
	requireDeviceExecution bool
	deviceOnlyBypass       string
	deviceOnlyBypassHolder atomic.Pointer[model.Session]
	// kvPrecision is the realized storage tier installed on every session this planner
	// builds (model.KVPrecisionFP32 default; Q8_0 for the dense mixed layout). Fixed at
	// construction from InKernelPlannerConfig.KVPrecision.
	kvPrecision model.KVPrecision
	// kvPrecisionFallback records why a requested 16-bit device tier fell back to f32
	// at construction ("" when the request was realized as asked).
	kvPrecisionFallback string
	// kvPrecisionErr is ValidateKVPrecision's construction-time verdict for the
	// resolved tier; non-nil refuses every request with a typed *KVPrecisionUnrealizedError.
	kvPrecisionErr error
	// expertSpill is the resolved graded expert placement (`--n-cpu-moe`, #5612) this planner
	// installs on every session it builds: how many MoE layers spill to host and how many device
	// bytes the routed-expert ring may hold. nil ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â the default and every planner that was never
	// given a grade ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â leaves placement exactly as cpuOffloadExperts alone decided it. Resolved once
	// by SetExpertSpill (inkernel_expert_spill.go), never per request.
	expertSpill *model.ExpertSpillPlacement
	// contextTokens is the runtime ceiling; zero delegates to MaxPositionEmbeddings.
	contextTokens int
	maxNew        int
	temp          float64
	seed          int64
	// seedExplicit records an operator-set FAK_INKERNEL_SEED. Seedless sampled
	// requests then take successive streams derived from seed, so a fresh planner
	// replays a run; otherwise each one draws fresh entropy.
	seedExplicit bool
	seedRequests atomic.Uint64
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

	// hostStages is the current turn's host-stage window, reported as host= on
	// a device-backed inkernel_chat summary line.
	hostStages inKernelHostStageWindow

	// tree is the process-scoped RadixAttention prefix cache (internal/radixkv): the
	// multi-thousand-token static system+tool-schema prefix is prefilled once and the
	// next turn REUSES its KV, prefilling only the divergent suffix ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â the candidate-#13
	// win, bit-identical to a full recompute (proven in internal/model's KV-prefix-reuse
	// rung). nil disables reuse (every turn full-prefills, the pre-#13 behavior); the
	// device-HAL path (backend != nil) never reuses (the reuse clone is a CPU session).
	// mu guards every tree access ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â the gateway can drive Complete concurrently, and the
	// tree is shared mutable state (radixkv itself is deliberately lock-free).
	//
	// MEMORY NOTE: radixkv stores the FULL-prefix KV per node, so a long single growing
	// conversation accumulates nested KV clones (see radixkv's Tokens-vs-PrefixTokens
	// note). InKernelPlannerConfig.RadixBudgetTokens sets the edge-token budget
	// (negative = unbounded, the historical default ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â the maximal-reuse regime the
	// witnesses measure; zero derives one context window), and
	// FAK_NATIVE_KV_VICTIM_RULE=cost-aware selects the KVBM victim rule for budget pressure.
	// Operators serving long sessions should set a budget; bounding the deep-chain
	// footprint is tracked.
	mu         sync.Mutex
	tree       *radixkv.Tree
	scopedTree *radixkv.ScopedTree
	// radixHostL2 is the host-DRAM L2 KV reserve resolved for tree at construction.
	radixHostL2 InKernelRadixHostL2
	// radixLocalSSD is the local-SSD KV overflow budget resolved for tree at construction;
	// zero when the tier is off (CPU-only planner, unknown disk capacity, or operator 0).
	radixLocalSSD InKernelRadixLocalSSD
	// pagedPrefix is the device planner's paged prefix page ledger
	// (inkernel_paged_prefix.go); nil on a CPU planner or with FAK_INKERNEL_PAGED_PREFIX=off.
	pagedPrefix *inKernelPagedPrefixLedger
	// prefixFlights bridges the cold lookup-to-admission gap for concurrent
	// plain-CPU requests. Its zero value is ready for constructor and bare-test
	// planners alike; persistent ownership remains in tree/scopedTree.
	prefixFlights radixkv.PrefixFlightGroup

	// radixAttrMu guards the opt-in cold-prefill-tax attribution ledger
	// (inkernel_radix_attribution.go). It is its own mutex, never shared with mu:
	// the ledger is written at admission seams that already hold or release mu,
	// and folding it into mu would couple observation to the hot lock. Disarmed
	// (the default) every seam pays one bool read and nothing else.
	radixAttrMu   sync.Mutex
	radixDebug    bool
	radixDebugLog bool
	radixAttr     RadixAttribution
	// radixOffLockAdmit selects the off-lock payload-clone admission fast path on
	// the unscoped CPU seam (ticket prefill-radix-cold-prefill-tax-20261108): the
	// O(prefix-bytes) session-cache clone runs outside mu, validated by a
	// re-lookup before attach. DEFAULT ON; tests flip it to witness parity against
	// the historical fully-locked section.
	radixOffLockAdmit bool

	// devMu serializes the WHOLE device forward pass (Prefill + the decode loop) when a
	// backend or native Metal is wired. These accelerators have one shared command stream and
	// model-level buffers. The CUDA backend's Go-side cudaMu makes
	// each INDIVIDUAL op atomic ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â but NOT a whole multi-op forward. Two Complete calls driven
	// concurrently by the gateway would interleave their per-token op sequences on that shared
	// device state and stomp each other's activation/KV buffers, faulting the kernel with an
	// illegal memory access that then poisons the CUDA context for every later request until a
	// process restart (observed live on an L4: a 2-way concurrent burst took the GPU serve down
	// with thousands of sticky cuda_kernels.cu illegal-access errors). The plain CPU path is
	// already session-local per turn and guards only its shared tree with p.mu, so devMu leaves
	// it untouched.
	// This serializes concurrent device requests into safe queuing ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â correct for a single-stream
	// device ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â instead of crashing; batched multi-user device decode is the separate throughput
	// follow-up (internal/model/batch.go), not a correctness fix.
	devMu inKernelDeviceGate

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
	// planLedgerDiffState is the once-per-serve first-prefill planner-vs-allocation-ledger
	// diff (inkernel_plan_ledger_diff.go). Observability only.
	planLedgerDiffState

	// inKernelTurnTaxState is the per-turn cache-decision ledger (#1538, inkernel_turntax.go),
	// embedded for the same reason as moeResidencyState directly above: the decision is taken
	// inside a session this planner builds and closes PER REQUEST, so planner-scoped storage is
	// the only place it survives that teardown. Its zero value is a usable empty ledger, so
	// every constructor ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â including a bare &InKernelPlanner{ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â¦} ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â records from its first turn.
	inKernelTurnTaxState

	oomRetryMu sync.Mutex
	oomRetry   map[string]*inKernelOOMRetryClassStats

	pressureTrimMu sync.Mutex
	pressureTrim   map[requestPressureTrimKey]*requestPressureTrimStats

	deviceBudgetMu      sync.Mutex
	deviceReservedBytes int64

	// hostBudget is the armed host-session memory ceiling (#13267, inkernel_host_budget.go):
	// nil keeps the historical unbounded host path; SetHostMemoryBudget arms it.
	hostBudgetMu sync.Mutex
	hostBudget   *hostMemoryBudget
	// Outstanding session/beam charges survive ceiling replacement and disarming.
	hostReservedBytes int64

	// kvSpanEvict gates the model-side KV-quarantine eviction BRIDGE (internal/kvmmu)
	// on the live serve path (issue #579). When on, a tool-result QUARANTINE drives a
	// real model.KVCache.Evict of the result's K/V span over a fresh model.Session built
	// from the loaded model ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â the bit-exact re-RoPE + renumber the kvmmu witnesses prove,
	// now fired by a live request instead of only a synthetic-model unit test. DEFAULT OFF
	// (FAK_INKERNEL_KVMMU=on opts in); off it is an inert no-op, so the served path is
	// byte-for-byte the pre-bridge behavior. It is independent of and additive to the
	// radixkv prefix-cache eviction above ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â that drops a reusable PREFIX node; this evicts
	// the per-session SPAN and is the model-independent KV-MMU floor.
	kvSpanEvict bool

	// batchDecode routes a request's decode through the continuous-batch
	// BatchSession.StepBatchActive machinery (as a batch of one on the resident chat serve)
	// instead of the serial Session.Step loop. It is the OPT-IN wiring seam
	// (FAK_INKERNEL_BATCH=on) that lets a future cross-request coalescer co-batch concurrent
	// chat turns onto the shared weight stream. DEFAULT OFF: unset, the decode path is the
	// byte-identical pre-seam serial loop, and even ON at B=1 StepBatchActive is exactly
	// Seqs[0].Step, so the served tokens are unchanged ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â the batched glm_moe_dsa GEMM that
	// yields throughput is the separate, box-gated lever, not this flag.
	batchDecode bool
	// batchSerialPerLaneNS is the planner-level per-token decode-step floor in
	// nanoseconds: a ratcheting MIN over timed serial steps AND healthy batched
	// steps (both folded as wall/lanes), seeded by the first cohort episode's
	// calibration step and refreshed by serial probes. The cohort batched->serial
	// degradation policy (inkernel_batch_health.go) budgets batched steps'
	// per-token cost against it.
	batchSerialPerLaneNS atomic.Int64
	// batchSeedSamples counts the per-token samples folded into
	// batchSerialPerLaneNS while the calibration window (batchStepSeedSamples)
	// is open: the window folds as MAX so one warmup micro-forward or one
	// backlog-drained step cannot alone define the floor.
	batchSeedSamples atomic.Int32
	// batchSeedCalibrating records that the calibration window opened on the
	// first fold into an unknown floor; a preset floor never opens it.
	batchSeedCalibrating atomic.Bool
	// v41EngramLookAhead is an explicit serial-only construction setting.
	v41EngramLookAhead bool

	// kvPrefixEverAdmitted is the #3391 first-prefill latch: false until a reuse-enabled
	// turn has admitted its prompt into the radix tree (generateReused step 3). While
	// false, a turn's prompt tokens are booked as INELIGIBLE in the cacheobs hit-rate
	// denominator ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â nothing has been admitted, so nothing could match: the always-cold
	// first prefill the raw ratio unfairly counts against the cache. Deliberately one-way
	// and planner-scoped (matching the tree's scope): a tree later evicted back to empty
	// keeps counting prompts as eligible, and on the multi-tenant scoped tree a tenant's
	// own cold first prefill after another tenant warmed the latch counts as eligible too
	// ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â both over-count the denominator, which can only UNDER-state the filtered ratio,
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
	// idle state, so every constructor ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â including a bare &InKernelPlanner{} ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â can
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
	// warmDisk fixes the local restart-cache policy and loaded artifact identity.
	// Empty config selects the bounded per-user default; an empty model digest
	// declines persistence because a filename is not a model-byte identity.
	deepSeekV41ChatProfile string              // immutable construction-time prompt contract
	nativeChatTemplate     *NativeChatTemplate // immutable, explicit text renderer
	warmDiskConfig         WarmDiskConfig
	warmDiskModelIdentity  string
	warmDiskCapture        bool
	warmDiskCaptureBudget  int64
	warmDiskCaptured       *model.PrefixSnapshot
	warmDiskCapturedTokens []int
	warmDiskCapturedLogits []float32
	warmDiskCapturedScope  radixkv.CacheIdentity

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
	warmClaimNow   func() time.Time
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
	// system-context sources (inkernel_incremental.go). nil ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â the default and every
	// planner that never calls SetIncrementalContext ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â leaves the historical planner
	// byte-for-byte inert. ctxSnapshot is the durable comparison state advanced on each
	// applied update; ctxMu guards both fields so a concurrent reconcile cannot race a
	// snapshot advance.
	incrementalContext *IncrementalContext
	ctxSnapshot        ContextSnapshot
	// ctxLayerMemo is the OPTIONAL per-session Source/Registry/Snapshot layer memo
	// (inkernel_incremental.go). nil ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â the default and every planner that never calls
	// SetContextLayerMemo ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â leaves RenderIncrementalContext on the historical full
	// rebuild. It is guarded by ctxMu alongside the registry fields above; its own
	// internal state is guarded by the memo's own mutex.
	ctxLayerMemo *ContextLayerMemo
	ctxMu        sync.Mutex
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
// none ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â the observation then keys the empty bucket rather than fabricating an id (#13120).
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

// ValidateSamplingSeed checks route support before a streaming response commits.
func (p *InKernelPlanner) ValidateSamplingSeed(seed *int64) error {
	if seed != nil && p.dflash2 != nil {
		return ErrSeedUnsupported
	}
	return nil
}

// nextSeedlessSampleSeed gives one seedless sampled request its own RNG stream.
// Adapted from DeepInfer's seedless sampling-seed rule (Apache-2.0,
// deepseek-ai/DeepInfer at 774ce244b87cc0eb8d4b94ab9f5de2ab22049f01); no
// upstream source is copied.
func (p *InKernelPlanner) nextSeedlessSampleSeed() int64 {
	if !p.seedExplicit {
		return int64(randv2.Uint64())
	}
	n := p.seedRequests.Add(1) - 1
	return int64(model.SplitMix64(uint64(p.seed) + n*model.SplitMixConstant))
}

// requestSampleSeed seeds one decode's planner RNG. A greedy or request-keyed
// decode never draws from it, so only a seedless sampled decode consumes a stream.
func (p *InKernelPlanner) requestSampleSeed(ctx context.Context, temp float64) int64 {
	if seed, ok := nativeSeedlessSampleSeedFromContext(ctx); ok {
		return seed
	}
	if _, keyed := nativeTargetSampleSeedFromContext(ctx); keyed || temp <= 0 {
		return p.seed
	}
	return p.nextSeedlessSampleSeed()
}

// validateNativeSamplingParams refuses sampling fields the native decode lanes
// cannot honor, before any model work or stream commit.
func validateNativeSamplingParams(sp SampleParams) error {
	if sp.MinP != nil && !(*sp.MinP >= 0 && *sp.MinP <= 1) {
		return ErrNativeMinPUnsupported
	}
	return nil
}

// validateV41EngramLookAhead refuses incompatible explicit configuration before
// any forward or output. It never disables an existing batch/speculative route.
func (p *InKernelPlanner) validateV41EngramLookAhead() error {
	if !p.v41EngramLookAhead {
		return nil
	}
	if p.m == nil || !p.m.Cfg.IsDeepSeekV41() || !p.m.V41EngramAttached() {
		return fmt.Errorf("%w: Engram look-ahead requires an attached V4.1 model", model.ErrV41NativeUnsupported)
	}
	if p.batchDecode {
		return fmt.Errorf("%w: Engram look-ahead requires explicit serial decode (FAK_INKERNEL_BATCH=off for serve)", model.ErrV41NativeUnsupported)
	}
	if p.dflash2 != nil || p.speculativeEngine != nil || p.vulkanMTP || p.vulkanMTPDraftFactory != nil || p.MetalMTPCoordinator() != nil || p.m.ExpertParallelRanks() > 1 || p.m.IsExpertParallelRankLocal() {
		return fmt.Errorf("%w: Engram look-ahead does not support speculative, DFlash2 or expert-parallel execution", model.ErrV41NativeUnsupported)
	}
	return nil
}

func (p *InKernelPlanner) Complete(ctx context.Context, messages []Message, tools []ToolDef, opts ...SampleOpt) (comp *Completion, err error) {
	if err := p.validateV41EngramLookAhead(); err != nil {
		return nil, err
	}
	offered := offeredToolsFor(ctx, tools)
	defer func() { err = p.executionDeadlineError(ctx, err) }()
	sp := applySampleOpts(opts...)
	return p.completeResolvedNative(ctx, messages, tools, sp, offered, &ctx, &err, opts...)
}

// completeResolvedNative consumes the ordinary invocation's resolved controls.
// It does not detach inputs, prepare another replica or grant cache credit.
// Publish the final context and named error on every unwind before Complete's
// existing deadline defer runs. That defer observes both mutable variables;
// normal-return assignments alone would lose their state on panic or Goexit.
func resolveInKernelTemperature(plannerTemp float64, sp SampleParams) float64 {
	if sp.Temperature != nil {
		return *sp.Temperature
	}
	if plannerTemp == 0 && sp.InheritUpstreamTemperature {
		return defaultInheritedTemperature
	}
	return plannerTemp
}

func (p *InKernelPlanner) completeResolvedNative(ctx context.Context, messages []Message, tools []ToolDef, sp SampleParams, offered OfferedTools, callerCtx *context.Context, callerErr *error, opts ...SampleOpt) (comp *Completion, err error) {
	defer func() { *callerCtx, *callerErr = ctx, err }()

	if sp.NativeBeamSearch != nil {
		return p.completeNativeBeamSearch(ctx, messages, tools, sp, opts...)
	}
	if err := p.ValidateSamplingSeed(sp.Seed); err != nil {
		return nil, err
	}
	if err := refuseNativeGuidedCarriers(sp.GuidedDecode); err != nil {
		return nil, err
	}
	if err := validateNativeSamplingParams(sp); err != nil {
		return nil, err
	}
	ctx = withNativeTargetSampleSeed(ctx, sp.Seed)
	ctx = withNativeMinP(ctx, sp.MinP)
	structured, err := p.newNativeRequestConstraint(sp, nil)
	if err != nil {
		return nil, err
	}
	if structured == nil {
		structured = p.newNativeForcedToolConstraint(sp, tools)
	}
	if structured != nil {
		if sp.NativeInferenceReceipt {
			return nil, fmt.Errorf("%w: %w", ErrNativeStructuredUnsupported, &model.NativeInferenceReceiptUnsupportedError{Reason: "structured output masks model logits"})
		}
		if len(tools) > 0 && !structured.forcedTool {
			return nil, fmt.Errorf("%w: structured output with tools", ErrNativeStructuredUnsupported)
		}
		if p.dflash2 != nil {
			return nil, fmt.Errorf("%w: dflash2 decode", ErrNativeStructuredUnsupported)
		}
		ctx = context.WithValue(ctx, nativeStructuredConstraintContextKey{}, structured)
	}
	// An in-kernel device-allocation failure (e.g. OOM on a small GPU under a large Claude
	// Code system prompt) panics deep below a CGO boundary with no error channel. Recover it
	// HERE ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â the narrowest Go frame that wraps the whole device decode (generateReused's
	// Prefill/Step + NewBackendSession's NewKV) AND returns the error the gateway already maps
	// to a client response ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â converting it into a typed error instead of crashing the serving
	// goroutine. The same seam recovers the two declared Flash attention-admission
	// refusals from direct Prefill/Step calls. Everything else re-panics, preserving
	// today's crash/stack behavior for bugs.
	return recoverDeviceCall(func() (comp *Completion, err error) {
		defer enginestep.Default.RequestStart()()
		releaseDFlash2Domain, domainErr := p.acquireDFlash2Domain(ctx)
		if domainErr != nil {
			return nil, domainErr
		}
		defer releaseDFlash2Domain()
		if p.kvPrecisionErr != nil {
			return nil, p.kvPrecisionErr
		}
		if p.qwenQ4KPrefillChunkTarget() && p.qwenQ4KPrefillChunkConfigErr != nil {
			return nil, p.qwenQ4KPrefillChunkConfigErr
		}
		if p.qwenQ4KPrefillChunkExplicit && p.qwenQ4KPrefillChunkTarget() {
			if _, capacityErr := p.deviceBoundedQwenQ4KPrefillChunkTokens(); capacityErr != nil {
				return nil, capacityErr
			}
		}
		if observer := sp.DecodeTokenObserver; observer != nil {
			delivered := &atomic.Bool{}
			ctx = context.WithValue(ctx, inKernelObserverDeliveryContextKey{}, delivered)
			sp.DecodeTokenObserver = func(piece, raw string) {
				// Mark before calling out: an observer can mutate a stream projector,
				// deliver output, or fail itself. None of those effects are rewindable.
				delivered.Store(true)
				observer(piece, raw)
			}
		}
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
		temp := resolveInKernelTemperature(p.temp, sp)
		// An omitted top_p keeps the full softmax draw; an explicit 0 keeps only the
		// most probable token instead of meaning "no nucleus".
		topP := nativeTopP(sp.TopP)
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
		if sp.NativeInferenceReceipt && (temp != 0 || (sp.TopP != nil && *sp.TopP != 0) || topK > 0 || (sp.MinP != nil && *sp.MinP != 0) || len(logitBias) > 0 || freqPenalty != 0 || presPenalty != 0) {
			return nil, &model.NativeInferenceReceiptUnsupportedError{Reason: "requires greedy sampling over unmodified logits"}
		}
		if err := p.validateDFlash2Request(sp, temp, freqPenalty, presPenalty); err != nil {
			return nil, err
		}
		if sp.Seed == nil && temp > 0 {
			// Resolved once per request so an OOM retry replays the same stream.
			ctx = withNativeSeedlessSampleSeed(ctx, p.nextSeedlessSampleSeed())
		}
		prepared, err := p.preparePrompt(ctx, messages, tools, sp, opts...)
		if err != nil {
			return nil, err
		}
		if prepared.multimodal != nil && sp.NativeInferenceReceipt {
			return nil, &model.NativeInferenceReceiptUnsupportedError{Reason: "multimodal prompt contains embedding rows rather than token ids"}
		}
		messages, tools = prepared.messages, prepared.tools
		chat, ids, maxNew := prepared.rendered, prepared.ids, prepared.maxNew
		ctx, err = p.withNativeVisionPrefill(ctx, prepared)
		if err != nil {
			return nil, err
		}
		ctx = withInKernelSharedPrefixBoundary(ctx, prepared.sharedBoundary)
		promptTokens := prepared.promptTokens()
		if err := p.refuseContextLength(promptTokens, maxNew); err != nil {
			return nil, err
		}
		stops := StopIDs(p.tok, p.m.Cfg)
		if p.dflash2 != nil {
			stops = dflash2StopIDs(p.tok, p.m.Cfg)
		}
		var turnContext []TurnAssessment
		if ta, ok := AssessTranscriptTurn(messages); ok {
			turnContext = append(turnContext, ta)
		}
		effectiveBudget := ResolveEffortBudget(sp.ReasoningEffort, sp.ThinkingBudget, turnContext...)
		startInSpan := strings.HasSuffix(chat, qwenThinkAssistantSeed)
		if structured != nil && (effectiveBudget > 0 || startInSpan) {
			if err := structured.armReasoning(p, startInSpan, stops); err != nil {
				if !structured.forcedTool {
					return nil, err
				}
				structured = nil
				ctx = context.WithValue(ctx, nativeStructuredConstraintContextKey{}, structured)
			}
		}
		var thinking *nativeThinkingConstraint
		if p.dflash2 != nil && effectiveBudget > 0 {
			return nil, errors.New("dflash2: thinking constraints are not supported")
		}
		if effectiveBudget > 0 {
			if sp.NativeInferenceReceipt {
				return nil, &model.NativeInferenceReceiptUnsupportedError{Reason: "thinking budget constrains token selection"}
			}
			thinking, err = p.newNativeThinkingConstraint(effectiveBudget, startInSpan, stops)
			if err != nil {
				return nil, err
			}
			thinking.spanByToken = structured.reasoningArmed()
			ctx = context.WithValue(ctx, nativeThinkingConstraintContextKey{}, thinking)
		}
		// emit runs per generated token: decode the piece, accumulate the text, and apply the
		// per-request string-suffix Stop (orthogonal to the token-ID stops). Returning true
		// ends the turn with the token counted and its text trimmed (the stop string is not
		// echoed back, matching the HTTP wires). Factoring decode into this closure keeps the
		// unbudgeted reuse/decode core (generateReused) tokenizer-free, so the candidate-#13
		// reuse and #14 eviction are witnessable on a synthetic model with no tokenizer fixture.
		//
		// Delta accrual: tokens append into the accumulator once; the stop probe scans only a
		// bounded tail window (the longest stop length) per token instead of re-materializing
		// the whole buffer per token ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â O(N) total string work per completion instead of
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
		// structuredContentStart is the byte offset just past a structured
		// response's close trigger; reasoning before it never reaches the stream.
		structuredContentStart := -1
		emit := func(next int) bool {
			if piece, derr := p.tok.Decode([]int{next}); derr == nil {
				sb.WriteString(piece)
				visible := true
				if structured.takeReasoningClosed() {
					structuredContentStart, visible = sb.Len(), false
				} else if structured.reasoningActive() {
					visible = false
				}
				if sp.DecodeTokenObserver != nil && visible {
					sp.DecodeTokenObserver(piece, "")
				}
				if !visible {
					// Stop strings apply to the response, never to hidden reasoning.
					return false
				}
				contentStart := max(structuredContentStart, 0)
				if scanner.appendPiece(piece) {
					if trimmed, hit := checkStop(sb.String()[contentStart:], sp.Stop); hit {
						kept := sb.String()[:contentStart] + trimmed
						sb.Reset()
						sb.WriteString(kept)
						return true
					}
				}
			}
			return false
		}

		// Resolve one request-local mode before dropping the ordinary planner gate.
		// Flash is target-only and opt-in; unsupported exact layouts retain the
		// lease through ordinary fallback, and never silently suppress speculation.
		complete := false
		coalesced := prepared.multimodal == nil && p.dflash2 == nil && p.coalescesQwenDecode()
		coalesceKind := inKernelCoalesceQwen
		if !coalesced && p.batchDecode && p.backend != nil && p.m.Cfg.IsDeepSeekV41() &&
			p.speculativeEngine == nil && !p.vulkanMTP && p.vulkanMTPDraftFactory == nil && p.MetalMTPCoordinator() == nil {
			eligibility, admissionErr := p.m.AcquireV41BatchEligibility(ctx, p.backend, 4)
			if admissionErr != nil {
				return nil, admissionErr
			}
			defer func() { eligibility.FinishCompleted(&err, complete) }()
			if eligibility.Eligible() {
				coalesced, coalesceKind = true, inKernelCoalesceV41
			}
		}

		comp, err = func() (*Completion, error) {
			// Serialize the entire device forward pass: a single-stream accelerator cannot run two
			// forwards at once without the concurrent op-streams corrupting shared device buffers
			// (see devMu). The plain CPU path owns a per-turn session and guards the shared radix tree
			// with p.mu itself, so it remains concurrent. Held across Prefill + decode.
			if p.requiresDeviceSerialization() && !coalesced {
				// #1589 phase-timeline probe: record the fan-out boundary and the
				// serialized critical section. Both calls are nil-safe no-ops unless a
				// caller opted into a concurrency profile.
				phase := p.concurrencyProfile.admit()
				deviceWait := time.Now()
				if err := p.devMu.LockContext(ctx); err != nil {
					enginestep.Default.ObservePhase(enginestep.PhaseDeviceWait, time.Since(deviceWait))
					return nil, err
				}
				enginestep.Default.ObservePhase(enginestep.PhaseDeviceWait, time.Since(deviceWait))
				p.concurrencyProfile.forwardEnter(phase, true)
				defer func() {
					p.concurrencyProfile.forwardExit(phase)
					p.devMu.Unlock()
				}()
				if err := p.refuseOversizeRequest(promptTokens, maxNew); err != nil {
					return nil, err
				}
				ctx = withPromptPanelHeadroom(ctx, p.promptPanelHeadroomGrantBytes(promptTokens, maxNew))
			} else if p.concurrencyProfile != nil {
				// Concurrent-unfused path (coalescing or CPU): still record the boundary
				// so the timeline can distinguish "no mutex serialized us" from "no
				// overlap happened at all".
				phase := p.concurrencyProfile.admit()
				p.concurrencyProfile.forwardEnter(phase, false)
				defer p.concurrencyProfile.forwardExit(phase)
			}
			if coalesced {
				releaseDeviceMemory, err := p.admitCoalescedDeviceMemory(promptTokens, maxNew)
				if err != nil {
					return nil, err
				}
				defer releaseDeviceMemory()
			}
			// #13267: the host-session (Metal) seam has no compute.Backend for refuseOversizeRequest
			// to ask, so an armed host ceiling prices it here, before any session or clone exists.
			var releaseHostMemory func()
			var hostErr error
			ctx, releaseHostMemory, hostErr = p.admitHostMemory(ctx, promptTokens, maxNew)
			if hostErr != nil {
				return nil, hostErr
			}
			defer releaseHostMemory()
			var measurement *nativeInferenceMeasurement
			if sp.NativeInferenceReceipt || (sp.DecodeTrace && sp.DecodeTokenObserver == nil) || sp.NativeDecodeTokenIDs || sp.NativeDecodeTopLogprobs > 0 {
				measurement = &nativeInferenceMeasurement{
					startedAt:             requestStarted,
					inferenceDisabled:     !sp.NativeInferenceReceipt,
					decodeTokenIDsEnabled: sp.NativeDecodeTokenIDs,
					topLogprobsK:          sp.NativeDecodeTopLogprobs,
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
				if p.dflash2 != nil {
					return p.generateDFlash2(runCtx, ids, maxNew, func(tokens []int) (dflash2StopPreview, error) {
						return previewDFlash2Stop(tokens, stops, scanner, p.tok.Decode)
					}, func(plan dflash2StopPreview) error {
						// Every piece and the exact stop prefix were decided before Commit.
						// The existing stream projector owns cross-block suffix withholding.
						for _, piece := range plan.pieces {
							if err := runCtx.Err(); err != nil {
								return err
							}
							sb.WriteString(piece)
							if sp.DecodeTokenObserver != nil {
								sp.DecodeTokenObserver(piece, "")
							}
						}
						if plan.trim > 0 {
							text := sb.String()
							if plan.trim > len(text) {
								return errors.New("dflash2: invalid committed stop trim")
							}
							text = text[:len(text)-plan.trim]
							sb.Reset()
							sb.WriteString(text)
						}
						scanner = plan.scanner
						return runCtx.Err()
					})
				}
				return p.generateReusedWithOOMRetry(runCtx, ids, maxNew, temp, topP, topK, logitBias, freqPenalty, presPenalty, stops, emit, func() {
					sb.Reset()
					structuredContentStart = -1
					scanner.reset()
					if thinking != nil {
						thinking.reset()
					}
					if structured != nil {
						structured.reset()
					}
					if measurement != nil {
						measurement.reset()
					}
				}, measurement)
			}
			var genRes inKernelGenerateResult
			p.hostStages.begin()
			if coalesced {
				genRes, err = p.runCoalescedGenerateKind(ctx, generate, coalesceKind)
			} else {
				genRes, err = generate(ctx)
			}
			if err != nil {
				// A generate path that bailed before its decode boundary (prefix-snapshot, admission,
				// context cancel) never reached the in-loop terminal emit, so the request still gets a
				// terminal observation here ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â bound to the same trace id (#13120).
				p.recordNativePhase(nativePhaseTraceID(ctx), NativePhaseTerminal, time.Now(), 0, false)
				if genRes.vulkanMTP != nil || genRes.specCost != nil {
					return &Completion{Model: p.modelID, VulkanMTP: genRes.vulkanMTP, NativeDecode: newNativeDecodeSummary(genRes)}, err
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
			// over the COMPUTED suffix (prompt minus the reused prefix) ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â the work actually done.
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
			// fast decode GEMV engaged) and the effective decode-worker count, so an operator can SEE ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â
			// without wall-clock guessing ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â that the AVX2/AVX-512 lane fired (not the reference path) and
			// that decode parallelizes across a modest, capped stream count rather than either 1 core or
			// an oversubscribed all-core dispatch (the pathology 40e0afd fixed on many-core amd64).
			q8kern, q8fused := model.Q8DecodeKernel()
			q8fusedMark := ""
			if q8fused {
				q8fusedMark = "+fused"
			}
			if p.dflash2 == nil {
				p.logExecutionSummary(q8kern, q8fusedMark, promptTok, genRes.cacheable, matched, computed, prefillS, prefTPS, gen, decodeS, decTPS, inKernelCohortReceiptClause(genRes.batchReceipt))
			}
			// Feed the process-global KV-prefix reuse tap so this turn's split hit-rate reaches
			// /metrics, not just this log line ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â the live measurement of the frozen-trajectory
			// cache cliff (docs/explainers/frozen-trajectory-cache-cliff.md). #3390: BOTH halves ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â
			// the lookup-side index match (cacheability) and the realized serve (matched/prompt) ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â
			// so the gap lost to eviction/admission is observable, not folded into "miss".
			// #3391 adds (a) the eligibility-filtered denominator ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â sampled BEFORE this turn is
			// latched as admitted, so the always-cold first prefill books zero eligible tokens
			// instead of depressing the fair hit-rate ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â and (b) the (model, tenant) attribution,
			// with the tenant drawn from the SAME authenticated prefix-cache identity the scoped
			// tree isolates on (unscoped traffic normalizes to "unknown" inside the tap).
			eligibleTok := p.kvPrefixEligiblePromptTokens(promptTok)
			p.noteKVPrefixAdmitted()
			tenant := ""
			if owner, scoped := prefixCacheIdentityFromContext(ctx); scoped {
				tenant = owner.Tenant
			}
			// A boot probe (readiness warmup, coherence probe) is not a served turn; booking it
			// would put kv_prefix turns ahead of the latency histograms and the perf ledger.
			if !IsBootProbe(ctx) {
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
			}
			compReuseEntry := cachemeta.FromProviderCache(cachemeta.ProviderCache{Provider: "fak-inkernel", ModelID: p.modelID, PromptTokens: int64(promptTok), CachedTokens: int64(matched)})

			// Split a Qwen3.5 reasoning block off the decoded text BEFORE it becomes Content
			// (and before the tool-call lift below reads it). A reasoning model (Ornith) opens
			// the turn with the open reasoning tag, then its final answer; renderChatMLTools does
			// NOT pre-seed the open tag, so the model emits both. splitReasoning is the in-kernel
			// equivalent of vLLM's --reasoning-parser qwen3: the reasoning lands in
			// ReasoningContent and only the post-reasoning answer flows into Content (and thus
			// into Claude Code's context). It is gated ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â a non-reasoning turn (no reasoning tags)
			// returns the decoded text untouched, so this is byte-identical to today for any
			// model that does not open a reasoning span.
			//
			// Raw-observer junction: the raw-text callback (rawText) carries the byte-identical
			// buffer the post-decode pipeline is about to consume, issued BEFORE splitReasoning
			// reads it. It is not called at all when no observer was requested.
			if sp.DecodeTokenObserver != nil {
				sp.DecodeTokenObserver("", sb.String())
			}
			var reasoning, content string
			if structured != nil {
				reasoning, content = structured.splitOutput(sb.String(), structuredContentStart)
			} else if p.m.Cfg.IsInkling2Loop() {
				reasoning, content = SplitInkling2Output(sb.String())
			} else {
				reasoning, content = splitNativeBudgetReasoning(sb.String(), thinking != nil && startInSpan)
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
				if thinking != nil && thinking.forced {
					content = StripReasoning(content)
				}
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
			populateReasoningTokensUsage(comp, p.tok)
			if p.dflash2 != nil {
				// This diagnostic host draft path has no qualified performance counter
				// mapping. Do not publish ordinary serial TPS or invented proposal counts.
				comp.Timings = nil
				comp.NativeDecode = &NativeDecodeSummary{Path: NativeDecodePathSpeculative}
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
				if err := validateNativeInferenceReceiptForwardCoverage(p.m.Cfg, receipt); err != nil {
					return nil, err
				}
				renderedSum := sha256.Sum256([]byte(prepared.rendered))
				receipt.PromptTokenIDs = append([]int(nil), prepared.ids...)
				receipt.TokenizerID = p.tok.Identity()
				receipt.RendererID = p.promptRendererID(sp)
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
			if sp.NativeDecodeTopLogprobs > 0 && measurement != nil {
				var decode func([]int) (string, error)
				if p.tok != nil {
					decode = p.tok.Decode
				}
				comp.NativeDecodeTopLogprobs = buildNativeDecodeTopLogprobs(measurement, decode)
			}
			// Lift the model's text-form <tool_call> emissions into structured Message.ToolCalls
			// (Hermes dialect == Qwen2.5 native), set FinishReason="tool_calls", and flag a
			// claimed-but-unparseable call ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â the SAME normalization every proxy adapter runs, so
			// the in-kernel forward becomes a first-class tool-calling planner. Without this the
			// gateway adjudicates nothing (it reads Message.ToolCalls) and the Anthropic wire never
			// emits a tool_use block, so Claude Code's agent loop has nothing to execute.
			if structured == nil || structured.forcedTool {
				comp = normalizeInKernelToolCalls(comp, offered, sp.ToolChoice, tools, messages)
			}
			return comp, nil
		}()
		complete = true
		return comp, err
	})
}

func (p *InKernelPlanner) requiresDeviceSerialization() bool {
	return p != nil && (p.backend != nil || p.metal)
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
//  4. An abstract RELEASE callback â€” the residency claim handover owned by the later warm
//     API (fak#13341). This leaf never dereferences a radix claim; it only guarantees the
//     callback fires at most once, idempotently, on Complete or Release.
//
// The zero value is a usable idle state, so every constructor â€” including a bare
// &InKernelPlanner{} in a test â€” can Begin a startup warm with no initialization step to
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
// startup warm can begin. It is idempotent â€” the second and later calls are no-ops, and the
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

// DeviceTransientPool reports the backend's request-transient reuse pool; ok is
// false when the backend keeps none (CPU, CUDA, or a non-pooling build).
func (p *InKernelPlanner) DeviceTransientPool() (DeviceTransientPoolStats, bool) {
	if p == nil || p.backend == nil {
		return DeviceTransientPoolStats{}, false
	}
	pool, ok := p.backend.(TransientPoolBackend)
	if !ok {
		return DeviceTransientPoolStats{}, false
	}
	buffers, bytes, peak := pool.TransientPoolStats()
	return DeviceTransientPoolStats{Backend: p.backend.Name(), PooledBuffers: buffers, PooledBytes: bytes, PeakPooledBytes: peak}, true
}
