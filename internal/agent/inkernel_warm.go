package agent

// inkernel_warm.go — #13344 (agent-startup-cache-warm CW-04): the operator-facing
// native KV cache WARM API with exact restore readback.
//
// WHY this exists. CW-02 (#13352, warm_prefix.go) derives the canonical stable
// descriptor of the prefix a startup warm targets, and CW-03 (#13351,
// inkernel_decode.go's PrimeCacheState) materializes a prompt's KV state through the
// exact production lookup/prefill/admission path without generating a token. What was
// still missing is the SEAM BETWEEN THEM: a single call that takes the descriptor and
// returns EVIDENCE that the admitted state is genuinely restorable by the request that
// will arrive next.
//
// CONTRACT (the load-bearing properties the witness pins):
//
//  1. Restore, not presence. Success requires reading the state back — the exact
//     prefix the spec names, under the spec's own tenant/agent cache scope, with a
//     real (non-nil) KV payload. A structural MatchLen match or a nil-KV entry is
//     explicitly insufficient: WarmPrefix refuses to report a restore it did not
//     observe.
//  2. Zero generation. A warm is a populate, never a turn: it samples nothing,
//     emits nothing, runs no tool and books no demand usage. The receipt records
//     ZeroGenerated so a caller cannot mistake a warm for a served request.
//  3. Closed, bounded receipt. The receipt carries identity/generation, the
//     requested/admitted/restored token counts, the source tier, a closed
//     status/reason and the prime duration. It NEVER carries prompt text or
//     instruction contents — only counts, digests and closed tokens.
//  4. Fail-closed. An unusable planner, an unbounded descriptor or an unsupported
//     (bare-KV / recompute-only / V4.1) runtime is refused with an explicit
//     unsupported status. A disabled path never reads as warm readiness, and the
//     first-demand latch flips ONLY after a successful restore readback.
//
// This file owns the WARM ORCHESTRATION only: it reuses the descriptor (CW-02), the
// populate purpose (CW-03) and the read-back probe (cachedPrefixLen / tree lookup). It
// adds no new cache backend and no throughput algorithm.

import (
	"context"
	"errors"
	"time"

	"github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/radixkv"
)

// Warm status vocabulary — the closed set a WarmReceipt.Status may carry. The set is
// deliberately small: a caller branches on ready vs cold vs unsupported and never has to
// parse a free-text reason to know whether the cache is ready.
const (
	// WarmStatusReady means the descriptor's prefix was materialized AND read back
	// through the same scoped cache the next demand turn will consult.
	WarmStatusReady = "ready"
	// WarmStatusCold means the path ran but did not restore the full prefix (a
	// legitimate, explicitly-labelled miss — never a silent hit).
	WarmStatusCold = "cold"
	// WarmStatusUnsupported means this planner cannot warm at all (no tree, a
	// recompute-only model, an unqualified backend, or the V4.1 continuation chain
	// that is not yet landed). It is fail-closed.
	WarmStatusUnsupported = "unsupported"
)

// ErrWarmPrefixUnsupported is the fail-closed refusal for a WarmPrefix call on a planner
// that cannot materialize or restore a native prefix cache. It mirrors the descriptor's
// own ErrWarmPrefixUnavailable: a caller must never read a zero-value receipt as a warm.
var ErrWarmPrefixUnsupported = errors.New("agent: warm prefix is unsupported on this planner")

// WarmReceipt is the closed result of a WarmPrefix call. It is a VALUE handle: every
// field is a scalar or a copied string, so a caller cannot accidentally mutate shared
// state, and it deliberately carries NO prompt text, instruction bytes or tool schemas.
//
// The single load-bearing bit is Ready. The rest is evidence a caller can log or bind to
// a startup receipt without ever leaking the prompt's contents.
type WarmReceipt struct {
	// Ready is true only when a real restore readback observed the full stable prefix.
	Ready bool `json:"ready"`
	// Status is the closed token (ready | cold | unsupported).
	Status string `json:"status"`
	// Reason is the closed sub-reason for a non-ready status (e.g. "no_tree",
	// "recompute_only", "v41_continuation_pending", "partial_restore"). Empty when ready.
	Reason string `json:"reason,omitempty"`

	// Identity is the descriptor's folded identity digest — the one key the warm store
	// is indexed on. Carried so a caller can bind the receipt to the descriptor it came
	// from without re-deriving it.
	Identity string `json:"identity"`
	// RuntimeGeneration is the decode-path/runtime generation the warm was captured on.
	RuntimeGeneration string `json:"runtime_generation"`
	// StableTokenDigest is the content digest of the stable prefix (never the tokens).
	StableTokenDigest string `json:"stable_token_digest"`

	// Scope is the authenticated cache owner the restore readback was performed under.
	Scope radixkv.CacheIdentity `json:"scope"`

	// RequestedTokens is the descriptor's stable boundary the warm asked to materialize.
	RequestedTokens int `json:"requested_tokens"`
	// AdmittedTokens is how many prompt tokens the populate purpose admitted (0 on a
	// refused/unsupported planner).
	AdmittedTokens int `json:"admitted_tokens"`
	// RestoredTokens is how many leading tokens the READBACK observed resident under the
	// scope — the number a next request could actually reuse. It is the count that decides
	// Ready.
	RestoredTokens int `json:"restored_tokens"`

	// SourceTier names where the restored state was read from (device_l1 | host_dram_l2 |
	// remote_http_l3), or empty on a miss. It is the truthful tier attribution, never an
	// optimistic default.
	SourceTier radixkv.SnapshotTier `json:"source_tier,omitempty"`

	// ZeroGenerated records that warming generated no token. It is always true; it is
	// carried explicitly so a receipt reader cannot mistake a populate for a served turn.
	ZeroGenerated bool `json:"zero_generated"`
	// PrimeDuration is the wall-clock time the materialize+readback took. It is a
	// [SW-VERIFIED] host timing observation, not a hardware cache-hit claim.
	PrimeDuration time.Duration `json:"prime_duration_ns"`

	// Aux carries the OPTIONAL auxiliary-cache warming evidence (expert-profile and
	// V4.1 Engram layers). It is present only when the caller configured auxiliary
	// warming via SetAuxWarmConfig; a planner without it leaves Aux nil, so the
	// historical KV-only receipt is byte-for-byte unchanged. Each layer carries its
	// OWN status/counters — a KV-ready warm with a declined expert layer reports
	// ready + expert:unsupported, never a single conflated bit.
	Aux *AuxWarmReceipt `json:"aux,omitempty"`

	// Claim carries the OPTIONAL bounded-residency evidence (CW-06, #13341): the finite
	// startup claim the warm acquired over the restored prefix. It is present only when
	// the caller configured a claim via SetWarmClaimConfig; a planner without it leaves
	// Claim nil, so the historical CW-04 receipt is byte-for-byte unchanged. A ready warm
	// whose claim could NOT be acquired is downgraded to cold (a warm that may be evicted
	// before the first request is not a durable warm), so Ready==true implies a live
	// claim whenever Claim is configured.
	Claim *WarmClaimReceipt `json:"claim,omitempty"`
}

// WarmClaimReceipt is the machine-checkable evidence that a warm's restored prefix is
// held by a FINITE, expiring startup claim — not merely resident at the moment of an
// immediate readback. It is a value handle (scalars + a copied deadline); it carries no
// prompt text and no model weights.
//
// The load-bearing bit is Live. A claim that has expired, been released, or whose node
// incarnation was replaced validates false, so a stale receipt can never report warm
// readiness the next request cannot actually reuse.
type WarmClaimReceipt struct {
	// Live is true only when the claim validated against the tree at receipt time.
	Live bool `json:"live"`
	// Generation is the claim's monotonic incarnation stamp. Two warms over the same
	// token prefix carry different generations, so a handle from the older warm is
	// distinguishable from the newer incarnation.
	Generation uint64 `json:"generation"`
	// Tokens is the claimed prefix length in tokens.
	Tokens int `json:"tokens"`
	// Bytes is the claimed prefix's resident payload bytes — the finite byte budget the
	// claim may pin. It draws ONLY on spare capacity, never on demand residency.
	Bytes int64 `json:"bytes"`
	// ExpiresAtMillis is the claim's deadline in Unix milliseconds, or 0 when the claim
	// carries no TTL (a caller that promises an explicit Release). A non-zero deadline is
	// the moment after which Validate reports the claim stale.
	ExpiresAtMillis int64 `json:"expires_at_ms,omitempty"`
	// Reason is the closed sub-reason for a non-live claim (e.g. "quota", "expired",
	// "stale_incarnation", "released", "no_tree"). Empty when live.
	Reason string `json:"reason,omitempty"`
}

// WarmClaimConfig bounds the optional finite-residency startup claim a warm acquires
// (CW-06, #13341). SpareBytes/SpareTokens draw ONLY on capacity that is not already
// holding demand residency, so a claim can never displace a real request's prefix; TTL
// bounds how long the claim may pin. It is inert by default: a planner with no config
// warms exactly as CW-04 did and leaves WarmReceipt.Claim nil.
type WarmClaimConfig struct {
	// SpareBytes is the total spare byte budget claims may draw on. Zero means the byte
	// axis is unbounded (only the token quota applies).
	SpareBytes int64
	// SpareTokens is the total spare token budget claims may draw on. Zero means the
	// token axis is unbounded.
	SpareTokens int
	// TTL bounds how long a claim may pin before it lazily expires. Zero means the claim
	// never expires on its own (the caller promises an explicit Release on shutdown).
	TTL time.Duration
}

// Aux status vocabulary — the closed set an AuxLayerReceipt.Status may carry. It mirrors
// the WarmStatus set so a receipt reader branches on the same ready/cold/unsupported
// trichotomy for every layer, plus "skipped" for optional work a cancellation or the
// remaining-reserve gate stopped before it could run.
const (
	// AuxStatusReady means the layer's optional warming ran and established residency.
	AuxStatusReady = "ready"
	// AuxStatusCold means the layer ran but retained/restored nothing usable (an honest
	// miss — e.g. an empty profile or a zero remaining reserve).
	AuxStatusCold = "cold"
	// AuxStatusUnsupported means the model cannot warm this layer at all (no expert
	// checkpoint tier, no Engram stage). It is fail-closed.
	AuxStatusUnsupported = "unsupported"
	// AuxStatusSkipped means the layer was configured but optional work did not run:
	// the request was cancelled, or the remaining reserve after KV/request headroom
	// was non-positive. Optional warming must never run on a cancel or a spent budget.
	AuxStatusSkipped = "skipped"
)

// AuxLayerReceipt is the machine-checkable result of warming ONE optional auxiliary
// layer. Every field is a scalar/copied value; it carries no prompt text and no model
// weights. Ready is the single bit a caller branches on; the counters are the layer's
// own independent ledger, so a layer that ran but seated nothing cannot read as ready.
type AuxLayerReceipt struct {
	// Status is the closed token (ready | cold | unsupported | skipped).
	Status string `json:"status"`
	// Reason is the closed sub-reason for a non-ready status (e.g. "no_checkpoint_tier",
	// "no_engram_stage", "cancelled", "no_reserve"). Empty when ready.
	Reason string `json:"reason,omitempty"`
	// Requested is how many units the layer was asked to warm (expert projections
	// selected, or Engram row addresses presented).
	Requested int `json:"requested"`
	// Retained is how many units the layer actually seated (retained projections or
	// resident Engram rows). Zero on a declined/unsupported/skipped layer.
	Retained int `json:"retained"`
	// BytesRead is the backing bytes the layer's warm read, from the layer's own
	// counter snapshot — never inferred from configuration.
	BytesRead int64 `json:"bytes_read"`
	// BytesRetained is the resident bytes the layer established (expert layers only;
	// zero for Engram, which reports residency in rows).
	BytesRetained int64 `json:"bytes_retained,omitempty"`
}

// AuxWarmReceipt folds the two optional auxiliary layers into one bounded receipt, each
// with its own independent status and counters. It is attached to WarmReceipt.Aux only
// when auxiliary warming is configured, so a KV-only warm never grows this block.
type AuxWarmReceipt struct {
	// Expert is the expert-profile (Model.WarmExpertProfile) layer result.
	Expert AuxLayerReceipt `json:"expert"`
	// Engram is the V4.1 Engram-prefix (Model.WarmV41EngramPrefix) layer result.
	Engram AuxLayerReceipt `json:"engram"`
	// ReserveBytes is the remaining startup reserve this call was allowed to spend on
	// optional warming, after KV/request headroom was deducted. It is the bound the
	// layer budgets were clamped against.
	ReserveBytes int64 `json:"reserve_bytes"`
}

// AuxWarmConfig configures OPTIONAL auxiliary-cache warming performed by WarmPrefix after
// the KV prefix is restored. It is inert by default: a planner with no config warms only
// KV and leaves WarmReceipt.Aux nil. The expert identity is presented (not derived) so a
// stale recorded profile is refused by the model's own identity check rather than warming
// the wrong projections.
type AuxWarmConfig struct {
	// ExpertProfile is the recorded demand profile to warm the expert cache from. An
	// empty profile warms nothing (the model reports no_profile).
	ExpertProfile model.ExpertWarmProfile
	// ExpertIdentity is the (checkpoint, quantization, layout) identity the profile was
	// recorded against; a mismatch makes the profile stale and the warm declines.
	ExpertIdentity model.ExpertWarmProfileIdentity
	// EngramTokens is the token prefix whose Engram rows are prefetched on a V4.1 model.
	// Empty (with EngramMask) skips the Engram layer explicitly.
	EngramTokens []int
	// EngramMask is the optional per-token mask for the Engram prefix hash.
	EngramMask []bool
	// ReserveBytes is the total startup reserve available to optional warming, BEFORE
	// KV/request headroom is deducted. The orchestrator spends what remains after the
	// caller-declared KVHeadroomBytes; a non-positive remainder skips optional work.
	ReserveBytes int64
	// KVHeadroomBytes is the slice of ReserveBytes reserved for the KV cache and request
	// headroom. It is deducted before any optional layer runs, so a reserve that only
	// covers KV never speculatively warms the expert/Engram caches.
	KVHeadroomBytes int64
}

// auxWarmAdapter is the narrow seam the warm orchestrator calls for OPTIONAL auxiliary
// warming. It is satisfied by *model.Model (the real adapter, retained on the planner for
// real requests) and lets a witness inject a counting fake without constructing a model
// with a checkpoint tier or an Engram stage. Every method mirrors an exported model
// capability; the orchestrator adds no selection, faulting or cache math of its own.
type auxWarmAdapter interface {
	WarmExpertProfile(profile model.ExpertWarmProfile, identity model.ExpertWarmProfileIdentity, budgetBytes int64) model.ExpertWarmResult
	WarmV41EngramPrefix(tokens []int, mask []bool) (model.V41EngramWarmDelta, error)
}

// Usable reports whether this receipt describes a restorable warm. It is the AND of the
// ready bit, the closed ready status, and a non-empty restored prefix, so a cancelled or
// partial warm can never read as usable.
func (r WarmReceipt) Usable() bool {
	return r.Ready && r.Status == WarmStatusReady && r.RestoredTokens > 0
}

// WarmPrefix materializes the KV state for spec's stable prefix through the existing
// cache-populate purpose and then READS IT BACK under the spec's own tenant/agent scope,
// returning a bounded receipt. It never generates a token and never carries prompt text.
//
// The readback is the point. A populate that reports admitted state but cannot be read
// back through the scoped cache is a cold result, not a warm one: the next demand turn
// would prefill the prefix again. WarmPrefix therefore requires the readback length to
// reach the descriptor's stable boundary before it reports Ready, and only then flips the
// first-demand eligibility latch (noteKVPrefixAdmitted) so the following turn's prompt is
// counted as reusable.
//
// V4.1 (and any recompute-only or bare-KV runtime) remains unsupported here: the native
// continuation-capability chain (fak#13342 / #13338 / #13334) owns that admission, and a
// warm must never claim readiness for a state the runtime cannot restore.
func (p *InKernelPlanner) WarmPrefix(ctx context.Context, spec WarmPrefixSpec) (WarmReceipt, error) {
	started := time.Now()
	receipt := WarmReceipt{
		Status:            WarmStatusUnsupported,
		Identity:          spec.Identity,
		RuntimeGeneration: spec.RuntimeGeneration,
		StableTokenDigest: spec.StableTokenDigest,
		Scope:             spec.Scope,
		RequestedTokens:   spec.StableTokens,
		ZeroGenerated:     true,
	}

	if p == nil {
		return finishWarm(receipt, started), ErrWarmPrefixUnsupported
	}
	if ctx == nil {
		ctx = context.Background()
	}
	if err := ctx.Err(); err != nil {
		receipt.Status = WarmStatusCold
		receipt.Reason = "cancelled"
		return finishWarm(receipt, started), err
	}

	// Fail-closed gates, each with a distinct closed reason so a caller can tell WHICH
	// precondition failed without parsing prose.
	if !spec.Bounded() {
		receipt.Reason = "unbounded_descriptor"
		return finishWarm(receipt, started), ErrWarmPrefixUnsupported
	}
	p.mu.Lock()
	tree := p.tree
	inputs := p.warmInputs
	p.mu.Unlock()
	if tree == nil {
		receipt.Reason = "no_tree"
		return finishWarm(receipt, started), ErrWarmPrefixUnsupported
	}
	if !inKernelPlannerPrefixReuseSupported(p.m, p.backend) {
		receipt.Reason = "recompute_only"
		return finishWarm(receipt, started), ErrWarmPrefixUnsupported
	}
	if p.warmIsV41() {
		receipt.Reason = "v41_continuation_pending"
		return finishWarm(receipt, started), ErrWarmPrefixUnsupported
	}

	// The descriptor's stable boundary is the token sequence we materialize. Reconstruct
	// it through the SAME production encoder the descriptor used (CW-02), so the tokens we
	// warm are exactly the tokens the descriptor's digest covers.
	tokens, err := p.warmStableTokens(ctx, inputs)
	if err != nil {
		receipt.Status = WarmStatusCold
		receipt.Reason = "encode_failed"
		return finishWarm(receipt, started), err
	}
	if len(tokens) != spec.StableTokens {
		// The boundary the descriptor reported and the boundary we can re-encode must
		// agree; a mismatch means the stable inputs changed under the descriptor.
		receipt.Status = WarmStatusCold
		receipt.Reason = "boundary_mismatch"
		return finishWarm(receipt, started), nil
	}

	prime, err := p.PrimeCacheState(ctx, tokens)
	if err != nil {
		receipt.Status = WarmStatusCold
		receipt.Reason = "prime_failed"
		receipt.AdmittedTokens = prime.Tokens
		return finishWarm(receipt, started), err
	}
	receipt.AdmittedTokens = prime.Tokens

	// The READBACK: a real restore observation, not the populate's own bookkeeping. We walk
	// the same scoped/unscoped cache the next demand turn consults and require the matched
	// prefix to reach the stable boundary with a live (non-nil) payload.
	restored, tier, ok := p.warmRestoreReadback(spec.Scope, tokens)
	receipt.RestoredTokens = restored
	receipt.SourceTier = tier
	if !ok || restored < spec.StableTokens {
		receipt.Status = WarmStatusCold
		receipt.Reason = "partial_restore"
		return finishWarm(receipt, started), nil
	}

	receipt.Ready = true
	receipt.Status = WarmStatusReady
	receipt.Reason = ""
	// CW-06 (#13341): bind the restored prefix to a FINITE residency claim, so the warm
	// survives the gap before the first real request instead of being the first LRU
	// victim. This runs BEFORE the readiness latch: a configured claim that cannot be
	// acquired means the prefix may be evicted before demand looks, so an unclaimed warm
	// is downgraded to cold rather than reporting a readiness the next request cannot use.
	if p.warmClaimConfigured() {
		claim, _ := p.acquireWarmClaim(tokens, spec)
		receipt.Claim = claim
		if claim == nil || !claim.Live {
			receipt.Ready = false
			receipt.Status = WarmStatusCold
			receipt.Reason = "claim_unavailable"
			return finishWarm(receipt, started), nil
		}
	}
	// Flip the first-demand latch ONLY now: a real restore was observed (and, when
	// configured, durably claimed), so the following turn's prompt is genuinely eligible
	// to reuse this prefix.
	p.noteKVPrefixAdmitted()
	// Optional auxiliary warming runs LAST, after the KV prefix is genuinely restorable,
	// so a failed optional layer can never cost the KV warm its readiness. It is inert
	// unless the caller configured it, and it stops on the request's own cancellation.
	receipt.Aux = p.warmAuxiliaryCaches(ctx, p.warmIsV41())
	return finishWarm(receipt, started), nil
}

// warmClaimConfigured reports whether a finite-residency startup claim was configured.
// It reads under mu alongside every other planner-owned warm field.
func (p *InKernelPlanner) warmClaimConfigured() bool {
	if p == nil {
		return false
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.warmClaimSet
}

// SetWarmClaimConfig installs the optional finite-residency startup claim config (CW-06,
// #13341). A planner that never calls it warms exactly as CW-04 did and leaves
// WarmReceipt.Claim nil.
func (p *InKernelPlanner) SetWarmClaimConfig(cfg WarmClaimConfig) {
	if p == nil {
		return
	}
	p.mu.Lock()
	p.warmClaim = radixkv.StartupClaimConfig{
		SpareBytes:  cfg.SpareBytes,
		SpareTokens: cfg.SpareTokens,
		TTL:         cfg.TTL,
	}
	p.warmClaimSet = true
	p.mu.Unlock()
}

// acquireWarmClaim takes a bounded, expiring lease on the just-restored prefix through the
// shared claim boundary, binds its release to the planner's startup-warm lifecycle, and
// returns the machine-checkable receipt. It returns a non-live receipt (never a nil claim)
// with a closed reason when the claim cannot be honored, so a caller can always tell WHY.
//
// The claim boundary SHARES p.mu as its locker (the same ScopedTree discipline), so claim
// mutation serializes with every other tree access. It must therefore never be reached
// while p.mu is held; WarmPrefix releases the lock before calling here.
func (p *InKernelPlanner) acquireWarmClaim(tokens []int, spec WarmPrefixSpec) (*WarmClaimReceipt, string) {
	if p == nil || len(tokens) == 0 {
		return &WarmClaimReceipt{Reason: "no_tree"}, "no_tree"
	}

	// Resolve the claim boundary and config under the lock, then call OUTSIDE it: the
	// StartupCache re-acquires p.mu itself, so holding it here would self-deadlock.
	p.mu.Lock()
	tree := p.tree
	cfg := p.warmClaim
	cache := p.warmClaimCache
	if tree == nil {
		p.mu.Unlock()
		return &WarmClaimReceipt{Reason: "no_tree"}, "no_tree"
	}
	// A previous claim over this planner may still be bound: capture it here and release
	// it AFTER unlocking (the boundary shares p.mu, so releasing under the lock would
	// self-deadlock). Repeated warm calls thus cannot multiply reservations.
	prev := p.warmClaimHandle
	p.warmClaimHandle = radixkv.StartupClaim{}
	if cache == nil {
		// Share p.mu as the boundary's locker: claim acquisition/validation/release then
		// serialize with every other tree access over this planner, exactly as scopedTree
		// does. A separate lock would race the demand path.
		cache = radixkv.NewStartupCacheWithLocker(tree, &p.mu)
		p.warmClaimCache = cache
	}
	p.mu.Unlock()

	if prev.Gen() != 0 {
		cache.Release(prev)
	}

	// WarmInsert the restored prefix (idempotent: an already-cached prefix is upgraded,
	// never double-counted) and take the bounded lease. The kv argument is nil: the warm
	// path already admitted the payload through the scoped/unscoped tree, and the claim's
	// byte accounting falls back to the 4-bytes/token bound, which is the finite quota
	// that matters here.
	claim, err := cache.AcquireStartup(radixkv.StartupClaimConfig{
		SpareBytes:  cfg.SpareBytes,
		SpareTokens: cfg.SpareTokens,
		TTL:         cfg.TTL,
	}, tokens, nil)
	if err != nil {
		reason := "quota"
		if errors.Is(err, radixkv.ErrStartupClaimExpired) {
			reason = "expired"
		}
		return &WarmClaimReceipt{Reason: reason}, reason
	}

	// Bind the claim's release into the planner's startup-warm lifecycle: a shutdown
	// (ReleaseStartupWarm) or a completed slot releases the lease exactly once, so a
	// prepared prefix returns to normal demand residency rather than pinning forever.
	bound := claim
	p.warmState.BindRelease(func() { cache.Release(bound) })

	// Validate the lease in the same serialized turn that acquired it. A live claim is
	// the readiness evidence; an immediate validation failure is reported honestly.
	live := cache.Validate(claim)
	rec := &WarmClaimReceipt{
		Live:       live,
		Generation: claim.Gen(),
		Tokens:     claim.Tokens(),
		Bytes:      claim.Bytes(),
	}
	if exp, ok := claim.ExpiresAt(); ok {
		rec.ExpiresAtMillis = exp.UnixMilli()
	}
	if !live {
		rec.Reason = "stale_incarnation"
	}

	p.mu.Lock()
	p.warmClaimHandle = claim
	p.warmClaimScope = spec.Scope
	p.warmClaimScopeSet = true
	p.mu.Unlock()
	return rec, rec.Reason
}

// warmHandoffScopeMatches reports whether the planner's bound startup claim was acquired
// under an authenticated cache scope compatible with the request's owner. A claim warmed
// for a DIFFERENT tenant (or a different agent within a tenant) must never be handed to a
// request that did not ask for it, so a scope mismatch is an identity mismatch. The served
// path always binds a scope, so the common case is an exact tenant+agent match.
func (p *InKernelPlanner) warmHandoffScopeMatches(owner radixkv.CacheIdentity) bool {
	if p == nil {
		return false
	}
	p.mu.Lock()
	scope := p.warmClaimScope
	set := p.warmClaimScopeSet
	p.mu.Unlock()
	if !set {
		return false
	}
	return scope.Tenant == owner.Tenant && scope.Agent == owner.Agent
}

// WarmHandoffReceipt is the closed result of a demand-path startup-warm handoff (CW-16,
// #13336). It records, in machine-checkable form, whether THIS request adopted the
// prepared startup prefix and — critically — whether doing so RELEASED the startup claim
// so the request owns the continuation. It carries no prompt text and no model weights.
type WarmHandoffReceipt struct {
	// HandedOff is true only when this request adopted the startup warm's prefix: the
	// claim was live and matched this request's prompt, and the continuation was taken
	// under the existing synchronization.
	HandedOff bool `json:"handed_off"`
	// Reason is the closed sub-reason for a non-handoff (e.g. "no_claim",
	// "identity_mismatch", "stale_claim", "unsupported"). Empty when handed off.
	Reason string `json:"reason,omitempty"`
	// MatchedTokens is the leading token prefix this request adopted from the warm.
	MatchedTokens int `json:"matched_tokens"`
	// Generation is the startup claim incarnation that was handed off (0 when none).
	Generation uint64 `json:"generation,omitempty"`
	// Released is true exactly when the handoff released the startup claim. A handoff that
	// did not release would leak the startup lease past the first real request.
	Released bool `json:"released"`
}

// demandWarmHandoff is the CW-16 (#13336) first-demand seam: when the ordinary scoped
// lookup did not already serve this request from a cached prefix, the planner offers the
// request any live startup warm claim over the SAME token prefix. On an exact-identity
// match the request takes the prepared continuation under the claim boundary's own
// serialization and the startup claim is RELEASED — ownership moves from startup to this
// demand turn, so the prepared prefix is no longer pinned.
//
// Fail-closed contract:
//   - A planner with no bound claim, or a claim that no longer validates (expired,
//     evicted, replaced incarnation), reports a closed reason and hands off nothing; the
//     caller follows the ordinary truthful miss path.
//   - A claim whose token prefix does not cover this request's prompt is an identity
//     mismatch: the request must never adopt state it did not ask for.
//   - A prime/populate turn is never a demand: it owns its own claim and must not consume
//     it, so the handoff is a no-op when p.cachePopulate is set.
//
// The matched prefix is the claim's token length; the caller still performs its own
// suffix prefill for ids[matched:]. Releasing the claim does NOT evict the prefix — it
// returns it to normal demand residency, exactly as StartupCache.Release documents.
func (p *InKernelPlanner) demandWarmHandoff(ids []int) WarmHandoffReceipt {
	if p == nil || len(ids) == 0 {
		return WarmHandoffReceipt{Reason: "unsupported"}
	}
	// A populate purpose owns its own startup claim; it must not consume it as demand.
	if p.cachePopulate {
		return WarmHandoffReceipt{Reason: "populate"}
	}
	p.mu.Lock()
	cache := p.warmClaimCache
	handle := p.warmClaimHandle
	p.mu.Unlock()
	if cache == nil || handle.Gen() == 0 {
		return WarmHandoffReceipt{Reason: "no_claim"}
	}
	// Validate outside p.mu: the claim boundary SHARES p.mu as its locker, so validating
	// under the lock would self-deadlock (the same discipline acquireWarmClaim documents).
	if !cache.Validate(handle) {
		return WarmHandoffReceipt{Reason: "stale_claim"}
	}
	// Exact-identity match: the claim's prepared prefix must be a leading prefix of this
	// request's prompt. A shorter claim can still be handed off (the caller prefills the
	// suffix); a claim that is not a prefix at all is refused.
	claimed := handle.Tokens()
	if claimed <= 0 || claimed > len(ids) {
		return WarmHandoffReceipt{Reason: "identity_mismatch"}
	}
	// Release the startup claim: ownership of the continuation transfers to this demand
	// turn. Clear the bound handle under p.mu so WarmClaimLive reports false from here on,
	// then release outside the lock (the boundary re-acquires p.mu itself).
	p.mu.Lock()
	p.warmClaimHandle = radixkv.StartupClaim{}
	p.mu.Unlock()
	cache.Release(handle)
	// The startup slot's replacement warm would have renewed a claim; the demand turn now
	// owns residency, so mark the slot complete so the release callback cannot double-run.
	p.warmState.Complete()
	return WarmHandoffReceipt{
		HandedOff:     true,
		MatchedTokens: claimed,
		Generation:    handle.Gen(),
		Released:      true,
	}
}

// WarmClaimLive reports whether the planner's currently bound startup claim still
// validates: the claimed leaf is still attached, still carries this claim's incarnation,
// and has not expired. A planner with no bound claim reports false. It is the readback a
// first-demand handoff consults before treating prepared cache state as reusable.
func (p *InKernelPlanner) WarmClaimLive() bool {
	if p == nil {
		return false
	}
	p.mu.Lock()
	cache := p.warmClaimCache
	handle := p.warmClaimHandle
	p.mu.Unlock()
	if cache == nil || handle.Gen() == 0 {
		return false
	}
	return cache.Validate(handle)
}

// warmAuxiliaryCaches orchestrates the OPTIONAL auxiliary-cache layers (expert profile and
// V4.1 Engram) on top of a successful KV warm. It is inert when no config is installed
// (Aux stays nil) and never mutates the KV readiness. Each layer gets an INDEPENDENT
// status: a declined expert warm does not make the Engram layer cold, and vice versa.
//
// The remaining reserve after KV/request headroom is the aggregate budget the layers
// share: the expert layer is clamped to it, and a non-positive remainder skips optional
// work entirely rather than reading bytes that cannot become residency. A cancelled
// request stops before doing optional work. No selection, faulting or cache math happens
// here — every producer is the model's own exported capability.
func (p *InKernelPlanner) warmAuxiliaryCaches(ctx context.Context, isV41 bool) *AuxWarmReceipt {
	cfg, ok := p.auxWarmConfigSnapshot()
	if !ok {
		return nil
	}
	reserve := cfg.ReserveBytes - cfg.KVHeadroomBytes
	aux := &AuxWarmReceipt{
		ReserveBytes: reserve,
		Expert:       AuxLayerReceipt{Status: AuxStatusSkipped, Reason: "no_reserve"},
		Engram:       AuxLayerReceipt{Status: AuxStatusSkipped, Reason: "no_reserve"},
	}
	// A cancelled request does no optional work: the KV warm it already observed stands,
	// but speculative reads must never run against a caller that has gone away.
	if ctx != nil && ctx.Err() != nil {
		aux.Expert.Reason = "cancelled"
		aux.Engram.Reason = "cancelled"
		return aux
	}
	if reserve <= 0 {
		return aux
	}

	adapter := p.auxWarmAdapter()
	aux.Expert = warmExpertLayer(adapter, cfg, reserve)
	// The Engram layer runs only for a genuinely V4.1 runtime whose stage is wired; a
	// non-V4.1 model or an unwired stage is reported unsupported, never a silent zero.
	aux.Engram = warmEngramLayer(adapter, isV41, cfg)
	return aux
}

// warmExpertLayer runs the expert-profile warm clamped to the remaining reserve and folds
// the model's own counters into a layer receipt. A nil adapter or a tier-less model is
// reported unsupported; a zero/empty plan is an honest cold result.
func warmExpertLayer(adapter auxWarmAdapter, cfg AuxWarmConfig, reserve int64) AuxLayerReceipt {
	if adapter == nil {
		return AuxLayerReceipt{Status: AuxStatusUnsupported, Reason: "no_model"}
	}
	budget := reserve
	if budget > cfg.ReserveBytes {
		budget = cfg.ReserveBytes
	}
	res := adapter.WarmExpertProfile(cfg.ExpertProfile, cfg.ExpertIdentity, budget)
	layer := AuxLayerReceipt{
		Requested:     res.Selected,
		Retained:      res.Retained,
		BytesRead:     res.ReadBytes,
		BytesRetained: res.RetainedBytes,
	}
	switch {
	case res.Retained > 0:
		layer.Status = AuxStatusReady
		layer.Reason = string(res.Reason)
	case string(res.Reason) == "no_indexed_expert":
		// A model with no checkpoint tier is not a miss: the capability is absent.
		layer.Status = AuxStatusUnsupported
		layer.Reason = string(res.Reason)
	default:
		layer.Status = AuxStatusCold
		layer.Reason = string(res.Reason)
	}
	return layer
}

// warmEngramLayer runs the V4.1 Engram-prefix warm. It is unsupported on a non-V4.1
// runtime or when the caller supplied no prefix; a warm that reads nothing (an already
// resident prefix) is an honest ready-with-zero-read result, distinguished from a miss by
// the delta's own Requested/Resident accounting.
func warmEngramLayer(adapter auxWarmAdapter, isV41 bool, cfg AuxWarmConfig) AuxLayerReceipt {
	if !isV41 {
		return AuxLayerReceipt{Status: AuxStatusUnsupported, Reason: "not_v41"}
	}
	if adapter == nil {
		return AuxLayerReceipt{Status: AuxStatusUnsupported, Reason: "no_model"}
	}
	if len(cfg.EngramTokens) == 0 {
		return AuxLayerReceipt{Status: AuxStatusSkipped, Reason: "no_prefix"}
	}
	delta, err := adapter.WarmV41EngramPrefix(cfg.EngramTokens, cfg.EngramMask)
	if err != nil {
		// The model refused the warm (no V4.1 config, no Engram layers, an unwired or
		// inconsistent stage): an explicit unsupported layer, never a silent zero.
		return AuxLayerReceipt{Status: AuxStatusUnsupported, Reason: "engram_unsupported"}
	}
	layer := AuxLayerReceipt{
		Requested: int(delta.Requested),
		Retained:  int(delta.Requested - delta.Misses),
		BytesRead: delta.BytesRead,
	}
	if delta.Requested > 0 && delta.Misses == 0 {
		layer.Status = AuxStatusReady
		return layer
	}
	if delta.Requested > 0 {
		layer.Status = AuxStatusReady
		layer.Reason = "partial_residency"
		return layer
	}
	layer.Status = AuxStatusCold
	layer.Reason = "no_rows"
	return layer
}

// auxWarmAdapter returns the seam the orchestrator calls: the injected test adapter when
// set, else the planner's own model (the real adapter retained for real requests). A nil
// model yields a nil adapter, which every layer reports as unsupported.
func (p *InKernelPlanner) auxWarmAdapter() auxWarmAdapter {
	if p.auxAdapter != nil {
		return p.auxAdapter
	}
	if p.m == nil {
		return nil
	}
	return p.m
}

// auxWarmConfigSnapshot returns the configured auxiliary warming plus whether any is
// configured. It reads under mu alongside every other planner-owned warm field.
func (p *InKernelPlanner) auxWarmConfigSnapshot() (AuxWarmConfig, bool) {
	p.mu.Lock()
	defer p.mu.Unlock()
	if !p.auxWarmSet {
		return AuxWarmConfig{}, false
	}
	return p.auxWarm, true
}

// SetAuxWarmConfig installs the optional auxiliary-cache warming configuration. It copies
// every slice so the caller cannot mutate the planner's config after the fact. A planner
// that never calls it warms only KV and leaves WarmReceipt.Aux nil.
func (p *InKernelPlanner) SetAuxWarmConfig(cfg AuxWarmConfig) {
	if p == nil {
		return
	}
	stored := cfg
	stored.EngramTokens = append([]int(nil), cfg.EngramTokens...)
	stored.EngramMask = append([]bool(nil), cfg.EngramMask...)
	stored.ExpertProfile.Entries = append([]model.ExpertWarmDemandEntry(nil), cfg.ExpertProfile.Entries...)
	p.mu.Lock()
	p.auxWarm = stored
	p.auxWarmSet = true
	p.mu.Unlock()
}

// warmIsV41 reports whether this planner's runtime is the DeepSeek V4.1 family, whose
// complete-snapshot continuation admission is owned by a later, unlanded leaf
// (fak#13342/#13338/#13334). The check reads the model config's own V4.1 identity, never
// a caller flag, so it cannot be bypassed by a mis-set option.
func (p *InKernelPlanner) warmIsV41() bool {
	return p != nil && p.m != nil && p.m.Cfg.IsDeepSeekV41()
}

// warmStableTokens re-encodes the descriptor's stable source into the exact token
// boundary the descriptor reported. It deliberately does NOT re-run an AGENTS scanner or
// re-derive instructions: it re-renders the SAME stable messages the descriptor used, so
// the sequence is a pure function of the descriptor's own content.
//
// The descriptor is bounded and text-free by contract, so it does not carry the input
// bytes. The caller therefore installs the same WarmPrefixInputs it derived from via
// SetWarmPrefixInputs at the startup seam, where both are in hand. A planner with no
// carrier refuses closed rather than guessing the boundary from prompt text.
func (p *InKernelPlanner) warmStableTokens(ctx context.Context, inputs *warmInputsCarrier) ([]int, error) {
	if inputs == nil {
		return nil, ErrWarmPrefixUnsupported
	}
	enc, err := p.EncodePrompt(ctx, inputs.stableMessages(), inputs.Tools)
	if err != nil {
		return nil, err
	}
	return enc.TokenIDs, nil
}

// warmRestoreReadback walks the same scoped/unscoped prefix cache the next demand turn
// consults and returns (matched, tier, ok). ok is false when the matched entry owns no
// reusable payload (a structural match or a nil-KV intermediate), which is exactly the
// "MatchLen alone is insufficient" case the issue names.
func (p *InKernelPlanner) warmRestoreReadback(scope radixkv.CacheIdentity, tokens []int) (int, radixkv.SnapshotTier, bool) {
	if p == nil || len(tokens) == 0 {
		return 0, radixkv.SnapshotTierMiss, false
	}
	// The scoped tree wraps p.mu as its own locker, so it must be called WITHOUT holding
	// p.mu or the lookup self-deadlocks. Snapshot both references under the lock, release,
	// then probe; the scoped tree re-acquires p.mu internally (the same contract the
	// production generateReusedContextWithBias path relies on).
	p.mu.Lock()
	scopedTree := p.scopedTree
	tree := p.tree
	p.mu.Unlock()
	if scope.Tenant != "" && scopedTree != nil {
		return p.warmScopedReadback(scopedTree, scope, tokens)
	}
	if tree == nil {
		return 0, radixkv.SnapshotTierMiss, false
	}
	node, snap, matched, tier, err := tree.LookupSnapshotTiered(tokens)
	if err != nil {
		return 0, radixkv.SnapshotTierMiss, false
	}
	if node != nil {
		tree.Done(node)
	}
	// A real restore requires a live payload: the tiered snapshot when the runtime keeps
	// one, else the node's own KV cache. A match with neither is a structural hit only.
	if snap != nil {
		return matched, tier, true
	}
	if node != nil && node.KV() != nil {
		return matched, tier, true
	}
	if matched > 0 {
		return matched, radixkv.SnapshotTierMiss, false
	}
	return matched, tier, false
}

// warmScopedReadback performs the readback through the private scoped tree, requiring a
// real payload exactly as the unscoped path does. It must be called WITHOUT holding p.mu:
// the scoped tree wraps p.mu as its locker, so it acquires it internally.
func (p *InKernelPlanner) warmScopedReadback(scopedTree *radixkv.ScopedTree, scope radixkv.CacheIdentity, tokens []int) (int, radixkv.SnapshotTier, bool) {
	snap, _, matched, _, tier, err := scopedTree.LookupSnapshotTiered(scope, tokens)
	if err != nil {
		return 0, radixkv.SnapshotTierMiss, false
	}
	if snap != nil {
		return matched, tier, true
	}
	// Scoped trees may hold a bare KVCache (AdmitPrivate) rather than a snapshot; probe
	// that path too so a private warm is not misread as a miss when a payload is present.
	kv, _, matchedKV, _, err := scopedTree.Lookup(scope, tokens)
	if err != nil {
		return matched, radixkv.SnapshotTierMiss, false
	}
	if kv != nil {
		return matchedKV, tier, true
	}
	if matched > 0 || matchedKV > 0 {
		best := matched
		if matchedKV > best {
			best = matchedKV
		}
		return best, radixkv.SnapshotTierMiss, false
	}
	return matched, tier, false
}

// finishWarm stamps the elapsed prime duration and returns the receipt. Centralising it
// keeps every exit path timing-consistent.
func finishWarm(r WarmReceipt, started time.Time) WarmReceipt {
	r.PrimeDuration = time.Since(started)
	return r
}

// warmInputsCarrier carries the stable inputs a descriptor was derived from, so the warm
// can re-encode the exact stable boundary without re-reading prompt text. It is installed
// on the planner at the startup seam (planner.warmInputs) where descriptor + inputs are
// both in hand; a planner with no carrier refuses the warm closed.
type warmInputsCarrier struct {
	Instructions []byte
	SystemBlocks [][]byte
	Tools        []ToolDef
}

// stableMessages reconstructs the stable-only message list the descriptor rendered from.
// The order matches DeriveWarmPrefix exactly (instructions, then resident blocks), so the
// re-encoded boundary is the descriptor's boundary.
func (w *warmInputsCarrier) stableMessages() []Message {
	msgs := make([]Message, 0, len(w.SystemBlocks)+1)
	if len(w.Instructions) > 0 {
		msgs = append(msgs, Message{Role: RoleSystem, Content: string(w.Instructions)})
	}
	for _, block := range w.SystemBlocks {
		if len(block) == 0 {
			continue
		}
		msgs = append(msgs, Message{Role: RoleSystem, Content: string(block)})
	}
	return msgs
}

// SetWarmPrefixInputs installs the stable inputs on the planner so a subsequent WarmPrefix
// can re-encode the descriptor's stable boundary. It copies every slice so the caller
// cannot mutate the planner's carrier after the fact.
func (p *InKernelPlanner) SetWarmPrefixInputs(in WarmPrefixInputs) {
	if p == nil {
		return
	}
	c := &warmInputsCarrier{
		Instructions: append([]byte(nil), in.Instructions...),
		Tools:        append([]ToolDef(nil), in.Tools...),
	}
	for _, block := range in.SystemBlocks {
		c.SystemBlocks = append(c.SystemBlocks, append([]byte(nil), block...))
	}
	p.mu.Lock()
	p.warmInputs = c
	p.mu.Unlock()
}
