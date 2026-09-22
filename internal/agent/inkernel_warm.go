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
	// Flip the first-demand latch ONLY now: a real restore was observed, so the following
	// turn's prompt is genuinely eligible to reuse this prefix.
	p.noteKVPrefixAdmitted()
	return finishWarm(receipt, started), nil
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
