package model

import (
	"fmt"
	"sync"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// SessionFromPrefix starts a session whose cache is a clone of an already-computed
// prefix, so only the suffix needs prefilling (real prefix reuse). For
// GLM-MoE-DSA the clone carries the DSA attention/index cache instead of the dense
// GQA K/V rows.
//
// It refuses, by name, an architecture whose session state is NOT the KVCache (#5548). A
// recompute session carries its prefix as a token history and leaves the cache empty, so
// this clone would hand back a session that has ingested nothing while its caller believes
// it holds the prefix — and then prefills only the suffix. Nothing downstream can catch
// that: an empty cache is a well-formed zero-length prefix at every consumer. The refusal
// is loud for the same reason requireGemma4Session's is: a wrong answer with no error is
// the failure this whole family of guards exists to prevent.
func (m *Model) SessionFromPrefix(prefix *KVCache) *Session {
	if !m.Cfg.KVPrefixReuseSupported() {
		panic("model: SessionFromPrefix is not available for architecture " + m.Cfg.archFamilyKey() +
			": its session state is the token history, not the KV cache, so a cache clone carries no prefix")
	}
	return &Session{M: m, Cache: prefix.Clone()}
}

// Session drives generation over a kernel-owned KV cache. Prefill ingests a prompt;
// Step decodes one token. Both share the exact per-token math the verified full
// forward pass uses, so cached decode is provably identical to full prefill.
// GLM-MoE-DSA uses a separate DSA attention/index cache carried inside KVCache.
//
// Naming: this is the token-decoder sense of "session" (a generator over a KV
// cache), NOT the drive-state session. The canonical "session" — run-state,
// budget, priority, pace — is internal/session.Table / session.State; the wire
// DTO of that drive state is gateway.SessionState. See the vocabulary worklist at
// docs/notes/VOCAB-DISAMBIGUATION-WORKLIST-2026-06-24.md.
// PrefixSnapshot is an independently owned inference prefix. For legacy/host
// sessions Cache is sufficient. Device hybrid sessions additionally carry attention
// KV and recurrent Qwen state; keeping all three in one owner prevents partial restores.
type PrefixSnapshot struct {
	mu     sync.Mutex
	closed bool // Protected by mu; ownership detached, cleanup may still be running.
	owner  *Session
	epoch  uint64
	Cache  *KVCache
	halKV  compute.KVStore
	// halKVPrecision is the device tier halKV holds, restored onto the session with it.
	halKVPrecision compute.KVPrecision
	halLineage     tokenLineage
	qwen35         *qwen35HALState
	Backend        compute.Backend
	Tokens         int
	// DenseGPULayers / GPULayers preserve layer placement across prefix snapshots.
	DenseGPULayers  int
	GPULayers       int
	ExecutionPolicy ExecutionPolicy
	// Native MTP consumes the exact pre-final-norm residual history. It is part of
	// session state just as surely as KV: restoring one without the other can make
	// a rejected draft visible to the next proposal even though attention rolled back.
	captureTargetHidden bool
	targetHidden        [][]float32
	targetHiddenTokens  []int
	// v41 captures the DeepSeek V4.1 session-owned continuation state (the
	// v41ForwardState committed token history plus each layer's bounded temporal
	// V41AttentionState) delivered by #13305/#13313. The generic Cache alone does
	// not carry it, so restoring a V4.1 prefix from Cache is incomplete; this
	// field is what makes the clone a COMPLETE V4.1 prefix (fak#13342).
	v41 *v41ForwardSnapshot
	// coverage describes the logits saved beside this exact prefix.
	// It is cache provenance, not arithmetic continuation or newly executed work.
	// Capture and clone own this value independently of every live session.
	coverage *NativeForwardCoverage
	// v41DeviceIdentity records the exact backend this snapshot's V4.1 state was
	// captured on. It is the snapshot-side half of the backend-prefix contract:
	// a V4.1 continuation restored onto a DIFFERENT backend has no continuity
	// guarantee, so a mismatch is refused rather than silently reused. It is
	// meaningful only when hasV41DeviceIdentity is true; a host-only session
	// (Backend == nil) records the absent contract rather than inventing an
	// identity for a backend it never ran on (fak#13337).
	v41DeviceIdentity    compute.Backend
	hasV41DeviceIdentity bool
	// v41Tokens is the exact position authority -- the committed token count --
	// this snapshot's V4.1 continuation state represents. V4.1 keeps its position
	// authority in the committed history rather than in the generic KVCache, so a
	// restore must refuse a target whose resident length disagrees with the
	// carried history instead of installing a state the arithmetic will index
	// past. hasV41Tokens distinguishes a snapshot that carries no V4.1 state
	// (absent contract) from one whose authority is a legitimate zero.
	v41Tokens    int
	hasV41Tokens bool
}

// v41BackendSnapshotSupported reports whether the exact backend this snapshot
// was captured on owns every byte of a complete V4.1 continuation prefix. It is
// the snapshot-side, backend-aware counterpart of
// Config.InKernelBackendPrefixReuseSupported and is deliberately fail-closed:
//
//   - a host-only capture (no device identity) is refused -- a backend prefix
//     must name the backend it was prepared on;
//   - a backend whose name is empty is refused -- an unnamed device cannot be
//     compared, so the identity cannot be trusted;
//   - a V4.1 backend is admitted ONLY when it is an explicit member of the
//     qualified set. The qualified set is the one backend identity witnessed to
//     own the complete V4.1 continuation contract in-tree (the cpu-ref counting
//     fixture); a different or unknown backend must be qualified on its own
//     before reuse, so the default answer is refuse (fak#13337).
//
// It never consults the generic Config capability, and it does not by itself
// enable serving reuse: the consuming planner owns that (fak#13330).
func (p *PrefixSnapshot) v41BackendSnapshotSupported() bool {
	if p == nil || !p.hasV41DeviceIdentity || p.v41DeviceIdentity == nil {
		return false
	}
	if p.v41DeviceIdentity.Name() == "" {
		return false
	}
	return v41QualifiedBackend(p.v41DeviceIdentity.Name())
}

// v41QualifiedBackend admits the exact backend identities whose complete V4.1
// continuation contract is witnessed (fak#13337). Admission is by explicit name
// rather than capability inference: "owns a KVStore" is not evidence that a
// backend owns the V4.1 shared/publication state, so every other backend stays
// refused until it is witnessed on its own. cpu-ref is witnessed by
// TestV41BackendPrefixSnapshot; vulkan by TestV41BackendPrefixSnapshotVulkan on
// the physical device, whose resident KV mirror and bindings rebuild from the
// restored host continuation state (fak-private#3432).
func v41QualifiedBackend(name string) bool {
	return name == "cpu-ref" || name == "vulkan"
}

// v41ForwardSnapshot owns a session's writable V4.1 continuation state. It
// mirrors exactly the mutable continuation fields the arithmetic
// reads -- token history, the per-layer temporal attention states, and the
// shared source-publication registry -- and NEVER the expertGateUp callback,
// which is a per-session device binding rather than continuation data, and
// never any weight table. Capture and cache-restore forks may retain already
// sealed immutable publications; every writable continuation field remains
// branch-owned. Ordinary Clone still deep-copies every row (fak#13342).
type v41ForwardSnapshot struct {
	history []int
	attn    *V41AttentionState
	layers  []*V41AttentionState
	// hadState records whether the owner session held a live v41ForwardState at
	// capture. A snapshot of a session that never ran a V4.1 forward carries the
	// nil-state contract, so restoring it leaves the target session's V4.1 state
	// absent rather than inventing an empty-but-present one.
	hadState bool
}

// PrefixSnapshot captures an independently owned prefix. V4.1 captures retain
// already sealed immutable publications and copy every writable row. Qwen recurrent
// layers share immutable device pairs until a branch first mutates that layer. On a
// backend-nil Qwen session whose resident GDN owners hold the live recurrent
// state, the owners are read back into the snapshot's host cache.
func (s *Session) PrefixSnapshot() (*PrefixSnapshot, error) {
	return s.prefixSnapshot(true)
}

// PrefixSnapshotHostOnly captures the prefix exactly as PrefixSnapshot did before
// resident GDN owner readback existed: the host Cache.linear is cloned as-is and
// live resident owners are neither read nor consulted. It is for callers that
// checkpoint and restore GDN state through their own transaction state (the P4
// MTP speculative-round checkpoint), where a per-round device readback of every
// owner would be pure overhead. Such a snapshot is NOT a publishable prefix of
// an owner-backed session; use PrefixSnapshot for anything restored elsewhere.
func (s *Session) PrefixSnapshotHostOnly() (*PrefixSnapshot, error) {
	return s.prefixSnapshot(false)
}

func (s *Session) prefixSnapshot(readResidentGDN bool) (*PrefixSnapshot, error) {
	return acquireCompletePrefixSnapshot(s.Backend, func(scope *backendExecutionScope) (*PrefixSnapshot, error) {
		scope.bind(s)
		return s.prefixSnapshotOwned(scope, readResidentGDN)
	})
}

// A returned candidate stays owned here until execution-scope retirement also
// returns normally. Failure cleanup runs after that scope releases its mutex.
func acquireCompletePrefixSnapshot(backend compute.Backend, body func(*backendExecutionScope) (*PrefixSnapshot, error)) (out *PrefixSnapshot, err error) {
	var candidate *PrefixSnapshot
	published := false
	defer func() {
		if !published && candidate != nil {
			suppressBackendCleanupPanic(candidate.Close)
		}
	}()
	func() {
		scope := beginBackendExecution(backend)
		complete := false
		defer scope.finishCompleted(&complete)
		candidate, err = body(scope)
		complete = err == nil
	}()
	if err != nil {
		return nil, err
	}
	published = true
	return candidate, nil
}

func (s *Session) prefixSnapshotOwned(scope *backendExecutionScope, readResidentGDN bool) (*PrefixSnapshot, error) {
	// A failed submission can leave host state ahead of executable backend
	// state. Refuse it before cloning any cache or recurrent publication.
	if s != nil && s.halClosed {
		if s.halFailure != nil {
			return nil, s.halFailure
		}
		return nil, fmt.Errorf("model: cannot snapshot closed backend session")
	}
	var out *PrefixSnapshot
	published := false
	// Registered before source locks, so cleanup never invokes a backend while
	// the source geometry or recurrent mutex is held.
	defer func() {
		if !published && out != nil {
			suppressBackendCleanupPanic(func() { out.closeOwned(scope) })
		}
	}()
	if s != nil && s.M.v41EPRequested() {
		return nil, fmt.Errorf("%w: distributed prefix snapshot is not supported", ErrV41ExpertParallel)
	}
	s.cacheGeometryMu.RLock()
	defer s.cacheGeometryMu.RUnlock()
	if s == nil || s.Cache == nil {
		return nil, fmt.Errorf("model: cannot snapshot nil session cache")
	}
	tokens := s.Cache.Len()
	if s.Backend != nil && s.halKV != nil {
		// HAL attention KV is the physical position authority. Hybrid device paths
		// may keep the host model cache positionless while all prefix state lives in
		// halKV plus recurrent tensors.
		tokens = s.halKV.Len()
	}
	out = &PrefixSnapshot{owner: s, epoch: s.cacheGeometryEpoch, Cache: s.Cache.Clone(), halLineage: s.halLineage.clone(0), Backend: s.Backend, Tokens: tokens, DenseGPULayers: s.DenseGPULayers, GPULayers: s.GPULayers, ExecutionPolicy: s.executionPolicy}
	s.targetHiddenMu.RLock()
	out.captureTargetHidden = s.captureTargetHidden
	out.targetHidden = cloneTargetHidden(s.targetHidden)
	out.targetHiddenTokens = append([]int(nil), s.targetHiddenTokens...)
	s.targetHiddenMu.RUnlock()
	// V4.1 session state is part of the prefix just as surely as KV: the generic
	// Cache does not carry the token history or the per-layer temporal states the
	// assembly reads, so a snapshot that copied Cache alone would restore a
	// session that has ingested nothing while its caller believes it holds the
	// prefix (#13342, same failure family as #5548). Capture it whenever the
	// session has one; a session that never ran a V4.1 forward keeps the absent
	// contract.
	out.v41 = captureV41ForwardSnapshotWithPublications(s.v41Forward, true)
	if out.v41 == nil && s.M != nil && s.M.Cfg.IsDeepSeekV41() {
		// Preserve absence explicitly: Restore must clear any target continuation,
		// rather than replacing generic KV while leaving stale V4.1 history live.
		out.v41 = &v41ForwardSnapshot{}
	}
	if s.v41Forward != nil && s.v41Forward.forwardCoverage != nil {
		coverage := *s.v41Forward.forwardCoverage
		out.coverage = &coverage
	}
	// A V4.1 prefix is not portable across backends and its position authority is
	// the committed history, not the generic cache. Record both at capture so
	// Restore can fail closed on a mismatched backend or token count instead of
	// installing state the target arithmetic cannot own (fak#13337). The identity
	// and count are recorded only when there IS V4.1 state to describe; a bare or
	// non-V4.1 snapshot keeps the absent contract and is unaffected.
	if out.v41 != nil && out.v41.hadState {
		out.v41Tokens = len(out.v41.history)
		out.hasV41Tokens = true
		// V4.1 advances its committed history, not the generic HAL KV store.
		// Consumers validate this complete snapshot's length before admitting
		// generated continuations, so publish the same position authority here.
		out.Tokens = out.v41Tokens
		if s.Backend != nil {
			out.v41DeviceIdentity = s.Backend
			out.hasV41DeviceIdentity = true
		}
	}
	if s.Backend == nil && !readResidentGDN {
		published = true
		return out, nil
	}
	if s.Backend == nil {
		// A backend-nil Qwen session whose resident GDN owners (admitted sequence
		// or promoted decode) hold the live recurrent state never advanced its host
		// Cache.linear while they ran. Read the owners into the CLONE so the
		// snapshot restores the true prefix state into any host session; the live
		// owners stay authoritative and unchanged, so a following prefill or decode
		// on this session continues from the same state. Fail closed rather than
		// publish a stale prefix.
		if s.qwen35GDNOwnersHoldLiveState() {
			snapshots, layer, stage, err := s.snapshotQwen35GDNOwners()
			if err != nil {
				return nil, fmt.Errorf("model: prefix snapshot cannot read resident GDN layer %d (%s): %w", layer, stage, err)
			}
			writeQwen35GDNSnapshotsToHost(s.M.Cfg, out.Cache, snapshots)
		} else if s.qwen35HAL != nil && s.qwen35HAL.sequenceFailure != nil {
			return nil, fmt.Errorf("model: cannot snapshot prefix after resident GDN failure: %w", s.qwen35HAL.sequenceFailure)
		}
		published = true
		return out, nil
	}
	if s.halKV == nil {
		return nil, fmt.Errorf("model: backend session has no device KV store")
	}
	out.halKV, out.halKVPrecision = s.halKV.Clone(), s.halKVPrecision
	var err error
	out.qwen35, err = cloneQwen35HALState(s.qwen35HAL, s.Backend)
	if err != nil {
		return nil, err
	}
	published = true
	return out, nil
}

// PrepareForCache seals independently owned V4.1 publication rows for reuse.
// The cache admission owner calls this before publishing the snapshot. It must
// serialize admission/accounting as it does insertion; cache hits never seal.
// Other continuation fields and malformed publication stores stay unchanged.
func (p *PrefixSnapshot) PrepareForCache() {
	if p == nil {
		return
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.Cache == nil || p.v41 == nil || !p.v41.hadState {
		return
	}
	p.v41.attn.sealPublications()
	for _, layer := range p.v41.layers {
		layer.sealPublications()
	}
}

// Clone makes a fully materialized, independent writable owner. Sealed V4.1
// publication rows are copied just like ordinary mutable publication rows.
func (p *PrefixSnapshot) Clone() (out *PrefixSnapshot, err error) {
	if p == nil {
		return nil, nil
	}
	return acquireCompletePrefixSnapshot(p.Backend, p.cloneOwned)
}

// CloneForRestore keeps sealed V4.1 publication rows immutable and shared while
// privately cloning every mutable continuation field. Unprepared rows retain
// Clone's deep-copy behavior. The result is consumed by Restore as usual.
func (p *PrefixSnapshot) CloneForRestore() (out *PrefixSnapshot, err error) {
	if p == nil {
		return nil, nil
	}
	return acquireCompletePrefixSnapshot(p.Backend, func(scope *backendExecutionScope) (*PrefixSnapshot, error) {
		return p.cloneOwnedWithPublications(scope, true)
	})
}

func (p *PrefixSnapshot) cloneOwned(scope *backendExecutionScope) (*PrefixSnapshot, error) {
	return p.cloneOwnedWithPublications(scope, false)
}

func (p *PrefixSnapshot) cloneOwnedWithPublications(scope *backendExecutionScope, shareSealed bool) (*PrefixSnapshot, error) {
	var out *PrefixSnapshot
	published := false
	defer func() {
		if !published && out != nil {
			suppressBackendCleanupPanic(func() { out.closeOwned(scope) })
		}
	}()
	if p == nil {
		return nil, nil
	}
	err := func() error {
		p.mu.Lock()
		defer p.mu.Unlock()
		started := prefixProfileStart()
		defer func() { emitPrefixProfile(started, "device_clone", "complete", p, nil) }()
		if p.Cache == nil {
			return nil
		}
		out = &PrefixSnapshot{
			owner: p.owner, epoch: p.epoch, Cache: p.Cache.Clone(), halLineage: p.halLineage.clone(0), Backend: p.Backend, Tokens: p.Tokens, ExecutionPolicy: p.ExecutionPolicy,
			DenseGPULayers: p.DenseGPULayers, GPULayers: p.GPULayers,
			captureTargetHidden: p.captureTargetHidden,
			targetHidden:        cloneTargetHidden(p.targetHidden),
			targetHiddenTokens:  append([]int(nil), p.targetHiddenTokens...),
			v41:                 p.v41.cloneWithPublications(shareSealed),
			// A backend identity is copied by reference (it names the device the state
			// belongs to, never device memory this snapshot owns), so a clone keeps the
			// same backend-prefix contract as its source.
			v41DeviceIdentity:    p.v41DeviceIdentity,
			hasV41DeviceIdentity: p.hasV41DeviceIdentity,
			v41Tokens:            p.v41Tokens,
			hasV41Tokens:         p.hasV41Tokens,
		}
		if p.coverage != nil {
			coverage := *p.coverage
			out.coverage = &coverage
		}
		if p.Backend == nil {
			return nil
		}
		if p.halKV == nil {
			return fmt.Errorf("model: device prefix snapshot has no KV store")
		}
		out.halKV, out.halKVPrecision = p.halKV.Clone(), p.halKVPrecision
		var err error
		out.qwen35, err = cloneQwen35HALState(p.qwen35, p.Backend)
		if err != nil {
			return err
		}
		return nil
	}()
	if err != nil {
		return nil, err
	}
	// The source lock and deferred profile callback have both completed.
	published = true
	return out, nil
}

// Restore installs this snapshot into a fresh backend session and transfers ownership.
func (p *PrefixSnapshot) Restore(s *Session) (err error) {
	if p == nil {
		return fmt.Errorf("model: invalid prefix snapshot restore")
	}
	scope := beginBackendExecution(p.Backend)
	defer scope.finishError(&err)
	return p.restoreOwned(scope, s)
}

func (p *PrefixSnapshot) restoreOwned(scope *backendExecutionScope, s *Session) error {
	if s != nil && s.M.v41EPRequested() {
		return fmt.Errorf("%w: distributed prefix restore is not supported", ErrV41ExpertParallel)
	}
	if p == nil {
		return fmt.Errorf("model: invalid prefix snapshot restore")
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.owner != nil {
		p.owner.cacheGeometryMu.RLock()
		defer p.owner.cacheGeometryMu.RUnlock()
	}
	if p.owner != nil && p.epoch != p.owner.cacheGeometryEpoch {
		return fmt.Errorf("model: stale prefix snapshot after cache rebuild")
	}
	if s == nil || p.Cache == nil {
		return fmt.Errorf("model: invalid prefix snapshot restore")
	}
	if p.Backend != s.Backend {
		return fmt.Errorf("model: prefix snapshot backend mismatch")
	}
	// State-only V4.1 prefill deliberately has no logits coverage receipt.
	// Its retained attention state still belongs to the model that produced it;
	// missing coverage must not authorize reuse across same-shape artifacts.
	if p.coverage != nil || p.v41 != nil && p.v41.hadState {
		if p.owner == nil || p.owner.M == nil || p.owner.M != s.M {
			return fmt.Errorf("%w: prefix snapshot model identity mismatch", ErrNativeForwardCoverageInvalid)
		}
	}
	if p.coverage != nil {
		coverage := p.coverage
		if err := coverage.Validate(); err != nil {
			return err
		}
		if coverage.ModelType != s.M.Cfg.ModelType || coverage.ConfiguredLayers != s.M.Cfg.NumLayers || coverage.HeadDim != s.M.Cfg.HeadDim {
			return fmt.Errorf("%w: prefix snapshot model geometry mismatch", ErrNativeForwardCoverageInvalid)
		}
	}
	scope.bind(s)
	// A V4.1 continuation is a backend-session prefix, not a portable host blob:
	// the arithmetic reads state bound to the device it was prepared on, so a
	// DEVICE capture that names no device, names an unqualified device, or names a
	// different device than the target is refused atomically -- the snapshot must
	// survive so a valid target can still consume it (fak#13337). The check is
	// scoped to snapshots that actually carry device-backed V4.1 state, so a
	// host-only or non-V4.1 prefix keeps its prior restore behavior byte-for-byte.
	deviceBackedV41 := p.v41 != nil && p.v41.hadState && p.hasV41DeviceIdentity
	if deviceBackedV41 {
		if !p.v41BackendSnapshotSupported() {
			return fmt.Errorf("model: V4.1 backend prefix snapshot has no qualified device identity")
		}
		if s.Backend == nil {
			return fmt.Errorf("model: V4.1 backend prefix snapshot requires a device session")
		}
		if p.v41DeviceIdentity != s.Backend {
			return fmt.Errorf("model: V4.1 prefix snapshot device mismatch")
		}
	}
	if p.v41 != nil && p.v41.hadState && p.hasV41Tokens {
		// A target that has never committed V4.1 positions is a legitimate
		// restore destination -- Restore replaces its continuation state. Only a
		// target that HAS a committed history is required to agree with the
		// snapshot's, so a diverged branch is refused rather than silently
		// re-based.
		if s.v41Forward != nil && len(s.v41Forward.history) != p.v41Tokens {
			return fmt.Errorf("model: V4.1 prefix snapshot token authority mismatch: committed=%d snapshot=%d", len(s.v41Forward.history), p.v41Tokens)
		}
	}
	if s.Backend != nil {
		if p.halKV == nil {
			return fmt.Errorf("model: device prefix snapshot missing KV")
		}
		// Detach attempted releases before callbacks. A Free may complete and
		// then panic or Goexit; recovery must never retry that old handle.
		oldKV, oldRecurrent := s.halKV, s.qwen35HAL
		s.halKV, s.qwen35HAL = nil, nil
		s.qwen35MetalStateIdentity = nil
		runBackendSiblingCleanup(
			func() {
				if oldKV != nil {
					oldKV.Free()
				}
			},
			func() {
				if oldRecurrent != nil {
					oldRecurrent.free(s.Backend)
				}
			},
		)
		// The source still owns both replacements until all old cleanup returns.
		s.halKV, s.halLineage, s.qwen35HAL = p.halKV, p.halLineage, p.qwen35
		s.halKVPrecision = p.halKVPrecision
		p.halKV, p.qwen35 = nil, nil
		p.halLineage = tokenLineage{}
		// A target built by NewBackendSessionCheckedDeferredState carries no eager
		// zero recurrent state (fak#13801). A hybrid snapshot always carries its own,
		// so this only fires for a hybrid model whose snapshot has none: materialize
		// the historical zero state rather than leaving the session unable to forward.
		if s.qwen35HAL == nil && s.qwen35StateDeferred && s.M != nil && s.M.Cfg.IsQwen35Hybrid() {
			s.initQwen35HALState(s.Backend.(Qwen35GDNBackend))
		}
	}
	if s.DenseGPULayers == 0 && p.DenseGPULayers != 0 {
		s.DenseGPULayers = p.DenseGPULayers
	}
	if s.GPULayers == 0 && p.GPULayers != 0 {
		s.GPULayers = p.GPULayers
	}
	if p.ExecutionPolicy == ExecutionPolicyDeviceOnly {
		s.executionPolicy = p.ExecutionPolicy
	}
	s.Cache = p.Cache
	p.Cache = nil
	p.halLineage = tokenLineage{}
	s.targetHiddenMu.Lock()
	s.captureTargetHidden = p.captureTargetHidden
	s.targetHidden, s.targetHiddenTokens = p.targetHidden, p.targetHiddenTokens
	s.targetHiddenMu.Unlock()
	p.targetHidden, p.targetHiddenTokens = nil, nil
	// Install the complete V4.1 continuation state and transfer ownership. A
	// snapshot captured from a session with no V4.1 state restores the absent
	// contract (nil), so the target never appears to hold a prefix it does not.
	if p.v41 != nil {
		if p.v41.hadState {
			s.v41Forward = p.v41.restore()
			s.v41Forward.forwardCoverage = p.coverage
			s.v41State()
		} else {
			s.v41Forward = nil
		}
		p.v41 = nil
	}
	p.coverage = nil
	return nil
}

// Close releases all device ownership held by a cache node or failed clone.
// Repeated calls after ownership is detached need not wait for cleanup to finish.
func (p *PrefixSnapshot) Close() {
	if p == nil {
		return
	}
	// A terminal call has no backend work. Release mu before acquiring the
	// domain so a live caller keeps the domain-then-mu cleanup order, while a
	// cleanup callback can re-enter Close without reacquiring its own domain.
	p.mu.Lock()
	closed := p.closed
	p.mu.Unlock()
	if closed {
		return
	}
	scope := beginBackendExecution(p.Backend)
	complete := false
	defer scope.finishCompleted(&complete)
	p.closeOwned(scope)
	complete = true
}

func (p *PrefixSnapshot) closeOwned(scope *backendExecutionScope) {
	if p == nil {
		return
	}
	p.mu.Lock()
	// Detach ownership and publish terminal state before callbacks. Recursive
	// public Close then needs neither this call's scope nor its backend domain.
	// The caller's execution scope, when present, still owns physical cleanup.
	kv, recurrent, backend := p.halKV, p.qwen35, p.Backend
	p.halKV, p.qwen35 = nil, nil
	p.halLineage = tokenLineage{}
	p.Cache = nil
	p.targetHidden, p.targetHiddenTokens = nil, nil
	p.v41, p.coverage = nil, nil
	p.closed = true
	p.mu.Unlock()
	runBackendSiblingCleanup(
		func() {
			if kv != nil {
				kv.Free()
			}
		},
		func() {
			if recurrent != nil {
				recurrent.free(backend)
			}
		},
	)
}

// snapshotResidentPositions reports a restore target's resident position count
// from its authoritative store: the device KV store on a HAL session, the host
// KVCache otherwise. It mirrors PrefixSnapshot's own Tokens resolution so the
// token-authority guard compares like with like.
func snapshotResidentPositions(s *Session) int {
	if s == nil {
		return 0
	}
	if s.Backend != nil && s.halKV != nil {
		return s.halKV.Len()
	}
	if s.Cache == nil {
		return 0
	}
	return s.Cache.Len()
}

// captureV41ForwardSnapshot deep-copies a session's V4.1 continuation state, or
// returns nil when the session never ran a V4.1 forward. The per-layer temporal
// states and the shared source-publication registry are copied row-by-row so the
// snapshot owns them independently of the session; the expertGateUp callback is
// deliberately NOT captured (it is a per-session device binding, not
// continuation data), matching the step-local run state the assembly builds.
// Last-logits coverage is also excluded: failed attempts clear it while restoring
// arithmetic continuation. PrefixSnapshot owns cached-logit provenance separately.
func captureV41ForwardSnapshot(st *v41ForwardState) *v41ForwardSnapshot {
	return captureV41ForwardSnapshotWithPublications(st, false)
}

// captureV41ForwardSnapshotWithPublications keeps an existing sealed base when
// requested, without sealing or changing the live session. Mutable publication
// overlays and all other continuation state remain independently owned.
func captureV41ForwardSnapshotWithPublications(st *v41ForwardState, shareSealed bool) *v41ForwardSnapshot {
	if st == nil {
		return nil
	}
	out := &v41ForwardSnapshot{
		history:  append([]int(nil), st.history...),
		attn:     st.attn.cloneWithPublications(shareSealed),
		hadState: true,
	}
	if len(st.layers) > 0 {
		out.layers = make([]*V41AttentionState, len(st.layers))
		for i, layer := range st.layers {
			out.layers[i] = layer.cloneWithPublications(shareSealed)
		}
	}
	return out
}

// clone makes a second deep-owned copy, so two branches of one snapshot never
// alias mutable rows. A nil snapshot clones to nil.
func (n *v41ForwardSnapshot) clone() *v41ForwardSnapshot {
	return n.cloneWithPublications(false)
}

func (n *v41ForwardSnapshot) cloneWithPublications(shareSealed bool) *v41ForwardSnapshot {
	if n == nil {
		return nil
	}
	out := &v41ForwardSnapshot{
		history:  append([]int(nil), n.history...),
		attn:     n.attn.cloneWithPublications(shareSealed),
		hadState: n.hadState,
	}
	if len(n.layers) > 0 {
		out.layers = make([]*V41AttentionState, len(n.layers))
		for i, layer := range n.layers {
			out.layers[i] = layer.cloneWithPublications(shareSealed)
		}
	}
	return out
}

// restore builds a fresh v41ForwardState that takes ownership of the snapshot's
// writable rows and any sealed base. Capture and clone isolate history from other
// owners, so transfer it with the attention state rather than copying the full
// prefix again. Clear all transferred references before returning. Session
// callbacks are left nil for the destination session to bind
// through Session.v41State, eagerly in PrefixSnapshot.Restore or on first use.
func (n *v41ForwardSnapshot) restore() *v41ForwardState {
	if n == nil {
		return nil
	}
	out := &v41ForwardState{
		history: n.history,
		attn:    n.attn,
		layers:  n.layers,
	}
	n.history, n.attn, n.layers = nil, nil, nil
	return out
}

// clone deep-copies one temporal attention state, including the window ring, the
// incomplete compressor group and positions, the shared compressed/index
// publications and their ranges, and the latest candidate/top-k selections. It
// returns nil for nil so an absent per-layer state stays absent.
func (s *V41AttentionState) clone() *V41AttentionState {
	return s.cloneWithPublications(false)
}

func (s *V41AttentionState) cloneWithPublications(shareSealed bool) *V41AttentionState {
	if s == nil {
		return nil
	}
	out := &V41AttentionState{
		windowSize:         s.windowSize,
		headDim:            s.headDim,
		indexHeadDim:       s.indexHeadDim,
		ratioCap:           s.ratioCap,
		nextWindowPos:      s.nextWindowPos,
		nextCompressRow:    s.nextCompressRow,
		retainedWindowRows: s.retainedWindowRows,
		retainedCopies:     s.retainedCopies,
		kvPublishedEnd:     make(map[int]int, len(s.kvPublishedEnd)),
		indexPublishedEnd:  make(map[int]int, len(s.indexPublishedEnd)),
		candidatesSet:      s.candidatesSet,
		candidateRatio:     s.candidateRatio,
		topkSet:            s.topkSet,
		topkRatio:          s.topkRatio,
	}
	out.window = cloneV41Rows(s.window)
	out.partialKV = cloneV41Rows(s.partialKV)
	out.partialInputs = cloneV41Rows(s.partialInputs)
	out.partialPositions = append([]int(nil), s.partialPositions...)
	if shareSealed || s.publicationPrefix == nil {
		out.publicationPrefix = s.publicationPrefix
		out.kvPublications = cloneV41PublicationRows(s.kvPublications)
		out.indexPublications = cloneV41PublicationRows(s.indexPublications)
	} else {
		out.kvPublications = cloneV41PublicationView(s.publicationPrefix.kv, s.kvPublications)
		out.indexPublications = cloneV41PublicationView(s.publicationPrefix.index, s.indexPublications)
	}
	for k, v := range s.kvPublishedEnd {
		out.kvPublishedEnd[k] = v
	}
	for k, v := range s.indexPublishedEnd {
		out.indexPublishedEnd[k] = v
	}
	if s.candidates != nil {
		out.candidates = append([]bool(nil), s.candidates...)
	}
	if s.topk != nil {
		out.topk = make([][]int32, len(s.topk))
		for i, row := range s.topk {
			out.topk[i] = append([]int32(nil), row...)
		}
	}
	return out
}

func cloneV41Rows(in [][]float32) [][]float32 {
	if in == nil {
		return nil
	}
	out := make([][]float32, len(in))
	for i, row := range in {
		out[i] = append([]float32(nil), row...)
	}
	return out
}

func cloneV41PublicationRows(in map[v41AttentionPublicationKey][]float32) map[v41AttentionPublicationKey][]float32 {
	if in == nil {
		return nil
	}
	out := make(map[v41AttentionPublicationKey][]float32, len(in))
	for k, row := range in {
		out[k] = append([]float32(nil), row...)
	}
	return out
}

func cloneTargetHidden(in [][]float32) [][]float32 {
	out := make([][]float32, len(in))
	for i := range in {
		out[i] = append([]float32(nil), in[i]...)
	}
	return out
}
