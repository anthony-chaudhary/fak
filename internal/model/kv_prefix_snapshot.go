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
	mu         sync.Mutex
	owner      *Session
	epoch      uint64
	Cache      *KVCache
	halKV      compute.KVStore
	halLineage tokenLineage
	qwen35     *qwen35HALState
	Backend    compute.Backend
	Tokens     int
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
	return p.v41DeviceIdentity.Name() == v41QualifiedBackendName
}

// v41QualifiedBackendName is the exact backend identity whose complete V4.1
// continuation contract is witnessed in-tree (fak#13337). It is a single
// explicit name rather than a capability inference: "owns a KVStore" is not
// evidence that a backend owns the V4.1 shared/publication state, so admission
// is by named qualification and every other backend stays refused until it is
// witnessed on its own.
const v41QualifiedBackendName = "cpu-ref"

// v41ForwardSnapshot is the deep-owned copy of a session's V4.1 continuation
// state. It mirrors exactly the mutable continuation fields the arithmetic
// reads -- token history, the per-layer temporal attention states, and the
// shared source-publication registry -- and NEVER the expertGateUp callback,
// which is a per-session device binding rather than continuation data, and
// never any weight table. Ownership is deep: every row is copied, so mutating
// or closing one branch cannot alter another (fak#13342).
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

// PrefixSnapshot captures an independently owned prefix. Qwen recurrent layers
// share immutable device pairs until a branch first mutates that layer. On a
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
	out := &PrefixSnapshot{owner: s, epoch: s.cacheGeometryEpoch, Cache: s.Cache.Clone(), halLineage: s.halLineage.clone(0), Backend: s.Backend, Tokens: tokens, DenseGPULayers: s.DenseGPULayers, GPULayers: s.GPULayers, ExecutionPolicy: s.executionPolicy}
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
	out.v41 = captureV41ForwardSnapshot(s.v41Forward)
	// A V4.1 prefix is not portable across backends and its position authority is
	// the committed history, not the generic cache. Record both at capture so
	// Restore can fail closed on a mismatched backend or token count instead of
	// installing state the target arithmetic cannot own (fak#13337). The identity
	// and count are recorded only when there IS V4.1 state to describe; a bare or
	// non-V4.1 snapshot keeps the absent contract and is unaffected.
	if out.v41 != nil && out.v41.hadState {
		out.v41Tokens = len(out.v41.history)
		out.hasV41Tokens = true
		if s.Backend != nil {
			out.v41DeviceIdentity = s.Backend
			out.hasV41DeviceIdentity = true
		}
	}
	if s.Backend == nil && !readResidentGDN {
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
				out.Close()
				return nil, fmt.Errorf("model: prefix snapshot cannot read resident GDN layer %d (%s): %w", layer, stage, err)
			}
			writeQwen35GDNSnapshotsToHost(s.M.Cfg, out.Cache, snapshots)
		} else if s.qwen35HAL != nil && s.qwen35HAL.sequenceFailure != nil {
			out.Close()
			return nil, fmt.Errorf("model: cannot snapshot prefix after resident GDN failure: %w", s.qwen35HAL.sequenceFailure)
		}
		return out, nil
	}
	if s.halKV == nil {
		return nil, fmt.Errorf("model: backend session has no device KV store")
	}
	out.halKV = s.halKV.Clone()
	var err error
	out.qwen35, err = cloneQwen35HALState(s.qwen35HAL, s.Backend)
	if err != nil {
		out.Close()
		return nil, err
	}
	return out, nil
}

// Clone makes a second independent owner for lookup; the cache retains the original.
func (p *PrefixSnapshot) Clone() (*PrefixSnapshot, error) {
	if p == nil {
		return nil, nil
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	started := prefixProfileStart()
	defer func() { emitPrefixProfile(started, "device_clone", "complete", p, nil) }()
	if p.Cache == nil {
		return nil, nil
	}
	out := &PrefixSnapshot{
		owner: p.owner, epoch: p.epoch, Cache: p.Cache.Clone(), halLineage: p.halLineage.clone(0), Backend: p.Backend, Tokens: p.Tokens, ExecutionPolicy: p.ExecutionPolicy,
		DenseGPULayers: p.DenseGPULayers, GPULayers: p.GPULayers,
		captureTargetHidden: p.captureTargetHidden,
		targetHidden:        cloneTargetHidden(p.targetHidden),
		targetHiddenTokens:  append([]int(nil), p.targetHiddenTokens...),
		v41:                 p.v41.clone(),
		// A backend identity is copied by reference (it names the device the state
		// belongs to, never device memory this snapshot owns), so a clone keeps the
		// same backend-prefix contract as its source.
		v41DeviceIdentity:    p.v41DeviceIdentity,
		hasV41DeviceIdentity: p.hasV41DeviceIdentity,
		v41Tokens:            p.v41Tokens,
		hasV41Tokens:         p.hasV41Tokens,
	}
	if p.Backend == nil {
		return out, nil
	}
	if p.halKV == nil {
		return nil, fmt.Errorf("model: device prefix snapshot has no KV store")
	}
	out.halKV = p.halKV.Clone()
	var err error
	out.qwen35, err = cloneQwen35HALState(p.qwen35, p.Backend)
	if err != nil {
		out.Close()
		return nil, err
	}
	return out, nil
}

// Restore installs this snapshot into a fresh backend session and transfers ownership.
func (p *PrefixSnapshot) Restore(s *Session) error {
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
		if s.halKV != nil {
			s.halKV.Free()
		}
		s.closeQwen35HALState()
		s.halKV, s.halLineage, s.qwen35HAL = p.halKV, p.halLineage, p.qwen35
		p.halKV, p.qwen35 = nil, nil
		p.halLineage = tokenLineage{}
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
			s.v41State()
		} else {
			s.v41Forward = nil
		}
		p.v41 = nil
	}
	return nil
}

// Close releases all device ownership held by a cache node or failed clone.
func (p *PrefixSnapshot) Close() {
	if p == nil {
		return
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.halKV != nil {
		p.halKV.Free()
		p.halKV = nil
	}
	if p.qwen35 != nil {
		p.qwen35.free(p.Backend)
		p.qwen35 = nil
	}
	p.Cache = nil
	p.targetHidden, p.targetHiddenTokens = nil, nil
	p.v41 = nil
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
func captureV41ForwardSnapshot(st *v41ForwardState) *v41ForwardSnapshot {
	if st == nil {
		return nil
	}
	out := &v41ForwardSnapshot{
		history:  append([]int(nil), st.history...),
		attn:     st.attn.clone(),
		hadState: true,
	}
	if len(st.layers) > 0 {
		out.layers = make([]*V41AttentionState, len(st.layers))
		for i, layer := range st.layers {
			out.layers[i] = layer.clone()
		}
	}
	return out
}

// clone makes a second deep-owned copy, so two branches of one snapshot never
// alias mutable rows. A nil snapshot clones to nil.
func (n *v41ForwardSnapshot) clone() *v41ForwardSnapshot {
	if n == nil {
		return nil
	}
	out := &v41ForwardSnapshot{
		history:  append([]int(nil), n.history...),
		attn:     n.attn.clone(),
		hadState: n.hadState,
	}
	if len(n.layers) > 0 {
		out.layers = make([]*V41AttentionState, len(n.layers))
		for i, layer := range n.layers {
			out.layers[i] = layer.clone()
		}
	}
	return out
}

// restore builds a fresh v41ForwardState that takes ownership of the snapshot's
// deep-owned rows. It never aliases the snapshot: the history is copied and the
// layer slice is handed over, then the snapshot's references are cleared by the
// caller. Session callbacks are left nil for the destination session to bind
// through Session.v41State, eagerly in PrefixSnapshot.Restore or on first use.
func (n *v41ForwardSnapshot) restore() *v41ForwardState {
	if n == nil {
		return nil
	}
	out := &v41ForwardState{
		history: append([]int(nil), n.history...),
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
	out.kvPublications = cloneV41PublicationRows(s.kvPublications)
	out.indexPublications = cloneV41PublicationRows(s.indexPublications)
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
