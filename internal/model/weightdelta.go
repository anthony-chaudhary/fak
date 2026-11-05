package model

// weightdelta.go — WeightDelta: typed, CAS-named, reversible weight-state mutation
// applied hot against a live loaded Model (continual-engine §4 P2, phase 1).
//
// Weights move through one funnel: ModelWeightSource adapts a Model to
// compute.WeightSource (Weight(name, want Dtype)). A WeightDeltaSource sits INSIDE
// that funnel: it serves patched tensors under UNCHANGED names, so the forward loop
// never learns deltas exist. The first delta kind is LoRA-as-delta — a re-expression
// of the proven LoRA merge path (LoRASet.MergeInto) with zero base mutation: the
// base f32 blob is never touched, the patched copy is materialized off-band and
// swapped in atomically, so a round trip through the delta layer is bit-identical to
// the existing LoRA merge path by construction (same math, same order, same inputs).
//
// Weight-epoch accounting: every committed transition advances a monotonically
// increasing generation (never reused or rewound) and recomputes the state's content
// digest (CAS identity: a delta is identified by its result digest, and re-applying
// an already-active delta is idempotent). KV spans stamp the epoch they were computed
// under (kvmmu.Segment); each delta declares its kv-compat impact, and serving a span
// across an epoch_bump boundary fails closed — the never-silent-recompute doctrine
// (internal/engine/cacheevents.go SilentRecompute; the engine package cites it, it
// does not own this seam).

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"math"
	"sort"
	"sync"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// ---- Delta kinds -----------------------------------------------------------------

// WeightDeltaKind names a class of weight-state mutation. Kinds are a closed set:
// an unknown kind is refused with ErrUnknownWeightDeltaKind, never guessed at.
type WeightDeltaKind uint8

const (
	// WeightDeltaLoRAAdapterAdd composes additive LoRA adapters into the current
	// weight state (ΔW = Σ scaling·B·A per target, the proven merge math). Zero base
	// mutation: the base f32 bytes stay untouched and the patched copy is served from
	// the delta layer.
	WeightDeltaLoRAAdapterAdd WeightDeltaKind = iota
)

func (k WeightDeltaKind) String() string {
	switch k {
	case WeightDeltaLoRAAdapterAdd:
		return "lora_adapter_add"
	default:
		return fmt.Sprintf("unknown_weight_delta_kind(%d)", uint8(k))
	}
}

// WeightDeltaKVCompat declares how a delta affects KV spans computed under earlier
// weight epochs. Compatible means the prefix cache survives (representation-only
// changes such as quant re-packs); EpochBump means KV under the old epoch must be
// evicted or lazily recomputed — silently serving it is exactly the failure this
// declaration exists to prevent.
type WeightDeltaKVCompat uint8

const (
	// KVCompatCompatible: spans stamped under prior epochs remain servable.
	KVCompatCompatible WeightDeltaKVCompat = iota
	// KVCompatEpochBump: spans stamped under prior epochs must not be served.
	KVCompatEpochBump
)

func (c WeightDeltaKVCompat) String() string {
	switch c {
	case KVCompatCompatible:
		return "compatible"
	case KVCompatEpochBump:
		return "epoch_bump"
	default:
		return fmt.Sprintf("unknown_kv_compat(%d)", uint8(c))
	}
}

// WeightDelta is one typed, CAS-named mutation against a live weight state.
type WeightDelta struct {
	Kind       WeightDeltaKind     // closed set; unknown kinds fail with ErrUnknownWeightDeltaKind
	BaseDigest string              // CAS guard: the state digest the delta was computed against
	Adapters   []*LoRAAdapter      // KindLoRAAdapterAdd: factors composed additively, in add order
	KVCompat   WeightDeltaKVCompat // declared impact on KV spans stamped under earlier epochs
}

// WeightDeltaReceipt is the per-apply witness: a delta without a receipt did not happen.
type WeightDeltaReceipt struct {
	Kind         WeightDeltaKind // meaningful for applies; zero for reverts
	BaseDigest   string          // state digest before the transition
	ResultDigest string          // state digest after the transition (the delta's CAS identity)
	KVCompat     WeightDeltaKVCompat
	MovedBytes   int64 // materialized patched-tensor bytes resident after the transition
	EpochFrom    uint64
	EpochTo      uint64
	NoOp         bool // true when the delta was already active (idempotent re-apply)
	Reverted     bool // true when the transition restored the last-known-good state
}

// WeightEpochStamp identifies a weight state for KV-span accounting: the monotonic
// generation plus the content digest of the state a span was computed under.
type WeightEpochStamp struct {
	Generation uint64
	Digest     string
}

// Sentinels: errors.Is-assertable, closed vocabulary (anti-brittle: callers match
// these, never error prose).
var (
	// ErrUnknownWeightDeltaKind: the delta's kind is outside the closed set.
	ErrUnknownWeightDeltaKind = errors.New("model: unknown weight-delta kind")
	// ErrEmptyWeightDelta: the delta carries no adapters to apply.
	ErrEmptyWeightDelta = errors.New("model: weight delta carries no adapters")
	// ErrWeightDeltaBaseMismatch: the CAS guard failed — the delta was computed
	// against a different weight state than the live one. Refused, never merged.
	ErrWeightDeltaBaseMismatch = errors.New("model: weight-delta base digest does not match the live weight state")
	// ErrWeightDeltaTargetNotResident: a delta target is absent from the f32 manifest
	// or not 2-D [out,in] — merging needs the f32 weight.
	ErrWeightDeltaTargetNotResident = errors.New("model: weight-delta target is not an f32-resident 2-D weight")
	// ErrWeightDeltaRevertEmpty: Revert with an empty active chain.
	ErrWeightDeltaRevertEmpty = errors.New("model: nothing to revert: weight-state history is empty")
	// ErrPatchedWeightQuantStoreRefused: a patched name requested from a quantized
	// store the delta layer cannot patch (Q4_K/Q4_0/AWQ/...): refusing beats serving
	// the unpatched quantized bytes silently.
	ErrPatchedWeightQuantStoreRefused = errors.New("model: patched weight cannot be served from a quantized store (delta layer serves f32 and Q8_0 only)")
)

// ---- State machine ---------------------------------------------------------------

// WeightDeltaSource fronts a Model's weight funnel with typed, reversible deltas.
// It wraps ModelWeightSource (the Weight(name, want Dtype) funnel) and serves
// patched tensors under unchanged names; every other name passes through to the base
// funnel untouched. The live state is the ACTIVE DELTA CHAIN: a deduplicated,
// ordered list of committed deltas, and the patched tensors are always re-derived
// from the PRISTINE base plus the whole chain (never incrementally from a prior
// patched state), so revert is exact and the state is a pure function of the chain.
// Construction pins the base-inventory digest; Apply/Revert advance the monotonic
// generation and recompute the state digest. All exported methods are safe for
// concurrent use.
type WeightDeltaSource struct {
	m     *Model
	be    compute.Backend
	base  *ModelWeightSource
	invFP string // base-inventory fingerprint: the state-digest seed

	mu      sync.RWMutex
	epoch   uint64                // monotonic generation; never reused or rewound
	digest  string                // content identity of the current state
	active  []WeightDelta         // committed chain, in apply order (deduplicated by payload)
	patched map[string][]float32  // materialized patched f32 per target name (nil when none)
	decls   []WeightDeltaKVCompat // decls[i] = the declaration of the transition that produced epoch i+1
	q8cache map[string]*q8Tensor  // memoized on-demand Q8_0 of patched f32; cleared on every transition
}

// NewWeightDeltaSource wraps the model's weight funnel (be, or the registered Default
// reference backend when nil — the same rule Model.WeightSource applies) with an
// empty delta layer at generation 0. The initial digest is the base-state digest:
// the stable identity of the un-patched state.
func NewWeightDeltaSource(m *Model, be compute.Backend) *WeightDeltaSource {
	if be == nil {
		be = compute.Default()
	}
	invFP := baseInventoryFingerprint(m)
	return &WeightDeltaSource{
		m:      m,
		be:     be,
		base:   m.WeightSource(be),
		invFP:  invFP,
		digest: baseStateDigest(invFP),
	}
}

// baseInventoryFingerprints the model's weight inventory (every tensor name + shape,
// sorted) into the state-digest seed. It anchors the CAS chain to WHICH checkpoint
// the state belongs to; geometry is additionally hard-validated per target at apply
// time, so a structurally identical checkpoint still applies cleanly (the operator
// chose the adapter for those values) while a different layout refuses at the CAS or
// geometry check.
func baseInventoryFingerprint(m *Model) string {
	names := make([]string, 0, len(m.manifest))
	for name := range m.manifest {
		names = append(names, name)
	}
	sort.Strings(names)
	h := sha256.New()
	for _, name := range names {
		io.WriteString(h, name)
		io.WriteString(h, "\x00")
		for _, d := range m.manifest[name].Shape {
			fmt.Fprintf(h, "%d,", d)
		}
		io.WriteString(h, "\n")
	}
	return hex.EncodeToString(h.Sum(nil))
}

// baseStateDigest is the content identity of the UN-patched state: the
// base-inventory fingerprint folded through one hash round (the zero-delta chain),
// so a span stamped before any delta layer is installed matches the generation-0
// digest of an installed layer exactly.
func baseStateDigest(invFP string) string {
	sum := sha256.Sum256([]byte(invFP))
	return hex.EncodeToString(sum[:])
}

// stateDigestOf computes the content identity of a weight state: the base-inventory
// fingerprint chained with every active delta's payload, in active order. Two states
// with the same active chain hash identically; popping a delta (revert) hashes back
// to the earlier value, which is what makes re-apply idempotence and revert
// restoration provable by equality.
func (s *WeightDeltaSource) stateDigestOf(active []WeightDelta) string {
	if len(active) == 0 {
		return baseStateDigest(s.invFP)
	}
	h := sha256.New()
	io.WriteString(h, s.invFP)
	for _, d := range active {
		h.Write([]byte{byte(d.Kind)})
		for _, a := range d.Adapters {
			if a == nil {
				continue
			}
			var buf [8]byte
			binary.LittleEndian.PutUint32(buf[:4], uint32(a.In))
			binary.LittleEndian.PutUint32(buf[4:], uint32(a.Out))
			h.Write(buf[:])
			binary.LittleEndian.PutUint32(buf[:4], uint32(a.Rank))
			h.Write(buf[:4])
			binary.LittleEndian.PutUint64(buf[:], math.Float64bits(a.Alpha))
			h.Write(buf[:])
			io.WriteString(h, a.Target)
			writeF32s(h, a.A)
			writeF32s(h, a.B)
		}
	}
	return hex.EncodeToString(h.Sum(nil))
}

// writeF32s writes a float32 slice into h as little-endian IEEE-754 bytes — the
// stable binary encoding the state digest hashes over.
func writeF32s(h io.Writer, v []float32) {
	for _, f := range v {
		var b [4]byte
		binary.LittleEndian.PutUint32(b[:], math.Float32bits(f))
		_, _ = h.Write(b[:])
	}
}

// f32sEqual reports bitwise equality of two float32 slices (bit patterns compared,
// so NaN payloads distinguish too).
func f32sEqual(a, b []float32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if math.Float32bits(a[i]) != math.Float32bits(b[i]) {
			return false
		}
	}
	return true
}

// weightDeltaSamePayload reports whether two deltas carry the same kind and the same
// factor payloads in the same order — the active-chain dedup key. Bundle names count:
// they are the unit of dynamic switching in the underlying LoRASet.
func weightDeltaSamePayload(a, b WeightDelta) bool {
	if a.Kind != b.Kind || len(a.Adapters) != len(b.Adapters) {
		return false
	}
	for i := range a.Adapters {
		x, y := a.Adapters[i], b.Adapters[i]
		if x == nil || y == nil {
			if x != y {
				return false
			}
			continue
		}
		if x.Target != y.Target || x.Adapter != y.Adapter || x.In != y.In || x.Out != y.Out || x.Rank != y.Rank || x.Alpha != y.Alpha {
			return false
		}
		if !f32sEqual(x.A, y.A) || !f32sEqual(x.B, y.B) {
			return false
		}
	}
	return true
}

// targetNames returns the distinct base-weight names a delta patches, sorted — the
// deterministic iteration order candidate materialization runs in.
func targetNames(d WeightDelta) []string {
	seen := map[string]bool{}
	var out []string
	for _, a := range d.Adapters {
		if a != nil && !seen[a.Target] {
			seen[a.Target] = true
			out = append(out, a.Target)
		}
	}
	sort.Strings(out)
	return out
}

// materialize derives the full patched state from the PRISTINE base plus the active
// delta chain: one fresh copy per target, merged with every active delta that touches
// it, in active order — the EXACT production merge math (LoRASet.MergeInto), so the
// result is bit-identical to the existing LoRA path by construction. Deriving from
// the base (never incrementally from a prior patched state) keeps revert exact and
// makes the state a pure function of the chain.
func (s *WeightDeltaSource) materialize(active []WeightDelta) (map[string][]float32, error) {
	targets := map[string]bool{}
	for _, d := range active {
		for _, a := range d.Adapters {
			if a != nil {
				targets[a.Target] = true
			}
		}
	}
	if len(targets) == 0 {
		return nil, nil
	}
	out := make(map[string][]float32, len(targets))
	for target := range targets {
		w := append([]float32(nil), s.m.tensor(target)...)
		set := NewLoRASet()
		for _, d := range active {
			for _, a := range d.Adapters {
				if a == nil || a.Target != target {
					continue
				}
				if err := set.Add(a); err != nil {
					return nil, err
				}
			}
		}
		if err := set.MergeInto(target, w); err != nil {
			return nil, err
		}
		out[target] = w
	}
	return out, nil
}

// movedBytes prices a state's materialized patched footprint: the bytes the delta
// layer holds beyond the base blob (the transfer side of a delta's true cost).
func movedBytes(patched map[string][]float32) int64 {
	var n int64
	for _, w := range patched {
		n += int64(len(w)) * 4
	}
	return n
}

// Epoch reports the current monotonic generation (0 until the first committed
// transition).
func (s *WeightDeltaSource) Epoch() uint64 {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.epoch
}

// Digest reports the content identity of the current weight state.
func (s *WeightDeltaSource) Digest() string {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.digest
}

// Stamp reports the weight-epoch pair KV spans record: the generation plus the
// content digest of the state they were computed under.
func (s *WeightDeltaSource) Stamp() WeightEpochStamp {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return WeightEpochStamp{Generation: s.epoch, Digest: s.digest}
}

// DeclarationsSince returns the kv-compat declarations of every transition committed
// AFTER the stamped generation, oldest first — the window a span-check consults. A
// stamped generation at or past the current one yields an empty window (the check
// then decides on digests alone, and fails closed on a rewind).
func (s *WeightDeltaSource) DeclarationsSince(stamped uint64) []WeightDeltaKVCompat {
	s.mu.RLock()
	defer s.mu.RUnlock()
	if stamped >= uint64(len(s.decls)) {
		return nil
	}
	return s.decls[stamped:]
}

// Apply validates and commits one delta against the live state, returning the
// receipt. The protocol is whole-old-or-whole-new: every validation (kind, CAS
// guard, target geometry) and the entire candidate materialization run BEFORE any
// state field changes; a failure leaves the state — weights, digest, epoch,
// declarations — exactly as it was. The commit is one locked section: swap the
// patched map, advance the generation, extend the active chain, append the
// declaration. Re-applying a delta that is already active is an idempotent no-op
// (CAS identity): same receipt, no generation bump, no chain growth.
func (s *WeightDeltaSource) Apply(d WeightDelta) (WeightDeltaReceipt, error) {
	if d.Kind != WeightDeltaLoRAAdapterAdd {
		return WeightDeltaReceipt{}, fmt.Errorf("%w: %v", ErrUnknownWeightDeltaKind, d.Kind)
	}
	if len(d.Adapters) == 0 {
		return WeightDeltaReceipt{}, ErrEmptyWeightDelta
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	// Idempotence (CAS identity) comes FIRST: a payload-identical delta already in
	// the active chain changes nothing — no generation bump, no chain growth — even
	// when the re-dispatch carries its original (now historical) BaseDigest. Its
	// effect is already in the state, so refusing it would punish a retry of an
	// applied delta while accepting a genuinely stale one below.
	for _, prev := range s.active {
		if weightDeltaSamePayload(prev, d) {
			return WeightDeltaReceipt{
				Kind: d.Kind, BaseDigest: s.digest, ResultDigest: s.digest,
				KVCompat: d.KVCompat, MovedBytes: movedBytes(s.patched),
				EpochFrom: s.epoch, EpochTo: s.epoch, NoOp: true,
			}, nil
		}
	}

	if d.BaseDigest != s.digest {
		return WeightDeltaReceipt{}, ErrWeightDeltaBaseMismatch
	}

	// Geometry: every adapter must target a 2-D f32-resident weight whose shape it
	// matches — the same hard check MergeLoRA runs, so a bad adapter is named here,
	// not as an out-of-range panic deep in a matvec.
	for _, a := range d.Adapters {
		if a == nil {
			return WeightDeltaReceipt{}, fmt.Errorf("%w: nil adapter", ErrWeightDeltaTargetNotResident)
		}
		meta, ok := s.m.manifest[a.Target]
		if !ok || len(meta.Shape) != 2 {
			return WeightDeltaReceipt{}, fmt.Errorf("%w: %q", ErrWeightDeltaTargetNotResident, a.Target)
		}
		if a.Out != meta.Shape[0] || a.In != meta.Shape[1] {
			return WeightDeltaReceipt{}, fmt.Errorf("model: lora %q adapter dims [out=%d,in=%d] != base %q [out=%d,in=%d]",
				a.Adapter, a.Out, a.In, a.Target, meta.Shape[0], meta.Shape[1])
		}
	}

	// Candidate materialization, off-band and complete: the whole state re-derived
	// from the pristine base plus the would-be active chain.
	nextActive := append(append([]WeightDelta{}, s.active...), d)
	candidate, err := s.materialize(nextActive)
	if err != nil {
		return WeightDeltaReceipt{}, err
	}
	resultDigest := s.stateDigestOf(nextActive)

	fromDigest := s.digest
	s.active = nextActive
	s.patched = candidate
	s.digest = resultDigest
	s.epoch++
	s.decls = append(s.decls, d.KVCompat)
	s.q8cache = nil
	return WeightDeltaReceipt{
		Kind: d.Kind, BaseDigest: fromDigest, ResultDigest: s.digest,
		KVCompat: d.KVCompat, MovedBytes: movedBytes(candidate),
		EpochFrom: s.epoch - 1, EpochTo: s.epoch,
	}, nil
}

// Revert restores the last-known-good state (the one before the most recent
// committed transition): both the weights AND their digest identity come back, so KV
// spans stamped under that state are servable again — while the generation still
// advances (monotonic: the counter records transitions, the digest records content).
// Returns ErrWeightDeltaRevertEmpty when the active chain is empty.
func (s *WeightDeltaSource) Revert() (WeightDeltaReceipt, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if len(s.active) == 0 {
		return WeightDeltaReceipt{}, ErrWeightDeltaRevertEmpty
	}
	// Materialize FIRST: a failure leaves the state exactly as it was.
	prevActive := s.active[:len(s.active)-1]
	candidate, err := s.materialize(prevActive)
	if err != nil {
		return WeightDeltaReceipt{}, err
	}
	fromDigest := s.digest
	s.active = prevActive
	s.patched = candidate
	s.digest = s.stateDigestOf(prevActive)
	s.epoch++
	s.decls = append(s.decls, KVCompatEpochBump) // a revert is a weight change: spans stamped between prev and now refuse
	s.q8cache = nil
	return WeightDeltaReceipt{
		BaseDigest: fromDigest, ResultDigest: s.digest,
		MovedBytes: movedBytes(candidate),
		EpochFrom:  s.epoch - 1, EpochTo: s.epoch, Reverted: true,
	}, nil
}

// Weight serves the named weight at the requested dtype through the delta layer:
// patched names come back as the materialized patched tensor (an f32 view, or an
// on-demand Q8_0 of it — the same quantize-on-request rule the base funnel applies
// to f32-resident weights), and every other name passes through to the base funnel
// untouched. A patched name requested from a quantized store the delta layer cannot
// patch (Q4_K/Q4_0/AWQ/...) is refused, never answered from the stale store.
func (s *WeightDeltaSource) Weight(name string, want compute.Dtype) (compute.Tensor, error) {
	switch want {
	case compute.F32, compute.F16, compute.BF16, compute.Q8_0:
	default:
		if s.isPatched(name) {
			return compute.Tensor{}, fmt.Errorf("%w: %s", ErrPatchedWeightQuantStoreRefused, name)
		}
		return s.base.Weight(name, want)
	}

	s.mu.RLock()
	p, ok := s.patched[name]
	var cached *q8Tensor
	if want == compute.Q8_0 && ok {
		cached = s.q8cache[name]
	}
	s.mu.RUnlock()

	if !ok {
		return s.base.Weight(name, want)
	}
	if want == compute.Q8_0 {
		if cached != nil {
			return compute.NewQ8(s.be, []int{cached.out, cached.in}, cached.q, cached.d, qBlk), nil
		}
		meta := s.m.manifest[name] // present: Apply validated it
		qt := quantizeQ8(p, meta.Shape[0], meta.Shape[1])
		s.mu.Lock()
		if s.q8cache == nil {
			s.q8cache = map[string]*q8Tensor{}
		}
		s.q8cache[name] = qt
		s.mu.Unlock()
		return compute.NewQ8(s.be, []int{meta.Shape[0], meta.Shape[1]}, qt.q, qt.d, qBlk), nil
	}
	meta := s.m.manifest[name]
	return compute.NewF32(s.be, append([]int(nil), meta.Shape...), p), nil
}

// isPatched reports whether the name has a materialized patched form.
func (s *WeightDeltaSource) isPatched(name string) bool {
	s.mu.RLock()
	defer s.mu.RUnlock()
	_, ok := s.patched[name]
	return ok
}

// static assertion: *WeightDeltaSource satisfies the compute.WeightSource seam — it
// IS the funnel the forward loop consumes, with deltas invisible behind it.
var _ compute.WeightSource = (*WeightDeltaSource)(nil)

// ---- Model / loader wiring -------------------------------------------------------

// SetWeightDeltaSource installs (or clears, with nil) the delta layer fronting this
// model's weight funnel. Mirrors SetLoRA: the loader stays the sole place weights
// are named; installing the delta layer is an explicit operator action.
func (m *Model) SetWeightDeltaSource(wds *WeightDeltaSource) {
	if m == nil {
		return
	}
	m.wds = wds
}

// WeightEpoch reports the model's current weight-state stamp: generation 0 with the
// base-state digest when no delta layer is installed, the live layer's stamp
// otherwise. The KV-MMU stamps it onto appended spans through the abi.KVBackend seam
// (kvBackend.WeightEpoch).
func (m *Model) WeightEpoch() WeightEpochStamp {
	if m == nil {
		return WeightEpochStamp{}
	}
	if m.wds != nil {
		return m.wds.Stamp()
	}
	return WeightEpochStamp{Digest: baseStateDigest(baseInventoryFingerprint(m))}
}

// LoadLoRADelta loads a PEFT adapter directory through the proven AddLoRAAdapter
// path and packages its factors as a ready-to-apply WeightDelta: the CAS guard is
// the baseDigest the caller read from the live state, and kvCompat declares the
// delta's impact on earlier-epoch KV spans. This is the "LoRA-as-first-delta" entry
// point: no new loading machinery, the existing loader re-expressed as a delta.
func LoadLoRADelta(dir, baseDigest string, kvCompat WeightDeltaKVCompat) (WeightDelta, error) {
	set := NewLoRASet()
	if err := AddLoRAAdapter(set, dir); err != nil {
		return WeightDelta{}, err
	}
	out := WeightDelta{Kind: WeightDeltaLoRAAdapterAdd, BaseDigest: baseDigest, KVCompat: kvCompat}
	for _, target := range set.Targets() {
		for _, a := range set.activeFor(target) {
			out.Adapters = append(out.Adapters, a)
		}
	}
	if len(out.Adapters) == 0 {
		return WeightDelta{}, ErrEmptyWeightDelta
	}
	return out, nil
}
