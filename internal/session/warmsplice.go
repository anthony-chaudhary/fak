package session

// warmsplice.go — the CONCRETE warm-KV mover behind the resume loop (issue #916, epic
// #912 "one machine"). resume.go owns the BLOCK-on-Paused / re-admit-at-boundary half and
// the abstract WarmKVSplicer seam; this file is the half that performs the real splice the
// issue names: it clones the paused session's attention state (model.KVCache.Clone) and
// drives the cachemeta lifecycle's promote (cachemeta.Lifecycle.MoveTo, which emits a
// cachemeta.KVRestore directive) so a resumed turn REUSES warm KV instead of cold
// re-prefilling.
//
// WHY THIS IS THE BUILDABLE SPLICE.
// A Paused session is a preempted sequence whose KV was OFFLOADED to a colder tier (HBM ->
// DRAM/CXL) while it was held. Resuming it warm is exactly a PROMOTE back to the hot tier:
//
//   - the BYTES are reattached by KVCache.Clone — an exact deep copy, so the resumed
//     session is bit-identical to one that re-prefilled the prefix (the Clone doc's "spliced
//     into the next session, skipping its prefill entirely" guarantee), and
//   - the cachemeta LIFECYCLE records the move: MoveTo(TierHBM) on a span resident in a
//     colder tier returns KVRestore (TierRank(HBM) < TierRank(DRAM)), the same vocabulary a
//     live-engine residency adapter lowers into engine.CacheEvent.
//
// It deliberately stays in-lane: it imports internal/model READ-ONLY (KVCache.Clone) and
// internal/cachemeta (Lifecycle / tiers). It performs no GPU work and writes no bytes off
// the clone — the physical device transfer is the host engine's job; this records the WARM
// decision and reattaches the kernel-owned cache so the resume loop can report ResumeWarm.
//
// DEGRADE-SAFE. A trace with no parked warm KV (it was evicted while paused, or the session
// never offloaded one) splices nothing and the seam reports cold — resume.go then falls back
// to today's cold re-prefill. Correctness never depends on the warm path: the worst case is
// slower, never wrong.

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"sync"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/cachemeta"
	"github.com/anthony-chaudhary/fak/internal/model"
)

// WarmKV is one paused session's offloaded attention state, parked for a warm resume. The
// Cache is the kernel-owned KVCache the session held when it was preempted; ColdTier is the
// tier it was offloaded to while paused (the tier MoveTo promotes it FROM). A session offloads
// its KV here at pause and reclaims it on resume; if it is evicted while paused, the entry is
// dropped and the resume degrades to cold.
const KindKVSpan = "kv_span"

type KVSpanPointer struct {
	Kind string `json:"kind,omitempty"`
	Ref  string `json:"ref,omitempty"`
}

func (p KVSpanPointer) IsZero() bool { return p.Kind == "" && p.Ref == "" }

type WarmKV struct {
	Cache      *model.KVCache
	ColdTier   cachemeta.ResidencyTier
	SpanDigest string
	Residency  abi.KVResidency
}

// SpliceResult is the typed record of one warm-KV splice. Warm is true exactly when a parked
// cache was found and reattached; Restored is the cloned cache the resumed turn attends (nil
// on a cold miss); Direction is the cachemeta transfer the promote emitted (KVRestore on a
// warm splice); RestoredPositions is the reattached span length (KVCache.Len). It is the
// auditable witness that the splice ran — a test (and an observability sink) reads it to
// prove the resumed turn reused warm KV instead of re-prefilling.
type SpliceResult struct {
	Warm              bool
	Restored          *model.KVCache
	Direction         cachemeta.KVTransferDirection
	FromTier          cachemeta.ResidencyTier
	ToTier            cachemeta.ResidencyTier
	RestoredPositions int
	SpanPointer       KVSpanPointer
	Residency         abi.KVResidency
}

// WarmKVStore parks the offloaded KV of paused sessions and performs the concrete warm splice
// on resume. It is the host-side object the gateway constructs once and wires into a Table via
// Splicer(): Park(trace, kv) at pause, and the returned WarmKVSplicer reattaches it on the
// Paused->Running edge. Safe for concurrent use (a gateway pauses/resumes many sessions).
//
// HotTier is the tier a resume promotes warm KV back TO (default TierHBM — device memory, the
// hottest tier a served decode attends from). profiles is the tier characteristics map MoveTo
// consults to land the restored span Resident (default cachemeta.DefaultTierProfiles).
type WarmKVStore struct {
	mu       sync.Mutex
	parked   map[string]WarmKV
	HotTier  cachemeta.ResidencyTier
	profiles map[cachemeta.ResidencyTier]cachemeta.TierProfile
	last     map[string]SpliceResult
	backend  abi.KVBackend
}

// NewWarmKVStore builds an empty store promoting to TierHBM with the default tier profiles.
func NewWarmKVStore() *WarmKVStore {
	return &WarmKVStore{
		parked:   map[string]WarmKV{},
		HotTier:  cachemeta.TierHBM,
		profiles: cachemeta.DefaultTierProfiles(),
		last:     map[string]SpliceResult{},
	}
}

// NewWarmKVStoreWithBackend injects the residency backend used across relaunch.
func NewWarmKVStoreWithBackend(backend abi.KVBackend) *WarmKVStore {
	s := NewWarmKVStore()
	s.backend = backend
	return s
}

// CarrySpan installs a durable pointer in a fresh store after process relaunch.
func (s *WarmKVStore) CarrySpan(trace string, pointer KVSpanPointer) {
	if s == nil || pointer.Kind != KindKVSpan || pointer.Ref == "" {
		return
	}
	s.mu.Lock()
	if s.parked == nil {
		s.parked = map[string]WarmKV{}
	}
	s.parked[trace] = WarmKV{ColdTier: cachemeta.TierDRAM, SpanDigest: pointer.Ref}
	s.mu.Unlock()
}

// Park records a paused session's offloaded KV under its trace, to be reattached warm on
// resume. coldTier is the tier the KV was offloaded to while held (the promote source). A nil
// cache or a nil store is a no-op (the resume then degrades to cold). Calling Park again for a
// trace replaces the prior parked cache.
func (s *WarmKVStore) Park(trace string, cache *model.KVCache, coldTier cachemeta.ResidencyTier) {
	if s == nil || cache == nil {
		return
	}
	if coldTier == "" {
		coldTier = cachemeta.TierDRAM
	}
	s.mu.Lock()
	if s.parked == nil {
		s.parked = map[string]WarmKV{}
	}
	parked := cache.Clone()
	digest := warmKVSpanDigest(parked)
	entry := WarmKV{Cache: parked, ColdTier: coldTier, SpanDigest: digest}
	backend := s.backend
	s.mu.Unlock()
	if backend != nil && digest != "" {
		residency, err := backend.StageSpan(context.Background(), digest, 0, parked.Len())
		if err == nil {
			entry.Residency = residency
		}
	}
	s.mu.Lock()
	s.parked[trace] = entry
	s.mu.Unlock()
}

// Evict drops a trace's parked warm KV — the "evicted while paused" path. After an Evict the
// trace's next resume finds no warm cache and degrades to cold. A no-op for an unknown trace.
func (s *WarmKVStore) Evict(trace string) {
	if s == nil {
		return
	}
	s.mu.Lock()
	delete(s.parked, trace)
	s.mu.Unlock()
}

// Splice reattaches a trace's parked warm KV: it CLONES the cache (an exact deep copy, so the
// resumed turn is bit-identical to a re-prefill) and drives the cachemeta lifecycle promote
// (MoveTo(HotTier)), which emits KVRestore because the hot tier outranks the cold tier the KV
// was parked at. It returns a SpliceResult witnessing the move. A trace with no parked cache
// returns a cold (Warm=false) result and reattaches nothing — the resume loop then falls back
// to cold re-prefill. The parked entry is consumed on a warm splice (a resume reclaims it
// once); a host that wants to keep it re-Parks at the next pause.
func (s *WarmKVStore) Splice(trace string) SpliceResult {
	if s == nil {
		return SpliceResult{}
	}
	s.mu.Lock()
	warm, ok := s.parked[trace]
	if !ok {
		s.mu.Unlock()
		return SpliceResult{}
	}
	delete(s.parked, trace)
	profiles := s.profiles
	hot := s.HotTier
	backend := s.backend
	s.mu.Unlock()

	pointer := KVSpanPointer{Kind: KindKVSpan, Ref: warm.SpanDigest}
	if warm.Cache == nil {
		if backend == nil || warm.SpanDigest == "" {
			return SpliceResult{}
		}
		residency, err := backend.RestoreSpan(context.Background(), warm.SpanDigest)
		if err != nil || residency.Outcome != abi.KVResidencyOK {
			return SpliceResult{SpanPointer: pointer, Residency: residency}
		}
		return SpliceResult{Warm: true, SpanPointer: pointer, Residency: residency}
	}

	if profiles == nil {
		profiles = cachemeta.DefaultTierProfiles()
	}
	if hot == "" {
		hot = cachemeta.TierHBM
	}

	// Reattach the BYTES: an exact deep copy of the offloaded cache. The resumed turn attends
	// this clone, skipping the re-prefill entirely (KVCache.Clone's reuse guarantee).
	restored := warm.Cache.Clone()

	// Record the MOVE on the cachemeta lifecycle: a span resident in the cold tier promoted to
	// the hot tier. MoveTo to a hotter tier (TierRank(hot) < TierRank(cold)) returns KVRestore.
	lc := cachemeta.NewLifecycle(warm.ColdTier, 0)
	lc = lc.MarkResident(profiles, 0)
	_, dir := lc.MoveTo(hot, profiles, 1)

	res := SpliceResult{
		Warm:              true,
		Restored:          restored,
		Direction:         dir,
		FromTier:          warm.ColdTier,
		ToTier:            hot,
		RestoredPositions: restored.Len(),
		SpanPointer:       pointer,
		Residency:         warm.Residency,
	}
	s.mu.Lock()
	if s.last == nil {
		s.last = map[string]SpliceResult{}
	}
	s.last[trace] = res
	s.mu.Unlock()
	return res
}

// LastSplice returns the most recent SpliceResult recorded for a trace and whether one exists.
// It is the observability read a supervisor / test uses to confirm a resume reused warm KV
// (Warm && Direction == cachemeta.KVRestore) rather than re-prefilling cold.
func (s *WarmKVStore) LastSplice(trace string) (SpliceResult, bool) {
	if s == nil {
		return SpliceResult{}, false
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	res, ok := s.last[trace]
	return res, ok
}

// Splicer adapts the store into the WarmKVSplicer the Table consults on a Paused->Running
// edge: it splices the resuming session's trace and reports true iff warm KV was reattached
// (so resume.go returns ResumeWarm). A store with no parked cache for the trace reports false
// and the resume degrades to cold. Wire it with table.WatchResumeSplice(store.Splicer()).
func (s *WarmKVStore) Splicer() WarmKVSplicer {
	return func(st State) SpliceResult { return s.Splice(st.TraceID) }
}

func warmKVSpanDigest(kv *model.KVCache) string {
	if kv == nil {
		return ""
	}
	h := sha256.New()
	_, _ = h.Write([]byte("fak-warm-kv-span\x00"))
	if span, err := kv.SerializeSpan(0, kv.Len()); err == nil {
		_, _ = h.Write(span)
	} else {
		// Some in-process cache variants are intentionally not serializable yet.
		// They still need a stable pointer identity for the additive producer seam;
		// the backend wiring ticket decides whether that pointer can be staged.
		_, _ = h.Write([]byte(fmt.Sprintf("%#v", kv)))
	}
	return hex.EncodeToString(h.Sum(nil))
}
