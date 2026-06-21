package vdso

import (
	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/cachemeta"
)

// CacheEventKind names a tier-2 vDSO cache lifecycle event that is now observable
// as a first-class cachemeta entry (§2.5: "vDSO still needs first-class cachemeta
// event emission"). These events turn the strongest local cache into observable
// ground truth, in the same stream as tool/context entries.
type CacheEventKind string

const (
	CacheFill   CacheEventKind = "fill"   // a tier-2 entry was stored from an EvComplete
	CacheHit    CacheEventKind = "hit"    // a tier-2 entry was served on Lookup
	CacheEvict  CacheEventKind = "evict"  // an entry was dropped by LRU pressure
	CacheRevoke CacheEventKind = "revoke" // an entry was evicted by witness refutation
)

// CacheEvent is a tier-2 lifecycle event lowered into a cachemeta entry. The Entry
// is built with cachemeta.FromVDSOKey, so it carries tool, args digest, admission
// epoch, witness, and (when a witness is present) external-refutation invalidation.
type CacheEvent struct {
	Kind  CacheEventKind
	Entry cachemeta.Entry
}

// WitnessFunc is a per-tool adapter that extracts the external world-state witness a
// call/result was admitted under — e.g. an etag for a web read, a git SHA for a
// file read, a DB row version, or a sandbox snapshot id. Registering one lets a
// tool govern its own witness instead of relying on the internal epoch (§2.5:
// "per-tool witness adapters instead of relying mainly on internal epochs").
type WitnessFunc func(c *abi.ToolCall, r *abi.Result) string

// SetCacheEventSink installs an observer of tier-2 cache lifecycle events. It is
// opt-in: a nil sink (the default) means the vDSO behavior is unchanged. Like the
// coherence-bus subscribers, the sink is invoked AFTER the cache mutation and
// OUTSIDE v.mu, so it may re-enter the vDSO. The sink must not block the hot path.
func (v *VDSO) SetCacheEventSink(fn func(CacheEvent)) {
	v.regMu.Lock()
	v.cacheSink = fn
	v.regMu.Unlock()
}

// RegisterWitness installs a per-tool witness adapter. An empty/nil result falls
// through to the default witness extraction (the call's declared "witness" meta,
// then the result's). The adapter is consulted on every EvComplete fill.
func (v *VDSO) RegisterWitness(tool string, fn WitnessFunc) {
	v.regMu.Lock()
	if v.witnessAdapters == nil {
		v.witnessAdapters = map[string]WitnessFunc{}
	}
	v.witnessAdapters[tool] = fn
	v.regMu.Unlock()
}

// resolveWitness consults a per-tool adapter first, then the default extraction.
// Called on the fill path so a registered tool can pin its own external witness.
func (v *VDSO) resolveWitness(c *abi.ToolCall, r *abi.Result) string {
	if c != nil {
		v.regMu.RLock()
		fn := v.witnessAdapters[c.Tool]
		v.regMu.RUnlock()
		if fn != nil {
			if w := fn(c, r); w != "" {
				return w
			}
		}
	}
	return defaultWitnessOf(c, r)
}

// emitCache builds a cachemeta entry from a tier-2 identity and dispatches it to the
// installed sink, if any. It is safe to call with a key that fails to parse (the
// event is dropped); in practice every emit site holds a well-formed tier-2 key.
// Called OUTSIDE v.mu by every emit site.
func (v *VDSO) emitCache(kind CacheEventKind, key string, ref abi.Ref, witness string) {
	v.regMu.RLock()
	sink := v.cacheSink
	v.regMu.RUnlock()
	if sink == nil {
		return
	}
	entry, err := cachemeta.FromVDSOKey(key, ref, cachemeta.WithWitness(witness))
	if err != nil {
		return
	}
	sink(CacheEvent{Kind: kind, Entry: entry})
}
