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

// Consumer-attribution meta keys (§2.5: "which agent/turn consumed a cached tool
// result"). These are OPTIONAL, OPEN ToolCall.Meta keys a caller may set so a tier-2
// hit names the agent/turn that reused the result. They are forward-compatible: a
// call that sets none is still attributed by its TraceID alone, and a call with no
// identity at all attaches no (empty) consumer. The vDSO neither requires nor invents
// them — it lowers exactly the identity the call already carries.
const (
	MetaAgentID = "agent_id"   // the agent that issued the consuming call
	MetaTurn    = "turn"       // the model turn the consuming call rode
	MetaSession = "session_id" // the session the consuming call belongs to
)

// consumerOf derives the cachemeta Consumer that reused a cached tool result from the
// LOOKUP call's identity: its TraceID (the trace seam every call carries) plus the
// OPEN agent/turn/session meta keys when present. ok=false when the call names no
// identity at all, so an anonymous lookup attaches no empty consumer to the hit event
// (the consumer graph stays a record of who actually reused a span, never noise).
func consumerOf(c *abi.ToolCall) (cachemeta.Consumer, bool) {
	if c == nil {
		return cachemeta.Consumer{}, false
	}
	cons := cachemeta.Consumer{Kind: "agent", TraceID: c.TraceID}
	if c.Meta != nil {
		cons.AgentID = c.Meta[MetaAgentID]
		if cons.ID = c.Meta[MetaTurn]; cons.ID == "" {
			cons.ID = c.Meta[MetaSession]
		}
	}
	if cons.TraceID == "" && cons.AgentID == "" && cons.ID == "" {
		return cachemeta.Consumer{}, false
	}
	return cons, true
}

// consumerOpt is the cachemeta.Option that attributes a tier-2 hit to its consumer,
// or nil when the consuming call names no identity (cachemeta.apply skips nil opts).
func consumerOpt(c *abi.ToolCall) cachemeta.Option {
	if cons, ok := consumerOf(c); ok {
		return cachemeta.WithConsumer(cons)
	}
	return nil
}

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
func (v *VDSO) emitCache(kind CacheEventKind, key string, ref abi.Ref, witness string, opts ...cachemeta.Option) {
	v.regMu.RLock()
	sink := v.cacheSink
	v.regMu.RUnlock()
	if sink == nil {
		return
	}
	entry, err := cachemeta.FromVDSOKey(key, ref, append([]cachemeta.Option{cachemeta.WithWitness(witness)}, opts...)...)
	if err != nil {
		return
	}
	sink(CacheEvent{Kind: kind, Entry: entry})
}
