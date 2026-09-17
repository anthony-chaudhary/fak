package model

import "hash/fnv"

// DomainTag namespaces a keyed draw so the accept coin, the bonus-token draw,
// and the residual-replacement draw can never collide on the same (request, step)
// key even when they share a step index.
type DomainTag uint64

const (
	DomainAcceptCoin DomainTag = 0x1
	DomainBonusToken DomainTag = 0x2
	DomainResidual   DomainTag = 0x3
)

// HashRequestID folds a request id to a 64-bit key via FNV-1a (hash/fnv). An
// empty id folds to 0, which callers treat as "no key" (legacy path).
func HashRequestID(requestID string) uint64 {
	if requestID == "" {
		return 0
	}
	h := fnv.New64a()
	_, _ = h.Write([]byte(requestID))
	return h.Sum64()
}

// KeyedUniform derives a deterministic uniform float32 in [0,1) from
// (requestHash, domain, step). The step is woven in as step*SplitMixConstant
// (the splitmix64 gamma) and the whole key is finalized by ONE SplitMix64 call
// -- never by chaining SplitMix64 outputs, which would double-add the constant
// and weaken the stream. KeyedUniform(0, ...) is defined but callers must not
// use it as a keyed draw: KeyedOn() gates that.
func KeyedUniform(requestHash uint64, domain DomainTag, step int) float32 {
	seed := requestHash ^ uint64(domain) ^ (uint64(step+1) * SplitMixConstant)
	u := SplitMix64(seed)
	return float32(u>>40) * (1.0 / 16777216.0)
}
