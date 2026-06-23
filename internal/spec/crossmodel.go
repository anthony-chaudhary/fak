package spec

// crossmodel.go is the #534 wiring of polymodel.CanShare into cachemeta's verdict
// layer plus the model.KVCache.Clone splice — the consumer side of "host many, share
// the prefill". It is the off-defconfig bridge the plan names: cachemeta (on the live
// request path) deliberately does NOT import the off-defconfig polymodel leaf, so the
// CanShare decision crosses into the on-path verdict layer HERE, the same shape as an
// integrator computing SignatureVerified for cachemeta.CheckResidentClaim.
//
// Two halves:
//   - PrefillSharePolicyFor turns a (provider, consumer) model pair into the pure-data
//     cachemeta.PrefillSharePolicy (the verdict-layer barrier lift). The rule lives in
//     polymodel.CanShare; cachemeta trusts the Allowed verdict but still verifies every
//     non-ModelID axis, so a lying CanShare can never relax the rest of the binding.
//   - SplicePrefillShare is the byte-moving half: on a lossless share it forks the
//     provider's already-computed prefix cache into the consumer via model.KVCache.Clone
//     (the proven bit-exact fork), so the consumer prefills only the suffix and skips
//     the shared prefix's prefill FLOPs.
//
// This leaf is off the defconfig and reached only under FAK_POLYMODEL (epic #529); the
// functions are deterministic library calls that never consult the flag themselves,
// matching spec's existing gating convention.

import (
	"github.com/anthony-chaudhary/fak/internal/cachemeta"
	"github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/polymodel"
)

// PrefillSharePolicyFor builds the cachemeta cross-model prefill-share policy for a
// (provider, consumer) model pair: it consults polymodel.CanShare (same non-empty
// Family + byte-identical PrefixDigest ⇒ the prefix KV is bit-identical, so reuse is
// lossless) and, on a yes, carries the family/digest into cachemeta's verdict for the
// HIT's audit trail. On a no it returns the zero policy, so cachemeta's exact-ModelID
// barrier stands unchanged. This is where the CanShare decision the plan ships as
// "decision only" crosses into the on-path verdict layer (#534 verdict half).
func PrefillSharePolicyFor(provider, consumer polymodel.Model) cachemeta.PrefillSharePolicy {
	if !polymodel.CanShare(provider, consumer) {
		return cachemeta.PrefillSharePolicy{}
	}
	return cachemeta.PrefillSharePolicy{
		Allowed:      true,
		Family:       provider.Family,
		PrefixDigest: provider.PrefixDigest,
	}
}

// SplicePrefillShare is the #534 "KVCache.Clone splice": fork the provider model's
// already-computed prefix cache into the consumer via model.KVCache.Clone when the two
// are polymodel.CanShare-compatible. Because a compatible pair shares a byte-identical
// prefill band, the cloned prefix is bit-identical to what the consumer would have
// prefilled itself, so the consumer may skip the shared prefix's prefill and prefill
// only the suffix. Returns the cloned cache and true on a lossless share; (nil, false)
// when the pair is not share-compatible or providerCache is nil, so the caller falls
// back to a full prefill. The clone is the proven bit-exact fork
// (TestKVPrefixReuseMatchesRecompute), never an approximate copy.
func SplicePrefillShare(provider, consumer polymodel.Model, providerCache *model.KVCache) (*model.KVCache, bool) {
	if providerCache == nil || !polymodel.CanShare(provider, consumer) {
		return nil, false
	}
	return providerCache.Clone(), true
}
