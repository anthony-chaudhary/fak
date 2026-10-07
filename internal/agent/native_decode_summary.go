package agent

import "github.com/anthony-chaudhary/fak/internal/radixkv"

// Native decode paths, mirroring enginestep's closed path vocabulary so a
// per-request row and the process-wide fak_engine_* families name the same thing.
const (
	NativeDecodePathSerial      = "serial"
	NativeDecodePathBatched     = "batched"
	NativeDecodePathSpeculative = "speculative"
)

// SpeculativeDecodeTally is one request's verified draft-verify rounds: how many
// rounds ran, how many draft tokens they proposed, and how many the target accepted.
type SpeculativeDecodeTally struct {
	Rounds         int `json:"rounds"`
	DraftTokens    int `json:"draft_tokens"`
	AcceptedTokens int `json:"accepted_tokens"`
}

// NativeDecodeSummary is the request-local view of the native engine's decode
// cycle. It is built from the request's own generate result, never from the
// shared enginestep recorder, so concurrent requests cannot bleed into each other.
type NativeDecodeSummary struct {
	// Path is the dominant decode path: speculative when any verify round ran,
	// batched when the request decoded inside a multi-lane cohort, else serial.
	Path string `json:"path"`
	// CohortSize is the coalesced cohort this request decoded in (0 when it
	// decoded alone).
	CohortSize int `json:"cohort_size,omitempty"`
	// Speculative is set only when at least one verify round ran.
	Speculative *SpeculativeDecodeTally `json:"speculative,omitempty"`
	// CacheTier is the KV-prefix tier this request's reused prompt was restored
	// from (radixkv.SnapshotTier: device_l1, host_dram_l2, remote_http_l3), or
	// NativeCacheTierNone when nothing was reused. "" means a reuse happened but
	// the planner did not resolve its tier; it is never imputed.
	CacheTier string `json:"cache_tier,omitempty"`
	// CacheRestore is this request's restore outcome: hit (some prefix served
	// from cache), miss (the lookup matched nothing), or unserved (the lookup
	// matched a prefix but none of it was served: a restore fault, an eviction
	// race, or a servability gate fell back to full prefill).
	CacheRestore string `json:"cache_restore,omitempty"`
}

// Native KV-prefix restore vocabulary carried on NativeDecodeSummary.
const (
	NativeCacheTierNone        = "none"
	NativeCacheRestoreHit      = "hit"
	NativeCacheRestoreMiss     = "miss"
	NativeCacheRestoreUnserved = "unserved"
)

// nativeCacheRestore classifies one request's prefix-cache restore from its own
// generate facts: matched (tokens served from cache), cacheable (tokens the
// lookup matched before servability), and the tier the served prefix came from.
func nativeCacheRestore(matched, cacheable int, tier radixkv.SnapshotTier) (tierLabel, outcome string) {
	switch {
	case matched > 0:
		return string(tier), NativeCacheRestoreHit
	case cacheable > 0:
		return NativeCacheTierNone, NativeCacheRestoreUnserved
	default:
		return NativeCacheTierNone, NativeCacheRestoreMiss
	}
}

func newNativeDecodeSummary(res inKernelGenerateResult) *NativeDecodeSummary {
	sum := &NativeDecodeSummary{Path: NativeDecodePathSerial}
	if res.batchReceipt.CohortID != 0 {
		sum.CohortSize = res.batchReceipt.CohortSize
		if sum.CohortSize > 1 {
			sum.Path = NativeDecodePathBatched
		}
	}
	if res.spec.Rounds > 0 {
		spec := res.spec
		sum.Speculative = &spec
		sum.Path = NativeDecodePathSpeculative
	}
	sum.CacheTier, sum.CacheRestore = nativeCacheRestore(res.matched, res.cacheable, res.sourceTier)
	return sum
}
