package cachewitness

// managed_cache_selfcheck.go — the managed-cache (epic #1844 C6) trust-but-verify
// self-check, sibling of the in-kernel KV-prefix witness in cachewitness.go and a
// concrete loop under the managed-cache verification epic #3569.
//
// WHAT IT ANSWERS. "Is the managed cache actually working on the trajectory that just
// ran?" The managed-cache lever's one job on the outbound Anthropic wire is to extend an
// existing stable-head cache_control breakpoint to Anthropic's 1h TTL tier (or place +
// upgrade it in one turn) so a session idling past the 5m window re-enters on a cache
// READ instead of re-writing the whole prefix. fak WITNESSES that it authored that
// transform (fak_gateway_cache_ttl_upgrade_total, by outcome); whether the provider then
// honored the longer TTL is OBSERVED on the provider's cache_read counter. This self-check
// reconciles the two over one /metrics scrape of a real served session.
//
// THE PROVENANCE FENCE (the same line cachewitness.go and metrics_render.go draw). The
// authored-upgrade count and the provider cache_read are DISTINCT trust classes over
// DISTINCT surfaces, and a provider cache_read is NEVER proof fak's upgrade caused it — a
// cache_read can equally come from the client's own breakpoints or an ordinary warm turn.
// So this report:
//   - keeps WITNESSED (authored) and OBSERVED (provider) numbers in separate fields and
//     never sums or derives one from the other;
//   - HARD-FAILS only on fak's OWN bug outcomes (splice_failed / redecode_failed), which the
//     transform contract says must stay 0 — those mean fak corrupted a real body;
//   - flags an authored-but-zero-provider-reuse trajectory as SUSPECT (worth a look:
//     TTL expiry, eviction, or a session too short to ever idle), NOT as fak's fault; and
//   - falls open (INSUFFICIENT, OK=true) when the lever was passive/off or recorded no
//     attempt — a thin signal is never a failure, matching cvregress and the sibling gates.
//
// It reuses splitMetricLine (this package) so it reads the SAME Prometheus exposition the
// KV-prefix witness does, and it does not touch Record — the golden-pinned KV-prefix
// evidence record stays exactly as it was.

import (
	"bufio"
	"fmt"
	"sort"
	"strings"
)

// ManagedCacheVerdict is the closed-vocabulary outcome of the self-check.
type ManagedCacheVerdict string

const (
	// ManagedCacheConsistent: fak authored ≥1 stable-head TTL upgrade AND the provider
	// served cache_read tokens on this trajectory — the lever fired and provider reuse is
	// present. Consistency, not causation (see the provenance fence). OK.
	ManagedCacheConsistent ManagedCacheVerdict = "CONSISTENT"
	// ManagedCacheSuspect: fak authored ≥1 upgrade but the provider served ZERO cache_read
	// tokens. fak did its part; the missing provider effect is worth investigating (TTL
	// expiry, eviction, or a session too short to idle past the window) but is NOT
	// attributable to fak. OK (fall-open) — flagged, not failed.
	ManagedCacheSuspect ManagedCacheVerdict = "SUSPECT"
	// ManagedCacheBroken: fak's OWN transform failed on a real body (splice_failed or
	// redecode_failed > 0). The contract says those must stay 0; any occurrence is a fak
	// bug that corrupts the outbound wire. The ONLY failing verdict. NOT OK.
	ManagedCacheBroken ManagedCacheVerdict = "BROKEN"
	// ManagedCacheNotEngaged: the lever was active but authored zero upgrades — every
	// attempt bailed for a NON-bug reason (no_stable_breakpoint, already_1h,
	// ttl_already_set, volatile_head, non_json). DominantBailReason names why. OK.
	ManagedCacheNotEngaged ManagedCacheVerdict = "NOT_ENGAGED"
	// ManagedCacheInsufficient: no managed-cache TTL-upgrade series was present (the lever
	// was passive/off, or this is not a fak gateway) or it recorded no attempt at all —
	// nothing to reconcile. OK (fall-open).
	ManagedCacheInsufficient ManagedCacheVerdict = "INSUFFICIENT"
)

// The managed-cache TTL-upgrade outcome tokens (the `outcome=` label values of
// fak_gateway_cache_ttl_upgrade_total). The two AUTHORING outcomes count turns fak actually
// extended a stable-head breakpoint to the 1h tier; every other value is a bail reason.
const (
	ttlOutcomeUpgraded          = "upgraded"            // WITNESSED: ttl:"1h" spliced into an existing stable-head cache_control.
	ttlOutcomePlacedAndUpgraded = "placed_and_upgraded" // WITNESSED: no breakpoint existed; fak placed AND upgraded it in one turn (#2175).
	ttlOutcomeSpliceFailed      = "splice_failed"       // fak bug: the target block was not spliceable. Must stay 0.
	ttlOutcomeRedecodeFailed    = "redecode_failed"     // fak bug: the spliced body failed to re-decode. Must stay 0.
)

// managedCacheTTLSeries is the WITNESSED-authoring metric this self-check folds.
const managedCacheTTLSeries = "fak_gateway_cache_ttl_upgrade_total"

// ManagedCacheReport is the provenance-labeled verdict of the managed-cache self-check over
// one gateway /metrics scrape. WITNESSED (fak-authored) and OBSERVED (provider-relayed)
// numbers are kept in separate fields and never summed.
type ManagedCacheReport struct {
	GatewayURL string              `json:"gateway_url"`
	Verdict    ManagedCacheVerdict `json:"verdict"`
	OK         bool                `json:"ok"`

	// --- WITNESSED: fak's own managed-cache authoring (fak_gateway_cache_ttl_upgrade_total) ---

	// Upgraded / PlacedAndUpgraded are the two authoring outcomes; Authored is their sum —
	// the count of turns fak actually put a stable-head breakpoint on the 1h tier.
	Upgraded          uint64 `json:"upgraded"`
	PlacedAndUpgraded uint64 `json:"placed_and_upgraded"`
	Authored          uint64 `json:"authored"`
	// Bails maps every NON-authoring outcome to its count (the refusal breakdown), including
	// the two fak-bug outcomes so a reader sees the full family. Empty when nothing bailed.
	Bails map[string]uint64 `json:"bails,omitempty"`
	// CorruptionOutcomes is splice_failed + redecode_failed — fak's own bug slice, which
	// forces the BROKEN verdict. Must be 0 on a healthy gateway.
	CorruptionOutcomes uint64 `json:"corruption_outcomes"`
	// DominantBailReason is the highest-count non-authoring, non-bug outcome (ties broken by
	// name), set for the NOT_ENGAGED verdict so the finding names WHY nothing was upgraded.
	DominantBailReason string `json:"dominant_bail_reason,omitempty"`

	// --- OBSERVED: the upstream provider's prompt cache (relayed, never fak's) ---

	// ProviderCacheReadTokens is the cumulative cache_read the provider served
	// (fak_gateway_inference_cached_prompt_tokens_total), relayed verbatim. The OBSERVED
	// effect this self-check reconciles the authoring against — consistency, not causation.
	ProviderCacheReadTokens uint64 `json:"provider_cache_read_tokens"`
	// ProviderCacheCreationTokens is the provider's cache_creation WRITE axis
	// (fak_vcache_cache_creation_tokens_total), relayed verbatim. Reported for context (the
	// 1h tier costs a 2x write premium) — not part of the verdict.
	ProviderCacheCreationTokens uint64 `json:"provider_cache_creation_tokens"`

	// Provenance maps each number to its trust class, so the report is self-describing.
	Provenance map[string]Provenance `json:"provenance"`
	// Finding is the one-line human-readable explanation of the verdict.
	Finding string `json:"finding"`
}

// SelfCheckManagedCache folds a gateway /metrics body (Prometheus text exposition) into the
// managed-cache self-check verdict. It reads only the TTL-upgrade family and the provider
// read/creation counters, ignoring every other series, so it is robust to the rest of the
// gateway metric surface changing. It is PURE — no clock, no I/O — mirroring cachewitness.Parse.
// gatewayURL is recorded verbatim for provenance.
func SelfCheckManagedCache(gatewayURL, metricsBody string) ManagedCacheReport {
	rep := ManagedCacheReport{
		GatewayURL: gatewayURL,
		Provenance: map[string]Provenance{
			"authored":                       Witnessed,
			"bails":                          Witnessed,
			"provider_cache_read_tokens":     Observed,
			"provider_cache_creation_tokens": Observed,
		},
	}

	familyPresent := false
	bails := map[string]uint64{}
	sc := bufio.NewScanner(strings.NewReader(metricsBody))
	sc.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue // skip HELP/TYPE/comment lines
		}
		name, label, val, ok := splitMetricLine(line)
		if !ok {
			continue
		}
		switch name {
		case managedCacheTTLSeries:
			familyPresent = true
			switch labelValue(label, "outcome") {
			case ttlOutcomeUpgraded:
				rep.Upgraded = val
			case ttlOutcomePlacedAndUpgraded:
				rep.PlacedAndUpgraded = val
			default:
				if o := labelValue(label, "outcome"); o != "" {
					bails[o] = val
				}
			}
		case mProviderRd:
			rep.ProviderCacheReadTokens = val
		case mProviderCacheCreation:
			rep.ProviderCacheCreationTokens = val
		}
	}

	rep.Authored = rep.Upgraded + rep.PlacedAndUpgraded
	if len(bails) > 0 {
		rep.Bails = bails
	}
	rep.CorruptionOutcomes = bails[ttlOutcomeSpliceFailed] + bails[ttlOutcomeRedecodeFailed]

	// Verdict order is load-bearing: a fak-bug corruption outcome fails HARD even when some
	// upgrades also succeeded; otherwise reconcile authoring against the observed provider
	// effect; a lever that authored nothing is explained by its dominant bail reason; and an
	// absent/empty family falls open.
	switch {
	case !familyPresent:
		rep.Verdict = ManagedCacheInsufficient
		rep.OK = true
		rep.Finding = "INSUFFICIENT — no managed-cache TTL-upgrade series present; the --managed-cache lever was passive/off, or this is not a fak gateway"
	case rep.CorruptionOutcomes > 0:
		rep.Verdict = ManagedCacheBroken
		rep.OK = false
		rep.Finding = fmt.Sprintf("BROKEN — fak's own TTL-upgrade transform failed on %d real body(ies) (splice_failed=%d redecode_failed=%d); these must stay 0",
			rep.CorruptionOutcomes, bails[ttlOutcomeSpliceFailed], bails[ttlOutcomeRedecodeFailed])
	case rep.Authored > 0 && rep.ProviderCacheReadTokens > 0:
		rep.Verdict = ManagedCacheConsistent
		rep.OK = true
		rep.Finding = fmt.Sprintf("CONSISTENT — fak authored %d stable-head 1h-TTL upgrade(s) (WITNESSED) and the provider served %d cache_read token(s) (OBSERVED); the managed cache fired and provider reuse is present (consistency, not causation)",
			rep.Authored, rep.ProviderCacheReadTokens)
	case rep.Authored > 0:
		rep.Verdict = ManagedCacheSuspect
		rep.OK = true
		rep.Finding = fmt.Sprintf("SUSPECT — fak authored %d 1h-TTL upgrade(s) (WITNESSED) but the provider served 0 cache_read tokens (OBSERVED); fak did its part — investigate TTL expiry, eviction, or a session too short to idle past the 5m window (not attributable to fak)",
			rep.Authored)
	default:
		// Authored == 0 with the family present. If there is a non-bug bail, name it; else
		// nothing was attempted and there is nothing to reconcile.
		if reason, count, ok := dominantNonBugBail(bails); ok {
			rep.Verdict = ManagedCacheNotEngaged
			rep.OK = true
			rep.DominantBailReason = reason
			rep.Finding = fmt.Sprintf("NOT_ENGAGED — the lever was active but authored 0 upgrades; the dominant refusal was %q (%d); no stable-head breakpoint was eligible to extend",
				reason, count)
		} else {
			rep.Verdict = ManagedCacheInsufficient
			rep.OK = true
			rep.Finding = "INSUFFICIENT — the managed-cache lever recorded no upgrade attempt on this trajectory; nothing to reconcile"
		}
	}
	return rep
}

// mProviderCacheCreation is the provider cache_creation WRITE-axis counter, the companion to
// mProviderRd (the READ axis). Reported for context only, never part of the verdict.
const mProviderCacheCreation = "fak_vcache_cache_creation_tokens_total"

// labelValue pulls value out of a Prometheus label block like `outcome="upgraded"`, for the
// named key. Returns "" when the key is absent — mirrors regimeLabel in cachewitness.go but
// keyed by an arbitrary label name.
func labelValue(label, key string) string {
	needle := key + `="`
	i := strings.Index(label, needle)
	if i < 0 {
		return ""
	}
	rest := label[i+len(needle):]
	if j := strings.IndexByte(rest, '"'); j >= 0 {
		return rest[:j]
	}
	return ""
}

// dominantNonBugBail returns the highest-count bail outcome that is NOT one of fak's own bug
// outcomes (splice_failed / redecode_failed — those drive BROKEN, not NOT_ENGAGED), with ties
// broken by outcome name for determinism. ok is false when there is no non-bug bail.
func dominantNonBugBail(bails map[string]uint64) (string, uint64, bool) {
	names := make([]string, 0, len(bails))
	for name := range bails {
		if name == ttlOutcomeSpliceFailed || name == ttlOutcomeRedecodeFailed {
			continue
		}
		if bails[name] == 0 {
			continue
		}
		names = append(names, name)
	}
	if len(names) == 0 {
		return "", 0, false
	}
	sort.Strings(names)
	best, bestN := "", uint64(0)
	for _, name := range names {
		if bails[name] > bestN {
			best, bestN = name, bails[name]
		}
	}
	return best, bestN, true
}
