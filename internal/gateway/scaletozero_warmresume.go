package gateway

import "strings"

// scaletozero_warmresume.go — cache-WARM scale-to-zero resume (#2853, Track D of
// epic #2834). Sibling of session_prefix_index.go (#2852): that leaf carries the warm
// prefix across a PLATFORM hop; this one carries it across a SCALE-TO-ZERO hop.
//
// Hermes mechanism. Hermes' serverless backends (Modal/Daytona) hibernate the agent's
// ENVIRONMENT when idle and wake it on demand via relay scale-to-zero primitives
// (going_idle / inbound_ack / wake-poke), "costing nearly nothing between sessions."
// But hibernation snapshots the filesystem/process — the provider-side KV cache is COLD
// on wake, so the first post-hibernate turn re-writes (cold-creates) the whole prefix and
// serves nothing from cache. "Nearly nothing when idle", but a full re-prefill on wake.
//
// What fak does better. fak owns the KV cache, so scale-to-zero can be made cache-AWARE:
// on going-idle, persist the metadata to reconstruct the prefix cheaply (a descriptor, not
// the KV bytes); on wake, restore warm so the first post-hibernate turn READS the prefix
// from cache instead of cold-writing it. "Nearly nothing when idle" AND "warm on wake" —
// the second half only a kernel that owns the cache can deliver.
//
// PROVENANCE (the same split session_prefix_index.go and cache_pricing.go keep). The
// going-idle descriptor and the restore decision are WITNESSED (fak authors the prefix
// content-address and the persist/restore hooks over inputs it controls). The cache-read
// FRACTION on the first post-wake turn is priced from the cache_read tokens the turn
// serves — OBSERVED when fed a live provider bill — so a "warm" verdict is evidence the
// resumed turn actually hit the cache, never a trust claim. A cold hibernate re-send reads
// fraction 0 on the wake turn, so the cold baseline can never be mislabeled warm.
//
// GATED / gen/next posture. These are PURE hooks (no wire, no clock): the going-idle and
// wake seams are modeled deterministically here and are INERT until a host wires them to
// the relay's going_idle / wake-poke — the same default-off posture resume_projection.go
// ships under. Nothing here hibernates, persists, or resumes a live session.
//
//   PROMOTION EVIDENCE (what moves this toward now): a dogfood run that feeds the live
//     first-post-wake provider bill into WakeWarmRestore's cacheReadTokens and shows the
//     witnessed fraction clears WarmResumeFloor on real traffic, i.e. fak's restore landed
//     a real cache_read where a Hermes-shaped hibernate would have read 0.
//   DEMOTION / RETIREMENT EVIDENCE: a wake path where the provider cannot be pre-primed
//     (it only caches on first WRITE, never on a restore) so the warm arm collapses to the
//     cold arm (lift ~0) — then this witness is retired as unbacked by the provider.
//   INVALIDATING ASSUMPTION: that fak's wake-restore can land the resident prefix as a
//     cache_read on the FIRST billed post-wake turn (pre-prime before the turn, or the
//     provider honors a restored prefix). If the provider only warms on a prior write, the
//     first post-wake turn still cold-creates and the modeled lift does not materialize.

// GoingIdleKVDescriptor is what a scale-to-zero GOING-IDLE hook persists so a later wake
// can restore the prefix WARM instead of cold-re-prefilling it. It is deliberately NOT the
// KV bytes — it is the cheap reconstruction metadata fak already holds: the prefix
// content-address, its prompt-token length, and the cache tier it was warmed under. This
// is the "or the metadata to reconstruct it cheaply" half of the issue's ask.
type GoingIdleKVDescriptor struct {
	Schema string `json:"schema"`
	// Trace is the session/trace the prefix belongs to (retained for the audit row; not
	// part of the prefix identity).
	Trace string `json:"trace,omitempty"`
	// PrefixDigest is fak's content-address for the resident prefix — the handle a wake
	// restore reconstructs the warm prefix from. Empty means no prefix was captured, so a
	// wake off this descriptor can never be witnessed warm.
	PrefixDigest string `json:"prefix_digest"`
	// ResidentPrefixTokens is the prompt-token length of the prefix warm at going-idle —
	// the tokens a cache-aware wake serves from cache that a cold hibernate re-writes.
	ResidentPrefixTokens int `json:"resident_prefix_tokens"`
	// TTLTier is the cache tier the prefix was warmed under (5m default), carried so the
	// wake restore re-primes under the same tier.
	TTLTier CacheTTL `json:"ttl_tier"`
}

const scaleToZeroWarmResumeSchema = "fak.gateway.scaletozero_warm_resume/v1"

// GoingIdlePersist is the going-idle KV-persist hook: at the moment the relay signals
// going_idle, the host captures the resident prefix's content-address and length into a
// descriptor the wake path can restore from. Pure — it mints the descriptor, persists
// nothing itself (the host owns the sink), and takes no clock. An empty digest or a
// non-positive length yields a descriptor a wake can only read cold (guarded downstream).
func GoingIdlePersist(trace, prefixDigest string, residentPrefixTokens int, ttl CacheTTL) GoingIdleKVDescriptor {
	if ttl != CacheTTL1h {
		ttl = CacheTTL5m
	}
	return GoingIdleKVDescriptor{
		Schema:               scaleToZeroWarmResumeSchema,
		Trace:                strings.TrimSpace(trace),
		PrefixDigest:         strings.TrimSpace(prefixDigest),
		ResidentPrefixTokens: maxNonNeg(residentPrefixTokens),
		TTLTier:              ttl,
	}
}

// ScaleToZeroResumeArm is one arm's FIRST-POST-WAKE turn accounting: what the resumed turn
// served from cache (CacheReadTokens) versus what it cold-created (CacheCreationTokens), and
// the witnessed cache-read fraction. Warm is true iff the fraction clears WarmResumeFloor —
// the same floor the cross-platform continuity witness (#2852) and cacheobs's cold-cliff
// alarm use, so "warm on wake" means the same thing across Track D.
type ScaleToZeroResumeArm struct {
	Arm                 string  `json:"arm"`
	PromptTokens        int     `json:"prompt_tokens"`
	CacheReadTokens     int     `json:"cache_read_tokens"`
	CacheCreationTokens int     `json:"cache_creation_tokens"`
	CacheReadFraction   float64 `json:"cache_read_fraction"`
	Warm                bool    `json:"warm"`
}

// WakeWarmRestore is the wake-time warm-restore: given the going-idle descriptor and the
// first post-wake turn's prompt length, it models the turn served against the restored
// prefix — the resident prefix (capped at the prompt) is a cache READ, and only the new
// tokens beyond it are uncached/cold-created. Feed the OBSERVED provider cache_read for a
// live wake by passing that as resumeCacheReadTokens (>= 0); pass -1 to MODEL the restore
// as serving the whole resident prefix. An empty-digest descriptor cannot be restored, so
// it reads cold (fraction 0) — a captured-nothing going-idle is never mislabeled warm.
func WakeWarmRestore(desc GoingIdleKVDescriptor, resumeTurnPromptTokens, resumeCacheReadTokens int) ScaleToZeroResumeArm {
	prompt := maxNonNeg(resumeTurnPromptTokens)
	served := 0
	if desc.PrefixDigest != "" {
		if resumeCacheReadTokens >= 0 {
			// OBSERVED: the provider-relayed cache_read on the live wake turn.
			served = resumeCacheReadTokens
		} else {
			// MODELED: the restore serves the whole resident prefix (capped at the prompt).
			served = desc.ResidentPrefixTokens
		}
	}
	if served > prompt {
		served = prompt
	}
	return ScaleToZeroResumeArm{
		Arm:                 "fak_warm_restore",
		PromptTokens:        prompt,
		CacheReadTokens:     served,
		CacheCreationTokens: prompt - served,
		CacheReadFraction:   cacheReadFraction(served, prompt),
		Warm:                cacheReadFraction(served, prompt) >= WarmResumeFloor,
	}
}

// coldHibernateBaseline is the Hermes arm: the provider KV cache is COLD on wake, so the
// first post-wake turn serves NOTHING from cache and cold-creates the whole prompt. Read
// fraction is 0 by construction — the cold baseline the warm restore is witnessed against.
func coldHibernateBaseline(resumeTurnPromptTokens int) ScaleToZeroResumeArm {
	prompt := maxNonNeg(resumeTurnPromptTokens)
	return ScaleToZeroResumeArm{
		Arm:                 "hermes_cold_hibernate",
		PromptTokens:        prompt,
		CacheReadTokens:     0,
		CacheCreationTokens: prompt,
		CacheReadFraction:   0,
		Warm:                false,
	}
}

// ScaleToZeroScenario is a deterministic going-idle -> wake cycle: a session whose resident
// prefix is ResidentPrefixTokens long at going-idle, woken by a first turn of
// ResumeTurnPromptTokens (the resident prefix plus a short new user turn appended to it).
type ScaleToZeroScenario struct {
	Session                string   `json:"session"`
	ResidentPrefixTokens   int      `json:"resident_prefix_tokens"`
	ResumeTurnPromptTokens int      `json:"resume_turn_prompt_tokens"`
	TTLTier                CacheTTL `json:"ttl_tier"`
}

// ScaleToZeroWarmResumeResult is the committed witness: the SAME first-post-wake turn priced
// two ways. HermesColdHibernate is the serverless-hibernate baseline — the KV cache is cold
// on wake, so the turn cold-creates the whole prefix and reads fraction 0. FakWarmRestore is
// fak's cache-aware wake — the going-idle descriptor restores the prefix, so the turn reads
// it from cache. CacheReadFractionLift is the fak fraction minus the Hermes fraction: the
// warm-on-wake continuity Hermes structurally cannot get. Pure and deterministic.
type ScaleToZeroWarmResumeResult struct {
	Harness               string                `json:"harness"`
	Scenario              ScaleToZeroScenario   `json:"scenario"`
	Descriptor            GoingIdleKVDescriptor `json:"descriptor"`
	HermesColdHibernate   ScaleToZeroResumeArm  `json:"hermes_cold_hibernate"`
	FakWarmRestore        ScaleToZeroResumeArm  `json:"fak_warm_restore"`
	CacheReadFractionLift float64               `json:"cache_read_fraction_lift"`
	WarmResumeFloor       float64               `json:"warm_resume_floor"`
}

// MeasureScaleToZeroWarmResume runs one going-idle -> wake cycle both ways and returns the
// comparison. The going-idle hook persists a descriptor for the resident prefix; the Hermes
// arm wakes with a cold provider cache (cache_read 0); the fak arm wakes through the
// descriptor and serves the resident prefix from the restored cache. Both price the
// IDENTICAL first-post-wake turn, so CacheReadFractionLift isolates the cache-aware restore —
// the warm continuity the issue asks be witnessed against a cold baseline.
func MeasureScaleToZeroWarmResume(sc ScaleToZeroScenario) ScaleToZeroWarmResumeResult {
	desc := GoingIdlePersist(sc.Session, "prefix:"+sc.Session, sc.ResidentPrefixTokens, sc.TTLTier)
	cold := coldHibernateBaseline(sc.ResumeTurnPromptTokens)
	// MODELED wake: pass -1 so the restore serves the whole resident prefix (capped at the
	// prompt). A dogfood run passes the OBSERVED provider cache_read instead.
	warm := WakeWarmRestore(desc, sc.ResumeTurnPromptTokens, -1)
	return ScaleToZeroWarmResumeResult{
		Harness:               "TestScaleToZeroWarmResume",
		Scenario:              sc,
		Descriptor:            desc,
		HermesColdHibernate:   cold,
		FakWarmRestore:        warm,
		CacheReadFractionLift: warm.CacheReadFraction - cold.CacheReadFraction,
		WarmResumeFloor:       WarmResumeFloor,
	}
}

// DefaultScaleToZeroScenario is the standing witness workload: a session hibernated with a
// warm 8,000-token resident prefix, woken by an 8,192-token first turn (the resident prefix
// plus a short new user turn). The fak arm serves 8,000 of 8,192 tokens from the restored
// prefix (fraction ~0.977, well past the warm floor); the Hermes cold-hibernate arm serves 0.
var DefaultScaleToZeroScenario = ScaleToZeroScenario{
	Session:                "sess-track-d-2853",
	ResidentPrefixTokens:   8000,
	ResumeTurnPromptTokens: 8192,
	TTLTier:                CacheTTL5m,
}
