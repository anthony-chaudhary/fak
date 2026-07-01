// Package cachewitness reads a live fak gateway's /metrics surface and folds the
// in-kernel KV-prefix cache family into ONE provenance-labeled evidence record:
// the cache VALUE a fak-served model (e.g. GLM-5.2 on the pure kernel) realized
// across a run, split by what fak CONTROLS from what it only OBSERVES.
//
// This is the observation seam for the GLM-5.2-fak-kernel-cache epic (#1010,
// child #1011): an agentic run against a fak serve gateway (the Claude harness
// driving `fak swebench run --agent fleet`, or `fak guard --base-url`) repeats a
// large system+tools+repo prefix across the run; fak's RadixAttention reports
// aggregate reused prefix tokens from the cached KV — the prefill the kernel did
// NOT redo. That reused-token count is the cache value the epic measures.
//
// The provenance discipline (the DOS / conflation-scorecard line, drawn the same
// way internal/gateway/metrics.go draws it):
//
//   - WITNESSED — fak_gateway_kv_prefix_reused_tokens_total. fak's OWN cache: the
//     RadixAttention prefix match, authored by fak's planner on every in-kernel
//     turn. fak controls this number, so it is WITNESSED.
//   - OBSERVED — fak_gateway_inference_cached_prompt_tokens_total. The upstream
//     PROVIDER's cache_read, relayed verbatim. On the pure in-kernel path there is
//     no provider, so this reads 0; on a proxy path it is the provider's doing,
//     never fak's. Either way it is OBSERVED, not proof fak preserved anything.
//
// The two are DISTINCT signals over distinct caches; a record that summed them
// would conflate the trust classes, so this package keeps them in separate fields
// and never derives one from the other.
package cachewitness

import (
	"bufio"
	"fmt"
	"strconv"
	"strings"
)

// Provenance labels the trust class of a cache number: who authored it.
type Provenance string

const (
	// Witnessed marks a number fak itself produced (its own kernel/cache).
	Witnessed Provenance = "WITNESSED"
	// Observed marks a number relayed from an external party (the provider).
	Observed Provenance = "OBSERVED"
	// Modeled marks a projected number that has not been measured on the device.
	Modeled Provenance = "MODELED"
)

// Record is the provenance-labeled cache-value evidence folded from a gateway's
// /metrics. It is the unit that graduates into a result packet: every cache
// number carries its provenance so a downstream reader (or the conflation
// scorecard) can never mistake fak's own reuse for the provider's.
type Record struct {
	// GatewayURL is the /metrics endpoint this record was scraped from.
	GatewayURL string `json:"gateway_url"`

	// WitnessWindow records the baseline/end scrape pair when this record is a
	// run delta rather than a whole-gateway cumulative snapshot.
	WitnessWindow *WitnessWindow `json:"witness_window,omitempty"`

	// GatewayUptimeTurns is the end-scrape cumulative gateway turn counter. When
	// WitnessWindow is present, KVPrefix.Turns is the run delta and this field
	// shows how much prior gateway lifetime was excluded.
	GatewayUptimeTurns uint64 `json:"gateway_uptime_turns"`

	// --- WITNESSED: fak's OWN in-kernel KV-prefix cache (the epic's lever) ---

	// KVPrefix is the in-kernel RadixAttention cache family. Provenance: WITNESSED.
	KVPrefix KVPrefixWitness `json:"kv_prefix"`

	// CacheBitScope names the exact guarantee behind CacheBit(): aggregate
	// run/window-level KV-prefix reuse, not solved-ticket turn attribution.
	CacheBitScope string `json:"cache_bit_scope"`

	// --- OBSERVED: the upstream provider's prompt cache (relayed, not fak's) ---

	// ProviderCacheReadTokens is the cumulative cache_read the upstream provider
	// served, relayed verbatim. Provenance: OBSERVED. 0 on the pure in-kernel path.
	ProviderCacheReadTokens uint64 `json:"provider_cache_read_tokens"`

	// CacheValue is the PUBLISHABLE view of the realized reuse, framed in the only
	// family #1066's honesty fence permits (marginal-over-tuned-warm-KV) and
	// self-fencing against the forbidden vs-naive multiple. Derived from KVPrefix.
	CacheValue CacheValue `json:"cache_value"`

	// Provenance maps each top-level number to its trust class, so the record is
	// self-describing — a reader never has to know which field is fak's.
	Provenance map[string]Provenance `json:"provenance"`
}

// WitnessWindow names the cumulative metrics scrape used as the start baseline
// and the scrape used as the end of the measured run window.
type WitnessWindow struct {
	StartScrape string `json:"start_scrape"`
	EndScrape   string `json:"end_scrape"`
}

// WarmKVMarginalFamily names the ONLY cache-value framing #1066's honesty fence
// permits for a *published* number: fak's marginal over a tuned warm-KV server
// (the internal/swebench cost.go B→C lever). A tuned warm-KV server (SGLang/llama
// with kv_unified) already keeps per-agent KV across turns, so it ALREADY earns a
// long trajectory's turn-over-turn reuse; fak's *marginal* over it is the
// cross-worker shared-prefix win, which is ~1.0x on a single session and grows to
// ~1.0–1.31x at modest fleets, reaching ~4.1x only on a 50×5 fleet.
const WarmKVMarginalFamily = "marginal-over-tuned-warm-KV (internal/swebench cost.go B/C; ~1.0x single-session, ~1.0-1.31x modest fleet, ~4.1x only on a 50x5 fleet)"

// CacheBitScopeAggregateRun is the scope the gateway /metrics scrape can
// witness: aggregate KV-prefix reuse across the parsed run or witness window.
// It deliberately does not claim per-turn solved-ticket attribution.
const CacheBitScopeAggregateRun = "aggregate-run-kv-prefix-reuse"

// CacheValue expresses the realized in-kernel reuse as a *publishable* cache
// number under #1066's honesty fence. The trap it guards: a long R2E-Gym
// trajectory has high turn-over-turn reuse (ReuseRatio → 0.9+), and 1/(1-reuse)
// reads as a big "speedup" — but that is the vs-NAIVE re-prefill multiple (cost.go
// A→C, the ~17.9–23.4x band), NOT a cache value, because no real server re-prefills
// the whole context every turn. The honest published number is the marginal over a
// *tuned warm-KV* server (B→C), so this view reports the realized reuse as
// WITNESSED data while pinning the publishable framing to that family and refusing
// to surface the vs-naive multiple.
type CacheValue struct {
	// ReusedTokens / PromptTokens / ReuseRatio echo the WITNESSED realized reuse —
	// fak's own RadixAttention prefix match on this run. ReuseRatio is the fraction
	// of this session's prefill the kernel did not redo (the warm-KV behavior itself).
	ReusedTokens uint64  `json:"kv_prefix_reused_tokens"`
	PromptTokens uint64  `json:"prompt_tokens"`
	ReuseRatio   float64 `json:"realized_reuse_ratio"`

	// PublishableValueFamily is the ONLY framing a published cache-value may take
	// (WarmKVMarginalFamily); SingleSessionMarginalX is the honest marginal for a
	// single live trajectory (1.0 — a tuned warm-KV server gets the same turn reuse);
	// FleetMarginalSource points at where the >1.0x fleet number is actually computed
	// (cost.AggregatePrefill B/C across worker counts), since a single-session
	// /metrics scrape cannot see cross-worker sharing.
	PublishableValueFamily string  `json:"publishable_value_family"`
	SingleSessionMarginalX float64 `json:"single_session_marginal_over_warm_kv_x"`
	FleetMarginalSource    string  `json:"fleet_marginal_source"`

	// VsNaiveMultipleExcluded records that the forbidden 1/(1-reuse) vs-naive
	// re-prefill multiple is deliberately NOT surfaced here (the #1066 fence), and
	// Note carries the one-line reason a reader can cite.
	VsNaiveMultipleExcluded bool   `json:"vs_naive_multiple_excluded"`
	Note                    string `json:"note"`
}

// cacheValue derives the publishable CacheValue view from the witnessed KV-prefix
// reuse. It never computes the vs-naive multiple — the omission is the fence.
func cacheValue(k KVPrefixWitness) CacheValue {
	return CacheValue{
		ReusedTokens:            k.ReusedTokens,
		PromptTokens:            k.PromptTokens,
		ReuseRatio:              k.ReuseRatio(),
		PublishableValueFamily:  WarmKVMarginalFamily,
		SingleSessionMarginalX:  1.0,
		FleetMarginalSource:     "internal/swebench cost.AggregatePrefill (B/C across worker counts)",
		VsNaiveMultipleExcluded: true,
		Note:                    "Realized reuse is WITNESSED; a tuned warm-KV server earns the same turn-over-turn reuse on a single trajectory, so fak's marginal over it is ~1.0x here. The >1.0x cache value is cross-worker shared-prefix (cost.go B/C). The vs-naive re-prefill multiple (1/(1-reuse), the ~17.9-23.4x band) is NOT a cache value and is excluded per #1066.",
	}
}

// KVPrefixWitness is fak's own in-kernel KV-prefix reuse, all WITNESSED.
type KVPrefixWitness struct {
	// Turns is the number of in-kernel model turns observed for prefix reuse.
	Turns uint64 `json:"turns"`
	// PromptTokens is the prefill tokens summed across those turns (the denominator).
	PromptTokens uint64 `json:"prompt_tokens"`
	// ReusedTokens is THE cache value: prefill tokens served from the cached KV
	// prefix — the work the kernel did not redo. This is the epic's headline datum.
	ReusedTokens uint64 `json:"reused_tokens"`
	// FrozenTurns/PartialTurns/ColdTurns is the cliff distribution: frozen turns
	// (reuse >= 0.90) are the append-only regime the cache value comes from; cold
	// turns (reuse < 0.10) are first prefills or head-mutated/fanned-out turns.
	FrozenTurns  uint64 `json:"frozen_turns"`
	PartialTurns uint64 `json:"partial_turns"`
	ColdTurns    uint64 `json:"cold_turns"`
}

// ReuseRatio is the realized in-kernel cache-hit: reused / prompt tokens. It is
// the fraction of prefill work the kernel skipped. Returns 0 when no in-kernel
// turn was observed (PromptTokens == 0), never a divide-by-zero.
func (k KVPrefixWitness) ReuseRatio() float64 {
	if k.PromptTokens == 0 {
		return 0
	}
	return float64(k.ReusedTokens) / float64(k.PromptTokens)
}

// CacheBit reports whether fak's own cache engaged in the aggregate run/window:
// the parsed /metrics scrape has non-zero KV-prefix reused tokens. The gateway
// metrics consumed here do not identify which solved-ticket turn produced reuse,
// so this bit is intentionally not a per-turn attribution witness. A run with
// ReusedTokens == 0 means the cache never engaged (all-cold), which is reported
// honestly as "did not bite".
func (r Record) CacheBit() bool {
	return r.KVPrefix.ReusedTokens > 0
}

// Sub returns the per-run delta from a cumulative end scrape and a cumulative
// baseline scrape. Prometheus counters may reset across a gateway restart; when
// an end counter is lower than the baseline, the end value is treated as the
// post-reset run value rather than underflowing.
func (r Record) Sub(baseline Record) Record {
	out := r
	out.KVPrefix = r.KVPrefix.Sub(baseline.KVPrefix)
	out.ProviderCacheReadTokens = counterDelta(r.ProviderCacheReadTokens, baseline.ProviderCacheReadTokens)
	out.GatewayUptimeTurns = r.GatewayUptimeTurns
	if out.GatewayUptimeTurns == 0 {
		out.GatewayUptimeTurns = r.KVPrefix.Turns
	}
	out.WitnessWindow = &WitnessWindow{StartScrape: baseline.GatewayURL, EndScrape: r.GatewayURL}
	if out.CacheBitScope == "" {
		out.CacheBitScope = CacheBitScopeAggregateRun
	}
	out.CacheValue = cacheValue(out.KVPrefix)
	if out.Provenance == nil {
		out.Provenance = map[string]Provenance{
			"kv_prefix":                  Witnessed,
			"provider_cache_read_tokens": Observed,
			"cache_value":                Witnessed,
		}
	}
	return out
}

// Sub returns a per-window in-kernel KV-prefix delta from cumulative counters.
func (k KVPrefixWitness) Sub(baseline KVPrefixWitness) KVPrefixWitness {
	return KVPrefixWitness{
		Turns:        counterDelta(k.Turns, baseline.Turns),
		PromptTokens: counterDelta(k.PromptTokens, baseline.PromptTokens),
		ReusedTokens: counterDelta(k.ReusedTokens, baseline.ReusedTokens),
		FrozenTurns:  counterDelta(k.FrozenTurns, baseline.FrozenTurns),
		PartialTurns: counterDelta(k.PartialTurns, baseline.PartialTurns),
		ColdTurns:    counterDelta(k.ColdTurns, baseline.ColdTurns),
	}
}

func counterDelta(end, start uint64) uint64 {
	if end >= start {
		return end - start
	}
	return end
}

// metric name → field the scraper folds it into.
const (
	mTurns      = "fak_gateway_kv_prefix_turns_total"
	mPromptTok  = "fak_gateway_kv_prefix_prompt_tokens_total"
	mReusedTok  = "fak_gateway_kv_prefix_reused_tokens_total"
	mByRegime   = "fak_gateway_kv_prefix_turns_by_regime_total"
	mProviderRd = "fak_gateway_inference_cached_prompt_tokens_total"
)

var requiredKVPrefixSeries = []string{mTurns, mPromptTok, mReusedTok}

// Parse folds a gateway /metrics body (Prometheus text exposition) into a Record.
// It reads only the cache family this package owns and ignores every other
// series, so it is robust to the rest of the gateway's metric surface changing.
// gatewayURL is recorded verbatim for provenance.
func Parse(gatewayURL string, metricsBody string) (Record, error) {
	r := Record{
		GatewayURL:    gatewayURL,
		CacheBitScope: CacheBitScopeAggregateRun,
		Provenance: map[string]Provenance{
			"kv_prefix":                  Witnessed,
			"provider_cache_read_tokens": Observed,
			"cache_value":                Witnessed,
		},
	}
	sc := bufio.NewScanner(strings.NewReader(metricsBody))
	sc.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	seen := false
	seenKVPrefix := false
	seenRequired := map[string]bool{}
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
		case mTurns:
			r.KVPrefix.Turns = val
			seen = true
			seenKVPrefix = true
			seenRequired[name] = true
		case mPromptTok:
			r.KVPrefix.PromptTokens = val
			seen = true
			seenKVPrefix = true
			seenRequired[name] = true
		case mReusedTok:
			r.KVPrefix.ReusedTokens = val
			seen = true
			seenKVPrefix = true
			seenRequired[name] = true
		case mProviderRd:
			r.ProviderCacheReadTokens = val
			seen = true
		case mByRegime:
			seenKVPrefix = true
			switch regimeLabel(label) {
			case "frozen":
				r.KVPrefix.FrozenTurns = val
			case "partial":
				r.KVPrefix.PartialTurns = val
			case "cold":
				r.KVPrefix.ColdTurns = val
			}
			seen = true
		}
	}
	if err := sc.Err(); err != nil {
		return Record{}, fmt.Errorf("scan metrics: %w", err)
	}
	if !seen {
		return Record{}, fmt.Errorf("no fak_gateway_kv_prefix_* / inference cache series found in %s — is this a fak gateway /metrics body?", gatewayURL)
	}
	if seenKVPrefix {
		var missing []string
		for _, name := range requiredKVPrefixSeries {
			if !seenRequired[name] {
				missing = append(missing, name)
			}
		}
		if len(missing) > 0 {
			return Record{}, fmt.Errorf("missing required cachewitness series in %s: %s", gatewayURL, strings.Join(missing, ", "))
		}
	}
	r.GatewayUptimeTurns = r.KVPrefix.Turns
	r.CacheValue = cacheValue(r.KVPrefix)
	return r, nil
}

// splitMetricLine parses one Prometheus sample line "name{labels} value" into
// the metric name, the raw label block (without braces, "" if none), and the
// integer value. A non-integer / malformed value yields ok=false (skipped) so a
// histogram bucket or float gauge elsewhere in the body never aborts the scrape.
func splitMetricLine(line string) (name, label string, val uint64, ok bool) {
	// Value is the last whitespace-separated token.
	sp := strings.LastIndexAny(line, " \t")
	if sp < 0 {
		return "", "", 0, false
	}
	head := strings.TrimSpace(line[:sp])
	valStr := strings.TrimSpace(line[sp+1:])
	// Counters are integers in this family; parse as float then floor to be
	// tolerant of a "123.0" rendering, but reject genuinely fractional values.
	f, err := strconv.ParseFloat(valStr, 64)
	if err != nil || f < 0 || f != float64(uint64(f)) {
		return "", "", 0, false
	}
	val = uint64(f)
	if i := strings.IndexByte(head, '{'); i >= 0 {
		name = head[:i]
		label = strings.TrimSuffix(head[i+1:], "}")
	} else {
		name = head
	}
	return name, label, val, true
}

// regimeLabel pulls the regime value out of a label block like `regime="frozen"`.
func regimeLabel(label string) string {
	const key = `regime="`
	i := strings.Index(label, key)
	if i < 0 {
		return ""
	}
	rest := label[i+len(key):]
	if j := strings.IndexByte(rest, '"'); j >= 0 {
		return rest[:j]
	}
	return ""
}
