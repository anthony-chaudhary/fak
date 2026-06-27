// Package cachewitness reads a live fak gateway's /metrics surface and folds the
// in-kernel KV-prefix cache family into ONE provenance-labeled evidence record:
// the cache VALUE a fak-served model (e.g. GLM-5.2 on the pure kernel) realized
// across a run, split by what fak CONTROLS from what it only OBSERVES.
//
// This is the observation seam for the GLM-5.2-fak-kernel-cache epic (#1010,
// child #1011): an agentic run against a fak serve gateway (the Claude harness
// driving `fak swebench run --agent fleet`, or `fak guard --base-url`) repeats a
// large system+tools+repo prefix every turn; fak's RadixAttention serves that
// prefix from the cached KV on turns 2..N — the prefill the kernel did NOT redo.
// That reused-token count is the cache value the epic measures.
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
)

// Record is the provenance-labeled cache-value evidence folded from a gateway's
// /metrics. It is the unit that graduates into a result packet: every cache
// number carries its provenance so a downstream reader (or the conflation
// scorecard) can never mistake fak's own reuse for the provider's.
type Record struct {
	// GatewayURL is the /metrics endpoint this record was scraped from.
	GatewayURL string `json:"gateway_url"`

	// --- WITNESSED: fak's OWN in-kernel KV-prefix cache (the epic's lever) ---

	// KVPrefix is the in-kernel RadixAttention cache family. Provenance: WITNESSED.
	KVPrefix KVPrefixWitness `json:"kv_prefix"`

	// --- OBSERVED: the upstream provider's prompt cache (relayed, not fak's) ---

	// ProviderCacheReadTokens is the cumulative cache_read the upstream provider
	// served, relayed verbatim. Provenance: OBSERVED. 0 on the pure in-kernel path.
	ProviderCacheReadTokens uint64 `json:"provider_cache_read_tokens"`

	// Provenance maps each top-level number to its trust class, so the record is
	// self-describing — a reader never has to know which field is fak's.
	Provenance map[string]Provenance `json:"provenance"`
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

// CacheBit reports whether fak's own cache actually bit on this run: at least one
// turn reused a non-zero prefix. This is the milestone-2 witness for #1012 — the
// cache biting on the prefix of a real solved-ticket turn — independent of
// whether the full patch was generated. A run with ReusedTokens == 0 means the
// cache never engaged (all-cold), which is reported honestly as "did not bite".
func (r Record) CacheBit() bool {
	return r.KVPrefix.ReusedTokens > 0
}

// metric name → field the scraper folds it into.
const (
	mTurns      = "fak_gateway_kv_prefix_turns_total"
	mPromptTok  = "fak_gateway_kv_prefix_prompt_tokens_total"
	mReusedTok  = "fak_gateway_kv_prefix_reused_tokens_total"
	mByRegime   = "fak_gateway_kv_prefix_turns_by_regime_total"
	mProviderRd = "fak_gateway_inference_cached_prompt_tokens_total"
)

// Parse folds a gateway /metrics body (Prometheus text exposition) into a Record.
// It reads only the cache family this package owns and ignores every other
// series, so it is robust to the rest of the gateway's metric surface changing.
// gatewayURL is recorded verbatim for provenance.
func Parse(gatewayURL string, metricsBody string) (Record, error) {
	r := Record{
		GatewayURL: gatewayURL,
		Provenance: map[string]Provenance{
			"kv_prefix":                  Witnessed,
			"provider_cache_read_tokens": Observed,
		},
	}
	sc := bufio.NewScanner(strings.NewReader(metricsBody))
	sc.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	seen := false
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
		case mPromptTok:
			r.KVPrefix.PromptTokens = val
			seen = true
		case mReusedTok:
			r.KVPrefix.ReusedTokens = val
			seen = true
		case mProviderRd:
			r.ProviderCacheReadTokens = val
			seen = true
		case mByRegime:
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
