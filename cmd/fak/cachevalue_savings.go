package main

import (
	"fmt"
	"strconv"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/cacheobs"
	"github.com/anthony-chaudhary/fak/internal/cachevalueledger"
	"github.com/anthony-chaudhary/fak/internal/cachevaluereport"
	"github.com/anthony-chaudhary/fak/internal/gateway"

	"github.com/anthony-chaudhary/fak/internal/strmatch"
)

const (
	cachevalueInputPriceEnv    = "FAK_CACHEVALUE_INPUT_PER_MTOK_USD"
	cachevalueOutputPriceEnv   = "FAK_CACHEVALUE_OUTPUT_PER_MTOK_USD"
	cachevalueEnvPricingSource = "env:FAK_CACHEVALUE_INPUT_PER_MTOK_USD/FAK_CACHEVALUE_OUTPUT_PER_MTOK_USD"
)

func appendObservedCacheSavings(sessionType, provider, context string, sum gateway.AdjudicationSummary) {
	_ = appendObservedCacheSavingsTo(nightrunLedgerPath(cachevaluereport.DefaultSavingsLedgerRel), sessionType, provider, context, sum, time.Now())
}

type cacheValueAppendResult struct {
	RowsPlanned             int
	RowsWritten             int
	ProviderTokenEquiv      float64
	FakCompactionTokenEquiv float64
	TotalTokenEquiv         float64
	CacheReadTokens         uint64
	CompactionShedTokens    uint64
	Err                     error
}

func appendObservedCacheSavingsTo(path, sessionType, provider, context string, sum gateway.AdjudicationSummary, now time.Time) cacheValueAppendResult {
	rows := cachevaluereport.NewSavingsRows(cachevaluereport.SavingsObservation{
		SessionType:                 sessionType,
		Provider:                    provider,
		Context:                     context,
		InputTokens:                 sum.InputTokens,
		CacheReadTokens:             sum.CachedPromptTokens,
		CacheCreationTokens:         sum.CacheCreationTokens,
		CacheCreationTokensUpgraded: sum.CacheCreationTokensUpgraded,
		OutputTokens:                sum.OutputTokens,
		CompactionShedTokens:        sum.CompactionShedTokens,
		// CompactionCacheReadTokens is the warm witness cacheprice.ShedTokenEquiv prices the
		// shed on. Without it the durable Track-2 row always sees warmWitness=0 and books every
		// shed at FULL_INPUT (1.0x) even on a warm session, diverging ~10x from the live split
		// and guard banner, which DO thread it (cache_pricing.go, serve.go). It is the ONE
		// argument that kept "ShedTokenEquiv is one source" from holding across surfaces by
		// construction (#2794/#2798).
		CompactionCacheReadTokens: sum.CompactionCacheReadTokens,
		CompactionFired:           sum.CompactionFired,
		CompactionBailed:          sum.CompactionBailed,
		CompactionAnchorStarved:   sum.CompactionAnchorStarved,
		CompactionBudget:          sum.CompactionBudget,
		// The seat these list-priced dollars were billed to (#3664), resolved once at
		// startup from the same credential fields the managed-cache posture reads. Blank
		// on a front door that never classified one, which folds NOTIONAL.
		BillingMode: resolvedBillingMode(),
		Pricing:     cachevalueSavingsPricing(provider, context),
	}, now)
	res := cacheValueAppendResult{RowsPlanned: len(rows)}
	for _, row := range rows {
		res.ProviderTokenEquiv += providerTokenEquiv(row)
		if row.Mechanism == "compaction_shed" {
			res.FakCompactionTokenEquiv += row.SavedTokenEquiv
		}
		res.CacheReadTokens += row.CacheReadTokens
		res.CompactionShedTokens += row.CompactionShedTokens
	}
	res.TotalTokenEquiv = res.ProviderTokenEquiv + res.FakCompactionTokenEquiv
	for _, row := range rows {
		if err := cachevaluereport.AppendSavings(path, row); err != nil {
			res.Err = err
			return res
		}
		res.RowsWritten++
	}
	return res
}

func providerTokenEquiv(row cachevaluereport.SavingsRow) float64 {
	if row.Mechanism != "provider_prompt_cache" {
		return 0
	}
	return row.NetSavedTokenEquiv
}

func cachevalueSavingsPricing(provider, context string) cachevaluereport.SavingsPricing {
	input, inputSet := cachevaluePriceFromEnv(cachevalueInputPriceEnv)
	output, outputSet := cachevaluePriceFromEnv(cachevalueOutputPriceEnv)
	if inputSet || outputSet {
		return cachevaluereport.SavingsPricing{
			InputPerMTokUSD:  input,
			OutputPerMTokUSD: output,
			Source:           cachevalueEnvPricingSource,
			DollarBlind:      input == 0 && output == 0,
		}
	}
	if p, source, ok := gateway.DefaultCachePricing(provider, context); ok {
		return cachevaluereport.SavingsPricing{
			InputPerMTokUSD:  p.InputPerMTokUSD,
			OutputPerMTokUSD: p.OutputPerMTokUSD,
			Source:           source,
		}
	}
	// No env override and no known price table for this model: the row is dollar-blind.
	// Stamp an explicit "unpriced:<provider>/<context>" source rather than an opaque
	// "none" so the ledger row names WHICH model it could not price (#2782) — the lint
	// and audit key off DollarBlind, but a human reading the raw JSONL gets the model too.
	return cachevaluereport.SavingsPricing{Source: unpricedSource(provider, context), DollarBlind: true}
}

// unpricedSource renders the "unpriced:<provider>/<context>" pricing_source stamped on
// a dollar-blind row, degrading gracefully when either dimension is blank.
func unpricedSource(provider, context string) string {
	model := strings.TrimSpace(provider)
	if c := strings.TrimSpace(context); c != "" {
		if model == "" {
			model = c
		} else {
			model += "/" + c
		}
	}
	if model == "" {
		return "unpriced"
	}
	return "unpriced:" + model
}

func cachevaluePriceFromEnv(name string) (float64, bool) {
	return priceFromEnv(name, false)
}

type cacheValuePersistenceReport struct {
	Since  string
	Track1 cacheValueTrack1Result
	Track2 cacheValueTrack2Result
}

type cacheValueTrack1Result struct {
	Path         string
	RowsPlanned  int
	RowsWritten  int
	Turns        uint64
	PromptTokens uint64
	ReusedTokens uint64
}

type cacheValueTrack2Result struct {
	Path                    string
	RowsPlanned             int
	RowsWritten             int
	ProviderTokenEquiv      float64
	FakCompactionTokenEquiv float64
	TotalTokenEquiv         float64
	CacheReadTokens         uint64
	CompactionShedTokens    uint64
}

// buildCacheValuePersistenceReport persists BOTH cache-value tracks for a finished
// session and returns a report naming what landed, so a live exit path can SURFACE the
// savings the operator just earned instead of writing it silently to the ledger and
// throwing the result away. Track 1 is the WITNESSED KV-prefix-reuse kernel row
// (cachevalueledger); Track 2 is the OBSERVED-$ provider-cache + compaction rows
// (appendObservedCacheSavingsTo). Before this wire, guard and serve exits computed both
// results and discarded them (`_ = ...`), leaving the dollar-aware summary formatter
// below orphaned — referenced only by its own test, never by a running session (the
// classic fak detection-without-enforcement gap). Best-effort: a ledger write failure
// never fails the session — the report still names the planned rows, and a
// RowsWritten<RowsPlanned gap tells the reader the write did not land.
//
// track1Rel is passed in because guard and serve resolve the Track-1 ledger path
// differently today (guard writes the bare relative rel; serve wraps it in
// nightrunLedgerPath) — this helper preserves each caller's existing target rather than
// silently relocating either ledger.
func buildCacheValuePersistenceReport(srv *gateway.Server, kind, name, provider, track1Rel string, now time.Time) cacheValuePersistenceReport {
	rep := cacheValuePersistenceReport{Since: now.UTC().Format("2006-01-02")}

	stats := cacheobs.Default.Snapshot()
	rep.Track1 = cacheValueTrack1Result{Path: track1Rel}
	if stats.Turns > 0 {
		rep.Track1.RowsPlanned = 1
		rep.Track1.Turns = stats.Turns
		rep.Track1.PromptTokens = stats.PromptTokens
		rep.Track1.ReusedTokens = stats.ReusedTokens
		if err := cachevalueledger.Append(kind, name, track1Rel, stats); err == nil {
			rep.Track1.RowsWritten = 1
		}
	}

	track2Path := nightrunLedgerPath(cachevaluereport.DefaultSavingsLedgerRel)
	t2 := appendObservedCacheSavingsTo(track2Path, kind, provider, name, srv.AdjudicationSummary(), now)
	rep.Track2 = cacheValueTrack2Result{
		Path:                    track2Path,
		RowsPlanned:             t2.RowsPlanned,
		RowsWritten:             t2.RowsWritten,
		ProviderTokenEquiv:      t2.ProviderTokenEquiv,
		FakCompactionTokenEquiv: t2.FakCompactionTokenEquiv,
		TotalTokenEquiv:         t2.TotalTokenEquiv,
		CacheReadTokens:         t2.CacheReadTokens,
		CompactionShedTokens:    t2.CompactionShedTokens,
	}
	return rep
}

func formatCacheValuePersistenceSummary(label string, rep cacheValuePersistenceReport) string {
	if strings.TrimSpace(label) == "" {
		label = "fak"
	}
	var b strings.Builder
	fmt.Fprintf(&b, "%s: cache-value evidence\n", label)
	if rep.Track1.RowsWritten > 0 {
		fmt.Fprintf(&b, "  Track 1 WITNESSED kernel row: wrote %d/%d to %s (%s turns, %s prompt tok, %s reused tok)\n",
			rep.Track1.RowsWritten, rep.Track1.RowsPlanned, rep.Track1.Path,
			formatWhole(rep.Track1.Turns), formatWhole(rep.Track1.PromptTokens), formatWhole(rep.Track1.ReusedTokens))
	} else {
		fmt.Fprintf(&b, "  Track 1 WITNESSED kernel row not written: no KV-prefix reuse turns for %s\n",
			strmatch.DashIfBlank(rep.Track1.Path))
	}
	if rep.Track2.RowsWritten > 0 {
		fmt.Fprintf(&b, "  Track 2 OBSERVED-$ rows: wrote %d/%d to %s (provider %s tok-eq, fak compaction %s tok-eq, total %s tok-eq; cache_read %s tok, compact_shed %s tok)\n",
			rep.Track2.RowsWritten, rep.Track2.RowsPlanned, rep.Track2.Path,
			formatSignedWholeFloat(rep.Track2.ProviderTokenEquiv),
			formatSignedWholeFloat(rep.Track2.FakCompactionTokenEquiv),
			formatSignedWholeFloat(rep.Track2.TotalTokenEquiv),
			formatWhole(rep.Track2.CacheReadTokens),
			formatWhole(rep.Track2.CompactionShedTokens))
		if rep.Since != "" {
			fmt.Fprintf(&b, "  next: fak cachevalue report --since %s\n", rep.Since)
		}
	} else {
		fmt.Fprintf(&b, "  Track 2 OBSERVED-$ row not written: no provider-cache or compaction tokens for %s\n",
			strmatch.DashIfBlank(rep.Track2.Path))
	}
	return b.String()
}

func formatSignedWholeFloat(v float64) string {
	if v < 0 {
		return "-" + formatWhole(uint64(-v+0.5))
	}
	return "+" + formatWhole(uint64(v+0.5))
}

func formatWhole(v uint64) string {
	s := strconv.FormatUint(v, 10)
	if len(s) <= 3 {
		return s
	}
	var b strings.Builder
	pre := len(s) % 3
	if pre == 0 {
		pre = 3
	}
	b.WriteString(s[:pre])
	for i := pre; i < len(s); i += 3 {
		b.WriteByte(',')
		b.WriteString(s[i : i+3])
	}
	return b.String()
}
