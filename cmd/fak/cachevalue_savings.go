package main

import (
	"fmt"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/cachevaluereport"
	"github.com/anthony-chaudhary/fak/internal/gateway"
)

const (
	cachevalueInputPriceEnv  = "FAK_CACHEVALUE_INPUT_PER_MTOK_USD"
	cachevalueOutputPriceEnv = "FAK_CACHEVALUE_OUTPUT_PER_MTOK_USD"
)

func appendObservedCacheSavings(sessionType, provider, context string, sum gateway.AdjudicationSummary) {
	_ = appendObservedCacheSavingsTo(cachevaluereport.DefaultSavingsLedgerRel, sessionType, provider, context, sum, time.Now())
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
		SessionType:          sessionType,
		Provider:             provider,
		Context:              context,
		InputTokens:          sum.InputTokens,
		CacheReadTokens:      sum.CachedPromptTokens,
		CacheCreationTokens:  sum.CacheCreationTokens,
		OutputTokens:         sum.OutputTokens,
		CompactionShedTokens: sum.CompactionShedTokens,
		Pricing:              cachevalueSavingsPricingFromEnv(),
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

func cachevalueSavingsPricingFromEnv() cachevaluereport.SavingsPricing {
	return cachevaluereport.SavingsPricing{
		InputPerMTokUSD:  cachevaluePriceFromEnv(cachevalueInputPriceEnv),
		OutputPerMTokUSD: cachevaluePriceFromEnv(cachevalueOutputPriceEnv),
	}
}

func cachevaluePriceFromEnv(name string) float64 {
	raw := strings.TrimSpace(os.Getenv(name))
	if raw == "" {
		return 0
	}
	v, err := strconv.ParseFloat(raw, 64)
	if err != nil || v < 0 {
		return 0
	}
	return v
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
			dashIfEmpty(rep.Track1.Path))
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
			dashIfEmpty(rep.Track2.Path))
	}
	return b.String()
}

func dashIfEmpty(s string) string {
	if strings.TrimSpace(s) == "" {
		return "-"
	}
	return s
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
