package main

import (
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
	}, time.Now())
	for _, row := range rows {
		_ = cachevaluereport.AppendSavings(cachevaluereport.DefaultSavingsLedgerRel, row)
	}
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
