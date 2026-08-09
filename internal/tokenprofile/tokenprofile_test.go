package tokenprofile

import "testing"

func TestPriceSeparatesEconomicAndSchedulerDominance(t *testing.T) {
	got, err := Price(Forecast{
		InputTokens: 100000, CachedInputTokens: 90000, MaxOutputTokens: 2000,
		Prices:  Prices{InputUncachedPerMillion: 3, InputCachedPerMillion: .3, OutputPerMillion: 10},
		Weights: DefaultWeights(),
	})
	if err != nil {
		t.Fatal(err)
	}
	if got.TotalTokens != 102000 {
		t.Fatalf("total=%d", got.TotalTokens)
	}
	if got.DominantCostClass != InputUncached {
		t.Fatalf("cost class=%s", got.DominantCostClass)
	}
	if got.DominantLoadClass != InputCached {
		t.Fatalf("load class=%s", got.DominantLoadClass)
	}
	if got.WorstCaseUSD != .077 {
		t.Fatalf("cost=%v", got.WorstCaseUSD)
	}
	if got.SchedulerUnits != 40500 {
		t.Fatalf("load=%v", got.SchedulerUnits)
	}
}

func TestPriceRejectsImpossibleCacheForecast(t *testing.T) {
	_, err := Price(Forecast{InputTokens: 3, CachedInputTokens: 4})
	if err == nil {
		t.Fatal("expected error")
	}
}
