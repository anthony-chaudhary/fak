package pricebook

import (
	"errors"
	"math"
	"testing"
)

// approx compares two analytical figures within relative tolerance; rates
// are floating-point by contract, so exact equality is not asserted.
func approx(t *testing.T, got, want float64) {
	t.Helper()
	if math.Abs(got-want) > 1e-9*math.Max(1, math.Abs(want)) {
		t.Fatalf("got %v, want %v", got, want)
	}
}

// fullyPricedCard prices all four lines with obviously fake rates.
func fullyPricedCard() ModelCard {
	c := testCard("model-x")
	c.CacheWriteRate = PricedRate(0.02)
	return c
}

func TestZeroCountAgainstUnpricedRateIsKnownZero(t *testing.T) {
	card := testCard("model-a") // cache write unpriced
	cost, err := card.Valuate(Counts{FreshInput: 1_000_000, CacheWrite: 0})
	if err != nil {
		t.Fatalf("Valuate() = %v", err)
	}
	if !cost.IsKnown() {
		t.Fatalf("zero write count with unpriced write rate must be known, got %#v", cost)
	}
	approx(t, cost.Amount, 0.01) // only the fresh line bills
}

func TestNonzeroCountAgainstUnpricedRateIsUnknown(t *testing.T) {
	card := testCard("model-a") // cache write unpriced
	cost, err := card.Valuate(Counts{CacheWrite: 1_000_000})
	if err != nil {
		t.Fatalf("Valuate() = %v", err)
	}
	if cost.State != CostUnknown {
		t.Fatalf("nonzero writes with unpriced write rate must be unknown, got %#v", cost)
	}
	if cost.Amount != 0 {
		t.Fatalf("unknown cost must carry no amount, got %v", cost.Amount)
	}
}

func TestAllLinesKnownSum(t *testing.T) {
	card := fullyPricedCard()
	cost, err := card.Valuate(Counts{
		FreshInput: 1_000_000,
		CacheRead:  4_000_000,
		CacheWrite: 1_000_000,
		Output:     100_000,
	})
	if err != nil {
		t.Fatalf("Valuate() = %v", err)
	}
	if !cost.IsKnown() {
		t.Fatalf("fully priced inputs must be known, got %#v", cost)
	}
	// 1M*0.01 + 4M*0.001 + 1M*0.02 + 0.1M*0.02 = 0.01+0.004+0.02+0.002
	approx(t, cost.Amount, 0.036)
}

func TestUnknownModelIsUnknownNotZero(t *testing.T) {
	b := &PriceBook{SchemaVersion: SchemaVersionV1, Models: []ModelCard{testCard("model-a")}}
	cost, err := b.Valuate("model-z", Counts{FreshInput: 1_000_000})
	if err != nil {
		t.Fatalf("Valuate() = %v", err)
	}
	if cost.State != CostUnknown {
		t.Fatalf("unpriced model must yield unknown cost, never a known zero, got %#v", cost)
	}
}

func TestNilBookAndNilCardAreUnknown(t *testing.T) {
	var b *PriceBook
	cost, err := b.Valuate("model-a", Counts{FreshInput: 1_000_000})
	if err != nil {
		t.Fatalf("nil book Valuate() = %v", err)
	}
	if cost.State != CostUnknown {
		t.Fatalf("nil book must yield unknown cost, got %#v", cost)
	}

	var c *ModelCard
	cost, err = c.Valuate(Counts{FreshInput: 1_000_000})
	if err != nil {
		t.Fatalf("nil card Valuate() = %v", err)
	}
	if cost.State != CostUnknown {
		t.Fatalf("nil card must yield unknown cost, got %#v", cost)
	}
}

func TestNegativeCountsRefused(t *testing.T) {
	card := testCard("model-a")
	invalid := []Counts{{FreshInput: -1}, {CacheRead: -1}, {CacheWrite: -1}, {Output: -2}}
	for _, counts := range invalid {
		_, err := card.Valuate(counts)
		if !errors.Is(err, ErrInvalidCounts) {
			t.Fatalf("Valuate(%#v) = %v, want ErrInvalidCounts", counts, err)
		}
		if errors.Is(err, ErrInvalidPriceBook) {
			t.Fatalf("ErrInvalidCounts must not wrap the book root sentinel, got %v", err)
		}
	}
}

func TestDeclaredZeroRateIsKnownZero(t *testing.T) {
	card := testCard("model-a")
	card.OutputRate = PricedRate(0) // declared zero, distinct from unpriced
	cost, err := card.Valuate(Counts{Output: 1_000_000})
	if err != nil {
		t.Fatalf("Valuate() = %v", err)
	}
	if !cost.IsKnown() || cost.Amount != 0 {
		t.Fatalf("declared zero rate times nonzero count must be known zero, got %#v", cost)
	}
}

func TestAllZeroCountsAreKnownZero(t *testing.T) {
	card := testCard("model-a") // cache write unpriced
	cost, err := card.Valuate(Counts{})
	if err != nil {
		t.Fatalf("Valuate() = %v", err)
	}
	if !cost.IsKnown() || cost.Amount != 0 {
		t.Fatalf("all-zero counts must be known zero even with an unpriced line, got %#v", cost)
	}
}

func TestCostAddPropagatesUnknown(t *testing.T) {
	if ZeroCost().Add(UnknownCost()).State != CostUnknown {
		t.Fatal("known plus unknown must be unknown")
	}
	if UnknownCost().Add(UnknownCost()).State != CostUnknown {
		t.Fatal("unknown plus unknown must be unknown")
	}
	sum := KnownCost(0.01).Add(KnownCost(0.02))
	if sum.State != CostKnown {
		t.Fatalf("known plus known must be known, got %#v", sum)
	}
	approx(t, sum.Amount, 0.03)
}

func TestValuateRefusesInvalidCard(t *testing.T) {
	card := testCard("model-a")
	card.FreshInputRate = PricedRate(-1)
	_, err := card.Valuate(Counts{FreshInput: 1})
	if !errors.Is(err, ErrInvalidRate) {
		t.Fatalf("Valuate() on invalid card = %v, want ErrInvalidRate", err)
	}
}
