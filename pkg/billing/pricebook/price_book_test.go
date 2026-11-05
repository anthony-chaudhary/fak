package pricebook

import (
	"errors"
	"math"
	"testing"
	"time"
)

// testTime is a fixed, obviously fake provenance time for fixtures.
var testTime = time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC)

// testCard returns a valid card with obviously fake rates and a generic
// model name: fresh input and output priced, cache read priced low, cache
// write unpriced (the common shape: a read discount does not establish a
// write price).
func testCard(id string) ModelCard {
	return ModelCard{
		ID:             id,
		Version:        1,
		SourceRevision: "test-revision-1",
		SourceTime:     testTime,
		Currency:       "USD",
		Units:          UnitsPerMillionTokens,
		FreshInputRate: PricedRate(0.01),
		CacheReadRate:  PricedRate(0.001),
		CacheWriteRate: UnpricedRate(),
		OutputRate:     PricedRate(0.02),
	}
}

func TestNilBookValidate(t *testing.T) {
	var b *PriceBook
	if err := b.Validate(); !errors.Is(err, ErrInvalidPriceBook) {
		t.Fatalf("nil book Validate() = %v, want ErrInvalidPriceBook", err)
	}
}

func TestBookValidationRules(t *testing.T) {
	base := func() *PriceBook {
		return &PriceBook{SchemaVersion: SchemaVersionV1, Models: []ModelCard{testCard("model-a")}}
	}
	cases := []struct {
		name   string
		mutate func(b *PriceBook)
		want   error
	}{
		{"unknown schema version", func(b *PriceBook) { b.SchemaVersion = 2 }, ErrUnknownSchemaVersion},
		{"zero schema version", func(b *PriceBook) { b.SchemaVersion = 0 }, ErrUnknownSchemaVersion},
		{"missing model id", func(b *PriceBook) { b.Models[0].ID = "" }, ErrMissingModelID},
		{"duplicate model id", func(b *PriceBook) {
			b.Models = append(b.Models, testCard("model-a"))
		}, ErrDuplicateModelID},
		{"zero model version", func(b *PriceBook) { b.Models[0].Version = 0 }, ErrInvalidModelVersion},
		{"negative model version", func(b *PriceBook) { b.Models[0].Version = -3 }, ErrInvalidModelVersion},
		{"missing source revision", func(b *PriceBook) { b.Models[0].SourceRevision = "" }, ErrMissingSourceRevision},
		{"missing source time", func(b *PriceBook) { b.Models[0].SourceTime = time.Time{} }, ErrMissingSourceTime},
		{"lowercase currency", func(b *PriceBook) { b.Models[0].Currency = "usd" }, ErrInvalidCurrency},
		{"short currency", func(b *PriceBook) { b.Models[0].Currency = "US" }, ErrInvalidCurrency},
		{"long currency", func(b *PriceBook) { b.Models[0].Currency = "USD1" }, ErrInvalidCurrency},
		{"currency with digit", func(b *PriceBook) { b.Models[0].Currency = "U5D" }, ErrInvalidCurrency},
		{"unknown units", func(b *PriceBook) { b.Models[0].Units = "per_token" }, ErrUnknownUnits},
		{"empty units", func(b *PriceBook) { b.Models[0].Units = "" }, ErrUnknownUnits},
		{"negative fresh rate", func(b *PriceBook) { b.Models[0].FreshInputRate = PricedRate(-0.01) }, ErrInvalidRate},
		{"unpriced carrying value", func(b *PriceBook) {
			b.Models[0].CacheWriteRate = Rate{Status: RateUnpriced, Value: 0.5}
		}, ErrInvalidRate},
		{"unknown rate status", func(b *PriceBook) {
			b.Models[0].OutputRate = Rate{Status: "bogus"}
		}, ErrInvalidRate},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			b := base()
			tc.mutate(b)
			err := b.Validate()
			if !errors.Is(err, tc.want) {
				t.Fatalf("Validate() = %v, want errors.Is(%v)", err, tc.want)
			}
			if !errors.Is(err, ErrInvalidPriceBook) {
				t.Fatalf("Validate() = %v, want root ErrInvalidPriceBook", err)
			}
		})
	}
}

func TestValidBookPasses(t *testing.T) {
	b := &PriceBook{
		SchemaVersion: SchemaVersionV1,
		Models:        []ModelCard{testCard("model-a"), testCard("model-b")},
	}
	if err := b.Validate(); err != nil {
		t.Fatalf("Validate() = %v, want nil", err)
	}
}

func TestEmptyBookIsValid(t *testing.T) {
	b := &PriceBook{SchemaVersion: SchemaVersionV1}
	if err := b.Validate(); err != nil {
		t.Fatalf("empty book Validate() = %v, want nil", err)
	}
}

func TestCardLookup(t *testing.T) {
	b := &PriceBook{
		SchemaVersion: SchemaVersionV1,
		Models:        []ModelCard{testCard("model-a"), testCard("model-b")},
	}
	if _, ok := b.Card("model-a"); !ok {
		t.Fatal("Card(model-a): want found")
	}
	if _, ok := b.Card("model-z"); ok {
		t.Fatal("Card(model-z): want miss")
	}
	var nilBook *PriceBook
	if _, ok := nilBook.Card("model-a"); ok {
		t.Fatal("nil book Card: want miss")
	}
}

func TestRateValid(t *testing.T) {
	valid := []Rate{PricedRate(0.01), PricedRate(0), UnpricedRate()}
	for _, r := range valid {
		if err := r.Valid(); err != nil {
			t.Fatalf("%#v Valid() = %v, want nil", r, err)
		}
	}
	invalid := []Rate{
		PricedRate(-0.01),
		PricedRate(math.NaN()),
		PricedRate(math.Inf(1)),
		Rate{Status: RateUnpriced, Value: 0.5},
		Rate{Status: "bogus"},
	}
	for _, r := range invalid {
		if err := r.Valid(); !errors.Is(err, ErrInvalidRate) {
			t.Fatalf("%#v Valid() = %v, want ErrInvalidRate", r, err)
		}
	}
}

func TestClosedVocabularies(t *testing.T) {
	if !RatePriced.Valid() || !RateUnpriced.Valid() || RateStatus("bogus").Valid() {
		t.Fatal("rate-status vocabulary broken")
	}
	if len(RateStatuses) != 2 {
		t.Fatalf("RateStatuses has %d members, want 2", len(RateStatuses))
	}
	if !CostKnown.Valid() || !CostUnknown.Valid() || CostState("bogus").Valid() {
		t.Fatal("cost-state vocabulary broken")
	}
	if len(CostStates) != 2 {
		t.Fatalf("CostStates has %d members, want 2", len(CostStates))
	}
	if !UnitsPerMillionTokens.Valid() || Units("per_token").Valid() {
		t.Fatal("units vocabulary broken")
	}
}
