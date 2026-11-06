package pricebook

import (
	"regexp"
	"testing"
)

var hex64 = regexp.MustCompile(`^[0-9a-f]{64}$`)

func TestStampFormat(t *testing.T) {
	b := &PriceBook{SchemaVersion: SchemaVersionV1, Models: []ModelCard{testCard("model-a")}}
	if !hex64.MatchString(b.VersionStamp()) {
		t.Fatalf("stamp %q is not 64 lowercase hex chars", b.VersionStamp())
	}
}

func TestNilBookStampIsEmpty(t *testing.T) {
	var b *PriceBook
	if got := b.VersionStamp(); got != "" {
		t.Fatalf("nil book stamp = %q, want empty", got)
	}
}

func TestStampDeterministic(t *testing.T) {
	a, err := Decode([]byte(sampleBookJSON))
	if err != nil {
		t.Fatalf("Decode() = %v", err)
	}
	b, err := Decode([]byte(sampleBookJSON))
	if err != nil {
		t.Fatalf("Decode() = %v", err)
	}
	if a.VersionStamp() != b.VersionStamp() {
		t.Fatalf("same bytes, different stamps: %s vs %s", a.VersionStamp(), b.VersionStamp())
	}
}

func TestStampIgnoresModelOrder(t *testing.T) {
	a, err := Decode([]byte(sampleBookJSON))
	if err != nil {
		t.Fatalf("Decode() = %v", err)
	}
	reordered := &PriceBook{
		SchemaVersion: SchemaVersionV1,
		Models:        []ModelCard{a.Models[1], a.Models[0]},
	}
	if a.VersionStamp() != reordered.VersionStamp() {
		t.Fatalf("model order must not change the stamp: %s vs %s",
			a.VersionStamp(), reordered.VersionStamp())
	}
}

func TestStampChangesWithContent(t *testing.T) {
	base, err := Decode([]byte(sampleBookJSON))
	if err != nil {
		t.Fatalf("Decode() = %v", err)
	}
	stamp := base.VersionStamp()
	mutate := func(mutate func(*ModelCard)) string {
		b := &PriceBook{SchemaVersion: SchemaVersionV1, Models: []ModelCard{base.Models[0]}}
		mutate(&b.Models[0])
		return b.VersionStamp()
	}
	cases := map[string]func(*ModelCard){
		"rate value":          func(c *ModelCard) { c.FreshInputRate = PricedRate(0.02) },
		"unpriced to priced":  func(c *ModelCard) { c.CacheWriteRate = PricedRate(0.01) },
		"provenance revision": func(c *ModelCard) { c.SourceRevision = "other-revision" },
		"provenance time":     func(c *ModelCard) { c.SourceTime = c.SourceTime.AddDate(0, 1, 0) },
		"card version":        func(c *ModelCard) { c.Version = 2 },
		"currency":            func(c *ModelCard) { c.Currency = "EUR" },
	}
	for name, m := range cases {
		t.Run(name, func(t *testing.T) {
			if got := mutate(m); got == stamp {
				t.Fatalf("%s change did not move the stamp", name)
			}
		})
	}
}

func TestEmptyBookStamps(t *testing.T) {
	b := &PriceBook{SchemaVersion: SchemaVersionV1}
	if !hex64.MatchString(b.VersionStamp()) {
		t.Fatalf("empty book stamp %q is not 64 lowercase hex chars", b.VersionStamp())
	}
}
