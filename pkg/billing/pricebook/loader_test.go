package pricebook

import (
	"errors"
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

// sampleBookJSON is a complete v1 operator file with obviously fake rates
// and generic model names: model-a leaves its cache-write rate unpriced
// (null), model-b prices all four lines.
const sampleBookJSON = `{
  "schema_version": 1,
  "models": [
    {
      "id": "model-a",
      "version": 1,
      "source_revision": "test-revision-1",
      "source_time": "2026-01-01T00:00:00Z",
      "currency": "USD",
      "units": "per_million_tokens",
      "fresh_input_rate": 0.01,
      "cache_read_rate": 0.001,
      "cache_write_rate": null,
      "output_rate": 0.02
    },
    {
      "id": "model-b",
      "version": 2,
      "source_revision": "test-revision-2",
      "source_time": "2026-01-02T08:30:00Z",
      "currency": "USD",
      "units": "per_million_tokens",
      "fresh_input_rate": 0.03,
      "cache_read_rate": 0.003,
      "cache_write_rate": 0.04,
      "output_rate": 0.05
    }
  ]
}`

func TestDecodeRoundTrip(t *testing.T) {
	first, err := Decode([]byte(sampleBookJSON))
	if err != nil {
		t.Fatalf("first Decode() = %v", err)
	}
	out, err := first.MarshalJSON()
	if err != nil {
		t.Fatalf("MarshalJSON() = %v", err)
	}
	second, err := Decode(out)
	if err != nil {
		t.Fatalf("second Decode() = %v", err)
	}
	if !reflect.DeepEqual(first, second) {
		t.Fatalf("round trip not identical:\nfirst=%#v\nsecond=%#v", first, second)
	}
	if first.VersionStamp() != second.VersionStamp() {
		t.Fatalf("stamp mismatch after round trip: %s vs %s",
			first.VersionStamp(), second.VersionStamp())
	}
	if !first.Models[0].CacheWriteRate.IsUnpriced() {
		t.Fatalf("null cache_write_rate must stay unpriced, got %#v",
			first.Models[0].CacheWriteRate)
	}
	if first.Models[1].CacheWriteRate.IsUnpriced() {
		t.Fatalf("declared write rate must stay priced, got %#v",
			first.Models[1].CacheWriteRate)
	}
	if first.Models[1].CacheWriteRate.Value != 0.04 {
		t.Fatalf("declared write rate value = %v, want 0.04",
			first.Models[1].CacheWriteRate.Value)
	}
}

func TestDecodeRefusesForeignInput(t *testing.T) {
	cases := map[string]string{
		"top-level unknown field": `{"schema_version":1,"models":[],"extra_field":123}`,
		"card unknown field":      `{"schema_version":1,"models":[{"id":"m","version":1,"source_revision":"r","source_time":"2026-01-01T00:00:00Z","currency":"USD","units":"per_million_tokens","fresh_input_rate":0.01,"cache_read_rate":0.001,"cache_write_rate":null,"output_rate":0.02,"extra_field":true}]}`,
		"trailing data":           sampleBookJSON + `{"schema_version":1}`,
		"truncated json":          `{"schema_version":1,`,
		"empty input":             ``,
	}
	for name, in := range cases {
		t.Run(name, func(t *testing.T) {
			_, err := Decode([]byte(in))
			if !errors.Is(err, ErrMalformedPayload) {
				t.Fatalf("Decode() = %v, want ErrMalformedPayload", err)
			}
		})
	}
}

func TestDecodeRateShapes(t *testing.T) {
	card := func(rate string) string {
		return `{"schema_version":1,"models":[{"id":"m","version":1,"source_revision":"r","source_time":"2026-01-01T00:00:00Z","currency":"USD","units":"per_million_tokens","fresh_input_rate":` + rate + `,"cache_read_rate":null,"cache_write_rate":null,"output_rate":null}]}`
	}
	b, err := Decode([]byte(card(`0.01`)))
	if err != nil {
		t.Fatalf("number rate Decode() = %v", err)
	}
	if b.Models[0].FreshInputRate != PricedRate(0.01) {
		t.Fatalf("number rate = %#v, want PricedRate(0.01)", b.Models[0].FreshInputRate)
	}
	if !b.Models[0].CacheReadRate.IsUnpriced() {
		t.Fatalf("null rate = %#v, want unpriced", b.Models[0].CacheReadRate)
	}

	invalid := []string{`"0.01"`, `true`, `{"value":1}`, `[0.01]`}
	for _, shape := range invalid {
		t.Run(shape, func(t *testing.T) {
			_, err := Decode([]byte(card(shape)))
			if !errors.Is(err, ErrInvalidRate) {
				t.Fatalf("Decode() = %v, want ErrInvalidRate", err)
			}
		})
	}
}

func TestLoadFile(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "book.json")
	if err := os.WriteFile(path, []byte(sampleBookJSON), 0o600); err != nil {
		t.Fatalf("WriteFile: %v", err)
	}
	b, err := LoadFile(path)
	if err != nil {
		t.Fatalf("LoadFile() = %v", err)
	}
	if len(b.Models) != 2 {
		t.Fatalf("LoadFile() loaded %d models, want 2", len(b.Models))
	}
	if _, err := LoadFile(filepath.Join(dir, "missing.json")); err == nil {
		t.Fatal("LoadFile(missing): want error")
	}
}

func TestMarshalJSONRefusesInvalidBook(t *testing.T) {
	b := &PriceBook{SchemaVersion: SchemaVersionV1, Models: []ModelCard{{ID: ""}}}
	if _, err := b.MarshalJSON(); !errors.Is(err, ErrMissingModelID) {
		t.Fatalf("MarshalJSON() = %v, want ErrMissingModelID", err)
	}
}
