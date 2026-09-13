package moecache_test

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/pkg/moecache"
)

// TestSchemaConstant pins the live-engine wire-format identifier. It is
// deliberately distinct from the benchmark receipt `fak.moe-expert-cache/v1`
// (those pin one offline measurement) while sharing its hit/recall vocabulary.
func TestSchemaConstant(t *testing.T) {
	if moecache.Schema != "fak.moe-expert-cache-telemetry/v1" {
		t.Fatalf("schema constant drifted: %q", moecache.Schema)
	}
	if got := moecache.New().Schema; got != moecache.Schema {
		t.Fatalf("New().Schema = %q, want %q", got, moecache.Schema)
	}
	if moecache.Schema == "fak.moe-expert-cache/v1" {
		t.Fatal("telemetry schema must not collide with the benchmark receipt schema")
	}
}

// TestTierVocabularyIsClosed pins the ordered waterfall vocabulary and the
// Valid() gate.
func TestTierVocabularyIsClosed(t *testing.T) {
	want := []moecache.Tier{moecache.TierDRAM, moecache.TierCheckpoint, moecache.TierNVMe}
	if len(moecache.Tiers) != len(want) {
		t.Fatalf("tier vocabulary has %d members, want %d", len(moecache.Tiers), len(want))
	}
	for i, tier := range want {
		if moecache.Tiers[i] != tier {
			t.Fatalf("tiers[%d] = %q, want %q", i, moecache.Tiers[i], tier)
		}
		if !tier.Valid() {
			t.Fatalf("declared tier %q reports Valid()=false", tier)
		}
	}
	if moecache.Tier("bogus").Valid() {
		t.Fatal("unknown tier reports Valid()=true")
	}
	if moecache.Tier("").Valid() {
		t.Fatal("empty tier reports Valid()=true")
	}
}

// TestUnknownMetricNeverSerializesAsZero is the load-bearing invariant: an
// unknown metric must carry known:false + a reason and MUST NOT emit a number,
// so absent can never be read as a fabricated 0.
func TestUnknownMetricNeverSerializesAsZero(t *testing.T) {
	cases := []struct {
		name string
		m    any
	}{
		{"float", moecache.Unknown[float64](moecache.ReasonNoMeasuredAccess)},
		{"int", moecache.Unknown[int](moecache.ReasonNoRing)},
		{"int64", moecache.Unknown[int64](moecache.ReasonNotDerivable)},
	}
	for _, c := range cases {
		b, err := json.Marshal(c.m)
		if err != nil {
			t.Fatalf("%s: marshal unknown metric: %v", c.name, err)
		}
		var raw map[string]json.RawMessage
		if err := json.Unmarshal(b, &raw); err != nil {
			t.Fatalf("%s: not valid JSON: %v\n%s", c.name, err, b)
		}
		if _, present := raw["value"]; present {
			t.Fatalf("%s: unknown metric emitted a value: %s", c.name, b)
		}
		if string(raw["known"]) != "false" {
			t.Fatalf("%s: unknown metric known=%s, want false: %s", c.name, raw["known"], b)
		}
		var reason string
		if err := json.Unmarshal(raw["reason"], &reason); err != nil || reason == "" {
			t.Fatalf("%s: unknown metric carries no reason: %s", c.name, b)
		}
	}
}

// TestZeroValueMetricIsUnknown pins that a producer cannot forget the
// discipline: the zero Metric is unknown, not a known zero.
func TestZeroValueMetricIsUnknown(t *testing.T) {
	var m moecache.Metric[float64]
	if _, known := m.Get(); known {
		t.Fatal("zero Metric reports known=true; a producer could ship a phantom 0")
	}
	b, err := json.Marshal(m)
	if err != nil {
		t.Fatalf("marshal zero metric: %v", err)
	}
	if strings.Contains(string(b), `"value"`) {
		t.Fatalf("zero metric serialized a value: %s", b)
	}
	if !strings.Contains(string(b), `"known":false`) {
		t.Fatalf("zero metric did not serialize known:false: %s", b)
	}
}

// TestMetricRoundTripPreservesKnownAndUnknown pins that marshal/unmarshal
// preserves the boundary in both directions, including a known zero (a measured
// 0 is NOT the same as an absent value).
func TestMetricRoundTripPreservesKnownAndUnknown(t *testing.T) {
	type record struct {
		Hits    moecache.Metric[int64]   `json:"hits"`
		HitRate moecache.Metric[float64] `json:"hit_rate"`
	}
	in := record{
		Hits:    moecache.Known[int64](0),
		HitRate: moecache.Unknown[float64](moecache.ReasonNoMeasuredAccess),
	}
	b, err := json.Marshal(in)
	if err != nil {
		t.Fatalf("marshal record: %v", err)
	}
	var out record
	if err := json.Unmarshal(b, &out); err != nil {
		t.Fatalf("unmarshal record: %v\n%s", err, b)
	}
	if v, known := out.Hits.Get(); !known || v != 0 {
		t.Fatalf("known zero lost its known flag: value=%d known=%v", v, known)
	}
	if _, known := out.HitRate.Get(); known {
		t.Fatalf("unknown metric became known on round-trip: %s", b)
	}
	if out.HitRate.Reason != moecache.ReasonNoMeasuredAccess {
		t.Fatalf("unknown reason drifted: %q", out.HitRate.Reason)
	}
}

// TestUnknownNormalizesEmptyReason pins that an empty reason is never emitted.
func TestUnknownNormalizesEmptyReason(t *testing.T) {
	m := moecache.Unknown[int]("")
	if m.Known {
		t.Fatal("Unknown() returned a known metric")
	}
	if m.Reason != moecache.ReasonNotDerivable {
		t.Fatalf("empty reason normalized to %q, want %q", m.Reason, moecache.ReasonNotDerivable)
	}
}

// TestTelemetryRoundTripAndTierLookup pins the record-level round trip, the
// tier accessor, and that an unknown axis survives serialization as unknown.
func TestTelemetryRoundTripAndTierLookup(t *testing.T) {
	tel := moecache.New()
	tel.Shape = moecache.ModelShape{Experts: 384, TopK: 6, Layers: 40, ActivatedFraction: 6.0 / 384.0}

	dram := tel.Tier(moecache.TierDRAM)
	dram.HitRate = moecache.Known(0.9)
	dram.MissBytes = moecache.Known[int64](1024)
	dram.HitBytes = moecache.Unknown[int64](moecache.ReasonNotDerivable)
	dram.BeladyRegret = moecache.Unknown[float64](moecache.ReasonNoReplayTrace)

	tel.Coverage.Lookups = moecache.Known(200)
	tel.Coverage.AsyncOverlap = moecache.Unknown[float64](moecache.ReasonBackendNotAsync)

	b, err := json.Marshal(tel)
	if err != nil {
		t.Fatalf("marshal telemetry: %v", err)
	}
	var back moecache.ExpertCacheTelemetry
	if err := json.Unmarshal(b, &back); err != nil {
		t.Fatalf("unmarshal telemetry: %v\n%s", err, b)
	}
	if back.Schema != moecache.Schema {
		t.Fatalf("schema lost on round-trip: %q", back.Schema)
	}
	got, ok := back.Find(moecache.TierDRAM)
	if !ok {
		t.Fatal("DRAM tier lost on round-trip")
	}
	if rate, known := got.HitRate.Get(); !known || rate != 0.9 {
		t.Fatalf("hit rate round-trip: value=%v known=%v", rate, known)
	}
	if _, known := got.HitBytes.Get(); known {
		t.Fatal("unknown hit-bytes became known on round-trip")
	}
	if got.BeladyRegret.Reason != moecache.ReasonNoReplayTrace {
		t.Fatalf("regret reason drifted: %q", got.BeladyRegret.Reason)
	}
	if _, known := back.Coverage.AsyncOverlap.Get(); known {
		t.Fatal("unknown async-overlap became known on round-trip")
	}
	if _, ok := back.Find(moecache.TierNVMe); ok {
		t.Fatal("Find reported a tier slot that was never written")
	}
}
