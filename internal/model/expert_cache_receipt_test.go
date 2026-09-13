package model

import (
	"encoding/json"
	"math"
	"testing"
)

// populatedExpertCacheReceipt is a fully populated receipt at the pinned V4.1
// text shape. It exists so the field-lock test can marshal a receipt that
// touches every locked key.
func populatedExpertCacheReceipt() ExpertCacheReceipt {
	return ExpertCacheReceipt{
		Schema:            ExpertCacheReceiptSchema,
		Measurement:       ExpertCacheMeasurementSimulated,
		ModelID:           "deepseek-v4.1-flash",
		Arch:              "deepseek_v41_text",
		Precision:         "fp8",
		Quant:             "q2_k",
		Layers:            40,
		Experts:           384,
		TopK:              6,
		MoeInter:          2304,
		RingByteCap:       72 << 30,
		HitCount:          900,
		MissCount:         100,
		HitRate:           0.9,
		HitRateKnown:      true,
		BytesReadDRAM:     123456789,
		BytesReadNVMe:     987654321,
		PrefillToksPerSec: 42.5,
		DecodeToksPerSec:  18.25,
		TTFTMillis:        750.5,
		PeakMemoryBytes:   96 << 30,
	}
}

// TestExpertCacheReceiptRequiredFields is the FIELD LOCK: a fully populated
// receipt must carry every key in ExpertCacheRequiredFields and no others. If a
// field is renamed, dropped, or added without updating the required-field list,
// this fails - the schema cannot silently drift.
func TestExpertCacheReceiptRequiredFields(t *testing.T) {
	b, err := json.Marshal(populatedExpertCacheReceipt())
	if err != nil {
		t.Fatalf("marshal receipt: %v", err)
	}
	var m map[string]json.RawMessage
	if err := json.Unmarshal(b, &m); err != nil {
		t.Fatalf("receipt not valid JSON: %v\n%s", err, b)
	}
	for _, want := range ExpertCacheRequiredFields() {
		if _, ok := m[want]; !ok {
			t.Fatalf("missing required field %q:\n%s", want, b)
		}
	}
	if len(m) != len(ExpertCacheRequiredFields()) {
		t.Fatalf("receipt has %d fields, schema locks %d:\n%s", len(m), len(ExpertCacheRequiredFields()), b)
	}
}

// TestExpertCacheReceiptSchemaConstant pins the wire-format identifier; other
// suite leaves key their receipts and comparisons on this exact string.
func TestExpertCacheReceiptSchemaConstant(t *testing.T) {
	if ExpertCacheReceiptSchema != "fak.moe-expert-cache/v1" {
		t.Fatalf("schema constant drifted: %q", ExpertCacheReceiptSchema)
	}
	got := populatedExpertCacheReceipt().Schema
	if got != ExpertCacheReceiptSchema {
		t.Fatalf("populated receipt schema = %q, want %q", got, ExpertCacheReceiptSchema)
	}
}

// TestExpertCacheReceiptZeroHitRateUnknown pins the no-phantom-rate invariant: a
// receipt with no measured access (zero denominator) must report
// hit_rate_known=false, never a fabricated 0% hit rate.
func TestExpertCacheReceiptZeroHitRateUnknown(t *testing.T) {
	zero := ExpertCacheReceipt{Schema: ExpertCacheReceiptSchema, Measurement: ExpertCacheMeasurementSimulated}
	if zero.HitRateKnown {
		t.Fatalf("zero receipt claims a known hit rate")
	}
	b, err := json.Marshal(zero)
	if err != nil {
		t.Fatalf("marshal zero receipt: %v", err)
	}
	var m map[string]json.RawMessage
	if err := json.Unmarshal(b, &m); err != nil {
		t.Fatalf("zero receipt not valid JSON: %v\n%s", err, b)
	}
	for _, want := range ExpertCacheRequiredFields() {
		if _, ok := m[want]; !ok {
			t.Fatalf("zero receipt missing required field %q:\n%s", want, b)
		}
	}

	rate, known := ExpertCacheHitRate(0, 0)
	if known {
		t.Fatalf("ExpertCacheHitRate(0,0) reported known=true")
	}
	if rate != 0 {
		t.Fatalf("ExpertCacheHitRate(0,0) = %v, want 0", rate)
	}
	if ExpertCacheHitRateKnown(0, 0) {
		t.Fatalf("ExpertCacheHitRateKnown(0,0) = true, want false")
	}
}

// TestExpertCacheHitRate pins the arithmetic and the known/unknown boundary.
func TestExpertCacheHitRate(t *testing.T) {
	cases := []struct {
		hit, miss int64
		wantRate  float64
		wantKnown bool
	}{
		{900, 100, 0.9, true},
		{0, 100, 0.0, true},
		{100, 0, 1.0, true},
		{0, 0, 0.0, false},
		{1, 3, 0.25, true},
	}
	for _, c := range cases {
		rate, known := ExpertCacheHitRate(c.hit, c.miss)
		if known != c.wantKnown {
			t.Errorf("ExpertCacheHitRate(%d,%d) known=%v, want %v", c.hit, c.miss, known, c.wantKnown)
		}
		if math.Abs(rate-c.wantRate) > 1e-9 {
			t.Errorf("ExpertCacheHitRate(%d,%d) rate=%v, want %v", c.hit, c.miss, rate, c.wantRate)
		}
	}
}
