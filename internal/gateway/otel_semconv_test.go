package gateway

import (
	"math"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/pkg/stepspan"
)

// fak-test:runtime fast est=1s
func TestSemconvLookupAndAliases(t *testing.T) {
	cases := []struct {
		tag           string
		wantMetric    string
		wantUnit      string
		wantKind      OTelInstrumentKind
		wantTokenType string
	}{
		{"time_to_first_token", GenAIMetricServerTimeToFirstToken, "s", OTelInstrumentHistogram, ""},
		{"ttft", GenAIMetricServerTimeToFirstToken, "s", OTelInstrumentHistogram, ""},
		{"time_per_output_token", GenAIMetricServerTimePerOutputToken, "s", OTelInstrumentHistogram, ""},
		{"inter_token_latency", GenAIMetricServerTimePerOutputToken, "s", OTelInstrumentHistogram, ""},
		{"itl", GenAIMetricServerTimePerOutputToken, "s", OTelInstrumentHistogram, ""},
		{"request_latency", GenAIMetricClientOperationDuration, "s", OTelInstrumentHistogram, ""},
		{"server_request_duration", GenAIMetricServerRequestDuration, "s", OTelInstrumentHistogram, ""},
		{"input_sequence_length", GenAIMetricClientTokenUsage, "{token}", OTelInstrumentHistogram, "input"},
		{"isl", GenAIMetricClientTokenUsage, "{token}", OTelInstrumentHistogram, "input"},
		{"output_sequence_length", GenAIMetricClientTokenUsage, "{token}", OTelInstrumentHistogram, "output"},
		{"osl", GenAIMetricClientTokenUsage, "{token}", OTelInstrumentHistogram, "output"},
	}

	for _, tc := range cases {
		spec, ok := LookupGenAISemconv(tc.tag)
		if !ok {
			t.Fatalf("LookupGenAISemconv(%q) returned false", tc.tag)
		}
		if spec.MetricName != tc.wantMetric || spec.Unit != tc.wantUnit || spec.Instrument != tc.wantKind || spec.TokenType != tc.wantTokenType {
			t.Fatalf("LookupGenAISemconv(%q) = %+v, want metric=%q unit=%q kind=%q tokenType=%q",
				tc.tag, spec, tc.wantMetric, tc.wantUnit, tc.wantKind, tc.wantTokenType)
		}
	}

	if _, ok := LookupGenAISemconv(""); ok {
		t.Fatal("empty tag must not resolve")
	}
	if _, ok := LookupGenAISemconv("unknown_metric_tag"); ok {
		t.Fatal("unknown tag must not resolve")
	}
}

// fak-test:runtime fast est=1s
func TestSemconvBucketBoundariesAndStepspanAlignment(t *testing.T) {
	if GenAIAttrOperationName != stepspan.GenAIOperationName ||
		GenAIAttrProviderName != stepspan.GenAIProviderName ||
		GenAIAttrRequestModel != stepspan.GenAIRequestModel ||
		GenAIAttrErrorType != stepspan.GenAIErrorType {
		t.Fatal("GenAI attribute constants drifted from pkg/stepspan/genai_map.go")
	}

	specs := AllGenAISemconvSpecs()
	if len(specs) == 0 {
		t.Fatal("expected non-empty GenAI semconv table")
	}

	for _, spec := range specs {
		if !strings.HasPrefix(spec.MetricName, "gen_ai.") {
			t.Fatalf("%s metric %q must use gen_ai.* prefix", spec.Tag, spec.MetricName)
		}
		if spec.Instrument == OTelInstrumentHistogram {
			if len(spec.BucketBoundaries) == 0 {
				t.Fatalf("%s histogram missing spec bucket boundaries", spec.Tag)
			}
			prev := 0.0
			for i, b := range spec.BucketBoundaries {
				if math.IsNaN(b) || math.IsInf(b, 0) || b <= prev {
					t.Fatalf("%s invalid bucket boundary[%d]=%v (prev=%v)", spec.Tag, i, b, prev)
				}
				prev = b
			}
			direct, ok := GenAISemconvBucketBoundaries(spec.MetricName)
			if !ok || len(direct) != len(spec.BucketBoundaries) {
				t.Fatalf("GenAISemconvBucketBoundaries(%q) failed: ok=%v len=%d", spec.MetricName, ok, len(direct))
			}
		}
		for _, attr := range spec.RequiredAttrs {
			if stepspan.IsGenAIContentKey(attr) {
				t.Fatalf("%s required attr %q must not emit opt-in content keys", spec.Tag, attr)
			}
		}
	}
}

// fak-test:runtime fast est=1s
func TestSemconvPureDataDefensiveCopies(t *testing.T) {
	spec, ok := LookupGenAISemconv("ttft")
	if !ok {
		t.Fatal("LookupGenAISemconv(ttft) returned false")
	}
	origFirst := spec.BucketBoundaries[0]
	origAttr := spec.RequiredAttrs[0]
	spec.BucketBoundaries[0] = -999
	spec.RequiredAttrs[0] = "mutated"

	again, ok := LookupGenAISemconv("ttft")
	if !ok {
		t.Fatal("LookupGenAISemconv(ttft) second call failed")
	}
	if again.BucketBoundaries[0] != origFirst || again.RequiredAttrs[0] != origAttr {
		t.Fatalf("mutating returned slice corrupted package table: got bucket[0]=%v attr[0]=%q",
			again.BucketBoundaries[0], again.RequiredAttrs[0])
	}

	bounds, ok := GenAISemconvBucketBoundaries(GenAIMetricServerTimeToFirstToken)
	if !ok {
		t.Fatal("GenAISemconvBucketBoundaries failed")
	}
	bounds[0] = -1
	bounds2, ok := GenAISemconvBucketBoundaries(GenAIMetricServerTimeToFirstToken)
	if !ok || bounds2[0] != origFirst {
		t.Fatalf("mutating boundary copy corrupted package table: got %v want %v", bounds2[0], origFirst)
	}
}
