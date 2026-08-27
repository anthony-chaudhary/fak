package modelperfobs

import (
	"encoding/json"
	"math"
	"strings"
	"testing"
)

func TestLongContextEnvelopeContextAndPrefillDecodeDemandMatrix(t *testing.T) {
	for _, contextTokens := range []uint64{35_000, 64_000, 128_000, 200_000} {
		for _, ratio := range []uint64{200, 300} {
			in := longContextFixture()
			in.ResidentContextTokens = contextTokens
			in.DecodeTokens = contextTokens / (ratio + 1)
			in.PrefillTokens = ratio * in.DecodeTokens
			// Integer demand ratios need not consume the final remainder of context.
			if in.PrefillTokens+in.DecodeTokens > contextTokens {
				t.Fatalf("invalid test demand for context=%d ratio=%d", contextTokens, ratio)
			}
			activeParameters := in.ActiveParameters
			got, err := EstimateLongContextEnvelope(in)
			if err != nil {
				t.Fatalf("context=%d demand=%d:1: %v", contextTokens, ratio, err)
			}
			if in.ActiveParameters != activeParameters {
				t.Fatalf("context=%d demand=%d:1 mutated active parameters", contextTokens, ratio)
			}
			if in.PrefillTokens/in.DecodeTokens != ratio || in.PrefillTokens%in.DecodeTokens != 0 {
				t.Fatalf("context=%d demand ratio = %d:%d, want %d:1", contextTokens, in.PrefillTokens, in.DecodeTokens, ratio)
			}
			assertOrderedEnvelope(t, got)
			wantKVMin := float64(contextTokens*in.ResidentAgents) * in.KVBytesPerToken.Min
			if got.KVMemoryBytes.Min != wantKVMin {
				t.Errorf("context=%d demand=%d:1: KV minimum = %g, want %g", contextTokens, ratio, got.KVMemoryBytes.Min, wantKVMin)
			}
		}
	}
}

func TestLongContextEnvelopeSchemaAndJSONShape(t *testing.T) {
	got, err := EstimateLongContextEnvelope(longContextFixture())
	if err != nil {
		t.Fatal(err)
	}
	if got.Schema != "fak-long-context-envelope/2" {
		t.Fatalf("schema = %q", got.Schema)
	}
	raw, err := json.Marshal(got)
	if err != nil {
		t.Fatal(err)
	}
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(raw, &fields); err != nil {
		t.Fatal(err)
	}
	for _, key := range []string{"processed_tokens_per_second", "decode_tokens_per_second"} {
		if _, ok := fields[key]; !ok {
			t.Fatalf("JSON missing %q: %s", key, raw)
		}
	}
	if _, ok := fields["throughput_tokens_per_second"]; ok {
		t.Fatalf("JSON retained obsolete throughput key: %s", raw)
	}
}

func TestLongContextEnvelopeValidatesInputs(t *testing.T) {
	base := longContextFixture()
	tests := []struct {
		name string
		edit func(*LongContextEstimatorInput)
		want string
	}{
		{"non-finite", func(in *LongContextEstimatorInput) { in.TotalParameters = math.NaN() }, "finite and positive"},
		{"reversed range", func(in *LongContextEstimatorInput) { in.WeightBits = ClosedRange{8, 4} }, "minimum cannot exceed maximum"},
		{"wrong unit sign", func(in *LongContextEstimatorInput) { in.KVBytesPerToken.Min = 0 }, "must be positive"},
		{"fraction", func(in *LongContextEstimatorInput) { in.Efficiency.Max = 1.1 }, "fractions in [0,1]"},
		{"zero denominator", func(in *LongContextEstimatorInput) { in.BandwidthBytesPerSec.Min = 0 }, "must be positive"},
		{"active exceeds total", func(in *LongContextEstimatorInput) { in.ActiveParameters = in.TotalParameters + 1 }, "cannot exceed"},
		{"concurrency exceeds agents", func(in *LongContextEstimatorInput) { in.Concurrency = in.ResidentAgents + 1 }, "cannot exceed"},
		{"tokens exceed resident context", func(in *LongContextEstimatorInput) { in.PrefillTokens = in.ResidentContextTokens }, "cannot exceed"},
		{"no service tokens", func(in *LongContextEstimatorInput) { in.PrefillTokens, in.DecodeTokens = 0, 0 }, "cannot both be zero"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			in := base
			tt.edit(&in)
			_, err := EstimateLongContextEnvelope(in)
			if err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("error = %v, want substring %q", err, tt.want)
			}
		})
	}
}

func TestLongContextEnvelopeResidentContextBoundary(t *testing.T) {
	in := longContextFixture()
	in.ResidentContextTokens = in.PrefillTokens + in.DecodeTokens
	if _, err := EstimateLongContextEnvelope(in); err != nil {
		t.Fatalf("exact resident-context boundary rejected: %v", err)
	}

	in.ResidentContextTokens--
	_, err := EstimateLongContextEnvelope(in)
	if err == nil || !strings.Contains(err.Error(), "cannot exceed resident_context_tokens") {
		t.Fatalf("error = %v, want resident-context overflow rejection", err)
	}
}

func TestLongContextEnvelopeZeroWorkPrefillRemainsFinite(t *testing.T) {
	tests := []struct {
		name         string
		prefill      uint64
		cacheHit     ClosedRange
		wantPrefill0 bool
	}{
		{name: "fully cached prefill", prefill: 34_000, cacheHit: ClosedRange{1, 1}, wantPrefill0: true},
		{name: "zero prefill", prefill: 0, cacheHit: ClosedRange{0, 0}, wantPrefill0: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			in := longContextFixture()
			in.PrefillTokens = tt.prefill
			in.PrefillCacheHit = tt.cacheHit
			got, err := EstimateLongContextEnvelope(in)
			if err != nil {
				t.Fatal(err)
			}
			assertOrderedEnvelope(t, got)
			if tt.wantPrefill0 && got.Prefill.TimeSeconds != (ClosedRange{}) {
				t.Fatalf("prefill time = %v, want zero", got.Prefill.TimeSeconds)
			}
			if got.Decode.TimeSeconds.Min <= 0 || got.ServiceTimeSeconds.Min <= 0 || got.ProcessedTokensPerSec.Min <= 0 {
				t.Fatalf("non-finite decode service envelope: %+v", got)
			}
			if got.Bottleneck != got.Decode.Bottleneck {
				t.Fatalf("overall bottleneck = %q, want decode %q", got.Bottleneck, got.Decode.Bottleneck)
			}
		})
	}
}

func TestLongContextEnvelopeFitHeadroomAndResidency(t *testing.T) {
	in := longContextFixture()
	probe, err := EstimateLongContextEnvelope(in)
	if err != nil {
		t.Fatal(err)
	}

	in.UsableMemoryBytes = probe.TotalResidentMemoryBytes.Max + 1
	fit, err := EstimateLongContextEnvelope(in)
	if err != nil {
		t.Fatal(err)
	}
	if !fit.Fits || fit.FitUncertain || fit.HeadroomBytes.Min <= 0 {
		t.Fatalf("certain fit = %+v", fit)
	}
	if fit.MaxResidentAgents.Min < float64(in.ResidentAgents) {
		t.Fatalf("minimum resident agents = %g, want at least %d", fit.MaxResidentAgents.Min, in.ResidentAgents)
	}
	if math.Trunc(fit.MaxResidentAgents.Min) != fit.MaxResidentAgents.Min || math.Trunc(fit.MaxResidentAgents.Max) != fit.MaxResidentAgents.Max {
		t.Fatalf("max resident-agent endpoints must be integer-valued counts: %v", fit.MaxResidentAgents)
	}

	in.UsableMemoryBytes = (probe.TotalResidentMemoryBytes.Min + probe.TotalResidentMemoryBytes.Max) / 2
	uncertain, err := EstimateLongContextEnvelope(in)
	if err != nil {
		t.Fatal(err)
	}
	if uncertain.Fits || !uncertain.FitUncertain || uncertain.HeadroomBytes.Min >= 0 || uncertain.HeadroomBytes.Max <= 0 {
		t.Fatalf("uncertain fit = %+v", uncertain)
	}

	in.UsableMemoryBytes = probe.TotalResidentMemoryBytes.Min - 1
	noFit, err := EstimateLongContextEnvelope(in)
	if err != nil {
		t.Fatal(err)
	}
	if noFit.Fits || noFit.FitUncertain || noFit.HeadroomBytes.Max >= 0 {
		t.Fatalf("non-fit = %+v", noFit)
	}
}

func TestLongContextEnvelopeSeparatesProcessedAndDecodeThroughput(t *testing.T) {
	in := longContextFixture()
	in.PrefillCacheHit = ClosedRange{}

	got, err := EstimateLongContextEnvelope(in)
	if err != nil {
		t.Fatal(err)
	}
	processed := float64(in.Concurrency) * float64(in.PrefillTokens+in.DecodeTokens)
	decoded := float64(in.Concurrency) * float64(in.DecodeTokens)
	wantProcessed := ClosedRange{processed / got.ServiceTimeSeconds.Max, processed / got.ServiceTimeSeconds.Min}
	wantDecode := ClosedRange{decoded / got.Decode.TimeSeconds.Max, decoded / got.Decode.TimeSeconds.Min}
	if got.ProcessedTokensPerSec != wantProcessed {
		t.Fatalf("processed throughput = %v, want %v", got.ProcessedTokensPerSec, wantProcessed)
	}
	if got.DecodeTokensPerSec != wantDecode {
		t.Fatalf("decode throughput = %v, want %v", got.DecodeTokensPerSec, wantDecode)
	}
}

func TestLongContextEnvelopeCacheHitReducesPrefillOnly(t *testing.T) {
	miss := longContextFixture()
	miss.PrefillCacheHit = ClosedRange{0, 0}
	hit := miss
	hit.PrefillCacheHit = ClosedRange{0.75, 0.75}

	withoutCache, err := EstimateLongContextEnvelope(miss)
	if err != nil {
		t.Fatal(err)
	}
	withCache, err := EstimateLongContextEnvelope(hit)
	if err != nil {
		t.Fatal(err)
	}
	if withCache.Prefill.TimeSeconds.Max >= withoutCache.Prefill.TimeSeconds.Min {
		t.Fatalf("cached prefill %v did not improve over %v", withCache.Prefill.TimeSeconds, withoutCache.Prefill.TimeSeconds)
	}
	if withCache.Decode.ComputeSeconds != withoutCache.Decode.ComputeSeconds {
		t.Fatalf("decode FLOP-derived compute time changed with prefill cache: before=%v after=%v", withoutCache.Decode.ComputeSeconds, withCache.Decode.ComputeSeconds)
	}
	if withCache.Decode.TimeSeconds != withoutCache.Decode.TimeSeconds {
		t.Fatalf("decode time changed with prefill cache: before=%v after=%v", withoutCache.Decode.TimeSeconds, withCache.Decode.TimeSeconds)
	}
	if withCache.DecodeTokensPerSec != withoutCache.DecodeTokensPerSec {
		t.Fatalf("decode throughput changed with prefill cache: before=%v after=%v", withoutCache.DecodeTokensPerSec, withCache.DecodeTokensPerSec)
	}
	if withCache.ServiceTimeSeconds.Max >= withoutCache.ServiceTimeSeconds.Min {
		t.Fatalf("service time did not decrease with cache: before=%v after=%v", withoutCache.ServiceTimeSeconds, withCache.ServiceTimeSeconds)
	}
	if withCache.ProcessedTokensPerSec.Min <= withoutCache.ProcessedTokensPerSec.Min {
		t.Fatalf("processed throughput did not improve with cache: before=%v after=%v", withoutCache.ProcessedTokensPerSec, withCache.ProcessedTokensPerSec)
	}
	if withCache.KVMemoryBytes != withoutCache.KVMemoryBytes {
		t.Fatalf("resident KV changed with prefill cache: before=%v after=%v", withoutCache.KVMemoryBytes, withCache.KVMemoryBytes)
	}
}

func TestLongContextEnvelopeReportsComputeAndBandwidthBottlenecks(t *testing.T) {
	compute := longContextFixture()
	compute.ComputeFLOPS = ClosedRange{1e9, 1e9}
	compute.BandwidthBytesPerSec = ClosedRange{1e15, 1e15}
	gotCompute, err := EstimateLongContextEnvelope(compute)
	if err != nil {
		t.Fatal(err)
	}
	if gotCompute.Bottleneck != "compute" {
		t.Fatalf("bottleneck = %q, want compute", gotCompute.Bottleneck)
	}

	bandwidth := longContextFixture()
	bandwidth.ComputeFLOPS = ClosedRange{1e20, 1e20}
	bandwidth.BandwidthBytesPerSec = ClosedRange{1e6, 1e6}
	gotBandwidth, err := EstimateLongContextEnvelope(bandwidth)
	if err != nil {
		t.Fatal(err)
	}
	if gotBandwidth.Bottleneck != "bandwidth" {
		t.Fatalf("bottleneck = %q, want bandwidth", gotBandwidth.Bottleneck)
	}
	if gotBandwidth.Prefill.TimeSeconds.Min != math.Max(gotBandwidth.Prefill.ComputeSeconds.Min, gotBandwidth.Prefill.MemorySeconds.Min) {
		t.Fatal("prefill roofline is not max(compute, memory)")
	}
}

func longContextFixture() LongContextEstimatorInput {
	return LongContextEstimatorInput{
		TotalParameters: 300e9, ActiveParameters: 1.5e9,
		WeightBits: ClosedRange{4, 5}, MetadataOverhead: ClosedRange{0.02, 0.08},
		KVBytesPerToken: ClosedRange{256 * 1024, 384 * 1024}, ResidentContextTokens: 35_000,
		ResidentAgents: 8, Concurrency: 4, PrefillTokens: 34_000, DecodeTokens: 1_000,
		UsableMemoryBytes: 512e9, BandwidthBytesPerSec: ClosedRange{700e9, 900e9},
		ComputeFLOPS: ClosedRange{100e12, 140e12}, Efficiency: ClosedRange{0.45, 0.70},
		PrefillCacheHit: ClosedRange{0.10, 0.30},
	}
}

func assertOrderedEnvelope(t *testing.T, got LongContextEnvelope) {
	t.Helper()
	ranges := []ClosedRange{
		got.ModelMemoryBytes, got.KVMemoryPerAgentBytes, got.KVMemoryBytes,
		got.TotalResidentMemoryBytes, got.HeadroomBytes, got.MaxResidentAgents,
		got.Prefill.ComputeSeconds, got.Prefill.MemorySeconds, got.Prefill.TimeSeconds,
		got.Decode.ComputeSeconds, got.Decode.MemorySeconds, got.Decode.TimeSeconds,
		got.ServiceTimeSeconds, got.ProcessedTokensPerSec, got.DecodeTokensPerSec,
	}
	for i, r := range ranges {
		if !finite(r.Min) || !finite(r.Max) || r.Min > r.Max {
			t.Fatalf("range %d is invalid: %+v", i, r)
		}
	}
}
