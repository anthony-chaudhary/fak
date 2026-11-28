// Adapted from ai-dynamo/aiperf
// src/aiperf/post_processors/strategies/genai_semconv.py:69
// Pure-data, import-safe OTel GenAI semantic-convention metric mapping table
// aligned with pkg/stepspan/genai_map.go without modifying span definitions.

package gateway

import (
	"slices"
	"strings"

	"github.com/anthony-chaudhary/fak/pkg/stepspan"
)

// OTelInstrumentKind identifies the OpenTelemetry metric instrument kind.
type OTelInstrumentKind string

const (
	OTelInstrumentHistogram     OTelInstrumentKind = "histogram"
	OTelInstrumentCounter       OTelInstrumentKind = "counter"
	OTelInstrumentUpDownCounter OTelInstrumentKind = "up_down_counter"
	OTelInstrumentGauge         OTelInstrumentKind = "gauge"
)

// Canonical OpenTelemetry GenAI metric names.
const (
	GenAIMetricClientOperationDuration  = "gen_ai.client.operation.duration"
	GenAIMetricClientTokenUsage         = "gen_ai.client.token.usage"
	GenAIMetricServerTimeToFirstToken   = "gen_ai.server.time_to_first_token"
	GenAIMetricServerTimePerOutputToken = "gen_ai.server.time_per_output_token"
	GenAIMetricServerRequestDuration    = "gen_ai.server.request.duration"
)

// Canonical attribute keys aligned with pkg/stepspan/genai_map.go.
const (
	GenAIAttrOperationName = stepspan.GenAIOperationName
	GenAIAttrProviderName  = stepspan.GenAIProviderName
	GenAIAttrRequestModel  = stepspan.GenAIRequestModel
	GenAIAttrErrorType     = stepspan.GenAIErrorType
	GenAIAttrTokenType     = "gen_ai.token.type"
	GenAIAttrResponseModel = "gen_ai.response.model"
	GenAIAttrServerAddress = "server.address"
	GenAIAttrServerPort    = "server.port"
)

// GenAISemconvSpec maps a benchmark/gateway metric tag onto an OTel GenAI
// semantic-convention metric descriptor and spec histogram boundaries.
type GenAISemconvSpec struct {
	Tag              string
	Aliases          []string
	MetricName       string
	Unit             string
	Instrument       OTelInstrumentKind
	TokenType        string
	BucketBoundaries []float64
	RequiredAttrs    []string
	Description      string
}

var (
	genAITTFTBucketBoundaries = []float64{
		0.001, 0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1,
		0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0,
	}
	genAITPOTBucketBoundaries = []float64{
		0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25,
		0.3, 0.4, 0.5, 0.75, 1.0, 2.5,
	}
	genAIDurationBucketBoundaries = []float64{
		0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28,
		2.56, 5.12, 10.24, 20.48, 40.96, 81.92,
	}
	genAITokenUsageBucketBoundaries = []float64{
		1, 4, 16, 64, 256, 1024, 4096, 16384,
		65536, 262144, 1048576, 4194304, 16777216, 67108864,
	}
)

var genAISemconvSpecs = []GenAISemconvSpec{
	{
		Tag:              "time_to_first_token",
		Aliases:          []string{"ttft"},
		MetricName:       GenAIMetricServerTimeToFirstToken,
		Unit:             "s",
		Instrument:       OTelInstrumentHistogram,
		BucketBoundaries: genAITTFTBucketBoundaries,
		RequiredAttrs:    []string{GenAIAttrOperationName, GenAIAttrProviderName, GenAIAttrRequestModel},
		Description:      "Time to generate first token for successful responses",
	},
	{
		Tag:              "time_per_output_token",
		Aliases:          []string{"inter_token_latency", "itl", "tpot"},
		MetricName:       GenAIMetricServerTimePerOutputToken,
		Unit:             "s",
		Instrument:       OTelInstrumentHistogram,
		BucketBoundaries: genAITPOTBucketBoundaries,
		RequiredAttrs:    []string{GenAIAttrOperationName, GenAIAttrProviderName, GenAIAttrRequestModel},
		Description:      "Time per output token generated after the first token",
	},
	{
		Tag:              "request_latency",
		Aliases:          []string{"client_operation_duration"},
		MetricName:       GenAIMetricClientOperationDuration,
		Unit:             "s",
		Instrument:       OTelInstrumentHistogram,
		BucketBoundaries: genAIDurationBucketBoundaries,
		RequiredAttrs:    []string{GenAIAttrOperationName, GenAIAttrProviderName, GenAIAttrRequestModel},
		Description:      "GenAI client operation duration",
	},
	{
		Tag:              "server_request_duration",
		MetricName:       GenAIMetricServerRequestDuration,
		Unit:             "s",
		Instrument:       OTelInstrumentHistogram,
		BucketBoundaries: genAIDurationBucketBoundaries,
		RequiredAttrs:    []string{GenAIAttrOperationName, GenAIAttrProviderName, GenAIAttrRequestModel},
		Description:      "Generative AI server request duration",
	},
	{
		Tag:              "input_sequence_length",
		Aliases:          []string{"input_tokens", "isl"},
		MetricName:       GenAIMetricClientTokenUsage,
		Unit:             "{token}",
		Instrument:       OTelInstrumentHistogram,
		TokenType:        "input",
		BucketBoundaries: genAITokenUsageBucketBoundaries,
		RequiredAttrs:    []string{GenAIAttrOperationName, GenAIAttrProviderName, GenAIAttrTokenType, GenAIAttrRequestModel},
		Description:      "Measures number of input tokens used",
	},
	{
		Tag:              "output_sequence_length",
		Aliases:          []string{"output_tokens", "osl"},
		MetricName:       GenAIMetricClientTokenUsage,
		Unit:             "{token}",
		Instrument:       OTelInstrumentHistogram,
		TokenType:        "output",
		BucketBoundaries: genAITokenUsageBucketBoundaries,
		RequiredAttrs:    []string{GenAIAttrOperationName, GenAIAttrProviderName, GenAIAttrTokenType, GenAIAttrRequestModel},
		Description:      "Measures number of output tokens used",
	},
}

func cloneGenAISemconvSpec(in GenAISemconvSpec) GenAISemconvSpec {
	out := in
	out.Aliases = slices.Clone(in.Aliases)
	out.BucketBoundaries = slices.Clone(in.BucketBoundaries)
	out.RequiredAttrs = slices.Clone(in.RequiredAttrs)
	return out
}

// LookupGenAISemconv looks up a metric tag or alias and returns a defensive
// copy of its OTel GenAI semantic-convention mapping.
func LookupGenAISemconv(tag string) (GenAISemconvSpec, bool) {
	norm := strings.ToLower(strings.TrimSpace(tag))
	if norm == "" {
		return GenAISemconvSpec{}, false
	}
	for _, spec := range genAISemconvSpecs {
		if spec.Tag == norm {
			return cloneGenAISemconvSpec(spec), true
		}
		for _, alias := range spec.Aliases {
			if alias == norm {
				return cloneGenAISemconvSpec(spec), true
			}
		}
	}
	return GenAISemconvSpec{}, false
}

// AllGenAISemconvSpecs returns a defensive copy of all registered GenAI semconv mappings.
func AllGenAISemconvSpecs() []GenAISemconvSpec {
	out := make([]GenAISemconvSpec, len(genAISemconvSpecs))
	for i, spec := range genAISemconvSpecs {
		out[i] = cloneGenAISemconvSpec(spec)
	}
	return out
}

// GenAISemconvBucketBoundaries returns a defensive copy of the spec histogram
// bucket boundaries for an OTel GenAI metric name.
func GenAISemconvBucketBoundaries(metricName string) ([]float64, bool) {
	norm := strings.ToLower(strings.TrimSpace(metricName))
	for _, spec := range genAISemconvSpecs {
		if spec.MetricName == norm {
			return slices.Clone(spec.BucketBoundaries), true
		}
	}
	return nil, false
}
