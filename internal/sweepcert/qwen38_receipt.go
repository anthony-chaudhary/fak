package sweepcert

import (
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"slices"
	"strconv"
	"strings"
)

// Qwen38PrefillSample joins authoritative prompt encoding to the executed
// receipt. Pointer fields distinguish missing attestations from measured zero.
// Source identifies the scrubbed underlying receipt, never an inferred run.
type Qwen38PrefillSample struct {
	Source               string   `json:"source"`
	PromptTokens         int      `json:"prompt_tokens"`
	TokenIDs             []int    `json:"token_ids"`
	ExecutedTokenIDs     []int    `json:"executed_token_ids"`
	Renderer             string   `json:"renderer"`
	RenderedDigest       string   `json:"rendered_digest"`
	ReusedTokens         *int     `json:"reused_tokens"`
	ContextTokens        int      `json:"context_tokens"`
	ReservedOutputTokens int      `json:"reserved_output_tokens"`
	Throughput           *float64 `json:"throughput"`
	Failure              string   `json:"failure,omitempty"`
}

// Qwen38PrefillReceipt describes one source/configuration arm. Callers must
// authenticate the underlying receipts; this pure adapter neither dispatches
// hardware nor establishes that caller-supplied attestations are true.
type Qwen38PrefillReceipt struct {
	Provenance           Qwen38NativeProvenance `json:"provenance"`
	BinaryDigest         string                 `json:"binary_digest"`
	ShaderDigest         string                 `json:"shader_digest"`
	ChunkTokens          int                    `json:"chunk_tokens"`
	AdmissionEvidence    string                 `json:"admission_evidence"`
	AdmittedPromptTokens int                    `json:"admitted_prompt_tokens"`
	Samples              []Qwen38PrefillSample  `json:"samples"`
}

// Qwen38PrefillResponse preserves one raw observation per length. Single
// observations carry N=1, never distribution percentiles. The receipt is retained
// so that token identity and memory admission remain reviewable after export.
type Qwen38PrefillResponse struct {
	Receipt      Qwen38PrefillReceipt `json:"receipt"`
	Evidence     Evidence             `json:"evidence"`
	SampleCounts []int                `json:"sample_counts"`
	Finding      Finding              `json:"finding"`
}

// Qwen38PrefillReport adapts a cold native arm to sweepcert. It does not infer
// range closure from a failed allocation or a last sampled prompt length.
// Oversized failed targets remain null; measured points must fit admission.
func Qwen38PrefillReport(receipt Qwen38PrefillReceipt) (Qwen38PrefillResponse, error) {
	bad := func(reason string) (Qwen38PrefillResponse, error) {
		return Qwen38PrefillResponse{}, fmt.Errorf("Qwen3.8 receipt: %s", reason)
	}
	if strings.TrimSpace(receipt.BinaryDigest) == "" || strings.TrimSpace(receipt.ShaderDigest) == "" || strings.TrimSpace(receipt.AdmissionEvidence) == "" || receipt.ChunkTokens <= 0 || receipt.AdmittedPromptTokens <= 0 {
		return bad("binary, shader, chunk size and memory admission are required")
	}
	coordinates := make([]float64, len(receipt.Samples))
	for i, sample := range receipt.Samples {
		coordinates[i] = float64(sample.PromptTokens)
		if sample.PromptTokens <= 0 || strings.TrimSpace(sample.Source) == "" {
			return bad("each target needs a positive token count and receipt source")
		}
		if sample.Failure != "" {
			if strings.TrimSpace(sample.Failure) == "" || sample.Throughput != nil {
				return bad("failed targets require an explicit reason and null throughput")
			}
			continue
		}
		if sample.Throughput == nil || !finite(*sample.Throughput) || *sample.Throughput <= 0 {
			return bad("successful targets require positive finite throughput")
		}
		if sample.PromptTokens > receipt.AdmittedPromptTokens || sample.ReservedOutputTokens <= 0 || sample.ContextTokens <= sample.ReservedOutputTokens || sample.PromptTokens > sample.ContextTokens-sample.ReservedOutputTokens {
			return bad("measured target exceeds admitted memory/context boundary")
		}
		if len(sample.TokenIDs) != sample.PromptTokens || !slices.Equal(sample.TokenIDs, sample.ExecutedTokenIDs) {
			return bad("ordered encoded and executed token IDs must match the consumed count")
		}
		for _, id := range sample.TokenIDs {
			if id < 0 {
				return bad("negative token ID")
			}
		}
		if sample.ReusedTokens == nil || *sample.ReusedTokens != 0 {
			return bad("measured target requires observed zero reuse")
		}
		if strings.TrimSpace(sample.Renderer) == "" || strings.TrimSpace(sample.RenderedDigest) == "" {
			return bad("authoritative renderer and rendered prompt digest are required")
		}
	}
	envelope, err := NewQwen38NativeEnvelope(coordinates, receipt.Provenance, RangeClosure{})
	if err != nil {
		return Qwen38PrefillResponse{}, err
	}
	// Bind every per-request attestation, including ordered IDs, rather than
	// letting two arms compare solely because they used the same target labels.
	raw, err := json.Marshal(receipt)
	if err != nil {
		return Qwen38PrefillResponse{}, err
	}
	// Own the exported receipt: later caller mutations must not invalidate its digest.
	var owned Qwen38PrefillReceipt
	if err := json.Unmarshal(raw, &owned); err != nil {
		return Qwen38PrefillResponse{}, err
	}
	envelope.Bindings = append(envelope.Bindings, Binding{Name: "receipt_digest", Value: fmt.Sprintf("sha256:%x", sha256.Sum256(raw))})
	digest, err := CanonicalEnvelopeDigest(envelope)
	if err != nil {
		return Qwen38PrefillResponse{}, err
	}
	report := Qwen38PrefillResponse{Receipt: owned, Evidence: Evidence{Envelope: envelope, EnvelopeDigest: digest}, SampleCounts: make([]int, len(owned.Samples))}
	for i, sample := range owned.Samples {
		point := Point{ID: "prompt-" + strconv.Itoa(sample.PromptTokens), Coordinate: coordinates[i], EnvelopeDigest: digest, Status: PointNotMeasured}
		observation := Observation{Status: ObservationNotMeasured, Reason: sample.Failure}
		if sample.Failure == "" {
			point.Status = PointMeasured
			report.SampleCounts[i] = 1
			value := *sample.Throughput
			observation = Observation{Status: ObservationMeasured, Value: &value, Provenance: Provenance{Source: sample.Source, Method: "native-cold-prefill-single-observation", Unit: "tok/s", EnvelopeDigest: digest}}
		}
		point.Observations = map[string]Observation{"prefill_throughput": observation}
		report.Evidence.Points = append(report.Evidence.Points, point)
	}
	if err := ValidateQwen38NativeEvidence(report.Evidence); err != nil {
		return Qwen38PrefillResponse{}, err
	}
	report.Finding = ObservedExtremum(report.Evidence, "prefill_throughput", Maximum)
	return report, ValidateFinding(report.Evidence, report.Finding)
}
