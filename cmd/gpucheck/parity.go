package main

import "fmt"

const (
	gpucheckReceiptSchema = "fak.gpucheck/1"
	gpucheckNativeEngine  = "fak-native"
)

type parityReasonCode string

const (
	reasonInvalidArguments    parityReasonCode = "invalid_arguments"
	reasonInvalidTokenCount   parityReasonCode = "invalid_token_count"
	reasonInvalidVocabulary   parityReasonCode = "invalid_vocabulary"
	reasonEmptyStream         parityReasonCode = "empty_stream"
	reasonExcessTokens        parityReasonCode = "excess_tokens"
	reasonTokenOutOfRange     parityReasonCode = "token_out_of_range"
	reasonBackendUnobserved   parityReasonCode = "backend_unobserved"
	reasonBackendMismatch     parityReasonCode = "backend_mismatch"
	reasonBackendNotDevice    parityReasonCode = "backend_not_device"
	reasonAsymmetricStreams   parityReasonCode = "asymmetric_streams"
	reasonTokenMismatch       parityReasonCode = "token_mismatch"
	reasonShortWithoutEOS     parityReasonCode = "short_without_eos"
	reasonConfiguration       parityReasonCode = "configuration_error"
	reasonBackendUnavailable  parityReasonCode = "backend_unavailable"
	reasonUnsupportedOptions  parityReasonCode = "unsupported_options"
	reasonModelLoad           parityReasonCode = "model_load_error"
	reasonMetalWitnessNotHere parityReasonCode = "metal_witness_not_here"
)

type backendIdentity struct {
	Name         string
	Tier         string
	DeviceMemory bool
}

type parityInput struct {
	Requested        int
	VocabSize        int
	Reference        []int
	Device           []int
	RequestedBackend string
	Observed         backendIdentity
	ReferenceLabel   string
	RequireNative    bool
	IsEOS            func(int) bool
}

type parityReceipt struct {
	Schema           string           `json:"schema"`
	Engine           string           `json:"engine"`
	Model            string           `json:"model,omitempty"`
	RequestedBackend string           `json:"requested_backend"`
	ObservedBackend  string           `json:"observed_backend,omitempty"`
	ObservedTier     string           `json:"observed_tier,omitempty"`
	DeviceMemory     bool             `json:"device_memory"`
	BackendObserved  bool             `json:"backend_observed"`
	FallbackActive   bool             `json:"fallback_active"`
	FallbackScope    string           `json:"fallback_scope"`
	ReferenceLabel   string           `json:"reference_label,omitempty"`
	RequestedTokens  int              `json:"requested_tokens"`
	ComparedTokens   int              `json:"compared_tokens"`
	ReferenceTokens  []int            `json:"reference_tokens"`
	DeviceTokens     []int            `json:"device_tokens"`
	Pass             bool             `json:"pass"`
	ReasonCode       parityReasonCode `json:"reason_code,omitempty"`
	Reason           string           `json:"reason,omitempty"`
}

func newParityReceipt(in parityInput) parityReceipt {
	observed := in.Observed.Name != ""
	return parityReceipt{
		Schema:           gpucheckReceiptSchema,
		Engine:           gpucheckNativeEngine,
		RequestedBackend: in.RequestedBackend,
		ObservedBackend:  in.Observed.Name,
		ObservedTier:     in.Observed.Tier,
		DeviceMemory:     in.Observed.DeviceMemory,
		BackendObserved:  observed,
		FallbackActive:   observed && in.Observed.Name != in.RequestedBackend,
		FallbackScope:    "backend_session_selection",
		ReferenceLabel:   in.ReferenceLabel,
		RequestedTokens:  in.Requested,
		ReferenceTokens:  append([]int(nil), in.Reference...),
		DeviceTokens:     append([]int(nil), in.Device...),
	}
}

func evaluateParity(in parityInput) parityReceipt {
	r := newParityReceipt(in)
	fail := func(code parityReasonCode, reason string) parityReceipt {
		r.ReasonCode = code
		r.Reason = reason
		return r
	}
	if in.Requested <= 0 {
		return fail(reasonInvalidTokenCount, "-n must be positive")
	}
	if in.VocabSize <= 0 {
		return fail(reasonInvalidVocabulary, "vocabulary size must be positive")
	}
	if len(in.Reference) == 0 || len(in.Device) == 0 {
		return fail(reasonEmptyStream, "empty token stream")
	}
	if len(in.Reference) > in.Requested || len(in.Device) > in.Requested {
		return fail(reasonExcessTokens, fmt.Sprintf("token stream exceeds requested count %d: reference=%d device=%d", in.Requested, len(in.Reference), len(in.Device)))
	}
	for _, stream := range []struct {
		source string
		tokens []int
	}{{"reference", in.Reference}, {"device", in.Device}} {
		for i, id := range stream.tokens {
			if id < 0 || id >= in.VocabSize {
				return fail(reasonTokenOutOfRange, fmt.Sprintf("%s token %d at index %d is outside vocabulary [0,%d)", stream.source, id, i, in.VocabSize))
			}
		}
	}
	if in.RequireNative {
		if !r.BackendObserved {
			return fail(reasonBackendUnobserved, "native identity requires an observed session backend")
		}
		if r.ObservedBackend != r.RequestedBackend {
			return fail(reasonBackendMismatch, fmt.Sprintf("observed backend %q does not match requested backend %q", r.ObservedBackend, r.RequestedBackend))
		}
		if !r.DeviceMemory {
			return fail(reasonBackendNotDevice, fmt.Sprintf("observed backend %q does not advertise device memory", r.ObservedBackend))
		}
	}
	if len(in.Reference) != len(in.Device) {
		return fail(reasonAsymmetricStreams, fmt.Sprintf("asymmetric token counts: reference=%d device=%d", len(in.Reference), len(in.Device)))
	}
	for i := range in.Reference {
		r.ComparedTokens = i + 1
		if in.Reference[i] != in.Device[i] {
			return fail(reasonTokenMismatch, fmt.Sprintf("token mismatch at %d: reference=%d device=%d", i, in.Reference[i], in.Device[i]))
		}
	}
	if len(in.Reference) < in.Requested {
		last := in.Reference[len(in.Reference)-1]
		if in.IsEOS == nil || !in.IsEOS(last) {
			return fail(reasonShortWithoutEOS, fmt.Sprintf("short token streams ended at %d without reference EOS (requested %d)", len(in.Reference), in.Requested))
		}
	}
	r.Pass = true
	return r
}
