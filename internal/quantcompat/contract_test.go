package quantcompat

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/quantmeta"
)

func descriptor(format quantmeta.Format, method string) quantmeta.Descriptor {
	return quantmeta.Descriptor{Schema: quantmeta.SchemaVersion,
		Artifact:   &quantmeta.ArtifactSpec{ContainerID: string(format)},
		Weight:     &quantmeta.Weight{Format: quantmeta.Format("int4"), Bits: 4},
		Provenance: quantmeta.ProvenanceSpec{MethodID: method}}
}

func TestAdjudicateOperatingEnvelope(t *testing.T) {
	q4 := descriptor(quantmeta.Format("gguf"), "groupwise")
	tests := []struct {
		name   string
		req    Request
		status Status
		reason Reason
	}{
		{"supported", Request{q4, Runtime{"llama.cpp", []quantmeta.Format{quantmeta.Format("gguf")}, []string{"groupwise"}, []string{"cpu"}, nil, nil}, "cpu"}, StatusDirect, ReasonCompatible},
		{"delegated", Request{q4, Runtime{"gateway", nil, nil, []string{"cpu"}, []quantmeta.Format{quantmeta.Format("gguf")}, nil}, "cpu"}, StatusExternalRuntime, ReasonExternalRuntime},
		{"conversion required", Request{q4, Runtime{"cuda-engine", nil, nil, []string{"cuda-sm90"}, nil, []quantmeta.Format{quantmeta.Format("gguf")}}, "cuda-sm90"}, StatusConversionRequired, ReasonConversionAvailable},
		{"unsupported format", Request{q4, Runtime{"native", []quantmeta.Format{quantmeta.Format("safetensors")}, []string{"groupwise"}, []string{"cpu"}, nil, nil}, "cpu"}, StatusRejected, ReasonFormatRejected},
		{"unsupported method", Request{q4, Runtime{"native", []quantmeta.Format{quantmeta.Format("gguf")}, []string{"tensorwise"}, []string{"cpu"}, nil, nil}, "cpu"}, StatusRejected, ReasonMethodRejected},
		{"unsupported hardware", Request{q4, Runtime{"native", []quantmeta.Format{quantmeta.Format("gguf")}, []string{"groupwise"}, []string{"cuda-sm90"}, nil, nil}, "cpu"}, StatusRejected, ReasonHardwareRejected},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := Adjudicate(tt.req)
			if got.Status != tt.status || got.Reason != tt.reason {
				t.Fatalf("Adjudicate() = %+v, want %s/%s", got, tt.status, tt.reason)
			}
		})
	}
}

func TestUnknownAndInvalidInputsFailClosed(t *testing.T) {
	base := descriptor(quantmeta.Format("gguf"), "groupwise")
	runtime := Runtime{"runtime", []quantmeta.Format{quantmeta.Format("gguf")}, []string{"groupwise"}, []string{"cpu"}, nil, nil}
	tests := []struct {
		name   string
		req    Request
		reason Reason
	}{
		{"unknown schema", Request{quantmeta.Descriptor{Schema: "quantmeta/v99", Artifact: &quantmeta.ArtifactSpec{ContainerID: "gguf"}}, runtime, "cpu"}, ReasonFormatUnknown},
		{"missing runtime", Request{base, Runtime{}, "cpu"}, ReasonRuntimeUnknown},
		{"missing hardware", Request{base, runtime, ""}, ReasonHardwareUnknown},
		{"invalid artifact", Request{quantmeta.Descriptor{Schema: quantmeta.SchemaVersion, Artifact: &quantmeta.ArtifactSpec{ContainerID: "gguf"}, Weight: &quantmeta.Weight{Format: quantmeta.Format("int4"), Granularity: quantmeta.GranularityPerGroup}, Provenance: quantmeta.ProvenanceSpec{MethodID: "groupwise"}}, runtime, "cpu"}, ReasonArtifactInvalid},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := Adjudicate(tt.req)
			if got.Status != StatusRejected || got.Reason != tt.reason {
				t.Fatalf("got %+v", got)
			}
		})
	}
}

func TestEveryDecisionHasStableReason(t *testing.T) {
	valid := map[Reason]bool{ReasonCompatible: true, ReasonExternalRuntime: true, ReasonConversionAvailable: true, ReasonArtifactInvalid: true, ReasonFormatUnknown: true, ReasonRuntimeUnknown: true, ReasonHardwareUnknown: true, ReasonFormatRejected: true, ReasonMethodRejected: true, ReasonHardwareRejected: true}
	got := Adjudicate(Request{})
	if !valid[got.Reason] {
		t.Fatalf("unregistered reason %q", got.Reason)
	}
}
