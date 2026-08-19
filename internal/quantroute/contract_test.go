package quantroute

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/quantcompat"
	"github.com/anthony-chaudhary/fak/internal/quantmeta"
)

func artifact() quantcompat.Request {
	return quantcompat.Request{Artifact: quantmeta.Descriptor{
		Schema:     quantmeta.SchemaVersion,
		Artifact:   &quantmeta.ArtifactSpec{ContainerID: "gguf"},
		Weight:     &quantmeta.Weight{Format: quantmeta.Format("int4"), Bits: 4},
		Provenance: quantmeta.ProvenanceSpec{MethodID: "groupwise"},
	}}
}

func runtime(id string, formats, external, conversions []quantmeta.Format) quantcompat.Runtime {
	return quantcompat.Runtime{ID: id, Formats: formats, Methods: []string{"groupwise"}, Hardware: []string{"cpu"}, ExternalFormats: external, ConvertibleFormats: conversions}
}

func TestSelectPreservesOrderAndFallsBackCompatibly(t *testing.T) {
	candidates := []Candidate{
		{Provider: "preferred", Runtime: runtime("preferred", []quantmeta.Format{quantmeta.Format("safetensors")}, nil, nil), Hardware: "cpu"},
		{Provider: "fallback", Runtime: runtime("fallback", []quantmeta.Format{quantmeta.Format("gguf")}, nil, nil), Hardware: "cpu"},
		{Provider: "later", Runtime: runtime("later", []quantmeta.Format{quantmeta.Format("gguf")}, nil, nil), Hardware: "cpu"},
	}
	got := Select(artifact(), candidates)
	if got.Code != CodeSelected || got.Index != 1 || got.Candidate == nil || got.Candidate.Provider != "fallback" {
		t.Fatalf("Select() = %+v", got)
	}
	if len(got.Evaluated) != 2 {
		t.Fatalf("evaluated %d candidates, want 2", len(got.Evaluated))
	}
}

func TestSelectAcceptsExplicitExternalRuntime(t *testing.T) {
	got := Select(artifact(), []Candidate{{Provider: "gateway", Runtime: runtime("gateway", nil, []quantmeta.Format{quantmeta.Format("gguf")}, nil), Hardware: "cpu"}})
	if got.Code != CodeSelected || got.Evaluated[0].Status != quantcompat.StatusExternalRuntime {
		t.Fatalf("Select() = %+v", got)
	}
}

func TestSelectNeverSilentlyConverts(t *testing.T) {
	got := Select(artifact(), []Candidate{{Provider: "converter", Runtime: runtime("converter", nil, nil, []quantmeta.Format{quantmeta.Format("gguf")}), Hardware: "cpu"}})
	if got.Code != CodeConversionOnly || got.Candidate != nil || got.Index != -1 {
		t.Fatalf("Select() = %+v", got)
	}
}

func TestSelectUnknownAndUnsupportedRefuseExplicitly(t *testing.T) {
	tests := []struct {
		name       string
		request    quantcompat.Request
		candidates []Candidate
		code       Code
	}{
		{"empty", artifact(), nil, CodeEmptyInput},
		{"unknown artifact", quantcompat.Request{Artifact: quantmeta.Descriptor{Schema: "quantmeta/v99"}}, []Candidate{{Provider: "p", Runtime: runtime("p", []quantmeta.Format{quantmeta.Format("gguf")}, nil, nil), Hardware: "cpu"}}, CodeNoCompatibleTarget},
		{"unsupported", artifact(), []Candidate{{Provider: "p", Runtime: runtime("p", []quantmeta.Format{quantmeta.Format("safetensors")}, nil, nil), Hardware: "cpu"}}, CodeNoCompatibleTarget},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := Select(tt.request, tt.candidates)
			if got.Code != tt.code || got.Candidate != nil {
				t.Fatalf("Select() = %+v, want %s", got, tt.code)
			}
		})
	}
}
