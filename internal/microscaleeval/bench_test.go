package microscaleeval

import "testing"

func BenchmarkMicroScaleEval(b *testing.B) {
	ocpOperand := Operand{ElementFormat: "e2m1", Recovery: "none"}
	req := Request{
		Descriptor: Descriptor{
			Schema:      SchemaV1,
			Family:      "ocp-mx-v1",
			BlockSize:   32,
			ScaleFormat: "e8m0",
			Weights:     ocpOperand,
			Activations: ocpOperand,
		},
		Capabilities: RuntimeCapabilities{
			Profiles: []NativeProfile{{
				Family:      "ocp-mx-v1",
				BlockSize:   32,
				ScaleFormat: "e8m0",
				Weights:     ocpOperand,
				Activations: ocpOperand,
			}},
		},
		Provenance: Provenance{
			Artifact: Pin{ID: "artifact", Revision: "v1", SHA256: digest},
			Recipe:   Pin{ID: "recipe", Revision: "v1", SHA256: digest},
			Runtime:  Pin{ID: "runtime", Revision: "13.3", SHA256: digest},
			Model:    Pin{ID: "model", Revision: "commit-1", SHA256: digest},
		},
		Evidence: Evidence{
			Kind:   EvidenceModeled,
			Source: "arxiv:2608.03867v1",
		},
	}

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		v := Evaluate(req)
		if v.Outcome != Native {
			b.Fatalf("unexpected benchmark evaluation outcome: %v", v.Outcome)
		}
	}
}

func TestBenchmarkSmoke(t *testing.T) {
	ocpOperand := Operand{ElementFormat: "e2m1", Recovery: "none"}
	req := Request{
		Descriptor: Descriptor{
			Schema:      SchemaV1,
			Family:      "ocp-mx-v1",
			BlockSize:   32,
			ScaleFormat: "e8m0",
			Weights:     ocpOperand,
			Activations: ocpOperand,
		},
		Capabilities: RuntimeCapabilities{
			Profiles: []NativeProfile{{
				Family:      "ocp-mx-v1",
				BlockSize:   32,
				ScaleFormat: "e8m0",
				Weights:     ocpOperand,
				Activations: ocpOperand,
			}},
		},
		Provenance: Provenance{
			Artifact: Pin{ID: "artifact", Revision: "v1", SHA256: digest},
			Recipe:   Pin{ID: "recipe", Revision: "v1", SHA256: digest},
			Runtime:  Pin{ID: "runtime", Revision: "13.3", SHA256: digest},
			Model:    Pin{ID: "model", Revision: "commit-1", SHA256: digest},
		},
		Evidence: Evidence{
			Kind:   EvidenceModeled,
			Source: "arxiv:2608.03867v1",
		},
	}
	v := Evaluate(req)
	if v.Outcome != Native {
		t.Fatalf("unexpected outcome: %v", v.Outcome)
	}
}
