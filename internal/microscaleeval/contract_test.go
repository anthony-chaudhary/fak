package microscaleeval

import "testing"

const digest = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"

func pinned() Provenance {
	return Provenance{
		Artifact: Pin{ID: "artifact", Revision: "v1", SHA256: digest},
		Recipe:   Pin{ID: "recipe", Revision: "v1", SHA256: digest},
		Runtime:  Pin{ID: "runtime", Revision: "13.3", SHA256: digest},
		Model:    Pin{ID: "model", Revision: "commit-1", SHA256: digest},
	}
}

func modeled() Evidence { return Evidence{Kind: EvidenceModeled, Source: "arxiv:2608.03867v1"} }

func TestNamedWitnessProducesClosedOutcomes(t *testing.T) {
	ocpOperand := Operand{ElementFormat: "e2m1", Recovery: "none"}
	tests := []struct {
		name   string
		req    Request
		want   Disposition
		reason Reason
	}{
		{
			name: "supported exact OCP MXFP4 profile",
			req: Request{
				Descriptor:   Descriptor{Schema: SchemaV1, Family: "ocp-mx-v1", BlockSize: 32, ScaleFormat: "e8m0", Weights: ocpOperand, Activations: ocpOperand},
				Capabilities: RuntimeCapabilities{Profiles: []NativeProfile{{Family: "ocp-mx-v1", BlockSize: 32, ScaleFormat: "e8m0", Weights: ocpOperand, Activations: ocpOperand}}},
				Provenance:   pinned(), Evidence: modeled(),
			},
			want: Native, reason: ReasonNativeMatch,
		},
		{
			name: "delegate AdaMX per-block and per-operand heterogeneity",
			req: Request{
				Descriptor:   Descriptor{Schema: SchemaV1, Family: "adamx-paper-v1", BlockSize: 16, ScaleFormat: "e8m0", Weights: Operand{ElementFormat: "adaptive-fp4-int4", Recovery: "adaptive-outlier"}, Activations: Operand{ElementFormat: "adaptive-fp4", Recovery: "adaptive-microexponent"}, PerBlockMode: true},
				Capabilities: RuntimeCapabilities{Profiles: []NativeProfile{{Family: "ocp-mx-v1", BlockSize: 32, ScaleFormat: "e8m0", Weights: ocpOperand, Activations: ocpOperand}}, Delegate: "adamx-reference-decoder@arxiv:2608.03867v1"},
				Provenance:   pinned(), Evidence: modeled(),
			},
			want: Forward, reason: ReasonHeterogeneousRuntime,
		},
		{
			name: "unsupported unknown family",
			req: Request{
				Descriptor: Descriptor{Schema: SchemaV1, Family: "future-mx-v9", BlockSize: 7, ScaleFormat: "mystery", Weights: ocpOperand, Activations: ocpOperand},
				Provenance: pinned(), Evidence: modeled(),
			},
			want: Refuse, reason: ReasonUnsupportedFormat,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := Evaluate(tt.req)
			if got.Outcome != tt.want || got.Reason != tt.reason {
				t.Fatalf("got %s/%s, want %s/%s: %s", got.Outcome, got.Reason, tt.want, tt.reason, got.Detail)
			}
			if got.Provenance != tt.req.Provenance || got.Evidence != tt.req.Evidence {
				t.Fatal("verdict did not preserve provenance/evidence")
			}
		})
	}
}

func TestUnknownSchemaAndIncompletePinsRefuse(t *testing.T) {
	d := Descriptor{Schema: "fak.microscale-eval/v2", Family: "ocp-mx-v1", BlockSize: 32, ScaleFormat: "e8m0", Weights: Operand{"e2m1", "none"}, Activations: Operand{"e2m1", "none"}}
	got := Evaluate(Request{Descriptor: d, Provenance: pinned(), Evidence: modeled()})
	if got.Outcome != Refuse || got.Reason != ReasonUnknownSchema {
		t.Fatalf("got %#v", got)
	}

	d.Schema = SchemaV1
	p := pinned()
	p.Recipe.SHA256 = "unpinned"
	got = Evaluate(Request{Descriptor: d, Provenance: p, Evidence: modeled()})
	if got.Reason != ReasonInvalidProvenance {
		t.Fatalf("got %#v", got)
	}
}

func TestObservedCannotBeClaimedWithoutRunAndHardwareWitness(t *testing.T) {
	op := Operand{"e2m1", "none"}
	r := Request{Descriptor: Descriptor{Schema: SchemaV1, Family: "ocp-mx-v1", BlockSize: 32, ScaleFormat: "e8m0", Weights: op, Activations: op}, Provenance: pinned(), Evidence: Evidence{Kind: EvidenceObserved, Source: "local"}}
	got := Evaluate(r)
	if got.Outcome != Refuse || got.Reason != ReasonInvalidProvenance {
		t.Fatalf("got %#v", got)
	}

	r.Evidence = Evidence{Kind: EvidenceObserved, Source: "lab-run-1", RunSHA256: digest, HardwareFingerprint: "gpu-arch=example;runtime=example"}
	got = Evaluate(r)
	if got.Reason == ReasonInvalidProvenance {
		t.Fatalf("complete observed witness rejected: %#v", got)
	}
}

func TestVendorFP4DoesNotSilentlyMatchOCP(t *testing.T) {
	op := Operand{"e2m1", "none"}
	r := Request{
		Descriptor:   Descriptor{Schema: SchemaV1, Family: "nvidia-nvfp4", BlockSize: 16, ScaleFormat: "ue4m3", Weights: op, Activations: op},
		Capabilities: RuntimeCapabilities{Profiles: []NativeProfile{{Family: "ocp-mx-v1", BlockSize: 32, ScaleFormat: "e8m0", Weights: op, Activations: op}}},
		Provenance:   pinned(), Evidence: modeled(),
	}
	got := Evaluate(r)
	if got.Outcome != Refuse || got.Reason != ReasonRuntimeMismatch {
		t.Fatalf("got %#v", got)
	}
}
