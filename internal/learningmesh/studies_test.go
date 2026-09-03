package learningmesh

import (
	"reflect"
	"testing"
)

// TestIngestStudiesAllFiveProviders verifies the core requirement of #9887:
// studies from llama.cpp, vLLM, Dynamo, SGLang, and MLX are ingested into
// deterministic provider-neutral mechanisms and cross-framework targets with no duplicate IDs.
func TestIngestStudiesAllFiveProviders(t *testing.T) {
	studies := []ProviderStudyInput{
		{
			Provider:     "llama.cpp",
			StudyDocPath: "docs/notes/CONCEPT-STUDY-LLAMACPP-2026-08-26.md",
			Revision:     "05454413d0a05e5ff6c22b43be7c25344c96d35b",
			RecordDigest: "355f189af78bb7a680f0d3541c2092a4b63e6f2fc7de973d00547441b28f85b9",
			Mechanisms: []StudyMechanism{
				{
					ID:                 "llamacpp-gqa-layout",
					Name:               "contiguous GQA layout",
					Rule:               "Borrow layout mechanism, not comparator runtime.",
					Hardware:           "amd",
					Backend:            "hip",
					Model:              "qwen3.8",
					Workload:           "decode",
					DefaultDisposition: Adapt,
				},
			},
		},
		{
			Provider:     "vllm",
			StudyDocPath: "docs/notes/CONCEPT-STUDY-VLLM-2026-07-18.md",
			Revision:     "db6f55e35bcd4f43326110be29bf52aa48e19a88",
			RecordDigest: "e4b1d66263157c948fa8aef0bfd2f029350d33e3464b92d0f1000af16c6f1090",
			Mechanisms: []StudyMechanism{
				{
					ID:                 "vllm-paged-blocks",
					Name:               "paged attention block table",
					Rule:               "Decouple logical sequence from contiguous memory.",
					Hardware:           "nvidia",
					Backend:            "cuda",
					Model:              "qwen3.8",
					Workload:           "serving",
					DefaultDisposition: Copy,
				},
			},
		},
		{
			Provider:     "dynamo",
			StudyDocPath: "docs/notes/CONCEPT-STUDY-DYNAMO-EXHAUSTIVE-2026-08-26.md",
			Revision:     "cd0da5fc3f28c2d968737c542bc562641464e451",
			RecordDigest: "ce2bec1330bff542e335faf477f95f90a706a728868e8cdbd44aa8c0acf6fc5c",
			Mechanisms: []StudyMechanism{
				{
					ID:                 "dynamo-nixl-transport",
					Name:               "NIXL non-blocking KV transfer",
					Rule:               "Unified communication backend over UCX/RDMA.",
					Hardware:           "nvidia",
					Backend:            "cuda",
					Model:              "qwen3.8",
					Workload:           "serving",
					DefaultDisposition: Adapt,
				},
			},
		},
		{
			Provider:     "sglang",
			StudyDocPath: "docs/notes/CONCEPT-STUDY-SGLANG-2026-07-18.md",
			Revision:     "b8ec544946f1c5b6e17a919a691b05c5b3e7af84",
			RecordDigest: "b8ec544946f1c5b6e17a919a691b05c5b3e7af84",
			Mechanisms: []StudyMechanism{
				{
					ID:                 "sglang-radix-lpm",
					Name:               "longest-prefix-match queue ordering",
					Rule:               "Order request queue by shared prefix length.",
					Hardware:           "nvidia",
					Backend:            "cuda",
					Model:              "qwen3.8",
					Workload:           "serving",
					DefaultDisposition: Adapt,
				},
			},
		},
		{
			Provider:     "mlx",
			StudyDocPath: "docs/notes/study-mlx-dspark-borrow-scout-2026-07-10.md",
			Revision:     "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2",
			RecordDigest: "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2",
			Mechanisms: []StudyMechanism{
				{
					ID:                 "mlx-unified-memory-stage",
					Name:               "zero-copy unified memory staging",
					Rule:               "Leverage Apple silicon unified memory architecture.",
					Hardware:           "apple",
					Backend:            "metal",
					Model:              "qwen3.8",
					Workload:           "decode",
					DefaultDisposition: Adapt,
				},
			},
		},
	}

	targets := []Envelope{
		{
			ID:        "target-fak-native-nvidia",
			Hardware:  "nvidia",
			Backend:   "cuda",
			Framework: "fak-native",
			Engine:    "fak-native",
			Model:     "qwen3.8",
			Workload:  "serving",
			Role:      "product",
		},
		{
			ID:        "target-fak-native-apple",
			Hardware:  "apple",
			Backend:   "metal",
			Framework: "fak-native",
			Engine:    "fak-native",
			Model:     "qwen3.8",
			Workload:  "decode",
			Role:      "product",
		},
	}

	ledger1, err := IngestStudies(studies, targets)
	if err != nil {
		t.Fatalf("IngestStudies failed: %v", err)
	}

	ledger2, err := IngestStudies(studies, targets)
	if err != nil {
		t.Fatalf("IngestStudies failed: %v", err)
	}

	if !reflect.DeepEqual(ledger1, ledger2) {
		t.Fatalf("IngestStudies is not deterministic")
	}

	if len(ledger1.Mechanisms) != 5 { //boundarylint:ignore CHANGE_DETECTOR_TEST a deliberate fixed-width invariant
		t.Fatalf("expected 5 mechanisms, got %d", len(ledger1.Mechanisms))
	}

	// Verify all 5 providers are present in the compiled mechanisms
	providersSeen := make(map[string]bool)
	for _, m := range ledger1.Mechanisms {
		providersSeen[m.Source.Framework] = true
		if m.Source.Role != "baseline" {
			t.Errorf("mechanism %s source role = %q, want baseline", m.ID, m.Source.Role)
		}
		if len(m.Rules) == 0 || m.Rules[0].Target.Engine != "fak-native" {
			t.Errorf("mechanism %s target engine = %q, want fak-native", m.ID, m.Rules[0].Target.Engine)
		}
	}

	for _, p := range []string{"llama.cpp", "vllm", "dynamo", "sglang", "mlx"} {
		if !providersSeen[p] {
			t.Errorf("missing provider %q in compiled mechanisms", p)
		}
	}

	// Compile through learningmesh.Compile
	candidateSet, err := Compile(ledger1)
	if err != nil {
		t.Fatalf("Compile failed: %v", err)
	}
	if candidateSet.CandidateCount == 0 {
		t.Fatalf("Compile produced 0 candidates")
	}
}

// TestIngestStudiesDuplicateIDRefused verifies that duplicate mechanism IDs are caught and refused.
func TestIngestStudiesDuplicateIDRefused(t *testing.T) {
	studies := []ProviderStudyInput{
		{
			Provider:     "vllm",
			StudyDocPath: "docs/notes/vllm.md",
			Revision:     "rev1",
			Mechanisms:   []StudyMechanism{{ID: "dup-id", Name: "mech 1"}},
		},
		{
			Provider:     "sglang",
			StudyDocPath: "docs/notes/sglang.md",
			Revision:     "rev2",
			Mechanisms:   []StudyMechanism{{ID: "dup-id", Name: "mech 2"}},
		},
	}
	targets := []Envelope{{ID: "t1", Engine: "fak-native"}}

	_, err := IngestStudies(studies, targets)
	if err == nil {
		t.Fatalf("expected error on duplicate mechanism ID")
	}
}
