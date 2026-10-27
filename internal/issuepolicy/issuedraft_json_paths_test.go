package issuepolicy

import (
	"reflect"
	"testing"
)

// fak-test:runtime fast est=10ms lane=default
func TestCandidateFromIssueDraftJSONRoutingPaths(t *testing.T) {
	tests := []struct {
		name  string
		key   string
		value string
		want  []string
	}{
		{
			name:  "cuda_header_and_go_witness",
			value: `["internal/compute/cuda/quant_matmul.cuh", "internal/compute/cuda_kquant_panel_test.go"]`,
			want:  []string{"internal/compute/cuda/quant_matmul.cuh", "internal/compute/cuda_kquant_panel_test.go"},
		},
		{
			name:  "cuda_source_header_docs_and_go",
			value: `["internal/compute/cuda/qwen35_attention_postprocess.cuh", "internal/compute/cuda_qwen35_attention_postprocess.go", "internal/compute/cuda_qwen35_attention_postprocess_test.go", "internal/compute/cuda_kernels.cu", "internal/compute/cuda_qwen35_sequence.go", "docs/model-engine-env.md", "internal/sotamatrix/sotamatrix.go"]`,
			want: []string{
				"docs/model-engine-env.md",
				"internal/compute/cuda/qwen35_attention_postprocess.cuh",
				"internal/compute/cuda_kernels.cu",
				"internal/compute/cuda_qwen35_attention_postprocess.go",
				"internal/compute/cuda_qwen35_attention_postprocess_test.go",
				"internal/compute/cuda_qwen35_sequence.go",
				"internal/sotamatrix/sotamatrix.go",
			},
		},
		{
			name:  "singular_key_duplicates_and_normalization",
			key:   "path",
			value: `["internal/compute/cuda/quant_matmul.cuh", "docs/model-engine-env.md", "internal\\compute\\cuda\\quant_matmul.cuh", "internal/compute/cuda/quant_matmul.cuh"]`,
			want:  []string{"docs/model-engine-env.md", "internal/compute/cuda/quant_matmul.cuh"},
		},
		{
			name:  "unchanged_path_validation",
			value: `["README.md", "internal/compute/bad name.go", "internal/compute/bad;name.go", "internal/compute/bad:12.go", "internal/compute/quant_matmul.cuh"]`,
			want:  []string{"internal/compute/quant_matmul.cuh"},
		},
		{
			name:  "legacy_comma_space_and_backticks",
			value: "internal/compute/cuda/quant_matmul.cuh, `internal/compute/cuda_kernels.cu`\tdocs/model-engine-env.md internal/compute/cuda_kernels.cu",
			want:  []string{"docs/model-engine-env.md", "internal/compute/cuda/quant_matmul.cuh", "internal/compute/cuda_kernels.cu"},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			key := tt.key
			if key == "" {
				key = "paths"
			}
			body := "```routing\nlane: compute\n" + key + ": " + tt.value + "\nexpected_steps: 3\n```\n" +
				"## Likely files\ninternal/fallback/prose.go\n## Lane\nfallback\n## Expected steps\n7\n"
			got := CandidateFromIssueDraft(IssueDraft{Number: 14059, Title: "perf(cuda): retain exact paths", Body: body})
			if !reflect.DeepEqual(got.Paths, tt.want) {
				t.Fatalf("Paths = %#v, want %#v", got.Paths, tt.want)
			}
			if got.Lane != "compute" || got.ExpectedSteps != 3 {
				t.Fatalf("routing precedence changed: lane=%q steps=%d", got.Lane, got.ExpectedSteps)
			}
		})
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestCandidateFromIssueDraftJSONRoutingPathsUnusableFallback(t *testing.T) {
	tests := []struct {
		name  string
		value string
	}{
		{"empty_array", `[]`},
		{"empty_strings", `["", "   "]`},
		{"all_paths_rejected", `["README.md", "internal/compute/bad name.go"]`},
		{"missing_close", `["internal/partial/header.cuh"`},
		{"trailing_data", `["internal/partial/header.cuh"] ignored`},
		{"numeric_member", `["internal/partial/header.cuh", 7]`},
		{"boolean_member", `["internal/partial/header.cuh", true]`},
		{"null_member", `["internal/partial/header.cuh", null]`},
		{"object_member", `["internal/partial/header.cuh", {"path":"internal/partial/kernel.cu"}]`},
		{"nested_array", `["internal/partial/header.cuh", ["internal/partial/kernel.cu"]]`},
		{"unquoted_members", `[internal/partial/header.cuh, internal/partial/other.cuh, internal/partial/kernel.cu]`},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			body := "```routing\nlane: compute\npaths: " + tt.value + "\nexpected_steps: 3\n```\n" +
				"## Likely files\ninternal/fallback/prose.go\n"
			got := CandidateFromIssueDraft(IssueDraft{Number: 14061, Title: "perf(cuda): preserve fallback", Body: body})
			want := []string{"internal/fallback/prose.go"}
			if !reflect.DeepEqual(got.Paths, want) {
				t.Fatalf("Paths = %#v, want fallback %#v (no partial JSON paths)", got.Paths, want)
			}
			if got.Lane != "compute" || got.ExpectedSteps != 3 {
				t.Fatalf("unusable paths changed other routing keys: lane=%q steps=%d", got.Lane, got.ExpectedSteps)
			}
		})
	}
	got := CandidateFromIssueDraft(IssueDraft{Number: 14061, Body: "```routing\npaths: []\n```\n## Scope\ninternal/fallback/body.go\ninternal/fallback/header.cuh\n"})
	if want := []string{"internal/fallback/body.go"}; !reflect.DeepEqual(got.Paths, want) {
		t.Fatalf("existing Go-only body fallback changed: got %#v, want %#v", got.Paths, want)
	}
}
