// Tests for the pure helpers in cmd/pipelinegen/main.go: parseIDs (CSV token
// parsing), isGLMDsa (architecture detection), and snapToFullIndexer (pipeline
// cut snapping). All are deterministic and need no model file, network, or
// process — they operate purely on their inputs.
package main

import (
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/model"
)

func TestParseIDs(t *testing.T) {
	tests := []struct {
		name    string
		in      string
		want    []int
		wantErr bool
	}{
		{name: "simple", in: "3,1,4,1", want: []int{3, 1, 4, 1}},
		{name: "spaces trimmed", in: "3, 1 , 4", want: []int{3, 1, 4}},
		{name: "empty fields skipped", in: "3,,4", want: []int{3, 4}},
		{name: "trailing comma", in: "7,8,", want: []int{7, 8}},
		{name: "negative ids", in: "-1,0,2", want: []int{-1, 0, 2}},
		{name: "all empty", in: "", want: []int{}},
		{name: "only commas", in: ",,", want: []int{}},
		{name: "non numeric", in: "3,x,4", wantErr: true},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got, err := parseIDs(tc.in)
			if tc.wantErr {
				if err == nil {
					t.Fatalf("parseIDs(%q) = %v, want error", tc.in, got)
				}
				return
			}
			if err != nil {
				t.Fatalf("parseIDs(%q) unexpected error: %v", tc.in, err)
			}
			if !reflect.DeepEqual(got, tc.want) {
				t.Fatalf("parseIDs(%q) = %v, want %v", tc.in, got, tc.want)
			}
		})
	}
}

func TestIsGLMDsa(t *testing.T) {
	tests := []struct {
		name string
		cfg  model.Config
		want bool
	}{
		{
			name: "model_type glm_moe_dsa",
			cfg:  model.Config{ModelType: "glm_moe_dsa"},
			want: true,
		},
		{
			name: "glm via arch dsa via arch",
			cfg:  model.Config{ModelType: "glm_moe", Architectures: []string{"GlmMoeDsaForCausalLM"}},
			want: true,
		},
		{
			name: "glm only no dsa",
			cfg:  model.Config{ModelType: "glm_moe", Architectures: []string{"GlmMoeForCausalLM"}},
			want: false,
		},
		{
			name: "dsa only no glm",
			cfg:  model.Config{ModelType: "dsa_something"},
			want: false,
		},
		{
			name: "unrelated",
			cfg:  model.Config{ModelType: "llama", Architectures: []string{"LlamaForCausalLM"}},
			want: false,
		},
		{
			name: "uppercase model_type",
			cfg:  model.Config{ModelType: "GLM_MOE_DSA"},
			want: true,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if got := isGLMDsa(tc.cfg); got != tc.want {
				t.Fatalf("isGLMDsa(%+v) = %v, want %v", tc.cfg, got, tc.want)
			}
		})
	}
}

func TestSnapToFullIndexer(t *testing.T) {
	glm := model.Config{
		ModelType:    "glm_moe_dsa",
		NumLayers:    3,
		IndexerTypes: []string{"full", "shared", "full"},
	}
	tests := []struct {
		name string
		cfg  model.Config
		c    int
		want int
	}{
		// c=0 is already a "full" layer -> unchanged.
		{name: "already full", cfg: glm, c: 0, want: 0},
		// c=1 is "shared" -> advance to next non-shared layer (index 2, "full").
		{name: "snap past shared", cfg: glm, c: 1, want: 2},
		// c=2 is "full" -> unchanged.
		{name: "full stays", cfg: glm, c: 2, want: 2},
		// c beyond IndexerTypes length -> returned as-is (index guard).
		{name: "beyond indexer types", cfg: glm, c: 3, want: 3},
		// Non-GLM model -> always passthrough.
		{
			name: "non glm passthrough",
			cfg:  model.Config{ModelType: "llama", NumLayers: 4, IndexerTypes: []string{"shared", "shared"}},
			c:    1,
			want: 1,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if got := snapToFullIndexer(tc.cfg, tc.c); got != tc.want {
				t.Fatalf("snapToFullIndexer(c=%d) = %d, want %d", tc.c, got, tc.want)
			}
		})
	}
}
