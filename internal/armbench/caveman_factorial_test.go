package armbench

import (
	"encoding/json"
	"testing"
)

func TestCavemanFactorialCrossesFullMatrixAndSeparatesEvidence(t *testing.T) {
	m, err := RunCavemanFactorial(FactorialOptions{Pressures: []int{1, 4, 12}})
	if err != nil {
		t.Fatal(err)
	}
	if got, want := len(m.Cells), 2*5*2*3; got != want {
		t.Fatalf("cells=%d want=%d", got, want)
	}
	seen := map[string]bool{}
	for _, c := range m.Cells {
		seen[c.Style+"/"+string(c.Treatment)+"/"+c.Workload] = true
		if c.ProviderInputTokens != nil || c.ProviderCacheReadTokens != nil || c.ProviderCacheWriteTokens != nil {
			t.Fatalf("deterministic cell fabricated provider receipt: %+v", c)
		}
		if c.RetainedFacts > c.TotalFacts || c.Quality < 0 || c.Quality > 1 {
			t.Fatalf("bad quality: %+v", c)
		}
		if len(c.Stages) != 3 {
			t.Fatalf("stages=%d", len(c.Stages))
		}
		for _, s := range c.Stages {
			if s.CPUTimeNS < 1 || s.BytesBefore < 1 || s.BytesAfter < 1 || s.TokensBefore < 1 || s.TokensAfter < 1 {
				t.Fatalf("incomplete stage receipt: %+v", s)
			}
		}
	}
	if len(seen) != 20 {
		t.Fatalf("style/treatment/workload combinations=%d want=20", len(seen))
	}
	if len(m.Interactions) != 2*3*4 {
		t.Fatalf("interactions=%d want=24", len(m.Interactions))
	}
	if len(m.QualityFrontier) == 0 {
		t.Fatal("missing quality frontier")
	}
	b, err := json.Marshal(m)
	if err != nil {
		t.Fatal(err)
	}
	for _, needle := range []string{"provider_input_tokens", "provider_cache_read_tokens", "answer_output_tokens_estimated", "retained_facts", "interaction_effects", "quality_frontier"} {
		if !json.Valid(b) || !containsBytes(b, []byte(needle)) {
			t.Fatalf("manifest missing %q", needle)
		}
	}
}

func TestCavemanFactorialTransformsGrowingContext(t *testing.T) {
	m, err := RunCavemanFactorial(FactorialOptions{Pressures: []int{1, 4, 12}})
	if err != nil {
		t.Fatal(err)
	}
	var pass, both FactorialCell
	for _, c := range m.Cells {
		if c.Style == "normal" && c.Workload == "multi-turn-growing-tool-results" && c.Pressure == 12 {
			if c.Treatment == TreatmentPassthrough {
				pass = c
			}
			if c.Treatment == TreatmentBoth {
				both = c
			}
		}
	}
	t.Logf("pass=%+v both=%+v", pass.Stages, both.Stages)
	if finalBytes(both) >= finalBytes(pass) {
		t.Fatalf("both bytes=%d, passthrough=%d; transformation spine inactive", finalBytes(both), finalBytes(pass))
	}
	if both.RetainedFacts == 0 {
		t.Fatal("both treatment retained no task facts")
	}
}

func containsBytes(haystack, needle []byte) bool {
	for i := 0; i+len(needle) <= len(haystack); i++ {
		ok := true
		for j := range needle {
			if haystack[i+j] != needle[j] {
				ok = false
				break
			}
		}
		if ok {
			return true
		}
	}
	return false
}
