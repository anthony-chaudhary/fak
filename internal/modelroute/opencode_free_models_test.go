package modelroute

import "testing"

func TestParseOpenCodeFreeModels(t *testing.T) {
	lines := []string{
		"opencode/beam-free",
		"opencode/deepseek-v4.1-flash-free",
		"opencode/ll-alpha-free",
		"opencode/glm-5.3-flash-free",
		"opencode/space-bunny-free",
		"opencode/qwen3.6-27b",      // not free suffix -> skipped
		"opencode-go/ox-alpha-free", // free suffix + opencode-go provider
		"opencode/some-1b-free",     // deny token 1b -> rejected
		"opencode/tiny-model-free",  // deny token tiny -> rejected
		"opencode/compound-free",    // deny token compound -> rejected
		"",
		"   ",
		"not-a-provider-model", // no slash -> skipped
		"/leading-slash",       // empty provider -> skipped
		"trailing/",            // empty name -> skipped
	}
	got := ParseOpenCodeFreeModels(lines, "-free")
	survivors, rejected := SplitOpenCodeFreeModels(got)
	if len(survivors) != 6 {
		t.Fatalf("expected 6 survivors, got %d", len(survivors))
	}
	if len(rejected) != 3 {
		t.Fatalf("expected 3 rejected, got %d", len(rejected))
	}

	// Verify a specific survivor.
	found := false
	for _, s := range survivors {
		if s.ID == "opencode/ll-alpha-free" {
			found = true
			if !s.QualityOK {
				t.Fatalf("ll-alpha-free should pass quality gate, got reason: %s", s.Reason)
			}
			if s.Provider != "opencode" || s.Name != "ll-alpha-free" {
				t.Fatalf("bad parse: %+v", s)
			}
		}
	}
	if !found {
		t.Fatal("opencode/ll-alpha-free not in survivors")
	}

	// Verify rejected reasons mention deny tokens.
	for _, r := range rejected {
		if r.Reason == "" {
			t.Fatalf("rejected candidate %s has empty reason", r.ID)
		}
	}
}

func TestParseOpenCodeFreeModelsDefaultSuffix(t *testing.T) {
	lines := []string{"opencode/foo-free", "opencode/bar"}
	got := ParseOpenCodeFreeModels(lines, "")
	if len(got) != 1 {
		t.Fatalf("expected 1 candidate with default suffix, got %d", len(got))
	}
	if got[0].ID != "opencode/foo-free" {
		t.Fatalf("bad id: %s", got[0].ID)
	}
}

func TestParseOpenCodeFreeModelsCustomSuffix(t *testing.T) {
	lines := []string{"opencode/foo-preview", "opencode/bar-free"}
	got := ParseOpenCodeFreeModels(lines, "-preview")
	if len(got) != 1 || got[0].ID != "opencode/foo-preview" {
		t.Fatalf("custom suffix parse failed: %+v", got)
	}
}
