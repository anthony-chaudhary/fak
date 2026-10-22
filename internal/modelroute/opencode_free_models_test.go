package modelroute

import "testing"

func TestParseOpenCodeFreeModels(t *testing.T) {
	type disposition string
	const (
		survives disposition = "survives"
		rejected disposition = "rejected"
		skipped  disposition = "skipped"
	)
	cases := []struct {
		line string
		want disposition
	}{
		{"opencode/beam-free", survives},
		{"opencode/deepseek-v4.1-flash-free", survives},
		{"opencode/ll-alpha-free", survives},
		{"opencode/glm-5.3-flash-free", survives},
		{"opencode/space-bunny-free", survives},
		{"opencode/qwen3.6-27b", skipped},
		{"opencode-go/ox-alpha-free", survives},
		{"opencode/some-1b-free", rejected},
		{"opencode/tiny-model-free", rejected},
		{"opencode/compound-free", rejected},
		{"", skipped},
		{"   ", skipped},
		{"not-a-provider-model", skipped},
		{"/leading-slash", skipped},
		{"trailing/", skipped},
	}
	lines := make([]string, 0, len(cases))
	wantSurvivors := map[string]bool{}
	wantRejected := map[string]bool{}
	for _, tc := range cases {
		lines = append(lines, tc.line)
		switch tc.want {
		case survives:
			wantSurvivors[tc.line] = true
		case rejected:
			wantRejected[tc.line] = true
		}
	}
	got := ParseOpenCodeFreeModels(lines, "-free")
	survivorModels, rejectedModels := SplitOpenCodeFreeModels(got)

	found := false
	for _, s := range survivorModels {
		if !wantSurvivors[s.ID] {
			t.Fatalf("unexpected survivor: %+v", s)
		}
		delete(wantSurvivors, s.ID)
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
	if len(wantSurvivors) != 0 {
		t.Fatalf("missing survivors: %v", wantSurvivors)
	}

	for _, r := range rejectedModels {
		if !wantRejected[r.ID] {
			t.Fatalf("unexpected rejection: %+v", r)
		}
		delete(wantRejected, r.ID)
		if r.Reason == "" {
			t.Fatalf("rejected candidate %s has empty reason", r.ID)
		}
	}
	if len(wantRejected) != 0 {
		t.Fatalf("missing rejections: %v", wantRejected)
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
