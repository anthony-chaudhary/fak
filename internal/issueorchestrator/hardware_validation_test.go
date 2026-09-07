package issueorchestrator

import (
	"strings"
	"testing"
)

func TestFormatOpencodePromptHardwareValidation(t *testing.T) {
	for _, tc := range []struct {
		name  string
		issue Issue
		want  bool
	}{
		{"AMD title", Issue{Title: "perf: accelerate Strix Halo decode"}, true},
		{"compute path", Issue{Paths: []string{`internal\compute\matmul.go`}}, true},
		{"native performance lane", Issue{Lane: "nativeperf"}, true},
		{"model performance", Issue{Title: "perf: reduce decode latency", Paths: []string{"internal/model/decode.go"}}, true},
		{"model correctness unrelated", Issue{Title: "fix: input error", Lane: "model"}, false},
		{"Vulkan label", Issue{Labels: []string{"area/vulkan"}}, true},
		{"unrelated", Issue{Title: "fix: memory leak", Lane: "ctxmmu"}, false},
		{"shallow copy software", Issue{Title: "fix: prevent shallow copy in ctxmmu", Lane: "ctxmmu"}, false},
		{"prefix sibling", Issue{Paths: []string{"internal/computercatalog/catalog.go"}}, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			prompt := FormatOpencodePrompt(tc.issue)
			if got := strings.Contains(prompt, "Physical hardware validation:"); got != tc.want {
				t.Fatalf("hardware guidance=%v, want %v", got, tc.want)
			}
			if !tc.want {
				return
			}
			for _, required := range []string{"During investigation", "fak-dev amd-strix-probe", "shared lease", "exclusive lease", "flock -w 30 -x", "do not nest locks", "source-bound receipt", "dirty patch digest", "built executable digest", "fak-native", "matched baseline/candidate", "hardware validation PENDING"} {
				if !strings.Contains(prompt, required) {
					t.Errorf("missing %q", required)
				}
			}
		})
	}
}
