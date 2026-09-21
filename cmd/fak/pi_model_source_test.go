package main

import (
	"strings"
	"testing"
)

// TestPiModelSource pins the dry-run provenance label. It is display-only, but a
// wrong label is exactly how the auto-detect clobber stayed invisible (fak#13467):
// the operator could not tell a deliberate default from a /healthz adoption.
func TestPiModelSource(t *testing.T) {
	cases := []struct {
		name       string
		explicit   string
		adopted    bool
		wantSubstr string
	}{
		{"explicit flag wins", "deepseek-ai/DeepSeek-V4.1-Flash", false, "explicit --model"},
		{"explicit beats adopted", "deepseek-ai/DeepSeek-V4.1-Flash", true, "explicit --model"},
		{"adopted detect", "", true, "auto-detected"},
		{"configured default", "", false, "configured"},
		{"whitespace explicit is not explicit", "   ", false, "configured"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := piModelSource(tc.explicit, tc.adopted)
			if !strings.Contains(got, tc.wantSubstr) {
				t.Fatalf("piModelSource(%q, %v) = %q, want it to contain %q",
					tc.explicit, tc.adopted, got, tc.wantSubstr)
			}
		})
	}
}
