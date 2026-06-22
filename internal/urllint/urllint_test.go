package urllint

import (
	"os"
	"path/filepath"
	"testing"
)

// TestDownloadURLMatcher verifies the witness's matcher: a /resolve/ download url is
// caught, while a plain model-page / search link (which downloads nothing) is not.
func TestDownloadURLMatcher(t *testing.T) {
	cases := []struct {
		url   string
		match bool
	}{
		{"https://huggingface.co/mradermacher/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/x.gguf", true},
		{"https://hf-mirror.com/Qwen/Qwen2.5-0.5B-Instruct/resolve/main/tokenizer.json", true},
		{"https://huggingface.co/%s/resolve/main/%s", true}, // a Sprintf format counts
		{"https://huggingface.co/models?search=gguf qwen2.5 instruct", false},
		{"https://example.com/whatever/resolve/main/x", false},
		{"not a url at all", false},
	}
	for _, tc := range cases {
		if got := downloadURLRe.MatchString(tc.url); got != tc.match {
			t.Errorf("downloadURLRe.MatchString(%q) = %v, want %v", tc.url, got, tc.match)
		}
	}
}

// TestNoUnchokepointedDownloadURLs is the live enforcement: every model/tokenizer
// download url literal in the repo's Go source must live in the audited chokepoint
// (cmd/simpledemo/main.go, where modelDownload derives them and the network-gated
// reachability test verifies them). A new command pasting its own url fails here.
func TestNoUnchokepointedDownloadURLs(t *testing.T) {
	root := repoRoot(t)
	allow := map[string]bool{
		"cmd/simpledemo/main.go": true, // the single audited download-url builder
	}
	offenses, err := ScanForDownloadURLs(root, allow)
	if err != nil {
		t.Fatalf("scan: %v", err)
	}
	if len(offenses) > 0 {
		t.Errorf("%d hardcoded download url(s) outside the audited builder:", len(offenses))
		for _, o := range offenses {
			t.Errorf("  %s", o)
		}
		t.Errorf("fix: derive the url via cmd/simpledemo's modelDownload (covered by the reachability test), or add the file to the allowlist with a reason")
	}
}

// repoRoot walks up from the test's working directory to the module root (go.mod).
func repoRoot(t *testing.T) string {
	t.Helper()
	dir, err := os.Getwd()
	if err != nil {
		t.Fatalf("getwd: %v", err)
	}
	for {
		if _, err := os.Stat(filepath.Join(dir, "go.mod")); err == nil {
			return dir
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			t.Fatalf("no go.mod found above %s", dir)
		}
		dir = parent
	}
}
