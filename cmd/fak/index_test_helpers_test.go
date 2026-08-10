package main

import (
	"os"
	"path/filepath"
	"testing"
)

// writeIndexRepo lays down the minimal index fixtures shared by command tests.
func writeIndexRepo(t *testing.T) string {
	t.Helper()
	root := t.TempDir()
	files := map[string]string{
		"dos.toml":  "[lanes.trees]\ngateway = [\"internal/gateway/**\"]\nsession = [\"internal/session/**\"]\n",
		"CLAIMS.md": "# CLAIMS.md\n## Gateway\n- [SHIPPED] internal/gateway speaks OpenAI at the front door.\n- [STUB] internal/gateway streaming backpressure is deferred.\n## Session\n- [SIMULATED] internal/session cost ring uses stand-in data.\n",
	}
	for name, body := range files {
		if err := os.WriteFile(filepath.Join(root, name), []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	if err := os.MkdirAll(filepath.Join(root, "docs"), 0o755); err != nil {
		t.Fatal(err)
	}
	generation := "# Generation Contract\n\n| Stream | Label | Milestone | Meaning |\n|---|---|---|---|\n| now | `gen/now` | `Generation G0 - Now / Immediate` | Current product work. |\n| next | `gen/next` | `Generation G1 - Next Gen` | Near-term foundation. |\n| second-next | `gen/second-next` | `Generation G2 - Second Next Gen` | Architectural option. |\n| future | `gen/future` | `Generation G3 - Future` | Long-horizon research. |\n"
	if err := os.WriteFile(filepath.Join(root, "docs", "generation.md"), []byte(generation), 0o644); err != nil {
		t.Fatal(err)
	}
	return root
}
