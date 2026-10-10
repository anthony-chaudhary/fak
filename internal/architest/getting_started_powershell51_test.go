package architest

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestGettingStartedPreservesJSONInWindowsPowerShell51(t *testing.T) {
	body, err := os.ReadFile(filepath.Join(repoRoot(t), "GETTING-STARTED.md"))
	if err != nil {
		t.Fatal(err)
	}
	text := string(body)
	for _, want := range []string{"Windows PowerShell 5.1", `--args '{\"_positional\":[\"alice\"]}'`, "reason=DEFAULT_DENY"} {
		if !strings.Contains(text, want) {
			t.Fatalf("GETTING-STARTED.md missing Windows preflight witness %q", want)
		}
	}
	if strings.Contains(text, "git-bash / PowerShell, where the shown syntax works unchanged") {
		t.Fatal("guide still claims one quoting form works across Windows PowerShell versions")
	}
}
