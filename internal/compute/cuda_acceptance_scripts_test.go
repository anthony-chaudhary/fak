package compute_test

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestGPUAcceptanceScriptsDoNotRequireHOME(t *testing.T) {
	matches, err := filepath.Glob(filepath.Join("..", "..", "tools", "run_*_acceptance_on_gpu.sh"))
	if err != nil {
		t.Fatal(err)
	}
	if len(matches) == 0 {
		t.Fatal("no GPU acceptance scripts found")
	}
	for _, path := range matches {
		data, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		if strings.Contains(string(data), "${CUDA_HOME:-$HOME/") {
			t.Errorf("%s expands unset HOME under `set -u`; use ${HOME:-fallback} or a system-toolkit fallback", filepath.Base(path))
		}
	}
}
