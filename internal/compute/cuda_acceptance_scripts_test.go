package compute_test

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestGPUAcceptanceScriptsResolveCleanEnvironmentToolkit(t *testing.T) {
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
		script := string(data)
		if strings.Contains(script, "${CUDA_HOME:-$HOME/") {
			t.Errorf("%s expands unset HOME under `set -u`", filepath.Base(path))
		}
		if !strings.Contains(script, "[ -x /usr/local/cuda/bin/nvcc ]") {
			t.Errorf("%s does not probe the standard CUDA toolkit outside PATH", filepath.Base(path))
		}
	}
}
