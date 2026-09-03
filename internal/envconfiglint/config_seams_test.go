package envconfiglint

import (
	"os"
	"path/filepath"
	"testing"
)

func TestReleaseGateConfigReadsStayOnExplicitSeams(t *testing.T) {
	root := filepath.Join("..", "..")
	for file, forbidden := range map[string]string{
		"internal/guardsessions/guardsessions.go": "FAK_HOST_RECOVERY_SESSION",
		"cmd/fak/service_windows.go":              "FAK_INTERACTIVE_REGISTRY_DIR",
	} {
		src, err := os.ReadFile(filepath.Join(root, filepath.FromSlash(file)))
		if err != nil {
			t.Fatalf("read %s: %v", file, err)
		}
		for _, name := range ScanGoEnvReads(string(src)) {
			if name == forbidden {
				t.Errorf("%s: hidden config read %s returned; use the explicit parameter/config seam", file, name)
			}
		}
	}
}

func TestAdmittedPostFreezeContainsNoDuplicates(t *testing.T) {
	seen := make(map[string]bool, len(admittedPostFreeze))
	for _, name := range admittedPostFreeze {
		if seen[name] {
			t.Errorf("duplicate entry in admittedPostFreeze: %s", name)
		}
		seen[name] = true
	}
}
