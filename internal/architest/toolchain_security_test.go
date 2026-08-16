package architest

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestPinnedToolchainIncludesGo126SecurityFixes(t *testing.T) {
	root := filepath.Dir(internalDir(t))
	data, err := os.ReadFile(filepath.Join(root, "go.mod"))
	if err != nil {
		t.Fatalf("read go.mod: %v", err)
	}

	var directive string
	for line := range strings.Lines(string(data)) {
		fields := strings.Fields(line)
		if len(fields) == 2 && fields[0] == "toolchain" {
			directive = fields[1]
			break
		}
	}
	if directive == "" {
		t.Fatal("go.mod has no toolchain directive; security-audit must scan the toolchain the repository ships")
	}

	var major, minor, patch int
	if _, err := fmt.Sscanf(directive, "go%d.%d.%d", &major, &minor, &patch); err != nil {
		t.Fatalf("parse go.mod toolchain %q: %v", directive, err)
	}
	if major < 1 || major == 1 && (minor < 26 || minor == 26 && patch < 6) {
		t.Fatalf("go.mod toolchain %s predates go1.26.6, which fixes the reachable standard-library vulnerabilities reported by security-audit run 31955764842", directive)
	}
}
