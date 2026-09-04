//go:build darwin

package main

import (
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/sessionaudit"
)

// TestDarwinVarPrivateVarSymlinkNormalization asserts that macOS /var vs /private/var
// symlink differences normalize cleanly across sessionaudit namespace derivation,
// validate --mine path normalization, and validate overlay containment (#5364).
func TestDarwinVarPrivateVarSymlinkNormalization(t *testing.T) {
	if runtime.GOOS != "darwin" {
		t.Skip("Darwin-specific symlink normalization test")
	}

	resolvedVar, err := filepath.EvalSymlinks("/var")
	if err != nil || resolvedVar != "/private/var" {
		t.Skipf("/var does not resolve to /private/var on this host: %v (%s)", err, resolvedVar)
	}

	tempDir := t.TempDir()
	var varPath, privateVarPath string
	if strings.HasPrefix(tempDir, "/var/") {
		varPath = tempDir
		privateVarPath = "/private" + tempDir
	} else if strings.HasPrefix(tempDir, "/private/var/") {
		privateVarPath = tempDir
		varPath = strings.TrimPrefix(tempDir, "/private")
	} else {
		t.Skipf("t.TempDir() %q is not under /var or /private/var", tempDir)
	}

	// 1. Assert sessionaudit.ProjectNamespace produces identical namespaces
	nsVar := sessionaudit.ProjectNamespace(varPath)
	nsPrivateVar := sessionaudit.ProjectNamespace(privateVarPath)
	if nsVar != nsPrivateVar {
		t.Fatalf("ProjectNamespace mismatch:\n  /var form:         %q\n  /private/var form: %q", nsVar, nsPrivateVar)
	}

	// Create a package fixture under tempDir
	subDir := filepath.Join(tempDir, "pkg")
	if err := os.MkdirAll(subDir, 0o755); err != nil {
		t.Fatal(err)
	}
	filePath := filepath.Join(subDir, "p.go")
	if err := os.WriteFile(filePath, []byte("package pkg\n"), 0o644); err != nil {
		t.Fatal(err)
	}

	// 2. normalizeMinePaths: root under /var, mine path under /private/var
	relVarToPriv, err := normalizeMinePaths(varPath, []string{filepath.Join(privateVarPath, "pkg", "p.go")})
	if err != nil {
		t.Fatalf("normalizeMinePaths(root=/var, mine=/private/var): %v", err)
	}
	if len(relVarToPriv) != 1 || relVarToPriv[0] != "pkg/p.go" {
		t.Fatalf("normalizeMinePaths(root=/var, mine=/private/var) = %v, want [pkg/p.go]", relVarToPriv)
	}

	// 3. normalizeMinePaths: root under /private/var, mine path under /var
	relPrivToVar, err := normalizeMinePaths(privateVarPath, []string{filepath.Join(varPath, "pkg", "p.go")})
	if err != nil {
		t.Fatalf("normalizeMinePaths(root=/private/var, mine=/var): %v", err)
	}
	if len(relPrivToVar) != 1 || relPrivToVar[0] != "pkg/p.go" {
		t.Fatalf("normalizeMinePaths(root=/private/var, mine=/var) = %v, want [pkg/p.go]", relPrivToVar)
	}

	// 4. normalizeMinePaths: directory under /private/var with root under /var
	relDir, err := normalizeMinePaths(varPath, []string{filepath.Join(privateVarPath, "pkg")})
	if err != nil {
		t.Fatalf("normalizeMinePaths(root=/var, mine-dir=/private/var): %v", err)
	}
	if len(relDir) != 1 || relDir[0] != "pkg/p.go" {
		t.Fatalf("normalizeMinePaths(root=/var, mine-dir=/private/var) = %v, want [pkg/p.go]", relDir)
	}

	// 5. overlayMinePaths: srcRoot under /var, dstRoot under /private/var
	dst1 := filepath.Join(t.TempDir(), "dst1")
	if err := overlayMinePaths(varPath, dst1, []string{"pkg/p.go"}); err != nil {
		t.Fatalf("overlayMinePaths(srcRoot=/var): %v", err)
	}
	if _, err := os.Stat(filepath.Join(dst1, "pkg", "p.go")); err != nil {
		t.Fatalf("file not overlaid properly into dst1: %v", err)
	}

	// 6. overlayMinePaths: srcRoot under /private/var, dstRoot under /var
	dst2 := filepath.Join(t.TempDir(), "dst2")
	if err := overlayMinePaths(privateVarPath, dst2, []string{"pkg/p.go"}); err != nil {
		t.Fatalf("overlayMinePaths(srcRoot=/private/var): %v", err)
	}
	if _, err := os.Stat(filepath.Join(dst2, "pkg", "p.go")); err != nil {
		t.Fatalf("file not overlaid properly into dst2: %v", err)
	}
}
