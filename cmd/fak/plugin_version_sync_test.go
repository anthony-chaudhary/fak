package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

// TestPluginManifestVersionMatchesVERSION guards against the plugin manifest silently
// drifting behind the release. The release cut only ever writes the repo-level VERSION file;
// plugins/fak/.claude-plugin/plugin.json carries an INDEPENDENT "version" that no automation
// bumps, so it froze at 0.34.0 while the binary shipped 0.37.0 — anyone reading the plugin's
// advertised version got a stale answer. This test fails the moment the two diverge, so a
// future release must move both (or consciously exempt the manifest) instead of leaving the
// plugin misreporting.
func TestPluginManifestVersionMatchesVERSION(t *testing.T) {
	root := repoRootForTest(t)

	versionBytes, err := os.ReadFile(filepath.Join(root, "VERSION"))
	if err != nil {
		t.Fatalf("read VERSION: %v", err)
	}
	want := strings.TrimSpace(string(versionBytes))
	if want == "" {
		t.Fatal("VERSION file is empty")
	}

	manifestBytes, err := os.ReadFile(filepath.Join(root, "plugins", "fak", ".claude-plugin", "plugin.json"))
	if err != nil {
		t.Fatalf("read plugin.json: %v", err)
	}
	var manifest struct {
		Version string `json:"version"`
	}
	if err := json.Unmarshal(manifestBytes, &manifest); err != nil {
		t.Fatalf("parse plugin.json: %v", err)
	}

	if got := strings.TrimSpace(manifest.Version); got != want {
		t.Fatalf("plugin.json version = %q, want %q (the VERSION file); bump the manifest in the same release cut so the plugin does not advertise a stale version", got, want)
	}
}

// repoRootForTest walks up from this test file's own location to the repo root (the dir
// holding go.mod), so the test does not depend on the working directory the runner chose.
func repoRootForTest(t *testing.T) string {
	t.Helper()
	_, file, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("runtime.Caller failed")
	}
	dir := filepath.Dir(file)
	for {
		if _, err := os.Stat(filepath.Join(dir, "go.mod")); err == nil {
			return dir
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			t.Fatal("could not find repo root (go.mod) above test file")
		}
		dir = parent
	}
}
