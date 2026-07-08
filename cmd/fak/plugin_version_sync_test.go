package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

// TestPluginManifestVersionMatchesVERSION guards against EVERY plugin-manifest version field
// silently drifting behind the release. The release cut only ever writes the repo-level VERSION
// file; the Claude-plugin manifests carry INDEPENDENT "version" fields that no automation bumps,
// so they froze at 0.34.0 while the binary shipped 0.37.0 — anyone reading the plugin's or the
// marketplace's advertised version got a stale answer, and the marketplace one is what Claude
// Code actually reads to install. This test fails the moment ANY of them diverges from VERSION,
// so a future release must move them all (or consciously exempt one) instead of misreporting.
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

	// plugins/fak/.claude-plugin/plugin.json — the plugin's own manifest.
	var plugin struct {
		Version string `json:"version"`
	}
	readJSON(t, filepath.Join(root, "plugins", "fak", ".claude-plugin", "plugin.json"), &plugin)
	assertVersion(t, "plugin.json version", plugin.Version, want)

	// .claude-plugin/marketplace.json — what Claude Code reads to install; has TWO version
	// fields (the marketplace metadata and the embedded plugin entry), both of which drift.
	var market struct {
		Metadata struct {
			Version string `json:"version"`
		} `json:"metadata"`
		Plugins []struct {
			Name    string `json:"name"`
			Version string `json:"version"`
		} `json:"plugins"`
	}
	readJSON(t, filepath.Join(root, ".claude-plugin", "marketplace.json"), &market)
	assertVersion(t, "marketplace.json metadata.version", market.Metadata.Version, want)
	for _, p := range market.Plugins {
		assertVersion(t, "marketplace.json plugins["+p.Name+"].version", p.Version, want)
	}
}

func readJSON(t *testing.T, path string, v any) {
	t.Helper()
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read %s: %v", filepath.Base(path), err)
	}
	if err := json.Unmarshal(b, v); err != nil {
		t.Fatalf("parse %s: %v", filepath.Base(path), err)
	}
}

func assertVersion(t *testing.T, label, got, want string) {
	t.Helper()
	if g := strings.TrimSpace(got); g != want {
		t.Fatalf("%s = %q, want %q (the VERSION file); bump it in the same release cut so the plugin does not advertise a stale version", label, g, want)
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
