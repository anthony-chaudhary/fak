package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/modelroute"
)

func writeAgentRouteManifest(t *testing.T, body string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "route.json")
	if err := os.WriteFile(path, []byte(body), 0o600); err != nil {
		t.Fatalf("write route manifest: %v", err)
	}
	return path
}

func TestLoadAgentRouteOptionsEmptyPath(t *testing.T) {
	manifest, opts, err := loadAgentRouteOptions("   ")
	if err != nil {
		t.Fatalf("loadAgentRouteOptions empty path: %v", err)
	}
	if manifest != nil || opts != nil {
		t.Fatalf("empty --route-manifest should install nothing, got manifest=%v opts=%d", manifest, len(opts))
	}
}

func TestLoadAgentRouteOptionsValidManifest(t *testing.T) {
	path := writeAgentRouteManifest(t, `{
  "version": "fak-route/v1",
  "default": {"members": [{"model": "localtools"}]},
  "rules": [
    {
      "name": "book-to-guard",
      "match": {"aspect": "tool_call", "tool": "book_flight"},
      "plan": {"members": [{"model": "guard-engine"}]}
    }
  ]
}`)
	manifest, opts, err := loadAgentRouteOptions(path)
	if err != nil {
		t.Fatalf("loadAgentRouteOptions valid manifest: %v", err)
	}
	if manifest == nil || manifest.Default.Primary() != "localtools" || len(opts) != 1 {
		t.Fatalf("route manifest not installed: manifest=%+v opts=%d", manifest, len(opts))
	}
	if got := manifest.Route(modelroute.Subject{Aspect: modelroute.AspectToolCall, Tool: "book_flight"}).Plan.Primary(); got != "guard-engine" {
		t.Fatalf("book_flight route = %q, want guard-engine", got)
	}
}

func TestLoadAgentRouteOptionsInvalidManifestNamesFlag(t *testing.T) {
	path := writeAgentRouteManifest(t, `{not-json`)
	manifest, opts, err := loadAgentRouteOptions(path)
	if err == nil {
		t.Fatalf("invalid route manifest unexpectedly loaded: manifest=%v opts=%d", manifest, len(opts))
	}
	if !strings.Contains(err.Error(), "fak agent: --route-manifest") {
		t.Fatalf("error should name the flag, got %q", err)
	}
}
