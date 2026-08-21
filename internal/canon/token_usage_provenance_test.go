package canon

import (
	"encoding/json"
	"os"
	"strings"
	"testing"
)

type providerBaselineProvenance struct {
	Schema      string `json:"schema"`
	RetrievedAt string `json:"retrieved_at"`
	Sources     []struct {
		Provider   string   `json:"provider"`
		Repository string   `json:"repository"`
		Revision   string   `json:"revision"`
		Version    string   `json:"version,omitempty"`
		SourcePath string   `json:"source_path"`
		Symbols    []string `json:"symbols"`
		License    string   `json:"license"`
		Reuse      string   `json:"reuse"`
	} `json:"sources"`
}

func TestTokenUsageProviderBaselinesArePinned(t *testing.T) {
	raw, err := os.ReadFile("testdata/token_usage/upstream_provenance.json")
	if err != nil {
		t.Fatal(err)
	}
	var manifest providerBaselineProvenance
	if err := json.Unmarshal(raw, &manifest); err != nil {
		t.Fatal(err)
	}
	if manifest.Schema != "fak/provider-baseline-provenance/v1" || manifest.RetrievedAt == "" {
		t.Fatalf("incomplete provider baseline manifest: %+v", manifest)
	}
	if len(manifest.Sources) < 2 {
		t.Fatalf("sources=%d want at least OpenAI and Anthropic", len(manifest.Sources))
	}
	seen := make(map[string]bool)
	for _, source := range manifest.Sources {
		if source.Provider == "" || !strings.HasPrefix(source.Repository, "https://") || len(source.Revision) != 40 || source.SourcePath == "" || len(source.Symbols) == 0 || source.License == "" || source.Reuse == "" {
			t.Errorf("incomplete source pin: %+v", source)
		}
		seen[source.Provider] = true
	}
	for _, provider := range []string{"openai", "anthropic"} {
		if !seen[provider] {
			t.Errorf("missing %s baseline", provider)
		}
	}
}
