package canon

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/borrowprovenance"
)

type usageProvenanceManifest struct {
	Schema      string                  `json:"schema"`
	RetrievedAt string                  `json:"retrieved_at"`
	Sources     []usageProvenanceSource `json:"sources"`
}

type usageProvenanceSource struct {
	Provider        string   `json:"provider"`
	Repository      string   `json:"repository"`
	Revision        string   `json:"revision"`
	Version         string   `json:"version,omitempty"`
	SourcePath      string   `json:"source_path"`
	SourceStartLine int      `json:"source_start_line"`
	SourceEndLine   int      `json:"source_end_line"`
	Symbols         []string `json:"symbols"`
	License         string   `json:"license"`
	Reuse           string   `json:"reuse"`
	ExcerptPath     string   `json:"excerpt_path"`
	ExcerptSHA256   string   `json:"excerpt_sha256"`
	LicensePath     string   `json:"license_path"`
}

func TestTokenUsageUpstreamProvenanceIsPinned(t *testing.T) {
	manifest := readUsageProvenanceManifest(t)
	if manifest.Schema != "fak/provider-baseline-provenance/v1" || manifest.RetrievedAt == "" {
		t.Fatalf("incomplete manifest header: %+v", manifest)
	}
	wantFields := map[string][]string{
		"openai":    {`json:"input_tokens"`, `json:"output_tokens"`, `json:"cache_write_tokens"`, `json:"cached_tokens"`, `json:"reasoning_tokens"`},
		"anthropic": {`json:"input_tokens"`, `json:"output_tokens"`, `json:"cache_creation_input_tokens"`, `json:"cache_read_input_tokens"`},
		"gemini":    {`json:"promptTokenCount,omitempty"`, `json:"cachedContentTokenCount,omitempty"`, `json:"toolUsePromptTokenCount,omitempty"`, `json:"candidatesTokenCount,omitempty"`, `json:"thoughtsTokenCount,omitempty"`, `json:"totalTokenCount,omitempty"`},
	}
	seen := make(map[string]bool, len(wantFields))
	for _, source := range manifest.Sources {
		if _, duplicate := seen[source.Provider]; duplicate {
			t.Fatalf("duplicate provider provenance: %s", source.Provider)
		}
		seen[source.Provider] = true
		if source.Repository == "" || source.Revision == "" || source.SourcePath == "" || source.SourceStartLine < 1 || source.SourceEndLine < source.SourceStartLine || len(source.Symbols) == 0 || source.License == "" || source.Reuse == "" {
			t.Fatalf("incomplete source provenance: %+v", source)
		}
		excerpt := readUsageProvenanceFile(t, source.ExcerptPath)
		record := borrowprovenance.Record{Schema: borrowprovenance.Schema, SourceURL: source.Repository, SourceRef: source.Revision, SourcePath: source.SourcePath, SourceSHA256: source.ExcerptSHA256, License: source.License, Transformation: source.Reuse}
		verification, err := borrowprovenance.Verify(record, excerpt)
		if err != nil {
			t.Fatalf("verify %s excerpt: %v", source.Provider, err)
		}
		if !verification.Match {
			t.Fatalf("%s excerpt drifted: %+v", source.Provider, verification)
		}
		license := readUsageProvenanceFile(t, source.LicensePath)
		if len(bytes.TrimSpace(license)) == 0 {
			t.Fatalf("%s license notice is empty", source.Provider)
		}
		for _, symbol := range source.Symbols {
			if !bytes.Contains(excerpt, []byte("type "+symbol+" struct")) {
				t.Errorf("%s excerpt missing symbol %s", source.Provider, symbol)
			}
		}
		fields, ok := wantFields[source.Provider]
		if !ok {
			t.Fatalf("unexpected provider provenance %q", source.Provider)
		}
		for _, field := range fields {
			if !bytes.Contains(excerpt, []byte(field)) {
				t.Errorf("%s copied SDK excerpt missing adapter field %s", source.Provider, field)
			}
		}
	}
	for provider := range wantFields {
		if !seen[provider] {
			t.Errorf("missing %s provenance", provider)
		}
	}
}

func TestTokenUsageUpstreamExcerptDriftIsDetected(t *testing.T) {
	manifest := readUsageProvenanceManifest(t)
	if len(manifest.Sources) == 0 {
		t.Fatal("manifest has no sources")
	}
	source := manifest.Sources[0]
	excerpt := append(readUsageProvenanceFile(t, source.ExcerptPath), []byte("// upstream changed\n")...)
	record := borrowprovenance.Record{Schema: borrowprovenance.Schema, SourceURL: source.Repository, SourceRef: source.Revision, SourcePath: source.SourcePath, SourceSHA256: source.ExcerptSHA256}
	verification, err := borrowprovenance.Verify(record, excerpt)
	if err != nil {
		t.Fatal(err)
	}
	if verification.Match || verification.ExpectedSHA256 == verification.ActualSHA256 {
		t.Fatalf("changed copied source was accepted: %+v", verification)
	}
}

func readUsageProvenanceManifest(t *testing.T) usageProvenanceManifest {
	t.Helper()
	raw, err := os.ReadFile("testdata/token_usage/upstream_provenance.json")
	if err != nil {
		t.Fatal(err)
	}
	var manifest usageProvenanceManifest
	if err := json.Unmarshal(raw, &manifest); err != nil {
		t.Fatal(err)
	}
	return manifest
}

func readUsageProvenanceFile(t *testing.T, relative string) []byte {
	t.Helper()
	if relative == "" || filepath.IsAbs(relative) || strings.Contains(filepath.ToSlash(relative), "../") {
		t.Fatalf("unsafe provenance path %q", relative)
	}
	raw, err := os.ReadFile(filepath.Join("testdata", "token_usage", filepath.FromSlash(relative)))
	if err != nil {
		t.Fatal(err)
	}
	return raw
}
