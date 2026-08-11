package quantmatrix

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestPublicDocumentMatchesRegistry(t *testing.T) {
	docPath := filepath.Join("..", "..", "docs", "quantization-support.md")
	data, err := os.ReadFile(docPath)
	if err != nil {
		t.Fatalf("read public matrix: %v", err)
	}
	rows, err := ParseDocument(string(data))
	if err != nil {
		t.Fatalf("parse public matrix: %v", err)
	}
	if len(rows) < 4 {
		t.Fatalf("public matrix has %d rows; want at least supported, delegated, research-only, and unsupported", len(rows))
	}
	seenClass := map[Class]bool{}
	seenID := map[EntryID]bool{}
	for _, row := range rows {
		capability, ok := Lookup(row.ID)
		if !ok {
			t.Errorf("documented capability %q is not registered", row.ID)
			continue
		}
		if seenID[row.ID] {
			t.Errorf("duplicate documented capability %q", row.ID)
		}
		seenID[row.ID] = true
		seenClass[capability.Class] = true
		if row != capability {
			t.Errorf("documented row %q differs from registry\n document: %#v\n registry: %#v", row.ID, row, capability)
		}
		if capability.Artifact == "" || capability.Recipe == "" || capability.Runtime == "" || capability.Hardware == "" {
			t.Errorf("capability %q does not separate artifact, recipe, runtime, and hardware claims", row.ID)
		}
		if !strings.HasPrefix(capability.Evidence, "../") {
			t.Errorf("capability %q evidence must be a repository-relative Markdown link, got %q", row.ID, capability.Evidence)
			continue
		}
		target := filepath.Clean(filepath.Join("..", "..", "docs", filepath.FromSlash(capability.Evidence)))
		if _, err := os.Stat(target); err != nil {
			t.Errorf("capability %q evidence link %q: %v", row.ID, capability.Evidence, err)
		}
	}
	for _, status := range []Class{ClassNative, ClassExternal, ClassExperimental, ClassUnavailable} {
		if !seenClass[status] {
			t.Errorf("public matrix has no %s row", status)
		}
	}
	for _, capability := range Entries() {
		if !seenID[capability.ID] {
			t.Errorf("registered capability %q is missing from public matrix", capability.ID)
		}
	}
}

func TestAdjudicateTypedUnknownAndUnavailable(t *testing.T) {
	tests := []struct {
		name string
		req  Request
		want Decision
	}{
		{
			name: "known native combination",
			req:  Request{ID: EntryGGUFQ4KCPU, ArtifactVersion: "gguf-v3", Runtime: "fak-native-cpu"},
			want: Decision{Outcome: OutcomeAllow, Reason: CodeNative},
		},
		{
			name: "known delegated combination",
			req:  Request{ID: EntryGPTQExternal, ArtifactVersion: "gptq-model", Runtime: "external-runtime"},
			want: Decision{Outcome: OutcomeDelegate, Reason: CodeExternal},
		},
		{
			name: "unknown capability",
			req:  Request{ID: "quant.future.v9", ArtifactVersion: "v9", Runtime: "future"},
			want: Decision{Outcome: OutcomeAbstain, Reason: CodeUnknownID},
		},
		{
			name: "unknown artifact version",
			req:  Request{ID: EntryGGUFQ4KCPU, ArtifactVersion: "gguf-v99", Runtime: "fak-native-cpu"},
			want: Decision{Outcome: OutcomeAbstain, Reason: CodeUnknownVersion},
		},
		{
			name: "unsupported combination",
			req:  Request{ID: EntryBitsAndBytesInProcess, ArtifactVersion: "bnb-4bit", Runtime: "fak-native"},
			want: Decision{Outcome: OutcomeRefuse, Reason: CodeUnavailable},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := Adjudicate(tt.req); got != tt.want {
				t.Fatalf("Adjudicate(%+v) = %+v, want %+v", tt.req, got, tt.want)
			}
		})
	}
}
