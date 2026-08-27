package qwen4exp

import (
	_ "embed"
	"encoding/json"
	"strings"
	"testing"
)

//go:embed testdata/qwen38_flash_next_manifest.json
var capturedManifest []byte

func TestCapturedFlashNextManifestIsCompleteAndBound(t *testing.T) {
	m, err := ParseManifest(capturedManifest)
	if err != nil {
		t.Fatal(err)
	}
	if len(m.Artifacts) != 144 || m.TensorInventory.Count != 1658 {
		t.Fatalf("artifacts=%d tensors=%d", len(m.Artifacts), m.TensorInventory.Count)
	}
	b, err := m.Binding("fak-native/qwen4exp")
	if err != nil {
		t.Fatal(err)
	}
	if b.ManifestIdentity == "" || b.ArtifactIdentity == "" || b.TokenizerIdentity == "" || b.TemplateIdentity == "" || b.EngineIdentity != "fak-native/qwen4exp" {
		t.Fatalf("binding=%+v", b)
	}
}

func TestManifestRejectsDigestRevisionAndMissingArtifactBeforeBinding(t *testing.T) {
	var m Manifest
	if err := json.Unmarshal(capturedManifest, &m); err != nil {
		t.Fatal(err)
	}
	tests := []struct {
		name   string
		mutate func(*Manifest)
	}{
		{"mutable-revision", func(m *Manifest) { m.Revision = "main" }},
		{"changed-digest", func(m *Manifest) { m.Artifacts[0].SHA256 = strings.Repeat("z", 64) }},
		{"missing-tokenizer", func(m *Manifest) {
			for i, a := range m.Artifacts {
				if a.Path == "tokenizer.json" {
					m.Artifacts = append(m.Artifacts[:i], m.Artifacts[i+1:]...)
					break
				}
			}
		}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			x := m
			x.Artifacts = append([]ManifestArtifact(nil), m.Artifacts...)
			tt.mutate(&x)
			if _, err := x.Binding("fak-native/qwen4exp"); err == nil {
				t.Fatal("unsafe manifest accepted")
			}
		})
	}
}

func TestManifestIdentityDetectsOneChangedValidDigest(t *testing.T) {
	m, err := ParseManifest(capturedManifest)
	if err != nil {
		t.Fatal(err)
	}
	before, _ := m.Identity()
	m.Artifacts[0].SHA256 = strings.Repeat("a", 64)
	after, err := m.Identity()
	if err != nil {
		t.Fatal(err)
	}
	if before == after {
		t.Fatal("identity did not change")
	}
}
