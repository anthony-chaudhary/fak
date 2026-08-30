package qwen4exp

import (
	_ "embed"
	"encoding/json"
	"fmt"
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
	shards := make(map[int]struct{})
	shardTotal := 0
	for _, artifact := range m.Artifacts {
		if !strings.HasPrefix(artifact.Path, "model-") || !strings.HasSuffix(artifact.Path, ".safetensors") {
			continue
		}
		var shard, total int
		if _, err := fmt.Sscanf(artifact.Path, "model-%05d-of-%05d.safetensors", &shard, &total); err != nil {
			t.Fatalf("malformed checkpoint shard %q: %v", artifact.Path, err)
		}
		if canonical := fmt.Sprintf("model-%05d-of-%05d.safetensors", shard, total); artifact.Path != canonical {
			t.Fatalf("checkpoint shard %q is not canonical %q", artifact.Path, canonical)
		}
		if shardTotal == 0 {
			shardTotal = total
		}
		if total != shardTotal {
			t.Fatalf("checkpoint shard %q declares total %d, want %d", artifact.Path, total, shardTotal)
		}
		shards[shard] = struct{}{}
	}
	if shardTotal == 0 || len(shards) != shardTotal {
		t.Fatalf("checkpoint shards=%d declared=%d", len(shards), shardTotal)
	}
	for shard := 1; shard <= shardTotal; shard++ {
		if _, ok := shards[shard]; !ok {
			t.Fatalf("checkpoint shard %d of %d is missing", shard, shardTotal)
		}
	}
	if m.TensorInventory.Count != 1658 {
		t.Fatalf("tensors=%d", m.TensorInventory.Count)
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
