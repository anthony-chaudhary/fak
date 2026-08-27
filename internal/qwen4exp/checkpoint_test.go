package qwen4exp

import (
	"crypto/sha256"
	"encoding/hex"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestVerifyArtifactsBindsEveryByte(t *testing.T) {
	dir := t.TempDir()
	required := []string{"config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "LICENSE", "model.safetensors.index.json", "model-00001-of-00001.safetensors"}
	arts := []ManifestArtifact{}
	for _, name := range required {
		raw := []byte("exact-" + name)
		sum := sha256.Sum256(raw)
		os.WriteFile(filepath.Join(dir, name), raw, 0600)
		arts = append(arts, ManifestArtifact{Path: name, Size: int64(len(raw)), SHA256: hex.EncodeToString(sum[:])})
	}
	inv := sha256.Sum256([]byte("inventory"))
	rev := strings.Repeat("a", 40)
	m := Manifest{Schema: ManifestSchema, Repository: "m", Revision: rev, SourceRepository: "s", SourceRevision: rev, Architecture: "Qwen3_5NextForCausalLM", DType: "bfloat16", Artifacts: arts, TensorInventory: TensorInventory{Count: 1658, SHA256: hex.EncodeToString(inv[:])}}
	r, err := VerifyArtifacts(dir, m)
	if err != nil {
		t.Fatal(err)
	}
	if err := r.ValidateForExecution(); err != nil {
		t.Fatal(err)
	}
	os.WriteFile(filepath.Join(dir, required[0]), []byte("wrong"), 0600)
	if _, err := VerifyArtifacts(dir, m); err == nil {
		t.Fatal("mutated artifact accepted")
	}
}
