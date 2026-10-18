package ggufload

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

// fak-test:runtime fast est=100ms lane=default
func TestAuthenticatedArtifactReachesQuantModel(t *testing.T) {
	// Both revisions have identical tensor names, geometry and file size.
	// Only one packed weight byte changes. These are synthetic artifacts, not
	// evidence that the published 1017 checkpoint has been loaded.
	fixture := func(weight byte) []byte {
		var b bytes.Buffer
		writeMinimalHeader(&b, 1, 9)
		writeKVString(&b, "general.architecture", "llama")
		writeKVUint32(&b, "general.alignment", 32)
		writeKVUint32(&b, "llama.embedding_length", 256)
		writeKVUint32(&b, "llama.block_count", 1)
		writeKVUint32(&b, "llama.attention.head_count", 1)
		writeKVUint32(&b, "llama.attention.key_length", 256)
		writeKVUint32(&b, "llama.feed_forward_length", 256)
		writeKVFloat32(&b, "llama.attention.layer_norm_rms_epsilon", 1e-6)
		writeKVString(&b, "tokenizer.chat_template", "exact template\n")
		writeTensorInfoForTest(&b, "blk.0.attn_v.weight", []uint64{256, 1}, TensorQ4_K, 0)
		padToAlignment(&b, 32)
		payload := make([]byte, 144)
		payload[143] = weight
		b.Write(payload)
		return b.Bytes()
	}
	digest := func(b []byte) string { s := sha256.Sum256(b); return hex.EncodeToString(s[:]) }
	path := filepath.Join(t.TempDir(), "tiny.gguf")
	data := fixture(1)
	write := func(data []byte) {
		t.Helper()
		if err := os.WriteFile(path, data, 0600); err != nil {
			t.Fatal(err)
		}
	}
	write(data)
	shards := []ShardIdentity{{Path: path, Size: int64(len(data)), RawFileSHA256: digest(data)}}
	manifest := func() string {
		b, err := json.Marshal(shards)
		if err != nil {
			t.Fatal(err)
		}
		return digest(b)
	}
	oldManifest := manifest()
	ws, err := OpenWeightsAuthenticated(t.Context(), shards, oldManifest)
	if err != nil {
		t.Fatal(err)
	}
	defer ws.Close()
	m, err := ws.QuantModelQ4KProfileOptionsContext(t.Context(), nil)
	if err != nil {
		t.Fatal(err)
	}
	old, ok := m.AuthenticatedArtifactIdentity()
	if !ok || old.ManifestSHA256 != oldManifest || old.ChatTemplateSHA256 != digest([]byte("exact template\n")) {
		t.Fatalf("missing loader receipt: %+v %t", old, ok)
	}
	portable, _ := json.Marshal([]authenticatedShardContent{{shards[0].Size, shards[0].RawFileSHA256}})
	if old.ShardSetSHA256 != digest(portable) {
		t.Fatalf("portable digest = %s", old.ShardSetSHA256)
	}
	// Returned receipt is a value, never mutable model state.
	old.ShardSetSHA256 = "changed copy"
	old, _ = m.AuthenticatedArtifactIdentity()
	if old.ShardSetSHA256 == "changed copy" {
		t.Fatal("getter exposed mutable identity")
	}
	if err := ws.Close(); err != nil {
		t.Fatal(err)
	}
	// Close retained readers before changing the file, honoring immutability.
	data = fixture(2)
	write(data)
	if stale, err := OpenWeightsAuthenticated(t.Context(), shards, oldManifest); err == nil {
		stale.Close()
		t.Fatal("old manifest admitted same-size changed weights")
	}
	shards[0].RawFileSHA256 = digest(data)
	ws2, err := OpenWeightsAuthenticated(t.Context(), shards, manifest())
	if err != nil {
		t.Fatal(err)
	}
	defer ws2.Close()
	m2, err := ws2.QuantModelQ4KProfileOptionsContext(t.Context(), nil)
	if err != nil {
		t.Fatal(err)
	}
	newer, ok := m2.AuthenticatedArtifactIdentity()
	if !ok || newer.ShardSetSHA256 == old.ShardSetSHA256 {
		t.Fatal("changed bytes retained old model identity")
	}
	plain, err := OpenWeights(path)
	if err != nil {
		t.Fatal(err)
	}
	defer plain.Close()
	unverified, err := plain.QuantModelQ4KProfileOptionsContext(t.Context(), nil)
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := unverified.AuthenticatedArtifactIdentity(); ok {
		t.Fatal("ordinary loader fabricated authentication")
	}
}
