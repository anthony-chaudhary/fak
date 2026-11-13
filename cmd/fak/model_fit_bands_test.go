package main

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/fitplan"
	"github.com/anthony-chaudhary/fak/internal/ggufload"
	"github.com/anthony-chaudhary/fak/internal/model"
)

// fak-test:runtime fast est=1s
func TestFitBandsPreservesPinsAndBoundsWeights(t *testing.T) {
	cfg := model.Config{NumLayers: 4}
	c := ggufload.LayerStorageCensus{Layers: []int64{8, 12, 5, 15}, Embedding: 4, Head: 6}
	in := fitplan.DeploymentInput{Pools: []fitplan.MemoryPool{{ID: "a", UsableBytes: 24}, {ID: "b", UsableBytes: 26}}, Placement: fitplan.Placement{Name: "p", Weights: fitplan.WeightPartition{ShardedBytes: 40, ShardCount: 4}, Ranks: []fitplan.RankPlacement{{Rank: 0, PoolID: "a", Shards: 2, ExtraBytes: 4}, {Rank: 1, PoolID: "b", Shards: 2, ExtraBytes: 6}}}}
	b, err := fitBandsFromCensus(cfg, c, in)
	if err != nil || len(b) != 2 || b[0].Lo != 0 || b[0].Hi != 2 || b[1].Lo != 2 || b[1].Hi != 4 || b[0].PoolID != "a" || b[1].PoolID != "b" {
		t.Fatalf("bands=%+v error=%v", b, err)
	}
	c.Layers[0] = 9
	if _, err := fitBandsFromCensus(cfg, c, in); err == nil {
		t.Fatal("undercharged uneven band accepted")
	}
	c.Layers[0] = 8
	in.Placement.Ranks[0].ExtraBytes--
	if _, err := fitBandsFromCensus(cfg, c, in); err == nil {
		t.Fatal("embedding undercharge accepted")
	}
	in.Placement.Ranks[0].ExtraBytes++
	in.Placement.Ranks[0].Rank = 1
	if _, err := fitBandsFromCensus(cfg, c, in); err == nil {
		t.Fatal("rank pin reordered")
	}
}

// fak-test:runtime fast est=1s
func TestFitBandsAuthenticatesCheckpoint(t *testing.T) {
	t.Setenv("FAK_W3_MLP", "")
	path := writeServePackedEmbeddingFixture(t, "fixture.gguf", false)
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	h := sha256.Sum256(data)
	shards := []ggufload.ShardIdentity{{Path: path, Size: int64(len(data)), RawFileSHA256: hex.EncodeToString(h[:])}}
	manifest, err := json.Marshal(shards)
	if err != nil {
		t.Fatal(err)
	}
	manifestHash := sha256.Sum256(manifest)
	manifestPath := filepath.Join(t.TempDir(), "manifest.json")
	if err := os.WriteFile(manifestPath, manifest, 0600); err != nil {
		t.Fatal(err)
	}
	ws, err := ggufload.OpenWeights(path)
	if err != nil {
		t.Fatal(err)
	}
	c, err := ws.CensusQ4KLayerStorage(fitBandLoaderOptions()...)
	if err != nil {
		t.Fatal(err)
	}
	if err := ws.Close(); err != nil {
		t.Fatal(err)
	}
	in := fitplan.DeploymentInput{Pools: []fitplan.MemoryPool{{ID: "gpu", UsableBytes: 1 << 30}}, Placement: fitplan.Placement{Name: "fixed", Weights: fitplan.WeightPartition{ShardedBytes: c.Layers[0] + c.Layers[1], ShardCount: 2}, Ranks: []fitplan.RankPlacement{{Rank: 0, PoolID: "gpu", Shards: 2, ExtraBytes: c.Embedding + c.Head}}}}
	raw, err := json.Marshal(in)
	if err != nil {
		t.Fatal(err)
	}
	args := []string{"bands", "--input", "-", "--manifest", manifestPath, "--manifest-sha256", hex.EncodeToString(manifestHash[:])}
	var out, stderr bytes.Buffer
	if rc := runModelFit(bytes.NewReader(raw), &out, &stderr, args); rc != 0 {
		t.Fatalf("rc=%d stderr=%s", rc, stderr.String())
	}
	var artifact struct {
		ModeledOnly   bool      `json:"modeled_only"`
		Qualification string    `json:"qualification"`
		Bands         []fitBand `json:"bands"`
	}
	if err := json.Unmarshal(out.Bytes(), &artifact); err != nil || !artifact.ModeledOnly || artifact.Qualification != "unverified" || len(artifact.Bands) != 1 {
		t.Fatalf("output=%s err=%v", out.String(), err)
	}
	args[len(args)-1] = hex.EncodeToString(make([]byte, 32))
	out.Reset()
	stderr.Reset()
	if rc := runModelFit(bytes.NewReader(raw), &out, &stderr, args); rc == 0 || out.Len() != 0 {
		t.Fatalf("mismatched identity accepted rc=%d out=%s", rc, out.String())
	}
}
