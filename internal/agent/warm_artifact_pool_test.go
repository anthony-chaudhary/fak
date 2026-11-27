package agent

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"testing"
)

func TestWarmArtifactPoolTwoNodeStartupReusesWithoutRecompute(t *testing.T) {
	pool := NewWarmArtifactPool()
	keys := []StartupArtifactKey{
		{ModelID: "qwen3.8-32b", RuntimeGeneration: "gen-1", Kind: ArtifactWeights, Variant: "nvfp4"},
		{ModelID: "qwen3.8-32b", RuntimeGeneration: "gen-1", Kind: ArtifactMetadata},
		{ModelID: "qwen3.8-32b", RuntimeGeneration: "gen-1", Kind: ArtifactCompileCache, Variant: "sm90"},
		{ModelID: "qwen3.8-32b", RuntimeGeneration: "gen-1", Kind: ArtifactCompilerKey, Variant: "sm90"},
		{ModelID: "qwen3.8-32b", RuntimeGeneration: "gen-1", Kind: ArtifactCUDAGraphMemEst, Variant: "bs1"},
	}

	node1Setups := 0
	res1, err := pool.StartupOrReuse(context.Background(), "node-1", keys, func(_ context.Context, k StartupArtifactKey) ([]byte, error) {
		node1Setups++
		return []byte(fmt.Sprintf("built:%s:%s", k.Kind, k.Variant)), nil
	})
	if err != nil {
		t.Fatalf("node-1 startup failed: %v", err)
	}
	if res1.Misses != len(keys) || res1.Hits != 0 || res1.LocalSetups != len(keys) || node1Setups != len(keys) {
		t.Fatalf("node-1 result = %+v, setups=%d; want %d misses/setups", res1, node1Setups, len(keys))
	}

	// Node 2 starts the same model: all entries hit in place, zero local setups.
	node2Setups := 0
	res2, err := pool.StartupOrReuse(context.Background(), "node-2", keys, func(_ context.Context, k StartupArtifactKey) ([]byte, error) {
		node2Setups++
		return []byte("should-not-run"), nil
	})
	if err != nil {
		t.Fatalf("node-2 startup failed: %v", err)
	}
	if res2.Hits != len(keys) || res2.Misses != 0 || res2.LocalSetups != 0 || node2Setups != 0 {
		t.Fatalf("node-2 result = %+v, setups=%d; want %d hits and 0 setups", res2, node2Setups, len(keys))
	}
	for i, e := range res2.Entries {
		if e.PublishedBy != "node-1" {
			t.Fatalf("entry %d PublishedBy = %q, want node-1", i, e.PublishedBy)
		}
		if !bytes.Equal(e.PayloadCopy(), res1.Entries[i].PayloadCopy()) {
			t.Fatalf("entry %d payload mismatch", i)
		}
	}

	// Immutability: duplicate publish of an existing key is refused and original bytes remain intact.
	if err := pool.Publish(keys[0], []byte("tampered"), "node-2"); !errors.Is(err, ErrArtifactImmutable) {
		t.Fatalf("duplicate publish err = %v, want ErrArtifactImmutable", err)
	}
	got, ok := pool.ReadInPlace(keys[0])
	if !ok || !bytes.Equal(got.PayloadCopy(), []byte("built:weights:nvfp4")) {
		t.Fatalf("entry mutated after refused publish: ok=%v payload=%q", ok, got.PayloadCopy())
	}

	// Pool miss on a new variant falls back to local setup.
	missKey := StartupArtifactKey{ModelID: "qwen3.8-32b", RuntimeGeneration: "gen-1", Kind: ArtifactCompileCache, Variant: "sm100"}
	res3, err := pool.StartupOrReuse(context.Background(), "node-2", []StartupArtifactKey{keys[0], missKey}, func(_ context.Context, k StartupArtifactKey) ([]byte, error) {
		node2Setups++
		return []byte("fallback:" + k.Variant), nil
	})
	if err != nil {
		t.Fatalf("mixed hit/miss startup failed: %v", err)
	}
	if res3.Hits != 1 || res3.Misses != 1 || res3.LocalSetups != 1 || node2Setups != 1 {
		t.Fatalf("mixed hit/miss result = %+v, setups=%d; want 1 hit, 1 miss, 1 setup", res3, node2Setups)
	}
}
