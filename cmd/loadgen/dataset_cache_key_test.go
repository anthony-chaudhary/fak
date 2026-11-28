package main

import "testing"

// fak-test:runtime fast est=5ms lane=default
func TestDatasetCacheKey(t *testing.T) {
	base := DatasetCacheKeyInput{
		InputBytes:  []byte(`{"prompt":"hello"}`),
		TokenizerID: "Qwen/Qwen3.8-8B",
		Settings: map[string]any{
			"isl": 512,
			"osl": 128,
			"nested": map[string]any{
				"b": 2,
				"a": 1,
			},
		},
		ManifestVersion: DatasetCacheManifestVersion,
		CacheBust:       "run-1",
	}

	k1, err := ComputeDatasetCacheKey(base)
	if err != nil {
		t.Fatalf("ComputeDatasetCacheKey: %v", err)
	}
	if len(k1) != 64 {
		t.Fatalf("key len=%d, want 64 hex chars", len(k1))
	}

	// CacheBust must be excluded from the content-addressed key.
	withBust := base
	withBust.CacheBust = "run-999"
	kBust, err := ComputeDatasetCacheKey(withBust)
	if err != nil {
		t.Fatal(err)
	}
	if kBust != k1 {
		t.Fatalf("CacheBust changed cache key: %s != %s", kBust, k1)
	}

	// Changing input bytes, tokenizer ID, any setting, or manifest version changes the key.
	diffInput := base
	diffInput.InputBytes = []byte(`{"prompt":"world"}`)
	if k, _ := ComputeDatasetCacheKey(diffInput); k == k1 {
		t.Fatal("input bytes change did not alter key")
	}

	diffTok := base
	diffTok.TokenizerID = "meta-llama/Llama-3"
	if k, _ := ComputeDatasetCacheKey(diffTok); k == k1 {
		t.Fatal("tokenizer change did not alter key")
	}

	diffSetting := base
	diffSetting.Settings = map[string]any{"isl": 512, "osl": 256, "nested": map[string]any{"a": 1, "b": 2}}
	if k, _ := ComputeDatasetCacheKey(diffSetting); k == k1 {
		t.Fatal("settings change did not alter key")
	}

	diffVer := base
	diffVer.ManifestVersion = DatasetCacheManifestVersion + 1
	if k, _ := ComputeDatasetCacheKey(diffVer); k == k1 {
		t.Fatal("manifest version bump did not alter key")
	}

	if !DatasetManifestVersionCompatible(DatasetCacheManifestVersion) {
		t.Fatal("current manifest version rejected")
	}
	if DatasetManifestVersionCompatible(DatasetCacheManifestVersion + 1) {
		t.Fatal("bumped manifest version accepted")
	}
}
