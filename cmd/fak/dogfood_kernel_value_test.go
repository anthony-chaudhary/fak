package main

import (
	"os"
	"path/filepath"
	"testing"
)

func TestCollectDogfoodKernelValuePreservesNotYetAxes(t *testing.T) {
	dir := t.TempDir()
	got := collectDogfoodKernelValue(dir, "", 5)
	if got.Cohort.Status != "not-yet" || got.Cache.Status != "not-yet" || got.RepoPulseLaunches != 0 {
		t.Fatalf("got=%+v", got)
	}
}

func TestCollectDogfoodKernelValueFoldsTypedCacheWitness(t *testing.T) {
	dir := t.TempDir()
	receipt := filepath.Join(dir, "cache.json")
	raw := `{"schema":"fak-micro-cache-affinity-witness/1","verdict":"not-yet","reason":"need two seats","affinity_on":{"cached_prompt_tokens":120},"affinity_off":{"cached_prompt_tokens":80}}`
	if err := os.WriteFile(receipt, []byte(raw), 0o600); err != nil {
		t.Fatal(err)
	}
	got := collectDogfoodKernelValue(dir, receipt, 5)
	if got.Cache.Status != "not-yet" || got.CacheCachedPromptTokens != 200 || got.Cache.Reason != "need two seats" {
		t.Fatalf("got=%+v", got)
	}
}
