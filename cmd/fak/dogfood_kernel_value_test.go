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

func TestCollectDogfoodVelocityMatchesAdjacentCohorts(t *testing.T) {
	dir := t.TempDir()
	rows := map[string]string{
		"resolve-1-20260813-122201.witness": `{"claim":"CLAIM_WITNESSED"}`,
		"resolve-2-20260813-122200.witness": `{"claim":"CLAIM_NO_COMMIT"}`,
		"resolve-3-20260813-122159.witness": `{"claim":"CLAIM_WITNESSED"}`,
		"resolve-4-20260813-122202.witness": `{"claim":"CLAIM_NO_COMMIT"}`,
		"resolve-5-20260813-122203.witness": `{"claim":"CLAIM_NO_COMMIT"}`,
		"resolve-6-20260813-122204.witness": `{"claim":"CLAIM_WITNESSED"}`,
	}
	for name, body := range rows {
		if err := os.WriteFile(filepath.Join(dir, name), []byte(body), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	got := collectDogfoodVelocity(dir, 3)
	if got.Status != "regressed" || got.MatchedSamples != 3 || got.Pre.Shipped != 2 || got.Post.Shipped != 1 || got.ShipRateDelta >= 0 {
		t.Fatalf("got=%+v", got)
	}
}

func TestCollectDogfoodVelocityRefusesThinMatchedEvidence(t *testing.T) {
	dir := t.TempDir()
	for name, body := range map[string]string{
		"resolve-1-20260813-122201.witness": `{"claim":"CLAIM_WITNESSED"}`,
		"resolve-2-20260813-122202.witness": `{"claim":"CLAIM_WITNESSED"}`,
	} {
		if err := os.WriteFile(filepath.Join(dir, name), []byte(body), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	got := collectDogfoodVelocity(dir, 5)
	if got.Status != "not-yet" || got.MatchedSamples != 1 || got.Pre.Shipped != 1 || got.Post.Shipped != 1 {
		t.Fatalf("got=%+v", got)
	}
}
