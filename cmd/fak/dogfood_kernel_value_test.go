package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
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
	raw := `{"schema":"fak-micro-cache-affinity-witness/1","captured_at":"` + time.Now().UTC().Format(time.RFC3339Nano) + `","verdict":"not-yet","reason":"need two seats","affinity_on":{"cached_prompt_tokens":120},"affinity_off":{"cached_prompt_tokens":80}}`
	if err := os.WriteFile(receipt, []byte(raw), 0o600); err != nil {
		t.Fatal(err)
	}
	got := collectDogfoodKernelValue(dir, receipt, 5)
	if got.Cache.Status != "not-yet" || got.CacheCachedPromptTokens != 200 || got.Cache.Reason != "need two seats" {
		t.Fatalf("got=%+v", got)
	}
}

func TestCollectDogfoodKernelValueRefusesWrongSchemaAndStaleWitness(t *testing.T) {
	dir := t.TempDir()
	receipt := filepath.Join(dir, "cache.json")
	now := time.Date(2026, 8, 14, 12, 0, 0, 0, time.UTC)
	oldNow := dogfoodNow
	dogfoodNow = func() time.Time { return now }
	t.Cleanup(func() { dogfoodNow = oldNow })
	for _, tc := range []struct {
		name       string
		raw        string
		wantReason string
	}{
		{name: "wrong-schema", raw: `{"schema":"other/1","captured_at":"` + now.Format(time.RFC3339Nano) + `","verdict":"ready"}`, wantReason: `schema "other/1" is unsupported`},
		{name: "stale", raw: `{"schema":"fak-micro-cache-affinity-witness/1","captured_at":"` + now.Add(-25*time.Hour).Format(time.RFC3339Nano) + `","verdict":"ready"}`, wantReason: "stale (25h0m0s old; maximum 24h0m0s)"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if err := os.WriteFile(receipt, []byte(tc.raw), 0o600); err != nil {
				t.Fatal(err)
			}
			got := collectDogfoodKernelValue(dir, receipt, 5)
			if got.Cache.Status != "not-yet" || !strings.Contains(got.Cache.Reason, tc.wantReason) || got.CacheCachedPromptTokens != 0 {
				t.Fatalf("got=%+v, want explicit reason containing %q", got, tc.wantReason)
			}
		})
	}
}

func TestDogfoodScoreDefaultCacheReadbackAndExplicitPrecedence(t *testing.T) {
	root := t.TempDir()
	if err := os.WriteFile(filepath.Join(root, "go.mod"), []byte("module example.test/fak\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	t.Chdir(root)
	now := time.Now().UTC()
	canonical := canonicalDogfoodCacheWitnessPath()
	if err := os.MkdirAll(filepath.Dir(canonical), 0o755); err != nil {
		t.Fatal(err)
	}
	write := func(path, schema, verdict, reason string) {
		t.Helper()
		raw := `{"schema":"` + schema + `","captured_at":"` + now.Format(time.RFC3339Nano) + `","verdict":"` + verdict + `","reason":"` + reason + `"}`
		if err := os.WriteFile(path, []byte(raw), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	write(canonical, microCacheWitnessSchema, "not-yet", "canonical receipt consumed")

	run := func(extra ...string) dogfoodKernelValue {
		t.Helper()
		args := []string{"--json", "--claude-home", filepath.Join(root, "claude"), "--runs-dir", filepath.Join(root, "runs")}
		args = append(args, extra...)
		var stdout, stderr bytes.Buffer
		code := runDogfoodScore(&stdout, &stderr, args)
		if code != 0 && code != 1 {
			t.Fatalf("dogfood-score exit=%d stderr=%s", code, stderr.String())
		}
		var payload struct {
			KernelValue dogfoodKernelValue `json:"kernel_value"`
		}
		if err := json.Unmarshal(stdout.Bytes(), &payload); err != nil {
			t.Fatalf("decode dogfood-score: %v\n%s", err, stdout.String())
		}
		return payload.KernelValue
	}
	if got := run(); got.Cache.Status != "not-yet" || got.Cache.Reason != "canonical receipt consumed" {
		t.Fatalf("default readback got=%+v", got.Cache)
	}

	explicit := filepath.Join(root, "explicit.json")
	write(canonical, "wrong/1", "ready", "must not win")
	write(explicit, microCacheWitnessSchema, "ready", "explicit receipt wins")
	if got := run("--cache-witness", explicit); got.Cache.Status != "ready" || got.Cache.Reason != "explicit receipt wins" {
		t.Fatalf("explicit precedence got=%+v", got.Cache)
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
