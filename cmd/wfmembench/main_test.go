package main

import (
	"context"
	"encoding/json"
	"os"
	"strings"
	"testing"
)

func mustCompare(t *testing.T) Comparison {
	t.Helper()
	ctx := context.Background()
	im, dir, err := attachFixtureImage(ctx)
	if err != nil {
		t.Fatalf("attach: %v", err)
	}
	t.Cleanup(func() { os.RemoveAll(dir) }) // reap the wfmem-* image tree (#3508)
	tombstoneStalePreference(im)
	return Compare(ctx, im, benchRequest(), reproCommand)
}

// TestAttachFixtureImageReturnsReapableDir is the #3508 regression: attachFixtureImage
// must hand its temp dir back so the caller can reap it (the old signature swallowed the
// dir, leaking one wfmem-* image tree per invocation). Because the attached image resolves
// from memory, removing the dir after attach is safe AND leaves the image usable — the
// invariant that lets the call-site bound reclaim the leak without a reaper.
func TestAttachFixtureImageReturnsReapableDir(t *testing.T) {
	ctx := context.Background()
	im, dir, err := attachFixtureImage(ctx)
	if err != nil {
		t.Fatalf("attach: %v", err)
	}
	if dir == "" {
		t.Fatal("attachFixtureImage returned an empty dir — the caller cannot reap the temp image tree (#3508 leak)")
	}
	if _, err := os.Stat(dir); err != nil {
		t.Fatalf("temp image dir missing before reap: %v", err)
	}
	if err := os.RemoveAll(dir); err != nil {
		t.Fatalf("RemoveAll(dir) failed — the call-site bound cannot reclaim the leak: %v", err)
	}
	if _, err := os.Stat(dir); !os.IsNotExist(err) {
		t.Fatalf("temp image dir survived reap: stat err = %v", err)
	}
	if got := im.Pages(); got == 0 {
		t.Fatal("attached image reports 0 pages after reap — it should resolve from memory, not disk")
	}
}

func armsByName(cmp Comparison) map[string]ArmReport {
	m := make(map[string]ArmReport, len(cmp.Arms))
	for _, a := range cmp.Arms {
		m[a.Name] = a
	}
	return m
}

// TestCompare_Deterministic verifies the model-free workload is reproducible: two
// independent attach+compare passes produce byte-identical JSON.
func TestCompare_Deterministic(t *testing.T) {
	j1, _ := json.Marshal(mustCompare(t))
	j2, _ := json.Marshal(mustCompare(t))
	if string(j1) != string(j2) {
		t.Fatalf("workflow-memory comparison is not deterministic:\n j1=%s\n j2=%s", j1, j2)
	}
}

// TestFixtureHazards is acceptance #1: the fixture must carry the workflow-memory
// hazards, including the two sealed pages and the runtime tombstone.
func TestFixtureHazards(t *testing.T) {
	cmp := mustCompare(t)
	if cmp.Fixture.Sealed < 2 {
		t.Fatalf("fixture must have >=2 sealed pages (injection + secret), got %d", cmp.Fixture.Sealed)
	}
	if cmp.Fixture.Tombstoned < 1 {
		t.Fatalf("tombstone hazard not applied: tombstoned=%d", cmp.Fixture.Tombstoned)
	}
	if cmp.Fixture.Benign < 1 || cmp.Fixture.RawBytes <= 0 {
		t.Fatalf("fixture decomposition not reported: %+v", cmp.Fixture)
	}
}

// TestThreeArms is acceptance #2/#3: three arms, each reporting the full metric set,
// each arm faithful to its defining hazard behavior.
func TestThreeArms(t *testing.T) {
	cmp := mustCompare(t)
	if len(cmp.Arms) != 3 {
		t.Fatalf("want 3 arms, got %d", len(cmp.Arms))
	}
	arms := armsByName(cmp)

	full, ok := arms["full-transcript"]
	if !ok {
		t.Fatal("missing full-transcript arm")
	}
	// The baseline must visibly fail the trust hazards so the comparison is not vacuous.
	if full.PoisonLeak < 2 {
		t.Fatalf("full transcript must leak both sealed pages, got %d", full.PoisonLeak)
	}
	if full.StaleReuse < 1 {
		t.Fatalf("full transcript must replay the tombstoned page, got stale_reuse=%d", full.StaleReuse)
	}
	if full.FallbackToRaw != 1.0 {
		t.Fatalf("full transcript fallback-to-raw must be 1.0, got %v", full.FallbackToRaw)
	}

	naive, ok := arms["naive-global-summary"]
	if !ok {
		t.Fatal("missing naive-global-summary arm")
	}
	if naive.PoisonLeak < 2 {
		t.Fatalf("naive summary must fold both sealed pages, got %d", naive.PoisonLeak)
	}
	if naive.SourceCoverage != 0.0 {
		t.Fatalf("naive global summary must destroy provenance (source_coverage=0), got %v", naive.SourceCoverage)
	}
	if naive.FallbackToRaw != 0.0 {
		t.Fatalf("naive summary cannot fall back to raw, got %v", naive.FallbackToRaw)
	}
	if naive.ResidentBytes >= full.ResidentBytes {
		t.Fatalf("naive summary must be leaner than the full transcript: naive=%d full=%d", naive.ResidentBytes, full.ResidentBytes)
	}

	views, ok := arms["provenance-bound-virtual-views"]
	if !ok {
		t.Fatal("missing provenance-bound-virtual-views arm")
	}
	if views.Kind != "measured" {
		t.Fatalf("virtual-views arm must be measured, got %q", views.Kind)
	}
	// The fail-closed witnesses: zero stale reuse, zero poison leakage.
	if views.StaleReuse != 0 {
		t.Fatalf("virtual views must NOT reuse stale views (fail-closed), got %d", views.StaleReuse)
	}
	if views.PoisonLeak != 0 {
		t.Fatalf("virtual views must NOT leak poison (fail-closed), got %d", views.PoisonLeak)
	}
	if !views.TaskSuccess {
		t.Fatal("virtual views must still answer the goal")
	}
	// A real raw fallback must be exercised, else the fault metric is vacuous.
	if views.FallbackToRaw <= 0 {
		t.Fatalf("virtual views must exercise a real raw fallback, got %v", views.FallbackToRaw)
	}
	// The headline win: the safe arm carries fewer bytes than the full transcript
	// precisely because the sealed and tombstoned pages are paged out.
	if views.ResidentBytes >= full.ResidentBytes {
		t.Fatalf("virtual views must carry fewer resident bytes than the full transcript: views=%d full=%d", views.ResidentBytes, full.ResidentBytes)
	}
}

// TestReplays is acceptance #4/#5: the stale replay rejects the old view, and the
// poison replay contains the sealed source.
func TestReplays(t *testing.T) {
	cmp := mustCompare(t)
	if cmp.Stale.Recomputes < 1 {
		t.Fatalf("stale replay must RECOMPUTE the policy-drifted view, got %d", cmp.Stale.Recomputes)
	}
	if !cmp.Stale.OldViewRejected {
		t.Fatalf("stale replay: old view not rejected: %+v", cmp.Stale)
	}
	if cmp.Poison.SealedRefused < 1 {
		t.Fatalf("poison replay must REFUSE the sealed probe, got %d", cmp.Poison.SealedRefused)
	}
	if !cmp.Poison.SealedContained {
		t.Fatalf("poison replay: sealed source not contained: %+v", cmp.Poison)
	}
}

// TestRenderMarkdown sanity-checks the summary carries the command and the arms.
func TestRenderMarkdown(t *testing.T) {
	cmp := mustCompare(t)
	md := renderMarkdown(cmp)
	for _, want := range []string{reproCommand, "full-transcript", "naive-global-summary", "provenance-bound-virtual-views", "Stale replay", "Poison replay"} {
		if !strings.Contains(md, want) {
			t.Fatalf("markdown summary missing %q", want)
		}
	}
}
