package main

import (
	"bytes"
	"path/filepath"
	"strings"
	"testing"
)

func TestCachevalueFeedRendersFabricTrack1(t *testing.T) {
	var out, errb bytes.Buffer
	fixture := filepath.Join("..", "..", "experiments", "microcontext", "s2b-gcp-inkernel-prefix-ab-pass-2026-08-07.json")
	dir := t.TempDir()
	code := runCachevalueFeed(&out, &errb, []string{"--ledger", filepath.Join(dir, "empty.jsonl"), "--savings-ledger", filepath.Join(dir, "empty-savings.jsonl"), "--usage-ledger", filepath.Join(dir, "empty-usage.jsonl"), "--microcontext-ledger", fixture, "--dry-run", "--source", "agent"})
	if code != 0 {
		t.Fatalf("code=%d stderr=%s", code, errb.String())
	}
	if !strings.Contains(out.String(), "86.6%") {
		t.Fatalf("Track-1 micro-context reuse absent: %s", out.String())
	}
}

func TestCachevalueFeedRejectsSyntheticFabric(t *testing.T) {
	var out, errb bytes.Buffer
	code := runCachevalueFeed(&out, &errb, []string{"--microcontext-ledger", filepath.Join("..", "..", "experiments", "microcontext", "s0-local-10k-quality-ledger-2026-08-06.json"), "--dry-run"})
	if code == 0 || !strings.Contains(errb.String(), "micro-context provenance") {
		t.Fatalf("code=%d stderr=%s", code, errb.String())
	}
}
