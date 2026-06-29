package main

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// TestCachevalueSurfaceReadyAgainstChannel is the issue #1306 witness: `fak slack check`
// lists `cachevalue` READY against C0BDSB81XDZ once a token resolves.
func TestCachevalueSurfaceReadyAgainstChannel(t *testing.T) {
	clearSlackEnv(t)
	t.Setenv("FAK_SCOREBOARD_TOKEN", "bottok-sb") // shared workspace token

	reports := buildSurfaceReports()
	cv := reportByName(reports, "cachevalue")
	if cv == nil {
		t.Fatal("cachevalue surface not registered in slackSurfaces")
	}
	if cv.Channel != "C0BDSB81XDZ" {
		t.Fatalf("cachevalue should default to the public channel C0BDSB81XDZ, got %q", cv.Channel)
	}
	if cv.ChannelSource != "built-in default" {
		t.Fatalf("cachevalue channel should come from the built-in default, got %q", cv.ChannelSource)
	}
	if !cv.TokenSet || !strings.Contains(cv.TokenSource, "scoreboard-fallback") {
		t.Fatalf("cachevalue should fall back to the scoreboard token: %+v", cv)
	}
	if !cv.Ready {
		t.Fatalf("cachevalue should be READY (token + channel resolved): %+v", cv)
	}
}

// TestCachevalueFeedDryRunRendersCard is the issue #1306 witness: `fak cachevalue feed
// --dry-run` renders the exact card from the dogfooded ledger, posting nothing.
func TestCachevalueFeedDryRunRendersCard(t *testing.T) {
	clearSlackEnv(t)
	dir := t.TempDir()
	ledger := filepath.Join(dir, "cache-value.jsonl")
	// Two multi-turn weeks, trending up (60% -> 80% realized reuse).
	body := `{"date":"2026-06-15","session_type":"guard","turns":10,"prompt_tokens":1000,"reused_tokens":600}
{"date":"2026-06-22","session_type":"guard","turns":10,"prompt_tokens":1000,"reused_tokens":800}
`
	if err := os.WriteFile(ledger, []byte(body), 0o600); err != nil {
		t.Fatal(err)
	}

	var out, errb bytes.Buffer
	code := runCachevalueFeed(&out, &errb, []string{"--ledger", ledger, "--dry-run", "--source", "agent"})
	if code != 0 {
		t.Fatalf("feed --dry-run exit = %d, stderr=%s", code, errb.String())
	}
	got := out.String()
	if !strings.Contains(got, "MEASURED") {
		t.Fatalf("dry-run card should be MEASURED on a multi-turn corpus:\n%s", got)
	}
	if !strings.Contains(got, "80.0%") || !strings.Contains(got, "improved") {
		t.Fatalf("dry-run card dropped the latest reuse / trend:\n%s", got)
	}
	if !strings.Contains(got, "marginal-over-tuned-warm-KV") {
		t.Fatalf("dry-run card dropped the #1066 honesty fence:\n%s", got)
	}
	if !strings.Contains(got, "_posted by agent_") {
		t.Fatalf("dry-run card dropped the source:\n%s", got)
	}
}

// TestCachevalueFeedEmptyLedgerIsHonest checks a missing ledger folds to the INSUFFICIENT
// card (an honest "nothing to trend yet") rather than failing.
func TestCachevalueFeedEmptyLedgerIsHonest(t *testing.T) {
	clearSlackEnv(t)
	var out, errb bytes.Buffer
	code := runCachevalueFeed(&out, &errb, []string{"--ledger", filepath.Join(t.TempDir(), "absent.jsonl"), "--dry-run"})
	if code != 0 {
		t.Fatalf("feed --dry-run on a missing ledger should still render, exit=%d stderr=%s", code, errb.String())
	}
	if !strings.Contains(out.String(), "INSUFFICIENT") {
		t.Fatalf("missing ledger should fold to INSUFFICIENT:\n%s", out.String())
	}
}

// TestCachevaluePostRequiresReportJSON checks `post` with no payload exits 2 (it has no
// ledger to fall back to — the caller must say what to post).
func TestCachevaluePostRequiresReportJSON(t *testing.T) {
	clearSlackEnv(t)
	var out, errb bytes.Buffer
	if code := runCachevaluePost(&out, &errb, []string{"--dry-run"}); code != 2 {
		t.Fatalf("post with no --report-json should exit 2, got %d", code)
	}
}
