package main

import (
	"bytes"
	"strings"
	"testing"
)

func TestSlackWalkIncludesNewsRefreshCommand(t *testing.T) {
	clearSlackEnv(t)
	var out, errb bytes.Buffer
	code := runSlackWalk(&out, &errb, nil)
	if code != 0 {
		t.Fatalf("slack walk exit = %d, stderr=%s", code, errb.String())
	}
	got := out.String()
	if !strings.Contains(got, "news") || !strings.Contains(got, "fak news post --title TITLE --notes-file FILE") {
		t.Fatalf("walk output missing news refresh command:\n%s", got)
	}
}

func TestSlackRefreshNewsNeedsDigestInput(t *testing.T) {
	clearSlackEnv(t)
	var out, errb bytes.Buffer
	code := runSlackRefresh(&out, &errb, []string{"--surface", "news"})
	if code != 0 {
		t.Fatalf("slack refresh news exit = %d, stderr=%s", code, errb.String())
	}
	got := out.String()
	if !strings.Contains(got, "news: SKIP") || !strings.Contains(got, "needs --news-title and --news-file") {
		t.Fatalf("refresh output should skip news without digest input:\n%s", got)
	}
}

func TestSlackRefreshBlockersNeedsIssuePayload(t *testing.T) {
	clearSlackEnv(t)
	var out, errb bytes.Buffer
	code := runSlackRefresh(&out, &errb, []string{"--surface", "blockers"})
	if code != 0 {
		t.Fatalf("slack refresh blockers exit = %d, stderr=%s", code, errb.String())
	}
	got := out.String()
	if !strings.Contains(got, "blockers: SKIP") || !strings.Contains(got, "needs --blockers-issues") {
		t.Fatalf("refresh output should skip blockers without issue payload:\n%s", got)
	}
}

func TestSlackRefreshUnknownSurface(t *testing.T) {
	clearSlackEnv(t)
	var out, errb bytes.Buffer
	code := runSlackRefresh(&out, &errb, []string{"--surface", "missing"})
	if code != 2 {
		t.Fatalf("unknown surface exit = %d, stderr=%s stdout=%s", code, errb.String(), out.String())
	}
	if !strings.Contains(errb.String(), "unknown surface") {
		t.Fatalf("unknown surface error not surfaced: %s", errb.String())
	}
}
