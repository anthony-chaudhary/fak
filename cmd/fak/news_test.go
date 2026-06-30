package main

import (
	"bytes"
	"strings"
	"testing"
)

func TestNewsPostDryRun(t *testing.T) {
	clearSlackEnv(t)
	t.Setenv("FAK_SCOREBOARD_TOKEN", "bottok-news-secret")
	t.Setenv("FAK_SCOREBOARD_SOURCE", "test-agent")

	var out, errb bytes.Buffer
	code := runNewsPost(&out, &errb, []string{
		"--title", "AI infra digest",
		"--notes", "Source-linked item one",
		"--dry-run",
	})
	if code != 0 {
		t.Fatalf("news dry-run exit = %d, stderr=%s", code, errb.String())
	}
	got := out.String()
	for _, want := range []string{"news - AI infra digest", "Source-linked item one", "posted by test-agent"} {
		if !strings.Contains(got, want) {
			t.Fatalf("dry-run output missing %q:\n%s", want, got)
		}
	}
	if strings.Contains(got, "bottok-news-secret") {
		t.Fatalf("dry-run must not print the raw token:\n%s", got)
	}
}

func TestNewsPostRequiresTitleAndBody(t *testing.T) {
	clearSlackEnv(t)
	var out, errb bytes.Buffer
	if code := runNewsPost(&out, &errb, []string{"--notes", "body", "--dry-run"}); code != 2 {
		t.Fatalf("missing title should exit 2, got %d", code)
	}
	out.Reset()
	errb.Reset()
	if code := runNewsPost(&out, &errb, []string{"--title", "digest", "--dry-run"}); code != 2 {
		t.Fatalf("missing body should exit 2, got %d", code)
	}
}
