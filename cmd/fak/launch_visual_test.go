package main

import (
	"bytes"
	"path/filepath"
	"strings"
	"testing"
)

func TestLaunchVisualCheckPassesOnFixture(t *testing.T) {
	data := filepath.Join("..", "..", "internal", "launchvisual", "testdata", "charts.data.json")
	index := filepath.Join("..", "..", "internal", "launchvisual", "testdata", "headlines.golden.json")
	var out, errOut bytes.Buffer
	if code := runLaunchVisual(&out, &errOut, []string{"--check", "--data", data, "--index", index}); code != 0 {
		t.Fatalf("check on current fixture exit %d: %s%s", code, out.String(), errOut.String())
	}
	if !strings.Contains(out.String(), "launch docs cite only index-traced numbers") {
		t.Errorf("check success line missing: %q", out.String())
	}
}

func TestLaunchVisualUsageAndMissingInputs(t *testing.T) {
	var out, errOut bytes.Buffer
	if code := runLaunchVisual(&out, &errOut, []string{"--no-such-flag"}); code != 2 {
		t.Errorf("unknown flag exit %d, want 2", code)
	}
	if code := runLaunchVisual(&out, &errOut, []string{"stray"}); code != 2 {
		t.Errorf("stray argument exit %d, want 2", code)
	}
	if code := runLaunchVisual(&out, &errOut, []string{"--check", "--data", "nope.json", "--index", "nope.json"}); code != 1 {
		t.Errorf("missing data exit %d, want 1", code)
	}
}
