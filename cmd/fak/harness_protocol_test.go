package main

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestHarnessProtocolCLIAndTUIUseSameFixture(t *testing.T) {
	fixture := filepath.Join("..", "..", "docs", "_witnesses", "harness-protocol", "roundtrip-events.jsonl")
	if _, err := os.Stat(fixture); err != nil {
		t.Fatal(err)
	}
	var cli, tui, errb bytes.Buffer
	if code := runHarness(&cli, &errb, []string{"protocol", "project", "--input", fixture, "--view", "cli"}); code != 0 {
		t.Fatalf("cli code=%d %s", code, errb.String())
	}
	errb.Reset()
	if code := runHarness(&tui, &errb, []string{"protocol", "project", "--input", fixture, "--view", "tui"}); code != 0 {
		t.Fatalf("tui code=%d %s", code, errb.String())
	}
	for _, want := range []string{"witness-run-6789", "stable protocol", "search", "12", "7"} {
		if !strings.Contains(cli.String(), want) || !strings.Contains(tui.String(), want) {
			t.Fatalf("missing %q\nCLI:\n%s\nTUI:\n%s", want, cli.String(), tui.String())
		}
	}
	if cli.String() == tui.String() {
		t.Fatal("consumers must be independent renderers")
	}
}
