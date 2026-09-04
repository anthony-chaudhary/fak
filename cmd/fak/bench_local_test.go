package main

import (
	"bytes"
	"strings"
	"testing"
)

func TestBenchLocalHelp(t *testing.T) {
	var stdout, stderr bytes.Buffer
	if code := runBenchLocal(&stdout, &stderr, []string{"--help"}); code != 0 {
		t.Fatalf("exit = %d, stderr = %s", code, stderr.String())
	}
	for _, want := range []string{"fak bench local inventory", "--benchmark NAME", "--engine LABEL", "never uploads"} {
		if !strings.Contains(stdout.String(), want) {
			t.Fatalf("help missing %q:\n%s", want, stdout.String())
		}
	}
}

func TestBenchLocalErrorExit(t *testing.T) {
	var stdout, stderr bytes.Buffer
	if code := runBenchLocal(&stdout, &stderr, []string{"run"}); code != 1 {
		t.Fatalf("exit = %d, want 1", code)
	}
	if !strings.Contains(stderr.String(), "explicit child command is required after --") {
		t.Fatalf("stderr = %q", stderr.String())
	}
}

func TestBenchLocalAttestAndScoreboardHelp(t *testing.T) {
	var stdout, stderr bytes.Buffer
	if code := runBenchLocal(&stdout, &stderr, []string{"attest", "-h"}); code != 0 {
		t.Fatalf("exit = %d, stderr = %s", code, stderr.String())
	}
	for _, want := range []string{"-receipt", "-key", "-generate-key"} {
		if !strings.Contains(stderr.String(), want) && !strings.Contains(stdout.String(), want) {
			t.Fatalf("attest help missing %q", want)
		}
	}

	stdout.Reset()
	stderr.Reset()
	if code := runBenchLocal(&stdout, &stderr, []string{"scoreboard"}); code != 1 {
		t.Fatalf("exit = %d, want 1", code)
	}
	if !strings.Contains(stderr.String(), "scoreboard subcommand required") {
		t.Fatalf("stderr missing usage: %q", stderr.String())
	}
}
