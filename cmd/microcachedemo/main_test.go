package main

import (
	"bytes"
	"strings"
	"testing"
)

func TestMicrocacheSpine(t *testing.T) {
	got, err := run()
	if err != nil {
		t.Fatal(err)
	}
	if err := check(got); err != nil {
		t.Fatal(err)
	}
	if got.FakUpstreamCalls != 4 || got.VDSOHits != 252 {
		t.Fatalf("engine/hits = %d/%d, want 4/252", got.FakUpstreamCalls, got.VDSOHits)
	}
}

func TestRenderWitness(t *testing.T) {
	got, err := run()
	if err != nil {
		t.Fatal(err)
	}
	var b bytes.Buffer
	render(&b, got)
	for _, want := range []string{"32 agents x 8 calls = 256", "256 -> 4 calls", "98.4% upstream work avoided", "denied tool reached engine 0 time(s)", "private A/B engine calls = 1/1", "go run ./cmd/microcachedemo -selfcheck"} {
		if !strings.Contains(b.String(), want) {
			t.Errorf("render missing %q:\n%s", want, b.String())
		}
	}
}
