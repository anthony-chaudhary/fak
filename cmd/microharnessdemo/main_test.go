package main

import (
	"bytes"
	"context"
	"strings"
	"testing"
)

func TestMicroharnessSpine(t *testing.T) {
	r, err := run(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if err := check(r); err != nil {
		t.Fatal(err)
	}
	turns := map[string]int{}
	for _, rec := range r.Receipts {
		turns[rec.TaskID] = rec.Turns
	}
	if turns["architecture"] != 2 || turns["tools"] != 1 || turns["proof"] != 3 {
		t.Fatalf("turn envelope = %#v", turns)
	}
}

func TestRenderWitness(t *testing.T) {
	r, err := run(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	var got bytes.Buffer
	render(&got, r)
	for _, want := range []string{
		"FAK MICROHARNESS",
		"receipt architecture depth=1 turns=2",
		"receipt tools        depth=2 turns=1",
		"receipt proof        depth=2 turns=3",
		"full child transcripts in root=false",
		"depth<=2; turns/child<=3",
		"PASS — go run ./cmd/microharnessdemo -selfcheck",
	} {
		if !strings.Contains(got.String(), want) {
			t.Errorf("captured render missing %q:\n%s", want, got.String())
		}
	}
}
