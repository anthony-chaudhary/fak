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

func TestDecisionClassCorpusMapsTurnsAndRefusesMasterOnlyWork(t *testing.T) {
	r, err := run(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if err := check(r); err != nil {
		t.Fatal(err)
	}
	var got bytes.Buffer
	render(&got, r)
	for _, want := range []string{
		"task class one_turn case=capability-selection turns=1 outcome=completed",
		"task class bounded_correction case=witness-correction turns=3 outcome=completed",
		"task class root_only case=irreversible-goal turns=0 outcome=refused-delegation",
	} {
		if !strings.Contains(got.String(), want) {
			t.Errorf("captured task-class corpus missing %q:\n%s", want, got.String())
		}
	}
	for _, rec := range r.Receipts {
		if rec.TaskID == "irreversible-goal" {
			t.Fatal("master-context work crossed the delegated receipt boundary")
		}
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

func TestBenchmarkComparesMonolithWithReceiptOnlyRecursion(t *testing.T) {
	r, err := run(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if err := check(r); err != nil {
		t.Fatal(err)
	}
	if r.Benchmark.Method == "" {
		t.Fatal("benchmark method is unlabeled")
	}
	if r.Benchmark.Monolith.QualityPassed != r.Benchmark.ReceiptOnly.QualityPassed {
		t.Fatalf("quality differs: monolith=%d receipt_only=%d", r.Benchmark.Monolith.QualityPassed, r.Benchmark.ReceiptOnly.QualityPassed)
	}
	if r.Benchmark.ReceiptOnly.RetainedBytes >= r.Benchmark.Monolith.RetainedBytes {
		t.Fatalf("root context did not shrink: monolith=%d receipt_only=%d", r.Benchmark.Monolith.RetainedBytes, r.Benchmark.ReceiptOnly.RetainedBytes)
	}
	if r.Benchmark.ReceiptOnly.CacheReadTokens == 0 || r.Benchmark.ReceiptOnly.CostMicroUSD >= r.Benchmark.Monolith.CostMicroUSD {
		t.Fatalf("cache/cost receipt is not useful: %#v", r.Benchmark)
	}
}

func TestReductionPctRefusesInvalidEnvelope(t *testing.T) {
	for _, tc := range []struct{ baseline, candidate int }{{0, 0}, {10, -1}, {10, 11}} {
		if got := reductionPct(tc.baseline, tc.candidate); got != 0 {
			t.Fatalf("reductionPct(%d, %d) = %d, want refusal value 0", tc.baseline, tc.candidate, got)
		}
	}
}
