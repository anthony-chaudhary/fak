package main

import (
	"context"
	"testing"
)

func TestDescriptorBenchmarkUsesExistingRuntime(t *testing.T) {
	r, err := runDescriptorBenchmark(context.Background(), 1000)
	if err != nil {
		t.Fatal(err)
	}
	a := r.Arms[len(r.Arms)-1]
	if a.ModelCalls != 1000 || a.BytesPerDescriptor <= 0 || r.BaseInstalls != 1 {
		t.Fatalf("bad report: %+v", r)
	}
}
