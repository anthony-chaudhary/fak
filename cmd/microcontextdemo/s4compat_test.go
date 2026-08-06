package main

import "testing"

func TestCompatibilityWitness(t *testing.T) {
	r, err := buildCompatReport()
	if err != nil {
		t.Fatal(err)
	}
	if r.PaddingTax > .10 || r.SingletonFallbacks != 1 || r.Scheduled != 97 {
		t.Fatalf("bad report: %+v", r)
	}
}
