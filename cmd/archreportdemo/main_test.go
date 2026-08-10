package main

import "testing"

func TestRunDemo(t *testing.T) {
	got, err := runDemo(true)
	if err != nil {
		t.Fatal(err)
	}
	const want = `fak architecture report demo
schema: fak-architecture/1
healthy leaves: 4 across 3 tiers
upward violations: 1 (primitive -> composite)
direct fan-in hotspots: abi=2, policy=2
diagnostic: retired has a stale tier declaration
selfcheck: PASS (real archreport seam, deterministic fixture)
`
	if got != want {
		t.Fatalf("output mismatch\nwant:\n%s\ngot:\n%s", want, got)
	}
}
