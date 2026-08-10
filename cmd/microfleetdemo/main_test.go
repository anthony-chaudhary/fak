package main

import (
	"bytes"
	"strings"
	"testing"
)

func TestMicrofleetSpine(t *testing.T) {
	r, err := run()
	if err != nil {
		t.Fatal(err)
	}
	if err := check(r); err != nil {
		t.Fatal(err)
	}
	if r.PeakResidentAgents != 4 {
		t.Fatalf("peak resident = %d, want 4", r.PeakResidentAgents)
	}
	if r.Compactions == 0 {
		t.Fatal("compactions = 0, want native managed-context compaction")
	}
	if !r.DeniedNeverDispatched {
		t.Fatal("denied action reached backend")
	}
}

func TestRenderWitness(t *testing.T) {
	r, err := run()
	if err != nil {
		t.Fatal(err)
	}
	var got bytes.Buffer
	render(&got, r)
	for _, want := range []string{
		"FAK MICROFLEET",
		"peak resident 4; parked 20",
		"resident token-turns (92.2% avoided)",
		"192 compactions",
		"denied dispatches = false",
		"off-list destination refused",
		"go run ./cmd/microfleetdemo -selfcheck",
	} {
		if !strings.Contains(got.String(), want) {
			t.Errorf("captured render missing %q:\n%s", want, got.String())
		}
	}
}
