package hooks

import "testing"

func TestMicroharnessWitnessPolarity(t *testing.T) {
	violating := diffOf(t.TempDir(), map[string][]string{
		"internal/microagent/host.go": {"func changed() {}"},
	})
	got, err := gateMicroharnessWitness(violating)
	if err != nil {
		t.Fatal(err)
	}
	if !hasFindingFor(got, "MICROHARNESS_WITNESS", "bounded harness source changed") {
		t.Fatalf("violating findings = %+v", got)
	}

	clean := diffOf(t.TempDir(), map[string][]string{
		"internal/microagent/host.go":      {"func changed() {}"},
		"internal/microagent/host_test.go": {"func TestChanged(t *testing.T) {}"},
	})
	got, err = gateMicroharnessWitness(clean)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 0 {
		t.Fatalf("clean findings = %+v", got)
	}
}

func TestMicroharnessWitnessReceiptSuppressesNudge(t *testing.T) {
	d := diffOf(t.TempDir(), map[string][]string{
		"cmd/microharnessdemo/main.go": {"// Microharness-witness: go run ./cmd/microharnessdemo -selfcheck"},
	})
	got, err := gateMicroharnessWitness(d)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 0 {
		t.Fatalf("receipt findings = %+v", got)
	}
}

func TestMicroharnessWitnessRegisteredAdvisory(t *testing.T) {
	for _, gate := range PreCommitGates() {
		if gate.Name != "MICROHARNESS_WITNESS" {
			continue
		}
		if gate.DefaultMode != "warn" || gate.ModeEnv == "" || gate.EscapeEnv == "" {
			t.Fatalf("registration = %+v", gate)
		}
		return
	}
	t.Fatal("MICROHARNESS_WITNESS is not registered in PreCommitGates")
}
