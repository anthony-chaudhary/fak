package capabilitymatrix

import "testing"

// fak-test:runtime fast est=2ms lane=default
func TestTrainedLoopsCheckpointRegistrationDrivesLookup(t *testing.T) {
	const model = "inkling2-trained-loops-stub"
	t.Cleanup(func() { RegisterCheckpointTrainedLoops(model, 0) })

	RegisterCheckpointTrainedLoops(model, 3)
	if got := Lookup(model).TrainedLoops; got != 3 {
		t.Fatalf("registered loop.count 3, Lookup TrainedLoops = %d", got)
	}
	RegisterCheckpointTrainedLoops(model, 0)
	if got := Lookup(model).TrainedLoops; got != 0 {
		t.Fatalf("cleared registration, Lookup TrainedLoops = %d, want 0", got)
	}
}
