package tb4bench

import "testing"

// TestSpine drives the generated leaf's real surface end to end. Keep this
// representative path working while the proof envelope expands around it.
func TestSpine(t *testing.T) {
	if !Ready() {
		t.Fatal("generated leaf spine did not reach Ready")
	}
}
