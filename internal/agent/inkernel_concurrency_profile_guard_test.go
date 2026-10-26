package agent

import "testing"

// fak-test:runtime fast est=1ms lane=default
func TestInKernelConcurrencyProfileUnguardedSerialIsUnknown(t *testing.T) {
	for _, serialized := range [][]bool{{false, false}, {true, false}, {false, true}} {
		c := newConcurrencyProfiler()
		for _, guarded := range serialized {
			phase := c.admit()
			c.forwardEnter(phase, guarded)
			c.forwardExit(phase)
		}
		prof := c.build("cpu/reference")
		if prof.Mechanism != MechanismUnknown || prof.MaxConcurrent != 1 {
			t.Fatalf("serialized=%v: mechanism=%q max_concurrent_forward=%d, want unknown/1", serialized, prof.Mechanism, prof.MaxConcurrent)
		}
	}
}
