package main

import "testing"

// TestServeLlamaSlotAffinityDefaultsOn: `fak serve --base-url <llama-server>` keeps prefix
// slot pinning without a flag (the Strix proxy unit passes none); `=false` turns it off.
//
// fak-test:runtime fast est=20ms
func TestServeLlamaSlotAffinityDefaultsOn(t *testing.T) {
	for _, c := range []struct {
		argv []string
		want bool
	}{
		{nil, true},
		{[]string{"--llama-slot-affinity=false"}, false},
	} {
		fs, sf := newServeFlagSet()
		if !parseFlags(fs, c.argv) {
			t.Fatalf("parse %v failed", c.argv)
		}
		if *sf.llamaSlotAffinity != c.want {
			t.Fatalf("argv %v: --llama-slot-affinity = %v, want %v", c.argv, *sf.llamaSlotAffinity, c.want)
		}
	}
}
