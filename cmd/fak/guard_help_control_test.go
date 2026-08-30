package main

import "testing"

func TestGuardControlFlagsHaveHelpGroups(t *testing.T) {
	want := map[string]bool{
		"fleet-bus":      false,
		"output-profile": false, "work-profile": false,
	}
	for _, group := range guardFlagGroups {
		for _, name := range group.flags {
			if _, ok := want[name]; ok {
				want[name] = true
			}
		}
	}
	for name, found := range want {
		if !found {
			t.Errorf("guard control flag --%s has no help group", name)
		}
	}
}
