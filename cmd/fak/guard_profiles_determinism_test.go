package main

import (
	"reflect"
	"testing"
)

func TestGuardProfilesDeterministicAndInputSafe(t *testing.T) {
	command := []string{"codex", "exec", "task"}
	first, firstCapture, err := injectGuardProfiles(command, agentDefaultOutputStyle, agentDefaultWorkProfile, true)
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 100; i++ {
		got, capture, err := injectGuardProfiles(command, agentDefaultOutputStyle, agentDefaultWorkProfile, true)
		if err != nil || !reflect.DeepEqual(got, first) || !reflect.DeepEqual(capture, firstCapture) {
			t.Fatalf("run %d differs: got=%v capture=%+v err=%v", i, got, capture, err)
		}
		got[0] = "mutated"
		if command[0] != "codex" {
			t.Fatal("injected argv aliases caller input")
		}
	}
}
