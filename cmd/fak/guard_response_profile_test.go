package main

import (
	"encoding/json"
	"reflect"
	"strings"
	"testing"
)

func TestInjectGuardResponseProfileClaudeCapturesInstructionEnvelope(t *testing.T) {
	command := []string{"claude", "--model", "sonnet"}
	got, capture, err := injectGuardResponseProfile(command, "caveman:low")
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 5 || got[1] != "--append-system-prompt" || !strings.Contains(got[2], "<fak:steering v1>") {
		t.Fatalf("injected argv = %#v", got)
	}
	if !reflect.DeepEqual(got[3:], command[1:]) {
		t.Fatalf("child args changed: %#v", got)
	}
	if capture.Canonical != "caveman:native:low" || capture.Harness != "claude" || capture.ActivationSeam != "claude --append-system-prompt" || capture.FragmentDigest == "" || capture.DisableCommand == "" {
		t.Fatalf("capture = %+v", capture)
	}
	raw, err := marshalGuardResponseProfileCapture(capture)
	if err != nil {
		t.Fatal(err)
	}
	var decoded guardResponseProfileCapture
	if err := json.Unmarshal(raw, &decoded); err != nil {
		t.Fatal(err)
	}
	if decoded != *capture {
		t.Fatalf("capture round trip = %+v, want %+v", decoded, *capture)
	}
}

func TestInjectGuardResponseProfileDefaultIsByteIdentical(t *testing.T) {
	command := []string{"codex", "exec", "hello"}
	for _, selection := range []string{"", "full", " FULL "} {
		got, capture, err := injectGuardResponseProfile(command, selection)
		if err != nil || capture != nil || !reflect.DeepEqual(got, command) {
			t.Fatalf("selection %q = (%#v, %+v, %v)", selection, got, capture, err)
		}
	}
}

func TestInjectGuardResponseProfileRefusesUnsupportedHarness(t *testing.T) {
	got, capture, err := injectGuardResponseProfile([]string{"codex", "exec"}, "caveman:medium")
	if err == nil || !strings.Contains(err.Error(), "RESPONSE_PROFILE_UNSUPPORTED_HARNESS") || got != nil || capture != nil {
		t.Fatalf("got (%#v, %+v, %v)", got, capture, err)
	}
}

func TestInjectGuardResponseProfileRefusesUnknownBeforeLaunch(t *testing.T) {
	got, capture, err := injectGuardResponseProfile([]string{"claude"}, "caveman:original:auto")
	if err == nil || !strings.Contains(err.Error(), "RESPONSE_PROFILE_UNKNOWN") || got != nil || capture != nil {
		t.Fatalf("got (%#v, %+v, %v)", got, capture, err)
	}
}
