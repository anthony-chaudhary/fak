package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

func TestInjectGuardProfilesDefaultsInNonFakRepository(t *testing.T) {
	repo := t.TempDir()
	if err := os.WriteFile(filepath.Join(repo, "benchmark.txt"), []byte("non-fak fixture\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	old, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	if err := os.Chdir(repo); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = os.Chdir(old) })

	command := []string{"claude", "--model", "sonnet"}
	got, capture, err := injectGuardProfiles(command, agentDefaultOutputStyle, agentDefaultWorkProfile, false)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 5 || got[1] != "--append-system-prompt" {
		t.Fatalf("injected argv = %#v", got)
	}
	if !strings.Contains(got[2], "<fak:steering v1>") || !strings.Contains(got[2], "<fak:work-profile>") {
		t.Fatalf("composed fragment missing defaults: %q", got[2])
	}
	if !reflect.DeepEqual(got[3:], command[1:]) {
		t.Fatalf("child args changed: %#v", got)
	}
	if capture.OutputProfile != "caveman:native:medium" || capture.WorkProfile != "ponytail:native:medium" || !capture.DefaultActivation || capture.CompositeDigest == "" {
		t.Fatalf("capture = %+v", capture)
	}
	raw, err := marshalGuardProfileCapture(capture)
	if err != nil {
		t.Fatal(err)
	}
	var decoded guardProfileCapture
	if err := json.Unmarshal(raw, &decoded); err != nil || decoded != *capture {
		t.Fatalf("capture round trip = %+v, %v", decoded, err)
	}
}

func TestInjectGuardProfilesIndependentOptOuts(t *testing.T) {
	command := []string{"claude"}
	for _, tc := range []struct {
		name, output, work, want, absent string
	}{
		{name: "output off", output: "full", work: agentDefaultWorkProfile, want: "<fak:work-profile>", absent: "<fak:steering v1>"},
		{name: "work off", output: agentDefaultOutputStyle, work: "standard", want: "<fak:steering v1>", absent: "<fak:work-profile>"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got, capture, err := injectGuardProfiles(command, tc.output, tc.work, true)
			if err != nil || capture == nil || len(got) != 3 {
				t.Fatalf("got (%#v, %+v, %v)", got, capture, err)
			}
			if !strings.Contains(got[2], tc.want) || strings.Contains(got[2], tc.absent) {
				t.Fatalf("fragment = %q", got[2])
			}
		})
	}
	got, capture, err := injectGuardProfiles(command, "full", "standard", true)
	if err != nil || capture != nil || !reflect.DeepEqual(got, command) {
		t.Fatalf("both off = (%#v, %+v, %v)", got, capture, err)
	}
}

func TestInjectGuardProfilesDefaultsIntoCodexDeveloperInstructions(t *testing.T) {
	command := []string{"codex", "exec", "hello"}
	got, capture, err := injectGuardProfiles(command, agentDefaultOutputStyle, agentDefaultWorkProfile, false)
	if err != nil || capture == nil {
		t.Fatalf("default codex = (%#v, %+v, %v)", got, capture, err)
	}
	if len(got) != 5 || got[1] != "-c" || !strings.HasPrefix(got[2], "developer_instructions=") {
		t.Fatalf("codex injection argv = %#v", got)
	}
	if !strings.Contains(got[2], "<fak:steering v1>") || !strings.Contains(got[2], "<fak:work-profile>") {
		t.Fatalf("codex developer instructions missing profiles: %q", got[2])
	}
	if !reflect.DeepEqual(got[3:], command[1:]) {
		t.Fatalf("codex child args changed: %#v", got)
	}
	if capture.Harness != "codex" || capture.ActivationSeam != "codex -c developer_instructions" || !capture.DefaultActivation {
		t.Fatalf("codex capture = %+v", capture)
	}
}

func TestInjectGuardProfilesUnsupportedHarnessPreservesDefaults(t *testing.T) {
	command := []string{"other-agent", "hello"}
	got, capture, err := injectGuardProfiles(command, agentDefaultOutputStyle, agentDefaultWorkProfile, false)
	if err != nil || capture != nil || !reflect.DeepEqual(got, command) {
		t.Fatalf("default unsupported = (%#v, %+v, %v)", got, capture, err)
	}
	got, capture, err = injectGuardProfiles(command, "caveman:high", agentDefaultWorkProfile, true)
	if err == nil || !strings.Contains(err.Error(), "PROFILE_UNSUPPORTED_HARNESS") || got != nil || capture != nil {
		t.Fatalf("explicit unsupported = (%#v, %+v, %v)", got, capture, err)
	}
}

func TestInjectGuardProfilesRefusesUnknownBeforeLaunch(t *testing.T) {
	for _, tc := range []struct{ output, work, reason string }{
		{output: "caveman:original:auto", work: agentDefaultWorkProfile, reason: "RESPONSE_PROFILE_UNKNOWN"},
		{output: agentDefaultOutputStyle, work: "ponytail:original:auto", reason: "WORK_PROFILE_UNKNOWN"},
	} {
		got, capture, err := injectGuardProfiles([]string{"claude"}, tc.output, tc.work, true)
		if err == nil || !strings.Contains(err.Error(), tc.reason) || got != nil || capture != nil {
			t.Fatalf("got (%#v, %+v, %v), want %s", got, capture, err, tc.reason)
		}
	}
}

func TestInjectGuardProfilesPreservesCodexAuthManagementCommands(t *testing.T) {
	for _, command := range [][]string{
		{"codex", "login"},
		{"codex", "login", "status"},
		{"codex", "logout"},
	} {
		got, capture, err := injectGuardProfiles(command, agentDefaultOutputStyle, agentDefaultWorkProfile, false)
		if err != nil {
			t.Fatalf("injectGuardProfiles(%q): %v", command, err)
		}
		if !reflect.DeepEqual(got, command) {
			t.Fatalf("injectGuardProfiles(%q) = %q, want unchanged auth command", command, got)
		}
		if capture != nil {
			t.Fatalf("injectGuardProfiles(%q) capture=%+v, want nil", command, capture)
		}
		if repointed, _ := installGuardCodexConfig(got, true, "http://127.0.0.1:8137/v1", "OPENAI_API_KEY"); !reflect.DeepEqual(repointed, command) {
			t.Fatalf("installGuardCodexConfig(%q) = %q, want auth command to remain direct", got, repointed)
		}
	}
}
