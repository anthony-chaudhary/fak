package main

import (
	"os"
	"strings"
	"testing"
)

// TestGuardPiExtensionInstallsProviderRepoint is the acceptance witness: wrapping `pi` writes a
// session-scoped extension that registers the anthropic provider at the gateway origin and
// prepends `-e <path>` before the user args, so Pi routes through the kernel.
func TestGuardPiExtensionInstallsProviderRepoint(t *testing.T) {
	dir := t.TempDir()
	command, install, err := installGuardPiExtensionAt(
		[]string{"pi", "-p", "hello"},
		"http://127.0.0.1:4567",
		dir,
	)
	if err != nil {
		t.Fatalf("install pi extension: %v", err)
	}
	if !install.Applied {
		t.Fatalf("pi extension not applied: %+v", install)
	}
	if got, want := install.BaseURL, "http://127.0.0.1:4567"; got != want {
		t.Fatalf("install.BaseURL = %q, want %q (bare origin; Pi appends /v1/messages)", got, want)
	}
	// -e <path> must sit immediately after the executable, before the prompt args.
	if got, want := command[1], "-e"; got != want {
		t.Fatalf("command missing -e flag after executable: %v", command)
	}
	if got, want := command[2], install.ExtensionPath; got != want {
		t.Fatalf("extension path = %q, want %q", got, want)
	}
	if got, want := strings.Join(command[3:], "\x00"), strings.Join([]string{"-p", "hello"}, "\x00"); got != want {
		t.Fatalf("user args changed or -e was appended after prompt args: %v", command)
	}

	data, err := os.ReadFile(install.ExtensionPath)
	if err != nil {
		t.Fatalf("read pi extension: %v", err)
	}
	src := string(data)
	// The module must register the anthropic provider at the gateway origin with NO models
	// (so Pi keeps every Claude model it knows and swaps only the endpoint).
	for _, want := range []string{
		"export default function (pi)",
		`pi.registerProvider("anthropic"`,
		`baseUrl: "http://127.0.0.1:4567"`,
	} {
		if !strings.Contains(src, want) {
			t.Fatalf("extension missing %q:\n%s", want, src)
		}
	}
	if strings.Contains(src, "models") {
		t.Fatalf("extension should NOT set models (a baseUrl-only override preserves Pi's model list):\n%s", src)
	}
}

// TestGuardPiExtensionInjectsAcrossInvocationForms proves `-e <path>` lands immediately after
// the executable and before every user arg regardless of how Pi is named — a launcher-suffixed
// `pi.exe` or an absolute path resolves through the same profile gate as a bare `pi`, so the
// repoint is not silently skipped for a path-qualified invocation (the detection test covers
// these names; this locks the actual `-e` injection + ordering for them).
func TestGuardPiExtensionInjectsAcrossInvocationForms(t *testing.T) {
	for _, tc := range []struct {
		name    string
		command []string
		rest    []string
	}{
		{name: "exe-suffix", command: []string{"pi.exe", "chat", "hi"}, rest: []string{"chat", "hi"}},
		{name: "absolute-path", command: []string{"/usr/local/bin/pi", "-p", "hello"}, rest: []string{"-p", "hello"}},
		{name: "no-user-args", command: []string{"pi"}, rest: []string{}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			command, install, err := installGuardPiExtensionAt(tc.command, "http://127.0.0.1:4567", t.TempDir())
			if err != nil {
				t.Fatalf("install pi extension: %v", err)
			}
			if !install.Applied {
				t.Fatalf("pi extension not applied for %v: %+v", tc.command, install)
			}
			if got := command[0]; got != tc.command[0] {
				t.Fatalf("executable changed: %q -> %q", tc.command[0], got)
			}
			if command[1] != "-e" || command[2] != install.ExtensionPath {
				t.Fatalf("-e <path> not injected right after executable: %v", command)
			}
			if got, want := strings.Join(command[3:], "\x00"), strings.Join(tc.rest, "\x00"); got != want {
				t.Fatalf("user args changed or -e appended after them: %v", command)
			}
		})
	}
}

// TestGuardPiExtensionSkipsOffAndNonPi proves the install is inert for --pi-extension=false, a
// non-Pi child, and an empty command — the command is returned byte-identical, nothing written.
func TestGuardPiExtensionSkipsOffAndNonPi(t *testing.T) {
	for _, tc := range []struct {
		name    string
		enabled bool
		command []string
	}{
		{name: "off", enabled: false, command: []string{"pi"}},
		{name: "non-pi", enabled: true, command: []string{"claude"}},
		{name: "non-pi-codex", enabled: true, command: []string{"codex"}},
		{name: "empty", enabled: true, command: nil},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var (
				command []string
				install guardPiInstall
				err     error
			)
			if !tc.enabled {
				command, install, err = installGuardPiExtension(tc.command, false, "http://127.0.0.1:4567")
			} else {
				command, install, err = installGuardPiExtensionAt(tc.command, "http://127.0.0.1:4567", t.TempDir())
			}
			if err != nil {
				t.Fatalf("install pi extension: %v", err)
			}
			if install.Applied {
				t.Fatalf("pi extension applied unexpectedly: %+v", install)
			}
			if strings.Join(command, "\x00") != strings.Join(tc.command, "\x00") {
				t.Fatalf("command changed: %v -> %v", tc.command, command)
			}
		})
	}
}

// TestGuardIsPiMatchesProfileRegistry proves the gate is driven by the profile registry
// (RepointExtension), so exactly the pi profile takes the extension repoint and no other
// wrapped agent does.
func TestGuardIsPiMatchesProfileRegistry(t *testing.T) {
	for _, tc := range []struct {
		command string
		want    bool
	}{
		{"pi", true},
		{"pi.exe", true},
		{"/usr/local/bin/pi", true},
		{"claude", false},
		{"codex", false},
		{"opencode", false},
		{"vim", false},
		{"", false},
	} {
		if got := guardIsPi(tc.command); got != tc.want {
			t.Fatalf("guardIsPi(%q) = %v, want %v", tc.command, got, tc.want)
		}
	}
}

// TestGuardPiBaseURLTrimsTrailingSlash proves the base URL handed to Pi is the bare origin with
// any trailing slash trimmed — Pi appends /v1/messages itself, exactly like ANTHROPIC_BASE_URL.
func TestGuardPiBaseURLTrimsTrailingSlash(t *testing.T) {
	for _, tc := range []struct{ in, want string }{
		{"http://127.0.0.1:4567", "http://127.0.0.1:4567"},
		{"http://127.0.0.1:4567/", "http://127.0.0.1:4567"},
		{"  http://127.0.0.1:4567  ", "http://127.0.0.1:4567"},
	} {
		if got := guardPiBaseURL(tc.in); got != tc.want {
			t.Fatalf("guardPiBaseURL(%q) = %q, want %q", tc.in, got, tc.want)
		}
	}
}
