package servelaunch

import (
	"errors"
	"reflect"
	"strings"
	"testing"
)

// TestBuildSSHArgvShape proves the argv builder emits exactly the documented
// shape and keeps the hardening block in a fixed order, so a launch receipt
// that records the argv stays reproducible.
// fak-test:runtime fast est=1ms lane=default
func TestBuildSSHArgvShape(t *testing.T) {
	argv, err := BuildSSHArgv(SSHTarget{Host: "gpu-box-a", User: "bench", Port: 2222, Key: "/home/me/.ssh/id_ed25519"})
	if err != nil {
		t.Fatalf("err = %v, want nil", err)
	}
	want := []string{
		"ssh",
		"-p", "2222",
		"-i", "/home/me/.ssh/id_ed25519",
		"-o", "BatchMode=yes",
		"-o", "StrictHostKeyChecking=accept-new",
		"-o", "UserKnownHostsFile=/dev/null",
		"-o", "LogLevel=ERROR",
		"-o", "ClearAllForwardings=yes",
		"-o", "PermitLocalCommand=no",
		"-o", "ProxyCommand=none",
		"bench@gpu-box-a",
	}
	if !reflect.DeepEqual(argv, want) {
		t.Errorf("argv = %v, want %v", argv, want)
	}
}

// TestBuildSSHArgvDefaults proves the defaulting rules: empty user becomes
// root, the default port omits the -p flag, and no key omits -i. The hardening
// block is still present — defaults never relax the discipline.
// fak-test:runtime fast est=1ms lane=default
func TestBuildSSHArgvDefaults(t *testing.T) {
	argv, err := BuildSSHArgv(SSHTarget{Host: "10.0.0.5"})
	if err != nil {
		t.Fatalf("err = %v, want nil", err)
	}
	if argv[0] != "ssh" {
		t.Errorf("argv[0] = %q, want ssh", argv[0])
	}
	if argv[len(argv)-1] != "root@10.0.0.5" {
		t.Errorf("target = %q, want root@10.0.0.5", argv[len(argv)-1])
	}
	for _, a := range argv {
		if a == "-p" || a == "-i" {
			t.Errorf("argv = %v, must omit port/key flags at defaults", argv)
		}
	}
	if !containsPair(argv, "-o", "BatchMode=yes") {
		t.Errorf("argv = %v, missing BatchMode hardening", argv)
	}
}

// TestBuildSSHArgvRefusesEmptyHost proves the fail-closed target check: no host
// is a typed refusal, not a `user@` that ssh resolves unpredictably.
// fak-test:runtime fast est=1ms lane=default
func TestBuildSSHArgvRefusesEmptyHost(t *testing.T) {
	if _, err := BuildSSHArgv(SSHTarget{Host: "   "}); !errors.Is(err, ErrEmptyHost) {
		t.Errorf("err = %v, want errors.Is(err, ErrEmptyHost)", err)
	}
}

// TestSSHArgvPasswordStance proves the secret never reaches argv: the explicit
// refusal is typed, and no builder path emits the password anywhere.
// fak-test:runtime fast est=1ms lane=default
func TestSSHArgvPasswordStance(t *testing.T) {
	const secret = "hunter2-should-never-appear"
	if err := RefusePasswordAuth(secret); !errors.Is(err, ErrPasswordAuthUnsupported) {
		t.Errorf("err = %v, want errors.Is(err, ErrPasswordAuthUnsupported)", err)
	} else if strings.Contains(err.Error(), secret) {
		t.Errorf("refusal error leaked the secret: %q", err.Error())
	}
	// The empty password is a no-op: no password was requested, nothing to refuse.
	if err := RefusePasswordAuth(""); err != nil {
		t.Errorf("err = %v, want nil for an empty password", err)
	}
	// Every emitted argv is secret-free by construction (Password is not a field).
	for _, argv := range [][]string{
		mustArgv(t, func() ([]string, error) { return BuildSSHArgv(SSHTarget{Host: "h"}) }),
		mustArgv(t, func() ([]string, error) { return BuildSCPArgv(SSHTarget{Host: "h"}, "a", "b") }),
	} {
		for _, a := range argv {
			if strings.Contains(a, secret) {
				t.Errorf("argv leaked the secret: %v", argv)
			}
		}
	}
}

// TestBuildSCPArgvShape proves scp's capital -P port flag and the shared
// hardening block; getting the case wrong is the classic scp/ssh confusion.
// fak-test:runtime fast est=1ms lane=default
func TestBuildSCPArgvShape(t *testing.T) {
	argv, err := BuildSCPArgv(SSHTarget{Host: "gpu-box-a", Port: 2200, Key: "/k"}, "/local/f", "root@gpu-box-a:/remote/f")
	if err != nil {
		t.Fatalf("err = %v, want nil", err)
	}
	want := []string{
		"scp",
		"-P", "2200",
		"-i", "/k",
		"-o", "BatchMode=yes",
		"-o", "StrictHostKeyChecking=accept-new",
		"-o", "UserKnownHostsFile=/dev/null",
		"-o", "LogLevel=ERROR",
		"-o", "ClearAllForwardings=yes",
		"-o", "PermitLocalCommand=no",
		"-o", "ProxyCommand=none",
		"/local/f", "root@gpu-box-a:/remote/f",
	}
	if !reflect.DeepEqual(argv, want) {
		t.Errorf("argv = %v, want %v", argv, want)
	}
}

// TestBuildSCPArgvRefusesEmptyHost proves the scp builder shares the same
// fail-closed target check as the ssh builder.
// fak-test:runtime fast est=1ms lane=default
func TestBuildSCPArgvRefusesEmptyHost(t *testing.T) {
	if _, err := BuildSCPArgv(SSHTarget{}, "a", "b"); !errors.Is(err, ErrEmptyHost) {
		t.Errorf("err = %v, want errors.Is(err, ErrEmptyHost)", err)
	}
}

// TestBuildSSHArgvExtraOptions proves caller options append after the hardening
// block (so a host-specific knob can never displace the hardening) and blank
// entries are skipped rather than producing a dangling -o.
// fak-test:runtime fast est=1ms lane=default
func TestBuildSSHArgvExtraOptions(t *testing.T) {
	argv, err := BuildSSHArgv(SSHTarget{Host: "h", ExtraOptions: []string{"ConnectTimeout=10", "  "}})
	if err != nil {
		t.Fatalf("err = %v, want nil", err)
	}
	if !containsPair(argv, "-o", "ConnectTimeout=10") {
		t.Errorf("argv = %v, missing extra option", argv)
	}
	// The hardening ProxyCommand=none must appear BEFORE the extra option block.
	hard := indexOfPair(argv, "-o", "ProxyCommand=none")
	extra := indexOfPair(argv, "-o", "ConnectTimeout=10")
	if hard < 0 || extra < 0 || hard > extra {
		t.Errorf("argv = %v, want hardening before extras (hard=%d extra=%d)", argv, hard, extra)
	}
	// No dangling -o: every -o is followed by a non-empty K=V.
	for i, a := range argv {
		if a == "-o" && (i+1 >= len(argv) || argv[i+1] == "") {
			t.Errorf("argv = %v has a dangling -o at %d", argv, i)
		}
	}
}

func containsPair(argv []string, key, val string) bool { return indexOfPair(argv, key, val) >= 0 }

func indexOfPair(argv []string, key, val string) int {
	for i := 0; i+1 < len(argv); i++ {
		if argv[i] == key && argv[i+1] == val {
			return i
		}
	}
	return -1
}

func mustArgv(t *testing.T, f func() ([]string, error)) []string {
	t.Helper()
	argv, err := f()
	if err != nil {
		t.Fatalf("unexpected err: %v", err)
	}
	return argv
}
