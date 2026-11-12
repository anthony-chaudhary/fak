package servelaunch

import (
	"errors"
	"fmt"
	"strconv"
	"strings"
)

// Remote transport is the part of the launch lifecycle that runs the server on
// a host you do not sit in front of: build the ssh/scp argv, drive the remote
// shell, and keep the connection discipline that keeps automation from turning
// into an interactive prompt. This file is the argv half — a pure builder with
// no exec, no network, and no I/O.
//
// The hardening below is the portable lesson of the pinned study's transport
// (`server/_ssh.py`): automation must run with BatchMode (a missing key fails
// fast instead of blocking on a password prompt nobody can answer), must not
// mutate the host-key store on a fresh host (`StrictHostKeyChecking=no` plus a
// non-persistent known-hosts file is the harness default the pinned transport
// used for first-connect to a fresh DGX), and must never hand a secret to an
// argv that another local user can read out of `ps`.

var (
	// ErrEmptyHost reports a connection target with no host; a builder that
	// cannot name where it is connecting refuses rather than emitting a
	// `user@` that ssh resolves unpredictably.
	ErrEmptyHost = errors.New("servelaunch: ssh host is required")
	// ErrPasswordAuthUnsupported reports that the caller asked for password
	// auth. The pinned study's transport shipped this via `sshpass -p`, which
	// places the secret in argv where any process listing exposes it; this
	// builder refuses that stance and names the fix (provision a key) instead
	// of silently leaking the secret.
	ErrPasswordAuthUnsupported = errors.New("servelaunch: password auth refused — the secret would be visible in argv; provision an ssh key instead")
)

// DefaultSSHUser is the user the remote launch connects as when the caller
// names none: the pinned transport's root@ default for fresh GPU boxes.
const DefaultSSHUser = "root"

// DefaultSSHPort is the port ssh assumes when the caller names none.
const DefaultSSHPort = 22

// SSHTarget names the remote endpoint and the credentials a launch uses.
// Password is deliberately NOT a field: see ErrPasswordAuthUnsupported.
type SSHTarget struct {
	// Host is the IP or hostname ssh connects to. Required.
	Host string
	// User is the login user; empty takes DefaultSSHUser.
	User string
	// Port is the ssh port; <= 0 or DefaultSSHPort omits the port flag.
	Port int
	// Key is a private-key path for -i; empty omits the flag and lets ssh
	// pick from its own defaults (agent, ~/.ssh/config).
	Key string
	// ExtraOptions appends caller-supplied `-o Key=Value` pairs after the
	// hardening block. Reserved for the rare host that needs one more knob;
	// the hardening options are not overridable.
	ExtraOptions []string
}

// userHost renders the `user@host` target word.
func (t SSHTarget) userHost() (string, error) {
	host := strings.TrimSpace(t.Host)
	if host == "" {
		return "", ErrEmptyHost
	}
	user := strings.TrimSpace(t.User)
	if user == "" {
		user = DefaultSSHUser
	}
	return user + "@" + host, nil
}

// hardeningOptions is the fixed, ordered connection-options block every argv
// carries. It is the discipline, not a suggestion: BatchMode fails fast rather
// than blocking on an unanswerable prompt, host keys are checked but not
// persisted (fresh-host first-connect is expected), and every forwarding or
// local-command escape hatch ssh offers is closed so a launch cannot be
// silently redirected.
func hardeningOptions() []string {
	return []string{
		"-o", "BatchMode=yes",
		"-o", "StrictHostKeyChecking=accept-new",
		"-o", "UserKnownHostsFile=/dev/null",
		"-o", "LogLevel=ERROR",
		"-o", "ClearAllForwardings=yes",
		"-o", "PermitLocalCommand=no",
		"-o", "ProxyCommand=none",
	}
}

// BuildSSHArgv builds an argv prefix ending in `user@host`; the caller appends
// the remote command. Shape:
//
//	ssh [-p PORT] [-i KEY] -o K=V ... user@host
//
// The port flag is omitted at the default port, the key flag is omitted when no
// key is named, and the hardening block is always present in a fixed order so a
// receipt that records the argv is reproducible. An empty host is refused with
// ErrEmptyHost. There is no password stance: any password-auth request is
// refused with ErrPasswordAuthUnsupported (see the sentinel) and the returned
// argv is nil, so a caller cannot accidentally run the leaking form.
func BuildSSHArgv(t SSHTarget) ([]string, error) {
	target, err := t.userHost()
	if err != nil {
		return nil, err
	}
	argv := []string{"ssh"}
	argv = append(argv, portFlag(t.Port)...)
	if key := strings.TrimSpace(t.Key); key != "" {
		argv = append(argv, "-i", key)
	}
	argv = append(argv, hardeningOptions()...)
	argv = append(argv, extraOptionFlags(t.ExtraOptions)...)
	argv = append(argv, target)
	return argv, nil
}

// BuildSCPArgv builds a full scp argv moving src to dst. Either may be a local
// path or `user@host:path`. scp spells its port flag `-P` (capital), unlike
// ssh's `-p`; that difference is modelled here rather than guessed by callers.
// The same hardening block and the same refusals as BuildSSHArgv apply.
func BuildSCPArgv(t SSHTarget, src, dst string) ([]string, error) {
	if _, err := t.userHost(); err != nil {
		return nil, err
	}
	argv := []string{"scp"}
	if port := normalizedPort(t.Port); port != DefaultSSHPort {
		argv = append(argv, "-P", strconv.Itoa(port))
	}
	if key := strings.TrimSpace(t.Key); key != "" {
		argv = append(argv, "-i", key)
	}
	argv = append(argv, hardeningOptions()...)
	argv = append(argv, extraOptionFlags(t.ExtraOptions)...)
	argv = append(argv, src, dst)
	return argv, nil
}

// RefusePasswordAuth is the explicit password stance: it always returns the
// ErrPasswordAuthUnsupported sentinel and never a secret-bearing argv. Callers
// that inherited a password from a legacy config route through it to get the
// typed refusal and the provisioning hint.
func RefusePasswordAuth(password string) error {
	if password == "" {
		return nil
	}
	return fmt.Errorf("%w", ErrPasswordAuthUnsupported)
}

// normalizedPort maps an unset/zero port to the default.
func normalizedPort(port int) int {
	if port <= 0 {
		return DefaultSSHPort
	}
	return port
}

// portFlag returns ssh's `-p PORT` pair, omitted at the default port.
func portFlag(port int) []string {
	p := normalizedPort(port)
	if p == DefaultSSHPort {
		return nil
	}
	return []string{"-p", strconv.Itoa(p)}
}

// extraOptionFlags renders caller options as `-o Key=Value` pairs, skipping
// blanks so a trailing empty entry never produces a dangling `-o`.
func extraOptionFlags(opts []string) []string {
	var out []string
	for _, o := range opts {
		if s := strings.TrimSpace(o); s != "" {
			out = append(out, "-o", s)
		}
	}
	return out
}
