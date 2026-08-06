package main

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// serve_bind_safety_test.go — the #5373 matrix. Every case drives the PURE admission
// helpers against bind strings as data; nothing here opens a socket, so the suite never
// listens on a reachable interface (which on Windows pops a firewall prompt and can hang
// a headless run). The one case that needs the real flag surface parses it with
// newServeFlagSet and still binds nothing.
//
// All non-loopback addresses below are drawn from the documentation ranges
// (192.0.2.0/24, 198.51.100.0/24, RFC 3849's 2001:db8::/32) so no host's real address
// is ever written into the tree.

// TestServeBindReachesOffHost pins the reachability half of the conjunction: which
// listen addresses are provably reachable by someone other than this host, and which
// ambiguous ones are deliberately NOT (a refusal must never fire on a parse failure).
func TestServeBindReachesOffHost(t *testing.T) {
	cases := []struct {
		addr string
		want bool
		why  string
	}{
		// This host only.
		{"127.0.0.1:8080", false, "the shipped --addr default"},
		{"127.0.0.1:0", false, "the port-0 form every test binds"},
		{"127.0.0.1", false, "loopback with no port at all"},
		{"127.3.2.1:8080", false, "all of 127.0.0.0/8 is loopback"},
		{"localhost:8080", false, "the loopback name"},
		{"LocalHost:8080", false, "the loopback name is not case sensitive"},
		{"[::1]:8080", false, "IPv6 loopback, bracketed"},
		{"::1", false, "IPv6 loopback with no port"},
		{"", false, "no address at all — run() refuses that separately"},
		{"   ", false, "whitespace-only is the same as no address"},

		// Reachable by someone who is not this host.
		{"0.0.0.0:8080", true, "the wildcard the issue names"},
		{"0.0.0.0:0", true, "wildcard on an OS-picked port is still every interface"},
		{"[::]:8080", true, "the IPv6 wildcard"},
		{"::", true, "the IPv6 wildcard with no port"},
		{":8080", true, "a bare port: net.Listen binds ALL interfaces"},
		{":0", true, "a bare OS-picked port is still all interfaces"},
		{"192.0.2.10:8080", true, "a specific routable IPv4"},
		{"198.51.100.7", true, "a specific routable IPv4 with no port"},
		{"[2001:db8::1]:8080", true, "a specific routable IPv6"},
		{"[fe80::1]:8080", true, "link-local is reachable from the local link"},
		{"[fe80::1%eth0]:8080", true, "a zoned link-local literal still classifies"},

		// Not provable, so not refused.
		{"unresolved-host.example:8080", false, "a DNS name cannot be proven off-host at bind time"},
		{"127.0.0.1.evil.example:8080", false, "a loopback LOOKALIKE is not a parseable IP"},
		{"not an address", false, "garbage is refused by net.Listen, not by this rule"},
	}
	for _, c := range cases {
		if got := serveBindReachesOffHost(c.addr); got != c.want {
			t.Errorf("serveBindReachesOffHost(%q) = %v, want %v — %s", c.addr, got, c.want, c.why)
		}
	}
}

// TestServeBindRefusalMatrix walks the full cross product the issue specifies: the
// refusal fires on the CONJUNCTION (reachable off-host AND no token door) and on
// nothing else. The allow rows are the expensive half to get wrong — refusing any of
// them would break every local dev run in the tree.
func TestServeBindRefusalMatrix(t *testing.T) {
	offHost := []string{"0.0.0.0:8080", "[::]:8080", ":8080", "192.0.2.10:8080", "[2001:db8::1]:8080"}
	onHost := []string{"127.0.0.1:8080", "127.0.0.1:0", "localhost:8080", "[::1]:8080", ""}
	unproven := []string{"unresolved-host.example:8080", "127.0.0.1.evil.example:8080"}

	// Loopback binds are admitted with AND without a token door — the row that keeps
	// local development and the test suite working.
	for _, addr := range onHost {
		for _, authed := range []bool{false, true} {
			if got := serveBindRefusal(addr, authed, false); got != "" {
				t.Errorf("serveBindRefusal(%q, auth=%v) refused a loopback bind: %s", addr, authed, got)
			}
		}
	}

	// An off-host bind WITH a token door is admitted: the operator asked for exposure
	// and armed authentication, which is the supported production shape.
	for _, addr := range offHost {
		if got := serveBindRefusal(addr, true, false); got != "" {
			t.Errorf("serveBindRefusal(%q, auth=true) refused an authenticated off-host bind: %s", addr, got)
		}
	}

	// An off-host bind with NO token door is the one refusal, and it names its reason.
	for _, addr := range offHost {
		got := serveBindRefusal(addr, false, false)
		if got == "" {
			t.Fatalf("serveBindRefusal(%q, auth=false) admitted an UNAUTHENTICATED off-host bind — this is the whole issue", addr)
		}
		if !strings.Contains(got, serveBindRefusalToken) {
			t.Errorf("refusal for %q must name the %s reason, got: %s", addr, serveBindRefusalToken, got)
		}
		if !strings.Contains(got, addr) {
			t.Errorf("refusal for %q must quote the address the operator passed, got: %s", addr, got)
		}
		if !strings.Contains(got, serveUnsafeBindFlag) {
			t.Errorf("refusal for %q must name the --%s escape, got: %s", addr, serveUnsafeBindFlag, got)
		}
		for _, fix := range []string{"--require-key-env", "--key-principal", "--addr 127.0.0.1:8080"} {
			if !strings.Contains(got, fix) {
				t.Errorf("refusal for %q must offer %s as a fix, got: %s", addr, fix, got)
			}
		}
	}

	// The explicit operator escape suppresses exactly that refusal.
	for _, addr := range offHost {
		if got := serveBindRefusal(addr, false, true); got != "" {
			t.Errorf("serveBindRefusal(%q, auth=false, override=true) must be admitted, got: %s", addr, got)
		}
	}

	// An address whose interface cannot be determined is admitted either way: a
	// malformed-but-harmless --addr must not wedge startup.
	for _, addr := range unproven {
		for _, authed := range []bool{false, true} {
			if got := serveBindRefusal(addr, authed, false); got != "" {
				t.Errorf("serveBindRefusal(%q, auth=%v) refused on a parse failure alone: %s", addr, authed, got)
			}
		}
	}
}

// TestServeAuthConfiguredReadsBothDoors pins what "a token door is configured" means
// against the REAL flag surface: the two inbound-auth flags the gateway's withAuth
// condition (requireKey != "" || keyset != nil) is fed from, and nothing else. An
// outbound upstream credential is not an inbound door.
func TestServeAuthConfiguredReadsBothDoors(t *testing.T) {
	cases := []struct {
		name string
		argv []string
		want bool
	}{
		{"no auth flags at all", nil, false},
		{"the single bearer door", []string{"-require-key-env", "FAK_TEST_BEARER_ENV"}, true},
		{"one tenant key door", []string{"-key-principal", "acme=FAK_TEST_ACME_ENV"}, true},
		{"several tenant key doors", []string{"-key-principal", "acme=FAK_TEST_ACME_ENV", "-key-principal", "beta=FAK_TEST_BETA_ENV"}, true},
		{"both doors together", []string{"-require-key-env", "FAK_TEST_BEARER_ENV", "-key-principal", "acme=FAK_TEST_ACME_ENV"}, true},
		{"an empty bearer flag value is no door", []string{"-require-key-env", "   "}, false},
		{"an upstream key is not an inbound door", []string{"-api-key-env", "FAK_TEST_UPSTREAM_ENV"}, false},
		{"a cache admin key is not an inbound door", []string{"-engine-cache-admin-key-env", "FAK_TEST_ADMIN_ENV"}, false},
	}
	for _, c := range cases {
		fs, sf := newServeFlagSet()
		if err := fs.Parse(c.argv); err != nil {
			t.Fatalf("%s: parse %v: %v", c.name, c.argv, err)
		}
		if got := serveAuthConfigured(sf); got != c.want {
			t.Errorf("%s: serveAuthConfigured(%v) = %v, want %v", c.name, c.argv, got, c.want)
		}
	}
}

// TestAdmitServeBindOnRealFlagSurface drives the admission helper end to end through the
// parsed `fak serve` flags — proving the new flag is actually REGISTERED and read, not
// just declared. It binds no socket: admitServeBind only reads flag values.
func TestAdmitServeBindOnRealFlagSurface(t *testing.T) {
	cases := []struct {
		name      string
		argv      []string
		wantAdmit bool
		wantErr   string // a substring stderr must carry ("" = stderr must stay empty)
	}{
		{"the shipped default binds", nil, true, ""},
		{"explicit loopback with no auth binds", []string{"-addr", "127.0.0.1:9000"}, true, ""},
		{"stdio has no listener to expose", []string{"-stdio", "-addr", "0.0.0.0:8080"}, true, ""},
		{"auth-less wildcard is refused", []string{"-addr", "0.0.0.0:8080"}, false, serveBindRefusalToken},
		{"auth-less bare port is refused", []string{"-addr", ":8080"}, false, serveBindRefusalToken},
		{"auth-less routable address is refused", []string{"-addr", "192.0.2.10:8080"}, false, serveBindRefusalToken},
		{"a bearer door admits the wildcard", []string{"-addr", "0.0.0.0:8080", "-require-key-env", "FAK_TEST_BEARER_ENV"}, true, ""},
		{"a tenant key door admits the wildcard", []string{"-addr", "0.0.0.0:8080", "-key-principal", "acme=FAK_TEST_ACME_ENV"}, true, ""},
		{"the escape admits the wildcard, loudly", []string{"-addr", "0.0.0.0:8080", "-" + serveUnsafeBindFlag}, true, "WARNING"},
		{"the escape is silent when it changes nothing", []string{"-addr", "127.0.0.1:8080", "-" + serveUnsafeBindFlag}, true, ""},
	}
	for _, c := range cases {
		fs, sf := newServeFlagSet()
		if err := fs.Parse(c.argv); err != nil {
			t.Fatalf("%s: parse %v: %v", c.name, c.argv, err)
		}
		var stderr bytes.Buffer
		got := admitServeBind(sf, &stderr)
		if got != c.wantAdmit {
			t.Errorf("%s: admitServeBind(%v) = %v, want %v (stderr: %s)", c.name, c.argv, got, c.wantAdmit, stderr.String())
		}
		switch {
		case c.wantErr == "" && stderr.Len() != 0:
			t.Errorf("%s: admitServeBind(%v) must stay quiet, wrote: %s", c.name, c.argv, stderr.String())
		case c.wantErr != "" && !strings.Contains(stderr.String(), c.wantErr):
			t.Errorf("%s: admitServeBind(%v) stderr must contain %q, got: %s", c.name, c.argv, c.wantErr, stderr.String())
		}
	}
}

// repoDosToml reads the workspace refusal vocabulary, walking up from the package dir so the
// test does not care how deep cmd/fak sits.
func repoDosToml(t *testing.T) string {
	t.Helper()
	dir, err := filepath.Abs(".")
	if err != nil {
		t.Fatalf("abs: %v", err)
	}
	for {
		p := filepath.Join(dir, "dos.toml")
		if b, err := os.ReadFile(p); err == nil {
			return string(b)
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			t.Skip("no dos.toml above the package dir (detached build tree) — nothing to bind against")
		}
		dir = parent
	}
}

// A structured refusal is only structured if the vocabulary KNOWS the token. This gate shipped
// with its matrix well before UNAUTHENTICATED_OFF_HOST_BIND was declared in dos.toml, so
// `dos check-reason` classified fak's own security refusal as UNCLASSIFIED prose drift — the
// exact failure a closed vocabulary exists to kill, on the one refusal where a reader most needs
// a classification and a fix. This binds the emitted token to its declaration (#5373).
func TestServeBindRefusalTokenIsDeclaredInTheClosedVocabulary(t *testing.T) {
	body := repoDosToml(t)
	header := "[reasons." + serveBindRefusalToken + "]"
	i := strings.Index(body, header)
	if i < 0 {
		t.Fatalf("dos.toml declares no %s, so `dos check-reason %s` answers UNCLASSIFIED: "+
			"fak would emit a refusal token its own closed vocabulary does not recognise",
			header, serveBindRefusalToken)
	}
	block := body[i+len(header):]
	if j := strings.Index(block, "\n["); j >= 0 {
		block = block[:j]
	}
	for _, want := range []string{"OPERATOR_GATE", "refusal", "summary", "fix", "see_also"} {
		if !strings.Contains(block, want) {
			t.Errorf("dos.toml %s is missing %q — an operator who hits this refusal gets no classification or next step", header, want)
		}
	}
	// The declared fix has to name the same doors and escape the in-band refusal names, or the
	// vocabulary and the gate would tell an operator two different things.
	for _, want := range []string{"--require-key-env", "--key-principal", serveUnsafeBindFlag} {
		if !strings.Contains(block, want) {
			t.Errorf("dos.toml %s must name %q, the way the in-band refusal does", header, want)
		}
	}
}
