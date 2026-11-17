package main

// serve_bind_safety_hostname_test.go — #14448: the off-host bind-safety decision
// must RESOLVE hostnames before classifying, and the --allow-lan × --key-principal
// conjunction must be refused at boot. The hostname matrix drives an injected
// lookup, so nothing here consults a real resolver or opens a socket.

import (
	"bytes"
	"errors"
	"net"
	"strings"
	"testing"
)

// TestServeBindSafetyResolvesHostnameBeforeRefusal pins the resolution half of
// the admission rule: a hostname is classified by where it RESOLVES, not by the
// fact that it is not an IP literal. Any non-loopback resolution is off-host
// reachable; an unresolvable hostname fails closed; "localhost" and IP literals
// never consult the resolver.
func TestServeBindSafetyResolvesHostnameBeforeRefusal(t *testing.T) {
	ip := func(s string) net.IP { return net.ParseIP(s) }
	cases := []struct {
		name string
		addr string
		lk   func(string) ([]net.IP, error)
		want bool
	}{
		{"public hostname resolves off-host", "gw.internal.example:8080", func(string) ([]net.IP, error) { return []net.IP{ip("203.0.113.5")}, nil }, true},
		{"loopback-only hostname stays local", "gw.internal.example:8080", func(string) ([]net.IP, error) { return []net.IP{ip("127.0.0.1")}, nil }, false},
		{"mixed resolution binds off-host too", "gw.internal.example:8080", func(string) ([]net.IP, error) { return []net.IP{ip("127.0.0.1"), ip("10.1.2.3")}, nil }, true},
		{"unresolvable hostname fails closed", "gw.internal.example:8080", func(string) ([]net.IP, error) { return nil, errors.New("no such host") }, true},
		{"empty resolution fails closed", "gw.internal.example:8080", func(string) ([]net.IP, error) { return nil, nil }, true},
		{"localhost never consults the resolver", "localhost:8080", func(string) ([]net.IP, error) { return []net.IP{ip("203.0.113.5")}, nil }, false},
		{"loopback literal untouched", "127.0.0.1:8080", func(string) ([]net.IP, error) { return []net.IP{ip("203.0.113.5")}, nil }, false},
		{"wildcard untouched", "0.0.0.0:8080", func(string) ([]net.IP, error) { return []net.IP{ip("127.0.0.1")}, nil }, true},
	}
	for _, c := range cases {
		if got := serveBindOffHost(c.addr, c.lk); got != c.want {
			t.Errorf("%s: serveBindOffHost(%q) = %v, want %v", c.name, c.addr, got, c.want)
		}
	}
}

// TestServeBindSafetyRefusesAllowLANWithKeyPrincipal pins the boot refusal of
// the --allow-lan × --key-principal conjunction (#14448): the LAN exemption
// bypasses authentication on every route, so it cannot coexist with a tenant
// keyset. A loopback address keeps the case about the flags, not the bind.
func TestServeBindSafetyRefusesAllowLANWithKeyPrincipal(t *testing.T) {
	fs, sf := newServeFlagSet()
	if err := fs.Parse([]string{"--addr", "127.0.0.1:8080", "--allow-lan", "--key-principal", "acme=FAK_TEST_ACME_ENV"}); err != nil {
		t.Fatalf("parse: %v", err)
	}
	var stderr bytes.Buffer
	if admitServeBind(sf, &stderr) {
		t.Fatal("--allow-lan together with --key-principal was admitted")
	}
	if !strings.Contains(stderr.String(), serveLANKeysetRefusalToken) {
		t.Fatalf("refusal omitted %s: %s", serveLANKeysetRefusalToken, stderr.String())
	}
}

// TestServeBindSafetyAllowLanKeyPrincipalTokenIsDeclaredInTheClosedVocabulary
// binds the #14448 boot-refusal token to its dos.toml declaration, the same way
// TestServeBindRefusalTokenIsDeclaredInTheClosedVocabulary binds the off-host
// bind token: a structured refusal is only structured if the vocabulary knows
// the token.
func TestServeBindSafetyAllowLanKeyPrincipalTokenIsDeclaredInTheClosedVocabulary(t *testing.T) {
	body := repoDosToml(t)
	header := "[reasons." + serveLANKeysetRefusalToken + "]"
	i := strings.Index(body, header)
	if i < 0 {
		t.Fatalf("dos.toml declares no %s, so `dos man wedge %s --explain` answers UNCLASSIFIED: "+
			"fak would emit a refusal token its own closed vocabulary does not recognise",
			header, serveLANKeysetRefusalToken)
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
}
