package main

import (
	"bytes"
	"strings"
	"testing"
)

// fak-test:runtime fast est=5ms lane=default
// Estimate only: pure string classification, no sockets.

// TestL3ServeBindRefusal is the issue #14345 regression witness: fak l3-serve
// refuses an off-host bind with no auth door configured, admits loopback, and
// admits off-host once a door is configured or the operator overrides.
func TestL3ServeBindRefusal(t *testing.T) {
	t.Parallel()
	cases := []struct {
		name           string
		addr           string
		authConfigured bool
		override       bool
		wantRefusal    bool
	}{
		{"loopback_default", "127.0.0.1:18000", false, false, false},
		{"localhost_name", "localhost:18000", false, false, false},
		{"offhost_with_auth_secret", "0.0.0.0:18000", true, false, false},
		{"offhost_no_auth", "0.0.0.0:18000", false, false, true},
		{"offhost_override", "0.0.0.0:18000", false, true, false},
		{"routable_ip_no_auth", "192.168.1.10:18000", false, false, true},
		{"ipv6_wildcard_no_auth", "[::]:18000", false, false, true},
		{"unproven_dns_name", "caches.example:18000", false, false, false},
	}
	for _, tc := range cases {
		refusal := l3ServeBindRefusal(tc.addr, tc.authConfigured, tc.override)
		if (refusal != "") != tc.wantRefusal {
			t.Errorf("%s: refusal=%q wantRefusal=%v", tc.name, refusal, tc.wantRefusal)
		}
		if refusal != "" && !strings.Contains(refusal, l3BindRefusalToken) {
			t.Errorf("%s: refusal missing stable token: %q", tc.name, refusal)
		}
	}
}

// TestAdmitL3ServeBind covers the boot gate: the refusal exits, the override
// warns loudly, and clean binds print nothing.
func TestAdmitL3ServeBind(t *testing.T) {
	t.Parallel()
	var stderr bytes.Buffer
	if !admitL3ServeBind([]string{"127.0.0.1:18000"}, false, false, &stderr) {
		t.Fatal("loopback bind refused")
	}
	if stderr.Len() != 0 {
		t.Fatalf("clean bind wrote to stderr: %q", stderr.String())
	}

	stderr.Reset()
	if admitL3ServeBind([]string{"0.0.0.0:18000"}, false, false, &stderr) {
		t.Fatal("off-host unauthenticated bind admitted")
	}
	if !strings.Contains(stderr.String(), l3BindRefusalToken) {
		t.Fatalf("refusal missing stable token: %q", stderr.String())
	}

	stderr.Reset()
	if !admitL3ServeBind([]string{"0.0.0.0:18000"}, false, true, &stderr) {
		t.Fatal("explicit override refused")
	}
	if !strings.Contains(stderr.String(), "WARNING") || !strings.Contains(stderr.String(), l3UnsafeBindFlag) {
		t.Fatalf("override did not warn loudly: %q", stderr.String())
	}
}

// TestResolveSecretFlag covers the literal/env credential resolution: exactly
// one source, trimmed values, and a named-but-unset env var is fatal.
func TestResolveSecretFlag(t *testing.T) {
	// No t.Parallel: t.Setenv is incompatible with parallel tests.
	v, err := resolveSecretFlag("auth-secret", "  literal-secret  ", "")
	if err != nil || v != "literal-secret" {
		t.Fatalf("literal: %q, %v", v, err)
	}

	t.Setenv("FAK_TEST_L3_SECRET", " env-secret ")
	v, err = resolveSecretFlag("auth-secret", "", "FAK_TEST_L3_SECRET")
	if err != nil || v != "env-secret" {
		t.Fatalf("env: %q, %v", v, err)
	}

	if _, err := resolveSecretFlag("auth-secret", "", "FAK_TEST_L3_SECRET_UNSET"); err == nil {
		t.Fatal("named-but-unset env var not refused")
	}

	if _, err := resolveSecretFlag("auth-secret", "lit", "FAK_TEST_L3_SECRET"); err == nil {
		t.Fatal("both literal and env name not refused")
	}
}
