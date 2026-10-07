package accounts

import (
	"context"
	"errors"
	"strings"
	"sync"
	"testing"
	"time"
)

// keychain_state_test.go — witnesses for fak-private#3063: the keychain probe tells a
// missing item, a timed-out read (an unanswered securityd ACL prompt), and any other
// failure apart, and a needs_login seat surfaces the distinct reason. Every test drives a
// FAKE runner or a stubbed seam; none ever execs the real `security` binary.

type fakeKeychainRun struct {
	res  keychainExecResult
	name string
	args []string
}

func (f *fakeKeychainRun) run(_ context.Context, name string, args ...string) keychainExecResult {
	f.name, f.args = name, append([]string(nil), args...)
	return f.res
}

// fak-test:runtime fast est=100ms
func TestSecurityReadPasswordClassifiesEachState(t *testing.T) {
	cases := []struct {
		name  string
		res   keychainExecResult
		state KeychainState
		value string
	}{
		{"found", keychainExecResult{Out: []byte("secret\n")}, KeychainFound, "secret\n"},
		{"missing exit 44", keychainExecResult{ExitCode: 44, Err: errors.New("exit status 44")}, KeychainMissing, ""},
		{"timeout acl prompt", keychainExecResult{TimedOut: true, ExitCode: -1, Err: errors.New("signal: killed")}, KeychainTimeout, ""},
		{"other exit", keychainExecResult{ExitCode: 51, Err: errors.New("exit status 51")}, KeychainError, ""},
		{"exec failure", keychainExecResult{Err: errors.New("fork/exec: no such file")}, KeychainError, ""},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			f := &fakeKeychainRun{res: tc.res}
			b, err := securityReadPassword(f.run, time.Second)("svc", "alice")
			state, _ := classifyKeychainErr(err)
			if state != tc.state {
				t.Fatalf("state=%q (err=%v), want %q", state, err, tc.state)
			}
			if string(b) != tc.value {
				t.Fatalf("value=%q, want %q", b, tc.value)
			}
			if f.name != "/usr/bin/security" {
				t.Fatalf("runner got %q, want the pinned /usr/bin/security", f.name)
			}
			if got := strings.Join(f.args, " "); got != "find-generic-password -a alice -w -s svc" {
				t.Fatalf("argv=%q", got)
			}
		})
	}

	// An empty account omits -a so a fak-owned item matches under any account.
	f := &fakeKeychainRun{res: keychainExecResult{Out: []byte("k")}}
	if _, err := securityReadPassword(f.run, time.Second)("svc", ""); err != nil {
		t.Fatal(err)
	}
	if got := strings.Join(f.args, " "); got != "find-generic-password -w -s svc" {
		t.Fatalf("empty-account argv=%q, want no -a", got)
	}
}

// stubKeychainErr routes the seam so each service answers a fixed value or error.
func stubKeychainErr(t *testing.T, values map[string]string, errs map[string]error) {
	t.Helper()
	prev := claudeKeychainReadPassword
	claudeKeychainReadPassword = func(service, _ string) ([]byte, error) {
		if v, ok := values[service]; ok {
			return []byte(v), nil
		}
		if err, ok := errs[service]; ok {
			return nil, err
		}
		return nil, ErrKeychainItemNotFound
	}
	resetClaudeKeychainCache()
	t.Cleanup(func() {
		claudeKeychainReadPassword = prev
		resetClaudeKeychainCache()
	})
}

// stubOSVersion pins the macOS-version seam (and resets its once-cache) for the test.
func stubOSVersion(t *testing.T, v string) {
	t.Helper()
	prev := keychainOSVersionFunc
	keychainOSVersionFunc = func() string { return v }
	keychainOSVersionOnce, keychainOSVersionVal = sync.Once{}, ""
	t.Cleanup(func() {
		keychainOSVersionFunc = prev
		keychainOSVersionOnce, keychainOSVersionVal = sync.Once{}, ""
	})
}

// fak-test:runtime fast est=100ms
func TestClaudeKeychainCredProbeStates(t *testing.T) {
	stubOSVersion(t, "15.1")
	dir := t.TempDir()
	service := keychainServiceFor(dir)

	cases := []struct {
		name  string
		vals  map[string]string
		errs  map[string]error
		state KeychainState
	}{
		{"missing", nil, nil, KeychainMissing},
		{"timeout", nil, map[string]error{service: ErrKeychainTimeout}, KeychainTimeout},
		{"error", nil, map[string]error{service: errors.New("security find-generic-password exited 51")}, KeychainError},
		{"no token", map[string]string{service: `{"claudeAiOauth":{}}`}, nil, KeychainNoToken},
		{"found", map[string]string{service: `{"claudeAiOauth":{"accessToken":"t"}}`}, nil, KeychainFound},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			stubKeychainErr(t, tc.vals, tc.errs)
			_, probe := ClaudeKeychainCredProbe(dir)
			if probe.State != tc.state {
				t.Fatalf("state=%q, want %q", probe.State, tc.state)
			}
			if tc.state != KeychainFound && !strings.Contains(probe.Detail, "macOS 15.1") {
				t.Fatalf("miss detail=%q, want the macOS version recorded", probe.Detail)
			}
			if _, ok := ClaudeKeychainCred(dir); ok != (tc.state == KeychainFound) {
				t.Fatalf("bool wrapper ok=%v disagrees with state %q", ok, tc.state)
			}
		})
	}

	// No seam at all is "unsupported", not "missing", and records no OS version.
	prev := claudeKeychainReadPassword
	claudeKeychainReadPassword = nil
	resetClaudeKeychainCache()
	t.Cleanup(func() { claudeKeychainReadPassword = prev; resetClaudeKeychainCache() })
	if _, probe := ClaudeKeychainCredProbe(dir); probe.State != KeychainUnsupported || probe.Detail != "" {
		t.Fatalf("nil seam: probe=%+v, want unsupported with no detail", probe)
	}
}

// fak-test:runtime fast est=100ms
func TestKeychainProbeReportsMostInformativeMiss(t *testing.T) {
	stubOSVersion(t, "")
	// A probe across several candidate services keeps the most informative miss: a
	// timeout on one (the item is likely there behind an ACL) beats missing on another.
	got := KeychainProbe{State: KeychainUnsupported}.
		worse(KeychainProbe{State: KeychainMissing, Service: "a"}).
		worse(KeychainProbe{State: KeychainTimeout, Service: "b"}).
		worse(KeychainProbe{State: KeychainMissing, Service: "c"})
	if got.State != KeychainTimeout || got.Service != "b" {
		t.Fatalf("aggregate=%+v, want the timeout on service b", got)
	}
}

// fak-test:runtime fast est=100ms
func TestNeedsLoginSurfacesDistinctKeychainReason(t *testing.T) {
	stubOSVersion(t, "14.6")
	cases := []struct {
		name      string
		err       error
		state     KeychainState
		reasonHas string
		actionHas string
	}{
		{"missing", nil, KeychainMissing, "keychain item is missing", "/login"},
		{"timeout", ErrKeychainTimeout, KeychainTimeout, "timed out", "Always Allow"},
		{"error", errors.New("security find-generic-password exited 51"), KeychainError, "keychain read failed", "unlock-keychain"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			dir := t.TempDir()
			errs := map[string]error{}
			if tc.err != nil {
				errs[keychainServiceFor(dir)] = tc.err
			}
			stubKeychainErr(t, nil, errs)

			id := DeriveIdentity(dir)
			if id.HasCreds {
				t.Fatal("HasCreds=true with no credential anywhere")
			}
			if id.Keychain.State != tc.state {
				t.Fatalf("identity keychain state=%q, want %q", id.Keychain.State, tc.state)
			}
			report := (Registry{Homes: []Home{{Name: "s", Dir: dir, Identity: id}}}).LoginReport()
			seat := report.Seats[0]
			// Status and servability are preserved: still needs_login, still not servable.
			if seat.Status != LoginNeedsLogin || seat.CanServe {
				t.Fatalf("status=%q can_serve=%v, want needs_login false", seat.Status, seat.CanServe)
			}
			if seat.KeychainState != tc.state {
				t.Fatalf("observation keychain_state=%q, want %q", seat.KeychainState, tc.state)
			}
			if !strings.Contains(seat.Reason, tc.reasonHas) || !strings.Contains(seat.Reason, "macOS 14.6") {
				t.Fatalf("reason=%q, want %q and the macOS version", seat.Reason, tc.reasonHas)
			}
			if !strings.Contains(seat.NextAction, tc.actionHas) {
				t.Fatalf("next_action=%q, want %q", seat.NextAction, tc.actionHas)
			}
		})
	}
}

// fak-test:runtime fast est=100ms
func TestKeychainSecretReadsFakOwnedItem(t *testing.T) {
	stubOSVersion(t, "")
	gotAccount := "unset"
	prev := claudeKeychainReadPassword
	claudeKeychainReadPassword = func(service, account string) ([]byte, error) {
		gotAccount = account
		if service == "fak-ANTHROPIC_API_KEY" {
			return []byte("sk-ant-api03-durable\n"), nil
		}
		return nil, ErrKeychainItemNotFound
	}
	resetClaudeKeychainCache()
	t.Cleanup(func() { claudeKeychainReadPassword = prev; resetClaudeKeychainCache() })

	key, probe := KeychainSecret("fak-ANTHROPIC_API_KEY")
	if !probe.OK() || key != "sk-ant-api03-durable" {
		t.Fatalf("got (%q,%+v), want the trimmed durable key", key, probe)
	}
	if gotAccount != "" {
		t.Fatalf("fak-owned item read with account %q, want any-account (empty)", gotAccount)
	}
	if _, probe := KeychainSecret("fak-OTHER"); probe.State != KeychainMissing {
		t.Fatalf("absent item: state=%q, want missing", probe.State)
	}
}

// fak-test:runtime fast est=100ms
func TestSwVersProductVersionUsesRunner(t *testing.T) {
	f := &fakeKeychainRun{res: keychainExecResult{Out: []byte("15.1.1\n")}}
	if v := swVersProductVersion(f.run); v != "15.1.1" {
		t.Fatalf("version=%q, want 15.1.1", v)
	}
	if f.name != "/usr/bin/sw_vers" || strings.Join(f.args, " ") != "-productVersion" {
		t.Fatalf("runner got %q %v", f.name, f.args)
	}
	f.res = keychainExecResult{Err: errors.New("boom")}
	if v := swVersProductVersion(f.run); v != "" {
		t.Fatalf("failed run: version=%q, want empty", v)
	}
}
