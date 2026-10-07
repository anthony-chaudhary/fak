package main

import (
	"bytes"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/accounts"
)

// serve_upstream_key_test.go — witnesses for fak-private#3063's durable gateway key: an
// empty --api-key-env variable falls back to the fak-owned Keychain item, then Claude
// Code's saved key, and a gateway with no key anywhere fails fast. Every source is a
// fake; nothing reads the process env or execs `security`.

type fakeUpstreamKeySources struct {
	env       map[string]string
	items     map[string]string
	itemState accounts.KeychainState // state for an absent item (default missing)
	claude    string
	claudeSt  accounts.KeychainState
	itemCalls int
}

func (f *fakeUpstreamKeySources) sources() serveUpstreamKeySources {
	return serveUpstreamKeySources{
		getenv: func(k string) string { return f.env[k] },
		keychainItem: func(service string) (string, accounts.KeychainProbe) {
			f.itemCalls++
			if v, ok := f.items[service]; ok {
				return v, accounts.KeychainProbe{State: accounts.KeychainFound, Service: service}
			}
			st := f.itemState
			if st == "" {
				st = accounts.KeychainMissing
			}
			return "", accounts.KeychainProbe{State: st, Service: service}
		},
		claudeAPIKey: func(string) (string, accounts.KeychainProbe) {
			if f.claude != "" {
				return f.claude, accounts.KeychainProbe{State: accounts.KeychainFound}
			}
			st := f.claudeSt
			if st == "" {
				st = accounts.KeychainMissing
			}
			return "", accounts.KeychainProbe{State: st, Service: "Claude Code"}
		},
		claudeHome: func() string { return "/Users/op/.claude" },
	}
}

// fak-test:runtime fast est=100ms
func TestResolveServeUpstreamKeyOrder(t *testing.T) {
	const env = "ANTHROPIC_API_KEY"

	// 1. The env var wins and the durable store is never consulted.
	f := &fakeUpstreamKeySources{env: map[string]string{env: "sk-env"}, items: map[string]string{"fak-" + env: "sk-kc"}}
	if key, src, _ := resolveServeUpstreamKey(env, true, f.sources()); key != "sk-env" || src != "env $"+env || f.itemCalls != 0 {
		t.Fatalf("env set: got (%q,%q) itemCalls=%d", key, src, f.itemCalls)
	}

	// 2. Empty env (the post-reboot case) falls back to the fak-owned Keychain item.
	f = &fakeUpstreamKeySources{items: map[string]string{"fak-" + env: "sk-kc"}, claude: "sk-claude"}
	if key, src, _ := resolveServeUpstreamKey(env, true, f.sources()); key != "sk-kc" || !strings.Contains(src, "fak-"+env) {
		t.Fatalf("durable item: got (%q,%q)", key, src)
	}

	// 3. No fak item: Claude Code's saved API key (ANTHROPIC_API_KEY only).
	f = &fakeUpstreamKeySources{claude: "sk-claude"}
	if key, src, _ := resolveServeUpstreamKey(env, true, f.sources()); key != "sk-claude" || !strings.Contains(src, "Claude Code") {
		t.Fatalf("claude saved key: got (%q,%q)", key, src)
	}
	if key, _, _ := resolveServeUpstreamKey("OTHER_KEY", true, f.sources()); key != "" {
		t.Fatalf("Claude Code's key must only stand in for ANTHROPIC_API_KEY, got %q", key)
	}

	// 4. Nothing anywhere: no key, and each miss is named distinctly.
	f = &fakeUpstreamKeySources{itemState: accounts.KeychainTimeout, claudeSt: accounts.KeychainMissing}
	key, _, misses := resolveServeUpstreamKey(env, true, f.sources())
	if key != "" {
		t.Fatalf("no source: got key %q", key)
	}
	joined := strings.Join(misses, " | ")
	for _, want := range []string{"$" + env + " is empty", "timed out", "Claude Code saved API key", "missing"} {
		if !strings.Contains(joined, want) {
			t.Fatalf("misses %q lack %q", joined, want)
		}
	}

	// Without --require-upstream-key the durable store is never consulted (prior behavior).
	f = &fakeUpstreamKeySources{items: map[string]string{"fak-" + env: "sk-kc"}}
	if key, _, _ := resolveServeUpstreamKey(env, false, f.sources()); key != "" || f.itemCalls != 0 {
		t.Fatalf("durable=false: got %q itemCalls=%d, want env-only", key, f.itemCalls)
	}

	// No keychain on this platform: one honest line, no Claude Code probe.
	f = &fakeUpstreamKeySources{itemState: accounts.KeychainUnsupported, claude: "sk-claude"}
	if key, _, misses := resolveServeUpstreamKey(env, true, f.sources()); key != "" || !strings.Contains(strings.Join(misses, ";"), "no macOS login Keychain") {
		t.Fatalf("unsupported: got %q misses=%v", key, misses)
	}
}

// fak-test:runtime fast est=100ms
func TestServeNoGatewayKeyBailNamesDurableFix(t *testing.T) {
	var buf bytes.Buffer
	writeConfigBail(&buf, serveNoGatewayKeyBail("ANTHROPIC_API_KEY", []string{"$ANTHROPIC_API_KEY is empty", "keychain item is missing"}))
	out := buf.String()
	for _, want := range []string{"no gateway key", "keychain item is missing", "security add-generic-password -U -a \"$USER\" -s fak-ANTHROPIC_API_KEY -w", "KEY_ENV_UNSET"} {
		if !strings.Contains(out, want) {
			t.Fatalf("bail output lacks %q:\n%s", want, out)
		}
	}
	if strings.Contains(out, "launchctl setenv") {
		t.Fatalf("bail must not recommend the reboot-volatile launchctl setenv:\n%s", out)
	}
}

// fak-test:runtime fast est=100ms
func TestNodeDarwinUpstreamKeyGuidance(t *testing.T) {
	// Durable fak item resolves: arm fail-fast, name the source, never the value.
	f := &fakeUpstreamKeySources{env: map[string]string{"ANTHROPIC_API_KEY": "sk-shell"}, items: map[string]string{"fak-ANTHROPIC_API_KEY": "sk-secret-never-printed"}}
	var out, errb bytes.Buffer
	if !nodeDarwinUpstreamKeyGuidance(&out, &errb, true, f.sources()) {
		t.Fatal("durable fak item: want fail-fast armed")
	}
	if !strings.Contains(out.String(), "fak-ANTHROPIC_API_KEY") || errb.Len() != 0 {
		t.Fatalf("found: stdout=%q stderr=%q", out.String(), errb.String())
	}
	if strings.Contains(out.String()+errb.String(), "sk-secret") {
		t.Fatal("guidance must never print the key value")
	}

	// Claude Code's saved key also counts as durable.
	f = &fakeUpstreamKeySources{claude: "sk-claude"}
	out.Reset()
	errb.Reset()
	if !nodeDarwinUpstreamKeyGuidance(&out, &errb, false, f.sources()) || !strings.Contains(out.String(), "Claude Code") {
		t.Fatalf("claude saved key: want armed, stdout=%q", out.String())
	}

	// Nothing durable — even with the key in the installer shell, which a reboot
	// wipes: keep passthrough and tell the operator to store the item and re-run.
	for _, envSet := range []bool{true, false} {
		f = &fakeUpstreamKeySources{env: map[string]string{"ANTHROPIC_API_KEY": "sk-shell"}}
		out.Reset()
		errb.Reset()
		if nodeDarwinUpstreamKeyGuidance(&out, &errb, envSet, f.sources()) {
			t.Fatalf("envSet=%v: no durable key must NOT arm fail-fast (would crash-loop a passthrough install)", envSet)
		}
		msg := errb.String()
		for _, want := range []string{"passthrough", "security add-generic-password -U -a \"$USER\" -s fak-ANTHROPIC_API_KEY -w", "re-run `fak node install`"} {
			if !strings.Contains(msg, want) {
				t.Fatalf("envSet=%v: guidance lacks %q:\n%s", envSet, want, msg)
			}
		}
		if strings.Contains(msg, "launchctl setenv ANTHROPIC_API_KEY \"sk-ant") {
			t.Fatalf("envSet=%v: guidance still recommends launchctl setenv:\n%s", envSet, msg)
		}
	}
}

// fak-test:runtime fast est=100ms
func TestNodeDarwinPlistReadsUpstreamKeyDurably(t *testing.T) {
	d := nodeUnitData{Label: "l", WrapperPath: "/w", FakBin: "/f", Addr: "127.0.0.1:8080", PolicyPath: "/p", LogDir: "/l",
		Provider: "anthropic", BaseURL: "https://api.anthropic.com", UpstreamKeyEnv: "ANTHROPIC_API_KEY"}
	plist, err := nodeRenderDarwinPlist(d)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(plist, "<string>--api-key-env</string><string>ANTHROPIC_API_KEY</string>") || !strings.Contains(plist, "<string>--require-upstream-key</string>") {
		t.Fatalf("plist does not wire the durable upstream key:\n%s", plist)
	}
	if strings.Contains(plist, "sk-ant") {
		t.Fatal("plist must never carry a key value")
	}

	// No durable key at install time: today's passthrough plist, no fail-fast flags.
	d.UpstreamKeyEnv = ""
	plist, err = nodeRenderDarwinPlist(d)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(plist, "--api-key-env") || strings.Contains(plist, "--require-upstream-key") {
		t.Fatalf("passthrough plist must not arm the upstream key:\n%s", plist)
	}
}
