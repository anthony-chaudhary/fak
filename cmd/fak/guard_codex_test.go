package main

import (
	"bytes"
	"strings"
	"testing"
)

// guard_codex_test.go — coverage for the first-class `fak guard -- codex` install path
// (guard_codex.go). TestGuardDetectProvider already proves codex AUTODETECTS to the
// openai-responses upstream wire; what was untested is the other half of the seam — the
// argv rewrite that actually repoints Codex at the gateway. Codex does not read
// OPENAI_BASE_URL (the env var guard injects for the other OpenAI-wire agents), so
// installGuardCodexConfig is the ONLY thing that puts the kernel in front of Codex. An
// untested load-bearing path is not first-class, so these tests pin the override grammar,
// the /v1 base-URL math, the env-key resolution, the codex-only gating, and the
// subcommand-ordering invariant the gateway depends on.

// guardIsCodex must match the Codex CLI on its normalized base name alone — an absolute
// path, a Windows launcher suffix, or odd casing still matches — while never matching any
// other agent, because installGuardCodexConfig appends Codex-specific `-c` flags that would
// be garbage on a different argv.
func TestGuardIsCodex(t *testing.T) {
	cases := []struct {
		command string
		want    bool
	}{
		{"codex", true},
		{"Codex", true},                            // case-insensitive (guardAgentBaseName lowercases)
		{"codex.exe", true},                        // Windows launcher suffix stripped
		{"codex.cmd", true},                        // .cmd worker
		{"/usr/local/bin/codex", true},             // absolute POSIX path
		{`C:\Program Files\codex\codex.exe`, true}, // Windows absolute path + suffix
		{"  codex  ", true},                        // surrounding whitespace trimmed
		{"claude", false},
		{"opencode", false}, // contains "codex" as a substring but is NOT codex — base-name match, not substring
		{"aider", false},
		{"", false},
	}
	for _, tc := range cases {
		if got := guardIsCodex(tc.command); got != tc.want {
			t.Errorf("guardIsCodex(%q) = %v, want %v", tc.command, got, tc.want)
		}
	}
}

// guardCodexBaseURL gives Codex's Responses client the single `/v1` it appends `/responses`
// to. It must add exactly one /v1, be idempotent on a base that already carries it, trim a
// trailing slash so it never doubles up, and leave an empty base empty.
func TestGuardCodexBaseURL(t *testing.T) {
	cases := []struct {
		in, want string
	}{
		{"http://127.0.0.1:8137", "http://127.0.0.1:8137/v1"},     // bare origin gets /v1
		{"http://127.0.0.1:8137/", "http://127.0.0.1:8137/v1"},    // trailing slash trimmed first
		{"http://127.0.0.1:8137/v1", "http://127.0.0.1:8137/v1"},  // idempotent
		{"http://127.0.0.1:8137/v1/", "http://127.0.0.1:8137/v1"}, // /v1 + trailing slash -> /v1, not /v1/v1
		{"  http://h:1 ", "http://h:1/v1"},                        // surrounding whitespace trimmed
		{"", ""},                                                  // empty stays empty
	}
	for _, tc := range cases {
		if got := guardCodexBaseURL(tc.in); got != tc.want {
			t.Errorf("guardCodexBaseURL(%q) = %q, want %q", tc.in, got, tc.want)
		}
	}
}

// guardCodexEnvKey resolves the env var Codex reads the upstream bearer from: an explicit
// --api-key-env wins, an empty/whitespace value falls back to the OPENAI_API_KEY convention.
func TestGuardCodexEnvKey(t *testing.T) {
	cases := []struct {
		in, want string
	}{
		{"", guardCodexDefaultEnvKey},      // default convention
		{"   ", guardCodexDefaultEnvKey},   // whitespace-only is treated as unset
		{"MY_OPENAI_KEY", "MY_OPENAI_KEY"}, // explicit override wins
		{"  PADDED_KEY  ", "PADDED_KEY"},   // trimmed
	}
	for _, tc := range cases {
		if got := guardCodexEnvKey(tc.in); got != tc.want {
			t.Errorf("guardCodexEnvKey(%q) = %q, want %q", tc.in, got, tc.want)
		}
	}
	if guardCodexDefaultEnvKey != "OPENAI_API_KEY" {
		t.Errorf("guardCodexDefaultEnvKey = %q, want OPENAI_API_KEY (the OpenAI SDK convention)", guardCodexDefaultEnvKey)
	}
}

// guardCodexConfigArgs builds the ordered `-c key=value` overrides that define the `fak`
// provider in Codex's config. The provider id is used bare in model_provider= (Codex reads
// it as the id), while name/base_url/wire_api/env_key are TOML string literals carrying
// their own double quotes (guard execs the child directly, so Codex's TOML parser — not a
// shell — consumes the quotes). wire_api MUST be "responses" for the first-class guard
// path because the current Codex docs prefer Responses while Chat Completions is
// deprecated for future removal. This test pins the exact emitted sequence.
func TestGuardCodexConfigArgs(t *testing.T) {
	got := guardCodexConfigArgs("http://127.0.0.1:8137", "")
	want := []string{
		"-c", "model_provider=fak",
		"-c", `model_providers.fak.name="fak (kernel-adjudicated)"`,
		"-c", `model_providers.fak.base_url="http://127.0.0.1:8137/v1"`,
		"-c", `model_providers.fak.wire_api="responses"`,
		"-c", `model_providers.fak.env_key="OPENAI_API_KEY"`,
	}
	if len(got) != len(want) {
		t.Fatalf("guardCodexConfigArgs len = %d, want %d\n got=%v", len(got), len(want), got)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("guardCodexConfigArgs[%d] = %q, want %q", i, got[i], want[i])
		}
	}

	// An explicit --api-key-env threads through into env_key, and a base that already
	// carries /v1 is not doubled.
	gotKey := guardCodexConfigArgs("http://h:1/v1", "ALT_KEY")
	if !containsArg(gotKey, `model_providers.fak.env_key="ALT_KEY"`) {
		t.Errorf("guardCodexConfigArgs with --api-key-env ALT_KEY did not emit env_key=\"ALT_KEY\": %v", gotKey)
	}
	if !containsArg(gotKey, `model_providers.fak.base_url="http://h:1/v1"`) {
		t.Errorf("guardCodexConfigArgs did not keep the /v1 base undoubled: %v", gotKey)
	}

	// wire_api is responses on every code path for the first-class guard route.
	if !containsArg(got, `model_providers.fak.wire_api="responses"`) {
		t.Errorf("guardCodexConfigArgs must pin wire_api=\"responses\": %v", got)
	}
}

// installGuardCodexConfig is the rewrite that makes `fak guard -- codex` route through the
// gateway: it inserts the `-c` overrides immediately AFTER the codex executable and BEFORE
// any subcommand (`exec`) or user args, because Codex's global `-c` flag must precede the
// subcommand. It must be inert for a non-codex agent, inert when disabled, inert on an empty
// command, and it must report what it did in the guardCodexInstall struct for the banner.
func TestInstallGuardCodexConfigCodexOnlyRewrite(t *testing.T) {
	const gw = "http://127.0.0.1:8137"

	t.Run("codex enabled rewrites argv with overrides before the subcommand", func(t *testing.T) {
		in := []string{"codex", "exec", "do the thing"}
		out, info := installGuardCodexConfig(in, true, gw, "")
		if !info.Applied {
			t.Fatalf("install not Applied for codex: %+v", info)
		}
		if out[0] != "codex" {
			t.Errorf("executable must stay first, got %q", out[0])
		}
		// The user's subcommand + args survive, in order, AFTER the overrides.
		if out[len(out)-2] != "exec" || out[len(out)-1] != "do the thing" {
			t.Errorf("user subcommand/args not preserved at the tail: %v", out)
		}
		// The critical ordering invariant: every -c override sits before the `exec` subcommand.
		ix := indexOf(out, "exec")
		lastC := lastIndexOf(out, "-c")
		if ix < 0 || lastC < 0 || lastC > ix {
			t.Errorf("`-c` overrides must precede the subcommand: lastC=%d exec=%d argv=%v", lastC, ix, out)
		}
		// The struct the banner reads is fully populated.
		if info.ProviderID != "fak" || info.EnvKey != "OPENAI_API_KEY" || info.BaseURL != gw+"/v1" {
			t.Errorf("guardCodexInstall fields = %+v, want provider=fak env=OPENAI_API_KEY base=%s/v1", info, gw)
		}
	})

	t.Run("no subcommand still inserts overrides after the executable", func(t *testing.T) {
		out, info := installGuardCodexConfig([]string{"codex"}, true, gw, "")
		if !info.Applied || out[0] != "codex" || !containsArg(out, "model_provider=fak") {
			t.Errorf("bare `codex` not rewritten: out=%v info=%+v", out, info)
		}
	})

	t.Run("disabled is a no-op", func(t *testing.T) {
		in := []string{"codex", "exec"}
		out, info := installGuardCodexConfig(in, false, gw, "")
		if info.Applied || !equalArgs(out, in) {
			t.Errorf("--codex-config=false must leave argv unchanged: out=%v info=%+v", out, info)
		}
	})

	t.Run("non-codex agent is a no-op", func(t *testing.T) {
		in := []string{"claude", "--dangerously-skip-permissions"}
		out, info := installGuardCodexConfig(in, true, gw, "")
		if info.Applied || !equalArgs(out, in) {
			t.Errorf("a non-codex agent must never get codex `-c` flags: out=%v info=%+v", out, info)
		}
	})

	t.Run("empty command is a no-op", func(t *testing.T) {
		out, info := installGuardCodexConfig(nil, true, gw, "")
		if info.Applied || len(out) != 0 {
			t.Errorf("empty command must be inert: out=%v info=%+v", out, info)
		}
	})
}

// printGuardCodexNote stays silent unless the install actually applied (so the banner does
// not lie about a codex repoint for a non-codex agent), and when it applied it must name the
// provider, the responses wire, the base URL, and the honest auth fence (the env key, and
// that a ChatGPT-subscription `codex login` is not yet wired).
func TestPrintGuardCodexNote(t *testing.T) {
	var quiet bytes.Buffer
	printGuardCodexNote(&quiet, guardCodexInstall{}) // Applied=false
	if quiet.Len() != 0 {
		t.Errorf("printGuardCodexNote must be silent when not applied, wrote: %q", quiet.String())
	}

	var b bytes.Buffer
	printGuardCodexNote(&b, guardCodexInstall{
		Applied:    true,
		ProviderID: "fak",
		EnvKey:     "OPENAI_API_KEY",
		BaseURL:    "http://127.0.0.1:8137/v1",
	})
	out := b.String()
	for _, want := range []string{"fak", "responses", "http://127.0.0.1:8137/v1", "OPENAI_API_KEY", "codex login"} {
		if !strings.Contains(out, want) {
			t.Errorf("printGuardCodexNote output missing %q\n got: %s", want, out)
		}
	}
}

// --- small slice helpers (local to keep the assertions readable) ---

func containsArg(args []string, want string) bool { return indexOf(args, want) >= 0 }

func indexOf(args []string, want string) int {
	for i, a := range args {
		if a == want {
			return i
		}
	}
	return -1
}

func lastIndexOf(args []string, want string) int {
	for i := len(args) - 1; i >= 0; i-- {
		if args[i] == want {
			return i
		}
	}
	return -1
}

func equalArgs(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
