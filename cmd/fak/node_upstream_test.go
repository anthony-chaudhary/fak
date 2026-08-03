package main

// Tests for #5555: `fak node install` owned one launchd label AND pinned the Anthropic
// upstream into every rendered unit, so the sanctioned installer could never produce a
// local-model gateway. The fix keeps ONE label and teaches the installer the upstream
// (--base-url / --provider / --model / --env).
//
// WHAT THESE TESTS WITNESS, AND WHAT THEY DO NOT. The ticket's Verify offers two
// alternatives: loading both units on one Mac, or `fak node install --help` being able to
// produce both. These tests witness the SECOND. There is no launchd here — nothing below
// installs, loads, or registers anything — so the rendered bytes are pinned instead:
// nodeRenderDarwinPlist / nodeRenderLinuxUnit / nodeRenderWindowsRunner are pure functions
// of one nodeUnitData, and both headline outputs (the default Anthropic adjudication proxy
// and a local-model gateway) are frozen as fixtures that run identically on any OS. That a
// launchd unit built from those bytes actually loads is NOT witnessed here.
//
// Every secret in these fixtures is an obvious placeholder; no test prints a value.

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// ── upstream resolution + the credential-egress bound ─────────────────────────

func TestNodeResolveUpstreamDefaultsAndRefusals(t *testing.T) {
	const remoteAnthropicShim = "https://gateway.example.net/v1"
	cases := []struct {
		name                      string
		provider, baseURL, model  string
		wantProvider, wantBaseURL string
		wantModel                 string
		wantErrSubstr             string
		wantWarnSubstrs           []string
		wantNoWarn                bool
	}{
		{
			name:         "no flags keeps the historical Anthropic upstream",
			wantProvider: "anthropic",
			wantBaseURL:  "https://api.anthropic.com",
			wantNoWarn:   true,
		},
		{
			name:         "a loopback base-url defaults to the OpenAI-compatible wire",
			baseURL:      "http://127.0.0.1:8131/v1",
			model:        "qwen3.6-27b",
			wantProvider: "openai",
			wantBaseURL:  "http://127.0.0.1:8131/v1",
			wantModel:    "qwen3.6-27b",
			wantNoWarn:   true, // loopback: nothing leaves the machine
		},
		{
			name:         "an explicit Anthropic base-url still defaults to the anthropic wire",
			baseURL:      "https://api.anthropic.com",
			wantProvider: "anthropic",
			wantBaseURL:  "https://api.anthropic.com",
			wantNoWarn:   true,
		},
		{
			name:          "the anthropic wire may NOT be pointed at an arbitrary remote host",
			provider:      "anthropic",
			baseURL:       remoteAnthropicShim,
			wantErrSubstr: "forwards the CALLER's own API key upstream",
		},
		{
			name:         "the anthropic wire MAY be pointed at loopback (the key never leaves the host)",
			provider:     "anthropic",
			baseURL:      "http://127.0.0.1:4000",
			wantProvider: "anthropic",
			wantBaseURL:  "http://127.0.0.1:4000",
			wantNoWarn:   true,
		},
		{
			name:            "a remote OpenAI-wire upstream is allowed but names the egress",
			baseURL:         "http://100.64.0.7:8131/v1",
			wantProvider:    "openai",
			wantBaseURL:     "http://100.64.0.7:8131/v1",
			wantWarnSubstrs: []string{"100.64.0.7", "is NOT on this machine", "plain http"},
		},
		{
			name:            "a remote https OpenAI-wire upstream warns about egress only",
			baseURL:         "https://models.example.net/v1",
			wantProvider:    "openai",
			wantBaseURL:     "https://models.example.net/v1",
			wantWarnSubstrs: []string{"models.example.net", "is NOT on this machine"},
		},
		{
			name:          "a base-url with no scheme is refused, not silently installed",
			baseURL:       "127.0.0.1:8131",
			wantErrSubstr: "absolute http:// or https:// URL",
		},
		{
			name:          "an unknown provider is refused",
			provider:      "llamacpp",
			baseURL:       "http://127.0.0.1:8131/v1",
			wantErrSubstr: "is not an upstream wire",
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, warns, err := nodeResolveUpstream(tc.provider, tc.baseURL, tc.model)
			if tc.wantErrSubstr != "" {
				if err == nil {
					t.Fatalf("nodeResolveUpstream(%q,%q,%q) = %+v, want a refusal containing %q",
						tc.provider, tc.baseURL, tc.model, got, tc.wantErrSubstr)
				}
				if !strings.Contains(err.Error(), tc.wantErrSubstr) {
					t.Fatalf("refusal = %q, want it to contain %q", err.Error(), tc.wantErrSubstr)
				}
				return
			}
			if err != nil {
				t.Fatalf("nodeResolveUpstream(%q,%q,%q): %v", tc.provider, tc.baseURL, tc.model, err)
			}
			if got.Provider != tc.wantProvider || got.BaseURL != tc.wantBaseURL || got.Model != tc.wantModel {
				t.Fatalf("got %+v, want provider=%q base=%q model=%q",
					got, tc.wantProvider, tc.wantBaseURL, tc.wantModel)
			}
			joined := strings.Join(warns, " | ")
			if tc.wantNoWarn && len(warns) != 0 {
				t.Fatalf("want no warnings for an on-host upstream, got %q", joined)
			}
			for _, want := range tc.wantWarnSubstrs {
				if !strings.Contains(joined, want) {
					t.Fatalf("warnings %q missing %q", joined, want)
				}
			}
		})
	}
}

// TestNodeInstallRefusesCredentialEgressFromTheCLI proves the refusal is wired into the
// command, not just reachable in the helper. It is written so a REGRESSION cannot install
// anything: --port 99999 is out of range, so if the upstream guard were ever removed the
// bind-address check that runs immediately after it still returns 2 before any config dir,
// unit file, or service registration is touched. The assertion is on the credential text,
// which only the upstream guard emits.
func TestNodeInstallRefusesCredentialEgressFromTheCLI(t *testing.T) {
	var stdout, stderr bytes.Buffer
	rc := nodeInstall(&stdout, &stderr, []string{
		"--provider", "anthropic",
		"--base-url", "https://gateway.example.net/v1",
		"--port", "99999",
	})
	if rc != 2 {
		t.Fatalf("nodeInstall rc = %d, want 2 (refused)", rc)
	}
	if !strings.Contains(stderr.String(), "forwards the CALLER's own API key upstream") {
		t.Fatalf("stderr = %q, want the credential-egress refusal", stderr.String())
	}
	if stdout.Len() != 0 {
		t.Fatalf("a refused install must print nothing to stdout, got %q", stdout.String())
	}
}

// TestNodeInstallHelpProducesBothUpstreams is the ticket's second Verify branch, run
// literally: `fak node install --help` must show how to produce BOTH the Anthropic
// adjudication proxy and a local-model gateway.
func TestNodeInstallHelpProducesBothUpstreams(t *testing.T) {
	var stdout, stderr bytes.Buffer
	if rc := nodeInstall(&stdout, &stderr, []string{"--help"}); rc != 0 {
		t.Fatalf("`fak node install --help` rc = %d, want 0", rc)
	}
	help := stderr.String()
	for _, want := range []string{
		"-base-url",
		"-provider",
		"-model",
		"-env",
		"fak node install\n", // the Anthropic default
		"fak node install --base-url http://127.0.0.1:8131/v1", // the local-model unit
		"--model qwen3.6-27b",
		"does not need --uninstall first",
	} {
		if !strings.Contains(help, want) {
			t.Fatalf("`fak node install --help` is missing %q\n---\n%s", want, help)
		}
	}
}

// ── the --env flag never echoes a value ───────────────────────────────────────

func TestNodeEnvFlagParsesAndNeverEchoesValues(t *testing.T) {
	// An obviously fake placeholder standing in for something an operator might pass.
	const placeholder = "placeholder-not-a-real-secret"
	var f nodeEnvFlag
	if err := f.Set("FAK_PLANNER_TIMEOUT_S=1800"); err != nil {
		t.Fatalf("Set: %v", err)
	}
	if err := f.Set("FAK_DEMO_TOKEN=" + placeholder); err != nil {
		t.Fatalf("Set: %v", err)
	}
	if len(f) != 2 || f[0].Name != "FAK_PLANNER_TIMEOUT_S" || f[1].Value != placeholder {
		t.Fatalf("parsed entries = %+v, want both NAME=VALUE pairs in order", f)
	}
	if s := f.String(); !strings.Contains(s, "FAK_DEMO_TOKEN") || strings.Contains(s, placeholder) {
		t.Fatalf("nodeEnvFlag.String() = %q — it must report NAMES and never a value", s)
	}
	for _, bad := range []string{"", "NOEQUALS", "=novalue", "bad-name=1", "FAK_AUDIT_JOURNAL=/tmp/x"} {
		if err := f.Set(bad); err == nil {
			t.Errorf("Set(%q) = nil, want a rejection", bad)
		}
	}
	if err := f.Set("FAK_DEMO_TOKEN=again"); err == nil {
		t.Error("Set of a duplicate name = nil, want a rejection")
	}
}

func TestNodePrintUnitSummaryReportsNamesNotValues(t *testing.T) {
	const placeholder = "placeholder-not-a-real-secret"
	in := nodeInstallParams{
		upstream: nodeUpstream{Provider: "openai", BaseURL: "http://127.0.0.1:8131/v1", Model: "qwen3.6-27b"},
		env:      []nodeEnvVar{{Name: "FAK_DEMO_TOKEN", Value: placeholder}},
	}
	var out bytes.Buffer
	nodePrintUnitSummary(&out, in)
	got := out.String()
	if !strings.Contains(got, "openai http://127.0.0.1:8131/v1 (model qwen3.6-27b)") {
		t.Fatalf("summary = %q, want the resolved upstream", got)
	}
	if !strings.Contains(got, "FAK_DEMO_TOKEN") {
		t.Fatalf("summary = %q, want the env NAME reported", got)
	}
	if strings.Contains(got, placeholder) {
		t.Fatalf("summary printed an --env VALUE: %q", got)
	}
}

// ── the two pinned units ──────────────────────────────────────────────────────

// nodeFixtureUnit builds the renderer input the way `fak node install` does — through
// nodeResolveUpstream and nodeUnitDataFor — so the goldens pin the whole path from flags
// to unit bytes, not just a template. Paths are fixed placeholders so the output is
// byte-identical on every machine and OS.
func nodeFixtureUnit(t *testing.T, provider, baseURL, model string, env []nodeEnvVar) nodeUnitData {
	t.Helper()
	up, _, err := nodeResolveUpstream(provider, baseURL, model)
	if err != nil {
		t.Fatalf("nodeResolveUpstream(%q,%q,%q): %v", provider, baseURL, model, err)
	}
	in := nodeInstallParams{addr: "127.0.0.1:8080", localPort: "8080", upstream: up, env: env}
	return nodeUnitDataFor(in,
		nodeGatewayLabel,
		"/Users/youruser/.config/fak/serve-wrapper.sh",
		"/Users/youruser/.local/bin/fak",
		"/Users/youruser/.config/fak/node-policy.json",
		"/Users/youruser/.config/fak/logs",
		"", "")
}

// nodeRenderAllUnits renders one nodeUnitData through all three platform renderers. The
// units are pure functions, so a Windows box can pin the launchd plist exactly.
func nodeRenderAllUnits(t *testing.T, d nodeUnitData) string {
	t.Helper()
	plist, err := nodeRenderDarwinPlist(d)
	if err != nil {
		t.Fatalf("render launchd plist: %v", err)
	}
	unit, err := nodeRenderLinuxUnit(d)
	if err != nil {
		t.Fatalf("render systemd unit: %v", err)
	}
	runner, err := nodeRenderWindowsRunner(d)
	if err != nil {
		t.Fatalf("render windows runner: %v", err)
	}
	return "=== launchd (macOS) ===\n" + plist +
		"\n=== systemd --user (Linux) ===\n" + unit +
		"\n=== Scheduled Task runner (Windows) ===\n" + runner
}

// nodeCheckGolden compares got against the committed fixture. Regenerate deliberately
// after an intentional change with:
//
//	FAK_UPDATE_GOLDEN=1 go test ./cmd/fak -run TestNodeInstall -timeout=25m
func nodeCheckGolden(t *testing.T, name, got string) {
	t.Helper()
	path := filepath.Join("testdata", name)
	if os.Getenv("FAK_UPDATE_GOLDEN") != "" {
		if err := os.WriteFile(path, []byte(got), 0o644); err != nil {
			t.Fatalf("write golden %s: %v", path, err)
		}
		t.Logf("updated golden %s", path)
		return
	}
	want, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read golden %s: %v (regenerate with FAK_UPDATE_GOLDEN=1)", path, err)
	}
	if got != strings.ReplaceAll(string(want), "\r\n", "\n") {
		t.Fatalf("rendered units drifted from %s — regenerate with FAK_UPDATE_GOLDEN=1 and review the diff.\n--- got ---\n%s", path, got)
	}
}

// TestNodeInstallRendersDefaultAnthropicUnit pins the unit the unchanged invocation
// produces, so the #5555 flags cannot silently move the default upstream.
func TestNodeInstallRendersDefaultAnthropicUnit(t *testing.T) {
	got := nodeRenderAllUnits(t, nodeFixtureUnit(t, "", "", "", nil))
	for _, want := range []string{
		"<string>--provider</string><string>anthropic</string>",
		"<string>--base-url</string><string>https://api.anthropic.com</string>",
		"--provider anthropic --base-url https://api.anthropic.com",
	} {
		if !strings.Contains(got, want) {
			t.Fatalf("default unit is missing %q", want)
		}
	}
	if strings.Contains(got, "--model") {
		t.Fatal("the default unit must not pin --model (it keeps `fak serve`'s own default)")
	}
	nodeCheckGolden(t, "node_install_unit_anthropic.golden", got)
}

// TestNodeInstallRendersLocalModelUnit is the other half of the pair: the SAME installer,
// given the local-model flags, renders the unit docs/fak/mac-agent-ui.md previously told
// operators to hand-write under a colliding label.
func TestNodeInstallRendersLocalModelUnit(t *testing.T) {
	got := nodeRenderAllUnits(t, nodeFixtureUnit(t, "", "http://127.0.0.1:8131/v1", "qwen3.6-27b", []nodeEnvVar{
		{Name: "FAK_PLANNER_TIMEOUT_S", Value: "1800"},
		{Name: "FAK_HTTP_WRITE_TIMEOUT_S", Value: "1800"},
		{Name: "FAK_PROVIDER_EXTRA_BODY_JSON", Value: `{"top_k":20,"chat_template_kwargs":{"preserve_thinking":true}}`},
	}))
	for _, want := range []string{
		"<string>--provider</string><string>openai</string>",
		"<string>--base-url</string><string>http://127.0.0.1:8131/v1</string>",
		"<string>--model</string><string>qwen3.6-27b</string>",
		"<key>FAK_PROVIDER_EXTRA_BODY_JSON</key>",
		"--provider openai --base-url http://127.0.0.1:8131/v1 --model qwen3.6-27b",
	} {
		if !strings.Contains(got, want) {
			t.Fatalf("local-model unit is missing %q", want)
		}
	}
	if strings.Contains(got, "api.anthropic.com") {
		t.Fatal("the local-model unit must not carry the Anthropic upstream")
	}
	// Both units carry the SAME label: one gateway per host is the shape chosen in #5555,
	// so `--uninstall` and `status` keep naming exactly the unit that exists.
	if !strings.Contains(got, "<string>"+nodeGatewayLabel+"</string>") {
		t.Fatalf("local-model unit must keep the %s label", nodeGatewayLabel)
	}
	nodeCheckGolden(t, "node_install_unit_localmodel.golden", got)
}

// TestNodeUnitEscapingSurvivesHostileValues proves the per-format escapers are wired: a
// value carrying XML, systemd and cmd metacharacters must not be able to corrupt the
// rendered unit. The value is meaningless punctuation, not a secret.
func TestNodeUnitEscapingSurvivesHostileValues(t *testing.T) {
	d := nodeFixtureUnit(t, "", "http://127.0.0.1:8131/v1", "", []nodeEnvVar{
		{Name: "FAK_DEMO", Value: `a&b<c>"d"|e`},
	})
	plist, err := nodeRenderDarwinPlist(d)
	if err != nil {
		t.Fatalf("render plist: %v", err)
	}
	if !strings.Contains(plist, "a&amp;b&lt;c&gt;&quot;d&quot;|e") {
		t.Fatalf("plist did not XML-escape the env value:\n%s", plist)
	}
	unit, err := nodeRenderLinuxUnit(d)
	if err != nil {
		t.Fatalf("render unit: %v", err)
	}
	if !strings.Contains(unit, `Environment="FAK_DEMO=a&b<c>\"d\"|e"`) {
		t.Fatalf("systemd unit did not escape the quotes in the env value:\n%s", unit)
	}
	runner, err := nodeRenderWindowsRunner(d)
	if err != nil {
		t.Fatalf("render runner: %v", err)
	}
	if !strings.Contains(runner, `set FAK_DEMO=a^&b^<c^>"d"^|e`) {
		t.Fatalf("windows runner did not escape the cmd metacharacters:\n%s", runner)
	}
}
