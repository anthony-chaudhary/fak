package main

import (
	"bytes"
	"context"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"runtime"
	"slices"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/harnessprofile"
)

func TestGuardLaunchPlanIdentityComposition(t *testing.T) {
	tests := []struct {
		name         string
		command      []string
		wantProfile  string
		wantWire     harnessprofile.Wire
		wantRepoints []harnessprofile.RepointMechanism
	}{
		{
			name:         "claude",
			command:      []string{"claude", "-p", "inspect the repository"},
			wantProfile:  "claude",
			wantWire:     harnessprofile.WireAnthropic,
			wantRepoints: []harnessprofile.RepointMechanism{harnessprofile.RepointEnv, harnessprofile.RepointSettingsFile},
		},
		{
			name:         "codex",
			command:      []string{"codex", "exec", "inspect the repository"},
			wantProfile:  "codex",
			wantWire:     harnessprofile.WireOpenAIResponses,
			wantRepoints: []harnessprofile.RepointMechanism{harnessprofile.RepointEnv, harnessprofile.RepointCLIConfig},
		},
		{
			name:         "openai-generic",
			command:      []string{"opencode", "run", "inspect the repository"},
			wantProfile:  "openai-generic",
			wantWire:     harnessprofile.WireOpenAI,
			wantRepoints: []harnessprofile.RepointMechanism{harnessprofile.RepointEnv},
		},
	}
	transforms := []struct {
		name  string
		apply func([]string) []string
	}{
		{"broker", func(argv []string) []string { return append([]string(nil), argv...) }},
		{"prompt-stdin", func(argv []string) []string { return append([]string{argv[0], "--prompt-stdin"}, argv[1:]...) }},
		{"landlock", func(argv []string) []string { return append([]string{"fak", "landlock-exec", "--"}, argv...) }},
		{"terminal", func(argv []string) []string { return append([]string{"terminal-host", "--"}, argv...) }},
		{"os-exec", func(argv []string) []string {
			return append([]string{"node.exe", "agent-entrypoint.js", "--"}, argv...)
		}},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			plan := newGuardLaunchPlan(tc.command)
			if got := plan.semanticCommand(); !slices.Equal(got, tc.command) {
				t.Fatalf("semantic argv = %v, want %v", got, tc.command)
			}
			if got := plan.harnessProfile(); got.Name != tc.wantProfile || got.Wire != tc.wantWire || !reflect.DeepEqual(got.Repoint, tc.wantRepoints) {
				t.Fatalf("profile = %+v, want name=%q wire=%q repoint=%v", got, tc.wantProfile, tc.wantWire, tc.wantRepoints)
			}
			for _, transform := range transforms {
				plan = plan.withExecutableCommand(transform.apply(plan.executableCommand()))
				if got := plan.semanticCommand(); !slices.Equal(got, tc.command) {
					t.Fatalf("%s changed semantic argv to %v, want %v", transform.name, got, tc.command)
				}
				if got := plan.harnessProfile(); got.Name != tc.wantProfile || got.Wire != tc.wantWire || !reflect.DeepEqual(got.Repoint, tc.wantRepoints) {
					t.Fatalf("%s changed profile to %+v", transform.name, got)
				}
			}
			semantic := plan.semanticCommand()
			semantic[0] = "mutated"
			if got := plan.semanticCommand()[0]; got != tc.command[0] {
				t.Fatalf("semantic argv escaped immutability: got %q want %q", got, tc.command[0])
			}
			executable := plan.executableCommand()
			executable[0] = "mutated"
			if got := plan.executableCommand()[0]; got == "mutated" {
				t.Fatal("executable argv escaped immutability")
			}
			profile := plan.harnessProfile()
			profile.Repoint[0] = "mutated"
			if got := plan.harnessProfile().Repoint[0]; got == "mutated" {
				t.Fatal("harness profile escaped immutability")
			}
		})
	}

	codex := newGuardLaunchPlan([]string{"codex", "resume", "thread-id"})
	if provider, autodetected := codex.resolveProvider("openai"); provider != "openai-responses" || autodetected {
		t.Fatalf("explicit Codex OpenAI provider = %q autodetected=%v, want openai-responses false", provider, autodetected)
	}
	claude := newGuardLaunchPlan([]string{"claude"})
	if provider, autodetected := claude.resolveProvider("openai"); provider != "openai" || autodetected {
		t.Fatalf("non-Codex explicit OpenAI provider = %q autodetected=%v, want openai false", provider, autodetected)
	}

	unknown := newGuardLaunchPlan([]string{"custom-agent", "run"})
	if unknown.recognized() || unknown.harnessProfile().Recognized() {
		t.Fatalf("unknown command resolved a profile: %+v", unknown.harnessProfile())
	}
	if provider, autodetected := unknown.resolveProvider(""); provider != "anthropic" || autodetected {
		t.Fatalf("unknown provider fallback = %q autodetected=%v, want anthropic false", provider, autodetected)
	}
	for _, path := range []string{"guard.go", "guard_child.go", "guard_upstream_posture.go", "guard_codex_oauth.go", "guard_sessionstart.go"} {
		source, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		if bytes.Contains(source, []byte("guardIsCodex(")) {
			t.Fatalf("%s re-detects Codex identity after launch-plan admission", path)
		}
	}
}

func TestManagedGuardResolvesWindowsShimOnlyAtExecBoundary(t *testing.T) {
	guardSource, err := os.ReadFile("guard.go")
	if err != nil {
		t.Fatal(err)
	}
	if bytes.Contains(guardSource, []byte("resolveWindowsBatchCommand(")) {
		t.Fatal("cmdManageCommand resolves the Windows shim before Codex wiring; keep executable resolution inside buildGuardChild")
	}

	childSource, err := os.ReadFile("guard_child.go")
	if err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{
		"plan, env := guardChildPlanCommandEnv(newGuardLaunchPlan(command)",
		"child := newResolvedExecCommand(plan.executableCommand())",
		"command = resolveWindowsBatchCommand(command)",
		"return exec.Command(command[0]",
		"child := newResolvedExecCommand(grant.Argv)",
	} {
		if !bytes.Contains(childSource, []byte(want)) {
			t.Fatalf("guard child preparation lost semantic-command wiring %q", want)
		}
	}
}

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

func TestGuardCodexGatewayModel(t *testing.T) {
	// #10669: the guarded Astra default effort is medium; xhigh is opt-in.
	if guardCodexDefaultModelID != "gpt-6-astra" || guardCodexDefaultReasoningEffort != "medium" {
		t.Fatalf("Codex defaults = model %q effort %q, want gpt-6-astra/medium",
			guardCodexDefaultModelID, guardCodexDefaultReasoningEffort)
	}
	if got := guardCodexGatewayModel([]string{"codex"}, "", "openai-responses"); got != guardCodexDefaultModelID {
		t.Fatalf("default Codex gateway model = %q, want %q", got, guardCodexDefaultModelID)
	}
	if got := guardCodexGatewayModel([]string{"codex"}, "gpt-custom", "openai-responses"); got != "gpt-custom" {
		t.Fatalf("explicit Codex gateway model = %q, want gpt-custom", got)
	}
	if got := guardCodexGatewayModel([]string{"codex"}, "gpt 6 astra", "openai-responses"); got != "gpt-6-astra" {
		t.Fatalf("aliased Codex gateway model = %q, want gpt-6-astra", got)
	}
	if got := guardCodexGatewayModel([]string{"codex"}, "astra", "openai-responses"); got != "gpt-6-astra" {
		t.Fatalf("bare astra gateway model = %q, want gpt-6-astra", got)
	}
	if got := guardCodexGatewayModel([]string{"claude"}, "", "anthropic"); got != "" {
		t.Fatalf("non-Codex gateway model = %q, want empty passthrough default", got)
	}
}

func TestGuardCodexLoopGateConfigCodexOnly(t *testing.T) {
	cfg, ok := guardCodexLoopGateConfig([]string{"codex", "exec"}, "loop", "C:/tmp/codex", 12, 7, true)
	if !ok {
		t.Fatal("codex command did not enable loop gate config")
	}
	if cfg.Threshold != "loop" || cfg.CodexHome != "C:/tmp/codex" || cfg.SinceHours != 12 || cfg.Limit != 7 || !cfg.Quiet {
		t.Fatalf("wrong loop gate config: %+v", cfg)
	}
	if got := cfg.BypassCommand; got != "fak m --codex-loop-gate off -- codex exec" {
		t.Fatalf("bypass command=%q", got)
	}
	for _, command := range [][]string{
		nil,
		{"claude"},
		{"opencode"},
	} {
		if cfg, ok := guardCodexLoopGateConfig(command, "loop", "", 24, 20, false); ok {
			t.Fatalf("non-Codex command got loop gate config: command=%v cfg=%+v", command, cfg)
		}
	}
}

func TestGuardCodexLoopGateDefaultOffAndEnvironmentOptIn(t *testing.T) {
	t.Setenv("CODEX_THREAD_ID", "")
	home := codexLauncherLoopFixtureForProvider(t, "fak")
	if err := writeCodexGuardWitness(home, "loop-session"); err != nil {
		t.Fatal(err)
	}

	t.Run("default off skips transcript audit", func(t *testing.T) {
		t.Setenv("FLEET_CODEX_LOOP_GATE", "")
		cfg, ok := guardCodexLoopGateConfig([]string{"codex", "exec"}, dispatchCodexLoopGateDefaultThreshold(), home, 0, 20, true)
		if !ok || cfg.Threshold != "off" {
			t.Fatalf("default guard loop-gate config = %+v ok=%v, want off", cfg, ok)
		}
		var errb bytes.Buffer
		if rc := runCodexLoopGate(&errb, cfg); rc != 0 || errb.Len() != 0 {
			t.Fatalf("default-off guard gate rc=%d stderr=%s", rc, errb.String())
		}
	})

	t.Run("environment opt-in retains refusal", func(t *testing.T) {
		t.Setenv("FLEET_CODEX_LOOP_GATE", "loop")
		cfg, ok := guardCodexLoopGateConfig([]string{"codex", "exec"}, dispatchCodexLoopGateDefaultThreshold(), home, 0, 20, true)
		if !ok || cfg.Threshold != "loop" {
			t.Fatalf("environment guard loop-gate config = %+v ok=%v, want loop", cfg, ok)
		}
		var errb bytes.Buffer
		if rc := runCodexLoopGate(&errb, cfg); rc != 1 || !strings.Contains(errb.String(), "loop gate REFUSE fail-on=loop verdict=LOOP") {
			t.Fatalf("environment-opted-in guard gate rc=%d stderr=%s", rc, errb.String())
		}
	})
}

func TestGuardCodexLoopGateHelpSaysOptInDefaultOff(t *testing.T) {
	source, err := os.ReadFile("guard.go")
	if err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{"Codex-only opt-in launch gate", "else off", "loop|action"} {
		if !bytes.Contains(source, []byte(want)) {
			t.Fatalf("guard help source missing %q", want)
		}
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
	t.Setenv(guardCodexReasoningEffortEnv, "")
	t.Setenv("CODEX_HOME", t.TempDir())
	got := guardCodexConfigArgs("http://127.0.0.1:8137", "", "")
	want := []string{
		"-c", "model_provider=fak",
		"-c", `model="gpt-6-astra"`,
		"-c", `model_providers.fak.name="fak (kernel-adjudicated)"`,
		"-c", `model_providers.fak.base_url="http://127.0.0.1:8137/v1"`,
		"-c", `model_providers.fak.wire_api="responses"`,
		"-c", `model_providers.fak.env_key="OPENAI_API_KEY"`,
		"-c", `mcp_servers.fak_guard.url="http://127.0.0.1:8137/mcp"`,
		"-c", `model_reasoning_effort="medium"`,
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
	gotKey := guardCodexConfigArgs("http://h:1/v1", "ALT_KEY", "gpt-custom")
	if !containsArg(gotKey, `model_providers.fak.env_key="ALT_KEY"`) {
		t.Errorf("guardCodexConfigArgs with --api-key-env ALT_KEY did not emit env_key=\"ALT_KEY\": %v", gotKey)
	}
	if !containsArg(gotKey, `model_providers.fak.base_url="http://h:1/v1"`) {
		t.Errorf("guardCodexConfigArgs did not keep the /v1 base undoubled: %v", gotKey)
	}
	if !containsArg(gotKey, `mcp_servers.fak_guard.url="http://h:1/mcp"`) {
		t.Errorf("guardCodexConfigArgs did not derive the gateway MCP route: %v", gotKey)
	}
	if !containsArg(gotKey, `model="gpt-custom"`) || containsArg(gotKey, "model_reasoning_effort=") {
		t.Errorf("custom model must be pinned without a managed effort value (#10669: only an explicit opt-in may pin one): %v", gotKey)
	}

	// wire_api is responses on every code path for the first-class guard route.
	if !containsArg(got, `model_providers.fak.wire_api="responses"`) {
		t.Errorf("guardCodexConfigArgs must pin wire_api=\"responses\": %v", got)
	}
}

func TestGuardCodexConfigArgsDisablesFakMCPServerWhenPresent(t *testing.T) {
	t.Setenv(guardCodexReasoningEffortEnv, "")
	dir := t.TempDir()
	cfg := filepath.Join(dir, "config.toml")
	content := `[model_providers.fak]
name = "fak"

[mcp_servers.fak]
command = "fak"
args = ["serve", "--stdio"]
`
	if err := os.WriteFile(cfg, []byte(content), 0o600); err != nil {
		t.Fatal(err)
	}

	t.Run("via codexHome arg", func(t *testing.T) {
		t.Setenv("CODEX_HOME", "")
		got := guardCodexConfigArgs("http://127.0.0.1:8137", "", "", dir)
		if !containsArg(got, "mcp_servers.fak.enabled=false") {
			t.Errorf("expected mcp_servers.fak.enabled=false when [mcp_servers.fak] is present: %v", got)
		}
		// Confirm mcp_servers.fak_guard.url is still emitted
		if !containsArg(got, `mcp_servers.fak_guard.url="http://127.0.0.1:8137/mcp"`) {
			t.Errorf("expected mcp_servers.fak_guard.url to be emitted: %v", got)
		}
	})

	t.Run("via CODEX_HOME env", func(t *testing.T) {
		t.Setenv("CODEX_HOME", dir)
		got := guardCodexConfigArgs("http://127.0.0.1:8137", "", "")
		if !containsArg(got, "mcp_servers.fak.enabled=false") {
			t.Errorf("expected mcp_servers.fak.enabled=false when [mcp_servers.fak] is present: %v", got)
		}
	})

	t.Run("omitted when mcp_servers.fak is absent", func(t *testing.T) {
		emptyDir := t.TempDir()
		emptyCfg := filepath.Join(emptyDir, "config.toml")
		if err := os.WriteFile(emptyCfg, []byte(`[mcp_servers.node_repl]
command = "node"
`), 0o600); err != nil {
			t.Fatal(err)
		}
		got := guardCodexConfigArgs("http://127.0.0.1:8137", "", "", emptyDir)
		if containsArg(got, "mcp_servers.fak.enabled=false") {
			t.Errorf("unexpected mcp_servers.fak.enabled=false when [mcp_servers.fak] is absent: %v", got)
		}
	})
}

func TestGuardCodexMCPConfigSuppressesDuplicateFak(t *testing.T) {
	t.Setenv(guardCodexReasoningEffortEnv, "")
	dir := t.TempDir()
	t.Setenv("CODEX_HOME", dir)
	cfg := filepath.Join(dir, "config.toml")
	content := `[mcp_servers.fak]
command = "fak"
args = ["serve", "--stdio"]
`
	if err := os.WriteFile(cfg, []byte(content), 0o600); err != nil {
		t.Fatal(err)
	}

	got := guardCodexConfigArgs("http://127.0.0.1:8137", "", "", dir)
	found := false
	for i := 0; i < len(got)-1; i++ {
		if got[i] == "-c" && got[i+1] == "mcp_servers.fak.enabled=false" {
			found = true
			break
		}
	}
	if !found {
		t.Fatalf("expected guardCodexConfigArgs to include -c mcp_servers.fak.enabled=false when [mcp_servers.fak] is present: %v", got)
	}
}

func TestGuardCodexMCPConfigOmitsFakDisabledWhenNotPresent(t *testing.T) {
	t.Setenv(guardCodexReasoningEffortEnv, "")
	dir := t.TempDir()
	t.Setenv("CODEX_HOME", dir)
	cfg := filepath.Join(dir, "config.toml")
	content := `[mcp_servers.other]
command = "node"
`
	if err := os.WriteFile(cfg, []byte(content), 0o600); err != nil {
		t.Fatal(err)
	}

	got := guardCodexConfigArgs("http://127.0.0.1:8137", "", "", dir)
	if containsArg(got, "mcp_servers.fak.enabled=false") {
		t.Fatalf("unexpected mcp_servers.fak.enabled=false when [mcp_servers.fak] is absent: %v", got)
	}
}

// installGuardCodexConfig is the rewrite that makes `fak guard -- codex` route through the
// gateway: it inserts the `-c` overrides immediately AFTER the codex executable and BEFORE
// any subcommand (`exec`) or user args, because Codex's global `-c` flag must precede the
// subcommand. It must be inert for a non-codex agent, inert when disabled, inert on an empty
// command, and it must report what it did in the guardCodexInstall struct for the banner.
func TestInstallGuardCodexConfigCodexOnlyRewrite(t *testing.T) {
	const gw = "http://127.0.0.1:8137"
	t.Setenv(guardCodexReasoningEffortEnv, "")

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
		// The struct the banner reads is fully populated (#10669: the resolved effort, and
		// that it is the configured default rather than an operator opt-in).
		if info.ProviderID != "fak" || info.EnvKey != "OPENAI_API_KEY" || info.BaseURL != gw+"/v1" ||
			info.Model != "gpt-6-astra" || info.Reasoning != "medium" || info.ReasoningOptIn {
			t.Errorf("guardCodexInstall fields = %+v, want provider=fak env=OPENAI_API_KEY base=%s/v1 model=gpt-6-astra effort=medium not-opt-in", info, gw)
		}
	})

	t.Run("no subcommand still inserts overrides after the executable", func(t *testing.T) {
		out, info := installGuardCodexConfig([]string{"codex"}, true, gw, "")
		if !info.Applied || out[0] != "codex" || !containsArg(out, "model_provider=fak") {
			t.Errorf("bare `codex` not rewritten: out=%v info=%+v", out, info)
		}
	})

	t.Run("authentication management commands stay direct", func(t *testing.T) {
		for _, in := range [][]string{
			{"codex", "login"},
			{"codex", "login", "status"},
			{"codex", "logout"},
		} {
			t.Run(strings.Join(in[1:], "_"), func(t *testing.T) {
				out, info := installGuardCodexConfig(in, true, gw, "")
				if info.Applied || !equalArgs(out, in) {
					t.Errorf("auth management command must stay direct: out=%v info=%+v", out, info)
				}
			})
		}
	})

	t.Run("similar command remains guarded", func(t *testing.T) {
		in := []string{"codex", "login-extra"}
		out, info := installGuardCodexConfig(in, true, gw, "")
		if !info.Applied || !containsArg(out, "model_provider=fak") {
			t.Errorf("non-auth command must remain repointed: out=%v info=%+v", out, info)
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

func TestGuardCodexAuthManagementCommand(t *testing.T) {
	for _, tc := range []struct {
		name    string
		command []string
		want    bool
	}{
		{name: "login", command: []string{"codex", "login"}, want: true},
		{name: "login status", command: []string{"codex", "login", "status"}, want: true},
		{name: "logout", command: []string{"codex", "logout"}, want: true},
		{name: "bare", command: []string{"codex"}},
		{name: "exec", command: []string{"codex", "exec", "task"}},
		{name: "login prefix", command: []string{"codex", "login-extra"}},
		{name: "login extra arg", command: []string{"codex", "login", "status", "extra"}},
		{name: "other executable", command: []string{"other", "login"}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if got := guardCodexAuthManagementCommand(tc.command); got != tc.want {
				t.Fatalf("guardCodexAuthManagementCommand(%v) = %v, want %v", tc.command, got, tc.want)
			}
		})
	}
}
func TestGuardCodexAuthEnv(t *testing.T) {
	applied := guardCodexInstall{Applied: true, ProviderID: "fak", EnvKey: "OPENAI_API_KEY", BaseURL: "http://127.0.0.1:1/v1"}

	t.Run("disabled install needs no env", func(t *testing.T) {
		env, err := guardCodexAuthEnv(guardCodexInstall{}, "", false, nil)
		if err != nil || len(env) != 0 {
			t.Fatalf("guardCodexAuthEnv disabled = env=%v err=%v, want no-op", env, err)
		}
	})

	t.Run("resolved upstream key is explicit child grant", func(t *testing.T) {
		env, err := guardCodexAuthEnv(applied, "sk-test", false, nil)
		if err != nil {
			t.Fatalf("guardCodexAuthEnv resolved key: %v", err)
		}
		if len(env) != 1 || env[0] != [2]string{"OPENAI_API_KEY", "sk-test"} {
			t.Fatalf("env grant = %v, want OPENAI_API_KEY=sk-test", env)
		}
	})

	t.Run("chatgpt subscription keeps upstream token in guard", func(t *testing.T) {
		chatgpt := applied
		chatgpt.AuthMode = "chatgpt"
		env, err := guardCodexAuthEnv(chatgpt, "oauth-access-token", false, nil)
		if err != nil {
			t.Fatalf("guardCodexAuthEnv chatgpt: %v", err)
		}
		if len(env) != 1 || env[0] != [2]string{"OPENAI_API_KEY", guardCodexOAuthPlaceholderAPIKey} {
			t.Fatalf("chatgpt env = %v, want OPENAI_API_KEY placeholder", env)
		}
	})

	t.Run("ambient key can pass through", func(t *testing.T) {
		env, err := guardCodexAuthEnv(applied, "", false, func(name string) string {
			if name == "OPENAI_API_KEY" {
				return "sk-ambient"
			}
			return ""
		})
		if err != nil || len(env) != 0 {
			t.Fatalf("ambient key env = %v err=%v, want no extra grant", env, err)
		}
	})

	t.Run("local only gets placeholder", func(t *testing.T) {
		env, err := guardCodexAuthEnv(applied, "", true, func(string) string { return "" })
		if err != nil {
			t.Fatalf("local-only placeholder: %v", err)
		}
		if len(env) != 1 || env[0] != [2]string{"OPENAI_API_KEY", guardCodexLocalPlaceholderAPIKey} {
			t.Fatalf("local-only env = %v, want placeholder", env)
		}
	})

	t.Run("chatgpt subscription gets placeholder not token", func(t *testing.T) {
		chatgpt := applied
		chatgpt.AuthMode = "chatgpt"
		env, err := guardCodexAuthEnv(chatgpt, "oauth-token", false, func(string) string { return "" })
		if err != nil {
			t.Fatalf("chatgpt placeholder: %v", err)
		}
		if len(env) != 1 || env[0] != [2]string{"OPENAI_API_KEY", guardCodexOAuthPlaceholderAPIKey} {
			t.Fatalf("chatgpt env = %v, want OAuth placeholder", env)
		}
	})

	t.Run("missing api key fails before spawning codex", func(t *testing.T) {
		env, err := guardCodexAuthEnv(applied, "", false, func(string) string { return "" })
		if err == nil {
			t.Fatalf("guardCodexAuthEnv missing key returned env=%v and no error", env)
		}
		msg := err.Error()
		for _, want := range []string{"$OPENAI_API_KEY", "codex login", "--api-key-env"} {
			if !strings.Contains(msg, want) {
				t.Fatalf("missing-key error %q missing %q", msg, want)
			}
		}
	})
}

func TestResolveWindowsBatchCommandUsesCommandInterpreter(t *testing.T) {
	if runtime.GOOS != "windows" {
		t.Skip("Windows batch resolution")
	}
	got := resolveWindowsBatchCommand([]string{"codex", "exec", "-"})
	if len(got) < 5 || !strings.EqualFold(filepath.Base(got[0]), "node.exe") || filepath.Base(got[1]) != "codex.js" || got[2] != "exec" || got[3] != "-" {
		t.Fatalf("extensionless codex was not fronted by node and its npm entrypoint: %v", got)
	}
}

func TestResolveWindowsBatchCommandKeepsExplicitCommand(t *testing.T) {
	command := []string{"codex.cmd", "exec"}
	got := resolveWindowsBatchCommand(command)
	if got[0] != command[0] || got[1] != command[1] {
		t.Fatalf("explicit batch command changed: got %v want %v", got, command)
	}
}

func TestPinnedCodexChildGetsOpenAIPlaceholderOnly(t *testing.T) {
	t.Setenv("OPENAI_API_KEY", "")
	t.Setenv("ANTHROPIC_API_KEY", "")
	child := buildGuardChild([]string{"codex"}, nil, true)
	env := strings.Join(child.Env, "\n")
	if !strings.Contains(env, "OPENAI_API_KEY="+guardCodexOAuthPlaceholderAPIKey) {
		t.Fatalf("pinned Codex child must carry OPENAI_API_KEY placeholder; env=%v", child.Env)
	}
	if strings.Contains(env, "ANTHROPIC_API_KEY=fak-guard-oauth-placeholder") {
		t.Fatalf("pinned Codex child must not receive the Anthropic placeholder; env=%v", child.Env)
	}
}

// printGuardCodexNote stays silent unless the install actually applied (so the banner does
// not lie about a codex repoint for a non-codex agent), and when it applied it must name the
// provider, the responses wire, the base URL, and the credential posture.
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
	if strings.Contains(out, "not yet wired") {
		t.Fatalf("default Codex note still claims subscription auth is unwired: %s", out)
	}

	b.Reset()
	printGuardCodexNote(&b, guardCodexInstall{
		Applied:    true,
		ProviderID: "fak",
		EnvKey:     "OPENAI_API_KEY",
		BaseURL:    "http://127.0.0.1:8137/v1",
		AuthMode:   "chatgpt",
		AuthSource: "C:/tmp/codex/auth.json",
	})
	out = b.String()
	for _, want := range []string{"ChatGPT subscription", "C:/tmp/codex/auth.json", "placeholder"} {
		if !strings.Contains(out, want) {
			t.Errorf("chatgpt Codex note missing %q\n got: %s", want, out)
		}
	}
}

func TestResolveNodeBatchCommandFromPathSkipsManagedCodexWrapper(t *testing.T) {
	managed := t.TempDir()
	npm := t.TempDir()
	if err := os.WriteFile(filepath.Join(managed, "codex.cmd"), []byte("@fak-launch launch codex -- %*\r\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	entrypoint := filepath.Join(npm, "node_modules", "@openai", "codex", "bin", "codex.js")
	if err := os.MkdirAll(filepath.Dir(entrypoint), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(npm, "codex.cmd"), []byte("@node codex.js %*\r\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(entrypoint, []byte("// fixture\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	pathEnv := managed + string(os.PathListSeparator) + npm

	got := resolveNodeBatchCommandFromPath([]string{"codex", "--version"}, filepath.Join("node_modules", "@openai", "codex", "bin", "codex.js"), "node-fixture", pathEnv)
	if len(got) != 3 {
		t.Fatalf("resolved argv = %#v, want node + npm entrypoint + args", got)
	}
	if got[1] != entrypoint {
		t.Fatalf("entrypoint = %q, want npm entrypoint %q (managed wrapper must be skipped)", got[1], entrypoint)
	}
	if got[2] != "--version" {
		t.Fatalf("forwarded arg = %q, want --version", got[2])
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

func TestGuardCodexProtectedPaths(t *testing.T) {
	for _, p := range []string{
		".git", ".git/HEAD", "repo/.git/config",
		".agents", ".agents/skills", "path/to/.agents/memories",
		".codex", ".codex/config.toml", "home/.codex/sessions",
	} {
		if !isGuardCodexProtectedPath(p) {
			t.Errorf("isGuardCodexProtectedPath(%q) = false, want true", p)
		}
	}
	for _, p := range []string{
		"main.go", "src/file.txt", "git_helper.go", "agents.md", "codex.go",
	} {
		if isGuardCodexProtectedPath(p) {
			t.Errorf("isGuardCodexProtectedPath(%q) = true, want false", p)
		}
	}
}

func TestGuardCodexSandboxProtectedPathContainment(t *testing.T) {
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git unavailable")
	}

	root := t.TempDir()
	repo := filepath.Join(root, "repo")
	if err := os.MkdirAll(repo, 0o755); err != nil {
		t.Fatal(err)
	}

	runGit := func(dir string, args ...string) string {
		t.Helper()
		cmd := exec.Command("git", append([]string{"-C", dir}, args...)...)
		cmd.Env = append(os.Environ(),
			"GIT_AUTHOR_NAME=codex", "GIT_AUTHOR_EMAIL=codex@example.com",
			"GIT_COMMITTER_NAME=codex", "GIT_COMMITTER_EMAIL=codex@example.com")
		out, err := cmd.CombinedOutput()
		if err != nil {
			t.Fatalf("git %s: %v: %s", strings.Join(args, " "), err, out)
		}
		return strings.TrimSpace(string(out))
	}

	runGit(repo, "init", "-q", "-b", "main")
	runGit(repo, "config", "user.email", "codex@example.com")
	runGit(repo, "config", "user.name", "codex")

	fileA := filepath.Join(repo, "fileA.txt")
	if err := os.WriteFile(fileA, []byte("initial line\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	runGit(repo, "add", "fileA.txt")
	runGit(repo, "commit", "-qm", "feat: initial commit")

	agentsSkill := filepath.Join(repo, ".agents", "skills", "audit", "SKILL.md")
	if err := os.MkdirAll(filepath.Dir(agentsSkill), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(agentsSkill, []byte("# Audit Skill\n"), 0o644); err != nil {
		t.Fatal(err)
	}

	codexCfg := filepath.Join(repo, ".codex", "config.toml")
	if err := os.MkdirAll(filepath.Dir(codexCfg), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(codexCfg, []byte("model = \"gpt-6-astra\"\n"), 0o644); err != nil {
		t.Fatal(err)
	}

	session, err := NewGuardCodexSandboxSession(repo, true)
	if err != nil {
		t.Fatalf("NewGuardCodexSandboxSession: %v", err)
	}
	t.Cleanup(func() { _ = session.Close() })

	ctx := context.Background()

	// 1. Test git status inside sandboxed session
	statusRes, err := session.ExecuteCommand(ctx, []string{"git", "status"})
	if err != nil {
		t.Fatalf("git status under sandbox returned error: %v", err)
	}
	if statusRes.ExitCode != 0 {
		t.Fatalf("git status exit code = %d, want 0; stderr=%s", statusRes.ExitCode, statusRes.Stderr)
	}
	if !statusRes.Proxied {
		t.Fatalf("git status was not proxied via containment shim")
	}

	// 2. Modify file and verify git diff and git status reflect changes under sandbox
	if err := os.WriteFile(fileA, []byte("initial line\nmodified line\n"), 0o644); err != nil {
		t.Fatal(err)
	}

	diffRes, err := session.ExecuteCommand(ctx, []string{"git", "diff"})
	if err != nil {
		t.Fatalf("git diff under sandbox returned error: %v", err)
	}
	if diffRes.ExitCode != 0 {
		t.Fatalf("git diff exit code = %d, want 0; stderr=%s", diffRes.ExitCode, diffRes.Stderr)
	}
	if !diffRes.Proxied {
		t.Fatalf("git diff was not proxied via containment shim")
	}
	if !strings.Contains(diffRes.Stdout, "+modified line") {
		t.Fatalf("git diff output missing modified line: %s", diffRes.Stdout)
	}

	// 3. Test git log inside sandboxed session
	logRes, err := session.ExecuteCommand(ctx, []string{"git", "log", "-n", "1"})
	if err != nil {
		t.Fatalf("git log under sandbox returned error: %v", err)
	}
	if logRes.ExitCode != 0 {
		t.Fatalf("git log exit code = %d, want 0; stderr=%s", logRes.ExitCode, logRes.Stderr)
	}
	if !logRes.Proxied {
		t.Fatalf("git log was not proxied via containment shim")
	}
	if !strings.Contains(logRes.Stdout, "initial commit") {
		t.Fatalf("git log output missing initial commit: %s", logRes.Stdout)
	}

	// 4. Test protected path reads for .git and agent state (.agents, .codex)
	headBytes, err := session.ReadProtected(".git/HEAD")
	if err != nil {
		t.Fatalf("ReadProtected(.git/HEAD): %v", err)
	}
	if !strings.Contains(string(headBytes), "refs/heads/") && len(headBytes) == 0 {
		t.Fatalf("unexpected .git/HEAD content: %s", string(headBytes))
	}

	skillBytes, err := session.ReadProtected(".agents/skills/audit/SKILL.md")
	if err != nil {
		t.Fatalf("ReadProtected(.agents/skills/audit/SKILL.md): %v", err)
	}
	if !strings.Contains(string(skillBytes), "Audit Skill") {
		t.Fatalf("unexpected skill content: %s", string(skillBytes))
	}

	codexCfgBytes, err := session.ReadProtected(".codex/config.toml")
	if err != nil {
		t.Fatalf("ReadProtected(.codex/config.toml): %v", err)
	}
	if !strings.Contains(string(codexCfgBytes), "gpt-6-astra") {
		t.Fatalf("unexpected codex config content: %s", string(codexCfgBytes))
	}

	// 5. Verify sandbox isolation simulation:
	// Even if host .git is locked/inaccessible (simulating strict sandbox where raw .git access trips error/panic),
	// the containment proxy enables inspection commands to succeed.
	rawGitDir := filepath.Join(repo, ".git")
	blockedGitDir := filepath.Join(root, "repo_git_blocked")
	if err := os.Rename(rawGitDir, blockedGitDir); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if _, err := os.Stat(rawGitDir); os.IsNotExist(err) {
			_ = os.Rename(blockedGitDir, rawGitDir)
		}
	})

	isolatedStatus, err := session.ExecuteCommand(ctx, []string{"git", "status"})
	if err != nil || isolatedStatus.ExitCode != 0 {
		t.Fatalf("git status failed under strict sandbox isolation without raw .git: exit=%d err=%v stderr=%s",
			isolatedStatus.ExitCode, err, isolatedStatus.Stderr)
	}
	if !isolatedStatus.Proxied {
		t.Fatalf("isolated git status was not proxied via containment shim")
	}

	// 6. Verify SafeSandboxAccess catches access violation panics
	err = session.SafeSandboxAccess(".git", func() error {
		panic("sandbox access violation on protected path .git: permission denied")
	})
	if err != nil {
		t.Fatalf("SafeSandboxAccess returned error for contained violation: %v", err)
	}
}
