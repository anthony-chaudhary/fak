package main

// `fak node` — durable, self-contained tooling for setting up and connecting to an
// always-on fak serve gateway. Replaces the shell scripts in tools/ and scripts/
// that required sourcing, chmod, platform-specific quoting, and external sed/PlistBuddy.
//
// Subcommands:
//
//	fak node install [--remote] [--addr ADDR] [--port N] [--key-env VAR] [--uninstall]
//	                 [--base-url URL] [--provider NAME] [--model ID] [--env NAME=VALUE]
//	                           Install the fak serve gateway as a system service on this
//	                           machine. macOS: launchd KeepAlive agent with caffeinate
//	                           wrapper. Linux: systemd --user unit. Windows: an ONSTART
//	                           Scheduled Task. Prints client env lines after install.
//	                           --base-url/--provider/--model/--env select the UPSTREAM the
//	                           installed unit proxies, so the SAME installer produces both
//	                           the default Anthropic adjudication proxy and a local-model
//	                           gateway (#5555) instead of only the former.
//	fak node status            Show service state + gateway health (no flags).
//	fak node use HOST[:PORT] [--key KEY] [--env] [--no-check]
//	                           Write ~/.config/fak/node.json and print the two export
//	                           lines to paste in your shell / CLAUDE.md. --env skips
//	                           the write and just prints the lines. By default probes
//	                           GET <url>/healthz and warns (without blocking) if the
//	                           node is unreachable; --no-check skips the probe.
//	fak node run -- CMD [ARGS...]
//	                           Launch CMD with ANTHROPIC_BASE_URL (and ANTHROPIC_API_KEY,
//	                           when a key is configured) pointed at the node from
//	                           ~/.config/fak/node.json — e.g. `fak node run -- claude`.
//	                           Consumes the config `use` writes; exits with the child's
//	                           status. Requires a prior `fak node use`.
//	fak node forget            Clear ~/.config/fak/node.json (undo `use`).

import (
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"time"
)

// nodeDefaultPolicyJSON is a copy of examples/dogfood-claude-policy.json baked in
// at compile time. go:embed cannot traverse parent dirs, so we inline it here.
// Keep in sync with examples/dogfood-claude-policy.json when the policy evolves.
var nodeDefaultPolicyJSON = []byte(`{
  "version": "fak-policy/v1",

  "allow": [
    "Bash",
    "BashOutput",
    "KillShell",
    "Read",
    "Edit",
    "Write",
    "NotebookEdit",
    "Glob",
    "Grep",
    "LS",
    "TodoWrite",
    "Task",
    "WebFetch",
    "WebSearch",
    "ExitPlanMode",
    "Skill",
    "SlashCommand"
  ],

  "allow_prefix": [
    "read_",
    "get_",
    "search_",
    "list_",
    "lookup_",
    "find_"
  ],

  "arg_rules": [
    { "tool": "Bash", "arg": "command", "deny_regex": "\\brm\\s+-[A-Za-z]*[rRfF]", "reason": "POLICY_BLOCK" },
    { "tool": "Bash", "arg": "command", "deny_regex": "\\bsudo\\b", "reason": "POLICY_BLOCK" },
    { "tool": "Bash", "arg": "command", "deny_regex": "\\bmkfs\\b|\\bdd\\s+if=|>\\s*/dev/sd", "reason": "POLICY_BLOCK" },
    { "tool": "Bash", "arg": "command", "deny_regex": ":\\(\\)\\s*\\{", "reason": "POLICY_BLOCK" },
    { "tool": "Bash", "arg": "command", "deny_regex": "\\b(curl|wget)\\b[^|]*\\|\\s*(sudo\\s+)?(ba)?sh\\b", "reason": "POLICY_BLOCK" },
    { "tool": "Bash", "arg": "command", "deny_regex": "\\bgit\\s+push\\b", "reason": "POLICY_BLOCK" },
    { "tool": "Bash", "arg": "command", "deny_regex": "-o\\s+\\.\\.[\\\\/]", "reason": "POLICY_BLOCK" },
    { "tool": "Bash", "arg": "command", "deny_regex": "--output[= ]\\s*\\.\\.[\\\\/]", "reason": "POLICY_BLOCK" },
    { "tool": "Bash", "arg": "command", "deny_regex": ">>?\\s*\\.\\.[\\\\/]", "reason": "POLICY_BLOCK" },
    { "tool": "Bash", "arg": "command", "deny_regex": "\\b(cp|mv|install|tee|rsync|ln)\\b[^|;&]*\\s\\.\\.[\\\\/]", "reason": "POLICY_BLOCK" }
  ],

  "self_modify_globs": [
    "internal/abi/",
    "internal/kernel/",
    "internal/adjudicator/",
    "internal/policy/",
    "internal/registrations/",
    ".git/",
    ".dos/",
    "VERSION",
    "id_rsa",
    "/etc/"
  ],

  "redact_fields": [
    "password",
    "secret",
    "api_key",
    "token",
    "authorization"
  ]
}`)

// nodeGatewayLabel is the launchd / systemd service name.
//
// ONE label, deliberately (#5555, shape 2). The label names "the fak serve gateway on
// this host", and the installer — not a second hand-written artifact — is what decides
// which UPSTREAM that one gateway proxies (--base-url / --provider / --model / --env).
// The alternative shape, deriving a per-upstream label so two gateways can load at once,
// was rejected: two gateways also contend for one port, one node-install.json install
// record, and one client-side node.json, so coexistence is a much larger change than a
// label suffix — and it would STILL need these upstream flags to be reachable at all.
// Keeping one label means `--uninstall` and `status` continue to name exactly the unit
// that exists, with no stale-label discovery path to get wrong.
const nodeGatewayLabel = "com.fak.serve-gateway"

// nodeWindowsTaskName is the Scheduled Task name the Windows install registers.
const nodeWindowsTaskName = "FakServeGateway"

// Upstream defaults for the installed unit. The historical install pinned these two
// values into every rendered unit; they are now the DEFAULTS of --provider/--base-url
// so the unchanged invocation renders byte-identical argv to what it always did.
const (
	nodeDefaultProvider    = "anthropic"
	nodeDefaultUpstreamURL = "https://api.anthropic.com"
	// nodeLocalWireProvider is the provider assumed when an operator repoints --base-url
	// without naming a wire: an OpenAI-compatible /v1 endpoint, which is what llama-server,
	// vLLM, LM Studio and Ollama all speak. Chosen as the default over "anthropic" because
	// it is the CREDENTIAL-SAFE direction — see nodeResolveUpstream.
	nodeLocalWireProvider = "openai"
)

// nodeUpstream is the resolved upstream the installed unit proxies: the wire, the base
// URL, and the model id to advertise. It is the only part of the unit that differs
// between the Anthropic adjudication proxy and a local-model gateway, which is why it
// is a value threaded into the pure renderers rather than three template literals.
type nodeUpstream struct {
	Provider string
	BaseURL  string
	Model    string // empty ⇒ --model is omitted and `fak serve` keeps its own default
}

// nodeEnvVar is one extra EnvironmentVariables entry for the installed unit — the
// launchd `EnvironmentVariables` dict, the systemd `Environment=` lines, or the Windows
// runner's `set` lines. A local-model gateway needs these (long-turn timeouts, provider
// extra-body tuning), and before #5555 there was no way to pass them.
type nodeEnvVar struct{ Name, Value string }

// nodeEnvFlag is the repeatable --env NAME=VALUE flag.
//
// String() reports the NAMES only and never a value: the flag package calls String() to
// render defaults in `--help`, and an installer that echoed the value of every --env
// would print whatever an operator passed there into terminal scrollback and CI logs.
type nodeEnvFlag []nodeEnvVar

func (f *nodeEnvFlag) String() string {
	if f == nil || len(*f) == 0 {
		return ""
	}
	names := make([]string, 0, len(*f))
	for _, e := range *f {
		names = append(names, e.Name)
	}
	return strings.Join(names, ",")
}

func (f *nodeEnvFlag) Set(s string) error {
	name, value, ok := strings.Cut(s, "=")
	name = strings.TrimSpace(name)
	if !ok || name == "" {
		return fmt.Errorf("want NAME=VALUE (the name only is reported back; the value is never echoed)")
	}
	if !nodeValidEnvName(name) {
		return fmt.Errorf("%q is not a valid environment variable name (want [A-Za-z_][A-Za-z0-9_]*)", name)
	}
	if name == "FAK_AUDIT_JOURNAL" {
		return fmt.Errorf("FAK_AUDIT_JOURNAL is set by the installer itself; drop the --env entry")
	}
	for _, e := range *f {
		if e.Name == name {
			return fmt.Errorf("--env %s given twice", name)
		}
	}
	*f = append(*f, nodeEnvVar{Name: name, Value: value})
	return nil
}

// nodeValidEnvName reports whether s is a POSIX-shaped environment variable name. The
// name lands verbatim in a plist key / a systemd Environment= line / a cmd `set`, so a
// name carrying a quote, a newline, or an `=` would corrupt the unit rather than fail.
func nodeValidEnvName(s string) bool {
	if s == "" {
		return false
	}
	for i, r := range s {
		switch {
		case r == '_':
		case r >= 'A' && r <= 'Z':
		case r >= 'a' && r <= 'z':
		case r >= '0' && r <= '9' && i > 0:
		default:
			return false
		}
	}
	return true
}

// nodeResolveUpstream resolves --provider/--base-url/--model into the upstream the
// rendered unit proxies, and is where this installer's CREDENTIAL EGRESS policy lives.
//
// What the installed unit actually carries, and where it can go:
//
//   - The gateway's own inbound bearer (--key-env, default FAK_GATEWAY_KEY) is a DOOR
//     key. `fak serve` compares it to the inbound x-api-key / Bearer and never sends it
//     anywhere. It is separable from — and is not — the upstream provider credential.
//   - The upstream credential is NOT installed at all: the unit never passes
//     `fak serve --api-key-env`, so the gateway holds no upstream key of its own. On the
//     ANTHROPIC wire only, the gateway forwards the CALLER's inbound key upstream (the
//     transparent hop: gateway.anthropicUpstreamCredential → anthropicInboundKey, reached
//     only when the live planner's provider is anthropic). On every other wire the
//     caller's credential is never forwarded; the request body still is.
//
// So `--base-url` is exactly a "point a credential-bearing unit at another host" flag on
// the anthropic wire, and a "point the prompts at another host" flag everywhere else.
// The bound this function enforces:
//
//  1. --base-url with no --provider defaults to the OpenAI-compatible wire, not anthropic
//     — the direction in which no caller credential is forwarded.
//  2. The anthropic wire may only be pointed at Anthropic itself or at LOOPBACK. A remote
//     non-Anthropic host on the anthropic wire is REFUSED, because that combination sends
//     the caller's Anthropic key to a host of the flag's choosing. Loopback stays allowed
//     (a local anthropic-wire shim); the credential still never leaves the machine.
//  3. Any other non-loopback upstream is allowed but WARNED, naming the host, because
//     prompts and transcripts egress there — and warned again when the scheme is plain
//     http, which puts them on the wire in the clear.
//
// It returns the resolved upstream plus operator-facing warnings; an error means refuse.
func nodeResolveUpstream(providerFlag, baseURLFlag, modelFlag string) (nodeUpstream, []string, error) {
	provider := strings.ToLower(strings.TrimSpace(providerFlag))
	baseURL := strings.TrimSpace(baseURLFlag)
	model := strings.TrimSpace(modelFlag)

	if baseURL == "" {
		baseURL = nodeDefaultUpstreamURL
		if provider == "" {
			provider = nodeDefaultProvider
		}
	} else if provider == "" {
		// An operator who repointed the upstream without naming a wire gets the
		// OpenAI-compatible one: it is what local model servers speak, and it is the
		// arm that forwards no caller credential.
		provider = nodeLocalWireProvider
		if nodeIsAnthropicUpstream(baseURL) {
			provider = nodeDefaultProvider
		}
	}

	switch provider {
	case "openai", "anthropic", "gemini", "xai":
	default:
		return nodeUpstream{}, nil, fmt.Errorf("--provider %q is not an upstream wire (want openai, anthropic, gemini, or xai)", provider)
	}

	host, scheme, err := nodeUpstreamHost(baseURL)
	if err != nil {
		return nodeUpstream{}, nil, fmt.Errorf("--base-url %q: %w", baseURL, err)
	}
	loopback := nodeHostIsLoopback(host)

	if provider == nodeDefaultProvider && !loopback && !nodeIsAnthropicUpstream(baseURL) {
		return nodeUpstream{}, nil, fmt.Errorf(
			"refusing to install --provider anthropic against --base-url %q: on the anthropic wire this gateway forwards the CALLER's own API key upstream, so that host would receive it. Point --base-url at %s, keep it on loopback, or name the wire the other endpoint actually speaks (e.g. --provider openai for an OpenAI-compatible /v1 server)",
			baseURL, nodeDefaultUpstreamURL)
	}

	var warns []string
	if !loopback && !nodeIsAnthropicUpstream(baseURL) {
		warns = append(warns, fmt.Sprintf("upstream %s is NOT on this machine — every prompt and tool transcript this gateway proxies is sent to that host", host))
		if scheme == "http" {
			warns = append(warns, fmt.Sprintf("upstream %s uses plain http — those prompts and transcripts cross the network unencrypted", baseURL))
		}
	}
	return nodeUpstream{Provider: provider, BaseURL: baseURL, Model: model}, warns, nil
}

// nodeUpstreamHost extracts the host and scheme of an upstream base URL, rejecting the
// shapes that would silently install a broken (or differently-targeted) unit: a missing
// scheme, a non-http(s) scheme, or an empty host.
func nodeUpstreamHost(baseURL string) (host, scheme string, err error) {
	u, err := url.Parse(baseURL)
	if err != nil {
		// A bare "host:port" fails to parse here, so carry the same remedy the
		// missing-scheme branch below gives rather than leaking a bare parser message.
		return "", "", fmt.Errorf("want an absolute http:// or https:// URL (%v)", err)
	}
	scheme = strings.ToLower(u.Scheme)
	if scheme != "http" && scheme != "https" {
		return "", "", errors.New("want an absolute http:// or https:// URL")
	}
	if u.Hostname() == "" {
		return "", "", errors.New("no host in URL")
	}
	return u.Hostname(), scheme, nil
}

// nodeIsAnthropicUpstream reports whether a base URL names Anthropic's own API — the one
// destination the anthropic wire may forward a caller's key to off this machine.
func nodeIsAnthropicUpstream(baseURL string) bool {
	host, _, err := nodeUpstreamHost(baseURL)
	if err != nil {
		return false
	}
	host = strings.ToLower(host)
	return host == "anthropic.com" || strings.HasSuffix(host, ".anthropic.com")
}

// nodeCfg is the on-disk config written by `fak node use`.
type nodeCfg struct {
	URL string `json:"url"`
	Key string `json:"key,omitempty"`
}

func cmdNode(argv []string) { os.Exit(runNode(os.Stdout, os.Stderr, argv)) }

func runNode(stdout, stderr io.Writer, argv []string) int {
	if len(argv) == 0 {
		fmt.Fprintln(stderr, "usage: fak node <install|status|use|run|forget> [flags]")
		fmt.Fprintln(stderr, "       fak node install --help")
		return 2
	}
	switch argv[0] {
	case "install":
		return nodeInstall(stdout, stderr, argv[1:])
	case "status":
		return nodeStatus(stdout, stderr, argv[1:])
	case "use":
		return nodeUse(stdout, stderr, argv[1:])
	case "run":
		return nodeRun(stdout, stderr, argv[1:])
	case "forget":
		return nodeForget(stdout, stderr)
	default:
		fmt.Fprintf(stderr, "fak node: unknown subcommand %q (want install|status|use|run|forget)\n", argv[0])
		return 2
	}
}

// ── status ────────────────────────────────────────────────────────────────────

func nodeStatus(stdout, stderr io.Writer, _ []string) int {
	rc := 0

	// Service state.
	switch runtime.GOOS {
	case "darwin":
		out, _ := exec.Command("launchctl", "list", nodeGatewayLabel).Output()
		if len(out) > 0 {
			fmt.Fprintf(stdout, "[fak node] launchd: %s\n", strings.TrimSpace(string(out)))
		} else {
			fmt.Fprintf(stdout, "[fak node] launchd: %s not loaded\n", nodeGatewayLabel)
			rc = 1
		}
	case "linux":
		out, _ := exec.Command("systemctl", "--user", "status", "fak-serve-gateway", "--no-pager").Output()
		fmt.Fprintf(stdout, "[fak node] systemd:\n%s\n", string(out))
	case "windows":
		out, _ := nodeHelperCommand("schtasks", "/Query", "/TN", nodeWindowsTaskName, "/FO", "LIST").Output()
		if len(out) > 0 {
			fmt.Fprintf(stdout, "[fak node] schtasks %s:\n%s\n", nodeWindowsTaskName, strings.TrimSpace(string(out)))
		} else {
			fmt.Fprintf(stdout, "[fak node] schtasks: %s not installed\n", nodeWindowsTaskName)
			rc = 1
		}
	}

	// Gateway health — probe the loopback URL for the port we ACTUALLY installed with, read
	// from the persisted install state, instead of the literal :8080 that false-reported a
	// custom-port gateway as down (#1). Fall back to :8080 only when no install state exists
	// (e.g. status run on a client that never installed). Then also probe the node config URL.
	// nodeProbeHealth takes a base URL and appends /healthz itself, so the candidates are bases.
	loopback := "http://127.0.0.1:8080"
	if cfgDir, derr := nodeConfigDir(); derr == nil {
		if st, serr := nodeReadInstallState(cfgDir); serr == nil && st.Port != "" {
			loopback = "http://127.0.0.1:" + st.Port
		}
	}
	candidates := []string{loopback}
	if cfg, err := nodeReadCfg(); err == nil && cfg.URL != "" {
		candidates = append(candidates, cfg.URL)
	}
	for _, base := range candidates {
		status, ok := nodeProbeHealth(base)
		fmt.Fprintf(stdout, "[fak node] healthz %s/healthz: %s\n", strings.TrimRight(base, "/"), status)
		if !ok {
			rc = 1
		}
	}

	// Node config.
	cfg, err := nodeReadCfg()
	if err == nil && cfg.URL != "" {
		fmt.Fprintf(stdout, "[fak node] configured remote: %s", cfg.URL)
		if cfg.Key != "" {
			fmt.Fprintf(stdout, " (key set)")
		}
		fmt.Fprintln(stdout)
	}
	return rc
}

// ── use ───────────────────────────────────────────────────────────────────────

func nodeUse(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("node use", flag.ContinueOnError)
	fs.SetOutput(stderr)
	key := fs.String("key", "", "bearer key for an authenticated gateway")
	envOnly := fs.Bool("env", false, "just print the export lines; don't write config")
	noCheck := fs.Bool("no-check", false, "skip the GET /healthz reachability probe")
	// Go's flag package stops at the first non-flag token, so `fak node use host --key k`
	// would silently drop the flags after the positional HOST. Parse in passes that let
	// flag itself handle flag VALUES (so `--key k` is never mistaken for the host): each
	// Parse consumes the flags up to the next positional; we take the first positional as
	// the host and keep parsing the tail until no args remain.
	var host string
	args := argv
	for {
		if err := fs.Parse(args); err != nil {
			return 2
		}
		if fs.NArg() == 0 {
			break
		}
		if host == "" {
			host = fs.Arg(0)
		}
		args = fs.Args()[1:]
	}
	if host == "" {
		fmt.Fprintln(stderr, "usage: fak node use HOST[:PORT] [--key KEY] [--env] [--no-check]")
		return 2
	}
	u := host
	if !strings.Contains(u, "://") {
		u = "http://" + u
	}
	// Default the port to 8080 (the documented `fak serve` addr) only for an http:// URL
	// with no explicit port — the loopback / tailnet case. An https:// node is assumed to
	// sit behind TLS on its own port (443 by default), so it is never given :8080.
	if rest, ok := strings.CutPrefix(u, "http://"); ok && !strings.Contains(rest, ":") {
		u += ":8080"
	}
	cfg := nodeCfg{URL: u, Key: *key}

	// Reachability preflight: a node that is down or rejecting the key at config time is
	// the most common surprise, and a warning here beats a 502 on the client's first
	// turn. It never blocks — the node may legitimately be off when you configure it — so
	// the config is still written and exit stays 0.
	if !*noCheck {
		if status, ok := nodeProbeHealth(u); !ok {
			fmt.Fprintf(stderr, "[fak node] WARNING: %s not reachable (%s) — config written anyway; start it with `fak node install` on the host\n", u, status)
		} else {
			fmt.Fprintf(stdout, "[fak node] %s healthy (%s)\n", u, status)
		}
	}

	if !*envOnly {
		if err := nodeWriteCfg(cfg); err != nil {
			fmt.Fprintf(stderr, "fak node use: %v\n", err)
			return 1
		}
		cfgDir, _ := nodeConfigDir()
		fmt.Fprintf(stdout, "[fak node] wrote %s/node.json\n", cfgDir)
		fmt.Fprintln(stdout, "")
	}

	fmt.Fprintln(stdout, "Add to your shell (or run inline for a single session):")
	fmt.Fprintln(stdout, "")
	fmt.Fprintf(stdout, "  bash/zsh:\n")
	fmt.Fprintf(stdout, "    export ANTHROPIC_BASE_URL=\"%s\"\n", u)
	if *key != "" {
		fmt.Fprintf(stdout, "    export ANTHROPIC_API_KEY=\"%s\"\n", *key)
	}
	fmt.Fprintln(stdout, "")
	fmt.Fprintf(stdout, "  PowerShell:\n")
	fmt.Fprintf(stdout, "    $env:ANTHROPIC_BASE_URL = \"%s\"\n", u)
	if *key != "" {
		fmt.Fprintf(stdout, "    $env:ANTHROPIC_API_KEY  = \"%s\"\n", *key)
	}
	fmt.Fprintln(stdout, "")
	fmt.Fprintf(stdout, "Then run: claude\n")
	fmt.Fprintf(stdout, "Or skip the exports: fak node run -- claude\n")
	return 0
}

// ── run ─────────────────────────────────────────────────────────────────────────

// nodeRun launches a client command with its inference pointed at the configured node —
// the consumer that makes `fak node use` more than a print statement. It reads the
// node.json `use` wrote, injects ANTHROPIC_BASE_URL (+ ANTHROPIC_API_KEY when a key is
// configured) into ONLY the child's environment, and execs the command with stdio wired
// through so an interactive agent (Claude Code) runs normally. The child's exit status is
// propagated so `fak node run -- <cmd>` is a transparent wrapper.
func nodeRun(stdout, stderr io.Writer, argv []string) int {
	if len(argv) == 0 || argv[0] == "-h" || argv[0] == "--help" {
		fmt.Fprintln(stderr, "usage: fak node run -- CMD [ARGS...]")
		fmt.Fprintln(stderr, "       launches CMD pointed at the node from `fak node use`")
		return 2
	}
	// Accept an optional leading "--" separator (the idiomatic argv boundary) so both
	// `fak node run -- claude` and `fak node run claude` work.
	cmd := argv
	if argv[0] == "--" {
		cmd = argv[1:]
	}
	if len(cmd) == 0 {
		fmt.Fprintln(stderr, "fak node run: no command given (usage: fak node run -- CMD [ARGS...])")
		return 2
	}

	cfg, err := nodeReadCfg()
	if err != nil || strings.TrimSpace(cfg.URL) == "" {
		fmt.Fprintln(stderr, "fak node run: no node configured — run `fak node use HOST[:PORT]` first")
		return 2
	}

	child := exec.Command(cmd[0], cmd[1:]...)
	child.Env = os.Environ()
	for _, kv := range nodeChildEnv(cfg) {
		child.Env = append(child.Env, kv[0]+"="+kv[1])
	}
	child.Stdin, child.Stdout, child.Stderr = os.Stdin, stdout, stderr

	keyNote := ""
	if cfg.Key != "" {
		keyNote = " (key set)"
	}
	fmt.Fprintf(stderr, "[fak node] → %s%s\n", cfg.URL, keyNote)

	if err := child.Run(); err != nil {
		// Propagate the child's own exit code when it ran but failed; otherwise the
		// command could not be launched (not found, not executable) — report and 127,
		// the conventional "command not found" status.
		var ee *exec.ExitError
		if errors.As(err, &ee) {
			return ee.ExitCode()
		}
		fmt.Fprintf(stderr, "fak node run: %v\n", err)
		return 127
	}
	return 0
}

// ── forget ────────────────────────────────────────────────────────────────────

func nodeForget(stdout, stderr io.Writer) int {
	cfgDir, err := nodeConfigDir()
	if err != nil {
		fmt.Fprintf(stderr, "fak node forget: %v\n", err)
		return 1
	}
	p := filepath.Join(cfgDir, "node.json")
	if err := os.Remove(p); err != nil {
		if os.IsNotExist(err) {
			fmt.Fprintln(stdout, "[fak node] no node config to forget")
			return 0
		}
		fmt.Fprintf(stderr, "fak node forget: %v\n", err)
		return 1
	}
	fmt.Fprintf(stdout, "[fak node] removed %s\n", p)
	return 0
}

// ── helpers ───────────────────────────────────────────────────────────────────

func nodeConfigDir() (string, error) {
	home, err := os.UserHomeDir()
	if err != nil {
		return "", fmt.Errorf("home dir: %w", err)
	}
	switch runtime.GOOS {
	case "windows":
		if d := os.Getenv("APPDATA"); d != "" {
			return filepath.Join(d, "fak"), nil
		}
	}
	return filepath.Join(home, ".config", "fak"), nil
}

func nodeReadCfg() (nodeCfg, error) {
	d, err := nodeConfigDir()
	if err != nil {
		return nodeCfg{}, err
	}
	data, err := os.ReadFile(filepath.Join(d, "node.json"))
	if err != nil {
		return nodeCfg{}, err
	}
	var cfg nodeCfg
	return cfg, json.Unmarshal(data, &cfg)
}

func nodeWriteCfg(cfg nodeCfg) error {
	d, err := nodeConfigDir()
	if err != nil {
		return err
	}
	if err := os.MkdirAll(d, 0755); err != nil {
		return err
	}
	data, err := json.MarshalIndent(cfg, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(filepath.Join(d, "node.json"), data, 0600)
}

// nodeInstallState is the on-disk record of WHAT THE HOST INSTALLED, written by `install`
// (distinct from nodeCfg, which `use` writes on the CLIENT). It lets `status` probe the real
// installed port instead of the literal :8080 (#1) and lets a re-install reuse the existing
// bearer key instead of silently rotating it and breaking every configured client (#4).
type nodeInstallState struct {
	Addr    string `json:"addr"`              // the bind address `fak serve --addr` was given
	Port    string `json:"port"`              // the local port, for the loopback health URL
	Key     string `json:"key,omitempty"`     // the generated bearer (off-host installs only)
	KeyEnv  string `json:"key_env,omitempty"` // the env var name carrying the bearer
	OffHost bool   `json:"off_host"`          // whether the install required a bearer key
}

// nodeInstallStatePath is where the host-side install state lives (next to node-policy.json
// in the config dir). It is host state, NOT the client's node.json.
func nodeInstallStatePath(cfgDir string) string {
	return filepath.Join(cfgDir, "node-install.json")
}

// nodeReadInstallState reads the host install state, or a zero value + error when none exists.
func nodeReadInstallState(cfgDir string) (nodeInstallState, error) {
	data, err := os.ReadFile(nodeInstallStatePath(cfgDir))
	if err != nil {
		return nodeInstallState{}, err
	}
	var st nodeInstallState
	return st, json.Unmarshal(data, &st)
}

// nodeWriteInstallState persists the host install state (0600 — it can carry the bearer key),
// creating the config dir if needed so a first install (or a test against a fresh temp dir)
// does not fail on a missing parent.
func nodeWriteInstallState(cfgDir string, st nodeInstallState) error {
	if err := os.MkdirAll(cfgDir, 0755); err != nil {
		return err
	}
	data, err := json.MarshalIndent(st, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(nodeInstallStatePath(cfgDir), data, 0600)
}

// nodeResolveKey resolves the bearer key for an off-host install, in priority order so a
// re-install never silently rotates the key out from under configured clients (#4):
//
//  1. an explicit env-var value (the operator passing the key in deliberately), or --rotate-key
//     ⇒ mint a fresh key (the only paths that change the key);
//  2. else the key already persisted from a prior install ⇒ REUSE it (clients keep working);
//  3. else (first install, nothing to reuse) ⇒ mint a fresh key.
//
// It returns the key and whether it was freshly minted (so the installer can flag a rotation
// loudly in its output rather than letting it pass silently — the exact #4 failure).
func nodeResolveKey(cfgDir, keyEnv string, rotate bool) (key string, minted bool, err error) {
	if env := strings.TrimSpace(os.Getenv(keyEnv)); env != "" {
		return env, false, nil // operator supplied it explicitly — honor it, not a silent rotation
	}
	if !rotate {
		if st, rerr := nodeReadInstallState(cfgDir); rerr == nil && st.Key != "" {
			return st.Key, false, nil // reuse the persisted key so configured clients keep working
		}
	}
	b := make([]byte, 32)
	if _, err := rand.Read(b); err != nil {
		return "", false, fmt.Errorf("generate key: %w", err)
	}
	return hex.EncodeToString(b), true, nil
}

// nodeChildEnv builds the base-URL (and bearer-key, when set) env pairs that point a
// client at the configured node. It is the pure core of `fak node run` — the wire fak
// node use prints by hand — so it can be unit-tested without exec'ing a child. Anthropic
// clients (Claude Code) read ANTHROPIC_BASE_URL; the configured key, when present, is the
// gateway bearer they present as ANTHROPIC_API_KEY (the `fak serve --require-key-env`
// secret an off-host install generates), so a keyed remote node Just Works.
func nodeChildEnv(cfg nodeCfg) [][2]string {
	pairs := [][2]string{{"ANTHROPIC_BASE_URL", cfg.URL}}
	if cfg.Key != "" {
		pairs = append(pairs, [2]string{"ANTHROPIC_API_KEY", cfg.Key})
	}
	return pairs
}

// nodeHTTPClient is the http.Client nodeProbeHealth uses; a test swaps its Transport for
// an in-memory RoundTripper so the /healthz probe is exercised with no real network. It
// is a package var rather than a constructed-per-call client purely to expose that seam.
var nodeHTTPClient = &http.Client{Timeout: 3 * time.Second}

// nodeProbeHealth does a single GET <url>/healthz and reports the HTTP status line plus a
// reachable bit. "ok" means a live gateway answered with 2xx — the wire is up AND any
// required bearer was accepted. A connection-level failure (box down, wrong port) returns
// the error text as the status with ok=false. It is shared by `use` (a non-blocking
// warning at config time) and `status` (the live health line), so both agree on what
// "reachable" means and there is one probe to maintain.
func nodeProbeHealth(url string) (status string, ok bool) {
	resp, err := nodeHTTPClient.Get(strings.TrimRight(url, "/") + "/healthz") //nolint:noctx
	if err != nil {
		return err.Error(), false
	}
	defer resp.Body.Close()
	return resp.Status, resp.StatusCode >= 200 && resp.StatusCode < 300
}

// nodeTailscaleIP returns the machine's Tailscale IPv4 address, or "" if unavailable.
func nodeTailscaleIP() string {
	out, err := exec.Command("tailscale", "ip", "-4").Output()
	if err != nil {
		return ""
	}
	return strings.TrimSpace(strings.Split(string(out), "\n")[0])
}
