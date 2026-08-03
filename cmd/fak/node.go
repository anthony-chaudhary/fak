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
	"net"
	"net/http"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"text/template"
	"time"

	"github.com/anthony-chaudhary/fak/internal/windowgate"
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

// ── install ──────────────────────────────────────────────────────────────────

func nodeInstall(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("node install", flag.ContinueOnError)
	fs.SetOutput(stderr)
	remote := fs.Bool("remote", false, "bind to 0.0.0.0 (off-host access); auto-generates bearer key")
	addr := fs.String("addr", "", "gateway bind address (default 127.0.0.1:8080, or 0.0.0.0:8080 with --remote)")
	port := fs.Int("port", 8080, "gateway port")
	keyEnv := fs.String("key-env", "FAK_GATEWAY_KEY", "env var name for the bearer key (only used with --remote or non-loopback --addr)")
	uninstall := fs.Bool("uninstall", false, "remove the gateway service")
	rotateKey := fs.Bool("rotate-key", false, "mint a fresh bearer key even if one is already persisted (off-host installs reuse the existing key by default so clients keep working)")
	baseURL := fs.String("base-url", "", "UPSTREAM base URL the installed gateway proxies (default "+nodeDefaultUpstreamURL+"). A local-model gateway is --base-url http://127.0.0.1:8131/v1. On the anthropic wire this gateway forwards the CALLER's own key upstream, so an anthropic-wire --base-url is accepted only for Anthropic itself or a loopback address")
	provider := fs.String("provider", "", "upstream wire: openai, anthropic, gemini, or xai (default anthropic; with a non-Anthropic --base-url the default is openai, the wire local model servers speak and the one that forwards no caller credential)")
	model := fs.String("model", "", "model id the gateway advertises and asks the upstream for (e.g. qwen3.6-27b); empty keeps `fak serve`'s own default")
	var envVars nodeEnvFlag
	fs.Var(&envVars, "env", "extra environment entry for the installed unit as NAME=VALUE; repeat for N entries (e.g. --env FAK_PLANNER_TIMEOUT_S=1800). The NAME is echoed back, never the value — do not pass an upstream secret here; the unit reads its own bearer from --key-env")
	fs.Usage = func() { nodeInstallUsage(stderr, fs) }
	if code, ok := parseFlagsOrHelp(fs, argv); !ok {
		return code
	}

	upstream, upstreamWarns, err := nodeResolveUpstream(*provider, *baseURL, *model)
	if err != nil {
		fmt.Fprintf(stderr, "fak node install: %v\n", err)
		return 2
	}
	for _, e := range envVars {
		if e.Name == *keyEnv {
			fmt.Fprintf(stderr, "fak node install: --env %s collides with --key-env %s, which the installer populates itself; drop one\n", e.Name, *keyEnv)
			return 2
		}
	}
	for _, w := range upstreamWarns {
		fmt.Fprintf(stderr, "[fak node] WARNING: %s\n", w)
	}

	// Resolve and validate the bind address. parseNodeAddr decomposes --addr/--port with
	// net.SplitHostPort so a host-only --addr keeps the --port (#5 case 1), and classifies
	// loopback with net.IP.IsLoopback so [::1] and localhost are local (#5 case 2) — replacing
	// the old verbatim-string handling that dropped the port and mis-bound an IPv6 loopback.
	bindAddr, localPort, offHost, err := parseNodeAddr(*addr, *port, *remote)
	if err != nil {
		fmt.Fprintf(stderr, "fak node install: %v\n", err)
		return 2
	}

	cfgDir, err := nodeConfigDir()
	if err != nil {
		fmt.Fprintf(stderr, "fak node install: %v\n", err)
		return 1
	}

	in := nodeInstallParams{
		addr:      bindAddr,
		localPort: localPort,
		offHost:   offHost,
		keyEnv:    *keyEnv,
		uninstall: *uninstall,
		cfgDir:    cfgDir,
		rotateKey: *rotateKey,
		upstream:  upstream,
		env:       envVars,
	}
	switch runtime.GOOS {
	case "darwin":
		return nodeInstallDarwin(stdout, stderr, in)
	case "linux":
		return nodeInstallLinux(stdout, stderr, in)
	case "windows":
		return nodeInstallWindows(stdout, stderr, in)
	default:
		fmt.Fprintf(stderr, "fak node install: not yet supported on %s — use scripts/dogfood-claude.sh\n", runtime.GOOS)
		return 1
	}
}

// nodeInstallParams carries the resolved install inputs through the per-platform installers.
// Bundling them keeps the three platform signatures aligned and makes a new field (rotateKey,
// the persisted localPort) a one-line change rather than a three-way signature churn.
type nodeInstallParams struct {
	addr      string // the validated bind address passed to `fak serve --addr`
	localPort string // the parsed port, for the loopback health URL + persisted state (#1/#5)
	offHost   bool   // bind reaches beyond loopback ⇒ a bearer key is required
	keyEnv    string // env var name carrying the bearer secret
	uninstall bool
	cfgDir    string
	rotateKey bool         // mint a fresh bearer even when one is already persisted (#4)
	upstream  nodeUpstream // which upstream the rendered unit proxies (#5555)
	env       []nodeEnvVar // extra unit environment entries (#5555)
}

// nodeInstallUsage is `fak node install --help`. It prints the flag table and then the
// two invocations the flags exist to make possible — the default Anthropic adjudication
// proxy and a local-model gateway — because "one installer that can produce both" is the
// property #5555 is about, and a help page that only lists flags does not show it.
func nodeInstallUsage(stderr io.Writer, fs *flag.FlagSet) {
	fmt.Fprintln(stderr, "usage: fak node install [flags]")
	fmt.Fprintln(stderr, "")
	fmt.Fprintln(stderr, "Install the fak serve gateway as an always-on service on this machine.")
	fmt.Fprintln(stderr, "One unit, labelled "+nodeGatewayLabel+"; the flags below choose the upstream it proxies.")
	fmt.Fprintln(stderr, "")
	fs.PrintDefaults()
	fmt.Fprintln(stderr, "")
	fmt.Fprintln(stderr, "EXAMPLES")
	fmt.Fprintln(stderr, "  # the Anthropic adjudication proxy (the default upstream)")
	fmt.Fprintln(stderr, "  fak node install")
	fmt.Fprintln(stderr, "")
	fmt.Fprintln(stderr, "  # a LOCAL-MODEL gateway in front of an OpenAI-compatible server on this host")
	fmt.Fprintln(stderr, "  fak node install --base-url http://127.0.0.1:8131/v1 --model qwen3.6-27b \\")
	fmt.Fprintln(stderr, "      --env FAK_PLANNER_TIMEOUT_S=1800 --env FAK_HTTP_WRITE_TIMEOUT_S=1800")
	fmt.Fprintln(stderr, "")
	fmt.Fprintln(stderr, "  # off-host access over Tailscale (binds 0.0.0.0 and requires a bearer)")
	fmt.Fprintln(stderr, "  fak node install --remote")
	fmt.Fprintln(stderr, "")
	fmt.Fprintln(stderr, "Re-running install replaces the unit under the same label, so switching upstreams")
	fmt.Fprintln(stderr, "does not need --uninstall first. There is one gateway per host, not one per upstream.")
}

// parseNodeAddr resolves the gateway bind address from --addr and --port, returning the
// validated `host:port` to bind, the port alone (for the loopback health URL + persisted
// state), and whether the bind reaches beyond loopback (so a bearer key is required).
//
// It fixes two #5 bugs the old verbatim handling had: a host-only --addr ("0.0.0.0") now
// keeps --port instead of dropping it and producing a `:0.0.0.0` health URL; and loopback is
// detected by parsing the host with net.IP.IsLoopback (plus a literal "localhost"), so an
// IPv6 loopback [::1] is correctly local rather than being forced an off-host bearer key.
func parseNodeAddr(addr string, port int, remote bool) (bindAddr, localPort string, offHost bool, err error) {
	if port < 0 || port > 65535 {
		return "", "", false, fmt.Errorf("--port %d out of range (0-65535)", port)
	}
	host := ""
	if addr == "" {
		// No --addr: bind loopback (or 0.0.0.0 with --remote) on --port.
		if remote {
			host = "0.0.0.0"
		} else {
			host = "127.0.0.1"
		}
		localPort = strconv.Itoa(port)
	} else if h, p, splitErr := net.SplitHostPort(addr); splitErr == nil {
		// --addr carried a port ("0.0.0.0:9000", "[::1]:8080") — it wins over --port.
		host, localPort = h, p
	} else {
		// --addr is host-only ("0.0.0.0", "::1", "localhost") — keep --port (the #5 case-1 fix).
		host = strings.Trim(addr, "[]") // tolerate a bracketed IPv6 host with no port
		localPort = strconv.Itoa(port)
	}
	if strings.TrimSpace(host) == "" {
		host = "0.0.0.0" // a ":9000" wildcard bind is off-host
	}
	if localPort == "" || localPort == "0" {
		return "", "", false, fmt.Errorf("could not resolve a gateway port from --addr %q / --port %d", addr, port)
	}
	if _, convErr := strconv.Atoi(localPort); convErr != nil {
		return "", "", false, fmt.Errorf("invalid port %q in --addr %q", localPort, addr)
	}
	bindAddr = net.JoinHostPort(host, localPort)
	offHost = remote || !nodeHostIsLoopback(host)
	return bindAddr, localPort, offHost, nil
}

// nodeHostIsLoopback reports whether a bind host is loopback-only — the literal "localhost", or
// an IP that net.IP.IsLoopback accepts (127.0.0.0/8 and ::1). A non-IP, non-localhost host
// (a wildcard 0.0.0.0, a routable address, an empty host) is treated as NOT loopback, so a
// bearer key is required — the conservative direction for an exposed gateway.
func nodeHostIsLoopback(host string) bool {
	host = strings.TrimSpace(host)
	if strings.EqualFold(host, "localhost") {
		return true
	}
	ip := net.ParseIP(host)
	return ip != nil && ip.IsLoopback()
}

// ── macOS (launchd) ──────────────────────────────────────────────────────────

// caffeinateWrapperScript is written by install to ~/.config/fak/serve-wrapper.sh.
// It holds idle+system sleep assertions via caffeinate -is -w $$, then execs fak serve
// so launchd's direct child is fak serve (KeepAlive tracks the right process) and
// fak serve's stdio flows to the plist's StandardOutPath/StandardErrorPath.
const caffeinateWrapperScript = `#!/usr/bin/env bash
# Written by: fak node install — do not edit manually.
# caffeinate -is -w $$ holds idle + system sleep assertions for this PID.
# exec replaces this shell process with the target so launchd tracks fak serve.
caffeinate -is -w $$ &
exec "$@"
`

const darwinPlistTemplate = `<?xml version="1.0" encoding="UTF-8"?>
<!-- Written by: fak node install — regenerate with: fak node install -->
<plist version="1.0">
  <dict>
    <key>Label</key>
    <string>{{x .Label}}</string>

    <key>ProgramArguments</key>
    <array>
      <string>{{x .WrapperPath}}</string>
      <string>{{x .FakBin}}</string>
      <string>serve</string>
      <string>--provider</string><string>{{x .Provider}}</string>
      <string>--base-url</string><string>{{x .BaseURL}}</string>
      {{- if .Model}}
      <string>--model</string><string>{{x .Model}}</string>
      {{- end}}
      <string>--addr</string><string>{{x .Addr}}</string>
      <string>--policy</string><string>{{x .PolicyPath}}</string>
      {{- if .RequireKeyEnv}}
      <string>--require-key-env</string>
      <string>{{x .RequireKeyEnv}}</string>
      {{- end}}
    </array>

    <key>EnvironmentVariables</key>
    <dict>
      <key>FAK_AUDIT_JOURNAL</key>
      <string>{{x .LogDir}}/serve_audit.jsonl</string>
      {{- range .Env}}
      <key>{{x .Name}}</key>
      <string>{{x .Value}}</string>
      {{- end}}
      {{- if .GatewayKey}}
      <key>{{x .RequireKeyEnv}}</key>
      <string>{{x .GatewayKey}}</string>
      {{- end}}
    </dict>

    <key>WorkingDirectory</key>
    <string>{{x .LogDir}}</string>
    <key>KeepAlive</key>
    <true/>
    <key>RunAtLoad</key>
    <true/>
    <key>StandardOutPath</key>
    <string>{{x .LogDir}}/serve.log</string>
    <key>StandardErrorPath</key>
    <string>{{x .LogDir}}/serve.err</string>
  </dict>
</plist>
`

// nodeUnitData is the complete input to every platform's unit renderer: the paths and
// secrets the install resolved, plus the upstream selection and extra environment.
// Bundling it makes the three renderers PURE functions of one value, which is what lets
// both the default Anthropic unit and the local-model unit be pinned as fixtures on any
// OS — the only cross-platform witness available for a launchd/systemd/schtasks artifact.
type nodeUnitData struct {
	Label, WrapperPath, FakBin, Addr, PolicyPath, LogDir string
	RequireKeyEnv, GatewayKey                            string
	Provider, BaseURL, Model                             string
	Env                                                  []nodeEnvVar
}

// nodeUnitDataFor assembles the renderer input from the resolved install params. Pure:
// every value is already resolved by the caller, so a test can build the same input
// without touching the filesystem, launchd, or the network.
func nodeUnitDataFor(in nodeInstallParams, label, wrapperPath, fakBin, policyPath, logDir, gatewayKey, requireKeyEnv string) nodeUnitData {
	return nodeUnitData{
		Label:         label,
		WrapperPath:   wrapperPath,
		FakBin:        fakBin,
		Addr:          in.addr,
		PolicyPath:    policyPath,
		LogDir:        logDir,
		RequireKeyEnv: requireKeyEnv,
		GatewayKey:    gatewayKey,
		Provider:      in.upstream.Provider,
		BaseURL:       in.upstream.BaseURL,
		Model:         in.upstream.Model,
		Env:           in.env,
	}
}

// nodeXMLEscape escapes a value for a plist <string> body. Applied to every interpolated
// field: a repo path with an `&`, or an --env value carrying JSON with a `<`, would
// otherwise produce a plist launchd refuses to parse rather than a loud install error.
func nodeXMLEscape(s string) string {
	return strings.NewReplacer("&", "&amp;", "<", "&lt;", ">", "&gt;", `"`, "&quot;", "'", "&apos;").Replace(s)
}

// nodeRenderDarwinPlist renders the launchd agent. Pure function of its input — no file,
// no launchctl — so `fak node install`'s two headline outputs (the default Anthropic
// adjudication proxy and a local-model gateway) can both be pinned as fixtures.
func nodeRenderDarwinPlist(d nodeUnitData) (string, error) {
	return nodeRenderUnit("plist", darwinPlistTemplate, d)
}

// nodeRenderLinuxUnit renders the systemd --user unit. Same purity contract as
// nodeRenderDarwinPlist.
func nodeRenderLinuxUnit(d nodeUnitData) (string, error) {
	return nodeRenderUnit("unit", linuxUnitTemplate, d)
}

// nodeRenderWindowsRunner renders the .cmd the Scheduled Task launches. Same purity
// contract as nodeRenderDarwinPlist.
func nodeRenderWindowsRunner(d nodeUnitData) (string, error) {
	return nodeRenderUnit("runner", nodeWindowsRunnerTemplate, d)
}

// nodeRenderUnit is the shared parse-and-execute behind the three renderers, carrying the
// per-format escapers the templates call: `x` for plist XML, `sd` for a systemd quoted
// Environment= value, and `cmdv` for a cmd.exe `set` value.
func nodeRenderUnit(name, text string, d nodeUnitData) (string, error) {
	tmpl, err := template.New(name).Funcs(template.FuncMap{
		"x":    nodeXMLEscape,
		"sd":   nodeSystemdEscape,
		"cmdv": nodeCmdEscape,
	}).Parse(text)
	if err != nil {
		return "", err
	}
	var b strings.Builder
	if err := tmpl.Execute(&b, d); err != nil {
		return "", err
	}
	return b.String(), nil
}

// nodeSystemdEscape escapes a value for a double-quoted systemd `Environment="N=V"`
// entry. systemd's own unit-file quoting recognises backslash escapes inside double
// quotes, so a value containing a quote (JSON, which is exactly what the local-model
// tuning entries carry) survives instead of truncating the assignment.
func nodeSystemdEscape(s string) string {
	return strings.NewReplacer(`\`, `\\`, `"`, `\"`).Replace(s)
}

// nodeCmdEscape escapes a value for an UNQUOTED cmd.exe `set NAME=VALUE`. The unquoted
// form is used for --env entries specifically because cmd offers no escape for a `"`
// inside `set "NAME=VALUE"`, and JSON tuning values are full of quotes; unquoted, a `"`
// is literal and only the shell metacharacters need a `^`.
func nodeCmdEscape(s string) string {
	return strings.NewReplacer("^", "^^", "&", "^&", "|", "^|", "<", "^<", ">", "^>").Replace(s)
}

func nodeInstallDarwin(stdout, stderr io.Writer, in nodeInstallParams) int {
	uninstall, cfgDir := in.uninstall, in.cfgDir
	agentsDir := filepath.Join(os.Getenv("HOME"), "Library", "LaunchAgents")
	plistPath := filepath.Join(agentsDir, nodeGatewayLabel+".plist")

	if uninstall {
		_ = exec.Command("launchctl", "unload", "-w", plistPath).Run()
		_ = os.Remove(plistPath)
		_ = os.Remove(nodeInstallStatePath(cfgDir))
		fmt.Fprintf(stdout, "[fak node] unloaded and removed %s\n", plistPath)
		return 0
	}

	// Ensure dirs exist and write the default policy.
	logDir, policyPath, ok := nodeInstallDirs(stderr, cfgDir, agentsDir)
	if !ok {
		return 1
	}

	// Write the caffeinate wrapper script.
	wrapperPath := filepath.Join(cfgDir, "serve-wrapper.sh")
	if err := os.WriteFile(wrapperPath, []byte(caffeinateWrapperScript), 0755); err != nil {
		fmt.Fprintf(stderr, "fak node install: write wrapper: %v\n", err)
		return 1
	}

	// Resolve our own binary and (for off-host installs) the bearer key.
	fakBin, gatewayKey, requireKeyEnv, ok := nodeInstallBinAndKey(stdout, stderr, in)
	if !ok {
		return 1
	}

	// Render the plist (pure), then write it. Rendering before any launchctl call means a
	// template failure never leaves a half-written unit behind an unloaded service.
	plist, err := nodeRenderDarwinPlist(nodeUnitDataFor(in, nodeGatewayLabel, wrapperPath, fakBin, policyPath, logDir, gatewayKey, requireKeyEnv))
	if err != nil {
		fmt.Fprintf(stderr, "fak node install: render plist: %v\n", err)
		return 1
	}

	// Unload any existing unit before overwriting. One label per host: switching upstreams
	// re-renders THIS unit rather than adding a second one (#5555).
	_ = exec.Command("launchctl", "unload", "-w", plistPath).Run()

	if err := os.WriteFile(plistPath, []byte(plist), 0644); err != nil {
		fmt.Fprintf(stderr, "fak node install: write plist: %v\n", err)
		return 1
	}

	// Set ANTHROPIC_API_KEY in login env if provided — ONLY for the Anthropic wire. A
	// local-model unit has no use for it, and pushing an Anthropic credential into the
	// login environment of a host installed to talk to a local model is gratuitous
	// credential spread (#5555).
	if in.upstream.Provider == nodeDefaultProvider {
		if key := os.Getenv("ANTHROPIC_API_KEY"); key != "" {
			_ = exec.Command("launchctl", "setenv", "ANTHROPIC_API_KEY", key).Run()
			fmt.Fprintf(stdout, "[fak node] set ANTHROPIC_API_KEY in login environment\n")
		} else {
			fmt.Fprintf(stderr, "[fak node] WARNING: ANTHROPIC_API_KEY not set — set it with:\n")
			fmt.Fprintf(stderr, "           launchctl setenv ANTHROPIC_API_KEY \"sk-ant-...\"\n")
		}
	}

	// Load the unit.
	if out, err := exec.Command("launchctl", "load", "-w", plistPath).CombinedOutput(); err != nil {
		fmt.Fprintf(stderr, "fak node install: launchctl load: %v\n%s\n", err, out)
		return 1
	}
	fmt.Fprintf(stdout, "[fak node] loaded %s\n", nodeGatewayLabel)
	fmt.Fprintf(stdout, "           plist:   %s\n", plistPath)
	fmt.Fprintf(stdout, "           log:     %s/serve.log\n", logDir)

	// Persist what we installed so `status` probes the real port (#1) and a re-install can
	// reuse the bearer key (#4). Written before the health wait so the record exists even if
	// the gateway is slow to come up — then health-gate honestly and print the client lines.
	return nodeInstallFinalize(stdout, stderr, in, gatewayKey, requireKeyEnv, logDir)
}

// nodeInstallFinalize is the shared tail of every platform install path: it persists the
// install record (so `status` probes the real port and a re-install can reuse the bearer
// key), then health-gates the gateway HONESTLY — on a no-answer it warns at the logs and
// returns 1 instead of printing the client lines as if it were up (#2); otherwise it prints
// the client lines and returns 0.
func nodeInstallFinalize(stdout, stderr io.Writer, in nodeInstallParams, gatewayKey, requireKeyEnv, logDir string) int {
	nodePrintUnitSummary(stdout, in)
	nodePersistInstallState(stderr, in, gatewayKey, requireKeyEnv)
	localPort := in.localPort
	if !nodeWaitHealthy(stdout, "http://127.0.0.1:"+localPort) {
		nodeWarnUnhealthy(stderr, logDir)
		return 1
	}
	nodePrintClientLines(stdout, stderr, in.offHost, gatewayKey, localPort)
	return 0
}

// nodePrintUnitSummary reports which upstream the unit was installed against and which
// extra environment entries it carries. Entries are reported by NAME ONLY — the values
// live in the unit file, and echoing them would put whatever an operator passed to --env
// into terminal scrollback and CI logs. Makes a re-install that switched upstreams
// visible, since one label now serves both (#5555).
func nodePrintUnitSummary(stdout io.Writer, in nodeInstallParams) {
	fmt.Fprintf(stdout, "[fak node] upstream: %s\n", nodeUpstreamSummary(in.upstream))
	if names := nodeEnvNames(in.env); names != "" {
		fmt.Fprintf(stdout, "[fak node] unit env:  FAK_AUDIT_JOURNAL, %s (names only; values are in the unit file)\n", names)
	}
}

// nodeUpstreamSummary is the one-line, secret-free description of the resolved upstream.
func nodeUpstreamSummary(u nodeUpstream) string {
	s := u.Provider + " " + u.BaseURL
	if u.Model != "" {
		s += " (model " + u.Model + ")"
	}
	return s
}

// nodeEnvNames joins the extra environment entry NAMES for display. Never the values.
func nodeEnvNames(env []nodeEnvVar) string {
	names := make([]string, 0, len(env))
	for _, e := range env {
		names = append(names, e.Name)
	}
	return strings.Join(names, ", ")
}

// nodeInstallDirs is the shared head of every platform install path: it creates the
// config and log dirs (plus any platform-specific extras) and writes the default node
// policy. It returns the resolved log dir and policy path, or ok=false after reporting
// the failure to stderr.
func nodeInstallDirs(stderr io.Writer, cfgDir string, extraDirs ...string) (logDir, policyPath string, ok bool) {
	logDir = filepath.Join(cfgDir, "logs")
	for _, d := range append([]string{cfgDir, logDir}, extraDirs...) {
		if err := os.MkdirAll(d, 0755); err != nil {
			fmt.Fprintf(stderr, "fak node install: mkdir %s: %v\n", d, err)
			return "", "", false
		}
	}
	policyPath = filepath.Join(cfgDir, "node-policy.json")
	if err := os.WriteFile(policyPath, nodeDefaultPolicyJSON, 0644); err != nil {
		fmt.Fprintf(stderr, "fak node install: write policy: %v\n", err)
		return "", "", false
	}
	return logDir, policyPath, true
}

// nodeInstallBinAndKey resolves the running fak binary and, for off-host installs, the
// bearer key WITHOUT silently rotating it: a re-install reuses the persisted key so
// configured clients keep working (#4); a fresh mint is flagged. It returns ok=false
// after reporting the failure to stderr.
func nodeInstallBinAndKey(stdout, stderr io.Writer, in nodeInstallParams) (fakBin, gatewayKey, requireKeyEnv string, ok bool) {
	fakBin, err := os.Executable()
	if err != nil {
		fmt.Fprintf(stderr, "fak node install: resolve binary: %v\n", err)
		return "", "", "", false
	}
	if in.offHost {
		requireKeyEnv = in.keyEnv
		key, minted, kerr := nodeResolveKey(in.cfgDir, in.keyEnv, in.rotateKey)
		if kerr != nil {
			fmt.Fprintf(stderr, "fak node install: %v\n", kerr)
			return "", "", "", false
		}
		gatewayKey = key
		nodeReportKeyDisposition(stdout, minted, in.rotateKey)
	}
	return fakBin, gatewayKey, requireKeyEnv, true
}

// nodeWaitHealthy polls <base>/healthz up to 20 times (~10s) and returns whether a live
// gateway answered 2xx. It is the shared, HONEST install health gate: a caller that gets
// false must warn and fail rather than print the client lines as if the gateway were up (#2).
func nodeWaitHealthy(stdout io.Writer, base string) bool {
	fmt.Fprintf(stdout, "[fak node] waiting for gateway...\n")
	for i := 0; i < 20; i++ {
		time.Sleep(500 * time.Millisecond)
		if status, ok := nodeProbeHealth(base); ok {
			fmt.Fprintf(stdout, "[fak node] gateway healthy at %s/healthz (%s)\n", strings.TrimRight(base, "/"), status)
			return true
		}
	}
	return false
}

// nodeWarnUnhealthy prints the loud, actionable failure banner when the gateway never came
// up — pointing at the serve log/err files — so a silent install failure (a bad policy, a
// bound port, a missing upstream credential) is reported instead of a false success (#2).
func nodeWarnUnhealthy(stderr io.Writer, logDir string) {
	fmt.Fprintf(stderr, "\n[fak node] ERROR: the gateway did not become healthy within ~10s.\n")
	fmt.Fprintf(stderr, "           It may have failed to start (bad policy, port already bound, or a\n")
	fmt.Fprintf(stderr, "           missing upstream credential). Check the logs:\n")
	fmt.Fprintf(stderr, "             %s\n", filepath.Join(logDir, "serve.log"))
	fmt.Fprintf(stderr, "             %s\n", filepath.Join(logDir, "serve.err"))
	fmt.Fprintf(stderr, "           Fix the cause and re-run `fak node install`.\n")
}

// nodeReportKeyDisposition makes the bearer-key decision VISIBLE so a key rotation is never
// silent (#4): a reused key is noted, a freshly-minted key on a re-install (an explicit
// --rotate-key) is flagged loudly because every client still presenting the old key will 401.
func nodeReportKeyDisposition(stdout io.Writer, minted, rotate bool) {
	switch {
	case minted && rotate:
		fmt.Fprintf(stdout, "[fak node] NOTE: minted a NEW bearer key (--rotate-key) — every client must re-run `fak node use` with the new key below.\n")
	case minted:
		fmt.Fprintf(stdout, "[fak node] generated a new bearer key (save it below).\n")
	default:
		fmt.Fprintf(stdout, "[fak node] reusing the existing bearer key (configured clients keep working; pass --rotate-key to mint a fresh one).\n")
	}
}

// nodePersistInstallState records what the host installed (addr, port, key, off-host) so
// `status` can probe the real port (#1) and a re-install can reuse the key (#4). A write
// failure is non-fatal (the gateway is already up) but warned, since it degrades both fixes.
func nodePersistInstallState(stderr io.Writer, in nodeInstallParams, gatewayKey, requireKeyEnv string) {
	st := nodeInstallState{
		Addr:    in.addr,
		Port:    in.localPort,
		Key:     gatewayKey,
		KeyEnv:  requireKeyEnv,
		OffHost: in.offHost,
	}
	if err := nodeWriteInstallState(in.cfgDir, st); err != nil {
		fmt.Fprintf(stderr, "[fak node] WARNING: could not persist install state (%v) — `status` will fall back to :8080 and a re-install may rotate the key\n", err)
	}
}

// nodePrintClientLines emits the post-install guidance shared by every platform's
// installer: the Tailscale-routable client export lines (off-host) or the single-machine
// next step (loopback). Keeping it in one place means all three OS paths print identical,
// copy-pasteable instructions and there is one banner to maintain. localPort is the
// gateway's port; gatewayKey is the generated bearer (empty for a loopback install).
func nodePrintClientLines(stdout, stderr io.Writer, offHost bool, gatewayKey, localPort string) {
	fmt.Fprintln(stdout, "")
	if offHost {
		tailscaleIP := nodeTailscaleIP()
		if tailscaleIP == "" {
			tailscaleIP = "<this-machine-tailscale-ip>"
			fmt.Fprintf(stderr, "[fak node] WARNING: Tailscale not running — get IP with: tailscale ip -4\n")
		}
		fmt.Fprintln(stdout, "=== client connection (paste on any Tailscale-connected client) ===")
		fmt.Fprintln(stdout, "")
		fmt.Fprintf(stdout, "  bash/zsh:\n")
		fmt.Fprintf(stdout, "    export ANTHROPIC_BASE_URL=\"http://%s:%s\"\n", tailscaleIP, localPort)
		fmt.Fprintf(stdout, "    export ANTHROPIC_API_KEY=\"%s\"\n", gatewayKey)
		fmt.Fprintf(stdout, "    claude\n")
		fmt.Fprintln(stdout, "")
		fmt.Fprintf(stdout, "  PowerShell:\n")
		fmt.Fprintf(stdout, "    $env:ANTHROPIC_BASE_URL = \"http://%s:%s\"\n", tailscaleIP, localPort)
		fmt.Fprintf(stdout, "    $env:ANTHROPIC_API_KEY  = \"%s\"\n", gatewayKey)
		fmt.Fprintf(stdout, "    claude\n")
		fmt.Fprintln(stdout, "")
		fmt.Fprintf(stdout, "  Or use fak node use + run from the client:\n")
		fmt.Fprintf(stdout, "    fak node use %s:%s --key %s\n", tailscaleIP, localPort, gatewayKey)
		fmt.Fprintf(stdout, "    fak node run -- claude\n")
		fmt.Fprintln(stdout, "")
		fmt.Fprintf(stdout, "FAK_GATEWAY_KEY=%s\n", gatewayKey)
		fmt.Fprintf(stdout, "(save this — it is not stored in plaintext on remote clients)\n")
	} else {
		fmt.Fprintln(stdout, "=== single-machine use ===")
		fmt.Fprintln(stdout, "")
		fmt.Fprintf(stdout, "  fak guard -- claude    # guarded interactive session\n")
		fmt.Fprintln(stdout, "")
		fmt.Fprintf(stdout, "For off-host access (e.g. connecting a client over Tailscale):\n")
		fmt.Fprintf(stdout, "  fak node install --remote\n")
	}
}

// ── Linux (systemd --user) ────────────────────────────────────────────────────

const linuxUnitTemplate = `[Unit]
Description=fak serve gateway (always-on adjudicating proxy for {{.BaseURL}})
After=network.target

[Service]
ExecStart={{.FakBin}} serve --provider {{.Provider}} --base-url {{.BaseURL}}{{if .Model}} --model {{.Model}}{{end}} --addr {{.Addr}} --policy {{.PolicyPath}}{{if .RequireKeyEnv}} --require-key-env {{.RequireKeyEnv}}{{end}}
Restart=always
RestartSec=3
Environment=FAK_AUDIT_JOURNAL={{.LogDir}}/serve_audit.jsonl
{{- range .Env}}
Environment="{{.Name}}={{sd .Value}}"
{{- end}}
{{- if .GatewayKey}}
Environment={{.RequireKeyEnv}}={{.GatewayKey}}
{{- end}}
StandardOutput=append:{{.LogDir}}/serve.log
StandardError=append:{{.LogDir}}/serve.err

[Install]
WantedBy=default.target
`

func nodeInstallLinux(stdout, stderr io.Writer, in nodeInstallParams) int {
	uninstall, cfgDir := in.uninstall, in.cfgDir
	unitDir := filepath.Join(os.Getenv("HOME"), ".config", "systemd", "user")
	unitPath := filepath.Join(unitDir, "fak-serve-gateway.service")

	if uninstall {
		_ = exec.Command("systemctl", "--user", "disable", "--now", "fak-serve-gateway").Run()
		_ = os.Remove(unitPath)
		_ = os.Remove(nodeInstallStatePath(cfgDir))
		fmt.Fprintf(stdout, "[fak node] disabled and removed %s\n", unitPath)
		return 0
	}

	logDir, policyPath, ok := nodeInstallDirs(stderr, cfgDir, unitDir)
	if !ok {
		return 1
	}

	fakBin, gatewayKey, requireKeyEnv, ok := nodeInstallBinAndKey(stdout, stderr, in)
	if !ok {
		return 1
	}

	unit, err := nodeRenderLinuxUnit(nodeUnitDataFor(in, nodeGatewayLabel, "", fakBin, policyPath, logDir, gatewayKey, requireKeyEnv))
	if err != nil {
		fmt.Fprintf(stderr, "fak node install: render unit: %v\n", err)
		return 1
	}
	if err := os.WriteFile(unitPath, []byte(unit), 0644); err != nil {
		fmt.Fprintf(stderr, "fak node install: write unit: %v\n", err)
		return 1
	}

	_ = exec.Command("systemctl", "--user", "daemon-reload").Run()
	if out, err := exec.Command("systemctl", "--user", "enable", "--now", "fak-serve-gateway").CombinedOutput(); err != nil {
		fmt.Fprintf(stderr, "fak node install: systemctl enable: %v\n%s\n", err, out)
		return 1
	}
	fmt.Fprintf(stdout, "[fak node] enabled fak-serve-gateway (systemd --user)\n")

	// Add the post-enable health probe the linux path was missing entirely (#2): if the unit
	// loaded but fak serve never answered, warn at the logs and fail rather than print the
	// client lines as if it were up.
	return nodeInstallFinalize(stdout, stderr, in, gatewayKey, requireKeyEnv, logDir)
}

// ── Windows (Scheduled Task) ──────────────────────────────────────────────────

// nodeWindowsRunnerTemplate is the .cmd the Scheduled Task launches. A Scheduled Task
// runs an exe, not an env-carrying service, so the runner sets FAK_AUDIT_JOURNAL (and the
// bearer secret, off-host) before exec'ing fak serve — the Windows analog of the plist's
// EnvironmentVariables / the unit's Environment= lines.
const nodeWindowsRunnerTemplate = `@echo off
rem Written by: fak node install — regenerate with: fak node install
set "FAK_AUDIT_JOURNAL={{.LogDir}}\serve_audit.jsonl"
{{- range .Env}}
set {{.Name}}={{cmdv .Value}}
{{- end}}
{{- if .GatewayKey}}
set "{{.RequireKeyEnv}}={{.GatewayKey}}"
{{- end}}
"{{.FakBin}}" serve --provider {{.Provider}} --base-url {{.BaseURL}}{{if .Model}} --model {{.Model}}{{end}} --addr {{.Addr}} --policy "{{.PolicyPath}}"{{if .RequireKeyEnv}} --require-key-env {{.RequireKeyEnv}}{{end}} >> "{{.LogDir}}\serve.log" 2>> "{{.LogDir}}\serve.err"
`

func nodeHelperCommand(name string, args ...string) *exec.Cmd {
	cmd := exec.Command(name, args...)
	windowgate.ConfigureBackgroundCommand(cmd)
	return cmd
}

func nodeInstallWindows(stdout, stderr io.Writer, in nodeInstallParams) int {
	uninstall, cfgDir := in.uninstall, in.cfgDir
	if uninstall {
		_ = nodeHelperCommand("schtasks", "/End", "/TN", nodeWindowsTaskName).Run()
		out, err := nodeHelperCommand("schtasks", "/Delete", "/TN", nodeWindowsTaskName, "/F").CombinedOutput()
		if err != nil {
			fmt.Fprintf(stderr, "fak node install: schtasks /Delete: %v\n%s\n", err, out)
			return 1
		}
		_ = os.Remove(nodeInstallStatePath(cfgDir))
		fmt.Fprintf(stdout, "[fak node] removed Scheduled Task %s\n", nodeWindowsTaskName)
		return 0
	}

	logDir, policyPath, ok := nodeInstallDirs(stderr, cfgDir)
	if !ok {
		return 1
	}

	fakBin, gatewayKey, requireKeyEnv, ok := nodeInstallBinAndKey(stdout, stderr, in)
	if !ok {
		return 1
	}

	// Render the runner .cmd the task launches.
	runnerPath := filepath.Join(cfgDir, "serve-runner.cmd")
	runner, err := nodeRenderWindowsRunner(nodeUnitDataFor(in, nodeGatewayLabel, "", fakBin, policyPath, logDir, gatewayKey, requireKeyEnv))
	if err != nil {
		fmt.Fprintf(stderr, "fak node install: render runner: %v\n", err)
		return 1
	}
	if err := os.WriteFile(runnerPath, []byte(runner), 0644); err != nil {
		fmt.Fprintf(stderr, "fak node install: write runner: %v\n", err)
		return 1
	}

	localPort := in.localPort

	// Stop any prior instance and confirm the port is free BEFORE (re)starting (#3) — the
	// macOS path already `launchctl unload`s first; Windows did not, so a stale fak serve
	// kept the port and answered the health probe, making install falsely report the OLD
	// process healthy. End the task and wait for the port to free; if a foreign process still
	// holds it, fail loudly rather than blessing whatever answers the probe.
	_ = nodeHelperCommand("schtasks", "/End", "/TN", nodeWindowsTaskName).Run()
	if !nodeWaitPortFree(localPort) {
		fmt.Fprintf(stderr, "\n[fak node] ERROR: 127.0.0.1:%s is still in use after stopping the task.\n", localPort)
		fmt.Fprintf(stderr, "           Another process holds the port; the install would falsely report IT healthy.\n")
		fmt.Fprintf(stderr, "           Free the port (or choose another with --port) and re-run `fak node install`.\n")
		return 1
	}

	// Register an always-on Scheduled Task via an XML definition so it carries restart-on-
	// failure semantics (#6) — the Windows analog of launchd KeepAlive / systemd Restart=always
	// the simple `schtasks /Create /SC ONSTART` form cannot express. The XML triggers at boot
	// (BootTrigger) AND restarts the runner on failure (RestartOnFailure, 1-min interval).
	taskXML := nodeWindowsTaskXML(runnerPath)
	xmlPath := filepath.Join(cfgDir, "serve-task.xml")
	if err := os.WriteFile(xmlPath, []byte(taskXML), 0644); err != nil {
		fmt.Fprintf(stderr, "fak node install: write task xml: %v\n", err)
		return 1
	}
	if out, err := nodeHelperCommand("schtasks", "/Create", "/TN", nodeWindowsTaskName,
		"/XML", xmlPath, "/F").CombinedOutput(); err != nil {
		fmt.Fprintf(stderr, "fak node install: schtasks /Create: %v\n%s\n", err, out)
		return 1
	}
	if out, err := nodeHelperCommand("schtasks", "/Run", "/TN", nodeWindowsTaskName).CombinedOutput(); err != nil {
		// Non-fatal: the task is registered and will start at boot even if the immediate
		// run could not be kicked off.
		fmt.Fprintf(stderr, "[fak node] note: schtasks /Run did not start the task now (%v): %s\n", err, out)
	}
	fmt.Fprintf(stdout, "[fak node] registered Scheduled Task %s (boot + restart-on-failure)\n", nodeWindowsTaskName)
	fmt.Fprintf(stdout, "           runner:  %s\n", runnerPath)
	fmt.Fprintf(stdout, "           log:     %s\\serve.log\n", logDir)

	// Only the Anthropic wire needs an Anthropic credential; a local-model unit does not.
	if in.upstream.Provider == nodeDefaultProvider {
		if key := os.Getenv("ANTHROPIC_API_KEY"); key == "" {
			fmt.Fprintf(stderr, "[fak node] WARNING: ANTHROPIC_API_KEY not set for the task's user — set it (e.g. setx ANTHROPIC_API_KEY \"sk-ant-...\") and re-run, or the gateway has no upstream credential\n")
		}
	}

	// Honest health gate (#2): on failure warn at the logs and return non-zero instead of
	// printing the client lines as if the gateway were up.
	return nodeInstallFinalize(stdout, stderr, in, gatewayKey, requireKeyEnv, logDir)
}

// nodeWindowsTaskXML builds the Scheduled Task definition that gives the Windows gateway the
// crash-restart the simple `schtasks /SC ONSTART` form lacks (#6): a BootTrigger keeps it
// resident across reboots and RestartOnFailure relaunches the runner if fak serve exits,
// matching launchd KeepAlive / systemd Restart=always. The runner path is XML-escaped.
func nodeWindowsTaskXML(runnerPath string) string {
	esc := func(s string) string {
		r := strings.NewReplacer("&", "&amp;", "<", "&lt;", ">", "&gt;", `"`, "&quot;", "'", "&apos;")
		return r.Replace(s)
	}
	return `<?xml version="1.0" encoding="UTF-16"?>
<Task version="1.2" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">
  <RegistrationInfo>
    <Description>fak serve gateway (always-on adjudicating proxy) — written by fak node install</Description>
  </RegistrationInfo>
  <Triggers>
    <BootTrigger><Enabled>true</Enabled></BootTrigger>
  </Triggers>
  <Principals>
    <Principal id="Author">
      <LogonType>InteractiveToken</LogonType>
      <RunLevel>LeastPrivilege</RunLevel>
    </Principal>
  </Principals>
  <Settings>
    <MultipleInstancesPolicy>IgnoreNew</MultipleInstancesPolicy>
    <DisallowStartIfOnBatteries>false</DisallowStartIfOnBatteries>
    <StopIfGoingOnBatteries>false</StopIfGoingOnBatteries>
    <ExecutionTimeLimit>PT0S</ExecutionTimeLimit>
    <RestartOnFailure>
      <Interval>PT1M</Interval>
      <Count>999</Count>
    </RestartOnFailure>
  </Settings>
  <Actions Context="Author">
    <Exec>
      <Command>cmd.exe</Command>
      <Arguments>/c "` + esc(runnerPath) + `"</Arguments>
    </Exec>
  </Actions>
</Task>
`
}

// nodeWaitPortFree polls 127.0.0.1:<port> up to ~3s and returns true once nothing is
// listening (a fresh dial is refused). It is the pre-(re)start check that stops the Windows
// installer from blessing a stale/foreign process already bound to the port (#3).
func nodeWaitPortFree(port string) bool {
	for i := 0; i < 12; i++ {
		conn, err := net.DialTimeout("tcp", net.JoinHostPort("127.0.0.1", port), 200*time.Millisecond)
		if err != nil {
			return true // refused/timeout ⇒ nothing is listening ⇒ the port is free
		}
		_ = conn.Close()
		time.Sleep(250 * time.Millisecond)
	}
	return false
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
