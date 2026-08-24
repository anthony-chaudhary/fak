package main

// node_install.go — the `fak node install` subcommand: flag parsing and address
// resolution, the per-OS service-unit templates plus the pure renderers/escapers that
// fill them, and the launchd / systemd --user / Scheduled Task installers with the
// platform-independent install tail they share. Split verbatim out of node.go — no
// behaviour change — to hold the internal/godfileceiling 1500-line god-file cap;
// node.go keeps the node.json client side (status/use/run/forget) and the helpers.

import (
	"flag"
	"fmt"
	"io"
	"net"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"text/template"
	"time"
)

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
	return backgroundCommand(name, args...)
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
