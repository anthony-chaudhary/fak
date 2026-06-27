package main

// `fak node` — durable, self-contained tooling for setting up and connecting to an
// always-on fak serve gateway. Replaces the shell scripts in tools/ and scripts/
// that required sourcing, chmod, platform-specific quoting, and external sed/PlistBuddy.
//
// Subcommands:
//
//	fak node install [--remote] [--addr ADDR] [--port N] [--key-env VAR] [--uninstall]
//	                           Install the fak serve gateway as a system service on this
//	                           machine. macOS: launchd KeepAlive agent with caffeinate
//	                           wrapper. Linux: systemd --user unit. Prints client env
//	                           lines after install.
//	fak node status            Show service state + gateway health (no flags).
//	fak node use HOST[:PORT] [--key KEY] [--env]
//	                           Write ~/.config/fak/node.json and print the two export
//	                           lines to paste in your shell / CLAUDE.md. --env skips
//	                           the write and just prints the lines.
//	fak node forget            Clear ~/.config/fak/node.json (undo `use`).

import (
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"text/template"
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
    ".ssh/",
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
const nodeGatewayLabel = "com.fak.serve-gateway"

// nodeCfg is the on-disk config written by `fak node use`.
type nodeCfg struct {
	URL string `json:"url"`
	Key string `json:"key,omitempty"`
}

func cmdNode(argv []string) { os.Exit(runNode(os.Stdout, os.Stderr, argv)) }

func runNode(stdout, stderr io.Writer, argv []string) int {
	if len(argv) == 0 {
		fmt.Fprintln(stderr, "usage: fak node <install|status|use|forget> [flags]")
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
	case "forget":
		return nodeForget(stdout, stderr)
	default:
		fmt.Fprintf(stderr, "fak node: unknown subcommand %q (want install|status|use|forget)\n", argv[0])
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
	if err := fs.Parse(argv); err != nil {
		return 2
	}

	// Resolve bind address.
	bindAddr := *addr
	if bindAddr == "" {
		if *remote {
			bindAddr = fmt.Sprintf("0.0.0.0:%d", *port)
		} else {
			bindAddr = fmt.Sprintf("127.0.0.1:%d", *port)
		}
	}
	offHost := !strings.HasPrefix(bindAddr, "127.") && !strings.HasPrefix(bindAddr, "localhost")

	cfgDir, err := nodeConfigDir()
	if err != nil {
		fmt.Fprintf(stderr, "fak node install: %v\n", err)
		return 1
	}

	switch runtime.GOOS {
	case "darwin":
		return nodeInstallDarwin(stdout, stderr, bindAddr, offHost, *keyEnv, *uninstall, cfgDir)
	case "linux":
		return nodeInstallLinux(stdout, stderr, bindAddr, offHost, *keyEnv, *uninstall, cfgDir)
	default:
		fmt.Fprintf(stderr, "fak node install: not yet supported on %s — use scripts/dogfood-claude.sh or the Windows Scheduled Task\n", runtime.GOOS)
		return 1
	}
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
    <string>{{.Label}}</string>

    <key>ProgramArguments</key>
    <array>
      <string>{{.WrapperPath}}</string>
      <string>{{.FakBin}}</string>
      <string>serve</string>
      <string>--provider</string><string>anthropic</string>
      <string>--base-url</string><string>https://api.anthropic.com</string>
      <string>--addr</string><string>{{.Addr}}</string>
      <string>--policy</string><string>{{.PolicyPath}}</string>
      {{- if .RequireKeyEnv}}
      <string>--require-key-env</string>
      <string>{{.RequireKeyEnv}}</string>
      {{- end}}
    </array>

    <key>EnvironmentVariables</key>
    <dict>
      <key>FAK_AUDIT_JOURNAL</key>
      <string>{{.LogDir}}/serve_audit.jsonl</string>
      {{- if .GatewayKey}}
      <key>{{.RequireKeyEnv}}</key>
      <string>{{.GatewayKey}}</string>
      {{- end}}
    </dict>

    <key>WorkingDirectory</key>
    <string>{{.LogDir}}</string>
    <key>KeepAlive</key>
    <true/>
    <key>RunAtLoad</key>
    <true/>
    <key>StandardOutPath</key>
    <string>{{.LogDir}}/serve.log</string>
    <key>StandardErrorPath</key>
    <string>{{.LogDir}}/serve.err</string>
  </dict>
</plist>
`

func nodeInstallDarwin(stdout, stderr io.Writer, addr string, offHost bool, keyEnv string, uninstall bool, cfgDir string) int {
	agentsDir := filepath.Join(os.Getenv("HOME"), "Library", "LaunchAgents")
	plistPath := filepath.Join(agentsDir, nodeGatewayLabel+".plist")

	if uninstall {
		_ = exec.Command("launchctl", "unload", "-w", plistPath).Run()
		_ = os.Remove(plistPath)
		fmt.Fprintf(stdout, "[fak node] unloaded and removed %s\n", plistPath)
		return 0
	}

	// Ensure dirs exist.
	logDir := filepath.Join(cfgDir, "logs")
	for _, d := range []string{cfgDir, logDir, agentsDir} {
		if err := os.MkdirAll(d, 0755); err != nil {
			fmt.Fprintf(stderr, "fak node install: mkdir %s: %v\n", d, err)
			return 1
		}
	}

	// Write the default policy to the config dir.
	policyPath := filepath.Join(cfgDir, "node-policy.json")
	if err := os.WriteFile(policyPath, nodeDefaultPolicyJSON, 0644); err != nil {
		fmt.Fprintf(stderr, "fak node install: write policy: %v\n", err)
		return 1
	}

	// Write the caffeinate wrapper script.
	wrapperPath := filepath.Join(cfgDir, "serve-wrapper.sh")
	if err := os.WriteFile(wrapperPath, []byte(caffeinateWrapperScript), 0755); err != nil {
		fmt.Fprintf(stderr, "fak node install: write wrapper: %v\n", err)
		return 1
	}

	// Find our own binary.
	fakBin, err := os.Executable()
	if err != nil {
		fmt.Fprintf(stderr, "fak node install: resolve binary: %v\n", err)
		return 1
	}

	// Generate bearer key for off-host installs.
	gatewayKey := ""
	requireKeyEnv := ""
	if offHost {
		requireKeyEnv = keyEnv
		existing := os.Getenv(keyEnv)
		if existing != "" {
			gatewayKey = existing
		} else {
			b := make([]byte, 32)
			if _, err := rand.Read(b); err != nil {
				fmt.Fprintf(stderr, "fak node install: generate key: %v\n", err)
				return 1
			}
			gatewayKey = hex.EncodeToString(b)
		}
	}

	// Render the plist.
	tmpl, err := template.New("plist").Parse(darwinPlistTemplate)
	if err != nil {
		fmt.Fprintf(stderr, "fak node install: parse plist template: %v\n", err)
		return 1
	}
	plistData := struct {
		Label, WrapperPath, FakBin, Addr, PolicyPath, LogDir string
		RequireKeyEnv, GatewayKey                            string
	}{
		Label:         nodeGatewayLabel,
		WrapperPath:   wrapperPath,
		FakBin:        fakBin,
		Addr:          addr,
		PolicyPath:    policyPath,
		LogDir:        logDir,
		RequireKeyEnv: requireKeyEnv,
		GatewayKey:    gatewayKey,
	}

	// Unload any existing unit before overwriting.
	_ = exec.Command("launchctl", "unload", "-w", plistPath).Run()

	pf, err := os.Create(plistPath)
	if err != nil {
		fmt.Fprintf(stderr, "fak node install: create plist: %v\n", err)
		return 1
	}
	if err := tmpl.Execute(pf, plistData); err != nil {
		pf.Close()
		fmt.Fprintf(stderr, "fak node install: render plist: %v\n", err)
		return 1
	}
	pf.Close()

	// Set ANTHROPIC_API_KEY in login env if provided.
	if key := os.Getenv("ANTHROPIC_API_KEY"); key != "" {
		_ = exec.Command("launchctl", "setenv", "ANTHROPIC_API_KEY", key).Run()
		fmt.Fprintf(stdout, "[fak node] set ANTHROPIC_API_KEY in login environment\n")
	} else {
		fmt.Fprintf(stderr, "[fak node] WARNING: ANTHROPIC_API_KEY not set — set it with:\n")
		fmt.Fprintf(stderr, "           launchctl setenv ANTHROPIC_API_KEY \"sk-ant-...\"\n")
	}

	// Load the unit.
	if out, err := exec.Command("launchctl", "load", "-w", plistPath).CombinedOutput(); err != nil {
		fmt.Fprintf(stderr, "fak node install: launchctl load: %v\n%s\n", err, out)
		return 1
	}
	fmt.Fprintf(stdout, "[fak node] loaded %s\n", nodeGatewayLabel)
	fmt.Fprintf(stdout, "           plist:   %s\n", plistPath)
	fmt.Fprintf(stdout, "           log:     %s/serve.log\n", logDir)

	// Wait briefly for the gateway to start.
	localPort := addr[strings.LastIndex(addr, ":")+1:]
	healthURL := "http://127.0.0.1:" + localPort + "/healthz"
	fmt.Fprintf(stdout, "[fak node] waiting for gateway...\n")
	for i := 0; i < 20; i++ {
		time.Sleep(500 * time.Millisecond)
		resp, err := http.Get(healthURL) //nolint:noctx
		if err == nil {
			resp.Body.Close()
			if resp.StatusCode == 200 {
				fmt.Fprintf(stdout, "[fak node] gateway healthy at %s\n", healthURL)
				break
			}
		}
	}

	// Print client connection instructions.
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
		fmt.Fprintf(stdout, "  Or use fak node use from the client:\n")
		fmt.Fprintf(stdout, "    fak node use %s:%s --key %s\n", tailscaleIP, localPort, gatewayKey)
		fmt.Fprintln(stdout, "")
		fmt.Fprintf(stdout, "FAK_GATEWAY_KEY=%s\n", gatewayKey)
		fmt.Fprintf(stdout, "(save this — it is not stored in the plist in plaintext on remote clients)\n")
	} else {
		fmt.Fprintln(stdout, "=== single-machine use ===")
		fmt.Fprintln(stdout, "")
		fmt.Fprintf(stdout, "  fak guard -- claude    # guarded interactive session\n")
		fmt.Fprintln(stdout, "")
		fmt.Fprintf(stdout, "For off-host access (e.g. connecting a Windows client over Tailscale):\n")
		fmt.Fprintf(stdout, "  fak node install --remote\n")
	}
	return 0
}

// ── Linux (systemd --user) ────────────────────────────────────────────────────

const linuxUnitTemplate = `[Unit]
Description=fak serve gateway (always-on Anthropic proxy with tool adjudication)
After=network.target

[Service]
ExecStart={{.FakBin}} serve --provider anthropic --base-url https://api.anthropic.com --addr {{.Addr}} --policy {{.PolicyPath}}{{if .RequireKeyEnv}} --require-key-env {{.RequireKeyEnv}}{{end}}
Restart=always
RestartSec=3
Environment=FAK_AUDIT_JOURNAL={{.LogDir}}/serve_audit.jsonl
{{- if .GatewayKey}}
Environment={{.RequireKeyEnv}}={{.GatewayKey}}
{{- end}}
StandardOutput=append:{{.LogDir}}/serve.log
StandardError=append:{{.LogDir}}/serve.err

[Install]
WantedBy=default.target
`

func nodeInstallLinux(stdout, stderr io.Writer, addr string, offHost bool, keyEnv string, uninstall bool, cfgDir string) int {
	unitDir := filepath.Join(os.Getenv("HOME"), ".config", "systemd", "user")
	unitPath := filepath.Join(unitDir, "fak-serve-gateway.service")

	if uninstall {
		_ = exec.Command("systemctl", "--user", "disable", "--now", "fak-serve-gateway").Run()
		_ = os.Remove(unitPath)
		fmt.Fprintf(stdout, "[fak node] disabled and removed %s\n", unitPath)
		return 0
	}

	logDir := filepath.Join(cfgDir, "logs")
	for _, d := range []string{cfgDir, logDir, unitDir} {
		if err := os.MkdirAll(d, 0755); err != nil {
			fmt.Fprintf(stderr, "fak node install: mkdir %s: %v\n", d, err)
			return 1
		}
	}

	policyPath := filepath.Join(cfgDir, "node-policy.json")
	if err := os.WriteFile(policyPath, nodeDefaultPolicyJSON, 0644); err != nil {
		fmt.Fprintf(stderr, "fak node install: write policy: %v\n", err)
		return 1
	}

	fakBin, err := os.Executable()
	if err != nil {
		fmt.Fprintf(stderr, "fak node install: resolve binary: %v\n", err)
		return 1
	}

	gatewayKey, requireKeyEnv := "", ""
	if offHost {
		requireKeyEnv = keyEnv
		b := make([]byte, 32)
		if _, rerr := rand.Read(b); rerr != nil {
			fmt.Fprintf(stderr, "fak node install: generate key: %v\n", rerr)
			return 1
		}
		gatewayKey = hex.EncodeToString(b)
	}

	tmpl, _ := template.New("unit").Parse(linuxUnitTemplate)
	data := struct{ FakBin, Addr, PolicyPath, LogDir, RequireKeyEnv, GatewayKey string }{
		FakBin: fakBin, Addr: addr, PolicyPath: policyPath, LogDir: logDir,
		RequireKeyEnv: requireKeyEnv, GatewayKey: gatewayKey,
	}
	uf, err := os.Create(unitPath)
	if err != nil {
		fmt.Fprintf(stderr, "fak node install: create unit: %v\n", err)
		return 1
	}
	_ = tmpl.Execute(uf, data)
	uf.Close()

	_ = exec.Command("systemctl", "--user", "daemon-reload").Run()
	if out, err := exec.Command("systemctl", "--user", "enable", "--now", "fak-serve-gateway").CombinedOutput(); err != nil {
		fmt.Fprintf(stderr, "fak node install: systemctl enable: %v\n%s\n", err, out)
		return 1
	}
	fmt.Fprintf(stdout, "[fak node] enabled fak-serve-gateway (systemd --user)\n")
	if offHost {
		fmt.Fprintf(stdout, "FAK_GATEWAY_KEY=%s\n", gatewayKey)
	}
	return 0
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
	}

	// Gateway health — check loopback 8080, then node config URL.
	candidates := []string{"http://127.0.0.1:8080/healthz"}
	if cfg, err := nodeReadCfg(); err == nil && cfg.URL != "" {
		candidates = append(candidates, strings.TrimRight(cfg.URL, "/")+"/healthz")
	}
	for _, u := range candidates {
		cl := &http.Client{Timeout: 3 * time.Second}
		resp, err := cl.Get(u) //nolint:noctx
		if err != nil {
			fmt.Fprintf(stdout, "[fak node] healthz %s: unreachable (%v)\n", u, err)
			rc = 1
			continue
		}
		resp.Body.Close()
		fmt.Fprintf(stdout, "[fak node] healthz %s: %s\n", u, resp.Status)
		if resp.StatusCode != 200 {
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
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	if fs.NArg() < 1 {
		fmt.Fprintln(stderr, "usage: fak node use HOST[:PORT] [--key KEY] [--env]")
		return 2
	}
	hostPort := fs.Arg(0)
	u := hostPort
	if !strings.Contains(u, "://") {
		u = "http://" + u
	}
	// Default port 8080 if absent.
	if !strings.Contains(strings.TrimPrefix(u, "http://"), ":") {
		u += ":8080"
	}
	cfg := nodeCfg{URL: u, Key: *key}

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

// nodeTailscaleIP returns the machine's Tailscale IPv4 address, or "" if unavailable.
func nodeTailscaleIP() string {
	out, err := exec.Command("tailscale", "ip", "-4").Output()
	if err != nil {
		return ""
	}
	return strings.TrimSpace(strings.Split(string(out), "\n")[0])
}
