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
//	                           wrapper. Linux: systemd --user unit. Windows: an ONSTART
//	                           Scheduled Task. Prints client env lines after install.
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

// nodeWindowsTaskName is the Scheduled Task name the Windows install registers.
const nodeWindowsTaskName = "FakServeGateway"

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
	case "windows":
		return nodeInstallWindows(stdout, stderr, bindAddr, offHost, *keyEnv, *uninstall, cfgDir)
	default:
		fmt.Fprintf(stderr, "fak node install: not yet supported on %s — use scripts/dogfood-claude.sh\n", runtime.GOOS)
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

	nodePrintClientLines(stdout, stderr, offHost, gatewayKey, localPort)
	return 0
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
		if existing := os.Getenv(keyEnv); existing != "" {
			gatewayKey = existing
		} else {
			b := make([]byte, 32)
			if _, rerr := rand.Read(b); rerr != nil {
				fmt.Fprintf(stderr, "fak node install: generate key: %v\n", rerr)
				return 1
			}
			gatewayKey = hex.EncodeToString(b)
		}
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
	localPort := addr[strings.LastIndex(addr, ":")+1:]
	nodePrintClientLines(stdout, stderr, offHost, gatewayKey, localPort)
	return 0
}

// ── Windows (Scheduled Task) ──────────────────────────────────────────────────

// nodeWindowsRunnerTemplate is the .cmd the Scheduled Task launches. A Scheduled Task
// runs an exe, not an env-carrying service, so the runner sets FAK_AUDIT_JOURNAL (and the
// bearer secret, off-host) before exec'ing fak serve — the Windows analog of the plist's
// EnvironmentVariables / the unit's Environment= lines.
const nodeWindowsRunnerTemplate = `@echo off
rem Written by: fak node install — regenerate with: fak node install
set "FAK_AUDIT_JOURNAL={{.LogDir}}\serve_audit.jsonl"
{{- if .GatewayKey}}
set "{{.RequireKeyEnv}}={{.GatewayKey}}"
{{- end}}
"{{.FakBin}}" serve --provider anthropic --base-url https://api.anthropic.com --addr {{.Addr}} --policy "{{.PolicyPath}}"{{if .RequireKeyEnv}} --require-key-env {{.RequireKeyEnv}}{{end}} >> "{{.LogDir}}\serve.log" 2>> "{{.LogDir}}\serve.err"
`

func nodeInstallWindows(stdout, stderr io.Writer, addr string, offHost bool, keyEnv string, uninstall bool, cfgDir string) int {
	if uninstall {
		_ = exec.Command("schtasks", "/End", "/TN", nodeWindowsTaskName).Run()
		out, err := exec.Command("schtasks", "/Delete", "/TN", nodeWindowsTaskName, "/F").CombinedOutput()
		if err != nil {
			fmt.Fprintf(stderr, "fak node install: schtasks /Delete: %v\n%s\n", err, out)
			return 1
		}
		fmt.Fprintf(stdout, "[fak node] removed Scheduled Task %s\n", nodeWindowsTaskName)
		return 0
	}

	logDir := filepath.Join(cfgDir, "logs")
	for _, d := range []string{cfgDir, logDir} {
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
		if existing := os.Getenv(keyEnv); existing != "" {
			gatewayKey = existing
		} else {
			b := make([]byte, 32)
			if _, rerr := rand.Read(b); rerr != nil {
				fmt.Fprintf(stderr, "fak node install: generate key: %v\n", rerr)
				return 1
			}
			gatewayKey = hex.EncodeToString(b)
		}
	}

	// Render the runner .cmd the task launches.
	runnerPath := filepath.Join(cfgDir, "serve-runner.cmd")
	tmpl, err := template.New("runner").Parse(nodeWindowsRunnerTemplate)
	if err != nil {
		fmt.Fprintf(stderr, "fak node install: parse runner template: %v\n", err)
		return 1
	}
	data := struct{ FakBin, Addr, PolicyPath, LogDir, RequireKeyEnv, GatewayKey string }{
		FakBin: fakBin, Addr: addr, PolicyPath: policyPath, LogDir: logDir,
		RequireKeyEnv: requireKeyEnv, GatewayKey: gatewayKey,
	}
	rf, err := os.Create(runnerPath)
	if err != nil {
		fmt.Fprintf(stderr, "fak node install: create runner: %v\n", err)
		return 1
	}
	if err := tmpl.Execute(rf, data); err != nil {
		rf.Close()
		fmt.Fprintf(stderr, "fak node install: render runner: %v\n", err)
		return 1
	}
	rf.Close()

	// Register an always-on Scheduled Task: /SC ONSTART keeps the gateway resident across
	// reboots (it is a daemon, not a periodic tick), /RL LIMITED runs it as the current
	// user without elevation, /F overwrites a prior install. Then /Run it once so the
	// gateway comes up now, not only after the next boot.
	taskRun := fmt.Sprintf("cmd.exe /c \"%s\"", runnerPath)
	if out, err := exec.Command("schtasks", "/Create", "/TN", nodeWindowsTaskName,
		"/SC", "ONSTART", "/RL", "LIMITED", "/F", "/TR", taskRun).CombinedOutput(); err != nil {
		fmt.Fprintf(stderr, "fak node install: schtasks /Create: %v\n%s\n", err, out)
		return 1
	}
	if out, err := exec.Command("schtasks", "/Run", "/TN", nodeWindowsTaskName).CombinedOutput(); err != nil {
		// Non-fatal: the task is registered and will start at boot even if the immediate
		// run could not be kicked off.
		fmt.Fprintf(stderr, "[fak node] note: schtasks /Run did not start the task now (%v): %s\n", err, out)
	}
	fmt.Fprintf(stdout, "[fak node] registered Scheduled Task %s (/SC ONSTART)\n", nodeWindowsTaskName)
	fmt.Fprintf(stdout, "           runner:  %s\n", runnerPath)
	fmt.Fprintf(stdout, "           log:     %s\\serve.log\n", logDir)

	// Wait briefly for the gateway to answer.
	localPort := addr[strings.LastIndex(addr, ":")+1:]
	fmt.Fprintf(stdout, "[fak node] waiting for gateway...\n")
	for i := 0; i < 20; i++ {
		time.Sleep(500 * time.Millisecond)
		if _, ok := nodeProbeHealth("http://127.0.0.1:" + localPort); ok {
			fmt.Fprintf(stdout, "[fak node] gateway healthy at http://127.0.0.1:%s/healthz\n", localPort)
			break
		}
	}

	if key := os.Getenv("ANTHROPIC_API_KEY"); key == "" {
		fmt.Fprintf(stderr, "[fak node] WARNING: ANTHROPIC_API_KEY not set for the task's user — set it (e.g. setx ANTHROPIC_API_KEY \"sk-ant-...\") and re-run, or the gateway has no upstream credential\n")
	}

	nodePrintClientLines(stdout, stderr, offHost, gatewayKey, localPort)
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
	case "windows":
		out, _ := exec.Command("schtasks", "/Query", "/TN", nodeWindowsTaskName, "/FO", "LIST").Output()
		if len(out) > 0 {
			fmt.Fprintf(stdout, "[fak node] schtasks %s:\n%s\n", nodeWindowsTaskName, strings.TrimSpace(string(out)))
		} else {
			fmt.Fprintf(stdout, "[fak node] schtasks: %s not installed\n", nodeWindowsTaskName)
			rc = 1
		}
	}

	// Gateway health — check loopback 8080, then node config URL. nodeProbeHealth takes a
	// base URL and appends /healthz itself, so the candidates are base URLs.
	candidates := []string{"http://127.0.0.1:8080"}
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
