package main

import (
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"
)

const (
	mcpConfigSchema  = "fak.mcp.config/1"
	mcpStatusSchema  = "fak.mcp.status/1"
	mcpInstallSchema = "fak.mcp.install/1"
)

func cmdMCP(argv []string) {
	os.Exit(runMCP(os.Stdout, os.Stderr, argv))
}

func runMCP(stdout, stderr io.Writer, argv []string) int {
	if len(argv) == 0 {
		fmt.Fprintln(stdout, mcpUsage())
		return 0
	}
	sub := strings.ToLower(strings.TrimSpace(argv[0]))
	rest := argv[1:]
	switch sub {
	case "config":
		return runMCPConfig(stdout, stderr, rest)
	case "install":
		return runMCPInstall(stdout, stderr, rest)
	case "status":
		return runMCPStatus(stdout, stderr, rest)
	case "doctor":
		return runMCPDoctor(stdout, stderr, rest)
	case "uninstall":
		return runMCPUninstall(stdout, stderr, rest)
	case "-h", "--help", "help":
		fmt.Fprintln(stdout, mcpUsage())
		return 0
	default:
		fmt.Fprintf(stderr, "fak mcp: unknown subcommand %q\n%s\n", sub, mcpUsage())
		return 2
	}
}

func mcpUsage() string {
	return strings.TrimSpace(`usage: fak mcp <config|install|status|doctor|uninstall> [flags]
  config     print ready-to-paste MCP server configuration without modifying disk
  install    auto-configure fak as an MCP server (project/claude .mcp.json, cursor .cursor/mcp.json, opencode.json, or codex)
  status     inspect configured MCP server entries and executable resolution
  doctor     run a live MCP initialize handshake against the configured server
  uninstall  remove only the fak MCP server entry (preserves all other servers)`)
}

type mcpJSONServer struct {
	Command string            `json:"command"`
	Args    []string          `json:"args,omitempty"`
	Cwd     string            `json:"cwd,omitempty"`
	Env     map[string]string `json:"env,omitempty"`
}

type opencodeMCPServer struct {
	Type        string            `json:"type"`
	Enabled     bool              `json:"enabled"`
	Command     []string          `json:"command"`
	Environment map[string]string `json:"environment"`
}

func parseMCPFlags(fs *flag.FlagSet, stdout, stderr io.Writer, argv []string) (int, bool) {
	fs.SetOutput(stderr)
	fs.Usage = func() {
		w := stderr
		for _, a := range argv {
			if a == "-h" || a == "--help" || a == "-help" {
				w = stdout
				break
			}
		}
		fmt.Fprintf(w, "Usage of fak %s:\n", fs.Name())
		fs.SetOutput(w)
		fs.PrintDefaults()
		fs.SetOutput(stderr)
	}
	return parseFlagsOrHelp(fs, argv)
}

func normalizeMCPClient(raw, pathHint string) (string, error) {
	c := strings.ToLower(strings.TrimSpace(raw))
	if c == "" || c == "auto" {
		if hinted := inferMCPClientFromPath(pathHint); hinted != "" {
			return hinted, nil
		}
		if _, err := os.Stat(".mcp.json"); err == nil {
			return "project", nil
		}
		if _, err := os.Stat("opencode.json"); err == nil {
			return "opencode", nil
		}
		if info, err := os.Stat(filepath.Join(".cursor", "mcp.json")); err == nil && !info.IsDir() {
			return "cursor", nil
		}
		if info, err := os.Stat(".cursor"); err == nil && info.IsDir() {
			return "cursor", nil
		}
		return "project", nil
	}
	switch c {
	case "project", "claude", "cursor", "opencode", "codex":
		return c, nil
	default:
		return "", fmt.Errorf("unsupported --client %q (want auto, project, claude, cursor, opencode, or codex)", raw)
	}
}

func inferMCPClientFromPath(pathHint string) string {
	p := strings.ToLower(strings.TrimSpace(pathHint))
	if p == "" {
		return ""
	}
	base := filepath.Base(p)
	switch {
	case base == "opencode.json":
		return "opencode"
	case strings.HasSuffix(base, ".toml"):
		return "codex"
	case strings.Contains(filepath.ToSlash(p), ".cursor/"):
		return "cursor"
	case base == ".mcp.json":
		return "project"
	default:
		return ""
	}
}

func defaultMCPConfigPath(client, explicit string) string {
	if p := strings.TrimSpace(explicit); p != "" {
		return p
	}
	switch client {
	case "cursor":
		return filepath.Join(".cursor", "mcp.json")
	case "opencode":
		return "opencode.json"
	case "codex":
		return defaultCodexConfigPath()
	default:
		return ".mcp.json"
	}
}

func resolveMCPCommandAndArgs(portable bool, explicitPolicy, expose string) (string, []string, string) {
	cmd := "fak"
	if !portable {
		if exe, err := os.Executable(); err == nil && strings.TrimSpace(exe) != "" {
			cmd = exe
		}
	}
	args := []string{"serve", "--stdio"}
	policy := strings.TrimSpace(explicitPolicy)
	if policy == "" {
		// In a fak repo checkout, wire examples/dev-agent-policy.json if present;
		// in standalone / external projects, omit --policy so serve --stdio runs
		// cleanly with default_open and zero repo-relative dependencies.
		if wm := detectWorkspaceMode(); wm.HasDevPolicy {
			if portable {
				policy = filepath.ToSlash(filepath.Join("examples", "dev-agent-policy.json"))
			} else {
				policy = filepath.Join(wm.RepoRoot, "examples", "dev-agent-policy.json")
			}
		}
	}
	if policy != "" {
		args = append(args, "--policy", policy)
	}
	if exp := strings.TrimSpace(expose); exp != "" {
		args = append(args, "--expose", exp)
	}
	return cmd, args, policy
}

func formatCodexMCPSnippet(server, command string, args []string) string {
	quotedArgs := make([]string, len(args))
	for i, a := range args {
		b, _ := json.Marshal(a)
		quotedArgs[i] = string(b)
	}
	cmdJSON, _ := json.Marshal(command)
	return fmt.Sprintf("[mcp_servers.%s]\ncommand = %s\nargs = [%s]\n", server, string(cmdJSON), strings.Join(quotedArgs, ", "))
}

func runMCPConfig(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("mcp config", flag.ContinueOnError)
	clientFlag := fs.String("client", "auto", "target client: auto, project, claude, cursor, opencode, or codex")
	server := fs.String("server", "fak", "MCP server key name")
	policy := fs.String("policy", "", "optional capability-floor policy JSON path")
	expose := fs.String("expose", "", "optional MCP tool allowlist glob (e.g. 'fak_capabilities,fak_memory_*')")
	portable := fs.Bool("portable", false, "emit bare 'fak' command instead of the current executable path")
	asJSON := fs.Bool("json", false, "emit structured JSON report")
	if rc, ok := parseMCPFlags(fs, stdout, stderr, argv); !ok {
		return rc
	}
	client, err := normalizeMCPClient(*clientFlag, "")
	if err != nil {
		fmt.Fprintf(stderr, "fak mcp config: %v\n", err)
		return 2
	}
	cmd, args, resolvedPolicy := resolveMCPCommandAndArgs(*portable, *policy, *expose)
	targetPath := defaultMCPConfigPath(client, "")

	if client == "codex" {
		tomlSnippet := formatCodexMCPSnippet(*server, cmd, args)
		if *asJSON {
			writeJSONOut(stdout, map[string]any{
				"schema":         mcpConfigSchema,
				"client":         client,
				"server":         *server,
				"target_path":    targetPath,
				"command":        cmd,
				"args":           args,
				"policy_path":    resolvedPolicy,
				"workspace_mode": detectWorkspaceMode(),
				"toml":           tomlSnippet,
			})
			return 0
		}
		fmt.Fprint(stdout, tomlSnippet)
		return 0
	}

	var snippet map[string]any
	if client == "opencode" {
		fullCmd := append([]string{cmd}, args...)
		snippet = map[string]any{
			"mcp": map[string]any{
				*server: opencodeMCPServer{
					Type:        "local",
					Enabled:     true,
					Command:     fullCmd,
					Environment: map[string]string{},
				},
			},
		}
	} else {
		snippet = map[string]any{
			"mcpServers": map[string]any{
				*server: mcpJSONServer{
					Command: cmd,
					Args:    args,
					Cwd:     ".",
				},
			},
		}
	}
	if *asJSON {
		writeJSONOut(stdout, map[string]any{
			"schema":         mcpConfigSchema,
			"client":         client,
			"server":         *server,
			"target_path":    targetPath,
			"command":        cmd,
			"args":           args,
			"policy_path":    resolvedPolicy,
			"workspace_mode": detectWorkspaceMode(),
			"config":         snippet,
		})
		return 0
	}
	writeJSONOut(stdout, snippet)
	return 0
}

func runMCPInstall(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("mcp install", flag.ContinueOnError)
	clientFlag := fs.String("client", "auto", "target client: auto, project, claude, cursor, opencode, or codex")
	pathFlag := fs.String("path", "", "target config file path (default depends on --client)")
	server := fs.String("server", "fak", "MCP server key name")
	policy := fs.String("policy", "", "optional capability-floor policy JSON path")
	expose := fs.String("expose", "", "optional MCP tool allowlist glob")
	portable := fs.Bool("portable", false, "write bare 'fak' command instead of the current executable path")
	dryRun := fs.Bool("dry-run", false, "preview changes without writing files")
	force := fs.Bool("force", false, "overwrite an existing entry for --server")
	skipVerify := fs.Bool("skip-verify", false, "skip post-install MCP initialize handshake verification")
	timeout := fs.Duration("timeout", 10*time.Second, "initialize verification timeout")
	asJSON := fs.Bool("json", false, "emit structured JSON report")
	if rc, ok := parseMCPFlags(fs, stdout, stderr, argv); !ok {
		return rc
	}
	client, err := normalizeMCPClient(*clientFlag, *pathFlag)
	if err != nil {
		return mcpFail(stderr, *asJSON, "install", "INVALID_CLIENT", err.Error())
	}
	if client == "codex" {
		if *dryRun {
			cfgArgv := []string{"--client", "codex", "--server", *server, "--json"}
			if *policy != "" {
				cfgArgv = append(cfgArgv, "--policy", *policy)
			}
			if *expose != "" {
				cfgArgv = append(cfgArgv, "--expose", *expose)
			}
			if *portable {
				cfgArgv = append(cfgArgv, "--portable")
			}
			return runMCPConfig(stdout, stderr, cfgArgv)
		}
		forwarded := []string{"install", "--server", *server, "--timeout", timeout.String()}
		if *pathFlag != "" {
			forwarded = append(forwarded, "--config", *pathFlag)
		}
		if *policy != "" {
			forwarded = append(forwarded, "--policy", *policy)
		}
		if *skipVerify {
			forwarded = append(forwarded, "--skip-verify")
		}
		if *asJSON {
			forwarded = append(forwarded, "--json")
		}
		return runCodexMCP(stdout, stderr, forwarded)
	}

	targetPath := defaultMCPConfigPath(client, *pathFlag)
	if absTarget, err := filepath.Abs(targetPath); err == nil {
		targetPath = absTarget
	}

	cmd, args, resolvedPolicy := resolveMCPCommandAndArgs(*portable, *policy, *expose)

	rootObj := map[string]any{}
	existed := false
	if raw, readErr := os.ReadFile(targetPath); readErr == nil && len(bytes.TrimSpace(raw)) > 0 {
		existed = true
		if err := json.Unmarshal(raw, &rootObj); err != nil {
			return mcpFail(stderr, *asJSON, "install", "CONFIG_MALFORMED", fmt.Sprintf("parse %s: %v", targetPath, err))
		}
	} else if readErr != nil && !os.IsNotExist(readErr) {
		return mcpFail(stderr, *asJSON, "install", "CONFIG_READ", readErr.Error())
	}

	sectionKey := "mcpServers"
	if client == "opencode" {
		sectionKey = "mcp"
	}

	serversMap := map[string]any{}
	if existingServers, ok := rootObj[sectionKey]; ok && existingServers != nil {
		cast, ok := existingServers.(map[string]any)
		if !ok {
			return mcpFail(stderr, *asJSON, "install", "CONFIG_MALFORMED", fmt.Sprintf("%s: %s must be a JSON object", targetPath, sectionKey))
		}
		for k, v := range cast {
			serversMap[k] = v
		}
	}

	if _, already := serversMap[*server]; already && !*force {
		return mcpFail(stderr, *asJSON, "install", "ALREADY_CONFIGURED",
			fmt.Sprintf("%s already defines %s.%s (pass --force to overwrite)", targetPath, sectionKey, *server))
	}

	if client == "opencode" {
		serversMap[*server] = opencodeMCPServer{
			Type:        "local",
			Enabled:     true,
			Command:     append([]string{cmd}, args...),
			Environment: map[string]string{},
		}
	} else {
		serversMap[*server] = mcpJSONServer{
			Command: cmd,
			Args:    args,
			Cwd:     ".",
		}
	}
	rootObj[sectionKey] = serversMap

	encoded, err := json.MarshalIndent(rootObj, "", "  ")
	if err != nil {
		return mcpFail(stderr, *asJSON, "install", "CONFIG_ENCODE", err.Error())
	}
	encoded = append(encoded, '\n')

	// Verify the live MCP stdio initialize handshake BEFORE mutating the config file,
	// so a broken binary/policy never leaves a partially verified config on disk.
	verified := false
	if !*dryRun && !*skipVerify {
		rep := diagnoseMCP(*server, "", cmd, args, *timeout)
		verified = rep.OK
		if !rep.OK {
			verifyCause := ""
			for _, s := range rep.Stages {
				if s.Status == "fail" {
					verifyCause = s.Cause + ": " + s.Detail
					break
				}
			}
			if verifyCause == "" {
				verifyCause = "MCP initialize probe failed"
			}
			return mcpFail(stderr, *asJSON, "install", "HANDSHAKE_FAILED", verifyCause)
		}
	}

	if !*dryRun {
		if dir := filepath.Dir(targetPath); dir != "" && dir != "." {
			if err := os.MkdirAll(dir, 0o755); err != nil {
				return mcpFail(stderr, *asJSON, "install", "MKDIR_FAILED", err.Error())
			}
		}
		if err := writeFileAtomic(targetPath, encoded, 0o644); err != nil {
			return mcpFail(stderr, *asJSON, "install", "CONFIG_WRITE", err.Error())
		}
	}

	report := map[string]any{
		"schema":         mcpInstallSchema,
		"ok":             true,
		"client":         client,
		"server":         *server,
		"config_path":    targetPath,
		"existed":        existed,
		"dry_run":        *dryRun,
		"command":        cmd,
		"args":           args,
		"policy_path":    resolvedPolicy,
		"verified":       verified,
		"workspace_mode": detectWorkspaceMode(),
	}
	if *asJSON {
		writeJSONOut(stdout, report)
		return 0
	}
	action := "configured"
	if *dryRun {
		action = "would configure"
	}
	fmt.Fprintf(stdout, "fak mcp: %s %s (server=%s, client=%s, verified=%t)\n", action, targetPath, *server, client, verified)
	return 0
}

func runMCPStatus(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("mcp status", flag.ContinueOnError)
	clientFlag := fs.String("client", "auto", "target client: auto, project, claude, cursor, opencode, or codex")
	pathFlag := fs.String("path", "", "explicit config path")
	server := fs.String("server", "fak", "MCP server key name")
	asJSON := fs.Bool("json", false, "emit structured JSON report")
	if rc, ok := parseMCPFlags(fs, stdout, stderr, argv); !ok {
		return rc
	}
	client, err := normalizeMCPClient(*clientFlag, *pathFlag)
	if err != nil {
		return mcpFail(stderr, *asJSON, "status", "INVALID_CLIENT", err.Error())
	}
	if client == "codex" {
		forwarded := []string{"status", "--server", *server}
		if *pathFlag != "" {
			forwarded = append(forwarded, "--config", *pathFlag)
		}
		if *asJSON {
			forwarded = append(forwarded, "--json")
		}
		return runCodexMCP(stdout, stderr, forwarded)
	}
	targetPath := defaultMCPConfigPath(client, *pathFlag)
	absTarget, _ := filepath.Abs(targetPath)

	status := map[string]any{
		"schema":         mcpStatusSchema,
		"client":         client,
		"server":         *server,
		"config_path":    absTarget,
		"configured":     false,
		"workspace_mode": detectWorkspaceMode(),
	}

	entry, ok, readErr := readMCPConfigServer(client, absTarget, *server)
	if readErr != nil {
		status["error"] = readErr.Error()
	} else if ok {
		status["configured"] = true
		status["command"] = entry.Command
		status["args"] = entry.Args
		resolved, lookErr := exec.LookPath(entry.Command)
		status["executable_ok"] = lookErr == nil
		if lookErr == nil {
			status["resolved_command"] = resolved
		} else {
			status["executable_error"] = lookErr.Error()
		}
	}

	if *asJSON {
		writeJSONOut(stdout, status)
		return 0
	}
	if ok {
		fmt.Fprintf(stdout, "fak mcp: configured in %s (server=%s, client=%s, command=%s, executable_ok=%v)\n",
			absTarget, *server, client, entry.Command, status["executable_ok"])
	} else {
		fmt.Fprintf(stdout, "fak mcp: not configured in %s (server=%s, client=%s)\n  run: fak mcp install --client %s\n",
			absTarget, *server, client, client)
	}
	return 0
}

func runMCPDoctor(stdout, stderr io.Writer, argv []string) int {
	// If --config or --command is passed, delegate straight to runDoctorMCP (Codex/explicit).
	for _, a := range argv {
		if strings.HasPrefix(a, "--config") || strings.HasPrefix(a, "--command") {
			return runDoctorMCP(stdout, stderr, argv)
		}
	}
	fs := flag.NewFlagSet("mcp doctor", flag.ContinueOnError)
	clientFlag := fs.String("client", "auto", "target client: auto, project, claude, cursor, opencode, or codex")
	pathFlag := fs.String("path", "", "explicit config path")
	server := fs.String("server", "fak", "MCP server key name")
	timeout := fs.Duration("timeout", 10*time.Second, "initialize response timeout")
	asJSON := fs.Bool("json", false, "emit structured JSON report")
	if rc, ok := parseMCPFlags(fs, stdout, stderr, argv); !ok {
		return rc
	}
	client, err := normalizeMCPClient(*clientFlag, *pathFlag)
	if err != nil {
		return mcpFail(stderr, *asJSON, "doctor", "INVALID_CLIENT", err.Error())
	}
	if client == "codex" {
		forwarded := []string{"--server", *server, "--timeout", timeout.String()}
		if *pathFlag != "" {
			forwarded = append(forwarded, "--config", *pathFlag)
		}
		if *asJSON {
			forwarded = append(forwarded, "--json")
		}
		return runDoctorMCP(stdout, stderr, forwarded)
	}

	targetPath := defaultMCPConfigPath(client, *pathFlag)
	absTarget, _ := filepath.Abs(targetPath)
	entry, ok, readErr := readMCPConfigServer(client, absTarget, *server)
	if readErr == nil && ok {
		rep := diagnoseMCP(*server, "", entry.Command, entry.Args, *timeout)
		rep.ConfigPath = absTarget
		if *asJSON {
			writeJSONOut(stdout, rep)
		} else {
			writeDoctorMCPHuman(stdout, rep)
		}
		if !rep.OK {
			return 1
		}
		return 0
	}

	// If explicit client was requested or target path doesn't exist and --client wasn't auto,
	// report the missing project/opencode/cursor config clearly; otherwise try Codex fallback.
	if readErr != nil || *clientFlag != "auto" || *pathFlag != "" {
		detail := fmt.Sprintf("%s has no entry for server %q", absTarget, *server)
		cause := "NOT_CONFIGURED"
		if readErr != nil {
			detail = readErr.Error()
			cause = "CONFIG_READ_ERROR"
		}
		rep := doctorMCPReport{
			Schema:     doctorMCPSchema,
			Server:     *server,
			ConfigPath: absTarget,
			OK:         false,
			Stages: []doctorMCPStage{{
				Name:        "config",
				Status:      "fail",
				Cause:       cause,
				Detail:      detail,
				Remediation: fmt.Sprintf("fak mcp install --client %s", client),
			}},
		}
		if *asJSON {
			writeJSONOut(stdout, rep)
		} else {
			writeDoctorMCPHuman(stdout, rep)
		}
		return 1
	}

	forwarded := []string{"--server", *server, "--timeout", timeout.String()}
	if *asJSON {
		forwarded = append(forwarded, "--json")
	}
	return runDoctorMCP(stdout, stderr, forwarded)
}

func runMCPUninstall(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("mcp uninstall", flag.ContinueOnError)
	clientFlag := fs.String("client", "auto", "target client: auto, project, claude, cursor, opencode, or codex")
	pathFlag := fs.String("path", "", "target config file path")
	server := fs.String("server", "fak", "MCP server key name")
	asJSON := fs.Bool("json", false, "emit structured JSON report")
	if rc, ok := parseMCPFlags(fs, stdout, stderr, argv); !ok {
		return rc
	}
	client, err := normalizeMCPClient(*clientFlag, *pathFlag)
	if err != nil {
		return mcpFail(stderr, *asJSON, "uninstall", "INVALID_CLIENT", err.Error())
	}
	if client == "codex" {
		forwarded := []string{"uninstall", "--server", *server}
		if *pathFlag != "" {
			forwarded = append(forwarded, "--config", *pathFlag)
		}
		if *asJSON {
			forwarded = append(forwarded, "--json")
		}
		return runCodexMCP(stdout, stderr, forwarded)
	}
	targetPath := defaultMCPConfigPath(client, *pathFlag)
	absTarget, _ := filepath.Abs(targetPath)
	raw, err := os.ReadFile(absTarget)
	if err != nil {
		if os.IsNotExist(err) {
			if *asJSON {
				writeJSONOut(stdout, map[string]any{"ok": true, "removed": false, "client": client, "config_path": absTarget})
			} else {
				fmt.Fprintf(stdout, "fak mcp: %s does not exist (nothing to remove)\n", absTarget)
			}
			return 0
		}
		return mcpFail(stderr, *asJSON, "uninstall", "CONFIG_READ", err.Error())
	}
	var rootObj map[string]any
	if err := json.Unmarshal(raw, &rootObj); err != nil {
		return mcpFail(stderr, *asJSON, "uninstall", "CONFIG_MALFORMED", err.Error())
	}
	sectionKey := "mcpServers"
	if client == "opencode" {
		sectionKey = "mcp"
	}
	serversMap, _ := rootObj[sectionKey].(map[string]any)
	if serversMap == nil {
		if *asJSON {
			writeJSONOut(stdout, map[string]any{"ok": true, "removed": false, "client": client, "config_path": absTarget})
		} else {
			fmt.Fprintf(stdout, "fak mcp: no %s.%s in %s\n", sectionKey, *server, absTarget)
		}
		return 0
	}
	if _, exists := serversMap[*server]; !exists {
		if *asJSON {
			writeJSONOut(stdout, map[string]any{"ok": true, "removed": false, "client": client, "config_path": absTarget})
		} else {
			fmt.Fprintf(stdout, "fak mcp: no %s.%s in %s\n", sectionKey, *server, absTarget)
		}
		return 0
	}
	delete(serversMap, *server)
	rootObj[sectionKey] = serversMap
	encoded, err := json.MarshalIndent(rootObj, "", "  ")
	if err != nil {
		return mcpFail(stderr, *asJSON, "uninstall", "CONFIG_ENCODE", err.Error())
	}
	encoded = append(encoded, '\n')
	if err := writeFileAtomic(absTarget, encoded, 0o644); err != nil {
		return mcpFail(stderr, *asJSON, "uninstall", "CONFIG_WRITE", err.Error())
	}
	if *asJSON {
		writeJSONOut(stdout, map[string]any{"ok": true, "removed": true, "client": client, "server": *server, "config_path": absTarget})
	} else {
		fmt.Fprintf(stdout, "fak mcp: removed %s.%s from %s\n", sectionKey, *server, absTarget)
	}
	return 0
}

func readMCPConfigServer(client, path, server string) (mcpJSONServer, bool, error) {
	if client == "opencode" {
		return readOpenCodeMCPServer(path, server)
	}
	return readMCPJSONServer(path, server)
}

func readMCPJSONServer(path, server string) (mcpJSONServer, bool, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return mcpJSONServer{}, false, nil
		}
		return mcpJSONServer{}, false, err
	}
	var parsed struct {
		MCPServers map[string]mcpJSONServer `json:"mcpServers"`
	}
	if err := json.Unmarshal(raw, &parsed); err != nil {
		return mcpJSONServer{}, false, err
	}
	entry, ok := parsed.MCPServers[server]
	return entry, ok, nil
}

func readOpenCodeMCPServer(path, server string) (mcpJSONServer, bool, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return mcpJSONServer{}, false, nil
		}
		return mcpJSONServer{}, false, err
	}
	var parsed struct {
		MCP map[string]opencodeMCPServer `json:"mcp"`
	}
	if err := json.Unmarshal(raw, &parsed); err != nil {
		return mcpJSONServer{}, false, err
	}
	entry, ok := parsed.MCP[server]
	if !ok || len(entry.Command) == 0 {
		return mcpJSONServer{}, false, nil
	}
	return mcpJSONServer{
		Command: entry.Command[0],
		Args:    entry.Command[1:],
		Env:     entry.Environment,
	}, true, nil
}

func writeJSONOut(w io.Writer, v any) {
	enc := json.NewEncoder(w)
	enc.SetEscapeHTML(false)
	enc.SetIndent("", "  ")
	_ = enc.Encode(v)
}

func mcpFail(stderr io.Writer, asJSON bool, op, code, msg string) int {
	if asJSON {
		writeJSONOut(stderr, map[string]any{
			"ok":    false,
			"op":    op,
			"code":  code,
			"error": msg,
		})
	} else {
		fmt.Fprintf(stderr, "fak mcp %s: %s (%s)\n", op, msg, code)
	}
	return 1
}
