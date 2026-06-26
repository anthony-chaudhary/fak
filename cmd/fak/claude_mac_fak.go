package main

import (
	"flag"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
)

const (
	defaultClaudeMacGateway = "http://node-macos-a.local:8080"
	defaultClaudeMacModel   = "lmstudio-community/Qwen3.6-27B-GGUF:Q4_K_M"
	defaultClaudeMacSSHHost = "user@node-macos-a.local"
)

func cmdClaudeMacFak(argv []string) {
	os.Exit(runClaudeMacFak(os.Stdout, os.Stderr, argv))
}

func runClaudeMacFak(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("claude-mac-fak", flag.ContinueOnError)
	fs.SetOutput(stderr)
	gatewayURL := fs.String("gateway-url", envOrDefault("FAK_MAC_GATEWAY", defaultClaudeMacGateway), "fak serve gateway on the Mac")
	model := fs.String("model", envOrDefault("FAK_MAC_MODEL", defaultClaudeMacModel), "model id served by the Mac gateway")
	keyEnv := fs.String("gateway-key-env", "FAK_GATEWAY_KEY", "env var holding the gateway bearer")
	fetchKey := fs.Bool("fetch-key", true, "when gateway key env is empty, fetch ~/.fak-gateway-key from the Mac over ssh")
	sshHost := fs.String("ssh-host", envOrDefault("FAK_MAC_SSH_HOST", defaultClaudeMacSSHHost), "ssh host used by --fetch-key")
	sshKey := fs.String("ssh-key", defaultClaudeMacSSHKey(), "ssh identity used by --fetch-key; empty uses ssh defaults")
	configDir := fs.String("claude-config-dir", defaultClaudeMacConfigDir(), "isolated CLAUDE_CONFIG_DIR for this Mac gateway session")
	prompt := fs.String("prompt", "Reply with exactly: OK", "probe prompt; used with --probe")
	probe := fs.Bool("probe", false, "run one headless JSON probe instead of interactive Claude Code")
	interactive := fs.Bool("interactive", false, "explicitly open interactive Claude Code; this is the default")
	dryRun := fs.Bool("dry-run", false, "show the launch plan without starting Claude Code")
	asJSON := fs.Bool("json", false, "emit the launch model as JSON and do not start Claude Code")
	apiTimeoutMS := fs.Int("api-timeout-ms", 1800000, "Claude Code API_TIMEOUT_MS")
	width := fs.Int("width", 120, "dry-run width")
	command := fs.String("command", "claude", "Claude Code command or path to execute")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	if *gatewayURL == "" {
		fmt.Fprintln(stderr, "fak claude-mac-fak: --gateway-url must not be empty")
		return 2
	}
	if *model == "" {
		fmt.Fprintln(stderr, "fak claude-mac-fak: --model must not be empty")
		return 2
	}
	if *apiTimeoutMS < 0 {
		fmt.Fprintln(stderr, "fak claude-mac-fak: --api-timeout-ms must be non-negative")
		return 2
	}
	if *probe && *interactive {
		fmt.Fprintln(stderr, "fak claude-mac-fak: pass either --probe or --interactive, not both")
		return 2
	}
	if err := os.MkdirAll(*configDir, 0o700); err != nil {
		fmt.Fprintf(stderr, "fak claude-mac-fak: create --claude-config-dir: %v\n", err)
		return 1
	}
	if err := ensureClaudeMacGatewayKey(*keyEnv, *fetchKey, *sshHost, *sshKey); err != nil {
		fmt.Fprintf(stderr, "fak claude-mac-fak: %v\n", err)
		return 2
	}

	tuiArgs := []string{
		"agent",
		"--command", *command,
		"--claude-config-dir", *configDir,
		"--gateway-url", *gatewayURL,
		"--gateway-key-env", *keyEnv,
		"--model", *model,
		"--api-timeout-ms", strconv.Itoa(*apiTimeoutMS),
		"--width", strconv.Itoa(*width),
	}
	if *dryRun {
		tuiArgs = append(tuiArgs, "--dry-run")
	}
	if *asJSON {
		tuiArgs = append(tuiArgs, "--json")
	}
	if *probe {
		tuiArgs = append(tuiArgs, "--prompt", *prompt)
	}
	passthrough := fs.Args()
	if len(passthrough) == 0 && *probe && !*asJSON {
		passthrough = []string{"--output-format", "json"}
	}
	if len(passthrough) > 0 {
		tuiArgs = append(tuiArgs, "--")
		tuiArgs = append(tuiArgs, passthrough...)
	}
	return runTUI(stdout, stderr, tuiArgs)
}

func envOrDefault(name, def string) string {
	if v := strings.TrimSpace(os.Getenv(name)); v != "" {
		return v
	}
	return def
}

func defaultClaudeMacConfigDir() string {
	if v := strings.TrimSpace(os.Getenv("FAK_CLAUDE_CONFIG_DIR")); v != "" {
		return v
	}
	return filepath.Join(os.TempDir(), "fak-claude-mac")
}

func defaultClaudeMacSSHKey() string {
	if v := strings.TrimSpace(os.Getenv("FAK_MAC_SSH_KEY")); v != "" {
		return v
	}
	home, err := os.UserHomeDir()
	if err != nil || home == "" {
		return ""
	}
	return filepath.Join(home, ".ssh", "id_ed25519_prod_to_laptop")
}

func ensureClaudeMacGatewayKey(envName string, fetch bool, host, keyPath string) error {
	envName = strings.TrimSpace(envName)
	if envName == "" {
		envName = "FAK_GATEWAY_KEY"
	}
	if strings.TrimSpace(os.Getenv(envName)) != "" {
		return nil
	}
	if !fetch {
		return fmt.Errorf("%s is empty; set it or leave --fetch-key enabled", envName)
	}
	host = strings.TrimSpace(host)
	if host == "" {
		return fmt.Errorf("%s is empty and --ssh-host is empty", envName)
	}
	args := []string{}
	if strings.TrimSpace(keyPath) != "" {
		args = append(args, "-i", keyPath)
	}
	args = append(args, host, "cat ~/.fak-gateway-key")
	cmd := exec.Command("ssh", args...)
	out, err := cmd.Output()
	if err != nil {
		return fmt.Errorf("fetch gateway key over ssh: %w", err)
	}
	secret := strings.TrimSpace(string(out))
	if secret == "" {
		return fmt.Errorf("fetch gateway key over ssh: empty key")
	}
	return os.Setenv(envName, secret)
}
