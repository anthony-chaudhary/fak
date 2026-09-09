package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/childprocess"
	"github.com/anthony-chaudhary/fak/internal/pathutil"
	"github.com/anthony-chaudhary/fak/internal/procguard"
	"github.com/anthony-chaudhary/fak/internal/projectassets"
)

// claudeLaunchOptions contains configuration for launching raw Claude Code against fak serve.
type claudeLaunchOptions struct {
	dryRun          bool
	printEnv        bool
	probePrompt     string
	serverURL       string
	model           string
	apiKey          string
	apiKeyEnv       string
	apiTimeoutMS    int
	claudeConfigDir string
	command         string
	skipPermissions bool
	quiet           bool
	noProbe         bool
	passthrough     []string
}

type claudeStatusReport struct {
	OK      bool   `json:"ok"`
	Model   string `json:"model"`
	Backend string `json:"engine"`
}

var (
	claudeLaunchRun     = execClaudeLaunchChild
	claudeStatusFetcher = fetchClaudeServerStatus
)

func cmdClaude(argv []string) {
	os.Exit(runClaude(os.Stdout, os.Stderr, argv))
}

func defaultClaudeAddr() string {
	if v := strings.TrimSpace(os.Getenv("ANTHROPIC_BASE_URL")); v != "" {
		return projectassets.NormalizeClaudeBaseURL(v)
	}
	if v := strings.TrimSpace(os.Getenv("FAK_MAC_GATEWAY")); v != "" {
		return projectassets.NormalizeClaudeBaseURL(v)
	}
	return projectassets.DefaultClaudeBaseURL
}

func defaultClaudeModel() string {
	if v := strings.TrimSpace(os.Getenv("ANTHROPIC_MODEL")); v != "" {
		return v
	}
	if v := strings.TrimSpace(os.Getenv("FAK_MAC_MODEL")); v != "" {
		return v
	}
	return projectassets.DefaultClaudeModelID
}

func defaultClaudeAPIKey(keyEnv string) string {
	if keyEnv != "" {
		if v := strings.TrimSpace(os.Getenv(keyEnv)); v != "" {
			return v
		}
	}
	if v := strings.TrimSpace(os.Getenv("FAK_GATEWAY_KEY")); v != "" {
		return v
	}
	if v := strings.TrimSpace(os.Getenv("ANTHROPIC_API_KEY")); v != "" {
		return v
	}
	return projectassets.DefaultClaudeAPIKey
}

func runClaude(stdout, stderr io.Writer, argv []string) int {
	if len(argv) > 0 && argv[0] == "config" {
		return runClaudeConfig(stdout, stderr, argv[1:])
	}

	fs := flag.NewFlagSet("claude", flag.ContinueOnError)
	fs.SetOutput(stderr)
	verbFlagUsage(fs, "claude")

	dryRun := fs.Bool("dry-run", false, "print the Claude Code command and environment and exit without launching")
	printEnv := fs.Bool("print-env", false, "print shell export statements for Claude Code environment and exit")
	probePrompt := fs.String("probe", "", "run a single headless probe turn with this prompt and exit with JSON output")
	promptFlag := fs.String("prompt", "", "alias for probe prompt (or pass -p)")
	fs.StringVar(promptFlag, "p", "", "alias for probe prompt")
	serverURL := fs.String("gateway-url", defaultClaudeAddr(), "fak serve backend address URL")
	baseURL := fs.String("base-url", "", "alias for --gateway-url")
	model := fs.String("model", "", "model identifier (default: auto-detected from fak serve, or qwen38:27b-q4 on Mac)")
	apiKeyEnv := fs.String("api-key-env", "", "env var holding the server bearer/API key")
	apiKey := fs.String("api-key", "", "explicit API key string (default: fak-local-dogfood or $FAK_GATEWAY_KEY)")
	apiTimeoutMS := fs.Int("api-timeout-ms", 1800000, "Claude Code API timeout in milliseconds (default: 1800000 = 30m)")
	claudeConfigDir := fs.String("claude-config-dir", "", "isolated CLAUDE_CONFIG_DIR for this session")
	command := fs.String("command", "claude", "Claude Code executable name or path")
	skipPermissions := fs.Bool("skip-permissions", true, "pass --dangerously-skip-permissions to Claude Code for non-interactive execution")
	dangerouslySkipPermissions := fs.Bool("dangerously-skip-permissions", true, "pass --dangerously-skip-permissions to Claude Code")
	quiet := fs.Bool("quiet", false, "suppress launcher status messages")
	noProbe := fs.Bool("no-probe", false, "skip preflight health probe of fak serve backend")

	fs.Usage = func() {
		fmt.Fprintln(stderr, "usage: fak claude [launcher flags] [-- <claude args...>]")
		fmt.Fprintln(stderr, "       fak claude config [--write] [--addr ADDR] [--model MODEL]")
		fmt.Fprintln(stderr, "")
		fmt.Fprintln(stderr, "First-class support for Claude Code as harness with fak serve on Mac as backend.")
		fmt.Fprintln(stderr, "Runs Claude Code directly against the native Anthropic /v1/messages server without guard.")
		fmt.Fprintln(stderr, "")
		fmt.Fprintln(stderr, "examples:")
		fmt.Fprintln(stderr, "  fak claude                                    # interactive session on local Mac fak serve")
		fmt.Fprintln(stderr, "  fak claude --dry-run                          # preview environment and command")
		fmt.Fprintln(stderr, "  fak claude --print-env                        # print shell export lines")
		fmt.Fprintln(stderr, "  fak claude --probe \"Reply with: pong\"         # headless JSON probe turn")
		fmt.Fprintln(stderr, "  fak claude config --write                     # write .claude/settings.json")
		fmt.Fprintln(stderr, "")
		fs.PrintDefaults()
	}

	if !parseFlags(fs, argv) {
		return 2
	}

	effectiveServer := *serverURL
	if *baseURL != "" {
		effectiveServer = *baseURL
	}
	effectiveServer = projectassets.NormalizeClaudeBaseURL(effectiveServer)

	effectivePrompt := *probePrompt
	if effectivePrompt == "" && *promptFlag != "" {
		effectivePrompt = *promptFlag
	}

	effectiveKey := *apiKey
	if effectiveKey == "" {
		effectiveKey = defaultClaudeAPIKey(*apiKeyEnv)
	}

	effectiveModel := strings.TrimSpace(*model)
	var statusInfo *claudeStatusReport
	var probeErr error

	if !*noProbe && !*printEnv {
		statusInfo, probeErr = claudeStatusFetcher(effectiveServer, 2*time.Second)
	}

	// Auto-detect served model if caller did not supply one explicitly
	if effectiveModel == "" {
		if statusInfo != nil && strings.TrimSpace(statusInfo.Model) != "" {
			effectiveModel = statusInfo.Model
		} else {
			effectiveModel = defaultClaudeModel()
		}
	}

	launch := claudeLaunchOptions{
		dryRun:          *dryRun,
		printEnv:        *printEnv,
		probePrompt:     effectivePrompt,
		serverURL:       effectiveServer,
		model:           effectiveModel,
		apiKey:          effectiveKey,
		apiKeyEnv:       *apiKeyEnv,
		apiTimeoutMS:    *apiTimeoutMS,
		claudeConfigDir: pathutil.ExpandTilde(*claudeConfigDir),
		command:         *command,
		skipPermissions: *skipPermissions && *dangerouslySkipPermissions,
		quiet:           *quiet,
		noProbe:         *noProbe,
		passthrough:     fs.Args(),
	}

	if launch.printEnv {
		printClaudeShellEnv(stdout, launch)
		return 0
	}

	argvOut := buildClaudeLaunchArgv(launch)
	envMap := buildClaudeLaunchEnvMap(launch)
	envList := mergeEnv(os.Environ(), envMap)

	if launch.dryRun {
		fmt.Fprintln(stderr, "fak claude: dry-run - not launching")
		fmt.Fprintf(stderr, "  gateway     = %s\n", launch.serverURL)
		fmt.Fprintf(stderr, "  model       = %s\n", launch.model)
		fmt.Fprintf(stderr, "  harness     = claude (raw, without guard)\n")
		fmt.Fprintf(stderr, "  backend     = fak serve on Mac (/v1/messages)\n")
		fmt.Fprintln(stderr, "  environment =")
		for _, k := range []string{
			"ANTHROPIC_BASE_URL",
			"ANTHROPIC_API_KEY",
			"ANTHROPIC_MODEL",
			"ANTHROPIC_DEFAULT_OPUS_MODEL",
			"ANTHROPIC_DEFAULT_SONNET_MODEL",
			"ANTHROPIC_DEFAULT_HAIKU_MODEL",
			"ANTHROPIC_SMALL_FAST_MODEL",
			"CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC",
			"API_TIMEOUT_MS",
		} {
			if v, ok := envMap[k]; ok {
				fmt.Fprintf(stderr, "    %s=%s\n", k, v)
			}
		}
		if launch.claudeConfigDir != "" {
			fmt.Fprintf(stderr, "    CLAUDE_CONFIG_DIR=%s\n", launch.claudeConfigDir)
		}
		fmt.Fprintf(stderr, "  command     = %s\n", strings.Join(argvOut, " "))
		fmt.Fprintln(stdout, strings.Join(argvOut, " "))
		return 0
	}

	// In live mode, verify that the backend is reachable
	if !launch.noProbe && probeErr != nil {
		fmt.Fprintf(stderr, "fak claude: fak serve backend is unreachable at %s: %v\n", launch.serverURL, probeErr)
		if runtime.GOOS == "darwin" && (strings.Contains(launch.serverURL, "127.0.0.1") || strings.Contains(launch.serverURL, "localhost")) {
			fmt.Fprintln(stderr, "")
			fmt.Fprintln(stderr, "To start the local Mac Metal GPU backend:")
			fmt.Fprintln(stderr, "    fak serve --gguf qwen38:27b-q4")
			fmt.Fprintln(stderr, "")
			fmt.Fprintln(stderr, "Or point to an existing remote gateway:")
			fmt.Fprintln(stderr, "    fak claude --gateway-url http://<host>:8080")
		}
		return 1
	}

	if !launch.quiet {
		srvLabel := "fak serve"
		if statusInfo != nil && statusInfo.Backend != "" {
			srvLabel = fmt.Sprintf("fak serve (%s)", statusInfo.Backend)
		}
		fmt.Fprintf(stderr, "fak claude: connected to %s at %s (model: %s)\n", srvLabel, launch.serverURL, launch.model)
		fmt.Fprintln(stderr, "fak claude: launching Claude Code directly (raw harness, without guard) ...")
	}

	started := time.Now()
	code := claudeLaunchRun(stdout, stderr, argvOut, envList)
	if code == 0 && !launch.quiet {
		fmt.Fprintf(stderr, "fak claude: Claude Code session ended successfully (%s)\n", time.Since(started).Round(time.Millisecond))
	}
	return code
}

func fetchClaudeServerStatus(serverURL string, timeout time.Duration) (*claudeStatusReport, error) {
	healthURL := strings.TrimRight(serverURL, "/") + "/healthz"
	client := &http.Client{Timeout: timeout}
	resp, err := client.Get(healthURL)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("HTTP status %d", resp.StatusCode)
	}

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}

	var info claudeStatusReport
	if err := json.Unmarshal(body, &info); err != nil {
		return &claudeStatusReport{OK: true}, nil
	}
	return &info, nil
}

func buildClaudeLaunchArgv(o claudeLaunchOptions) []string {
	command := o.command
	if command == "" {
		command = "claude"
	}
	argv := []string{command}

	if o.probePrompt != "" {
		argv = append(argv, "-p", o.probePrompt, "--output-format", "json")
		if !hasArg(o.passthrough, "--safe-mode") {
			argv = append(argv, "--safe-mode")
		}
		if !hasArg(o.passthrough, "--no-session-persistence") {
			argv = append(argv, "--no-session-persistence")
		}
	}

	if o.skipPermissions && !hasArg(o.passthrough, "--dangerously-skip-permissions") && !hasArg(o.passthrough, "--permission-mode") {
		argv = append(argv, "--dangerously-skip-permissions")
	}

	argv = append(argv, o.passthrough...)
	return argv
}

func buildClaudeLaunchEnvMap(o claudeLaunchOptions) map[string]string {
	envMap := projectassets.ClaudeEnvMap(o.serverURL, o.model, o.apiKey)
	if o.apiTimeoutMS > 0 {
		envMap["API_TIMEOUT_MS"] = strconv.Itoa(o.apiTimeoutMS)
	}
	if o.claudeConfigDir != "" {
		envMap["CLAUDE_CONFIG_DIR"] = o.claudeConfigDir
	}
	return envMap
}

func mergeEnv(existing []string, overrides map[string]string) []string {
	seen := make(map[string]bool, len(overrides))
	var result []string

	for _, entry := range existing {
		parts := strings.SplitN(entry, "=", 2)
		key := parts[0]
		if val, override := overrides[key]; override {
			result = append(result, fmt.Sprintf("%s=%s", key, val))
			seen[key] = true
		} else {
			result = append(result, entry)
		}
	}

	for k, v := range overrides {
		if !seen[k] {
			result = append(result, fmt.Sprintf("%s=%s", k, v))
		}
	}
	return result
}

func printClaudeShellEnv(w io.Writer, o claudeLaunchOptions) {
	envMap := buildClaudeLaunchEnvMap(o)
	fmt.Fprintln(w, "# Environment configuration for Claude Code with fak serve backend on Mac")
	for _, k := range []string{
		"ANTHROPIC_BASE_URL",
		"ANTHROPIC_API_KEY",
		"ANTHROPIC_MODEL",
		"ANTHROPIC_DEFAULT_OPUS_MODEL",
		"ANTHROPIC_DEFAULT_SONNET_MODEL",
		"ANTHROPIC_DEFAULT_HAIKU_MODEL",
		"ANTHROPIC_SMALL_FAST_MODEL",
		"CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC",
		"API_TIMEOUT_MS",
	} {
		if v, ok := envMap[k]; ok {
			fmt.Fprintf(w, "export %s=%q\n", k, v)
		}
	}
	if o.claudeConfigDir != "" {
		fmt.Fprintf(w, "export CLAUDE_CONFIG_DIR=%q\n", o.claudeConfigDir)
	}
}

func execClaudeLaunchChild(stdout, stderr io.Writer, argv, env []string) int {
	ctx, stop := signal.NotifyContext(context.Background(), terminatingSignals()...)
	defer stop()
	if len(argv) == 0 {
		fmt.Fprintln(stderr, "fak claude: empty command")
		return 2
	}
	cmd := exec.CommandContext(ctx, argv[0], argv[1:]...)
	cmd.Env = env
	cmd.Stdin, cmd.Stdout, cmd.Stderr = os.Stdin, stdout, stderr
	cmd.WaitDelay = 5 * time.Second
	cmd.Cancel = func() error {
		if cmd.Process != nil && cmd.Process.Pid > 0 {
			procguard.KillPID(cmd.Process.Pid)
		}
		return nil
	}
	if err := cmd.Run(); err != nil {
		code := childprocess.ExitCode(err, 1)
		if code == 1 {
			fmt.Fprintf(stderr, "fak claude: %v\n", err)
		}
		return code
	}
	return 0
}

func runClaudeConfig(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("claude config", flag.ContinueOnError)
	fs.SetOutput(stderr)
	addr := fs.String("addr", projectassets.DefaultClaudeBaseURL, "fak serve gateway address")
	model := fs.String("model", projectassets.DefaultClaudeModelID, "served model ID")
	key := fs.String("key", projectassets.DefaultClaudeAPIKey, "API key for Claude Code")
	write := fs.Bool("write", false, "write or update .claude/settings.json in the current workspace")
	dir := fs.String("dir", ".", "workspace directory containing .claude/settings.json")

	if !parseFlags(fs, argv) {
		return 2
	}

	baseURL := projectassets.NormalizeClaudeBaseURL(*addr)
	modelID := projectassets.NormalizeClaudeModelID(*model)
	apiKey := projectassets.NormalizeClaudeAPIKey(*key)

	if *write {
		modified, err := projectassets.EnsureClaudeSettingsConfig(*dir, baseURL, modelID, apiKey)
		if err != nil {
			fmt.Fprintf(stderr, "fak claude config: %v\n", err)
			return 1
		}
		targetPath := filepath.Join(*dir, ".claude", "settings.json")
		if modified {
			fmt.Fprintf(stdout, "fak claude config: updated %s with fak serve backend (baseURL: %s, model: %s)\n", targetPath, baseURL, modelID)
		} else {
			fmt.Fprintf(stdout, "fak claude config: %s already configured with fak serve backend\n", targetPath)
		}
		return 0
	}

	out, err := projectassets.GenerateClaudeSettings(baseURL, modelID, apiKey)
	if err != nil {
		fmt.Fprintf(stderr, "fak claude config: %v\n", err)
		return 1
	}
	fmt.Fprintln(stdout, string(out))
	return 0
}
