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
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/childprocess"
	"github.com/anthony-chaudhary/fak/internal/procguard"
	"github.com/anthony-chaudhary/fak/internal/projectassets"
)

type piLaunchOptions struct {
	dryRun       bool
	probePrompt  string
	addr         string
	baseURL      string
	model        string
	configPath   string
	writeConfig  bool
	checkBackend bool
	thinking     string
	tools        string
	quiet        bool
	command      string
	provider     string
	passthrough  []string
}

var piLaunchRun = execPiLaunchChild

func cmdPi(argv []string) {
	os.Exit(runPi(os.Stdout, os.Stderr, argv))
}

func runPi(stdout, stderr io.Writer, argv []string) int {
	if len(argv) > 0 && argv[0] == "config" {
		return runPiConfig(stdout, stderr, argv[1:])
	}

	fs := flag.NewFlagSet("pi", flag.ContinueOnError)
	fs.SetOutput(stderr)
	verbFlagUsage(fs, "pi")

	dryRun := fs.Bool("dry-run", false, "print the Pi launch command and exit without executing")
	printEnv := fs.Bool("print-env", false, "print shell export statements for Pi environment and exit")
	probePrompt := fs.String("probe", "", "run a single headless probe turn with this prompt and exit")
	promptFlag := fs.String("prompt", "", "alias for probe prompt (or pass -p)")
	fs.StringVar(promptFlag, "p", "", "alias for probe prompt")
	addr := fs.String("addr", "127.0.0.1:8080", "fak serve gateway listen address (default: 127.0.0.1:8080 or FAK_SERVE_ADDR)")
	baseURL := fs.String("base-url", "", "fak serve provider base URL (default: http://<addr>/v1)")
	model := fs.String("model", "", "model ID (default: auto-detect from fak serve /healthz or qwen38:27b-q4)")
	configPath := fs.String("config-path", "", "custom destination path for Pi models.json (default: ~/.pi/agent/models.json)")
	writeConfig := fs.Bool("write-config", true, "ensure ~/.pi/agent/models.json is configured with provider 'fak' before launching")
	checkBackend := fs.Bool("check-backend", true, "verify fak serve backend is reachable before starting Pi")
	thinking := fs.String("thinking", "", "thinking/reasoning level passed to pi (--thinking <level>)")
	tools := fs.String("tools", "", "tool allowlist passed to pi (--tools <list>)")
	quiet := fs.Bool("quiet", false, "suppress launcher banner and diagnostics")
	command := fs.String("command", "pi", "executable name or path for Pi CLI")

	fs.Usage = func() {
		fmt.Fprintln(stderr, "usage: fak pi [launcher flags] [-- <pi args...>]")
		fmt.Fprintln(stderr, "       fak pi config [--write] [--addr ADDR] [--model MODEL] [--path PATH]")
		fmt.Fprintln(stderr, "")
		fmt.Fprintln(stderr, "First-class support for Pi coding agent as harness with fak serve on Mac as backend.")
		fmt.Fprintln(stderr, "Runs Pi directly targeting fak serve backend (for now raw, without guard).")
		fmt.Fprintln(stderr, "")
		fmt.Fprintln(stderr, "examples:")
		fmt.Fprintln(stderr, "  fak pi                                        # interactive session on local Mac fak serve")
		fmt.Fprintln(stderr, "  fak pi --dry-run                              # preview configuration and command")
		fmt.Fprintln(stderr, "  fak pi --print-env                            # print shell export lines")
		fmt.Fprintln(stderr, "  fak pi --probe \"Explain unified memory\"       # headless probe turn")
		fmt.Fprintln(stderr, "  fak pi config --write                         # write ~/.pi/agent/models.json")
		fmt.Fprintln(stderr, "")
		fs.PrintDefaults()
	}

	if !parseFlags(fs, argv) {
		return 2
	}

	effectivePrompt := *probePrompt
	if effectivePrompt == "" && *promptFlag != "" {
		effectivePrompt = *promptFlag
	}

	targetAddr := *addr
	if envAddr := os.Getenv("FAK_SERVE_ADDR"); envAddr != "" && targetAddr == "127.0.0.1:8080" {
		targetAddr = envAddr
	}

	targetBaseURL := *baseURL
	if targetBaseURL == "" {
		targetBaseURL = projectassets.NormalizePiBaseURL(targetAddr)
	} else {
		targetBaseURL = projectassets.NormalizePiBaseURL(targetBaseURL)
	}

	targetModel := strings.TrimSpace(*model)
	backendActive := false

	// Probe backend to verify reachability and auto-detect served model if unspecified
	detectedModel, reachable := probePiBackend(targetBaseURL, 1500*time.Millisecond)
	if reachable {
		backendActive = true
		if targetModel == "" && detectedModel != "" && detectedModel != "mock" {
			targetModel = detectedModel
		}
	}

	if targetModel == "" {
		targetModel = projectassets.DefaultPiModelID
	}

	if *printEnv {
		fmt.Fprintln(stdout, "# Environment configuration for Pi with fak serve backend on Mac")
		fmt.Fprintf(stdout, "export PI_PROVIDER=\"%s\"\n", projectassets.DefaultPiProviderID)
		fmt.Fprintf(stdout, "export PI_MODEL=\"%s\"\n", targetModel)
		fmt.Fprintf(stdout, "export OPENAI_BASE_URL=\"%s\"\n", targetBaseURL)
		return 0
	}

	if *checkBackend && !backendActive && !*dryRun {
		fmt.Fprintf(stderr, "fak pi: fak serve backend at %s is not responding.\n", targetBaseURL)
		fmt.Fprintln(stderr, "  To start the backend on macOS with Apple Silicon Metal acceleration:")
		fmt.Fprintf(stderr, "    fak serve --gguf %s --pi\n", targetModel)
		fmt.Fprintln(stderr, "  Or pass --check-backend=false to bypass backend liveness preflight.")
		return 1
	}

	launch := piLaunchOptions{
		dryRun:       *dryRun,
		probePrompt:  effectivePrompt,
		addr:         targetAddr,
		baseURL:      targetBaseURL,
		model:        targetModel,
		configPath:   *configPath,
		writeConfig:  *writeConfig,
		checkBackend: *checkBackend,
		thinking:     *thinking,
		tools:        *tools,
		quiet:        *quiet,
		command:      *command,
		provider:     projectassets.DefaultPiProviderID,
		passthrough:  fs.Args(),
	}

	if launch.writeConfig {
		resolvedPath, modified, err := projectassets.EnsurePiProviderConfig(launch.configPath, launch.baseURL, launch.model)
		if err != nil && !launch.quiet {
			fmt.Fprintf(stderr, "fak pi: warning: could not update Pi config %s: %v\n", resolvedPath, err)
		} else if modified && !launch.quiet {
			fmt.Fprintf(stderr, "fak pi: updated %s with provider \"fak\" (baseURL: %s, model: %s)\n", resolvedPath, launch.baseURL, launch.model)
		}
	}

	argvOut := buildPiLaunchArgv(launch)

	if launch.dryRun {
		fmt.Fprintln(stderr, "fak pi: dry-run - not launching")
		fmt.Fprintf(stderr, "  backend     = %s (raw without guard)\n", launch.baseURL)
		fmt.Fprintf(stderr, "  provider    = %s\n", launch.provider)
		fmt.Fprintf(stderr, "  model       = %s\n", launch.model)
		fmt.Fprintln(stderr, "  command     = "+strings.Join(argvOut, " "))
		fmt.Fprintln(stdout, strings.Join(argvOut, " "))
		return 0
	}

	env := os.Environ()
	if launch.configPath != "" {
		dir := launch.configPath
		if strings.HasSuffix(strings.ToLower(dir), ".json") {
			dir = filepath.Dir(dir)
		}
		env = append(env, "PI_CODING_AGENT_DIR="+dir)
	}

	if !launch.quiet {
		fmt.Fprintf(stderr, "fak pi: launching Pi (raw without guard) -> backend %s (model: %s)\n", launch.baseURL, launch.model)
	}

	return piLaunchRun(stdout, stderr, argvOut, env)
}

func probePiBackend(baseURL string, timeout time.Duration) (string, bool) {
	client := &http.Client{Timeout: timeout}
	healthURL := strings.TrimRight(strings.TrimSuffix(baseURL, "/v1"), "/") + "/healthz"
	resp, err := client.Get(healthURL)
	if err == nil && resp.StatusCode == http.StatusOK {
		defer resp.Body.Close()
		var h struct {
			OK    bool   `json:"ok"`
			Model string `json:"model"`
		}
		if json.NewDecoder(resp.Body).Decode(&h) == nil && (h.OK || h.Model != "") {
			return h.Model, true
		}
		return "", true
	}

	// Fallback to /v1/models
	modelsURL := strings.TrimRight(baseURL, "/") + "/models"
	mResp, mErr := client.Get(modelsURL)
	if mErr == nil && mResp.StatusCode == http.StatusOK {
		defer mResp.Body.Close()
		var catalog struct {
			Data []struct {
				ID string `json:"id"`
			} `json:"data"`
		}
		if json.NewDecoder(mResp.Body).Decode(&catalog) == nil && len(catalog.Data) > 0 {
			return catalog.Data[0].ID, true
		}
		return "", true
	}

	return "", false
}

func buildPiLaunchArgv(opts piLaunchOptions) []string {
	cmdName := opts.command
	if cmdName == "" {
		if envBin := os.Getenv("PI_BIN"); envBin != "" {
			cmdName = envBin
		} else {
			cmdName = "pi"
		}
	}
	argv := []string{cmdName}
	if opts.provider != "" {
		argv = append(argv, "--provider", opts.provider)
	}
	if opts.model != "" {
		argv = append(argv, "--model", opts.model)
	}
	if opts.probePrompt != "" {
		argv = append(argv, "-p", opts.probePrompt)
	}
	if opts.thinking != "" {
		argv = append(argv, "--thinking", opts.thinking)
	}
	if opts.tools != "" {
		argv = append(argv, "--tools", opts.tools)
	}
	argv = append(argv, opts.passthrough...)
	return argv
}

func execPiLaunchChild(stdout, stderr io.Writer, argv, env []string) int {
	if len(argv) == 0 {
		fmt.Fprintln(stderr, "fak pi: empty command")
		return 2
	}
	ctx, stop := signal.NotifyContext(context.Background(), terminatingSignals()...)
	defer stop()
	cmd := exec.CommandContext(ctx, argv[0], argv[1:]...)
	cmd.Env = env
	cmd.Stdin, cmd.Stdout, cmd.Stderr = os.Stdin, stdout, stderr
	cmd.Cancel = func() error {
		if cmd.Process != nil && cmd.Process.Pid > 0 {
			procguard.KillPID(cmd.Process.Pid)
		}
		return nil
	}
	if err := cmd.Run(); err != nil {
		code := childprocess.ExitCode(err, 1)
		if code == 1 {
			fmt.Fprintf(stderr, "fak pi: %v\n", err)
		}
		return code
	}
	return 0
}

func runPiConfig(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("pi config", flag.ContinueOnError)
	fs.SetOutput(stderr)
	addr := fs.String("addr", "127.0.0.1:8080", "fak serve gateway listen address")
	model := fs.String("model", projectassets.DefaultPiModelID, "served model ID")
	write := fs.Bool("write", false, "write or update ~/.pi/agent/models.json (or --path)")
	path := fs.String("path", "", "destination path or directory for models.json")
	if !parseFlags(fs, argv) {
		return 2
	}
	baseURL := projectassets.NormalizePiBaseURL(*addr)
	if *write {
		resolvedPath, modified, err := projectassets.EnsurePiProviderConfig(*path, baseURL, *model)
		if err != nil {
			fmt.Fprintf(stderr, "fak pi config: %v\n", err)
			return 1
		}
		if modified {
			fmt.Fprintf(stdout, "fak pi config: updated %s with provider \"fak\" (baseURL: %s, model: %s)\n", resolvedPath, baseURL, *model)
		} else {
			fmt.Fprintf(stdout, "fak pi config: %s already has up-to-date provider \"fak\"\n", resolvedPath)
		}
		return 0
	}
	out, err := projectassets.GeneratePiConfig(baseURL, *model)
	if err != nil {
		fmt.Fprintf(stderr, "fak pi config: %v\n", err)
		return 1
	}
	fmt.Fprintln(stdout, string(out))
	return 0
}
