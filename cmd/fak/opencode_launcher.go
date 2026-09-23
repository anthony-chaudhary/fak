package main

import (
	"context"
	"flag"
	"fmt"
	"io"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"runtime"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/childprocess"
	"github.com/anthony-chaudhary/fak/internal/pathutil"
	"github.com/anthony-chaudhary/fak/internal/procguard"
	"github.com/anthony-chaudhary/fak/internal/projectassets"
)

type opencodeLaunchOptions struct {
	dryRun          bool
	probePrompt     string
	splitMode       string
	splitWhere      string
	splitInterval   time.Duration
	policyPath      string
	apiKeyEnv       string
	baseURL         string
	remoteServe     string
	model           string
	auditPath       string
	noAudit         bool
	quiet           bool
	localAuto       bool
	halo            bool
	metal           bool
	ggufPath        string
	gpuBackend      string
	tokenizerPath   string
	pure            bool
	auto            bool
	skipPermissions bool
	passthrough     []string
}

var opencodeLaunchRun = execOpencodeLaunchChild

func cmdOpencode(argv []string) {
	os.Exit(runOpencode(os.Stdout, os.Stderr, argv))
}

func runOpencode(stdout, stderr io.Writer, argv []string) int {
	if len(argv) > 0 && argv[0] == "config" {
		return runOpencodeConfig(stdout, stderr, argv[1:])
	}

	fs := flag.NewFlagSet("opencode", flag.ContinueOnError)
	fs.SetOutput(stderr)
	verbFlagUsage(fs, "opencode")
	dryRun := fs.Bool("dry-run", false, "print the OpenCode command and exit without launching")
	probePrompt := fs.String("probe", "", "run a single headless probe turn with this prompt and exit")
	skipPermissions := fs.Bool("skip-permissions", true, "pass --auto to the OpenCode child when running unattended")
	pure := fs.Bool("pure", false, "pass --pure to opencode child to prevent reading untracked global state")
	auto := fs.Bool("auto", false, "pass --auto to opencode child for non-interactive execution")
	splitMode := fs.String("split", "auto", "open the 20% fak-info pane when possible: auto|on|off")
	splitWhere := fs.String("split-where", "bottom", "with --split: place the 20% fak-info pane as a bottom strip or right column")
	splitInterval := fs.Duration("split-interval", 2*time.Second, "with --split: fak-info refresh interval")
	policyPath := fs.String("policy", "", "capability-floor manifest to enforce (default: guard's embedded floor)")
	apiKeyEnv := fs.String("api-key-env", "", "env var holding the upstream OpenAI API key (default: OPENAI_API_KEY)")
	baseURL := fs.String("base-url", "", "upstream provider base URL; advanced override passed to fak guard")
	remoteServe := fs.String("remote-serve", "", "send inference to a remote fak serve (HOST or HOST:PORT), while this local guard adjudicates")
	model := fs.String("model", "", "OpenCode model id under the fak provider")
	auditPath := fs.String("audit", "", "write guard's decision journal to this file (or 'off')")
	noAudit := fs.Bool("no-audit", false, "disable guard's decision journal")
	quiet := fs.Bool("quiet", false, "suppress launcher startup banner")
	localAuto := fs.Bool("local", false, "auto-detect a local OpenAI-compatible model server for guard's upstream")
	halo := fs.Bool("halo", false, "target local AMD Strix Halo appliance server (http://127.0.0.1:8080/v1)")
	strix := fs.Bool("strix", false, "alias for --halo")
	metal := fs.Bool("metal", false, "with --gguf: require Apple Silicon Metal GPU acceleration")
	ggufPath := fs.String("gguf", "", "run a local in-kernel GGUF model as guard's upstream")
	gpuBackend := fs.String("backend", "", "with --gguf: compute backend")
	tokenizerPath := fs.String("tokenizer", "", "with --gguf: tokenizer override")

	fs.Usage = func() {
		fmt.Fprintln(stderr, "usage: fak opencode [launcher flags] [-- <opencode args...>]")
		fmt.Fprintln(stderr, "       fak opencode config [--write] [--addr ADDR] [--model MODEL]")
		fmt.Fprintln(stderr, "  e.g. fak opencode")
		fmt.Fprintln(stderr, "       fak opencode --dry-run")
		fmt.Fprintln(stderr, "       fak opencode --probe \"Use bash to print hello\"")
		fmt.Fprintln(stderr, "       fak opencode --model qwen38:27b -- run \"check the repo\"")
		fmt.Fprintln(stderr, "  Guard-only legacy launcher flags are rejected; configure provider routing with fak opencode config.")
		fmt.Fprintln(stderr, "")
		fs.PrintDefaults()
	}
	if !parseFlags(fs, argv) {
		return 2
	}
	if err := validateOpencodeLaunchSplit(*splitMode, *splitWhere); err != nil {
		fmt.Fprintf(stderr, "fak opencode: %v\n", err)
		return 2
	}
	// Guard-specific launcher flags have no meaning when OpenCode runs directly.
	// Refuse them instead of silently accepting settings that would be ignored.
	guardOnly := map[string]bool{
		"split": true, "split-where": true, "split-interval": true,
		"policy": true, "api-key-env": true, "base-url": true, "remote-serve": true,
		"audit": true, "no-audit": true, "local": true, "halo": true, "strix": true,
		"metal": true, "gguf": true, "backend": true, "tokenizer": true,
	}
	var unsupported []string
	fs.Visit(func(f *flag.Flag) {
		if guardOnly[f.Name] {
			unsupported = append(unsupported, "--"+f.Name)
		}
	})
	if len(unsupported) != 0 {
		fmt.Fprintf(stderr, "fak opencode: guard-only flags are unavailable for a direct launch: %s\n", strings.Join(unsupported, ", "))
		return 2
	}

	opencodeBin := resolveOpencodeLaunchBinary()
	launch := opencodeLaunchOptions{
		dryRun:          *dryRun,
		probePrompt:     *probePrompt,
		splitMode:       *splitMode,
		splitWhere:      *splitWhere,
		splitInterval:   *splitInterval,
		policyPath:      *policyPath,
		apiKeyEnv:       *apiKeyEnv,
		baseURL:         *baseURL,
		remoteServe:     *remoteServe,
		model:           *model,
		auditPath:       *auditPath,
		noAudit:         *noAudit,
		quiet:           *quiet,
		localAuto:       *localAuto,
		halo:            *halo || *strix,
		metal:           *metal,
		ggufPath:        *ggufPath,
		gpuBackend:      *gpuBackend,
		tokenizerPath:   *tokenizerPath,
		pure:            *pure,
		auto:            *auto,
		skipPermissions: *skipPermissions,
		passthrough:     fs.Args(),
	}
	argvOut := buildOpencodeLaunchArgv(opencodeBin, launch)

	if launch.dryRun {
		fmt.Fprintln(stderr, "fak opencode: dry-run - not launching")
		fmt.Fprintln(stderr, "  command     = "+strings.Join(argvOut, " "))
		fmt.Fprintln(stdout, strings.Join(argvOut, " "))
		return 0
	}

	if _, err := projectassets.Ensure(".", true); err != nil && !launch.quiet {
		fmt.Fprintf(stderr, "fak opencode: warning: %v\n", err)
	}
	if err := projectassets.VerifyOpenCodeSnapshot("."); err != nil && !launch.quiet {
		fmt.Fprintf(stderr, "fak opencode: warning: %v\n", err)
	}

	started := time.Now()
	if !launch.quiet {
		fmt.Fprintln(stderr, "fak opencode: launching OpenCode directly ...")
	}
	code := opencodeLaunchRun(stdout, stderr, argvOut, os.Environ())
	if code == 0 && !launch.quiet {
		fmt.Fprintf(stderr, "fak opencode: OpenCode completed successfully in %s\n", time.Since(started).Round(time.Millisecond))
	}
	return code
}

func validateOpencodeLaunchSplit(mode, where string) error {
	switch strings.TrimSpace(strings.ToLower(mode)) {
	case "", "auto", "on", "true", "1", "yes", "off", "false", "0", "no":
	default:
		return fmt.Errorf("--split must be auto|on|off, got %q", mode)
	}
	switch strings.TrimSpace(strings.ToLower(where)) {
	case "", "bottom", "right":
		return nil
	default:
		return fmt.Errorf("--split-where must be %q or %q, got %q", "bottom", "right", where)
	}
}

func buildOpencodeLaunchArgv(opencodeBin string, o opencodeLaunchOptions) []string {
	argv := []string{opencodeBin}
	if childModel := strings.TrimPrefix(strings.TrimSpace(o.model), "fak/"); childModel != "" {
		argv = append(argv, "--model", "fak/"+childModel)
	}
	if o.probePrompt != "" {
		argv = append(argv, "run", o.probePrompt, "--format", "json")
		if o.auto || o.skipPermissions {
			argv = append(argv, "--auto")
		}
		if o.pure {
			argv = append(argv, "--pure")
		}
	}
	return append(argv, o.passthrough...)
}

func resolveOpencodeLaunchBinary() string {
	if runtime.GOOS == "windows" {
		if appData := os.Getenv("APPDATA"); appData != "" {
			native := filepath.Join(appData, "npm", "node_modules", "opencode-ai", "bin", "opencode.exe")
			if info, err := os.Stat(native); err == nil && !info.IsDir() {
				return native
			}
		}
	}
	if binary, err := exec.LookPath("opencode"); err == nil {
		return binary
	}
	if runtime.GOOS != "windows" {
		if binary := resolvePOSIXOpenCodeBinary(""); binary != "" {
			return binary
		}
	}
	return "opencode"
}

func execOpencodeLaunchChild(stdout, stderr io.Writer, argv, env []string) int {
	ctx, stop := signal.NotifyContext(context.Background(), terminatingSignals()...)
	defer stop()
	return execOpencodeLaunchChildContext(ctx, stdout, stderr, argv, env)
}

func execOpencodeLaunchChildContext(ctx context.Context, stdout, stderr io.Writer, argv, env []string) int {
	if len(argv) == 0 {
		fmt.Fprintln(stderr, "fak opencode: empty command")
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
			fmt.Fprintf(stderr, "fak opencode: %v\n", err)
		}
		return code
	}
	return 0
}

func opencodeConfigModeLabel(mode projectassets.OpenCodeConfigMode) string {
	if mode == projectassets.OpenCodeModeMac {
		return "mac metal native"
	}
	if mode == projectassets.OpenCodeModeHalo {
		return "halo"
	}
	return "gateway"
}

func runOpencodeConfig(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("opencode config", flag.ContinueOnError)
	fs.SetOutput(stderr)
	addr := fs.String("addr", "127.0.0.1:8080", "fak serve gateway listen address")
	model := fs.String("model", projectassets.DefaultOpenCodeModelID, "served model ID")
	halo := fs.Bool("halo", false, "configure opencode.json for local AMD Strix Halo server")
	strix := fs.Bool("strix", false, "alias for --halo")
	mac := fs.Bool("mac", false, "configure opencode.json for native in-kernel Apple Silicon Metal inference")
	nativeMetal := fs.Bool("native-metal", false, "alias for --mac")
	write := fs.Bool("write", false, "write or update opencode.json in the current workspace")
	dir := fs.String("dir", ".", "workspace directory containing opencode.json")
	fs.Usage = func() {
		fmt.Fprintln(stderr, "usage: fak opencode config [--write] [--addr ADDR] [--model MODEL] [--halo|--strix|--mac|--native-metal]")
		fmt.Fprintln(stderr, "  e.g. fak opencode config --mac --write")
		fmt.Fprintln(stderr, "")
		fs.PrintDefaults()
	}
	if !parseFlags(fs, argv) {
		return 2
	}
	*dir = pathutil.ExpandTilde(*dir)
	macMode := *mac || *nativeMetal
	if macMode {
		if *model == projectassets.DefaultOpenCodeModelID || *model == "fak-local" {
			*model = projectassets.DefaultOpenCodeMacModelID
		}
	} else if (*halo || *strix) && *model == projectassets.DefaultOpenCodeModelID {
		*model = projectassets.ResolveDynamicHaloModel(*dir)
	}
	baseURL := *addr
	if !strings.HasPrefix(baseURL, "http://") && !strings.HasPrefix(baseURL, "https://") {
		baseURL = "http://" + baseURL
	}
	if !strings.HasSuffix(baseURL, "/v1") {
		baseURL = strings.TrimSuffix(baseURL, "/") + "/v1"
	}
	mode := projectassets.OpenCodeModeGateway
	switch {
	case macMode:
		mode = projectassets.OpenCodeModeMac
	case *halo || *strix:
		mode = projectassets.OpenCodeModeHalo
	}

	if *write {
		modified, err := projectassets.EnsureOpenCodeProviderConfigMode(*dir, baseURL, *model, mode)
		if err != nil {
			fmt.Fprintf(stderr, "fak opencode config: %v\n", err)
			return 1
		}
		label := opencodeConfigModeLabel(mode)
		if modified {
			fmt.Fprintf(stdout, "fak opencode config: updated %s with provider \"fak\" (%s, baseURL: %s, model: %s)\n", filepath.Join(*dir, "opencode.json"), label, baseURL, *model)
		} else {
			fmt.Fprintf(stdout, "fak opencode config: %s already has up-to-date provider \"fak\" (%s)\n", filepath.Join(*dir, "opencode.json"), label)
		}
		return 0
	}
	out, err := projectassets.GenerateOpenCodeConfigMode(baseURL, *model, mode)
	if err != nil {
		fmt.Fprintf(stderr, "fak opencode config: %v\n", err)
		return 1
	}
	fmt.Fprintln(stdout, string(out))
	return 0
}
