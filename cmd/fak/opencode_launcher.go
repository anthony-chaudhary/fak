package main

import (
	"flag"
	"fmt"
	"io"
	"os"
	"os/exec"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/childprocess"
	"github.com/anthony-chaudhary/fak/internal/pathutil"
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
	fs := flag.NewFlagSet("opencode", flag.ContinueOnError)
	fs.SetOutput(stderr)
	dryRun := fs.Bool("dry-run", false, "print the guarded OpenCode command and exit without launching")
	probePrompt := fs.String("probe", "", "run a single headless probe turn with this prompt and exit")
	skipPermissions := fs.Bool("skip-permissions", true, "pass --dangerously-skip-permissions to opencode child when running unattended")
	pure := fs.Bool("pure", false, "pass --pure to opencode child to prevent reading untracked global state")
	auto := fs.Bool("auto", false, "pass --auto to opencode child for non-interactive execution")
	splitMode := fs.String("split", "auto", "open the 20% fak-info pane when possible: auto|on|off")
	splitWhere := fs.String("split-where", "bottom", "with --split: place the 20% fak-info pane as a bottom strip or right column")
	splitInterval := fs.Duration("split-interval", 2*time.Second, "with --split: fak-info refresh interval")
	policyPath := fs.String("policy", "", "capability-floor manifest to enforce (default: guard's embedded floor)")
	apiKeyEnv := fs.String("api-key-env", "", "env var holding the upstream OpenAI API key (default: OPENAI_API_KEY)")
	baseURL := fs.String("base-url", "", "upstream provider base URL; advanced override passed to fak guard")
	remoteServe := fs.String("remote-serve", "", "send inference to a remote fak serve (HOST or HOST:PORT), while this local guard adjudicates")
	model := fs.String("model", "", "upstream model id override passed to fak guard")
	auditPath := fs.String("audit", "", "write guard's decision journal to this file (or 'off')")
	noAudit := fs.Bool("no-audit", false, "disable guard's decision journal")
	quiet := fs.Bool("quiet", false, "suppress guard's startup banner and exit summary")
	localAuto := fs.Bool("local", false, "auto-detect a local OpenAI-compatible model server for guard's upstream")
	ggufPath := fs.String("gguf", "", "run a local in-kernel GGUF model as guard's upstream")
	gpuBackend := fs.String("backend", "", "with --gguf: compute backend")
	tokenizerPath := fs.String("tokenizer", "", "with --gguf: tokenizer override")

	fs.Usage = func() {
		fmt.Fprintln(stderr, "usage: fak opencode [launcher flags] [-- <opencode args...>]")
		fmt.Fprintln(stderr, "  e.g. fak opencode")
		fmt.Fprintln(stderr, "       fak opencode --dry-run")
		fmt.Fprintln(stderr, "       fak opencode --probe \"Use bash to print hello\"")
		fmt.Fprintln(stderr, "       fak opencode --policy my-floor.json -- run \"check the repo\"")
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
	*ggufPath = pathutil.ExpandTilde(*ggufPath)
	*tokenizerPath = pathutil.ExpandTilde(*tokenizerPath)

	fakBin := tuiExecutable()
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
		ggufPath:        *ggufPath,
		gpuBackend:      *gpuBackend,
		tokenizerPath:   *tokenizerPath,
		pure:            *pure,
		auto:            *auto,
		skipPermissions: *skipPermissions,
		passthrough:     fs.Args(),
	}
	argvOut := buildOpencodeLaunchArgv(fakBin, launch)

	if launch.dryRun {
		fmt.Fprintln(stderr, "fak opencode: dry-run - not launching")
		fmt.Fprintln(stderr, "  view        = agent 80% / fak info 20% (--split "+launch.splitMode+")")
		fmt.Fprintln(stderr, "  provider    = openai (Chat Completions /v1/chat/completions)")
		fmt.Fprintln(stderr, "  command     = "+strings.Join(argvOut, " "))
		fmt.Fprintln(stdout, strings.Join(argvOut, " "))
		return 0
	}

	started := time.Now()
	fmt.Fprintln(stderr, "fak opencode: launching OpenCode through fak guard ...")
	code := opencodeLaunchRun(stdout, stderr, argvOut, os.Environ())
	if code == 0 {
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

func buildOpencodeLaunchArgv(fakBin string, o opencodeLaunchOptions) []string {
	argv := []string{
		fakBin,
		"guard",
		"--provider", "openai",
		"--split", firstNonEmpty(strings.TrimSpace(o.splitMode), "auto"),
		"--split-where", firstNonEmpty(strings.TrimSpace(o.splitWhere), "bottom"),
		"--split-interval", o.splitInterval.String(),
	}
	appendKV := func(flag, value string) {
		if strings.TrimSpace(value) != "" {
			argv = append(argv, flag, value)
		}
	}
	appendKV("--policy", o.policyPath)
	appendKV("--api-key-env", o.apiKeyEnv)
	appendKV("--base-url", o.baseURL)
	appendKV("--remote-serve", o.remoteServe)
	appendKV("--model", o.model)
	appendKV("--audit", o.auditPath)
	if o.noAudit {
		argv = append(argv, "--audit", "off")
	}
	if o.quiet {
		argv = append(argv, "--quiet")
	}
	if o.localAuto {
		argv = append(argv, "--local")
	}
	if o.probePrompt != "" {
		argv = append(argv, "--probe")
	}
	appendKV("--gguf", o.ggufPath)
	appendKV("--backend", o.gpuBackend)
	appendKV("--tokenizer", o.tokenizerPath)

	argv = append(argv, "--", "opencode")
	if o.probePrompt != "" {
		argv = append(argv, "run", o.probePrompt, "--format", "json")
		if o.auto || o.probePrompt != "" {
			argv = append(argv, "--auto")
		}
		if o.pure {
			argv = append(argv, "--pure")
		}
		if o.skipPermissions {
			argv = append(argv, "--dangerously-skip-permissions")
		}
	}
	return append(argv, o.passthrough...)
}

func execOpencodeLaunchChild(stdout, stderr io.Writer, argv, env []string) int {
	if len(argv) == 0 {
		fmt.Fprintln(stderr, "fak opencode: empty command")
		return 2
	}
	cmd := exec.Command(argv[0], argv[1:]...)
	cmd.Env = env
	cmd.Stdin, cmd.Stdout, cmd.Stderr = os.Stdin, stdout, stderr
	if err := cmd.Run(); err != nil {
		code := childprocess.ExitCode(err, 1)
		if code == 1 {
			fmt.Fprintf(stderr, "fak opencode: %v\n", err)
		}
		return code
	}
	return 0
}
