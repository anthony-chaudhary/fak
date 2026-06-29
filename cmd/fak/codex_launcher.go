package main

import (
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"os/exec"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/pathutil"
)

// fak codex is the short, operator-facing Codex launcher. It intentionally does not
// reimplement guard: it builds the same `fak guard -- codex` argv the long form uses, then
// lets guard own the in-process gateway, Codex provider injection, audit journal, and 80/20
// fak-info split pane. The only Codex-specific default here is permission authority: Codex is
// launched with its own bypass flag so fak's capability floor, not Codex's prompt/sandbox
// layer, is the permission system for this dogfood path.

type codexLaunchOptions struct {
	dryRun          bool
	skipPermissions bool
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
	codexConfig     bool
	passthrough     []string
}

var codexLaunchRun = execCodexLaunchChild

func cmdCodex(argv []string) {
	os.Exit(runCodex(os.Stdout, os.Stderr, argv))
}

func runCodex(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("codex", flag.ContinueOnError)
	fs.SetOutput(stderr)
	dryRun := fs.Bool("dry-run", false, "print the guarded Codex command and exit without launching")
	skipPermissions := fs.Bool("skip-permissions", true, "pass Codex's --dangerously-bypass-approvals-and-sandbox so fak's capability floor is the permission system")
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
	codexConfig := fs.Bool("codex-config", true, "let guard inject per-run Codex -c provider overrides (default true)")
	fs.Usage = func() {
		fmt.Fprintln(stderr, "usage: fak codex [launcher flags] [-- <codex args...>]")
		fmt.Fprintln(stderr, "  e.g. fak codex")
		fmt.Fprintln(stderr, "       fak codex -- exec \"summarize AGENTS.md\"")
		fmt.Fprintln(stderr, "       fak codex --policy my-floor.json -- exec --json \"check the repo\"")
		fmt.Fprintln(stderr, "")
		fs.PrintDefaults()
	}
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	if err := validateCodexLaunchSplit(*splitMode, *splitWhere); err != nil {
		fmt.Fprintf(stderr, "fak codex: %v\n", err)
		return 2
	}
	*ggufPath = pathutil.ExpandTilde(*ggufPath)
	*tokenizerPath = pathutil.ExpandTilde(*tokenizerPath)

	fakBin := tuiExecutable()
	launch := codexLaunchOptions{
		dryRun:          *dryRun,
		skipPermissions: *skipPermissions,
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
		codexConfig:     *codexConfig,
		passthrough:     fs.Args(),
	}
	argvOut := buildCodexLaunchArgv(fakBin, launch)

	fmt.Fprintln(stderr, "fak codex: launching Codex through fak guard")
	fmt.Fprintln(stderr, "  view        = agent 80% / fak info 20% (--split "+launch.splitMode+")")
	if launch.skipPermissions {
		fmt.Fprintln(stderr, "  permissions = fak floor is the permission system (Codex bypass flag passed)")
	} else {
		fmt.Fprintln(stderr, "  permissions = Codex keeps its own approval/sandbox layer (--skip-permissions=false)")
	}
	fmt.Fprintln(stderr, "  command     = "+strings.Join(argvOut, " "))
	if launch.dryRun {
		fmt.Fprintln(stderr, "  (dry-run - not launching)")
		fmt.Fprintln(stdout, strings.Join(argvOut, " "))
		return 0
	}
	return codexLaunchRun(stdout, stderr, argvOut, os.Environ())
}

func validateCodexLaunchSplit(mode, where string) error {
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

func buildCodexLaunchArgv(fakBin string, o codexLaunchOptions) []string {
	argv := []string{
		fakBin,
		"guard",
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
		argv = append(argv, "--no-audit")
	}
	if o.quiet {
		argv = append(argv, "--quiet")
	}
	if o.localAuto {
		argv = append(argv, "--local")
	}
	appendKV("--gguf", o.ggufPath)
	appendKV("--backend", o.gpuBackend)
	appendKV("--tokenizer", o.tokenizerPath)
	if !o.codexConfig {
		argv = append(argv, "--codex-config=false")
	}

	argv = append(argv, "--", "codex")
	if o.skipPermissions {
		if flag := launchSkipPermsFlag("codex"); flag != "" {
			argv = append(argv, flag)
		}
	}
	return append(argv, o.passthrough...)
}

func execCodexLaunchChild(stdout, stderr io.Writer, argv, env []string) int {
	if len(argv) == 0 {
		fmt.Fprintln(stderr, "fak codex: empty command")
		return 2
	}
	cmd := exec.Command(argv[0], argv[1:]...)
	cmd.Env = env
	cmd.Stdin, cmd.Stdout, cmd.Stderr = os.Stdin, stdout, stderr
	if err := cmd.Run(); err != nil {
		var ee *exec.ExitError
		if errors.As(err, &ee) {
			return ee.ExitCode()
		}
		fmt.Fprintf(stderr, "fak codex: %v\n", err)
		return 1
	}
	return 0
}
