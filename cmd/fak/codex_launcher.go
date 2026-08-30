package main

import (
	"flag"
	"fmt"
	"github.com/anthony-chaudhary/fak/internal/childprocess"
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
// fak-info split pane. Managed launches default to Codex's non-interactive approval/sandbox
// bypass while fak's independent routing, capacity, policy, hook, and loop gates remain active.
// Operators can explicitly restore Codex's native approval and sandbox layer.

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
	managedCache    string // managed-cache posture: auto|on|off (auto/"" => omit; carried for a future cache-capable wire)
	auditPath       string
	noAudit         bool
	quiet           bool
	localAuto       bool
	ggufPath        string
	gpuBackend      string
	tokenizerPath   string
	codexConfig     bool
	codexHome       string
	loopGateChecked bool // outer fak codex already ran the pre-spawn gate; do not repeat it inside guard
	passthrough     []string
}

type codexLoopGateConfig struct {
	Threshold     string
	CodexHome     string
	SinceHours    float64
	Limit         int
	WorkingDir    string
	Quiet         bool
	BypassCommand string
}

var codexLaunchRun = execCodexLaunchChild

func cmdCodex(argv []string) {
	// MCP lifecycle operations are local installer/status work, not Codex launches.
	// Route them before the seat/loop gate so `fak codex mcp ...` never consumes a
	// provider turn and remains usable while the launcher is otherwise held.
	if len(argv) > 0 && argv[0] == "mcp" {
		cmdCodexMCP(argv[1:])
		return
	}
	args, code, stop := runCodexFreshnessAdmission(argv)
	if stop {
		os.Exit(code)
	}
	os.Exit(runCodex(os.Stdout, os.Stderr, args))
}

func runCodex(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("codex", flag.ContinueOnError)
	fs.SetOutput(stderr)
	dryRun := fs.Bool("dry-run", false, "print the guarded Codex command and exit without launching")
	_ = fs.String("freshness-gate", "on", "require a current checkout launcher before admission (on|off; off is an explicit recovery override)")
	skipPermissions := fs.Bool("skip-permissions", true, "legacy explicit opt-in for Codex's full approval/sandbox bypass (default true for managed launches); fak routing, capacity, policy, hook, and loop gates still apply")
	nativePermissions := fs.Bool("native-permissions", false, "restore Codex's native approval prompts and sandbox; Codex subagents inherit this parent permission mode")
	splitMode := fs.String("split", "auto", "open the 20% fak-info pane when possible: auto|on|off")
	splitWhere := fs.String("split-where", "bottom", "with --split: place the 20% fak-info pane as a bottom strip or right column")
	splitInterval := fs.Duration("split-interval", 2*time.Second, "with --split: fak-info refresh interval")
	policyPath := fs.String("policy", "", "capability-floor manifest to enforce (default: guard's embedded floor)")
	apiKeyEnv := fs.String("api-key-env", "", "env var holding the upstream OpenAI API key (default: OPENAI_API_KEY)")
	baseURL := fs.String("base-url", "", "upstream provider base URL; advanced override passed to fak guard")
	remoteServe := fs.String("remote-serve", "", "send inference to a remote fak serve (HOST or HOST:PORT), while this local guard adjudicates")
	model := fs.String("model", "", "upstream model id override passed to fak guard")
	managedCache := fs.String("managed-cache", os.Getenv(fleetManagedCacheEnv), "managed-cache posture forwarded to fak guard: auto|on|off (default: $"+fleetManagedCacheEnv+", else auto). The openai wire has no cache_control today, so this is passive there; the flag is carried so a cache-capable wire lands managed")
	auditPath := fs.String("audit", "", "write guard's decision journal to this file (or 'off')")
	noAudit := fs.Bool("no-audit", false, "disable guard's decision journal")
	quiet := fs.Bool("quiet", false, "suppress guard's startup banner and exit summary")
	localAuto := fs.Bool("local", false, "auto-detect a local OpenAI-compatible model server for guard's upstream")
	ggufPath := fs.String("gguf", "", "run a local in-kernel GGUF model as guard's upstream")
	gpuBackend := fs.String("backend", "", "with --gguf: compute backend")
	tokenizerPath := fs.String("tokenizer", "", "with --gguf: tokenizer override")
	codexConfig := fs.Bool("codex-config", true, "let guard inject per-run Codex -c provider overrides (default true)")
	codexHome := fs.String("codex-home", "", "Codex home for auth and loop-gate transcript audit (default: $CODEX_HOME or ~/.codex)")
	loopGate := fs.String("loop-gate", dispatchCodexLoopGateDefaultThreshold(), "opt-in pre-launch audit of recent Codex sessions; refuse at threshold loop|action, or use off (default: $FLEET_CODEX_LOOP_GATE, else off)")
	loopGateSinceHours := fs.Float64("loop-gate-since-hours", 24, "with --loop-gate, only scan Codex sessions modified within N hours (0 = all)")
	loopGateLimit := fs.Int("loop-gate-limit", 20, "legacy compatibility value; the launch gate evaluates only the newest Codex session")
	fs.Usage = func() {
		fmt.Fprintln(stderr, "usage: fak codex [launcher flags] [-- <codex args...>]")
		fmt.Fprintln(stderr, "  e.g. fak codex")
		fmt.Fprintln(stderr, "       fak codex -- exec \"summarize AGENTS.md\"")
		fmt.Fprintln(stderr, "       fak codex --policy my-floor.json -- exec --json \"check the repo\"")
		fmt.Fprintln(stderr, "")
		fs.PrintDefaults()
	}
	if !parseFlags(fs, argv) {
		return 2
	}
	if err := validateCodexLaunchSplit(*splitMode, *splitWhere); err != nil {
		fmt.Fprintf(stderr, "fak codex: %v\n", err)
		return 2
	}
	mcMode, mcErr := normalizeManagedCacheMode(*managedCache)
	if mcErr != nil {
		fmt.Fprintf(stderr, "fak codex: %v\n", mcErr)
		return 2
	}
	*ggufPath = pathutil.ExpandTilde(*ggufPath)
	*tokenizerPath = pathutil.ExpandTilde(*tokenizerPath)
	*codexHome = pathutil.ExpandTilde(*codexHome)

	fakBin := tuiExecutable()
	launch := codexLaunchOptions{
		dryRun:          *dryRun,
		skipPermissions: *skipPermissions && !*nativePermissions,
		splitMode:       *splitMode,
		splitWhere:      *splitWhere,
		splitInterval:   *splitInterval,
		policyPath:      *policyPath,
		apiKeyEnv:       *apiKeyEnv,
		baseURL:         *baseURL,
		remoteServe:     *remoteServe,
		model:           *model,
		managedCache:    mcMode,
		auditPath:       *auditPath,
		noAudit:         *noAudit,
		quiet:           *quiet,
		localAuto:       *localAuto,
		ggufPath:        *ggufPath,
		gpuBackend:      *gpuBackend,
		tokenizerPath:   *tokenizerPath,
		codexConfig:     *codexConfig,
		codexHome:       *codexHome,
		passthrough:     fs.Args(),
	}
	if !launch.dryRun {
		workingDir, cwdErr := os.Getwd()
		if cwdErr != nil {
			fmt.Fprintf(stderr, "fak codex: resolve launch directory: %v\n", cwdErr)
			return 1
		}
		if rc := runCodexLoopGate(stderr, codexLoopGateConfig{
			Threshold:     *loopGate,
			CodexHome:     *codexHome,
			SinceHours:    *loopGateSinceHours,
			Limit:         *loopGateLimit,
			WorkingDir:    workingDir,
			Quiet:         *quiet,
			BypassCommand: "fak codex --loop-gate off",
		}); rc != 0 {
			return rc
		}
		launch.loopGateChecked = true
	}
	argvOut := buildCodexLaunchArgv(fakBin, launch)

	if launch.dryRun {
		fmt.Fprintln(stderr, "fak codex: dry-run - not launching")
		fmt.Fprintln(stderr, "  view        = agent 80% / fak info 20% (--split "+launch.splitMode+")")
		if launch.skipPermissions {
			fmt.Fprintln(stderr, "  permissions = Codex approval/sandbox bypass (managed default); fak gates remain active; Codex subagents inherit this mode")
		} else {
			fmt.Fprintln(stderr, "  permissions = Codex native approvals + sandbox explicitly restored; Codex subagents inherit this mode; fak gates remain active")
		}
		fmt.Fprintln(stderr, "  command     = "+strings.Join(argvOut, " "))
		fmt.Fprintln(stdout, strings.Join(argvOut, " "))
		return 0
	}

	started := time.Now()
	fmt.Fprintln(stderr, "fak codex: launching Codex through fak guard ...")
	code := codexLaunchRun(stdout, stderr, argvOut, os.Environ())
	if code == 0 {
		fmt.Fprintf(stderr, "fak codex: Codex completed successfully in %s\n", time.Since(started).Round(time.Millisecond))
	}
	return code
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

const codexCompactTokenLimit = 96000

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
	// Managed-cache posture: emit only a non-default on|off (auto is guard's own default), so an
	// unconfigured `fak codex` argv is byte-identical to before this flag existed.
	if m := strings.TrimSpace(o.managedCache); m != "" && m != guardManagedCacheAuto {
		argv = append(argv, "--managed-cache", m)
	}
	appendKV("--audit", o.auditPath)
	// `fak guard` no longer carries a --no-audit alias; 'off' is the spelling --audit
	// documents. Emitted after the --audit passthrough above so it wins on a flag parse.
	if o.noAudit {
		argv = append(argv, "--audit", "off")
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
	appendKV("--codex-home", o.codexHome)
	if o.loopGateChecked {
		// fak codex just ran this same gate before spawning guard. Disable the
		// nested copy so one launch produces one verdict instead of two scans and
		// two identical operator messages. Direct `fak guard -- codex` launches
		// still run their own gate because they never set this internal bit.
		argv = append(argv, "--codex-loop-gate", "off")
	}

	argv = append(argv, "--", "codex")
	// fak's history compactor is intentionally attached to the Anthropic Messages
	// wire. Codex uses the OpenAI Responses wire, so its equivalent shed line must
	// be configured in Codex itself. Without this override fakc inherits Codex's
	// near-window default (~245K on a 258K effective window), which made guarded
	// sessions appear never to compact and left headless workers above the 96K
	// budget tracked by #4253. A later user-supplied -c remains authoritative.
	argv = append(argv, "-c", fmt.Sprintf("model_auto_compact_token_limit=%d", codexCompactTokenLimit))
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
		code := childprocess.ExitCode(err, 1)
		if code == 1 {
			fmt.Fprintf(stderr, "fak codex: %v\n", err)
		}
		return code
	}
	return 0
}

func printCodexOperatorOverride(stderr io.Writer, cfg codexLoopGateConfig) {
	command := strings.TrimSpace(cfg.BypassCommand)
	if command == "" {
		command = "fak codex --loop-gate off"
	}
	fmt.Fprintf(stderr, "fak codex: operator override: rerun as `%s` (the flag belongs after the fak verb).\n", command)
}

func runCodexLoopGate(stderr io.Writer, cfg codexLoopGateConfig) int {
	threshold := strings.ToLower(strings.TrimSpace(cfg.Threshold))
	if threshold == "" || threshold == "off" || threshold == "none" || threshold == "false" || threshold == "0" {
		return 0
	}
	if _, ok := codexLoopFailOnRank(threshold); !ok {
		fmt.Fprintf(stderr, "fak codex: invalid --loop-gate %q (want loop, action, or off)\n", cfg.Threshold)
		return 2
	}
	if d, ok, err := diagnoseCurrentCodexLoop(cfg.CodexHome); ok {
		if err != nil {
			fmt.Fprintf(stderr, "fak codex: current-thread gate audit failed: %v\n", err)
			return 1
		}
		if codexLoopDiagnosisUnguarded(d) {
			fmt.Fprintf(stderr, "fak codex: current-thread gate REFUSE fail-on=unguarded verdict=%s reason=%s\n",
				d.Verdict, codexLoopDiagnosisGateReason(d, "unguarded"))
			fmt.Fprint(stderr, renderCodexLoopDiagnosis(d))
			printCodexOperatorOverride(stderr, cfg)
			return 1
		}
		if !cfg.Quiet {
			fmt.Fprintf(stderr, "fak codex: current-thread gate allow provider=%s verdict=%s session=%s\n",
				firstString(strings.TrimSpace(d.ModelProvider), "-"), d.Verdict, firstString(strings.TrimSpace(d.SessionID), "-"))
		}
	}
	rep, _, err := diagnoseNewestCodexLoopForLaunch(cfg.CodexHome, cfg.SinceHours, cfg.WorkingDir)
	if err != nil {
		fmt.Fprintf(stderr, "fak codex: loop gate audit failed: %v\n", err)
		return 1
	}
	gateCode, remediationCount, _ := codexLoopGuardedLaunchGate(rep, threshold)
	if gateCode != 0 {
		fmt.Fprintf(stderr, "fak codex: loop gate REFUSE fail-on=%s verdict=%s reason=%s\n",
			codexLoopFailOnName(threshold), rep.Verdict, rep.Reason)
		fmt.Fprint(stderr, renderCodexLoopRecentReport(rep))
		printCodexOperatorOverride(stderr, cfg)
		return 1
	}
	if remediationCount > 0 {
		if !cfg.Quiet {
			fmt.Fprintf(stderr, "fak codex: loop gate remediation allow fail-on=%s direct_matches=%d: prior matching rollouts bypassed fak; this child is entering through fak guard\n",
				codexLoopFailOnName(threshold), remediationCount)
		}
		return 0
	}
	if !cfg.Quiet {
		fmt.Fprintf(stderr, "fak codex: loop gate allow fail-on=%s verdict=%s scanned=%d\n",
			codexLoopFailOnName(threshold), rep.Verdict, rep.Scanned)
	}
	return 0
}
