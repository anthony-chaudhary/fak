package main

import (
	"errors"
	"flag"
	"fmt"
	"os"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/modelroute"
)

type agentFlags struct {
	task          *string
	outputStyle   *string
	consoleConfig *string
	workProfile   *string
	provider      *string
	baseURL       *string
	model         *string
	apiKeyEnv     *string
	anthropicAuth *string
	offline       *bool
	native        *bool
	maxTurns      *int
	out           *string
	logOut        *string
	policyPath    *string
	codeTools     *bool
	codeWorkspace *string
	routeManifest *string
	routeAccounts *string
	keepAwake     *string
}

func newAgentFlagSet() (*flag.FlagSet, *agentFlags) {
	fs := flag.NewFlagSet("agent", flag.ExitOnError)
	verbFlagUsage(fs, "agent")
	af := &agentFlags{}
	af.task = fs.String("task", agent.DefaultTask, "the user task the agent must complete")
	af.outputStyle = fs.String("output-style", agentDefaultOutputStyle, "response shape: full|native:{low|medium|high}|caveman:{low|medium|high}; defaults to caveman:medium, full disables it (see `fak agent profiles`)")
	af.consoleConfig = fs.String("console-config", defaultTUIConsoleFile(), "persisted operator preferences (default: FAK_CONSOLE_FILE, else ~/.fak/console.json)")
	af.workProfile = fs.String("work-profile", agentDefaultWorkProfile, "implementation policy: ponytail:{low|medium|high}|standard; defaults to ponytail:medium, standard disables it (see `fak agent profiles`)")
	af.provider = fs.String("provider", "openai", "provider transcript wire: openai, anthropic, gemini, or xai")
	af.baseURL = fs.String("base-url", "", "provider base URL (OpenAI-compatible: .../v1; Gemini native: .../v1beta; Anthropic native: https://api.anthropic.com)")
	af.model = fs.String("model", "gemini-2.5-flash", "model id")
	af.apiKeyEnv = fs.String("api-key-env", "GEMINI_API_KEY", "env var holding the API key")
	af.anthropicAuth = fs.String("anthropic-auth", "auto", "(--provider anthropic) how to present the credential: auto (sniff the token shape - correct for api.anthropic.com), bearer, or x-api-key. Pass bearer for a THIRD-PARTY Anthropic-compatible endpoint whose tenant token is not an sk-ant-* key: auto would send x-api-key and the call would 401 even with a correct base URL, model, and body")
	af.offline = fs.Bool("offline", false, "use the deterministic mock planner (no network)")
	af.native = fs.Bool("native", false, "run one kernel-mediated arm and print its final answer (basic terminal mode)")
	af.maxTurns = fs.Int("max-turns", 10, "max model turns per arm")
	af.out = fs.String("out", "agent-report.json", "report output path")
	af.logOut = fs.String("log", "", "optional path to write the per-call trace log")
	af.policyPath = fs.String("policy", "", "load the capability floor from a manifest (default: the built-in floor plus bounded repository code tools when enabled; see `fak policy --dump`)")
	af.codeTools = fs.Bool("code-tools", true, "arm bounded kernel Read/Write/Edit/Bash/Grep/Glob in the current repository; use --code-tools=false to disable")
	af.codeWorkspace = fs.String("code-workspace", "", "override the workspace root for default-on bounded repository code tools")
	af.routeManifest = fs.String("route-manifest", "", "model-routing policy to install for the fak arm; each tool call is classified and a single-model PICK binds abi.ToolCall.Engine before kernel submit")
	af.routeAccounts = fs.String("route-accounts", "", "model-account roster used to resolve routed model ids to account-bound engine routes")
	af.keepAwake = fs.String("keep-awake", KeepAwakeOff, "prevent OS sleep during execution: off|while-active|always (default off)")
	return fs, af
}

// fak agent  -  the LIVE agentic loop. A real model (or the offline mock) drives a
// multi-turn tool-calling conversation TWICE over the same task: once with every
// tool call mediated by the in-process kernel (fak arm), once naive (the "now"
// baseline). It reports turns, tokens, in-syscall repairs, vDSO dedup hits,
// adjudicator denies, and MMU quarantines for each arm  -  the real turn-use-vs-now
// measurement the static bench could not produce.
func cmdAgent(argv []string) {
	if len(argv) > 0 && argv[0] == "profiles" {
		if err := printAgentOutputProfiles(os.Stdout, argv[1:]); err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(2)
		}
		return
	}
	fs, af := newAgentFlagSet()
	_ = fs.Parse(argv)

	keepAwakeMode, err := validateKeepAwake(*af.keepAwake)
	if err != nil {
		fmt.Fprintf(os.Stderr, "fak agent: %v\n", err)
		os.Exit(2)
	}
	*af.keepAwake = keepAwakeMode

	processWakeReleaser, err := acquireProcessKeepAwake(*af.keepAwake, "fak agent (always)")
	if err != nil {
		fmt.Fprintf(os.Stderr, "fak agent: keep-awake: %v\n", err)
	}
	if processWakeReleaser != nil {
		defer processWakeReleaser.Release()
	}

	outputStyleExplicit := false
	fs.Visit(func(f *flag.Flag) {
		if f.Name == "output-style" {
			outputStyleExplicit = true
		}
	})
	preference, err := resolveAgentOutputStylePreference(*af.outputStyle, outputStyleExplicit, *af.consoleConfig)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(2)
	}
	restoreStyle, err := applyAgentOutputStyle(preference.Style)
	if err != nil {
		fmt.Fprintf(os.Stderr, "agent: set --output-style: %v\n", err)
		os.Exit(2)
	}
	defer restoreStyle()
	workProfileExplicit := false
	fs.Visit(func(f *flag.Flag) {
		if f.Name == "work-profile" {
			workProfileExplicit = true
		}
	})
	work, err := resolveAgentWorkProfilePreference(*af.workProfile, workProfileExplicit)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(2)
	}
	restoreWork, err := applyAgentWorkProfile(work.Profile)
	if err != nil {
		fmt.Fprintf(os.Stderr, "agent: set --work-profile: %v\n", err)
		os.Exit(2)
	}
	defer restoreWork()
	printAgentProfileValue(os.Stderr, preference, work)
	applyPolicy(*af.policyPath)
	loadedRoute, loadedAccounts, runOpts, err := loadAgentRouteOptionsWithAccounts(*af.routeManifest, *af.routeAccounts)
	must(err)
	runOpts = append(runOpts, agent.WithResponseProfileSource(preference.Source))
	if loadedRoute != nil {
		fmt.Fprintf(os.Stderr, "fak agent: loaded model-routing policy from %s\n", *af.routeManifest)
	}
	announceAgentRouteAccounts(os.Stderr, *af.routeAccounts, loadedAccounts)
	if *af.codeTools {
		root := strings.TrimSpace(*af.codeWorkspace)
		if root == "" {
			root, err = os.Getwd()
			must(err)
		}
		catalog, armErr := agent.ArmFocusedCodeTools(root)
		must(armErr)
		defer agent.DisarmCodeTools()
		runOpts = append(runOpts, agent.WithToolCatalog(catalog))
	}

	var planner agent.Planner
	if *af.offline || *af.baseURL == "" {
		if !*af.offline {
			fmt.Fprintln(os.Stderr, "fak agent: no --base-url given; using the offline mock planner (pass --base-url for a live run)")
		}
		planner = agent.NewMockPlanner(*af.model)
	} else {
		key := os.Getenv(*af.apiKeyEnv)
		if key == "" {
			// A local endpoint (e.g. the transformers shim) needs no key; a remote
			// one will return 401, which the planner surfaces clearly. Warn, proceed.
			fmt.Fprintf(os.Stderr, "fak agent: env %s is empty  -  proceeding with no auth header (fine for a local endpoint)\n", *af.apiKeyEnv)
		}
		p, err := agent.NewProviderHTTPPlanner(*af.provider, *af.baseURL, *af.model, key)
		must(err)
		scheme, ok := agent.ParseAnthropicAuthScheme(*af.anthropicAuth)
		if !ok {
			must(fmt.Errorf("--anthropic-auth %q: want auto, bearer, or x-api-key", *af.anthropicAuth))
		}
		p.AnthropicAuthScheme = scheme
		planner = p
	}

	if *af.native {
		if *af.logOut != "" {
			must(errors.New("fak agent: --native does not support --log; use --out for its receipt"))
		}
		activeWakeReleaser, _ := acquireAgentRunKeepAwake(*af.keepAwake)
		metrics, err := agent.RunArm(ctx(), planner, *af.task, true, *af.maxTurns, nil, runOpts...)
		if activeWakeReleaser != nil {
			_ = activeWakeReleaser.Release()
		}
		must(err)
		receipt := newNativeAgentReceipt(*af.task, planner.Model(), metrics)
		must(os.WriteFile(*af.out, jsonIndent(receipt), 0o644))
		fmt.Fprintln(os.Stdout, metrics.FinalAnswer)
		announceAgentReport(os.Stderr, *af.out)
		return
	}

	activeWakeReleaser, _ := acquireAgentRunKeepAwake(*af.keepAwake)
	res, trace, err := agent.Run(ctx(), planner, *af.task, *af.maxTurns, runOpts...)
	if activeWakeReleaser != nil {
		_ = activeWakeReleaser.Release()
	}
	must(err)

	must(os.WriteFile(*af.out, jsonIndent(res), 0o644))
	if *af.logOut != "" {
		_ = os.WriteFile(*af.logOut, agent.RenderTrace(trace), 0o644)
	}
	agent.PrintReport(os.Stdout, res, trace, *af.out)
	// The summary above names the file; this names the DIRECTORY it went to, so
	// the first-run proof never leaves an unfindable artifact behind (#5473).
	announceAgentReport(os.Stderr, *af.out)
}

func loadAgentRouteOptions(path string) (*modelroute.Manifest, []agent.RunOption, error) {
	path = strings.TrimSpace(path)
	if path == "" {
		return nil, nil, nil
	}
	manifest, err := modelroute.LoadManifest(path)
	if err != nil {
		return nil, nil, fmt.Errorf("fak agent: --route-manifest: %w", err)
	}
	return &manifest, []agent.RunOption{agent.WithRouteManifest(&manifest)}, nil
}
