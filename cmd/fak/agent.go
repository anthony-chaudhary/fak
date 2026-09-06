package main

import (
	"crypto/rand"
	"encoding/hex"
	"errors"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/adjudicator"
	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/dropin"
	"github.com/anthony-chaudhary/fak/internal/journal"
	"github.com/anthony-chaudhary/fak/internal/modelroute"
	"github.com/anthony-chaudhary/fak/internal/systools"
)

type agentFlags struct {
	task                  *string
	outputStyle           *string
	consoleConfig         *string
	workProfile           *string
	reasoningProfile      *string
	effort                *string
	thinkingBudget        *int
	provider              *string
	baseURL               *string
	model                 *string
	apiKeyEnv             *string
	anthropicAuth         *string
	offline               *bool
	native                *bool
	raw                   *bool
	mode                  *string
	maxTurns              *int
	out                   *string
	logOut                *string
	policyPath            *string
	codeTools             *bool
	codeWorkspace         *string
	sysTools              *bool
	mcpTools              *bool
	routeManifest         *string
	routeAccounts         *string
	keepAwake             *string
	skills                *bool
	skillsDir             *string
	workflow              *string
	workflowStep          *bool
	workflowCheckpointDir *string
	memory                *bool
	memoryStore           *string
	posture               *string
	mcpConfig             *string
	session               *string
	resume                *string
	sessionDir            *string
}

func newAgentFlagSet() (*flag.FlagSet, *agentFlags) {
	fs := flag.NewFlagSet("agent", flag.ExitOnError)
	verbFlagUsage(fs, "agent")
	af := &agentFlags{}
	af.task = fs.String("task", agent.DefaultTask, "the user task the agent must complete")
	af.outputStyle = fs.String("output-style", agentDefaultOutputStyle, "response shape: full|native:{low|medium|high}|caveman:{low|medium|high}; defaults to caveman:medium, full disables it (see `fak agent profiles`)")
	af.consoleConfig = fs.String("console-config", defaultTUIConsoleFile(), "persisted operator preferences (default: FAK_CONSOLE_FILE, else ~/.fak/console.json)")
	af.workProfile = fs.String("work-profile", agentDefaultWorkProfile, "implementation policy: ponytail:{low|medium|high}|standard; defaults to ponytail:medium, standard disables it (see `fak agent profiles`)")
	af.reasoningProfile = fs.String("reasoning-profile", agent.ReasoningProfileDefault, "named reasoning profile: default|baseline|deep-reason (default: default)")
	af.effort = fs.String("effort", "", "reasoning effort for model inference: none|low|medium|balanced|adaptive|high")
	af.thinkingBudget = fs.Int("thinking-budget", -1, "explicit thinking token budget ceiling (>=0 overrides --effort; 0 disables thinking)")
	af.provider = fs.String("provider", "openai", "provider transcript wire: openai, openai-responses, astra, anthropic, gemini, or xai")
	af.baseURL = fs.String("base-url", "", "provider base URL (OpenAI-compatible: .../v1; Gemini native: .../v1beta; Anthropic native: https://api.anthropic.com)")
	af.model = fs.String("model", "gemini-2.5-flash", "model id")
	af.apiKeyEnv = fs.String("api-key-env", "GEMINI_API_KEY", "env var holding the API key")
	af.anthropicAuth = fs.String("anthropic-auth", "auto", "(--provider anthropic) how to present the credential: auto (sniff the token shape - correct for api.anthropic.com), bearer, or x-api-key. Pass bearer for a THIRD-PARTY Anthropic-compatible endpoint whose tenant token is not an sk-ant-* key: auto would send x-api-key and the call would 401 even with a correct base URL, model, and body")
	af.offline = fs.Bool("offline", false, "use the deterministic mock planner (no network)")
	af.native = fs.Bool("native", false, "run one kernel-mediated arm and print its final answer (basic terminal mode)")
	af.raw = fs.Bool("raw", false, "run one unmediated baseline harness arm (skipping the mediated fak kernel arm)")
	af.mode = fs.String("mode", "", "execution mode: dual (default), native, or raw")
	af.maxTurns = fs.Int("max-turns", 10, "max model turns per arm")
	af.out = fs.String("out", "agent-report.json", "report output path")
	af.logOut = fs.String("log", "", "optional path to write the per-call trace log")
	af.policyPath = fs.String("policy", "", "load the capability floor from a manifest (default: the built-in floor plus bounded repository code tools when enabled; see `fak policy --dump`)")
	af.codeTools = fs.Bool("code-tools", true, "arm bounded kernel Read/Write/Edit/Bash/Grep/Glob in the current repository; use --code-tools=false to disable")
	af.codeWorkspace = fs.String("code-workspace", "", "override the workspace root for default-on bounded repository code tools")
	af.sysTools = fs.Bool("sys-tools", true, "arm safe read-only system and web utility tools (get_time, fetch_web, web_search); use --sys-tools=false to disable")
	af.mcpTools = fs.Bool("mcp-tools", true, "arm native fak MCP features (fak_read, fak_tools_search, fak_adjudicate, fak_syscall); use --mcp-tools=false to disable")
	af.routeManifest = fs.String("route-manifest", "", "model-routing policy to install for the fak arm; each tool call is classified and a single-model PICK binds abi.ToolCall.Engine before kernel submit")
	af.routeAccounts = fs.String("route-accounts", "", "model-account roster used to resolve routed model ids to account-bound engine routes")
	af.keepAwake = fs.String("keep-awake", KeepAwakeOff, "prevent OS sleep during execution: off|while-active|always (default off)")
	af.skills = fs.Bool("skills", true, "enable Agent Skills discovery and dynamic faulting")
	af.skillsDir = fs.String("skills-dir", "", "optional custom directory to search for SKILL.md definitions")
	af.workflow = fs.String("workflow", "", "name of workflow to execute (e.g. fleet-wave)")
	af.workflowStep = fs.Bool("workflow-step", false, "execute a single workflow phase step instead of full workflow")
	af.workflowCheckpointDir = fs.String("workflow-checkpoint-dir", ".fak/workflows", "directory for workflow state checkpoints")
	af.memory = fs.Bool("memory", true, "discover and inject verified workspace memory notes into agent prompt; use --memory=false to disable")
	af.memoryStore = fs.String("memory-store", "", "optional custom memory store path (directory or MEMORY.md); defaults to auto-discovery")
	af.posture = fs.String("posture", "default_open", "adjudication posture: default_open|fail_closed|admit_and_log (default: default_open; env: FAK_AGENT_POSTURE or FAK_GUARD_POSTURE)")
	af.mcpConfig = fs.String("mcp-config", os.Getenv("FAK_MCP_CONFIG"), "optional path to MCP client configuration file")
	af.session = fs.String("session", "", "session ID for durable checkpointing in .fak/sessions/ (generates unique ID if 'auto' or empty with flag)")
	af.resume = fs.String("resume", "", "resume from existing session ID or checkpoint path")
	af.sessionDir = fs.String("session-dir", ".fak/sessions", "directory for session checkpoints (default: .fak/sessions)")
	return fs, af
}

// fak agent  -  the LIVE agentic loop. A real model (or the offline mock) drives a
// multi-turn tool-calling conversation TWICE over the same task: once with every
// tool call mediated by the in-process kernel (fak arm), once naive (the "now"
// baseline). It reports turns, tokens, in-syscall repairs, vDSO dedup hits,
// adjudicator denies, and MMU quarantines for each arm  -  the real turn-use-vs-now
// measurement the static bench could not produce.
func cmdAgent(argv []string) {
	runAgent(argv)
}

func runAgent(argv []string) {
	if len(argv) > 0 && (argv[0] == "list" || argv[0] == "descriptors" || argv[0] == "declarative") {
		code := runAgentsList(os.Stdout, os.Stderr, argv[1:])
		if code != 0 {
			os.Exit(code)
		}
		return
	}
	if len(argv) > 0 && argv[0] == "profiles" {
		if err := printAgentOutputProfiles(os.Stdout, argv[1:]); err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(2)
		}
		return
	}
	if len(argv) > 0 && argv[0] == "resume" {
		if len(argv) < 2 || strings.HasPrefix(argv[1], "-") {
			fmt.Fprintln(os.Stderr, "fak agent resume: requires a session ID or checkpoint path")
			os.Exit(2)
		}
		argv = append([]string{"--resume", argv[1]}, argv[2:]...)
	}
	fs, af := newAgentFlagSet()
	_ = fs.Parse(argv)

	if *af.resume == "" && fs.NArg() > 0 && fs.Arg(0) == "resume" {
		if fs.NArg() < 2 || strings.HasPrefix(fs.Arg(1), "-") {
			fmt.Fprintln(os.Stderr, "fak agent resume: requires a session ID or checkpoint path")
			os.Exit(2)
		}
		*af.resume = fs.Arg(1)
	}

	postureExplicit := false
	taskExplicit := false
	modelExplicit := false
	providerExplicit := false
	baseURLExplicit := false
	sessionExplicit := false
	sessionDirExplicit := false
	fs.Visit(func(f *flag.Flag) {
		switch f.Name {
		case "posture":
			postureExplicit = true
		case "task":
			taskExplicit = true
		case "model":
			modelExplicit = true
		case "provider":
			providerExplicit = true
		case "base-url":
			baseURLExplicit = true
		case "session":
			sessionExplicit = true
		case "session-dir":
			sessionDirExplicit = true
		}
	})

	defaultSessionDir := *af.sessionDir
	if defaultSessionDir == "" {
		defaultSessionDir = agent.DefaultSessionCheckpointDir
	}

	var resumedCP *agent.SessionCheckpoint
	if *af.resume != "" {
		if !sessionDirExplicit {
			if dir := filepath.Dir(*af.resume); dir != "." && dir != "" && dir != string(filepath.Separator) {
				defaultSessionDir = dir
			}
		}
		cp, err := agent.LoadSessionCheckpoint(*af.resume, defaultSessionDir)
		must(err)
		resumedCP = cp

		if *af.session == "" {
			*af.session = cp.SessionID
		}
		if !taskExplicit && cp.Task != "" {
			*af.task = cp.Task
		}
		if !modelExplicit && cp.Model != "" {
			*af.model = cp.Model
		}
		if !providerExplicit && cp.Provider != "" {
			*af.provider = cp.Provider
		}
		if !baseURLExplicit && cp.BaseURL != "" {
			*af.baseURL = cp.BaseURL
		}
	} else if sessionExplicit || *af.session != "" {
		if *af.session == "" || strings.EqualFold(*af.session, "auto") {
			*af.session = generateAgentSessionID()
		}
	}

	rawPosture := *af.posture
	if !postureExplicit {
		if env := os.Getenv("FAK_AGENT_POSTURE"); env != "" {
			rawPosture = env
		} else if env := os.Getenv("FAK_GUARD_POSTURE"); env != "" {
			rawPosture = env
		}
	}
	switch strings.ToLower(strings.TrimSpace(rawPosture)) {
	case "fail_closed", "strict":
		agent.SetConfiguredPosture(adjudicator.PostureFailClosed)
	case "admit_and_log":
		agent.SetConfiguredPosture(adjudicator.PostureAdmitAndLog)
	default:
		agent.SetConfiguredPosture(adjudicator.PostureDefaultOpen)
	}

	isRaw, isNative, err := resolveAgentMode(*af.raw, *af.native, *af.mode)
	must(err)

	if af.reasoningProfile != nil && *af.reasoningProfile != "" {
		if err := validateReasoningProfile(*af.reasoningProfile); err != nil {
			fmt.Fprintf(os.Stderr, "fak agent: %v\n", err)
			os.Exit(2)
		}
	}

	if *af.workflow != "" {
		if err := runWorkflowCLI(*af.workflow, *af.workflowStep, *af.workflowCheckpointDir); err != nil {
			os.Exit(1)
		}
		return
	}

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
	if !isRaw {
		applyPolicy(*af.policyPath)
	}
	loadedRoute, loadedAccounts, runOpts, err := loadAgentRouteOptionsWithAccounts(*af.routeManifest, *af.routeAccounts)
	must(err)
	runOpts = append(runOpts, agent.WithResponseProfileSource(preference.Source))
	if loadedRoute != nil && !isRaw {
		fmt.Fprintf(os.Stderr, "fak agent: loaded model-routing policy from %s\n", *af.routeManifest)
	}
	if !isRaw {
		announceAgentRouteAccounts(os.Stderr, *af.routeAccounts, loadedAccounts)
	}
	root := strings.TrimSpace(*af.codeWorkspace)
	if root == "" {
		root, err = os.Getwd()
		must(err)
	}
	var catalog []agent.ToolDef
	if *af.codeTools {
		var extraDirs []string
		if *af.skillsDir != "" {
			extraDirs = append(extraDirs, *af.skillsDir)
		}
		codeCatalog, armErr := agent.ArmCodeToolsWithOptions(agent.CodeToolsOptions{
			Root:         root,
			Focused:      true,
			EnableSkills: *af.skills,
			ExtraDirs:    extraDirs,
		})
		must(armErr)
		defer agent.DisarmCodeTools()
		catalog = append(catalog, codeCatalog...)
	}
	if *af.sysTools {
		sysCatalog, sysErr := agent.ArmSysTools(systools.Config{})
		must(sysErr)
		defer agent.DisarmSysTools()
		catalog = append(catalog, sysCatalog...)
	}
	if *af.mcpTools {
		mcpCatalog, mcpErr := agent.ArmMCPTools()
		must(mcpErr)
		defer agent.DisarmMCPTools()
		catalog = append(catalog, mcpCatalog...)
	}
	if len(catalog) > 0 {
		runOpts = append(runOpts, agent.WithToolCatalog(catalog))
	}
	runOpts = append(runOpts, agentEffortRunOptions(af)...)
	if opt := agentReasoningProfileRunOption(af); opt != nil {
		runOpts = append(runOpts, opt)
	}
	if memOpt, _ := resolveAgentMemoryOption(*af.memory, *af.memoryStore, root); memOpt != nil {
		runOpts = append(runOpts, memOpt)
	}

	if resumedCP != nil {
		runOpts = append(runOpts, agent.WithSessionCheckpointState(resumedCP, defaultSessionDir))
		msgs := append([]agent.Message(nil), resumedCP.Messages...)
		if taskExplicit && *af.task != "" && *af.task != resumedCP.Task {
			if len(msgs) > 0 && msgs[len(msgs)-1].Role == agent.RoleAssistant {
				msgs = append(msgs, agent.Message{Role: agent.RoleUser, Content: *af.task})
			}
		}
		runOpts = append(runOpts, agent.WithConversation(msgs))
	} else if *af.session != "" {
		runOpts = append(runOpts, agent.WithSessionCheckpoint(*af.session, defaultSessionDir))
	}

	effectiveBaseURL := *af.baseURL
	if effectiveBaseURL == "" {
		if env := os.Getenv(dropin.EnvVar(*af.provider, "")); env != "" {
			effectiveBaseURL = env
		} else if providerExplicit && !*af.offline {
			effectiveBaseURL = dropin.DefaultBaseURL(*af.provider)
		}
	}
	if *af.provider != "" {
		runOpts = append(runOpts, agent.WithProvider(*af.provider))
	}
	if effectiveBaseURL != "" {
		runOpts = append(runOpts, agent.WithBaseURL(effectiveBaseURL))
	}

	var planner agent.Planner
	if *af.offline || effectiveBaseURL == "" {
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
		p, err := agent.NewProviderHTTPPlanner(*af.provider, effectiveBaseURL, *af.model, key)
		must(err)
		scheme, ok := agent.ParseAnthropicAuthScheme(*af.anthropicAuth)
		if !ok {
			must(fmt.Errorf("--anthropic-auth %q: want auto, bearer, or x-api-key", *af.anthropicAuth))
		}
		p.AnthropicAuthScheme = scheme
		planner = p
	}

	var auditJournal *journal.Journal
	if *af.logOut != "" && strings.HasSuffix(strings.ToLower(*af.logOut), ".jsonl") {
		if dir := filepath.Dir(*af.logOut); dir != "" && dir != "." {
			_ = os.MkdirAll(dir, 0o755)
		}
		j, err := journal.Open(*af.logOut)
		must(err)
		defer func() {
			_ = j.Flush()
			_ = j.Close()
		}()
		auditJournal = j
		runOpts = append(runOpts, agent.WithAuditJournal(j))
	}

	if isRaw {
		if *af.logOut != "" && auditJournal == nil {
			must(errors.New("fak agent: --raw does not support --log; use --out for its receipt"))
		}
		activeWakeReleaser, _ := acquireAgentRunKeepAwake(*af.keepAwake)
		metrics, err := agent.RunArm(ctx(), planner, *af.task, false, *af.maxTurns, nil, runOpts...)
		if activeWakeReleaser != nil {
			_ = activeWakeReleaser.Release()
		}
		must(err)
		if auditJournal != nil {
			_ = auditJournal.Flush()
		}
		receipt := newRawAgentReceipt(*af.task, planner.Model(), metrics)
		data := jsonIndent(receipt)
		if *af.out == "" || *af.out == "-" || *af.out == "stdout" {
			fmt.Fprintln(os.Stdout, string(data))
		} else {
			must(os.WriteFile(*af.out, data, 0o644))
			fmt.Fprintln(os.Stdout, metrics.FinalAnswer)
			announceAgentReport(os.Stderr, *af.out)
		}
		return
	}

	if isNative {
		if *af.logOut != "" && auditJournal == nil {
			must(errors.New("fak agent: --native does not support --log; use --out for its receipt"))
		}
		activeWakeReleaser, _ := acquireAgentRunKeepAwake(*af.keepAwake)
		metrics, err := agent.RunArm(ctx(), planner, *af.task, true, *af.maxTurns, nil, runOpts...)
		if activeWakeReleaser != nil {
			_ = activeWakeReleaser.Release()
		}
		must(err)
		if auditJournal != nil {
			_ = auditJournal.Flush()
		}
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
	if auditJournal != nil {
		_ = auditJournal.Flush()
	}

	must(os.WriteFile(*af.out, jsonIndent(res), 0o644))
	if *af.logOut != "" && auditJournal == nil {
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

func agentEffortRunOptions(af *agentFlags) []agent.RunOption {
	var opts []agent.RunOption
	if af == nil {
		return opts
	}
	if af.effort != nil && *af.effort != "" {
		opts = append(opts, agent.WithRunReasoningEffort(*af.effort))
	}
	if af.thinkingBudget != nil && *af.thinkingBudget >= 0 {
		opts = append(opts, agent.WithRunThinkingBudget(*af.thinkingBudget))
	}
	return opts
}

func agentReasoningProfileRunOption(af *agentFlags) agent.RunOption {
	if af == nil || af.reasoningProfile == nil || *af.reasoningProfile == "" {
		return nil
	}
	return agent.WithReasoningProfile(*af.reasoningProfile)
}

func validateReasoningProfile(profile string) error {
	switch strings.ToLower(strings.TrimSpace(profile)) {
	case agent.ReasoningProfileDefault, agent.ReasoningProfileBaseline, agent.ReasoningProfileDeepReason:
		return nil
	default:
		return fmt.Errorf("unknown --reasoning-profile %q (want default, baseline, or deep-reason)", profile)
	}
}

func resolveAgentMode(raw, native bool, mode string) (isRaw, isNative bool, err error) {
	modeVal := strings.ToLower(strings.TrimSpace(mode))
	switch modeVal {
	case "", "dual", "ab":
		// default dual/A-B mode unless explicit boolean flags set
	case "raw":
		if native {
			return false, false, errors.New("fak agent: cannot specify both native and raw execution modes")
		}
		raw = true
	case "native":
		if raw {
			return false, false, errors.New("fak agent: cannot specify both native and raw execution modes")
		}
		native = true
	default:
		return false, false, fmt.Errorf("fak agent: unknown --mode %q (want dual, ab, native, or raw)", mode)
	}
	if raw && native {
		return false, false, errors.New("fak agent: cannot specify both raw and native execution modes")
	}
	return raw, native, nil
}

func validateAgentMode(raw, native bool, mode string) (isRaw, isNative bool, err error) {
	return resolveAgentMode(raw, native, mode)
}

func generateAgentSessionID() string {
	var b [8]byte
	if _, err := rand.Read(b[:]); err == nil {
		return fmt.Sprintf("session-%s-%s", time.Now().UTC().Format("20060102-150405"), hex.EncodeToString(b[:]))
	}
	return fmt.Sprintf("session-%s-%x", time.Now().UTC().Format("20060102-150405"), time.Now().UnixNano())
}
