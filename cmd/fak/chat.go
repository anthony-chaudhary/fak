package main

import (
	"bufio"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/adjudicator"
	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/dropin"
	"github.com/anthony-chaudhary/fak/internal/policy"
	"github.com/anthony-chaudhary/fak/internal/systools"
	"github.com/anthony-chaudhary/fak/pkg/fakclient"
)

type chatFlags struct {
	provider              *string
	baseURL               *string
	model                 *string
	apiKeyEnv             *string
	codexAuth             *bool
	codexHome             *string
	anthropicAuth         *string
	offline               *bool
	maxTurns              *int
	policyPath            *string
	posture               *string
	task                  *string
	taskFile              *string
	effort                *string
	tools                 *string
	codeTools             *bool
	codeWorkspace         *string
	sysTools              *bool
	mcpTools              *bool
	skills                *bool
	skillsDir             *string
	workflow              *string
	workflowStep          *bool
	workflowCheckpointDir *string
	memory                *bool
	memoryStore           *string
	reasoningProfile      *string
	verbose               *bool
	asJSON                *bool
	receiptOut            *string
}

func newChatFlagSet() (*flag.FlagSet, *chatFlags) {
	fs := flag.NewFlagSet("chat", flag.ExitOnError)
	cf := &chatFlags{}
	cf.provider = fs.String("provider", "openai", "provider transcript wire: openai, anthropic, gemini, or xai")
	cf.baseURL = fs.String("base-url", "", "provider base URL (empty => offline mock planner; no upstream)")
	cf.model = fs.String("model", "gemini-3.8-flash", "model id")
	cf.apiKeyEnv = fs.String("api-key-env", "GEMINI_API_KEY", "env var holding the API key")
	cf.codexAuth = fs.Bool("codex-auth", false, "explicitly reuse a Codex-managed ChatGPT login read-only; Codex owns renewal")
	cf.codexHome = fs.String("codex-home", "", "Codex credential home for --codex-auth (default: existing discovery)")
	cf.anthropicAuth = fs.String("anthropic-auth", "auto", "(--provider anthropic) how to present the credential: auto (sniff the token shape), bearer, or x-api-key. Pass bearer for a THIRD-PARTY Anthropic-compatible endpoint whose tenant token is not an sk-ant-* key")
	cf.offline = fs.Bool("offline", false, "force the deterministic mock planner (no network)")
	cf.maxTurns = fs.Int("max-turns", 10, "max model turns the loop may take to resolve ONE human turn")
	cf.policyPath = fs.String("policy", "", "load the capability floor from a manifest (default: the built-in production capability floor)")
	cf.posture = fs.String("posture", "fail_closed", "adjudication posture: fail_closed|default_open|admit_and_log (default: fail_closed developer floor; env: FAK_AGENT_POSTURE or FAK_GUARD_POSTURE)")
	cf.task = fs.String("task", "", "run a single non-interactive task turn (headless mode) and exit")
	cf.taskFile = fs.String("task-file", "", "read a non-interactive UTF-8 task from a file (mutually exclusive with --task)")
	cf.effort = fs.String("effort", "", "reasoning effort for the native task")
	cf.tools = fs.String("tools", "code", "toolset to arm: code (Read/Write/Edit/Bash/Grep/Glob), demo, or none")
	cf.codeTools = fs.Bool("code-tools", true, "arm bounded kernel Read/Write/Edit/Bash/Grep/Glob in the workspace (alias for --tools=code)")
	cf.codeWorkspace = fs.String("code-workspace", "", "override workspace root for code tools (default: current directory)")
	cf.sysTools = fs.Bool("sys-tools", true, "arm safe read-only system and web utility tools (get_time, fetch_web, web_search); use --sys-tools=false to disable")
	cf.mcpTools = fs.Bool("mcp-tools", true, "arm native fak MCP features")
	cf.skills = fs.Bool("skills", true, "enable Agent Skills discovery and dynamic faulting")
	cf.skillsDir = fs.String("skills-dir", "", "optional custom directory to search for SKILL.md definitions")
	cf.workflow = fs.String("workflow", "", "name of workflow to execute (e.g. fleet-wave)")
	cf.workflowStep = fs.Bool("workflow-step", false, "execute a single workflow phase step instead of full workflow")
	cf.workflowCheckpointDir = fs.String("workflow-checkpoint-dir", ".fak/workflows", "directory for workflow state checkpoints")
	cf.memory = fs.Bool("memory", true, "discover and inject verified workspace memory notes into agent prompt; use --memory=false to disable")
	cf.memoryStore = fs.String("memory-store", "", "optional custom memory store path (directory or MEMORY.md); defaults to auto-discovery")
	cf.reasoningProfile = fs.String("reasoning-profile", agent.ReasoningProfileDefault, "named reasoning profile: default|baseline|deep-reason (default: default)")
	cf.verbose = fs.Bool("verbose", false, "show tool arguments and turn diagnostics in interactive chat")
	cf.asJSON = fs.Bool("json", false, "emit machine-readable JSON execution receipt in headless mode")
	cf.receiptOut = fs.String("receipt", "", "write machine-readable execution receipt JSON to file in headless mode")
	return fs, cf
}

// cmdChat is the minimal native TUI/REPL on the internal/agent seam (#1320, child
// of the #1315 native-harness program): the Apache-clean, single-binary operator
// front door for the OWNED loop. A human types a turn on stdin, agent.RunArm owns
// dispatch, and kernel.Syscall is the sole tool path — so a destructive call the
// capability floor denies lands as a STRUCTURED VALUE the model sees, never an
// executed effect and never an engine dispatch.
//
// It is deliberately NOT cmdAgent (a one-shot A/B benchmark) nor cmdTUI (a loops
// console): each line of input is one human turn, driven through the fak arm of
// RunArm in-process, with no upstream required (the offline mock planner is the
// default, matching `fak agent`). --base-url swaps in a live provider planner.
func cmdChat(argv []string) {
	fs, cf := newChatFlagSet()
	_ = fs.Parse(argv)
	apiKeyExplicit := false
	modelExplicit := false
	baseURLExplicit := false
	providerExplicit := false
	fs.Visit(func(f *flag.Flag) {
		if f.Name == "api-key-env" {
			apiKeyExplicit = true
		}
		if f.Name == "model" {
			modelExplicit = true
		}
		if f.Name == "base-url" {
			baseURLExplicit = true
		}
		if f.Name == "provider" {
			providerExplicit = true
		}
	})
	if *cf.codexHome != "" && !*cf.codexAuth {
		must(fmt.Errorf("fak chat: --codex-home requires --codex-auth"))
	}
	if *cf.codexAuth {
		if *cf.offline || (apiKeyExplicit && *cf.apiKeyEnv != "") {
			must(fmt.Errorf("fak chat: --codex-auth cannot be combined with --offline or an explicit API key environment"))
		}
		if *cf.provider != "openai-responses" || strings.TrimRight(*cf.baseURL, "/") != guardCodexChatGPTBackendBaseURL {
			must(fmt.Errorf("fak chat: --codex-auth requires --provider openai-responses and --base-url %s", guardCodexChatGPTBackendBaseURL))
		}
		*cf.apiKeyEnv = ""
	}
	if *cf.taskFile != "" {
		if *cf.task != "" {
			must(fmt.Errorf("fak chat: --task and --task-file are mutually exclusive"))
		}
		data, err := os.ReadFile(*cf.taskFile)
		must(err)
		if len(strings.TrimSpace(string(data))) == 0 {
			must(fmt.Errorf("fak chat: task file must be nonempty"))
		}
		*cf.task = string(data)
	}

	if *cf.workflow != "" {
		if err := runWorkflowCLI(*cf.workflow, *cf.workflowStep, *cf.workflowCheckpointDir); err != nil {
			os.Exit(1)
		}
		return
	}

	if cf.reasoningProfile != nil && *cf.reasoningProfile != "" {
		if err := validateReasoningProfile(*cf.reasoningProfile); err != nil {
			fmt.Fprintf(os.Stderr, "fak chat: %v\n", err)
			os.Exit(2)
		}
	}

	postureExplicit := false
	fs.Visit(func(f *flag.Flag) {
		if f.Name == "posture" {
			postureExplicit = true
		}
	})
	rawMode := *cf.posture
	if !postureExplicit {
		if env := os.Getenv("FAK_AGENT_POSTURE"); env != "" {
			rawMode = env
		} else if env := os.Getenv("FAK_GUARD_POSTURE"); env != "" {
			rawMode = env
		}
	}
	if strings.TrimSpace(rawMode) == "" {
		rawMode = "fail_closed"
	}

	var runOpts []agent.RunOption
	policyDigest := guardPolicyDigest(guardDefaultPolicyJSON)
	var effectivePosture adjudicator.Posture
	if *cf.policyPath != "" {
		policyDigest = configFileDigest(*cf.policyPath)
		if policyDigest == "" {
			must(fmt.Errorf("fak chat: policy digest unavailable"))
		}
		applyPolicy(*cf.policyPath)
		if loadedDigest := configFileDigest(*cf.policyPath); loadedDigest != policyDigest {
			must(fmt.Errorf("fak chat: policy changed while loading"))
		}
		effMode := parseChatMode(rawMode)
		agent.SetConfiguredPosture(effMode)
		snap := adjudicator.Default.PolicySnapshot()
		if postureExplicit {
			snap.Posture = effMode
		}
		effectivePosture = snap.Posture
		runOpts = append(runOpts, agent.WithPolicySnapshot(snap))
	} else {
		initDevRules(rawMode)
		effectivePosture = adjudicator.Default.PolicySnapshot().Posture
	}

	effectiveBaseURL := *cf.baseURL
	if effectiveBaseURL == "" && providerExplicit && !*cf.offline {
		effectiveBaseURL = dropin.DefaultBaseURL(*cf.provider)
	}
	if effectiveBaseURL == "" && !*cf.offline && !baseURLExplicit {
		if localModel, ok := probeLocalGateway("http://127.0.0.1:8080"); ok {
			effectiveBaseURL = "http://127.0.0.1:8080/v1"
			if !modelExplicit && localModel != "" && localModel != "mock" {
				*cf.model = localModel
			}
			fmt.Fprintln(os.Stderr, "fak chat: connected to local gateway")
		}
	} else if effectiveBaseURL != "" && !modelExplicit && !*cf.offline {
		if serverModel := detectServerModel(effectiveBaseURL); serverModel != "" && serverModel != "mock" {
			*cf.model = serverModel
			fmt.Fprintf(os.Stderr, "fak chat: connected to %s\n", effectiveBaseURL)
		}
	}

	if *cf.effort != "" {
		runOpts = append(runOpts, agent.WithRunReasoningEffort(*cf.effort))
	}
	var catalog []agent.ToolDef
	hasCustomCatalog := false
	root := strings.TrimSpace(*cf.codeWorkspace)
	if root == "" {
		var err error
		root, err = os.Getwd()
		must(err)
	}
	root, err := canonicalNativeReceiptWorkspace(root)
	must(err)
	receiptTools := nativeAgentToolCapabilities{}
	useCodeTools := *cf.codeTools && *cf.tools != "demo" && *cf.tools != "none"
	if *cf.tools == "code" || useCodeTools {
		var extraDirs []string
		if *cf.skillsDir != "" {
			extraDirs = append(extraDirs, *cf.skillsDir)
		}
		var exactCommands []string
		if *cf.policyPath != "" {
			exactCommands = extractPolicyExactCommands(adjudicator.Default.PolicySnapshot())
		}
		codeCat, armErr := agent.ArmCodeToolsWithOptions(agent.CodeToolsOptions{
			Root:                 root,
			Focused:              true,
			EnableSkills:         *cf.skills,
			ExtraDirs:            extraDirs,
			ExactAllowedCommands: exactCommands,
		})
		must(armErr)
		defer agent.DisarmCodeTools()
		catalog = append(catalog, codeCat...)
		hasCustomCatalog = true
		receiptTools.Skills = *cf.skills
	}
	if *cf.sysTools && *cf.tools != "demo" && *cf.tools != "none" {
		sysCatalog, sysErr := agent.ArmSysTools(systools.Config{})
		must(sysErr)
		defer agent.DisarmSysTools()
		catalog = append(catalog, sysCatalog...)
		hasCustomCatalog = true
		receiptTools.System = true
	}
	if *cf.mcpTools && *cf.tools != "demo" && *cf.tools != "none" {
		mcpCatalog, mcpErr := agent.ArmMCPTools()
		must(mcpErr)
		defer agent.DisarmMCPTools()
		catalog = append(catalog, mcpCatalog...)
		hasCustomCatalog = true
		receiptTools.MCP = true
	}
	if hasCustomCatalog {
		runOpts = append(runOpts, agent.WithToolCatalog(catalog))
	} else if *cf.tools == "none" {
		runOpts = append(runOpts, agent.WithToolCatalog(nil))
	}
	if memOpt, _ := resolveAgentMemoryOption(*cf.memory, *cf.memoryStore, root); memOpt != nil {
		runOpts = append(runOpts, memOpt)
		receiptTools.Memory = true
	}
	if cf.reasoningProfile != nil && *cf.reasoningProfile != "" {
		runOpts = append(runOpts, agent.WithReasoningProfile(*cf.reasoningProfile))
	}

	planner := chatPlanner(*cf.offline, effectiveBaseURL, *cf.provider, *cf.model, *cf.apiKeyEnv, *cf.anthropicAuth, *cf.codexAuth)
	if *cf.codexAuth {
		httpPlanner, ok := planner.(*agent.HTTPPlanner)
		if !ok {
			must(fmt.Errorf("fak chat: --codex-auth requires the native HTTP planner"))
		}
		must(configureChatCodexSubscription(httpPlanner, *cf.codexHome))
	}
	if *cf.task != "" {
		receiptContext := nativeAgentReceiptContext{}
		if guardPosture, supported := nativeAgentPostureName(effectivePosture); supported {
			receiptContext.Enforcement = nativeAgentEnforcement{
				Schema:          nativeAgentEnforcementSchema,
				GuardPosture:    guardPosture,
				PolicyDigest:    policyDigest,
				WorkspaceDigest: opsRunDigest(root),
				Tools:           receiptTools,
			}
		}
		if err := runChatHeadlessWithContext(os.Stdout, planner, *cf.task, *cf.maxTurns, *cf.asJSON, *cf.receiptOut, root, receiptContext, runOpts...); err != nil {
			os.Exit(1)
		}
		return
	}
	runChatWithDisplay(os.Stdin, os.Stdout, planner, *cf.maxTurns, *cf.verbose, runOpts...)
}

func nativeAgentPostureName(posture adjudicator.Posture) (string, bool) {
	switch posture {
	case adjudicator.PostureFailClosed:
		return "fail_closed", true
	case adjudicator.PostureAdmitAndLog:
		return "admit_and_log", true
	case adjudicator.PostureDefaultOpen:
		return "default_open", true
	default:
		return "", false
	}
}

// chatPlanner picks the planner the REPL drives: the offline mock (no upstream)
// unless a --base-url is given, mirroring `fak agent` exactly so `fak chat`
// runs with zero network by default.
func chatPlanner(offline bool, baseURL, provider, model, apiKeyEnv, anthropicAuth string, codexAuth bool) agent.Planner {
	return chatPlannerWithStderr(os.Stderr, offline, baseURL, provider, model, apiKeyEnv, anthropicAuth, codexAuth)
}

func chatPlannerWithStderr(stderr io.Writer, offline bool, baseURL, provider, model, apiKeyEnv, anthropicAuth string, codexAuth bool) agent.Planner {
	if stderr == nil {
		stderr = os.Stderr
	}
	effectiveBaseURL := baseURL
	if offline || effectiveBaseURL == "" {
		if !offline {
			fmt.Fprintln(stderr, "fak chat: no --base-url given; using the offline mock planner (pass --base-url for a live run)")
		}
		return agent.NewMockPlanner(model)
	}
	var key string
	if codexAuth {
		fmt.Fprintln(stderr, "fak chat: auth mode: codex-auth")
	} else {
		if apiKeyEnv != "" {
			key = os.Getenv(apiKeyEnv)
		}
		if key == "" {
			fmt.Fprintf(stderr, "fak chat: env %s is empty  -  proceeding with no auth header (fine for a local endpoint)\n", apiKeyEnv)
		}
	}
	p, err := agent.NewProviderHTTPPlanner(provider, effectiveBaseURL, model, key)
	must(err)
	scheme, ok := agent.ParseAnthropicAuthScheme(anthropicAuth)
	if !ok {
		must(fmt.Errorf("--anthropic-auth %q: want auto, bearer, or x-api-key", anthropicAuth))
	}
	p.AnthropicAuthScheme = scheme
	return p
}

// runChatHeadless executes a single turn non-interactively (headless mode), printing
// any executed tool calls and the final answer directly to out, or outputting/writing
// a structured execution receipt if asJSON is true or receiptOut is non-empty.
func runChatHeadless(out io.Writer, planner agent.Planner, task string, maxTurns int, asJSON bool, receiptOut string, workspace string, opts ...agent.RunOption) error {
	return runChatHeadlessWithContext(out, planner, task, maxTurns, asJSON, receiptOut, workspace, nativeAgentReceiptContext{}, opts...)
}

func runChatHeadlessWithContext(out io.Writer, planner agent.Planner, task string, maxTurns int, asJSON bool, receiptOut string, workspace string, receiptContext nativeAgentReceiptContext, opts ...agent.RunOption) error {
	m, calls, err := agent.RunGovernedArm(ctx(), planner, task, maxTurns, opts...)
	if asJSON || receiptOut != "" {
		model := ""
		if planner != nil {
			model = planner.Model()
		}
		receipt := newHeadlessAgentReceiptWithContext(task, model, m, calls, workspace, err, receiptContext)
		data, marshalErr := json.MarshalIndent(receipt, "", "  ")
		if marshalErr != nil {
			return marshalErr
		}
		if receiptOut != "" {
			if dir := filepath.Dir(receiptOut); dir != "" && dir != "." {
				_ = os.MkdirAll(dir, 0o755)
			}
			if writeErr := os.WriteFile(receiptOut, append(data, '\n'), 0o644); writeErr != nil {
				return writeErr
			}
		}
		if asJSON {
			fmt.Fprintf(out, "%s\n", data)
			return err
		}
	}

	if err != nil {
		renderChatTermination(out, err)
		return err
	}
	for _, c := range calls {
		if c.Verdict == "ALLOW" {
			fmt.Fprintf(out, "[tool] %s(%s) => ALLOW\n", c.Tool, c.Args)
		} else {
			fmt.Fprintf(out, "[tool] %s(%s) => %s (%s by %s)\n", c.Tool, c.Args, c.Verdict, c.Reason, c.By)
		}
	}
	fmt.Fprintf(out, "%s\n", strings.TrimSpace(m.FinalAnswer))
	return nil
}

// runChat is the REPL core, factored from cmdChat so an e2e test can script turns
// over an in-memory reader/writer with a deterministic planner. Each non-blank
// input line is ONE human turn driven through agent.RunGovernedArm with fak=true — the
// kernel mediates every tool call, so a denied destructive call is returned to the
// model as a value (recorded in ArmMetrics.Denies) and never executed
// (DestructiveExecuted stays false). The per-turn summary surfaces that boundary.
func runChat(in io.Reader, out io.Writer, planner agent.Planner, maxTurns int, opts ...agent.RunOption) {
	runChatWithDisplay(in, out, planner, maxTurns, false, opts...)
}

func runChatWithDisplay(in io.Reader, out io.Writer, planner agent.Planner, maxTurns int, verbose bool, opts ...agent.RunOption) {
	fmt.Fprintf(out, "fak chat | %s\n", chatModelLabel(planner))
	fmt.Fprintln(out, "Type a message. /help for commands; Ctrl-D or /exit to quit.")
	sc := bufio.NewScanner(in)
	sc.Buffer(make([]byte, 0, 64*1024), 1<<20)
	turn := 0
	var history []agent.Message
	for {
		fmt.Fprint(out, "you> ")
		if !sc.Scan() {
			break
		}
		line := strings.TrimSpace(sc.Text())
		if line == "" {
			continue
		}
		if line == "/exit" || line == "/quit" {
			break
		}
		if line == "/clear" || line == "/reset" {
			history = nil
			fmt.Fprintln(out, "fak> conversation cleared.")
			continue
		}
		if line == "/help" {
			renderChatHelp(out)
			continue
		}
		if line == "/status" {
			detail := "hidden"
			if verbose {
				detail = "visible"
			}
			fmt.Fprintf(out, "fak> Model: %s; tool details: %s; turn limit: %d steps.\n", chatModelLabel(planner), detail, maxTurns)
			continue
		}
		if line == "/verbose" || strings.HasPrefix(line, "/verbose ") {
			var message string
			verbose, message = updateChatVerbose(line, verbose)
			fmt.Fprintln(out, message)
			continue
		}
		turn++

		currentConv := append(history, agent.Message{Role: agent.RoleUser, Content: line})
		turnOpts := append([]agent.RunOption{agent.WithConversation(currentConv)}, opts...)

		stream := &chatStreamState{out: out}
		m, calls, err := agent.RunGovernedArmStream(ctx(), planner, line, maxTurns, stream.write, turnOpts...)
		stream.finish()
		if err != nil {
			renderInteractiveChatTermination(out, err, verbose)
			renderInteractiveChatActivity(out, calls, m, turn, verbose, true)
			continue
		}

		finalAnswer := strings.TrimSpace(m.FinalAnswer)
		if !stream.wrote && finalAnswer != "" {
			fmt.Fprintf(out, "fak> %s\n", finalAnswer)
		}
		if m.HitTurnCap {
			renderChatTurnLimit(out, maxTurns, stream.wrote || finalAnswer != "")
		} else if !stream.wrote && finalAnswer == "" {
			fmt.Fprintln(out, "fak> I could not produce an answer. Try again, or use /verbose for details.")
		}
		renderInteractiveChatActivity(out, calls, m, turn, verbose, false)

		history = currentConv
		if finalAnswer != "" {
			history = append(history, agent.Message{Role: agent.RoleAssistant, Content: m.FinalAnswer})
		}
	}
}

func chatModelLabel(planner agent.Planner) string {
	if planner == nil {
		return "model unavailable"
	}
	model := strings.TrimSpace(planner.Model())
	if model == "" {
		return "model unavailable"
	}
	model = strings.ReplaceAll(model, "\\", "/")
	if i := strings.LastIndex(model, "/"); i >= 0 {
		model = model[i+1:]
	}
	if len(model) > len(".gguf") && strings.EqualFold(model[len(model)-len(".gguf"):], ".gguf") {
		model = model[:len(model)-len(".gguf")]
	}
	if model == "" {
		return "model unavailable"
	}
	return model
}

func renderChatHelp(out io.Writer) {
	fmt.Fprintln(out, "fak> Commands:")
	fmt.Fprintln(out, "     /clear             clear conversation history")
	fmt.Fprintln(out, "     /status            show the model and detail level")
	fmt.Fprintln(out, "     /verbose [on|off]  show or hide tool details")
	fmt.Fprintln(out, "     /exit              leave chat")
}

func updateChatVerbose(line string, current bool) (bool, string) {
	fields := strings.Fields(line)
	if len(fields) == 1 {
		current = !current
	} else if len(fields) == 2 {
		switch strings.ToLower(fields[1]) {
		case "on":
			current = true
		case "off":
			current = false
		default:
			return current, "fak> Usage: /verbose [on|off]"
		}
	} else {
		return current, "fak> Usage: /verbose [on|off]"
	}
	state := "off"
	if current {
		state = "on"
	}
	return current, fmt.Sprintf("fak> Tool details are %s.", state)
}

type chatStreamState struct {
	out     io.Writer
	started bool
	wrote   bool
}

func (s *chatStreamState) write(delta string) error {
	if s == nil || s.out == nil || delta == "" {
		return nil
	}
	if !s.started {
		if _, err := io.WriteString(s.out, "fak> "); err != nil {
			return err
		}
		s.started = true
	}
	if _, err := io.WriteString(s.out, delta); err != nil {
		return err
	}
	s.wrote = true
	return nil
}

func (s *chatStreamState) finish() {
	if s != nil && s.out != nil && s.started {
		fmt.Fprintln(s.out)
	}
}

func renderChatTurnLimit(out io.Writer, maxTurns int, hasAnswer bool) {
	prefix := "fak> "
	if hasAnswer {
		prefix = "     "
	}
	fmt.Fprintf(out, "%sThis response is incomplete: I reached this turn's %d-step limit. Try a narrower request or raise --max-turns.\n", prefix, maxTurns)
}

func renderInteractiveChatTermination(out io.Writer, err error, verbose bool) {
	t := agent.ClassifyTermination(err)
	message := "I could not finish this turn. Try again, or use /verbose for details."
	switch t.Cause {
	case agent.TerminationCanceled:
		message = "The request stopped before I could answer. Try again."
	case agent.TerminationRateLimited:
		message = "The model is rate-limited. Wait a moment and try again."
	case agent.TerminationContextLimit:
		message = "This conversation is too long for the model. Use /clear, then try again."
	case agent.TerminationRefused:
		message = "fak blocked this turn. Rephrase the request or use /verbose for details."
	case agent.TerminationProvider:
		message = "I lost the model connection before it answered. This chat is still open; try again."
	}
	fmt.Fprintf(out, "fak> %s [%s]\n", message, t.Cause)
	if verbose {
		fmt.Fprintf(out, "     %s\n", t.Evidence)
	}
}

func renderInteractiveChatActivity(out io.Writer, calls []agent.CallTrace, m agent.ArmMetrics, turn int, verbose, interrupted bool) {
	if verbose {
		for _, c := range calls {
			if c.Verdict == "ALLOW" {
				fmt.Fprintf(out, "     [tool] %s(%s) => ALLOW\n", c.Tool, c.Args)
			} else {
				fmt.Fprintf(out, "     [tool] %s(%s) => %s (%s by %s)\n", c.Tool, c.Args, c.Verdict, c.Reason, c.By)
			}
		}
		fmt.Fprintf(out, "     [turn %d: %d model turns, %d engine calls, %d denied, %d served]\n",
			turn, m.Turns, m.EngineCalls, m.Denies, m.VDSOHits)
		return
	}
	if len(calls) == 0 {
		return
	}
	if len(calls) == 1 {
		verdict := strings.ToUpper(strings.TrimSpace(calls[0].Verdict))
		if verdict == "" {
			verdict = "UNKNOWN"
		}
		fmt.Fprintf(out, "     [tool] %s => %s", calls[0].Tool, verdict)
		if m.Denies > 0 {
			fmt.Fprintf(out, " (%d denied)", m.Denies)
		}
		if interrupted {
			fmt.Fprint(out, " before interruption")
		}
		fmt.Fprintln(out)
		return
	}

	const maxNames = 3
	names := make([]string, 0, maxNames)
	seen := make(map[string]struct{}, maxNames)
	for _, c := range calls {
		if _, ok := seen[c.Tool]; ok {
			continue
		}
		seen[c.Tool] = struct{}{}
		if len(names) < maxNames {
			names = append(names, c.Tool)
		}
	}
	fmt.Fprintf(out, "     [tools] %d actions: %s", len(calls), strings.Join(names, ", "))
	if len(seen) > len(names) {
		fmt.Fprintf(out, " (+%d more)", len(seen)-len(names))
	}
	if m.Denies > 0 {
		fmt.Fprintf(out, "; %d denied", m.Denies)
	}
	if interrupted {
		fmt.Fprint(out, " before interruption")
	}
	fmt.Fprintln(out)
}

func renderChatTermination(out io.Writer, err error) {
	t := agent.ClassifyTermination(err)
	fmt.Fprintf(out, "fak> turn terminated [%s]: %s\n", t.Cause, t.Evidence)
}

func parseChatMode(s string) adjudicator.Posture {
	switch strings.ToLower(strings.TrimSpace(s)) {
	case "default_open":
		return adjudicator.PostureDefaultOpen
	case "admit_and_log":
		return adjudicator.PostureAdmitAndLog
	case "fail_closed", "strict", "":
		return adjudicator.PostureFailClosed
	default:
		p, err := policy.ParsePosture(s)
		if err == nil {
			return p
		}
		return adjudicator.PostureFailClosed
	}
}

func initDevRules(mode string) {
	rt, err := policy.ParseRuntime(guardDefaultPolicyJSON)
	must(err)
	effMode := parseChatMode(mode)
	rt.Adjudicator.Posture = effMode
	rt.PolicyContext.Posture = effMode
	agent.SetConfiguredPosture(effMode)
	digest := guardPolicyDigest(guardDefaultPolicyJSON)
	policyReloadMu.Lock()
	defer policyReloadMu.Unlock()
	_, err = applyPolicyRuntimeLocked(rt, "embedded:developer", digest, "", false)
	must(err)
}

func extractPolicyExactCommands(p adjudicator.Policy) []string {
	if p.Posture != adjudicator.PostureFailClosed {
		return nil
	}
	if len(p.Complain) > 0 {
		return nil
	}
	var exacts []string
	for _, pred := range p.ArgPredicates {
		if pred.Advisory {
			continue
		}
		if p.AdvisoryReasons != nil && p.AdvisoryReasons[pred.Reason] {
			continue
		}
		if strings.EqualFold(pred.Tool, "bash") && pred.Arg == "command" && pred.Kind == adjudicator.ArgAllowExact && pred.Glob != "" {
			exacts = append(exacts, pred.Glob)
		}
	}
	return exacts
}

// probeGatewayOnce probes a single gateway origin: GET origin/healthz, decode
// the health payload, and fall back to origin/v1/models for model discovery.
func probeGatewayOnce(client *http.Client, origin string) (string, bool) {
	origin = strings.TrimRight(origin, "/")
	resp, err := client.Get(origin + "/healthz")
	if err != nil {
		return "", false
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return "", false
	}
	var health struct {
		OK    bool   `json:"ok"`
		Model string `json:"model"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&health); err != nil || !health.OK {
		return "", false
	}
	model := strings.TrimSpace(health.Model)
	if model == "" || model == "mock" {
		if discovered := probeServerModels(client, origin+"/v1/models"); discovered != "" {
			model = discovered
		}
	}
	return model, true
}

// probeLocalGateway probes a gateway origin, retrying once via a loopback
// family fallback (e.g. 127.0.0.1 -> localhost) when the first probe fails.
// On Windows the loopback may be reachable only on one address family, so an
// IPv4 literal can fail while "localhost" resolves to the live family.
func probeLocalGateway(addr string) (string, bool) {
	client := &http.Client{Timeout: 150 * time.Millisecond}
	if model, ok := probeGatewayOnce(client, addr); ok {
		return model, true
	}
	if fallback, ok := fakclient.LoopbackFallbackURL(addr); ok {
		if model, ok := probeGatewayOnce(client, fallback); ok {
			return model, true
		}
	}
	return "", false
}

func detectServerModel(baseURL string) string {
	client := &http.Client{Timeout: 250 * time.Millisecond}
	root := strings.TrimRight(baseURL, "/")
	root = strings.TrimSuffix(root, "/v1")
	if m, ok := probeLocalGateway(root); ok && m != "" && m != "mock" {
		return m
	}
	modelsURL := strings.TrimRight(baseURL, "/")
	if !strings.HasSuffix(modelsURL, "/models") {
		if strings.HasSuffix(modelsURL, "/v1") {
			modelsURL += "/models"
		} else {
			modelsURL += "/v1/models"
		}
	}
	return probeServerModels(client, modelsURL)
}

func probeServerModels(client *http.Client, modelsURL string) string {
	resp, err := client.Get(modelsURL)
	if err != nil || resp.StatusCode != http.StatusOK {
		if resp != nil && resp.Body != nil {
			_ = resp.Body.Close()
		}
		return ""
	}
	defer resp.Body.Close()
	var body struct {
		Data []struct {
			ID string `json:"id"`
		} `json:"data"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&body); err == nil && len(body.Data) > 0 {
		for _, m := range body.Data {
			id := strings.TrimSpace(m.ID)
			if id != "" && id != "mock" {
				return id
			}
		}
		return strings.TrimSpace(body.Data[0].ID)
	}
	return ""
}
