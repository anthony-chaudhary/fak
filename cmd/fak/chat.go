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
	asJSON                *bool
	receiptOut            *string
}

func newChatFlagSet() (*flag.FlagSet, *chatFlags) {
	fs := flag.NewFlagSet("chat", flag.ExitOnError)
	cf := &chatFlags{}
	cf.provider = fs.String("provider", "openai", "provider transcript wire: openai, anthropic, gemini, or xai")
	cf.baseURL = fs.String("base-url", "", "provider base URL (empty => offline mock planner; no upstream)")
	cf.model = fs.String("model", "gemini-2.5-flash", "model id")
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
	if *cf.policyPath != "" {
		applyPolicy(*cf.policyPath)
		effMode := parseChatMode(rawMode)
		agent.SetConfiguredPosture(effMode)
		snap := adjudicator.Default.PolicySnapshot()
		if postureExplicit {
			snap.Posture = effMode
		}
		runOpts = append(runOpts, agent.WithPolicySnapshot(snap))
	} else {
		initDevRules(rawMode)
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
			fmt.Fprintf(os.Stderr, "fak chat: auto-connected to local gateway at %s (model: %s)\n", effectiveBaseURL, *cf.model)
		}
	} else if effectiveBaseURL != "" && !modelExplicit && !*cf.offline {
		if serverModel := detectServerModel(effectiveBaseURL); serverModel != "" && serverModel != "mock" {
			*cf.model = serverModel
			fmt.Fprintf(os.Stderr, "fak chat: auto-detected model %q from %s\n", *cf.model, effectiveBaseURL)
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
	}
	if *cf.sysTools && *cf.tools != "demo" && *cf.tools != "none" {
		sysCatalog, sysErr := agent.ArmSysTools(systools.Config{})
		must(sysErr)
		defer agent.DisarmSysTools()
		catalog = append(catalog, sysCatalog...)
		hasCustomCatalog = true
	}
	if *cf.mcpTools && *cf.tools != "demo" && *cf.tools != "none" {
		mcpCatalog, mcpErr := agent.ArmMCPTools()
		must(mcpErr)
		defer agent.DisarmMCPTools()
		catalog = append(catalog, mcpCatalog...)
		hasCustomCatalog = true
	}
	if hasCustomCatalog {
		runOpts = append(runOpts, agent.WithToolCatalog(catalog))
	} else if *cf.tools == "none" {
		runOpts = append(runOpts, agent.WithToolCatalog(nil))
	}
	if memOpt, _ := resolveAgentMemoryOption(*cf.memory, *cf.memoryStore, root); memOpt != nil {
		runOpts = append(runOpts, memOpt)
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
		if err := runChatHeadless(os.Stdout, planner, *cf.task, *cf.maxTurns, *cf.asJSON, *cf.receiptOut, root, runOpts...); err != nil {
			os.Exit(1)
		}
		return
	}
	runChat(os.Stdin, os.Stdout, planner, *cf.maxTurns, runOpts...)
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
	m, calls, err := agent.RunGovernedArm(ctx(), planner, task, maxTurns, opts...)
	if asJSON || receiptOut != "" {
		model := ""
		if planner != nil {
			model = planner.Model()
		}
		receipt := newHeadlessAgentReceipt(task, model, m, calls, workspace, err)
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
	fmt.Fprintf(out, "fak chat — native REPL on the owned loop (model %s). One line = one turn; Ctrl-D to exit.\n", planner.Model())
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
		turn++

		currentConv := append(history, agent.Message{Role: agent.RoleUser, Content: line})
		turnOpts := append([]agent.RunOption{agent.WithConversation(currentConv)}, opts...)

		m, calls, err := agent.RunGovernedArm(ctx(), planner, line, maxTurns, turnOpts...)
		if err != nil {
			renderChatTermination(out, err)
			continue
		}
		for _, c := range calls {
			if c.Verdict == "ALLOW" {
				fmt.Fprintf(out, "     [tool] %s(%s) => ALLOW\n", c.Tool, c.Args)
			} else {
				fmt.Fprintf(out, "     [tool] %s(%s) => %s (%s by %s)\n", c.Tool, c.Args, c.Verdict, c.Reason, c.By)
			}
		}
		fmt.Fprintf(out, "fak> %s\n", strings.TrimSpace(m.FinalAnswer))
		fmt.Fprintf(out, "     [turn %d: %d model turns, %d engine calls, %d denied, %d served]\n",
			turn, m.Turns, m.EngineCalls, m.Denies, m.VDSOHits)

		history = append(currentConv, agent.Message{Role: agent.RoleAssistant, Content: m.FinalAnswer})
	}
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

func probeLocalGateway(addr string) (string, bool) {
	client := &http.Client{Timeout: 150 * time.Millisecond}
	resp, err := client.Get(strings.TrimRight(addr, "/") + "/healthz")
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
		if discovered := probeServerModels(client, strings.TrimRight(addr, "/")+"/v1/models"); discovered != "" {
			model = discovered
		}
	}
	return model, true
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
