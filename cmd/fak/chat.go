package main

import (
	"bufio"
	"flag"
	"fmt"
	"io"
	"os"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/dropin"
)

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
	fs := flag.NewFlagSet("chat", flag.ExitOnError)
	provider := fs.String("provider", "openai", "provider transcript wire: openai, anthropic, gemini, or xai")
	baseURL := fs.String("base-url", "", "provider base URL (empty => offline mock planner; no upstream)")
	model := fs.String("model", "gemini-2.5-flash", "model id")
	apiKeyEnv := fs.String("api-key-env", "GEMINI_API_KEY", "env var holding the API key")
	anthropicAuth := fs.String("anthropic-auth", "auto", "(--provider anthropic) how to present the credential: auto (sniff the token shape), bearer, or x-api-key. Pass bearer for a THIRD-PARTY Anthropic-compatible endpoint whose tenant token is not an sk-ant-* key")
	offline := fs.Bool("offline", false, "force the deterministic mock planner (no network)")
	maxTurns := fs.Int("max-turns", 10, "max model turns the loop may take to resolve ONE human turn")
	policyPath := fs.String("policy", "", "load the capability floor from a manifest (default: the built-in adjudicator floor)")
	task := fs.String("task", "", "run a single non-interactive task turn (headless mode) and exit")
	tools := fs.String("tools", "code", "toolset to arm: code (Read/Write/Edit/Bash/Grep/Glob), demo (airline fixture), or none")
	codeTools := fs.Bool("code-tools", true, "arm bounded kernel Read/Write/Edit/Bash/Grep/Glob in the workspace (alias for --tools=code)")
	codeWorkspace := fs.String("code-workspace", "", "override workspace root for code tools (default: current directory)")
	skills := fs.Bool("skills", true, "enable Agent Skills discovery and dynamic faulting")
	skillsDir := fs.String("skills-dir", "", "optional custom directory to search for SKILL.md definitions")
	_ = fs.Parse(argv)

	applyPolicy(*policyPath)

	providerExplicit := false
	fs.Visit(func(f *flag.Flag) {
		if f.Name == "provider" {
			providerExplicit = true
		}
	})
	effectiveBaseURL := *baseURL
	if effectiveBaseURL == "" && providerExplicit && !*offline {
		effectiveBaseURL = dropin.DefaultBaseURL(*provider)
	}

	var runOpts []agent.RunOption
	useCodeTools := *codeTools && *tools != "demo" && *tools != "none"
	if *tools == "code" || useCodeTools {
		root := strings.TrimSpace(*codeWorkspace)
		if root == "" {
			var err error
			root, err = os.Getwd()
			must(err)
		}
		var extraDirs []string
		if *skillsDir != "" {
			extraDirs = append(extraDirs, *skillsDir)
		}
		catalog, armErr := agent.ArmCodeToolsWithOptions(agent.CodeToolsOptions{
			Root:         root,
			Focused:      true,
			EnableSkills: *skills,
			ExtraDirs:    extraDirs,
		})
		must(armErr)
		defer agent.DisarmCodeTools()
		runOpts = append(runOpts, agent.WithToolCatalog(catalog))
	} else if *tools == "none" {
		runOpts = append(runOpts, agent.WithToolCatalog(nil))
	}

	planner := chatPlanner(*offline, effectiveBaseURL, *provider, *model, *apiKeyEnv, *anthropicAuth)
	if *task != "" {
		if err := runChatHeadless(os.Stdout, planner, *task, *maxTurns, runOpts...); err != nil {
			os.Exit(1)
		}
		return
	}
	runChat(os.Stdin, os.Stdout, planner, *maxTurns, runOpts...)
}

// chatPlanner picks the planner the REPL drives: the offline mock (no upstream)
// unless a --base-url is given, mirroring `fak agent` exactly so `fak chat`
// runs with zero network by default.
func chatPlanner(offline bool, baseURL, provider, model, apiKeyEnv, anthropicAuth string) agent.Planner {
	effectiveBaseURL := baseURL
	if effectiveBaseURL == "" && !offline {
		if u := dropin.DefaultBaseURL(provider); u != "" {
			effectiveBaseURL = u
		}
	}
	if offline || effectiveBaseURL == "" {
		if !offline {
			fmt.Fprintln(os.Stderr, "fak chat: no --base-url given; using the offline mock planner (pass --base-url for a live run)")
		}
		return agent.NewMockPlanner(model)
	}
	key := os.Getenv(apiKeyEnv)
	if key == "" {
		fmt.Fprintf(os.Stderr, "fak chat: env %s is empty  -  proceeding with no auth header (fine for a local endpoint)\n", apiKeyEnv)
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
// any executed tool calls and the final answer directly to out.
func runChatHeadless(out io.Writer, planner agent.Planner, task string, maxTurns int, opts ...agent.RunOption) error {
	m, calls, err := agent.RunGovernedArm(ctx(), planner, task, maxTurns, opts...)
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
