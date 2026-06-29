package main

import (
	"bufio"
	"flag"
	"fmt"
	"io"
	"os"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/agent"
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
	offline := fs.Bool("offline", false, "force the deterministic mock planner (no network)")
	maxTurns := fs.Int("max-turns", 10, "max model turns the loop may take to resolve ONE human turn")
	policyPath := fs.String("policy", "", "load the capability floor from a manifest (default: the built-in adjudicator floor)")
	_ = fs.Parse(argv)
	applyPolicy(*policyPath)

	planner := chatPlanner(*offline, *baseURL, *provider, *model, *apiKeyEnv)
	runChat(os.Stdin, os.Stdout, planner, *maxTurns)
}

// chatPlanner picks the planner the REPL drives: the offline mock (no upstream)
// unless a --base-url is given, mirroring `fak agent` exactly so `fak chat`
// runs with zero network by default.
func chatPlanner(offline bool, baseURL, provider, model, apiKeyEnv string) agent.Planner {
	if offline || baseURL == "" {
		if !offline {
			fmt.Fprintln(os.Stderr, "fak chat: no --base-url given; using the offline mock planner (pass --base-url for a live run)")
		}
		return agent.NewMockPlanner(model)
	}
	key := os.Getenv(apiKeyEnv)
	if key == "" {
		fmt.Fprintf(os.Stderr, "fak chat: env %s is empty  -  proceeding with no auth header (fine for a local endpoint)\n", apiKeyEnv)
	}
	p, err := agent.NewProviderHTTPPlanner(provider, baseURL, model, key)
	must(err)
	return p
}

// runChat is the REPL core, factored from cmdChat so an e2e test can script turns
// over an in-memory reader/writer with a deterministic planner. Each non-blank
// input line is ONE human turn driven through agent.RunArm with fak=true — the
// kernel mediates every tool call, so a denied destructive call is returned to the
// model as a value (recorded in ArmMetrics.Denies) and never executed
// (DestructiveExecuted stays false). The per-turn summary surfaces that boundary.
func runChat(in io.Reader, out io.Writer, planner agent.Planner, maxTurns int) {
	fmt.Fprintf(out, "fak chat — native REPL on the owned loop (model %s). One line = one turn; Ctrl-D to exit.\n", planner.Model())
	sc := bufio.NewScanner(in)
	sc.Buffer(make([]byte, 0, 64*1024), 1<<20)
	turn := 0
	for {
		fmt.Fprint(out, "you> ")
		if !sc.Scan() {
			break
		}
		line := strings.TrimSpace(sc.Text())
		if line == "" {
			continue
		}
		turn++
		// The per-turn ArmMetrics is the whole summary this REPL needs (turns,
		// engine calls, denies); the verbose trace log is for the A/B report path,
		// so pass nil and let RunArm skip it.
		m, err := agent.RunArm(ctx(), planner, line, true, maxTurns, nil)
		if err != nil {
			fmt.Fprintf(out, "fak> turn failed: %v\n", err)
			continue
		}
		fmt.Fprintf(out, "fak> %s\n", strings.TrimSpace(m.FinalAnswer))
		fmt.Fprintf(out, "     [turn %d: %d model turns, %d engine calls, %d denied, %d served]\n",
			turn, m.Turns, m.EngineCalls, m.Denies, m.VDSOHits)
	}
}
