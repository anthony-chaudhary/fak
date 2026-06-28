package main

import (
	"bufio"
	"context"
	"flag"
	"fmt"
	"os"
	"os/signal"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/cacheobs"
	"github.com/anthony-chaudhary/fak/internal/cachevalueledger"
	"github.com/anthony-chaudhary/fak/internal/hfhub"
	"github.com/anthony-chaudhary/fak/internal/modelreg"
	"github.com/anthony-chaudhary/fak/internal/pathutil"
)

// runChatModel is the `fak run <model> [prompt]` chat path — the daemon-less,
// Ollama-style one-shot/REPL surface. It loads a model directly into fak's
// in-kernel engine (the SAME loaders `fak serve --gguf` uses) and runs a chat
// completion with no HTTP gateway and no provider:
//
//	fak run smollm2 "explain mmap in one line"   # one-shot: print the answer, exit
//	fak run smollm2                              # REPL: read a line, answer, repeat
//
// The model ref is alias-aware (`smollm2` → its hf:// target), an hf:// URI is
// downloaded on demand, and a local .gguf path loads directly. This is a plain
// chat with no tools, so there is no tool-call adjudication in the loop — the value
// here is the in-kernel engine (prefix reuse, quantized resident decode) in one
// static binary with no server to stand up.
func runChatModel(argv []string) {
	fs := flag.NewFlagSet("run", flag.ExitOnError)
	fs.Usage = func() {
		fmt.Fprintln(os.Stderr, "usage: fak run <model> [prompt]")
		fmt.Fprintln(os.Stderr, "  <model>  a model alias (see `fak ls`), an hf://owner/repo/file.gguf URI, or a .gguf path")
		fmt.Fprintln(os.Stderr, "  [prompt] one-shot prompt; omit it for an interactive REPL")
		fmt.Fprintln(os.Stderr, "  (this is `fak run` CHAT mode; `fak run --trace FILE` / `fak replay` is the trace replayer)")
		fs.PrintDefaults()
	}
	backendName := fs.String("backend", "", "compute backend for decode: empty = the CPU reference path; a registered device like 'cuda' runs through the GPU HAL (needs a -tags cuda build + a reachable GPU)")
	system := fs.String("system", "", "optional system prompt prepended to the conversation")
	maxTokens := fs.Int("max-tokens", 512, "maximum number of tokens to generate per turn")
	temp := fs.Float64("temp", 0, "sampling temperature (0 = greedy/deterministic)")
	topP := fs.Float64("top-p", 0, "nucleus-sampling cutoff (0 = off)")
	topK := fs.Int("top-k", 0, "top-k truncation (0 = full distribution)")
	// fak's core value-add — KV-prefix reuse — is invisible by default everywhere else
	// (#333: only --debug-stats or /metrics show it). On the daemon-less `fak run` front
	// door we print a one-line WITNESSED cache-value summary per turn to STDERR (never
	// stdout, so the model answer stays pipe-clean) so a developer SEES the prefix the
	// kernel served from cache instead of recomputed. --quiet silences it for scripting.
	quiet := fs.Bool("quiet", false, "suppress the per-turn cache-value summary line on stderr (the kernel's WITNESSED KV-prefix reuse)")

	// argv[0] is the model ref (a non-flag, guaranteed by the cmdRun dispatch); the
	// rest are flags and/or the prompt words. Parse the flags out of argv[1:].
	modelRef := argv[0]
	if err := fs.Parse(argv[1:]); err != nil {
		os.Exit(2)
	}
	prompt := strings.TrimSpace(strings.Join(fs.Args(), " "))

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()

	planner := buildRunPlanner(ctx, modelRef, *backendName)

	opts := runSampleOpts(*maxTokens, *temp, *topP, *topK)
	if prompt != "" {
		// One-shot: answer and exit.
		runChatTurn(ctx, planner, *system, nil, prompt, opts, !*quiet)
	} else {
		// REPL: interactive session.
		runChatREPL(ctx, planner, *system, opts, !*quiet)
	}
	// Append cache-value observation to ledger (epic #1072, issue #1075).
	stats := cacheobs.Default.Snapshot()
	if stats.Turns > 0 {
		_ = cachevalueledger.Append("run", modelRef, cachevalueledger.DefaultLedgerRel, stats)
	}
}

// cacheValueLine renders one WITNESSED cache-value line from the per-turn delta of the
// process-global cacheobs tap (the SAME realized KV-prefix reuse the gateway scrapes onto
// /metrics). before/after bracket exactly one served turn, so the delta is this turn's
// prompt tokens and the prefix of them the kernel served from its cached KV — fak's own
// measurement (WITNESSED), not a provider's reported counter. It returns "" for an
// idle/empty turn (no prompt delta) so the caller prints nothing.
func cacheValueLine(before, after cacheobs.Stats) string {
	prompt := int64(after.PromptTokens) - int64(before.PromptTokens)
	reused := int64(after.ReusedTokens) - int64(before.ReusedTokens)
	if prompt <= 0 {
		return ""
	}
	if reused < 0 {
		reused = 0
	}
	ratio := float64(reused) / float64(prompt)
	regime := "cold"
	switch {
	case ratio >= cacheobs.FrozenFloor:
		regime = "frozen"
	case ratio >= cacheobs.ColdCeil:
		regime = "partial"
	}
	// by=vdso names the mechanism that served the reuse (the in-kernel RadixAttention
	// KV-prefix cache, fak's vDSO fast path) — closing #333's missing attribution on the
	// run surface. prompt−reused is the suffix the kernel actually recomputed this turn.
	return fmt.Sprintf("  cache: reused %d/%d prompt tok (%.0f%% %s, by=vdso) — computed %d",
		reused, prompt, ratio*100, regime, prompt-reused)
}

// cacheTurnLine is the show-gate over cacheValueLine: it returns the per-turn cache
// line to print, or "" when the caller asked to suppress it (--quiet => show=false) or
// the turn was idle. Splitting this out of runChatTurn lets the showCache gate be tested
// without standing up a planner — the #333 wire is a no-op unless this returns non-empty.
func cacheTurnLine(before, after cacheobs.Stats, show bool) string {
	if !show {
		return ""
	}
	return cacheValueLine(before, after)
}

// buildRunPlanner resolves the model ref, loads the weights + tokenizer through the
// shared serve loaders, and returns a ready in-kernel planner. It exits the process
// with a clear message on any load failure — there is no daemon to keep alive.
func buildRunPlanner(ctx context.Context, modelRef, backendName string) *agent.InKernelPlanner {
	ref, expanded := modelreg.Resolve(modelRef)
	if expanded {
		fmt.Fprintf(os.Stderr, "fak run: %s → %s\n", modelRef, ref)
	}
	ref = pathutil.ExpandTilde(ref)
	if hfhub.IsURI(ref) {
		resolved, err := hfhub.FetchURI(ctx, ref, os.Stderr)
		if err != nil {
			fmt.Fprintf(os.Stderr, "fak run: %v\n", err)
			os.Exit(1)
		}
		ref = resolved
	}
	if _, err := os.Stat(ref); err != nil {
		fmt.Fprintf(os.Stderr, "fak run: model %q is not a known alias, an hf:// URI, or an existing .gguf path\n", modelRef)
		os.Exit(2)
	}

	backend, err := resolveServeChatBackend(backendName)
	if err != nil {
		fmt.Fprintf(os.Stderr, "fak run: %v\n", err)
		os.Exit(2)
	}
	model, q4k, _, _ := loadServeInKernelModel(ref, backend, false, 0)
	if model == nil {
		fmt.Fprintf(os.Stderr, "fak run: failed to load %q into the in-kernel engine\n", ref)
		os.Exit(1)
	}
	tok, ok := resolveServeTokenizer("", ref)
	if !ok || tok == nil {
		fmt.Fprintf(os.Stderr, "fak run: %q has no usable tokenizer; pass a GGUF with an embedded tokenizer\n", ref)
		os.Exit(1)
	}
	// metal=false: `fak run`'s first cut targets the CPU reference path and the cuda
	// HAL; the Apple-Metal session forward is reachable through `fak serve --metal`.
	return agent.NewInKernelPlanner(model, tok, modelRef, q4k, backend, false)
}

// runSampleOpts folds the CLI sampling flags into planner SampleOpts. Temperature /
// top-p / top-k take pointers so "unset" (the zero default) stays a no-op rather than
// forcing greedy — only a value the user actually passed reaches the sampler.
func runSampleOpts(maxTokens int, temp, topP float64, topK int) []agent.SampleOpt {
	opts := []agent.SampleOpt{agent.WithMaxTokens(maxTokens)}
	if temp > 0 {
		t := temp
		opts = append(opts, agent.WithTemperature(&t))
	}
	if topP > 0 {
		p := topP
		opts = append(opts, agent.WithTopP(&p))
	}
	if topK > 0 {
		k := topK
		opts = append(opts, agent.WithTopK(&k))
	}
	return opts
}

// runChatTurn runs one completion and prints the assistant text to stdout. history
// is the prior conversation (nil for a one-shot); it returns the appended messages so
// the REPL can thread context across turns. When showCache is set, it brackets the
// turn with a cacheobs snapshot and prints the WITNESSED KV-prefix reuse line to
// stderr (the #333 value-add the run surface advertises) — stdout stays pipe-clean.
func runChatTurn(ctx context.Context, planner *agent.InKernelPlanner, system string, history []agent.Message, prompt string, opts []agent.SampleOpt, showCache bool) []agent.Message {
	msgs := history
	if len(msgs) == 0 && strings.TrimSpace(system) != "" {
		msgs = append(msgs, agent.Message{Role: "system", Content: system})
	}
	msgs = append(msgs, agent.Message{Role: "user", Content: prompt})

	// Bracket exactly this turn with the process-global cacheobs tap so the delta is
	// THIS turn's prompt tokens and the prefix the kernel served from its cached KV.
	// The planner feeds cacheobs.Default.Observe inside Complete; capturing before/after
	// here is what makes the per-turn line real rather than cumulative.
	before := cacheobs.Default.Snapshot()
	comp, err := planner.Complete(ctx, msgs, nil, opts...)
	if err != nil {
		fmt.Fprintf(os.Stderr, "fak run: %v\n", err)
		os.Exit(1)
	}
	out := comp.Message.Content
	fmt.Println(strings.TrimSpace(out))
	// The cache-value summary goes to STDERR (never stdout) so a `fak run ... | …` pipe
	// stays clean; --quiet (showCache=false) silences it for scripting. An idle/empty
	// turn (no prompt delta) renders "" and prints nothing.
	if line := cacheTurnLine(before, cacheobs.Default.Snapshot(), showCache); line != "" {
		fmt.Fprintln(os.Stderr, line)
	}
	return append(msgs, comp.Message)
}

// runChatREPL is the interactive mode: read a line, answer, repeat. EOF (Ctrl-D) or
// an interrupt ends it. Conversation context is threaded across turns.
func runChatREPL(ctx context.Context, planner *agent.InKernelPlanner, system string, opts []agent.SampleOpt, showCache bool) {
	fmt.Fprintf(os.Stderr, "fak run: interactive chat on %q — Ctrl-D to exit\n", planner.Model())
	var history []agent.Message
	sc := bufio.NewScanner(os.Stdin)
	sc.Buffer(make([]byte, 0, 64*1024), 4*1024*1024)
	for {
		fmt.Fprint(os.Stderr, ">>> ")
		if !sc.Scan() {
			fmt.Fprintln(os.Stderr)
			return
		}
		line := strings.TrimSpace(sc.Text())
		if line == "" {
			continue
		}
		history = runChatTurn(ctx, planner, system, history, line, opts, showCache)
	}
}
