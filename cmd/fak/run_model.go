package main

import (
	"bufio"
	"context"
	"errors"
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

type runAction int

const (
	runActionUsage runAction = iota // bare / missing required args
	runActionHelp                   // -h or --help
	runActionTrace                  // trace replay
	runActionChat                   // in-kernel model chat / REPL
)

type runFlags struct {
	// Chat / model controls
	backendName      *string
	metal            *bool
	system           *string
	maxTokens        *int
	temp             *float64
	topP             *float64
	topK             *int
	frequencyPenalty *float64
	presencePenalty  *float64
	effort           *string
	thinkingBudget   *int
	quiet            *bool
	nativeFlags      nativeControlFlags

	// Trace controls
	trace      *string
	engineID   *string
	vdso       *bool
	policyPath *string
}

type chatModelConfig struct {
	modelRef         string
	prompt           string
	backendName      string
	metal            bool
	system           string
	maxTokens        int
	temp             float64
	topP             float64
	topK             int
	frequencyPenalty float64
	presencePenalty  float64
	effort           string
	thinkingBudget   int
	quiet            bool
	nativeFlags      nativeControlFlags
}

type parsedRunCommand struct {
	action     runAction
	errMessage string
	modelRef   string
	prompt     string
	chatConfig chatModelConfig
	tracePath  string
	engineID   string
	vdso       bool
	policyPath string
}

func newRunFlagSet(name string, errorHandling flag.ErrorHandling) (*flag.FlagSet, *runFlags) {
	fs := flag.NewFlagSet(name, errorHandling)
	verbFlagUsage(fs, "run")
	flags := &runFlags{}

	// Chat / model controls
	flags.backendName = fs.String("backend", "", "compute backend for decode: empty = the CPU reference path; a registered device like 'cuda' runs through the GPU HAL (needs a -tags cuda build + a reachable GPU)")
	flags.metal = fs.Bool("metal", false, "run the in-kernel chat through the Apple-Silicon Metal GPU forward (auto-selected on darwin/arm64 with a usable Metal device; mutually exclusive with --backend)")
	flags.nativeFlags = registerRunNativeControlFlags(fs)
	flags.system = fs.String("system", "", "optional system prompt prepended to the conversation")
	flags.maxTokens = fs.Int("max-tokens", 512, "maximum number of tokens to generate per turn")
	flags.temp = fs.Float64("temp", 0, "sampling temperature (0 = greedy/deterministic)")
	flags.topP = fs.Float64("top-p", 0, "nucleus-sampling cutoff (0 = off)")
	flags.topK = fs.Int("top-k", 0, "top-k truncation (0 = full distribution)")
	flags.frequencyPenalty = fs.Float64("frequency-penalty", 0, "penalize tokens in proportion to their generated count (0 = off)")
	flags.presencePenalty = fs.Float64("presence-penalty", 0, "penalize tokens already generated this turn (0 = off)")
	flags.effort = fs.String("effort", "", "reasoning effort for model inference: none|low|medium|balanced|adaptive|high")
	flags.thinkingBudget = fs.Int("thinking-budget", -1, "explicit thinking token budget ceiling (>=0 overrides --effort; 0 disables thinking)")
	flags.quiet = fs.Bool("quiet", false, "suppress the per-turn cache-value summary line on stderr (the kernel's WITNESSED KV-prefix reuse)")

	// Trace replay controls
	flags.trace = fs.String("trace", "", "path to a trace JSON file to replay through the kernel")
	flags.engineID = fs.String("engine", "mock", "engine id for trace replay (default: mock; inkernel: the explicit fak-native model path; cassette)")
	flags.vdso = fs.Bool("vdso", true, "enable the vDSO fast path for trace replay")
	flags.policyPath = fs.String("policy", "", "load the capability floor from a manifest (default: the built-in production capability floor; see `fak policy --dump`)")

	return fs, flags
}

func parseRunArgs(fs *flag.FlagSet, flags *runFlags, argv []string) (parsedRunCommand, error) {
	if len(argv) == 0 {
		return parsedRunCommand{
			action:     runActionUsage,
			errMessage: "fak run: model or --trace is required",
		}, nil
	}

	var modelRef string
	var prompt string

	if !strings.HasPrefix(argv[0], "-") {
		// Model specified as the first argument: fak run <model> [flags] [prompt]
		modelRef = argv[0]
		if err := fs.Parse(argv[1:]); err != nil {
			if errors.Is(err, flag.ErrHelp) {
				return parsedRunCommand{action: runActionHelp}, nil
			}
			return parsedRunCommand{}, err
		}
		if flags.trace != nil && *flags.trace != "" {
			return parsedRunCommand{
				action:     runActionUsage,
				errMessage: "fak run: --trace cannot be combined with a model argument",
			}, nil
		}
		prompt = strings.TrimSpace(strings.Join(fs.Args(), " "))
	} else {
		// Flag specified as first argument: fak run [flags] [model] [prompt]
		if err := fs.Parse(argv); err != nil {
			if errors.Is(err, flag.ErrHelp) {
				return parsedRunCommand{action: runActionHelp}, nil
			}
			return parsedRunCommand{}, err
		}
		if flags.trace != nil && *flags.trace != "" {
			return parsedRunCommand{
				action:     runActionTrace,
				tracePath:  *flags.trace,
				engineID:   *flags.engineID,
				vdso:       *flags.vdso,
				policyPath: *flags.policyPath,
			}, nil
		}
		args := fs.Args()
		if len(args) == 0 {
			return parsedRunCommand{
				action:     runActionUsage,
				errMessage: "fak run: model is required (or pass --trace FILE)",
			}, nil
		}
		modelRef = args[0]
		prompt = strings.TrimSpace(strings.Join(args[1:], " "))
	}

	return parsedRunCommand{
		action:   runActionChat,
		modelRef: modelRef,
		prompt:   prompt,
		chatConfig: chatModelConfig{
			modelRef:         modelRef,
			prompt:           prompt,
			backendName:      *flags.backendName,
			metal:            *flags.metal,
			system:           *flags.system,
			maxTokens:        *flags.maxTokens,
			temp:             *flags.temp,
			topP:             *flags.topP,
			topK:             *flags.topK,
			frequencyPenalty: *flags.frequencyPenalty,
			presencePenalty:  *flags.presencePenalty,
			effort:           *flags.effort,
			thinkingBudget:   *flags.thinkingBudget,
			quiet:            *flags.quiet,
			nativeFlags:      flags.nativeFlags,
		},
	}, nil
}

func runUnifiedRun(argv []string) {
	fs, flags := newRunFlagSet("run", flag.ContinueOnError)
	cmd, err := parseRunArgs(fs, flags, argv)
	if err != nil {
		if errors.Is(err, flag.ErrHelp) {
			fs.Usage()
			os.Exit(0)
		}
		os.Exit(2)
	}
	switch cmd.action {
	case runActionHelp:
		os.Exit(0)
	case runActionUsage:
		fmt.Fprintln(os.Stderr, cmd.errMessage)
		fmt.Fprintln(os.Stderr)
		fmt.Fprintln(os.Stderr, "usage: fak run <model> [prompt] [flags]")
		fmt.Fprintln(os.Stderr, "       fak run --trace FILE [flags]")
		fmt.Fprintln(os.Stderr)
		fmt.Fprintln(os.Stderr, "Run an in-kernel model chat/REPL, or replay a recorded tool-call trace.")
		fmt.Fprintln(os.Stderr, "See 'fak run --help' or 'fak help run' for details.")
		os.Exit(2)
	case runActionTrace:
		executeTraceReplay(cmd.tracePath, cmd.engineID, cmd.vdso, cmd.policyPath)
	case runActionChat:
		executeChatModel(cmd.chatConfig)
	}
}

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
	runUnifiedRun(argv)
}

func executeChatModel(cfg chatModelConfig) {
	if cfg.nativeFlags.prefillChunk != nil {
		if err := validateNativeQwenQ4KPrefillChunk(*cfg.nativeFlags.prefillChunk); err != nil {
			fmt.Fprintln(os.Stderr, "fak run:", err)
			os.Exit(2)
		}
	}
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()

	planner := buildRunPlanner(ctx, cfg.modelRef, cfg.backendName, cfg.metal, cfg.nativeFlags.config())

	var extraOpts []agent.SampleOpt
	if cfg.effort != "" {
		extraOpts = append(extraOpts, agent.WithReasoningEffort(cfg.effort))
	}
	if cfg.thinkingBudget >= 0 {
		extraOpts = append(extraOpts, agent.WithThinkingBudget(cfg.thinkingBudget))
	}
	opts := runSampleOpts(cfg.maxTokens, cfg.temp, cfg.topP, cfg.topK, cfg.frequencyPenalty, cfg.presencePenalty, extraOpts...)
	if cfg.prompt != "" {
		// One-shot: answer and exit.
		runChatTurn(ctx, planner, cfg.system, nil, cfg.prompt, opts, !cfg.quiet)
	} else {
		// REPL: interactive session.
		runChatREPL(ctx, planner, cfg.system, opts, !cfg.quiet)
	}
	// Append cache-value observation to ledger (epic #1072, issue #1075).
	stats := cacheobs.Default.Snapshot()
	if stats.Turns > 0 {
		_ = cachevalueledger.Append("run", cfg.modelRef, nightrunLedgerPath(cachevalueledger.DefaultLedgerRel), stats)
	}
	// #1303 names this exit point alongside guard_child.go/serve.go for the Track-2
	// appendObservedCacheSavings(sessionType, provider, context, gateway.AdjudicationSummary)
	// OBSERVED-$ ledger, but `fak run` has no HTTP gateway and no provider (see the
	// doc comment above) -- there is no AdjudicationSummary to observe here, only the
	// Track-1 WITNESSED in-kernel reuse already appended above. Calling it with a
	// zero-value summary would write zero rows forever (cachevaluereport.NewSavingsRows
	// is a no-op unless CacheReadTokens/CacheCreationTokens/CompactionShedTokens is
	// nonzero), so it is deliberately omitted rather than wired as dead code.
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
func registerRunNativeControlFlags(fs *flag.FlagSet) nativeControlFlags {
	return registerNativeControlFlags(fs)
}

// resolveRunMetal decides whether `fak run` runs the in-kernel chat through the
// Apple-Silicon Metal GPU forward. It delegates to resolveServeMetal (the shared
// Apple-Silicon Metal decision seam) and adapts errors to the `fak run` surface.
func resolveRunMetal(flag, env bool, backendName string) (bool, error) {
	use, err := resolveServeMetal(flag, env, backendName)
	if err != nil {
		return false, errors.New(strings.ReplaceAll(err.Error(), "fak serve:", "fak run:"))
	}
	return use, nil
}

func buildRunPlanner(ctx context.Context, modelRef, backendName string, metalFlag bool, nativeConfig nativeControlConfig) *agent.InKernelPlanner {
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
	if err := applyNativeControls(backend, nativeConfig); err != nil {
		fmt.Fprintln(os.Stderr, "fak run:", err)
		os.Exit(2)
	}
	useMetal, err := resolveRunMetal(metalFlag, os.Getenv("FAK_METAL") != "", backendName)
	if err != nil {
		fmt.Fprintf(os.Stderr, "fak run: %v\n", err)
		os.Exit(2)
	}
	inKernelModel, q4k, _, _ := loadServeInKernelModel(ref, backend, false, 0, nil, 1)
	if inKernelModel == nil {
		fmt.Fprintf(os.Stderr, "fak run: failed to load %q into the in-kernel engine\n", ref)
		os.Exit(1)
	}
	tok, ok := resolveServeTokenizer("", ref)
	if !ok || tok == nil {
		fmt.Fprintf(os.Stderr, "fak run: %q has no usable tokenizer; pass a GGUF with an embedded tokenizer\n", ref)
		os.Exit(1)
	}
	return agent.NewInKernelPlannerWithConfig(inKernelModel, tok, modelRef, q4k, backend, useMetal, nativeConfig.Planner)
}

// runSampleOpts folds the CLI sampling flags into planner SampleOpts. Sampling and
// repetition controls stay no-ops at their zero defaults, so only a value the user
// actually passed reaches the sampler.
func runSampleOpts(maxTokens int, temp, topP float64, topK int, frequencyPenalty, presencePenalty float64, extraOpts ...agent.SampleOpt) []agent.SampleOpt {
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
	if frequencyPenalty != 0 {
		p := frequencyPenalty
		opts = append(opts, agent.WithFrequencyPenalty(&p))
	}
	if presencePenalty != 0 {
		p := presencePenalty
		opts = append(opts, agent.WithPresencePenalty(&p))
	}
	opts = append(opts, extraOpts...)
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
