package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"net"
	"net/http"
	"os"
	"os/signal"
	"strings"
	"sync"
	"sync/atomic"
	"syscall"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/allinone"
	"github.com/anthony-chaudhary/fak/internal/appversion"
	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/gpulease"
	"github.com/anthony-chaudhary/fak/internal/hfhub"
	"github.com/anthony-chaudhary/fak/internal/macfit"
	fakmodel "github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/modelreg"
	"github.com/anthony-chaudhary/fak/internal/pathutil"
	"github.com/anthony-chaudhary/fak/internal/tokenizer"
)

func printUpHelp(w io.Writer) {
	fmt.Fprintln(w, "Usage: fak up [flags]")
	fmt.Fprintln(w, "")
	fmt.Fprintln(w, "Turnkey zero-configuration Apple Silicon model provisioner and interactive server.")
	fmt.Fprintln(w, "Probes unified memory via macfit, auto-selects optimal model tier and context,")
	fmt.Fprintln(w, "launches local OpenAI-compatible server on :8080, and opens an interactive chat REPL.")
	fmt.Fprintln(w, "")
	fmt.Fprintln(w, "Flags:")
	fmt.Fprintln(w, "  --addr <addr>               address to bind HTTP server (default: 127.0.0.1:8080)")
	fmt.Fprintln(w, "  --mock                      enable mock completion responses for testing/offline")
	fmt.Fprintln(w, "  --headless                  run server in foreground without interactive REPL")
	fmt.Fprintln(w, "  --dry-run                   validate topology or profile hardware without running")
	fmt.Fprintln(w, "  --memory-gib <gib>          override detected unified memory in GiB")
	fmt.Fprintln(w, "  --model <tier>              override auto-selected model tier (e.g. 7B, 27B, 70B)")
	fmt.Fprintln(w, "  --context <tokens>          override auto-selected context budget tokens")
	fmt.Fprintln(w, "  --lock <path>               path to harness product lock JSON (v2) [all-in-one mode]")
	fmt.Fprintln(w, "  --bundle <path>             path to .fakpack bundle [all-in-one mode]")
	fmt.Fprintln(w, "  --bundle-verify-key <key>   optional public key / signature verification key for bundle")
	fmt.Fprintln(w, "  --policy <path>             path to security policy file")
	fmt.Fprintln(w, "  --engine <engine>           model engine ID (default: mock)")
	fmt.Fprintln(w, "  --help, -h                  show this help message")
}

func isServeDelegation(argv []string) bool {
	serveSpecificFlags := []string{
		"--session-registry", "-session-registry",
		"--session-state", "-session-state",
		"--gguf", "-gguf",
		"--base-url", "-base-url",
		"--metrics-snapshot", "-metrics-snapshot",
		"--require-key-env", "-require-key-env",
		"--policy-check", "-policy-check",
		"--plan-json", "-plan-json",
		"--profile", "-profile",
		"--serve", "-serve",
		"--config", "-config",
		"--backend", "-backend",
	}
	for _, arg := range argv {
		for _, f := range serveSpecificFlags {
			if arg == f || strings.HasPrefix(arg, f+"=") {
				return true
			}
		}
	}
	return false
}

// cmdUp is the product entry point for the unified deployable runtime and turnkey provisioner.
// When --lock or --bundle is supplied, it boots the all-in-one orchestrator.
// When raw serve-specific flags are passed, it delegates directly to serve.
// Otherwise, it runs the turnkey Apple Silicon model provisioner and interactive server.
func cmdUp(argv []string) {
	for _, arg := range argv {
		if arg == "--help" || arg == "-h" || arg == "help" {
			printUpHelp(os.Stdout)
			return
		}
	}

	if isServeDelegation(argv) {
		cmdServe(argv)
		return
	}

	hasLockOrBundle := false
	for _, arg := range argv {
		if arg == "--lock" || strings.HasPrefix(arg, "--lock=") ||
			arg == "-lock" || strings.HasPrefix(arg, "-lock=") ||
			arg == "--bundle" || strings.HasPrefix(arg, "--bundle=") ||
			arg == "-bundle" || strings.HasPrefix(arg, "-bundle=") {
			hasLockOrBundle = true
			break
		}
	}

	if hasLockOrBundle {
		runAllInOneUp(argv)
		return
	}

	runTurnkeyUp(os.Stdin, os.Stdout, os.Stderr, argv)
}

func runAllInOneUp(argv []string) {
	fs := flag.NewFlagSet("up", flag.ContinueOnError)
	fs.SetOutput(os.Stderr)
	lockPath := fs.String("lock", "", "path to harness product lock JSON (v2)")
	bundlePath := fs.String("bundle", "", "path to .fakpack bundle")
	bundleVerifyKey := fs.String("bundle-verify-key", "", "optional verification key for bundle")
	addr := fs.String("addr", "127.0.0.1:4000", "address to bind HTTP server")
	policyPath := fs.String("policy", "", "path to security policy file")
	engineID := fs.String("engine", "mock", "model engine ID")
	dryRun := fs.Bool("dry-run", false, "validate topology and print execution plan without running")
	mock := fs.Bool("mock", false, "enable mock engine and test components")

	if err := fs.Parse(argv); err != nil {
		os.Exit(2)
	}

	explicit := explicitFlagNames(fs)
	resolvedEngine := *engineID
	if !explicit["engine"] && !*mock {
		resolvedEngine = "inkernel"
	}

	cfg := allinone.Config{
		LockPath:        *lockPath,
		BundlePath:      *bundlePath,
		BundleVerifyKey: *bundleVerifyKey,
		Addr:            *addr,
		PolicyPath:      *policyPath,
		Engine:          resolvedEngine,
		DryRun:          *dryRun,
		Mock:            *mock,
	}

	sup, err := allinone.NewSupervisor(cfg)
	if err != nil {
		fmt.Fprintf(os.Stderr, "fak up: %v\n", err)
		os.Exit(1)
	}

	if cfg.DryRun {
		plan, err := sup.DryRunTopology()
		if err != nil {
			fmt.Fprintf(os.Stderr, "fak up dry-run failed: %v\n", err)
			os.Exit(1)
		}
		raw, err := json.MarshalIndent(plan, "", "  ")
		if err != nil {
			fmt.Fprintf(os.Stderr, "fak up json marshal: %v\n", err)
			os.Exit(1)
		}
		fmt.Println(string(raw))
		return
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	if err := sup.Start(ctx); err != nil {
		fmt.Fprintf(os.Stderr, "fak up start failed: %v\n", err)
		os.Exit(1)
	}
	ver := appversion.Current()
	if id := guardShortBuildID(); id != "" {
		ver += " (" + id + ")"
	}
	readyBadge := "[READY]"
	if guardFdIsTerminal(int(os.Stdout.Fd())) && os.Getenv("NO_COLOR") == "" {
		readyBadge = tuiSGRGreenBold + "[READY]" + tuiSGRReset
	}
	supAddr := sup.Addr()
	if !strings.HasPrefix(supAddr, "http://") && !strings.HasPrefix(supAddr, "https://") {
		supAddr = "http://" + supAddr
	}
	fmt.Printf("\n%s fak up %s running on %s\n", readyBadge, ver, supAddr)
	fmt.Printf("  • OpenAI-compatible endpoint: %s/v1/chat/completions\n", supAddr)
	fmt.Printf("  • Agent sessions endpoint:    %s/v1/fak/agent/sessions\n", supAddr)
	fmt.Printf("  • Health check endpoint:      %s/healthz\n\n", supAddr)
	_ = os.Stdout.Sync()

	<-ctx.Done()
	shutdownTimeout, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	_ = sup.Shutdown(shutdownTimeout)
}

func runTurnkeyUp(in io.Reader, stdout, stderr io.Writer, argv []string) {
	fs := flag.NewFlagSet("up", flag.ContinueOnError)
	fs.SetOutput(stderr)
	addr := fs.String("addr", "127.0.0.1:8080", "address to bind HTTP server")
	mock := fs.Bool("mock", false, "enable mock completion responses for testing/offline")
	headless := fs.Bool("headless", false, "run server only without interactive REPL")
	dryRun := fs.Bool("dry-run", false, "profile hardware and print execution plan without running")
	asJSON := fs.Bool("json", false, "emit plan in JSON format")
	memoryGiB := fs.Float64("memory-gib", 0, "override detected unified memory in GiB")
	modelOverride := fs.String("model", "", "override auto-selected model tier (e.g. 7B, 27B, 70B)")
	contextOverride := fs.Uint64("context", 0, "override auto-selected context budget tokens")
	engineID := fs.String("engine", "mock", "model engine ID")

	if err := fs.Parse(argv); err != nil {
		os.Exit(2)
	}

	explicit := explicitFlagNames(fs)
	if explicit["engine"] && *engineID == "mock" {
		*mock = true
	}

	var memoryBytes uint64
	if *memoryGiB > 0 {
		memoryBytes = uint64(*memoryGiB * float64(macfit.GiB))
	} else {
		var err error
		memoryBytes, err = macfit.DetectUnifiedMemory()
		if err != nil || memoryBytes == 0 {
			memoryBytes = 16 * macfit.GiB
		}
	}

	plan, err := macfit.ConfigureTurnkey(memoryBytes)
	if err != nil {
		fmt.Fprintf(stderr, "fak up: planning failed: %v\n", err)
		os.Exit(1)
	}

	if *modelOverride != "" {
		plan.Tier.ModelID = *modelOverride
		plan.Tier.Name = *modelOverride
	}
	if *contextOverride > 0 {
		plan.ContextBudgetTokens = *contextOverride
		allocated := plan.Tier.WeightBytes + (plan.ContextBudgetTokens * plan.KVBytesPerToken)
		if allocated < memoryBytes {
			plan.HeadroomBytes = memoryBytes - allocated
			plan.HeadroomRatio = float64(plan.HeadroomBytes) / float64(memoryBytes)
		}
	}

	if *dryRun {
		modelRef := plan.Tier.ModelID
		if modelRef == "" {
			modelRef = modelreg.DefaultAlias
		}
		ref := resolveTurnkeyModelRef(modelRef)
		resolvedURI, _ := modelreg.Resolve(ref)

		if *asJSON {
			enc := json.NewEncoder(stdout)
			enc.SetIndent("", "  ")
			_ = enc.Encode(plan)
			return
		}
		fmt.Fprintln(stdout, "fak up — Apple Silicon Turnkey Execution Plan")
		fmt.Fprintf(stdout, "Unified Memory : %.1f GiB\n", float64(plan.MemoryBytes)/float64(macfit.GiB))
		fmt.Fprintf(stdout, "Model Tier     : %s (%s, quant: %s)\n", plan.Tier.Name, plan.Tier.ModelID, plan.Tier.QuantTier)
		if resolvedURI != "" {
			fmt.Fprintf(stdout, "Resolved URI   : %s\n", resolvedURI)
		}
		fmt.Fprintf(stdout, "Weights Size   : %.2f GiB\n", float64(plan.Tier.WeightBytes)/float64(macfit.GiB))
		fmt.Fprintf(stdout, "Context Budget : %d tokens (KV: %.2f GiB)\n", plan.ContextBudgetTokens, float64(plan.ContextBudgetTokens*plan.KVBytesPerToken)/float64(macfit.GiB))
		fmt.Fprintf(stdout, "Headroom       : %.1f%% (>= 20.0%% guaranteed to prevent swap)\n", plan.HeadroomRatio*100)
		return
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	server, err := startTurnkeyServer(ctx, plan, *addr, *mock)
	if err != nil {
		fmt.Fprintf(stderr, "fak up: %v\n", err)
		os.Exit(1)
	}
	plan = server.Plan()
	defer func() {
		shutdownTimeout, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		_ = server.Shutdown(shutdownTimeout)
	}()

	ver := appversion.Current()
	if id := guardShortBuildID(); id != "" {
		ver += " (" + id + ")"
	}
	printTurnkeyReady(stdout, ver, server.Addr(), plan)

	if *headless || in == nil {
		<-ctx.Done()
		return
	}

	_ = runTurnkeyREPL(ctx, in, stdout, "http://"+server.Addr(), plan)
}

func printTurnkeyReady(w io.Writer, ver, addr string, plan macfit.TurnkeyProfile) {
	isTTY := false
	if f, ok := w.(*os.File); ok {
		isTTY = guardFdIsTerminal(int(f.Fd()))
	}
	readyBadge := "[READY]"
	if isTTY && os.Getenv("NO_COLOR") == "" {
		readyBadge = tuiSGRGreenBold + "[READY]" + tuiSGRReset
	}
	fmt.Fprintf(w, "\n%s fak up %s running on http://%s\n", readyBadge, ver, addr)
	fmt.Fprintf(w, "  • Model:                      %s (%s, quant: %s) | Context: %d tokens | Headroom: %.1f%%\n",
		plan.Tier.Name, plan.Tier.ModelID, plan.Tier.QuantTier, plan.ContextBudgetTokens, plan.HeadroomRatio*100)
	fmt.Fprintf(w, "  • OpenAI-compatible endpoint: http://%s/v1/chat/completions\n", addr)
	fmt.Fprintf(w, "  • Health check endpoint:      http://%s/healthz\n\n", addr)
	if f, ok := w.(*os.File); ok {
		_ = f.Sync()
	}
}

type turnkeyServer struct {
	plan             macfit.TurnkeyProfile
	mock             bool
	engineID         string
	planner          agent.Planner
	listener         net.Listener
	boundAddr        string
	httpServer       *http.Server
	residencyRelease func()
	requestCount     int64
	totalTokens      int64
	mu               sync.Mutex
	stopping         bool
	releaseRequested bool
	activeRequests   int
	residencyOnce    sync.Once
}

func (s *turnkeyServer) Addr() string {
	return s.boundAddr
}

func (s *turnkeyServer) Plan() macfit.TurnkeyProfile {
	return s.plan
}

func (s *turnkeyServer) Planner() *agent.InKernelPlanner {
	planner, _ := s.planner.(*agent.InKernelPlanner)
	return planner
}

func (s *turnkeyServer) Shutdown(ctx context.Context) error {
	s.mu.Lock()
	s.stopping = true
	s.mu.Unlock()
	err := s.httpServer.Shutdown(ctx)
	if err == nil {
		s.requestResidencyRelease()
	}
	return err
}

func (s *turnkeyServer) Close() error {
	// Close does not wait for active handlers, so gate new chat work first. After
	// a successful close, the last active handler releases residency; an idle
	// server releases immediately. A failed close retains residency for safety.
	s.mu.Lock()
	s.stopping = true
	s.mu.Unlock()
	err := s.httpServer.Close()
	if err == nil {
		s.requestResidencyRelease()
	}
	return err
}

func (s *turnkeyServer) releaseResidency() {
	if s.residencyRelease != nil {
		s.residencyOnce.Do(s.residencyRelease)
	}
}

func (s *turnkeyServer) requestResidencyRelease() {
	s.mu.Lock()
	s.releaseRequested = true
	idle := s.activeRequests == 0
	s.mu.Unlock()
	if idle {
		s.releaseResidency()
	}
}

func (s *turnkeyServer) beginChatRequest() bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.stopping {
		return false
	}
	s.activeRequests++
	return true
}

func (s *turnkeyServer) endChatRequest() {
	s.mu.Lock()
	s.activeRequests--
	release := s.releaseRequested && s.activeRequests == 0
	s.mu.Unlock()
	if release {
		s.releaseResidency()
	}
}

func newInKernelChatPlanner(model *fakmodel.Model, tok *tokenizer.Tokenizer, modelID string, q4k bool, backend compute.Backend, metal bool) *agent.InKernelPlanner {
	return newTurnkeyInKernelPlanner(model, tok, modelID, q4k, backend, metal, 0)
}

func newTurnkeyInKernelPlanner(model *fakmodel.Model, tok *tokenizer.Tokenizer, modelID string, q4k bool, backend compute.Backend, metal bool, contextTokens int) *agent.InKernelPlanner {
	return agent.NewInKernelPlannerWithConfig(model, tok, modelID, q4k, backend, metal, agent.InKernelPlannerConfig{
		ContextTokens: contextTokens,
	})
}

func turnkeyContextTokens(tokens uint64) (int, error) {
	maxInt := uint64(^uint(0) >> 1)
	if tokens > maxInt {
		return 0, fmt.Errorf("turnkey context budget %d exceeds the platform integer limit %d", tokens, maxInt)
	}
	return int(tokens), nil
}

func resolveTurnkeyModelRef(ref string) string {
	trimmed := strings.TrimSpace(ref)
	if trimmed == "" {
		return modelreg.DefaultAlias
	}
	if _, err := os.Stat(trimmed); err == nil {
		return trimmed
	}
	switch strings.ToLower(trimmed) {
	case "70b", "qwen3.8-70b-q4_k_m", "qwen3.8-70b", "qwen38-70b", "qwen38:70b", "qwen38:70b-q4_k_m":
		return "qwen38:70b"
	case "27b", "qwen3.8-27b-q4_k_m", "qwen3.8-27b", "qwen38-27b", "qwen38:27b", "qwen38:27b-q4_k_m":
		return "qwen38:27b"
	case "7b", "qwen3.8-7b-q4_k_m", "qwen3.8-7b":
		return "qwen2.5:7b"
	case "3b", "qwen3.8-3b-q4_k_m", "qwen3.8-3b":
		return "qwen2.5-coder:3b"
	}
	return trimmed
}

func startTurnkeyServer(ctx context.Context, plan macfit.TurnkeyProfile, addr string, mock bool, custom ...*agent.InKernelPlanner) (*turnkeyServer, error) {
	contextTokens, err := turnkeyContextTokens(plan.ContextBudgetTokens)
	if err != nil {
		return nil, err
	}

	var planner *agent.InKernelPlanner
	var residencyRelease func()
	residencyHandedOff := false
	defer func() {
		if residencyRelease != nil && !residencyHandedOff {
			residencyRelease()
		}
	}()
	if !mock {
		if len(custom) > 0 && custom[0] != nil {
			planner = custom[0]
		} else {
			modelRef := plan.Tier.ModelID
			if modelRef == "" {
				modelRef = modelreg.DefaultAlias
			}
			ref := resolveTurnkeyModelRef(modelRef)
			ref, _ = modelreg.Resolve(ref)
			ref = pathutil.ExpandTilde(ref)
			if hfhub.IsURI(ref) {
				resolved, err := hfhub.FetchURI(ctx, ref, os.Stderr)
				if err != nil {
					return nil, fmt.Errorf("fetch %s: %w", ref, err)
				}
				ref = resolved
			}
			if _, err := os.Stat(ref); err != nil {
				return nil, fmt.Errorf("model %q (%s) is not a known alias, an hf:// URI, or an existing .gguf path", plan.Tier.ModelID, ref)
			}

			backend, err := resolveServeChatBackend("")
			if err != nil {
				return nil, fmt.Errorf("backend: %w", err)
			}
			useMetal, _ := resolveServeMetal(false, false, "")

			var m *fakmodel.Model
			var q4k bool
			load := func() {
				m, q4k, _, _ = loadServeInKernelModel(ref, backend, false, contextTokens, nil, 1)
			}
			if backend != nil && backend.Name() == "vulkan" {
				residencyRelease, err = loadLocalLauncherModelWithVulkanLease(true, ref, gpulease.Options{}, load)
				if err != nil {
					return nil, fmt.Errorf("Vulkan model residency: %w", err)
				}
			} else {
				load()
			}
			if m == nil {
				return nil, fmt.Errorf("failed to load %q into the in-kernel engine", ref)
			}
			tok, ok := resolveServeTokenizer("", ref)
			if !ok || tok == nil {
				return nil, fmt.Errorf("%q has no usable tokenizer; pass a GGUF with an embedded tokenizer", ref)
			}

			planner = newTurnkeyInKernelPlanner(m, tok, plan.Tier.ModelID, q4k, backend, useMetal, contextTokens)
		}
	}
	if planner != nil {
		if effective := planner.ContextWindow(); effective > 0 {
			plan.ContextBudgetTokens = uint64(effective)
		}
	}

	if addr == "" {
		addr = "127.0.0.1:8080"
	}
	ln, err := net.Listen("tcp", addr)
	if err != nil {
		return nil, fmt.Errorf("listen on %s: %w", addr, err)
	}

	engineID := "inkernel"
	if mock {
		engineID = "mock"
	}
	var servedPlanner agent.Planner
	if planner != nil {
		servedPlanner = planner
	}

	ts := &turnkeyServer{
		plan:             plan,
		mock:             mock,
		engineID:         engineID,
		planner:          servedPlanner,
		listener:         ln,
		boundAddr:        ln.Addr().String(),
		residencyRelease: residencyRelease,
	}

	mux := http.NewServeMux()
	mux.HandleFunc("/healthz", ts.handleHealthz)
	mux.HandleFunc("/readyz", ts.handleReadyz)
	mux.HandleFunc("/v1/models", ts.handleModels)
	mux.HandleFunc("/v1/chat/completions", ts.handleChatCompletions)

	ts.httpServer = &http.Server{
		Handler: mux,
	}

	go func() {
		_ = ts.httpServer.Serve(ln)
	}()

	residencyHandedOff = true
	return ts, nil
}

func (s *turnkeyServer) handleHealthz(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	_ = json.NewEncoder(w).Encode(map[string]any{
		"status":         "ok",
		"mode":           "turnkey",
		"engine":         s.engineID,
		"tier":           s.plan.Tier.Name,
		"model":          s.plan.Tier.ModelID,
		"headroom_ratio": s.plan.HeadroomRatio,
	})
}

func (s *turnkeyServer) handleReadyz(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	_, _ = w.Write([]byte(`{"status":"ready"}`))
}

func (s *turnkeyServer) handleModels(w http.ResponseWriter, r *http.Request) {
	row := map[string]any{
		"id":         s.plan.Tier.ModelID,
		"object":     "model",
		"created":    time.Now().Unix(),
		"owned_by":   "fak",
		"permission": []any{},
	}
	if planner, ok := s.planner.(interface{ ContextWindow() int }); ok {
		if contextWindow := planner.ContextWindow(); contextWindow > 0 {
			row["context_length"] = contextWindow
		}
	}
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	_ = json.NewEncoder(w).Encode(map[string]any{
		"object": "list",
		"data":   []map[string]any{row},
	})
}

// Keep the turnkey wire message identical to the planner's canonical OpenAI
// message shape. In particular, assistant tool_calls and role=tool correlation
// fields must survive a client -> planner -> client continuation unchanged.
type chatCompletionMessage = agent.Message

type chatCompletionRequest struct {
	Model       string                  `json:"model"`
	Messages    []chatCompletionMessage `json:"messages"`
	Tools       []agent.ToolDef         `json:"tools,omitempty"`
	ToolChoice  json.RawMessage         `json:"tool_choice,omitempty"`
	Stream      bool                    `json:"stream"`
	MaxTokens   int                     `json:"max_tokens"`
	Temperature *float64                `json:"temperature,omitempty"`
}

type chatCompletionChoice struct {
	Index        int                   `json:"index"`
	Message      chatCompletionMessage `json:"message"`
	FinishReason string                `json:"finish_reason"`
}

type chatCompletionUsage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

type chatCompletionResponse struct {
	ID      string                 `json:"id"`
	Object  string                 `json:"object"`
	Created int64                  `json:"created"`
	Model   string                 `json:"model"`
	Choices []chatCompletionChoice `json:"choices"`
	Usage   chatCompletionUsage    `json:"usage"`
}

func (s *turnkeyServer) handleChatCompletions(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	if !s.beginChatRequest() {
		http.Error(w, "server stopping", http.StatusServiceUnavailable)
		return
	}
	defer s.endChatRequest()

	var req chatCompletionRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "bad request: "+err.Error(), http.StatusBadRequest)
		return
	}

	modelID := s.plan.Tier.ModelID
	if req.Model != "" {
		modelID = req.Model
	}

	promptTokens := 0
	compTokens := 0
	totalTokens := 0
	finishReason := "stop"
	var answerText string
	var answerToolCalls []agent.ToolCall

	if !s.mock && s.planner != nil {
		agentMsgs := append([]agent.Message(nil), req.Messages...)
		var sampleOpts []agent.SampleOpt
		if req.MaxTokens > 0 {
			sampleOpts = append(sampleOpts, agent.WithMaxTokens(req.MaxTokens))
		}
		if req.Temperature != nil {
			sampleOpts = append(sampleOpts, agent.WithTemperature(req.Temperature))
		}
		if len(req.ToolChoice) > 0 {
			sampleOpts = append(sampleOpts, agent.WithToolChoice(req.ToolChoice))
		}
		comp, err := s.planner.Complete(r.Context(), agentMsgs, req.Tools, sampleOpts...)
		if err != nil {
			writeTurnkeyInferenceError(w, err)
			return
		}
		if comp.ToolCallsDropped && len(comp.Message.ToolCalls) == 0 {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusBadGateway)
			_ = json.NewEncoder(w).Encode(map[string]any{
				"error": map[string]any{
					"message": "upstream tool-call format not recognized; refusing to skip adjudication",
					"type":    "server_error",
					"code":    "tool_call_conformance",
				},
			})
			return
		}
		answerText = comp.Message.Content
		answerToolCalls = comp.Message.ToolCalls
		promptTokens = comp.Usage.PromptTokens
		compTokens = comp.Usage.CompletionTokens
		totalTokens = comp.Usage.TotalTokens
		if comp.FinishReason != "" {
			finishReason = comp.FinishReason
		}
		if len(answerToolCalls) > 0 {
			finishReason = "tool_calls"
		}
	} else {
		lastUserContent := ""
		for _, m := range req.Messages {
			words := len(strings.Fields(m.Content))
			promptTokens += words + 4
			if m.Role == "user" {
				lastUserContent = m.Content
			}
		}
		if promptTokens == 0 {
			promptTokens = 8
		}

		if lastUserContent != "" {
			answerText = fmt.Sprintf("Turnkey %s completion on Apple Silicon (Metal). Probed unified RAM with %.1f%% headroom. Processed: %s",
				s.plan.Tier.Name, s.plan.HeadroomRatio*100, lastUserContent)
		} else {
			answerText = fmt.Sprintf("Turnkey %s completion on Apple Silicon via Metal. Ready to assist.", s.plan.Tier.Name)
		}

		compTokens = len(strings.Fields(answerText))
		if compTokens == 0 {
			compTokens = 12
		}
		totalTokens = promptTokens + compTokens
	}

	atomic.AddInt64(&s.requestCount, 1)
	atomic.AddInt64(&s.totalTokens, int64(compTokens))

	created := time.Now().Unix()
	cmplID := fmt.Sprintf("chatcmpl-%d", time.Now().UnixNano())

	if req.Stream {
		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("Cache-Control", "no-cache")
		w.Header().Set("Connection", "keep-alive")
		w.WriteHeader(http.StatusOK)
		flusher, _ := w.(http.Flusher)

		sendChunk := func(delta map[string]any, finishReason *string) {
			chunk := map[string]any{
				"id":      cmplID,
				"object":  "chat.completion.chunk",
				"created": created,
				"model":   modelID,
				"choices": []map[string]any{
					{
						"index":         0,
						"delta":         delta,
						"finish_reason": finishReason,
					},
				},
			}
			raw, _ := json.Marshal(chunk)
			_, _ = fmt.Fprintf(w, "data: %s\n\n", raw)
			if flusher != nil {
				flusher.Flush()
			}
		}

		sendChunk(map[string]any{"role": "assistant"}, nil)
		if answerText != "" {
			sendChunk(map[string]any{"content": answerText}, nil)
		}
		if len(answerToolCalls) > 0 {
			toolCalls := make([]map[string]any, 0, len(answerToolCalls))
			for i, call := range answerToolCalls {
				toolCalls = append(toolCalls, map[string]any{
					"index":    i,
					"id":       call.ID,
					"type":     call.Type,
					"function": call.Function,
				})
			}
			sendChunk(map[string]any{"tool_calls": toolCalls}, nil)
		}
		stop := finishReason
		sendChunk(map[string]any{}, &stop)
		_, _ = fmt.Fprint(w, "data: [DONE]\n\n")
		if flusher != nil {
			flusher.Flush()
		}
		return
	}

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	resp := chatCompletionResponse{
		ID:      cmplID,
		Object:  "chat.completion",
		Created: created,
		Model:   modelID,
		Choices: []chatCompletionChoice{
			{
				Index: 0,
				Message: chatCompletionMessage{
					Role:      "assistant",
					Content:   answerText,
					ToolCalls: answerToolCalls,
				},
				FinishReason: finishReason,
			},
		},
		Usage: chatCompletionUsage{
			PromptTokens:     promptTokens,
			CompletionTokens: compTokens,
			TotalTokens:      totalTokens,
		},
	}
	_ = json.NewEncoder(w).Encode(resp)
}

func writeTurnkeyInferenceError(w http.ResponseWriter, err error) {
	var contextErr *agent.InKernelContextLengthError
	if errors.As(err, &contextErr) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusBadRequest)
		_ = json.NewEncoder(w).Encode(map[string]any{
			"error": map[string]any{
				"message": contextErr.Error(),
				"type":    "invalid_request_error",
				"code":    "context_length_exceeded",
			},
		})
		return
	}
	http.Error(w, fmt.Sprintf("inference error: %v", err), http.StatusInternalServerError)
}

func runTurnkeyREPL(ctx context.Context, in io.Reader, out io.Writer, baseURL string, profile macfit.TurnkeyProfile) error {
	fmt.Fprintf(out, "\nInteractive chat on %s — Type prompt, press Enter. /quit or Ctrl-D to exit.\n", profile.Tier.ModelID)
	sc := bufio.NewScanner(in)
	sc.Buffer(make([]byte, 0, 64*1024), 1<<20)
	client := &http.Client{Timeout: 30 * time.Second}
	turn := 0

	promptUser := func() { fmt.Fprint(out, "you> ") }
	promptUser()
	for sc.Scan() {
		if ctx.Err() != nil {
			return ctx.Err()
		}
		rawInput := sc.Text()
		trimmedInput := strings.TrimSpace(rawInput)
		if trimmedInput == "" {
			promptUser()
			continue
		}
		if trimmedInput == "/exit" || trimmedInput == "/quit" {
			break
		}

		turn++
		t0 := time.Now()

		reqPayload := map[string]any{
			"model": profile.Tier.ModelID,
			"messages": []map[string]string{
				{"role": "user", "content": trimmedInput},
			},
			"stream": false,
		}
		reqBody, _ := json.Marshal(reqPayload)

		req, err := http.NewRequestWithContext(ctx, http.MethodPost, baseURL+"/v1/chat/completions", bytes.NewReader(reqBody))
		if err != nil {
			fmt.Fprintf(out, "fak> request error: %v\n", err)
			promptUser()
			continue
		}
		req.Header.Set("Content-Type", "application/json")

		resp, err := client.Do(req)
		if err != nil {
			fmt.Fprintf(out, "fak> connection error: %v\n", err)
			promptUser()
			continue
		}

		var compResp chatCompletionResponse
		err = json.NewDecoder(resp.Body).Decode(&compResp)
		_ = resp.Body.Close()

		if err != nil || len(compResp.Choices) == 0 {
			fmt.Fprintf(out, "fak> response parse error: %v\n", err)
			promptUser()
			continue
		}

		elapsed := time.Since(t0)
		answer := compResp.Choices[0].Message.Content
		compTokens := compResp.Usage.CompletionTokens
		if compTokens == 0 {
			compTokens = len(strings.Fields(answer))
		}
		secs := elapsed.Seconds()
		if secs <= 0 {
			secs = 0.001
		}
		toksPerSec := float64(compTokens) / secs

		fmt.Fprintf(out, "fak> %s\n", strings.TrimSpace(answer))
		fmt.Fprintf(out, "     [telemetry: %.1f tok/s | %d tokens | %s | context: %d/%d]\n",
			toksPerSec, compTokens, elapsed.Round(time.Millisecond), compResp.Usage.TotalTokens, profile.ContextBudgetTokens)
		promptUser()
	}
	return nil
}
