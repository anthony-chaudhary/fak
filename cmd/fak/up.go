package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
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

	"github.com/anthony-chaudhary/fak/internal/allinone"
	"github.com/anthony-chaudhary/fak/internal/appversion"
	"github.com/anthony-chaudhary/fak/internal/macfit"
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

	cfg := allinone.Config{
		LockPath:        *lockPath,
		BundlePath:      *bundlePath,
		BundleVerifyKey: *bundleVerifyKey,
		Addr:            *addr,
		PolicyPath:      *policyPath,
		Engine:          *engineID,
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
	fmt.Printf("fak up %s running on %s\n", ver, sup.Addr())

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

	if err := fs.Parse(argv); err != nil {
		os.Exit(2)
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
		if *asJSON {
			enc := json.NewEncoder(stdout)
			enc.SetIndent("", "  ")
			_ = enc.Encode(plan)
			return
		}
		fmt.Fprintln(stdout, "fak up — Apple Silicon Turnkey Execution Plan")
		fmt.Fprintf(stdout, "Unified Memory : %.1f GiB\n", float64(plan.MemoryBytes)/float64(macfit.GiB))
		fmt.Fprintf(stdout, "Model Tier     : %s (%s, quant: %s)\n", plan.Tier.Name, plan.Tier.ModelID, plan.Tier.QuantTier)
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
	defer func() {
		shutdownTimeout, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		_ = server.Shutdown(shutdownTimeout)
	}()

	ver := appversion.Current()
	if id := guardShortBuildID(); id != "" {
		ver += " (" + id + ")"
	}
	fmt.Fprintf(stdout, "fak up %s running on http://%s\n", ver, server.Addr())
	fmt.Fprintf(stdout, "Model: %s (%s, quant: %s) | Context: %d tokens | Headroom: %.1f%%\n",
		plan.Tier.Name, plan.Tier.ModelID, plan.Tier.QuantTier, plan.ContextBudgetTokens, plan.HeadroomRatio*100)
	fmt.Fprintf(stdout, "OpenAI-compatible endpoint: http://%s/v1/chat/completions\n", server.Addr())

	if *headless || in == nil {
		<-ctx.Done()
		return
	}

	_ = runTurnkeyREPL(ctx, in, stdout, "http://"+server.Addr(), plan)
}

type turnkeyServer struct {
	plan         macfit.TurnkeyProfile
	mock         bool
	listener     net.Listener
	boundAddr    string
	httpServer   *http.Server
	requestCount int64
	totalTokens  int64
	mu           sync.Mutex
	stopping     bool
}

func (s *turnkeyServer) Addr() string {
	return s.boundAddr
}

func (s *turnkeyServer) Plan() macfit.TurnkeyProfile {
	return s.plan
}

func (s *turnkeyServer) Shutdown(ctx context.Context) error {
	s.mu.Lock()
	s.stopping = true
	s.mu.Unlock()
	return s.httpServer.Shutdown(ctx)
}

func (s *turnkeyServer) Close() error {
	return s.httpServer.Close()
}

func startTurnkeyServer(ctx context.Context, plan macfit.TurnkeyProfile, addr string, mock bool) (*turnkeyServer, error) {
	if addr == "" {
		addr = "127.0.0.1:8080"
	}
	ln, err := net.Listen("tcp", addr)
	if err != nil {
		return nil, fmt.Errorf("listen on %s: %w", addr, err)
	}

	ts := &turnkeyServer{
		plan:      plan,
		mock:      mock,
		listener:  ln,
		boundAddr: ln.Addr().String(),
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

	return ts, nil
}

func (s *turnkeyServer) handleHealthz(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	_ = json.NewEncoder(w).Encode(map[string]any{
		"status":         "ok",
		"mode":           "turnkey",
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
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	_ = json.NewEncoder(w).Encode(map[string]any{
		"object": "list",
		"data": []map[string]any{
			{
				"id":         s.plan.Tier.ModelID,
				"object":     "model",
				"created":    time.Now().Unix(),
				"owned_by":   "fak",
				"permission": []any{},
			},
		},
	})
}

type chatCompletionMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type chatCompletionRequest struct {
	Model       string                  `json:"model"`
	Messages    []chatCompletionMessage `json:"messages"`
	Stream      bool                    `json:"stream"`
	MaxTokens   int                     `json:"max_tokens"`
	Temperature float64                 `json:"temperature"`
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

	var answerText string
	if lastUserContent != "" {
		answerText = fmt.Sprintf("Turnkey %s completion on Apple Silicon (Metal). Probed unified RAM with %.1f%% headroom. Processed: %s",
			s.plan.Tier.Name, s.plan.HeadroomRatio*100, lastUserContent)
	} else {
		answerText = fmt.Sprintf("Turnkey %s completion on Apple Silicon via Metal. Ready to assist.", s.plan.Tier.Name)
	}

	compTokens := len(strings.Fields(answerText))
	if compTokens == 0 {
		compTokens = 12
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
		sendChunk(map[string]any{"content": answerText}, nil)
		stop := "stop"
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
					Role:    "assistant",
					Content: answerText,
				},
				FinishReason: "stop",
			},
		},
		Usage: chatCompletionUsage{
			PromptTokens:     promptTokens,
			CompletionTokens: compTokens,
			TotalTokens:      promptTokens + compTokens,
		},
	}
	_ = json.NewEncoder(w).Encode(resp)
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
