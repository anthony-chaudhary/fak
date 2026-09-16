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
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"syscall"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/allinone"
	"github.com/anthony-chaudhary/fak/internal/appversion"
	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/gateway"
	"github.com/anthony-chaudhary/fak/internal/gpulease"
	"github.com/anthony-chaudhary/fak/internal/hfhub"
	"github.com/anthony-chaudhary/fak/internal/macfit"
	"github.com/anthony-chaudhary/fak/internal/metalgemm"
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
	fmt.Fprintln(w, "  --engine <engine>           model engine ID (default: inkernel; mock only with --mock)")
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
	engineID := fs.String("engine", "inkernel", "model engine ID (default inkernel; mock only with --mock)")
	dryRun := fs.Bool("dry-run", false, "validate topology and print execution plan without running")
	mock := fs.Bool("mock", false, "enable mock engine and test components")

	if err := fs.Parse(argv); err != nil {
		os.Exit(2)
	}

	explicit := explicitFlagNames(fs)
	resolvedEngine := *engineID
	if *mock && !explicit["engine"] {
		resolvedEngine = "mock"
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
	if sup.ServesChatCompletions() {
		fmt.Printf("  • OpenAI-compatible endpoint: %s/v1/chat/completions\n", supAddr)
	}
	fmt.Printf("  • Agent sessions endpoint:    %s/v1/fak/agent/sessions\n", supAddr)
	fmt.Printf("  • Health check endpoint:      %s/healthz\n\n", supAddr)
	_ = os.Stdout.Sync()

	<-ctx.Done()
	shutdownTimeout, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	_ = sup.Shutdown(shutdownTimeout)
}

// applyTurnkeyModelOverride applies a --model override to an already-planned
// TurnkeyProfile. When the override names a known tier (case-insensitive), the
// plan's ENTIRE geometry is replaced with that tier's StandardTiers entry and
// the KV bytes-per-token, context bucket, and headroom are recomputed for the
// new geometry so no stale auto-selected-tier accounting survives. When the
// override is not a known tier (a raw .gguf path, an hf:// URI, or an arbitrary
// alias), only the ModelID/Name labels change, preserving the historical
// passthrough behavior. An explicit --context override is applied by the caller
// afterwards, so it still wins over the recomputed bucket.
func applyTurnkeyModelOverride(plan *macfit.TurnkeyProfile, memoryBytes uint64, modelOverride string) error {
	tier, ok := macfit.LookupModelTier(modelOverride)
	if !ok {
		plan.Tier.ModelID = modelOverride
		plan.Tier.Name = modelOverride
		return nil
	}

	reserveBytes := (memoryBytes * 20) / 100
	if reserveBytes == 0 {
		reserveBytes = 1
	}
	// A named tier that cannot fit within 80% usable memory would otherwise emit a
	// physically impossible plan (weights alone exceeding RAM) while still printing
	// the ">= 20% headroom" guarantee. Refuse it exactly as the auto-selection path
	// does, rather than silently carrying the base plan's stale headroom.
	usableBytes := memoryBytes - reserveBytes
	if tier.WeightBytes >= usableBytes {
		return fmt.Errorf("model tier %s requires %.2f GiB but only %.2f GiB is usable on this host (20%% headroom reserved); choose a smaller tier or a raw .gguf/hf:// override",
			tier.Name, float64(tier.WeightBytes)/float64(macfit.GiB), float64(usableBytes)/float64(macfit.GiB))
	}

	plan.Tier = tier
	plan.ReserveBytes = reserveBytes

	kvpt, err := macfit.KVBytesPerTokenForTier(tier, plan.KVPrecision)
	if err != nil {
		return err
	}
	plan.KVBytesPerToken = kvpt

	kvPoolBytes := usableBytes - tier.WeightBytes
	plan.KVPoolBytes = kvPoolBytes

	maxTokens := uint64(0)
	if kvpt > 0 {
		maxTokens = kvPoolBytes / kvpt
	}
	var contextBudget uint64
	for _, bucket := range []uint64{65536, 32768, macfit.TurnkeySLOContextTokens, 16384, 8192, 4096, 2048, 1024, 512} {
		if maxTokens >= bucket {
			contextBudget = bucket
			break
		}
	}
	// Falling back to maxTokens (when no standard bucket fits) also covers the
	// empty-pool case: a hardcoded 512 floor could allocate more KV than the pool
	// holds and quietly break the >= 20% headroom guarantee, so the context budget
	// must never exceed what maxTokens actually permits.
	if contextBudget == 0 {
		contextBudget = maxTokens
	}
	// The named-tier override path honors the same flagship SLO default as the
	// auto-selected path, so `--model 27B` and auto-selection agree on 20k.
	if slo := macfit.TurnkeySLODefault(tier, kvpt, kvPoolBytes); slo > 0 {
		contextBudget = slo
		if used := contextBudget * kvpt; kvPoolBytes < used {
			kvPoolBytes = used
			plan.KVPoolBytes = kvPoolBytes
		}
	}
	plan.ContextBudgetTokens = contextBudget

	allocatedBytes := tier.WeightBytes + (contextBudget * kvpt)
	plan.HeadroomBytes = memoryBytes - allocatedBytes
	plan.HeadroomRatio = float64(plan.HeadroomBytes) / float64(memoryBytes)
	return nil
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
	kvPrecision := fs.String("kv-precision", "", "KV cache storage tier: f32 (default, exact) or q8_0 (dense mixed: f32 pre-RoPE K + q8_0 K/V; ~2x more context). Also settable via FAK_UP_KV_PRECISION.")
	engineID := fs.String("engine", "inkernel", "model engine ID (default inkernel; mock only with --mock)")
	gpuIdleExit := fs.Duration("gpu-idle-exit", defaultGPUIdleExit, "stop the resident server after this idle window (no in-flight request) so its GPU lease and model residency are released for a queued peer (e.g. modelbench, #13135); 0 keeps the historical process-lifetime holder")

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

	kvPrec, err := resolveUpKVPrecision(*kvPrecision)
	if err != nil {
		fmt.Fprintf(stderr, "fak up: %v\n", err)
		os.Exit(1)
	}

	plan, err := macfit.ConfigureTurnkeyWithOptions(memoryBytes, macfit.TurnkeyOptions{KVPrecision: kvPrec})
	if err != nil {
		fmt.Fprintf(stderr, "fak up: planning failed: %v\n", err)
		os.Exit(1)
	}

	if *modelOverride != "" {
		if err := applyTurnkeyModelOverride(&plan, memoryBytes, *modelOverride); err != nil {
			fmt.Fprintf(stderr, "fak up: %v\n", err)
			os.Exit(1)
		}
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
		explicit := turnkeyRefIsExplicit(modelRef)
		resolvedURI, _ := modelreg.Resolve(ref)
		if err := validateTurnkeyArtifact(plan.Tier, describeTurnkeyArtifact(resolvedURI, explicit)); err != nil {
			fmt.Fprintf(stderr, "fak up: %v\n", err)
			os.Exit(1)
		}

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
		kvPrecLabel := string(plan.KVPrecision)
		if kvPrecLabel == "" {
			kvPrecLabel = string(fakmodel.KVPrecisionFP16)
		}
		fmt.Fprintf(stdout, "Context Budget : %d tokens (KV: %s, %.2f GiB)\n", plan.ContextBudgetTokens, kvPrecLabel, float64(plan.ContextBudgetTokens*plan.KVBytesPerToken)/float64(macfit.GiB))
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
	server.armGPUIdleExit(*gpuIdleExit)
	printTurnkeyReady(stdout, ver, server.Addr(), plan)
	printTurnkeyBackendStamp(stdout, server.metalDecision, metalResidencyStampFrom(server.liveResidencyReport()))
	if *gpuIdleExit > 0 {
		fmt.Fprintf(stdout, "  • GPU idle-exit:             stops after %s idle so a queued GPU peer can run (#13135)\n", *gpuIdleExit)
	}

	if *headless || in == nil {
		select {
		case <-ctx.Done():
		case <-server.done:
			// The bounded idle exit already ran the graceful shutdown and
			// released the GPU lease; nothing further to unwind here.
		}
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
	kvPrecLabel := string(plan.KVPrecision)
	if kvPrecLabel == "" {
		kvPrecLabel = string(fakmodel.KVPrecisionFP16)
	}
	fmt.Fprintf(w, "  • Model:                      %s (%s, quant: %s) | Context: %d tokens | KV precision: %s | Headroom: %.1f%%\n",
		plan.Tier.Name, plan.Tier.ModelID, plan.Tier.QuantTier, plan.ContextBudgetTokens, kvPrecLabel, plan.HeadroomRatio*100)
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
	native           *turnkeyNativeResources
	metalDecision    serveMetalDecision
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
	ready            *readinessGate
	// idleExit stops the resident server after a bounded idle window so the GPU
	// lease and model residency are released instead of pinned for the process
	// lifetime (#13135). Nil preserves the historical lifetime holder.
	idleExit *gpuIdleExitGovernor
	// stop triggers the bounded stop from the idle governor; done is closed when
	// the stop has been requested so the run loop can return through the same
	// graceful-shutdown path a SIGTERM drives.
	stop     func()
	stopOnce sync.Once
	done     chan struct{}
}

// readinessGate is a small package-main equivalent of the gateway warmup gate
// (internal/gateway/readiness_warmup.go): it records whether a boot-time warmup
// phase is still in flight so /healthz and /readyz can tell the TRUTH about
// readiness instead of hardcoding ok/ready. The zero value means "not warming",
// so a bare &turnkeyServer{} is ready — existing tests that construct one stay
// byte-for-byte unaffected. Guarded by its own mutex; safe on a nil receiver.
type readinessGate struct {
	mu       sync.Mutex
	armed    bool
	complete bool
}

// armWarming declares that boot work (e.g. the synchronous model load) is in
// flight and the server is not ready until markReady is called.
func (g *readinessGate) armWarming() {
	if g == nil {
		return
	}
	g.mu.Lock()
	defer g.mu.Unlock()
	g.armed = true
	g.complete = false
}

// markReady records that boot work finished and the server is ready. The first
// completion wins; marking ready also overrides an armed gate.
func (g *readinessGate) markReady() {
	if g == nil {
		return
	}
	g.mu.Lock()
	defer g.mu.Unlock()
	g.armed = true
	g.complete = true
}

// pending reports whether readiness is being HELD for an incomplete warmup —
// true only when the gate was armed and warmup has not completed. A never-armed
// or already-complete gate returns false (readiness unaffected / already warm).
func (g *readinessGate) pending() bool {
	if g == nil {
		return false
	}
	g.mu.Lock()
	defer g.mu.Unlock()
	return g.armed && !g.complete
}

// readyState reports the readiness as a state string and a boolean: while
// armed-and-incomplete it is ("warming_up", false); otherwise ready ("ok", true).
func (g *readinessGate) readyState() (state string, isReady bool) {
	if g == nil {
		return "ok", true
	}
	g.mu.Lock()
	defer g.mu.Unlock()
	if g.armed && !g.complete {
		return "warming_up", false
	}
	return "ok", true
}

func (s *turnkeyServer) Addr() string {
	return s.boundAddr
}

// armGPUIdleExit installs the bounded idle-exit governor (#13135). A zero or
// negative window disables it, preserving the historical process-lifetime
// holder. It is a no-op when the server is mock (no GPU lease to release) or
// has no stop wired.
func (s *turnkeyServer) armGPUIdleExit(idle time.Duration) {
	if s == nil || s.stop == nil || idle <= 0 {
		return
	}
	s.mu.Lock()
	s.idleExit = newGPUIdleExitGovernor(idle, s.stop, func(format string, args ...any) {
		fmt.Fprintf(os.Stderr, format+"\n", args...)
	})
	s.mu.Unlock()
}

func (s *turnkeyServer) Plan() macfit.TurnkeyProfile {
	return s.plan
}

func (s *turnkeyServer) Planner() agent.Planner {
	return s.planner
}

func (s *turnkeyServer) Shutdown(ctx context.Context) error {
	s.mu.Lock()
	s.stopping = true
	idleExit := s.idleExit
	s.mu.Unlock()
	idleExit.close()
	err := s.httpServer.Shutdown(ctx)
	if err == nil {
		s.requestResidencyRelease()
	}
	return err
}

func (s *turnkeyServer) Close() error {
	// Close does not wait for active handlers. The final handler releases native
	// resources and their admission lease after a successful server close.
	s.mu.Lock()
	s.stopping = true
	idleExit := s.idleExit
	s.mu.Unlock()
	idleExit.close()
	err := s.httpServer.Close()
	if err == nil {
		s.requestResidencyRelease()
	}
	return err
}

func (s *turnkeyServer) releaseResidency() {
	s.residencyOnce.Do(func() {
		if s.residencyRelease != nil {
			s.residencyRelease()
		}
		_ = s.native.Close()
	})
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
	if s.stopping {
		s.mu.Unlock()
		return false
	}
	s.activeRequests++
	idleExit := s.idleExit
	s.mu.Unlock()
	if idleExit != nil {
		idleExit.requestBegan()
	}
	return true
}

func (s *turnkeyServer) endChatRequest() {
	s.mu.Lock()
	s.activeRequests--
	release := s.releaseRequested && s.activeRequests == 0
	idleExit := s.idleExit
	s.mu.Unlock()
	if idleExit != nil {
		idleExit.requestEnded()
	}
	if release {
		s.releaseResidency()
	}
}

// readiness reports whether the turnkey HTTP surface is accepting new work as
// the tuple (ready, state, reason). Readiness is a published contract (see
// help.go: GET /readyz is 503 until the server is up and passing its health
// gates), not merely a liveness flag, so it folds BOTH independent not-ready
// conditions: the boot warmup gate (s.ready, armed until the synchronous model
// load completes — #12984) and the stopping transition (the same s.stopping
// state beginChatRequest gates on). A zero-value turnkeyServer is ready:
// nothing has asked it to stop and no warmup was armed. Callers must not hold
// s.mu while encoding or writing the response.
func (s *turnkeyServer) readiness() (ready bool, state string, reason string) {
	if readyState, isReady := s.ready.readyState(); !isReady {
		return false, readyState, "boot warmup in flight"
	}
	s.mu.Lock()
	stopping := s.stopping
	s.mu.Unlock()
	if stopping {
		return false, "stopping", "server is stopping"
	}
	return true, "ok", ""
}

func newInKernelChatPlanner(model *fakmodel.Model, tok *tokenizer.Tokenizer, modelID string, q4k bool, backend compute.Backend, metal bool) *agent.InKernelPlanner {
	return newTurnkeyInKernelPlanner(model, tok, modelID, q4k, backend, metal, 0)
}

func newTurnkeyInKernelPlanner(model *fakmodel.Model, tok *tokenizer.Tokenizer, modelID string, q4k bool, backend compute.Backend, metal bool, contextTokens int) *agent.InKernelPlanner {
	// The realized KV tier is resolved once at startup (--kv-precision > FAK_UP_KV_PRECISION)
	// and published to FAK_UP_KV_PRECISION so the loader dep path (which carries no KV field)
	// and any nested serve planner agree on one value. Unset means f32, byte-identical.
	kvPrec, err := resolveUpKVPrecision("")
	if err != nil {
		panic(err)
	}
	return agent.NewInKernelPlannerWithConfig(model, tok, modelID, q4k, backend, metal, agent.InKernelPlannerConfig{ContextTokens: contextTokens, KVPrecision: kvPrec})
}

// resolveUpKVPrecision resolves the realized KV storage tier for `fak up`. Precedence:
// an explicit flag value wins; otherwise FAK_UP_KV_PRECISION; otherwise f32 (the exact
// default). It pins the resolved value back into FAK_UP_KV_PRECISION so the native
// loader dep path and the per-request planner cannot drift. Unknown tokens refuse
// rather than silently falling back, so a typo never buys a lossy cache unnoticed.
func resolveUpKVPrecision(flagValue string) (fakmodel.KVPrecision, error) {
	raw := strings.TrimSpace(flagValue)
	if raw == "" {
		raw = strings.TrimSpace(os.Getenv("FAK_UP_KV_PRECISION"))
	}
	if raw == "" {
		return fakmodel.KVPrecisionFP32, nil
	}
	prec, err := fakmodel.ParseKVPrecision(raw)
	if err != nil {
		return "", err
	}
	_ = os.Setenv("FAK_UP_KV_PRECISION", string(prec))
	return prec, nil
}

// turnkeyStableFitBudget builds the loader admission budget the turnkey path pins
// for a host whose total unified memory is memoryBytes. It is deliberately STABLE:
// the base is the physical total and the headroom is macfit's documented 20% OS/
// other-app reserve, so the number does not move with the instantaneous free-memory
// reading. That is the fix for a fresh-boot/cold-cache box spuriously refusing a
// context the reserve-based envelope admits. memoryBytes==0 (an unknown host, e.g. a
// unit-test profile) yields nil, preserving the historical live probe.
func turnkeyStableFitBudget(memoryBytes uint64) *serveFitBudget {
	if memoryBytes == 0 {
		return nil
	}
	return &serveFitBudget{Base: int64(memoryBytes), Headroom: macfit.DefaultMinHeadroomRatio}
}

func turnkeyContextTokens(tokens uint64) (int, error) {
	maxInt := uint64(^uint(0) >> 1)
	if tokens > maxInt {
		return 0, fmt.Errorf("turnkey context budget %d exceeds the platform integer limit %d", tokens, maxInt)
	}
	return int(tokens), nil
}

// turnkeyCanonicalQuantAlias maps a macfit tier's declared quant to the embedded
// catalog alias that unambiguously names that exact quant. Turnkey tier selection
// resolves to these aliases, NOT to a bare family alias like "qwen38:27b": the bare
// alias is user-shadowable via registry.json, so a user overlay could silently
// re-bind the 27B tier to a different quant (witnessed: a UD-Q2_K_XL overlay
// shadowed "qwen38:27b" while the tier budget still assumed 16 GiB Q4_K_M). The
// quant-qualified alias is not in the user overlay and always names Q4_K_M.
//
// Each alias MUST exist in modelreg.Catalog (see TestTurnkeyTierAliasesAreUnshadowed).
func turnkeyCanonicalQuantAlias(tierName string) string {
	switch strings.ToUpper(strings.TrimSpace(tierName)) {
	case "70B":
		return "qwen38:70b-q4_k_m"
	case "27B":
		return "qwen38:27b-q4_k_m"
	case "7B":
		return "qwen3.8-7b-q4_k_m"
	case "3B":
		return "qwen3.8-3b-q4_k_m"
	}
	return ""
}

func resolveTurnkeyModelRef(ref string) string {
	trimmed := strings.TrimSpace(ref)
	if trimmed == "" {
		return turnkeyCanonicalQuantAlias("27B")
	}
	if hfhub.IsURI(trimmed) {
		return trimmed
	}
	if _, err := os.Stat(trimmed); err == nil {
		return trimmed
	}
	switch strings.ToLower(trimmed) {
	case "70b", "qwen3.8-70b-q4_k_m", "qwen3.8-70b", "qwen38-70b", "qwen38:70b", "qwen38:70b-q4_k_m":
		return turnkeyCanonicalQuantAlias("70B")
	case "27b", "qwen3.8-27b-q4_k_m", "qwen3.8-27b", "qwen38-27b", "qwen38:27b", "qwen38:27b-q4_k_m":
		return turnkeyCanonicalQuantAlias("27B")
	case "7b", "qwen3.8-7b-q4_k_m", "qwen3.8-7b":
		return "qwen2.5:7b"
	case "3b", "qwen3.8-3b-q4_k_m", "qwen3.8-3b":
		return "qwen2.5-coder:3b"
	}
	return trimmed
}

// artifactDescriptor is the concrete model artifact a turnkey model ref resolves
// to, reduced to the two facts a tier-consistency gate needs: the quantization it
// names and its on-disk byte size. It is a plain value so the gate is a pure,
// network-free function unit-testable with a fabricated descriptor.
type artifactDescriptor struct {
	Ref       string // alias, hf:// URI, or local path as resolved
	Filename  string // basename of the artifact when known
	Quant     string // quant inferred from Filename (e.g. "Q4_K_M"); "" if unknown
	SizeBytes uint64 // on-disk size when known; 0 when unknown
	Explicit  bool   // operator typed a concrete path/hf:// URI (opinionated input)
}

// turnkeyTierSizeTolerance is how far a resolved artifact's byte size may sit from
// the tier's declared WeightBytes before it is treated as a different artifact. A
// lower bound catches a smaller quant masquerading as the tier default (e.g. a
// 9.3 GiB 2-bit file under a 16 GiB Q4_K_M tier); the upper bound catches a larger
// quant. Both are generous because file-size accounting varies slightly, but a
// full quant rung apart (Q2 vs Q4 ~= 0.58x) is far outside the band.
const turnkeyTierSizeTolerance = 0.25

// inferQuantFromFilename extracts a quantization token from a GGUF artifact name
// (e.g. "Qwen3.8-27B-Q4_K_M.gguf" -> "Q4_K_M", "...UD-Q2_K_XL.gguf" -> "Q2_K"). It
// normalizes the underscore/dash spelling to the K-quant form and returns "" when
// no recognizable quant token is present, so callers only fail on a real signal.
func inferQuantFromFilename(name string) string {
	base := strings.ToUpper(strings.TrimSuffix(filepath.Base(name), ".gguf"))
	base = strings.ReplaceAll(base, "_", "-")
	for _, q := range []string{"IQ1-S", "IQ1-M", "IQ2-XXS", "IQ2-XS", "IQ2-S", "IQ3-XXS", "IQ3-XS", "IQ3-S", "IQ4-XS", "IQ4-NL"} {
		if strings.Contains(base, q) {
			return strings.ReplaceAll(q, "-", "_")
		}
	}
	for _, q := range []string{"BF16", "F16", "F32", "FP8", "FP16"} {
		if strings.Contains(base, q) {
			return q
		}
	}
	for _, digits := range []string{"2", "3", "4", "5", "6", "8"} {
		stem := "Q" + digits
		idx := strings.Index(base, stem)
		if idx < 0 {
			continue
		}
		tail := base[idx:]
		var b strings.Builder
		b.WriteString(stem)
		for _, part := range strings.Split(tail, "-")[1:] {
			switch part {
			case "K", "M", "S", "L", "XL", "XS", "XXS", "NL":
				b.WriteString("_" + part)
			default:
				goto done
			}
		}
	done:
		return b.String()
	}
	return ""
}

// validateTurnkeyArtifact fails loud when the artifact a turnkey ref resolves to
// clearly contradicts the selected macfit tier's declared quant/size. An explicit
// operator-supplied full path or hf:// URI is an opinionated override and always
// passes through. An alias, by contrast, must be tier-consistent - it is the
// turnkey surface, so a mismatch means the alias was mis-bound (e.g. a user
// registry.json overlay shadowing the tier default with a 2-bit artifact).
func validateTurnkeyArtifact(tier macfit.ModelTier, art artifactDescriptor) error {
	if art.Explicit {
		return nil
	}
	if art.Quant != "" && tier.QuantTier != "" && !strings.EqualFold(art.Quant, tier.QuantTier) {
		return fmt.Errorf("turnkey model %q (alias %q) declares quant %s but tier %s requires %s; refusing to load a mismatched artifact (use an explicit .gguf path or hf:// URI to override)",
			art.Filename, art.Ref, art.Quant, tier.Name, tier.QuantTier)
	}
	if art.SizeBytes > 0 && tier.WeightBytes > 0 {
		expected := float64(tier.WeightBytes)
		actual := float64(art.SizeBytes)
		if actual < expected*(1-turnkeyTierSizeTolerance) || actual > expected*(1+turnkeyTierSizeTolerance) {
			return fmt.Errorf("turnkey model %q (alias %q) is %.2f GiB but tier %s budgets %.2f GiB for %s; refusing to load a mismatched artifact (use an explicit .gguf path or hf:// URI to override)",
				art.Filename, art.Ref, actual/float64(macfit.GiB), tier.Name, expected/float64(macfit.GiB), tier.QuantTier)
		}
	}
	return nil
}

// turnkeyRefIsExplicit reports whether the operator supplied a concrete model
// reference (an hf:// URI or an existing file) rather than a friendly tier alias.
// Explicit input is an opinionated override, so the tier-consistency gate passes it
// through; only alias-resolved turnkey refs are held to the tier's declared quant.
func turnkeyRefIsExplicit(ref string) bool {
	trimmed := strings.TrimSpace(ref)
	if trimmed == "" || strings.EqualFold(trimmed, "default") {
		return false
	}
	if hfhub.IsURI(trimmed) {
		return true
	}
	if _, err := os.Stat(pathutil.ExpandTilde(trimmed)); err == nil {
		return true
	}
	return false
}

// describeTurnkeyArtifact derives an artifact descriptor from a resolved model ref.
// explicit marks a concrete path/URI the operator typed directly (vs a tier alias),
// which the consistency gate treats as an opinionated override.
func describeTurnkeyArtifact(ref string, explicit bool) artifactDescriptor {
	art := artifactDescriptor{Ref: ref, Explicit: explicit}
	if hfhub.IsURI(ref) {
		if parsed, err := hfhub.ParseURI(ref); err == nil && parsed.File != "" {
			art.Filename = filepath.Base(parsed.File)
		}
	} else {
		art.Filename = filepath.Base(ref)
	}
	art.Quant = inferQuantFromFilename(art.Filename)
	if fi, err := os.Stat(pathutil.ExpandTilde(ref)); err == nil && !fi.IsDir() {
		art.SizeBytes = uint64(fi.Size())
	}
	return art
}

func startTurnkeyServer(ctx context.Context, plan macfit.TurnkeyProfile, addr string, mock bool, custom ...*agent.InKernelPlanner) (*turnkeyServer, error) {
	contextTokens, err := turnkeyContextTokens(plan.ContextBudgetTokens)
	if err != nil {
		return nil, err
	}
	var planner agent.Planner
	var native *turnkeyNativeResources
	var capturedMetalDecision serveMetalDecision
	// Declare the boot phase honestly: until the model/planner is constructed and
	// the listener is about to bind, a readiness probe must not claim ready.
	ready := &readinessGate{}
	ready.armWarming()
	if !mock {
		if len(custom) > 0 && custom[0] != nil {
			planner = custom[0]
		} else {
			modelRef := plan.Tier.ModelID
			if modelRef == "" {
				modelRef = modelreg.DefaultAlias
			}
			ref := resolveTurnkeyModelRef(modelRef)
			explicit := turnkeyRefIsExplicit(modelRef)
			ref, _ = modelreg.Resolve(ref)
			if err := validateTurnkeyArtifact(plan.Tier, describeTurnkeyArtifact(ref, explicit)); err != nil {
				return nil, err
			}
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
			// Re-check with the concrete on-disk size now that the artifact is local.
			if err := validateTurnkeyArtifact(plan.Tier, describeTurnkeyArtifact(ref, explicit)); err != nil {
				return nil, err
			}

			deps := defaultTurnkeyNativeLoadDeps()
			// Pin the loader's admission to a STABLE, reserve-based budget instead of
			// a live host-free probe. A momentary dip in reclaimable memory (cold
			// page cache, a just-loaded sibling) must not spuriously refuse the
			// turnkey plan the reserve-based envelope already admitted; the same 20%
			// reserve macfit plans against is the headroom here, so the two agree.
			// Genuine live pressure is still caught downstream by the local
			// admission reservation and the Metal GPU lease.
			deps.fitOverride = turnkeyStableFitBudget(plan.MemoryBytes)
			// The reservation lives below the loader and re-samples live memory,
			// so it needs the SAME stable envelope: otherwise a momentary dip
			// refuses a plan the loader just admitted (the macfit/loader and
			// reservation budgets disagreeing is the original defect).
			deps.fitFloor = deps.fitOverride
			resolveBackend := deps.resolveBackend
			admitMetal := deps.admitAndLoad
			var selectedBackend compute.Backend
			deps.resolveBackend = func() (compute.Backend, error) {
				backend, err := resolveBackend()
				selectedBackend = backend
				return backend, err
			}
			deps.admitAndLoad = func(metal bool, path string, load func(), fitFloor *serveFitBudget) (func(), error) {
				if selectedBackend != nil && selectedBackend.Name() == "vulkan" {
					return loadLocalLauncherModelWithVulkanLease(true, path, gpulease.Options{}, load)
				}
				return admitMetal(metal, path, load, fitFloor)
			}

			var err error
			native, err = loadTurnkeyNativeResourcesWith(ctx, ref, plan.Tier.ModelID, contextTokens, deps)
			if err != nil {
				return nil, err
			}
			metalDecision := serveMetalDecision{live: native.Startup.MetalLive}
			if !metalDecision.live && native.Startup.MetalCompiled {
				metalDecision.skippedBecause = skipReasonNoDevice
			} else if !metalDecision.live {
				metalDecision.skippedBecause = skipReasonNotCompiled
			}
			capturedMetalDecision = metalDecision
			planner = native.Planner
		}
	}
	if contextWindow := turnkeyPlannerContextWindow(planner); contextWindow > 0 {
		plan.ContextBudgetTokens = uint64(contextWindow)
	}

	if addr == "" {
		addr = "127.0.0.1:8080"
	}
	// Boot work (model load + planner construction) is complete here; the server
	// flips to ready immediately before binding the listener.
	ready.markReady()
	ln, err := net.Listen("tcp", addr)
	if err != nil {
		_ = native.Close()
		return nil, fmt.Errorf("listen on %s: %w", addr, err)
	}

	engineID := "inkernel"
	if mock {
		engineID = "mock"
	}
	ts := &turnkeyServer{
		plan:          plan,
		mock:          mock,
		engineID:      engineID,
		planner:       planner,
		metalDecision: capturedMetalDecision,
		native:        native,
		listener:      ln,
		boundAddr:     ln.Addr().String(),
		ready:         ready,
		done:          make(chan struct{}),
	}
	// The idle-exit stop and the signal-driven stop converge here: both request
	// the graceful shutdown and release the same residency/GPU lease exactly once.
	ts.stop = func() {
		ts.stopOnce.Do(func() {
			shutdownCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			defer cancel()
			_ = ts.Shutdown(shutdownCtx)
			close(ts.done)
		})
	}
	if native != nil {
		// Release the admission (including the machine-wide GPU lease) as the
		// residency half of the bounded stop; requestResidencyRelease fires it
		// when the last in-flight request drains, and Shutdown if already idle.
		ts.residencyRelease = func() { native.ReleaseAdmission() }
	}

	mux := http.NewServeMux()
	mux.HandleFunc("/healthz", ts.handleHealthz)
	mux.HandleFunc("/readyz", ts.handleReadyz)
	mux.HandleFunc("/v1/models", ts.handleModels)
	mux.HandleFunc("/v1/chat/completions", ts.handleChatCompletions)
	mux.HandleFunc("/v1/completions", ts.handleCompletions)
	mux.HandleFunc("/v1/fak/tokenize", ts.handleTokenize)

	ts.httpServer = &http.Server{
		Handler: mux,
	}

	go func() {
		_ = ts.httpServer.Serve(ln)
	}()

	return ts, nil
}

func (s *turnkeyServer) handleHealthz(w http.ResponseWriter, r *http.Request) {
	isReady, state, _ := s.readiness()
	var nativeStartup *turnkeyNativeStartup
	if s.native != nil {
		nativeStartup = &s.native.Startup
	}
	w.Header().Set("Content-Type", "application/json")
	// Liveness is preserved for a live-but-not-ready process: /healthz answers
	// 200 while the process can still respond, but the body's status must be
	// truthful about readiness — "warming_up" until the boot warmup gate
	// completes, "stopping" once shutdown began, otherwise "ok".
	w.WriteHeader(http.StatusOK)
	_ = json.NewEncoder(w).Encode(map[string]any{
		"status":         state,
		"ready":          isReady,
		"mode":           "turnkey",
		"engine":         s.engineID,
		"tier":           s.plan.Tier.Name,
		"model":          s.plan.Tier.ModelID,
		"headroom_ratio": s.plan.HeadroomRatio,
		"native_startup": nativeStartup,
		"live_residency": s.liveResidencyReport(),
	})
}

// liveResidencyReport reads the LIVE Metal/CPU routing state at request time (#12875) instead of
// replaying the snapshot frozen into native.Startup at load. The frozen `native_startup` stays for
// backward compatibility; this block is the truthful answer to "did decode run on Metal on this
// run?". It re-samples device-resident weight counts (a lazy later upload is now visible) and
// reports the model's process-wide promised-CPU-fallback tally, which survives the per-request
// session churn. It returns nil when no native model is loaded (mock/custom-server paths), so the
// key is absent rather than a fabricated zero.
func (s *turnkeyServer) liveResidencyReport() map[string]any {
	if s.native == nil || s.native.Model == nil {
		return nil
	}
	q6k, q8 := s.native.Model.RefreshMetalResidency()
	fallbacks := s.native.Model.MetalFallbackSnapshot()
	return map[string]any{
		"metal_live_q8_weights":    q8,
		"metal_live_q6_weights":    q6k,
		"promised_cpu_fallbacks":   fallbacks.Total,
		"fallbacks_observed":       fallbacks.Observed,
		"fallbacks_by_route":       fallbacks.ByRoute,
		"startup_q8_weights":       s.native.Startup.MetalLiveQ8Weights,
		"startup_q6_weights":       s.native.Startup.MetalLiveQ6Weights,
		"metal_q8_residency_error": s.native.Startup.MetalQ8ResidencyError,
	}
}

func (s *turnkeyServer) handleReadyz(w http.ResponseWriter, r *http.Request) {
	isReady, state, reason := s.readiness()
	w.Header().Set("Content-Type", "application/json")
	if !isReady {
		w.Header().Set("Retry-After", "1")
		w.WriteHeader(http.StatusServiceUnavailable)
		_ = json.NewEncoder(w).Encode(map[string]any{
			"status": state,
			"ready":  false,
			"reason": reason,
		})
		return
	}
	w.WriteHeader(http.StatusOK)
	_ = json.NewEncoder(w).Encode(map[string]any{
		"status": "ready",
		"ready":  true,
		"mode":   "turnkey",
	})
}

func (s *turnkeyServer) handleModels(w http.ResponseWriter, r *http.Request) {
	row := map[string]any{
		"id":         s.plan.Tier.ModelID,
		"object":     "model",
		"created":    time.Now().Unix(),
		"owned_by":   "fak",
		"permission": []any{},
	}
	if contextWindow := turnkeyPlannerContextWindow(s.planner); contextWindow > 0 {
		row["context_length"] = contextWindow
		row["context_window"] = contextWindow
		row["max_output_tokens"] = turnkeyMaxOutputTokens(uint64(contextWindow))
	}
	if turnkeyPromptEncodingAvailable(s) {
		row["fak_capabilities"] = map[string]any{
			"prompt_tokenization": map[string]any{"endpoint": "/v1/fak/tokenize"},
		}
	}
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	_ = json.NewEncoder(w).Encode(map[string]any{
		"object": "list",
		"data":   []map[string]any{row},
	})
}

func turnkeyPlannerContextWindow(planner agent.Planner) int {
	if contextual, ok := planner.(interface{ ContextWindow() int }); ok {
		return contextual.ContextWindow()
	}
	return 0
}

func turnkeyMaxOutputTokens(contextTokens uint64) int {
	const defaultMax = 1024
	if contextTokens > 0 && contextTokens < defaultMax {
		return int(contextTokens)
	}
	return defaultMax
}

// Keep the command's internal response and request names for the REPL and existing
// tests while using the gateway's canonical OpenAI wire shape.
type chatCompletionResponse = gateway.ChatResponse
type chatCompletionRequest = gateway.ChatRequest
type chatCompletionMessage = agent.Message

// turnkeyCompletionRequest is the LEGACY OpenAI text-completion wire for the
// turnkey server (POST /v1/completions). It mirrors gateway.CompletionRequest but is
// local to keep this file's wire surface explicit; `prompt` is raw because the wire
// allows a bare string or an array of strings.
type turnkeyCompletionRequest struct {
	Model       string          `json:"model"`
	Prompt      json.RawMessage `json:"prompt"`
	MaxTokens   int             `json:"max_tokens,omitempty"`
	Temperature *float64        `json:"temperature,omitempty"`
	TopP        *float64        `json:"top_p,omitempty"`
	Stream      bool            `json:"stream,omitempty"`
}

// turnkeyNormalizePrompt folds the legacy `prompt` field (bare string, or array of
// strings joined with newlines) into one prompt string.
func turnkeyNormalizePrompt(raw json.RawMessage) string {
	if len(raw) == 0 {
		return ""
	}
	var one string
	if err := json.Unmarshal(raw, &one); err == nil {
		return one
	}
	var many []string
	if err := json.Unmarshal(raw, &many); err == nil {
		return strings.Join(many, "\n")
	}
	return ""
}

// handleCompletions serves the LEGACY text-completion wire the turnkey server
// previously omitted: it wraps the request prompt as a single user message and reuses
// the same planner path as the chat route, then emits `text_completion` frames (bare
// `text`, never a chat delta). This is the surface vLLM, SGLang, llama.cpp-server,
// and the subagent fan-out harness all speak.
func (s *turnkeyServer) handleCompletions(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	if !s.beginChatRequest() {
		http.Error(w, "server stopping", http.StatusServiceUnavailable)
		return
	}
	defer s.endChatRequest()

	var req turnkeyCompletionRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "bad request: "+err.Error(), http.StatusBadRequest)
		return
	}
	prompt := turnkeyNormalizePrompt(req.Prompt)
	if strings.TrimSpace(prompt) == "" {
		http.Error(w, "prompt: field required", http.StatusBadRequest)
		return
	}
	if req.MaxTokens < 0 {
		http.Error(w, "max_tokens: must be a positive integer", http.StatusBadRequest)
		return
	}

	chatReq := gateway.ChatRequest{
		Model:       req.Model,
		Messages:    []agent.Message{{Role: agent.RoleUser, Content: prompt}},
		MaxTokens:   req.MaxTokens,
		Temperature: req.Temperature,
		TopP:        req.TopP,
		Stream:      req.Stream,
	}
	modelID := s.plan.Tier.ModelID
	if chatReq.Model != "" {
		modelID = chatReq.Model
	}

	if chatReq.Stream && !s.mock {
		if sp, ok := s.planner.(agent.StreamingPlanner); ok && sp.StreamingSupported() {
			s.handleCompletionsStream(w, r, chatReq, modelID, sp)
			return
		}
	}

	finishReason := "stop"
	answer := agent.Message{Role: agent.RoleAssistant}
	var usage agent.Usage

	if !s.mock && s.planner != nil {
		sampleOpts := turnkeyChatSampleOpts(chatReq, s.plan.ContextBudgetTokens)
		comp, err := s.planner.Complete(r.Context(), chatReq.Messages, nil, sampleOpts...)
		if err != nil {
			writeTurnkeyInferenceError(w, err)
			return
		}
		answer = comp.Message
		usage = comp.Usage
		if comp.FinishReason != "" {
			finishReason = comp.FinishReason
		}
	} else {
		answer.Content = fmt.Sprintf("Turnkey %s completion on Apple Silicon. Processed: %s", s.plan.Tier.Name, prompt)
	}
	if usage.CompletionTokens == 0 && answer.Content != "" {
		usage.CompletionTokens = len(strings.Fields(answer.Content))
		if usage.CompletionTokens == 0 {
			usage.CompletionTokens = 1
		}
		usage.TotalTokens = usage.PromptTokens + usage.CompletionTokens
	}
	atomic.AddInt64(&s.requestCount, 1)
	atomic.AddInt64(&s.totalTokens, int64(usage.CompletionTokens))
	created := time.Now().Unix()
	cmplID := fmt.Sprintf("cmpl-fak-%d", time.Now().UnixNano())

	if chatReq.Stream {
		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("Cache-Control", "no-cache")
		w.Header().Set("Connection", "keep-alive")
		w.WriteHeader(http.StatusOK)
		flusher, _ := w.(http.Flusher)
		sendChunk := func(text string, finish *string) {
			chunk := map[string]any{
				"id": cmplID, "object": "text_completion", "created": created, "model": modelID,
				"choices": []map[string]any{{"index": 0, "text": text, "finish_reason": finish}},
			}
			if finish != nil {
				chunk["usage"] = usage
			}
			raw, _ := json.Marshal(chunk)
			_, _ = fmt.Fprintf(w, "data: %s\n\n", raw)
			if flusher != nil {
				flusher.Flush()
			}
		}
		if answer.Content != "" {
			sendChunk(answer.Content, nil)
		}
		stop := finishReason
		sendChunk("", &stop)
		_, _ = fmt.Fprint(w, "data: [DONE]\n\n")
		if flusher != nil {
			flusher.Flush()
		}
		return
	}

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	_ = json.NewEncoder(w).Encode(gateway.CompletionResponse{
		ID:      cmplID,
		Object:  "text_completion",
		Created: created,
		Model:   modelID,
		Choices: []gateway.CompletionChoice{{Index: 0, Text: answer.Content, FinishReason: &finishReason}},
		Usage:   usage,
	})
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

	var req gateway.ChatRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "bad request: "+err.Error(), http.StatusBadRequest)
		return
	}
	if req.Stream && req.Fak != nil && req.Fak.NativeInferenceReceipt {
		http.Error(w, "native inference receipt requires a buffered response", http.StatusBadRequest)
		return
	}

	modelID := s.plan.Tier.ModelID
	if req.Model != "" {
		modelID = req.Model
	}
	if req.Stream && !s.mock {
		if sp, ok := s.planner.(agent.StreamingPlanner); ok && sp.StreamingSupported() {
			s.handleChatCompletionsStream(w, r, req, modelID, sp)
			return
		}
	}

	promptTokens := 0
	compTokens := 0
	totalTokens := 0
	finishReason := "stop"
	answer := agent.Message{Role: agent.RoleAssistant}
	var usage agent.Usage
	var nativeReceipt *fakmodel.NativeInferenceReceipt

	if !s.mock && s.planner != nil {
		sampleOpts := turnkeyChatSampleOpts(req, s.plan.ContextBudgetTokens)
		if req.Fak != nil {
			sampleOpts = append(sampleOpts, agent.WithNativeInferenceReceipt(req.Fak.NativeInferenceReceipt))
		}
		comp, err := s.planner.Complete(r.Context(), req.Messages, req.Tools, sampleOpts...)
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
		answer = comp.Message
		if answer.Role == "" {
			answer.Role = agent.RoleAssistant
		}
		promptTokens = comp.Usage.PromptTokens
		compTokens = comp.Usage.CompletionTokens
		totalTokens = comp.Usage.TotalTokens
		usage = comp.Usage
		nativeReceipt = comp.NativeInference
		if comp.FinishReason != "" {
			finishReason = comp.FinishReason
		}
		if len(answer.ToolCalls) > 0 {
			finishReason = "tool_calls"
		} else if compTokens == 0 && answer.Content != "" {
			compTokens = len(strings.Fields(answer.Content))
			if compTokens == 0 {
				compTokens = 1
			}
			totalTokens = promptTokens + compTokens
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

		chatWord := metalStampChatWord(s.metalDecision.live)
		if lastUserContent != "" {
			answer.Content = fmt.Sprintf("Turnkey %s completion on Apple Silicon (%s). Probed unified RAM with %.1f%% headroom. Processed: %s",
				s.plan.Tier.Name, chatWord, s.plan.HeadroomRatio*100, lastUserContent)
		} else {
			answer.Content = fmt.Sprintf("Turnkey %s completion on Apple Silicon via %s. Ready to assist.", s.plan.Tier.Name, chatWord)
		}

		compTokens = len(strings.Fields(answer.Content))
		if compTokens == 0 {
			compTokens = 12
		}
		totalTokens = promptTokens + compTokens
	}
	usage.PromptTokens = promptTokens
	usage.CompletionTokens = compTokens
	usage.TotalTokens = totalTokens

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
			if finishReason != nil {
				chunk["usage"] = usage
			}
			raw, _ := json.Marshal(chunk)
			_, _ = fmt.Fprintf(w, "data: %s\n\n", raw)
			if flusher != nil {
				flusher.Flush()
			}
		}

		sendChunk(map[string]any{"role": "assistant"}, nil)
		if answer.Content != "" {
			sendChunk(map[string]any{"content": answer.Content}, nil)
		}
		if len(answer.ToolCalls) > 0 {
			calls := make([]gateway.ChatDeltaToolCall, 0, len(answer.ToolCalls))
			for i, call := range answer.ToolCalls {
				calls = append(calls, gateway.ChatDeltaToolCall{Index: i, ID: call.ID, Type: call.Type, Function: call.Function})
			}
			sendChunk(map[string]any{"tool_calls": calls}, nil)
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
	resp := gateway.ChatResponse{
		ID:      cmplID,
		Object:  "chat.completion",
		Created: created,
		Model:   modelID,
		Choices: []gateway.ChatChoice{
			{
				Index:        0,
				Message:      answer,
				FinishReason: finishReason,
			},
		},
		Usage: usage,
	}
	if req.Fak != nil && req.Fak.NativeInferenceReceipt {
		resp.Fak = &gateway.FakExt{NativeInferenceReceipt: nativeReceipt}
	}
	_ = json.NewEncoder(w).Encode(resp)
}

func writeTurnkeyInferenceError(w http.ResponseWriter, err error) {
	var contextErr *agent.InKernelContextLengthError
	if errors.As(err, &contextErr) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusBadRequest)
		_ = json.NewEncoder(w).Encode(map[string]any{"error": map[string]any{
			"message": contextErr.Error(), "type": "invalid_request_error", "code": "context_length_exceeded",
		}})
		return
	}
	// A native Metal command-buffer wait that exceeded its bound is a local,
	// retryable resource stall — not a generic server fault. Surface it as 503
	// with a distinct code so a client can retry or shed load instead of
	// treating it as an opaque 500. The observation seam may wrap the typed
	// error, so errors.As (not a type assertion) recovers it.
	var stall metalgemm.MetalCommandBufferStallError
	if errors.As(err, &stall) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusServiceUnavailable)
		_ = json.NewEncoder(w).Encode(map[string]any{"error": map[string]any{
			"message": fmt.Sprintf("Metal command buffer stall during %s: waited %.3fms at/over %.3fms limit",
				stall.Operation, stall.WaitedMilliseconds, stall.LimitMilliseconds),
			"type": "server_error", "code": "metal_command_buffer_stalled",
		}})
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
