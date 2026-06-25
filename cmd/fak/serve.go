package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"time"

	"github.com/anthony-chaudhary/fak/internal/appversion"
	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/gateway"
	"github.com/anthony-chaudhary/fak/internal/ggufload"
	"github.com/anthony-chaudhary/fak/internal/hfhub"
	fakmodel "github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/modelengine"
	"github.com/anthony-chaudhary/fak/internal/pathutil"
	"github.com/anthony-chaudhary/fak/internal/policy"
	"github.com/anthony-chaudhary/fak/internal/tokenizer"
)

func cmdServe(argv []string) {
	// t0 anchors the boot timeline exposed as fak_gateway_time_to_ready_seconds; it
	// must be the FIRST statement so flag parse + policy + weight load are accounted.
	t0 := time.Now()
	fs := flag.NewFlagSet("serve", flag.ExitOnError)
	addr := fs.String("addr", "127.0.0.1:8080", "HTTP listen address (OpenAI + fak + /mcp surface); ignored with --stdio")
	stdio := fs.Bool("stdio", false, "serve MCP over stdin/stdout (newline-delimited JSON-RPC) instead of HTTP")
	provider := fs.String("provider", "openai", "upstream provider transcript wire: openai, anthropic, gemini, or xai")
	baseURL := fs.String("base-url", "", "upstream provider base URL for the /v1/chat/completions proxy (empty = offline mock planner)")
	model := fs.String("model", "mock", "model id (advertised by /v1/models; used for the upstream call)")
	apiKeyEnv := fs.String("api-key-env", "", "env var holding the upstream API key (proxy mode)")
	engineCacheEngine := fs.String("engine-cache-engine", "", "self-hosted upstream cache reset engine for quarantined provider-bound tool results: sglang|vllm (empty disables)")
	engineCacheBaseURL := fs.String("engine-cache-base-url", "", "serving-engine control/base URL for cache reset (default: --base-url when --engine-cache-engine is set)")
	engineCacheAdminKeyEnv := fs.String("engine-cache-admin-key-env", "", "env var holding the serving-engine admin API key for cache reset")
	engineCacheIdleTimeout := fs.Duration("engine-cache-idle-timeout", 0, "SGLang /flush_cache idle timeout, e.g. 30s (0 fails fast)")
	engineCacheRequireExactSpan := fs.Bool("engine-cache-require-exact-span", false, "require exact remote K/V/index span eviction; fail closed if the selected engine only supports whole-cache reset")
	engineID := fs.String("engine", "inkernel", "registered engine id that fak_syscall dispatches an allowed call to (default: the fused in-kernel model)")
	backendName := fs.String("backend", "", "compute backend for the in-kernel chat decode (with --gguf, no --base-url): empty = the CPU reference path; a registered device name like 'cuda' runs prefill+decode through the GPU HAL. Requires a `-tags cuda` build AND a reachable GPU at runtime; fails loud if named but unavailable so a typo never silently runs on CPU.")
	policyPath := fs.String("policy", "", "capability-floor manifest to load (default: built-in DefaultPolicy)")
	policyCheck := fs.Bool("policy-check", false, "validate --policy and exit without binding a listener")
	vdso := fs.Bool("vdso", true, "enable the vDSO dedup fast path")
	invalidation := fs.String("invalidation", "global", "vDSO tier-2 invalidation granularity for the live fleet: global|namespace|resource")
	requireKeyEnv := fs.String("require-key-env", "", "env var holding a bearer token to REQUIRE on every request (default: no auth)")
	ggufPath := fs.String("gguf", "", "load these GGUF weights into the in-kernel engine at boot; the load is part of the measured startup sequence and its phase breakdown is exposed on /metrics. Default path is lean-Q8 (Q4→f32→Q8 round-trip); set FAK_Q4K=1 for the direct-resident-Q4_K path (Qwen3.6-27B q4_k_m, the P1/P2 decode lever)")
	tokPath := fs.String("tokenizer", "", "OPTIONAL override for the in-kernel CHAT planner's tokenizer. With --gguf and no --base-url, /v1/chat/completions AND /v1/messages already serve the in-kernel model (real ChatML chat) using the GGUF's EMBEDDED tokenizer; pass this only to override it (e.g. an SPM-only checkpoint with no embedded BPE tokenizer, or a custom vocab). Accepts a tokenizer.json or its directory. e.g. ~/.cache/fak-models/tokenizers/qwen3.6")
	ctxViewBudget := fs.Int("ctx-view-budget", 0, "wire the ctxplan context PLANNER into the live serve loop: each buffered turn, re-materialize the forwarded history as an O(1) planned VIEW under this resident-token budget (a planned view in place of appending the whole transcript, #555). 0 (default) leaves the existing path byte-for-byte unchanged. OFF by default: it rewrites in-flight turn history, so gate it until you have watched a real session. The streaming fast-path bypasses this; the buffered turn path is what gets planned.")
	tParse := time.Now()
	_ = fs.Parse(argv)
	parseDur := time.Since(tParse)

	// Expand a leading ~ in the model/tokenizer paths: PowerShell and most quoting
	// pass ~ through literally and Go never expands it, so `--gguf ~/...` (as the
	// docs and the --tokenizer help itself show) would otherwise fail to open.
	*ggufPath = pathutil.ExpandTilde(*ggufPath)
	*tokPath = pathutil.ExpandTilde(*tokPath)

	// An hf:// --gguf resolves to a locally cached file before the loader sees it,
	// so `fak serve --gguf hf://owner/repo/model.gguf` works without a manual
	// `fak model load` first (issue #294). Download progress goes to stderr.
	if hfhub.IsURI(*ggufPath) {
		ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
		resolved, err := hfhub.FetchURI(ctx, *ggufPath, os.Stderr)
		stop()
		if err != nil {
			fmt.Fprintf(os.Stderr, "fak serve: --gguf %v\n", err)
			os.Exit(1)
		}
		*ggufPath = resolved
	}

	// --policy-check: validate the manifest and exit, binding no listener.
	if *policyCheck {
		if *policyPath == "" {
			fmt.Fprintln(os.Stderr, "fak serve: --policy-check requires --policy FILE")
			os.Exit(2)
		}
		rt, err := policy.LoadRuntime(*policyPath)
		if err != nil {
			fmt.Fprintln(os.Stderr, "fak serve:", err)
			os.Exit(1)
		}
		fmt.Printf("OK  %s  (manifest valid; every deny cites a closed-vocabulary reason)\n\n%s", *policyPath, policy.SummaryRuntime(rt))
		return
	}

	// Install the capability floor fail-loud: a bad manifest aborts startup rather
	// than silently falling back to a more permissive default. Time it as the first
	// startup phase.
	tPolicy := time.Now()
	applyPolicy(*policyPath)
	startupPhases := []gateway.StartupPhase{
		{Name: "flag-parse", Dur: parseDur},
		{Name: "policy-load", Dur: time.Since(tPolicy)},
	}

	// Eager GGUF load: pull the weights resident BEFORE binding the listener so the
	// (potentially multi-second) load is measured as part of time-to-ready and its
	// phase breakdown is on /metrics, rather than a lazy cost paid on first request.
	//
	// Two load paths, selected by the FAK_Q4K env (mirroring cmd/fakchat and
	// cmd/q4kdiag): the default lean-Q8 round-trip, or the direct-resident-Q4_K path
	// (QWEN36-NATIVE-PERF-PLAN P1/P2) that holds eligible Q4_K matmul tensors raw and
	// engages the NEON SDOT int8 decode GEMV — ~10× faster load and the Qwen3.6-27B
	// decode lever. The Q8 path stays byte-identical when the env is unset.
	//
	// The loaded *model.Model is ALSO kept for the gateway chat planner: with a tokenizer
	// (explicit --tokenizer or the GGUF's embedded one) and no --base-url,
	// /v1/chat/completions and /v1/messages serve it directly.
	inKernelModel, inKernelQ4K, loadProfile, loadPhase := loadServeInKernelModel(*ggufPath, *backendName)
	if loadPhase.Name != "" {
		startupPhases = append(startupPhases, loadPhase)
	}

	inKernelTok, tokLoaded := resolveServeTokenizer(*tokPath, *ggufPath)
	if tokLoaded {
		startupPhases = append(startupPhases, gateway.StartupPhase{Name: "tokenizer-load", Dur: 0})
	}

	apiKey := ""
	if *apiKeyEnv != "" {
		apiKey = os.Getenv(*apiKeyEnv)
	}
	engineCacheAdminKey, ok := resolveRequiredKey(*engineCacheAdminKeyEnv, os.Getenv)
	if !ok {
		fmt.Fprintf(os.Stderr, "fak serve: --engine-cache-admin-key-env %s is set but unset/empty — refusing to send cache-reset requests with NO admin auth (set the secret or omit the flag)\n", *engineCacheAdminKeyEnv)
		os.Exit(2)
	}
	if *engineCacheIdleTimeout < 0 {
		fmt.Fprintln(os.Stderr, "fak serve: --engine-cache-idle-timeout must be non-negative")
		os.Exit(2)
	}
	requireKey, ok := resolveRequiredKey(*requireKeyEnv, os.Getenv)
	if !ok {
		fmt.Fprintf(os.Stderr, "fak serve: --require-key-env %s is set but unset/empty — refusing to start a network-facing gateway with NO authentication (set the secret or omit the flag)\n", *requireKeyEnv)
		os.Exit(2)
	}

	// Resolve the optional in-kernel chat decode backend. Lookup (not Pick) so a typo
	// or an unbuilt/absent device fails loud instead of silently degrading to CPU and
	// masquerading as a GPU result. A device backend self-registers in its init() only
	// under the matching build tag AND when a device is actually reachable at runtime.
	var chatBackend compute.Backend
	if *backendName != "" {
		be, found := compute.Lookup(*backendName)
		if !found {
			fmt.Fprintf(os.Stderr, "fak serve: --backend %q is not available (registered backends: %v). A device backend needs both a matching build tag (e.g. -tags %s) and a reachable device at runtime.\n", *backendName, compute.Registered(), *backendName)
			os.Exit(2)
		}
		chatBackend = be
		fmt.Printf("fak: in-kernel chat decode → device backend %q\n", be.Name())
	}

	srv, err := gateway.New(gateway.Config{
		EngineID:                    *engineID,
		Model:                       *model,
		BaseURL:                     *baseURL,
		Provider:                    *provider,
		APIKey:                      apiKey,
		EngineCacheEngine:           *engineCacheEngine,
		EngineCacheBaseURL:          *engineCacheBaseURL,
		EngineCacheAdminKey:         engineCacheAdminKey,
		EngineCacheIdleTimeout:      *engineCacheIdleTimeout,
		EngineCacheRequireExactSpan: *engineCacheRequireExactSpan,
		InKernelModel:               inKernelModel,
		Tokenizer:                   inKernelTok,
		InKernelQ4K:                 inKernelQ4K,
		Backend:                     chatBackend,
		RequireKey:                  requireKey,
		VDSO:                        *vdso,
		Invalidation:                *invalidation,
		Version:                     appversion.Current(),
		ReloadPolicy:                policyReloader(*policyPath),
		ResetTrace:                  resetTrace,
		ObserveTrace:                observeTrace,
		ObserveSession:              observeSession,
		ControlSession:              controlSession,
		ListSessions:                listSessions,
		StartTime:                   t0,
		StartupPhases:               startupPhases,
		CtxViewBudget:               *ctxViewBudget,
	})
	must(err)
	srv.SetModelLoadProfile(loadProfile)

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()

	if *stdio {
		// MCP over stdio: stdout carries the protocol; the log package writes to
		// stderr, so diagnostics never corrupt the frames.
		if err := srv.ServeStdio(ctx, os.Stdin, os.Stdout); err != nil && !errors.Is(err, context.Canceled) {
			must(err)
		}
		return
	}
	if *addr == "" {
		fmt.Fprintln(os.Stderr, "fak serve: --addr is required (or pass --stdio)")
		os.Exit(2)
	}
	if err := srv.ListenAndServe(ctx, *addr); err != nil && !errors.Is(err, http.ErrServerClosed) {
		must(err)
	}
}

// toGatewayLoadProfile mirrors a ggufload.LoadProfile into the gateway's import-
// decoupled ModelLoadProfile so the boot-time weight-load breakdown surfaces on
// /metrics. Returns nil for a nil profile (no eager load happened).
func toGatewayLoadProfile(p *ggufload.LoadProfile) *gateway.ModelLoadProfile {
	if p == nil {
		return nil
	}
	out := &gateway.ModelLoadProfile{
		Source:       p.Source,
		Mode:         p.Mode,
		TotalSeconds: float64(p.TotalNanos) / 1e9,
		Tensors:      p.TensorCount,
		Bottleneck:   p.Bottleneck,
	}
	for _, ph := range p.Phases {
		out.Bytes += ph.Bytes
		out.Phases = append(out.Phases, gateway.ModelLoadPhase{
			Phase:   ph.Phase,
			Seconds: float64(ph.Nanos) / 1e9,
			Bytes:   ph.Bytes,
			Tensors: ph.Tensors,
		})
	}
	return out
}

// loadServeInKernelModel eagerly loads the GGUF weights (when ggufPath is set) BEFORE the
// listener binds, so the load counts toward time-to-ready and its phase breakdown reaches
// /metrics rather than being a lazy cost on first request. It returns the resident model
// (nil if no --gguf), whether the direct-resident-Q4_K path was taken, the load profile for
// /metrics, and the model-load startup phase (zero Name when no load happened). The path
// selection mirrors cmd/fakchat: a device --backend forces the F32 load (the compute HAL has
// no quantized-upload seam yet), FAK_Q4K takes the direct-resident-Q4_K path, and the
// default is the lean-Q8 round-trip; the Q8 path stays byte-identical when the env is unset.
func loadServeInKernelModel(ggufPath, backendName string) (inKernelModel *fakmodel.Model, inKernelQ4K bool, loadProfile *gateway.ModelLoadProfile, phase gateway.StartupPhase) {
	if ggufPath == "" {
		return nil, false, nil, gateway.StartupPhase{}
	}
	tLoad := time.Now()
	switch {
	case backendName != "":
		// A device backend (e.g. CUDA) consumes weights through the compute HAL Upload path,
		// which today only narrows F32 host data to VRAM (the quantized H2D / UploadDtype seam
		// is deferred — see internal/compute/cuda.go). The lean-Q8 and Q4_K loads keep ONLY
		// quantized weights (no F32 manifest entry), which makes weightHAL panic ("missing
		// tensor"). So when serving on a device we load the GGUF as F32 — the SAME path
		// cmd/modelbench uses before NewBackendSession — so the HAL takes its proven F32 GEMV.
		mm, err := ggufload.LoadModel(ggufPath)
		must(err)
		modelengine.Preload(mm)
		loadNanos := time.Since(tLoad).Nanoseconds()
		profile := toGatewayLoadProfile(&ggufload.LoadProfile{
			Mode:       "gguf-f32-device",
			Source:     ggufPath,
			TotalNanos: loadNanos,
			TotalMS:    float64(loadNanos) / 1e6,
			Phases:     []ggufload.LoadPhaseStat{{Phase: "f32-load", Calls: 1, Nanos: loadNanos, MS: float64(loadNanos) / 1e6, TimePct: 100}},
			Bottleneck: "f32-load",
		})
		return mm, false, profile, gateway.StartupPhase{Name: "model-load", Dur: time.Duration(loadNanos)}
	case os.Getenv("FAK_Q4K") != "":
		mm, err := ggufload.LoadModelQ4K(ggufPath)
		must(err)
		modelengine.PreloadQ4K(mm)
		loadNanos := time.Since(tLoad).Nanoseconds()
		// LoadModelQ4K does not thread a LoadProfiler (the direct-q4 path has no
		// dequant/re-quant phases to break down), so surface the load as a single
		// measured phase rather than an empty profile.
		profile := toGatewayLoadProfile(&ggufload.LoadProfile{
			Mode:       "gguf-resident-q4k",
			Source:     ggufPath,
			TotalNanos: loadNanos,
			TotalMS:    float64(loadNanos) / 1e6,
			Phases:     []ggufload.LoadPhaseStat{{Phase: "q4k-direct-load", Calls: 1, Nanos: loadNanos, MS: float64(loadNanos) / 1e6, TimePct: 100}},
			Bottleneck: "q4k-direct-load",
		})
		return mm, true, profile, gateway.StartupPhase{Name: "model-load", Dur: time.Duration(loadNanos)}
	default:
		prof := ggufload.NewLoadProfiler()
		mm, err := ggufload.LoadModelQuantProfile(ggufPath, prof)
		must(err)
		modelengine.Preload(mm)
		loadNanos := time.Since(tLoad).Nanoseconds()
		profile := toGatewayLoadProfile(prof.Snapshot("gguf-lean-q8", ggufPath, loadNanos))
		return mm, false, profile, gateway.StartupPhase{Name: "model-load", Dur: time.Duration(loadNanos)}
	}
}

// resolveServeTokenizer picks the in-kernel chat planner's tokenizer: an explicit
// --tokenizer (a tokenizer.json or its directory, matching cmd/fakchat's resolution) wins;
// otherwise, with --gguf set, the GGUF's EMBEDDED tokenizer is used so /v1/chat/completions
// and /v1/messages serve real in-kernel chat by default (like cmd/simpledemo); otherwise it
// returns nil, leaving the gateway's offline MockPlanner fallback. The bool reports whether
// a tokenizer-load startup phase should be recorded.
func resolveServeTokenizer(tokPath, ggufPath string) (*tokenizer.Tokenizer, bool) {
	if tokPath != "" {
		tokFile := tokPath
		if info, err := os.Stat(tokFile); err == nil && info.IsDir() {
			tokFile = filepath.Join(tokFile, "tokenizer.json")
		}
		tok, err := tokenizer.LoadJSON(tokFile)
		must(err)
		return tok, true
	}
	if ggufPath != "" {
		// No explicit --tokenizer: fall back to the tokenizer EMBEDDED in the GGUF. Virtually
		// every GGUF carries its full vocab+merges, so `fak serve --gguf X` (no --base-url)
		// serves real in-kernel chat out of the box instead of silently dropping
		// /v1/chat/completions to the offline MockPlanner. If the GGUF embeds no usable BPE
		// tokenizer (e.g. an SPM-only checkpoint), we keep the MockPlanner fallback — pass
		// --tokenizer to override.
		if tok, err := embeddedGGUFTokenizer(ggufPath); err == nil {
			return tok, true
		} else {
			fmt.Fprintf(os.Stderr, "fak serve: --gguf set without --tokenizer and no embedded BPE tokenizer (%v);\n"+
				"  /v1/chat/completions will use the offline mock planner. Pass --tokenizer <dir|file> for real chat.\n", err)
		}
	}
	return nil, false
}
