// serve_stages.go holds the `fak serve` boot pipeline: the serveRuntime stage
// methods cmdServe walks in order. Stage order is load-bearing: compute resolves
// before the weight load so a known device can refuse an oversize GGUF from its
// header; the session plane restores persisted drive state before the gateway
// binds; the observer seams are resolved before the gateway exists but installed
// only after (wireGateway), because the scheduler's Attach owns the table's single
// observer slots. The flag surface (serveFlags/newServeFlagSet) and buildGateway
// stay in serve.go: token_defaults_test.go and collectTokenDefaultsScorecard
// derive the default-on token-saver stack from that file's raw source.
package main

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"os"
	"os/signal"
	"strconv"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/gateway"
	"github.com/anthony-chaudhary/fak/internal/ggufload"
	"github.com/anthony-chaudhary/fak/internal/hfhub"
	"github.com/anthony-chaudhary/fak/internal/l3kv"
	fakmodel "github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/modelreg"
	"github.com/anthony-chaudhary/fak/internal/pathutil"
	"github.com/anthony-chaudhary/fak/internal/policy"
	"github.com/anthony-chaudhary/fak/internal/session"
	"github.com/anthony-chaudhary/fak/internal/tokenizer"
	"github.com/anthony-chaudhary/fak/internal/toolplugin"
)

// serveRuntime carries the state the serve boot stages resolve on the way to a
// bound listener: the compute plane, the resident model, the session/auth
// material, the observer seams, and finally the gateway itself.
type serveRuntime struct {
	t0              time.Time
	startupPhases   []gateway.StartupPhase
	startupMessages []gateway.StartupMessage

	chatBackend compute.Backend
	useMetal    bool
	ep          epRankConfig

	inKernelModel *fakmodel.Model
	inKernelQ4K   bool
	loadProfile   *gateway.ModelLoadProfile
	epGroup       *fakmodel.DistComm
	epRole        epDecodeRole
	epCoord       *fakmodel.EPDecodeCoordinator
	inKernelTok   *tokenizer.Tokenizer

	apiKey              string
	engineCacheAdminKey string
	requireKey          string
	defaultTraceID      string

	transObs  session.TransitionObserver
	budgetObs session.BudgetObserver

	toolPlugins     []toolplugin.Plugin
	toolPreferences toolplugin.PreferenceLayers
	srv             *gateway.Server
	qwen38Deps      *qwen38RuntimeDependencies
	llamaProcess    qwen38ChildProcess
}

func newServeStartupMessage(source, kind, level, text string) gateway.StartupMessage {
	return gateway.StartupMessage{Source: source, Kind: kind, Level: level, Text: text}
}

// serveNativeAdmissionPolicy changes only the token axis of the gateway's
// shipping policy. Keeping the other axes derived from DefaultAdmissionPolicy
// prevents this operator declaration from drifting scheduler semantics.
func serveNativeAdmissionPolicy(sf *serveFlags) (gateway.AdmissionPolicy, error) {
	if sf == nil || sf.nativeAdmissionTokenBudget == nil {
		return gateway.DefaultAdmissionPolicy(), errors.New("--native-admission-token-budget is unavailable")
	}
	return nativeAdmissionPolicyForBudget(*sf.nativeAdmissionTokenBudget)
}

// nativeAdmissionPolicyForBudget is the one seam every launcher that runs the
// in-kernel model derives its admission policy through (serve via serveFlags,
// guard/manage via the guard launch flag set), so the two front doors cannot
// drift apart on what an operator-declared budget means. A non-positive budget
// is refused loud — a typo must never silently shrink or disable the gate.
func nativeAdmissionPolicyForBudget(budget int) (gateway.AdmissionPolicy, error) {
	policy := gateway.DefaultAdmissionPolicy()
	if budget <= 0 {
		return policy, fmt.Errorf("--native-admission-token-budget must be positive (got %d)", budget)
	}
	policy.TokenBudget = budget
	return policy, nil
}

// newServeNativeAdmissionController is the production construction seam shared
// by buildGateway and the admission witness. The startup note is durable readback
// of the exact bound installed on this boot, not an inferred model/context limit.
func newServeNativeAdmissionController(sf *serveFlags) (*gateway.AdmissionController, gateway.StartupMessage, error) {
	policy, err := serveNativeAdmissionPolicy(sf)
	if err != nil {
		return nil, gateway.StartupMessage{}, err
	}
	message := newServeStartupMessage("serve", "native-admission-token-budget", "info",
		fmt.Sprintf("native scheduler admission token budget=%d", policy.TokenBudget))
	return gateway.NewAdmissionController(policy), message, nil
}

func (rt *serveRuntime) addStartupMessage(message gateway.StartupMessage) {
	if strings.TrimSpace(message.Text) == "" {
		return
	}
	rt.startupMessages = append(rt.startupMessages, message)
	if rt.srv != nil {
		rt.srv.AddStartupMessages(message)
	}
}

// resolveServeModelSources normalizes the --gguf/--tokenizer sources before any
// stage touches them: ~ expansion, registry alias resolution, and hf:// fetch.
func (rt *serveRuntime) resolveServeModelSources(sf *serveFlags) {
	// Expand a leading ~ in the model/tokenizer paths: PowerShell and most quoting
	// pass ~ through literally and Go never expands it, so `--gguf ~/...` (as the
	// docs and the --tokenizer help itself show) would otherwise fail to open.
	*sf.ggufPath = pathutil.ExpandTilde(*sf.ggufPath)
	*sf.tokPath = pathutil.ExpandTilde(*sf.tokPath)

	// A friendly alias (`--gguf smollm2`) resolves through the model registry to its
	// target ref (an hf:// URI or a local path) before anything else, so the run-by-name
	// surface (`fak pull` / `fak ls`) reaches `fak serve` too. A bare hf:// URI or an
	// existing path passes through unchanged.
	if *sf.ggufPath != "" {
		if resolved, expanded := modelreg.Resolve(*sf.ggufPath); expanded {
			rt.addStartupMessage(newServeStartupMessage("serve", "model-alias", "info",
				fmt.Sprintf("--gguf %s -> %s", *sf.ggufPath, resolved)))
			*sf.ggufPath = resolved
		}
	}

	// An hf:// --gguf resolves to a locally cached file before the loader sees it,
	// so `fak serve --gguf hf://owner/repo/model.gguf` works without a manual
	// `fak model load` first (issue #294). Download progress goes to stderr.
	if hfhub.IsURI(*sf.ggufPath) {
		ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
		resolved, err := hfhub.FetchURI(ctx, *sf.ggufPath, os.Stderr)
		stop()
		if err != nil {
			fmt.Fprintf(os.Stderr, "fak serve: --gguf %v\n", err)
			os.Exit(1)
		}
		*sf.ggufPath = resolved
	}
}

// runServePolicyCheck validates --policy for --policy-check and exits non-zero on
// a missing or invalid manifest; it binds no listener.
func runServePolicyCheck(policyPath string) {
	if policyPath == "" {
		writeConfigBail(os.Stderr, configBail{
			Verb:    "fak serve",
			Reason:  bailPolicyCheckNoFile,
			Summary: "--policy-check validates a manifest and none was given",
			Knobs: []bailKnob{
				bailFlag("policy-check", "true"),
				bailFlag("policy", "").want("the manifest to validate"),
			},
			Check: "fak policy --dump   # the default manifest, as a starting file",
		})
		os.Exit(2)
	}
	rt, err := policy.LoadRuntime(policyPath)
	if err != nil {
		fmt.Fprintln(os.Stderr, "fak serve:", err)
		os.Exit(1)
	}
	fmt.Printf("OK  %s  (manifest valid; every deny cites a closed-vocabulary reason)\n\n%s", policyPath, policy.SummaryRuntime(rt))
}

// resolveCompute resolves the decode compute plane before any weights load: the
// optional device --backend, the Apple-Silicon Metal seam, the expert-/tensor-
// parallel rank plan with its device-collective gate, and the CUDA-graph flip.
func (rt *serveRuntime) resolveCompute(sf *serveFlags) {
	// Resolve the optional in-kernel chat decode backend BEFORE eager model loading, so
	// a known device can refuse an oversize GGUF from its header instead of OOMing during
	// the load. Lookup (not Pick) keeps typos fail-loud rather than silently degrading to CPU.
	chatBackend, err := resolveServeChatBackend(*sf.backendName)
	if err != nil {
		writeBackendUnavailableBail(os.Stderr, "fak serve", *sf.backendName)
		os.Exit(2)
	}
	if chatBackend != nil {
		rt.addStartupMessage(newServeStartupMessage("serve", "compute-backend", "info",
			fmt.Sprintf("in-kernel chat decode -> device backend %q", chatBackend.Name())))
	}
	if err := applyNativeControls(chatBackend, serveNativeControlConfig(sf)); err != nil {
		fmt.Fprintln(os.Stderr, "fak serve:", err)
		os.Exit(2)
	}
	// Resolve the Apple-Silicon Metal GPU forward BEFORE eager loading. On an
	// Apple-Silicon+cgo binary with a usable device it is the default runtime path; an
	// explicit --metal/FAK_METAL=1 keeps the old fail-loud posture when Metal is unavailable.
	// Metal is the CPU-session seam, so it conflicts with a device --backend.
	useMetal, err := resolveServeMetal(*sf.metal, os.Getenv("FAK_METAL") != "", *sf.backendName)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(2)
	}
	if useMetal {
		rt.addStartupMessage(newServeStartupMessage("serve", "compute-backend", "info",
			"in-kernel chat decode -> Apple-Silicon Metal GPU (prefill + resident Q8 decode)"))
	}
	// Multi-GPU rank counts (#971). The EP arithmetic is host-proven bit-exact at ranks=1
	// (the no-op default), but a ranks>1 reduction is only a real multi-GPU serve when it
	// runs over a non-cpu-ref device collective. Fail loud on N>1 rather than silently reduce
	// through the single-box LocalCollective and mislabel it "multi-GPU".
	if *sf.expertParallel < 1 || *sf.tensorParallel < 1 {
		fmt.Fprintln(os.Stderr, "fak serve: --expert-parallel and --tensor-parallel must be >= 1")
		os.Exit(2)
	}
	// Resolve this process's place in a SHARDED expert-parallel serve (FAK_EP_RANK / FAK_EP_COORD_ADDR).
	// When ep.sharded, N separate processes each load only their band and reduce across a DistComm
	// process group (the host multi-process topology, #971) — so it takes the DistComm path below,
	// NOT the single-process device-collective gate. When not sharded, ep is inert and the serve is
	// byte-identical to today.
	ep, err := resolveEPRankConfig(*sf.expertParallel)
	if err != nil {
		fmt.Fprintf(os.Stderr, "fak serve: %v\n", err)
		os.Exit(2)
	}
	if *sf.expertParallel > 1 && ep.sharded && *sf.tensorParallel > 1 {
		fmt.Fprintln(os.Stderr, "fak serve: sharded --expert-parallel (FAK_EP_COORD_ADDR) does not combine with --tensor-parallel>1 yet (#971)")
		os.Exit(2)
	}
	if (*sf.expertParallel > 1 || *sf.tensorParallel > 1) && !ep.sharded {
		if *sf.expertParallel > 1 && *sf.tensorParallel > 1 && *sf.expertParallel != *sf.tensorParallel {
			fmt.Fprintln(os.Stderr, "fak serve: --expert-parallel and --tensor-parallel currently must match when both are >1; the single-process NCCL backend has one communicator world and no subgroup split yet (#971)")
			os.Exit(2)
		}
		collectiveRanks := *sf.expertParallel
		if *sf.tensorParallel > collectiveRanks {
			collectiveRanks = *sf.tensorParallel
		}
		if init, ok := chatBackend.(compute.CollectiveInitializer); ok {
			if err := init.InitCollective(collectiveRanks); err != nil {
				fmt.Fprintf(os.Stderr, "fak serve: initialize %d-rank device collective: %v\n", collectiveRanks, err)
				os.Exit(2)
			}
		}
		deviceCollective := chatBackend != nil && chatBackend.Caps().Collective
		if !deviceCollective {
			fmt.Fprintf(os.Stderr, "fak serve: --expert-parallel/--tensor-parallel N>1 requires a multi-device collective backend: no compute backend advertises Caps().Collective after initialization. Build with the device NCCL CollectiveBackend rung (CUDA: -tags cuda,nccl via FAK_CUDA_NCCL=1) and run on a box with enough visible GPUs (#971).\n")
			os.Exit(2)
		}
	}
	// --cuda-graph flips the (init-time, FAK_CUDA_GRAPH-gated) graph-replay decode path on
	// from a parsed flag. graphEnabled is consulted per token at GraphBegin, so this post-init
	// flip cleanly activates the fully-wired HAL capture/replay path. No-op on a non-cuda build.
	if *sf.cudaGraph {
		compute.EnableCUDAGraph()
		// Size the fixed device-KV prealloc to the served context so a real prompt never grows
		// the cache mid-capture (a cudaMalloc during capture is illegal — #932). Off-budget (0)
		// leaves the decode-bench default (1024). The prealloc is real VRAM (3 buffers × KV-heads
		// × head-dim × positions × 4B/layer), so an operator who wants a large graph context must
		// budget VRAM for it (or pair with the Q4_K weight lever to free room).
		compute.SetCUDAGraphKVCapacity(*sf.contextBudgetTokens)
		rt.addStartupMessage(newServeStartupMessage("serve", "cuda-graph", "info",
			fmt.Sprintf("decode replay enabled; KV graph capacity=%d positions", max(*sf.contextBudgetTokens, 1024))))
	}
	rt.chatBackend, rt.useMetal, rt.ep = chatBackend, useMetal, ep
}

// loadModel eagerly loads the GGUF weights and the in-kernel tokenizer before the
// listener binds. For a sharded expert-parallel rank it also dials the process
// group and wires the rank-local forward; the dialed group lands on rt.epGroup and
// is closed by cmdServe's deferred closeEPGroup.
func (rt *serveRuntime) loadModel(sf *serveFlags) {
	// This header-only forward gate precedes expert-shard derivation and
	// loadServeInKernelModel, which owns memory planning and tensor payload reads.
	// The graded expert spill is an OPERATOR grade, so it is validated at the terminal's expense,
	// not the load's: a mistyped --n-cpu-moe refuses here rather than after the weights are
	// resident. Carried to the planner through agent.ExpertSpillEnv (serve_ncpumoe.go).
	must(applyServeNCPUMoE(*sf.nCPUMoE))

	pf, err := preflightServeBackendForward(*sf.ggufPath, rt.chatBackend)
	must(err)
	rt.addStartupMessage(serveBackendForwardPreflightMessage(pf))

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
	//
	// For a SHARDED EP rank, size this process's expert band from the GGUF header BEFORE the load so
	// it admits only [Lo,Hi) into the resident store (the #971 residency). nil = the full model, as
	// today. numExperts is a cheap header read (no tensor bytes).
	var expertShard *ggufload.ExpertShard
	if rt.ep.sharded {
		numExperts, err := ggufNumExperts(*sf.ggufPath)
		if err != nil {
			fmt.Fprintf(os.Stderr, "fak serve: read GGUF expert count for the expert-parallel shard: %v\n", err)
			os.Exit(2)
		}
		shard, err := expertShardForConfig(rt.ep, numExperts)
		if err != nil {
			fmt.Fprintf(os.Stderr, "fak serve: %v\n", err)
			os.Exit(2)
		}
		expertShard = shard
		rt.addStartupMessage(newServeStartupMessage("expert-parallel", "shard-residency", "info",
			fmt.Sprintf("rank %d/%d loads experts [%d,%d) of %d resident", rt.ep.rank, rt.ep.ranks, shard.Lo, shard.Hi, numExperts)))
	}
	expertRanks := 1
	if rt.ep.sharded {
		expertRanks = rt.ep.ranks
	}
	if *sf.ggufPath != "" && rt.useMetal {
		if err := refuseOversubscribedMetalGGUF(*sf.ggufPath); err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(2)
		}
	}
	inKernelModel, inKernelQ4K, loadProfile, loadPhase := loadServeInKernelModel(*sf.ggufPath, rt.chatBackend, *sf.cpuOffloadExperts, *sf.contextBudgetTokens, expertShard, expertRanks)
	if loadPhase.Name != "" {
		rt.startupPhases = append(rt.startupPhases, loadPhase)
	}
	// A sharded EP rank now joins the DistComm process group and wires the rank-local forward: each
	// rank computes only its band and reduces its single [H] partial across the group. The group is
	// formed AFTER the load (a load failure must not block peers) and BEFORE binding the listener.
	// The rank-local forward path is entered ONLY here — an ordinary serve leaves the model on the
	// single-process all-band path, byte-identical to today.
	if rt.ep.sharded && inKernelModel != nil {
		group, err := dialEPGroup(rt.ep)
		if err != nil {
			fmt.Fprintf(os.Stderr, "fak serve: form the %d-rank expert-parallel group: %v\n", rt.ep.ranks, err)
			os.Exit(2)
		}
		rt.epGroup = group
		inKernelModel.SetExpertParallelRanks(rt.ep.ranks)
		inKernelModel.SetExpertParallelRank(rt.ep.rank)
		// Opt-in upgrade: on a backend that implements the multi-process NCCL process group
		// (compute.ProcessGroupBackend, -tags cuda,nccl on a real device), form it now and
		// reduce through the device-NCCL tensor rung instead of the host DistComm reduce. Any
		// other build (no cuda/nccl tag, or a backend without the seam) falls through unchanged
		// to today's NewDistCommCollective(group) — zero behavior change on every existing path.
		requireDevicePG := epRequireDevicePG()
		devColl, devErr := joinDevicePGIfSupported(rt.chatBackend, group, rt.ep)
		if devErr != nil {
			if requireDevicePG {
				fmt.Fprintf(os.Stderr, "fak serve: FAK_EP_REQUIRE_DEVICE_PG=1 but expert-parallel rank %d/%d could not join the device-NCCL process group: %v\n", rt.ep.rank, rt.ep.ranks, devErr)
				os.Exit(2)
			}
			rt.addStartupMessage(newServeStartupMessage("expert-parallel", "collective-fallback", "warning",
				fmt.Sprintf("rank %d/%d: device-NCCL process group unavailable (%v); using host DistComm reduce", rt.ep.rank, rt.ep.ranks, devErr)))
		}
		if devColl != nil {
			inKernelModel.SetExpertParallelCollective(devColl)
			rt.addStartupMessage(newServeStartupMessage("expert-parallel", "process-group", "info",
				fmt.Sprintf("rank %d/%d joined the device-NCCL tensor-reduce process group", rt.ep.rank, rt.ep.ranks)))
		} else {
			if requireDevicePG {
				backend := "<nil>"
				if rt.chatBackend != nil {
					backend = rt.chatBackend.Name()
				}
				fmt.Fprintf(os.Stderr, "fak serve: FAK_EP_REQUIRE_DEVICE_PG=1 but backend %q does not expose the multi-process device-NCCL ProcessGroupBackend. Build with FAK_CUDA_NCCL=1 (-tags cuda,nccl) and run one rank per visible GPU, or unset FAK_EP_REQUIRE_DEVICE_PG for the host DistComm development rung.\n", backend)
				os.Exit(2)
			}
			inKernelModel.SetExpertParallelCollective(fakmodel.NewDistCommCollective(group))
			rt.addStartupMessage(newServeStartupMessage("expert-parallel", "process-group", "info",
				fmt.Sprintf("rank %d/%d joined the host DistComm reduce process group", rt.ep.rank, rt.ep.ranks)))
		}
	}
	// Per-GPU residency pre-check for an expert-parallel serve: refuse an --expert-parallel N whose
	// per-card shard (replicated weights + the largest expert band) exceeds a GPU, BEFORE binding the
	// listener and letting rank r OOM uploading its band. Fail-open on cpu-ref / a non-probing backend
	// (the load above already gated host/aggregate fit); this adds only the per-rank VRAM check the
	// rank-count + Caps().Collective gate above does not make (#971).
	if err := refuseEPPlanIfUnfit(inKernelModel, rt.chatBackend, *sf.expertParallel, *sf.contextBudgetTokens); err != nil {
		fmt.Fprintf(os.Stderr, "fak serve: --expert-parallel %d does not fit resident across the GPUs: %v\n", *sf.expertParallel, err)
		os.Exit(2)
	}

	inKernelTok, tokLoaded := resolveServeTokenizer(*sf.tokPath, *sf.ggufPath, rt.addStartupMessage)
	if tokLoaded {
		rt.startupPhases = append(rt.startupPhases, gateway.StartupPhase{Name: "tokenizer-load", Dur: 0})
	}
	rt.inKernelModel, rt.inKernelQ4K, rt.loadProfile, rt.inKernelTok = inKernelModel, inKernelQ4K, loadProfile, inKernelTok
}

// closeEPGroup closes the expert-parallel process group if loadModel dialed one;
// nil-safe so cmdServe defers it unconditionally before the load stage runs.
func (rt *serveRuntime) closeEPGroup() {
	if rt.epGroup != nil {
		rt.epGroup.Close()
	}
	if rt.inKernelModel != nil {
		_ = rt.inKernelModel.CloseWeights()
	}
}

// resolveSessionPlane resolves the auth/key material, validates the budget flags,
// cold-resumes persisted session drive state (#629), and seeds the default trace's
// budget and durability registration.
func (rt *serveRuntime) resolveSessionPlane(sf *serveFlags) {
	apiKey := ""
	if *sf.apiKeyEnv != "" {
		apiKey = os.Getenv(*sf.apiKeyEnv)
	}
	engineCacheAdminKey := resolveServeRequiredKey(*sf.engineCacheAdminKeyEnv, "engine-cache-admin-key-env",
		"refusing to send cache-reset requests with no admin auth: the named admin-key variable is empty",
		"the engine-cache admin secret, or omit the flag")
	if *sf.engineCacheIdleTimeout < 0 {
		fmt.Fprintln(os.Stderr, "fak serve: --engine-cache-idle-timeout must be non-negative")
		os.Exit(2)
	}
	requireKey := resolveServeRequiredKey(*sf.requireKeyEnv, "require-key-env",
		"refusing to start a network-facing gateway with no authentication: the named bearer variable is empty",
		"the bearer token callers must present, or omit the flag")
	if *sf.contextBudgetTokens < 0 {
		writeConfigBail(os.Stderr, configBail{
			Verb:    "fak serve",
			Reason:  bailBudgetFlagIncoherent,
			Summary: "--context-budget-tokens must be non-negative",
			Knobs: []bailKnob{
				bailFlag("context-budget-tokens", strconv.Itoa(*sf.contextBudgetTokens)).want("0 to disable, or a positive token count"),
			},
		})
		os.Exit(2)
	}
	if *sf.resetOnBudget && *sf.contextBudgetTokens <= 0 {
		writeConfigBail(os.Stderr, configBail{
			Verb:    "fak serve",
			Reason:  bailBudgetFlagIncoherent,
			Summary: "--reset-on-budget has no budget to reset against",
			Knobs: []bailKnob{
				bailFlag("reset-on-budget", "true"),
				bailFlag("context-budget-tokens", strconv.Itoa(*sf.contextBudgetTokens)).want("a positive token count, or drop --reset-on-budget"),
			},
		})
		os.Exit(2)
	}
	// COLD resume (#629): re-attach the persisted drive state of every session BEFORE the
	// per-boot default-budget seed, so a restart resumes each session at the budget/
	// priority/run-state/pace it held — not its defaults — while an explicit
	// --context-budget-tokens on THIS boot still re-seeds the default trace. A STOPPED
	// session reloads STOPPED with its reason (session.Table.Restore), never silently
	// resurrected as RUNNING. A missing file is a clean first boot; a present-but-corrupt
	// file fails loud (a tampered drive record is worse than none).
	if err := restoreServeSessions(serveSessions, *sf.sessionStatePath); err != nil {
		fmt.Fprintln(os.Stderr, "fak serve:", err)
		os.Exit(1)
	}
	// The registry this resolves to IS the reach of every fanned lifecycle op: serveSessions is
	// hydrated from it, and `fak fleet control send --op pause --all` writes through that table.
	// A hard-coded "" here made --session-registry unreachable, so a serve armed with a private
	// --fleet-bus-dir still adopted every session on the host and paused peers' work (#5825).
	if err := configureServeSessionDurability(serveSessions, *sf.sessionRegistry, os.Stderr); err != nil {
		fmt.Fprintln(os.Stderr, "fak serve:", err)
		os.Exit(1)
	}

	defaultTraceID := strings.TrimSpace(*sf.sessionID)
	if *sf.contextBudgetTokens > 0 {
		if defaultTraceID == "" {
			defaultTraceID = "default"
		}
		serveSessions.SetBudget(defaultTraceID, session.Budget{
			TurnsLeft:         session.Unbounded,
			TokensLeft:        session.Unbounded,
			ContextTokensLeft: *sf.contextBudgetTokens,
		})
	}
	if err := registerServeSessionDurability(context.Background(), defaultTraceID); err != nil {
		fmt.Fprintln(os.Stderr, "fak serve:", err)
		os.Exit(1)
	}
	rt.apiKey, rt.engineCacheAdminKey, rt.requireKey, rt.defaultTraceID = apiKey, engineCacheAdminKey, requireKey, defaultTraceID
}

// resolveObservers resolves the optional operator webhook (#743) and the tiered
// stop-reason notifier (#761) into the two table observer seams; wireGateway
// installs them once the gateway exists.
func (rt *serveRuntime) resolveObservers(sf *serveFlags) {
	// Wire the optional operator webhook (#743) and the tiered stop-reason push notifier
	// (#761). The #743 budget webhook stays byte-identical when it is the only thing set:
	// combineBudgetObservers returns the lone observer unchanged, so WatchBudget is called
	// once exactly as before. The notifier (native default-on; webhook/Slack opt-in) adds a
	// SECOND budget fan-out plus the run-state TRANSITION observer that covers
	// PAUSED/DRAINING/STOPPED — the rest of the closed stop-reason vocabulary the budget seam
	// alone never sees. newNotifier returns nil when no sink is configured, leaving the
	// transition seam its byte-identical no-op default.
	notifier := newNotifier(*sf.notifyNative, os.Stderr, *sf.notifyWebhook, *sf.notifySlack)
	var transObs session.TransitionObserver
	if notifier != nil {
		transObs = notifier.transitionObserver()
	}
	var budgetObs session.BudgetObserver
	if obs := budgetWebhookObserver(*sf.budgetWebhook); obs != nil {
		budgetObs = obs
	}
	if notifier != nil {
		budgetObs = combineBudgetObservers(budgetObs, notifier.budgetObserver())
	}
	// The two table observer seams (transObs/budgetObs) are resolved here but INSTALLED below,
	// after srv exists: the #1095 slot-freed attach needs srv, and the scheduler's Attach owns the
	// table's single WatchTransitions/WatchBudget slots — so the install must be a single decision
	// (scheduler-takeover vs direct) that also has srv in hand. See the attach call after gateway.New.
	rt.transObs, rt.budgetObs = transObs, budgetObs
}

// wireGateway installs the observer seams and the KV reclaim/pressure edges over
// the live gateway, and streams drive-state revisions onto its revision ring.
func (rt *serveRuntime) wireGateway(sf *serveFlags) {
	// Slot-freed -> KV-free serve-path attach (#1095): with FAK_INKERNEL_KVMMU on, a
	// session.Scheduler is attached over the live serve table and the gateway's KV-reclaim edge
	// becomes its slot-freed observer, so a real drain/stop routes to rt.srv.ReclaimKVOnSlotFreed at
	// the next boundary — the FIRST non-test caller of wireSlotFreedKVReclaim + SetKVResidencyReclaimer.
	// The scheduler's Attach OWNS the table's single WatchTransitions/WatchBudget slots, so it takes
	// the resolved observers as pass-throughs (composing, never clobbering the notifier). When it
	// takes over (flag on), the direct Watch* installs are skipped; flag-off they run exactly as
	// before, so the served path is byte-identical until an operator opts in. The residency-backed
	// reclaimer it installs is nil today (servePathResidencyReclaimer) — the edge is reachable but
	// inert until the planner surfaces a trace-keyed residency to evict (#1074 / #987).
	if !attachServeSlotFreedReclaim(serveSessions, rt.srv, *sf.budgetWarnFraction, rt.budgetObs, rt.transObs) {
		if rt.transObs != nil {
			serveSessions.WatchTransitions(rt.transObs)
		}
		if rt.budgetObs != nil {
			serveSessions.WatchBudget(*sf.budgetWarnFraction, rt.budgetObs)
		}
	}

	// Install the post-decode capacity sweep over the complete native radix
	// snapshots the in-kernel planner actually owns. A direct native planner
	// exposes both the enumerable hot payloads and the host-DRAM stage/restore
	// backend; proxy/dual planners expose neither, so provider counters can never
	// be mistaken for fak-owned L2 bytes.
	// Seed the peer-DRAM lender roster from FAK_PEER_DRAM_LENDER (#5083) BEFORE the first
	// post-decode sweep runs probedTierProfilesForHost, so an operator declaration of a
	// neighbor's lendable DRAM makes the peer-DRAM-over-RDMA rung (#4306) reachable in this
	// live serve. Absent the env var this registers nothing and the rung stays out of the
	// ladder, byte-identical to today. (The live RDMA discovery transport is #3199.)
	if n := registerPeerDRAMLendersFromEnv(); n > 0 {
		rt.addStartupMessage(newServeStartupMessage("serve", "remote-dram", "info",
			fmt.Sprintf("registered %d peer-DRAM lender(s) from %s; remote-DRAM paging active", n, peerDRAMLenderEnvVar)))
	}

	prefixSource := rt.srv.InKernelKVPrefixPressureSource()
	if prefixSource != nil {
		store, configured, err := l3kv.ConfiguredRemoteStore()
		if err != nil {
			fmt.Fprintln(os.Stderr, "fak serve: native remote prefix L3:", err)
			os.Exit(1)
		}
		if configured {
			configurer, ok := prefixSource.(agent.KVPrefixRemoteConfigurer)
			if !ok {
				fmt.Fprintln(os.Stderr, "fak serve: native remote prefix L3: planner has no remote configuration seam")
				os.Exit(1)
			}
			if err := configurer.ConfigureKVPrefixRemote(store); err != nil {
				fmt.Fprintln(os.Stderr, "fak serve: native remote prefix L3:", err)
				os.Exit(1)
			}
			rt.addStartupMessage(newServeStartupMessage("serve", "prefix-store", "info",
				"native complete-prefix L3 -> l3kv/blobhttp (versioned, scoped, digest-verified)"))
		}
	}
	prefixBridge := newInKernelPrefixPressureBridge(prefixSource)
	if prefixBridge != nil {
		wireKVPressureRelief(rt.srv, rt.chatBackend, prefixBridge, prefixBridge)
	} else {
		wireKVPressureRelief(rt.srv, rt.chatBackend, nil, nil)
	}

	// Stream every drive-state revision on /v1/fak/session/changes (#630). Wired
	// AFTER gateway.New so rt.srv exists: each Rev bump of the process-local table
	// (a control verb, a debit, a continuation) is projected to the wire DTO and
	// pushed onto the gateway's bounded revision ring, where an operator drains it
	// by cursor — the live "what is every session doing right now" tail. The sink is
	// a cheap ring append and never re-enters the table (see session.RevisionObserver).
	serveSessions.WatchRevisions(func(s session.State) {
		rt.srv.PublishSessionRevision(toGatewaySessionState(s))
	})
}

// run serves until a terminating signal: it arms the route-manifest hot-reload
// watcher, the optional periodic usage snapshot, then blocks in stdio or HTTP
// mode and writes the shared shutdown tail (ledgers + session drive-state dump).
func (rt *serveRuntime) run(sf *serveFlags) {
	// Graceful drain on ANY terminating signal, not just Ctrl-C (#1359): SIGHUP is "the
	// terminal was closed" and SIGTERM is "an orchestrator asked us to stop" — both must
	// route through the same ctx-cancel → ListenAndServe-returns → dumpServeSessions flush
	// as SIGINT, or the most common "I closed the window" case (SIGHUP) silently loses the
	// live drive-state that had not been dumped on a prior clean exit. A SIGKILL (kill -9)
	// is uncatchable and still loses the un-journaled tail — that residue is the write-ahead
	// journal's job (#1363), not this signal handler's.
	ctx, stop := signal.NotifyContext(context.Background(), terminatingSignals()...)
	defer stop()

	// Hot-reload the routing policy (#842): when a manifest is installed, follow the
	// file and atomically swap the live policy on a validated edit — no restart. A
	// malformed edit is rejected and the last-good policy is kept (the fail-loud
	// startup contract extended to reload). The watcher reads the SAME atomic Live
	// the gateway classifies through, so a swap is visible on the hot path; it is
	// bound to ctx, so it stops with the server. Reloads/rejections are logged so an
	// operator can confirm the swap landed. armRouteHotReload also publishes that
	// watcher behind POST /v1/fak/route/reload (#4003), so an edit the poll loop's
	// size+mtime gate cannot see still has a production trigger — SIGHUP cannot be
	// that trigger here, it terminates (see the drain above).
	if watcher := armRouteHotReload(ctx, rt.srv, *sf.routeManifest); watcher != nil {
		// If --dojo is enabled, log the start of a live dojo episode.
		if *sf.dojoMode {
			if err := logDojoEpisodeStart("serve"); err != nil {
				fmt.Fprintf(os.Stderr, "fak: --dojo episode logging failed: %v (continuing without dojo)\n", err)
			}
		}
	}

	// #1610 (child B of epic #1601): the optional periodic gateway-usage snapshot runs
	// for the lifetime of ctx, appending an interim "periodic" counter row every
	// --metrics-snapshot tick so a crash before a clean exit still leaves a trail. Off
	// (0 duration, the default) is a byte-for-byte no-op; the exit-time "exit" row below
	// is always written regardless of this flag.
	stopMetricsSnapshot := startGatewayUsageSnapshotLoop(ctx, rt.srv, *sf.metricsSnapshot, "serve", rt.t0)
	defer stopMetricsSnapshot()

	// #5600 (epic #5599): with --fleet-bus, this serve joins the fleet control bus as an
	// INSTANCE — announcing presence and draining directives for the lifetime of ctx, so
	// a single `fak fleet control send` reaches it along with every peer. Off (the
	// default) is a byte-for-byte no-op. `*sf.native` is passed in rather than read off
	// the server because gateway.Server.ownsSessionLoop is unexported; it decides whether
	// a fanned `steer` can be delivered at all, and a non-native serve refuses it with the
	// same STEER_NO_OWNED_LOOP the single-session route refuses a 202 for.
	fleetIdentity := resolveFleetBusIdentity(sf)
	stopFleetBus := startFleetBusLoop(ctx, resolveFleetBusDir(sf), fleetIdentity.ID,
		*sf.fleetBusInterval, serveSessions, *sf.native, serveGwBusApplier{srv: rt.srv, addr: fleetIdentity.Addr})
	defer stopFleetBus()

	// Everything a finished serve must leave behind, whichever transport served it. The stdio
	// and HTTP exits below record the SAME four things and differ only in the transport word
	// they record it under, so the sequence lives here once: a new observation added for one
	// exit cannot be missing from the other, and a run that ends over stdio cannot silently
	// leave less evidence than one that ends over HTTP.
	persistServeExitObservations := func(transport string) {
		// Append the cache-value observation + the observed vcache window (#1072/#1075/#1090).
		persistCacheValueObservations(rt.srv, "serve", transport, *sf.provider)
		if *sf.dojoMode {
			_ = persistLiveDojoEpisode("serve", rt.srv)
		}
		// Append the full served-turn counter-family snapshot (#1610). The row is stamped
		// with a session id only when this process hosted exactly one session, so a
		// multiplexed gateway's total is never attributed to one of its traces — see
		// serveUsageSessionID.
		persistGatewayUsageObservation(rt.srv, "serve", transport, time.Since(rt.t0), serveUsageSessionID(serveSessions))
		dumpServeSessions(serveSessions, *sf.sessionStatePath) // #629: persist drive state for the next cold resume
	}

	if *sf.stdio {
		// MCP over stdio: stdout carries the protocol; the log package writes to
		// stderr, so diagnostics never corrupt the frames.
		if err := rt.srv.ServeStdio(ctx, os.Stdin, os.Stdout); err != nil && !errors.Is(err, context.Canceled) {
			must(err)
		}
		persistServeExitObservations("stdio")
		return
	}
	if *sf.addr == "" {
		writeConfigBail(os.Stderr, configBail{
			Verb:    "fak serve",
			Reason:  bailAddrRequired,
			Summary: "no transport: --addr is empty and --stdio was not passed",
			Knobs: []bailKnob{
				// --addr has a non-empty default, so an empty one was SET empty —
				// usually a shell variable that expanded to nothing.
				bailFlag("addr", "").want("HOST:PORT, e.g. 127.0.0.1:8080"),
				bailFlag("stdio", "false").want("true, to serve MCP over stdin/stdout instead"),
			},
			Check: "fak serve --help   # the transport flags and their defaults",
		})
		os.Exit(2)
	}
	// #3051/#3083: a local in-kernel GLM serve pays a one-time ~500s backend warmup
	// (weight load into VRAM, CUDA-graph capture, DeepGEMM/JIT compile) on its first
	// decode. Arm the readiness gate BEFORE binding the listener so /healthz reports
	// warmup_pending from the very first probe, then run a synthetic warm turn in the
	// background alongside ListenAndServe — the operator's first real turn is warm,
	// not a ~500s stall, and a client's cold-request timeout cannot cancel a
	// legitimate warmup by racing an early ready mark. The condition mirrors the one
	// serve.go uses to route chat in-kernel (local model, no upstream base-url, no
	// replica fleet); a proxy/replica serve pays no such tax and skips warm-start.
	if rt.inKernelModel != nil && rt.inKernelTok != nil && strings.TrimSpace(*sf.baseURL) == "" && len(sf.replicaBaseURLs.Values()) == 0 {
		rt.srv.ArmWarmupGate()
		go func() { _, _ = rt.srv.RunWarmup(ctx) }()
	}
	if err := rt.srv.ListenAndServe(ctx, *sf.addr); err != nil && !errors.Is(err, http.ErrServerClosed) {
		must(err)
	}
	persistServeExitObservations("http")
}

func resolveServeRequiredKey(envName, flagName, summary, want string) string {
	key, ok := resolveRequiredKey(envName, os.Getenv)
	if ok {
		return key
	}
	writeConfigBail(os.Stderr, configBail{
		Verb: "fak serve", Reason: bailKeyEnvUnset, Summary: summary,
		Knobs: []bailKnob{bailFlag(flagName, envName), bailEnv(envName, "").want(want)},
		Bind:  []string{"env=" + envName},
	})
	os.Exit(2)
	return ""
}
