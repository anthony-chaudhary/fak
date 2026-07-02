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
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/gateway"
	"github.com/anthony-chaudhary/fak/internal/ggufload"
	"github.com/anthony-chaudhary/fak/internal/hfhub"
	fakmodel "github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/modelreg"
	"github.com/anthony-chaudhary/fak/internal/modelroute"
	"github.com/anthony-chaudhary/fak/internal/pathutil"
	"github.com/anthony-chaudhary/fak/internal/policy"
	"github.com/anthony-chaudhary/fak/internal/session"
	"github.com/anthony-chaudhary/fak/internal/tokenizer"
)

// serveRuntime carries the state the serve boot stages resolve on the way to a
// bound listener: the compute plane, the resident model, the session/auth
// material, the observer seams, and finally the gateway itself.
type serveRuntime struct {
	t0            time.Time
	startupPhases []gateway.StartupPhase

	chatBackend compute.Backend
	useMetal    bool
	ep          epRankConfig

	inKernelModel *fakmodel.Model
	inKernelQ4K   bool
	loadProfile   *gateway.ModelLoadProfile
	epGroup       *fakmodel.DistComm
	inKernelTok   *tokenizer.Tokenizer

	apiKey              string
	engineCacheAdminKey string
	requireKey          string
	defaultTraceID      string

	transObs  session.TransitionObserver
	budgetObs session.BudgetObserver

	srv *gateway.Server
}

// resolveServeModelSources normalizes the --gguf/--tokenizer sources before any
// stage touches them: ~ expansion, registry alias resolution, and hf:// fetch.
func resolveServeModelSources(sf *serveFlags) {
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
			fmt.Fprintf(os.Stderr, "fak serve: --gguf %s → %s\n", *sf.ggufPath, resolved)
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
		fmt.Fprintln(os.Stderr, "fak serve: --policy-check requires --policy FILE")
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
		fmt.Fprintln(os.Stderr, err)
		os.Exit(2)
	}
	if chatBackend != nil {
		fmt.Printf("fak: in-kernel chat decode → device backend %q\n", chatBackend.Name())
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
		fmt.Println("fak: in-kernel chat decode → Apple-Silicon Metal GPU (prefill + resident Q8 decode)")
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
		fmt.Printf("fak: CUDA-graph decode replay enabled (#483), KV graph capacity=%d positions — witness tok/s before relying on it\n", max(*sf.contextBudgetTokens, 1024))
	}
	rt.chatBackend, rt.useMetal, rt.ep = chatBackend, useMetal, ep
}

// loadModel eagerly loads the GGUF weights and the in-kernel tokenizer before the
// listener binds. For a sharded expert-parallel rank it also dials the process
// group and wires the rank-local forward; the dialed group lands on rt.epGroup and
// is closed by cmdServe's deferred closeEPGroup.
func (rt *serveRuntime) loadModel(sf *serveFlags) {
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
		fmt.Printf("fak: expert-parallel rank %d/%d loads experts [%d,%d) of %d resident (sharded serve, #971)\n", rt.ep.rank, rt.ep.ranks, shard.Lo, shard.Hi, numExperts)
	}
	inKernelModel, inKernelQ4K, loadProfile, loadPhase := loadServeInKernelModel(*sf.ggufPath, rt.chatBackend, *sf.cpuOffloadExperts, *sf.contextBudgetTokens, expertShard)
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
		devColl, devErr := joinDevicePGIfSupported(rt.chatBackend, group, rt.ep)
		if devErr != nil {
			fmt.Printf("fak: expert-parallel rank %d/%d: device-NCCL process group unavailable (%v) — falling back to host DistComm reduce\n", rt.ep.rank, rt.ep.ranks, devErr)
		}
		if devColl != nil {
			inKernelModel.SetExpertParallelCollective(devColl)
			fmt.Printf("fak: expert-parallel rank %d/%d joined the process group (device-NCCL tensor reduce, #971 follow-on)\n", rt.ep.rank, rt.ep.ranks)
		} else {
			inKernelModel.SetExpertParallelCollective(fakmodel.NewDistCommCollective(group))
			fmt.Printf("fak: expert-parallel rank %d/%d joined the process group (host DistComm reduce, #971) — device-NCCL tensor rung stays separate\n", rt.ep.rank, rt.ep.ranks)
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

	inKernelTok, tokLoaded := resolveServeTokenizer(*sf.tokPath, *sf.ggufPath)
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
}

// resolveSessionPlane resolves the auth/key material, validates the budget flags,
// cold-resumes persisted session drive state (#629), and seeds the default trace's
// budget and durability registration.
func (rt *serveRuntime) resolveSessionPlane(sf *serveFlags) {
	apiKey := ""
	if *sf.apiKeyEnv != "" {
		apiKey = os.Getenv(*sf.apiKeyEnv)
	}
	engineCacheAdminKey, ok := resolveRequiredKey(*sf.engineCacheAdminKeyEnv, os.Getenv)
	if !ok {
		fmt.Fprintf(os.Stderr, "fak serve: --engine-cache-admin-key-env %s is set but unset/empty — refusing to send cache-reset requests with NO admin auth (set the secret or omit the flag)\n", *sf.engineCacheAdminKeyEnv)
		os.Exit(2)
	}
	if *sf.engineCacheIdleTimeout < 0 {
		fmt.Fprintln(os.Stderr, "fak serve: --engine-cache-idle-timeout must be non-negative")
		os.Exit(2)
	}
	requireKey, ok := resolveRequiredKey(*sf.requireKeyEnv, os.Getenv)
	if !ok {
		fmt.Fprintf(os.Stderr, "fak serve: --require-key-env %s is set but unset/empty — refusing to start a network-facing gateway with NO authentication (set the secret or omit the flag)\n", *sf.requireKeyEnv)
		os.Exit(2)
	}
	if *sf.contextBudgetTokens < 0 {
		fmt.Fprintln(os.Stderr, "fak serve: --context-budget-tokens must be non-negative")
		os.Exit(2)
	}
	if *sf.resetOnBudget && *sf.contextBudgetTokens <= 0 {
		fmt.Fprintln(os.Stderr, "fak serve: --reset-on-budget requires --context-budget-tokens N")
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
	if err := configureServeSessionDurability(serveSessions, "", os.Stderr); err != nil {
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

	// Install the #1073 post-decode KV pressure-relief sweep (#1094): the LIVE, non-test caller
	// of SetKVPressureRelief. The sweeper closure wraps the genuine engine capacity executor over
	// the live device backend; the gateway gates the edge on FAK_INKERNEL_KVMMU AND on a non-nil
	// provider, so installing it unconditionally is safe — with no provider the edge stays inert
	// (a no-op, byte-identical to today). The PROVIDER (the resident-span enumerator over
	// kvmmu.Segment{From,Len,KV}) is nil here because no durable cross-turn resident-span ledger
	// exists yet — the in-kernel planner builds a kvmmu.Context ephemerally per eviction and keeps
	// only a radixkv prefix-reuse tree, not enumerable per-span candidates. Building that
	// enumerator is the fenced follow-on #1074 / #987; when it lands, serve.go passes it here
	// instead of nil and the sweep fires on real residency with no other change. KV is nil for the
	// same reason (a nil-provider sweep never calls it). See wireKVPressureRelief's honest fence.
	wireKVPressureRelief(rt.srv, rt.chatBackend, nil, nil)

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
	// operator can confirm the swap landed.
	if live := rt.srv.RouteLive(); live != nil {
		watcher := modelroute.NewWatcher(*sf.routeManifest, live, 0, func(ev modelroute.ReloadEvent) {
			if ev.Err != nil {
				fmt.Fprintf(os.Stderr, "fak: route-manifest reload REJECTED: %v\n", ev.Err)
				return
			}
			if ev.Reloaded {
				fmt.Fprintf(os.Stderr, "fak: model-routing policy hot-reloaded from %s (reload #%d)\n", *sf.routeManifest, ev.Reloads)
			}
		})
		go func() { _ = watcher.Run(ctx) }()

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
	stopMetricsSnapshot := startGatewayUsageSnapshotLoop(ctx, rt.srv, *sf.metricsSnapshot, "serve")
	defer stopMetricsSnapshot()

	if *sf.stdio {
		// MCP over stdio: stdout carries the protocol; the log package writes to
		// stderr, so diagnostics never corrupt the frames.
		if err := rt.srv.ServeStdio(ctx, os.Stdin, os.Stdout); err != nil && !errors.Is(err, context.Canceled) {
			must(err)
		}
		// Append the cache-value observation + the observed vcache window (#1072/#1075/#1090).
		persistCacheValueObservations(rt.srv, "serve", "stdio", *sf.provider)
		if *sf.dojoMode {
			_ = persistLiveDojoEpisode("serve", rt.srv)
		}
		// Append the full served-turn counter-family snapshot (#1610).
		persistGatewayUsageObservation(rt.srv, "serve", "stdio")
		dumpServeSessions(serveSessions, *sf.sessionStatePath) // #629: persist drive state for the next cold resume
		return
	}
	if *sf.addr == "" {
		fmt.Fprintln(os.Stderr, "fak serve: --addr is required (or pass --stdio)")
		os.Exit(2)
	}
	if err := rt.srv.ListenAndServe(ctx, *sf.addr); err != nil && !errors.Is(err, http.ErrServerClosed) {
		must(err)
	}
	// Append the cache-value observation + the observed vcache window (#1072/#1075/#1090).
	persistCacheValueObservations(rt.srv, "serve", "http", *sf.provider)
	if *sf.dojoMode {
		_ = persistLiveDojoEpisode("serve", rt.srv)
	}
	// Append the full served-turn counter-family snapshot (#1610).
	persistGatewayUsageObservation(rt.srv, "serve", "http")
	dumpServeSessions(serveSessions, *sf.sessionStatePath) // #629: persist drive state for the next cold resume
}
