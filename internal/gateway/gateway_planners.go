package gateway

import (
	"errors"
	"fmt"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/enginecache"
	"github.com/anthony-chaudhary/fak/internal/model"
)

func newEngineCacheClient(cfg Config) (*enginecache.Client, error) {
	engineName := strings.ToLower(strings.TrimSpace(cfg.EngineCacheEngine))
	baseURL := strings.TrimSpace(cfg.EngineCacheBaseURL)
	if engineName == "" && baseURL == "" && strings.TrimSpace(cfg.EngineCacheAdminKey) == "" && cfg.EngineCacheIdleTimeout == 0 && !cfg.EngineCacheRequireExactSpan {
		return nil, nil
	}
	if engineName == "" {
		return nil, errors.New("gateway: engine cache reset requires EngineCacheEngine (sglang|vllm)")
	}
	if baseURL == "" {
		urls, err := proxyBaseURLs(cfg)
		if err != nil {
			return nil, err
		}
		if len(urls) > 1 {
			return nil, errors.New("gateway: engine cache reset with replica base URLs requires EngineCacheBaseURL")
		}
		if len(urls) == 1 {
			// urls[0] may carry a name=URL identity; the cache-control target is the URL.
			_, baseURL = parseReplicaEntry(urls[0])
		}
	}
	if baseURL == "" {
		return nil, errors.New("gateway: engine cache reset requires EngineCacheBaseURL or BaseURL")
	}
	engine := enginecache.Engine(engineName)
	switch engine {
	case enginecache.EngineSGLang, enginecache.EngineVLLM:
	default:
		return nil, fmt.Errorf("gateway: unsupported engine cache engine %q (want sglang|vllm)", cfg.EngineCacheEngine)
	}
	requiredScope := ""
	if cfg.EngineCacheRequireExactSpan {
		requiredScope = enginecache.ScopeExactSpan
	}
	return &enginecache.Client{
		Engine:        engine,
		BaseURL:       baseURL,
		AdminAPIKey:   cfg.EngineCacheAdminKey,
		IdleTimeout:   cfg.EngineCacheIdleTimeout,
		RequiredScope: requiredScope,
	}, nil
}

// HasProxyUpstream reports whether the gateway will select an upstream planner.
// It shares the planner's endpoint collection rules; provider validation remains
// part of planner construction.
func HasProxyUpstream(cfg Config) (bool, error) {
	urls, err := proxyBaseURLs(cfg)
	return len(urls) != 0, err
}

func proxyBaseURLs(cfg Config) ([]string, error) {
	urls := make([]string, 0, 1+len(cfg.ReplicaBaseURLs))
	if base := strings.TrimSpace(cfg.BaseURL); base != "" {
		urls = append(urls, base)
	}
	for i, base := range cfg.ReplicaBaseURLs {
		base = strings.TrimSpace(base)
		if base == "" {
			return nil, fmt.Errorf("gateway: replica base URL %d is empty", i+1)
		}
		urls = append(urls, base)
	}
	return urls, nil
}

// newInKernelChatPlanner builds the in-kernel chat planner (the model fused into the
// kernel) advertising modelID, wiring expert parallelism onto the model exactly as the
// single-planner path always has. Shared by the pure in-kernel case and the dual
// (local-alongside-API) case so the EP semantics cannot drift between them.
// Expert parallelism is model state, set on the in-kernel Model here (the EP rank
// lives on the Model, consumed by ffnForLayer); 0/1 is the no-op default.
func newInKernelChatPlanner(cfg Config, modelID string, logf func(string, ...any)) agent.Planner {
	if cfg.ExpertParallelRanks > 1 && !cfg.InKernelModel.IsExpertParallelRankLocal() {
		cfg.InKernelModel.SetExpertParallelRanks(cfg.ExpertParallelRanks)
		// Reduce the routed-expert partials through the DEVICE collective the serve
		// initialized — serve.go gates ranks>1 on a backend advertising Caps().Collective
		// (the NCCL CollectiveBackend), so the decode AllReduceSum must cross those GPUs,
		// not the hardcoded single-box LocalCollective glmMoeEPFFN reduced through before.
		// On cpu-ref the bridge is byte-identical to LocalCollective (collective_bridge_test.go),
		// so this changes no host-tested bytes; on the NCCL backend the SAME call all-reduces
		// across the rank fleet. Fail-soft: a backend without the seam leaves the bit-exact
		// LocalCollective default (the EP output stays correct, just reduced host-side).
		if cfg.Backend != nil {
			if err := cfg.InKernelModel.SetExpertParallelDeviceCollective(cfg.Backend); err == nil {
				logf("gateway: expert-parallel ranks=%d → routed-expert AllReduceSum reduces through device collective %q (Caps().Collective=%v)", cfg.ExpertParallelRanks, cfg.Backend.Name(), cfg.Backend.Caps().Collective)
			} else {
				logf("gateway: expert-parallel ranks=%d: backend %q exposes no device collective (%v) — reducing host-side via LocalCollective (correct, single-box)", cfg.ExpertParallelRanks, cfg.Backend.Name(), err)
			}
		}
	} else if cfg.ExpertParallelRanks > 1 && cfg.InKernelModel.IsExpertParallelRankLocal() {
		// A SHARDED EP rank: the serve already set the rank, the world size, and the DistComm
		// process-group collective (each rank holds only its band, reduces cross-process). Do
		// NOT re-wire a single-process device/Local collective here — it would clobber the
		// cross-process reduce and break the sharded serve (#971).
		logf("gateway: expert-parallel ranks=%d rank-local (sharded serve) — reducing through the serve's DistComm process group, device-collective wiring skipped", cfg.ExpertParallelRanks)
	}
	plannerCfg := cfg.InKernelPlanner
	plannerCfg.CPUOffloadExperts = cfg.CPUOffloadExperts
	if plannerCfg.CompactHistoryBudget == 0 && cfg.CompactHistoryBudget > 0 {
		plannerCfg.CompactHistoryBudget = cfg.CompactHistoryBudget
	}
	if !plannerCfg.ElideStaleReads && cfg.ElideStaleReads {
		plannerCfg.ElideStaleReads = cfg.ElideStaleReads
	}
	if !plannerCfg.DeferColdTools && cfg.DeferColdTools {
		plannerCfg.DeferColdTools = cfg.DeferColdTools
	}
	ikp := agent.NewInKernelPlannerWithConfig(cfg.InKernelModel, cfg.Tokenizer, modelID, cfg.InKernelQ4K, cfg.Backend, cfg.Metal, plannerCfg)
	if shouldEnableMetalMTP(cfg) {
		if err := ikp.EnableMetalMTP(); err != nil {
			logf("gateway: failed to enable Metal MTP coordinator: %v", err)
		}
	}
	if shouldEnableVulkanMTP(cfg) {
		if err := ikp.EnableVulkanMTP(model.Qwen35MTPMaxDraftDepth); err != nil {
			logf("gateway: failed to enable resident Vulkan MTP: %v", err)
		} else {
			logf("gateway: enabled request-bound resident Vulkan MTP with resident target verification")
		}
	} else if shouldEnableNGramSpeculative(cfg) {
		ikp.EnableSpeculativeDecoding(model.NewNGramProposalGenerator(model.NgramDrafter{
			Enabled:  true,
			MaxDraft: 4,
		}), 4)
		logf("gateway: enabled prompt n-gram speculation with resident Vulkan target verification")
	}
	return ikp
}

func newProxyPlanner(cfg Config, model string, baseURLs []string) (agent.Planner, error) {
	if len(baseURLs) == 1 {
		// A lone upstream needs no replica identity, but still honor a name=URL form
		// (an operator may pin one) by dialing only the URL part.
		_, dialURL := parseReplicaEntry(baseURLs[0])
		return newConfiguredHTTPPlanner(cfg, model, dialURL)
	}
	replicas := make([]PlannerReplica, 0, len(baseURLs))
	for _, base := range baseURLs {
		// Stable, order-independent identity (#3968): use the operator-chosen id from a
		// name=URL entry, else derive replica-<digest> from the endpoint so the same
		// upstream keeps one identity — and one set of metric/residency labels — no
		// matter its flag position or a membership change that drops a peer.
		name, dialURL := parseReplicaEntry(base)
		if name == "" {
			name = deriveReplicaName(dialURL)
		}
		p, err := newConfiguredHTTPPlanner(cfg, model, dialURL)
		if err != nil {
			return nil, err
		}
		replicas = append(replicas, PlannerReplica{
			Name:     name,
			Planner:  p,
			Endpoint: dialURL,
		})
	}
	router, err := NewReplicaDispatch(model, replicas)
	if err != nil {
		return nil, err
	}
	router.Hedge = cfg.HedgePolicy
	// Build the live health/drain registry over exactly these replicas and stash it
	// UNARMED (issue fak-private#2417): the router keeps its blind round-robin until
	// the Serve-run fleet-health loop probes once and arms it (runFleetHealthLoop), so
	// construction stays free of network I/O and a request racing startup is not
	// answered from a not-yet-probed roster. A failure to register is impossible after
	// NewReplicaDispatch validated unique, non-empty replica names, but it is surfaced
	// rather than dropped.
	fm, err := buildReplicaMembership(replicas, model)
	if err != nil {
		return nil, err
	}
	router.fleet = fm
	return router, nil
}

// replicaDispatchOf returns the ReplicaDispatch reachable from p — p itself, or the proxy
// side of a DualPlanner — or nil when the deployment is not a replica fleet (a lone
// upstream, the in-kernel model, or the mock). It is how New recovers the fleet handle
// newProxyPlanner stashed on the router, without widening newProxyPlanner's signature.
func replicaDispatchOf(p agent.Planner) *ReplicaDispatch {
	switch v := p.(type) {
	case *ReplicaDispatch:
		return v
	case *DualPlanner:
		if rr, ok := v.Proxy().(*ReplicaDispatch); ok {
			return rr
		}
	}
	return nil
}

// newConfiguredHTTPPlanner dials one upstream and applies every Config-derived knob to
// the resulting planner — the wiring newProxyPlanner performs once for a lone upstream
// and once per replica, so a new Config field is threaded through in exactly one place.
func newConfiguredHTTPPlanner(cfg Config, model, dialURL string) (*agent.HTTPPlanner, error) {
	p, err := agent.NewProviderHTTPPlanner(cfg.Provider, dialURL, model, cfg.APIKey)
	if err != nil {
		return nil, err
	}
	p.APIKeyFunc = cfg.APIKeyFunc
	p.AccountFailoverFunc = cfg.AccountFailoverFunc
	p.TransientTargetFunc = cfg.TransientTargetFunc
	p.ExtraHeaders = cloneConfigHeaders(cfg.ExtraHeaders)
	p.ExtraHeadersFunc = cfg.ExtraHeadersFunc
	p.ForceResponsesStream = cfg.ForceResponsesStream
	// Passed through verbatim: Config.StreamProgressTimeout carries the planner field's own
	// encoding (0 = the agent default, negative = disabled, out-of-band = the default), so a
	// Config nobody configures leaves the planner at the 300s default byte-for-byte.
	p.StreamProgressTimeout = cfg.StreamProgressTimeout
	// #10638: the soft diagnostic deadline rides through the SAME verbatim encoding as the
	// hard window above (0 = derive from hard, negative = off, positive = honored when earlier
	// than hard), resolved by streamSoftProgressWindow at arm time.
	p.StreamSoftProgressTimeout = cfg.StreamSoftProgressTimeout
	p.LlamaSlotAffinity = cfg.LlamaSlotAffinity
	p.LlamaSoftSlot = cfg.LlamaSoftSlot
	p.LlamaSoftSlotPolicy = cfg.LlamaSoftSlotPolicy
	wrapUpstreamObserver(p.Client, cfg.UpstreamResponseObserver, cfg.UpstreamTransportErrorObserver, cfg.UpstreamFailureObserver)
	return p, nil
}

func cloneConfigHeaders(in map[string]string) map[string]string {
	if len(in) == 0 {
		return nil
	}
	out := make(map[string]string, len(in))
	for k, v := range in {
		if strings.TrimSpace(k) != "" {
			out[k] = v
		}
	}
	return out
}
