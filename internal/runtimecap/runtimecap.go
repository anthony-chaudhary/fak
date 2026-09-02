package runtimecap

import (
	"runtime"
	"runtime/debug"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

const Schema = "fak-runtime-capabilities/1"

const (
	PlacementLocalOnly     = "local_only"
	PlacementPreferLocal   = "prefer_local"
	PlacementRemoteAllowed = "remote_allowed"
)

const (
	FallbackPolicyPinOrRefuse     = "pin_or_refuse"
	FallbackPolicyLocalCPUDegrade = "local_cpu_degraded"

	payloadCompatibilityNotChecked = "not_checked"
	payloadCompatibilitySupported  = "supported"
	payloadCompatibilityRefused    = "refused"

	executionModeStandard         = "standard"
	executionModeLocalCPUDegraded = FallbackPolicyLocalCPUDegrade
	executionModeRemote           = "remote"
)

type Reason struct {
	Code        string `json:"code"`
	Detail      string `json:"detail"`
	Remediation string `json:"remediation,omitempty"`
}

type Backend struct {
	Name       string `json:"name"`
	Registered bool   `json:"registered"`
	Class      string `json:"class,omitempty"`
	Tier       string `json:"tier,omitempty"`
}

type RequestedBackend struct {
	Name       string  `json:"name"`
	Status     string  `json:"status"`
	ExactMatch bool    `json:"exact_match"`
	Selected   string  `json:"selected,omitempty"`
	Reason     *Reason `json:"reason,omitempty"`
}

type HostMemory struct {
	Known      bool  `json:"known"`
	TotalBytes int64 `json:"total_bytes,omitempty"`
	FreeBytes  int64 `json:"free_bytes,omitempty"`
	FreeKnown  bool  `json:"free_known"`
}

type PayloadDemand struct {
	Class  string `json:"class"`
	Bytes  int64  `json:"bytes"`
	Detail string `json:"detail,omitempty"`
}

type CPUEnvelope struct {
	ID                          string          `json:"id"`
	Model                       string          `json:"model"`
	Quantization                string          `json:"quantization"`
	Engine                      string          `json:"engine"`
	Backend                     string          `json:"backend"`
	GOOS                        string          `json:"goos"`
	GOARCH                      string          `json:"goarch"`
	MinimumRAMBytes             int64           `json:"minimum_ram_bytes"`
	MinimumDiskBytes            int64           `json:"minimum_disk_bytes"`
	HeadroomRatio               float64         `json:"headroom_ratio"`
	ExpectedDecodeTokPerSec     float64         `json:"expected_decode_tok_per_sec,omitempty"`
	ExpectedPrefill256TokPerSec float64         `json:"expected_prefill_p256_tok_per_sec,omitempty"`
	PerformanceClass            string          `json:"performance_class"`
	QualityEquivalence          string          `json:"quality_equivalence"`
	Witness                     string          `json:"witness"`
	Demands                     []PayloadDemand `json:"demands"`
}

type LocalCPUDegradedReceipt struct {
	Mode               string     `json:"mode"`
	Policy             string     `json:"policy"`
	RequestedBackend   string     `json:"requested_backend,omitempty"`
	SelectedBackend    string     `json:"selected_backend"`
	Reason             *Reason    `json:"reason,omitempty"`
	EnvelopeID         string     `json:"envelope_id"`
	Model              string     `json:"model"`
	Quantization       string     `json:"quantization"`
	PerformanceClass   string     `json:"performance_class"`
	QualityEquivalence string     `json:"quality_equivalence"`
	HostMemory         HostMemory `json:"host_memory"`
}

type RemoteGate struct {
	State  string `json:"state"`
	Detail string `json:"detail,omitempty"`
}

type RemoteCredentialGate struct {
	Name    string `json:"name,omitempty"`
	Present bool   `json:"present"`
}

type RemotePlacementReceipt struct {
	Mode                  string               `json:"mode"`
	ControlPlaneOwner     string               `json:"control_plane_owner"`
	Target                string               `json:"target"`
	AuthorizedTarget      string               `json:"authorized_target"`
	Provider              string               `json:"provider"`
	Engine                string               `json:"engine"`
	Model                 string               `json:"model"`
	EndpointClass         string               `json:"endpoint_class,omitempty"`
	Region                string               `json:"region,omitempty"`
	LocalFailureTrigger   *Reason              `json:"local_failure_trigger"`
	StateCrossingBoundary []string             `json:"state_crossing_boundary"`
	Egress                RemoteGate           `json:"egress"`
	Credential            RemoteCredentialGate `json:"credential"`
	TLS                   RemoteGate           `json:"tls"`
	Proxy                 RemoteGate           `json:"proxy"`
	Reachability          RemoteGate           `json:"reachability"`
	TimeoutMilliseconds   int64                `json:"timeout_milliseconds"`
	RetryCeiling          int                  `json:"retry_ceiling"`
	BudgetMicroUSD        int64                `json:"budget_micro_usd"`
}

type Execution struct {
	Runnable             bool                     `json:"runnable"`
	Engine               string                   `json:"engine,omitempty"`
	Backend              string                   `json:"backend,omitempty"`
	Mode                 string                   `json:"mode,omitempty"`
	CPUEnvelope          string                   `json:"cpu_envelope,omitempty"`
	PayloadLoaded        bool                     `json:"payload_loaded"`
	PayloadCompatibility string                   `json:"payload_compatibility"`
	LocalCPUDegraded     *LocalCPUDegradedReceipt `json:"local_cpu_degraded,omitempty"`
	RemotePlacement      *RemotePlacementReceipt  `json:"remote_placement,omitempty"`
	Reason               *Reason                  `json:"reason,omitempty"`
}

type Report struct {
	Schema                string            `json:"schema"`
	GOOS                  string            `json:"goos"`
	GOARCH                string            `json:"goarch"`
	BinaryRunnable        bool              `json:"binary_runnable"`
	ControlPlaneRunnable  bool              `json:"control_plane_runnable"`
	BuildTags             []string          `json:"build_tags"`
	RegisteredBackends    []Backend         `json:"registered_backends"`
	PortableCPU           Backend           `json:"portable_cpu_reference"`
	HostMemory            HostMemory        `json:"host_memory"`
	RequestedBackend      *RequestedBackend `json:"requested_backend,omitempty"`
	PreferredBackend      *RequestedBackend `json:"preferred_backend,omitempty"`
	CPUFallbackPolicy     string            `json:"cpu_fallback_policy,omitempty"`
	PlacementMode         string            `json:"placement_mode"`
	SupportedCPUEnvelopes []CPUEnvelope     `json:"supported_cpu_envelopes,omitempty"`
	ModelExecution        Execution         `json:"model_execution"`
}

type Options struct {
	RequestedBackend          string
	PreferredBackend          string
	CPUFallbackPolicy         string
	CPUEnvelope               string
	GOOS                      string
	GOARCH                    string
	BuildTags                 []string
	Backends                  []compute.Backend
	HostMemory                HostMemory
	HostMemoryOverride        bool
	PlacementMode             string
	RemoteTarget              string
	AuthorizedTarget          string
	RemoteProvider            string
	RemoteEngine              string
	RemoteModel               string
	RemoteEndpointClass       string
	RemoteRegion              string
	RemoteStateBoundary       []string
	RemoteEgress              string
	RemoteCredentialName      string
	RemoteCredentialPresent   bool
	RemoteTLS                 string
	RemoteProxy               string
	RemoteReachability        string
	RemoteTimeoutMilliseconds int64
	RemoteRetryCeiling        int
	RemoteBudgetMicroUSD      int64
}

func Probe(opts Options) Report {
	goos, goarch := opts.GOOS, opts.GOARCH
	if goos == "" {
		goos = runtime.GOOS
	}
	if goarch == "" {
		goarch = runtime.GOARCH
	}
	tags := append([]string(nil), opts.BuildTags...)
	if opts.BuildTags == nil {
		tags = currentBuildTags()
	}
	sort.Strings(tags)

	backends := opts.Backends
	if backends == nil {
		for _, name := range compute.Registered() {
			if backend, ok := compute.Lookup(name); ok {
				backends = append(backends, backend)
			}
		}
	}
	records := make([]Backend, 0, len(backends))
	byName := make(map[string]compute.Backend, len(backends))
	for _, backend := range backends {
		if backend == nil {
			continue
		}
		byName[backend.Name()] = backend
		records = append(records, backendRecord(backend))
	}
	sort.Slice(records, func(i, j int) bool { return records[i].Name < records[j].Name })

	cpu := Backend{Name: "cpu-ref"}
	if backend, ok := byName["cpu-ref"]; ok {
		cpu = backendRecord(backend)
	}
	hostMemory := effectiveHostMemory(opts)
	policy := normalizeFallbackPolicy(opts.CPUFallbackPolicy)
	placement := normalizePlacementMode(opts.PlacementMode)
	report := Report{
		Schema:                Schema,
		GOOS:                  goos,
		GOARCH:                goarch,
		BinaryRunnable:        true,
		ControlPlaneRunnable:  true,
		BuildTags:             tags,
		RegisteredBackends:    records,
		PortableCPU:           cpu,
		HostMemory:            hostMemory,
		CPUFallbackPolicy:     policy,
		PlacementMode:         placement,
		SupportedCPUEnvelopes: supportedCPUEnvelopes(),
		ModelExecution: Execution{
			PayloadLoaded:        false,
			PayloadCompatibility: payloadCompatibilityNotChecked,
		},
	}

	requested := strings.TrimSpace(opts.RequestedBackend)
	preferred := strings.TrimSpace(opts.PreferredBackend)
	if requested != "" && preferred != "" {
		reason := Reason{
			Code:        "conflicting_backend_requests",
			Detail:      "--backend is exact and cannot be combined with --prefer-backend",
			Remediation: "choose either an exact backend pin or a preferred backend with explicit fallback policy",
		}
		report.RequestedBackend = &RequestedBackend{Name: requested, Status: reasonStatus(reason), Reason: &reason}
		report.PreferredBackend = &RequestedBackend{Name: preferred, Status: reasonStatus(reason), Reason: &reason}
		report.ModelExecution.Reason = &reason
		return report
	}

	if requested != "" {
		request := &RequestedBackend{Name: requested}
		report.RequestedBackend = request
		if backend, ok := byName[requested]; ok {
			request.Status, request.ExactMatch, request.Selected = "available", true, backend.Name()
			report.ModelExecution = executionForBackend(backend, opts.CPUEnvelope, goos, goarch, hostMemory, report.SupportedCPUEnvelopes, policy)
			return report
		}
		reason := unavailableReason(requested, goos, goarch, tags)
		request.Status, request.Reason = reasonStatus(reason), &reason
		report.ModelExecution.Reason = &reason
		return report
	}

	if preferred != "" {
		request := &RequestedBackend{Name: preferred}
		report.PreferredBackend = request
		if backend, ok := byName[preferred]; ok {
			request.Status, request.ExactMatch, request.Selected = "available", true, backend.Name()
			report.ModelExecution = executionForBackend(backend, opts.CPUEnvelope, goos, goarch, hostMemory, report.SupportedCPUEnvelopes, policy)
			return report
		}
		reason := unavailableReason(preferred, goos, goarch, tags)
		request.Status, request.Reason = reasonStatus(reason), &reason
		if placement == PlacementRemoteAllowed {
			report.ModelExecution = applyRemotePlacement(opts, reason)
			if report.ModelExecution.Runnable {
				request.Selected = "remote:" + strings.TrimSpace(opts.RemoteTarget)
			}
			return report
		}
		if policy != FallbackPolicyLocalCPUDegrade {
			report.ModelExecution.Reason = &reason
			return report
		}
		cpuBackend, ok := byName["cpu-ref"]
		if !ok {
			report.ModelExecution.Reason = &Reason{
				Code:        "portable_cpu_reference_unavailable",
				Detail:      "the portable fak-native CPU reference backend is not registered",
				Remediation: "use an official fak build that includes internal/compute/cpuref",
			}
			return report
		}
		report.ModelExecution = applyCPUEnvelope(availableExecution(cpuBackend), goos, goarch, hostMemory, opts.CPUEnvelope, report.SupportedCPUEnvelopes, preferred, &reason, policy)
		if report.ModelExecution.Runnable {
			request.Selected = "cpu-ref"
		}
		return report
	}

	if backend, ok := byName["cpu-ref"]; ok {
		report.ModelExecution = executionForBackend(backend, opts.CPUEnvelope, goos, goarch, hostMemory, report.SupportedCPUEnvelopes, policy)
	} else {
		report.ModelExecution.Reason = &Reason{
			Code:        "portable_cpu_reference_unavailable",
			Detail:      "the portable fak-native CPU reference backend is not registered",
			Remediation: "use an official fak build that includes internal/compute/cpuref",
		}
	}
	return report
}

func normalizePlacementMode(mode string) string {
	mode = strings.TrimSpace(mode)
	if mode == "" {
		return PlacementLocalOnly
	}
	return mode
}

func applyRemotePlacement(opts Options, localFailure Reason) Execution {
	target := strings.TrimSpace(opts.RemoteTarget)
	authorized := strings.TrimSpace(opts.AuthorizedTarget)
	proxy := strings.TrimSpace(opts.RemoteProxy)
	if proxy == "" {
		proxy = "none"
	}
	receipt := &RemotePlacementReceipt{
		Mode: PlacementRemoteAllowed, ControlPlaneOwner: "fak-local", Target: target, AuthorizedTarget: authorized,
		Provider: strings.TrimSpace(opts.RemoteProvider), Engine: strings.TrimSpace(opts.RemoteEngine), Model: strings.TrimSpace(opts.RemoteModel),
		EndpointClass: strings.TrimSpace(opts.RemoteEndpointClass), Region: strings.TrimSpace(opts.RemoteRegion), LocalFailureTrigger: &localFailure,
		StateCrossingBoundary: append([]string(nil), opts.RemoteStateBoundary...),
		Egress:                RemoteGate{State: strings.TrimSpace(opts.RemoteEgress)},
		Credential:            RemoteCredentialGate{Name: strings.TrimSpace(opts.RemoteCredentialName), Present: opts.RemoteCredentialPresent},
		TLS:                   RemoteGate{State: strings.TrimSpace(opts.RemoteTLS)}, Proxy: RemoteGate{State: proxy},
		Reachability: RemoteGate{State: strings.TrimSpace(opts.RemoteReachability)}, TimeoutMilliseconds: opts.RemoteTimeoutMilliseconds,
		RetryCeiling: opts.RemoteRetryCeiling, BudgetMicroUSD: opts.RemoteBudgetMicroUSD,
	}
	refuse := func(code, detail, remediation string) Execution {
		return Execution{
			PayloadLoaded: false, PayloadCompatibility: payloadCompatibilityRefused, RemotePlacement: receipt,
			Reason: &Reason{Code: code, Detail: detail, Remediation: remediation},
		}
	}
	if target == "" {
		return refuse("remote_target_required", "remote_allowed placement requires an exact named remote target", "set --remote-target and authorize the identical name")
	}
	if authorized == "" || authorized != target {
		return refuse("remote_target_unauthorized", "remote target "+target+" is not exactly authorized", "set --authorize-remote-target to the exact target name")
	}
	if receipt.Provider == "" || receipt.Engine == "" || receipt.Model == "" {
		return refuse("remote_identity_incomplete", "remote provider, engine, and model must be named before placement", "set --remote-provider, --remote-engine, and --remote-model")
	}
	if receipt.Egress.State != "allowed" {
		return refuse("remote_egress_denied", "remote egress is not explicitly allowed", "set --remote-egress allowed only under an approved data-egress policy")
	}
	if receipt.Credential.Name == "" || !receipt.Credential.Present {
		return refuse("remote_credential_missing", "the named remote credential is absent", "provide a credential through the approved secret store; only its name and presence are reported")
	}
	if receipt.TLS.State != "verified" {
		return refuse("remote_tls_unverifiable", "remote TLS state is not verified", "verify the target TLS chain before allowing placement")
	}
	if proxy != "none" && proxy != "verified" {
		return refuse("remote_proxy_unverifiable", "remote proxy state is not verified", "use none or a verified corporate proxy")
	}
	if receipt.Reachability.State != "reachable" {
		return refuse("remote_target_unreachable", "remote target reachability is not established", "supply an independently witnessed reachable state")
	}
	if receipt.TimeoutMilliseconds <= 0 {
		return refuse("remote_timeout_invalid", "remote timeout must be greater than zero", "set a positive --remote-timeout-ms")
	}
	if receipt.RetryCeiling < 0 {
		return refuse("remote_retry_invalid", "remote retry ceiling cannot be negative", "set --remote-retry-ceiling to zero or a bounded positive value")
	}
	if receipt.BudgetMicroUSD <= 0 {
		return refuse("remote_budget_invalid", "remote budget must be greater than zero", "set a positive --remote-budget-microusd")
	}
	if len(receipt.StateCrossingBoundary) == 0 {
		return refuse("remote_state_boundary_required", "state crossing the remote boundary must be declared", "set --remote-state-boundary to a comma-separated data-class list")
	}
	return Execution{
		Runnable: true, Engine: receipt.Engine, Backend: "remote:" + target, Mode: executionModeRemote,
		PayloadLoaded: false, PayloadCompatibility: payloadCompatibilitySupported, RemotePlacement: receipt,
	}
}

func backendRecord(backend compute.Backend) Backend {
	return Backend{Name: backend.Name(), Registered: true, Class: backend.Class().String(), Tier: backend.Tier()}
}

func availableExecution(backend compute.Backend) Execution {
	return Execution{
		Runnable:             true,
		Engine:               "fak-native",
		Backend:              backend.Name(),
		Mode:                 executionModeStandard,
		PayloadLoaded:        false,
		PayloadCompatibility: payloadCompatibilityNotChecked,
	}
}

func executionForBackend(backend compute.Backend, envelopeID, goos, goarch string, host HostMemory, catalog []CPUEnvelope, policy string) Execution {
	exec := availableExecution(backend)
	if backend.Name() == "cpu-ref" && strings.TrimSpace(envelopeID) != "" {
		return applyCPUEnvelope(exec, goos, goarch, host, envelopeID, catalog, "", nil, policy)
	}
	return exec
}

func applyCPUEnvelope(exec Execution, goos, goarch string, host HostMemory, envelopeID string, catalog []CPUEnvelope, fallbackBackend string, fallbackReason *Reason, policy string) Execution {
	envelopeID = strings.TrimSpace(envelopeID)
	if envelopeID == "" {
		return refuseCPUEnvelope(exec, "cpu_fallback_envelope_required", "local CPU admission requires an exact supported cpu envelope before payload load", "pass --cpu-envelope <id> from supported_cpu_envelopes")
	}
	env, ok := findCPUEnvelope(catalog, envelopeID)
	exec.CPUEnvelope = envelopeID
	if !ok {
		return refuseCPUEnvelope(exec, "unsupported_cpu_envelope", "the requested CPU envelope name is not recognized", "choose an exact id from supported_cpu_envelopes; fak will not silently substitute another model or engine")
	}
	if env.GOOS != "" && goos != env.GOOS {
		return refuseCPUEnvelope(exec, "unsupported_cpu_envelope", "the requested CPU envelope is not supported on this operating system", "choose a supported_cpu_envelopes row whose goos/goarch exactly matches the host")
	}
	if env.GOARCH != "" && goarch != env.GOARCH {
		return refuseCPUEnvelope(exec, "unsupported_cpu_envelope", "the requested CPU envelope is not supported on this architecture", "choose a supported_cpu_envelopes row whose goos/goarch exactly matches the host")
	}
	if host.Known && env.MinimumRAMBytes > 0 && host.TotalBytes > 0 && host.TotalBytes < env.MinimumRAMBytes {
		return refuseCPUEnvelope(exec, "cpu_fallback_minimum_ram_unmet", "the host total RAM is below this CPU envelope's supported minimum", "choose a smaller supported CPU envelope or dispatch to a host with more RAM")
	}
	plan := env.memoryPlan()
	if err := compute.RefuseMemoryPlanIfTooBigForReportedHost(plan, host.TotalBytes, hostFree(host), host.Known, env.HeadroomRatio); err != nil {
		return refuseCPUEnvelope(exec, "cpu_fallback_over_budget", "the requested CPU envelope exceeds the host memory budget before payload load: "+err.Error(), "choose a smaller supported CPU envelope or dispatch to a host with more RAM; fak will not load the payload first")
	}
	exec.Runnable = true
	exec.PayloadCompatibility = payloadCompatibilitySupported
	if fallbackReason != nil {
		exec.Mode = executionModeLocalCPUDegraded
		exec.LocalCPUDegraded = &LocalCPUDegradedReceipt{
			Mode:               executionModeLocalCPUDegraded,
			Policy:             policy,
			RequestedBackend:   fallbackBackend,
			SelectedBackend:    exec.Backend,
			Reason:             fallbackReason,
			EnvelopeID:         env.ID,
			Model:              env.Model,
			Quantization:       env.Quantization,
			PerformanceClass:   env.PerformanceClass,
			QualityEquivalence: env.QualityEquivalence,
			HostMemory:         host,
		}
	}
	return exec
}

func refuseCPUEnvelope(exec Execution, code, detail, remediation string) Execution {
	exec.Runnable = false
	exec.PayloadCompatibility = payloadCompatibilityRefused
	exec.Reason = &Reason{Code: code, Detail: detail, Remediation: remediation}
	return exec
}

func findCPUEnvelope(catalog []CPUEnvelope, id string) (CPUEnvelope, bool) {
	for _, env := range catalog {
		if env.ID == id {
			return env, true
		}
	}
	return CPUEnvelope{}, false
}

func hostFree(host HostMemory) int64 {
	if host.FreeKnown {
		return host.FreeBytes
	}
	return compute.FreeUnknown
}

func effectiveHostMemory(opts Options) HostMemory {
	if opts.HostMemoryOverride {
		return normalizeHostMemory(opts.HostMemory)
	}
	total, free, known := compute.HostSystemMemoryInfo()
	return normalizeHostMemory(HostMemory{Known: known, TotalBytes: total, FreeBytes: free, FreeKnown: known && free >= 0})
}

func normalizeHostMemory(host HostMemory) HostMemory {
	if !host.Known {
		return HostMemory{}
	}
	if host.TotalBytes < 0 {
		host.TotalBytes = 0
	}
	if host.FreeKnown {
		if host.FreeBytes < 0 {
			host.FreeBytes = 0
		}
	} else {
		host.FreeBytes = 0
	}
	return host
}

func supportedCPUEnvelopes() []CPUEnvelope {
	envelopes := []CPUEnvelope{
		cpuEnvelopeFixture(
			"qwen25-1p5b-q8-windows-amd64", "Qwen2.5-1.5B-Instruct.Q8_0.gguf", "Q8_0", "windows", 8<<30, 2<<30,
			18.42750846670304, 193.50481437710533, "local_cpu_degraded_small_q8",
			"experiments/gpu/crossover-qwen2.5-1.5b-cpu-q8-20260619.json",
			[]PayloadDemand{
				{Class: string(compute.MemoryWeights), Bytes: 2 << 30, Detail: "qwen25-1.5b-q8 weights"},
				{Class: string(compute.MemoryKVCache), Bytes: 256 << 20, Detail: "decode kv window"},
				{Class: string(compute.MemoryActivation), Bytes: 256 << 20, Detail: "prefill/decode activation"},
			},
		),
		cpuEnvelopeFixture(
			"qwen36-27b-q4k-linux-amd64", "Qwen3.6-27B-Q4_K_M.gguf", "Q4_K_M", "linux", 253<<30, 20<<30,
			1.05, 23.0, "local_cpu_degraded_large_q4k",
			"docs/notes/CPU-INFERENCE-SCALING-BOX873AF63B-2026-07-11.md",
			[]PayloadDemand{
				{Class: string(compute.MemoryWeights), Bytes: 24 << 30, Detail: "qwen36-27b-q4k resident weights"},
				{Class: string(compute.MemoryKVCache), Bytes: 2 << 30, Detail: "decode kv window"},
				{Class: string(compute.MemoryActivation), Bytes: 1 << 30, Detail: "prefill/decode activation"},
			},
		),
	}
	sort.Slice(envelopes, func(i, j int) bool { return envelopes[i].ID < envelopes[j].ID })
	return envelopes
}

func cpuEnvelopeFixture(id, model, quantization, goos string, minimumRAM, minimumDisk int64, decodeTPS, prefillTPS float64, performanceClass, witness string, demands []PayloadDemand) CPUEnvelope {
	return CPUEnvelope{
		ID: id, Model: model, Quantization: quantization, Engine: "fak-native", Backend: "cpu-ref",
		GOOS: goos, GOARCH: "amd64", MinimumRAMBytes: minimumRAM, MinimumDiskBytes: minimumDisk, HeadroomRatio: 0.15,
		ExpectedDecodeTokPerSec: decodeTPS, ExpectedPrefill256TokPerSec: prefillTPS, PerformanceClass: performanceClass,
		QualityEquivalence: "same fak-native engine and exact GGUF artifact; only latency degrades, never the engine or model identity",
		Witness:            witness, Demands: demands,
	}
}

func (e CPUEnvelope) memoryPlan() compute.MemoryPlan {
	plan := make(compute.MemoryPlan, 0, len(e.Demands))
	for _, demand := range e.Demands {
		if demand.Bytes <= 0 {
			continue
		}
		class := compute.MemoryClass(strings.TrimSpace(demand.Class))
		if class == "" {
			class = compute.MemoryUnknown
		}
		plan = append(plan, compute.MemoryDemand{
			Class:  class,
			Bytes:  demand.Bytes,
			Detail: demand.Detail,
			Scope:  compute.MemoryScopeHost,
			DType:  strings.ToLower(e.Quantization),
		})
	}
	return plan
}

func normalizeFallbackPolicy(policy string) string {
	policy = strings.TrimSpace(policy)
	if policy == "" {
		return FallbackPolicyPinOrRefuse
	}
	return policy
}

func reasonStatus(reason Reason) string {
	if strings.HasPrefix(reason.Code, "unsupported_") {
		return "unsupported"
	}
	return "unavailable"
}

func unavailableReason(name, goos, goarch string, tags []string) Reason {
	hasTag := func(want string) bool {
		for _, tag := range tags {
			if tag == want {
				return true
			}
		}
		return false
	}
	switch name {
	case "cpu-ref":
		return Reason{Code: "portable_cpu_reference_unavailable", Detail: "cpu-ref is expected in every official fak binary but is not registered", Remediation: "replace this binary with an official build"}
	case "metal":
		if goos != "darwin" || goarch != "arm64" {
			return Reason{Code: "unsupported_platform", Detail: "metal requires darwin/arm64", Remediation: "request cpu-ref explicitly or run on Apple Silicon"}
		}
		return Reason{Code: "backend_unavailable", Detail: "metal was not registered; the binary may lack cgo/Metal support or no Metal device is available", Remediation: "use a cgo-enabled darwin/arm64 build and verify Metal availability"}
	case "cuda":
		if !hasTag("cuda") {
			return Reason{Code: "backend_not_compiled", Detail: "cuda is not compiled into this binary", Remediation: "use a fak build produced with the cuda tag and its required CUDA runtime/static library"}
		}
		return Reason{Code: "backend_unavailable", Detail: "cuda is compiled but no reachable CUDA device/runtime registered", Remediation: "install a compatible NVIDIA driver/runtime and verify device access"}
	case "vulkan":
		if goos != "windows" {
			return Reason{Code: "unsupported_platform", Detail: "the fak Vulkan backend currently requires windows", Remediation: "request cpu-ref or a backend supported on this operating system"}
		}
		if !hasTag("vulkan") {
			return Reason{Code: "backend_not_compiled", Detail: "vulkan is not compiled into this binary", Remediation: "use a Windows fak build produced with the vulkan tag and bundled Vulkan shim"}
		}
		return Reason{Code: "backend_unavailable", Detail: "vulkan is compiled but no compatible Vulkan device/runtime registered", Remediation: "install a compatible Vulkan driver and verify device access"}
	default:
		return Reason{Code: "unsupported_backend", Detail: "the requested backend name is not recognized or registered", Remediation: "choose an exact name from registered_backends; fak will not silently substitute another backend"}
	}
}

func currentBuildTags() []string {
	info, ok := debug.ReadBuildInfo()
	if !ok {
		return []string{}
	}
	for _, setting := range info.Settings {
		if setting.Key != "-tags" {
			continue
		}
		fields := strings.FieldsFunc(setting.Value, func(r rune) bool { return r == ',' || r == ' ' })
		out := make([]string, 0, len(fields))
		for _, field := range fields {
			if field != "" {
				out = append(out, field)
			}
		}
		return out
	}
	return []string{}
}
