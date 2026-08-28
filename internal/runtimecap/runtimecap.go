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
	FallbackPolicyPinOrRefuse     = "pin_or_refuse"
	FallbackPolicyLocalCPUDegrade = "local_cpu_degraded"

	payloadCompatibilityNotChecked = "not_checked"
	payloadCompatibilitySupported  = "supported"
	payloadCompatibilityRefused    = "refused"

	executionModeStandard         = "standard"
	executionModeLocalCPUDegraded = FallbackPolicyLocalCPUDegrade
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

type Execution struct {
	Runnable             bool                     `json:"runnable"`
	Engine               string                   `json:"engine,omitempty"`
	Backend              string                   `json:"backend,omitempty"`
	Mode                 string                   `json:"mode,omitempty"`
	CPUEnvelope          string                   `json:"cpu_envelope,omitempty"`
	PayloadLoaded        bool                     `json:"payload_loaded"`
	PayloadCompatibility string                   `json:"payload_compatibility"`
	LocalCPUDegraded     *LocalCPUDegradedReceipt `json:"local_cpu_degraded,omitempty"`
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
	SupportedCPUEnvelopes []CPUEnvelope     `json:"supported_cpu_envelopes,omitempty"`
	ModelExecution        Execution         `json:"model_execution"`
}

type Options struct {
	RequestedBackend   string
	PreferredBackend   string
	CPUFallbackPolicy  string
	CPUEnvelope        string
	GOOS               string
	GOARCH             string
	BuildTags          []string
	Backends           []compute.Backend
	HostMemory         HostMemory
	HostMemoryOverride bool
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
			report.ModelExecution = availableExecution(backend)
			if backend.Name() == "cpu-ref" && strings.TrimSpace(opts.CPUEnvelope) != "" {
				report.ModelExecution = applyCPUEnvelope(report.ModelExecution, goos, goarch, hostMemory, opts.CPUEnvelope, report.SupportedCPUEnvelopes, "", nil, policy)
			}
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
			report.ModelExecution = availableExecution(backend)
			if backend.Name() == "cpu-ref" && strings.TrimSpace(opts.CPUEnvelope) != "" {
				report.ModelExecution = applyCPUEnvelope(report.ModelExecution, goos, goarch, hostMemory, opts.CPUEnvelope, report.SupportedCPUEnvelopes, "", nil, policy)
			}
			return report
		}
		reason := unavailableReason(preferred, goos, goarch, tags)
		request.Status, request.Reason = reasonStatus(reason), &reason
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
		report.ModelExecution = availableExecution(backend)
		if strings.TrimSpace(opts.CPUEnvelope) != "" {
			report.ModelExecution = applyCPUEnvelope(report.ModelExecution, goos, goarch, hostMemory, opts.CPUEnvelope, report.SupportedCPUEnvelopes, "", nil, policy)
		}
	} else {
		report.ModelExecution.Reason = &Reason{
			Code:        "portable_cpu_reference_unavailable",
			Detail:      "the portable fak-native CPU reference backend is not registered",
			Remediation: "use an official fak build that includes internal/compute/cpuref",
		}
	}
	return report
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

func applyCPUEnvelope(exec Execution, goos, goarch string, host HostMemory, envelopeID string, catalog []CPUEnvelope, fallbackBackend string, fallbackReason *Reason, policy string) Execution {
	envelopeID = strings.TrimSpace(envelopeID)
	if envelopeID == "" {
		exec.Runnable = false
		exec.PayloadCompatibility = payloadCompatibilityRefused
		exec.Reason = &Reason{
			Code:        "cpu_fallback_envelope_required",
			Detail:      "local CPU admission requires an exact supported cpu envelope before payload load",
			Remediation: "pass --cpu-envelope <id> from supported_cpu_envelopes",
		}
		return exec
	}
	env, ok := findCPUEnvelope(catalog, envelopeID)
	exec.CPUEnvelope = envelopeID
	if !ok {
		exec.Runnable = false
		exec.PayloadCompatibility = payloadCompatibilityRefused
		exec.Reason = &Reason{
			Code:        "unsupported_cpu_envelope",
			Detail:      "the requested CPU envelope name is not recognized",
			Remediation: "choose an exact id from supported_cpu_envelopes; fak will not silently substitute another model or engine",
		}
		return exec
	}
	if env.GOOS != "" && goos != env.GOOS {
		exec.Runnable = false
		exec.PayloadCompatibility = payloadCompatibilityRefused
		exec.Reason = &Reason{
			Code:        "unsupported_cpu_envelope",
			Detail:      "the requested CPU envelope is not supported on this operating system",
			Remediation: "choose a supported_cpu_envelopes row whose goos/goarch exactly matches the host",
		}
		return exec
	}
	if env.GOARCH != "" && goarch != env.GOARCH {
		exec.Runnable = false
		exec.PayloadCompatibility = payloadCompatibilityRefused
		exec.Reason = &Reason{
			Code:        "unsupported_cpu_envelope",
			Detail:      "the requested CPU envelope is not supported on this architecture",
			Remediation: "choose a supported_cpu_envelopes row whose goos/goarch exactly matches the host",
		}
		return exec
	}
	if host.Known && env.MinimumRAMBytes > 0 && host.TotalBytes > 0 && host.TotalBytes < env.MinimumRAMBytes {
		exec.Runnable = false
		exec.PayloadCompatibility = payloadCompatibilityRefused
		exec.Reason = &Reason{
			Code:        "cpu_fallback_minimum_ram_unmet",
			Detail:      "the host total RAM is below this CPU envelope's supported minimum",
			Remediation: "choose a smaller supported CPU envelope or dispatch to a host with more RAM",
		}
		return exec
	}
	plan := env.memoryPlan()
	if err := compute.RefuseMemoryPlanIfTooBigForReportedHost(plan, host.TotalBytes, hostFree(host), host.Known, env.HeadroomRatio); err != nil {
		exec.Runnable = false
		exec.PayloadCompatibility = payloadCompatibilityRefused
		exec.Reason = &Reason{
			Code:        "cpu_fallback_over_budget",
			Detail:      "the requested CPU envelope exceeds the host memory budget before payload load: " + err.Error(),
			Remediation: "choose a smaller supported CPU envelope or dispatch to a host with more RAM; fak will not load the payload first",
		}
		return exec
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
		{
			ID:                          "qwen25-1p5b-q8-windows-amd64",
			Model:                       "Qwen2.5-1.5B-Instruct.Q8_0.gguf",
			Quantization:                "Q8_0",
			Engine:                      "fak-native",
			Backend:                     "cpu-ref",
			GOOS:                        "windows",
			GOARCH:                      "amd64",
			MinimumRAMBytes:             8 << 30,
			MinimumDiskBytes:            2 << 30,
			HeadroomRatio:               0.15,
			ExpectedDecodeTokPerSec:     18.42750846670304,
			ExpectedPrefill256TokPerSec: 193.50481437710533,
			PerformanceClass:            "local_cpu_degraded_small_q8",
			QualityEquivalence:          "same fak-native engine and exact GGUF artifact; only latency degrades, never the engine or model identity",
			Witness:                     "experiments/gpu/crossover-qwen2.5-1.5b-cpu-q8-20260619.json",
			Demands: []PayloadDemand{
				{Class: string(compute.MemoryWeights), Bytes: 2 << 30, Detail: "qwen25-1.5b-q8 weights"},
				{Class: string(compute.MemoryKVCache), Bytes: 256 << 20, Detail: "decode kv window"},
				{Class: string(compute.MemoryActivation), Bytes: 256 << 20, Detail: "prefill/decode activation"},
			},
		},
		{
			ID:                          "qwen36-27b-q4k-linux-amd64",
			Model:                       "Qwen3.6-27B-Q4_K_M.gguf",
			Quantization:                "Q4_K_M",
			Engine:                      "fak-native",
			Backend:                     "cpu-ref",
			GOOS:                        "linux",
			GOARCH:                      "amd64",
			MinimumRAMBytes:             253 << 30,
			MinimumDiskBytes:            20 << 30,
			HeadroomRatio:               0.15,
			ExpectedDecodeTokPerSec:     1.05,
			ExpectedPrefill256TokPerSec: 23.0,
			PerformanceClass:            "local_cpu_degraded_large_q4k",
			QualityEquivalence:          "same fak-native engine and exact GGUF artifact; only latency degrades, never the engine or model identity",
			Witness:                     "docs/notes/CPU-INFERENCE-SCALING-BOX873AF63B-2026-07-11.md",
			Demands: []PayloadDemand{
				{Class: string(compute.MemoryWeights), Bytes: 24 << 30, Detail: "qwen36-27b-q4k resident weights"},
				{Class: string(compute.MemoryKVCache), Bytes: 2 << 30, Detail: "decode kv window"},
				{Class: string(compute.MemoryActivation), Bytes: 1 << 30, Detail: "prefill/decode activation"},
			},
		},
	}
	sort.Slice(envelopes, func(i, j int) bool { return envelopes[i].ID < envelopes[j].ID })
	return envelopes
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
