package runtimecap

import (
	"runtime"
	"runtime/debug"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

const Schema = "fak-runtime-capabilities/1"

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

type Execution struct {
	Runnable             bool    `json:"runnable"`
	Engine               string  `json:"engine,omitempty"`
	Backend              string  `json:"backend,omitempty"`
	PayloadLoaded        bool    `json:"payload_loaded"`
	PayloadCompatibility string  `json:"payload_compatibility"`
	Reason               *Reason `json:"reason,omitempty"`
}

type Report struct {
	Schema               string            `json:"schema"`
	GOOS                 string            `json:"goos"`
	GOARCH               string            `json:"goarch"`
	BinaryRunnable       bool              `json:"binary_runnable"`
	ControlPlaneRunnable bool              `json:"control_plane_runnable"`
	BuildTags            []string          `json:"build_tags"`
	RegisteredBackends   []Backend         `json:"registered_backends"`
	PortableCPU          Backend           `json:"portable_cpu_reference"`
	RequestedBackend     *RequestedBackend `json:"requested_backend,omitempty"`
	ModelExecution       Execution         `json:"model_execution"`
}

type Options struct {
	RequestedBackend string
	GOOS             string
	GOARCH           string
	BuildTags        []string
	Backends         []compute.Backend
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
	report := Report{
		Schema:               Schema,
		GOOS:                 goos,
		GOARCH:               goarch,
		BinaryRunnable:       true,
		ControlPlaneRunnable: true,
		BuildTags:            tags,
		RegisteredBackends:   records,
		PortableCPU:          cpu,
		ModelExecution: Execution{
			PayloadLoaded:        false,
			PayloadCompatibility: "not_checked",
		},
	}

	requested := strings.TrimSpace(opts.RequestedBackend)
	if requested == "" {
		if backend, ok := byName["cpu-ref"]; ok {
			report.ModelExecution = availableExecution(backend)
		} else {
			report.ModelExecution.Reason = &Reason{
				Code:        "portable_cpu_reference_unavailable",
				Detail:      "the portable fak-native CPU reference backend is not registered",
				Remediation: "use an official fak build that includes internal/compute/cpuref",
			}
		}
		return report
	}

	request := &RequestedBackend{Name: requested}
	report.RequestedBackend = request
	if backend, ok := byName[requested]; ok {
		request.Status, request.ExactMatch, request.Selected = "available", true, backend.Name()
		report.ModelExecution = availableExecution(backend)
		return report
	}

	reason := unavailableReason(requested, goos, goarch, tags)
	request.Status, request.Reason = reasonStatus(reason), &reason
	report.ModelExecution.Reason = &reason
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
		PayloadLoaded:        false,
		PayloadCompatibility: "not_checked",
	}
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
