package main

import (
	"fmt"
	"io"
	"os"
	"runtime"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/gateway"
	"github.com/anthony-chaudhary/fak/internal/ggufload"
	"github.com/anthony-chaudhary/fak/internal/metalgemm"
)

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
	if len(out.Phases) == 0 && (out.TotalSeconds > 0 || out.Tensors > 0) {
		b := out.Bottleneck
		if b == "" {
			b = "weights-load"
			out.Bottleneck = b
		}
		out.Phases = append(out.Phases, gateway.ModelLoadPhase{
			Phase:   b,
			Seconds: out.TotalSeconds,
			Bytes:   out.Bytes,
			Tensors: out.Tensors,
		})
	}
	for _, lp := range p.LoadPaths {
		out.LoadPaths = append(out.LoadPaths, gateway.ModelLoadPath{
			QuantType:       lp.QuantType,
			Expert:          lp.Expert,
			ResidentTensors: lp.ResidentTensors,
			ResidentBytes:   lp.ResidentBytes,
			DequantTensors:  lp.DequantTensors,
			DequantBytes:    lp.DequantBytes,
		})
	}
	for _, alert := range p.Alerts {
		out.Messages = append(out.Messages, gateway.StartupMessage{
			Source: "model-load",
			Kind:   alert.Kind,
			Level:  alert.Level,
			Text:   alert.Text,
		})
	}
	return out
}

func withServeGGUFMemoryProfile(p *gateway.ModelLoadProfile, plan compute.MemoryPlan, be compute.Backend) *gateway.ModelLoadProfile {
	if p == nil {
		return nil
	}
	p.MemoryPlan = toGatewayLoadMemoryPlan(plan)
	if be != nil {
		p.MemoryCapacities = toGatewayLoadMemoryCapacities(be)
		if len(p.MemoryPlan) > 0 {
			p.MemoryHeadroomRatio = serveGGUFDeviceHeadroom
		}
	}
	return p
}

func toGatewayLoadMemoryPlan(plan compute.MemoryPlan) []gateway.ModelLoadMemoryDemand {
	if len(plan) == 0 {
		return nil
	}
	out := make([]gateway.ModelLoadMemoryDemand, 0, len(plan))
	for _, d := range plan {
		if d.Bytes <= 0 {
			continue
		}
		class := d.Class
		if class == "" {
			class = compute.MemoryUnknown
		}
		out = append(out, gateway.ModelLoadMemoryDemand{
			Class:  string(class),
			Scope:  string(d.ScopeOrDefault()),
			Bytes:  d.Bytes,
			Detail: d.Detail,
			DType:  d.DType,
		})
	}
	return out
}

func toGatewayLoadMemoryCapacities(be compute.Backend) []gateway.ModelLoadMemoryCapacity {
	if be == nil {
		return nil
	}
	deviceTotal, deviceFree, deviceKnown := compute.DeviceMemoryInfo(be)
	hostTotal, hostFree, hostKnown := compute.HostMemoryInfo(be)
	return []gateway.ModelLoadMemoryCapacity{
		toGatewayLoadMemoryCapacity(string(compute.MemoryScopeDevice), deviceTotal, deviceFree, deviceKnown),
		toGatewayLoadMemoryCapacity(string(compute.MemoryScopeHost), hostTotal, hostFree, hostKnown),
	}
}

func toGatewayLoadMemoryCapacity(scope string, total, free int64, known bool) gateway.ModelLoadMemoryCapacity {
	cap := gateway.ModelLoadMemoryCapacity{
		Scope:      scope,
		TotalBytes: total,
		Known:      known,
		FreeKnown:  known && free >= 0,
	}
	if !known {
		cap.TotalBytes = 0
		return cap
	}
	if cap.FreeKnown {
		cap.FreeBytes = free
	}
	return cap
}

// Backend selection lives in the native CLI host rather than internal/compute so
// the portable registry stays independent of process environment and runtime OS.
const (
	serveBackendAuto = "auto"
	serveBackendCPU  = "cpu"
)

type serveBackendLookup func(string) (compute.Backend, error)

// serveBackendSelection is the side-effect-free result shared by the compute-HAL
// and Apple-Silicon Metal resolvers. The CPU floor is represented by name "cpu"
// and a nil backend because the served model's ordinary Session owns that path.
type serveBackendSelection struct {
	name     string
	backend  compute.Backend
	source   string
	explicit bool
}

// resolveServeBackendSelection applies the native backend policy without touching
// process globals, so callers can prove it with an injected registry and GOOS.
// An explicit --backend (including "auto") wins FAK_BACKEND. Linux and Windows
// automatic selection probes only Vulkan: registration already proves that the
// tagged backend initialized a usable device. Darwin leaves automatic execution to
// the existing metalgemm seam. The documented "cpu" name is the portable opt-out.
func resolveServeBackendSelection(requested, envValue, goos string, lookup serveBackendLookup) (serveBackendSelection, error) {
	requested = strings.ToLower(strings.TrimSpace(requested))
	envValue = strings.ToLower(strings.TrimSpace(envValue))
	goos = strings.ToLower(strings.TrimSpace(goos))
	explicit := requested != ""

	name := requested
	source := "explicit"
	if name == "" {
		name = envValue
		source = "env"
	}
	if name == "" {
		name = serveBackendAuto
		source = "auto"
	}

	if name == serveBackendAuto {
		switch goos {
		case "darwin":
			return serveBackendSelection{name: "metal", source: "auto", explicit: explicit}, nil
		case "linux", "windows":
			be, err := lookup("vulkan")
			if err == nil && be != nil {
				return serveBackendSelection{name: "vulkan", backend: be, source: "auto", explicit: explicit}, nil
			}
		}
		return serveBackendSelection{name: serveBackendCPU, source: "default", explicit: explicit}, nil
	}
	if name == serveBackendCPU {
		return serveBackendSelection{name: name, source: source, explicit: explicit}, nil
	}

	selection := serveBackendSelection{name: name, source: source, explicit: explicit}
	be, err := lookup(name)
	if err != nil || be == nil {
		if err == nil {
			err = fmt.Errorf("backend lookup returned no backend")
		}
		selector := "--backend"
		if source == "env" {
			selector = "FAK_BACKEND"
		}
		return selection, fmt.Errorf("fak serve: %s %q is not available: %w", selector, name, err)
	}
	selection.backend = be
	return selection, nil
}

func lookupRegisteredServeBackend(name string) (compute.Backend, error) {
	be, found := compute.Lookup(name)
	if !found {
		return nil, fmt.Errorf("registered backends: %v; a device backend needs both a matching build tag (e.g. -tags %s) and a reachable device at runtime", compute.Registered(), name)
	}
	return be, nil
}

func resolveServeChatBackend(backendName string) (compute.Backend, error) {
	selection, err := resolveServeBackendSelection(backendName, os.Getenv("FAK_BACKEND"), runtime.GOOS, lookupRegisteredServeBackend)
	return selection.backend, err
}

// writeBackendUnavailableBail renders the BACKEND_UNAVAILABLE bail for a --backend
// or FAK_BACKEND name this binary never registered. Shared by the serve entry points so both report
// the same knobs; resolveServeChatBackend fails for exactly this one reason, so a
// non-nil error from it is always this bail.
//
// The name is not silently downgraded to CPU: a typo that quietly served on the
// wrong device would misreport every throughput number taken from that run.
func writeBackendUnavailableBail(w io.Writer, verb, backendName string) {
	backendName = strings.ToLower(strings.TrimSpace(backendName))
	fromEnv := backendName == ""
	if fromEnv {
		backendName = strings.ToLower(strings.TrimSpace(os.Getenv("FAK_BACKEND")))
	}
	selector := "--backend"
	knob := bailFlag("backend", backendName).want(fmt.Sprintf("one of %v, %q, or %q", compute.Registered(), serveBackendAuto, serveBackendCPU))
	if fromEnv {
		selector = "FAK_BACKEND"
		knob = bailEnv("FAK_BACKEND", backendName).want(fmt.Sprintf("one of %v, %q, %q, or unset for platform automatic selection", compute.Registered(), serveBackendAuto, serveBackendCPU))
	}
	writeConfigBail(w, configBail{
		Verb:    verb,
		Reason:  bailBackendUnavailable,
		Summary: fmt.Sprintf("%s %q is not registered in this binary", selector, backendName),
		Knobs:   []bailKnob{knob},
		// Keep the build-tag half of the original message: "not registered" reads
		// as a runtime/device problem, but the usual cause is a binary compiled
		// without the tag, which no amount of checking the device will reveal.
		Check: fmt.Sprintf("fak doctor serve   # decode tier and serve readiness; a device backend needs BOTH a build tag (-tags %s) and a reachable device", backendName),
	})
}

// resolveServeMetal decides whether `fak serve` runs the in-kernel chat through the
// Apple-Silicon Metal GPU forward. Metal auto-selects when this binary has the backend
// linked and a usable device is present; --metal/FAK_METAL=1 only changes the unavailable
// case from CPU fallback to a fail-loud error. The error distinguishes a wrong build
// (`metalgemm.Compiled()` false → build on Apple Silicon with cgo) from a right build with
// no device (`Available()` false). Metal is the CPU-session seam (the served session keeps
// s.Backend nil and gets s.Metal=true), so it is mutually exclusive with a selected compute backend.
// Kept side-effect free (no os.Exit) so the decision is unit-testable; on a non-Metal build
// metalgemm.Available()/Compiled() are the stub's deterministic false.
func resolveServeMetal(flag, env bool, backendName string) (bool, error) {
	requested := flag || env
	selection, selectionErr := resolveServeBackendSelection(backendName, os.Getenv("FAK_BACKEND"), runtime.GOOS, lookupRegisteredServeBackend)
	// The main compute resolver reports explicit --backend failures first. Propagate
	// environment failures here as well so direct callers cannot turn a bad
	// FAK_BACKEND value into automatic Metal by accident.
	if selectionErr != nil && strings.TrimSpace(backendName) == "" {
		return false, selectionErr
	}
	if requested && selection.explicit {
		return false, fmt.Errorf("fak serve: --metal and --backend %q are mutually exclusive — Metal is the Apple-Silicon CPU-session forward, not a compute HAL device. Pass one.", backendName)
	}
	backendSelected := selection.backend != nil || selection.source == "explicit" || selection.source == "env"
	if backendSelected {
		if requested {
			return false, fmt.Errorf("fak serve: Metal and FAK_BACKEND=%q are mutually exclusive — Metal is the Apple-Silicon CPU-session forward, not a compute HAL device. Select one.", selection.name)
		}
		return false, nil
	}
	if !metalgemm.Available() {
		if !requested {
			return false, nil
		}
		if !metalgemm.Compiled() {
			return false, fmt.Errorf("fak serve: --metal requested but this binary has no Metal support — build on darwin/arm64 with cgo enabled.")
		}
		return false, fmt.Errorf("fak serve: --metal requested but no usable Metal device is available on this host.")
	}
	return true, nil
}
