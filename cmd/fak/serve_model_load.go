package main

import (
	"fmt"
	"io"
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

// loadServeInKernelModel eagerly loads the GGUF weights (when ggufPath is set) BEFORE the
// listener binds, so the load counts toward time-to-ready and its phase breakdown reaches
// /metrics rather than being a lazy cost on first request. It returns the resident model
// (nil if no --gguf), whether the direct-resident-Q4_K path was taken, the load profile for
// /metrics, and the model-load startup phase (zero Name when no load happened). The path
// selection mirrors cmd/fakchat with one device-specific split: a device --backend that
// advertises quantized upload takes the lean-Q8 load, because the served planner runs
// Session.Quant=true and the HAL can consume Q8_0 directly. Backends without UploadDtype keep
// the F32 fallback until they can consume quantized resident weights. FAK_Q4K takes the
// direct-resident-Q4_K CPU path, and the CPU default is the lean-Q8 round-trip; the Q8 path
// stays byte-identical when the env is unset.
func resolveServeChatBackend(backendName string) (compute.Backend, error) {
	backendName = strings.TrimSpace(backendName)
	if backendName == "" {
		return nil, nil
	}
	be, found := compute.Lookup(backendName)
	if !found {
		return nil, fmt.Errorf("fak serve: --backend %q is not available (registered backends: %v). A device backend needs both a matching build tag (e.g. -tags %s) and a reachable device at runtime.", backendName, compute.Registered(), backendName)
	}
	return be, nil
}

// writeBackendUnavailableBail renders the BACKEND_UNAVAILABLE bail for a --backend
// this binary never registered. Shared by the serve entry points so both report
// the same knobs; resolveServeChatBackend fails for exactly this one reason, so a
// non-nil error from it is always this bail.
//
// The name is not silently downgraded to CPU: a typo that quietly served on the
// wrong device would misreport every throughput number taken from that run.
func writeBackendUnavailableBail(w io.Writer, verb, backendName string) {
	writeConfigBail(w, configBail{
		Verb:    verb,
		Reason:  bailBackendUnavailable,
		Summary: fmt.Sprintf("--backend %q is not registered in this binary", backendName),
		Knobs: []bailKnob{
			bailFlag("backend", backendName).want(fmt.Sprintf("one of %v, or omit --backend to serve on the CPU path", compute.Registered())),
		},
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
// s.Backend nil and gets s.Metal=true), so it is mutually exclusive with a device --backend.
// Kept side-effect free (no os.Exit) so the decision is unit-testable; on a non-Metal build
// metalgemm.Available()/Compiled() are the stub's deterministic false.
func resolveServeMetal(flag, env bool, backendName string) (bool, error) {
	requested := flag || env
	if strings.TrimSpace(backendName) != "" {
		if requested {
			return false, fmt.Errorf("fak serve: --metal and --backend %q are mutually exclusive — Metal is the Apple-Silicon CPU-session forward, not a compute HAL device. Pass one.", backendName)
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
