package modelreg

import (
	"fmt"
	"regexp"
	"sort"
	"strings"
)

const LocalAdmissionDecisionSchema = "fak.harness.local-runtime-admission.v1"

const (
	LocalAdmissionAdmit  = "ADMIT"
	LocalAdmissionRefuse = "REFUSE"

	LocalDeviceCPU = "cpu"

	LocalRefusalInvalidDeclaration  = "INVALID_DECLARATION"
	LocalRefusalArtifactUnverified  = "ARTIFACT_UNVERIFIED"
	LocalRefusalArtifactMismatch    = "ARTIFACT_IDENTITY_MISMATCH"
	LocalRefusalRuntimeUnverified   = "RUNTIME_UNVERIFIED"
	LocalRefusalRuntimeIncompatible = "RUNTIME_INCOMPATIBLE"
	LocalRefusalRuntimeVersion      = "RUNTIME_VERSION_INCOMPATIBLE"
	LocalRefusalCapabilityMissing   = "RUNTIME_CAPABILITY_MISSING"
	LocalRefusalDiskUnmeasured      = "DISK_UNMEASURED"
	LocalRefusalDiskInsufficient    = "DISK_INSUFFICIENT"
	LocalRefusalRAMUnmeasured       = "RAM_UNMEASURED"
	LocalRefusalRAMInsufficient     = "RAM_INSUFFICIENT"
	LocalRefusalDeviceUnavailable   = "DEVICE_UNAVAILABLE"
	LocalRefusalVRAMUnmeasured      = "VRAM_UNMEASURED"
	LocalRefusalVRAMInsufficient    = "VRAM_INSUFFICIENT"
	LocalRefusalCPUFallbackInvalid  = "CPU_FALLBACK_INVALID"
)

var localAdmissionSHA256 = regexp.MustCompile(`^[0-9a-f]{64}$`)

// LocalResourceRequirements are declaration-authored launch reservations. They
// are requirements, not measurements; EvaluateLocalAdmission compares them with a
// separately observed LocalHostFacts value.
type LocalResourceRequirements struct {
	DiskBytes int64 `json:"disk_bytes"`
	RAMBytes  int64 `json:"ram_bytes"`
	VRAMBytes int64 `json:"vram_bytes,omitempty"`
}

// LocalDeviceTarget is one explicitly declared execution target. DeviceKind is
// cpu, cuda, metal, vulkan, or another runtime capability name. DeviceID narrows
// a non-CPU request to one measured device when set.
type LocalDeviceTarget struct {
	DeviceKind string                    `json:"device_kind"`
	DeviceID   string                    `json:"device_id,omitempty"`
	Resources  LocalResourceRequirements `json:"resources"`
}

// LocalAdmissionDeclaration contains only caller-authored facts. CPUFallback
// is a pointer by design: nil means refuse rather than invent a CPU path.
type LocalAdmissionDeclaration struct {
	ModelID                   string             `json:"model_id"`
	ArtifactSHA256            string             `json:"artifact_sha256"`
	ArtifactBytes             int64              `json:"artifact_bytes"`
	RuntimeID                 string             `json:"runtime_id"`
	RuntimeVersion            string             `json:"runtime_version"`
	RequiredRuntimeCapability string             `json:"required_runtime_capability"`
	Requested                 LocalDeviceTarget  `json:"requested"`
	CPUFallback               *LocalDeviceTarget `json:"cpu_fallback,omitempty"`
}

// LocalVerifiedArtifactFacts are produced by the verified acquisition/cache
// stage. Verified is explicit: a plausible path, digest, or byte count alone is
// never promoted into verified identity by this planner.
type LocalVerifiedArtifactFacts struct {
	Path     string `json:"path"`
	SHA256   string `json:"sha256"`
	Bytes    int64  `json:"bytes"`
	Verified bool   `json:"verified"`
}

// LocalRuntimeFacts describe an already-observed runtime. The planner never
// searches for, installs, starts, or probes a runtime.
type LocalRuntimeFacts struct {
	ID           string   `json:"id"`
	Version      string   `json:"version"`
	Capabilities []string `json:"capabilities"`
	Verified     bool     `json:"verified"`
}

// LocalDeviceFacts are measured device availability and free VRAM. VRAMKnown
// distinguishes an observed zero from a missing measurement.
type LocalDeviceFacts struct {
	Kind          string `json:"kind"`
	ID            string `json:"id,omitempty"`
	Available     bool   `json:"available"`
	VRAMKnown     bool   `json:"vram_known"`
	FreeVRAMBytes int64  `json:"free_vram_bytes,omitempty"`
}

// LocalHostFacts are the measurements used for this one decision. No accessor
// or callback is embedded, keeping planning deterministic and effect-free.
type LocalHostFacts struct {
	DiskKnown     bool               `json:"disk_known"`
	FreeDiskBytes int64              `json:"free_disk_bytes,omitempty"`
	RAMKnown      bool               `json:"ram_known"`
	FreeRAMBytes  int64              `json:"free_ram_bytes,omitempty"`
	Devices       []LocalDeviceFacts `json:"devices,omitempty"`
}

type LocalAdmissionRequest struct {
	Declaration LocalAdmissionDeclaration  `json:"declaration"`
	Artifact    LocalVerifiedArtifactFacts `json:"artifact"`
	Runtime     LocalRuntimeFacts          `json:"runtime"`
	Host        LocalHostFacts             `json:"host"`
}

type LocalAdmissionRefusal struct {
	Code      string `json:"code"`
	Resource  string `json:"resource,omitempty"`
	Required  int64  `json:"required,omitempty"`
	Available int64  `json:"available,omitempty"`
	Detail    string `json:"detail"`
}

// LocalLaunchResourceReservation is the fixed, evidenced capacity envelope an
// admitted launcher receives. It is not an execution plan or scheduler: it
// orders no steps and owns no process lifecycle.
type LocalLaunchResourceReservation struct {
	ModelID        string                    `json:"model_id"`
	ArtifactPath   string                    `json:"artifact_path"`
	ArtifactSHA256 string                    `json:"artifact_sha256"`
	ArtifactBytes  int64                     `json:"artifact_bytes"`
	RuntimeID      string                    `json:"runtime_id"`
	RuntimeVersion string                    `json:"runtime_version"`
	DeviceKind     string                    `json:"device_kind"`
	DeviceID       string                    `json:"device_id,omitempty"`
	CPUFallback    bool                      `json:"cpu_fallback"`
	FallbackReason *LocalAdmissionRefusal    `json:"fallback_reason,omitempty"`
	Required       LocalResourceRequirements `json:"required"`
	FreeDiskBytes  int64                     `json:"free_disk_bytes"`
	FreeRAMBytes   int64                     `json:"free_ram_bytes"`
	FreeVRAMBytes  int64                     `json:"free_vram_bytes,omitempty"`
}

type LocalAdmissionDecision struct {
	Schema   string                          `json:"schema"`
	Verdict  string                          `json:"verdict"`
	Plan     *LocalLaunchResourceReservation `json:"plan,omitempty"`
	Refusals []LocalAdmissionRefusal         `json:"refusals"`
}

// LocalAdmissionRefusalError lets an integration seam fail closed without
// throwing away the complete typed decision a diagnostic or receipt consumes.
type LocalAdmissionRefusalError struct {
	Decision LocalAdmissionDecision
}

func (e *LocalAdmissionRefusalError) Error() string {
	if e == nil || len(e.Decision.Refusals) == 0 {
		return "local runtime admission refused"
	}
	r := e.Decision.Refusals[0]
	return fmt.Sprintf("local runtime admission refused: %s: %s", r.Code, r.Detail)
}

func (d LocalAdmissionDecision) RefusalError() error {
	if d.Verdict == LocalAdmissionAdmit {
		return nil
	}
	return &LocalAdmissionRefusalError{Decision: d}
}

// EvaluateLocalAdmission is a pure decision evaluator over authored
// declaration, verified artifact/runtime facts, and measured host facts. Unlike
// an execution planner or launcher, it orders no work and performs no I/O,
// network, download, device discovery, or process launch.
func EvaluateLocalAdmission(req LocalAdmissionRequest) LocalAdmissionDecision {
	d := normalizedAdmissionDeclaration(req.Declaration)
	a := req.Artifact
	a.Path = strings.TrimSpace(a.Path)
	a.SHA256 = strings.ToLower(strings.TrimSpace(a.SHA256))
	rt := req.Runtime
	rt.ID = strings.TrimSpace(rt.ID)
	rt.Version = strings.TrimSpace(rt.Version)

	decision := LocalAdmissionDecision{Schema: LocalAdmissionDecisionSchema, Verdict: LocalAdmissionRefuse, Refusals: []LocalAdmissionRefusal{}}
	refuse := func(code, resource string, required, available int64, detail string) {
		decision.Refusals = append(decision.Refusals, LocalAdmissionRefusal{Code: code, Resource: resource, Required: required, Available: available, Detail: detail})
	}

	if d.ModelID == "" || !localAdmissionSHA256.MatchString(d.ArtifactSHA256) || d.ArtifactBytes <= 0 ||
		d.RuntimeID == "" || d.RuntimeVersion == "" || d.RequiredRuntimeCapability == "" || !validLocalTarget(d.Requested) {
		refuse(LocalRefusalInvalidDeclaration, "", 0, 0, "model, pinned artifact, runtime, capability, and requested resource target are required")
		return decision
	}
	if d.CPUFallback != nil && (d.CPUFallback.DeviceKind != LocalDeviceCPU || !validLocalTarget(*d.CPUFallback)) {
		refuse(LocalRefusalCPUFallbackInvalid, "cpu", 0, 0, "declared CPU fallback must be an explicit cpu target with positive RAM and no VRAM")
		return decision
	}
	if !a.Verified {
		refuse(LocalRefusalArtifactUnverified, "artifact", d.ArtifactBytes, a.Bytes, "artifact acquisition receipt is not verified")
		return decision
	}
	if a.Path == "" || a.SHA256 != d.ArtifactSHA256 || a.Bytes != d.ArtifactBytes {
		refuse(LocalRefusalArtifactMismatch, "artifact", d.ArtifactBytes, a.Bytes, "verified artifact path, SHA-256, and byte count must match the declaration")
		return decision
	}
	if !rt.Verified {
		refuse(LocalRefusalRuntimeUnverified, "runtime", 0, 0, "runtime facts are not verified")
		return decision
	}
	if rt.ID != d.RuntimeID {
		refuse(LocalRefusalRuntimeIncompatible, "runtime", 0, 0, fmt.Sprintf("runtime %q does not match declared %q", rt.ID, d.RuntimeID))
		return decision
	}
	if rt.Version != d.RuntimeVersion {
		refuse(LocalRefusalRuntimeVersion, "runtime_version", 0, 0, fmt.Sprintf("runtime version %q does not match declared %q", rt.Version, d.RuntimeVersion))
		return decision
	}
	if !containsFold(rt.Capabilities, d.RequiredRuntimeCapability) {
		refuse(LocalRefusalCapabilityMissing, "runtime_capability", 0, 0, fmt.Sprintf("runtime lacks declared capability %q", d.RequiredRuntimeCapability))
		return decision
	}

	primaryPlan, primaryRefusal := planLocalTarget(d, a, rt, req.Host, d.Requested, false)
	if primaryRefusal == nil {
		decision.Verdict, decision.Plan = LocalAdmissionAdmit, primaryPlan
		return decision
	}
	if d.CPUFallback != nil && fallbackEligible(primaryRefusal.Code) {
		fallbackPlan, fallbackRefusal := planLocalTarget(d, a, rt, req.Host, *d.CPUFallback, true)
		if fallbackRefusal == nil {
			fallbackPlan.FallbackReason = primaryRefusal
			decision.Verdict, decision.Plan = LocalAdmissionAdmit, fallbackPlan
			return decision
		}
		decision.Refusals = append(decision.Refusals, *primaryRefusal, *fallbackRefusal)
		return decision
	}
	decision.Refusals = append(decision.Refusals, *primaryRefusal)
	return decision
}

func normalizedAdmissionDeclaration(in LocalAdmissionDeclaration) LocalAdmissionDeclaration {
	in.ModelID = strings.TrimSpace(in.ModelID)
	in.ArtifactSHA256 = strings.ToLower(strings.TrimSpace(in.ArtifactSHA256))
	in.RuntimeID = strings.TrimSpace(in.RuntimeID)
	in.RuntimeVersion = strings.TrimSpace(in.RuntimeVersion)
	in.RequiredRuntimeCapability = strings.TrimSpace(in.RequiredRuntimeCapability)
	in.Requested = normalizedLocalTarget(in.Requested)
	if in.CPUFallback != nil {
		fallback := normalizedLocalTarget(*in.CPUFallback)
		in.CPUFallback = &fallback
	}
	return in
}

func normalizedLocalTarget(in LocalDeviceTarget) LocalDeviceTarget {
	in.DeviceKind = strings.ToLower(strings.TrimSpace(in.DeviceKind))
	in.DeviceID = strings.TrimSpace(in.DeviceID)
	return in
}

func validLocalTarget(t LocalDeviceTarget) bool {
	if t.DeviceKind == "" || t.Resources.DiskBytes < 0 || t.Resources.RAMBytes <= 0 || t.Resources.VRAMBytes < 0 {
		return false
	}
	if t.DeviceKind == LocalDeviceCPU {
		return t.DeviceID == "" && t.Resources.VRAMBytes == 0
	}
	return t.Resources.VRAMBytes > 0
}

func planLocalTarget(d LocalAdmissionDeclaration, a LocalVerifiedArtifactFacts, rt LocalRuntimeFacts, host LocalHostFacts, target LocalDeviceTarget, fallback bool) (*LocalLaunchResourceReservation, *LocalAdmissionRefusal) {
	required := target.Resources
	if !containsFold(rt.Capabilities, target.DeviceKind) {
		return nil, &LocalAdmissionRefusal{Code: LocalRefusalCapabilityMissing, Resource: "runtime_capability", Detail: fmt.Sprintf("runtime lacks requested device capability %q", target.DeviceKind)}
	}
	if required.DiskBytes < a.Bytes {
		required.DiskBytes = a.Bytes
	}
	if !host.DiskKnown {
		return nil, &LocalAdmissionRefusal{Code: LocalRefusalDiskUnmeasured, Resource: "disk", Required: required.DiskBytes, Detail: "free disk was not measured"}
	}
	if host.FreeDiskBytes < required.DiskBytes {
		return nil, &LocalAdmissionRefusal{Code: LocalRefusalDiskInsufficient, Resource: "disk", Required: required.DiskBytes, Available: host.FreeDiskBytes, Detail: "measured free disk is below the launch reservation"}
	}
	if !host.RAMKnown {
		return nil, &LocalAdmissionRefusal{Code: LocalRefusalRAMUnmeasured, Resource: "ram", Required: required.RAMBytes, Detail: "free RAM was not measured"}
	}
	if host.FreeRAMBytes < required.RAMBytes {
		return nil, &LocalAdmissionRefusal{Code: LocalRefusalRAMInsufficient, Resource: "ram", Required: required.RAMBytes, Available: host.FreeRAMBytes, Detail: "measured free RAM is below the launch reservation"}
	}

	freeVRAM := int64(0)
	if target.DeviceKind != LocalDeviceCPU {
		device, ok := matchingLocalDevice(host.Devices, target)
		if !ok || !device.Available {
			return nil, &LocalAdmissionRefusal{Code: LocalRefusalDeviceUnavailable, Resource: target.DeviceKind, Detail: fmt.Sprintf("requested device %q was not observed available", localTargetName(target))}
		}
		if !device.VRAMKnown {
			return nil, &LocalAdmissionRefusal{Code: LocalRefusalVRAMUnmeasured, Resource: "vram", Required: required.VRAMBytes, Detail: "free VRAM was not measured for the requested device"}
		}
		freeVRAM = device.FreeVRAMBytes
		if freeVRAM < required.VRAMBytes {
			return nil, &LocalAdmissionRefusal{Code: LocalRefusalVRAMInsufficient, Resource: "vram", Required: required.VRAMBytes, Available: freeVRAM, Detail: "measured free VRAM is below the launch reservation"}
		}
	}

	return &LocalLaunchResourceReservation{
		ModelID: d.ModelID, ArtifactPath: a.Path, ArtifactSHA256: a.SHA256, ArtifactBytes: a.Bytes,
		RuntimeID: rt.ID, RuntimeVersion: rt.Version, DeviceKind: target.DeviceKind, DeviceID: target.DeviceID,
		CPUFallback: fallback, Required: required, FreeDiskBytes: host.FreeDiskBytes, FreeRAMBytes: host.FreeRAMBytes, FreeVRAMBytes: freeVRAM,
	}, nil
}

func matchingLocalDevice(devices []LocalDeviceFacts, target LocalDeviceTarget) (LocalDeviceFacts, bool) {
	ordered := append([]LocalDeviceFacts(nil), devices...)
	sort.Slice(ordered, func(i, j int) bool {
		if ordered[i].Kind != ordered[j].Kind {
			return ordered[i].Kind < ordered[j].Kind
		}
		return ordered[i].ID < ordered[j].ID
	})
	for _, device := range ordered {
		if !strings.EqualFold(strings.TrimSpace(device.Kind), target.DeviceKind) {
			continue
		}
		if target.DeviceID == "" || strings.TrimSpace(device.ID) == target.DeviceID {
			return device, true
		}
	}
	return LocalDeviceFacts{}, false
}

func fallbackEligible(code string) bool {
	switch code {
	case LocalRefusalDeviceUnavailable, LocalRefusalVRAMUnmeasured, LocalRefusalVRAMInsufficient,
		LocalRefusalRAMUnmeasured, LocalRefusalRAMInsufficient, LocalRefusalCapabilityMissing:
		return true
	default:
		return false
	}
}

func containsFold(values []string, want string) bool {
	for _, value := range values {
		if strings.EqualFold(strings.TrimSpace(value), want) {
			return true
		}
	}
	return false
}

func localTargetName(target LocalDeviceTarget) string {
	if target.DeviceID == "" {
		return target.DeviceKind
	}
	return target.DeviceKind + ":" + target.DeviceID
}
