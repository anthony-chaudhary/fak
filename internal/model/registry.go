package model

import (
	"errors"
	"strconv"
	"strings"
	"sync"
)

type FormatSignature struct {
	Container   string   `json:"container"`    // e.g. "gguf", "safetensors"
	Quant       string   `json:"quant"`        // e.g. "q4_k_m", "q8_0", "fp8", "f16"
	TensorOrder string   `json:"tensor_order"` // e.g. "canonical", "transposed"
	Features    []string `json:"features,omitempty"`
}

type DeviceRequirement struct {
	BackendName      string   `json:"backend_name"`      // "cpu-ref", "metal", "cuda", "vulkan"
	MinDeviceArch    string   `json:"min_device_arch"`   // e.g. "sm_80", "metal3"
	RequiredFeatures []string `json:"required_features"` // e.g. "fp16", "dotprod", "simd"
	MinMemoryBytes   int64    `json:"min_memory_bytes"`
}

type BackendSpec struct {
	Name      string            `json:"name"`
	DeviceReq DeviceRequirement `json:"device_req"`
	Priority  int               `json:"priority"` // higher number = higher priority
}

type ModelSpec struct {
	ID       string            `json:"id"`
	Family   string            `json:"family"`
	Formats  []FormatSignature `json:"formats"`
	Backends []BackendSpec     `json:"backends"`
}

type HostProfile struct {
	Arch       string          `json:"arch"`     // "amd64", "arm64"
	OS         string          `json:"os"`       // "darwin", "linux", "windows"
	Features   map[string]bool `json:"features"` // "fp16", "dotprod", "simd", "avx2"
	HasCUDA    bool            `json:"has_cuda"`
	CUDAArch   string          `json:"cuda_arch"` // e.g. "sm_89"
	HasMetal   bool            `json:"has_metal"`
	HasVulkan  bool            `json:"has_vulkan"`
	TotalRAM   int64           `json:"total_ram"`
	DeviceVRAM int64           `json:"device_vram"`
}

var (
	ErrNoMatchingBackend = errors.New("declarative registry: no matching backend for host profile")
	ErrModelNotFound     = errors.New("declarative registry: model not found")
)

// MatchDeviceRequirement verifies whether a host profile meets the device requirement.
func MatchDeviceRequirement(req DeviceRequirement, host HostProfile) bool {
	switch req.BackendName {
	case "cuda":
		if !host.HasCUDA {
			return false
		}
	case "metal":
		if !host.HasMetal {
			return false
		}
	case "vulkan":
		if !host.HasVulkan {
			return false
		}
	}

	if req.MinMemoryBytes > 0 && host.DeviceVRAM < req.MinMemoryBytes && host.TotalRAM < req.MinMemoryBytes {
		return false
	}

	for _, feat := range req.RequiredFeatures {
		if host.Features == nil || !host.Features[feat] {
			return false
		}
	}

	if req.BackendName == "cuda" && req.MinDeviceArch != "" && host.CUDAArch == "" {
		return false
	}
	if req.MinDeviceArch != "" && host.CUDAArch != "" {
		if !cudaArchSatisfied(req.MinDeviceArch, host.CUDAArch) {
			return false
		}
	}

	return true
}

func cudaArchSatisfied(minArch, hostArch string) bool {
	minVal, minOk := parseArchNumber(minArch)
	hostVal, hostOk := parseArchNumber(hostArch)
	if minOk && hostOk {
		return hostVal >= minVal
	}
	return hostArch >= minArch
}

func parseArchNumber(arch string) (int, bool) {
	s := strings.TrimPrefix(arch, "sm_")
	v, err := strconv.Atoi(s)
	if err != nil {
		return 0, false
	}
	return v, true
}

// ResolveBackend ranks matching backends for a model spec by priority descending.
func ResolveBackend(spec ModelSpec, host HostProfile) (BackendSpec, error) {
	var (
		found bool
		best  BackendSpec
	)
	for _, b := range spec.Backends {
		if MatchDeviceRequirement(b.DeviceReq, host) {
			if !found || b.Priority > best.Priority {
				best = b
				found = true
			}
		}
	}
	if !found {
		return BackendSpec{}, ErrNoMatchingBackend
	}
	return best, nil
}

var (
	registryMu    sync.RWMutex
	registrySpecs = make(map[string]ModelSpec)
)

// RegisterModelSpec registers a model specification into the thread-safe registry.
func RegisterModelSpec(spec ModelSpec) {
	registryMu.Lock()
	defer registryMu.Unlock()
	registrySpecs[spec.ID] = spec
}

// LookupModelSpec retrieves a model specification by ID.
func LookupModelSpec(id string) (ModelSpec, bool) {
	registryMu.RLock()
	defer registryMu.RUnlock()
	spec, ok := registrySpecs[id]
	return spec, ok
}

// ResolveModelBackend finds the model spec and resolves the best matching backend.
func ResolveModelBackend(id string, host HostProfile) (BackendSpec, error) {
	spec, ok := LookupModelSpec(id)
	if !ok {
		return BackendSpec{}, ErrModelNotFound
	}
	return ResolveBackend(spec, host)
}

// ClearModelSpecs resets the registry for testing.
func ClearModelSpecs() {
	registryMu.Lock()
	defer registryMu.Unlock()
	registrySpecs = make(map[string]ModelSpec)
}
