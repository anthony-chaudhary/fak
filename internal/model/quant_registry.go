package model

import (
	"fmt"
	"sync"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// QuantDescriptor describes a resident quantized tensor format and its HAL weight staging capabilities.
type QuantDescriptor interface {
	// Kind returns the underlying kQuantKind enum.
	Kind() kQuantKind
	// Name returns the canonical format name (e.g., "Q5_K", "Q6_K", "Q2_K").
	Name() string
	// Dtype returns the corresponding compute.Dtype used on compute backends.
	Dtype() compute.Dtype
	// KeyPrefix returns the cache/staging key prefix for halW (e.g., "kquant-raw:").
	KeyPrefix() string
	// SupportsHAL reports whether this format is supported for device HAL weight staging.
	SupportsHAL() bool
	// BlockBytes returns the byte length of one super-block.
	BlockBytes() int
	// BlockWeights returns the element count per super-block.
	BlockWeights() int
	// NewHostTensor constructs an un-uploaded host compute.Tensor wrapping the raw bytes.
	NewHostTensor(out, in int, raw []byte) compute.Tensor
}

// BaseQuantDescriptor is a concrete value implementation of QuantDescriptor.
type BaseQuantDescriptor struct {
	QuantKind     kQuantKind
	QuantName     string
	ComputeDtype  compute.Dtype
	Prefix        string
	HALSupported  bool
	BytesPerBlk   int
	WeightsPerBlk int
	HostTensorFn  func(out, in int, raw []byte) compute.Tensor
}

func (d BaseQuantDescriptor) Kind() kQuantKind     { return d.QuantKind }
func (d BaseQuantDescriptor) Name() string         { return d.QuantName }
func (d BaseQuantDescriptor) Dtype() compute.Dtype { return d.ComputeDtype }
func (d BaseQuantDescriptor) KeyPrefix() string {
	if d.Prefix != "" {
		return d.Prefix
	}
	return "kquant-raw:"
}
func (d BaseQuantDescriptor) SupportsHAL() bool { return d.HALSupported }
func (d BaseQuantDescriptor) BlockBytes() int   { return d.BytesPerBlk }
func (d BaseQuantDescriptor) BlockWeights() int { return d.WeightsPerBlk }
func (d BaseQuantDescriptor) NewHostTensor(out, in int, raw []byte) compute.Tensor {
	if d.HostTensorFn != nil {
		return d.HostTensorFn(out, in, raw)
	}
	panic(fmt.Sprintf("model: host tensor construction not supported for %s", d.QuantName))
}

type quantRegistry struct {
	mu     sync.RWMutex
	byKind map[kQuantKind]QuantDescriptor
	byName map[string]QuantDescriptor
}

var globalQuantRegistry = newQuantRegistry()

func newQuantRegistry() *quantRegistry {
	return &quantRegistry{
		byKind: make(map[kQuantKind]QuantDescriptor),
		byName: make(map[string]QuantDescriptor),
	}
}

func init() {
	resetDefaultQuantDescriptors()
}

func resetDefaultQuantDescriptors() {
	globalQuantRegistry.mu.Lock()
	defer globalQuantRegistry.mu.Unlock()
	globalQuantRegistry.byKind = make(map[kQuantKind]QuantDescriptor)
	globalQuantRegistry.byName = make(map[string]QuantDescriptor)

	registerDefaultLocked(BaseQuantDescriptor{
		QuantKind:     kindQ5K,
		QuantName:     "Q5_K",
		ComputeDtype:  compute.Q5_K,
		Prefix:        "kquant-raw:",
		HALSupported:  true,
		BytesPerBlk:   q5kBlockBytes,
		WeightsPerBlk: qkK,
		HostTensorFn: func(out, in int, raw []byte) compute.Tensor {
			return compute.NewQ5K(compute.Default(), []int{out, in}, raw)
		},
	})
	registerDefaultLocked(BaseQuantDescriptor{
		QuantKind:     kindQ6K,
		QuantName:     "Q6_K",
		ComputeDtype:  compute.Q6_K,
		Prefix:        "kquant-raw:",
		HALSupported:  true,
		BytesPerBlk:   q6kBlockBytes,
		WeightsPerBlk: qkK,
		HostTensorFn: func(out, in int, raw []byte) compute.Tensor {
			return compute.NewQ6K(compute.Default(), []int{out, in}, raw)
		},
	})
	registerDefaultLocked(BaseQuantDescriptor{
		QuantKind:     kindQ2K,
		QuantName:     "Q2_K",
		ComputeDtype:  compute.Q2_K,
		Prefix:        "kquant-raw:",
		HALSupported:  true,
		BytesPerBlk:   q2kBlockBytes,
		WeightsPerBlk: qkK,
		HostTensorFn: func(out, in int, raw []byte) compute.Tensor {
			return compute.NewQ2K(compute.Default(), []int{out, in}, raw)
		},
	})

	nonHAL := []struct {
		kind  kQuantKind
		name  string
		dtype compute.Dtype
	}{
		{kindIQ3XXS, "IQ3_XXS", 0},
		{kindIQ4XS, "IQ4_XS", 0},
		{kindIQ2XXS, "IQ2_XXS", 0},
		{kindIQ2XS, "IQ2_XS", 0},
		{kindIQ1S, "IQ1_S", 0},
		{kindIQ2S, "IQ2_S", 0},
		{kindIQ1M, "IQ1_M", 0},
		{kindQ8_0, "Q8_0", compute.Q8_0},
		{kindQ4_0, "Q4_0", 0},
	}
	for _, item := range nonHAL {
		registerDefaultLocked(BaseQuantDescriptor{
			QuantKind:     item.kind,
			QuantName:     item.name,
			ComputeDtype:  item.dtype,
			Prefix:        "kquant-raw:",
			HALSupported:  false,
			BytesPerBlk:   item.kind.blockBytes(),
			WeightsPerBlk: item.kind.blockWeights(),
		})
	}
}

func registerDefaultLocked(desc QuantDescriptor) {
	globalQuantRegistry.byKind[desc.Kind()] = desc
	globalQuantRegistry.byName[desc.Name()] = desc
}

// RegisterQuantDescriptor registers a descriptor in the global quant registry.
// If a descriptor for the same kind or name already exists, it is overwritten.
func RegisterQuantDescriptor(desc QuantDescriptor) {
	if desc == nil {
		panic("model: cannot register nil QuantDescriptor")
	}
	globalQuantRegistry.mu.Lock()
	defer globalQuantRegistry.mu.Unlock()
	globalQuantRegistry.byKind[desc.Kind()] = desc
	globalQuantRegistry.byName[desc.Name()] = desc
}

// LookupQuantDescriptor looks up a descriptor by its kQuantKind.
func LookupQuantDescriptor(kind kQuantKind) (QuantDescriptor, bool) {
	globalQuantRegistry.mu.RLock()
	defer globalQuantRegistry.mu.RUnlock()
	desc, ok := globalQuantRegistry.byKind[kind]
	return desc, ok
}

// LookupQuantDescriptorByName looks up a descriptor by its canonical name (e.g. "Q5_K").
func LookupQuantDescriptorByName(name string) (QuantDescriptor, bool) {
	globalQuantRegistry.mu.RLock()
	defer globalQuantRegistry.mu.RUnlock()
	desc, ok := globalQuantRegistry.byName[name]
	return desc, ok
}

// SupportsHALKQuant reports whether kind is registered and supports device HAL weight staging.
func SupportsHALKQuant(kind kQuantKind) bool {
	desc, ok := LookupQuantDescriptor(kind)
	return ok && desc.SupportsHAL()
}

// RegisteredQuantDescriptors returns a snapshot slice of all registered descriptors.
func RegisteredQuantDescriptors() []QuantDescriptor {
	globalQuantRegistry.mu.RLock()
	defer globalQuantRegistry.mu.RUnlock()
	descs := make([]QuantDescriptor, 0, len(globalQuantRegistry.byKind))
	for _, d := range globalQuantRegistry.byKind {
		descs = append(descs, d)
	}
	return descs
}

// ResetDefaultQuantDescriptors resets the global registry to its default built-in descriptors.
// This is primarily intended for test isolation.
func ResetDefaultQuantDescriptors() {
	resetDefaultQuantDescriptors()
}
