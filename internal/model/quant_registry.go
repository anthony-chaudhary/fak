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
		{kindIQ3XXS, "IQ3_XXS", compute.IQ3_XXS},
		{kindIQ4XS, "IQ4_XS", 0},
		{kindIQ2XXS, "IQ2_XXS", 0},
		{kindIQ2XS, "IQ2_XS", 0},
		{kindIQ1S, "IQ1_S", 0},
		{kindIQ2S, "IQ2_S", 0},
		{kindIQ1M, "IQ1_M", 0},
		{kindQ8_0, "Q8_0", compute.Q8_0},
		{kindQ4_0, "Q4_0", 0},
		{kindQ3K, "Q3_K", 0},
		{kindIQ3S, "IQ3_S", compute.IQ3_S},
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

// IQ3XXSCapabilityBackend is the optional capability interface that a device backend
// implements to declare native execution and staging support for IQ3_XXS weights.
type IQ3XXSCapabilityBackend interface {
	SupportsIQ3XXS() bool
}

// IQ3XXSHALRefusalReason represents a closed-vocabulary failure reason for IQ3_XXS HAL admission.
type IQ3XXSHALRefusalReason string

const (
	IQ3XXSRefusalNilBackend       IQ3XXSHALRefusalReason = "NIL_BACKEND"
	IQ3XXSRefusalNoCapability     IQ3XXSHALRefusalReason = "NO_BACKEND_CAPABILITY"
	IQ3XXSRefusalCapabilityDenied IQ3XXSHALRefusalReason = "CAPABILITY_DENIED"
	IQ3XXSRefusalHALNotRegistered IQ3XXSHALRefusalReason = "HAL_NOT_REGISTERED"
)

// IQ3XXSHALAdmissionRefusal is the typed, fail-closed refusal returned when
// IQ3_XXS HAL admission cannot be granted.
type IQ3XXSHALAdmissionRefusal struct {
	Kind   kQuantKind
	Dtype  compute.Dtype
	Reason IQ3XXSHALRefusalReason
	Detail string
}

func (e *IQ3XXSHALAdmissionRefusal) Error() string {
	return fmt.Sprintf("model: IQ3_XXS HAL admission refused (kind=%s, dtype=%s): %s - %s",
		e.Kind, e.Dtype, e.Reason, e.Detail)
}

// IQ3XXSHALAdmissionVerdict represents the result of evaluating IQ3_XXS HAL admission.
type IQ3XXSHALAdmissionVerdict struct {
	Admitted bool
	Dtype    compute.Dtype
	Refusal  *IQ3XXSHALAdmissionRefusal
}

// AdmitIQ3XXSHAL evaluates the model-to-HAL admission contract for IQ3_XXS weights.
// By default, execution remains denied until a capable backend (implementing
// IQ3XXSCapabilityBackend with SupportsIQ3XXS() == true) is provided or a registered
// HAL descriptor is present.
func AdmitIQ3XXSHAL(be compute.Backend) IQ3XXSHALAdmissionVerdict {
	desc, ok := LookupQuantDescriptor(kindIQ3XXS)
	dtype := compute.IQ3_XXS
	if ok && desc.Dtype() != 0 {
		dtype = desc.Dtype()
	}

	if be == nil {
		return IQ3XXSHALAdmissionVerdict{
			Admitted: false,
			Dtype:    dtype,
			Refusal: &IQ3XXSHALAdmissionRefusal{
				Kind:   kindIQ3XXS,
				Dtype:  dtype,
				Reason: IQ3XXSRefusalNilBackend,
				Detail: "nil compute backend provided",
			},
		}
	}

	// 1. Check if backend implements explicit capability
	capable, isCapable := be.(IQ3XXSCapabilityBackend)
	if isCapable && capable.SupportsIQ3XXS() {
		return IQ3XXSHALAdmissionVerdict{
			Admitted: true,
			Dtype:    dtype,
		}
	}

	// 2. Check if descriptor in registry was explicitly registered to support HAL
	if ok && desc.SupportsHAL() {
		return IQ3XXSHALAdmissionVerdict{
			Admitted: true,
			Dtype:    dtype,
		}
	}

	var reason IQ3XXSHALRefusalReason = IQ3XXSRefusalNoCapability
	var detail = "backend does not implement SupportsIQ3XXS capability"
	if isCapable && !capable.SupportsIQ3XXS() {
		reason = IQ3XXSRefusalCapabilityDenied
		detail = "backend explicitly reported SupportsIQ3XXS() == false"
	}

	return IQ3XXSHALAdmissionVerdict{
		Admitted: false,
		Dtype:    dtype,
		Refusal: &IQ3XXSHALAdmissionRefusal{
			Kind:   kindIQ3XXS,
			Dtype:  dtype,
			Reason: reason,
			Detail: detail,
		},
	}
}

// AdmitHALQuant evaluates the model-to-HAL admission contract for a quantized format.
// Returns the compute.Dtype if admitted, or a typed refusal error if denied.
func AdmitHALQuant(kind kQuantKind, be compute.Backend) (compute.Dtype, error) {
	if kind == kindIQ3XXS {
		v := AdmitIQ3XXSHAL(be)
		if !v.Admitted {
			return v.Dtype, v.Refusal
		}
		return v.Dtype, nil
	}
	if kind == kindIQ3S {
		v := AdmitIQ3SHAL(be)
		if !v.Admitted {
			return v.Dtype, v.Refusal
		}
		return v.Dtype, nil
	}
	desc, ok := LookupQuantDescriptor(kind)
	if !ok {
		return 0, fmt.Errorf("model: quant kind %s not registered", kind)
	}
	if !desc.SupportsHAL() {
		return desc.Dtype(), fmt.Errorf("model: quant kind %s does not support device HAL staging", kind)
	}
	return desc.Dtype(), nil
}

// IQ3SCapabilityBackend is the optional capability interface that a device backend
// implements to declare native execution and staging support for IQ3_S weights.
type IQ3SCapabilityBackend interface {
	SupportsIQ3S() bool
}

// IQ3SHALRefusalReason represents a closed-vocabulary failure reason for IQ3_S HAL admission.
type IQ3SHALRefusalReason string

const (
	IQ3SRefusalNilBackend       IQ3SHALRefusalReason = "NIL_BACKEND"
	IQ3SRefusalNoCapability     IQ3SHALRefusalReason = "NO_BACKEND_CAPABILITY"
	IQ3SRefusalCapabilityDenied IQ3SHALRefusalReason = "CAPABILITY_DENIED"
	IQ3SRefusalHALNotRegistered IQ3SHALRefusalReason = "HAL_NOT_REGISTERED"
)

// IQ3SHALAdmissionRefusal is the typed, fail-closed refusal returned when
// IQ3_S HAL admission cannot be granted.
type IQ3SHALAdmissionRefusal struct {
	Kind   kQuantKind
	Dtype  compute.Dtype
	Reason IQ3SHALRefusalReason
	Detail string
}

func (e *IQ3SHALAdmissionRefusal) Error() string {
	return fmt.Sprintf("model: IQ3_S HAL admission refused (kind=%s, dtype=%s): %s - %s",
		e.Kind, e.Dtype, e.Reason, e.Detail)
}

// IQ3SHALAdmissionVerdict represents the result of evaluating IQ3_S HAL admission.
type IQ3SHALAdmissionVerdict struct {
	Admitted bool
	Dtype    compute.Dtype
	Refusal  *IQ3SHALAdmissionRefusal
}

// AdmitIQ3SHAL evaluates the model-to-HAL admission contract for IQ3_S weights.
// By default, execution remains denied until a capable backend (implementing
// IQ3SCapabilityBackend with SupportsIQ3S() == true) is provided.
func AdmitIQ3SHAL(be compute.Backend) IQ3SHALAdmissionVerdict {
	desc, ok := LookupQuantDescriptor(kindIQ3S)
	dtype := compute.IQ3_S
	if ok && desc.Dtype() != 0 {
		dtype = desc.Dtype()
	}

	if be == nil {
		return IQ3SHALAdmissionVerdict{
			Admitted: false,
			Dtype:    dtype,
			Refusal: &IQ3SHALAdmissionRefusal{
				Kind:   kindIQ3S,
				Dtype:  dtype,
				Reason: IQ3SRefusalNilBackend,
				Detail: "nil compute backend provided",
			},
		}
	}

	// 1. Check if backend implements explicit capability
	capable, isCapable := be.(IQ3SCapabilityBackend)
	if !isCapable {
		return IQ3SHALAdmissionVerdict{
			Admitted: false,
			Dtype:    dtype,
			Refusal: &IQ3SHALAdmissionRefusal{
				Kind:   kindIQ3S,
				Dtype:  dtype,
				Reason: IQ3SRefusalNoCapability,
				Detail: "backend does not implement SupportsIQ3S capability",
			},
		}
	}

	if !capable.SupportsIQ3S() {
		return IQ3SHALAdmissionVerdict{
			Admitted: false,
			Dtype:    dtype,
			Refusal: &IQ3SHALAdmissionRefusal{
				Kind:   kindIQ3S,
				Dtype:  dtype,
				Reason: IQ3SRefusalCapabilityDenied,
				Detail: "backend explicitly reported SupportsIQ3S() == false",
			},
		}
	}

	return IQ3SHALAdmissionVerdict{
		Admitted: true,
		Dtype:    dtype,
	}
}
