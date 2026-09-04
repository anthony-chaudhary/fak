package compute

import (
	"errors"
	"fmt"
	"strings"
)

// Typed sentinel errors for Vulkan compute failures.
var (
	// ErrVulkanDeviceLost indicates that the Vulkan logical or physical device was lost,
	// reset, or encountered an unrecoverable driver hang.
	ErrVulkanDeviceLost = errors.New("compute: vulkan device lost")

	// ErrVulkanAllocationFailed indicates that a device-local or host-visible buffer allocation failed
	// (e.g. out of memory, fvk_malloc returned nil, or host-visible fallback failed).
	ErrVulkanAllocationFailed = errors.New("compute: vulkan allocation failed")

	// ErrVulkanSubmissionFailed indicates that command buffer recording, queue submission,
	// batch flush, or fence synchronization failed.
	ErrVulkanSubmissionFailed = errors.New("compute: vulkan submission failed")

	// ErrVulkanInvalidGeometry indicates tensor shape, dimension, dtype, stride, or
	// index incompatibility rejected before or during Vulkan operation dispatch.
	ErrVulkanInvalidGeometry = errors.New("compute: vulkan invalid geometry")

	// ErrVulkanResourceExhausted indicates that a requested buffer exceeds device limits
	// (e.g. maxStorageBufferRange, maxMemoryAllocationSize, single-resource cap) or residency budget.
	ErrVulkanResourceExhausted = errors.New("compute: vulkan resource exhausted")

	// ErrVulkanExecutionFailed indicates a shader invocation, pipeline execution, or compute dispatch failure.
	ErrVulkanExecutionFailed = errors.New("compute: vulkan execution failed")
)

// VulkanErrorClass categorizes the failure domain of a Vulkan compute operation.
type VulkanErrorClass string

const (
	VulkanClassDeviceLost        VulkanErrorClass = "device_lost"
	VulkanClassAllocationFailed  VulkanErrorClass = "allocation_failed"
	VulkanClassSubmissionFailed  VulkanErrorClass = "submission_failed"
	VulkanClassInvalidGeometry   VulkanErrorClass = "invalid_geometry"
	VulkanClassResourceExhausted VulkanErrorClass = "resource_exhausted"
	VulkanClassExecutionFailed   VulkanErrorClass = "execution_failed"
	VulkanClassUnknown           VulkanErrorClass = "unknown"
)

// BackendError is the typed error returned when a compute backend operation fails
// or recovers from an in-flight request panic.
type BackendError struct {
	Backend   string           // backend identifier, e.g. "vulkan"
	Class     VulkanErrorClass // failure classification
	Site      string           // call site or operation name (e.g. "MatMul", "dalloc", "Upload")
	Message   string           // human-readable message or original panic representation
	Err       error            // typed sentinel error (e.g. ErrVulkanAllocationFailed)
	Recovered any              // original panic value or wrapped cause
}

// VulkanBackendError is an alias for BackendError when attributing to Vulkan.
type VulkanBackendError = BackendError

// Error formats the backend failure for operator logs and gateway diagnostic returns.
func (e *BackendError) Error() string {
	if e == nil {
		return "compute: nil backend error"
	}
	var b strings.Builder
	b.WriteString("compute: ")
	if e.Backend != "" {
		b.WriteString(e.Backend)
		b.WriteString(" ")
	}
	if e.Class != "" {
		b.WriteString("[")
		b.WriteString(string(e.Class))
		b.WriteString("] ")
	}
	if e.Site != "" {
		b.WriteString("at ")
		b.WriteString(e.Site)
		b.WriteString(": ")
	}
	if e.Message != "" {
		b.WriteString(e.Message)
	} else if e.Err != nil {
		b.WriteString(e.Err.Error())
	} else {
		b.WriteString("unknown failure")
	}
	return b.String()
}

// Unwrap supports errors.Is and errors.As against both the typed sentinel and any wrapped cause.
func (e *BackendError) Unwrap() []error {
	if e == nil {
		return nil
	}
	var errs []error
	if e.Err != nil {
		errs = append(errs, e.Err)
	}
	if recErr, ok := e.Recovered.(error); ok && recErr != nil && recErr != e.Err {
		errs = append(errs, recErr)
	}
	return errs
}

// Is reports whether target matches the sentinel error or recovered cause.
func (e *BackendError) Is(target error) bool {
	if e == nil || target == nil {
		return false
	}
	if e.Err != nil && errors.Is(e.Err, target) {
		return true
	}
	if recErr, ok := e.Recovered.(error); ok && errors.Is(recErr, target) {
		return true
	}
	return false
}

// IsDeviceLost reports whether the error represents a lost or reset device.
func (e *BackendError) IsDeviceLost() bool {
	return e != nil && (e.Class == VulkanClassDeviceLost || errors.Is(e.Err, ErrVulkanDeviceLost))
}

// IsAllocation reports whether the error represents an allocation failure.
func (e *BackendError) IsAllocation() bool {
	return e != nil && (e.Class == VulkanClassAllocationFailed || errors.Is(e.Err, ErrVulkanAllocationFailed))
}

// IsSubmissionFailed reports whether the error represents a submission or fence synchronization failure.
func (e *BackendError) IsSubmissionFailed() bool {
	return e != nil && (e.Class == VulkanClassSubmissionFailed || errors.Is(e.Err, ErrVulkanSubmissionFailed))
}

// IsInvalidGeometry reports whether the error represents a shape, dimension, or dtype mismatch.
func (e *BackendError) IsInvalidGeometry() bool {
	return e != nil && (e.Class == VulkanClassInvalidGeometry || errors.Is(e.Err, ErrVulkanInvalidGeometry))
}

// IsResourceExhausted reports whether the error represents a hardware resource cap or budget limit.
func (e *BackendError) IsResourceExhausted() bool {
	return e != nil && (e.Class == VulkanClassResourceExhausted || errors.Is(e.Err, ErrVulkanResourceExhausted))
}

// IsRecoverable reports whether the failure is recoverable at request boundaries without restarting the device.
func (e *BackendError) IsRecoverable() bool {
	if e == nil {
		return true
	}
	return !e.IsDeviceLost()
}

// VulkanErrorRule defines one entry in the Vulkan error classification table.
type VulkanErrorRule struct {
	Domain      string           // failure domain: "device", "allocation", "capacity", "submission", "geometry", "execution"
	Class       VulkanErrorClass // error class
	TypedError  error            // canonical sentinel error
	Keywords    []string         // case-insensitive substrings matching this rule
	Description string           // explanation of this failure classification
}

// VulkanErrorInventory maintains the checked inventory of Vulkan error mappings.
type VulkanErrorInventory struct {
	Rules []VulkanErrorRule
}

// DefaultVulkanErrorInventory returns the static, checked inventory of Vulkan error classifications.
func DefaultVulkanErrorInventory() VulkanErrorInventory {
	return VulkanErrorInventory{
		Rules: []VulkanErrorRule{
			{
				Domain:     "device",
				Class:      VulkanClassDeviceLost,
				TypedError: ErrVulkanDeviceLost,
				Keywords: []string{
					"device lost",
					"vk_error_device_lost",
					"device reset",
					"device removed",
					"driver timeout",
					"device hung",
				},
				Description: "Vulkan logical or physical device lost or driver reset",
			},
			{
				Domain:     "capacity",
				Class:      VulkanClassResourceExhausted,
				TypedError: ErrVulkanResourceExhausted,
				Keywords: []string{
					"single-resource cap",
					"exceeds device single-resource cap",
					"maxstoragebufferrange",
					"maxmemoryallocationsize",
					"maxbufferbytes",
					"budget exceeded",
					"resource cap",
					"resource exhausted",
				},
				Description: "Buffer or allocation exceeds device limits or residency budget",
			},
			{
				Domain:     "allocation",
				Class:      VulkanClassAllocationFailed,
				TypedError: ErrVulkanAllocationFailed,
				Keywords: []string{
					"allocation of",
					"out of memory",
					"vk_error_out_of_device_memory",
					"vk_error_out_of_host_memory",
					"deviceallocerror",
					"fvk_malloc",
					"dalloc",
					"allocation failed",
					"malloc failed",
				},
				Description: "Buffer or memory allocation failure on host or device",
			},
			{
				Domain:     "submission",
				Class:      VulkanClassSubmissionFailed,
				TypedError: ErrVulkanSubmissionFailed,
				Keywords: []string{
					"submission",
					"queue submit",
					"vk_error_initialization_failed",
					"fvk_batch",
					"fence wait timeout",
					"command buffer",
					"queue wait",
					"flush batch",
					"submission failed",
				},
				Description: "Command buffer recording, queue submission, or synchronization failure",
			},
			{
				Domain:     "geometry",
				Class:      VulkanClassInvalidGeometry,
				TypedError: ErrVulkanInvalidGeometry,
				Keywords: []string{
					"unsupported weight dtype",
					"unsupported weight",
					"supports only",
					"shape mismatch",
					"does not match shape",
					"divisible",
					"out of range",
					"missing quantspec",
					"expects host data",
					"expects a 2d",
					"expects one input row",
					"input dims differ",
					"gate/up shapes differ",
					"dst shape does not match",
					"missing q8 chunk",
					"missing device scale buffer",
					"decode-only",
					"norm weight must be",
					"missing quant",
					"row out of range",
					"invalid geometry",
					"does not match kv config",
					"shape does not match",
					"missing q8 chunk device buffers",
				},
				Description: "Tensor shape, dimension, layout, stride, or data type mismatch",
			},
			{
				Domain:     "execution",
				Class:      VulkanClassExecutionFailed,
				TypedError: ErrVulkanExecutionFailed,
				Keywords: []string{
					"execution failed",
					"shader invocation",
					"pipeline execution",
					"compute dispatch failed",
					"kernel error",
				},
				Description: "Shader, kernel, or pipeline dispatch execution failure",
			},
		},
	}
}

// Validate checks the internal consistency, coverage, and uniqueness of the inventory.
func (inv VulkanErrorInventory) Validate() error {
	if len(inv.Rules) == 0 {
		return errors.New("vulkan error inventory contains no rules")
	}
	seenKeywords := make(map[string]string)
	seenDomains := make(map[string]bool)

	for i, rule := range inv.Rules {
		if rule.Domain == "" {
			return fmt.Errorf("vulkan error rule #%d has empty domain", i)
		}
		if rule.Class == "" {
			return fmt.Errorf("vulkan error rule #%d (%s) has empty class", i, rule.Domain)
		}
		if rule.TypedError == nil {
			return fmt.Errorf("vulkan error rule #%d (%s) has nil typed error", i, rule.Domain)
		}
		if len(rule.Keywords) == 0 {
			return fmt.Errorf("vulkan error rule #%d (%s) has no keywords", i, rule.Domain)
		}
		if rule.Description == "" {
			return fmt.Errorf("vulkan error rule #%d (%s) has empty description", i, rule.Domain)
		}
		seenDomains[rule.Domain] = true

		for _, kw := range rule.Keywords {
			kwLower := strings.ToLower(strings.TrimSpace(kw))
			if kwLower == "" {
				return fmt.Errorf("vulkan error rule (%s) contains blank keyword", rule.Domain)
			}
			if existingDomain, exists := seenKeywords[kwLower]; exists {
				return fmt.Errorf("vulkan error keyword %q in domain %q duplicated from domain %q", kw, rule.Domain, existingDomain)
			}
			seenKeywords[kwLower] = rule.Domain
		}
	}

	requiredDomains := []string{"device", "capacity", "allocation", "submission", "geometry", "execution"}
	for _, req := range requiredDomains {
		if !seenDomains[req] {
			return fmt.Errorf("vulkan error inventory missing required domain %q", req)
		}
	}
	return nil
}

// ClassifyVulkanPanic converts an arbitrary recovered panic value into a typed *BackendError.
func ClassifyVulkanPanic(recovered any, site string) *BackendError {
	if recovered == nil {
		return nil
	}

	// 1. If already a BackendError, preserve it and update site if empty.
	if be, ok := recovered.(*BackendError); ok && be != nil {
		if be.Site == "" {
			be.Site = site
		}
		return be
	}

	// 2. Direct check for known typed error types.
	if dae, ok := recovered.(*DeviceAllocError); ok && dae != nil {
		allocSite := site
		if allocSite == "" {
			allocSite = dae.Site
		}
		return &BackendError{
			Backend:   "vulkan",
			Class:     VulkanClassAllocationFailed,
			Site:      allocSite,
			Message:   dae.Error(),
			Err:       ErrVulkanAllocationFailed,
			Recovered: dae,
		}
	}

	if dfe, ok := recovered.(*DeviceFaultError); ok && dfe != nil {
		faultSite := site
		if faultSite == "" {
			faultSite = dfe.Site
		}
		return &BackendError{
			Backend:   "vulkan",
			Class:     VulkanClassDeviceLost,
			Site:      faultSite,
			Message:   dfe.Error(),
			Err:       ErrVulkanDeviceLost,
			Recovered: dfe,
		}
	}

	// 3. If it's an error, check if it wraps a known typed error or sentinel.
	if err, ok := recovered.(error); ok && err != nil {
		var be *BackendError
		if errors.As(err, &be) && be != nil {
			if be.Site == "" {
				be.Site = site
			}
			return be
		}
		var dae *DeviceAllocError
		if errors.As(err, &dae) && dae != nil {
			allocSite := site
			if allocSite == "" {
				allocSite = dae.Site
			}
			return &BackendError{
				Backend:   "vulkan",
				Class:     VulkanClassAllocationFailed,
				Site:      allocSite,
				Message:   err.Error(),
				Err:       ErrVulkanAllocationFailed,
				Recovered: err,
			}
		}
		var dfe *DeviceFaultError
		if errors.As(err, &dfe) && dfe != nil {
			faultSite := site
			if faultSite == "" {
				faultSite = dfe.Site
			}
			return &BackendError{
				Backend:   "vulkan",
				Class:     VulkanClassDeviceLost,
				Site:      faultSite,
				Message:   err.Error(),
				Err:       ErrVulkanDeviceLost,
				Recovered: err,
			}
		}

		switch {
		case errors.Is(err, ErrVulkanDeviceLost):
			return &BackendError{
				Backend:   "vulkan",
				Class:     VulkanClassDeviceLost,
				Site:      site,
				Message:   err.Error(),
				Err:       ErrVulkanDeviceLost,
				Recovered: err,
			}
		case errors.Is(err, ErrVulkanResourceExhausted):
			return &BackendError{
				Backend:   "vulkan",
				Class:     VulkanClassResourceExhausted,
				Site:      site,
				Message:   err.Error(),
				Err:       ErrVulkanResourceExhausted,
				Recovered: err,
			}
		case errors.Is(err, ErrVulkanAllocationFailed):
			return &BackendError{
				Backend:   "vulkan",
				Class:     VulkanClassAllocationFailed,
				Site:      site,
				Message:   err.Error(),
				Err:       ErrVulkanAllocationFailed,
				Recovered: err,
			}
		case errors.Is(err, ErrVulkanSubmissionFailed):
			return &BackendError{
				Backend:   "vulkan",
				Class:     VulkanClassSubmissionFailed,
				Site:      site,
				Message:   err.Error(),
				Err:       ErrVulkanSubmissionFailed,
				Recovered: err,
			}
		case errors.Is(err, ErrVulkanInvalidGeometry):
			return &BackendError{
				Backend:   "vulkan",
				Class:     VulkanClassInvalidGeometry,
				Site:      site,
				Message:   err.Error(),
				Err:       ErrVulkanInvalidGeometry,
				Recovered: err,
			}
		case errors.Is(err, ErrVulkanExecutionFailed):
			return &BackendError{
				Backend:   "vulkan",
				Class:     VulkanClassExecutionFailed,
				Site:      site,
				Message:   err.Error(),
				Err:       ErrVulkanExecutionFailed,
				Recovered: err,
			}
		}
	}

	// 4. Extract message string for keyword matching against inventory.
	var msg string
	switch v := recovered.(type) {
	case string:
		msg = v
	case fmt.Stringer:
		msg = v.String()
	case error:
		msg = v.Error()
	default:
		msg = fmt.Sprintf("%v", v)
	}

	msgLower := strings.ToLower(msg)
	inv := DefaultVulkanErrorInventory()

	// Match against inventory rules in priority order.
	for _, rule := range inv.Rules {
		for _, kw := range rule.Keywords {
			if strings.Contains(msgLower, kw) {
				return &BackendError{
					Backend:   "vulkan",
					Class:     rule.Class,
					Site:      site,
					Message:   msg,
					Err:       rule.TypedError,
					Recovered: recovered,
				}
			}
		}
	}

	// Default fallback: unknown class
	return &BackendError{
		Backend:   "vulkan",
		Class:     VulkanClassUnknown,
		Site:      site,
		Message:   msg,
		Err:       ErrVulkanExecutionFailed,
		Recovered: recovered,
	}
}

// ClassifyVulkanError classifies an error into a typed *BackendError.
// If err is already a *BackendError, it is returned unchanged.
// If err is nil, it returns nil.
func ClassifyVulkanError(err error, site string) *BackendError {
	if err == nil {
		return nil
	}
	return ClassifyVulkanPanic(err, site)
}

// CatchVulkanPanic recovers from a panic and assigns the converted *BackendError to *err.
// It is intended to be called with defer:
//
//	defer CatchVulkanPanic("MatMul", &err)
func CatchVulkanPanic(site string, err *error) {
	if r := recover(); r != nil {
		if err != nil {
			*err = ClassifyVulkanPanic(r, site)
		}
	}
}

// CatchVulkanPanicHandler recovers from a panic and passes the converted *BackendError to handler.
func CatchVulkanPanicHandler(site string, handler func(*BackendError)) {
	if r := recover(); r != nil {
		if handler != nil {
			handler(ClassifyVulkanPanic(r, site))
		}
	}
}

// SafeVulkanOp executes fn and converts any panic into a *BackendError.
func SafeVulkanOp(site string, fn func()) (err error) {
	defer CatchVulkanPanic(site, &err)
	fn()
	return nil
}

// SafeVulkanRun executes fn and converts any panic into a *BackendError,
// returning existing errors unchanged if no panic occurred.
func SafeVulkanRun(site string, fn func() error) (err error) {
	defer CatchVulkanPanic(site, &err)
	return fn()
}

// SafeVulkanCall executes fn and converts any panic into a *BackendError.
func SafeVulkanCall[T any](site string, fn func() (T, error)) (val T, err error) {
	defer CatchVulkanPanic(site, &err)
	return fn()
}

// SafeVulkanValue executes fn and converts any panic into a *BackendError.
func SafeVulkanValue[T any](site string, fn func() T) (val T, err error) {
	defer CatchVulkanPanic(site, &err)
	return fn(), nil
}

// ExecuteVulkanRequest executes a request function fn under a Vulkan request boundary.
// It establishes a RequestLifetime, catches any request-time panic, converts it into
// a typed *BackendError, and guarantees request resource retirement so subsequent requests
// can proceed.
func ExecuteVulkanRequest[T any](backend Backend, site string, fn func() (T, error)) (res T, err error) {
	lifetime := BeginRequest(backend)
	defer lifetime.Retire()
	defer CatchVulkanPanic(site, &err)

	return fn()
}
