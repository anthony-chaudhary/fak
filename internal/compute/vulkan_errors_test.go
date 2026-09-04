package compute

import (
	"errors"
	"fmt"
	"strings"
	"sync"
	"testing"
)

func TestVulkanErrorsSentinelsAndUnwrapping(t *testing.T) {
	sentinels := []struct {
		name string
		err  error
		want string
	}{
		{"DeviceLost", ErrVulkanDeviceLost, "compute: vulkan device lost"},
		{"AllocationFailed", ErrVulkanAllocationFailed, "compute: vulkan allocation failed"},
		{"SubmissionFailed", ErrVulkanSubmissionFailed, "compute: vulkan submission failed"},
		{"InvalidGeometry", ErrVulkanInvalidGeometry, "compute: vulkan invalid geometry"},
		{"ResourceExhausted", ErrVulkanResourceExhausted, "compute: vulkan resource exhausted"},
		{"ExecutionFailed", ErrVulkanExecutionFailed, "compute: vulkan execution failed"},
	}

	for _, s := range sentinels {
		t.Run(s.name, func(t *testing.T) {
			if s.err.Error() != s.want {
				t.Fatalf("sentinel error text = %q, want %q", s.err.Error(), s.want)
			}
		})
	}

	// Test BackendError formatting, Unwrap, and errors.Is
	be := &BackendError{
		Backend:   "vulkan",
		Class:     VulkanClassAllocationFailed,
		Site:      "dallocForClass",
		Message:   "device out of memory",
		Err:       ErrVulkanAllocationFailed,
		Recovered: errors.New("underlying driver failure"),
	}

	if !errors.Is(be, ErrVulkanAllocationFailed) {
		t.Fatal("errors.Is(be, ErrVulkanAllocationFailed) = false, want true")
	}

	// Check unwrapping of recovered cause
	if !errors.Is(be, be.Recovered.(error)) {
		t.Fatal("errors.Is(be, recoveredError) = false, want true")
	}

	// Check errors.As for *BackendError
	var targetBE *BackendError
	if !errors.As(be, &targetBE) {
		t.Fatal("errors.As(be, &targetBE) = false, want true")
	}
	if targetBE.Site != "dallocForClass" {
		t.Fatalf("targetBE.Site = %q, want dallocForClass", targetBE.Site)
	}

	// Check helper predicates
	if !be.IsAllocation() {
		t.Fatal("be.IsAllocation() = false, want true")
	}
	if be.IsDeviceLost() {
		t.Fatal("be.IsDeviceLost() = true, want false")
	}
	if be.IsSubmissionFailed() {
		t.Fatal("be.IsSubmissionFailed() = true, want false")
	}
	if be.IsInvalidGeometry() {
		t.Fatal("be.IsInvalidGeometry() = true, want false")
	}
	if be.IsResourceExhausted() {
		t.Fatal("be.IsResourceExhausted() = true, want false")
	}
	if !be.IsRecoverable() {
		t.Fatal("be.IsRecoverable() = false, want true")
	}

	// DeviceLost should report IsRecoverable() == false
	lostErr := &BackendError{
		Backend: "vulkan",
		Class:   VulkanClassDeviceLost,
		Err:     ErrVulkanDeviceLost,
	}
	if lostErr.IsRecoverable() {
		t.Fatal("lostErr.IsRecoverable() = true, want false for DeviceLost")
	}
	if !lostErr.IsDeviceLost() {
		t.Fatal("lostErr.IsDeviceLost() = false, want true")
	}

	// Nil receiver safety
	var nilBE *BackendError
	if nilBE.Error() != "compute: nil backend error" {
		t.Fatalf("nilBE.Error() = %q", nilBE.Error())
	}
	if nilBE.Unwrap() != nil {
		t.Fatal("nilBE.Unwrap() != nil")
	}
	if nilBE.Is(ErrVulkanAllocationFailed) {
		t.Fatal("nilBE.Is(...) = true, want false")
	}
	if nilBE.IsAllocation() {
		t.Fatal("nilBE.IsAllocation() = true, want false")
	}
	if !nilBE.IsRecoverable() {
		t.Fatal("nilBE.IsRecoverable() = false, want true")
	}
}

func TestVulkanErrorsCheckedInventory(t *testing.T) {
	inv := DefaultVulkanErrorInventory()
	if err := inv.Validate(); err != nil {
		t.Fatalf("DefaultVulkanErrorInventory().Validate() error = %v", err)
	}

	// Corrupted inventory validations
	t.Run("EmptyRules", func(t *testing.T) {
		emptyInv := VulkanErrorInventory{}
		if err := emptyInv.Validate(); err == nil {
			t.Fatal("expected error for empty inventory, got nil")
		}
	})

	t.Run("EmptyDomain", func(t *testing.T) {
		bad := inv
		bad.Rules = append([]VulkanErrorRule(nil), inv.Rules...)
		bad.Rules[0].Domain = ""
		if err := bad.Validate(); err == nil || !strings.Contains(err.Error(), "empty domain") {
			t.Fatalf("Validate() = %v, want empty domain error", err)
		}
	})

	t.Run("NilTypedError", func(t *testing.T) {
		bad := inv
		bad.Rules = append([]VulkanErrorRule(nil), inv.Rules...)
		bad.Rules[0].TypedError = nil
		if err := bad.Validate(); err == nil || !strings.Contains(err.Error(), "nil typed error") {
			t.Fatalf("Validate() = %v, want nil typed error", err)
		}
	})

	t.Run("DuplicateKeywordsAcrossRules", func(t *testing.T) {
		bad := inv
		bad.Rules = append([]VulkanErrorRule(nil), inv.Rules...)
		bad.Rules[1].Keywords = append(bad.Rules[1].Keywords, bad.Rules[0].Keywords[0])
		if err := bad.Validate(); err == nil || !strings.Contains(err.Error(), "duplicated") {
			t.Fatalf("Validate() = %v, want duplicate keyword error", err)
		}
	})

	t.Run("MissingRequiredDomain", func(t *testing.T) {
		bad := VulkanErrorInventory{
			Rules: []VulkanErrorRule{
				{
					Domain:      "device",
					Class:       VulkanClassDeviceLost,
					TypedError:  ErrVulkanDeviceLost,
					Keywords:    []string{"device lost"},
					Description: "device lost",
				},
			},
		}
		if err := bad.Validate(); err == nil || !strings.Contains(err.Error(), "missing required domain") {
			t.Fatalf("Validate() = %v, want missing required domain error", err)
		}
	})
}

func TestVulkanErrorsStaticPanicSitesInventory(t *testing.T) {
	tests := []struct {
		name          string
		recovered     any
		wantClass     VulkanErrorClass
		wantSentinel  error
		wantPredicate func(*BackendError) bool
	}{
		// 1. Capacity & single-resource cap
		{
			name:         "SingleResourceCapExceeded",
			recovered:    "compute: vulkan storage buffer allocation of 5000000000 bytes exceeds device single-resource cap 2147483648 bytes (maxStorageBufferRange=2147483648, maxMemoryAllocationSize=2147483648)",
			wantClass:    VulkanClassResourceExhausted,
			wantSentinel: ErrVulkanResourceExhausted,
			wantPredicate: func(e *BackendError) bool {
				return e.IsResourceExhausted()
			},
		},
		{
			name:         "Q8RowCapExceeded",
			recovered:    "compute: vulkan Q8_0 weight row [1, 4096] allocation of 2000000000 bytes exceeds device single-resource cap 1000000000 bytes",
			wantClass:    VulkanClassResourceExhausted,
			wantSentinel: ErrVulkanResourceExhausted,
			wantPredicate: func(e *BackendError) bool {
				return e.IsResourceExhausted()
			},
		},
		{
			name:         "BudgetExceeded",
			recovered:    "compute: vulkan budget exceeded for weight storage",
			wantClass:    VulkanClassResourceExhausted,
			wantSentinel: ErrVulkanResourceExhausted,
			wantPredicate: func(e *BackendError) bool {
				return e.IsResourceExhausted()
			},
		},

		// 2. Allocation failures
		{
			name:         "DeviceAllocErrorStruct",
			recovered:    &DeviceAllocError{Bytes: 1048576, Site: "vulkan:storage buffer", Class: MemoryWeights},
			wantClass:    VulkanClassAllocationFailed,
			wantSentinel: ErrVulkanAllocationFailed,
			wantPredicate: func(e *BackendError) bool {
				var dae *DeviceAllocError
				return e.IsAllocation() && errors.As(e, &dae) && dae.Bytes == 1048576
			},
		},
		{
			name:         "HostVisibleAllocFailed",
			recovered:    &DeviceAllocError{Bytes: 2097152, Site: "vulkan:host-visible storage buffer", Class: MemoryOffload},
			wantClass:    VulkanClassAllocationFailed,
			wantSentinel: ErrVulkanAllocationFailed,
			wantPredicate: func(e *BackendError) bool {
				return e.IsAllocation()
			},
		},
		{
			name:         "FvkMallocFailed",
			recovered:    "compute: vulkan fvk_malloc returned nil for allocation of 65536 bytes",
			wantClass:    VulkanClassAllocationFailed,
			wantSentinel: ErrVulkanAllocationFailed,
			wantPredicate: func(e *BackendError) bool {
				return e.IsAllocation()
			},
		},
		{
			name:         "OutOfDeviceMemory",
			recovered:    "VK_ERROR_OUT_OF_DEVICE_MEMORY: failed to allocate buffer memory",
			wantClass:    VulkanClassAllocationFailed,
			wantSentinel: ErrVulkanAllocationFailed,
			wantPredicate: func(e *BackendError) bool {
				return e.IsAllocation()
			},
		},
		{
			name:         "OutOfHostMemory",
			recovered:    "VK_ERROR_OUT_OF_HOST_MEMORY: host staging buffer allocation failed",
			wantClass:    VulkanClassAllocationFailed,
			wantSentinel: ErrVulkanAllocationFailed,
			wantPredicate: func(e *BackendError) bool {
				return e.IsAllocation()
			},
		},

		// 3. Device Lost
		{
			name:         "DeviceLostVK",
			recovered:    "VK_ERROR_DEVICE_LOST: logical device lost during execution",
			wantClass:    VulkanClassDeviceLost,
			wantSentinel: ErrVulkanDeviceLost,
			wantPredicate: func(e *BackendError) bool {
				return e.IsDeviceLost() && !e.IsRecoverable()
			},
		},
		{
			name:         "DeviceReset",
			recovered:    "vulkan device reset detected by watchdog",
			wantClass:    VulkanClassDeviceLost,
			wantSentinel: ErrVulkanDeviceLost,
			wantPredicate: func(e *BackendError) bool {
				return e.IsDeviceLost()
			},
		},
		{
			name:         "DeviceFaultErrorStruct",
			recovered:    &DeviceFaultError{Backend: "vulkan", Site: "queue-submit", Class: DeviceFaultContext},
			wantClass:    VulkanClassDeviceLost,
			wantSentinel: ErrVulkanDeviceLost,
			wantPredicate: func(e *BackendError) bool {
				return e.IsDeviceLost()
			},
		},

		// 4. Submission & Synchronization failures
		{
			name:         "QueueSubmitFailed",
			recovered:    "compute: vulkan queue submit failed with error",
			wantClass:    VulkanClassSubmissionFailed,
			wantSentinel: ErrVulkanSubmissionFailed,
			wantPredicate: func(e *BackendError) bool {
				return e.IsSubmissionFailed()
			},
		},
		{
			name:         "FenceWaitTimeout",
			recovered:    "compute: vulkan fence wait timeout waiting for command buffer",
			wantClass:    VulkanClassSubmissionFailed,
			wantSentinel: ErrVulkanSubmissionFailed,
			wantPredicate: func(e *BackendError) bool {
				return e.IsSubmissionFailed()
			},
		},
		{
			name:         "BatchFlushFailure",
			recovered:    "compute: vulkan fvk_batch flush failed",
			wantClass:    VulkanClassSubmissionFailed,
			wantSentinel: ErrVulkanSubmissionFailed,
			wantPredicate: func(e *BackendError) bool {
				return e.IsSubmissionFailed()
			},
		},

		// 5. Invalid Geometry & Incompatible Operands
		{
			name:         "UploadExpectsHostData",
			recovered:    "compute: vulkan Upload expects host data",
			wantClass:    VulkanClassInvalidGeometry,
			wantSentinel: ErrVulkanInvalidGeometry,
			wantPredicate: func(e *BackendError) bool {
				return e.IsInvalidGeometry()
			},
		},
		{
			name:         "UploadMissingQuantSpec",
			recovered:    "compute: vulkan Upload Q8 tensor missing QuantSpec",
			wantClass:    VulkanClassInvalidGeometry,
			wantSentinel: ErrVulkanInvalidGeometry,
			wantPredicate: func(e *BackendError) bool {
				return e.IsInvalidGeometry()
			},
		},
		{
			name:         "UploadUnsupportedDtype",
			recovered:    "compute: vulkan Upload supports only F32 today (got f16)",
			wantClass:    VulkanClassInvalidGeometry,
			wantSentinel: ErrVulkanInvalidGeometry,
			wantPredicate: func(e *BackendError) bool {
				return e.IsInvalidGeometry()
			},
		},
		{
			name:         "Q4KByteLengthMismatch",
			recovered:    "compute: vulkan Q4_K raw byte length does not match shape",
			wantClass:    VulkanClassInvalidGeometry,
			wantSentinel: ErrVulkanInvalidGeometry,
			wantPredicate: func(e *BackendError) bool {
				return e.IsInvalidGeometry()
			},
		},
		{
			name:         "Q8Expects2D",
			recovered:    "compute: vulkan Q8 upload expects a 2D weight tensor",
			wantClass:    VulkanClassInvalidGeometry,
			wantSentinel: ErrVulkanInvalidGeometry,
			wantPredicate: func(e *BackendError) bool {
				return e.IsInvalidGeometry()
			},
		},
		{
			name:         "Q8DivisibleBlock",
			recovered:    "compute: vulkan Q8 upload supports only Q8_0 block=32 with divisible input dim",
			wantClass:    VulkanClassInvalidGeometry,
			wantSentinel: ErrVulkanInvalidGeometry,
			wantPredicate: func(e *BackendError) bool {
				return e.IsInvalidGeometry()
			},
		},
		{
			name:         "MatMulUnsupportedDtype",
			recovered:    "compute: vulkan MatMul unsupported weight dtype bf16",
			wantClass:    VulkanClassInvalidGeometry,
			wantSentinel: ErrVulkanInvalidGeometry,
			wantPredicate: func(e *BackendError) bool {
				return e.IsInvalidGeometry()
			},
		},
		{
			name:         "MatMulMissingQ8Chunks",
			recovered:    "compute: vulkan MatMul missing Q8 chunk device buffers",
			wantClass:    VulkanClassInvalidGeometry,
			wantSentinel: ErrVulkanInvalidGeometry,
			wantPredicate: func(e *BackendError) bool {
				return e.IsInvalidGeometry()
			},
		},
		{
			name:         "MatMulMissingScaleBuffer",
			recovered:    "compute: vulkan MatMul missing device scale buffer",
			wantClass:    VulkanClassInvalidGeometry,
			wantSentinel: ErrVulkanInvalidGeometry,
			wantPredicate: func(e *BackendError) bool {
				return e.IsInvalidGeometry()
			},
		},
		{
			name:         "MatMulArgmaxRowMismatch",
			recovered:    "compute: vulkan MatMulArgmax expects one input row matching the weight input dim",
			wantClass:    VulkanClassInvalidGeometry,
			wantSentinel: ErrVulkanInvalidGeometry,
			wantPredicate: func(e *BackendError) bool {
				return e.IsInvalidGeometry()
			},
		},
		{
			name:         "RMSNormMatMulArgmaxShapeMismatch",
			recovered:    "compute: vulkan RMSNormMatMulArgmax norm weight shape does not match projection input dim",
			wantClass:    VulkanClassInvalidGeometry,
			wantSentinel: ErrVulkanInvalidGeometry,
			wantPredicate: func(e *BackendError) bool {
				return e.IsInvalidGeometry()
			},
		},
		{
			name:         "EmbeddingRowOutOfRange",
			recovered:    "compute: vulkan EmbeddingRow row out of range",
			wantClass:    VulkanClassInvalidGeometry,
			wantSentinel: ErrVulkanInvalidGeometry,
			wantPredicate: func(e *BackendError) bool {
				return e.IsInvalidGeometry()
			},
		},
		{
			name:         "MatMulAddInPlaceShapeMismatch",
			recovered:    "compute: vulkan MatMulAddInPlace dst shape does not match projection output",
			wantClass:    VulkanClassInvalidGeometry,
			wantSentinel: ErrVulkanInvalidGeometry,
			wantPredicate: func(e *BackendError) bool {
				return e.IsInvalidGeometry()
			},
		},
		{
			name:         "MatMul2InputDimsDiffer",
			recovered:    "compute: vulkan MatMul2 weight input dims differ",
			wantClass:    VulkanClassInvalidGeometry,
			wantSentinel: ErrVulkanInvalidGeometry,
			wantPredicate: func(e *BackendError) bool {
				return e.IsInvalidGeometry()
			},
		},
		{
			name:         "MatMul3DecodeOnly",
			recovered:    "compute: vulkan MatMul3 is decode-only today",
			wantClass:    VulkanClassInvalidGeometry,
			wantSentinel: ErrVulkanInvalidGeometry,
			wantPredicate: func(e *BackendError) bool {
				return e.IsInvalidGeometry()
			},
		},
		{
			name:         "SwiGLUShapesDiffer",
			recovered:    "compute: vulkan SwiGLUMatMulAddInPlace gate/up shapes differ",
			wantClass:    VulkanClassInvalidGeometry,
			wantSentinel: ErrVulkanInvalidGeometry,
			wantPredicate: func(e *BackendError) bool {
				return e.IsInvalidGeometry()
			},
		},
		{
			name:         "AppendKVRoPEShapeMismatch",
			recovered:    "compute: vulkan AppendKVRoPE shape does not match KV config",
			wantClass:    VulkanClassInvalidGeometry,
			wantSentinel: ErrVulkanInvalidGeometry,
			wantPredicate: func(e *BackendError) bool {
				return e.IsInvalidGeometry()
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			be := ClassifyVulkanPanic(tt.recovered, "testSite")
			if be == nil {
				t.Fatalf("ClassifyVulkanPanic(%v) = nil, want *BackendError", tt.recovered)
			}
			if be.Class != tt.wantClass {
				t.Errorf("be.Class = %v, want %v", be.Class, tt.wantClass)
			}
			if !errors.Is(be, tt.wantSentinel) {
				t.Errorf("errors.Is(be, %v) = false, want true", tt.wantSentinel)
			}
			if tt.wantPredicate != nil && !tt.wantPredicate(be) {
				t.Errorf("wantPredicate failed for %v", be)
			}
		})
	}
}

func TestVulkanErrorsSimulatedRequestTimePanicRecovery(t *testing.T) {
	// 1. SafeVulkanOp
	t.Run("SafeVulkanOp", func(t *testing.T) {
		err := SafeVulkanOp("addInPlace", func() {
			panic("compute: vulkan MatMul unsupported weight dtype i4")
		})
		if err == nil {
			t.Fatal("SafeVulkanOp did not return error for panic")
		}
		if !errors.Is(err, ErrVulkanInvalidGeometry) {
			t.Fatalf("SafeVulkanOp err = %v, want ErrVulkanInvalidGeometry", err)
		}

		// Normal success should return nil
		normalErr := SafeVulkanOp("addInPlace", func() {})
		if normalErr != nil {
			t.Fatalf("SafeVulkanOp normal returned error = %v, want nil", normalErr)
		}
	})

	// 2. SafeVulkanRun
	t.Run("SafeVulkanRun", func(t *testing.T) {
		err := SafeVulkanRun("uploadWeights", func() error {
			panic(&DeviceAllocError{Bytes: 4096, Site: "vulkan:upload", Class: MemoryWeights})
		})
		if err == nil {
			t.Fatal("SafeVulkanRun did not return error for panic")
		}
		if !errors.Is(err, ErrVulkanAllocationFailed) {
			t.Fatalf("SafeVulkanRun err = %v, want ErrVulkanAllocationFailed", err)
		}

		// Non-panic error should be preserved
		customErr := errors.New("custom error")
		returnedErr := SafeVulkanRun("uploadWeights", func() error {
			return customErr
		})
		if !errors.Is(returnedErr, customErr) {
			t.Fatalf("SafeVulkanRun returnedErr = %v, want customErr", returnedErr)
		}
	})

	// 3. SafeVulkanCall
	t.Run("SafeVulkanCall", func(t *testing.T) {
		val, err := SafeVulkanCall[int]("matmul", func() (int, error) {
			panic("compute: vulkan storage buffer allocation of 999999999 bytes exceeds device single-resource cap 500000000 bytes")
		})
		if err == nil {
			t.Fatal("SafeVulkanCall did not return error for panic")
		}
		if val != 0 {
			t.Fatalf("SafeVulkanCall val = %d, want 0", val)
		}
		if !errors.Is(err, ErrVulkanResourceExhausted) {
			t.Fatalf("SafeVulkanCall err = %v, want ErrVulkanResourceExhausted", err)
		}

		// Normal call
		successVal, successErr := SafeVulkanCall[int]("matmul", func() (int, error) {
			return 42, nil
		})
		if successErr != nil || successVal != 42 {
			t.Fatalf("SafeVulkanCall successVal=%d, err=%v, want 42, nil", successVal, successErr)
		}
	})

	// 4. SafeVulkanValue
	t.Run("SafeVulkanValue", func(t *testing.T) {
		val, err := SafeVulkanValue[string]("argmax", func() string {
			panic("compute: vulkan fence wait timeout waiting for command buffer")
		})
		if err == nil {
			t.Fatal("SafeVulkanValue did not return error for panic")
		}
		if val != "" {
			t.Fatalf("SafeVulkanValue val = %q, want empty", val)
		}
		if !errors.Is(err, ErrVulkanSubmissionFailed) {
			t.Fatalf("SafeVulkanValue err = %v, want ErrVulkanSubmissionFailed", err)
		}
	})

	// 5. CatchVulkanPanicHandler
	t.Run("CatchVulkanPanicHandler", func(t *testing.T) {
		var handled *BackendError
		func() {
			defer CatchVulkanPanicHandler("handlerSite", func(be *BackendError) {
				handled = be
			})
			panic("VK_ERROR_DEVICE_LOST: driver timeout")
		}()
		if handled == nil {
			t.Fatal("CatchVulkanPanicHandler was not called")
		}
		if !handled.IsDeviceLost() {
			t.Fatalf("handled.IsDeviceLost() = false, want true for %v", handled)
		}
		if handled.Site != "handlerSite" {
			t.Fatalf("handled.Site = %q, want handlerSite", handled.Site)
		}
	})
}

// mockVulkanBackend simulates a stateful Vulkan backend implementing Backend and BackendRequestRetirer.
type mockVulkanBackend struct {
	mu           sync.Mutex
	retireCalls  int
	teardownCall int
}

func (m *mockVulkanBackend) Name() string            { return "vulkan" }
func (m *mockVulkanBackend) Tier() string            { return "integrated:mock" }
func (m *mockVulkanBackend) Class() CorrectnessClass { return Approx }
func (m *mockVulkanBackend) Caps() Caps              { return Caps{DeviceMemory: true} }

func (m *mockVulkanBackend) Upload(t Tensor, as Dtype) Tensor { return t }
func (m *mockVulkanBackend) Host(t Tensor) ([]float32, bool)  { return nil, false }
func (m *mockVulkanBackend) Read(t Tensor) []float32          { return nil }
func (m *mockVulkanBackend) Free(t Tensor)                    {}
func (m *mockVulkanBackend) NewKV(cfg KVConfig) KVStore       { return nil }

func (m *mockVulkanBackend) MatMul(w, x Tensor) Tensor                                     { return w }
func (m *mockVulkanBackend) BatchedMatMul(w, X Tensor, P int) Tensor                       { return w }
func (m *mockVulkanBackend) RMSNorm(x, weight Tensor, eps float32) Tensor                  { return x }
func (m *mockVulkanBackend) RoPE(x Tensor, pos, nHeads, headDim int, theta float64) Tensor { return x }
func (m *mockVulkanBackend) SwiGLU(gate, up Tensor) Tensor                                 { return gate }
func (m *mockVulkanBackend) AddInPlace(dst, src Tensor)                                    {}
func (m *mockVulkanBackend) AddBias(dst, bias Tensor)                                      {}
func (m *mockVulkanBackend) Attention(q Tensor, kv KVStore, layer int, causal bool, grp int, scale float32) Tensor {
	return q
}
func (m *mockVulkanBackend) Argmax(logits Tensor) int { return 0 }

func (m *mockVulkanBackend) RetireRequestResources() {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.retireCalls++
}

func (m *mockVulkanBackend) TeardownResources() error {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.teardownCall++
	return nil
}

func TestVulkanErrorsSubsequentOperationsSucceedAfterRecovery(t *testing.T) {
	backend := &mockVulkanBackend{}

	// Sequence of 5 requests:
	// Request 1: Normal success
	res1, err1 := ExecuteVulkanRequest(backend, "req1", func() (int, error) {
		return 100, nil
	})
	if err1 != nil || res1 != 100 {
		t.Fatalf("Request 1 failed: res=%d, err=%v", res1, err1)
	}

	// Request 2: Panics with single-resource cap exceeded
	res2, err2 := ExecuteVulkanRequest(backend, "req2", func() (int, error) {
		panic("compute: vulkan storage buffer allocation of 8000000000 bytes exceeds device single-resource cap 2000000000 bytes")
	})
	if err2 == nil {
		t.Fatal("Request 2 succeeded, want panic recovery error")
	}
	if res2 != 0 {
		t.Fatalf("Request 2 res = %d, want 0", res2)
	}
	if !errors.Is(err2, ErrVulkanResourceExhausted) {
		t.Fatalf("Request 2 err = %v, want ErrVulkanResourceExhausted", err2)
	}

	// Request 3: Subsequent normal request MUST SUCCEED cleanly
	res3, err3 := ExecuteVulkanRequest(backend, "req3", func() (int, error) {
		return 300, nil
	})
	if err3 != nil || res3 != 300 {
		t.Fatalf("Request 3 failed after Request 2 recovery: res=%d, err=%v", res3, err3)
	}

	// Request 4: Panics with allocation failure (*DeviceAllocError)
	res4, err4 := ExecuteVulkanRequest(backend, "req4", func() (int, error) {
		panic(&DeviceAllocError{Bytes: 1048576, Site: "vulkan:dalloc", Class: MemoryWeights})
	})
	if err4 == nil {
		t.Fatal("Request 4 succeeded, want panic recovery error")
	}
	if res4 != 0 {
		t.Fatalf("Request 4 res = %d, want 0", res4)
	}
	if !errors.Is(err4, ErrVulkanAllocationFailed) {
		t.Fatalf("Request 4 err = %v, want ErrVulkanAllocationFailed", err4)
	}

	// Request 5: Subsequent normal request MUST SUCCEED cleanly
	res5, err5 := ExecuteVulkanRequest(backend, "req5", func() (int, error) {
		return 500, nil
	})
	if err5 != nil || res5 != 500 {
		t.Fatalf("Request 5 failed after Request 4 recovery: res=%d, err=%v", res5, err5)
	}

	// Every request (both successful and failed) must have had its RequestLifetime retired
	backend.mu.Lock()
	retires := backend.retireCalls
	backend.mu.Unlock()

	if retires != 5 {
		t.Fatalf("total retire calls = %d, want 5 (one per request)", retires)
	}
}

func TestVulkanErrorsSafetyAndEdgeCases(t *testing.T) {
	t.Run("NilPanicRecovery", func(t *testing.T) {
		be := ClassifyVulkanPanic(nil, "site")
		if be != nil {
			t.Fatalf("ClassifyVulkanPanic(nil) = %v, want nil", be)
		}
	})

	t.Run("ArbitraryTypePanic", func(t *testing.T) {
		be := ClassifyVulkanPanic(12345, "site")
		if be == nil {
			t.Fatal("ClassifyVulkanPanic(12345) = nil")
		}
		if be.Class != VulkanClassUnknown {
			t.Fatalf("be.Class = %v, want VulkanClassUnknown", be.Class)
		}
		if !errors.Is(be, ErrVulkanExecutionFailed) {
			t.Fatalf("be.Err = %v, want ErrVulkanExecutionFailed", be.Err)
		}
	})

	t.Run("PreconstructedBackendErrorPreserved", func(t *testing.T) {
		original := &BackendError{
			Backend: "vulkan",
			Class:   VulkanClassDeviceLost,
			Site:    "existingSite",
			Err:     ErrVulkanDeviceLost,
		}
		be := ClassifyVulkanPanic(original, "newSite")
		if be != original {
			t.Fatalf("be = %v, want identical pointer %v", be, original)
		}
		if be.Site != "existingSite" {
			t.Fatalf("be.Site = %q, want existingSite", be.Site)
		}
	})

	t.Run("ConcurrentPanicRecovery", func(t *testing.T) {
		const goroutines = 50
		var wg sync.WaitGroup
		wg.Add(goroutines)

		backend := &mockVulkanBackend{}

		for i := 0; i < goroutines; i++ {
			go func(id int) {
				defer wg.Done()
				switch id % 3 {
				case 0:
					res, err := ExecuteVulkanRequest(backend, "concurrentSuccess", func() (int, error) {
						return id, nil
					})
					if err != nil || res != id {
						t.Errorf("concurrent success failed: %v", err)
					}
				case 1:
					_, err := ExecuteVulkanRequest(backend, "concurrentAllocPanic", func() (int, error) {
						panic(&DeviceAllocError{Bytes: id * 1024, Site: "vulkan:test", Class: MemoryWeights})
					})
					if !errors.Is(err, ErrVulkanAllocationFailed) {
						t.Errorf("concurrent alloc failed to recover properly: %v", err)
					}
				case 2:
					_, err := ExecuteVulkanRequest(backend, "concurrentGeometryPanic", func() (int, error) {
						panic("compute: vulkan MatMul unsupported weight dtype f16")
					})
					if !errors.Is(err, ErrVulkanInvalidGeometry) {
						t.Errorf("concurrent geometry failed to recover properly: %v", err)
					}
				}
			}(i)
		}

		wg.Wait()
	})

	t.Run("WrappedErrorClassification", func(t *testing.T) {
		// Wrapped *DeviceAllocError
		dae := &DeviceAllocError{Bytes: 8192, Site: "vulkan:wrappedAlloc", Class: MemoryActivation}
		wrappedDAE := fmt.Errorf("outer failure: %w", dae)
		beDAE := ClassifyVulkanPanic(wrappedDAE, "site")
		if beDAE == nil || !beDAE.IsAllocation() {
			t.Fatalf("expected allocation error from wrappedDAE, got %v", beDAE)
		}
		var targetDAE *DeviceAllocError
		if !errors.As(beDAE, &targetDAE) || targetDAE.Bytes != 8192 {
			t.Fatalf("errors.As(*DeviceAllocError) failed on beDAE: %v", targetDAE)
		}

		// Wrapped *DeviceFaultError
		dfe := &DeviceFaultError{Backend: "vulkan", Site: "vulkan:wrappedFault", Class: DeviceFaultContext}
		wrappedDFE := fmt.Errorf("outer failure: %w", dfe)
		beDFE := ClassifyVulkanPanic(wrappedDFE, "site")
		if beDFE == nil || !beDFE.IsDeviceLost() {
			t.Fatalf("expected device lost from wrappedDFE, got %v", beDFE)
		}

		// Wrapped Sentinel Error
		wrappedSentinel := fmt.Errorf("outer failure: %w", ErrVulkanInvalidGeometry)
		beSent := ClassifyVulkanPanic(wrappedSentinel, "site")
		if beSent == nil || !beSent.IsInvalidGeometry() {
			t.Fatalf("expected invalid geometry from wrappedSentinel, got %v", beSent)
		}

		// ClassifyVulkanError helper
		if ClassifyVulkanError(nil, "site") != nil {
			t.Fatal("ClassifyVulkanError(nil) != nil")
		}
		beHelper := ClassifyVulkanError(wrappedDAE, "helperSite")
		if beHelper == nil || !beHelper.IsAllocation() {
			t.Fatalf("expected allocation error from ClassifyVulkanError, got %v", beHelper)
		}
	})
}
