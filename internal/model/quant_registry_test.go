package model

import (
	"errors"
	"sync"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

func TestQuantRegistryBuiltinRegistration(t *testing.T) {
	ResetDefaultQuantDescriptors()

	tests := []struct {
		kind         kQuantKind
		name         string
		dtype        compute.Dtype
		supportsHAL  bool
		blockBytes   int
		blockWeights int
	}{
		{kindQ5K, "Q5_K", compute.Q5_K, true, q5kBlockBytes, qkK},
		{kindQ6K, "Q6_K", compute.Q6_K, true, q6kBlockBytes, qkK},
		{kindQ2K, "Q2_K", compute.Q2_K, true, q2kBlockBytes, qkK},
		{kindQ8_0, "Q8_0", compute.Q8_0, false, q8_0BlockBytes, q8_0BlockWeights},
		{kindQ4_0, "Q4_0", 0, false, q4_0BlockBytes, q4_0BlockWeights},
		{kindIQ3XXS, "IQ3_XXS", compute.IQ3_XXS, false, iq3xxsBlockBytes, qkK},
		{kindIQ4XS, "IQ4_XS", 0, false, iq4xsBlockBytes, qkK},
		{kindIQ2XXS, "IQ2_XXS", 0, false, iq2xxsBlockBytes, qkK},
		{kindIQ2XS, "IQ2_XS", 0, false, iq2xsBlockBytes, qkK},
		{kindIQ1S, "IQ1_S", 0, false, iq1sBlockBytes, qkK},
		{kindIQ2S, "IQ2_S", 0, false, iq2sBlockBytes, qkK},
		{kindIQ1M, "IQ1_M", 0, false, iq1mBlockBytes, qkK},
		{kindQ3K, "Q3_K", 0, false, q3kBlockBytes, qkK},
		{kindIQ3S, "IQ3_S", compute.IQ3_S, false, iq3sBlockBytes, qkK},
	}

	for _, tc := range tests {
		desc, ok := LookupQuantDescriptor(tc.kind)
		if !ok {
			t.Errorf("expected descriptor for %s (%d) to be registered", tc.name, tc.kind)
			continue
		}
		if desc.Kind() != tc.kind {
			t.Errorf("%s: got Kind %d, want %d", tc.name, desc.Kind(), tc.kind)
		}
		if desc.Name() != tc.name {
			t.Errorf("%s: got Name %q, want %q", tc.name, desc.Name(), tc.name)
		}
		if desc.Dtype() != tc.dtype {
			t.Errorf("%s: got Dtype %v, want %v", tc.name, desc.Dtype(), tc.dtype)
		}
		if desc.SupportsHAL() != tc.supportsHAL {
			t.Errorf("%s: got SupportsHAL %v, want %v", tc.name, desc.SupportsHAL(), tc.supportsHAL)
		}
		if desc.BlockBytes() != tc.blockBytes {
			t.Errorf("%s: got BlockBytes %d, want %d", tc.name, desc.BlockBytes(), tc.blockBytes)
		}
		if desc.BlockWeights() != tc.blockWeights {
			t.Errorf("%s: got BlockWeights %d, want %d", tc.name, desc.BlockWeights(), tc.blockWeights)
		}
		if desc.KeyPrefix() != "kquant-raw:" {
			t.Errorf("%s: got KeyPrefix %q, want 'kquant-raw:'", tc.name, desc.KeyPrefix())
		}
		if SupportsHALKQuant(tc.kind) != tc.supportsHAL {
			t.Errorf("%s: SupportsHALKQuant got %v, want %v", tc.name, SupportsHALKQuant(tc.kind), tc.supportsHAL)
		}
	}
}

func TestQuantRegistryLookupByName(t *testing.T) {
	ResetDefaultQuantDescriptors()

	for _, name := range []string{"Q5_K", "Q6_K", "Q2_K", "Q8_0", "Q4_0", "IQ3_XXS", "IQ3_S"} {
		desc, ok := LookupQuantDescriptorByName(name)
		if !ok || desc == nil {
			t.Fatalf("expected descriptor lookup by name for %q to succeed", name)
		}
		if desc.Name() != name {
			t.Fatalf("got Name %q, want %q", desc.Name(), name)
		}
	}

	if _, ok := LookupQuantDescriptorByName("UNKNOWN_QUANT"); ok {
		t.Fatal("expected LookupQuantDescriptorByName to return false for unknown name")
	}
}

func TestQuantRegistryHostTensorBuilders(t *testing.T) {
	ResetDefaultQuantDescriptors()

	// Q5_K host tensor construction
	descQ5, ok := LookupQuantDescriptor(kindQ5K)
	if !ok {
		t.Fatal("Q5_K not found")
	}
	rawQ5 := make([]byte, 2*q5kBlockBytes)
	tensorQ5 := descQ5.NewHostTensor(2, 256, rawQ5)
	if tensorQ5.Dtype != compute.Q5_K {
		t.Fatalf("expected Q5_K tensor, got %v", tensorQ5.Dtype)
	}

	// Q6_K host tensor construction
	descQ6, ok := LookupQuantDescriptor(kindQ6K)
	if !ok {
		t.Fatal("Q6_K not found")
	}
	rawQ6 := make([]byte, 2*q6kBlockBytes)
	tensorQ6 := descQ6.NewHostTensor(2, 256, rawQ6)
	if tensorQ6.Dtype != compute.Q6_K {
		t.Fatalf("expected Q6_K tensor, got %v", tensorQ6.Dtype)
	}

	// Q2_K host tensor construction
	descQ2, ok := LookupQuantDescriptor(kindQ2K)
	if !ok {
		t.Fatal("Q2_K not found")
	}
	rawQ2 := make([]byte, 2*q2kBlockBytes)
	tensorQ2 := descQ2.NewHostTensor(2, 256, rawQ2)
	if tensorQ2.Dtype != compute.Q2_K {
		t.Fatalf("expected Q2_K tensor, got %v", tensorQ2.Dtype)
	}

	// Unsupported host tensor construction panics cleanly
	descQ8, ok := LookupQuantDescriptor(kindQ8_0)
	if !ok {
		t.Fatal("Q8_0 not found")
	}
	defer func() {
		r := recover()
		if r == nil {
			t.Fatal("expected panic when constructing host tensor for unsupported descriptor")
		}
	}()
	descQ8.NewHostTensor(1, 32, make([]byte, 34))
}

func TestQuantRegistryCustomRegistrationAndConcurrency(t *testing.T) {
	ResetDefaultQuantDescriptors()
	defer ResetDefaultQuantDescriptors()

	const customKind kQuantKind = 250
	customDesc := BaseQuantDescriptor{
		QuantKind:     customKind,
		QuantName:     "CUSTOM_Q",
		ComputeDtype:  compute.F32,
		Prefix:        "custom:",
		HALSupported:  true,
		BytesPerBlk:   128,
		WeightsPerBlk: 256,
		HostTensorFn: func(out, in int, raw []byte) compute.Tensor {
			return compute.NewF32(compute.Default(), []int{out, in}, make([]float32, out*in))
		},
	}

	RegisterQuantDescriptor(customDesc)

	desc, ok := LookupQuantDescriptor(customKind)
	if !ok || desc == nil {
		t.Fatal("custom descriptor lookup failed")
	}
	if desc.Name() != "CUSTOM_Q" || !desc.SupportsHAL() || desc.KeyPrefix() != "custom:" {
		t.Fatalf("custom descriptor attributes mismatch: %+v", desc)
	}

	// Concurrent lookups and registrations must not race
	var wg sync.WaitGroup
	for i := 0; i < 30; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			_, _ = LookupQuantDescriptor(kindQ5K)
			_, _ = LookupQuantDescriptor(customKind)
			_, _ = LookupQuantDescriptorByName("Q6_K")
			_ = SupportsHALKQuant(kindQ2K)
			_ = RegisteredQuantDescriptors()
		}(i)
	}
	wg.Wait()
}

type quantUploadTestBackend struct {
	compute.Backend
	uploads []compute.Dtype
}

func (r *quantUploadTestBackend) Caps() compute.Caps {
	c := r.Backend.Caps()
	c.UploadDtype = true
	return c
}

func (r *quantUploadTestBackend) Upload(t compute.Tensor, as compute.Dtype) compute.Tensor {
	r.uploads = append(r.uploads, as)
	return r.Backend.Upload(t, as)
}

func TestWeightHALKQuantRegistryDispatch(t *testing.T) {
	ResetDefaultQuantDescriptors()
	defer ResetDefaultQuantDescriptors()

	rec := &quantUploadTestBackend{Backend: compute.Default()}
	s := &Session{
		Backend: rec,
		halW:    map[string]compute.Tensor{},
	}

	const out, in = 4, 256
	rawQ5 := make([]byte, out*(in/256)*q5kBlockBytes)
	qtQ5 := &kQuantTensor{out: out, in: in, nblk: in / qkK, kind: kindQ5K, raw: rawQ5}

	tensorQ5 := s.weightHALKQuant("w_q5", qtQ5)
	if tensorQ5.Dtype != compute.Q5_K {
		t.Fatalf("expected Q5_K, got %v", tensorQ5.Dtype)
	}
	if _, ok := s.halW["kquant-raw:w_q5"]; !ok {
		t.Fatalf("expected cached key 'kquant-raw:w_q5' in halW")
	}

	// Repeated call must be cached
	_ = s.weightHALKQuant("w_q5", qtQ5)
	if len(rec.uploads) != 1 {
		t.Fatalf("expected 1 upload due to cache hit, got %d", len(rec.uploads))
	}
}

func TestWeightHALKQuantRegistryRefusal(t *testing.T) {
	ResetDefaultQuantDescriptors()
	defer ResetDefaultQuantDescriptors()

	rec := &quantUploadTestBackend{Backend: compute.Default()}
	s := &Session{
		Backend: rec,
		halW:    map[string]compute.Tensor{},
	}

	qtUnsupported := &kQuantTensor{out: 4, in: 256, nblk: 1, kind: kindIQ3XXS, raw: make([]byte, 100)}
	defer func() {
		r := recover()
		if r == nil {
			t.Fatalf("expected panic on unsupported kind")
		}
		msg, ok := r.(string)
		if !ok || msg != "model: unsupported resident expert k-quant: IQ3_XXS" {
			t.Fatalf("unexpected panic message: %v", r)
		}
	}()
	s.weightHALKQuant("unsupported", qtUnsupported)
}

func TestWeightHALKQuantCustomRegistrationDispatch(t *testing.T) {
	ResetDefaultQuantDescriptors()
	defer ResetDefaultQuantDescriptors()

	const customKind kQuantKind = 240
	customDesc := BaseQuantDescriptor{
		QuantKind:     customKind,
		QuantName:     "CUSTOM_HAL_Q",
		ComputeDtype:  compute.F32,
		Prefix:        "custom-raw:",
		HALSupported:  true,
		BytesPerBlk:   128,
		WeightsPerBlk: 256,
		HostTensorFn: func(out, in int, raw []byte) compute.Tensor {
			return compute.NewF32(compute.Default(), []int{out, in}, make([]float32, out*in))
		},
	}
	RegisterQuantDescriptor(customDesc)

	rec := &quantUploadTestBackend{Backend: compute.Default()}
	s := &Session{
		Backend: rec,
		halW:    map[string]compute.Tensor{},
	}

	qtCustom := &kQuantTensor{out: 2, in: 256, nblk: 1, kind: customKind, raw: make([]byte, 2*128)}
	tensorCustom := s.weightHALKQuant("custom_layer", qtCustom)
	if tensorCustom.Dtype != compute.F32 {
		t.Fatalf("expected F32, got %v", tensorCustom.Dtype)
	}
	if _, ok := s.halW["custom-raw:custom_layer"]; !ok {
		t.Fatalf("expected cached key 'custom-raw:custom_layer' in halW")
	}

	// expertWeight halKey must use the custom prefix
	ew := expertWeight{name: "custom_layer", kq: qtCustom}
	if key := ew.halKey(); key != "custom-raw:custom_layer" {
		t.Fatalf("expertWeight.halKey() = %q, want 'custom-raw:custom_layer'", key)
	}
}

func TestMatWeightHALAndLMHeadHALRegistryIntegration(t *testing.T) {
	ResetDefaultQuantDescriptors()
	defer ResetDefaultQuantDescriptors()

	const out, in = 4, 256
	rawQ6 := make([]byte, out*(in/256)*q6kBlockBytes)
	qtQ6 := &kQuantTensor{out: out, in: in, nblk: in / qkK, kind: kindQ6K, raw: rawQ6}

	m := &Model{
		kqw: map[string]*kQuantTensor{
			"attn.weight":    qtQ6,
			"lm_head.weight": qtQ6,
		},
	}
	rec := &quantUploadTestBackend{Backend: compute.Default()}
	s := &Session{
		M:       m,
		Backend: rec,
		halW:    map[string]compute.Tensor{},
	}

	// matWeightHAL routing
	t1 := s.matWeightHAL("attn.weight")
	if t1.Dtype != compute.Q6_K {
		t.Fatalf("expected matWeightHAL to stage Q6_K, got %v", t1.Dtype)
	}

	// lmHeadMatHAL routing
	t2 := s.lmHeadMatHAL()
	if t2.Dtype != compute.Q6_K {
		t.Fatalf("expected lmHeadMatHAL to stage Q6_K, got %v", t2.Dtype)
	}
}

type mockIQ3XXSCapableBackend struct {
	compute.Backend
	capable bool
}

func (m *mockIQ3XXSCapableBackend) SupportsIQ3XXS() bool {
	return m.capable
}

func TestIQ3XXSHALAdmissionRequiresRegisteredCapability(t *testing.T) {
	ResetDefaultQuantDescriptors()
	defer ResetDefaultQuantDescriptors()

	// 1. Exact block metadata
	desc, ok := LookupQuantDescriptor(kindIQ3XXS)
	if !ok {
		t.Fatalf("LookupQuantDescriptor(kindIQ3XXS) not found")
	}
	if desc.Name() != "IQ3_XXS" {
		t.Errorf("Name = %q, want %q", desc.Name(), "IQ3_XXS")
	}
	if desc.Kind() != kindIQ3XXS {
		t.Errorf("Kind = %v, want %v", desc.Kind(), kindIQ3XXS)
	}
	if desc.BlockBytes() != iq3xxsBlockBytes {
		t.Errorf("BlockBytes = %d, want %d (iq3xxsBlockBytes)", desc.BlockBytes(), iq3xxsBlockBytes)
	}
	if desc.BlockBytes() != 98 {
		t.Errorf("BlockBytes = %d, want 98 (2 + 3*256/8)", desc.BlockBytes())
	}
	if desc.BlockWeights() != qkK {
		t.Errorf("BlockWeights = %d, want %d (qkK=256)", desc.BlockWeights(), qkK)
	}
	if desc.KeyPrefix() != "kquant-raw:" {
		t.Errorf("KeyPrefix = %q, want 'kquant-raw:'", desc.KeyPrefix())
	}

	// 2. Stable dtype identity
	if desc.Dtype() != compute.IQ3_XXS {
		t.Fatalf("desc.Dtype() = %v, want compute.IQ3_XXS", desc.Dtype())
	}
	if compute.IQ3_XXS.String() != "iq3_xxs" {
		t.Errorf("IQ3_XXS.String() = %q, want 'iq3_xxs'", compute.IQ3_XXS.String())
	}
	if !compute.IQ3_XXS.Quantized() {
		t.Errorf("IQ3_XXS.Quantized() = false, want true")
	}
	if compute.IQ3_XXS.Bytes() != 1 {
		t.Errorf("IQ3_XXS.Bytes() = %d, want 1", compute.IQ3_XXS.Bytes())
	}

	// 3. Default state is fail-closed (HALSupported == false)
	if desc.SupportsHAL() {
		t.Fatalf("IQ3_XXS default descriptor must NOT support HAL before capability is registered")
	}
	if SupportsHALKQuant(kindIQ3XXS) {
		t.Fatalf("SupportsHALKQuant(kindIQ3XXS) must be false by default")
	}

	// 4. Typed refusal without capability
	// 4a. Nil backend
	vNil := AdmitIQ3XXSHAL(nil)
	if vNil.Admitted {
		t.Fatalf("AdmitIQ3XXSHAL(nil) admitted, want refusal")
	}
	if vNil.Refusal == nil || vNil.Refusal.Reason != IQ3XXSRefusalNilBackend {
		t.Fatalf("AdmitIQ3XXSHAL(nil) refusal = %v, want reason %s", vNil.Refusal, IQ3XXSRefusalNilBackend)
	}

	// 4b. Incapable default backend
	defaultBE := compute.Default()
	vDefault := AdmitIQ3XXSHAL(defaultBE)
	if vDefault.Admitted {
		t.Fatalf("AdmitIQ3XXSHAL(defaultBE) admitted without capability, want refusal")
	}
	if vDefault.Refusal == nil || vDefault.Refusal.Reason != IQ3XXSRefusalNoCapability {
		t.Fatalf("AdmitIQ3XXSHAL(defaultBE) refusal = %v, want reason %s", vDefault.Refusal, IQ3XXSRefusalNoCapability)
	}
	if vDefault.Dtype != compute.IQ3_XXS {
		t.Errorf("vDefault.Dtype = %v, want compute.IQ3_XXS", vDefault.Dtype)
	}

	// 4c. Backend explicitly denying capability
	deniedBE := &mockIQ3XXSCapableBackend{Backend: compute.Default(), capable: false}
	vDenied := AdmitIQ3XXSHAL(deniedBE)
	if vDenied.Admitted {
		t.Fatalf("AdmitIQ3XXSHAL(deniedBE) admitted, want refusal")
	}
	if vDenied.Refusal == nil || vDenied.Refusal.Reason != IQ3XXSRefusalCapabilityDenied {
		t.Fatalf("AdmitIQ3XXSHAL(deniedBE) refusal = %v, want reason %s", vDenied.Refusal, IQ3XXSRefusalCapabilityDenied)
	}

	// 4d. Typed refusal via AdmitHALQuant
	_, err := AdmitHALQuant(kindIQ3XXS, defaultBE)
	if err == nil {
		t.Fatalf("AdmitHALQuant(kindIQ3XXS, defaultBE) want typed refusal, got nil")
	}
	var refusal *IQ3XXSHALAdmissionRefusal
	if !errors.As(err, &refusal) {
		t.Fatalf("AdmitHALQuant error %T is not *IQ3XXSHALAdmissionRefusal", err)
	}

	// 5. Admission only when a matching backend capability is injected
	capableBE := &mockIQ3XXSCapableBackend{Backend: compute.Default(), capable: true}
	vCapable := AdmitIQ3XXSHAL(capableBE)
	if !vCapable.Admitted {
		t.Fatalf("AdmitIQ3XXSHAL(capableBE) refused: %v, want admitted", vCapable.Refusal)
	}
	if vCapable.Dtype != compute.IQ3_XXS {
		t.Errorf("vCapable.Dtype = %v, want compute.IQ3_XXS", vCapable.Dtype)
	}
	if vCapable.Refusal != nil {
		t.Errorf("vCapable.Refusal = %v, want nil", vCapable.Refusal)
	}

	admittedDtype, err := AdmitHALQuant(kindIQ3XXS, capableBE)
	if err != nil {
		t.Fatalf("AdmitHALQuant(kindIQ3XXS, capableBE) returned error: %v", err)
	}
	if admittedDtype != compute.IQ3_XXS {
		t.Errorf("admittedDtype = %v, want compute.IQ3_XXS", admittedDtype)
	}

	// 6. Fail-closed: descriptor with HALSupported: true must NOT bypass backend capability requirement
	capableDesc := BaseQuantDescriptor{
		QuantKind:     kindIQ3XXS,
		QuantName:     "IQ3_XXS",
		ComputeDtype:  compute.IQ3_XXS,
		Prefix:        "kquant-raw:",
		HALSupported:  true,
		BytesPerBlk:   iq3xxsBlockBytes,
		WeightsPerBlk: qkK,
	}
	RegisterQuantDescriptor(capableDesc)
	if !SupportsHALKQuant(kindIQ3XXS) {
		t.Fatalf("SupportsHALKQuant(kindIQ3XXS) must be true after registering descriptor with HALSupported")
	}
	vBypass := AdmitIQ3XXSHAL(defaultBE)
	if vBypass.Admitted {
		t.Fatalf("AdmitIQ3XXSHAL(defaultBE) must not admit when backend lacks capability even if descriptor has HALSupported")
	}
	if vBypass.Refusal == nil || vBypass.Refusal.Reason != IQ3XXSRefusalNoCapability {
		t.Fatalf("AdmitIQ3XXSHAL(defaultBE) refusal = %v, want %s", vBypass.Refusal, IQ3XXSRefusalNoCapability)
	}
}

type mockIQ3SCapableBackend struct {
	compute.Backend
	capable bool
}

func (m *mockIQ3SCapableBackend) SupportsIQ3S() bool {
	return m.capable
}

func TestIQ3SHALAdmissionRequiresRegisteredCapability(t *testing.T) {
	ResetDefaultQuantDescriptors()
	defer ResetDefaultQuantDescriptors()

	// 1. Exact block metadata
	desc, ok := LookupQuantDescriptor(kindIQ3S)
	if !ok {
		t.Fatalf("LookupQuantDescriptor(kindIQ3S) not found")
	}
	if desc.Name() != "IQ3_S" {
		t.Errorf("Name = %q, want %q", desc.Name(), "IQ3_S")
	}
	if desc.Kind() != kindIQ3S {
		t.Errorf("Kind = %v, want %v", desc.Kind(), kindIQ3S)
	}
	if desc.BlockBytes() != iq3sBlockBytes {
		t.Errorf("BlockBytes = %d, want %d (iq3sBlockBytes)", desc.BlockBytes(), iq3sBlockBytes)
	}
	if desc.BlockBytes() != 110 {
		t.Errorf("BlockBytes = %d, want 110", desc.BlockBytes())
	}
	if desc.BlockWeights() != qkK {
		t.Errorf("BlockWeights = %d, want %d (qkK=256)", desc.BlockWeights(), qkK)
	}
	if desc.KeyPrefix() != "kquant-raw:" {
		t.Errorf("KeyPrefix = %q, want 'kquant-raw:'", desc.KeyPrefix())
	}

	// 2. Stable dtype identity
	if desc.Dtype() != compute.IQ3_S {
		t.Fatalf("desc.Dtype() = %v, want compute.IQ3_S", desc.Dtype())
	}
	if compute.IQ3_S.String() != "iq3_s" {
		t.Errorf("IQ3_S.String() = %q, want 'iq3_s'", compute.IQ3_S.String())
	}
	if !compute.IQ3_S.Quantized() {
		t.Errorf("IQ3_S.Quantized() = false, want true")
	}
	if compute.IQ3_S.Bytes() != 1 {
		t.Errorf("IQ3_S.Bytes() = %d, want 1", compute.IQ3_S.Bytes())
	}

	// 3. Default state is fail-closed (HALSupported == false)
	if desc.SupportsHAL() {
		t.Fatalf("IQ3_S default descriptor must NOT support HAL before capability is registered")
	}
	if SupportsHALKQuant(kindIQ3S) {
		t.Fatalf("SupportsHALKQuant(kindIQ3S) must be false by default")
	}

	// 4. Typed refusal without capability
	// 4a. Nil backend
	vNil := AdmitIQ3SHAL(nil)
	if vNil.Admitted {
		t.Fatalf("AdmitIQ3SHAL(nil) admitted, want refusal")
	}
	if vNil.Refusal == nil || vNil.Refusal.Reason != IQ3SRefusalNilBackend {
		t.Fatalf("AdmitIQ3SHAL(nil) refusal = %v, want reason %s", vNil.Refusal, IQ3SRefusalNilBackend)
	}

	// 4b. Incapable default backend
	defaultBE := compute.Default()
	vDefault := AdmitIQ3SHAL(defaultBE)
	if vDefault.Admitted {
		t.Fatalf("AdmitIQ3SHAL(defaultBE) admitted without capability, want refusal")
	}
	if vDefault.Refusal == nil || vDefault.Refusal.Reason != IQ3SRefusalNoCapability {
		t.Fatalf("AdmitIQ3SHAL(defaultBE) refusal = %v, want reason %s", vDefault.Refusal, IQ3SRefusalNoCapability)
	}
	if vDefault.Dtype != compute.IQ3_S {
		t.Errorf("vDefault.Dtype = %v, want compute.IQ3_S", vDefault.Dtype)
	}

	// 4c. Backend explicitly denying capability
	deniedBE := &mockIQ3SCapableBackend{Backend: compute.Default(), capable: false}
	vDenied := AdmitIQ3SHAL(deniedBE)
	if vDenied.Admitted {
		t.Fatalf("AdmitIQ3SHAL(deniedBE) admitted, want refusal")
	}
	if vDenied.Refusal == nil || vDenied.Refusal.Reason != IQ3SRefusalCapabilityDenied {
		t.Fatalf("AdmitIQ3SHAL(deniedBE) refusal = %v, want reason %s", vDenied.Refusal, IQ3SRefusalCapabilityDenied)
	}

	// 4d. Typed refusal via AdmitHALQuant
	_, err := AdmitHALQuant(kindIQ3S, defaultBE)
	if err == nil {
		t.Fatalf("AdmitHALQuant(kindIQ3S, defaultBE) want typed refusal, got nil")
	}
	var refusal *IQ3SHALAdmissionRefusal
	if !errors.As(err, &refusal) {
		t.Fatalf("AdmitHALQuant error %T is not *IQ3SHALAdmissionRefusal", err)
	}

	// 4e. Fail-closed: descriptor with HALSupported: true must NOT bypass backend capability requirement
	capableDesc := BaseQuantDescriptor{
		QuantKind:     kindIQ3S,
		QuantName:     "IQ3_S",
		ComputeDtype:  compute.IQ3_S,
		Prefix:        "kquant-raw:",
		HALSupported:  true,
		BytesPerBlk:   iq3sBlockBytes,
		WeightsPerBlk: qkK,
	}
	RegisterQuantDescriptor(capableDesc)
	if !SupportsHALKQuant(kindIQ3S) {
		t.Fatalf("SupportsHALKQuant(kindIQ3S) must be true after registering descriptor with HALSupported")
	}
	vBypass := AdmitIQ3SHAL(defaultBE)
	if vBypass.Admitted {
		t.Fatalf("AdmitIQ3SHAL(defaultBE) must not admit when backend lacks capability even if descriptor has HALSupported")
	}
	if vBypass.Refusal == nil || vBypass.Refusal.Reason != IQ3SRefusalNoCapability {
		t.Fatalf("AdmitIQ3SHAL(defaultBE) refusal = %v, want %s", vBypass.Refusal, IQ3SRefusalNoCapability)
	}

	// 5. Admission only when a matching backend capability is injected
	capableBE := &mockIQ3SCapableBackend{Backend: compute.Default(), capable: true}
	vCapable := AdmitIQ3SHAL(capableBE)
	if !vCapable.Admitted {
		t.Fatalf("AdmitIQ3SHAL(capableBE) refused: %v, want admitted", vCapable.Refusal)
	}
	if vCapable.Dtype != compute.IQ3_S {
		t.Errorf("vCapable.Dtype = %v, want compute.IQ3_S", vCapable.Dtype)
	}
	if vCapable.Refusal != nil {
		t.Errorf("vCapable.Refusal = %v, want nil", vCapable.Refusal)
	}

	admittedDtype, err := AdmitHALQuant(kindIQ3S, capableBE)
	if err != nil {
		t.Fatalf("AdmitHALQuant(kindIQ3S, capableBE) returned error: %v", err)
	}
	if admittedDtype != compute.IQ3_S {
		t.Errorf("admittedDtype = %v, want compute.IQ3_S", admittedDtype)
	}
}
