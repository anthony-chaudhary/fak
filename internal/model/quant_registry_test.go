package model

import (
	"sync"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

func TestQuantRegistryBuiltinRegistration(t *testing.T) {
	ResetDefaultQuantDescriptors()

	tests := []struct {
		kind        kQuantKind
		name        string
		dtype       compute.Dtype
		supportsHAL bool
		blockBytes  int
		blockWeights int
	}{
		{kindQ5K, "Q5_K", compute.Q5_K, true, q5kBlockBytes, qkK},
		{kindQ6K, "Q6_K", compute.Q6_K, true, q6kBlockBytes, qkK},
		{kindQ2K, "Q2_K", compute.Q2_K, true, q2kBlockBytes, qkK},
		{kindQ8_0, "Q8_0", compute.Q8_0, false, q8_0BlockBytes, q8_0BlockWeights},
		{kindQ4_0, "Q4_0", 0, false, q4_0BlockBytes, q4_0BlockWeights},
		{kindIQ3XXS, "IQ3_XXS", 0, false, iq3xxsBlockBytes, qkK},
		{kindIQ4XS, "IQ4_XS", 0, false, iq4xsBlockBytes, qkK},
		{kindIQ2XXS, "IQ2_XXS", 0, false, iq2xxsBlockBytes, qkK},
		{kindIQ2XS, "IQ2_XS", 0, false, iq2xsBlockBytes, qkK},
		{kindIQ1S, "IQ1_S", 0, false, iq1sBlockBytes, qkK},
		{kindIQ2S, "IQ2_S", 0, false, iq2sBlockBytes, qkK},
		{kindIQ1M, "IQ1_M", 0, false, iq1mBlockBytes, qkK},
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

	for _, name := range []string{"Q5_K", "Q6_K", "Q2_K", "Q8_0", "Q4_0", "IQ3_XXS"} {
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
