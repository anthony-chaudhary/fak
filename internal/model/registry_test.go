package model

import (
	"errors"
	"sync"
	"testing"
)

func TestDeclarativeRegistry(t *testing.T) {
	ClearModelSpecs()

	spec := ModelSpec{
		ID:     "qwen3.8-test",
		Family: "qwen",
		Formats: []FormatSignature{
			{
				Container:   "gguf",
				Quant:       "q4_k_m",
				TensorOrder: "canonical",
				Features:    []string{"simd"},
			},
		},
		Backends: []BackendSpec{
			{
				Name:     "cpu-ref",
				Priority: 10,
				DeviceReq: DeviceRequirement{
					BackendName: "cpu-ref",
				},
			},
			{
				Name:     "metal",
				Priority: 50,
				DeviceReq: DeviceRequirement{
					BackendName:      "metal",
					RequiredFeatures: []string{"simd"},
				},
			},
			{
				Name:     "cuda",
				Priority: 100,
				DeviceReq: DeviceRequirement{
					BackendName:      "cuda",
					MinDeviceArch:    "sm_80",
					RequiredFeatures: []string{"fp16"},
					MinMemoryBytes:   8 * 1024 * 1024 * 1024,
				},
			},
		},
	}

	RegisterModelSpec(spec)

	gotSpec, ok := LookupModelSpec("qwen3.8-test")
	if !ok {
		t.Fatalf("LookupModelSpec returned false, want true")
	}
	if gotSpec.ID != "qwen3.8-test" || len(gotSpec.Backends) != 3 {
		t.Fatalf("LookupModelSpec returned unexpected spec: %+v", gotSpec)
	}

	t.Run("metal on arm64 selects metal over cpu", func(t *testing.T) {
		hostMetal := HostProfile{
			Arch:       "arm64",
			OS:         "darwin",
			HasMetal:   true,
			Features:   map[string]bool{"simd": true},
			TotalRAM:   16 * 1024 * 1024 * 1024,
			DeviceVRAM: 0,
		}
		backend, err := ResolveModelBackend("qwen3.8-test", hostMetal)
		if err != nil {
			t.Fatalf("ResolveModelBackend unexpected error: %v", err)
		}
		if backend.Name != "metal" {
			t.Fatalf("ResolveModelBackend selected %q, want %q", backend.Name, "metal")
		}
	})

	t.Run("cuda on linux selects cuda", func(t *testing.T) {
		hostCUDA := HostProfile{
			Arch:       "amd64",
			OS:         "linux",
			HasCUDA:    true,
			CUDAArch:   "sm_89",
			Features:   map[string]bool{"fp16": true, "simd": true},
			TotalRAM:   32 * 1024 * 1024 * 1024,
			DeviceVRAM: 16 * 1024 * 1024 * 1024,
		}
		backend, err := ResolveModelBackend("qwen3.8-test", hostCUDA)
		if err != nil {
			t.Fatalf("ResolveModelBackend unexpected error: %v", err)
		}
		if backend.Name != "cuda" {
			t.Fatalf("ResolveModelBackend selected %q, want %q", backend.Name, "cuda")
		}
	})

	t.Run("cpu only host selects cpu-ref fallback", func(t *testing.T) {
		hostCPU := HostProfile{
			Arch:     "amd64",
			OS:       "linux",
			Features: map[string]bool{"simd": true},
			TotalRAM: 16 * 1024 * 1024 * 1024,
		}
		backend, err := ResolveModelBackend("qwen3.8-test", hostCPU)
		if err != nil {
			t.Fatalf("ResolveModelBackend unexpected error: %v", err)
		}
		if backend.Name != "cpu-ref" {
			t.Fatalf("ResolveModelBackend selected %q, want %q", backend.Name, "cpu-ref")
		}
	})

	t.Run("missing required features fails closed", func(t *testing.T) {
		specFeatureReq := ModelSpec{
			ID:     "feature-bound-model",
			Family: "test",
			Backends: []BackendSpec{
				{
					Name:     "cpu-ref",
					Priority: 10,
					DeviceReq: DeviceRequirement{
						BackendName:      "cpu-ref",
						RequiredFeatures: []string{"dotprod"},
					},
				},
			},
		}
		RegisterModelSpec(specFeatureReq)

		hostNoFeature := HostProfile{
			Arch:     "amd64",
			OS:       "linux",
			Features: map[string]bool{"simd": true},
		}
		_, err := ResolveModelBackend("feature-bound-model", hostNoFeature)
		if !errors.Is(err, ErrNoMatchingBackend) {
			t.Fatalf("ResolveModelBackend error = %v, want %v", err, ErrNoMatchingBackend)
		}
	})

	t.Run("model not in registry returns ErrModelNotFound", func(t *testing.T) {
		host := HostProfile{
			Arch: "amd64",
			OS:   "linux",
		}
		_, err := ResolveModelBackend("non-existent-model-id", host)
		if !errors.Is(err, ErrModelNotFound) {
			t.Fatalf("ResolveModelBackend error = %v, want %v", err, ErrModelNotFound)
		}
	})
}

func TestMatchDeviceRequirement(t *testing.T) {
	tests := []struct {
		name string
		req  DeviceRequirement
		host HostProfile
		want bool
	}{
		// Memory
		{
			name: "memory: device vram meets requirement",
			req:  DeviceRequirement{MinMemoryBytes: 4000},
			host: HostProfile{DeviceVRAM: 8000, TotalRAM: 1000},
			want: true,
		},
		{
			name: "memory: total ram meets requirement when vram insufficient",
			req:  DeviceRequirement{MinMemoryBytes: 4000},
			host: HostProfile{DeviceVRAM: 1000, TotalRAM: 8000},
			want: true,
		},
		{
			name: "memory: neither ram nor vram meets requirement",
			req:  DeviceRequirement{MinMemoryBytes: 4000},
			host: HostProfile{DeviceVRAM: 2000, TotalRAM: 3000},
			want: false,
		},
		{
			name: "memory: zero minimum memory bytes always matches",
			req:  DeviceRequirement{MinMemoryBytes: 0},
			host: HostProfile{DeviceVRAM: 0, TotalRAM: 0},
			want: true,
		},

		// Architecture
		{
			name: "arch: cuda exact arch match",
			req:  DeviceRequirement{BackendName: "cuda", MinDeviceArch: "sm_80"},
			host: HostProfile{HasCUDA: true, CUDAArch: "sm_80"},
			want: true,
		},
		{
			name: "arch: cuda newer arch satisfies requirement",
			req:  DeviceRequirement{BackendName: "cuda", MinDeviceArch: "sm_80"},
			host: HostProfile{HasCUDA: true, CUDAArch: "sm_89"},
			want: true,
		},
		{
			name: "arch: cuda older arch fails requirement",
			req:  DeviceRequirement{BackendName: "cuda", MinDeviceArch: "sm_80"},
			host: HostProfile{HasCUDA: true, CUDAArch: "sm_75"},
			want: false,
		},
		{
			name: "arch: cuda cross-generation arch sm_100 satisfies sm_80",
			req:  DeviceRequirement{BackendName: "cuda", MinDeviceArch: "sm_80"},
			host: HostProfile{HasCUDA: true, CUDAArch: "sm_100"},
			want: true,
		},
		{
			name: "arch: cuda missing host cuda arch fails",
			req:  DeviceRequirement{BackendName: "cuda", MinDeviceArch: "sm_80"},
			host: HostProfile{HasCUDA: true, CUDAArch: ""},
			want: false,
		},
		{
			name: "arch: non-cuda arch requirement does not fail",
			req:  DeviceRequirement{BackendName: "metal", MinDeviceArch: "metal3"},
			host: HostProfile{HasMetal: true},
			want: true,
		},

		// Features
		{
			name: "features: all required features present",
			req:  DeviceRequirement{RequiredFeatures: []string{"fp16", "dotprod"}},
			host: HostProfile{Features: map[string]bool{"fp16": true, "dotprod": true, "avx2": true}},
			want: true,
		},
		{
			name: "features: required feature missing from map",
			req:  DeviceRequirement{RequiredFeatures: []string{"fp16", "dotprod"}},
			host: HostProfile{Features: map[string]bool{"fp16": true}},
			want: false,
		},
		{
			name: "features: required feature present but false",
			req:  DeviceRequirement{RequiredFeatures: []string{"fp16"}},
			host: HostProfile{Features: map[string]bool{"fp16": false}},
			want: false,
		},
		{
			name: "features: nil features map fails when required feature non-empty",
			req:  DeviceRequirement{RequiredFeatures: []string{"fp16"}},
			host: HostProfile{Features: nil},
			want: false,
		},
		{
			name: "features: empty required features matches empty host features",
			req:  DeviceRequirement{RequiredFeatures: nil},
			host: HostProfile{Features: nil},
			want: true,
		},

		// Platform flags
		{
			name: "platform: cuda matches when has_cuda is true",
			req:  DeviceRequirement{BackendName: "cuda"},
			host: HostProfile{HasCUDA: true},
			want: true,
		},
		{
			name: "platform: cuda fails when has_cuda is false",
			req:  DeviceRequirement{BackendName: "cuda"},
			host: HostProfile{HasCUDA: false},
			want: false,
		},
		{
			name: "platform: metal matches when has_metal is true",
			req:  DeviceRequirement{BackendName: "metal"},
			host: HostProfile{HasMetal: true},
			want: true,
		},
		{
			name: "platform: metal fails when has_metal is false",
			req:  DeviceRequirement{BackendName: "metal"},
			host: HostProfile{HasMetal: false},
			want: false,
		},
		{
			name: "platform: vulkan matches when has_vulkan is true",
			req:  DeviceRequirement{BackendName: "vulkan"},
			host: HostProfile{HasVulkan: true},
			want: true,
		},
		{
			name: "platform: vulkan fails when has_vulkan is false",
			req:  DeviceRequirement{BackendName: "vulkan"},
			host: HostProfile{HasVulkan: false},
			want: false,
		},
		{
			name: "platform: cpu backend does not require gpu flags",
			req:  DeviceRequirement{BackendName: "cpu-ref"},
			host: HostProfile{},
			want: true,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := MatchDeviceRequirement(tc.req, tc.host)
			if got != tc.want {
				t.Errorf("MatchDeviceRequirement() = %v, want %v", got, tc.want)
			}
		})
	}
}

func TestResolveBackendRanking(t *testing.T) {
	spec := ModelSpec{
		ID:     "multi-backend-model",
		Family: "test",
		Backends: []BackendSpec{
			{
				Name:     "low-priority-cpu",
				Priority: 5,
				DeviceReq: DeviceRequirement{
					BackendName: "cpu-ref",
				},
			},
			{
				Name:     "high-priority-cpu",
				Priority: 25,
				DeviceReq: DeviceRequirement{
					BackendName: "cpu-ref",
				},
			},
			{
				Name:     "medium-priority-cpu",
				Priority: 15,
				DeviceReq: DeviceRequirement{
					BackendName: "cpu-ref",
				},
			},
		},
	}

	host := HostProfile{
		Arch: "amd64",
		OS:   "linux",
	}

	resolved, err := ResolveBackend(spec, host)
	if err != nil {
		t.Fatalf("ResolveBackend unexpected error: %v", err)
	}
	if resolved.Name != "high-priority-cpu" {
		t.Fatalf("ResolveBackend selected %q, want %q", resolved.Name, "high-priority-cpu")
	}
}

func TestRegistryConcurrency(t *testing.T) {
	ClearModelSpecs()

	const workers = 8
	const iterations = 50

	var wg sync.WaitGroup
	wg.Add(workers)

	for w := 0; w < workers; w++ {
		go func(workerID int) {
			defer wg.Done()
			spec := ModelSpec{
				ID:     "concurrent-model",
				Family: "test",
				Backends: []BackendSpec{
					{
						Name:     "cpu-ref",
						Priority: 10,
						DeviceReq: DeviceRequirement{
							BackendName: "cpu-ref",
						},
					},
				},
			}

			host := HostProfile{
				Arch: "amd64",
				OS:   "linux",
			}

			for i := 0; i < iterations; i++ {
				RegisterModelSpec(spec)
				_, _ = LookupModelSpec("concurrent-model")
				_, _ = ResolveModelBackend("concurrent-model", host)
			}
		}(w)
	}

	wg.Wait()
}
