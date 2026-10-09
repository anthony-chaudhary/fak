package main

import (
	"math"
	"runtime"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/ggufload"
)

type mappedResidencyBackend struct {
	compute.Backend
	name                 string
	deviceMemory, upload bool
}

func (b mappedResidencyBackend) Name() string { return b.name }

func (b mappedResidencyBackend) Caps() compute.Caps {
	caps := b.Backend.Caps()
	caps.DeviceMemory = b.deviceMemory
	caps.UploadDtype = b.upload
	return caps
}

// fak-test:runtime fast est=3ms lane=default
func TestServeDeviceMappedQ4KResidencyRequiresCompatibleBackendAndPolicy(t *testing.T) {
	qualified := mappedResidencyBackend{Backend: compute.Default(), name: "vulkan", deviceMemory: true, upload: true}
	onLinux := runtime.GOOS == "linux"
	for _, tc := range []struct {
		name    string
		backend compute.Backend
		env     map[string]string
		opts    []ggufload.Q4KLoadOption
		want    bool
	}{
		{name: "qualified Vulkan device", backend: qualified, want: onLinux},
		{name: "host backend", backend: nil},
		{name: "different device API", backend: mappedResidencyBackend{Backend: compute.Default(), name: "cuda", deviceMemory: true, upload: true}},
		{name: "host memory Vulkan", backend: mappedResidencyBackend{Backend: compute.Default(), name: "vulkan", upload: true}},
		{name: "no quantized upload", backend: mappedResidencyBackend{Backend: compute.Default(), name: "vulkan", deviceMemory: true}},
		{name: "mmap zero", backend: qualified, env: map[string]string{"FAK_GGUF_MMAP": "0"}},
		{name: "mmap off", backend: qualified, env: map[string]string{"FAK_GGUF_MMAP": "off"}},
		{name: "mmap false", backend: qualified, env: map[string]string{"FAK_GGUF_MMAP": "false"}},
		{name: "dense stream environment", backend: qualified, env: map[string]string{"FAK_STREAM_Q4K": "1"}},
		{name: "Metal stream environment", backend: qualified, env: map[string]string{"FAK_METAL_STREAM_Q4K": "1"}},
		{name: "unbounded dense option", backend: qualified, opts: []ggufload.Q4KLoadOption{ggufload.WithStreamedDenseQ4K(true)}},
		{name: "bounded dense option", backend: qualified, opts: []ggufload.Q4KLoadOption{ggufload.WithStreamedDenseQ4KWorkingSet(1 << 20)}},
		{name: "streamed experts", backend: qualified, opts: []ggufload.Q4KLoadOption{ggufload.WithStreamedExperts(0)}},
		{name: "expert shard", backend: qualified, opts: []ggufload.Q4KLoadOption{ggufload.WithExpertShard(0, 1)}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			for _, key := range []string{"FAK_GGUF_MMAP", "FAK_STREAM_Q4K", "FAK_METAL_STREAM_Q4K"} {
				t.Setenv(key, tc.env[key])
			}
			if got := serveDeviceMappedQ4KResidency(tc.backend, tc.opts); got != tc.want {
				t.Fatalf("mapped residency = %v, want %v on %s", got, tc.want, runtime.GOOS)
			}
		})
	}
}

// fak-test:runtime fast est=400ms lane=default
func TestServeDeviceMappedQ4KLoadPreservesOwnedCPULogits(t *testing.T) {
	mappedQ4KServeCPUEnv(t)
	path := createMappedQ4KServeCompleteGGUF(t)
	backend := mappedResidencyBackend{Backend: compute.Default(), name: "vulkan", deviceMemory: true, upload: true}
	opts := serveDenseKQuantOptions(backend)

	owned, err := ggufload.LoadModelQ4KProfileOptions(path, nil, opts...)
	if err != nil {
		t.Fatalf("owned load: %v", err)
	}
	defer owned.CloseWeights()
	want := mappedQ4KServeCPULogits(t, owned)

	mapped, err := ggufload.LoadModelQ4KMappedResident(path, nil, opts...)
	if err != nil {
		t.Fatalf("mapped load: %v", err)
	}
	defer mapped.CloseWeights()
	got := mappedQ4KServeCPULogits(t, mapped)
	if len(got) != len(want) {
		t.Fatalf("mapped steps = %d, want %d", len(got), len(want))
	}
	for step := range want {
		if len(got[step]) != len(want[step]) {
			t.Fatalf("step %d logits = %d, want %d", step, len(got[step]), len(want[step]))
		}
		for token := range want[step] {
			if math.Float32bits(got[step][token]) != math.Float32bits(want[step][token]) {
				t.Fatalf("step %d token %d mapped bits=%x owned bits=%x", step, token, math.Float32bits(got[step][token]), math.Float32bits(want[step][token]))
			}
		}
	}
}
