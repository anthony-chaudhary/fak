package main

import (
	"errors"
	"fmt"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

type haloGPUBackend struct {
	compute.Backend
	name   string
	device bool
}

func (b haloGPUBackend) Name() string { return b.name }

func (b haloGPUBackend) Caps() compute.Caps {
	caps := b.Backend.Caps()
	caps.DeviceMemory = b.device
	return caps
}

// fak-test:runtime fast est=2ms lane=default
func TestResolveServeChatBackendHaloRequiresDeviceExecution(t *testing.T) {
	device := haloGPUBackend{Backend: compute.Default(), name: "vulkan", device: true}
	host := haloGPUBackend{Backend: compute.Default(), name: "vulkan", device: false}
	if !device.Caps().DeviceMemory {
		t.Fatal("device test backend does not advertise device memory")
	}

	tests := []struct {
		name      string
		requested string
		halo      bool
		lookup    serveBackendLookup
		want      compute.Backend
		wantError string
	}{
		{
			name: "Halo omission selects available device backend",
			halo: true,
			lookup: func(name string) (compute.Backend, error) {
				if name == "vulkan" {
					return device, nil
				}
				return nil, fmt.Errorf("missing %s", name)
			},
			want: device,
		},
		{
			name:      "Halo omission refuses missing accelerator",
			halo:      true,
			lookup:    func(string) (compute.Backend, error) { return nil, errors.New("not linked") },
			wantError: "no-device-backend",
		},
		{
			name:      "Halo explicit CPU refuses",
			requested: "cpu",
			halo:      true,
			lookup:    func(string) (compute.Backend, error) { return device, nil },
			wantError: "cpu-backend",
		},
		{
			name: "Halo refuses backend without device memory",
			halo: true,
			lookup: func(name string) (compute.Backend, error) {
				return host, nil
			},
			wantError: "no-device-backend",
		},
		{
			name:   "non-Halo keeps portable CPU default",
			halo:   false,
			lookup: func(string) (compute.Backend, error) { return nil, errors.New("not linked") },
			want:   nil,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := resolveServeChatBackendForHost(tt.requested, "", "linux", tt.lookup, tt.halo)
			if tt.wantError != "" {
				var typed *haloGPURequiredError
				if !errors.As(err, &typed) {
					t.Fatalf("error=%T %v, want *haloGPURequiredError", err, err)
				}
				if !strings.Contains(err.Error(), tt.wantError) {
					t.Fatalf("error=%q, want reason %q", err, tt.wantError)
				}
				if got != nil {
					t.Fatalf("backend=%q after refusal, want nil", got.Name())
				}
				return
			}
			if err != nil {
				t.Fatalf("resolve: %v", err)
			}
			if got != tt.want {
				t.Fatalf("backend=%v, want %v", got, tt.want)
			}
		})
	}
}

// fak-test:runtime fast est=2ms lane=default
func TestValidateServeHaloCPUOffloadRefusesBeforeLoad(t *testing.T) {
	tests := []struct {
		name       string
		halo       bool
		offload    bool
		flagGrade  string
		envGrade   string
		wantRefuse bool
	}{
		{name: "cpu offload experts", halo: true, offload: true, wantRefuse: true},
		{name: "n cpu moe auto", halo: true, flagGrade: "auto", wantRefuse: true},
		{name: "n cpu moe positive", halo: true, flagGrade: "2", wantRefuse: true},
		{name: "environment n cpu moe positive", halo: true, envGrade: "4", wantRefuse: true},
		{name: "explicit zero overrides environment", halo: true, flagGrade: "0", envGrade: "4"},
		{name: "Halo GPU-only controls", halo: true, flagGrade: "0"},
		{name: "portable non-Halo CPU offload", offload: true, flagGrade: "auto"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := validateServeHaloCPUOffload(tt.halo, tt.offload, tt.flagGrade, tt.envGrade)
			if tt.wantRefuse {
				var typed *haloGPURequiredError
				if !errors.As(err, &typed) || !strings.Contains(err.Error(), "cpu-expert-offload") {
					t.Fatalf("error=%T %v, want cpu-expert-offload *haloGPURequiredError", err, err)
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected refusal: %v", err)
			}
		})
	}
}
