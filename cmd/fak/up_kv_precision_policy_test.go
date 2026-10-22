package main

import (
	"context"
	"errors"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

type turnkeyKVPolicyBackend struct {
	compute.Backend
	name string
}

func (b turnkeyKVPolicyBackend) Name() string { return b.name }

// fak-test:runtime fast est=10ms lane=default
func TestTurnkeyKVPrecisionRefusesBeforeModelAdmission(t *testing.T) {
	for _, tc := range []struct {
		name, env, flag, backend string
		metal, refuse            bool
	}{
		{name: "metal env q8", env: "q8_0", metal: true, refuse: true},
		{name: "metal flag q8 wins over fp32 env", env: "fp32", flag: "q8_0", metal: true, refuse: true},
		{name: "vulkan q8", env: "q8_0", backend: "vulkan", refuse: true},
		{name: "vulkan fp16", env: "fp16", backend: "vulkan", refuse: true},
		{name: "host fp16", env: "fp16", refuse: true},
		{name: "host q8", env: "q8_0"},
		{name: "host unset"},
		{name: "metal unset", metal: true},
		{name: "vulkan unset", backend: "vulkan"},
		{name: "metal fp32", env: "fp32", metal: true},
		{name: "vulkan fp32", env: "fp32", backend: "vulkan"},
		{name: "explicit fp32 wins", env: "q8_0", flag: "fp32", metal: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Setenv("FAK_UP_KV_PRECISION", tc.env)
			if _, err := resolveUpKVPrecision(tc.flag); err != nil {
				t.Fatal(err)
			}
			reachedAdmission := false
			stop := errors.New("test stops before model load")
			deps := turnkeyNativeLoadDeps{
				resolveBackend: func() (compute.Backend, error) {
					if tc.backend == "" {
						return nil, nil
					}
					return turnkeyKVPolicyBackend{name: tc.backend}, nil
				},
				resolveMetal: func() (serveMetalDecision, error) {
					return serveMetalDecision{live: tc.metal}, nil
				},
				refusePeak: func(string) error {
					reachedAdmission = true
					return stop
				},
				admitAndLoad: func(bool, string, func(), *serveFitBudget) (func(), error) {
					reachedAdmission = true
					return nil, stop
				},
			}
			resources, err := loadTurnkeyNativeResourcesWith(context.Background(), "not-loaded.gguf", "test", 128, deps)
			if resources != nil {
				t.Fatal("unexpected loaded resources")
			}
			if tc.refuse {
				if !errors.Is(err, ErrKVPrecisionUnrealized) || reachedAdmission {
					t.Fatalf("err=%v reachedAdmission=%v; want typed refusal before admission", err, reachedAdmission)
				}
			} else if !errors.Is(err, stop) || !reachedAdmission {
				t.Fatalf("err=%v reachedAdmission=%v; want realized tier admitted to the loader boundary", err, reachedAdmission)
			}
		})
	}
}
