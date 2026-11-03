//go:build cuda && cgo

package main

import (
	"bytes"
	"flag"
	"fmt"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/ggufload"
	fakmodel "github.com/anthony-chaudhary/fak/internal/model"
)

var qwen35Q56KMMQRequired = flag.String("qwen35-q56k-mmq-required", "",
	"require actual retained Q5/Q6 sequence MMQ (1) or separate default archive (default)")

type serveQ56KMMQDevice interface {
	serveIQSequenceDevice
	Q56KMMQLaunchCounts() (uint64, uint64)
}

// Ordinary serve loader -> retained canonical GGUF -> Session.Prefill ->
// full sequence -> two continuing decode steps. No direct-kernel substitute.
// The shared four-layer fixture keeps actual 24Q/4KV/256 attention geometry;
// it is not a full Qwen3.8 model, a latency cell, or a failed-drain injection.
// fak-test:runtime slow est=30s lane=optin
func TestServeCUDAQwen35RetainedQ56KMMQ(t *testing.T) {
	mode := *qwen35Q56KMMQRequired
	if mode == "" {
		t.Skip("set required mode 1 or default on an admitted CUDA node")
	}
	if mode != "1" && mode != "default" {
		t.Fatal("required mode must be 1 or default")
	}
	selected := compute.CUDAQ56KMMQQualificationEnabled()
	if selected != (mode == "1") {
		t.Fatal("required matching rebuilt native archive is absent")
	}
	backend, ok := compute.Lookup("cuda")
	if !ok || backend == nil || backend.Name() != "cuda" {
		t.Fatal("required real CUDA backend absent")
	}
	be, ok := backend.(serveQ56KMMQDevice)
	if !ok {
		t.Fatalf("required native witness interface absent: %T", backend)
	}
	t.Setenv("FAK_W3_MLP", "0") // existing loader policy, identical in both arms
	for _, kind := range []ggufload.TensorType{ggufload.TensorQ5_K, ggufload.TensorQ6_K} {
		t.Run(fmt.Sprint(kind), func(t *testing.T) {
			m, raw := serveIQSequenceFixture(t, be, kind)
			if len(raw) != 15 {
				t.Fatalf("retained fixture projection count=%d", len(raw))
			}
			for name, want := range raw {
				got, resident := m.KQuantRaw(name)
				if !resident || !bytes.Equal(got, want) || m.HasQ8(name) || m.HasQ4K(name) {
					t.Fatalf("ordinary serve loader widened/changed retained %s %s", kind, name)
				}
			}
			for _, p := range []int{128, 129} {
				t.Run(fmt.Sprint(p), func(t *testing.T) {
					prompt := make([]int, p)
					for i := range prompt {
						prompt[i] = (3 + i*7) % 32
					}
					cpu := m.NewSession()
					defer cpu.Close()
					want := append([]float32(nil), cpu.Prefill(prompt)...)
					next1 := append([]float32(nil), cpu.Step(17)...)
					next2 := append([]float32(nil), cpu.Step(19)...)
					device, err := m.NewBackendSessionChecked(be)
					if err != nil {
						t.Fatal(err)
					}
					defer device.Close()
					defer be.Recycle()
					device.SetExecutionPolicy(fakmodel.ExecutionPolicyDeviceOnly)
					be.ResetQwen35GDNOperationCount()
					q5, q6 := be.Q56KMMQLaunchCounts()
					got := device.Prefill(prompt)
					if err := device.RequireQwen35SequencePrefillNativePerformance(); err != nil {
						t.Fatal(err)
					}
					if be.Qwen35GDNOperationCount() != uint64(3*p) {
						t.Fatal("ordinary caller missed complete GDN sequence")
					}
					a5, a6 := be.Q56KMMQLaunchCounts()
					want5, want6 := uint64(0), uint64(0)
					if selected && kind == ggufload.TensorQ5_K {
						want5 = 14
					}
					if selected && kind == ggufload.TensorQ6_K {
						want6 = 14
					}
					if a5-q5 != want5 || a6-q6 != want6 {
						t.Fatalf("ordinary retained projection launch deltas=%d/%d want %d/%d", a5-q5, a6-q6, want5, want6)
					}
					serveIQSequenceParity(t, "prefill", want, got)
					serveIQSequenceParity(t, "first continuing decode", next1, device.Step(17))
					serveIQSequenceParity(t, "second continuing decode", next2, device.Step(19))
					d5, d6 := be.Q56KMMQLaunchCounts()
					if d5 != a5 || d6 != a6 {
						t.Fatal("short decode unexpectedly selected prompt MMQ")
					}
					if device.HostFallbackObserved() || device.BackendSessionClosed() {
						t.Fatal("device-only sequence fell back or retired")
					}
					for name, want := range raw {
						got, resident := m.KQuantRaw(name)
						if !resident || !bytes.Equal(got, want) {
							t.Fatal("packed owner changed after sequence/decode")
						}
					}
				})
			}
		})
	}
}
