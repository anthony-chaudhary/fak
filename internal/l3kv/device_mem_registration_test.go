package l3kv

import (
	"errors"
	"math"
	"reflect"
	"testing"
)

// fak-test:runtime fast est=1s
func TestDeviceMemRegistration(t *testing.T) {
	host := RegistrationBuffer{Class: RegistrationMemoryHost, Address: 0x10000, Length: 4096, CUDADevice: -1}
	device := RegistrationBuffer{Class: RegistrationMemoryCUDA, Address: 0x20000, Length: 8192, CUDADevice: 0}
	ready := RegistrationContextCaps{Ready: true, CUDAMemory: true}

	t.Run("class_and_device_zero_are_preserved", func(t *testing.T) {
		fake := memoryRegistrationOracle{caps: MemoryRegistrationCaps{CUDAEnabled: true, Contexts: []RegistrationContextCaps{ready, ready}}}
		for _, buffer := range []RegistrationBuffer{host, device} {
			if err := fake.admit(buffer); err != nil {
				t.Fatal(err)
			}
		}
		if len(fake.plans) != 2 || fake.plans[0].Interface != RegistrationInterfaceSystem || fake.plans[1].Interface != RegistrationInterfaceCUDA || fake.plans[0].Buffer != host || fake.plans[1].Buffer != device || fake.plans[1].ContextCount != 2 {
			t.Fatalf("registration class/ordinal lost: %+v", fake.plans)
		}
		// Plans contain owned scalar descriptors, never references to the
		// caller's mutable descriptor or its capability slice.
		before := fake.plans[1]
		deviceCopy := device
		deviceCopy.CUDADevice = 7
		if err := fake.admit(deviceCopy); err != nil || fake.plans[2].Buffer.CUDADevice != 7 {
			t.Fatalf("explicit CUDA ordinal lost: %v", err)
		}
		fake.caps.Contexts[0].CUDAMemory = false
		deviceCopy.Address = 1
		if fake.plans[1] != before || fake.plans[2].Buffer.Address != device.Address {
			t.Fatal("plan aliases mutable caller metadata")
		}
	})

	t.Run("unsupported_device_never_becomes_host", func(t *testing.T) {
		for _, caps := range []MemoryRegistrationCaps{
			{Contexts: []RegistrationContextCaps{ready}}, // no CUDA build
			{CUDAEnabled: true, Contexts: []RegistrationContextCaps{{Ready: true}}},
			{CUDAEnabled: true, Contexts: []RegistrationContextCaps{ready, {Ready: true}}}, // all, not any
		} {
			fake := memoryRegistrationOracle{caps: caps}
			if err := fake.admit(host); err != nil {
				t.Fatalf("host should not require CUDA support: %v", err)
			}
			before := append([]MemoryRegistrationPlan(nil), fake.plans...)
			if err := fake.admit(device); !errors.Is(err, ErrDeviceMemoryUnsupported) || !reflect.DeepEqual(fake.plans, before) {
				t.Fatalf("unsupported device registered or fell back: plans=%+v, err=%v", fake.plans, err)
			}
		}
	})

	t.Run("fail_closed_metadata_and_transport", func(t *testing.T) {
		caps := MemoryRegistrationCaps{CUDAEnabled: true, Contexts: []RegistrationContextCaps{ready}}
		for _, bad := range []RegistrationBuffer{
			{}, {Class: 99, Address: 1, Length: 1, CUDADevice: -1},
			{Class: RegistrationMemoryHost, Address: 1, Length: 1, CUDADevice: 0},
			{Class: RegistrationMemoryCUDA, Address: 1, Length: 1, CUDADevice: -1},
			{Class: RegistrationMemoryHost, Address: 0, Length: 1, CUDADevice: -1},
			{Class: RegistrationMemoryHost, Address: 1, Length: 0, CUDADevice: -1},
			{Class: RegistrationMemoryHost, Address: math.MaxUint64 - 3, Length: 4, CUDADevice: -1},
		} {
			out, err := PlanMemoryRegistration(bad, caps)
			if !errors.Is(err, ErrMemoryRegistrationInput) || out != (MemoryRegistrationPlan{}) {
				t.Fatalf("invalid descriptor admitted: %+v, %v", out, err)
			}
		}
		for _, bad := range []MemoryRegistrationCaps{{}, {CUDAEnabled: true, Contexts: []RegistrationContextCaps{ready, {}}}} {
			for _, buffer := range []RegistrationBuffer{host, device} {
				out, err := PlanMemoryRegistration(buffer, bad)
				if !errors.Is(err, ErrMemoryTransportUnavailable) || out != (MemoryRegistrationPlan{}) {
					t.Fatalf("unavailable transport admitted: %+v, %v", out, err)
				}
			}
		}
		boundary := RegistrationBuffer{Class: RegistrationMemoryHost, Address: math.MaxUint64 - 1, Length: 1, CUDADevice: -1}
		if out, err := PlanMemoryRegistration(boundary, caps); err != nil || out.Buffer != boundary {
			t.Fatalf("representable exclusive endpoint rejected: %+v, %v", out, err)
		}
	})
}

// This records accepted plans only. It is not a fake CUDA/NIC allocation and
// never dereferences synthetic addresses or performs actual registration.
type memoryRegistrationOracle struct {
	caps  MemoryRegistrationCaps
	plans []MemoryRegistrationPlan
}

func (o *memoryRegistrationOracle) admit(buffer RegistrationBuffer) error {
	plan, err := PlanMemoryRegistration(buffer, o.caps)
	if err != nil {
		return err
	}
	o.plans = append(o.plans, plan)
	return nil
}
