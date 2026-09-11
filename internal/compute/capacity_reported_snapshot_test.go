package compute

import "testing"

func TestRefuseMemoryPlanIfTooBigForReportedDevice(t *testing.T) {
	t.Run("supplied device budget overrides live backend memory", func(t *testing.T) {
		backend := capDevice{total: 2 << 30, free: 1 << 30, known: true}
		plan := MemoryPlan{{Class: MemoryWeights, Bytes: 4 << 30}}

		if err := RefuseMemoryPlanIfTooBigForReportedDevice(backend, plan, 16<<30, 12<<30, true, 0); err != nil {
			t.Fatalf("plan fits supplied device budget despite smaller live backend budget: %v", err)
		}
	})

	t.Run("unknown supplied budget fails open for device demands", func(t *testing.T) {
		backend := capDevice{total: 2 << 30, free: 1 << 30, known: true}
		plan := MemoryPlan{{Class: MemoryWeights, Bytes: 4 << 30}}

		if err := RefuseMemoryPlanIfTooBigForReportedDevice(backend, plan, 0, FreeUnknown, false, 0); err != nil {
			t.Fatalf("unknown supplied device budget must fail open: %v", err)
		}
	})

	t.Run("host demands still use known host capacity", func(t *testing.T) {
		backend := capDevice{
			total: 16 << 30, free: 12 << 30, known: true,
			hostTotal: 8 << 30, hostFree: 1 << 30, hostKnown: true, hostProbe: true,
		}
		plan := MemoryPlan{{Class: MemoryOffload, Bytes: 2 << 30, Scope: MemoryScopeHost}}

		err := RefuseMemoryPlanIfTooBigForReportedDevice(backend, plan, 16<<30, 12<<30, true, 0)
		if err == nil {
			t.Fatal("known-too-small host capacity must refuse host-scoped demand")
		}
		fitErr, ok := err.(*FitError)
		if !ok {
			t.Fatalf("host refusal type = %T, want *FitError", err)
		}
		if fitErr.Scope != MemoryScopeHost || fitErr.Want != 2<<30 || fitErr.Avail != 1<<30 {
			t.Fatalf("host refusal = %+v, want scope=host want=2GiB avail=1GiB", fitErr)
		}
	})
}
