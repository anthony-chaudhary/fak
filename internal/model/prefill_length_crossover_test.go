package model

import "testing"

// Witnesses for the length-driven prefill compute-placement switch (#5279): below the crossover ->
// host-hybrid, at or above -> device-layerwise, boundary honored exactly, degenerate inputs fail
// closed to host-hybrid. Pure, no device, no clock.

func TestChoosePrefillPlacementInjectedBoundaryExact(t *testing.T) {
	base := PrefillPlacementInputs{InjectedCrossover: 4096, Batch: 1}

	below := base
	below.PrefillTokens = 4095
	if got := ChoosePrefillPlacement(below); got.StageToDevice {
		t.Fatalf("below crossover: want host-hybrid (StageToDevice=false), got %+v", got)
	}

	at := base
	at.PrefillTokens = 4096
	got := ChoosePrefillPlacement(at)
	if !got.StageToDevice {
		t.Fatalf("at crossover: want device-layerwise (StageToDevice=true), got %+v", got)
	}
	if !got.FromInjected || got.CrossoverTokens != 4096 || !got.EverStages {
		t.Fatalf("at crossover: want injected crossover 4096, got %+v", got)
	}

	above := base
	above.PrefillTokens = 4097
	if got := ChoosePrefillPlacement(above); !got.StageToDevice {
		t.Fatalf("above crossover: want device-layerwise, got %+v", got)
	}
}

func TestChoosePrefillPlacementDerivedBreakEven(t *testing.T) {
	// slope = HostPerToken - DevicePerToken = 1; break-even = StagingOverhead/slope + 1 = 4001.
	in := PrefillPlacementInputs{
		HostPerToken:    2,
		DevicePerToken:  1,
		StagingOverhead: 4000,
		Batch:           1,
	}

	below := in
	below.PrefillTokens = 4000
	if got := ChoosePrefillPlacement(below); got.StageToDevice || got.FromInjected {
		t.Fatalf("below derived break-even: want host-hybrid, derived, got %+v", got)
	}

	at := in
	at.PrefillTokens = 4001
	got := ChoosePrefillPlacement(at)
	if !got.StageToDevice || got.CrossoverTokens != 4001 || got.FromInjected || !got.EverStages {
		t.Fatalf("at derived break-even: want device at 4001, derived, got %+v", got)
	}
}

func TestChoosePrefillPlacementNeverStagesFailsClosed(t *testing.T) {
	// Device per-token >= host per-token: streaming never pays for itself, no injected threshold.
	in := PrefillPlacementInputs{
		PrefillTokens:   1 << 20,
		HostPerToken:    1,
		DevicePerToken:  1,
		StagingOverhead: 10,
		Batch:           1,
	}
	got := ChoosePrefillPlacement(in)
	if got.StageToDevice || got.EverStages || got.CrossoverTokens != 0 {
		t.Fatalf("never-worthwhile: want host-hybrid pinned, EverStages=false, crossover=0, got %+v", got)
	}
}

func TestChoosePrefillPlacementDegenerateFailsClosed(t *testing.T) {
	// Zero-length prefill fails closed to host-hybrid even with a low crossover.
	zero := PrefillPlacementInputs{PrefillTokens: 0, InjectedCrossover: 1, Batch: 4}
	if got := ChoosePrefillPlacement(zero); got.StageToDevice || got.TotalTokens != 0 {
		t.Fatalf("zero prefill: want host-hybrid, TotalTokens=0, got %+v", got)
	}

	// Negative length fails closed too.
	neg := PrefillPlacementInputs{PrefillTokens: -5, InjectedCrossover: 1}
	if got := ChoosePrefillPlacement(neg); got.StageToDevice {
		t.Fatalf("negative prefill: want host-hybrid, got %+v", got)
	}

	// Zero costs with no injected threshold -> no derivable break-even -> host-hybrid.
	zc := PrefillPlacementInputs{PrefillTokens: 100000, HostPerToken: 0, DevicePerToken: 0}
	if got := ChoosePrefillPlacement(zc); got.StageToDevice || got.EverStages {
		t.Fatalf("zero costs: want host-hybrid pinned, got %+v", got)
	}
}

func TestChoosePrefillPlacementBatchScales(t *testing.T) {
	// 2048 tokens * 2 sequences = 4096 total, hits the injected crossover exactly.
	in := PrefillPlacementInputs{PrefillTokens: 2048, Batch: 2, InjectedCrossover: 4096}
	got := ChoosePrefillPlacement(in)
	if got.TotalTokens != 4096 || !got.StageToDevice {
		t.Fatalf("batch scaling: want TotalTokens=4096 staged to device, got %+v", got)
	}

	// Batch omitted (<=0) is treated as a single sequence, not zero work.
	single := PrefillPlacementInputs{PrefillTokens: 4096, Batch: 0, InjectedCrossover: 4096}
	if got := ChoosePrefillPlacement(single); got.TotalTokens != 4096 || !got.StageToDevice {
		t.Fatalf("omitted batch: want single-sequence TotalTokens=4096, got %+v", got)
	}
}

func TestChoosePrefillPlacementMonotone(t *testing.T) {
	base := PrefillPlacementInputs{InjectedCrossover: 512, Batch: 1}
	prevStaged := false
	for n := 1; n <= 4096; n++ {
		in := base
		in.PrefillTokens = n
		staged := ChoosePrefillPlacement(in).StageToDevice
		if prevStaged && !staged {
			t.Fatalf("non-monotone at n=%d: staging flipped back to host-hybrid", n)
		}
		prevStaged = staged
		if n == 511 && staged {
			t.Fatalf("n=511 below crossover 512 must be host-hybrid")
		}
		if n == 512 && !staged {
			t.Fatalf("n=512 at crossover 512 must be device-layerwise")
		}
	}
}
