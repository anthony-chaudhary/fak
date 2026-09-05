package harnessres

import (
	"encoding/json"
	"sync"
	"testing"
	"time"
)

// TestGovernorDynamicWaveExpansion tests expansion under low memory pressure
// up to 40-80+ seats on standard developer workstations (16GB, 32GB, 64GB).
func TestGovernorDynamicWaveExpansion(t *testing.T) {
	t.Run("64GB workstation ramps up to max capacity 80 seats", func(t *testing.T) {
		cfg := DefaultGovernorConfig()
		cfg.MaxWorkers = 80
		cfg.MinWorkers = 2
		gov := NewDensityGovernor(cfg)

		if got := gov.CurrentConcurrency(); got != 2 {
			t.Fatalf("expected initial concurrency 2, got %d", got)
		}

		// 64 GiB total, 50 GiB available (~78% free RAM)
		sample := HostSample{
			TotalRAMBytes: 64 * 1024 * 1024 * 1024,
			AvailRAMBytes: 50 * 1024 * 1024 * 1024,
			HaveRAM:       true,
			HavePSI:       true,
			PSIAvg10:      0.0,
			Timestamp:     time.Now(),
		}

		// Successive updates should monotonically ramp up towards 80
		prev := gov.CurrentConcurrency()
		for i := 0; i < 20; i++ {
			c := gov.Update(sample)
			if c < prev {
				t.Fatalf("step %d: concurrency decreased from %d to %d under low pressure", i, prev, c)
			}
			admit, reason := gov.ShouldAdmit()
			if !admit {
				t.Fatalf("step %d: ShouldAdmit refused under low pressure: %s", i, reason)
			}
			prev = c
		}

		if prev != 80 {
			t.Fatalf("expected concurrency to ramp up to MaxWorkers 80, got %d", prev)
		}
	})

	t.Run("32GB workstation scales safely to proportional capacity", func(t *testing.T) {
		cfg := DefaultGovernorConfig()
		cfg.MaxWorkers = 80
		cfg.MinWorkers = 2
		gov := NewDensityGovernor(cfg)

		// 32 GiB total, 24 GiB available (75% free RAM)
		// Target free = 25% (8 GiB). Surplus = 16 GiB.
		// 16 GiB / 350 MiB ~= 46 seats.
		sample := HostSample{
			TotalRAMBytes: 32 * 1024 * 1024 * 1024,
			AvailRAMBytes: 24 * 1024 * 1024 * 1024,
			HaveRAM:       true,
			HavePSI:       true,
			PSIAvg10:      0.0,
			Timestamp:     time.Now(),
		}

		for i := 0; i < 25; i++ {
			gov.Update(sample)
		}

		c := gov.CurrentConcurrency()
		if c < 40 || c > 55 {
			t.Fatalf("expected concurrency between 40 and 55 on 32GB workstation, got %d", c)
		}
	})

	t.Run("16GB workstation scales within conservative bounds", func(t *testing.T) {
		cfg := DefaultGovernorConfig()
		cfg.MaxWorkers = 80
		cfg.MinWorkers = 2
		gov := NewDensityGovernor(cfg)

		// 16 GiB total, 10 GiB available (62.5% free RAM)
		// Target free = 25% (4 GiB). Surplus = 6 GiB.
		// 6 GiB / 350 MiB ~= 17 seats + 2 min = ~19 seats.
		sample := HostSample{
			TotalRAMBytes: 16 * 1024 * 1024 * 1024,
			AvailRAMBytes: 10 * 1024 * 1024 * 1024,
			HaveRAM:       true,
			Timestamp:     time.Now(),
		}

		for i := 0; i < 20; i++ {
			gov.Update(sample)
		}

		c := gov.CurrentConcurrency()
		if c < 15 || c > 25 {
			t.Fatalf("expected concurrency between 15 and 25 on 16GB workstation, got %d", c)
		}
	})
}

// TestGovernorRapidBackoff tests rapid back-off circuit under memory squeeze (<15%),
// high PSI stall rate, PSI stall duration delta, and swap surge.
func TestGovernorRapidBackoff(t *testing.T) {
	t.Run("RAM squeeze below 15% throttles and cuts concurrency", func(t *testing.T) {
		cfg := DefaultGovernorConfig()
		cfg.MaxWorkers = 80
		gov := NewDensityGovernor(cfg)

		// Expand first to 80
		healthy := HostSample{
			TotalRAMBytes: 64 * 1024 * 1024 * 1024,
			AvailRAMBytes: 50 * 1024 * 1024 * 1024,
			HaveRAM:       true,
			Timestamp:     time.Now(),
		}
		for i := 0; i < 20; i++ {
			gov.Update(healthy)
		}
		if gov.CurrentConcurrency() != 80 {
			t.Fatalf("setup failed: expected 80, got %d", gov.CurrentConcurrency())
		}

		// Squeeze: Available RAM drops to 10% (< 15% threshold)
		squeezed := HostSample{
			TotalRAMBytes: 64 * 1024 * 1024 * 1024,
			AvailRAMBytes: (64 * 1024 * 1024 * 1024) / 10,
			HaveRAM:       true,
			Timestamp:     time.Now(),
		}

		newConc := gov.Update(squeezed)
		// Multiplicative decrease cuts 80 -> 40
		if newConc > 45 {
			t.Fatalf("expected rapid backoff concurrency <= 45, got %d", newConc)
		}

		admit, reason := gov.ShouldAdmit()
		if admit {
			t.Fatalf("expected ShouldAdmit = false during memory squeeze, got true")
		}
		if reason == "" {
			t.Fatalf("expected non-empty reason for ShouldAdmit refusal")
		}

		// Repeated squeezed samples drive concurrency down to floor
		for i := 0; i < 10; i++ {
			gov.Update(squeezed)
		}
		if got := gov.CurrentConcurrency(); got != 2 {
			t.Fatalf("expected concurrency to reach MinWorkers 2 under sustained squeeze, got %d", got)
		}

		// Recovery: RAM returns to 60%
		recovered := HostSample{
			TotalRAMBytes: 64 * 1024 * 1024 * 1024,
			AvailRAMBytes: (64 * 1024 * 1024 * 1024 * 6) / 10,
			HaveRAM:       true,
			Timestamp:     time.Now(),
		}
		gov.Update(recovered)
		admitAfter, reasonAfter := gov.ShouldAdmit()
		if !admitAfter {
			t.Fatalf("expected ShouldAdmit = true after recovery, got false (%s)", reasonAfter)
		}
	})

	t.Run("PSI avg10 stall surge triggers backoff", func(t *testing.T) {
		cfg := DefaultGovernorConfig()
		cfg.MaxWorkers = 80
		cfg.PSITriggerThreshold = 10.0
		gov := NewDensityGovernor(cfg)

		// Expand to 40
		healthy := HostSample{
			TotalRAMBytes: 64 * 1024 * 1024 * 1024,
			AvailRAMBytes: 40 * 1024 * 1024 * 1024,
			HaveRAM:       true,
			HavePSI:       true,
			PSIAvg10:      0.5,
			Timestamp:     time.Now(),
		}
		for i := 0; i < 15; i++ {
			gov.Update(healthy)
		}
		startConc := gov.CurrentConcurrency()
		if startConc < 30 {
			t.Fatalf("expected start concurrency >= 30, got %d", startConc)
		}

		// PSI stall surge: avg10 = 22.5%
		psiSurg := healthy
		psiSurg.PSIAvg10 = 22.5
		newConc := gov.Update(psiSurg)

		if newConc >= startConc {
			t.Fatalf("expected concurrency to drop on PSI surge, got %d -> %d", startConc, newConc)
		}
		admit, reason := gov.ShouldAdmit()
		if admit {
			t.Fatalf("expected ShouldAdmit = false on PSI surge")
		}
		if reason == "" {
			t.Fatalf("expected non-empty refusal reason")
		}
	})

	t.Run("PSI total stall duration delta triggers backoff", func(t *testing.T) {
		cfg := DefaultGovernorConfig()
		cfg.PSITriggerDuration = 80 * time.Millisecond
		gov := NewDensityGovernor(cfg)

		base := HostSample{
			TotalRAMBytes: 64 * 1024 * 1024 * 1024,
			AvailRAMBytes: 40 * 1024 * 1024 * 1024,
			HaveRAM:       true,
			HavePSI:       true,
			PSISomeTotal:  100 * time.Millisecond,
			Timestamp:     time.Now(),
		}
		gov.Update(base)

		// Next sample: PSISomeTotal jumped by 150ms (> 80ms trigger)
		spike := base
		spike.PSISomeTotal = 250 * time.Millisecond
		gov.Update(spike)

		admit, reason := gov.ShouldAdmit()
		if admit {
			t.Fatalf("expected backoff from PSI stall delta spike")
		}
		if reason == "" {
			t.Fatalf("expected non-empty refusal reason")
		}
	})

	t.Run("Swap activity surge triggers backoff", func(t *testing.T) {
		cfg := DefaultGovernorConfig()
		gov := NewDensityGovernor(cfg)

		initial := HostSample{
			TotalRAMBytes:  64 * 1024 * 1024 * 1024,
			AvailRAMBytes:  40 * 1024 * 1024 * 1024,
			HaveRAM:        true,
			TotalSwapBytes: 16 * 1024 * 1024 * 1024,
			AvailSwapBytes: 15 * 1024 * 1024 * 1024, // 1 GiB used
			HaveSwap:       true,
			Timestamp:      time.Now(),
		}
		gov.Update(initial)

		// Surge: Swap used increases by 2 GiB (AvailSwap drops to 13 GiB)
		surged := initial
		surged.AvailSwapBytes = 13 * 1024 * 1024 * 1024
		gov.Update(surged)

		admit, reason := gov.ShouldAdmit()
		if admit {
			t.Fatalf("expected backoff on swap surge")
		}
		if reason == "" {
			t.Fatalf("expected non-empty reason")
		}
	})
}

// TestGovernorFailClosed proves the governor fails closed to conservative concurrency
// floor (min 2 workers) whenever metrics are missing or corrupted.
func TestGovernorFailClosed(t *testing.T) {
	t.Run("missing RAM metrics", func(t *testing.T) {
		gov := NewDensityGovernor(DefaultGovernorConfig())
		// First expand
		gov.Update(HostSample{
			TotalRAMBytes: 32 * 1024 * 1024 * 1024,
			AvailRAMBytes: 24 * 1024 * 1024 * 1024,
			HaveRAM:       true,
		})

		// Now send missing RAM
		c := gov.Update(HostSample{HaveRAM: false})
		if c != 2 {
			t.Fatalf("expected fail closed to floor 2 on missing RAM, got %d", c)
		}
		admit, _ := gov.ShouldAdmit()
		if admit {
			t.Fatalf("expected ShouldAdmit = false when metrics missing")
		}
	})

	t.Run("corrupted RAM metrics (available > total)", func(t *testing.T) {
		gov := NewDensityGovernor(DefaultGovernorConfig())
		c := gov.Update(HostSample{
			TotalRAMBytes: 16 * 1024 * 1024 * 1024,
			AvailRAMBytes: 32 * 1024 * 1024 * 1024, // impossible
			HaveRAM:       true,
		})
		if c != 2 {
			t.Fatalf("expected fail closed on corrupted RAM, got %d", c)
		}
		admit, _ := gov.ShouldAdmit()
		if admit {
			t.Fatalf("expected ShouldAdmit = false on corrupted RAM")
		}
	})

	t.Run("explicitly marked corrupted sample", func(t *testing.T) {
		gov := NewDensityGovernor(DefaultGovernorConfig())
		c := gov.Update(HostSample{
			TotalRAMBytes: 32 * 1024 * 1024 * 1024,
			AvailRAMBytes: 20 * 1024 * 1024 * 1024,
			HaveRAM:       true,
			Corrupted:     true,
		})
		if c != 2 {
			t.Fatalf("expected fail closed on marked corrupted sample, got %d", c)
		}
	})

	t.Run("missing PSI when RequirePSI is true", func(t *testing.T) {
		cfg := DefaultGovernorConfig()
		cfg.RequirePSI = true
		gov := NewDensityGovernor(cfg)

		c := gov.Update(HostSample{
			TotalRAMBytes: 32 * 1024 * 1024 * 1024,
			AvailRAMBytes: 25 * 1024 * 1024 * 1024,
			HaveRAM:       true,
			HavePSI:       false, // missing PSI
		})
		if c != 2 {
			t.Fatalf("expected fail closed to floor 2 when RequirePSI is true and PSI is missing, got %d", c)
		}
		admit, _ := gov.ShouldAdmit()
		if admit {
			t.Fatalf("expected ShouldAdmit = false when RequirePSI is unsatisfied")
		}
	})
}

// TestGovernorArgumentResolution tests ResolveConcurrency for "auto", positive integers,
// and invalid arguments.
func TestGovernorArgumentResolution(t *testing.T) {
	gov := NewDensityGovernor(DefaultGovernorConfig())
	gov.Update(HostSample{
		TotalRAMBytes: 64 * 1024 * 1024 * 1024,
		AvailRAMBytes: 48 * 1024 * 1024 * 1024,
		HaveRAM:       true,
	})

	current := gov.CurrentConcurrency()

	// "auto" returns dynamic limit
	if got := gov.ResolveConcurrency("auto"); got != current {
		t.Fatalf("expected ResolveConcurrency('auto') = %d, got %d", current, got)
	}
	if got := gov.ResolveConcurrency("AUTO"); got != current {
		t.Fatalf("expected ResolveConcurrency('AUTO') = %d, got %d", current, got)
	}
	if got := gov.ResolveConcurrency("dynamic"); got != current {
		t.Fatalf("expected ResolveConcurrency('dynamic') = %d, got %d", current, got)
	}
	if got := gov.ResolveConcurrency(""); got != current {
		t.Fatalf("expected ResolveConcurrency('') = %d, got %d", current, got)
	}

	// Positive integers parsed
	if got := gov.ResolveConcurrency("4"); got != 4 {
		t.Fatalf("expected ResolveConcurrency('4') = 4, got %d", got)
	}
	if got := gov.ResolveConcurrency("64"); got != 64 {
		t.Fatalf("expected ResolveConcurrency('64') = 64, got %d", got)
	}

	// Invalid inputs fallback to conservative floor
	if got := gov.ResolveConcurrency("0"); got != 2 {
		t.Fatalf("expected ResolveConcurrency('0') = 2, got %d", got)
	}
	if got := gov.ResolveConcurrency("-10"); got != 2 {
		t.Fatalf("expected ResolveConcurrency('-10') = 2, got %d", got)
	}
	if got := gov.ResolveConcurrency("banana"); got != 2 {
		t.Fatalf("expected ResolveConcurrency('banana') = 2, got %d", got)
	}

	// Package-level ResolveConcurrency
	pkgAuto := ResolveConcurrency("auto")
	if pkgAuto < 2 {
		t.Fatalf("expected package-level ResolveConcurrency('auto') >= 2, got %d", pkgAuto)
	}
	if got := ResolveConcurrency("12"); got != 12 {
		t.Fatalf("expected package-level ResolveConcurrency('12') = 12, got %d", got)
	}
	if got := ResolveConcurrency("invalid"); got != 2 {
		t.Fatalf("expected package-level ResolveConcurrency('invalid') = 2, got %d", got)
	}
}

// TestGovernorConcurrentSafety exercises the governor under high goroutine concurrency.
func TestGovernorConcurrentSafety(t *testing.T) {
	gov := NewDensityGovernor(DefaultGovernorConfig())

	const numGoroutines = 24
	const iterations = 150

	var wg sync.WaitGroup
	wg.Add(numGoroutines)

	for g := 0; g < numGoroutines; g++ {
		go func(id int) {
			defer wg.Done()
			for i := 0; i < iterations; i++ {
				switch id % 5 {
				case 0:
					gov.Update(HostSample{
						TotalRAMBytes: 32 * 1024 * 1024 * 1024,
						AvailRAMBytes: uint64(10+((i+id)%18)) * 1024 * 1024 * 1024,
						HaveRAM:       true,
						HavePSI:       true,
						PSIAvg10:      float64((i + id) % 15),
						Timestamp:     time.Now(),
					})
				case 1:
					_ = gov.CurrentConcurrency()
				case 2:
					_, _ = gov.ShouldAdmit()
				case 3:
					_ = gov.ResolveConcurrency("auto")
				case 4:
					_ = gov.Telemetry()
				}
			}
		}(g)
	}

	wg.Wait()

	// Verify state remains consistent and valid after high contention
	c := gov.CurrentConcurrency()
	if c < 2 || c > 80 {
		t.Fatalf("invalid final concurrency after concurrent test: %d", c)
	}
}

// TestGovernorTelemetryAndHostReading validates telemetry structure and live ReadHostSample.
func TestGovernorTelemetryAndHostReading(t *testing.T) {
	gov := NewDensityGovernor(DefaultGovernorConfig())
	sample := ReadHostSample()

	// On Windows, sample should have valid RAM
	t.Logf("ReadHostSample result: haveRAM=%v totalRAM=%d availRAM=%d havePSI=%v",
		sample.HaveRAM, sample.TotalRAMBytes, sample.AvailRAMBytes, sample.HavePSI)

	gov.Update(sample)
	tel := gov.Telemetry()

	b, err := json.Marshal(tel)
	if err != nil {
		t.Fatalf("telemetry JSON marshal failed: %v", err)
	}
	if len(b) == 0 {
		t.Fatalf("empty telemetry JSON")
	}

	var parsed map[string]any
	if err := json.Unmarshal(b, &parsed); err != nil {
		t.Fatalf("telemetry JSON unmarshal failed: %v", err)
	}

	if _, ok := parsed["concurrency"]; !ok {
		t.Fatalf("expected 'concurrency' field in telemetry JSON")
	}
	if _, ok := parsed["admit_allowed"]; !ok {
		t.Fatalf("expected 'admit_allowed' field in telemetry JSON")
	}
}

// TestGovernorUpdateHost verifies UpdateHost adapter.
func TestGovernorUpdateHost(t *testing.T) {
	gov := NewDensityGovernor(DefaultGovernorConfig())
	h := Host{
		TotalRAMBytes: 32 * 1024 * 1024 * 1024,
		AvailRAMBytes: 20 * 1024 * 1024 * 1024,
		HaveRAM:       true,
	}

	c := gov.UpdateHost(h)
	if c < 2 {
		t.Fatalf("expected concurrency >= 2, got %d", c)
	}
}
