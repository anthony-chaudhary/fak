package compute

import (
	"context"
	"errors"
	"math"
	"testing"
)

// fak-test:runtime fast est=1s
func TestWatchdogPrefillRejectsNonfiniteConfiguration(t *testing.T) {
	fields := []struct {
		name string
		set  func(*ChunkingProfile, *ModelGeometry, *APUHardwareProfile, float64)
		want error
	}{
		{"ceiling", func(p *ChunkingProfile, _ *ModelGeometry, _ *APUHardwareProfile, v float64) {
			p.MaxExecutionCeilingMs = v
		}, ErrInvalidChunkingProfile},
		{"watchdog", func(p *ChunkingProfile, _ *ModelGeometry, _ *APUHardwareProfile, v float64) { p.WatchdogTimeoutMs = v }, ErrInvalidChunkingProfile},
		{"margin", func(p *ChunkingProfile, _ *ModelGeometry, _ *APUHardwareProfile, v float64) { p.SafetyMarginRatio = v }, ErrInvalidChunkingProfile},
		{"weight_bytes", func(_ *ChunkingProfile, g *ModelGeometry, _ *APUHardwareProfile, v float64) {
			g.WeightBytesPerParam = v
		}, ErrInvalidModelGeometry},
		{"compute_rate", func(_ *ChunkingProfile, _ *ModelGeometry, h *APUHardwareProfile, v float64) { h.PeakTFLOPS = v }, ErrInvalidHardwareProfile},
		{"compute_efficiency", func(_ *ChunkingProfile, _ *ModelGeometry, h *APUHardwareProfile, v float64) {
			h.SustainedComputeEff = v
		}, ErrInvalidHardwareProfile},
		{"memory_rate", func(_ *ChunkingProfile, _ *ModelGeometry, h *APUHardwareProfile, v float64) {
			h.TheoreticalPeakBWGBps = v
		}, ErrInvalidHardwareProfile},
		{"memory_efficiency", func(_ *ChunkingProfile, _ *ModelGeometry, h *APUHardwareProfile, v float64) { h.SustainedMemoryEff = v }, ErrInvalidHardwareProfile},
		{"launch_overhead", func(_ *ChunkingProfile, _ *ModelGeometry, h *APUHardwareProfile, v float64) {
			h.KernelLaunchOverheadMs = v
		}, ErrInvalidHardwareProfile},
	}
	for _, field := range fields {
		for _, value := range []struct {
			name  string
			value float64
		}{{"nan", math.NaN()}, {"positive_inf", math.Inf(1)}, {"negative_inf", math.Inf(-1)}} {
			t.Run(field.name+"/"+value.name, func(t *testing.T) {
				p, g, h := DefaultSingleSequenceProfile(), ModelGeometry27B(), DefaultAPUHardwareProfile()
				field.set(&p, &g, &h, value.value)
				scheduler, err := NewWatchdogPrefillScheduler(p, g, h)
				if scheduler != nil || !errors.Is(err, field.want) {
					t.Fatalf("invalid configuration admitted: scheduler=%v error=%v want=%v", scheduler, err, field.want)
				}
			})
		}
	}
}

// fak-test:runtime fast est=1s
func TestWatchdogPrefillFiniteDefaultExecutesActualSchedule(t *testing.T) {
	p, g, h := DefaultSingleSequenceProfile(), ModelGeometry27B(), DefaultAPUHardwareProfile()
	// Zero launch overhead is a valid finite boundary and remains admitted.
	h.KernelLaunchOverheadMs = 0
	scheduler, err := NewWatchdogPrefillScheduler(p, g, h)
	if err != nil {
		t.Fatal(err)
	}
	schedule, err := scheduler.PlanSchedule(1)
	if err != nil {
		t.Fatal(err)
	}
	if !schedule.IsWatchdogSafe || math.IsNaN(schedule.MaxChunkDurationMs) || math.IsInf(schedule.MaxChunkDurationMs, 0) {
		t.Fatalf("invalid default schedule: %+v", schedule)
	}
	calls, tokens := 0, 0
	receipt, err := scheduler.Execute(context.Background(), schedule, func(_ context.Context, c PrefillChunk) error { calls++; tokens += c.TokenCount; return nil }, nil)
	if err != nil || receipt == nil || !receipt.CompletedClean {
		t.Fatalf("execute receipt=%+v error=%v", receipt, err)
	}
	if calls != len(schedule.Chunks) || tokens != schedule.TotalTokens || receipt.ChunksExecuted != calls {
		t.Fatal("actual callback coverage differs from planned schedule")
	}
}
