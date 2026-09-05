package loaddebounce

import (
	"fmt"
	"testing"
	"time"
)

var (
	benchArmedSink bool
	benchValSink   int
	benchTimeSink  time.Time
)

func BenchmarkCoalescer_Observe(b *testing.B) {
	now := time.Date(2026, 9, 1, 12, 0, 0, 0, time.UTC)

	b.Run("DedupSteadyState", func(b *testing.B) {
		c := New[int](DefaultDebounce)
		c.Prime(42)

		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchArmedSink = c.Observe(42, now)
		}
	})

	b.Run("BurstChanging", func(b *testing.B) {
		c := New[int](DefaultDebounce)
		c.Prime(0)

		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			// Each sample is distinct, resetting the deadline.
			benchArmedSink = c.Observe(i+1, now.Add(time.Duration(i)*time.Microsecond))
		}
	})

	b.Run("RepeatingPending", func(b *testing.B) {
		c := New[int](DefaultDebounce)
		c.Prime(0)
		c.Observe(99, now)

		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			// Already waiting out the window for 99; maintains current deadline.
			benchArmedSink = c.Observe(99, now)
		}
	})

	b.Run("CycleCancellation", func(b *testing.B) {
		c := New[int](DefaultDebounce)
		c.Prime(10)

		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			// Oscillate between a changed value and the published baseline.
			if i%2 == 0 {
				benchArmedSink = c.Observe(20, now)
			} else {
				benchArmedSink = c.Observe(10, now)
			}
		}
	})
}

func BenchmarkCoalescer_Emit(b *testing.B) {
	now := time.Date(2026, 9, 1, 12, 0, 0, 0, time.UTC)

	b.Run("NotDue", func(b *testing.B) {
		c := New[int](DefaultDebounce)
		c.Observe(1, now)

		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			val, _ := c.Emit(now)
			benchValSink = val
		}
	})

	b.Run("DueAndEmitted", func(b *testing.B) {
		c := New[int](DefaultDebounce)
		c.Prime(0)
		emitTime := now.Add(2 * DefaultDebounce)

		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			c.Observe(i+1, now)
			val, _ := c.Emit(emitTime)
			benchValSink = val
		}
	})
}

func BenchmarkPublisher_Sample(b *testing.B) {
	clk := newFakeClock()
	var publishedVal int
	emitFn := func(v int) { publishedVal = v }

	b.Run("SteadyStateDedup", func(b *testing.B) {
		p := NewPublisher[int](DefaultDebounce, clk.now, emitFn)
		p.Prime(16)

		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			p.Sample(16)
		}
		benchValSink = publishedVal
	})

	b.Run("SettledLifecycle", func(b *testing.B) {
		p := NewPublisher[int](DefaultDebounce, clk.now, emitFn)
		p.Prime(0)

		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			p.Sample(i + 1)
			clk.advance(2 * DefaultDebounce)
			p.Flush()
		}
		benchValSink = publishedVal
	})

	b.Run("RapidBurstCoalesce", func(b *testing.B) {
		p := NewPublisher[int](DefaultDebounce, clk.now, emitFn)
		p.Prime(0)

		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			for step := 1; step <= 5; step++ {
				p.Sample(i*10 + step)
				clk.advance(100 * time.Microsecond)
			}
			clk.advance(2 * DefaultDebounce)
			p.Flush()
		}
		benchValSink = publishedVal
	})
}

func BenchmarkDispatchTickSimulation(b *testing.B) {
	// Simulates production usage in cmd/fak/dispatch_tick_load_debounce.go:
	// Ticks fire periodically; 95% of ticks sample steady load, 5% observe
	// temporary fluctuation bursts that coalesce.
	clk := newFakeClock()
	var publishedCount int
	emitFn := func(v int) { publishedCount++ }

	const tickCount = 100
	samples := make([]int, tickCount)
	for t := 0; t < tickCount; t++ {
		if t >= 40 && t < 45 {
			samples[t] = 8 + (t - 40) // burst up
		} else if t >= 45 && t < 50 {
			samples[t] = 12 // settle at 12
		} else {
			samples[t] = 8 // steady state
		}
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		p := NewPublisher[int](DefaultDebounce, clk.now, emitFn)
		p.Prime(samples[0])

		for _, s := range samples[1:] {
			clk.advance(500 * time.Millisecond) // tick interval
			p.Sample(s)
			p.Flush()
		}
	}
	benchValSink = publishedCount
}

func BenchmarkFleetScale(b *testing.B) {
	workerCounts := []int{10, 50, 200}
	for _, count := range workerCounts {
		b.Run(fmt.Sprintf("%d_workers", count), func(b *testing.B) {
			clk := newFakeClock()
			publishers := make([]*Publisher[int], count)
			for w := 0; w < count; w++ {
				publishers[w] = NewPublisher[int](DefaultDebounce, clk.now, func(v int) {
					benchValSink = v
				})
				publishers[w].Prime(w)
			}

			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				for w := 0; w < count; w++ {
					// Mostly steady state with occasional load shift
					val := w
					if (i+w)%10 == 0 {
						val = w + 1
					}
					publishers[w].Sample(val)
				}
				clk.advance(2 * DefaultDebounce)
				for w := 0; w < count; w++ {
					publishers[w].Flush()
				}
			}
		})
	}
}

func BenchmarkPublisher_Parallel(b *testing.B) {
	now := time.Date(2026, 9, 1, 12, 0, 0, 0, time.UTC)

	b.Run("ParallelIndependentWorkers", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		b.RunParallel(func(pb *testing.PB) {
			p := NewPublisher[int](DefaultDebounce, func() time.Time { return now }, func(v int) {
				benchValSink = v
			})
			p.Prime(1)
			idx := 0
			for pb.Next() {
				// Steady state interspersed with changes
				if idx%8 == 0 {
					p.Sample(idx)
				} else {
					p.Sample(1)
				}
				idx++
			}
		})
	})
}
