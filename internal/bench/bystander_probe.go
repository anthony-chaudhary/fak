package bench

import (
	"context"
	"math"
	"sort"
	"sync"
	"time"
)

// BystanderProbeConfig configures probe-request dispatch during serve benchmarks (#10726).
type BystanderProbeConfig struct {
	Interval time.Duration `json:"interval"` // dispatch interval between single-token probes
	Timeout  time.Duration `json:"timeout"`  // per-probe timeout
}

// DefaultBystanderProbeConfig returns standard probe measurement defaults.
func DefaultBystanderProbeConfig() BystanderProbeConfig {
	return BystanderProbeConfig{
		Interval: 50 * time.Millisecond,
		Timeout:  2 * time.Second,
	}
}

// BystanderInterference captures single-token TTFT latency percentiles for concurrent bystander
// probe requests (#10726, borrowed from vLLM serve benchmark methodology).
// It directly isolates and measures scheduler head-of-line interference that heavy background
// prefill/batching traffic inflicts on unrelated interactive requests.
type BystanderInterference struct {
	Samples int     `json:"samples"`
	P50MS   float64 `json:"p50_ms"`
	P95MS   float64 `json:"p95_ms"`
	P99MS   float64 `json:"p99_ms"`
	MaxMS   float64 `json:"max_ms"`
	MeanMS  float64 `json:"mean_ms"`
}

// ProbeRunner executes periodic 1-token probe requests concurrently alongside a workload.
type ProbeRunner struct {
	cfg       BystanderProbeConfig
	mu        sync.Mutex
	latencies []time.Duration
	stopCh    chan struct{}
	doneCh    chan struct{}
}

// NewProbeRunner constructs a runner under the given config.
func NewProbeRunner(cfg BystanderProbeConfig) *ProbeRunner {
	if cfg.Interval <= 0 {
		cfg.Interval = 50 * time.Millisecond
	}
	if cfg.Timeout <= 0 {
		cfg.Timeout = 2 * time.Second
	}
	return &ProbeRunner{
		cfg:    cfg,
		stopCh: make(chan struct{}),
		doneCh: make(chan struct{}),
	}
}

// Start launches the probe worker in the background invoking probeFn periodically.
func (p *ProbeRunner) Start(probeFn func(ctx context.Context) (time.Duration, error)) {
	go func() {
		defer close(p.doneCh)
		ticker := time.NewTicker(p.cfg.Interval)
		defer ticker.Stop()

		for {
			select {
			case <-p.stopCh:
				return
			case <-ticker.C:
				ctx, cancel := context.WithTimeout(context.Background(), p.cfg.Timeout)
				lat, err := probeFn(ctx)
				cancel()
				if err == nil {
					p.mu.Lock()
					p.latencies = append(p.latencies, lat)
					p.mu.Unlock()
				}
			}
		}
	}()
}

// Stop stops the probe runner and returns the computed BystanderInterference metrics.
func (p *ProbeRunner) Stop() BystanderInterference {
	close(p.stopCh)
	<-p.doneCh
	return p.Compute()
}

// Compute calculates the BystanderInterference distribution from collected latencies.
func (p *ProbeRunner) Compute() BystanderInterference {
	p.mu.Lock()
	defer p.mu.Unlock()
	return ComputeBystanderInterference(p.latencies)
}

// ComputeBystanderInterference calculates exact percentiles from raw latency samples.
func ComputeBystanderInterference(raw []time.Duration) BystanderInterference {
	if len(raw) == 0 {
		return BystanderInterference{}
	}
	sorted := append([]time.Duration(nil), raw...)
	sort.Slice(sorted, func(i, j int) bool { return sorted[i] < sorted[j] })

	n := len(sorted)
	var sum time.Duration
	for _, d := range sorted {
		sum += d
	}

	percentile := func(pct float64) float64 {
		if n == 1 {
			return float64(sorted[0].Microseconds()) / 1000.0
		}
		idx := int(math.Round(pct * float64(n-1)))
		if idx < 0 {
			idx = 0
		}
		if idx >= n {
			idx = n - 1
		}
		return float64(sorted[idx].Microseconds()) / 1000.0
	}

	return BystanderInterference{
		Samples: n,
		P50MS:   percentile(0.50),
		P95MS:   percentile(0.95),
		P99MS:   percentile(0.99),
		MaxMS:   float64(sorted[n-1].Microseconds()) / 1000.0,
		MeanMS:  float64(sum.Microseconds()) / float64(n) / 1000.0,
	}
}
