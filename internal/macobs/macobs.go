package macobs

import (
	"context"
	"net/http"
	"os"
	"time"
)

// Option configures Collector behavior.
type Option func(*Collector)

// WithMetricsURL configures the Prometheus endpoint URL for MLX metrics.
func WithMetricsURL(url string) Option {
	return func(c *Collector) {
		c.metricsURL = url
	}
}

// WithHTTPClient overrides the HTTP client for scraping MLX metrics.
func WithHTTPClient(client *http.Client) Option {
	return func(c *Collector) {
		if client != nil {
			c.httpClient = client
		}
	}
}

// WithCommandRunner configures a custom command runner for hardware telemetry probing.
func WithCommandRunner(runner CommandRunner) Option {
	return func(c *Collector) {
		if runner != nil {
			c.runner = runner
		}
	}
}

// WithHeadroomConfig configures the model architecture and context assumptions for headroom modeling.
func WithHeadroomConfig(cfg HeadroomConfig) Option {
	return func(c *Collector) {
		c.headroomConfig = cfg
	}
}

// WithRequestedAgents sets the target concurrency to evaluate in diagnosis.
func WithRequestedAgents(n int) Option {
	return func(c *Collector) {
		if n > 0 {
			c.requestedAgents = n
		}
	}
}

// Collector coordinates Mac hardware, MLX serving, and headroom observations.
type Collector struct {
	metricsURL      string
	httpClient      *http.Client
	runner          CommandRunner
	headroomConfig  HeadroomConfig
	requestedAgents int
}

// NewCollector instantiates a Collector with sensible defaults and optional overrides.
func NewCollector(opts ...Option) *Collector {
	c := &Collector{
		metricsURL:      os.Getenv("FAK_MLX_METRICS_URL"),
		httpClient:      &http.Client{Timeout: 3 * time.Second},
		runner:          DefaultCommandRunner,
		headroomConfig:  DefaultHeadroomConfig(),
		requestedAgents: 1,
	}
	for _, opt := range opts {
		opt(c)
	}
	return c
}

// Observe samples Mac hardware, MLX runtime metrics, and produces an integrated Snapshot.
func (c *Collector) Observe(ctx context.Context) (Snapshot, error) {
	// 1. Collect hardware counters
	hw := CollectHardwareWithRunner(ctx, c.runner)

	// 2. Scrape serving metrics if endpoint configured
	var srv MLXServingTelemetry
	var prefix PrefixCacheTelemetry
	if c.metricsURL != "" {
		s, p, err := ScrapeMLXMetrics(ctx, c.metricsURL, c.httpClient)
		if err == nil {
			srv = s
			prefix = p
		}
	}

	// 3. Compute headroom under hardware limits
	head := ComputeHeadroom(hw, c.headroomConfig)

	// 4. Diagnose bottlenecks and synthesize action verdict
	diag := Diagnose(hw, srv, head, c.requestedAgents)

	// 5. Determine data provenance honestly
	provenance := ProvenanceUnavailable
	if hw.Available {
		provenance = ProvenanceWitnessed
	} else if head.Available {
		provenance = ProvenanceModeled
	}

	return Snapshot{
		Schema:      SchemaV1,
		Timestamp:   time.Now().UTC(),
		Provenance:  provenance,
		Hardware:    hw,
		MLXServing:  srv,
		Headroom:    head,
		PrefixCache: prefix,
		Analysis:    diag,
	}, nil
}
