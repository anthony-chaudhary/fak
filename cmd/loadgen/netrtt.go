// Adapted from ai-dynamo/aiperf
// modified from the original): pre-flight unpooled TCP/RTT calibration probe
// (DNS resolved once at construction, no TLS handshake even for https) plus
// non-destructive network_adjusted_* latency computation clamped >= 0 with
// ITL explicitly exempted because RTT cancels in (request_latency - ttft).
package main

import (
	"context"
	"errors"
	"fmt"
	"math"
	"net"
	"net/url"
	"sort"
	"time"
)

const (
	defaultNetRTTProbes  = 5
	defaultNetRTTTimeout = 2 * time.Second
)

var (
	errMissingNetRTTBaseline = errors.New("loadgen: missing network RTT baseline (zero successful probes)")
	errNonPositiveNetRTT     = errors.New("loadgen: network RTT baseline must be positive and finite")
)

// NetRTTProbeOptions customizes resolution, dialing, and timing hooks for
// pre-flight network RTT calibration. Zero values use stdlib defaults.
type NetRTTProbeOptions struct {
	Probes  int
	Timeout time.Duration
	Resolve func(ctx context.Context, host string) ([]net.IPAddr, error)
	Dial    func(ctx context.Context, network, addr string) (net.Conn, error)
	Now     func() time.Time
}

// NetRTTReport summarizes pre-flight TCP-connect calibration probes.
type NetRTTReport struct {
	TargetURL    string        `json:"target_url"`
	ResolvedAddr string        `json:"resolved_addr"`
	Probes       int           `json:"probes"`
	Samples      int           `json:"samples"`
	Errors       int           `json:"errors"`
	MeanRTT      time.Duration `json:"-"`
	MeanRTTMS    float64       `json:"network_rtt_ms"`
	MinRTTMS     float64       `json:"min_rtt_ms,omitempty"`
	MaxRTTMS     float64       `json:"max_rtt_ms,omitempty"`
}

// NetRTTProber performs fresh unpooled TCP-connect probes against an address
// resolved once at construction so probes measure network connect RTT rather
// than repeated DNS resolution or TLS negotiation.
type NetRTTProber struct {
	targetURL    string
	resolvedAddr string
	probes       int
	timeout      time.Duration
	dial         func(ctx context.Context, network, addr string) (net.Conn, error)
	now          func() time.Time
}

// NewNetRTTProber parses rawURL and resolves the target host once.
func NewNetRTTProber(ctx context.Context, rawURL string, opts NetRTTProbeOptions) (*NetRTTProber, error) {
	u, err := url.Parse(rawURL)
	if err != nil || u.Scheme == "" || u.Hostname() == "" {
		return nil, fmt.Errorf("loadgen: invalid probe target URL %q", rawURL)
	}
	port := u.Port()
	if port == "" {
		switch u.Scheme {
		case "https":
			port = "443"
		case "http":
			port = "80"
		default:
			return nil, fmt.Errorf("loadgen: probe URL %q needs port or http/https scheme", rawURL)
		}
	}

	resolveFn := opts.Resolve
	if resolveFn == nil {
		resolveFn = net.DefaultResolver.LookupIPAddr
	}
	ips, err := resolveFn(ctx, u.Hostname())
	if err != nil || len(ips) == 0 {
		return nil, fmt.Errorf("loadgen: resolve %q: %w", u.Hostname(), err)
	}

	probes := opts.Probes
	if probes <= 0 {
		probes = defaultNetRTTProbes
	}
	timeout := opts.Timeout
	if timeout <= 0 {
		timeout = defaultNetRTTTimeout
	}
	dialFn := opts.Dial
	if dialFn == nil {
		d := &net.Dialer{Timeout: timeout}
		dialFn = d.DialContext
	}
	nowFn := opts.Now
	if nowFn == nil {
		nowFn = time.Now
	}

	return &NetRTTProber{
		targetURL:    rawURL,
		resolvedAddr: net.JoinHostPort(ips[0].IP.String(), port),
		probes:       probes,
		timeout:      timeout,
		dial:         dialFn,
		now:          nowFn,
	}, nil
}

// ResolvedAddr returns the host:port resolved once at construction.
func (p *NetRTTProber) ResolvedAddr() string {
	if p == nil {
		return ""
	}
	return p.resolvedAddr
}

// Probe executes p.probes fresh unpooled TCP connections against resolvedAddr.
// Individual probe failures are counted in Errors and never abort the run.
func (p *NetRTTProber) Probe(ctx context.Context) NetRTTReport {
	rep := NetRTTReport{
		TargetURL:    p.targetURL,
		ResolvedAddr: p.resolvedAddr,
		Probes:       p.probes,
	}
	if p == nil {
		return rep
	}

	var total time.Duration
	var minRTT, maxRTT time.Duration
	for i := 0; i < p.probes; i++ {
		if ctx.Err() != nil {
			rep.Errors += p.probes - i
			break
		}
		probeCtx, cancel := context.WithTimeout(ctx, p.timeout)
		start := p.now()
		conn, err := p.dial(probeCtx, "tcp", p.resolvedAddr)
		elapsed := p.now().Sub(start)
		cancel()
		if err != nil {
			rep.Errors++
			continue
		}
		_ = conn.Close()
		if elapsed <= 0 || math.IsNaN(float64(elapsed)) {
			rep.Errors++
			continue
		}
		rep.Samples++
		total += elapsed
		if rep.Samples == 1 || elapsed < minRTT {
			minRTT = elapsed
		}
		if elapsed > maxRTT {
			maxRTT = elapsed
		}
	}

	if rep.Samples > 0 {
		rep.MeanRTT = total / time.Duration(rep.Samples)
		rep.MeanRTTMS = float64(total.Nanoseconds()) / 1e6 / float64(rep.Samples)
		rep.MinRTTMS = float64(minRTT.Nanoseconds()) / 1e6
		rep.MaxRTTMS = float64(maxRTT.Nanoseconds()) / 1e6
	}
	return rep
}

// ValidateNetRTTBaseline checks that baseline contains at least one successful
// probe and a strictly positive, finite mean RTT.
func ValidateNetRTTBaseline(baseline NetRTTReport) error {
	if baseline.Samples <= 0 {
		return errMissingNetRTTBaseline
	}
	if math.IsNaN(baseline.MeanRTTMS) || math.IsInf(baseline.MeanRTTMS, 0) || baseline.MeanRTTMS <= 0 {
		return fmt.Errorf("%w: got %v ms", errNonPositiveNetRTT, baseline.MeanRTTMS)
	}
	return nil
}

// AdjustLatenciesForNetRTT subtracts baseline.MeanRTTMS from each finite
// request-start-anchored latency in rawMS, clamping negative residuals at 0.
// Non-positive or missing baselines are refused cleanly.
func AdjustLatenciesForNetRTT(rawMS []float64, baseline NetRTTReport) ([]float64, error) {
	if err := ValidateNetRTTBaseline(baseline); err != nil {
		return nil, err
	}
	out := make([]float64, len(rawMS))
	for i, v := range rawMS {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			return nil, fmt.Errorf("loadgen: non-finite latency sample at index %d: %v", i, v)
		}
		adj := v - baseline.MeanRTTMS
		if adj < 0 {
			adj = 0
		}
		out[i] = adj
	}
	return out, nil
}

// NetworkAdjustedPoint reports non-destructive network_adjusted_* latency
// variants alongside the network_rtt scalar. ITL remains unmodified because
// RTT cancels across (request_latency - ttft).
type NetworkAdjustedPoint struct {
	NetworkRTTMS                 float64 `json:"network_rtt_ms"`
	NetworkAdjustedMeanLatencyMS float64 `json:"network_adjusted_mean_latency_ms"`
	NetworkAdjustedP50LatencyMS  float64 `json:"network_adjusted_p50_latency_ms"`
	NetworkAdjustedP95LatencyMS  float64 `json:"network_adjusted_p95_latency_ms"`
	NetworkAdjustedTTFTP50MS     float64 `json:"network_adjusted_ttft_p50_ms,omitempty"`
	NetworkAdjustedTTFTP95MS     float64 `json:"network_adjusted_ttft_p95_ms,omitempty"`
	MeanITLMS                    float64 `json:"mean_itl_ms,omitempty"`
	ITLP50MS                     float64 `json:"itl_p50_ms,omitempty"`
	ITLP95MS                     float64 `json:"itl_p95_ms,omitempty"`
}

// ComputeNetworkAdjustedPoint produces network_adjusted_* latencies from raw
// latency/TTFT slices (when provided) or PointResult aggregates, leaving ITL
// metrics unchanged.
func ComputeNetworkAdjustedPoint(pr PointResult, rawLatenciesMS, rawTTFTsMS []float64, baseline NetRTTReport) (NetworkAdjustedPoint, error) {
	if err := ValidateNetRTTBaseline(baseline); err != nil {
		return NetworkAdjustedPoint{}, err
	}

	out := NetworkAdjustedPoint{
		NetworkRTTMS: baseline.MeanRTTMS,
		MeanITLMS:    pr.MeanITLMS,
		ITLP50MS:     pr.ITLP50MS,
		ITLP95MS:     pr.ITLP95MS,
	}

	if len(rawLatenciesMS) > 0 {
		adj, err := AdjustLatenciesForNetRTT(rawLatenciesMS, baseline)
		if err != nil {
			return NetworkAdjustedPoint{}, err
		}
		sum := 0.0
		for _, v := range adj {
			sum += v
		}
		sort.Float64s(adj)
		out.NetworkAdjustedMeanLatencyMS = sum / float64(len(adj))
		out.NetworkAdjustedP50LatencyMS = percentile(adj, 50)
		out.NetworkAdjustedP95LatencyMS = percentile(adj, 95)
	} else {
		out.NetworkAdjustedMeanLatencyMS = clampNonNegative(pr.MeanLatencyMS - baseline.MeanRTTMS)
		out.NetworkAdjustedP50LatencyMS = clampNonNegative(pr.P50LatencyMS - baseline.MeanRTTMS)
		out.NetworkAdjustedP95LatencyMS = clampNonNegative(pr.P95LatencyMS - baseline.MeanRTTMS)
	}

	if len(rawTTFTsMS) > 0 {
		adjTTFT, err := AdjustLatenciesForNetRTT(rawTTFTsMS, baseline)
		if err != nil {
			return NetworkAdjustedPoint{}, err
		}
		sort.Float64s(adjTTFT)
		out.NetworkAdjustedTTFTP50MS = percentile(adjTTFT, 50)
		out.NetworkAdjustedTTFTP95MS = percentile(adjTTFT, 95)
	} else if pr.Streamed {
		if pr.TTFTP50MS > 0 {
			out.NetworkAdjustedTTFTP50MS = clampNonNegative(pr.TTFTP50MS - baseline.MeanRTTMS)
		}
		if pr.TTFTP95MS > 0 {
			out.NetworkAdjustedTTFTP95MS = clampNonNegative(pr.TTFTP95MS - baseline.MeanRTTMS)
		}
	}

	return out, nil
}

func clampNonNegative(v float64) float64 {
	if v < 0 {
		return 0
	}
	return v
}
