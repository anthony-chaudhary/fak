// fak-test:runtime fast est=1s lane=default
package main

import (
	"context"
	"errors"
	"math"
	"net"
	"sync/atomic"
	"testing"
	"time"
)

func TestNetRTT(t *testing.T) {
	t.Run("resolves DNS once and runs unpooled TCP probes against local listener", func(t *testing.T) {
		ln, err := net.Listen("tcp", "127.0.0.1:0")
		if err != nil {
			t.Fatalf("net.Listen: %v", err)
		}
		defer ln.Close()

		acceptDone := make(chan struct{})
		var accepted atomic.Int32
		go func() {
			defer close(acceptDone)
			for {
				c, err := ln.Accept()
				if err != nil {
					return
				}
				n := accepted.Add(1)
				_ = c.Close()
				if n == 4 {
					return
				}
			}
		}()

		var resolveCalls int
		_, port, _ := net.SplitHostPort(ln.Addr().String())
		prober, err := NewNetRTTProber(context.Background(), "https://mock-endpoint.local:"+port+"/v1/chat/completions", NetRTTProbeOptions{
			Probes:  4,
			Timeout: time.Second,
			Resolve: func(ctx context.Context, host string) ([]net.IPAddr, error) {
				resolveCalls++
				return []net.IPAddr{{IP: net.ParseIP("127.0.0.1")}}, nil
			},
			// Advance simulated clock by 2.5ms per dial so OS timer resolution never yields 0ns on loopback.
			Now: func() func() time.Time {
				base := time.Unix(1700000000, 0)
				step := 0
				return func() time.Time {
					ts := base.Add(time.Duration(step) * 2500 * time.Microsecond)
					step++
					return ts
				}
			}(),
		})
		if err != nil {
			t.Fatalf("NewNetRTTProber: %v", err)
		}
		rep := prober.Probe(context.Background())
		select {
		case <-acceptDone:
		case <-time.After(2 * time.Second):
			t.Fatalf("timed out waiting for server accept (accepted=%d)", accepted.Load())
		}
		if resolveCalls != 1 {
			t.Fatalf("DNS resolve calls = %d, want 1 (resolve once at construction)", resolveCalls)
		}
		if rep.Samples != 4 || rep.Errors != 0 || accepted.Load() != 4 {
			t.Fatalf("unexpected probe report=%+v accepted=%d", rep, accepted.Load())
		}
		if math.Abs(rep.MeanRTTMS-2.5) > 1e-6 {
			t.Fatalf("MeanRTTMS = %v, want 2.5", rep.MeanRTTMS)
		}
	})

	t.Run("partial probe failures do not crash and non-positive/missing baselines are rejected", func(t *testing.T) {
		attempt := 0
		prober, err := NewNetRTTProber(context.Background(), "http://example.local/v1/chat/completions", NetRTTProbeOptions{
			Probes: 3,
			Resolve: func(ctx context.Context, host string) ([]net.IPAddr, error) {
				return []net.IPAddr{{IP: net.ParseIP("127.0.0.1")}}, nil
			},
			Dial: func(ctx context.Context, network, addr string) (net.Conn, error) {
				attempt++
				if attempt == 2 {
					return nil, errors.New("dial refused")
				}
				c1, c2 := net.Pipe()
				_ = c2.Close()
				return c1, nil
			},
			Now: func() func() time.Time {
				t0 := time.Unix(1700000000, 0)
				calls := 0
				return func() time.Time {
					// Probe 1: 4ms, Probe 2: error, Probe 3: 6ms => mean 5ms.
					offsets := []time.Duration{
						0, 4 * time.Millisecond,
						10 * time.Millisecond, 11 * time.Millisecond,
						20 * time.Millisecond, 26 * time.Millisecond,
					}
					ts := t0.Add(offsets[calls])
					calls++
					return ts
				}
			}(),
		})
		if err != nil {
			t.Fatal(err)
		}
		rep := prober.Probe(context.Background())
		if rep.Samples != 2 || rep.Errors != 1 || math.Abs(rep.MeanRTTMS-5.0) > 1e-6 {
			t.Fatalf("partial-failure report = %+v, want samples=2 errors=1 mean=5ms", rep)
		}

		// Reject missing / zero / negative / NaN baselines cleanly.
		for _, bad := range []NetRTTReport{
			{Samples: 0, MeanRTTMS: 5.0},
			{Samples: 3, MeanRTTMS: 0},
			{Samples: 3, MeanRTTMS: -1.5},
			{Samples: 3, MeanRTTMS: math.NaN()},
		} {
			if _, err := AdjustLatenciesForNetRTT([]float64{10, 20}, bad); err == nil {
				t.Errorf("expected rejection for bad baseline %+v", bad)
			}
			if _, err := ComputeNetworkAdjustedPoint(PointResult{MeanLatencyMS: 10}, nil, nil, bad); err == nil {
				t.Errorf("expected ComputeNetworkAdjustedPoint rejection for bad baseline %+v", bad)
			}
		}
	})

	t.Run("network_adjusted clamps >= 0 and leaves ITL unmodified", func(t *testing.T) {
		baseline := NetRTTReport{Samples: 3, MeanRTTMS: 10.0}
		pr := PointResult{
			MeanLatencyMS: 25.0,
			P50LatencyMS:  20.0,
			P95LatencyMS:  40.0,
			Streamed:      true,
			TTFTP50MS:     8.0, // below RTT -> clamps to 0
			TTFTP95MS:     15.0,
			MeanITLMS:     3.2,
			ITLP50MS:      3.0,
			ITLP95MS:      4.5,
		}
		adj, err := ComputeNetworkAdjustedPoint(pr, []float64{5.0, 20.0, 40.0}, []float64{8.0, 15.0}, baseline)
		if err != nil {
			t.Fatalf("ComputeNetworkAdjustedPoint: %v", err)
		}
		if adj.NetworkRTTMS != 10.0 {
			t.Fatalf("NetworkRTTMS = %v, want 10", adj.NetworkRTTMS)
		}
		// Adjusted raw latencies: [0, 10, 30] -> mean 40/3, p50 10, p95 30.
		if math.Abs(adj.NetworkAdjustedMeanLatencyMS-(40.0/3.0)) > 1e-6 ||
			adj.NetworkAdjustedP50LatencyMS != 10.0 ||
			adj.NetworkAdjustedP95LatencyMS != 30.0 {
			t.Fatalf("unexpected adjusted latencies: %+v", adj)
		}
		// TTFT 8ms - 10ms clamps to 0; 15ms - 10ms = 5ms.
		if adj.NetworkAdjustedTTFTP50MS != 0 || adj.NetworkAdjustedTTFTP95MS != 5.0 {
			t.Fatalf("unexpected adjusted TTFTs: %+v", adj)
		}
		// ITL must remain completely exempt from RTT subtraction.
		if adj.MeanITLMS != 3.2 || adj.ITLP50MS != 3.0 || adj.ITLP95MS != 4.5 {
			t.Fatalf("ITL was unexpectedly modified: %+v", adj)
		}
		// Raw PointResult stays non-destructively untouched.
		if pr.P50LatencyMS != 20.0 || pr.TTFTP50MS != 8.0 {
			t.Fatalf("raw PointResult mutated: %+v", pr)
		}
	})
}
