package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/macobs"
)

func cmdMacObs(argv []string) {
	os.Exit(runMacObs(os.Stdout, os.Stderr, argv))
}

func runMacObs(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("macobs", flag.ContinueOnError)
	fs.SetOutput(stderr)

	asJSON := fs.Bool("json", false, "emit the fak.macobs.v1 JSON envelope")
	checkHeadroom := fs.Bool("check-headroom", false, "emit concise agent-focused admission report")
	agents := fs.Int("agents", 4, "target concurrent agents to evaluate")
	prefixTokens := fs.Uint64("prefix-tokens", 4096, "shared prefix tokens (system prompt + tools)")
	tailTokens := fs.Uint64("tail-tokens", 2048, "private agent turn tokens")
	mlxEndpoint := fs.String("mlx-endpoint", "", "MLX server metrics Prometheus endpoint URL")
	watch := fs.Bool("watch", false, "watch mode: periodically refresh dashboard")
	interval := fs.Duration("interval", 2*time.Second, "refresh interval in watch mode")

	if rc, ok := parseFlagsOrHelp(fs, argv); !ok {
		return rc
	}

	if fs.NArg() != 0 {
		fmt.Fprintln(stderr, "fak macobs: unexpected positional arguments")
		return 2
	}
	if *asJSON && *checkHeadroom {
		fmt.Fprintln(stderr, "fak macobs: cannot specify both --json and --check-headroom")
		return 2
	}
	if *agents <= 0 {
		fmt.Fprintln(stderr, "fak macobs: --agents must be positive")
		return 2
	}
	if *prefixTokens == 0 {
		fmt.Fprintln(stderr, "fak macobs: --prefix-tokens must be positive")
		return 2
	}
	if *tailTokens == 0 {
		fmt.Fprintln(stderr, "fak macobs: --tail-tokens must be positive")
		return 2
	}
	if *interval <= 0 {
		fmt.Fprintln(stderr, "fak macobs: --interval must be positive")
		return 2
	}

	cfg := macobs.DefaultHeadroomConfig()
	cfg.SharedPrefixTokens = *prefixTokens
	cfg.PrivateTailTokens = *tailTokens

	opts := []macobs.Option{
		macobs.WithHeadroomConfig(cfg),
		macobs.WithRequestedAgents(*agents),
	}
	if *mlxEndpoint != "" {
		opts = append(opts, macobs.WithMetricsURL(*mlxEndpoint))
	}
	col := macobs.NewCollector(opts...)

	for {
		snap, err := col.Observe(context.Background())
		if err != nil {
			fmt.Fprintf(stderr, "fak macobs: observation failed: %v\n", err)
			return 1
		}

		if *asJSON {
			data, err := json.MarshalIndent(snap, "", "  ")
			if err != nil {
				fmt.Fprintf(stderr, "fak macobs: marshal json failed: %v\n", err)
				return 1
			}
			fmt.Fprintln(stdout, string(data))
		} else if *checkHeadroom {
			bName := string(snap.Analysis.PrimaryBottleneck)
			if !strings.HasPrefix(bName, "BOTTLENECK_") {
				bName = "BOTTLENECK_" + bName
			}
			gatePassed := snap.Analysis.Verdict == macobs.VerdictHeadroomOK
			fmt.Fprintf(stdout, "macobs: [%s] recommended_agents=%d (shared=%d, isolated=%d) available_kv=%dMB bottleneck=%s gate_passed=%t\n",
				snap.Analysis.Verdict,
				snap.Analysis.RecommendedAgents,
				snap.Headroom.MaxSharedAgents,
				snap.Headroom.MaxIsolatedAgents,
				snap.Headroom.AvailableKVPoolBytes/(1024*1024),
				bName,
				gatePassed,
			)
			if !gatePassed {
				return 1
			}
		} else {
			renderMacObsDashboard(stdout, snap)
		}

		if !*watch {
			break
		}
		time.Sleep(*interval)
	}

	return 0
}

func renderMacObsDashboard(w io.Writer, snap macobs.Snapshot) {
	fmt.Fprintln(w, "fak macobs — Apple Silicon & MLX Agent Observability")
	fmt.Fprintln(w, "Hardware:")
	if snap.Hardware.Available {
		fmt.Fprintf(w, "  Physical Memory: %.2f GB (Wired Limit: %.2f GB, Metal Allocated: %.2f GB)\n",
			float64(snap.Hardware.TotalSystemMemoryBytes)/(1<<30),
			float64(snap.Hardware.WiredMemoryLimitBytes)/(1<<30),
			float64(snap.Hardware.AllocSystemMemoryBytes)/(1<<30),
		)
		fmt.Fprintf(w, "  Swap: %d MB used / %d MB total (Pageouts: %d)\n",
			snap.Hardware.SwapUsedBytes/(1024*1024),
			snap.Hardware.SwapTotalBytes/(1024*1024),
			snap.Hardware.PageOuts,
		)
		fmt.Fprintf(w, "  Thermal State: %s | Power: %s\n",
			snap.Hardware.ThermalState,
			snap.Hardware.PowerSource,
		)
	} else {
		fmt.Fprintln(w, "  Status: unavailable (non-Darwin host or hardware counters unreadable)")
	}

	fmt.Fprintln(w, "MLX Serving:")
	if snap.MLXServing.Available {
		serverType := snap.MLXServing.ServerType
		if serverType == "" {
			serverType = "unknown"
		}
		fmt.Fprintf(w, "  Active Requests: %d | Queued: %d | KV Cache: %.1f%% (%s)\n",
			snap.MLXServing.ActiveRequests,
			snap.MLXServing.QueuedRequests,
			snap.MLXServing.KVCacheUsagePct,
			serverType,
		)
		fmt.Fprintf(w, "  Throughput: %.1f prompt tok/s, %.1f decode tok/s | Avg TTFT: %.1f ms | Avg ITL: %.1f ms\n",
			snap.MLXServing.PromptTokensPerSec,
			snap.MLXServing.DecodeTokensPerSec,
			snap.MLXServing.AvgTTFTMs,
			snap.MLXServing.AvgITLMs,
		)
		if snap.PrefixCache.Available {
			fmt.Fprintf(w, "  Prefix Cache: %.1f%% hit ratio (%d hits, %d misses)\n",
				snap.PrefixCache.HitRatio,
				snap.PrefixCache.Hits,
				snap.PrefixCache.Misses,
			)
		}
	} else {
		fmt.Fprintln(w, "  Status: unavailable (no endpoint configured or server unreachable)")
	}

	fmt.Fprintln(w, "Agent Concurrency Headroom:")
	if snap.Headroom.Available {
		fmt.Fprintf(w, "  Available KV Pool: %d MB (KV/token: %d B)\n",
			snap.Headroom.AvailableKVPoolBytes/(1024*1024),
			snap.Headroom.ModelKVBytesPerToken,
		)
		fmt.Fprintf(w, "  Max Concurrency: %d shared prefix agents vs %d isolated agents (%.1fx advantage)\n",
			snap.Headroom.MaxSharedAgents,
			snap.Headroom.MaxIsolatedAgents,
			snap.Headroom.ConcurrencyAdvantage,
		)
		fmt.Fprintf(w, "  Context Model: %d prefix tokens shared + %d tail tokens/agent\n",
			snap.Headroom.SharedPrefixTokens,
			snap.Headroom.PrivateTailTokens,
		)
	} else {
		fmt.Fprintln(w, "  Status: unavailable (insufficient unified memory for KV pool)")
	}

	bName := string(snap.Analysis.PrimaryBottleneck)
	if !strings.HasPrefix(bName, "BOTTLENECK_") {
		bName = "BOTTLENECK_" + bName
	}
	fmt.Fprintln(w, "Verdict & Action:")
	fmt.Fprintf(w, "  Verdict: [%s] (Recommended Concurrency: %d agents)\n",
		snap.Analysis.Verdict,
		snap.Analysis.RecommendedAgents,
	)
	fmt.Fprintf(w, "  Bottleneck: %s (%s)\n",
		bName,
		snap.Analysis.BottleneckReason,
	)
	fmt.Fprintf(w, "  Remediation: %s\n",
		snap.Analysis.Remediation,
	)
}
