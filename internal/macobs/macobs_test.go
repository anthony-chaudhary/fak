package macobs

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestCollectorObserve(t *testing.T) {
	// 1. Mock MLX Prometheus endpoint
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(`
vllm:num_requests_running 2
vllm:num_requests_waiting 0
vllm:kv_cache_usage_perc 40.0
vllm:time_to_first_token_seconds_sum 0.5
vllm:time_to_first_token_seconds_count 5
vllm:inter_token_latency_seconds_sum 0.1
vllm:inter_token_latency_seconds_count 10
vllm:avg_prompt_throughput_tok_per_s 150.0
vllm:avg_generation_throughput_tok_per_s 35.0
vllm:prefix_cache_hits 100
vllm:prefix_cache_queries 120
`))
	}))
	defer ts.Close()

	// 2. Mock command runner
	mockRunner := func(ctx context.Context, name string, args ...string) ([]byte, error) {
		cmdStr := name + " " + strings.Join(args, " ")
		switch {
		case strings.Contains(cmdStr, "hw.memsize"):
			return []byte("38654705664\n"), nil // 36GB
		case strings.Contains(cmdStr, "iogpu.wired_limit_mb"):
			return []byte("27648\n"), nil // 27GB
		case strings.Contains(cmdStr, "vm.swapusage"):
			return []byte("total = 1024.00M used = 0.00M free = 1024.00M\n"), nil
		case strings.Contains(cmdStr, "kern.memorystatus_level"):
			return []byte("75\n"), nil
		case strings.Contains(cmdStr, "vm_stat"):
			return []byte("page size of 16384 bytes\nPages free: 20000.\nPages wired down: 50000.\nPages occupied by compressor: 10000.\nPageins: 500.\nPageouts: 0.\n"), nil
		case strings.Contains(cmdStr, "therm"):
			return []byte("Note: No thermal warning level has been recorded\n"), nil
		case strings.Contains(cmdStr, "ps"):
			return []byte("Now drawing from 'AC Power'\n -InternalBattery-0 100%;\n"), nil
		default:
			return nil, nil
		}
	}

	col := NewCollector(
		WithMetricsURL(ts.URL),
		WithHTTPClient(ts.Client()),
		WithCommandRunner(mockRunner),
		WithRequestedAgents(4),
		WithHeadroomConfig(DefaultHeadroomConfig()),
	)

	snap, err := col.Observe(context.Background())
	if err != nil {
		t.Fatalf("Observe failed: %v", err)
	}

	if snap.Schema != SchemaV1 {
		t.Errorf("got schema %s, want %s", snap.Schema, SchemaV1)
	}
	if snap.Timestamp.IsZero() {
		t.Errorf("expected non-zero timestamp")
	}

	// Hardware checks (on Darwin runner was mocked)
	if snap.Hardware.TotalSystemMemoryBytes != 38654705664 {
		t.Errorf("got hardware total mem %d, want 38654705664", snap.Hardware.TotalSystemMemoryBytes)
	}

	// MLX Serving checks
	if !snap.MLXServing.Available {
		t.Fatalf("expected MLXServing to be available")
	}
	if snap.MLXServing.ActiveRequests != 2 {
		t.Errorf("got active requests %d, want 2", snap.MLXServing.ActiveRequests)
	}
	if snap.MLXServing.KVCacheUsagePct != 40.0 {
		t.Errorf("got kv cache usage %f, want 40.0", snap.MLXServing.KVCacheUsagePct)
	}

	// Prefix Cache checks
	if !snap.PrefixCache.Available {
		t.Fatalf("expected PrefixCache to be available")
	}
	if snap.PrefixCache.Hits != 100 {
		t.Errorf("got prefix hits %d, want 100", snap.PrefixCache.Hits)
	}

	// Headroom checks
	if !snap.Headroom.Available {
		t.Fatalf("expected Headroom to be available")
	}
	if snap.Headroom.MaxSharedAgents <= 0 {
		t.Errorf("expected positive MaxSharedAgents, got %d", snap.Headroom.MaxSharedAgents)
	}

	// Analysis checks
	if snap.Analysis.Verdict != VerdictHeadroomOK {
		t.Errorf("got verdict %s, want HEADROOM_OK", snap.Analysis.Verdict)
	}
}

func TestCollectorObserveNoEndpoints(t *testing.T) {
	col := NewCollector(
		WithMetricsURL(""),
	)
	snap, err := col.Observe(context.Background())
	if err != nil {
		t.Fatalf("Observe failed: %v", err)
	}
	if snap.Schema != SchemaV1 {
		t.Errorf("got schema %s, want %s", snap.Schema, SchemaV1)
	}
	if snap.MLXServing.Available {
		t.Errorf("expected MLXServing available=false when no URL provided")
	}
}

func TestCoordinatorSubagentAdmissionScenario(t *testing.T) {
	// Coordinator plans a wave of 8 subagents with:
	// - 4096-token shared preamble (system prompt + shared tools)
	// - 2048-token private turn (agent reasoning scratchpad + tool outputs)
	// - Context tokens = 8192
	// - Model architecture: Qwen2.5 7B GQA (28 layers, 4 KV heads, 128 head dim, 2 bytes/elem fp16)
	//   KV bytes per token = 2 * 28 * 4 * 128 * 2 = 57,344 bytes.
	// - Model weights = 5GB, OS reserve = 3GB (required base = 8GB).
	cfg := HeadroomConfig{
		Layers:             28,
		KVHeads:            4,
		HeadDim:            128,
		KVBytesPerElement:  2,
		ModelWeightBytes:   5 * 1024 * 1024 * 1024,
		ContextTokens:      8192,
		SharedPrefixTokens: 4096,
		PrivateTailTokens:  2048,
		OSReserveBytes:     3 * 1024 * 1024 * 1024,
	}

	requestedAgents := 8

	t.Run("PrefixSharingEnabled_Admits8Subagents_VerdictHeadroomOK", func(t *testing.T) {
		// Mock hardware: 36GB M-series host with 24GB wired limit.
		// Base: 8GB -> Available KV pool = 16GB (17,179,869,184 bytes).
		// Shared prefix KV = 4096 * 57,344 = 234,881,024 bytes (~224 MB).
		// Private tail KV per agent = 2048 * 57,344 = 117,440,512 bytes (~112 MB).
		// Isolated KV per agent (without prefix sharing) = 8192 * 57,344 = 469,762,048 bytes (~448 MB).
		// Max Isolated = 16GB / 469,762,048 = 36 agents.
		// Max Shared = (16GB - 234,881,024) / 117,440,512 = 144 agents.
		// Concurrency Advantage = 144 / 36 = 4.0x.
		// With requestedAgents = 8 <= 144, headroom is plentiful:
		// Verdict must be VerdictHeadroomOK with RecommendedAgents = 8.
		mockRunner := func(ctx context.Context, name string, args ...string) ([]byte, error) {
			cmdStr := name + " " + strings.Join(args, " ")
			switch {
			case strings.Contains(cmdStr, "hw.memsize"):
				return []byte("38654705664\n"), nil // 36GB
			case strings.Contains(cmdStr, "iogpu.wired_limit_mb"):
				return []byte("24576\n"), nil // 24GB
			case strings.Contains(cmdStr, "vm.swapusage"):
				return []byte("total = 1024.00M used = 0.00M free = 1024.00M\n"), nil
			case strings.Contains(cmdStr, "kern.memorystatus_level"):
				return []byte("75\n"), nil
			case strings.Contains(cmdStr, "therm"):
				return []byte("Note: No thermal warning level has been recorded\n"), nil
			case strings.Contains(cmdStr, "ps"):
				return []byte("Now drawing from 'AC Power'\n"), nil
			default:
				return nil, nil
			}
		}

		col := NewCollector(
			WithCommandRunner(mockRunner),
			WithHeadroomConfig(cfg),
			WithRequestedAgents(requestedAgents),
		)

		snap, err := col.Observe(context.Background())
		if err != nil {
			t.Fatalf("Observe failed: %v", err)
		}

		if !snap.Headroom.Available {
			t.Fatalf("expected Headroom.Available = true")
		}
		wantPool := uint64(16 * 1024 * 1024 * 1024)
		if snap.Headroom.AvailableKVPoolBytes != wantPool {
			t.Errorf("AvailableKVPoolBytes = %d, want %d", snap.Headroom.AvailableKVPoolBytes, wantPool)
		}
		if snap.Headroom.MaxIsolatedAgents != 36 {
			t.Errorf("MaxIsolatedAgents = %d, want 36", snap.Headroom.MaxIsolatedAgents)
		}
		if snap.Headroom.MaxSharedAgents != 144 {
			t.Errorf("MaxSharedAgents = %d, want 144", snap.Headroom.MaxSharedAgents)
		}
		if snap.Headroom.ConcurrencyAdvantage != 4.0 {
			t.Errorf("ConcurrencyAdvantage = %f, want 4.0", snap.Headroom.ConcurrencyAdvantage)
		}

		// Verdict checks
		if snap.Analysis.Verdict != VerdictHeadroomOK {
			t.Errorf("Verdict = %s, want HEADROOM_OK", snap.Analysis.Verdict)
		}
		if snap.Analysis.PrimaryBottleneck != BottleneckNone {
			t.Errorf("PrimaryBottleneck = %s, want NONE", snap.Analysis.PrimaryBottleneck)
		}
		if snap.Analysis.RecommendedAgents != 8 {
			t.Errorf("RecommendedAgents = %d, want 8", snap.Analysis.RecommendedAgents)
		}
		wantRemediation := "Headroom is nominal; system can support concurrent agent runs or burst tool loops."
		if snap.Analysis.Remediation != wantRemediation {
			t.Errorf("Remediation = %q, want %q", snap.Analysis.Remediation, wantRemediation)
		}
	})

	t.Run("MemoryConstrained_TriggersReduceConcurrency_RecommendsExactSafeConcurrency", func(t *testing.T) {
		// Constrained memory scenario:
		// Model weights: 5GB + OS reserve: 3GB = 8GB base (8,589,934,592 bytes).
		// Wired limit: 9000 MB = 9,437,184,000 bytes.
		// Available KV pool = 9,437,184,000 - 8,589,934,592 = 847,249,408 bytes (~808 MB).
		// Shared prefix: 4096 * 57,344 = 234,881,024 bytes (~224 MB).
		// Remainder for private tails: 847,249,408 - 234,881,024 = 612,368,384 bytes.
		// Private tail per agent: 2048 * 57,344 = 117,440,512 bytes (~112 MB).
		// Max Shared Agents = 612,368,384 / 117,440,512 = 5 agents.
		// Max Isolated Agents = 847,249,408 / 469,762,048 = 1 agent.
		// Concurrency advantage = 5.0 / 1.0 = 5.0x.
		// Coordinator assesses launching 8 subagents.
		// 8 > MaxSharedAgents (5) -> must trigger VerdictReduceConcurrency with RecommendedAgents = 5.
		mockRunner := func(ctx context.Context, name string, args ...string) ([]byte, error) {
			cmdStr := name + " " + strings.Join(args, " ")
			switch {
			case strings.Contains(cmdStr, "hw.memsize"):
				return []byte("17179869184\n"), nil // 16GB
			case strings.Contains(cmdStr, "iogpu.wired_limit_mb"):
				return []byte("9000\n"), nil // 9000 MB
			case strings.Contains(cmdStr, "vm.swapusage"):
				return []byte("total = 1024.00M used = 0.00M free = 1024.00M\n"), nil
			case strings.Contains(cmdStr, "kern.memorystatus_level"):
				return []byte("75\n"), nil
			case strings.Contains(cmdStr, "therm"):
				return []byte("Note: No thermal warning level has been recorded\n"), nil
			case strings.Contains(cmdStr, "ps"):
				return []byte("Now drawing from 'AC Power'\n"), nil
			default:
				return nil, nil
			}
		}

		col := NewCollector(
			WithCommandRunner(mockRunner),
			WithHeadroomConfig(cfg),
			WithRequestedAgents(requestedAgents),
		)

		snap, err := col.Observe(context.Background())
		if err != nil {
			t.Fatalf("Observe failed: %v", err)
		}

		if !snap.Headroom.Available {
			t.Fatalf("expected Headroom.Available = true")
		}
		wantPool := uint64(9000*1024*1024) - uint64(8*1024*1024*1024)
		if snap.Headroom.AvailableKVPoolBytes != wantPool {
			t.Errorf("AvailableKVPoolBytes = %d, want %d", snap.Headroom.AvailableKVPoolBytes, wantPool)
		}
		if snap.Headroom.MaxIsolatedAgents != 1 {
			t.Errorf("MaxIsolatedAgents = %d, want 1", snap.Headroom.MaxIsolatedAgents)
		}
		if snap.Headroom.MaxSharedAgents != 5 {
			t.Errorf("MaxSharedAgents = %d, want 5", snap.Headroom.MaxSharedAgents)
		}
		if snap.Headroom.ConcurrencyAdvantage != 5.0 {
			t.Errorf("ConcurrencyAdvantage = %f, want 5.0", snap.Headroom.ConcurrencyAdvantage)
		}

		// Analysis assertions:
		// Must diagnose REDUCE_CONCURRENCY and recommend exactly 5 safe agents
		if snap.Analysis.Verdict != VerdictReduceConcurrency {
			t.Errorf("Verdict = %s, want REDUCE_CONCURRENCY", snap.Analysis.Verdict)
		}
		if snap.Analysis.PrimaryBottleneck != BottleneckMemoryCapacity {
			t.Errorf("PrimaryBottleneck = %s, want MEMORY_CAPACITY", snap.Analysis.PrimaryBottleneck)
		}
		if snap.Analysis.RecommendedAgents != 5 {
			t.Errorf("RecommendedAgents = %d, want 5", snap.Analysis.RecommendedAgents)
		}
		wantReason := "Requested 8 concurrent agents exceeds modeled headroom of 5 shared agents"
		if snap.Analysis.BottleneckReason != wantReason {
			t.Errorf("BottleneckReason = %q, want %q", snap.Analysis.BottleneckReason, wantReason)
		}
		wantRemediation := "Limit active concurrent subagents to 5 to remain within wired unified memory limits."
		if snap.Analysis.Remediation != wantRemediation {
			t.Errorf("Remediation = %q, want %q", snap.Analysis.Remediation, wantRemediation)
		}
	})
}
