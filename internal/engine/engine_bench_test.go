package engine

import (
	"context"
	"encoding/json"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
	_ "github.com/anthony-chaudhary/fak/internal/blob"
)

func BenchmarkEngine(b *testing.B) {
	ctx := context.Background()

	b.Run("Mock_Complete", func(b *testing.B) {
		m := &Mock{}
		call := &abi.ToolCall{
			Tool: "benchmark_tool",
			Args: abi.Ref{Kind: abi.RefInline, Inline: []byte(`{"action":"eval","value":123}`)},
		}
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			res, err := m.Complete(ctx, call)
			if err != nil || res.Status != abi.StatusOK {
				b.Fatalf("unexpected completion: res=%v, err=%v", res, err)
			}
		}
	})

	b.Run("Cassette_Hit", func(b *testing.B) {
		tool := "benchmark_tool"
		args := []byte(`{"action":"eval","value":123}`)
		entry := CassetteEntry{
			Tool:     tool,
			Args:     args,
			Response: json.RawMessage(`{"tool":"benchmark_tool","ok":true,"result":"hit"}`),
			Usage:    Usage{InputTokens: 10, OutputTokens: 5, TotalTokens: 15},
		}
		c := NewCassette([]CassetteEntry{entry})
		engine := NewCassetteEngine(c)
		call := &abi.ToolCall{
			Tool: tool,
			Args: abi.Ref{Kind: abi.RefInline, Inline: args},
		}
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			res, err := engine.Complete(ctx, call)
			if err != nil || res.Status != abi.StatusOK {
				b.Fatalf("unexpected cassette hit completion: res=%v, err=%v", res, err)
			}
		}
	})

	b.Run("Cassette_Miss", func(b *testing.B) {
		c := NewCassette(nil)
		engine := NewCassetteEngine(c)
		call := &abi.ToolCall{
			Tool: "missing_tool",
			Args: abi.Ref{Kind: abi.RefInline, Inline: []byte(`{"missing":true}`)},
		}
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			res, err := engine.Complete(ctx, call)
			if err != nil || res.Status != abi.StatusError {
				b.Fatalf("unexpected cassette miss completion: res=%v, err=%v", res, err)
			}
		}
	})
}

func BenchmarkEngineDispatch(b *testing.B) {
	ctx := context.Background()
	m := &Mock{}

	payloads := []struct {
		name string
		size int
	}{
		{"Payload_64B", 64},
		{"Payload_1KB", 1024},
		{"Payload_64KB", 64 * 1024},
	}

	for _, p := range payloads {
		b.Run(p.name, func(b *testing.B) {
			data := make([]byte, p.size)
			for j := range data {
				data[j] = byte('a' + (j % 26))
			}
			call := &abi.ToolCall{
				Tool: "dispatch_test",
				Args: abi.Ref{Kind: abi.RefInline, Inline: data},
			}
			b.SetBytes(int64(p.size))
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				res, err := m.Complete(ctx, call)
				if err != nil || res.Status != abi.StatusOK {
					b.Fatalf("dispatch failure: res=%v, err=%v", res, err)
				}
			}
		})
	}

	b.Run("Parallel_Throughput", func(b *testing.B) {
		call := &abi.ToolCall{
			Tool: "dispatch_parallel",
			Args: abi.Ref{Kind: abi.RefInline, Inline: []byte(`{"parallel":true}`)},
		}
		b.ReportAllocs()
		b.ResetTimer()
		b.RunParallel(func(pb *testing.PB) {
			for pb.Next() {
				res, err := m.Complete(ctx, call)
				if err != nil || res.Status != abi.StatusOK {
					b.Fatalf("parallel dispatch failure: res=%v, err=%v", res, err)
				}
			}
		})
	})
}

func BenchmarkEngineResidency(b *testing.B) {
	ctx := context.Background()
	gate := residencyGate{}

	cases := []struct {
		name string
		call *abi.ToolCall
	}{
		{
			name: "Local_Tenant_Admitted",
			call: &abi.ToolCall{
				Engine: "inkernel",
				Args:   abi.Ref{Scope: abi.ScopeTenant, Kind: abi.RefInline, Inline: []byte(`{}`)},
			},
		},
		{
			name: "Remote_Tenant_Denied",
			call: &abi.ToolCall{
				Engine: "litellm/gpt-4o",
				Args:   abi.Ref{Scope: abi.ScopeTenant, Kind: abi.RefInline, Inline: []byte(`{}`)},
			},
		},
		{
			name: "Remote_SensitiveTag_Denied",
			call: &abi.ToolCall{
				Engine: "anthropic/claude-3-5",
				Args:   abi.Ref{Kind: abi.RefInline, Inline: []byte(`{}`)},
				Meta:   map[string]string{"sensitivity": "confidential"},
			},
		},
		{
			name: "Remote_ZDR_Denied",
			call: &abi.ToolCall{
				Engine: "openai/o3-mini",
				Args:   abi.Ref{Kind: abi.RefInline, Inline: []byte(`{}`)},
				Meta:   map[string]string{"zdr": "true"},
			},
		},
		{
			name: "Remote_Unsensitive_Deferred",
			call: &abi.ToolCall{
				Engine: "openrouter/auto",
				Args:   abi.Ref{Kind: abi.RefInline, Inline: []byte(`{}`)},
			},
		},
	}

	for _, tc := range cases {
		b.Run(tc.name, func(b *testing.B) {
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_ = gate.Adjudicate(ctx, tc.call)
			}
		})
	}
}

func BenchmarkEngineCallKey(b *testing.B) {
	tool := "calculate_metrics"
	payloads := []struct {
		name string
		size int
	}{
		{"Short_16B", 16},
		{"Medium_256B", 256},
		{"Large_4KB", 4096},
	}

	for _, p := range payloads {
		b.Run(p.name, func(b *testing.B) {
			args := make([]byte, p.size)
			for j := range args {
				args[j] = byte('0' + (j % 10))
			}
			b.SetBytes(int64(p.size))
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_ = CallKey(tool, args)
			}
		})
	}
}

func BenchmarkEngineCacheAdmit(b *testing.B) {
	cases := []struct {
		name string
		cap  CacheCapability
		op   CacheVerdict
	}{
		{
			name: "Admitted_ActiveWarm",
			cap: CacheCapability{
				Engine:          "vllm",
				Verdict:         CacheActiveWarm,
				Provenance:      ProvenanceKernel,
				ColdPathCorrect: true,
			},
			op: CacheActiveWarm,
		},
		{
			name: "Refused_Unknown",
			cap: CacheCapability{
				Engine:     "unknown-backend",
				Verdict:    CacheUnknown,
				Provenance: ProvenanceProvider,
			},
			op: CacheActiveWarm,
		},
		{
			name: "Refused_ColdPathUnwitnessed",
			cap: CacheCapability{
				Engine:          "sglang",
				Verdict:         CacheExactEvict,
				Provenance:      ProvenanceKernel,
				ColdPathCorrect: false,
			},
			op: CacheExactEvict,
		},
		{
			name: "Refused_Forecast",
			cap: CacheCapability{
				Engine:          "llama.cpp",
				Verdict:         CachePrefixClone,
				Provenance:      ProvenanceForecast,
				ColdPathCorrect: true,
			},
			op: CachePrefixClone,
		},
	}

	for _, tc := range cases {
		b.Run(tc.name, func(b *testing.B) {
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_ = AdmitActiveCache(tc.cap, tc.op)
			}
		})
	}
}

func BenchmarkEngineKVQuantization(b *testing.B) {
	now := time.Now()
	policy := KVQuantizationThresholds{
		DemotePressure:  0.85,
		PromotePressure: 0.60,
		AccuracyBudget:  0.05,
		MinDwell:        100 * time.Millisecond,
	}

	stateDemote := KVQuantizationState{
		Precision:      KVPrecisionFP16,
		Eligible:       true,
		EstimatedError: 0.01,
		LastTransition: now.Add(-time.Second),
	}

	b.Run("DemoteDecision", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_ = ChooseKVQuantization(now, 0.90, stateDemote, policy)
		}
	})

	statePromote := KVQuantizationState{
		Precision:      KVPrecisionFP8,
		Eligible:       true,
		EstimatedError: 0.01,
		LastTransition: now.Add(-time.Second),
	}

	b.Run("PromoteDecision", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_ = ChooseKVQuantization(now, 0.40, statePromote, policy)
		}
	})
}
