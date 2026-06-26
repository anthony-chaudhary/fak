package gateway

import (
	"context"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/cacheobs"
)

// metrics_kvprefix_test.go — the in-kernel KV-prefix reuse family on /metrics
// (fak_gateway_kv_prefix_*). It is the live measurement of the frozen-trajectory cache
// cliff: the planner feeds cacheobs.Default the realized reuse on every served in-kernel
// turn, and the gateway scrapes it here. The process-global tap may carry counts from
// sibling tests, so the family-present checks assert the series exist; the live-read
// asserts an Observe moves the reused-tokens counter and the frozen-regime bucket.
func TestMetricsExposesKVPrefixReuse(t *testing.T) {
	srv := newTestServer(t)

	for _, want := range []string{
		"# TYPE fak_gateway_kv_prefix_turns_total counter",
		"# TYPE fak_gateway_kv_prefix_prompt_tokens_total counter",
		"# TYPE fak_gateway_kv_prefix_reused_tokens_total counter",
		"# TYPE fak_gateway_kv_prefix_turns_by_regime_total counter",
		`fak_gateway_kv_prefix_turns_by_regime_total{regime="frozen"} `,
		`fak_gateway_kv_prefix_turns_by_regime_total{regime="partial"} `,
		`fak_gateway_kv_prefix_turns_by_regime_total{regime="cold"} `,
		"# TYPE fak_gateway_kv_prefix_reuse_ratio gauge",
		"fak_gateway_kv_prefix_reuse_ratio ",
	} {
		if text := srv.renderMetrics(); !strings.Contains(text, want) {
			t.Fatalf("metrics missing %q\n--- metrics ---\n%s", want, text)
		}
	}

	// Live read: a frozen-regime turn (990/1000 reused) must move the reused-tokens
	// counter and increment the frozen bucket.
	before := cacheobs.Default.Snapshot()
	cacheobs.Default.Observe(1000, 990)
	after := cacheobs.Default.Snapshot()
	if after.ReusedTokens <= before.ReusedTokens {
		t.Fatalf("reused tokens did not rise after Observe: before=%d after=%d", before.ReusedTokens, after.ReusedTokens)
	}
	if after.FrozenTurns != before.FrozenTurns+1 {
		t.Fatalf("frozen bucket did not increment: before=%d after=%d", before.FrozenTurns, after.FrozenTurns)
	}

	text := srv.renderMetrics()
	line := metricLine(text, "fak_gateway_kv_prefix_reused_tokens_total")
	if line == "" {
		t.Fatalf("no fak_gateway_kv_prefix_reused_tokens_total line:\n%s", text)
	}
	n, err := strconv.Atoi(strings.TrimSpace(strings.TrimPrefix(line, "fak_gateway_kv_prefix_reused_tokens_total")))
	if err != nil {
		t.Fatalf("parse %q: %v", line, err)
	}
	if uint64(n) < after.ReusedTokens {
		t.Fatalf("scraped reused tokens %d < observed %d", n, after.ReusedTokens)
	}
}

type kvMemoryStatsPlanner struct {
	stats agent.KVMemoryStats
	req   agent.RequestMemoryStats
}

func (p kvMemoryStatsPlanner) Complete(context.Context, []agent.Message, []agent.ToolDef, ...agent.SampleOpt) (*agent.Completion, error) {
	return &agent.Completion{Message: agent.Message{Role: agent.RoleAssistant, Content: "ok"}}, nil
}

func (p kvMemoryStatsPlanner) Model() string { return "kv-memory-test" }

func (p kvMemoryStatsPlanner) KVMemoryStats() agent.KVMemoryStats { return p.stats }

func (p kvMemoryStatsPlanner) RequestMemoryStats() agent.RequestMemoryStats { return p.req }

func TestKVMemoryMetricsSuppressedWithoutReporter(t *testing.T) {
	srv := newTestServer(t)
	if text := srv.renderMetrics(); strings.Contains(text, "fak_gateway_kv_memory_") {
		t.Fatalf("resident KV memory metrics should be absent for a non-reporting planner:\n%s", text)
	}
	if vars := srv.debugVars(time.Now()); vars.KVMemory != nil {
		t.Fatalf("debug kv_memory should be absent for a non-reporting planner: %+v", vars.KVMemory)
	}
}

func TestKVMemoryMetricsAndDebugVars(t *testing.T) {
	srv := newTestServer(t)
	srv.planner = kvMemoryStatsPlanner{stats: agent.KVMemoryStats{
		Enabled:         true,
		Backend:         "radixkv",
		MemoryClass:     "kv_cache",
		Scope:           "host",
		DType:           "f32",
		BytesPerToken:   6144,
		ResidentTokens:  42,
		ResidentBytes:   258048,
		BudgetTokens:    64,
		LRUTokens:       18,
		MaxDepthTokens:  21,
		Nodes:           3,
		Leaves:          2,
		Evictions:       4,
		PolicyEvictions: 1,
		Splits:          5,
	}}

	text := srv.renderMetrics()
	for _, want := range []string{
		`fak_gateway_kv_memory_enabled{class="kv_cache",scope="host",backend="radixkv"} 1`,
		`fak_gateway_kv_memory_dtype_info{class="kv_cache",scope="host",backend="radixkv",dtype="f32"} 1`,
		`fak_gateway_kv_memory_bytes_per_token{class="kv_cache",scope="host",backend="radixkv"} 6144`,
		`fak_gateway_kv_memory_resident_tokens{class="kv_cache",scope="host",backend="radixkv"} 42`,
		`fak_gateway_kv_memory_resident_bytes{class="kv_cache",scope="host",backend="radixkv"} 258048`,
		`fak_gateway_kv_memory_lru_tokens{class="kv_cache",scope="host",backend="radixkv"} 18`,
		`fak_gateway_kv_memory_budget_tokens{class="kv_cache",scope="host",backend="radixkv"} 64`,
		`fak_gateway_kv_memory_evictions_total{class="kv_cache",scope="host",backend="radixkv",kind="lru"} 4`,
		`fak_gateway_kv_memory_evictions_total{class="kv_cache",scope="host",backend="radixkv",kind="policy"} 1`,
		`fak_gateway_kv_memory_splits_total{class="kv_cache",scope="host",backend="radixkv"} 5`,
	} {
		if !strings.Contains(text, want) {
			t.Fatalf("KV memory metrics missing %q\n--- metrics ---\n%s", want, text)
		}
	}

	vars := srv.debugVars(time.Now())
	if vars.KVMemory == nil {
		t.Fatal("/debug/vars missing kv_memory")
	}
	if vars.KVMemory.ResidentTokens != 42 || vars.KVMemory.ResidentBytes != 258048 ||
		vars.KVMemory.LRUTokens != 18 || vars.KVMemory.PolicyEvictions != 1 || vars.KVMemory.DType != "f32" {
		t.Fatalf("debug kv_memory = %+v, want resident/lru/eviction fields", vars.KVMemory)
	}
}

func TestKVMemoryMetricsDisabledReporterEmitsGeometryOnly(t *testing.T) {
	srv := newTestServer(t)
	srv.planner = kvMemoryStatsPlanner{stats: agent.KVMemoryStats{
		Enabled:       false,
		Backend:       "cpu-ref",
		MemoryClass:   "kv_cache",
		Scope:         "device",
		DType:         "f32",
		BytesPerToken: 4096,
	}}

	text := srv.renderMetrics()
	for _, want := range []string{
		`fak_gateway_kv_memory_enabled{class="kv_cache",scope="device",backend="cpu-ref"} 0`,
		`fak_gateway_kv_memory_dtype_info{class="kv_cache",scope="device",backend="cpu-ref",dtype="f32"} 1`,
		`fak_gateway_kv_memory_bytes_per_token{class="kv_cache",scope="device",backend="cpu-ref"} 4096`,
	} {
		if !strings.Contains(text, want) {
			t.Fatalf("disabled KV memory reporter missing %q\n--- metrics ---\n%s", want, text)
		}
	}
	for _, absent := range []string{
		"fak_gateway_kv_memory_resident_tokens",
		"fak_gateway_kv_memory_evictions_total",
		"fak_gateway_kv_memory_splits_total",
	} {
		if strings.Contains(text, absent) {
			t.Fatalf("disabled KV memory reporter should not emit %q\n--- metrics ---\n%s", absent, text)
		}
	}

	vars := srv.debugVars(time.Now())
	if vars.KVMemory == nil {
		t.Fatal("/debug/vars missing disabled kv_memory geometry")
	}
	if vars.KVMemory.Enabled || vars.KVMemory.Scope != "device" || vars.KVMemory.DType != "f32" || vars.KVMemory.BytesPerToken != 4096 || vars.KVMemory.ResidentTokens != 0 {
		t.Fatalf("debug disabled kv_memory = %+v, want geometry-only disabled snapshot", vars.KVMemory)
	}
}

func TestRequestMemoryMetricsAndDebugVars(t *testing.T) {
	srv := newTestServer(t)
	srv.planner = kvMemoryStatsPlanner{req: agent.RequestMemoryStats{
		Observed:      true,
		Backend:       "vulkan",
		PromptTokens:  13,
		MaxNewTokens:  4,
		PlannedTokens: 17,
		HeadroomRatio: 0.15,
		MemoryPlan: []agent.RequestMemoryDemand{
			{Class: "kv_cache", Scope: "device", DType: "f32", Bytes: 1462272, Detail: "hal-kv-store"},
			{Class: "scratchpad", Scope: "device", DType: "f32", Bytes: 4134912, Detail: "hal-token-scratch"},
			{Class: "kv_cache", Scope: "device", DType: "f32", Bytes: 1024, Detail: "second-kv-row"},
		},
		Capacities: []agent.RequestMemoryCapacity{
			{Scope: "device", TotalBytes: 8 << 30, Known: true},
			{Scope: "host", TotalBytes: 64 << 30, FreeBytes: 48 << 30, Known: true, FreeKnown: true},
		},
	}}

	text := srv.renderMetrics()
	for _, want := range []string{
		`fak_gateway_in_kernel_request_memory_plan_bytes{backend="vulkan",class="kv_cache",scope="device",dtype="f32"} 1463296`,
		`fak_gateway_in_kernel_request_memory_plan_bytes{backend="vulkan",class="scratchpad",scope="device",dtype="f32"} 4134912`,
		`fak_gateway_in_kernel_request_memory_tokens{backend="vulkan",kind="prompt"} 13`,
		`fak_gateway_in_kernel_request_memory_tokens{backend="vulkan",kind="max_new"} 4`,
		`fak_gateway_in_kernel_request_memory_tokens{backend="vulkan",kind="planned"} 17`,
		`fak_gateway_in_kernel_request_memory_headroom_ratio{backend="vulkan"} 0.15`,
		`fak_gateway_in_kernel_request_memory_capacity_known{backend="vulkan",scope="device"} 1`,
		`fak_gateway_in_kernel_request_memory_capacity_free_known{backend="vulkan",scope="device"} 0`,
		`fak_gateway_in_kernel_request_memory_capacity_bytes{backend="vulkan",scope="host",kind="free"} 51539607552`,
		`fak_gateway_in_kernel_request_memory_fit_bytes{backend="vulkan",scope="device",kind="want"} 5598208`,
		`fak_gateway_in_kernel_request_memory_fit_bytes{backend="vulkan",scope="device",kind="budget"} 7301444403`,
		`fak_gateway_in_kernel_request_memory_fit_bytes{backend="vulkan",scope="device",kind="margin"} 7295846195`,
		`fak_gateway_in_kernel_request_memory_fit_bytes{backend="vulkan",scope="host",kind="want"} 0`,
		`fak_gateway_in_kernel_request_memory_fit_bytes{backend="vulkan",scope="host",kind="budget"} 43808666419`,
		`fak_gateway_in_kernel_request_memory_fit_bytes{backend="vulkan",scope="host",kind="margin"} 43808666419`,
	} {
		if !strings.Contains(text, want) {
			t.Fatalf("request memory metrics missing %q\n--- metrics ---\n%s", want, text)
		}
	}

	vars := srv.debugVars(time.Now())
	if vars.RequestMemory == nil {
		t.Fatal("/debug/vars missing request_memory")
	}
	if vars.RequestMemory.Backend != "vulkan" || vars.RequestMemory.PlannedTokens != 17 || len(vars.RequestMemory.MemoryPlan) != 3 {
		t.Fatalf("debug request_memory = %+v, want backend/tokens/raw plan rows", vars.RequestMemory)
	}
	if vars.RequestMemory.MemoryPlan[0].DType != "f32" || vars.RequestMemory.Capacities[1].FreeBytes != 48<<30 {
		t.Fatalf("debug request_memory detail = %+v capacities=%+v", vars.RequestMemory.MemoryPlan, vars.RequestMemory.Capacities)
	}
	if len(vars.RequestMemory.Fit) != 2 || vars.RequestMemory.Fit[0].Scope != "device" ||
		vars.RequestMemory.Fit[0].WantBytes != 5_598_208 || vars.RequestMemory.Fit[0].MarginBytes != 7_295_846_195 ||
		vars.RequestMemory.Fit[1].Scope != "host" || vars.RequestMemory.Fit[1].WantBytes != 0 ||
		vars.RequestMemory.Fit[1].MarginBytes != 43_808_666_419 {
		t.Fatalf("debug request_memory fit = %+v, want device+host fit rows", vars.RequestMemory.Fit)
	}
}
