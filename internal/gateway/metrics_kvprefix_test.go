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

func TestRejectedTierMetricDeterminism(t *testing.T) {
	cacheobs.Default.ObserveTier(cacheobs.TierAccess{
		Tier:    cacheobs.CacheTier(255),
		Op:      cacheobs.OpRead,
		Outcome: cacheobs.OutcomeHit,
		Backend: cacheobs.BackendMemory,
	})

	render := func() string {
		var b strings.Builder
		writeKVPrefixMetrics(&b)
		return b.String()
	}

	first := render()
	second := render()
	if first != second {
		t.Fatalf("same rejected-tier input rendered different bytes:\n--- first ---\n%s--- second ---\n%s", first, second)
	}
}

func TestMetricsExposesRejectedTierAccesses(t *testing.T) {
	srv := newTestServer(t)
	const metric = "fak_gateway_kv_prefix_tier_accesses_rejected_total"

	before := cacheobs.Default.Snapshot()
	beforeTier := cacheobs.Default.TierSnapshot()
	beforeText := srv.renderMetrics()
	for _, want := range []string{
		"# HELP " + metric + " ",
		"# TYPE " + metric + " counter",
	} {
		if !strings.Contains(beforeText, want) {
			t.Fatalf("metrics missing %q\n--- metrics ---\n%s", want, beforeText)
		}
	}
	if got := metricUint64(t, beforeText, metric); got != before.RejectedTierAccesses {
		t.Fatalf("initial scraped rejected tier accesses = %d, snapshot = %d", got, before.RejectedTierAccesses)
	}

	cacheobs.Default.ObserveTier(cacheobs.TierAccess{
		Tier:    cacheobs.CacheTier(255),
		Op:      cacheobs.OpRead,
		Outcome: cacheobs.OutcomeHit,
		Backend: cacheobs.BackendMemory,
	})

	after := cacheobs.Default.Snapshot()
	afterTier := cacheobs.Default.TierSnapshot()
	if after.RejectedTierAccesses != before.RejectedTierAccesses+1 {
		t.Fatalf("rejected tier accesses did not advance by one: before=%d after=%d", before.RejectedTierAccesses, after.RejectedTierAccesses)
	}
	if after.Turns != before.Turns || after.PromptTokens != before.PromptTokens || after.ReusedTokens != before.ReusedTokens ||
		afterTier.Total.Requests != beforeTier.Total.Requests {
		t.Fatalf("invalid tier access changed accepted denominators: stats before=%+v after=%+v tier requests before=%d after=%d",
			before, after, beforeTier.Total.Requests, afterTier.Total.Requests)
	}

	afterText := srv.renderMetrics()
	if strings.Contains(afterText, metric+"{") {
		t.Fatalf("rejected tier counter must be unlabeled:\n%s", afterText)
	}
	if got := metricUint64(t, afterText, metric); got != after.RejectedTierAccesses {
		t.Fatalf("scraped rejected tier accesses = %d, snapshot = %d", got, after.RejectedTierAccesses)
	}
}

func metricUint64(t *testing.T, text, name string) uint64 {
	t.Helper()
	line := metricLine(text, name)
	if line == "" {
		t.Fatalf("no %s line:\n%s", name, text)
	}
	value := strings.TrimSpace(strings.TrimPrefix(line, name))
	if strings.ContainsAny(value, "{}") {
		t.Fatalf("%s row has labels or raw dimensions: %q", name, line)
	}
	n, err := strconv.ParseUint(value, 10, 64)
	if err != nil {
		t.Fatalf("parse %q as uint64: %v", line, err)
	}
	return n
}

// TestMetricsExposesKVPrefixBySource covers the provenance axis (#3896): the same in-kernel
// prompt tokens split by WHERE they were served from, orthogonal to the reuse-depth family.
// The process-global tap may carry counts from sibling tests, so the family-present check
// asserts the three source series exist, and the live-read asserts an ObserveBySource on the
// external-transfer bucket (the disaggregation dividend) moves the scraped counter.
func TestMetricsExposesKVPrefixBySource(t *testing.T) {
	srv := newTestServer(t)

	for _, want := range []string{
		"# TYPE fak_gateway_kv_prefix_prompt_tokens_by_source_total counter",
		`fak_gateway_kv_prefix_prompt_tokens_by_source_total{source="local_compute"} `,
		`fak_gateway_kv_prefix_prompt_tokens_by_source_total{source="local_cache_hit"} `,
		`fak_gateway_kv_prefix_prompt_tokens_by_source_total{source="external_kv_transfer"} `,
	} {
		if text := srv.renderMetrics(); !strings.Contains(text, want) {
			t.Fatalf("metrics missing %q\n--- metrics ---\n%s", want, text)
		}
	}

	// Live read: booking external-transfer tokens (the disaggregation dividend) must move the
	// external_kv_transfer series, and the scraped value must be >= what the tap now holds.
	before := cacheobs.Default.SourceSnapshot()
	cacheobs.Default.ObserveBySource(cacheobs.SourceExternalTransfer, 512)
	after := cacheobs.Default.SourceSnapshot()
	if after.ExternalTransferTokens != before.ExternalTransferTokens+512 {
		t.Fatalf("external-transfer bucket did not rise by 512: before=%d after=%d", before.ExternalTransferTokens, after.ExternalTransferTokens)
	}

	line := metricLine(srv.renderMetrics(), `fak_gateway_kv_prefix_prompt_tokens_by_source_total{source="external_kv_transfer"}`)
	if line == "" {
		t.Fatalf("no external_kv_transfer line after observe:\n%s", srv.renderMetrics())
	}
	n, err := strconv.Atoi(strings.TrimSpace(line[strings.LastIndex(line, "}")+1:]))
	if err != nil {
		t.Fatalf("parse %q: %v", line, err)
	}
	if uint64(n) < after.ExternalTransferTokens {
		t.Fatalf("scraped external-transfer %d < observed %d", n, after.ExternalTransferTokens)
	}
}

type kvMemoryStatsPlanner struct {
	stats        agent.KVMemoryStats
	req          agent.RequestMemoryStats
	oomRetry     agent.InKernelOOMRetryStats
	pressureTrim agent.InKernelMemoryPressureTrimStats
}

func (p kvMemoryStatsPlanner) Complete(context.Context, []agent.Message, []agent.ToolDef, ...agent.SampleOpt) (*agent.Completion, error) {
	return &agent.Completion{Message: agent.Message{Role: agent.RoleAssistant, Content: "ok"}}, nil
}

func (p kvMemoryStatsPlanner) Model() string { return "kv-memory-test" }

func (p kvMemoryStatsPlanner) KVMemoryStats() agent.KVMemoryStats { return p.stats }

func (p kvMemoryStatsPlanner) RequestMemoryStats() agent.RequestMemoryStats { return p.req }

func (p kvMemoryStatsPlanner) InKernelOOMRetryStats() agent.InKernelOOMRetryStats {
	return p.oomRetry
}

func (p kvMemoryStatsPlanner) InKernelMemoryPressureTrimStats() agent.InKernelMemoryPressureTrimStats {
	return p.pressureTrim
}

func TestKVMemoryMetricsSuppressedWithoutReporter(t *testing.T) {
	srv := newTestServer(t)
	if text := srv.renderMetrics(); strings.Contains(text, "fak_gateway_kv_memory_") {
		t.Fatalf("resident KV memory metrics should be absent for a non-reporting planner:\n%s", text)
	}
	if vars := srv.debugVars(time.Now()); vars.KVMemory != nil {
		t.Fatalf("debug kv_memory should be absent for a non-reporting planner: %+v", vars.KVMemory)
	}
}

func TestInKernelPressureTrimMetricsAndDebugVars(t *testing.T) {
	srv := newTestServer(t)
	srv.planner = kvMemoryStatsPlanner{pressureTrim: agent.InKernelMemoryPressureTrimStats{
		Backend: "vulkan",
		Rows: []agent.InKernelMemoryPressureTrimClassStats{
			{
				Scope:           "device",
				Class:           "kv_cache",
				Reason:          "capacity_precheck",
				Attempts:        2,
				Trimmed:         1,
				NoHooks:         1,
				Resolved:        1,
				LastWantBytes:   900,
				LastBudgetBytes: 850,
				LastMarginBytes: -50,
			},
		},
	}}

	text := srv.renderMetrics()
	for _, want := range []string{
		`fak_gateway_in_kernel_memory_pressure_trim_total{backend="vulkan",scope="device",class="kv_cache",reason="capacity_precheck",outcome="attempted"} 2`,
		`fak_gateway_in_kernel_memory_pressure_trim_total{backend="vulkan",scope="device",class="kv_cache",reason="capacity_precheck",outcome="trimmed"} 1`,
		`fak_gateway_in_kernel_memory_pressure_trim_total{backend="vulkan",scope="device",class="kv_cache",reason="capacity_precheck",outcome="no_hooks"} 1`,
		`fak_gateway_in_kernel_memory_pressure_trim_total{backend="vulkan",scope="device",class="kv_cache",reason="capacity_precheck",outcome="resolved"} 1`,
		`fak_gateway_in_kernel_memory_pressure_trim_last_bytes{backend="vulkan",scope="device",class="kv_cache",reason="capacity_precheck",kind="want"} 900`,
		`fak_gateway_in_kernel_memory_pressure_trim_last_bytes{backend="vulkan",scope="device",class="kv_cache",reason="capacity_precheck",kind="budget"} 850`,
		`fak_gateway_in_kernel_memory_pressure_trim_last_bytes{backend="vulkan",scope="device",class="kv_cache",reason="capacity_precheck",kind="margin"} -50`,
	} {
		if !strings.Contains(text, want) {
			t.Fatalf("pressure trim metrics missing %q\n--- metrics ---\n%s", want, text)
		}
	}

	vars := srv.debugVars(time.Now())
	rows := vars.Metrics.InKernelPressureTrim
	if len(rows) != 1 || rows[0].Backend != "vulkan" || rows[0].Class != "kv_cache" ||
		rows[0].Reason != "capacity_precheck" || rows[0].Attempts != 2 ||
		rows[0].Trimmed != 1 || rows[0].NoHooks != 1 || rows[0].Resolved != 1 ||
		rows[0].LastMarginBytes != -50 {
		t.Fatalf("debug pressure trim rows = %+v", rows)
	}
}

func TestInKernelOOMRetryMetricsAndDebugVars(t *testing.T) {
	srv := newTestServer(t)
	srv.planner = kvMemoryStatsPlanner{oomRetry: agent.InKernelOOMRetryStats{
		Backend: "vulkan",
		Rows: []agent.InKernelOOMRetryClassStats{
			{
				Class:           "scratchpad",
				Attempts:        2,
				Successes:       1,
				Failures:        1,
				LastFailedBytes: 1 << 20,
				LastSite:        "transient-scratch",
			},
		},
	}}

	text := srv.renderMetrics()
	for _, want := range []string{
		`fak_gateway_in_kernel_oom_retry_total{backend="vulkan",class="scratchpad",outcome="attempted"} 2`,
		`fak_gateway_in_kernel_oom_retry_total{backend="vulkan",class="scratchpad",outcome="succeeded"} 1`,
		`fak_gateway_in_kernel_oom_retry_total{backend="vulkan",class="scratchpad",outcome="failed"} 1`,
		`fak_gateway_in_kernel_oom_retry_last_failed_bytes{backend="vulkan",class="scratchpad"} 1048576`,
	} {
		if !strings.Contains(text, want) {
			t.Fatalf("OOM retry metrics missing %q\n--- metrics ---\n%s", want, text)
		}
	}

	vars := srv.debugVars(time.Now())
	if len(vars.Metrics.InKernelOOMRetries) != 1 {
		t.Fatalf("debug OOM retry rows = %+v, want one row", vars.Metrics.InKernelOOMRetries)
	}
	got := vars.Metrics.InKernelOOMRetries[0]
	if got.Backend != "vulkan" || got.Class != "scratchpad" || got.Attempts != 2 ||
		got.Successes != 1 || got.Failures != 1 || got.LastFailedBytes != 1<<20 ||
		got.LastSite != "transient-scratch" {
		t.Fatalf("debug OOM retry row = %+v, want vulkan/scratchpad 2/1/1 transient-scratch", got)
	}
}

func TestKVMemoryMetricsAndDebugVars(t *testing.T) {
	srv := newTestServer(t)
	srv.planner = kvMemoryStatsPlanner{stats: agent.KVMemoryStats{
		Enabled:               true,
		Backend:               "radixkv",
		MemoryClass:           "kv_cache",
		Scope:                 "host",
		DType:                 "f32",
		BytesPerToken:         6144,
		ResidentTokens:        42,
		ResidentBytes:         258048,
		CapacityKnown:         true,
		CapacityFreeKnown:     true,
		CapacityTotalBytes:    2 << 20,
		CapacityFreeBytes:     790528,
		HeadroomRatio:         0.25,
		FitBudgetBytes:        786432,
		FitMarginBytes:        528384,
		BudgetTokens:          64,
		LRUTokens:             18,
		MaxDepthTokens:        21,
		Nodes:                 3,
		Leaves:                2,
		Evictions:             4,
		PolicyEvictions:       1,
		Splits:                5,
		L1DeviceResidentBytes: 1000,
		L1HostResidentBytes:   200,
		L2HostResidentBytes:   3000,
		L2HostCapacityBytes:   4096,
		L1Hits:                4,
		L1Misses:              3,
		L1Faults:              1,
		L1HitTokens:           40,
		L2Hits:                2,
		L2Misses:              1,
		L2Faults:              1,
		L2HitTokens:           20,
		L2StageBytes:          5000,
		L2RestoreBytes:        4000,
		L2Evictions:           2,
		L3Enabled:             true,
		L3ReferencedBytes:     9000,
		L3Hits:                3,
		L3Misses:              2,
		L3Faults:              1,
		L3HitTokens:           30,
		L3StageBytes:          12000,
		L3RestoreBytes:        11000,
		L3StageNanos:          250000000,
		L3RestoreNanos:        500000000,
		L3StageFaults:         1,
		L3RestoreFaults:       2,
	}}

	text := srv.renderMetrics()
	for _, want := range []string{
		`fak_gateway_kv_memory_enabled{class="kv_cache",scope="host",backend="radixkv"} 1`,
		`fak_gateway_kv_memory_dtype_info{class="kv_cache",scope="host",backend="radixkv",dtype="f32"} 1`,
		`fak_gateway_kv_memory_bytes_per_token{class="kv_cache",scope="host",backend="radixkv"} 6144`,
		`fak_gateway_kv_memory_headroom_ratio{class="kv_cache",scope="host",backend="radixkv"} 0.25`,
		`fak_gateway_kv_memory_capacity_known{class="kv_cache",scope="host",backend="radixkv"} 1`,
		`fak_gateway_kv_memory_capacity_free_known{class="kv_cache",scope="host",backend="radixkv"} 1`,
		`fak_gateway_kv_memory_capacity_bytes{class="kv_cache",scope="host",backend="radixkv",kind="total"} 2097152`,
		`fak_gateway_kv_memory_capacity_bytes{class="kv_cache",scope="host",backend="radixkv",kind="free"} 790528`,
		`fak_gateway_kv_memory_fit_bytes{class="kv_cache",scope="host",backend="radixkv",kind="want"} 258048`,
		`fak_gateway_kv_memory_fit_bytes{class="kv_cache",scope="host",backend="radixkv",kind="budget"} 786432`,
		`fak_gateway_kv_memory_fit_bytes{class="kv_cache",scope="host",backend="radixkv",kind="margin"} 528384`,
		`fak_gateway_kv_memory_resident_tokens{class="kv_cache",scope="host",backend="radixkv"} 42`,
		`fak_gateway_kv_memory_resident_bytes{class="kv_cache",scope="host",backend="radixkv"} 258048`,
		`fak_gateway_kv_memory_lru_tokens{class="kv_cache",scope="host",backend="radixkv"} 18`,
		`fak_gateway_kv_memory_budget_tokens{class="kv_cache",scope="host",backend="radixkv"} 64`,
		`fak_gateway_kv_memory_evictions_total{class="kv_cache",scope="host",backend="radixkv",kind="lru"} 4`,
		`fak_gateway_kv_memory_evictions_total{class="kv_cache",scope="host",backend="radixkv",kind="policy"} 1`,
		`fak_gateway_kv_memory_splits_total{class="kv_cache",scope="host",backend="radixkv"} 5`,
		`fak_gateway_kv_prefix_tier_resident_bytes{backend="radixkv",tier="device_l1",scope="device"} 1000`,
		`fak_gateway_kv_prefix_tier_resident_bytes{backend="radixkv",tier="device_l1",scope="host_metadata"} 200`,
		`fak_gateway_kv_prefix_tier_resident_bytes{backend="radixkv",tier="host_dram_l2",scope="host"} 3000`,
		`fak_gateway_kv_prefix_tier_resident_bytes{backend="radixkv",tier="remote_http_l3",scope="remote_referenced"} 9000`,
		`fak_gateway_kv_prefix_tier_capacity_bytes{backend="radixkv",tier="host_dram_l2"} 4096`,
		`fak_gateway_kv_prefix_tier_lookups_total{backend="radixkv",tier="device_l1",outcome="hit"} 4`,
		`fak_gateway_kv_prefix_tier_lookups_total{backend="radixkv",tier="device_l1",outcome="miss"} 3`,
		`fak_gateway_kv_prefix_tier_lookups_total{backend="radixkv",tier="device_l1",outcome="fault"} 1`,
		`fak_gateway_kv_prefix_tier_lookups_total{backend="radixkv",tier="host_dram_l2",outcome="hit"} 2`,
		`fak_gateway_kv_prefix_tier_lookups_total{backend="radixkv",tier="host_dram_l2",outcome="miss"} 1`,
		`fak_gateway_kv_prefix_tier_lookups_total{backend="radixkv",tier="host_dram_l2",outcome="fault"} 1`,
		`fak_gateway_kv_prefix_tier_lookups_total{backend="radixkv",tier="remote_http_l3",outcome="hit"} 3`,
		`fak_gateway_kv_prefix_tier_lookups_total{backend="radixkv",tier="remote_http_l3",outcome="miss"} 2`,
		`fak_gateway_kv_prefix_tier_lookups_total{backend="radixkv",tier="remote_http_l3",outcome="fault"} 1`,
		`fak_gateway_kv_prefix_tier_hit_tokens_total{backend="radixkv",tier="device_l1"} 40`,
		`fak_gateway_kv_prefix_tier_hit_tokens_total{backend="radixkv",tier="host_dram_l2"} 20`,
		`fak_gateway_kv_prefix_tier_hit_tokens_total{backend="radixkv",tier="remote_http_l3"} 30`,
		`fak_gateway_kv_prefix_tier_transfer_bytes_total{backend="radixkv",tier="host_dram_l2",direction="stage"} 5000`,
		`fak_gateway_kv_prefix_tier_transfer_bytes_total{backend="radixkv",tier="host_dram_l2",direction="restore"} 4000`,
		`fak_gateway_kv_prefix_tier_transfer_bytes_total{backend="radixkv",tier="remote_http_l3",direction="stage"} 12000`,
		`fak_gateway_kv_prefix_tier_transfer_bytes_total{backend="radixkv",tier="remote_http_l3",direction="restore"} 11000`,
		`fak_gateway_kv_prefix_tier_transfer_latency_seconds_total{backend="radixkv",tier="remote_http_l3",direction="stage"} 0.25`,
		`fak_gateway_kv_prefix_tier_transfer_latency_seconds_total{backend="radixkv",tier="remote_http_l3",direction="restore"} 0.5`,
		`fak_gateway_kv_prefix_tier_transfer_faults_total{backend="radixkv",tier="remote_http_l3",direction="stage"} 1`,
		`fak_gateway_kv_prefix_tier_transfer_faults_total{backend="radixkv",tier="remote_http_l3",direction="restore"} 2`,
		`fak_gateway_kv_prefix_tier_evictions_total{backend="radixkv",tier="host_dram_l2"} 2`,
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
		vars.KVMemory.LRUTokens != 18 || vars.KVMemory.PolicyEvictions != 1 || vars.KVMemory.DType != "f32" ||
		!vars.KVMemory.CapacityKnown || !vars.KVMemory.CapacityFreeKnown ||
		vars.KVMemory.CapacityTotalBytes != 2<<20 || vars.KVMemory.CapacityFreeBytes != 790528 ||
		vars.KVMemory.FitBudgetBytes != 786432 || vars.KVMemory.FitMarginBytes != 528384 ||
		vars.KVMemory.L1DeviceResidentBytes != 1000 || vars.KVMemory.L1HostResidentBytes != 200 ||
		vars.KVMemory.L2HostResidentBytes != 3000 || vars.KVMemory.L2HostCapacityBytes != 4096 ||
		vars.KVMemory.L1Hits != 4 || vars.KVMemory.L1Misses != 3 || vars.KVMemory.L1Faults != 1 ||
		vars.KVMemory.L1HitTokens != 40 || vars.KVMemory.L2Hits != 2 || vars.KVMemory.L2Misses != 1 ||
		vars.KVMemory.L2Faults != 1 || vars.KVMemory.L2HitTokens != 20 ||
		vars.KVMemory.L2StageBytes != 5000 || vars.KVMemory.L2RestoreBytes != 4000 ||
		vars.KVMemory.L2Evictions != 2 || !vars.KVMemory.L3Enabled || vars.KVMemory.L3ReferencedBytes != 9000 ||
		vars.KVMemory.L3Hits != 3 || vars.KVMemory.L3Misses != 2 || vars.KVMemory.L3Faults != 1 ||
		vars.KVMemory.L3HitTokens != 30 || vars.KVMemory.L3StageBytes != 12000 || vars.KVMemory.L3RestoreBytes != 11000 ||
		vars.KVMemory.L3StageNanos != 250000000 || vars.KVMemory.L3RestoreNanos != 500000000 ||
		vars.KVMemory.L3StageFaults != 1 || vars.KVMemory.L3RestoreFaults != 2 {
		t.Fatalf("debug kv_memory = %+v, want resident/lru/eviction fields", vars.KVMemory)
	}
}

func TestKVMemoryMetricsDisabledReporterEmitsGeometryOnly(t *testing.T) {
	srv := newTestServer(t)
	srv.planner = kvMemoryStatsPlanner{stats: agent.KVMemoryStats{
		Enabled:            false,
		Backend:            "cpu-ref",
		MemoryClass:        "kv_cache",
		Scope:              "device",
		DType:              "f32",
		BytesPerToken:      4096,
		CapacityKnown:      true,
		CapacityTotalBytes: 8192,
		HeadroomRatio:      0.25,
		FitBudgetBytes:     6144,
		FitMarginBytes:     6144,
	}}

	text := srv.renderMetrics()
	for _, want := range []string{
		`fak_gateway_kv_memory_enabled{class="kv_cache",scope="device",backend="cpu-ref"} 0`,
		`fak_gateway_kv_memory_dtype_info{class="kv_cache",scope="device",backend="cpu-ref",dtype="f32"} 1`,
		`fak_gateway_kv_memory_bytes_per_token{class="kv_cache",scope="device",backend="cpu-ref"} 4096`,
		`fak_gateway_kv_memory_capacity_known{class="kv_cache",scope="device",backend="cpu-ref"} 1`,
		`fak_gateway_kv_memory_capacity_free_known{class="kv_cache",scope="device",backend="cpu-ref"} 0`,
		`fak_gateway_kv_memory_capacity_bytes{class="kv_cache",scope="device",backend="cpu-ref",kind="total"} 8192`,
		`fak_gateway_kv_memory_fit_bytes{class="kv_cache",scope="device",backend="cpu-ref",kind="want"} 0`,
		`fak_gateway_kv_memory_fit_bytes{class="kv_cache",scope="device",backend="cpu-ref",kind="budget"} 6144`,
		`fak_gateway_kv_memory_fit_bytes{class="kv_cache",scope="device",backend="cpu-ref",kind="margin"} 6144`,
	} {
		if !strings.Contains(text, want) {
			t.Fatalf("disabled KV memory reporter missing %q\n--- metrics ---\n%s", want, text)
		}
	}
	physicalTierText := strings.ReplaceAll(text, "fak_gateway_kv_prefix_tier_accesses_rejected_total", "")
	for _, absent := range []string{
		"fak_gateway_kv_memory_resident_tokens",
		"fak_gateway_kv_memory_evictions_total",
		"fak_gateway_kv_memory_splits_total",
		"fak_gateway_kv_prefix_tier_",
	} {
		if strings.Contains(physicalTierText, absent) {
			t.Fatalf("disabled KV memory reporter should not emit %q\n--- metrics ---\n%s", absent, text)
		}
	}

	vars := srv.debugVars(time.Now())
	if vars.KVMemory == nil {
		t.Fatal("/debug/vars missing disabled kv_memory geometry")
	}
	if vars.KVMemory.Enabled || vars.KVMemory.Scope != "device" || vars.KVMemory.DType != "f32" ||
		vars.KVMemory.BytesPerToken != 4096 || vars.KVMemory.ResidentTokens != 0 ||
		!vars.KVMemory.CapacityKnown || vars.KVMemory.CapacityFreeKnown ||
		vars.KVMemory.FitBudgetBytes != 6144 || vars.KVMemory.FitMarginBytes != 6144 {
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

func TestRequestMemoryAggregateMetricsAndDebugVars(t *testing.T) {
	srv := newTestServer(t)
	srv.planner = kvMemoryStatsPlanner{req: agent.RequestMemoryStats{
		Observed:      true,
		Backend:       "vulkan",
		PromptTokens:  10,
		MaxNewTokens:  5,
		PlannedTokens: 15,
		HeadroomRatio: 0.10,
		MemoryPlan: []agent.RequestMemoryDemand{
			{Class: "kv_cache", Scope: "device", DType: "f32", Bytes: 100},
			{Class: "scratchpad", Scope: "device", DType: "f32", Bytes: 50},
		},
		Capacities: []agent.RequestMemoryCapacity{
			{Scope: "device", TotalBytes: 1000, FreeBytes: 1000, Known: true, FreeKnown: true},
		},
	}}
	srv.observePlannerRequestMemory()
	srv.planner = kvMemoryStatsPlanner{req: agent.RequestMemoryStats{
		Observed:      true,
		Backend:       "vulkan",
		PromptTokens:  20,
		MaxNewTokens:  7,
		PlannedTokens: 27,
		HeadroomRatio: 0.10,
		MemoryPlan: []agent.RequestMemoryDemand{
			{Class: "activation", Scope: "device", DType: "f32", Bytes: 75},
			{Class: "kv_cache", Scope: "device", DType: "f32", Bytes: 200},
		},
		Capacities: []agent.RequestMemoryCapacity{
			{Scope: "device", TotalBytes: 1000, FreeBytes: 400, Known: true, FreeKnown: true},
		},
	}}
	srv.observePlannerRequestMemory()

	text := srv.renderMetrics()
	for _, want := range []string{
		`fak_gateway_in_kernel_request_memory_observations_total{backend="vulkan"} 2`,
		`fak_gateway_in_kernel_request_memory_plan_observations_total{backend="vulkan",class="kv_cache",scope="device",dtype="f32"} 2`,
		`fak_gateway_in_kernel_request_memory_plan_bytes_total{backend="vulkan",class="kv_cache",scope="device",dtype="f32"} 300`,
		`fak_gateway_in_kernel_request_memory_plan_high_water_bytes{backend="vulkan",class="kv_cache",scope="device",dtype="f32"} 200`,
		`fak_gateway_in_kernel_request_memory_tokens_total{backend="vulkan",kind="planned"} 42`,
		`fak_gateway_in_kernel_request_memory_tokens_high_water{backend="vulkan",kind="prompt"} 20`,
		`fak_gateway_in_kernel_request_memory_fit_observations_total{backend="vulkan",scope="device"} 2`,
		`fak_gateway_in_kernel_request_memory_fit_want_high_water_bytes{backend="vulkan",scope="device"} 275`,
		`fak_gateway_in_kernel_request_memory_fit_margin_low_water_bytes{backend="vulkan",scope="device"} 85`,
	} {
		if !strings.Contains(text, want) {
			t.Fatalf("request memory aggregate metrics missing %q\n--- metrics ---\n%s", want, text)
		}
	}

	vars := srv.debugVars(time.Now())
	var kv *debugRequestMemoryMetricVars
	for i := range vars.Metrics.RequestMemory {
		row := &vars.Metrics.RequestMemory[i]
		if row.Class == "kv_cache" {
			kv = row
			break
		}
	}
	if kv == nil || kv.Observations != 2 || kv.TotalBytes != 300 || kv.HighWaterBytes != 200 {
		t.Fatalf("debug request_memory aggregate kv row = %+v, want observations=2 total=300 high=200", kv)
	}
	if len(vars.Metrics.RequestMemoryFit) != 1 || vars.Metrics.RequestMemoryFit[0].WantHighWater != 275 ||
		vars.Metrics.RequestMemoryFit[0].MarginLowWater != 85 || !vars.Metrics.RequestMemoryFit[0].MarginLowWaterOK {
		t.Fatalf("debug request_memory_fit = %+v, want high=275 low_margin=85", vars.Metrics.RequestMemoryFit)
	}
	if len(vars.Metrics.RequestMemoryTokens) != 3 {
		t.Fatalf("debug request_memory_tokens = %+v, want prompt/max_new/planned rows", vars.Metrics.RequestMemoryTokens)
	}
}
