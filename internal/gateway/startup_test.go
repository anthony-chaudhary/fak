package gateway

import (
	"context"
	"net"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// newTestServerWithConfig builds a Server from an explicit Config over the same
// registered test ABI as newTestServer, so a test can exercise boot-timeline knobs
// (StartTime, StartupPhases) that the zero-config helper does not set.
func newTestServerWithConfig(t *testing.T, cfg Config) *Server {
	t.Helper()
	abi.ResetForTest()
	abi.RegisterRegionBackend(inlineBackend{})
	abi.RegisterEngine("test", echoEngine{})
	abi.RegisterAdjudicator(0, toolAdj{})
	srv, err := New(cfg)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	t.Cleanup(srv.Close)
	return srv
}

// TestStartupMetricsExposeBootTimeline checks that the gateway exposes the one-time
// boot timeline: the New-internal phases are always present, host-supplied pre-New
// phases are merged in, and MarkReady closes time_to_ready with a positive value.
func TestStartupMetricsExposeBootTimeline(t *testing.T) {
	srv := newTestServerWithConfig(t, Config{
		EngineID:      "test",
		Model:         "test-model",
		VDSO:          true,
		StartTime:     time.Now().Add(-50 * time.Millisecond),
		StartupPhases: []StartupPhase{{Name: "policy-load", Dur: 7 * time.Millisecond}},
	})

	// Before MarkReady the timeline is open: ready/time-to-ready read 0.
	pre := srv.renderMetrics()
	for _, want := range []string{
		"# TYPE fak_gateway_time_to_ready_seconds gauge",
		"fak_gateway_time_to_ready_seconds 0",
		"fak_gateway_ready_time_seconds 0",
		`fak_gateway_startup_phase_duration_seconds{phase="policy-load"} 0.007`,
		`fak_gateway_startup_phase_duration_seconds{phase="planner-init"}`,
		`fak_gateway_startup_phase_duration_seconds{phase="vdso-config"}`,
		`fak_gateway_startup_phase_duration_seconds{phase="kernel-init"}`,
	} {
		if !strings.Contains(pre, want) {
			t.Fatalf("pre-ready metrics missing %q\n--- metrics ---\n%s", want, pre)
		}
	}

	srv.MarkReady()
	post := srv.renderMetrics()
	if strings.Contains(post, "fak_gateway_time_to_ready_seconds 0\n") {
		t.Fatalf("time_to_ready still 0 after MarkReady\n--- metrics ---\n%s", post)
	}
	if strings.Contains(post, "fak_gateway_ready_time_seconds 0\n") {
		t.Fatalf("ready_time still 0 after MarkReady\n--- metrics ---\n%s", post)
	}

	// MarkReady is idempotent: a second call must not move the mark.
	firstReady := srv.startup.snapshot().ready
	srv.MarkReady()
	if got := srv.startup.snapshot().ready; !got.Equal(firstReady) {
		t.Fatalf("MarkReady not idempotent: ready moved %v -> %v", firstReady, got)
	}
}

func TestTimeToReadyIsPositiveOnceReadyOnSameTick(t *testing.T) {
	now := time.Unix(1, 0)
	if got := (startupSnapshot{start: now, ready: now}).timeToReady(); got <= 0 {
		t.Fatalf("same-tick ready time must be positive once ready is stamped, got %v", got)
	}
	if got := (startupSnapshot{start: now}).timeToReady(); got != 0 {
		t.Fatalf("unstamped ready time = %v, want 0", got)
	}
	if got := (startupSnapshot{start: now, ready: now.Add(-time.Nanosecond)}).timeToReady(); got != 0 {
		t.Fatalf("ready-before-start time = %v, want 0", got)
	}
}

// TestModelLoadMetricsSuppressedUntilSet asserts the fail-honest default: a serve
// that loaded no weights publishes NO fak_model_load_* series, and setting a
// profile then publishes the full family ordered bottleneck-first.
func TestModelLoadMetricsSuppressedUntilSet(t *testing.T) {
	srv := newTestServer(t)

	if got := srv.renderMetrics(); strings.Contains(got, "fak_model_load_") {
		t.Fatalf("model-load metrics present before any load was set:\n%s", got)
	}

	srv.SetModelLoadProfile(&ModelLoadProfile{
		Source:       "/models/qwen.gguf",
		Mode:         "gguf-lean-q8",
		TotalSeconds: 1.5,
		Bytes:        2_000_000,
		Tensors:      290,
		Bottleneck:   "dequant",
		Phases: []ModelLoadPhase{
			{Phase: "header", Seconds: 0.1, Bytes: 1_000, Tensors: 0},
			{Phase: "dequant", Seconds: 1.2, Bytes: 1_900_000, Tensors: 290},
		},
		MemoryPlan: []ModelLoadMemoryDemand{
			{Class: "weights", Scope: "device", Bytes: 1_750_000, Detail: "gguf-q8-load", DType: "q8_0"},
			{Class: "kv_cache", Scope: "device", Bytes: 240_000, Detail: "hal-kv-store", DType: "f32"},
			{Class: "offload", Scope: "host", Bytes: 10_000, Detail: "expert-weights", DType: "q8_0"},
			{Class: "weights", Scope: "device", Bytes: 250_000, Detail: "scale-buffer", DType: "f32"},
		},
		MemoryCapacities: []ModelLoadMemoryCapacity{
			{Scope: "device", TotalBytes: 8 << 30, Known: true},
			{Scope: "host", TotalBytes: 64 << 30, FreeBytes: 48 << 30, Known: true, FreeKnown: true},
		},
		MemoryHeadroomRatio: 0.15,
	})

	got := srv.renderMetrics()
	for _, want := range []string{
		`fak_model_load_info{source="/models/qwen.gguf",mode="gguf-lean-q8",bottleneck="dequant"} 1`,
		"fak_model_load_duration_seconds 1.5",
		"fak_model_load_bytes 2000000",
		"fak_model_load_tensors 290",
		`fak_model_load_phase_duration_seconds{phase="dequant"} 1.2`,
		`fak_model_load_phase_bytes{phase="dequant"} 1900000`,
		`fak_model_load_phase_tensors{phase="dequant"} 290`,
		`fak_model_load_memory_plan_bytes{class="weights",scope="device"} 2000000`,
		`fak_model_load_memory_plan_bytes{class="kv_cache",scope="device"} 240000`,
		`fak_model_load_memory_plan_bytes{class="offload",scope="host"} 10000`,
		`fak_model_load_memory_plan_dtype_bytes{class="weights",scope="device",dtype="q8_0"} 1750000`,
		`fak_model_load_memory_plan_dtype_bytes{class="weights",scope="device",dtype="f32"} 250000`,
		`fak_model_load_memory_plan_dtype_bytes{class="kv_cache",scope="device",dtype="f32"} 240000`,
		`fak_model_load_memory_plan_dtype_bytes{class="offload",scope="host",dtype="q8_0"} 10000`,
		"fak_model_load_memory_headroom_ratio 0.15",
		`fak_model_load_memory_capacity_known{scope="device"} 1`,
		`fak_model_load_memory_capacity_free_known{scope="device"} 0`,
		`fak_model_load_memory_capacity_known{scope="host"} 1`,
		`fak_model_load_memory_capacity_free_known{scope="host"} 1`,
		`fak_model_load_memory_capacity_bytes{scope="device",kind="total"} 8589934592`,
		`fak_model_load_memory_capacity_bytes{scope="host",kind="free"} 51539607552`,
		`fak_model_load_memory_fit_bytes{scope="device",kind="want"} 2240000`,
		`fak_model_load_memory_fit_bytes{scope="device",kind="budget"} 7301444403`,
		`fak_model_load_memory_fit_bytes{scope="device",kind="margin"} 7299204403`,
		`fak_model_load_memory_fit_bytes{scope="host",kind="want"} 10000`,
		`fak_model_load_memory_fit_bytes{scope="host",kind="budget"} 43808666419`,
		`fak_model_load_memory_fit_bytes{scope="host",kind="margin"} 43808656419`,
	} {
		if !strings.Contains(got, want) {
			t.Fatalf("model-load metrics missing %q\n--- metrics ---\n%s", want, got)
		}
	}

	vars := srv.debugVars(time.Now())
	if vars.ModelLoad == nil {
		t.Fatal("/debug/vars missing model_load after setting a profile")
	}
	if vars.ModelLoad.Mode != "gguf-lean-q8" || vars.ModelLoad.MemoryHeadroomRatio != 0.15 {
		t.Fatalf("debug model_load = %+v, want mode/headroom", vars.ModelLoad)
	}
	if len(vars.ModelLoad.MemoryPlan) != 4 || vars.ModelLoad.MemoryPlan[0].Detail != "gguf-q8-load" {
		t.Fatalf("debug memory plan = %+v, want detailed plan rows", vars.ModelLoad.MemoryPlan)
	}
	if vars.ModelLoad.MemoryPlan[0].DType != "q8_0" || vars.ModelLoad.MemoryPlan[1].DType != "f32" {
		t.Fatalf("debug memory plan dtypes = %+v, want q8_0/f32 rows", vars.ModelLoad.MemoryPlan)
	}
	if len(vars.ModelLoad.MemoryCapacities) != 2 || !vars.ModelLoad.MemoryCapacities[1].FreeKnown {
		t.Fatalf("debug capacities = %+v, want device+host rows with host free known", vars.ModelLoad.MemoryCapacities)
	}
	if len(vars.ModelLoad.MemoryFit) != 2 || vars.ModelLoad.MemoryFit[0].Scope != "device" ||
		vars.ModelLoad.MemoryFit[0].WantBytes != 2_240_000 || vars.ModelLoad.MemoryFit[0].MarginBytes != 7_299_204_403 ||
		!vars.ModelLoad.MemoryFit[1].FreeKnown {
		t.Fatalf("debug memory fit = %+v, want device+host fit rows", vars.ModelLoad.MemoryFit)
	}

	// Bottleneck-first ordering: dequant's phase row precedes header's.
	if iDeq, iHdr := strings.Index(got, `phase_duration_seconds{phase="dequant"}`), strings.Index(got, `phase_duration_seconds{phase="header"}`); iDeq < 0 || iHdr < 0 || iDeq > iHdr {
		t.Fatalf("phases not ordered bottleneck-first (dequant before header): %d vs %d", iDeq, iHdr)
	}

	// Clearing the profile restores the no-weights default.
	srv.SetModelLoadProfile(nil)
	if got := srv.renderMetrics(); strings.Contains(got, "fak_model_load_") {
		t.Fatalf("model-load metrics still present after clearing:\n%s", got)
	}
	if vars := srv.debugVars(time.Now()); vars.ModelLoad != nil {
		t.Fatalf("debug model_load still present after clearing: %+v", vars.ModelLoad)
	}
}

// gaugeValue parses the value of a bare (label-less) gauge line from a rendered
// /metrics body. Returns 0 when the series is absent (matching Prometheus's
// absence-as-zero convention for a gauge that has not yet been set).
func gaugeValue(metrics, name string) float64 {
	prefix := name + " "
	for _, line := range strings.Split(metrics, "\n") {
		line = strings.TrimSpace(line)
		if strings.HasPrefix(line, prefix) {
			v, _ := strconv.ParseFloat(strings.TrimPrefix(line, prefix), 64)
			return v
		}
	}
	return 0
}

// TestStartupUnaccountedReflectsUntimedBoot checks the "is startup fully
// instrumented" gauge: with the named phases under-explaining boot it is positive
// (an untimed gap is visible), and when they over-explain boot it is clamped to 0
// (scrape skew never renders a negative boot).
func TestStartupUnaccountedReflectsUntimedBoot(t *testing.T) {
	// StartTime 100ms ago with only a 10ms phase => ~90ms of boot is unaccounted.
	srv := newTestServerWithConfig(t, Config{
		EngineID:      "test",
		Model:         "test-model",
		StartTime:     time.Now().Add(-100 * time.Millisecond),
		StartupPhases: []StartupPhase{{Name: "policy-load", Dur: 10 * time.Millisecond}},
	})
	srv.MarkReady()
	got := srv.renderMetrics()
	if !strings.Contains(got, "# TYPE fak_gateway_startup_unaccounted_seconds gauge") {
		t.Fatalf("unaccounted family not emitted\n--- metrics ---\n%s", got)
	}
	if u := gaugeValue(got, "fak_gateway_startup_unaccounted_seconds"); u <= 0 {
		t.Fatalf("expected positive unaccounted (phases under-explain boot), got %v\n--- metrics ---\n%s", u, got)
	}

	// A phase larger than total boot must clamp to 0, never go negative.
	srv2 := newTestServerWithConfig(t, Config{
		EngineID:      "test",
		Model:         "test-model",
		StartTime:     time.Now(),
		StartupPhases: []StartupPhase{{Name: "policy-load", Dur: time.Hour}},
	})
	srv2.MarkReady()
	if u := gaugeValue(srv2.renderMetrics(), "fak_gateway_startup_unaccounted_seconds"); u != 0 {
		t.Fatalf("expected unaccounted clamped to 0 (phases over-explain boot), got %v", u)
	}
}

// TestListenAndServeBindsBeforeReady asserts the boot-timeline fix: ListenAndServe
// binds the socket SYNCHRONOUSLY and records a "listener-bind" phase, and MarkReady
// fires strictly AFTER the bind returns. So the moment time_to_ready goes positive
// the socket is already bound — a TCP dial must succeed at that instant (under the
// old async ListenAndServe, MarkReady raced the bind and ready could precede the
// socket being bound).
func TestListenAndServeBindsBeforeReady(t *testing.T) {
	srv := newTestServerWithConfig(t, Config{
		EngineID:  "test",
		Model:     "test-model",
		StartTime: time.Now(),
	})
	// Reserve a free port the OS hands out, then release it for ListenAndServe.
	probe, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("probe listen: %v", err)
	}
	addr := probe.Addr().String()
	probe.Close()

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	errc := make(chan error, 1)
	go func() { errc <- srv.ListenAndServe(ctx, addr) }()

	// Wait until the boot timeline closes (time_to_ready > 0).
	deadline := time.Now().Add(3 * time.Second)
	var ready bool
	for time.Now().Before(deadline) {
		if gaugeValue(srv.renderMetrics(), "fak_gateway_time_to_ready_seconds") > 0 {
			ready = true
			break
		}
		select {
		case err := <-errc:
			t.Fatalf("ListenAndServe returned before ready: %v", err)
		default:
		}
		time.Sleep(5 * time.Millisecond)
	}
	if !ready {
		t.Fatalf("time_to_ready never went positive; ListenAndServe did not come up")
	}

	// The socket must already be bound at the instant ready was observed.
	c, err := net.DialTimeout("tcp", addr, time.Second)
	if err != nil {
		t.Fatalf("dial %s failed right after ready observed (MarkReady raced the bind): %v", addr, err)
	}
	c.Close()

	got := srv.renderMetrics()
	if !strings.Contains(got, `fak_gateway_startup_phase_duration_seconds{phase="listener-bind"}`) {
		t.Fatalf("listener-bind boot phase missing from metrics\n--- metrics ---\n%s", got)
	}
}
