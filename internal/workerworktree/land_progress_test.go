package workerworktree

import (
	"context"
	"encoding/json"
	"os"
	"os/exec"
	"reflect"
	"runtime"
	"testing"
	"time"
)

// fak-test:runtime fast est=100ms lane=default
func TestLandCostReportsReapedChildResources(t *testing.T) {
	const childEnv = "FAK_TEST_LAND_COST_CHILD"
	if os.Getenv(childEnv) == "1" {
		// Touch at most 8 MiB and do fixed work, not a timing-sensitive burn.
		pages := make([]byte, 8<<20)
		for i := range pages {
			pages[i] = byte(i*31 + 7)
		}
		runtime.KeepAlive(pages)
		return
	}
	if runtime.GOOS != "linux" {
		t.Skip("waited-child resource sampling is Linux-only")
	}
	t.Setenv(IsolatedLandEnv, "0")
	executable, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	child := exec.CommandContext(ctx, executable, "-test.run=^TestLandCostReportsReapedChildResources$")
	child.Env = append(os.Environ(), childEnv+"=1")
	child.WaitDelay = time.Second
	g := newFakeGit().reply("diff", 0, "diff --git a/x b/x\n@@\n-old\n+new\n")
	result := Land("/trunk", "/wt", "base", "", nil, func(string) (bool, string) {
		if out, err := child.CombinedOutput(); err != nil {
			t.Fatalf("bounded child: %v: %s", err, out)
		}
		return false, "resource witness completed; fixture refuses source mutation"
	}, g.run)
	if result.OK || result.Cost == nil || child.ProcessState == nil {
		t.Fatalf("real Land terminal lost resource attribution: %+v", result)
	}
	cost := landCostJSONForTest(t, result)
	assertLandResourceScopesForTest(t, cost)
	if cost["resource_state"] != "available" || cost["child_resource_state"] != "available" {
		t.Fatalf("supported resource axes are not available: %v", cost)
	}
	for _, key := range []string{"cpu_time_ms", "peak_rss_bytes", "child_cpu_time_ms", "child_peak_rss_bytes"} {
		if value, ok := cost[key].(float64); !ok || value < 0 {
			t.Fatalf("missing or invalid numeric axis %s: %v", key, cost)
		}
	}
	// CPU may legitimately round to zero milliseconds. Check accounting of the
	// waited child, not an arbitrary positive-duration threshold.
	childMS := (child.ProcessState.UserTime() + child.ProcessState.SystemTime()).Milliseconds()
	if cost["child_cpu_time_ms"].(float64) < float64(childMS-1) {
		t.Fatalf("waited child CPU was lost: cost=%v child_ms=%d", cost, childMS)
	}
	// SysUsage's Unix type is inspected only on Linux so this test still builds
	// on unsupported platforms. Linux Maxrss is KiB, whereas the receipt is bytes.
	usage := reflect.ValueOf(child.ProcessState.SysUsage())
	if usage.Kind() != reflect.Pointer || usage.IsNil() {
		t.Fatalf("missing Linux child rusage: %v", child.ProcessState.SysUsage())
	}
	maxRSS := usage.Elem().FieldByName("Maxrss")
	if !maxRSS.IsValid() || !maxRSS.CanInt() || maxRSS.Int() <= 0 {
		t.Fatal("missing Linux child Maxrss")
	}
	if cost["child_peak_rss_bytes"].(float64) < float64(maxRSS.Int()*1024) {
		t.Fatalf("largest waited-child RSS lost or not converted from KiB: %v", cost)
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestLandCostScopesAndUnobservedChildren(t *testing.T) {
	observed := currentLandResources()
	for _, tc := range []struct {
		name                 string
		start, end           bool
		state                string
		wantCPU, wantPeakRSS bool
	}{
		{name: "neither endpoint", state: "unavailable"},
		{name: "missing start", end: true, state: "partial", wantPeakRSS: true},
		{name: "missing end", start: true, state: "unavailable"},
		{name: "both endpoints", start: true, end: true, state: "available", wantCPU: true, wantPeakRSS: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			calls := 0
			tracker := newLandProgressTracker(landConfig{resources: func() landResourceSample {
				calls++
				sample := landResourceSample{}
				if calls == 1 && tc.start || calls == 2 && tc.end {
					sample = observed
				}
				sample.cpuTime = time.Duration(100+7*(calls-1)) * time.Millisecond
				sample.peakRSSBytes = int64(2048 * calls)
				sample.cpuAvailable, sample.rssAvailable = true, true
				return sample
			}})
			cost := landCostJSONForTest(t, Result{Cost: tracker.receipt()})
			assertLandResourceScopesForTest(t, cost)
			if cost["cpu_time_ms"] != float64(7) || cost["peak_rss_bytes"] != float64(4096) || cost["resource_state"] != "available" {
				t.Fatalf("existing self interval CPU/lifetime RSS changed: %v", cost)
			}
			state, wantCPU, wantRSS := tc.state, tc.wantCPU, tc.wantPeakRSS
			if runtime.GOOS != "linux" {
				state, wantCPU, wantRSS = "unavailable", false, false
			}
			if cost["child_resource_state"] != state {
				t.Fatalf("child availability = %v, want %s: %v", cost["child_resource_state"], state, cost)
			}
			_, hasCPU := cost["child_cpu_time_ms"]
			_, hasRSS := cost["child_peak_rss_bytes"]
			if hasCPU != wantCPU || hasRSS != wantRSS {
				t.Fatalf("unobserved child axes must be absent, not zero: %v", cost)
			}
			if state != "available" {
				if reason, ok := cost["child_resource_reason"].(string); !ok || reason == "" {
					t.Fatalf("missing child unavailability reason: %v", cost)
				}
			}
			if wantCPU && cost["child_cpu_time_ms"] != float64(0) {
				t.Fatalf("equal cumulative snapshots must accrue zero CPU: %v", cost)
			}
		})
	}
}

func landCostJSONForTest(t *testing.T, result Result) map[string]any {
	t.Helper()
	data, err := json.Marshal(result)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("RESOURCE_RECEIPT %s", data)
	var report struct {
		Cost map[string]any `json:"cost"`
	}
	if err := json.Unmarshal(data, &report); err != nil || report.Cost == nil {
		t.Fatalf("decode terminal report: %v: %s", err, data)
	}
	return report.Cost
}

func assertLandResourceScopesForTest(t *testing.T, cost map[string]any) {
	t.Helper()
	for key, want := range map[string]string{
		"cpu_time_scope":       "process-self-execution-during-land",
		"peak_rss_scope":       "process-self-lifetime-maximum",
		"child_cpu_time_scope": "process-waited-children-cpu-accrued-during-land",
		"child_peak_rss_scope": "process-largest-waited-child-lifetime-maximum",
	} {
		if cost[key] != want {
			t.Errorf("%s = %v, want %q", key, cost[key], want)
		}
	}
}

func TestLandProgressTrackerFakeClockAndPhases(t *testing.T) {
	base := time.Date(2026, 8, 27, 12, 0, 0, 0, time.UTC)
	tick := -1
	now := func() time.Time {
		tick++
		return base.Add(time.Duration(tick) * 10 * time.Millisecond)
	}
	resourceCall := 0
	resources := func() landResourceSample {
		resourceCall++
		return landResourceSample{
			cpuTime:      time.Duration(100+7*(resourceCall-1)) * time.Millisecond,
			peakRSSBytes: int64(1024 * resourceCall), cpuAvailable: true, rssAvailable: true,
		}
	}
	var events []LandProgressEvent
	tracker := newLandProgressTracker(landConfig{now: now, resources: resources, progress: func(event LandProgressEvent) {
		events = append(events, event)
	}})
	admission := tracker.start("admission", 0)
	tracker.setPatchScope(2, 512)
	tracker.complete(admission)
	commit := tracker.start("commit", 0)
	tracker.setCache("fresh-isolated-index", false)
	tracker.complete(commit)
	receipt := tracker.receipt()

	var got []string
	for _, event := range events {
		got = append(got, event.Phase+":"+event.Status)
	}
	want := []string{"admission:started", "admission:completed", "commit:started", "commit:completed"}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("event order = %v, want %v", got, want)
	}
	if receipt.WallTimeMS != 50 || receipt.PhaseTotalMS != 20 || receipt.UnattributedMS != 30 {
		t.Fatalf("time reconciliation = %+v", receipt)
	}
	if receipt.CPUTimeMS == nil || *receipt.CPUTimeMS != 7 || receipt.PeakRSSBytes == nil || *receipt.PeakRSSBytes != 2048 || receipt.ResourceState != "available" {
		t.Fatalf("resource attribution = %+v", receipt)
	}
	if receipt.PatchScopeFiles != 2 || receipt.PatchScopeBytes != 512 || receipt.CacheState != "fresh-isolated-index" || receipt.Reused {
		t.Fatalf("work attribution = %+v", receipt)
	}
	if receipt.SlowestPhase == "" || len(receipt.Phases) != 2 {
		t.Fatalf("phase attribution = %+v", receipt)
	}
}

func TestLandCostMakesResourceProbeFailureVisible(t *testing.T) {
	tracker := newLandProgressTracker(landConfig{resources: func() landResourceSample {
		return landResourceSample{reason: "probe failed"}
	}})
	receipt := tracker.receipt()
	if receipt.ResourceState != "unavailable" || receipt.ResourceReason != "probe failed" {
		t.Fatalf("resource failure not visible: %+v", receipt)
	}
	if receipt.CPUTimeMS != nil || receipt.PeakRSSBytes != nil {
		t.Fatalf("unavailable resources must omit numeric readings: %+v", receipt)
	}
}

func TestLandProgressSurfacesProspectiveValidation(t *testing.T) {
	t.Setenv(IsolatedLandEnv, "0")
	g := newFakeGit().reply("diff", 0, "diff --git a/x b/x\n@@\n-old\n+new\n")
	base := time.Unix(0, 0)
	tick := -1
	now := func() time.Time {
		tick++
		return base.Add(time.Duration(tick) * time.Millisecond)
	}
	var events []LandProgressEvent
	res := Land("/trunk", "/wt", "base", "/tmp/msg", nil, func(string) (bool, string) {
		return false, "prospective build failed"
	}, g.run, WithLandClock(now), WithLandProgress(func(event LandProgressEvent) {
		events = append(events, event)
	}), withLandResourceSampler(func() landResourceSample { return landResourceSample{} }))
	if res.OK || res.Cost == nil {
		t.Fatalf("refused land must retain terminal cost: %+v", res)
	}
	var got []string
	for _, event := range events {
		got = append(got, event.Phase+":"+event.Status)
	}
	want := []string{
		"admission:started", "admission:completed",
		"prospective-validation:started", "prospective-validation:completed",
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("validation progress = %v, want %v", got, want)
	}
	if res.Cost.PatchScopeBytes == 0 || res.Cost.CacheState != "not-built" {
		t.Fatalf("refusal cost = %+v", res.Cost)
	}
}

func TestIsolatedLandProgressFollowsStateMachineOrder(t *testing.T) {
	var events []LandProgressEvent
	cfg := landConfig{
		progress:  func(event LandProgressEvent) { events = append(events, event) },
		resources: func() landResourceSample { return landResourceSample{} },
	}
	tracker := newLandProgressTracker(cfg)
	cfg.tracker = tracker
	g := isolatedHappyFake()
	msg := writeMsg(t, "feat(x): progress (fak x)")
	res, handled := landIsolated("/trunk", "/wt", "diff --git a/x b/x\n@@\n-o\n+n\n", msg, []string{"x"}, g.run, g.runEnv, cfg)
	if !handled || !res.OK {
		t.Fatalf("isolated land = %+v, handled=%v", res, handled)
	}
	var started []string
	for _, event := range events {
		if event.Status == "started" {
			started = append(started, event.Phase)
		}
	}
	want := []string{
		"isolated-admission",
		"index-construction",
		"candidate-scope-fence",
		"commit-construction",
		"recovery-ref-publication",
		"trunk-cas",
		"working-tree-sync",
	}
	if !reflect.DeepEqual(started, want) {
		t.Fatalf("started phase order = %v, want %v", started, want)
	}
	if len(events) != 2*len(want) {
		t.Fatalf("events = %d, want %d bounded started/completed events: %+v", len(events), 2*len(want), events)
	}
}
