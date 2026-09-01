package workerworktree

import (
	"reflect"
	"testing"
	"time"
)

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
