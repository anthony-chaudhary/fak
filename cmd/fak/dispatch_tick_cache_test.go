package main

import (
	"errors"
	"io"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/dispatchcache"
	"github.com/anthony-chaudhary/fak/internal/dispatchtick"
)

func TestDispatchRoutedBacklogCache(t *testing.T) {
	root := t.TempDir()
	if err := os.WriteFile(filepath.Join(root, "dos.toml"), []byte("[lanes]\ncmd = [\"cmd/fak/**\"]\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	oldView := dispatchTickView
	dispatchTickView = ""
	t.Cleanup(func() { dispatchTickView = oldView })

	now := time.Unix(100, 0)
	oldCache := dispatchRoutedBacklogCache
	dispatchRoutedBacklogCache = dispatchcache.New[dispatchtick.RouterPayload](func() time.Time { return now })
	t.Cleanup(func() { dispatchRoutedBacklogCache = oldCache })

	calls := 0
	stubDispatchIssueFetches(t, nil, func(string, int) ([]dispatchtick.Issue, error) {
		calls++
		return []dispatchtick.Issue{dispatchViewTestIssue(4168)}, nil
	})
	if _, err := dispatchRoutedBeforePrereqHold(root, io.Discard); err != nil {
		t.Fatal(err)
	}
	if _, err := dispatchRoutedBeforePrereqHold(root, io.Discard); err != nil {
		t.Fatal(err)
	}
	if calls != 1 {
		t.Fatalf("fetch calls inside TTL = %d, want 1", calls)
	}
	now = now.Add(dispatchRoutedBacklogTTL)
	if _, err := dispatchRoutedBeforePrereqHold(root, io.Discard); err != nil {
		t.Fatal(err)
	}
	if calls != 2 {
		t.Fatalf("fetch calls after TTL = %d, want 2", calls)
	}
}

func TestPersistDispatchLaneQueuesCarriesScoredOrderAcrossProcess(t *testing.T) {
	root := t.TempDir()
	now := time.Unix(100, 0).UTC()
	payload := dispatchtick.RouterPayload{Lanes: map[string]dispatchtick.RouterLaneGroup{
		"docs": {Issues: []int{42, 17}},
		"cmd":  {Issues: []int{99}},
	}}
	key := dispatchcache.Key(root, "current", 1000)
	if err := persistDispatchLaneQueues(root, key, payload, now); err != nil {
		t.Fatal(err)
	}
	got, ok, err := dispatchcache.PopLane(dispatchLaneQueuePath(root), key, "docs", dispatchLaneQueueTTL, now.Add(time.Minute))
	if err != nil || !ok || got != 42 {
		t.Fatalf("pop=%d,%v,%v", got, ok, err)
	}
	q, ok := dispatchcache.ReadQueues(dispatchLaneQueuePath(root), key, dispatchLaneQueueTTL, now.Add(time.Minute))
	if !ok || len(q.Lanes["docs"]) != 1 || q.Lanes["docs"][0] != 17 {
		t.Fatalf("persisted queue=%+v ok=%v", q, ok)
	}
}

func TestPickDispatchLaneConsumesPersistedExplicitLaneWithoutFetch(t *testing.T) {
	root := t.TempDir()
	oldView := dispatchTickView
	dispatchTickView = ""
	t.Cleanup(func() { dispatchTickView = oldView })
	key := dispatchcache.Key(root, "", 1000)
	if err := dispatchcache.WriteQueues(dispatchLaneQueuePath(root), key, map[string][]int{"docs": {42, 17}}, time.Now()); err != nil {
		t.Fatal(err)
	}
	oldTax := dispatchLoadLaneTaxonomy
	dispatchLoadLaneTaxonomy = func(string) (dispatchtick.LaneTaxonomy, error) {
		return dispatchtick.LaneTaxonomy{Trees: map[string][]string{"docs": {"docs/**"}}}, nil
	}
	t.Cleanup(func() { dispatchLoadLaneTaxonomy = oldTax })
	fetches := 0
	stubDispatchIssueFetches(t, nil, func(string, int) ([]dispatchtick.Issue, error) { fetches++; return nil, nil })
	pick, err := pickDispatchLane(root, io.Discard, "docs", nil, false, "", dispatchGoalProfileThroughput, 0)
	if err != nil {
		t.Fatal(err)
	}
	if fetches != 0 {
		t.Fatalf("full backlog fetches=%d, want 0", fetches)
	}
	if pick.Lane != "docs" || len(pick.Numbers) != 2 || pick.Numbers[0] != 42 || pick.Numbers[1] != 17 {
		t.Fatalf("pick=%+v", pick)
	}
}

func TestPickDispatchLaneFallsBackWhenPersistedQueueStale(t *testing.T) {
	root := t.TempDir()
	oldView := dispatchTickView
	dispatchTickView = ""
	t.Cleanup(func() { dispatchTickView = oldView })
	key := dispatchcache.Key(root, "", 1000)
	if err := dispatchcache.WriteQueues(dispatchLaneQueuePath(root), key, map[string][]int{"docs": {42}}, time.Now().Add(-dispatchLaneQueueTTL)); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(root, "dos.toml"), []byte("[lanes]\ndocs = [\"docs/**\"]\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	fetches := 0
	stubDispatchIssueFetches(t, nil, func(string, int) ([]dispatchtick.Issue, error) {
		fetches++
		return []dispatchtick.Issue{dispatchViewTestIssue(17)}, nil
	})
	if _, err := pickDispatchLane(root, io.Discard, "docs", nil, false, "", dispatchGoalProfileThroughput, 0); err != nil {
		t.Fatal(err)
	}
	if fetches != 1 {
		t.Fatalf("fallback fetches=%d, want 1", fetches)
	}
}

func TestDispatchBacklogIncrementalMergesUpdateAddAndClosure(t *testing.T) {
	root := t.TempDir()
	now := time.Unix(200, 0).UTC()
	key := dispatchcache.Key(root, "", 1000)
	if err := dispatchcache.WriteBacklog(dispatchBacklogSnapshotPath(root), key, now.Add(-time.Minute), dispatchIssueRows([]dispatchtick.Issue{{Number: 1, Title: "old"}, {Number: 2, Title: "close"}})); err != nil {
		t.Fatal(err)
	}
	oldDelta, oldFull := dispatchFetchBacklogDeltaIssues, dispatchFetchBacklogIssues
	dispatchFetchBacklogDeltaIssues = func(string, time.Time, int) (dispatchBacklogDelta, error) {
		return dispatchBacklogDelta{Issues: []dispatchtick.Issue{{Number: 1, Title: "new"}, {Number: 3, Title: "added"}}, Closed: []int{2}, Watermark: now}, nil
	}
	full := 0
	dispatchFetchBacklogIssues = func(string, int) ([]dispatchtick.Issue, error) { full++; return nil, nil }
	t.Cleanup(func() { dispatchFetchBacklogDeltaIssues, dispatchFetchBacklogIssues = oldDelta, oldFull })
	got, err := dispatchFetchBacklogIncremental(root, 1000, now)
	if err != nil {
		t.Fatal(err)
	}
	if full != 0 || len(got) != 2 || got[0].Number != 1 || got[0].Title != "new" || got[1].Number != 3 {
		t.Fatalf("got=%+v full=%d", got, full)
	}
}

func TestDispatchBacklogIncrementalFallsBackToFullRefresh(t *testing.T) {
	root := t.TempDir()
	now := time.Unix(200, 0).UTC()
	key := dispatchcache.Key(root, "", 1000)
	if err := dispatchcache.WriteBacklog(dispatchBacklogSnapshotPath(root), key, now.Add(-time.Minute), nil); err != nil {
		t.Fatal(err)
	}
	oldDelta, oldFull := dispatchFetchBacklogDeltaIssues, dispatchFetchBacklogIssues
	dispatchFetchBacklogDeltaIssues = func(string, time.Time, int) (dispatchBacklogDelta, error) {
		return dispatchBacklogDelta{}, errors.New("delta unavailable")
	}
	dispatchFetchBacklogIssues = func(string, int) ([]dispatchtick.Issue, error) {
		return []dispatchtick.Issue{{Number: 9, Title: "full"}}, nil
	}
	t.Cleanup(func() { dispatchFetchBacklogDeltaIssues, dispatchFetchBacklogIssues = oldDelta, oldFull })
	got, err := dispatchFetchBacklogIncremental(root, 1000, now)
	if err != nil || len(got) != 1 || got[0].Number != 9 {
		t.Fatalf("got=%+v err=%v", got, err)
	}
	snap, ok := dispatchcache.ReadBacklog(dispatchBacklogSnapshotPath(root), key)
	if !ok || !snap.Watermark.Equal(now) {
		t.Fatalf("snap=%+v ok=%v", snap, ok)
	}
}
