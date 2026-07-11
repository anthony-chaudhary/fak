package dispatchcache

import (
	"path/filepath"
	"testing"
	"time"
)

func TestPersistedLaneQueueSurvivesRestartAndPopsHead(t *testing.T) {
	path := filepath.Join(t.TempDir(), "lane-queues.json")
	now := time.Unix(100, 0).UTC()
	if err := WriteQueues(path, "key-a", map[string][]int{"docs": {9, 7}}, now); err != nil {
		t.Fatal(err)
	}
	got, ok, err := PopLane(path, "key-a", "docs", time.Hour, now.Add(time.Minute))
	if err != nil || !ok || got != 9 {
		t.Fatalf("pop=%d,%v,%v", got, ok, err)
	}
	q, ok := ReadQueues(path, "key-a", time.Hour, now.Add(time.Minute))
	if !ok || len(q.Lanes["docs"]) != 1 || q.Lanes["docs"][0] != 7 {
		t.Fatalf("queue=%+v ok=%v", q, ok)
	}
}

func TestPersistedLaneQueueInvalidatesOnInputsAndTTL(t *testing.T) {
	path := filepath.Join(t.TempDir(), "lane-queues.json")
	now := time.Unix(100, 0).UTC()
	if err := WriteQueues(path, "key-a", map[string][]int{"docs": {9}}, now); err != nil {
		t.Fatal(err)
	}
	if _, ok := ReadQueues(path, "key-b", time.Hour, now); ok {
		t.Fatal("accepted queue for different routed-input key")
	}
	if _, ok := ReadQueues(path, "key-a", time.Minute, now.Add(time.Minute)); ok {
		t.Fatal("accepted stale queue at TTL boundary")
	}
}
