package dispatchcache

import (
	"encoding/json"
	"path/filepath"
	"strconv"
	"testing"
	"time"
)

func BenchmarkDispatchCache(b *testing.B) {
	tempDir := b.TempDir()
	queuePath := filepath.Join(tempDir, "lane-queues.json")
	backlogPath := filepath.Join(tempDir, "backlog.json")
	key := "bench-key"
	now := time.Now().UTC()

	initialLanes := map[string][]int{
		"docs":    {1, 2, 3, 4, 5},
		"gateway": {10, 20, 30},
	}
	baseBacklog := []BacklogIssue{
		{Number: 1, Data: json.RawMessage(`{"title":"issue 1"}`)},
		{Number: 2, Data: json.RawMessage(`{"title":"issue 2"}`)},
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := WriteQueues(queuePath, key, initialLanes, now); err != nil {
			b.Fatal(err)
		}
		if _, _, err := PopLane(queuePath, key, "docs", time.Hour, now); err != nil {
			b.Fatal(err)
		}

		tick := now.Add(time.Duration(i) * time.Second)
		delta := []BacklogIssue{
			{Number: 100 + (i % 10), Data: json.RawMessage(`{"title":"delta ` + strconv.Itoa(i) + `"}`)},
		}
		if _, err := SyncBacklog(backlogPath, key, tick, baseBacklog, delta, nil); err != nil {
			b.Fatal(err)
		}
	}
}
