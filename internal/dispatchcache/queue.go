package dispatchcache

import (
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"time"
)

// Invariant: dispatch queue cache persistence is fail-closed and bounded; invalid schemas or expired records return empty snapshots.

// QueueSchema identifies the schema version for persisted lane queue snapshots.
const QueueSchema = "fak.dispatch-lane-queues.v1"

// QueueSnapshot records the state of lane queues at a specific generation point.
type QueueSnapshot struct {
	Schema      string           `json:"schema"`
	Key         string           `json:"key"`
	GeneratedAt time.Time        `json:"generated_at"`
	Lanes       map[string][]int `json:"lanes"`
}

// WriteQueues writes a snapshot of lane queues atomically to disk via a temporary file.
func WriteQueues(path, key string, lanes map[string][]int, now time.Time) error {
	if path == "" || key == "" {
		return errors.New("dispatchcache: queue path and key are required")
	}
	copyLanes := make(map[string][]int, len(lanes))
	for lane, issues := range lanes {
		copyLanes[lane] = append([]int(nil), issues...)
	}
	b, err := json.Marshal(QueueSnapshot{Schema: QueueSchema, Key: key, GeneratedAt: now.UTC(), Lanes: copyLanes})
	if err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	tmp, err := os.CreateTemp(filepath.Dir(path), ".lane-queues-*.tmp")
	if err != nil {
		return err
	}
	tmpName := tmp.Name()
	defer os.Remove(tmpName)
	if _, err = tmp.Write(append(b, '\n')); err == nil {
		err = tmp.Close()
	} else {
		_ = tmp.Close()
	}
	if err != nil {
		return err
	}
	return os.Rename(tmpName, path)
}

// ReadQueues reads a valid, unexpired queue snapshot matching the requested key and age bound.
func ReadQueues(path, key string, maxAge time.Duration, now time.Time) (QueueSnapshot, bool) {
	var q QueueSnapshot
	b, err := os.ReadFile(path)
	if err != nil || json.Unmarshal(b, &q) != nil || q.Schema != QueueSchema || q.Key != key {
		return QueueSnapshot{}, false
	}
	if maxAge <= 0 || q.GeneratedAt.IsZero() || now.Sub(q.GeneratedAt) < 0 || now.Sub(q.GeneratedAt) >= maxAge {
		return QueueSnapshot{}, false
	}
	return q, true
}

// PopLane pops the first issue from the specified lane queue and atomically persists the updated snapshot.
func PopLane(path, key, lane string, maxAge time.Duration, now time.Time) (int, bool, error) {
	q, ok := ReadQueues(path, key, maxAge, now)
	if !ok || len(q.Lanes[lane]) == 0 {
		return 0, false, nil
	}
	issue := q.Lanes[lane][0]
	q.Lanes[lane] = append([]int(nil), q.Lanes[lane][1:]...)
	return issue, true, WriteQueues(path, key, q.Lanes, now)
}
