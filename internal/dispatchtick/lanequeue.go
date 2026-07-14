package dispatchtick

import (
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
)

// LaneQueueSchema tags the on-disk per-lane dispatch queue file so a reader can
// reject a foreign or future-schema payload before trusting its lane order.
const LaneQueueSchema = "fak.dispatchtick.lane-queues.v1"

// LaneQueuesFileName is the fixed filename WriteLaneQueues writes under its dir
// and ReadLaneQueue reads back, so a worker knows the on-disk queue path without
// re-deriving it. It lives under a caller-chosen dir (e.g. RunsDirName).
const LaneQueuesFileName = "lane-queues.json"

// LaneQueueFile is the serialized projection of RouterPayload.Lanes: each lane
// mapped to its priority+recency-ordered issue refs, exactly as
// BuildRouterPayload's OrderLaneCandidates established them. It carries NO
// TTL/freshness metadata — the cache staleness gate is a sibling leaf's concern
// under epic #4160; this file persists only the ordered refs a worker pops.
type LaneQueueFile struct {
	Schema string           `json:"schema"`
	Lanes  map[string][]int `json:"lanes"`
}

// WriteLaneQueues projects payload.Lanes into a stable, schema-tagged JSON file
// under dir, copying each lane's RouterLaneGroup.Issues slice verbatim so the
// persisted order is byte-identical to the order BuildRouterPayload already
// produced. It re-orders nothing and re-routes nothing: the ordering happened
// in BuildRouterPayload (OrderLaneCandidates), and this serializer only mirrors
// it to disk. The write is atomic (temp file + rename) so a concurrent reader
// never sees a half-written queue.
func WriteLaneQueues(dir string, payload RouterPayload) error {
	if dir == "" {
		return errors.New("dispatchtick: lane queue dir is required")
	}
	// Non-nil map so an empty payload serializes to `"lanes":{}` (round-trip
	// stable), never JSON null.
	lanes := make(map[string][]int, len(payload.Lanes))
	for lane, grp := range payload.Lanes {
		lanes[lane] = append([]int(nil), grp.Issues...)
	}
	b, err := json.MarshalIndent(LaneQueueFile{Schema: LaneQueueSchema, Lanes: lanes}, "", "  ")
	if err != nil {
		return err
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	tmp, err := os.CreateTemp(dir, ".lane-queues-*.tmp")
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
	return os.Rename(tmpName, filepath.Join(dir, LaneQueuesFileName))
}

// ReadLaneQueue reads the persisted queue file under dir and returns the named
// lane's ordered issue refs and its head — the first ref a worker would pop —
// with no gh fetch and no RouteIssues call. ok is false when the file is
// absent/corrupt/wrong-schema or the lane has no persisted queue at all; a lane
// that is present but empty returns (nil, 0, true). The returned slice is a copy,
// so a caller may mutate it (e.g. pop the head) without touching another read.
func ReadLaneQueue(dir, lane string) (refs []int, head int, ok bool) {
	if dir == "" {
		return nil, 0, false
	}
	b, err := os.ReadFile(filepath.Join(dir, LaneQueuesFileName))
	if err != nil {
		return nil, 0, false
	}
	var f LaneQueueFile
	if json.Unmarshal(b, &f) != nil || f.Schema != LaneQueueSchema {
		return nil, 0, false
	}
	issues, present := f.Lanes[lane]
	if !present {
		return nil, 0, false
	}
	refs = append([]int(nil), issues...)
	if len(refs) == 0 {
		return nil, 0, true
	}
	return refs, refs[0], true
}
