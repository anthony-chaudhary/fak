package dispatch

import (
	"encoding/json"
	"fmt"
	"log"
	"time"

	"github.com/anthony-chaudhary/fak/internal/l3server/index"
	"github.com/anthony-chaudhary/fak/internal/l3server/shard"
	"github.com/anthony-chaudhary/fak/internal/l3server/snapshot"
	"github.com/anthony-chaudhary/fak/internal/l3server/transport/protocol"
	"github.com/anthony-chaudhary/fak/internal/l3server/version"
)

// maintenanceRequest is the JSON request body for OpMaintenance.
type maintenanceRequest struct {
	Action   string `json:"action"`
	Force    bool   `json:"force"`
	ShardIDs []int  `json:"shard_ids"`
}

// HandleMaintenance processes vacuum, autotune, and status requests.
// Requires Manager â€” the Dispatch() gate ensures this is only called when ready.
func (d *Dispatcher) HandleMaintenance(msg protocol.Message) protocol.Message {
	var req maintenanceRequest
	if err := json.Unmarshal(msg.Body, &req); err != nil {
		return ErrResponse(msg.Header.RequestID, "invalid maintenance JSON: "+err.Error())
	}

	var result interface{}
	switch req.Action {
	case "vacuum":
		result = d.Manager.OnDemandVacuum(req.Force, req.ShardIDs)
	case "autotune":
		result = d.Manager.OnDemandAutoTune(req.Force, req.ShardIDs)
	case "status":
		result = d.Manager.MaintenanceStatus()
	default:
		return ErrResponse(msg.Header.RequestID, "unknown maintenance action: "+req.Action)
	}

	body, err := json.Marshal(result)
	if err != nil {
		return ErrResponse(msg.Header.RequestID, "failed to marshal maintenance result: "+err.Error())
	}
	return protocol.Message{
		Header: protocol.Header{OpCode: protocol.RespOK, RequestID: msg.Header.RequestID},
		Body:   body,
	}
}

// HandleSnapshot triggers a server-side cache snapshot.
func (d *Dispatcher) HandleSnapshot(msg protocol.Message) protocol.Message {
	var req struct {
		Dir string `json:"dir"`
	}
	if len(msg.Body) > 0 {
		_ = json.Unmarshal(msg.Body, &req)
	}
	dir := req.Dir
	if dir == "" {
		dir = d.SnapshotDir
	}
	if dir == "" {
		return ErrResponse(msg.Header.RequestID, "no snapshot directory configured")
	}

	start := time.Now()
	n := d.Manager.NumShards()
	var totalKeys int64
	var totalBytes int64

	w := snapshot.NewWriter(dir)

	// Collect and write entries from all shards
	for i := 0; i < n; i++ {
		sh := d.Manager.Shard(i)
		result := sh.Submit(shard.ShardOp{
			Type:   shard.OpSnapshot,
			Result: make(chan shard.OpResult, 1),
		})
		if result.Err != nil {
			return ErrResponse(msg.Header.RequestID, fmt.Sprintf("snapshot shard %d: %v", i, result.Err))
		}
		if err := w.WriteShard(i, result.SnapshotEntries); err != nil {
			return ErrResponse(msg.Header.RequestID, fmt.Sprintf("snapshot write shard %d: %v", i, err))
		}
		totalKeys += int64(len(result.SnapshotEntries))
		for _, e := range result.SnapshotEntries {
			totalBytes += int64(len(e.Key) + len(e.Value))
		}
	}

	// Write manifest
	allocMode := "slab"
	if n > 0 {
		allocMode = d.Manager.Shard(0).Config().AllocatorMode
	}
	m := snapshot.Manifest{
		Version:       1,
		ServerVersion: version.ServerVersion,
		CreatedAt:     time.Now().Format(time.RFC3339),
		ShardCount:    n,
		AllocatorMode: allocMode,
		TotalKeys:     totalKeys,
		TotalBytes:    totalBytes,
	}
	for i := 0; i < n; i++ {
		m.Files = append(m.Files, fmt.Sprintf("shard-%d.dat", i))
	}
	if err := w.WriteManifest(m); err != nil {
		return ErrResponse(msg.Header.RequestID, fmt.Sprintf("snapshot manifest: %v", err))
	}

	durationMs := time.Since(start).Milliseconds()
	body, _ := json.Marshal(map[string]interface{}{
		"keys":        totalKeys,
		"dir":         dir,
		"duration_ms": durationMs,
		"shards":      n,
	})
	log.Printf("[snapshot] saved %d keys (%.1f MB) to %s in %dms",
		totalKeys, float64(totalBytes)/(1024*1024), dir, durationMs)
	return protocol.Message{
		Header: protocol.Header{OpCode: protocol.RespOK, RequestID: msg.Header.RequestID},
		Body:   body,
	}
}

// HandleRestore triggers a server-side cache restore.
func (d *Dispatcher) HandleRestore(msg protocol.Message) protocol.Message {
	var req struct {
		Dir string `json:"dir"`
	}
	if len(msg.Body) > 0 {
		_ = json.Unmarshal(msg.Body, &req)
	}
	dir := req.Dir
	if dir == "" {
		dir = d.SnapshotDir
	}
	if dir == "" {
		return ErrResponse(msg.Header.RequestID, "no snapshot directory configured")
	}

	start := time.Now()

	// Flush all shards first
	for i := 0; i < d.Manager.NumShards(); i++ {
		d.Manager.Shard(i).Submit(shard.ShardOp{
			Type:   shard.OpFlush,
			Result: make(chan shard.OpResult, 1),
		})
	}

	// Load and restore
	manifest, entries, err := snapshot.Load(dir, d.Manager.NumShards())
	if err != nil {
		return ErrResponse(msg.Header.RequestID, fmt.Sprintf("restore from %s: %v", dir, err))
	}

	// Group entries by current shard layout
	shardEntries := make([][]snapshot.KVEntry, d.Manager.NumShards())
	for i := range shardEntries {
		shardEntries[i] = []snapshot.KVEntry{}
	}
	for _, e := range entries {
		h := index.KeyHash(e.Key)
		sh := d.Manager.Route(h)
		shardEntries[sh.ID()] = append(shardEntries[sh.ID()], e)
	}

	var totalLoaded int64
	for i, se := range shardEntries {
		if len(se) == 0 {
			continue
		}
		result := d.Manager.Shard(i).Submit(shard.ShardOp{
			Type:           shard.OpRestore,
			RestoreEntries: se,
			Result:         make(chan shard.OpResult, 1),
		})
		if result.Err != nil {
			log.Printf("[snapshot] restore shard %d failed: %v", i, result.Err)
		}
		totalLoaded += int64(result.Loaded)
	}

	durationMs := time.Since(start).Milliseconds()
	body, _ := json.Marshal(map[string]interface{}{
		"keys":           totalLoaded,
		"dir":            dir,
		"duration_ms":    durationMs,
		"server_version": manifest.ServerVersion,
	})
	log.Printf("[snapshot] restored %d keys from %s in %dms (original: v%s, %d shards)",
		totalLoaded, dir, durationMs, manifest.ServerVersion, manifest.ShardCount)
	return protocol.Message{
		Header: protocol.Header{OpCode: protocol.RespOK, RequestID: msg.Header.RequestID},
		Body:   body,
	}
}
