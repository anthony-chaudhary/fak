package dispatch

import (
	"fmt"
	"log"
	"time"

	"github.com/anthony-chaudhary/fak/internal/l3server/metrics"
	"github.com/anthony-chaudhary/fak/internal/l3server/shard"
	"github.com/anthony-chaudhary/fak/internal/l3server/transport/protocol"
)

var errDispatchTimeout = fmt.Errorf("batch dispatch timeout")

// fanOutShards submits operations to multiple shards concurrently and collects
// results concurrently. Each shard result is received in a goroutine so a slow
// shard doesn't block collection of fast shards (fixes cumulative timeout).
func (d *Dispatcher) fanOutShards(groups []*batchGroup, makeOp func(*batchGroup) shard.ShardOp) ([]shard.OpResult, error) {
	ops := make([]shard.ShardOp, len(groups))
	for i, g := range groups {
		ops[i] = makeOp(g)
		if !g.shard.SubmitAsync(ops[i]) {
			log.Printf("[cama] WARNING: batch dispatch to shard %d dropped â€” queue full (%d groups total)", g.shard.ID(), len(groups))
		}
	}

	timeout := d.DispatchTimeout
	if timeout <= 0 {
		timeout = 30 * time.Second
	}

	results := make([]shard.OpResult, len(groups))
	remaining := len(groups)

	type indexedResult struct {
		idx    int
		result shard.OpResult
	}
	done := make(chan indexedResult, len(groups))
	for i := range groups {
		go func(idx int) {
			done <- indexedResult{idx: idx, result: <-ops[idx].Result}
		}(i)
	}

	timer := time.NewTimer(timeout)
	defer timer.Stop()
	for remaining > 0 {
		select {
		case ir := <-done:
			results[ir.idx] = ir.result
			remaining--
		case <-timer.C:
			completed := len(groups) - remaining
			log.Printf("[cama] WARNING: batch dispatch timed out â€” %d/%d completed", completed, len(groups))
			metrics.IncrBatchTimeouts()
			return nil, errDispatchTimeout
		}
	}
	return results, nil
}

// HandleMGet processes a batch MGET request.
// Cluster: fan-out to remote primaries for non-local keys.
func (d *Dispatcher) HandleMGet(msg protocol.Message) protocol.Message {
	keys, err := protocol.DecodeMGetBody(msg.Body)
	if err != nil {
		return ErrResponse(msg.Header.RequestID, err.Error())
	}

	values := make([][]byte, len(keys))
	founds := make([]bool, len(keys))

	if d.Ring != nil {
		// Cluster: group by node, fan out
		nodeGroups := d.groupKeysByNode(keys, nil)
		for _, ng := range nodeGroups {
			if ng.isLocal {
				// Local path
				localGroups := d.groupKeysByShard(ng.keys, nil, true)
				for _, g := range localGroups {
					result := g.shard.Submit(shard.ShardOp{
						Type:      shard.OpMGet,
						Keys:      g.keys,
						KeyHashes: g.hashes,
						Result:    make(chan shard.OpResult, 1),
					})
					if result.Err != nil {
						log.Printf("[cama] WARNING: MGET shard error (cluster local): %v", result.Err)
						continue
					}
					for j, localIdx := range g.indices {
						origIdx := ng.indices[localIdx]
						if j < len(result.Values) {
							values[origIdx] = result.Values[j]
						}
						if j < len(result.Founds) {
							founds[origIdx] = result.Founds[j]
						}
					}
				}
			} else {
				// Remote: proxy via MGet
				c, cerr := d.getProxyConn(ng.addr)
				if cerr != nil {
					continue // skip remote failures
				}
				rVals, rFounds, gerr := c.MGet(ng.keys)
				if gerr != nil {
					d.removeProxyConn(ng.addr)
					continue
				}
				for j, origIdx := range ng.indices {
					if j < len(rVals) {
						values[origIdx] = rVals[j]
						founds[origIdx] = rFounds[j]
					}
				}
			}
		}
	} else {
		// Standalone path
		groups := d.groupKeysByShard(keys, nil, true)
		results, err := d.fanOutShards(groups, func(g *batchGroup) shard.ShardOp {
			return shard.ShardOp{
				Type:      shard.OpMGet,
				Keys:      g.keys,
				KeyHashes: g.hashes,
				Result:    make(chan shard.OpResult, 1),
			}
		})
		if err != nil {
			return ErrResponse(msg.Header.RequestID, err.Error())
		}
		for i, g := range groups {
			if results[i].Err != nil {
				log.Printf("[cama] WARNING: MGET shard error (shard group %d): %v", i, results[i].Err)
				continue // leave values[origIdx]=nil, founds[origIdx]=false
			}
			for j, origIdx := range g.indices {
				if j < len(results[i].Values) {
					values[origIdx] = results[i].Values[j]
				}
				if j < len(results[i].Founds) {
					founds[origIdx] = results[i].Founds[j]
				}
			}
		}
	}

	return protocol.Message{
		Header: protocol.Header{OpCode: protocol.RespMultiValue},
		Body:   protocol.EncodeMultiValueResponse(values, founds),
	}
}

// HandleMGetWithAlloc processes a batch MGET and returns per-key AllocInfos
// alongside values and founds. Used by the RDMA transport for batch RDMA Read.
func (d *Dispatcher) HandleMGetWithAlloc(msg protocol.Message) ([]shard.OpResult, []*BatchGroup, error) {
	keys, err := protocol.DecodeMGetBody(msg.Body)
	if err != nil {
		return nil, nil, err
	}

	groups := d.groupKeysByShard(keys, nil, true)
	results, err := d.fanOutShards(groups, func(g *batchGroup) shard.ShardOp {
		return shard.ShardOp{
			Type:      shard.OpMGetWithAlloc,
			Keys:      g.keys,
			KeyHashes: g.hashes,
			Result:    make(chan shard.OpResult, 1),
		}
	})
	if err != nil {
		return nil, nil, err
	}
	return results, groups, nil
}

// HandleMSet processes a batch MSET request.
// Cluster: fan-out to remote primaries; replicate local keys.
// Returns RespOK on full success (backward compat) or RespMSetResult on partial failure.
func (d *Dispatcher) HandleMSet(msg protocol.Message) protocol.Message {
	keys, vals, err := protocol.DecodeMSetBody(msg.Body)
	if err != nil {
		return ErrResponse(msg.Header.RequestID, err.Error())
	}

	allStatuses := make([]byte, len(keys))
	anyFailed := false
	var oomShard *shard.Shard // first shard that returned OOM (for diagnostics)
	allOOM := true           // track if every shard group failed with OOM

	if d.Ring != nil {
		nodeGroups := d.groupKeysByNode(keys, vals)
		for _, ng := range nodeGroups {
			if ng.isLocal {
				localGroups := d.groupKeysByShard(ng.keys, ng.values, true)
				for _, g := range localGroups {
					result := g.shard.Submit(shard.ShardOp{
						Type:      shard.OpMSet,
						Keys:      g.keys,
						Values:    g.values,
						KeyHashes: g.hashes,
						TTLMs:     int64(msg.Header.Flags & protocol.FlagWithTTL),
						Result:    make(chan shard.OpResult, 1),
					})
					if result.Err != nil {
						// Whole-shard error: mark all keys in this group as failed
						for _, localIdx := range g.indices {
							origIdx := ng.indices[localIdx]
							allStatuses[origIdx] = 1
							anyFailed = true
						}
					} else if result.SetStatuses != nil {
						for j, localIdx := range g.indices {
							origIdx := ng.indices[localIdx]
							allStatuses[origIdx] = result.SetStatuses[j]
							if result.SetStatuses[j] != 0 {
								anyFailed = true
							}
						}
					}
				}
				// Replicate local keys
				if d.Replicator != nil && msg.Header.Flags&protocol.FlagReplicated == 0 {
					for j := range ng.keys {
						d.Replicator.ReplicateSet(ng.keys[j], ng.values[j], 0)
					}
				}
			} else {
				c, cerr := d.getProxyConn(ng.addr)
				if cerr != nil {
					for _, origIdx := range ng.indices {
						allStatuses[origIdx] = 1
						anyFailed = true
					}
					continue
				}
				remoteStatuses, rerr := c.MSet(ng.keys, ng.values)
				if rerr != nil {
					d.removeProxyConn(ng.addr)
					for _, origIdx := range ng.indices {
						allStatuses[origIdx] = 1
						anyFailed = true
					}
					continue
				}
				if remoteStatuses != nil {
					for j, origIdx := range ng.indices {
						if j < len(remoteStatuses) {
							allStatuses[origIdx] = remoteStatuses[j]
							if remoteStatuses[j] != 0 {
								anyFailed = true
							}
						}
					}
				}
			}
		}
	} else {
		groups := d.groupKeysByShard(keys, vals, true)
		results, err := d.fanOutShards(groups, func(g *batchGroup) shard.ShardOp {
			return shard.ShardOp{
				Type:      shard.OpMSet,
				Keys:      g.keys,
				Values:    g.values,
				KeyHashes: g.hashes,
				TTLMs:     int64(msg.Header.Flags & protocol.FlagWithTTL),
				Result:    make(chan shard.OpResult, 1),
			}
		})
		if err != nil {
			return ErrResponse(msg.Header.RequestID, err.Error())
		}
		for i, g := range groups {
			if results[i].Err != nil {
				isOOM := shard.IsOOM(results[i].Err)
				if isOOM && oomShard == nil {
					oomShard = g.shard
				}
				if !isOOM {
					allOOM = false
				}
				for _, origIdx := range g.indices {
					allStatuses[origIdx] = 1
					anyFailed = true
				}
			} else {
				allOOM = false
				if results[i].SetStatuses != nil {
					for j, origIdx := range g.indices {
						allStatuses[origIdx] = results[i].SetStatuses[j]
						if results[i].SetStatuses[j] != 0 {
							anyFailed = true
						}
					}
				}
			}
		}
	}

	if !anyFailed {
		return OKResponse(msg.Header.RequestID) // backward compat
	}

	// If ALL shard groups failed with OOM, return dedicated RespOOM for client backpressure
	if allOOM && oomShard != nil {
		return OOMResponse(msg.Header.RequestID, oomShard, "MSET rejected: all shards under memory pressure")
	}

	return protocol.Message{
		Header: protocol.Header{OpCode: protocol.RespMSetResult, RequestID: msg.Header.RequestID},
		Body:   protocol.EncodeMSetResultResponse(allStatuses),
	}
}

// HandleMTest processes a batch MTEST (EXISTS) request.
func (d *Dispatcher) HandleMTest(msg protocol.Message) protocol.Message {
	keys, err := protocol.DecodeMGetBody(msg.Body)
	if err != nil {
		return ErrResponse(msg.Header.RequestID, err.Error())
	}

	groups := d.groupKeysByShard(keys, nil, true)
	founds := make([]bool, len(keys))
	results, ferr := d.fanOutShards(groups, func(g *batchGroup) shard.ShardOp {
		return shard.ShardOp{
			Type:      shard.OpTest,
			Keys:      g.keys,
			KeyHashes: g.hashes,
			Result:    make(chan shard.OpResult, 1),
		}
	})
	if ferr != nil {
		return ErrResponse(msg.Header.RequestID, ferr.Error())
	}
	for i, g := range groups {
		if results[i].Err != nil {
			log.Printf("[cama] WARNING: MTEST shard error (shard group %d): %v", i, results[i].Err)
			continue // leave founds[origIdx]=false
		}
		for j, origIdx := range g.indices {
			if j < len(results[i].Founds) {
				founds[origIdx] = results[i].Founds[j]
			}
		}
	}

	values := make([][]byte, len(keys))
	return protocol.Message{
		Header: protocol.Header{OpCode: protocol.RespMultiValue},
		Body:   protocol.EncodeMultiValueResponse(values, founds),
	}
}

// HandleMDel processes a batch MDEL request.
func (d *Dispatcher) HandleMDel(msg protocol.Message) protocol.Message {
	keys, err := protocol.DecodeMDelBody(msg.Body)
	if err != nil {
		return ErrResponse(msg.Header.RequestID, err.Error())
	}

	groups := d.groupKeysByShard(keys, nil, false)
	results, ferr := d.fanOutShards(groups, func(g *batchGroup) shard.ShardOp {
		return shard.ShardOp{
			Type:      shard.OpMDel,
			Keys:      g.keys,
			KeyHashes: g.hashes,
			Result:    make(chan shard.OpResult, 1),
		}
	})
	if ferr != nil {
		return ErrResponse(msg.Header.RequestID, ferr.Error())
	}
	for _, r := range results {
		if r.Err != nil {
			return ErrResponse(msg.Header.RequestID, r.Err.Error())
		}
	}

	return OKResponse(msg.Header.RequestID)
}

// HandleKeys processes a KEYS scan request.
func (d *Dispatcher) HandleKeys(msg protocol.Message) protocol.Message {
	pattern, err := protocol.DecodeKeysBody(msg.Body)
	if err != nil {
		return ErrResponse(msg.Header.RequestID, err.Error())
	}

	var allKeys [][]byte
	for i := 0; i < d.Manager.NumShards(); i++ {
		sh := d.Manager.Shard(i)
		result := sh.Submit(shard.ShardOp{
			Type:    shard.OpKeys,
			Pattern: pattern,
			Result:  make(chan shard.OpResult, 1),
		})
		if result.Err != nil {
			return ErrResponse(msg.Header.RequestID, result.Err.Error())
		}
		allKeys = append(allKeys, result.MatchedKeys...)
	}

	return protocol.Message{
		Header: protocol.Header{OpCode: protocol.RespOK},
		Body:   protocol.EncodeKeysResponse(allKeys),
	}
}

// HandleFlush clears all data from every shard.
func (d *Dispatcher) HandleFlush(msg protocol.Message) protocol.Message {
	n := d.Manager.NumShards()

	// Snapshot pre-flush state for logging
	var totalKeys uint64
	var totalAllocBytes int64
	for i := 0; i < n; i++ {
		sh := d.Manager.Shard(i)
		totalKeys += sh.IndexCount()
		totalAllocBytes += sh.Allocator().AllocatedBytes()
	}
	log.Printf("[cama] flush: starting across %d shards (%d keys, %.2f GB allocated)",
		n, totalKeys, float64(totalAllocBytes)/(1<<30))

	start := time.Now()
	for i := 0; i < n; i++ {
		sh := d.Manager.Shard(i)
		result := sh.Submit(shard.ShardOp{
			Type:   shard.OpFlush,
			Result: make(chan shard.OpResult, 1),
		})
		if result.Err != nil {
			log.Printf("[cama] flush: shard %d failed: %v", i, result.Err)
			return ErrResponse(msg.Header.RequestID, result.Err.Error())
		}
	}
	// Reset all stats counters (shard metrics reset in handleFlush, wire metrics here)
	if d.ConnReg != nil {
		d.ConnReg.ResetAll()
	}

	log.Printf("[cama] flush: completed in %v (%d keys cleared, %.2f GB freed)",
		time.Since(start).Round(time.Millisecond), totalKeys, float64(totalAllocBytes)/(1<<30))
	return OKResponse(msg.Header.RequestID)
}
