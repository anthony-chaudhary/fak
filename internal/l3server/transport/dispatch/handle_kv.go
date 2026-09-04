package dispatch

import (
	"fmt"

	"github.com/anthony-chaudhary/fak/internal/l3server/index"
	"github.com/anthony-chaudhary/fak/internal/l3server/shard"
	"github.com/anthony-chaudhary/fak/internal/l3server/transport/protocol"
)

// HandleGet processes a single GET request.
// Cluster: read-from-any â€” try local first; on miss proxy to primary.
func (d *Dispatcher) HandleGet(msg protocol.Message) protocol.Message {
	key, err := protocol.DecodeKeyBody(msg.Body)
	if err != nil {
		return ErrResponse(msg.Header.RequestID, err.Error())
	}

	// Try local first â€” use pre-decoded key to avoid double DecodeKeyBody
	result := d.submitPreDecoded(key, shard.OpGet)
	if result.Found {
		return protocol.Message{
			Header: protocol.Header{OpCode: protocol.RespValue},
			Body:   protocol.EncodeValueResponse(result.Value, result.Found),
		}
	}

	// Miss locally â€” if clustered and not primary, proxy to primary
	if d.Ring != nil {
		primary, isLocal := d.isLocalKey(key)
		if !isLocal && primary != nil {
			c, cerr := d.getProxyConn(primary.Addr)
			if cerr == nil {
				val, gerr := c.Get(key)
				if gerr != nil {
					d.removeProxyConn(primary.Addr)
				} else if val != nil {
					return protocol.Message{
						Header: protocol.Header{OpCode: protocol.RespValue},
						Body:   protocol.EncodeValueResponse(val, true),
					}
				}
			}
		}
	}

	return protocol.Message{
		Header: protocol.Header{OpCode: protocol.RespValue},
		Body:   protocol.EncodeValueResponse(nil, false),
	}
}

// HandleGetWithAlloc processes a single GET and returns both the result and alloc metadata.
// Used by the RDMA transport to look up MR info for the RDMA Read path.
func (d *Dispatcher) HandleGetWithAlloc(msg protocol.Message) (shard.OpResult, error) {
	key, err := protocol.DecodeKeyBody(msg.Body)
	if err != nil {
		return shard.OpResult{}, err
	}
	return d.submitPreDecoded(key, shard.OpGet), nil
}

// HandleSet processes a SET request.
// Cluster: writes route to primary; after success, replicate to peers (unless FlagReplicated).
func (d *Dispatcher) HandleSet(msg protocol.Message) protocol.Message {
	key, value, ttlMs, err := protocol.DecodeKVBody(msg.Body, msg.Header.Flags)
	if err != nil {
		return ErrResponse(msg.Header.RequestID, err.Error())
	}

	// Cluster: route to primary if not local
	if d.Ring != nil {
		primary, isLocal := d.isLocalKey(key)
		if !isLocal && primary != nil {
			c, cerr := d.getProxyConn(primary.Addr)
			if cerr != nil {
				return ErrResponse(msg.Header.RequestID, fmt.Sprintf("proxy connect to %s: %v", primary.Addr, cerr))
			}
			if err := c.Set(key, value, ttlMs); err != nil {
				d.removeProxyConn(primary.Addr)
				return ErrResponse(msg.Header.RequestID, fmt.Sprintf("proxy SET to %s: %v", primary.Addr, err))
			}
			return OKResponse(msg.Header.RequestID)
		}
	}

	// Local: handle via shard
	keyHash := index.KeyHash(key)
	sh := d.Manager.Route(keyHash)
	result := sh.Submit(shard.ShardOp{
		Type:    shard.OpSet,
		Key:     key,
		KeyHash: keyHash,
		Value:   value,
		TTLMs:   ttlMs,
	})
	if result.Err != nil {
		if shard.IsOOM(result.Err) {
			return OOMResponse(msg.Header.RequestID, sh, result.Err.Error())
		}
		return ErrResponse(msg.Header.RequestID, result.Err.Error())
	}

	// Replicate to peers (async, fire-and-forget) â€” skip if FlagReplicated
	if d.Replicator != nil && msg.Header.Flags&protocol.FlagReplicated == 0 {
		d.Replicator.ReplicateSet(key, value, ttlMs)
	}

	return OKResponse(msg.Header.RequestID)
}

// HandleDelete processes a DELETE request.
// Cluster: route to primary; replicate to peers.
func (d *Dispatcher) HandleDelete(msg protocol.Message) protocol.Message {
	key, err := protocol.DecodeKeyBody(msg.Body)
	if err != nil {
		return ErrResponse(msg.Header.RequestID, err.Error())
	}

	// Cluster: route to primary if not local
	if d.Ring != nil {
		primary, isLocal := d.isLocalKey(key)
		if !isLocal && primary != nil {
			c, cerr := d.getProxyConn(primary.Addr)
			if cerr != nil {
				return ErrResponse(msg.Header.RequestID, fmt.Sprintf("proxy connect to %s: %v", primary.Addr, cerr))
			}
			if err := c.Delete(key); err != nil {
				d.removeProxyConn(primary.Addr)
				return ErrResponse(msg.Header.RequestID, fmt.Sprintf("proxy DELETE to %s: %v", primary.Addr, err))
			}
			return OKResponse(msg.Header.RequestID)
		}
	}

	// Local â€” use pre-decoded key to avoid double DecodeKeyBody
	result := d.submitPreDecoded(key, shard.OpDelete)
	if result.Err != nil {
		return ErrResponse(msg.Header.RequestID, result.Err.Error())
	}

	// Replicate delete (async)
	if d.Replicator != nil && msg.Header.Flags&protocol.FlagReplicated == 0 {
		d.Replicator.ReplicateDelete(key)
	}

	return OKResponse(msg.Header.RequestID)
}

// HandleTest processes a TEST (EXISTS) request.
// Cluster: read-from-any â€” check local first; on miss proxy to primary.
func (d *Dispatcher) HandleTest(msg protocol.Message) protocol.Message {
	key, err := protocol.DecodeKeyBody(msg.Body)
	if err != nil {
		return ErrResponse(msg.Header.RequestID, err.Error())
	}

	// Use pre-decoded key to avoid double DecodeKeyBody
	result := d.submitPreDecoded(key, shard.OpTest)

	if !result.Found && d.Ring != nil {
		primary, isLocal := d.isLocalKey(key)
		if !isLocal && primary != nil {
			c, cerr := d.getProxyConn(primary.Addr)
			if cerr == nil {
				found, gerr := c.Exists(key)
				if gerr != nil {
					d.removeProxyConn(primary.Addr)
				} else if found {
					return protocol.Message{
						Header: protocol.Header{OpCode: protocol.RespOK},
						Body:   []byte{1},
					}
				}
			}
		}
	}

	body := []byte{0}
	if result.Found {
		body[0] = 1
	}
	return protocol.Message{
		Header: protocol.Header{OpCode: protocol.RespOK},
		Body:   body,
	}
}

// HandleLease processes a LEASE request.
func (d *Dispatcher) HandleLease(msg protocol.Message) protocol.Message {
	key, durationMs, err := protocol.DecodeLeaseBody(msg.Body)
	if err != nil {
		return ErrResponse(msg.Header.RequestID, err.Error())
	}
	keyHash := index.KeyHash(key)
	sh := d.Manager.Route(keyHash)
	result := sh.Submit(shard.ShardOp{
		Type:    shard.OpLease,
		Key:     key,
		KeyHash: keyHash,
		LeaseMs: durationMs,
	})
	if result.Err != nil {
		return ErrResponse(msg.Header.RequestID, result.Err.Error())
	}
	return OKResponse(msg.Header.RequestID)
}

// HandlePin processes a PIN request.
func (d *Dispatcher) HandlePin(msg protocol.Message) protocol.Message {
	result, err := d.submitSingleKeyOp(msg.Body, shard.OpPin)
	if err != nil {
		return ErrResponse(msg.Header.RequestID, err.Error())
	}
	if result.Err != nil {
		return ErrResponse(msg.Header.RequestID, result.Err.Error())
	}
	return OKResponse(msg.Header.RequestID)
}

// HandleUnpin processes an UNPIN request.
func (d *Dispatcher) HandleUnpin(msg protocol.Message) protocol.Message {
	result, err := d.submitSingleKeyOp(msg.Body, shard.OpUnpin)
	if err != nil {
		return ErrResponse(msg.Header.RequestID, err.Error())
	}
	if result.Err != nil {
		return ErrResponse(msg.Header.RequestID, result.Err.Error())
	}
	return OKResponse(msg.Header.RequestID)
}
