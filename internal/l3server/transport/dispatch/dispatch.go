package dispatch

import (
	"sync"
	"sync/atomic"
	"time"

	"github.com/anthony-chaudhary/fak/internal/l3server/cluster"
	"github.com/anthony-chaudhary/fak/internal/l3server/index"
	"github.com/anthony-chaudhary/fak/internal/l3server/metrics"
	"github.com/anthony-chaudhary/fak/internal/l3server/shard"
	"github.com/anthony-chaudhary/fak/internal/l3server/transport/protocol"
	"github.com/anthony-chaudhary/fak/internal/l3server/client"
)

// globalInflightOps tracks total ops currently inside Dispatch() across all dispatchers.
var globalInflightOps atomic.Int64

// GlobalInflightOps returns the number of ops currently being processed.
func GlobalInflightOps() int64 { return globalInflightOps.Load() }

// RDMAEndpointInfo describes a single RDMA endpoint for client discovery.
type RDMAEndpointInfo struct {
	Device string `json:"device"`
	IP     string `json:"ip"`
	Port   int    `json:"port"`
}

const bytesPerGB = 1 << 30

// Dispatcher routes protocol messages to shard operations.
// Shared between TCP and RDMA transports. Transport-specific handlers
// (e.g. Get/MGet with RDMA Read path) can override by handling those
// opcodes before calling Dispatch.
type Dispatcher struct {
	Manager        *shard.Manager
	ConnReg        *metrics.ConnRegistry
	ClientStatsReg *metrics.ClientStatsRegistry
	StartedAt      time.Time
	RDMAEndpoints  []RDMAEndpointInfo
	MetricsAddr    string // address of the /ready endpoint (e.g. ":9090")

	// Cluster â€” nil when standalone
	Ring       *cluster.Ring
	Replicator *cluster.Replicator
	LocalID    string
	SnapshotDir string // for on-demand snapshot/restore

	// RDMA buffer sizes (bytes) â€” surfaced in INFO response for client diagnostics
	RDMASendBufSize int
	RDMARecvBufSize int

	// PollerMetrics returns per-NIC CQ poller stats (set after RDMA servers start).
	PollerMetrics metrics.PollerMetricsProvider

	// OpLatencyMetrics returns per-shard op latency data (set after shard allocation).
	OpLatencyMetrics metrics.OpLatencyProvider

	// DispatchTimeout is the timeout for batch fan-out operations.
	DispatchTimeout time.Duration

	// Early connection acceptance: Manager may be nil during startup.
	// managerReady is set to true via SetManager() when shard allocation completes.
	managerReady atomic.Bool
	numShards    int // set at construction time for NotReady progress responses

	proxyMu    sync.Mutex
	proxyPool  map[string]*client.Client
}

// SetManager wires the shard Manager into the dispatcher after background allocation.
// Must be called exactly once. After this call, data ops are routed to shards.
func (d *Dispatcher) SetManager(mgr *shard.Manager) {
	d.Manager = mgr
	d.numShards = mgr.NumShards()
	d.managerReady.Store(true)
}

// ManagerReady returns true if the shard Manager has been set via SetManager.
func (d *Dispatcher) ManagerReady() bool {
	return d.managerReady.Load()
}

// SetNumShards sets the expected shard count for NotReady progress responses.
// Called during early-connect setup when Manager is nil.
func (d *Dispatcher) SetNumShards(n int) {
	d.numShards = n
}

// NotReadyResponse builds a RespNotReady message with shard allocation progress.
func (d *Dispatcher) NotReadyResponse(reqID uint32) protocol.Message {
	var ready, total uint32
	if d.Manager != nil {
		ready = uint32(d.Manager.ReadyCount())
		total = uint32(d.Manager.NumShards())
	} else {
		total = uint32(d.numShards)
	}
	return protocol.Message{
		Header: protocol.Header{OpCode: protocol.RespNotReady, RequestID: reqID},
		Body:   protocol.EncodeNotReadyResponse(ready, total, "server starting â€” shards not yet allocated"),
	}
}

// SetClusterConfig sets cluster-mode fields on the dispatcher.
// ring must be *cluster.Ring, replicator must be *cluster.Replicator.
func (d *Dispatcher) SetClusterConfig(ring any, replicator any, localID string, snapshotDir string) {
	if ring != nil {
		d.Ring = ring.(*cluster.Ring)
	}
	if replicator != nil {
		d.Replicator = replicator.(*cluster.Replicator)
	}
	d.LocalID = localID
	d.SnapshotDir = snapshotDir
}

// batchGroup holds keys (and optionally values/indices) destined for a single shard.
type batchGroup struct {
	shard   *shard.Shard
	indices []int    // original positions, populated when trackIndices is true
	keys    [][]byte
	hashes  []uint64
	values  [][]byte // populated only when values are provided
}

// Shard returns the target shard for this group.
func (g *batchGroup) Shard() *shard.Shard { return g.shard }

// Indices returns the original key positions for result reordering.
func (g *batchGroup) Indices() []int { return g.indices }

// Hashes returns the key hashes for this group.
func (g *batchGroup) Hashes() []uint64 { return g.hashes }

// BatchGroup is exported for cross-package access (e.g., RDMA transport).
type BatchGroup = batchGroup

// Dispatch routes a message to the appropriate handler.
// Returns the response message. Get and MGet are included here for TCP;
// RDMA callers should intercept those opcodes before calling Dispatch.
//
// Opcodes that do NOT require Manager (always available): OpHandshake, OpInfo,
// OpStats, OpReportStats, OpCluster. All other opcodes return RespNotReady
// until SetManager() has been called.
func (d *Dispatcher) Dispatch(msg protocol.Message) protocol.Message {
	globalInflightOps.Add(1)
	defer globalInflightOps.Add(-1)

	// Always-available ops (no Manager required)
	switch msg.Header.OpCode {
	case protocol.OpHandshake:
		return d.HandleHandshake(msg)
	case protocol.OpInfo:
		return d.HandleInfo(msg)
	case protocol.OpCluster:
		return d.HandleCluster(msg)
	case protocol.OpStats:
		return d.HandleStats(msg)
	}

	// All remaining ops require a ready Manager
	if !d.managerReady.Load() {
		return d.NotReadyResponse(msg.Header.RequestID)
	}

	switch msg.Header.OpCode {
	case protocol.OpGet:
		return d.HandleGet(msg)
	case protocol.OpSet:
		return d.HandleSet(msg)
	case protocol.OpDelete:
		return d.HandleDelete(msg)
	case protocol.OpTest:
		return d.HandleTest(msg)
	case protocol.OpLease:
		return d.HandleLease(msg)
	case protocol.OpPin:
		return d.HandlePin(msg)
	case protocol.OpUnpin:
		return d.HandleUnpin(msg)
	case protocol.OpMGet:
		return d.HandleMGet(msg)
	case protocol.OpMSet:
		return d.HandleMSet(msg)
	case protocol.OpMTest:
		return d.HandleMTest(msg)
	case protocol.OpMDel:
		return d.HandleMDel(msg)
	case protocol.OpKeys:
		return d.HandleKeys(msg)
	case protocol.OpFlush:
		return d.HandleFlush(msg)
	case protocol.OpMaintenance:
		return d.HandleMaintenance(msg)
	case protocol.OpSnapshot:
		return d.HandleSnapshot(msg)
	case protocol.OpRestore:
		return d.HandleRestore(msg)
	default:
		return ErrResponse(msg.Header.RequestID, "unknown opcode")
	}
}

// submitSingleKeyOp decodes a key-only body, hashes, routes to the target shard, and submits.
func (d *Dispatcher) submitSingleKeyOp(body []byte, opType shard.OpType) (shard.OpResult, error) {
	key, err := protocol.DecodeKeyBody(body)
	if err != nil {
		return shard.OpResult{}, err
	}
	return d.submitPreDecoded(key, opType), nil
}

// submitPreDecoded routes a pre-decoded key to the target shard and submits.
// Avoids a second DecodeKeyBody call when the caller already has the key.
func (d *Dispatcher) submitPreDecoded(key []byte, opType shard.OpType) shard.OpResult {
	keyHash := index.KeyHash(key)
	sh := d.Manager.Route(keyHash)
	return sh.Submit(shard.ShardOp{
		Type:    opType,
		Key:     key,
		KeyHash: keyHash,
	})
}

// groupKeysByShard partitions keys (and optional values) by their target shard.
// When trackIndices is true, original positions are recorded for result reordering.
func (d *Dispatcher) groupKeysByShard(keys [][]byte, values [][]byte, trackIndices bool) []*batchGroup {
	m := make(map[*shard.Shard]*batchGroup)
	for i, key := range keys {
		h := index.KeyHash(key)
		sh := d.Manager.Route(h)
		g, ok := m[sh]
		if !ok {
			g = &batchGroup{shard: sh}
			m[sh] = g
		}
		if trackIndices {
			g.indices = append(g.indices, i)
		}
		g.keys = append(g.keys, key)
		g.hashes = append(g.hashes, h)
		if values != nil {
			g.values = append(g.values, values[i])
		}
	}
	groups := make([]*batchGroup, 0, len(m))
	for _, g := range m {
		groups = append(groups, g)
	}
	return groups
}

// ErrResponse creates an error response message.
func ErrResponse(reqID uint32, msg string) protocol.Message {
	return protocol.Message{
		Header: protocol.Header{OpCode: protocol.RespError, RequestID: reqID},
		Body:   protocol.EncodeErrorResponse(msg),
	}
}

// OKResponse creates an OK response message.
func OKResponse(reqID uint32) protocol.Message {
	return protocol.Message{
		Header: protocol.Header{OpCode: protocol.RespOK, RequestID: reqID},
		Body:   protocol.EncodeOKResponse(),
	}
}

// OOMResponse creates a RespOOM response with diagnostic information.
// The shard's allocator is queried for utilization data to help the client
// understand why the SET was rejected.
func OOMResponse(reqID uint32, sh *shard.Shard, errMsg string) protocol.Message {
	a := sh.Allocator()
	allocated := a.AllocatedBytes()
	var totalBytes uint64
	for _, r := range a.Regions() {
		totalBytes += r.Region.Size()
	}
	var utilPct uint8
	if totalBytes > 0 {
		utilPct = uint8(float64(allocated) / float64(totalBytes) * 100)
		if utilPct > 100 {
			utilPct = 100
		}
	}
	return protocol.Message{
		Header: protocol.Header{OpCode: protocol.RespOOM, RequestID: reqID},
		Body:   protocol.EncodeOOMResponse(utilPct, uint64(allocated), totalBytes, errMsg),
	}
}
