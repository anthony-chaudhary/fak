package metrics

import (
	"sync"
	"sync/atomic"
	"time"
)

// ConnMetrics holds per-connection wire byte counters.
type ConnMetrics struct {
	ID          uint64
	Transport   string
	RemoteAddr  string
	Device      string // RDMA device name (e.g. "mlx5_0"), empty for TCP
	DeviceIP    string // RDMA device IP, empty for TCP
	ConnectedAt time.Time
	bytesRecv   atomic.Int64
	bytesSent   atomic.Int64
	requests    atomic.Int64
}

// NewConnMetrics creates metrics for a connection.
func NewConnMetrics(id uint64, transport, remoteAddr, device, deviceIP string) *ConnMetrics {
	return &ConnMetrics{
		ID:          id,
		Transport:   transport,
		RemoteAddr:  remoteAddr,
		Device:      device,
		DeviceIP:    deviceIP,
		ConnectedAt: time.Now(),
	}
}

func (c *ConnMetrics) AddBytesRecv(n int64) { c.bytesRecv.Add(n) }
func (c *ConnMetrics) AddBytesSent(n int64) { c.bytesSent.Add(n) }
func (c *ConnMetrics) IncrRequests()        { c.requests.Add(1) }

func (c *ConnMetrics) BytesRecv() int64 { return c.bytesRecv.Load() }
func (c *ConnMetrics) BytesSent() int64 { return c.bytesSent.Load() }
func (c *ConnMetrics) Requests() int64  { return c.requests.Load() }

// Reset zeroes per-connection counters (called on FLUSH).
func (c *ConnMetrics) Reset() {
	c.bytesRecv.Store(0)
	c.bytesSent.Store(0)
	c.requests.Store(0)
}

// nicStats holds per-NIC cumulative wire byte counters for disconnected RDMA connections.
type nicStats struct {
	bytesRecv atomic.Int64
	bytesSent atomic.Int64
}

// ConnRegistry tracks active connections and accumulates lifetime totals.
type ConnRegistry struct {
	mu        sync.Mutex
	conns     map[uint64]*ConnMetrics
	connsPeak int // high-water mark for map compaction
	nextID    atomic.Uint64

	// Cumulative counters for disconnected connections, ensuring
	// global totals are monotonically increasing (Prometheus counter convention).
	totalBytesRecv atomic.Int64
	totalBytesSent atomic.Int64
	totalRequests  atomic.Int64

	// Per-NIC cumulative counters (for disconnected RDMA connections)
	nicMu        sync.Mutex
	nicTotals    map[string]*nicStats // keyed by device name
	nicIPs       map[string]string    // device -> IP (set once via RegisterNIC)
	nicLinkRates map[string]float64   // device -> link rate Gbps

	// Epoch tracking for throughput computation and flush tombstone.
	statsEpoch    atomic.Int64 // unix nanos — set to startedAt, updated on flush
	lastFlushedAt atomic.Int64 // unix nanos — 0 = never flushed

	// ClientStatsReg is set after construction; Deregister removes client stats on disconnect.
	ClientStatsReg *ClientStatsRegistry
}

// NewConnRegistry creates a connection registry.
func NewConnRegistry() *ConnRegistry {
	r := &ConnRegistry{
		conns:        make(map[uint64]*ConnMetrics),
		nicTotals:    make(map[string]*nicStats),
		nicIPs:       make(map[string]string),
		nicLinkRates: make(map[string]float64),
	}
	r.statsEpoch.Store(time.Now().UnixNano())
	return r
}

// RegisterNIC registers a known RDMA NIC so its metrics are always reported,
// even when no connections are active. Called once at startup per NIC.
func (r *ConnRegistry) RegisterNIC(device, ip string) {
	r.nicMu.Lock()
	defer r.nicMu.Unlock()
	if _, ok := r.nicTotals[device]; !ok {
		r.nicTotals[device] = &nicStats{}
	}
	r.nicIPs[device] = ip
}

// Register creates and registers a new connection, returning its metrics handle.
func (r *ConnRegistry) Register(transport, remoteAddr, device, deviceIP string) *ConnMetrics {
	id := r.nextID.Add(1)
	cm := NewConnMetrics(id, transport, remoteAddr, device, deviceIP)
	r.mu.Lock()
	r.conns[id] = cm
	if n := len(r.conns); n > r.connsPeak {
		r.connsPeak = n
	}
	r.mu.Unlock()
	return cm
}

// Deregister removes a connection and adds its final counts to cumulative totals.
func (r *ConnRegistry) Deregister(id uint64) {
	r.mu.Lock()
	cm, ok := r.conns[id]
	if ok {
		delete(r.conns, id)
	}
	r.conns, r.connsPeak = compactMap(r.conns, r.connsPeak)
	r.mu.Unlock()
	if r.ClientStatsReg != nil {
		r.ClientStatsReg.Remove(id)
	}
	if ok {
		recv := cm.BytesRecv()
		sent := cm.BytesSent()
		r.totalBytesRecv.Add(recv)
		r.totalBytesSent.Add(sent)
		r.totalRequests.Add(cm.Requests())

		if cm.Device != "" {
			r.nicMu.Lock()
			ns, exists := r.nicTotals[cm.Device]
			if !exists {
				ns = &nicStats{}
				r.nicTotals[cm.Device] = ns
			}
			ns.bytesRecv.Add(recv)
			ns.bytesSent.Add(sent)
			r.nicMu.Unlock()
		}
	}
}

// Snapshot returns a copy of active connections.
func (r *ConnRegistry) Snapshot() []*ConnMetrics {
	r.mu.Lock()
	defer r.mu.Unlock()
	out := make([]*ConnMetrics, 0, len(r.conns))
	for _, cm := range r.conns {
		out = append(out, cm)
	}
	return out
}

// ActiveCount returns the number of active connections.
func (r *ConnRegistry) ActiveCount() int {
	r.mu.Lock()
	defer r.mu.Unlock()
	return len(r.conns)
}

// TotalBytesRecv returns cumulative + active wire bytes received.
func (r *ConnRegistry) TotalBytesRecv() int64 {
	total := r.totalBytesRecv.Load()
	r.mu.Lock()
	for _, cm := range r.conns {
		total += cm.BytesRecv()
	}
	r.mu.Unlock()
	return total
}

// TotalBytesSent returns cumulative + active wire bytes sent.
func (r *ConnRegistry) TotalBytesSent() int64 {
	total := r.totalBytesSent.Load()
	r.mu.Lock()
	for _, cm := range r.conns {
		total += cm.BytesSent()
	}
	r.mu.Unlock()
	return total
}

// TotalRequests returns cumulative + active request count.
func (r *ConnRegistry) TotalRequests() int64 {
	total := r.totalRequests.Load()
	r.mu.Lock()
	for _, cm := range r.conns {
		total += cm.Requests()
	}
	r.mu.Unlock()
	return total
}

// NICWireBytes returns per-device [recv, sent] byte totals (cumulative + active).
func (r *ConnRegistry) NICWireBytes() map[string][2]int64 {
	result := make(map[string][2]int64)

	r.nicMu.Lock()
	for dev, ns := range r.nicTotals {
		result[dev] = [2]int64{ns.bytesRecv.Load(), ns.bytesSent.Load()}
	}
	r.nicMu.Unlock()

	r.mu.Lock()
	for _, cm := range r.conns {
		if cm.Device == "" {
			continue
		}
		cur := result[cm.Device]
		cur[0] += cm.BytesRecv()
		cur[1] += cm.BytesSent()
		result[cm.Device] = cur
	}
	r.mu.Unlock()

	return result
}

// NICIPs returns device -> IP mapping for registered NICs.
func (r *ConnRegistry) NICIPs() map[string]string {
	r.nicMu.Lock()
	defer r.nicMu.Unlock()
	out := make(map[string]string, len(r.nicIPs))
	for k, v := range r.nicIPs {
		out[k] = v
	}
	return out
}

// ResetAll zeroes all wire counters and resets the stats epoch (called on FLUSH).
func (r *ConnRegistry) ResetAll() {
	r.totalBytesRecv.Store(0)
	r.totalBytesSent.Store(0)
	r.totalRequests.Store(0)
	now := time.Now().UnixNano()
	r.lastFlushedAt.Store(now)
	r.statsEpoch.Store(now)
	r.mu.Lock()
	fresh := make(map[uint64]*ConnMetrics, len(r.conns))
	for k, cm := range r.conns {
		cm.Reset()
		fresh[k] = cm
	}
	r.conns = fresh
	r.connsPeak = len(fresh)
	r.mu.Unlock()
	r.nicMu.Lock()
	for _, ns := range r.nicTotals {
		ns.bytesRecv.Store(0)
		ns.bytesSent.Store(0)
	}
	r.nicMu.Unlock()
}

// RegisterNICLinkRate stores the detected (or overridden) link rate for a NIC.
func (r *ConnRegistry) RegisterNICLinkRate(device string, detectedGbps, overrideGbps float64) {
	r.nicMu.Lock()
	defer r.nicMu.Unlock()
	if overrideGbps > 0 {
		r.nicLinkRates[device] = overrideGbps
	} else {
		r.nicLinkRates[device] = detectedGbps
	}
}

// NICLinkRates returns device -> link rate in Gbps.
func (r *ConnRegistry) NICLinkRates() map[string]float64 {
	r.nicMu.Lock()
	defer r.nicMu.Unlock()
	out := make(map[string]float64, len(r.nicLinkRates))
	for k, v := range r.nicLinkRates {
		out[k] = v
	}
	return out
}

// NICThroughputGbps returns per-NIC bidirectional throughput in Gbps since epoch.
func (r *ConnRegistry) NICThroughputGbps() map[string]float64 {
	nicBytes := r.NICWireBytes()
	epochSec := time.Since(r.StatsEpoch()).Seconds()
	if epochSec <= 0 {
		epochSec = 1
	}
	result := make(map[string]float64, len(nicBytes))
	for dev, b := range nicBytes {
		result[dev] = float64(b[0]+b[1]) * 8 / (epochSec * 1e9)
	}
	return result
}

// StatsEpoch returns the time from which throughput is computed (start or last flush).
func (r *ConnRegistry) StatsEpoch() time.Time {
	return time.Unix(0, r.statsEpoch.Load())
}

// LastFlushedAt returns the time of the last flush, or zero if never flushed.
func (r *ConnRegistry) LastFlushedAt() time.Time {
	ns := r.lastFlushedAt.Load()
	if ns == 0 {
		return time.Time{}
	}
	return time.Unix(0, ns)
}
