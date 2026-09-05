package transport

import (
	"time"

	"github.com/anthony-chaudhary/fak/internal/l3server/metrics"
	"github.com/anthony-chaudhary/fak/internal/l3server/shard"
	"github.com/anthony-chaudhary/fak/internal/l3server/transport/dispatch"
)

// Transport is the common interface for all L3 transport servers (TCP, RDMA, CXL).
// Unifies lifecycle management and endpoint advertisement across transport types.
type Transport interface {
	Start() error
	Stop()
	Name() string
	SetRDMAEndpoints([]dispatch.RDMAEndpointInfo)
	SetClientStatsReg(*metrics.ClientStatsRegistry)
	SetMetricsAddr(string)
	SetPollerMetrics(metrics.PollerMetricsProvider)
	SetOpLatencyMetrics(metrics.OpLatencyProvider)
	SetDispatchTimeout(time.Duration)
	SetCluster(ring any, replicator any, localID string, snapshotDir string)
	SetManager(mgr *shard.Manager) // wire Manager after background allocation
}

// AllocAwareTransport extends Transport with allocator lifecycle hooks.
// Implemented by transports that register memory regions (e.g. RDMA with ibv_reg_mr).
type AllocAwareTransport interface {
	Transport
	shard.AllocChangeListener
	shard.AllocPreRegisterer
}
