package metrics

import "sync/atomic"

var (
	batchTimeoutsTotal       atomic.Int64
	rdmaConnectionRejections atomic.Int64
	tcpConnectionRejections  atomic.Int64

	tcpHandlerPanics   atomic.Int64
	rdmaDispatchPanics atomic.Int64
	rdmaPollerPanics   atomic.Int64
)

func IncrBatchTimeouts()              { batchTimeoutsTotal.Add(1) }
func BatchTimeouts() int64            { return batchTimeoutsTotal.Load() }
func IncrRDMAConnectionRejections()   { rdmaConnectionRejections.Add(1) }
func RDMAConnectionRejections() int64 { return rdmaConnectionRejections.Load() }
func IncrTCPConnectionRejections()    { tcpConnectionRejections.Add(1) }
func TCPConnectionRejections() int64  { return tcpConnectionRejections.Load() }

func IncrTCPHandlerPanics()     { tcpHandlerPanics.Add(1) }
func TCPHandlerPanics() int64   { return tcpHandlerPanics.Load() }
func IncrRDMADispatchPanics()   { rdmaDispatchPanics.Add(1) }
func RDMADispatchPanics() int64 { return rdmaDispatchPanics.Load() }
func IncrRDMAPollerPanics()     { rdmaPollerPanics.Add(1) }
func RDMAPollerPanics() int64   { return rdmaPollerPanics.Load() }

func compactMap[K comparable, V any](m map[K]V, peak int) (map[K]V, int) {
	cur := len(m)
	if peak > 64 && cur < peak/4 {
		fresh := make(map[K]V, cur)
		for k, v := range m {
			fresh[k] = v
		}
		return fresh, cur
	}
	return m, peak
}
