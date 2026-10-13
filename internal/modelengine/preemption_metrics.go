package modelengine

import "strings"

// KVPreemptionMetricWriter exposes the captured native scheduler's serving metrics.
type KVPreemptionMetricWriter interface {
	WriteKVPreemptionMetrics(*strings.Builder)
}

type nativePreemptionMetricWriter struct {
	scheduler *NativeScheduler
}

// KVPreemptionMetricWriter initializes and captures this engine's scheduler at startup.
// Reading the returned writer never initializes a model or a scheduler.
func (e *Engine) KVPreemptionMetricWriter() KVPreemptionMetricWriter {
	if e == nil {
		return nil
	}
	return nativePreemptionMetricWriter{scheduler: e.nativeScheduler()}
}

func (w nativePreemptionMetricWriter) WriteKVPreemptionMetrics(b *strings.Builder) {
	if w.scheduler == nil || b == nil {
		return
	}
	stats := w.scheduler.KVPreemptionStats()
	if stats.MaxBlocks <= 0 {
		return
	}
	writeNativePreemptionMetrics(b, stats)
}
